"""Database handles: :class:`Anatid` (one DuckDB file) and :class:`DatabasePool` (file-per-tenant).

See :class:`Anatid`'s docstring for the isolation, transaction and time-travel contract.  It is
the same text stored in the file's ``anatid_meta.contract`` column.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import duckdb

from . import recall as _recall
from . import schema as _schema
from .csr import CsrBackend
from .errors import (
    AnatidError,
    ConflictError,
    ExtensionUnavailable,
    TenantIsolationError,
)
from .schema import SchemaConfig, quote_ident
from .types import (
    FtsStatus,
    Isolation,
    Namespace,
    SchemaInfo,
)
from .verbs import MemoryVerbs

log = logging.getLogger("anatid")

__all__ = ["Anatid", "DatabasePool", "connect"]

__version__ = "0.1.0"

#: Substrings DuckDB uses for an MVCC abort.  Deliberately narrow, and it has to stay that way:
#: :class:`~anatid.errors.ConflictError` promises the caller that retrying the unit of work is
#: the right response, so only errors for which that is TRUE may be translated.
#:
#: In particular ``duckdb.TransactionException`` and the bare string "transactioncontext error"
#: are NOT sufficient: DuckDB raises the same family for deterministic programming errors --
#: "cannot start a transaction within a transaction", "cannot commit - no transaction is active",
#: "cannot rollback - no transaction is active" -- which fail identically on every retry.  A
#: retry loop keyed on ``ConflictError.retryable`` would spin forever on those.
_CONFLICT_MARKERS = ("conflict on update", "conflict on delete", "conflict on tuple",
                     "transaction is aborted", "could not serialize")

#: Deterministic ``TransactionContext`` errors that must keep their own type.  Checked first.
_NOT_CONFLICT_MARKERS = ("no transaction is active", "transaction within a transaction",
                         "transaction is launched in read-only mode",
                         "cannot write to database")


def _translate(exc: BaseException) -> BaseException:
    """Map a DuckDB error onto anatid's hierarchy where anatid gives it a stable meaning.

    Only the write-write abort is translated (to :class:`~anatid.errors.ConflictError`); every
    other DuckDB error keeps its own type so callers can tell a planner error, a misuse of the
    transaction API or a read-only database from a race.
    """
    if isinstance(exc, AnatidError):
        return exc
    if not isinstance(exc, duckdb.Error):
        return exc
    msg = str(exc).lower()
    if any(m in msg for m in _NOT_CONFLICT_MARKERS):
        return exc
    if any(m in msg for m in _CONFLICT_MARKERS):
        return ConflictError(str(exc), cause=exc)
    return exc


class Anatid(MemoryVerbs):
    """An open anatid database: one DuckDB file (or ``":memory:"``) plus one default namespace.

    ::

        with Anatid.open("agent.anatid", tenant=1, embedding_dim=1536) as db:
            m = db.remember("Ada prefers dark roast", entities=["Ada", "coffee"])
            hits = db.recall("coffee", embedding=vec, k=5)

    Isolation contract
    ------------------
    **DuckDB has no schema-level or row-level access control.**  Nothing in the engine can stop a
    connection from reading any row in the file it has open.  anatid therefore offers exactly two
    levels, and names them honestly:

    ``Isolation.FILE_PER_TENANT`` -- *real isolation.*
        The tenant gets its own DuckDB file.  The boundary is the filesystem plus this wrapper: a
        handle opened for tenant 7 raises :class:`~anatid.errors.TenantIsolationError` if any verb
        is asked for another tenant_id.  :class:`DatabasePool` manages a directory of such files.
        Cross-tenant reads are possible only by explicitly attaching another file READ ONLY
        (:meth:`attach_read_only`), which is an auditable act, not a default.

    ``Isolation.SCOPED`` (the default) -- *scoping, NOT isolation.*
        Several tenants share one file and are separated only by the ``tenant_id`` column that
        anatid puts into every predicate it generates.  Any code with a connection to that file --
        including any raw SQL a caller runs through :attr:`connection` -- can read every tenant in
        it.  Use this for a single user or a single trust domain.  Do not use it as a security
        boundary between mutually distrusting tenants; use file-per-tenant for that.

    Namespaces/schemas *inside* one file are likewise scoping, not isolation.

    Transaction contract
    --------------------
    DuckDB's MVCC is **optimistic and snapshot-isolated -- not serializable.**  Concretely:

    * Appends never conflict.  Concurrent ``remember()`` calls from many threads **in one
      process** all commit (the Phase 0 spike measured 4 writer + 2 reader threads, each on its
      own ``cursor()`` to the same file, for 30 s with 0 errors).  DuckDB allows exactly one
      read-write **process** per file: a second one fails to open it at all, with ``IO Error:
      Could not set lock on file ...: Conflicting lock is held``.  Multi-process means
      file-per-tenant (:class:`DatabasePool`), or one writer process and readers that open the
      file ``read_only=True``.
    * Two concurrent transactions that update the **same row** conflict.  The loser is aborted at
      statement time, not at COMMIT, with ``TransactionContext Error: Conflict on update!``.
      anatid raises that as :class:`~anatid.errors.ConflictError`, which is retryable: nothing was
      committed, so roll back and re-run the unit of work.
    * A transaction sees the snapshot it started with.  Two transactions can both read a value,
      both decide based on it and both commit if they write different rows -- that is snapshot
      isolation, and it permits write skew.  Do not describe it as serializable.
    * One writer transaction at a time makes progress per row; anatid never retries for you,
      because the correct retry depends on what your verb was doing.
    * Every **write** verb (``remember``, ``supersede``, ``forget``, ``relate``, ``reinforce``,
      ``episode``, ``entity_id``) is exactly one transaction.  ``prune`` is not: it is one
      ``SELECT`` plus one transaction per memory it forgets, so a failure part-way leaves the
      earlier deletions committed -- read ``PruneReport.memory_ids`` from a ``dry_run`` first.
      **Read** verbs (``recall``, ``recall_2hop``, ``context``, ``get``, ``provenance``,
      ``stats``) open no transaction at all: ``recall`` issues roughly six independent statements
      (staleness probe, up to three arms, hydrate, about-names), so a commit by another thread
      can land between them.  Wrap the call in ``with db.transaction():`` yourself when you need
      all of it on one snapshot.

    Time travel contract
    --------------------
    DuckDB has **no** ``AS OF SYSTEM TIME``.  ``as_of=`` is anatid's own filter over the
    ``valid_from``/``valid_to`` and ``tx_from``/``tx_to`` columns, compiled into the WHERE clause
    (:func:`anatid.schema.temporal_predicate`).  Intervals are half-open.  A hard purge
    (``forget(hard=True)``) removes the row from history too -- erasure beats auditability by
    design, and the receipt is handed back to the caller to log elsewhere.

    Full-text contract
    ------------------
    The DuckDB ``fts`` index is not incremental.  Rows written after the last
    :meth:`rebuild_fts_index` are invisible to the BM25 arm of :meth:`recall`, which reports it on
    the result (``hits.bm25_stale``, ``hits.pending_fts_rows``) and logs a warning.  Nothing is
    rebuilt implicitly: a rebuild is O(corpus) and belongs to your write path, not to an unlucky
    read.

    Threading
    ---------
    A ``DuckDBPyConnection`` is not safe for concurrent use from several threads.  anatid keeps
    one **connection per thread**, created lazily from the root connection with ``.cursor()``
    (same database, independent transaction state) and exposed as :attr:`connection`.  Verbs pick
    it up automatically, so an ``Anatid`` handle can be shared by a thread pool.  A transaction
    belongs to the thread that opened it.
    """

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        *,
        path: str,
        namespace: Namespace,
        config: SchemaConfig,
        backend: CsrBackend,
        read_only: bool = False,
        owns_connection: bool = True,
    ) -> None:
        self._root = con
        self.path = path
        self.namespace = namespace
        self.config = config
        self.csr = backend
        self.read_only = read_only
        self._owns = owns_connection
        self._closed = False
        self._local = threading.local()
        self._lock = threading.RLock()
        self._attached: dict[str, str] = {}
        self.erasure_hooks: list[Any] = []

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    def open(
        cls,
        path: str | os.PathLike = ":memory:",
        *,
        tenant: int | Namespace = 0,
        embedding_dim: int = _schema.DEFAULT_EMBEDDING_DIM,
        isolation: Isolation | None = None,
        read_only: bool = False,
        ensure: bool = True,
        indexes: Sequence[str] | None = None,
        threads: int | None = None,
        memory_limit: str | None = None,
        use_csr_extension: bool = False,
        extension_path: str | os.PathLike | None = None,
        require_extension: bool = False,
        duckdb_config: dict[str, Any] | None = None,
        fts: bool = True,
    ) -> "Anatid":
        """Open (and by default create) an anatid database.

        ``path``
            A file path, or ``":memory:"`` for a throwaway in-process database.
        ``tenant``
            The default namespace for every verb.  An int becomes a
            :class:`~anatid.types.Namespace` with ``isolation`` (default
            :attr:`~anatid.types.Isolation.SCOPED`).
        ``embedding_dim``
            ``N`` in ``FLOAT[N]``.  Used only when creating the schema; an existing file keeps
            the dimension recorded in ``anatid_meta`` and this argument is checked against it.
        ``use_csr_extension``
            Open with ``allow_unsigned_extensions`` and try to load the optional C++ graph
            extension.  This flag can only be set at open time -- DuckDB reads it from the connect
            config.  Failure falls back to the pure-SQL expansion (identical results) unless
            ``require_extension=True``, which raises :class:`~anatid.errors.ExtensionUnavailable`.
        ``fts``
            Install/load the ``fts`` extension so BM25 is available.  Set False for an air-gapped
            environment with no extension repository; ``recall()`` then runs without the text arm
            and says so.
        """
        p = ":memory:" if str(path) == ":memory:" else str(Path(path).expanduser())
        ns = Namespace.coerce(tenant)
        if isolation is not None:
            ns = Namespace(ns.tenant_id, ns.label, Isolation(isolation))

        cfg_kwargs: dict[str, Any] = dict(duckdb_config or {})
        if use_csr_extension or require_extension:
            cfg_kwargs["allow_unsigned_extensions"] = "true"
        if threads is not None:
            cfg_kwargs["threads"] = int(threads)
        if memory_limit is not None:
            cfg_kwargs["memory_limit"] = memory_limit

        con = duckdb.connect(p, read_only=read_only, config=cfg_kwargs) if cfg_kwargs \
            else duckdb.connect(p, read_only=read_only)

        if fts:
            try:
                con.execute("INSTALL fts")
                con.execute("LOAD fts")
            except duckdb.Error as exc:      # offline / no extension repository
                log.warning("fts extension unavailable (%s); BM25 recall will be disabled", exc)

        backend = CsrBackend(enabled=use_csr_extension or require_extension,
                             extension_path=extension_path, require=require_extension)
        if backend.enabled:
            backend.load(con)

        existing_dim = _schema.embedding_dim(con)
        if existing_dim is not None and int(existing_dim) != int(embedding_dim):
            log.info("using embedding_dim=%s recorded in %s (open() was given %s)",
                     existing_dim, p, embedding_dim)
            embedding_dim = int(existing_dim)

        idx = tuple(indexes) if indexes is not None else tuple(_schema.DEFAULT_INDEXES)
        cfg = SchemaConfig(embedding_dim=embedding_dim, indexes=idx)

        db = cls(con, path=p, namespace=ns, config=cfg, backend=backend, read_only=read_only)
        if ensure and not read_only:
            db.ensure_schema()
        return db

    def close(self) -> None:
        """Close every connection this handle owns.  Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            child = getattr(self._local, "con", None)
            if child is not None:
                try:
                    child.close()
                except duckdb.Error:
                    pass
                self._local.con = None
            if self._owns:
                try:
                    self._root.close()
                except duckdb.Error:
                    pass

    def register_erasure_hook(self, hook) -> None:
        """Extend ``forget(hard=True)`` to a table anatid does not own.

        ``hook(db, memory_id, tenant_id, content) -> int`` is called **inside** the purge
        transaction -- ``content`` is the memory's text, read before the row is deleted, because
        a verbatim copy of the text is as much a copy as the id is -- and returns how many rows
        it deleted; the total lands in
        :attr:`~anatid.types.ForgetReceipt.extra_rows_deleted`.  A hook that raises aborts the
        whole purge, so a partial erasure is never committed.

        Hooks are held on this handle for its lifetime (a bound method keeps its object alive),
        and registering the same one twice is a no-op.

        This exists because a right-to-erasure purge has to cover every copy of the content in
        the file, and anatid's verbs only know about the memory graph.  The obvious other copy is
        a conversation transcript: ``AnatidSession`` writes the tool call and its JSON result --
        including the memory's id *and* its verbatim content -- into ``agent_messages`` in the
        same file, and would replay it into the model's context on the next turn.  The session
        registers a hook for exactly that.  Any other table you write here is yours to cover.
        """
        if not callable(hook):
            raise TypeError("erasure hook must be callable")
        if hook not in self.erasure_hooks:
            self.erasure_hooks.append(hook)

    def __enter__(self) -> "Anatid":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"<Anatid path={self.path!r} tenant={self.namespace.tenant_id} "
                f"isolation={self.namespace.isolation.value} dim={self.config.embedding_dim} "
                f"expand={self.csr.active}{' closed' if self._closed else ''}>")

    @property
    def closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------------ connections

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """This thread's DuckDB connection (a ``cursor()`` on the same database).

        Created on first use per thread.  Raw SQL run through it bypasses every tenant filter
        anatid would have added -- with ``Isolation.SCOPED`` that means it can read every tenant
        in the file.  That is a property of DuckDB, not a bug in anatid.
        """
        if self._closed:
            raise AnatidError(f"database {self.path!r} is closed")
        con = getattr(self._local, "con", None)
        if con is None:
            con = self._root.cursor()
            self._local.con = con
            self._local.depth = 0
            if self.csr.enabled and self.csr.extension_path is not None:
                # each cursor is its own connection: extensions must be loaded per connection
                try:
                    con.execute(f"LOAD '{self.csr.extension_path.as_posix()}'")
                except duckdb.Error:
                    pass
        return con

    def execute(self, sql: str, params: Sequence[Any] | None = None, *, con=None):
        """Run one statement on this thread's connection, translating MVCC aborts.

        The only error translation is the one anatid promises: a DuckDB write-write conflict
        becomes :class:`~anatid.errors.ConflictError`.  Everything else propagates unchanged.
        """
        c = con if con is not None else self.connection
        try:
            return c.execute(sql, list(params) if params is not None else None)
        except Exception as exc:
            raise _translate(exc) from exc

    @contextmanager
    def transaction(self, con=None) -> Iterator[duckdb.DuckDBPyConnection]:
        """Explicit transaction on this thread's connection.

        Re-entrant: a nested ``with db.transaction()`` joins the outer one (DuckDB has no usable
        savepoints, so anatid does not pretend to offer nested rollback).  On any exception the
        outermost block rolls back and re-raises; a write-write conflict surfaces as
        :class:`~anatid.errors.ConflictError`, which is retryable because nothing was committed.
        """
        c = con if con is not None else self.connection
        depth = getattr(self._local, "depth", 0)
        if depth:
            self._local.depth = depth + 1
            try:
                yield c
            finally:
                self._local.depth -= 1
            return
        try:
            c.execute("BEGIN")
        except Exception as exc:
            raise _translate(exc) from exc
        self._local.depth = 1
        try:
            yield c
        except Exception as exc:
            self._local.depth = 0
            try:
                c.execute("ROLLBACK")
            except duckdb.Error:
                pass
            raise _translate(exc) from exc
        else:
            self._local.depth = 0
            try:
                c.execute("COMMIT")
            except Exception as exc:
                try:
                    c.execute("ROLLBACK")
                except duckdb.Error:
                    pass
                raise _translate(exc) from exc

    # ------------------------------------------------------------------ tenancy

    def resolve_tenant(self, tenant: int | Namespace | None = None) -> Namespace:
        """Resolve a verb's ``tenant=`` argument, enforcing the file-per-tenant boundary.

        With :attr:`~anatid.types.Isolation.FILE_PER_TENANT`, asking this handle for any tenant
        other than the one it was opened for raises
        :class:`~anatid.errors.TenantIsolationError`.  With ``SCOPED`` any tenant_id in the file
        is reachable -- that is the whole difference between the two levels.
        """
        ns = Namespace.coerce(tenant, self.namespace)
        if self.namespace.is_isolated and ns.tenant_id != self.namespace.tenant_id:
            raise TenantIsolationError(
                f"handle for {self.path!r} is bound to tenant {self.namespace.tenant_id} "
                f"(file-per-tenant isolation); it will not touch tenant {ns.tenant_id}. "
                f"Open that tenant's own file, or attach it read-only.")
        return ns

    def attach_read_only(self, path: str | os.PathLike, alias: str) -> str:
        """``ATTACH`` another anatid file READ ONLY under ``alias`` for cross-tenant reads.

        This is the only sanctioned way to read across a file-per-tenant boundary, and it is
        deliberately explicit: the caller names the file and the alias, the attachment is
        read-only, and :meth:`detach` removes it.  Query it as ``alias.memories`` etc. through
        :attr:`connection`; anatid's verbs always stay in ``main``.
        """
        quote_ident(alias)
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"no database file at {p}")
        # The path is a filesystem path we just verified, and single quotes in it are escaped.
        literal = str(p).replace("'", "''")
        self.execute(f"ATTACH '{literal}' AS {quote_ident(alias)} (READ_ONLY)")
        self._attached[alias] = str(p)
        return alias

    def detach(self, alias: str) -> None:
        """Undo :meth:`attach_read_only`."""
        quote_ident(alias)
        self.execute(f"DETACH {quote_ident(alias)}")
        self._attached.pop(alias, None)

    @property
    def attached(self) -> dict[str, str]:
        """Read-only databases currently attached: ``{alias: path}``."""
        return dict(self._attached)

    # ------------------------------------------------------------------ schema

    def ensure_schema(self) -> int:
        """Create the schema if absent, migrate it if behind.  Returns the schema version."""
        with self.transaction():
            return _schema.ensure_schema(self.connection, self.config,
                                         anatid_version=__version__)

    def info(self) -> SchemaInfo:
        """The ``anatid_meta`` catalog row, including the stored contract notes."""
        row = self.execute(
            "SELECT schema_version, created_at, embedding_dim, anatid_version, duckdb_version,"
            " fts_indexed_at, fts_indexed_rows, contract FROM anatid_meta LIMIT 1").fetchone()
        if row is None:
            raise AnatidError(f"{self.path!r} has no anatid_meta row; call ensure_schema()")
        return SchemaInfo(
            schema_version=int(row[0]), created_at=row[1], embedding_dim=int(row[2]),
            anatid_version=row[3], duckdb_version=row[4], fts_indexed_at=row[5],
            fts_indexed_rows=None if row[6] is None else int(row[6]), contract=row[7],
            extras={"path": self.path, "tenant": self.namespace.tenant_id,
                    "isolation": self.namespace.isolation.value,
                    "expand_path": self.csr.active})

    def create_node_label(
        self,
        label: str,
        properties: Sequence[tuple[str, str]] = (),
        *,
        id_column: str | None = None,
        system_columns: bool = True,
    ) -> str:
        """Create a table for a user-defined node label.  System columns are on by default."""
        ddl = _schema.node_table_ddl(label, properties, id_column=id_column,
                                     system_columns=system_columns)
        self.execute(ddl)
        return ddl

    def create_edge_type(
        self,
        edge_type: str,
        properties: Sequence[tuple[str, str]] = (),
        *,
        table: str | None = None,
        system_columns: bool = True,
    ) -> str:
        """Create a table for a user-defined edge type.  System columns are on by default."""
        ddl = _schema.edge_table_ddl(edge_type, properties, table=table,
                                     system_columns=system_columns)
        self.execute(ddl)
        return ddl

    def recluster(self, tables: Sequence[str] | None = None) -> dict[str, int]:
        """Rewrite tables in the physical order that made the Phase 0 benchmark fast.

        Bulk loads should insert with these ``ORDER BY``s already applied (see
        :meth:`load_parquet`); this rebuilds the ordering after a long run of appends, which land
        in arrival order.  Rows written later are still correct without it -- DuckDB filters them
        per row group as usual -- so treat this as compaction, not as a requirement.

        The replacement table is created from :func:`anatid.schema.table_ddl` and then filled by
        ``INSERT ... SELECT * ... ORDER BY``, **not** by ``CREATE TABLE AS SELECT``: a CTAS keeps
        only the column names and types and silently drops every ``NOT NULL`` and every column
        ``DEFAULT``, which would let ``memory_id``/``tenant_id``/``created_at`` go NULL
        afterwards and leave rows no verb can ever reach again.

        Returns ``{table: row_count}``.  Indexes on the rebuilt tables are recreated afterwards.
        """
        names = list(tables) if tables is not None else list(_schema.CLUSTER_ORDER)
        out: dict[str, int] = {}
        con = self.connection
        for name in names:
            order = _schema.CLUSTER_ORDER.get(name)
            if order is None:
                raise ValueError(f"no cluster order defined for table {name!r}")
            t = quote_ident(name)
            tmp_name = f"{name}__reclustered"
            tmp = quote_ident(tmp_name)
            ddl = _schema.table_ddl(name, self.config, as_table=tmp_name)
            cols = ", ".join(quote_ident(r[1]) for r in
                             con.execute(f"PRAGMA table_info({t})").fetchall())
            with self.transaction():
                self.execute(f"DROP TABLE IF EXISTS {tmp}")
                self.execute(ddl)
                self.execute(f"INSERT INTO {tmp} ({cols}) "
                             f"SELECT {cols} FROM {t} ORDER BY {order}")
                self.execute(f"DROP TABLE {t}")
                self.execute(f"ALTER TABLE {tmp} RENAME TO {quote_ident(name)}")
            out[name] = int(con.execute(f"SELECT count(*) FROM {t}").fetchone()[0])
        for stmt in _schema.index_statements(self.config):
            try:
                self.execute(stmt)
            except duckdb.Error as exc:      # index on a table we did not rebuild
                log.debug("recluster: skipping %s (%s)", stmt, exc)
        return out

    # ------------------------------------------------------------------ full text

    def fts_status(self, *, deep: bool = False) -> FtsStatus:
        """How far the (non-incremental) BM25 index has fallen behind ``memories``."""
        return _recall.fts_status(self.connection, deep=deep)

    def rebuild_fts_index(self, *, now=None, terms_index: bool = True) -> FtsStatus:
        """Rebuild the BM25 index and record the watermark.  See the class docstring."""
        with self.transaction():
            return _recall.rebuild_fts_index(self.connection, now=now, terms_index=terms_index)

    # ------------------------------------------------------------------ graph backend

    def build_csr(self) -> Any:
        """(Re)build the optional C++ extension's in-memory CSR snapshot of ``edges_relates``.

        No-op returning ``None`` when the extension is not loaded.  Call after a bulk load or a
        batch of RELATES_TO writes; anatid marks the snapshot stale on every such write and falls
        back to the pure-SQL expansion until it is rebuilt.
        """
        return self.csr.build(self.connection)

    @property
    def expand_path(self) -> str:
        """Which graph-expansion path current-state reads take: ``"extension"`` or ``"sql"``."""
        return self.csr.active

    def require_csr_extension(self) -> None:
        """Raise :class:`~anatid.errors.ExtensionUnavailable` unless the C++ path is live."""
        if not self.csr.fresh:
            raise ExtensionUnavailable(
                f"csr extension not active (state: {self.csr.describe()})")

    # ------------------------------------------------------------------ bulk load

    def load_parquet(
        self,
        directory: str | os.PathLike,
        *,
        tables: Sequence[str] | None = None,
        rebuild_fts: bool = True,
        build_csr: bool = True,
    ) -> dict[str, int]:
        """Bulk-load a directory of Parquet files written in anatid's own column order.

        Files are matched by table name (``memories.parquet``, ``entities.parquet``,
        ``edges_about.parquet``, ``edges_relates.parquet``, ``edges_supersedes.parquet``,
        ``episodes.parquet``).  Missing files are skipped.  Columns are matched **by name**, so a
        file may carry a subset; the rest take their defaults.  Each table is inserted with the
        :data:`anatid.schema.CLUSTER_ORDER` ``ORDER BY`` that the Phase 0 spike measured.

        Returns ``{table: rows_inserted}``.
        """
        d = Path(directory).expanduser()
        wanted = list(tables) if tables is not None else [
            "entities", "episodes", "memories", "edges_about", "edges_relates", "edges_supersedes"]
        dim = self.config.embedding_dim
        out: dict[str, int] = {}
        con = self.connection
        for name in wanted:
            f = d / f"{name}.parquet"
            if not f.is_file():
                continue
            src = str(f).replace("'", "''")
            have = [r[0] for r in con.execute(
                "SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet(?))",
                [str(f)]).fetchall()]
            cols = [r[1] for r in con.execute(
                f"PRAGMA table_info({quote_ident(name)})").fetchall()]
            shared = [c for c in cols if c in have]
            if not shared:
                continue
            proj = ", ".join(
                (f"{quote_ident(c)}::FLOAT[{dim}]" if c == "embedding" else quote_ident(c))
                for c in shared)
            order = _schema.CLUSTER_ORDER.get(name)
            order_sql = f" ORDER BY {order}" if order else ""
            before = int(con.execute(f"SELECT count(*) FROM {quote_ident(name)}").fetchone()[0])
            self.execute(
                f"INSERT INTO {quote_ident(name)} ({', '.join(quote_ident(c) for c in shared)}) "
                f"SELECT {proj} FROM read_parquet('{src}'){order_sql}")
            after = int(con.execute(f"SELECT count(*) FROM {quote_ident(name)}").fetchone()[0])
            out[name] = after - before
        if rebuild_fts and out.get("memories"):
            try:
                self.rebuild_fts_index()
            except duckdb.Error as exc:
                log.warning("fts index rebuild after load failed: %s", exc)
        if build_csr and self.csr.enabled:
            self.build_csr()
        return out


def connect(path: str | os.PathLike = ":memory:", **kwargs) -> Anatid:
    """Shorthand for :meth:`Anatid.open`."""
    return Anatid.open(path, **kwargs)


# --------------------------------------------------------------------------- pool

class DatabasePool:
    """A directory of per-tenant anatid files, opened lazily and closed LRU.

    This is how anatid delivers *real* tenant isolation: one DuckDB file per tenant, so a handle
    physically cannot see another tenant's rows (DuckDB has no row-level security to lean on).
    Opening a DuckDB file is cheap, so the pool keeps at most ``max_open`` of them and closes the
    least recently used beyond that.

    ::

        pool = DatabasePool("/var/lib/anatid/tenant_{tenant}.duckdb", embedding_dim=1536)
        db = pool.get(42)                       # opens (and creates) tenant 42's file
        db.remember("...", entities=["Ada"])    # tenant 42, enforced

    ``path_template`` may use ``{tenant}`` and ``{label}``.  Handles it hands out carry
    :attr:`~anatid.types.Isolation.FILE_PER_TENANT`, so any verb called with a different tenant
    raises :class:`~anatid.errors.TenantIsolationError`.

    Cross-tenant reads go through :meth:`attach_read_only`, which attaches another tenant's file
    READ ONLY under an alias -- explicit, auditable, and unable to write.

    The pool is thread-safe.  The :class:`Anatid` handles it returns are themselves thread-safe
    (connection per thread), so a request handler can fetch one per request.
    """

    def __init__(
        self,
        path_template: str | os.PathLike,
        *,
        max_open: int = 16,
        create_parents: bool = True,
        **open_kwargs: Any,
    ) -> None:
        self.path_template = str(path_template)
        if "{tenant}" not in self.path_template and "{label}" not in self.path_template:
            raise ValueError(
                "path_template must contain {tenant} (or {label}) so tenants get separate files")
        self.max_open = int(max_open)
        if self.max_open < 1:
            raise ValueError("max_open must be >= 1")
        self.create_parents = create_parents
        self.open_kwargs = open_kwargs
        self._open: "OrderedDict[int, Anatid]" = OrderedDict()
        self._lock = threading.RLock()
        self._closed = False

    # -------------------------------------------------------------- paths

    def path_for(self, tenant: int | Namespace) -> Path:
        """The file this pool would use for ``tenant`` (whether or not it exists yet)."""
        ns = Namespace.coerce(tenant)
        return Path(self.path_template.format(tenant=ns.tenant_id, label=ns.name)).expanduser()

    def known_tenants(self) -> list[int]:
        """Tenant ids currently held open by the pool, most recently used last."""
        with self._lock:
            return list(self._open)

    # -------------------------------------------------------------- handles

    def get(self, tenant: int | Namespace) -> Anatid:
        """Return (opening if needed) the handle for ``tenant``.  Marks it most-recently-used."""
        if self._closed:
            raise AnatidError("pool is closed")
        ns = Namespace.coerce(tenant)
        ns = Namespace(ns.tenant_id, ns.label, Isolation.FILE_PER_TENANT)
        with self._lock:
            db = self._open.get(ns.tenant_id)
            if db is not None and not db.closed:
                self._open.move_to_end(ns.tenant_id)
                return db
            path = self.path_for(ns)
            if self.create_parents:
                path.parent.mkdir(parents=True, exist_ok=True)
            kwargs = dict(self.open_kwargs)
            kwargs.pop("tenant", None)
            kwargs.pop("isolation", None)
            db = Anatid.open(path, tenant=ns, isolation=Isolation.FILE_PER_TENANT, **kwargs)
            self._open[ns.tenant_id] = db
            self._open.move_to_end(ns.tenant_id)
            self._evict()
            return db

    __getitem__ = get

    def _evict(self) -> None:
        while len(self._open) > self.max_open:
            _tid, victim = self._open.popitem(last=False)
            victim.close()

    def close(self, tenant: int | Namespace) -> bool:
        """Close one tenant's handle.  Returns True if it was open."""
        ns = Namespace.coerce(tenant)
        with self._lock:
            db = self._open.pop(ns.tenant_id, None)
        if db is None:
            return False
        db.close()
        return True

    def close_all(self) -> None:
        """Close every open handle.  Idempotent."""
        with self._lock:
            self._closed = True
            handles = list(self._open.values())
            self._open.clear()
        for db in handles:
            db.close()

    def __enter__(self) -> "DatabasePool":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close_all()

    def __len__(self) -> int:
        with self._lock:
            return len(self._open)

    def __repr__(self) -> str:
        return (f"<DatabasePool template={self.path_template!r} open={len(self)}"
                f"/{self.max_open}>")

    # -------------------------------------------------------------- cross-tenant

    def attach_read_only(
        self,
        host: int | Namespace,
        other: int | Namespace,
        *,
        alias: str | None = None,
    ) -> str:
        """Attach ``other``'s file READ ONLY inside ``host``'s handle and return the alias.

        The only supported cross-tenant read.  It is explicit by design: the resulting alias is
        visible in ``host.attached``, the attachment cannot write, and anatid's own verbs never
        look outside ``main``, so a cross-tenant query has to be written by hand::

            alias = pool.attach_read_only(1, 2)
            rows = pool.get(1).connection.execute(
                f"SELECT content FROM {alias}.memories LIMIT 5").fetchall()

        A DuckDB file cannot be attached while the same process holds it open read-write, so the
        pool closes ``other``'s handle first (it reopens on the next :meth:`get`).  If a *different*
        process has it open read-write, DuckDB refuses the attach -- that is the file lock, not
        anatid.
        """
        host_ns = Namespace.coerce(host)
        other_ns = Namespace.coerce(other)
        if host_ns.tenant_id == other_ns.tenant_id:
            raise TenantIsolationError("cannot attach a tenant to itself")
        name = alias or f"tenant_{other_ns.tenant_id}"
        path = self.path_for(other_ns)
        host_db = self.get(host_ns)
        self.close(other_ns)
        return host_db.attach_read_only(path, name)
