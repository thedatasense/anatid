"""Database handles: :class:`Anatid` (one DuckDB file) and :class:`DatabasePool` (file-per-tenant).

See :class:`Anatid`'s docstring for the isolation, transaction and time-travel contract.  It is
the same text stored in the file's ``anatid_meta.contract`` column.
"""

from __future__ import annotations

import logging
import os
import threading
import time
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
    IntegrityError,
    TenantIsolationError,
)
from .schema import SchemaConfig, quote_ident
from .types import (
    DOCTOR_SAMPLE_LIMIT,
    DoctorFinding,
    DoctorReport,
    FtsStatus,
    Isolation,
    Namespace,
    SchemaInfo,
    Severity,
    utcnow,
)
from .verbs import MemoryVerbs

log = logging.getLogger("anatid")

__all__ = ["Anatid", "DatabasePool", "connect"]

__version__ = "0.1.1"

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


def _scalar_int(result: duckdb.DuckDBPyConnection) -> int:
    """First column of the first row of an aggregate query, as an int (0 for no row / NULL).

    ``fetchone()`` is typed ``Optional``; every ``count(*)`` here has exactly one row, and this
    says so once instead of at every call site.
    """
    row = result.fetchone()
    return int(row[0]) if row is not None and row[0] is not None else 0


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
        registers a hook for exactly that, as ``RunStateStore`` does for ``agent_run_states``.

        Hooks are per handle, and a file outlives every handle, so the tables anatid's own
        integrations create are also purged **by name, on every handle**
        (:data:`anatid.erasure.BUNDLED_INTEGRATION_TABLES`): a ``forget(hard=True)`` from a
        second process or from the MCP server reaches them without any hook having been
        registered there.  Any other table you write here is yours to cover with a hook.
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

    @property
    def in_transaction(self) -> bool:
        """True when this thread already has a :meth:`transaction` open.

        The verbs read it to decide whether a lost entity-creation race is theirs to retry: a
        transaction the *caller* opened is the caller's to re-run, because only they know what
        else went into it.  See :meth:`anatid.verbs.MemoryVerbs._atomic`.
        """
        return bool(getattr(self._local, "depth", 0))

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

        Returns ``{table: row_count}``.  Indexes on a rebuilt table are recreated **inside the
        same transaction** as the rebuild, which matters for one of them: ``DROP TABLE entities``
        takes the ``UNIQUE (tenant_id, entity_key)`` index with it, and that index is the
        enforcement behind "one entity per canonical name per tenant".  Recreating it afterwards
        would leave a window in which two concurrent ``remember()`` calls could put two rows for
        one name into the table -- re-opening exactly the race schema v3 closed, during
        maintenance.
        """
        names = list(tables) if tables is not None else list(_schema.CLUSTER_ORDER)
        out: dict[str, int] = {}
        con = self.connection
        wanted = dict(_schema.REQUIRED_INDEXES)
        for index_name in self.config.indexes:
            sql = _schema.DEFAULT_INDEXES.get(index_name) or _schema.OPTIONAL_INDEXES.get(index_name)
            if sql:
                wanted[index_name] = sql
        for name in names:
            order = _schema.CLUSTER_ORDER.get(name)
            if order is None:
                raise ValueError(f"no cluster order defined for table {name!r}")
            t = quote_ident(name)
            tmp_name = f"{name}__reclustered"
            tmp = quote_ident(tmp_name)
            ddl = _schema.table_ddl(name, self.config, as_table=tmp_name)
            # NOT PRAGMA table_info directly: it lists generated columns (entities.entity_key)
            # like any other, and DuckDB rejects those in an INSERT column list with
            # "Binder Error: Cannot insert into a generated column".
            cols = ", ".join(quote_ident(c) for c in _schema.insertable_columns(con, name))
            with self.transaction():
                self.execute(f"DROP TABLE IF EXISTS {tmp}")
                self.execute(ddl)
                self.execute(f"INSERT INTO {tmp} ({cols}) "
                             f"SELECT {cols} FROM {t} ORDER BY {order}")
                self.execute(f"DROP TABLE {t}")
                self.execute(f"ALTER TABLE {tmp} RENAME TO {quote_ident(name)}")
                # Same transaction as the DROP: see the docstring.  `ON <table> (` is how the
                # statement names its table -- every one of these is generated in schema.py.
                for stmt in wanted.values():
                    if f" ON {name} (" in stmt:
                        self.execute(stmt)
            out[name] = _scalar_int(con.execute(f"SELECT count(*) FROM {t}"))
        # Anything the per-table pass did not cover (a user-configured index, or one whose
        # table was not rebuilt).  All of these are CREATE INDEX IF NOT EXISTS, so this is a
        # no-op for indexes that survived.
        for stmt in _schema.index_statements(self.config) + _schema.required_index_statements():
            try:
                self.execute(stmt)
            except duckdb.Error as exc:      # index on a table we did not rebuild
                log.debug("recluster: skipping %s (%s)", stmt, exc)
        return out

    # ------------------------------------------------------------------ health

    def doctor(
        self,
        *,
        tenant: int | Namespace | None = None,
        all_tenants: bool = False,
        deep: bool = True,
        samples: int = DOCTOR_SAMPLE_LIMIT,
        raise_on_error: bool = False,
    ) -> DoctorReport:
        """Check the file for the faults anatid's verbs assume cannot happen, and report them.

        ``doctor()`` **reads only**.  It never repairs, never rebuilds an index and never
        deletes a row, because every fault it finds has more than one defensible repair and
        picking one silently is how a health check destroys data.  It hands you a
        :class:`~anatid.types.DoctorReport`; what to do about it is yours.

        Why it exists: anatid 0.1.0 accepted duplicate memory ids inside one tenant,
        ``confidence=-1``, ``confidence=2`` and NaN embeddings, and wrote them all.  The verbs
        now refuse those at the boundary (:class:`~anatid.errors.ValidationError`), but a
        boundary check only covers rows that came through the boundary.  Bulk loads
        (:meth:`load_parquet`), raw SQL through :attr:`connection`, a file written by an older
        anatid and a half-finished migration all bypass it.  This is how you find out.

        The checks, by ``check`` name:

        ``schema_drift``
            The file's ``schema_version`` is not this build's, a built-in table is missing, or a
            REQUIRED index (the ``UNIQUE (tenant_id, entity_key)`` that makes entity creation
            safe) is absent.  Every other finding below is less meaningful while this one holds.
        ``duplicate_memory_ids`` / ``duplicate_entity_ids`` / ``duplicate_episode_ids``
            Two rows sharing an id **inside one tenant** (across tenants is legal and normal).
            ``get()`` then returns an arbitrary one of them and ``supersede`` closes both.
        ``duplicate_entity_names``
            Two entity rows whose names canonicalise to the same ``entity_key`` in one tenant --
            the fracture the v3 unique index exists to prevent.  Present only in a file the
            2->3 migration has not merged.
        ``dangling_edges``
            An edge whose ``src`` or ``dst`` has no row in the table it points at (same tenant).
            2-hop recall silently loses that hop.
        ``dangling_episode_references`` (warning)
            A memory, entity or edge whose ``episode_id`` names no episode in its tenant, so
            ``provenance()`` cannot reach the evidence behind it.
        ``duplicate_live_edges`` (warning)
            Two current ABOUT edges with the same ``(src, dst)``, or two current RELATES_TO
            edges with the same ``(src, dst, rel_kind)``, in one tenant.  Nothing is lost, but
            ``about_names()`` repeats the entity and a 2-hop weight is doubled.  The verbs never
            write one; a bulk load can, and the 2->3 merge leaves a pair alone when neither edge
            was repointed by it.
        ``embedding_dimension_mismatch``
            A stored vector whose length is not the file's ``anatid_meta.embedding_dim``, or a
            column whose declared ``FLOAT[N]`` disagrees with it.
        ``non_finite_embeddings`` (``deep=True`` only)
            A vector containing NaN or +/-inf.  ``array_cosine_similarity`` returns NaN against
            such a row for *every* query, so it sorts arbitrarily inside the vector arm.
        ``confidence_out_of_range`` / ``weight_out_of_range``
            A ``confidence`` or ABOUT ``weight`` outside ``[0, 1]``.
        ``timestamp_order``
            ``valid_from > valid_to`` or ``tx_from > tx_to``: an interval that was never open,
            so no ``as_of`` query can ever return the row.
        ``stale_fts_index`` / ``orphaned_fts_documents`` / ``fts_statistics_drift``
            BM25 upkeep.  The orphan check matters beyond ranking: ``anatid_fts_documents``
            holds ``content`` **verbatim**, so a document whose ``memories`` row is gone is a
            copy of erased text still sitting in the file.

        ``all_tenants=False`` (the default) scopes every row-level check to one tenant;
        ``schema_drift`` is file-wide either way.  ``raise_on_error=True`` raises
        :class:`~anatid.errors.IntegrityError` instead of returning a report with errors in it.

        Not a snapshot: the checks are separate statements, so a concurrent writer can land
        between two of them.  ``with db.transaction(): db.doctor()`` if you need one.
        """
        started = time.perf_counter()
        ns = self.resolve_tenant(tenant)
        con = self.connection
        findings: list[DoctorFinding] = []
        ran: list[str] = []
        skipped: dict[str, str] = {}
        present = _schema.table_names(con)
        n_samples = max(0, int(samples))

        def scope(alias: str = "") -> tuple[str, list[Any]]:
            """Tenant predicate for a row-level check, or a no-op with ``all_tenants``."""
            if all_tenants:
                return "TRUE", []
            col = f"{alias}.tenant_id" if alias else "tenant_id"
            return f"{col} = ?", [ns.tenant_id]

        def probe(check: str, severity: Severity, table: str | None, detail: str,
                  sql: str, params: Sequence[Any] = ()) -> None:
            """Run one check.  ``sql`` selects the offending rows; the report gets the count.

            Always fetches at least one row: ``samples=0`` means "no examples in the report",
            not "no report" -- ``LIMIT 0`` would have made every check pass.
            """
            ran.append(check)
            rows = con.execute(f"SELECT * FROM ({sql}) LIMIT {max(1, n_samples)}",
                               list(params)).fetchall()
            if not rows:
                return
            total = _scalar_int(con.execute(f"SELECT count(*) FROM ({sql})", list(params)))
            findings.append(DoctorFinding(
                check=check, severity=severity, count=total, detail=detail, table=table,
                samples=tuple(tuple(r) for r in rows[:n_samples])))

        # The dimension the FILE records, which is what every write is checked against; the
        # handle's config adopts it on open, so they agree unless someone edited anatid_meta.
        dim = int(self.config.embedding_dim)
        if "anatid_meta" in present:
            recorded = con.execute("SELECT embedding_dim FROM anatid_meta LIMIT 1").fetchone()
            if recorded is not None and recorded[0] is not None:
                dim = int(recorded[0])

        # -- schema drift ------------------------------------------------ file-wide
        ran.append("schema_drift")
        version = _schema.current_version(con)
        drift: list[tuple[Any, ...]] = []
        if version is None:
            drift.append(("no_anatid_meta_row", None, _schema.SCHEMA_VERSION))
        elif int(version) != _schema.SCHEMA_VERSION:
            drift.append(("schema_version", int(version), _schema.SCHEMA_VERSION))
        for missing in _schema.missing_tables(con):
            drift.append(("missing_table", missing))
        declared = con.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'memories' AND column_name = 'embedding'").fetchone()
        if declared is not None and str(declared[0]).upper() != f"FLOAT[{dim}]":
            # The column type IS part of the schema: anatid_meta.embedding_dim is what every
            # write is validated against, and a column that disagrees accepts what the verbs
            # refuse.  The per-row check below then says which rows actually differ.
            drift.append(("embedding_column_type", str(declared[0]), f"FLOAT[{dim}]"))
        have_idx = {r[0] for r in con.execute(
            "SELECT index_name FROM duckdb_indexes()").fetchall()}
        for name in _schema.REQUIRED_INDEXES:
            if name not in have_idx:
                drift.append(("missing_required_index", name))
        for name in self.config.indexes:
            if name not in have_idx and name in _schema.DEFAULT_INDEXES:
                drift.append(("missing_index", name))
        if drift:
            findings.append(DoctorFinding(
                check="schema_drift", severity=Severity.ERROR, count=len(drift),
                detail=(f"file is schema v{version} against this build's "
                        f"v{_schema.SCHEMA_VERSION}, and/or a built-in table, a required index "
                        f"or the declared embedding column type does not match; re-open the "
                        f"file to run the migration ladder"),
                table=None, samples=tuple(drift[:n_samples])))

        # -- duplicate ids ----------------------------------------------- per tenant
        for table, column, check in (("memories", "memory_id", "duplicate_memory_ids"),
                                     ("entities", "entity_id", "duplicate_entity_ids"),
                                     ("episodes", "episode_id", "duplicate_episode_ids")):
            if table not in present:
                skipped[check] = f"table {table} is missing"
                continue
            where, params = scope()
            probe(check, Severity.ERROR, table,
                  f"{table}.{column} is not unique within a tenant; get() returns an arbitrary "
                  f"one of the rows and supersede/forget act on all of them",
                  f"SELECT tenant_id, {column}, count(*) AS n FROM {table} WHERE {where} "
                  f"GROUP BY 1, 2 HAVING count(*) > 1 ORDER BY 3 DESC, 2", params)

        # -- duplicate entity names -------------------------------------- per tenant
        if "entities" in present:
            where, params = scope()
            # The canonicalisation EXPRESSION, not the generated column: on a v3 file they are
            # the same value, and on a v2 file opened read_only (so never migrated) only the
            # expression exists -- and that file is exactly the one this check matters for.
            probe("duplicate_entity_names", Severity.ERROR, "entities",
                  "two entity rows canonicalise to one name in a tenant, so the graph has two "
                  "nodes for one thing; the v3 UNIQUE (tenant_id, entity_key) index prevents "
                  "new ones and the 2->3 migration merges old ones",
                  f"SELECT tenant_id, {_schema.entity_key_sql('name')} AS entity_key, "
                  f"count(*) AS n FROM entities WHERE {where} AND name IS NOT NULL "
                  f"GROUP BY 1, 2 HAVING count(*) > 1 ORDER BY 3 DESC, 2", params)

        # -- dangling edges ---------------------------------------------- per tenant
        endpoints = (
            ("edges_about", "src", "memories", "memory_id"),
            ("edges_about", "dst", "entities", "entity_id"),
            ("edges_relates", "src", "entities", "entity_id"),
            ("edges_relates", "dst", "entities", "entity_id"),
            ("edges_supersedes", "src", "memories", "memory_id"),
            ("edges_supersedes", "dst", "memories", "memory_id"),
        )
        usable = [e for e in endpoints if e[0] in present and e[2] in present]
        if not usable:
            skipped["dangling_edges"] = "no edge tables present"
        else:
            parts, params = [], []
            for edge_table, column, target, target_id in usable:
                where, p = scope("e")
                parts.append(
                    f"SELECT '{edge_table}' AS edge_table, '{column}' AS endpoint, e.edge_id, "
                    f"e.{column} AS missing_id FROM {edge_table} e WHERE {where} "
                    f"AND NOT EXISTS (SELECT 1 FROM {target} t WHERE t.{target_id} = e.{column} "
                    f"AND t.tenant_id = e.tenant_id)")
                params += p
            probe("dangling_edges", Severity.ERROR, None,
                  "an edge points at a row that does not exist in its tenant; graph expansion "
                  "silently drops that hop",
                  " UNION ALL ".join(parts) + " ORDER BY 1, 3", params)

        # -- dangling evidence -------------------------------------------- per tenant
        # A row whose episode_id names no episode in its tenant.  forget(hard=True) deletes an
        # episode only when nothing cites it, so this is raw SQL or a bulk load.  A WARNING,
        # not an ERROR: provenance() tolerates it (the chain stays, the source text is gone).
        citing = [(t, "memory_id" if t == "memories" else "entity_id" if t == "entities"
                   else "edge_id")
                  for t in ("memories", "entities", "edges_about", "edges_relates")
                  if t in present]
        if "episodes" not in present or not citing:
            skipped["dangling_episode_references"] = "episodes or every citing table is missing"
        else:
            parts, params = [], []
            for table, id_col in citing:
                where, p = scope("r")
                parts.append(
                    f"SELECT '{table}' AS \"table\", r.tenant_id, r.{id_col} AS row_id, "
                    f"r.episode_id FROM {table} r WHERE {where} AND r.episode_id IS NOT NULL "
                    f"AND NOT EXISTS (SELECT 1 FROM episodes e WHERE e.episode_id = r.episode_id "
                    f"AND e.tenant_id = r.tenant_id)")
                params += p
            probe("dangling_episode_references", Severity.WARNING, None,
                  "a row cites an episode_id that has no episodes row in its tenant, so "
                  "provenance() cannot reach the evidence it was derived from",
                  " UNION ALL ".join(parts) + " ORDER BY 1, 3", params)

        # -- duplicate live edges ---------------------------------------- per tenant
        # Two current edges saying the same thing.  Not an integrity break (both endpoints
        # exist) but it doubles a 2-hop weight and repeats an entity in about_names(), so a
        # WARNING.  Only edges live now: a closed edge next to its replacement is history.
        live = "valid_to IS NULL AND tx_to IS NULL"
        edge_keys = [("edges_about", "src, dst"), ("edges_relates", "src, dst, rel_kind")]
        edge_keys = [(t, k) for t, k in edge_keys if t in present]
        if not edge_keys:
            skipped["duplicate_live_edges"] = "no edge tables present"
        else:
            parts, params = [], []
            for table, key in edge_keys:
                where, p = scope()
                key_cols = ", ".join(f"{c.strip()}" for c in key.split(","))
                parts.append(
                    f"SELECT '{table}' AS \"table\", tenant_id, src, dst, "
                    f"{'rel_kind' if 'rel_kind' in key else 'NULL AS rel_kind'}, "
                    f"count(*) AS n FROM {table} WHERE {where} AND {live} "
                    f"GROUP BY tenant_id, {key_cols} HAVING count(*) > 1")
                params += p
            probe("duplicate_live_edges", Severity.WARNING, None,
                  "two current edges in a tenant say the same thing (same src/dst[/rel_kind]); "
                  "about_names() repeats the entity and a 2-hop weight is doubled",
                  " UNION ALL ".join(parts) + " ORDER BY 1, 6 DESC, 3", params)

        # -- embeddings --------------------------------------------------- per tenant
        if "memories" not in present:
            skipped["embedding_dimension_mismatch"] = "table memories is missing"
        else:
            where, params = scope()
            probe("embedding_dimension_mismatch", Severity.ERROR, "memories",
                  f"a stored vector is not {dim}-dimensional, so array_cosine_similarity "
                  f"cannot compare it with a query vector",
                  f"SELECT tenant_id, memory_id, len(embedding) AS dim FROM memories "
                  f"WHERE {where} AND embedding IS NOT NULL AND len(embedding) <> ? "
                  f"ORDER BY memory_id", params + [dim])

        if "memories" not in present:
            skipped["non_finite_embeddings"] = "table memories is missing"
        elif not deep:
            skipped["non_finite_embeddings"] = "deep=False (this check reads every vector)"
        else:
            where, params = scope()
            probe("non_finite_embeddings", Severity.ERROR, "memories",
                  "a stored vector contains NaN, inf or a NULL element; "
                  "array_cosine_similarity returns NaN/NULL against that row for every query, "
                  "so it ranks arbitrarily instead of ranking badly",
                  f"SELECT tenant_id, memory_id FROM memories WHERE {where} "
                  f"AND embedding IS NOT NULL "
                  f"AND len(list_filter(embedding::DOUBLE[], "
                  f"                    x -> x IS NULL OR NOT isfinite(x))) > 0 "
                  f"ORDER BY memory_id", params)

        # -- value ranges -------------------------------------------------- per tenant
        conf_tables = [t for t in ("memories", "entities", "edges_about", "edges_relates")
                       if t in present]
        if conf_tables:
            parts, params = [], []
            for table in conf_tables:
                where, p = scope()
                parts.append(f"SELECT '{table}' AS \"table\", tenant_id, confidence FROM {table} "
                             f"WHERE {where} AND confidence IS NOT NULL "
                             f"AND (confidence < 0 OR confidence > 1 OR NOT isfinite(confidence))")
                params += p
            probe("confidence_out_of_range", Severity.ERROR, None,
                  "confidence is documented and validated as a number in [0, 1]; a row outside "
                  "it came from a bulk load or raw SQL and will skew any caller that weights by "
                  "it", " UNION ALL ".join(parts) + " ORDER BY 1", params)

        if "edges_about" in present:
            where, params = scope()
            probe("weight_out_of_range", Severity.ERROR, "edges_about",
                  "an ABOUT edge weight is outside [0, 1]",
                  f"SELECT tenant_id, edge_id, weight FROM edges_about WHERE {where} "
                  f"AND weight IS NOT NULL AND (weight < 0 OR weight > 1 OR NOT isfinite(weight))"
                  f" ORDER BY edge_id", params)

        # -- timestamp ordering -------------------------------------------- per tenant
        temporal = [t for t in ("memories", "entities", "episodes", "edges_about",
                                "edges_relates") if t in present]
        if temporal:
            parts, params = [], []
            for table in temporal:
                where, p = scope()
                parts.append(
                    f"SELECT '{table}' AS \"table\", tenant_id, valid_from, valid_to, tx_from, "
                    f"tx_to FROM {table} WHERE {where} AND ((valid_to IS NOT NULL AND "
                    f"valid_from IS NOT NULL AND valid_to < valid_from) OR (tx_to IS NOT NULL "
                    f"AND tx_from IS NOT NULL AND tx_to < tx_from))")
                params += p
            probe("timestamp_order", Severity.ERROR, None,
                  "a row's interval closes before it opens ([from, to) with to < from), so no "
                  "as_of query can ever return it",
                  " UNION ALL ".join(parts) + " ORDER BY 1", params)

        # -- full-text upkeep ----------------------------------------------- file-wide
        try:
            status = _recall.fts_status(con)
        except Exception as exc:                      # pragma: no cover - fts not installed
            skipped["stale_fts_index"] = f"fts status unavailable: {exc}"
            status = None
        if status is not None:
            ran.append("stale_fts_index")
            if status.available and status.stale:
                findings.append(DoctorFinding(
                    check="stale_fts_index", severity=Severity.WARNING,
                    count=int(status.pending_rows),
                    detail=("rows have been written since the last rebuild_fts_index(); "
                            "DuckDB's fts index is not incremental, so BM25 cannot see them"),
                    table=_schema.FTS_SOURCE_TABLE,
                    samples=((status.indexed_rows, status.current_rows, status.indexed_at),)))
            elif not status.available and status.current_rows:
                findings.append(DoctorFinding(
                    check="stale_fts_index", severity=Severity.WARNING,
                    count=int(status.current_rows),
                    detail=("no BM25 index has been built, so recall() runs without its text "
                            "arm; call rebuild_fts_index()"),
                    table=_schema.FTS_SOURCE_TABLE, samples=()))

        if _schema.FTS_SOURCE_TABLE in present and "memories" in present:
            where, params = scope("d")
            probe("orphaned_fts_documents", Severity.ERROR, _schema.FTS_SOURCE_TABLE,
                  "a BM25 document has no memories row, and anatid_fts_documents stores content "
                  "VERBATIM -- this is a copy of deleted text still in the file and still "
                  "findable by recall()",
                  f"SELECT d.tenant_id, d.memory_id FROM {_schema.FTS_SOURCE_TABLE} d "
                  f"WHERE {where} AND NOT EXISTS (SELECT 1 FROM memories m "
                  f"WHERE m.memory_id = d.memory_id AND m.tenant_id = d.tenant_id) "
                  f"ORDER BY d.tenant_id, d.memory_id", params)

        if _schema.FTS_DICT_TABLE in present and _schema.FTS_STATS_TABLE in present:
            where, params = scope("d")
            probe("fts_statistics_drift", Severity.WARNING, _schema.FTS_DICT_TABLE,
                  "a tenant has per-term document frequencies but no (num_docs, avgdl) row, so "
                  "its BM25 scores cannot be computed; rebuild_fts_index()",
                  f"SELECT DISTINCT d.tenant_id FROM {_schema.FTS_DICT_TABLE} d WHERE {where} "
                  f"AND NOT EXISTS (SELECT 1 FROM {_schema.FTS_STATS_TABLE} s "
                  f"WHERE s.tenant_id = d.tenant_id) ORDER BY 1", params)

        # -- counts, for context -------------------------------------------------------
        counts: dict[str, int] = {}
        for table in _schema.ALL_TABLES:
            if table not in present or table == "anatid_meta":
                continue
            has_tenant = "tenant_id" in {
                r[1] for r in con.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()}
            if all_tenants or not has_tenant:
                counts[table] = _scalar_int(con.execute(
                    f"SELECT count(*) FROM {quote_ident(table)}"))
            else:
                counts[table] = _scalar_int(con.execute(
                    f"SELECT count(*) FROM {quote_ident(table)} WHERE tenant_id = ?",
                    [ns.tenant_id]))

        report = DoctorReport(
            checked_at=utcnow(), schema_version=version,
            expected_schema_version=_schema.SCHEMA_VERSION,
            tenant_id=None if all_tenants else ns.tenant_id, all_tenants=all_tenants,
            findings=tuple(findings), counts=counts, checks_run=tuple(ran),
            checks_skipped=skipped,
            duration_ms=round((time.perf_counter() - started) * 1000, 3))
        if raise_on_error and not report.ok:
            raise IntegrityError(
                f"doctor() found {len(report.errors)} integrity fault(s) in {self.path!r}: "
                + ", ".join(f"{f.check}={f.count}" for f in report.errors),
                report=report, findings=report.errors)
        return report

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
            before = _scalar_int(con.execute(f"SELECT count(*) FROM {quote_ident(name)}"))
            self.execute(
                f"INSERT INTO {quote_ident(name)} ({', '.join(quote_ident(c) for c in shared)}) "
                f"SELECT {proj} FROM read_parquet('{src}'){order_sql}")
            after = _scalar_int(con.execute(f"SELECT count(*) FROM {quote_ident(name)}"))
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
