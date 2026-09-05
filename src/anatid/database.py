"""Database handles: :class:`Anatid` (one DuckDB file) and :class:`DatabasePool` (file-per-tenant).

See :class:`Anatid`'s docstring for the isolation, transaction and time-travel contract.  It is
the same text stored in the file's ``anatid_meta.contract`` column.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import logging
import os
import shutil
import sys
import threading
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import duckdb

from . import atomic as _atomic
from . import recall as _recall
from . import schema as _schema
from .csr import CsrBackend
from .derived import (
    HealthReason,
    HealthReport,
    IndexRegistry,
    MaintenancePolicy,
    MaintenanceReport,
)
from .embed import Embedder
from .errors import (
    AnatidError,
    BackupDestinationExists,
    ConflictError,
    EmbeddingDimensionError,
    ExtensionUnavailable,
    IntegrityError,
    NotFoundError,
    TenantIsolationError,
)
from .schema import SchemaConfig, quote_ident, quote_name
from .visibility import current_row_sql, live_row_sql, tenant_sql
from .types import (
    DOCTOR_SAMPLE_LIMIT,
    AsOf,
    DoctorFinding,
    DoctorReport,
    Edge,
    Entity,
    FtsStatus,
    Isolation,
    Memory,
    Namespace,
    SchemaInfo,
    Severity,
    to_utc_naive,
    utcnow,
)
from .verbs import MemoryVerbs

log = logging.getLogger("anatid")

__all__ = ["Anatid", "DatabasePool", "PoolEvent", "connect"]

__version__ = "0.4.0"

#: Substrings DuckDB uses for an MVCC abort.  Deliberately narrow, and it has to stay that way:
#: :class:`~anatid.errors.ConflictError` promises the caller that retrying the unit of work is
#: the right response, so only errors for which that is TRUE may be translated.
#:
#: In particular ``duckdb.TransactionException`` and the bare string "transactioncontext error"
#: are NOT sufficient: DuckDB raises the same family for deterministic programming errors --
#: "cannot start a transaction within a transaction", "cannot commit - no transaction is active",
#: "cannot rollback - no transaction is active" -- which fail identically on every retry.  A
#: retry loop keyed on ``ConflictError.retryable`` would spin forever on those.
_CONFLICT_MARKERS = (
    "conflict on update",
    "conflict on delete",
    "conflict on tuple",
    "transaction is aborted",
    "could not serialize",
    # A derived index's generation storage retired under a transaction that was
    # writing to it (a hard erasure racing a rebuild).  DuckDB names the other
    # transaction, so this is a race and not a deterministic misuse: retrying
    # finds the table gone and succeeds.  The sibling message about a table
    # another transaction ALTERED is deliberately NOT here: that one is the
    # deterministic "modified rows then ran DDL in one transaction" mistake.
    "has dropped this table",
)

#: Deterministic ``TransactionContext`` errors that must keep their own type.  Checked first.
_NOT_CONFLICT_MARKERS = (
    "no transaction is active",
    "transaction within a transaction",
    "transaction is launched in read-only mode",
    "cannot write to database",
)


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
    (:mod:`anatid.visibility`).  Intervals are half-open.  Rows of ``memories``,
    ``edges_about`` and ``edges_relates`` are immutable versions (schema v4): a correction
    closes the current version on the transaction axis and inserts the next one, so the
    ``tx_time`` half of an ``as_of`` returns what the database believed then, not what it
    believes now.  A hard purge (``forget(hard=True)``) removes every version from history
    too -- erasure beats auditability by design, and the receipt is handed back to the caller
    to log elsewhere.

    Full-text contract
    ------------------
    The DuckDB ``fts`` index is not incremental, so :mod:`anatid.fts` runs the BM25 arm as a
    derived index, which :meth:`open` attaches by default.  A published base generation carries
    the corpus and an ordered journal carries every document written or closed since, written
    inside the same transaction as the memory, so **a write is searchable by the very next**
    :meth:`recall` **with nothing rebuilt**: on this handle, on any other handle on the file,
    and for a ``supersede`` or a ``forget`` as well as an insert.
    :meth:`rebuild_fts_index` compacts the journal into a new generation.  That buys read
    latency, because a search rescans the journalled documents and the rescan is linear in the
    journal, and it is never required for a write to be found.  Nothing is rebuilt implicitly:
    a rebuild is O(corpus) and belongs to your write path rather than to an unlucky read.
    ``hits.pending_fts_rows`` counts the documents a search rescanned, all of them searched;
    ``hits.bm25_stale`` means the arm could not answer exactly, which on this path takes both no
    usable generation and a corpus above :data:`anatid.fts.SCAN_CEILING`.  A handle opened with
    ``accelerators=False`` keeps 0.1.1's single file-wide index, and there ``bm25_stale`` has
    its old meaning: rows written since the last :meth:`rebuild_fts_index` are invisible to the
    arm until the next one.

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
        #: The embedding model :meth:`open` was given, or None (:mod:`anatid.embed`).  With one
        #: set, ``remember`` and ``supersede`` embed content they were not handed an embedding
        #: for, and ``recall`` embeds the query so the vector arm runs.  Nothing here calls it.
        self.embedder: Embedder | None = None
        #: The derived indexes (:mod:`anatid.derived`).  The DEFINITIONS live in the file, so
        #: the verbs journal every write for every index the file defines whether or not this
        #: handle holds that accelerator's code; this object holds the implementations this
        #: handle does have, plus a placeholder per accelerator name.
        self.indexes = IndexRegistry(self)

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
        accelerators: bool = True,
        vector_backend: str = "exact",
        embedder: Embedder | None = None,
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
        ``accelerators``
            Attach the derived-index accelerators that cost nothing to have: the full-text index
            (:mod:`anatid.fts`) and the graph CSR (:mod:`anatid.csr`).  Attaching registers each
            one's DEFINITION in the file, which is what makes every handle journal writes for it
            inside the writing transaction, and it is what ``maintain_indexes()`` builds from.
            Nothing is built at open time and no generation is published, so a fresh database
            answers text search by scanning it exactly and expands the graph in SQL, as 0.1.1
            did.  The cost of having them is one journal row per write per index that derives
            from the table written (0.68 ms per ``remember`` for full text on this machine).
            Set False for a write-heavy database that never searches text.
        ``vector_backend``
            ``"exact"`` (the default) is the brute-force cosine scan, which is also the oracle
            every other backend is measured against.  ``"duckdb_vss"`` attaches an HNSW
            accelerator (:mod:`anatid.vector`).  It is opt in because DuckDB documents HNSW
            persistence as experimental, and because it only pays above roughly 15,000 rows per
            tenant; below that the scan is faster.  Build it with ``maintain_indexes()``.
        ``embedder``
            An :class:`~anatid.embed.Embedder` (``embed(texts)``, ``embed_one(text)``, ``dim``).
            Stored on the handle as :attr:`embedder`.  With one set, ``remember()`` and
            ``supersede()`` embed content they were not given an embedding for and ``recall()``
            embeds the query, so the vector arm runs without the caller producing vectors.  An
            embedding passed explicitly always wins.  Its ``dim`` must equal the database's
            ``embedding_dim``; a mismatch raises :class:`~anatid.errors.EmbeddingDimensionError`
            here rather than at the first write.  Without one nothing changes: anatid never calls
            a model on its own.
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

        con = (
            duckdb.connect(p, read_only=read_only, config=cfg_kwargs)
            if cfg_kwargs
            else duckdb.connect(p, read_only=read_only)
        )

        if fts:
            try:
                con.execute("INSTALL fts")
                con.execute("LOAD fts")
            except duckdb.Error as exc:  # offline / no extension repository
                log.warning("fts extension unavailable (%s); BM25 recall will be disabled", exc)

        backend = CsrBackend(
            enabled=use_csr_extension or require_extension,
            extension_path=extension_path,
            require=require_extension,
        )
        if backend.enabled:
            backend.load(con)

        existing_dim = _schema.embedding_dim(con)
        if existing_dim is not None and int(existing_dim) != int(embedding_dim):
            log.info(
                "using embedding_dim=%s recorded in %s (open() was given %s)",
                existing_dim,
                p,
                embedding_dim,
            )
            embedding_dim = int(existing_dim)

        idx = tuple(indexes) if indexes is not None else tuple(_schema.DEFAULT_INDEXES)
        cfg = SchemaConfig(embedding_dim=embedding_dim, indexes=idx)

        if embedder is not None:
            edim = getattr(embedder, "dim", None)
            if edim is not None and int(edim) != int(embedding_dim):
                con.close()
                raise EmbeddingDimensionError(
                    f"embedder {embedder!r} produces {int(edim)}-dimensional vectors, database "
                    f"{p!r} is FLOAT[{int(embedding_dim)}]",
                    expected=int(embedding_dim),
                    got=int(edim),
                )

        db = cls(con, path=p, namespace=ns, config=cfg, backend=backend, read_only=read_only)
        db.embedder = embedder
        if ensure and not read_only:
            db.ensure_schema()
        db._attach_accelerators(
            accelerators=accelerators, fts=fts, vector_backend=vector_backend
        )
        return db

    def _attach_accelerators(
        self, *, accelerators: bool, fts: bool, vector_backend: str
    ) -> None:
        """Register the derived indexes this handle should hold.  Called by :meth:`open`.

        Registering is two things: an object on this handle that can build and read a
        generation, and a row in the file's ``anatid_index_registry`` that makes every OTHER
        handle journal writes for the same index.  The second is why this runs on a handle that
        holds no schema yet as well: the definitions are what a later ``maintain_indexes()``
        builds from, and a journal with a hole in it is worse than no index at all.

        A read-only handle attaches too, because a definition already in the file decides how a
        READ is answered and this handle has to make the same decision the writer does; the
        registry's own write is a no-op there.  A failure to attach is logged and swallowed:
        every accelerator has an exact fallback, so an index that could not be set up must not
        stop the database from opening.

        Nothing is attached to a file that has no derived-index catalog: a pre-v4 file this
        handle could not migrate (``read_only`` or ``ensure=False``) has neither the registry
        nor the journal, so an accelerator there could not record a write even in principle,
        and half-attaching one would put an index on the handle whose every catalog read
        raises.  Those files read through the SQL path, which is what they did in 0.1.1.
        """
        try:
            present = _schema.table_names(self.connection)
        except duckdb.Error as exc:  # pragma: no cover - a handle whose connection is gone
            log.warning("accelerators not attached: %s", exc)
            return
        if not set(_schema.INDEX_TABLES) <= present:
            log.info(
                "%s has no derived-index catalog (schema v%d); reads take the SQL path",
                self.path,
                _schema.current_version(self.connection) or 0,
            )
            return
        if self.config.embedding_dim and vector_backend and vector_backend != "exact":
            try:
                from .vector import attach as _attach_vector

                _attach_vector(self, backend=vector_backend)
            except Exception as exc:  # noqa: BLE001 - the exact scan still answers
                log.warning("vector backend %r could not be attached: %s", vector_backend, exc)
        if not accelerators:
            return
        try:
            from .csr import attach_csr_index

            attach_csr_index(self, register=not self.read_only)
        except Exception as exc:  # noqa: BLE001 - the SQL expansion still answers
            log.warning("csr index could not be attached: %s", exc)
        if not fts:
            return
        try:
            from .fts import attach as _attach_fts

            _attach_fts(self)
        except Exception as exc:  # noqa: BLE001 - the exact scan still answers
            log.warning("full-text index could not be attached: %s", exc)

    def checkpoint(self) -> None:
        """Fold this file's write-ahead log into the file itself.

        It runs on the handle's ROOT connection, and that is the whole point of the method
        existing rather than callers writing ``db.execute("CHECKPOINT")``.  Measured on duckdb
        1.5.5: through :meth:`execute`, which uses this thread's cursor, ``CHECKPOINT`` succeeds
        on a handle for a file this process CREATED and raises ``TransactionException: Cannot
        CHECKPOINT: there are other write transactions active`` on a handle for a file that
        already existed, from any thread, on a handle that has run nothing else.  From the root
        connection it succeeds in both cases, in any order, and folds the log (measured: a
        2,146,504 byte ``.wal`` to 0).  ``FORCE CHECKPOINT``, which DuckDB's error message
        suggests, is worse than either: from a cursor it does not raise, it never returns.

        It does not force.  Another thread with a write transaction genuinely open is a reason to
        leave the log alone, and DuckDB raises ``TransactionException`` for it; that is a real
        answer and this method passes it through rather than blocking behind it.  Measured under
        four threads writing continuously: some calls checkpointed and some raised, and none
        hung.

        A read-only handle has nothing to fold, so this is a no-op there.  Closing a handle also
        folds its log, and closing additionally gives up DuckDB's exclusive lock on the file,
        which is what a process has to do before another process can open it at all.
        """
        if self._closed:
            raise AnatidError(f"database {self.path!r} is closed")
        if self.read_only:
            return
        self._root.execute("CHECKPOINT")

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
        return (
            f"<Anatid path={self.path!r} tenant={self.namespace.tenant_id} "
            f"isolation={self.namespace.isolation.value} dim={self.config.embedding_dim} "
            f"expand={self.csr.active}{' closed' if self._closed else ''}>"
        )

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

        This is anatid's own seam: the verbs and :mod:`anatid.recall` reach the cursor through
        it.  Administrative access from outside the package has a separate, honest name,
        :meth:`unsafe_connection`, and on a handle a :class:`DatabasePool` opened this property
        is guarded to say so (see the pool's ``raw_access`` argument).
        """
        if self._closed:
            raise AnatidError(f"database {self.path!r} is closed")
        if self._raw_guard is not None:
            self._check_raw_access()
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
                f"Open that tenant's own file, or attach it read-only."
            )
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
            return _schema.ensure_schema(self.connection, self.config, anatid_version=__version__)

    def info(self) -> SchemaInfo:
        """The ``anatid_meta`` catalog row, including the stored contract notes."""
        row = self.execute(
            "SELECT schema_version, created_at, embedding_dim, anatid_version, duckdb_version,"
            " fts_indexed_at, fts_indexed_rows, contract FROM anatid_meta LIMIT 1"
        ).fetchone()
        if row is None:
            raise AnatidError(f"{self.path!r} has no anatid_meta row; call ensure_schema()")
        return SchemaInfo(
            schema_version=int(row[0]),
            created_at=row[1],
            embedding_dim=int(row[2]),
            anatid_version=row[3],
            duckdb_version=row[4],
            fts_indexed_at=row[5],
            fts_indexed_rows=None if row[6] is None else int(row[6]),
            contract=row[7],
            extras={
                "path": self.path,
                "tenant": self.namespace.tenant_id,
                "isolation": self.namespace.isolation.value,
                "expand_path": self.csr.active,
            },
        )

    def create_node_label(
        self,
        label: str,
        properties: Sequence[tuple[str, str]] = (),
        *,
        id_column: str | None = None,
        system_columns: bool = True,
    ) -> str:
        """Create a table for a user-defined node label.  System columns are on by default."""
        ddl = _schema.node_table_ddl(
            label, properties, id_column=id_column, system_columns=system_columns
        )
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
        ddl = _schema.edge_table_ddl(
            edge_type, properties, table=table, system_columns=system_columns
        )
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
            sql = _schema.DEFAULT_INDEXES.get(index_name) or _schema.OPTIONAL_INDEXES.get(
                index_name
            )
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
                self.execute(f"INSERT INTO {tmp} ({cols}) SELECT {cols} FROM {t} ORDER BY {order}")
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
            except duckdb.Error as exc:  # index on a table we did not rebuild
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
            ``memories`` is versioned, so its check is two live versions of one id
            (``tx_to IS NULL`` twice) or two rows with one version number; the closed
            versions a correction leaves behind are the design, not a duplicate.
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
            BM25 upkeep for 0.1.1's non-incremental file-wide index.  The orphan check matters
            beyond ranking: ``anatid_fts_documents`` holds ``content`` **verbatim**, so a
            document whose ``memories`` row is gone is a copy of erased text still sitting in
            the file.
        ``unusable_derived_index``
            A derived index (:mod:`anatid.derived`) has a published generation that reads
            cannot use: invalidated, damaged, or its storage will not load.  Answers stay
            correct because the SQL path is the oracle; the sample names the index and the
            :class:`~anatid.derived.HealthReason`.  ``maintain_indexes()`` rebuilds.

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
            return tenant_sql(alias or None), [ns.tenant_id]

        def probe(
            check: str,
            severity: Severity,
            table: str | None,
            detail: str,
            sql: str,
            params: Sequence[Any] = (),
        ) -> None:
            """Run one check.  ``sql`` selects the offending rows; the report gets the count.

            Always fetches at least one row: ``samples=0`` means "no examples in the report",
            not "no report" -- ``LIMIT 0`` would have made every check pass.
            """
            ran.append(check)
            rows = con.execute(
                f"SELECT * FROM ({sql}) LIMIT {max(1, n_samples)}", list(params)
            ).fetchall()
            if not rows:
                return
            total = _scalar_int(con.execute(f"SELECT count(*) FROM ({sql})", list(params)))
            findings.append(
                DoctorFinding(
                    check=check,
                    severity=severity,
                    count=total,
                    detail=detail,
                    table=table,
                    samples=tuple(tuple(r) for r in rows[:n_samples]),
                )
            )

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
            "WHERE table_name = 'memories' AND column_name = 'embedding'"
        ).fetchone()
        if declared is not None and str(declared[0]).upper() != f"FLOAT[{dim}]":
            # The column type IS part of the schema: anatid_meta.embedding_dim is what every
            # write is validated against, and a column that disagrees accepts what the verbs
            # refuse.  The per-row check below then says which rows actually differ.
            drift.append(("embedding_column_type", str(declared[0]), f"FLOAT[{dim}]"))
        have_idx = {r[0] for r in con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
        for name in _schema.REQUIRED_INDEXES:
            if name not in have_idx:
                drift.append(("missing_required_index", name))
        for name in self.config.indexes:
            if name not in have_idx and name in _schema.DEFAULT_INDEXES:
                drift.append(("missing_index", name))
        # Last, so the samples an older file's report leads with (version, tables, indexes)
        # keep their places within the sample cap.
        versioned: set[str] = set()
        for table in _schema.VERSIONED_TABLES:
            if table not in present:
                continue
            table_cols = {
                r[1] for r in con.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
            }
            if _schema.VERSION_COLUMN in table_cols:
                versioned.add(table)
            else:
                # A v4 file written by a build that predates the version column, or a v2/v3
                # file opened read-only: the verbs cannot version its rows.  ensure_schema()
                # adds the column on the next writable open.
                drift.append(("missing_column", table, _schema.VERSION_COLUMN))
        if _schema.INDEX_GENERATIONS_TABLE in present:
            # Same story for the derived-index catalog: a v4 file written before the column
            # existed cannot tell a force-published generation from an invalidated one, so the
            # framework treats both as unusable until the next writable open repairs it.
            gen_cols = {
                r[1]
                for r in con.execute(
                    f"PRAGMA table_info({quote_ident(_schema.INDEX_GENERATIONS_TABLE)})"
                ).fetchall()
            }
            for name, _decl in _schema.INDEX_GENERATION_ADDED_COLUMNS:
                if name not in gen_cols:
                    drift.append(("missing_column", _schema.INDEX_GENERATIONS_TABLE, name))
        if drift:
            findings.append(
                DoctorFinding(
                    check="schema_drift",
                    severity=Severity.ERROR,
                    count=len(drift),
                    detail=(
                        f"file is schema v{version} against this build's "
                        f"v{_schema.SCHEMA_VERSION}, and/or a built-in table, a required index "
                        f"or the declared embedding column type does not match; re-open the "
                        f"file to run the migration ladder"
                    ),
                    table=None,
                    samples=tuple(drift[:n_samples]),
                )
            )

        # -- duplicate ids ----------------------------------------------- per tenant
        for table, column, check in (
            ("memories", "memory_id", "duplicate_memory_ids"),
            ("entities", "entity_id", "duplicate_entity_ids"),
            ("episodes", "episode_id", "duplicate_episode_ids"),
        ):
            if table not in present:
                skipped[check] = f"table {table} is missing"
                continue
            where, params = scope()
            if table in versioned:
                # Versions of one id are expected; two LIVE versions, or two rows carrying
                # the same version number, are not.  (A table without the column is checked
                # the plain way below; schema_drift already names the missing column.)
                live = live_row_sql()
                version_expr = f"coalesce({_schema.VERSION_COLUMN}, 1)"
                probe(
                    check,
                    Severity.ERROR,
                    table,
                    f"{table}.{column} has two live versions (tx_to IS NULL twice) or two "
                    f"rows with one version number within a tenant; get() returns an "
                    f"arbitrary one of the rows and supersede/forget act on all of them",
                    f"SELECT tenant_id, {column}, count(*) FILTER (WHERE {live}) AS live_rows, "
                    f"count(*) AS n FROM {table} WHERE {where} GROUP BY 1, 2 "
                    f"HAVING count(*) FILTER (WHERE {live}) > 1 "
                    f"OR count(*) > count(DISTINCT {version_expr}) ORDER BY 3 DESC, 4 DESC, 2",
                    params,
                )
                continue
            probe(
                check,
                Severity.ERROR,
                table,
                f"{table}.{column} is not unique within a tenant; get() returns an arbitrary "
                f"one of the rows and supersede/forget act on all of them",
                f"SELECT tenant_id, {column}, count(*) AS n FROM {table} WHERE {where} "
                f"GROUP BY 1, 2 HAVING count(*) > 1 ORDER BY 3 DESC, 2",
                params,
            )

        # -- duplicate entity names -------------------------------------- per tenant
        if "entities" in present:
            where, params = scope()
            # The canonicalisation EXPRESSION, not the generated column: on a v3 file they are
            # the same value, and on a v2 file opened read_only (so never migrated) only the
            # expression exists -- and that file is exactly the one this check matters for.
            probe(
                "duplicate_entity_names",
                Severity.ERROR,
                "entities",
                "two entity rows canonicalise to one name in a tenant, so the graph has two "
                "nodes for one thing; the v3 UNIQUE (tenant_id, entity_key) index prevents "
                "new ones and the 2->3 migration merges old ones",
                f"SELECT tenant_id, {_schema.entity_key_sql('name')} AS entity_key, "
                f"count(*) AS n FROM entities WHERE {where} AND name IS NOT NULL "
                f"GROUP BY 1, 2 HAVING count(*) > 1 ORDER BY 3 DESC, 2",
                params,
            )

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
                    f"AND t.tenant_id = e.tenant_id)"
                )
                params += p
            probe(
                "dangling_edges",
                Severity.ERROR,
                None,
                "an edge points at a row that does not exist in its tenant; graph expansion "
                "silently drops that hop",
                " UNION ALL ".join(parts) + " ORDER BY 1, 3",
                params,
            )

        # -- dangling evidence -------------------------------------------- per tenant
        # A row whose episode_id names no episode in its tenant.  forget(hard=True) deletes an
        # episode only when nothing cites it, so this is raw SQL or a bulk load.  A WARNING,
        # not an ERROR: provenance() tolerates it (the chain stays, the source text is gone).
        citing = [
            (t, "memory_id" if t == "memories" else "entity_id" if t == "entities" else "edge_id")
            for t in ("memories", "entities", "edges_about", "edges_relates")
            if t in present
        ]
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
                    f"AND e.tenant_id = r.tenant_id)"
                )
                params += p
            probe(
                "dangling_episode_references",
                Severity.WARNING,
                None,
                "a row cites an episode_id that has no episodes row in its tenant, so "
                "provenance() cannot reach the evidence it was derived from",
                " UNION ALL ".join(parts) + " ORDER BY 1, 3",
                params,
            )

        # -- duplicate live edges ---------------------------------------- per tenant
        # Two current edges saying the same thing.  Not an integrity break (both endpoints
        # exist) but it doubles a 2-hop weight and repeats an entity in about_names(), so a
        # WARNING.  Only edges live now: a closed edge next to its replacement is history.
        live = current_row_sql()
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
                    f"GROUP BY tenant_id, {key_cols} HAVING count(*) > 1"
                )
                params += p
            probe(
                "duplicate_live_edges",
                Severity.WARNING,
                None,
                "two current edges in a tenant say the same thing (same src/dst[/rel_kind]); "
                "about_names() repeats the entity and a 2-hop weight is doubled",
                " UNION ALL ".join(parts) + " ORDER BY 1, 6 DESC, 3",
                params,
            )

        # -- embeddings --------------------------------------------------- per tenant
        if "memories" not in present:
            skipped["embedding_dimension_mismatch"] = "table memories is missing"
        else:
            where, params = scope()
            probe(
                "embedding_dimension_mismatch",
                Severity.ERROR,
                "memories",
                f"a stored vector is not {dim}-dimensional, so array_cosine_similarity "
                f"cannot compare it with a query vector",
                f"SELECT tenant_id, memory_id, len(embedding) AS dim FROM memories "
                f"WHERE {where} AND embedding IS NOT NULL AND len(embedding) <> ? "
                f"ORDER BY memory_id",
                params + [dim],
            )

        if "memories" not in present:
            skipped["non_finite_embeddings"] = "table memories is missing"
        elif not deep:
            skipped["non_finite_embeddings"] = "deep=False (this check reads every vector)"
        else:
            where, params = scope()
            probe(
                "non_finite_embeddings",
                Severity.ERROR,
                "memories",
                "a stored vector contains NaN, inf or a NULL element; "
                "array_cosine_similarity returns NaN/NULL against that row for every query, "
                "so it ranks arbitrarily instead of ranking badly",
                f"SELECT tenant_id, memory_id FROM memories WHERE {where} "
                f"AND embedding IS NOT NULL "
                f"AND len(list_filter(embedding::DOUBLE[], "
                f"                    x -> x IS NULL OR NOT isfinite(x))) > 0 "
                f"ORDER BY memory_id",
                params,
            )

        # -- value ranges -------------------------------------------------- per tenant
        conf_tables = [
            t for t in ("memories", "entities", "edges_about", "edges_relates") if t in present
        ]
        if conf_tables:
            parts, params = [], []
            for table in conf_tables:
                where, p = scope()
                parts.append(
                    f"SELECT '{table}' AS \"table\", tenant_id, confidence FROM {table} "
                    f"WHERE {where} AND confidence IS NOT NULL "
                    f"AND (confidence < 0 OR confidence > 1 OR NOT isfinite(confidence))"
                )
                params += p
            probe(
                "confidence_out_of_range",
                Severity.ERROR,
                None,
                "confidence is documented and validated as a number in [0, 1]; a row outside "
                "it came from a bulk load or raw SQL and will skew any caller that weights by "
                "it",
                " UNION ALL ".join(parts) + " ORDER BY 1",
                params,
            )

        if "edges_about" in present:
            where, params = scope()
            probe(
                "weight_out_of_range",
                Severity.ERROR,
                "edges_about",
                "an ABOUT edge weight is outside [0, 1]",
                f"SELECT tenant_id, edge_id, weight FROM edges_about WHERE {where} "
                f"AND weight IS NOT NULL AND (weight < 0 OR weight > 1 OR NOT isfinite(weight))"
                f" ORDER BY edge_id",
                params,
            )

        # -- timestamp ordering -------------------------------------------- per tenant
        temporal = [
            t
            for t in ("memories", "entities", "episodes", "edges_about", "edges_relates")
            if t in present
        ]
        if temporal:
            parts, params = [], []
            for table in temporal:
                where, p = scope()
                parts.append(
                    f"SELECT '{table}' AS \"table\", tenant_id, valid_from, valid_to, tx_from, "
                    f"tx_to FROM {table} WHERE {where} AND ((valid_to IS NOT NULL AND "
                    f"valid_from IS NOT NULL AND valid_to < valid_from) OR (tx_to IS NOT NULL "
                    f"AND tx_from IS NOT NULL AND tx_to < tx_from))"
                )
                params += p
            probe(
                "timestamp_order",
                Severity.ERROR,
                None,
                "a row's interval closes before it opens ([from, to) with to < from), so no "
                "as_of query can ever return it",
                " UNION ALL ".join(parts) + " ORDER BY 1",
                params,
            )

        # -- full-text upkeep ----------------------------------------------- file-wide
        from .fts import staleness_message

        try:
            status = _recall.fts_status(con)
        except Exception as exc:  # pragma: no cover - fts not installed
            skipped["stale_fts_index"] = f"fts status unavailable: {exc}"
            status = None
        if status is not None:
            ran.append("stale_fts_index")
            if status.available and status.stale:
                findings.append(
                    DoctorFinding(
                        check="stale_fts_index",
                        severity=Severity.WARNING,
                        count=int(status.pending_rows),
                        # What "stale" means depends on which half of anatid.fts answered, so
                        # the sentence comes from there rather than being hardcoded here: on
                        # 0.1.1's index it is rows the arm cannot see, on the derived index it
                        # is that no generation was usable and the corpus was too big to scan.
                        detail=staleness_message(status),
                        table=_schema.FTS_SOURCE_TABLE,
                        samples=((status.indexed_rows, status.current_rows, status.indexed_at),),
                    )
                )
            elif not status.available and status.current_rows:
                findings.append(
                    DoctorFinding(
                        check="stale_fts_index",
                        severity=Severity.WARNING,
                        count=int(status.current_rows),
                        detail=(
                            "no BM25 index has been built, so recall() runs without its text "
                            "arm; call rebuild_fts_index()"
                        ),
                        table=_schema.FTS_SOURCE_TABLE,
                        samples=(),
                    )
                )

        if _schema.FTS_SOURCE_TABLE in present and "memories" in present:
            where, params = scope("d")
            probe(
                "orphaned_fts_documents",
                Severity.ERROR,
                _schema.FTS_SOURCE_TABLE,
                "a BM25 document has no memories row, and anatid_fts_documents stores content "
                "VERBATIM -- this is a copy of deleted text still in the file and still "
                "findable by recall()",
                f"SELECT d.tenant_id, d.memory_id FROM {_schema.FTS_SOURCE_TABLE} d "
                f"WHERE {where} AND NOT EXISTS (SELECT 1 FROM memories m "
                f"WHERE m.memory_id = d.memory_id AND m.tenant_id = d.tenant_id) "
                f"ORDER BY d.tenant_id, d.memory_id",
                params,
            )

        if _schema.FTS_DICT_TABLE in present and _schema.FTS_STATS_TABLE in present:
            where, params = scope("d")
            probe(
                "fts_statistics_drift",
                Severity.WARNING,
                _schema.FTS_DICT_TABLE,
                "a tenant has per-term document frequencies but no (num_docs, avgdl) row, so "
                "its BM25 scores cannot be computed; rebuild_fts_index()",
                f"SELECT DISTINCT d.tenant_id FROM {_schema.FTS_DICT_TABLE} d WHERE {where} "
                f"AND NOT EXISTS (SELECT 1 FROM {_schema.FTS_STATS_TABLE} s "
                f"WHERE s.tenant_id = d.tenant_id) ORDER BY 1",
                params,
            )

        # -- derived indexes ------------------------------------------------- per tenant
        # The framework's own upkeep signal.  `stale_fts_index` above is about 0.1.1's
        # non-incremental index and is silent on a database that has moved off it, because
        # nothing there is invisible; what an operator needs instead is which generation cannot
        # be used and why.  A usable-but-due generation is not reported: the read merges the
        # journal, so it is a performance note, not an integrity one.
        if set(_schema.INDEX_TABLES) <= present:
            try:
                reports = self.indexes.health(ns.tenant_id)
            except Exception as exc:  # noqa: BLE001 - a diagnostic must not be the thing that fails
                skipped["unusable_derived_index"] = f"index health unavailable: {exc}"
                reports = {}
            else:
                ran.append("unusable_derived_index")
            unusable = [
                (name, report.reason.value, report.detail)
                for name, report in sorted(reports.items())
                if not report.usable and report.reason is not HealthReason.ABSENT
            ]
            if unusable:
                findings.append(
                    DoctorFinding(
                        check="unusable_derived_index",
                        severity=Severity.WARNING,
                        count=len(unusable),
                        detail=(
                            "a derived index cannot serve reads, so they fall back to the SQL "
                            "path: correct, slower, and the reason is per index below; "
                            "maintain_indexes() rebuilds"
                        ),
                        table=_schema.INDEX_GENERATIONS_TABLE,
                        samples=tuple(unusable[:n_samples]),
                    )
                )

        # -- counts, for context -------------------------------------------------------
        counts: dict[str, int] = {}
        for table in _schema.ALL_TABLES:
            if table not in present or table == "anatid_meta":
                continue
            has_tenant = "tenant_id" in {
                r[1] for r in con.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
            }
            if all_tenants or not has_tenant:
                counts[table] = _scalar_int(
                    con.execute(f"SELECT count(*) FROM {quote_ident(table)}")
                )
            else:
                counts[table] = _scalar_int(
                    con.execute(
                        f"SELECT count(*) FROM {quote_ident(table)} WHERE {tenant_sql()}",
                        [ns.tenant_id],
                    )
                )

        report = DoctorReport(
            checked_at=utcnow(),
            schema_version=version,
            expected_schema_version=_schema.SCHEMA_VERSION,
            tenant_id=None if all_tenants else ns.tenant_id,
            all_tenants=all_tenants,
            findings=tuple(findings),
            counts=counts,
            checks_run=tuple(ran),
            checks_skipped=skipped,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        if raise_on_error and not report.ok:
            raise IntegrityError(
                f"doctor() found {len(report.errors)} integrity fault(s) in {self.path!r}: "
                + ", ".join(f"{f.check}={f.count}" for f in report.errors),
                report=report,
                findings=report.errors,
            )
        return report

    # ------------------------------------------------------------------ full text

    def fts_status(self, *, deep: bool = False) -> FtsStatus:
        """How the text arm will answer, as :class:`~anatid.types.FtsStatus`: which half of the
        library is running, how far its base generation is behind ``memories``, and what a
        rebuild would compact.  See the class docstring for what each field means on each half.
        """
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
        """Which path a CURRENT-STATE expansion for this handle's own tenant would take now.

        ``"extension"``, ``"csr"`` or ``"sql"``.  It is a forecast about the next such read, not
        a record of the last one: a read carrying ``as_of``, or a read for another tenant, makes
        its own decision and can take a different path while this still says ``"csr"``.  The
        value is an :class:`~anatid.csr.ExpandPath`, which IS that string and also carries
        ``.reason`` and ``.explain()``; :attr:`last_expansion` is the honest per-read record.
        """
        return self.csr.active

    @property
    def last_expansion(self):
        """The path the last graph expansion on this handle actually took, or ``None``.

        An :class:`~anatid.csr.ExpandPath`: ``str(...)`` is the path and ``.explain()`` says why
        it was chosen, which is how a fallback reports itself for the graph arm the way
        :class:`~anatid.derived.HealthReport` does for the others.
        """
        return self.csr.last_expansion

    def require_csr_extension(self) -> None:
        """Raise :class:`~anatid.errors.ExtensionUnavailable` unless the C++ path is live."""
        if not self.csr.fresh:
            raise ExtensionUnavailable(f"csr extension not active (state: {self.csr.describe()})")

    # ------------------------------------------------------------------ derived indexes

    def index_health(
        self,
        *,
        tenant: int | Namespace | None = None,
        policy: MaintenancePolicy | None = None,
        as_of: AsOf | _dt.datetime | None = None,
    ) -> dict[str, HealthReport]:
        """A :class:`~anatid.derived.HealthReport` per derived index, for one tenant.

        Covers every index DEFINED IN THE FILE, not only the ones this handle implements, so an
        operator can see the state of an accelerator another process builds.  The reason is
        machine-readable (:class:`~anatid.derived.HealthReason`): fresh, stale generation,
        unvalidated, historical query, rebuild in progress, load failure, damaged base or
        absent.

        ``as_of`` asks what a HISTORICAL read would do.  Most accelerators index current state
        and cannot answer one at all, so they report ``historical_query`` and the SQL path
        answers; the full-text index can, because its base holds document identity and content
        and the time predicate is applied to the canonical rows it joins.
        """
        ns = self.resolve_tenant(tenant)
        return self.indexes.health(ns.tenant_id, as_of=as_of, policy=policy)

    def maintain_indexes(
        self,
        *,
        tenant: int | Namespace | None = None,
        policy: MaintenancePolicy | None = None,
        now=None,
    ) -> dict[str, MaintenanceReport]:
        """Run :func:`anatid.derived.maintain` on every index THIS HANDLE implements.

        Building needs the accelerator's code, so unlike :meth:`index_health` this does not
        reach a definition another process owns.

        Explicitly callable, no background thread: call it after a batch of writes, on a timer
        of your own, or when :meth:`index_health` reports a stale generation.
        """
        ns = self.resolve_tenant(tenant)
        return self.indexes.maintain(ns.tenant_id, policy, now=now)

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

        A bulk load bypasses the verbs, so the derived indexes' journal cannot see it: every
        published generation over a loaded table is marked invalidated
        (:meth:`anatid.derived.IndexRegistry.invalidate`) and stops being usable until
        :meth:`maintain_indexes` rebuilds it.  The rows are correct throughout; only the
        accelerators fall back to the SQL path.

        The inserts and the invalidation are **one transaction**.  With two, there was a window
        in which the rows were committed and every generation was still marked validated, so a
        read in that window used a base generation that could not contain them; and an
        invalidation that failed left the generation trusted for good.  Now a failure rolls the
        load back with it.  The optional BM25 and CSR rebuilds run afterwards, outside the
        transaction, because they are rebuilds and not part of the load's atomicity.
        """
        d = Path(directory).expanduser()
        wanted = (
            list(tables)
            if tables is not None
            else [
                "entities",
                "episodes",
                "memories",
                "edges_about",
                "edges_relates",
                "edges_supersedes",
            ]
        )
        dim = self.config.embedding_dim
        out: dict[str, int] = {}
        con = self.connection
        with self.transaction():
            for name in wanted:
                f = d / f"{name}.parquet"
                if not f.is_file():
                    continue
                src = str(f).replace("'", "''")
                have = [
                    r[0]
                    for r in con.execute(
                        "SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet(?))", [str(f)]
                    ).fetchall()
                ]
                cols = [
                    r[1] for r in con.execute(f"PRAGMA table_info({quote_ident(name)})").fetchall()
                ]
                shared = [c for c in cols if c in have]
                if not shared:
                    continue
                proj = ", ".join(
                    (f"{quote_ident(c)}::FLOAT[{dim}]" if c == "embedding" else quote_ident(c))
                    for c in shared
                )
                order = _schema.CLUSTER_ORDER.get(name)
                order_sql = f" ORDER BY {order}" if order else ""
                before = _scalar_int(con.execute(f"SELECT count(*) FROM {quote_ident(name)}"))
                self.execute(
                    f"INSERT INTO {quote_ident(name)} "
                    f"({', '.join(quote_ident(c) for c in shared)}) "
                    f"SELECT {proj} FROM read_parquet('{src}'){order_sql}"
                )
                after = _scalar_int(con.execute(f"SELECT count(*) FROM {quote_ident(name)}"))
                out[name] = after - before
                if out[name]:
                    self.indexes.invalidate(
                        name, reason=f"bulk load of {out[name]} row(s) into {name}"
                    )
        if rebuild_fts and out.get("memories"):
            try:
                self.rebuild_fts_index()
            except duckdb.Error as exc:
                log.warning("fts index rebuild after load failed: %s", exc)
        if build_csr and self.csr.enabled:
            self.build_csr()
        return out

    # ------------------------------------------------------------------ conflicts

    #: Raw-cursor policy for :attr:`connection`.  ``None`` on a handle opened directly, which
    #: is the unguarded 0.1 behaviour; :class:`DatabasePool` sets ``"warn"`` or ``"deny"`` on
    #: the handles it opens so administrative access to a tenant file has to say its name.
    _raw_guard: str | None = None
    _raw_warned: bool = False
    #: The pool that opened this handle, when one did.  Audit events go there.
    _pool: "DatabasePool | None" = None

    def _check_raw_access(self) -> None:
        """Enforce the pool's ``raw_access`` policy for a caller outside :mod:`anatid`.

        Called from :attr:`connection`, so the frame two up is whoever asked for the cursor.
        anatid's own modules are always allowed: the verbs, ``recall`` and this module reach
        the cursor through that property on every call, and a policy that stopped them would
        stop the database.  Everyone else is warned once per handle, or refused, and the pool
        records it.

        A guardrail against reaching for the cursor by accident, not a sandbox.  Python has no
        private state, so a caller determined to have the connection object will get it; the
        point is that they cannot do it without either saying ``unsafe_connection`` or leaving
        a record.
        """
        caller = sys._getframe(2)  # noqa: SLF001 -- the documented way to see the caller
        module = caller.f_globals.get("__name__", "")
        if module == "anatid" or module.startswith("anatid."):
            return
        where = f"{caller.f_code.co_filename}:{caller.f_lineno}"
        pool = self._pool
        if self._raw_guard == "deny":
            if pool is not None:
                # DatabasePool and Anatid are two halves of one component in one module;
                # the audit log belongs to the pool, so the handle writes into it.
                pool._audit("raw_access_denied", self.namespace.tenant_id, self.path, where)  # noqa: SLF001
            raise TenantIsolationError(
                f"{where} asked a pooled handle for its raw DuckDB cursor. A cursor bypasses "
                f"every tenant and time predicate anatid compiles in, so this pool "
                f"(raw_access='deny') hands one out only through unsafe_connection(), which "
                f"says what it is and is audited."
            )
        if not self._raw_warned:
            self._raw_warned = True
            log.warning(
                "%s used the raw DuckDB cursor of pooled tenant %s; it bypasses every predicate "
                "anatid adds. Call unsafe_connection(reason=...) instead, which is audited.",
                where,
                self.namespace.tenant_id,
            )
            if pool is not None:
                pool._audit("raw_access", self.namespace.tenant_id, self.path, where)  # noqa: SLF001

    def unsafe_connection(self, *, reason: str | None = None) -> duckdb.DuckDBPyConnection:
        """This thread's raw DuckDB cursor, named for what it is.

        The administrative escape hatch, and the only sanctioned way to get a cursor out of a
        handle a :class:`DatabasePool` opened.  What comes back is a plain DuckDB connection:
        every statement run on it bypasses the tenant predicate, the valid-time and
        transaction-time predicates, the derived-index journal and the erasure path.  On a
        ``SCOPED`` handle that means it reads every tenant in the file; on a file-per-tenant
        handle it still means it can write rows anatid's verbs would have refused.

        ``reason`` is recorded with the pool's audit event, so an operator can see later why a
        cursor was taken.  Use the verbs for everything they cover.
        """
        pool = self._pool
        if pool is not None:
            pool._audit("unsafe_connection", self.namespace.tenant_id, self.path, reason)  # noqa: SLF001
        else:
            log.info("unsafe_connection on %s (%s)", self.path, reason or "no reason given")
        return self.connection

    def atomic(
        self,
        callback: Callable[..., Any],
        *,
        max_attempts: int = _atomic.DEFAULT_MAX_ATTEMPTS,
        backoff: float = _atomic.DEFAULT_BACKOFF,
        max_backoff: float = _atomic.DEFAULT_MAX_BACKOFF,
        sleep: Callable[[float], Any] | None = None,
        rng: Callable[[], float] | None = None,
    ) -> Any:
        """Run ``callback`` in one transaction, re-running the WHOLE callback on a conflict.

        ::

            def move():
                db.forget(old_id)
                return db.remember("Ada drinks tea now", entities=["Ada"])

            memory = db.atomic(move)

        Returns whatever the callback returns.  Between attempts anatid sleeps a jittered
        exponential backoff, so two writers that collided do not wake together and collide
        again.

        Re-running the *whole* callback is the only thing that can work.  DuckDB marks a
        transaction aborted at the first conflict, so the failed statement cannot be re-run
        inside it, and a unit of work that read a row before writing it has to read it again to
        be correct.  Write the callback so it is safe to run more than once: it must not depend
        on anything it computed in a previous attempt, and any id it mints should be minted
        inside it.

        Only a **retryable** :class:`~anatid.errors.ConflictError` is re-run.  A compare-and-swap
        failure (``expected_version=`` did not match, ``if_current=True`` found a closed row) is
        raised straight through, because the version the caller reasoned about is gone and the
        same callback would fail the same way every time.  No other exception is ever retried.
        The error that ends the last attempt carries ``attempt`` set to the number of attempts
        made.

        Inside a transaction the caller opened, this steps aside and runs the callback once:
        only the caller can decide to re-run the caller's transaction.  The callback is called
        with no arguments, or with an :class:`anatid.atomic.Attempt` if it accepts one.
        """
        return _atomic.run(
            self,
            callback,
            max_attempts=max_attempts,
            backoff=backoff,
            max_backoff=max_backoff,
            sleep=sleep,
            rng=rng,
        ).result

    def memory_version(self, memory_id: int, *, tenant: int | Namespace | None = None) -> int | None:
        """The version number of a memory's live row, or None when the id is not in this tenant.

        This is the number to hold for :meth:`update`'s ``expected_version``.  It is also
        ``db.get(memory_id).version``; this method reads one column instead of the row.
        """
        ns = self.resolve_tenant(tenant)
        return _atomic.version_of(self, "memories", "memory_id", int(memory_id), tenant_id=ns.tenant_id)

    def update(
        self,
        memory_id: int,
        content: str,
        *,
        expected_version: int | None = None,
        tenant: int | Namespace | None = None,
        **supersede_kwargs: Any,
    ) -> Memory:
        """Correct a memory, refusing the write if it is not at the version you read.

        ::

            m = db.get(mid)
            db.update(mid, "Ada drinks tea now", expected_version=m.version)

        A correction in anatid is a :meth:`~anatid.verbs.MemoryVerbs.supersede`: the new belief
        is a new memory with its own id, the old one is closed at ``now``, and a ``SUPERSEDES``
        edge records the link.  Every keyword :meth:`supersede` takes is passed through, and
        the new :class:`~anatid.types.Memory` comes back, so this is that verb plus the
        compare-and-swap the design asks for.

        ``expected_version`` is the version of the memory's live row
        (:meth:`memory_version`).  The check and the write are in one transaction, so nothing
        can slip between them: a writer that committed a correction first moves the version on
        and this call raises :class:`~anatid.errors.ConflictError` with ``expected_version``,
        ``current_version`` and ``retryable=False``.  A writer that is *concurrently*
        correcting the same row loses the write-write race instead and gets a retryable
        conflict from the engine.  Leave ``expected_version`` out and this is exactly
        ``supersede``.

        Retry policy stays with the caller, because only the caller knows whether a change
        computed from the old version still applies to the new one.  ``db.atomic(...)`` re-runs
        the engine-level conflict; the compare-and-swap failure is yours to handle.
        """
        ns = self.resolve_tenant(tenant)
        mid = int(memory_id)
        expected = None if expected_version is None else int(expected_version)

        def _work() -> Memory:
            with self.transaction():
                current = _atomic.version_of(
                    self, "memories", "memory_id", mid, tenant_id=ns.tenant_id
                )
                if current is None:
                    raise NotFoundError(f"memory {mid} not found in tenant {ns.tenant_id}")
                if expected is not None and current != expected:
                    raise _atomic.version_conflict(
                        f"memory {mid}",
                        expected_version=expected,
                        current_version=current,
                        action="update",
                    )
                return self.supersede(mid, content, tenant=ns, **supersede_kwargs)

        return self._atomic(_work)

    def relate(
        self,
        src: "int | str | Entity",
        dst: "int | str | Entity",
        *,
        if_current: bool = False,
        **kwargs: Any,
    ) -> Edge:
        """:meth:`~anatid.verbs.MemoryVerbs.relate`, with an optional guard on the endpoints.

        ``if_current=True`` refuses to write the edge unless both endpoints have a current,
        live row in this tenant, and raises a non-retryable
        :class:`~anatid.errors.ConflictError` naming the first one that does not.  Without it
        this is the verb unchanged, which takes an ``int`` endpoint verbatim and never checks
        that the entity exists: relating to an id that was purged writes an edge into a graph
        no traversal can explain.

        The check runs in the transaction that writes the edge, so it sees that transaction's
        own writes (an entity ``create_missing`` just minted counts as current) and one
        consistent snapshot.  It is snapshot isolation, not serializability: an endpoint closed
        by a transaction that commits after this one's snapshot opened is not seen, because
        reading a row does not conflict with writing it in DuckDB's MVCC.  What the guard
        removes is the serialized case, which is the one that happens: a caller acting on a
        stale read, or a retry after a purge.
        """
        if not if_current:
            return MemoryVerbs.relate(self, src, dst, **kwargs)
        ns = self.resolve_tenant(kwargs.pop("tenant", None))
        at = to_utc_naive(kwargs.pop("now", None)) or utcnow()
        create_missing = kwargs.get("create_missing", True)
        writer = kwargs.get("writer")

        def _work() -> Edge:
            with self.transaction():
                s = self.entity_id(src, tenant=ns, create=create_missing, now=at, writer=writer)
                d = self.entity_id(dst, tenant=ns, create=create_missing, now=at, writer=writer)
                found = _atomic.current_ids(
                    self, "entities", "entity_id", (s, d), tenant_id=ns.tenant_id
                )
                gone = _atomic.missing_or_closed(found, (s, d))
                if gone:
                    raise ConflictError(
                        f"entity {gone[0]} has no current row in tenant {ns.tenant_id}, so "
                        f"relate(if_current=True) will not attach an edge to it. Re-read the "
                        f"entity: it was purged, or its row was closed.",
                        resource=f"entity {gone[0]}",
                        retryable=False,
                    )
                return MemoryVerbs.relate(self, s, d, tenant=ns, now=at, **kwargs)

        return self._atomic(_work)


def connect(path: str | os.PathLike = ":memory:", **kwargs) -> Anatid:
    """Shorthand for :meth:`Anatid.open`."""
    return Anatid.open(path, **kwargs)


# --------------------------------------------------------------------------- pool

#: Mode a pool sets on a tenant file it creates.  Owner read/write, nothing for group or other.
POOL_FILE_MODE = 0o600

#: Mode a pool sets on a directory it creates.  The directory is the real protection for the
#: files DuckDB writes on its own schedule (the write-ahead log appears at the first write, and
#: nothing gives anatid a hook between its creation and its first byte).
POOL_DIR_MODE = 0o700

#: What a pool does when code outside :mod:`anatid` reads :attr:`Anatid.connection` on one of
#: its handles.  ``"warn"`` logs once per handle and records an audit event; ``"deny"`` raises
#: :class:`~anatid.errors.TenantIsolationError` and points at
#: :meth:`Anatid.unsafe_connection`; ``"allow"`` is the unguarded 0.1 behaviour.
RAW_ACCESS_MODES = ("allow", "warn", "deny")

#: Characters that must never reach a path component the pool builds.  ``os.sep`` and
#: ``os.altsep`` are added at run time so this is right on every platform.
_PATH_SEPARATORS = {"/", "\\", "\x00", os.sep, os.altsep or "/"}


@dataclass(frozen=True, slots=True)
class PoolEvent:
    """One thing a :class:`DatabasePool` did to a tenant's file.

    ``action`` is one of ``open``, ``close``, ``evict``, ``delete``, ``backup``, ``attach``,
    ``unsafe_connection``, ``raw_access``, ``raw_access_denied``, ``rejected``.  Events are
    kept in a bounded deque on the pool and handed to the ``audit=`` callback as they happen,
    so the record can go wherever the deployment keeps its audit log.
    """

    action: str
    tenant_id: int | None
    at: _dt.datetime
    path: str | None = None
    detail: str | None = None

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "tenant_id": self.tenant_id,
            "at": self.at.isoformat(),
            "path": self.path,
            "detail": self.detail,
        }

    def __str__(self) -> str:
        bits = [self.action, f"tenant={self.tenant_id}"]
        if self.path:
            bits.append(self.path)
        if self.detail:
            bits.append(f"({self.detail})")
        return " ".join(bits)


def _reject_component(text: str, *, what: str, tenant_id: int) -> str:
    """Return ``text`` when it is a single, harmless path component, or refuse it.

    A tenant label is caller data, and the pool turns caller data into a filesystem path.  A
    label of ``"../secrets"`` in a ``{label}`` template would name a file outside the pool's
    directory, which is the whole tenant boundary gone.  This refuses rather than sanitises:
    rewriting ``"../x"`` into ``"x"`` would silently map two different tenants onto one file,
    which is the same leak by another route.
    """
    bad = None
    if not text:
        bad = "it is empty"
    elif text in (".", ".."):
        bad = "it is a directory reference"
    elif any(sep in text for sep in _PATH_SEPARATORS):
        bad = "it contains a path separator"
    elif text.startswith("~"):
        bad = "it starts with ~, which expands to a home directory"
    elif len(text) > 128:
        bad = "it is longer than 128 characters"
    if bad is None:
        return text
    raise TenantIsolationError(
        f"tenant {tenant_id}'s {what} {text!r} cannot be part of a file name: {bad}. "
        f"A pool builds one file per tenant inside its own directory; a name that could "
        f"escape it would not be a tenant boundary. Use opaque=True, or a template that "
        f"only interpolates {{tenant}}."
    )


class DatabasePool:
    """A directory of per-tenant anatid files, opened lazily and closed LRU.

    This is how anatid delivers *real* tenant isolation: one DuckDB file per tenant, so a handle
    physically cannot see another tenant's rows (DuckDB has no row-level security to lean on).
    Opening a DuckDB file is cheap, so the pool keeps at most ``max_open`` of them and closes the
    least recently used beyond that.

    ::

        pool = DatabasePool("/var/lib/anatid/tenant_{tenant}.duckdb", embedding_dim=1536)
        db = pool.get(42)  # opens (and creates) tenant 42's file
        db.remember("...", entities=["Ada"])  # tenant 42, enforced

    ``path_template`` may use ``{tenant}`` (the integer tenant id) and ``{label}`` (the
    namespace's name).  Handles it hands out carry
    :attr:`~anatid.types.Isolation.FILE_PER_TENANT`, so any verb called with a different tenant
    raises :class:`~anatid.errors.TenantIsolationError`.

    Turning a tenant into a path
    ----------------------------
    A path built from caller data is an attack surface, so the pool never simply formats one in.
    Every component it interpolates is checked (:func:`_reject_component`) and the finished path
    must resolve inside the pool's :attr:`root`, which is the fixed prefix of the template.  A
    label of ``"../../etc/passwd"`` is refused, not sanitised.  With ``opaque=True`` no caller
    data reaches the path at all: both placeholders become a BLAKE2b digest of the tenant id
    (keyed with ``secret=`` when one is given), so the directory listing carries no customer
    names.  The digest is derived from the tenant id alone, so one tenant is one file however
    it is addressed, and :meth:`registry` maps the digests back.

    Permissions
    -----------
    A directory the pool creates is ``0o700`` and a file it creates is ``0o600``
    (``dir_mode`` / ``file_mode``, ``None`` to leave both alone).  DuckDB creates the file
    itself, so the pool chmods it immediately afterwards; the directory mode is what actually
    protects the write-ahead log, which DuckDB creates on its own schedule.

    Raw cursors
    -----------
    ``raw_access`` decides what happens when code outside anatid reads
    :attr:`Anatid.connection` on a pooled handle, since a cursor bypasses every predicate
    anatid compiles in.  The default warns once per handle and records it;
    ``raw_access="deny"`` refuses outright and is what a deployment should run, because then
    :meth:`Anatid.unsafe_connection` (or :meth:`unsafe_connection` here) is the ONLY way to get
    a cursor out of a pooled file, and it says what it is and is audited.  The default is the
    softer one only so that code written against 0.1 keeps working on upgrade.

    Audit
    -----
    Opens, evictions, closes, deletions, backups, attachments, refused paths and raw-cursor
    access are recorded as :class:`PoolEvent`s: kept in :meth:`events` (bounded by
    ``audit_size``) and passed to the ``audit=`` callback as they happen.  The callback runs on
    the calling thread and may hold the pool's lock, so it must not call back into the pool.

    Cross-tenant reads go through :meth:`attach_read_only`, which attaches another tenant's file
    READ ONLY under an alias -- explicit, auditable, and unable to write.

    The pool is thread-safe.  The :class:`Anatid` handles it returns are themselves thread-safe
    (connection per thread), so a request handler can fetch one per request.
    """

    def __init__(
        self,
        path_template: str | os.PathLike,
        *,
        root: str | os.PathLike | None = None,
        opaque: bool = False,
        secret: bytes | None = None,
        max_open: int = 16,
        create_parents: bool = True,
        file_mode: int | None = POOL_FILE_MODE,
        dir_mode: int | None = POOL_DIR_MODE,
        raw_access: str = "warn",
        audit: "Callable[[PoolEvent], Any] | None" = None,
        audit_size: int = 256,
        **open_kwargs: Any,
    ) -> None:
        self.path_template = str(path_template)
        if "{tenant}" not in self.path_template and "{label}" not in self.path_template:
            raise ValueError(
                "path_template must contain {tenant} (or {label}) so tenants get separate files"
            )
        self.max_open = int(max_open)
        if self.max_open < 1:
            raise ValueError("max_open must be >= 1")
        if raw_access not in RAW_ACCESS_MODES:
            raise ValueError(f"raw_access must be one of {RAW_ACCESS_MODES}, got {raw_access!r}")
        self.create_parents = create_parents
        self.opaque = bool(opaque)
        self.file_mode = file_mode
        self.dir_mode = dir_mode
        self.raw_access = raw_access
        self.open_kwargs = open_kwargs
        self.root = Path(root).expanduser() if root is not None else self._template_root()
        self._secret = bytes(secret) if secret else b""
        self._open: "OrderedDict[int, Anatid]" = OrderedDict()
        self._paths: dict[int, str] = {}
        self._audit_hook = audit
        self._events: "deque[PoolEvent]" = deque(maxlen=max(1, int(audit_size)))
        self._lock = threading.RLock()
        self._closed = False

    # -------------------------------------------------------------- paths

    def _template_root(self) -> Path:
        """The fixed directory prefix of the template: everything before the first ``{``.

        Every path this pool builds has to resolve inside it.  For
        ``/var/lib/anatid/t_{tenant}.duckdb`` that is ``/var/lib/anatid``; for
        ``/var/lib/anatid/{label}/db.anatid`` it is ``/var/lib/anatid`` as well.
        """
        head = self.path_template.split("{", 1)[0]
        p = Path(head).expanduser()
        return p if head.endswith(("/", os.sep)) else p.parent

    def _digest(self, tenant_id: int) -> str:
        """The opaque file name component for a tenant: a keyed BLAKE2b of its id.

        Derived from the id alone, so ``get(7)`` and ``get(Namespace(7, "acme"))`` are one
        file.  Without ``secret=`` the digest is stable but enumerable (a small integer space
        is cheap to walk); that is enough to keep customer names out of a directory listing,
        and ``secret=`` is what makes it unguessable.
        """
        h = hashlib.blake2b(digest_size=16, key=self._secret)
        h.update(f"anatid-tenant:{int(tenant_id)}".encode())
        return h.hexdigest()

    def path_for(self, tenant: int | Namespace) -> Path:
        """The file this pool would use for ``tenant`` (whether or not it exists yet).

        Refuses with :class:`~anatid.errors.TenantIsolationError` when the label cannot be a
        file name, or when the finished path would land outside :attr:`root`.
        """
        ns = Namespace.coerce(tenant)
        if self.opaque:
            token = self._digest(ns.tenant_id)
            fields = {"tenant": token, "label": token}
        else:
            try:
                label = _reject_component(ns.name, what="label", tenant_id=ns.tenant_id)
            except TenantIsolationError as exc:
                self._audit("rejected", ns.tenant_id, None, str(exc).split(":", 1)[-1].strip())
                raise
            fields = {"tenant": str(int(ns.tenant_id)), "label": label}
        try:
            raw = self.path_template.format(**fields)
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"path_template {self.path_template!r} uses a field this pool does not "
                f"provide ({exc}); only {{tenant}} and {{label}} are substituted"
            ) from exc
        path = Path(raw).expanduser()
        root = self.root.expanduser()
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            self._audit("rejected", ns.tenant_id, str(path), "outside the pool root")
            raise TenantIsolationError(
                f"tenant {ns.tenant_id} resolves to {path}, which is outside this pool's "
                f"directory {root}. A pool keeps every tenant file under one root; a path "
                f"that leaves it is not a tenant boundary."
            ) from None
        self._paths[ns.tenant_id] = str(path)
        return path

    def registry(self) -> dict[int, str]:
        """``{tenant_id: path}`` for every tenant this pool has resolved a path for.

        The way back from an opaque file name to the tenant it belongs to, for an operator
        looking at a directory listing.  It covers the tenants this pool has seen, not every
        file on disk: the mapping is a pure function of the tenant id (:meth:`path_for`), so
        anything missing can be recomputed by asking for it.
        """
        return dict(self._paths)

    def known_tenants(self) -> list[int]:
        """Tenant ids currently held open by the pool, most recently used last."""
        with self._lock:
            return list(self._open)

    # -------------------------------------------------------------- audit

    def _audit(
        self,
        action: str,
        tenant_id: int | None,
        path: str | None = None,
        detail: str | None = None,
    ) -> PoolEvent:
        event = PoolEvent(action=action, tenant_id=tenant_id, at=utcnow(), path=path, detail=detail)
        self._events.append(event)
        log.debug("pool %s", event)
        hook = self._audit_hook
        if hook is not None:
            try:
                hook(event)
            except Exception:  # an audit sink must never break the operation it records
                log.exception("pool audit hook failed for %s", event)
        return event

    def events(self, *, action: str | None = None, tenant: int | Namespace | None = None) -> list[PoolEvent]:
        """The recorded :class:`PoolEvent`s, oldest first, optionally filtered."""
        tid = None if tenant is None else Namespace.coerce(tenant).tenant_id
        return [
            e
            for e in list(self._events)
            if (action is None or e.action == action) and (tid is None or e.tenant_id == tid)
        ]

    # -------------------------------------------------------------- permissions

    def _chmod(self, path: Path, mode: int | None) -> bool:
        if mode is None:
            return False
        try:
            os.chmod(path, mode)
            return True
        except OSError as exc:  # a filesystem that has no modes, or a file that just went away
            log.warning("could not set mode %o on %s: %s", mode, path, exc)
            return False

    def harden(self, tenant: int | Namespace | None = None) -> list[str]:
        """Re-apply ``file_mode`` to a tenant's file and write-ahead log (or to every known one).

        The pool does this for a file it creates.  Call it after a restore, or on a schedule,
        for the write-ahead log: DuckDB creates that at the first write, with the process
        umask, and there is no hook in between.  A directory the pool created is ``0o700``, so
        the log is unreachable there whatever its own mode.
        """
        targets = (
            [self.path_for(tenant)]
            if tenant is not None
            else [Path(p) for p in self.registry().values()]
        )
        done: list[str] = []
        for path in targets:
            for candidate in (path, Path(f"{path}.wal")):
                if candidate.exists() and self._chmod(candidate, self.file_mode):
                    done.append(str(candidate))
        return done

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
            existed = path.exists()
            if self.create_parents and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                self._chmod(path.parent, self.dir_mode)
            kwargs = dict(self.open_kwargs)
            kwargs.pop("tenant", None)
            kwargs.pop("isolation", None)
            db = Anatid.open(path, tenant=ns, isolation=Isolation.FILE_PER_TENANT, **kwargs)
            # The handle is this pool's from here on: it carries the pool for its audit
            # events and the raw-cursor policy the pool was constructed with.
            db._pool = self  # noqa: SLF001
            db._raw_guard = None if self.raw_access == "allow" else self.raw_access  # noqa: SLF001
            if not existed:
                # DuckDB creates the file, so the mode is the process umask for the moment
                # between its creation and this call.  The 0o700 directory above is what makes
                # that window harmless.
                self._chmod(path, self.file_mode)
            for wal in (Path(f"{path}.wal"),):
                if wal.exists():
                    self._chmod(wal, self.file_mode)
            self._open[ns.tenant_id] = db
            self._open.move_to_end(ns.tenant_id)
            self._audit("open", ns.tenant_id, str(path), "created" if not existed else "existing")
            self._evict()
            return db

    __getitem__ = get

    def _evict(self) -> None:
        while len(self._open) > self.max_open:
            tid, victim = self._open.popitem(last=False)
            self._audit("evict", tid, victim.path, f"max_open={self.max_open}")
            victim.close()

    def unsafe_connection(
        self, tenant: int | Namespace, *, reason: str | None = None
    ) -> duckdb.DuckDBPyConnection:
        """The raw DuckDB cursor for one tenant's file, audited.

        The only way this pool hands out a cursor, and the only one that works at all under
        ``raw_access="deny"``.  Everything :meth:`Anatid.unsafe_connection` says applies: the
        statements you run on it bypass every tenant and time predicate anatid compiles in, the
        derived-index journal and the erasure path.
        """
        return self.get(tenant).unsafe_connection(reason=reason)

    def close(self, tenant: int | Namespace) -> bool:
        """Close one tenant's handle.  Returns True if it was open."""
        ns = Namespace.coerce(tenant)
        with self._lock:
            db = self._open.pop(ns.tenant_id, None)
        if db is None:
            return False
        db.close()
        self._audit("close", ns.tenant_id, db.path)
        return True

    def close_all(self) -> None:
        """Close every open handle.  Idempotent."""
        with self._lock:
            self._closed = True
            handles = list(self._open.items())
            self._open.clear()
        for tid, db in handles:
            db.close()
            self._audit("close", tid, db.path)

    def __enter__(self) -> "DatabasePool":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close_all()

    def __len__(self) -> int:
        with self._lock:
            return len(self._open)

    def __repr__(self) -> str:
        return (
            f"<DatabasePool template={self.path_template!r} open={len(self)}/{self.max_open}"
            f"{' opaque' if self.opaque else ''}>"
        )

    # -------------------------------------------------------------- lifecycle

    def delete(self, tenant: int | Namespace, *, missing_ok: bool = True) -> bool:
        """Close a tenant's handle and remove its file.  Returns True if a file was removed.

        Erasure at the granularity the file-per-tenant model actually gives you: the database,
        its write-ahead log and any temporary directory beside it are unlinked, so nothing of
        that tenant is left for another handle to open.  The audit event is written whether or
        not a file was there.

        This does not consult the tenant: whatever is in the file goes.  For erasing one
        document while the tenant keeps working, use ``forget(hard=True)``.
        """
        ns = Namespace.coerce(tenant)
        path = self.path_for(ns)
        self.close(ns)
        removed = False
        for extra in (path, Path(f"{path}.wal")):
            try:
                extra.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise AnatidError(f"could not delete {extra}: {exc}") from exc
            removed = removed or extra == path
        tmp = Path(f"{path}.tmp")
        if tmp.is_dir():
            shutil.rmtree(tmp, ignore_errors=True)
        self._paths.pop(ns.tenant_id, None)
        self._audit("delete", ns.tenant_id, str(path), "removed" if removed else "no file")
        if not removed and not missing_ok:
            raise FileNotFoundError(f"no database file for tenant {ns.tenant_id} at {path}")
        return removed

    def backup(
        self, tenant: int | Namespace, path: str | os.PathLike, *, overwrite: bool = False
    ) -> Path:
        """Copy one tenant's database to ``path`` and return it.

        The copy is made by DuckDB (``COPY FROM DATABASE``) inside the handle that owns the
        file, so it is a consistent snapshot of committed data: a transaction another thread
        has open but has not committed is not in it.  Copying the file with ``cp`` while a
        writer is running gives no such promise.

        The destination is created with the pool's ``file_mode``.  An existing destination is
        refused unless ``overwrite=True``, because a backup that silently replaced the previous
        one is one crash away from having neither.  The refusal is
        :class:`~anatid.errors.BackupDestinationExists`, which is a ``FileExistsError`` so code
        that already catches one keeps working, and an :class:`~anatid.errors.AnatidError` so a
        caller that maps anatid's errors to its own vocabulary (an exit code, an HTTP status) can
        tell it apart from every other ``OSError`` the filesystem might raise.

        The source catalog is quoted with :func:`~anatid.schema.quote_name`, not
        ``quote_ident``.  DuckDB derives the catalog from the file stem, so the two pool
        templates the documentation itself shows (``tenant-{tenant}.anatid`` and
        ``{tenant}.anatid``) produce ``tenant-1`` and ``1``, neither of which is a bare SQL
        identifier.  Validating them would refuse an ordinary pool; quoting them works.
        """
        dest = Path(path).expanduser()
        if dest.exists():
            if not overwrite:
                raise BackupDestinationExists(
                    f"{dest} exists; pass overwrite=True to replace it (a backup that "
                    f"overwrites silently can leave you with neither copy)",
                    path=dest,
                )
            dest.unlink()
        ns = Namespace.coerce(tenant)
        db = self.get(ns)
        if self.create_parents:
            dest.parent.mkdir(parents=True, exist_ok=True)
        alias = "anatid_backup"
        literal = str(dest).replace("'", "''")
        name = db.execute("SELECT current_database()").fetchone()
        catalog = quote_name(str(name[0]) if name else "memory")
        db.execute(f"ATTACH '{literal}' AS {quote_ident(alias)}")
        try:
            db.execute(f"COPY FROM DATABASE {catalog} TO {quote_ident(alias)}")
        finally:
            db.execute(f"DETACH {quote_ident(alias)}")
        self._chmod(dest, self.file_mode)
        self._audit("backup", ns.tenant_id, str(dest), f"from {db.path}")
        return dest

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
        visible in ``host.attached``, the attachment cannot write, anatid's own verbs never look
        outside ``main``, and the pool records an audit event.  A cross-tenant query has to be
        written by hand and run through the named administrative cursor::

            alias = pool.attach_read_only(1, 2)
            rows = (
                pool.unsafe_connection(1, reason="support ticket 91")
                .execute(f"SELECT content FROM {alias}.memories LIMIT 5")
                .fetchall()
            )

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
        out = host_db.attach_read_only(path, name)
        self._audit("attach", host_ns.tenant_id, str(path), f"read-only as {name}")
        return out
