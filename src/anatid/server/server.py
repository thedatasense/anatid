"""The anatid server: one process owns the files and answers verbs over a socket.

What this is
------------
``anatid.Anatid`` is one writer process with many writer threads.  That stays exactly as it is,
and this is a second deployment profile beside it, not a replacement.  It exists because DuckDB
allows one process to hold a database file read-write and, measured on duckdb 1.5.5, excludes
every other process from that file entirely -- a second process gets ``duckdb.IOException`` even
when it asks for read-only access.  So when several processes have to share one memory, one of
them has to own the files and the others have to ask it.

Both directions go over the wire, therefore.  There is no "write through the server, read the
file directly" arrangement, because while the server holds the file no other process can open it
at all.  What that costs, measured on a 3,001-memory tenant at 384 dimensions, p50, in process
against over a Unix socket: ``get`` 0.49 ms -> 1.01 ms, ``remember`` 2.83 ms -> 4.24 ms,
``recall`` with its vector arm 25.4 ms -> 25.0 ms.  About half a millisecond per call, fixed,
which doubles the cheapest read and vanishes into a recall.  Most of it is the JSON codec rather
than the socket (a bare socket echo is 7.5 us), and ``ServerConfig(embeddings="f32")`` removes
most of the codec share.

The shape
---------
::

    connection thread                   worker thread                     DuckDB
        |                                    |                               |
        +-- authenticate (once per conn)     |                               |
        +-- decode Request                   |                               |
        +-- principal.require(tenant)  <-- the tenant boundary, before any file is opened
        |                                    |                               |
        +-- read verb  --> read pool ------------------------------------->  SELECT
        |                                    |                               |
        +-- write verb --> WriteQueue ---> one transaction per batch ------>  INSERT
        |                                    |                               |
        +-- encode Response                  |                               |

Transports.  A Unix domain socket by default, authenticated by the permissions on the socket and
its directory plus peer credentials where the platform reports them.  HTTP optionally, which
requires a bearer token and refuses to bind anything but loopback without one.

Isolation.  A principal carries the tenants it may name and the dispatcher checks before it
resolves a handle, so a client authenticated for tenant 1 cannot read or write tenant 2 and
cannot learn from the error whether tenant 2 exists.  With a :class:`anatid.DatabasePool` that
check sits on top of file-per-tenant isolation.  With a single shared file it sits on top of
namespaces, which is not a security boundary, and :class:`ServerConfig` says so.

Shutdown.  SIGTERM closes the listeners, stops accepting writes at once, drains the queues under
``shutdown_timeout``, cancels the connections, releases every open file and exits.  Releasing is
what folds the write-ahead log into the file and, just as importantly, gives up DuckDB's exclusive
lock, so the replacement process can open the file at all.  Whatever the drain abandoned is named
per tenant in the :class:`~anatid.server.queue.DrainReport` and completed with
:class:`~anatid.server.protocol.ShuttingDown`, so no client is left waiting on a write nothing
will run.  The budget bounds the whole exit and not only the drain: a client that stays connected
does not extend it, and a second SIGTERM abandons the drain rather than queueing behind it.

Health and readiness are different questions.  Health is "this process is alive", which is what a
supervisor restarts on.  Readiness is "the files are open, the migrations are done and no queue
is at its high-water mark", which is what a load balancer takes out of rotation on.  A server
under backpressure is healthy and not ready, and collapsing the two would either restart a busy
server or keep sending traffic to one that cannot take it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import os
import signal
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable, Mapping

from .. import __version__ as _anatid_version
from .. import csr as _csr
from .. import derived as _derived
from .. import fts as _fts
from ..database import Anatid, DatabasePool
from ..errors import AnatidError, TenantIsolationError
from ..schema import SCHEMA_VERSION, current_version
from . import protocol
from .auth import (
    SOCKET_DIR_MODE,
    SOCKET_MODE,
    AllowAllAuthenticator,
    Authenticator,
    ConnectionContext,
    Principal,
    check_bind_address,
)
from .protocol import (
    MAX_FRAME_BYTES,
    AuthenticationError,
    AuthorizationError,
    BusyError,
    DeadlineExceeded,
    FrameError,
    ProtocolError,
    Request,
    Response,
    ShuttingDown,
    Status,
    UnsupportedTransport,
)
from .queue import (
    DEFAULT_BATCH_MAX,
    DEFAULT_IDEMPOTENCY_TTL,
    DEFAULT_MAX_DEPTH,
    DEFAULT_WORKERS,
    DrainReport,
    IdempotencyStore,
    WriteQueue,
)

if TYPE_CHECKING:  # pragma: no cover - backup.py takes an AnatidServer, so this is a cycle
    from .backup import BackupReport

log = logging.getLogger("anatid.server")

#: Bounds on the last wait in :meth:`AnatidServer.shutdown`: how long a listener gets to report
#: itself closed once every connection handler has been cancelled.  It is normally instant, and
#: the ceiling is what keeps a client that will not release a socket from holding the process
#: open the way it used to.
_MIN_LISTENER_GRACE = 0.05
_MAX_LISTENER_GRACE = 1.0

__all__ = [
    "VerbSpec",
    "VERBS",
    "ServerConfig",
    "Health",
    "Readiness",
    "AnatidServer",
    "serve",
]


# --------------------------------------------------------------------------- wire types
#
# The value types the dispatched verbs return that do not live in anatid.types.  Registered here
# rather than in protocol.py so that module keeps importing only anatid.types and anatid.errors.

for _dc in (
    _derived.Watermark,
    _derived.Generation,
    _derived.IndexDefinition,
    _derived.ValidationReport,
    _derived.HealthReport,
    _derived.MaintenanceReport,
    _derived.IndexEvent,
    _derived.ErasureResult,
    _fts.FtsSearch,
    _csr.CsrInfo,
):
    protocol.register_dataclass(_dc)
del _dc

protocol.register_enum(_derived.HealthReason)


def _encode_expand_path(value: _csr.ExpandPath, _opts: Any) -> Any:
    """``ExpandPath`` is a ``str`` subclass with four extra attributes, not a dataclass.

    Both ``stats`` and ``info`` put one in the mapping they return, so without this codec those
    two verbs raise while the server is encoding a reply it has already computed.  Encoding it as
    a bare string would be worse than a failure in a quieter way: the value would still compare
    equal to ``"sql"`` and the reason it was ``"sql"`` would be gone, which is the one thing the
    class exists to carry.
    """
    return {
        "path": str(value),
        "reason": value.reason.value,
        "detail": value.detail,
        "generation": value.generation,
        "tenant_id": value.tenant_id,
    }


def _decode_expand_path(raw: Any) -> _csr.ExpandPath:
    if not isinstance(raw, Mapping):
        raise ProtocolError(f"ExpandPath should carry an object, got {type(raw).__name__}")
    return _csr.ExpandPath(
        str(raw.get("path", "")),
        reason=_derived.HealthReason(raw.get("reason", _derived.HealthReason.ABSENT.value)),
        detail=str(raw.get("detail", "")),
        generation=raw.get("generation"),
        tenant_id=raw.get("tenant_id"),
    )


protocol.register_codec("ExpandPath", _encode_expand_path, _decode_expand_path)


# --------------------------------------------------------------------------- the verb table


@dataclass(frozen=True, slots=True)
class VerbSpec:
    """One callable a request may name.

    A request carries a verb NAME and the server looks it up here.  It is never an attribute
    lookup on the handle from client-supplied text: a table is the difference between "the
    client may call these fourteen things" and "the client may call anything that happens to be
    a method".

    ``write``
        True routes the call through the per-tenant write queue.  False runs it on the read
        pool, concurrently.
    ``batchable``
        False gives the call a transaction of its own even when other writes for the tenant are
        queued beside it.  Set for the verbs that rewrite tables, publish index generations or
        purge across every table, where sharing a transaction with an unrelated write buys
        nothing and couples two failures.
    ``tenant_arg``
        Whether the callable takes ``tenant=``.  The server always passes it when it does, so
        the handle's own tenant check runs as well as the dispatcher's.
    ``server``
        Answered by the server itself rather than by a database handle.
    """

    name: str
    write: bool = False
    batchable: bool = True
    tenant_arg: bool = True
    server: bool = False
    summary: str = ""


def _spec(name: str, **kw: Any) -> tuple[str, VerbSpec]:
    return name, VerbSpec(name=name, **kw)


#: Every verb a client may call, and how the server runs it.  Adding an entry is the only way to
#: expose a method; there is no fallback path that reaches an unlisted one.
VERBS: dict[str, VerbSpec] = dict(
    [
        # -- reads -------------------------------------------------------------------------
        _spec("get", summary="one memory by id"),
        _spec("versions", summary="every version of one memory"),
        _spec("recall", summary="fused vector, text and graph retrieval"),
        _spec("recall_2hop", summary="memories two hops from a seed entity"),
        _spec("recall_2hop_ids", summary="the same, as ids"),
        _spec("context", summary="memories about one entity"),
        _spec("entities_of", summary="the entities one memory is about"),
        _spec("provenance", summary="the evidence trail behind one memory"),
        _spec("get_entity", summary="one entity by id or name"),
        _spec("get_episode", summary="one episode by id"),
        _spec("memory_version", summary="the live version number of one memory"),
        _spec("stats", summary="row counts for a tenant"),
        _spec("doctor", summary="integrity report"),
        _spec("index_health", summary="per-index health and staleness"),
        _spec("info", tenant_arg=False, summary="the anatid_meta catalog row"),
        _spec("fts_status", tenant_arg=False, summary="state of the BM25 index"),
        # -- writes ------------------------------------------------------------------------
        _spec("remember", write=True, summary="write a memory"),
        _spec("supersede", write=True, summary="correct a memory with a newer one"),
        _spec("correct", write=True, summary="supersede a memory and move its edges, as one"),
        _spec("reinforce", write=True, summary="record an access, adjust confidence"),
        _spec("update", write=True, summary="correct a memory at an expected version"),
        _spec("relate", write=True, summary="an edge between two entities"),
        _spec("unrelate", write=True, summary="close an edge"),
        _spec("upsert_entity", write=True, summary="create or fetch an entity"),
        _spec("entity_id", write=True, summary="resolve a name to an id, optionally creating"),
        _spec("episode", write=True, summary="write raw source material"),
        # Not batchable: each of these rewrites tables, publishes a generation or purges across
        # every table, so a shared transaction couples two unrelated failures for no gain.
        _spec("forget", write=True, batchable=False, summary="close or hard-purge a memory"),
        _spec("prune", write=True, batchable=False, summary="close or purge in bulk"),
        _spec(
            "rebuild_fts_index",
            write=True,
            batchable=False,
            tenant_arg=False,
            summary="rebuild the BM25 index",
        ),
        _spec("maintain_indexes", write=True, batchable=False, summary="build derived indexes"),
        _spec(
            "recluster",
            write=True,
            batchable=False,
            tenant_arg=False,
            summary="rewrite by sort key",
        ),
        # -- the server itself -------------------------------------------------------------
        _spec("health", server=True, tenant_arg=False, summary="is this process alive"),
        _spec("ready", server=True, tenant_arg=False, summary="can it take traffic"),
        _spec("queue_stats", server=True, tenant_arg=False, summary="queue depths and counters"),
    ]
)


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """How one server process is set up.

    ``socket_path``
        The Unix socket to listen on.  Created with mode 0600 in a directory the server sets to
        0700, which is the actual access control for this transport.  None disables it.
    ``http_host`` / ``http_port``
        An optional HTTP listener.  ``http_host`` outside loopback needs an authenticator that
        requires a token; :func:`anatid.server.auth.check_bind_address` refuses the combination
        at construction, not at the first request.
    ``max_depth``, ``batch_max``, ``workers``
        Passed to the :class:`~anatid.server.queue.WriteQueue`.  ``workers`` is how many tenants
        can be written in parallel; no tenant is ever written by two at once.
    ``read_workers``
        Threads answering reads.  Reads do not queue: they run concurrently with each other and
        with writes, under DuckDB's MVCC.
    ``idempotency``
        Whether to keep idempotency keys at all.  On by default: a retried write that writes
        twice is the failure mode a network makes routine.
    ``default_deadline``
        Seconds of budget for a request that does not carry its own.  A queued write whose
        deadline passes before it starts is dropped and reported, never half-run.
    ``shutdown_timeout``
        How long a graceful shutdown drains before it abandons what is left and says what it was.
        It bounds the exit, not just the drain: a client that stays connected does not extend it,
        and the whole shutdown finishes inside this budget plus about a second for the listeners.
    ``embeddings``
        ``"list"`` sends embeddings as JSON arrays, ``"f32"`` as base64 little-endian float32.
        Measured on one ``Memory``: at 384 dimensions 7,725 bytes and 274 us to encode and decode
        as an array against 2,408 bytes and 69 us as f32; at 1536, 29,934 bytes and 1,050 us
        against 8,552 bytes and 211 us.  The codec is the larger half of what the server costs
        over an in-process call, so this is the setting that moves that number.  ``"f32"`` rounds
        to float32, which is what anatid's ``FLOAT[N]`` column already holds, so it is lossless
        for a value read back out of the database and lossy (measured maximum absolute error
        3.0e-08) for a float64 a caller computed.
    ``checkpoint_on_shutdown``
        Release every open file on the way out, which folds its write-ahead log in and gives up
        DuckDB's exclusive lock.  See :meth:`AnatidServer._checkpoint_all` for why it is a
        release and not a ``CHECKPOINT`` statement.
    ``create_tenants``
        Whether naming a tenant this server has never seen creates it.  True keeps the
        pool's own behaviour, which is what a service that provisions tenants from its own
        traffic wants.  False confines the server to ``tenants`` plus whatever files already
        exist, so a principal with no tenant restriction cannot turn a stream of requests into
        a directory of database files.  It bounds resource use and nothing else: it is not a
        tenant boundary, and it does not change what an authorised caller may reach.

    On tenant isolation: with a :class:`anatid.DatabasePool` each tenant is its own file and the
    principal check sits on top of a real boundary.  With ``database=`` (one shared file) the
    tenants are namespaces in one file, which is scoping and not a security boundary, exactly as
    :class:`anatid.types.Isolation` says.  The server does not change that either way.
    """

    socket_path: str | os.PathLike[str] | None = None
    http_host: str | None = None
    http_port: int = 8787
    max_depth: int = DEFAULT_MAX_DEPTH
    batch_max: int = DEFAULT_BATCH_MAX
    workers: int = DEFAULT_WORKERS
    read_workers: int = 8
    idempotency: bool = True
    idempotency_ttl: float = DEFAULT_IDEMPOTENCY_TTL
    default_deadline: float | None = 30.0
    shutdown_timeout: float = 30.0
    max_frame_bytes: int = MAX_FRAME_BYTES
    max_header_bytes: int = 16 * 1024
    socket_mode: int = SOCKET_MODE
    socket_dir_mode: int = SOCKET_DIR_MODE
    embeddings: str = "list"
    checkpoint_on_shutdown: bool = True
    create_tenants: bool = True
    high_water: float = 0.8
    #: Tenants to open when the server starts.  Readiness is false until each of them has an
    #: open handle at the current schema version, so a load balancer does not send traffic to a
    #: process that has not finished migrating a file yet.
    tenants: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.socket_path is None and self.http_host is None:
            raise ValueError(
                "a server needs at least one transport: pass socket_path, http_host, or both"
            )
        if self.embeddings not in ("list", "f32"):
            raise ValueError(f"embeddings must be 'list' or 'f32', got {self.embeddings!r}")


@dataclass(frozen=True, slots=True)
class Health:
    """Is this process alive.  Nothing more, on purpose.

    A supervisor restarts on this.  It stays true while the server is draining, because a
    draining server is finishing work and restarting it would throw that work away.
    """

    ok: bool
    status: str
    pid: int
    uptime_s: float
    anatid_version: str
    protocol_version: int


@dataclass(frozen=True, slots=True)
class Readiness:
    """Can this process take traffic.

    True only when every configured tenant's file is open, every one of them is at the schema
    version this build expects, the queue is accepting, and no tenant's queue has reached its
    high-water mark.  A load balancer reads this.  A server that is merely busy answers false
    here and true on :class:`Health`, which is the distinction that keeps a busy server from
    being restarted and an unmigrated one from being sent traffic.
    """

    ready: bool
    accepting: bool
    open_files: int
    tenants: tuple[int, ...]
    migrations_done: bool
    schema_versions: dict[int, int] = field(default_factory=dict)
    expected_schema_version: int = SCHEMA_VERSION
    queues_below_high_water: bool = True
    max_tenant_depth: int = 0
    high_water: int = 0
    detail: str = ""

    def scoped_to(self, tenants: Iterable[int] | None) -> "Readiness":
        """This answer with every tenant the caller may not name taken out.

        The verdict is not a secret and the detail is.  ``ready`` says whether the process can
        take traffic, which is what a load balancer reads and which says nothing about tenants.
        The tenant list, the per-tenant schema versions and the open-file count all say which
        tenants exist.  :mod:`anatid.server.auth` and :mod:`anatid.server.metrics` both state the
        rule this keeps: a principal cannot learn whether a tenant it may not name exists.  An
        unrestricted principal passes ``None`` and gets the whole record; a scoped one gets its
        own tenants; an anonymous caller passes ``()`` and gets the verdict alone.

        The reasons are rebuilt from what is left rather than filtered as text, so a caller is
        never handed half a sentence naming somebody else's tenant.
        """
        if tenants is None:
            return self
        allowed = {int(t) for t in tenants}
        kept = tuple(t for t in self.tenants if t in allowed)
        versions = {t: v for t, v in self.schema_versions.items() if t in allowed}
        missing = [t for t in kept if t not in versions]
        behind = [t for t, v in versions.items() if v != self.expected_schema_version]
        reasons: list[str] = []
        if missing:
            reasons.append(f"no open file for tenants {missing}")
        if behind:
            reasons.append(f"schema below version {self.expected_schema_version} for {behind}")
        if not self.queues_below_high_water:
            reasons.append(
                f"a tenant queue is at or above the high-water mark of {self.high_water}"
            )
        if not self.accepting:
            reasons.append("the server is not accepting writes")
        if self.ready:
            detail = "ready"
        else:
            detail = "; ".join(reasons) or "not ready for a reason this principal cannot be told"
        return Readiness(
            ready=self.ready,
            accepting=self.accepting,
            open_files=len(versions),
            tenants=kept,
            migrations_done=self.migrations_done,
            schema_versions=versions,
            expected_schema_version=self.expected_schema_version,
            queues_below_high_water=self.queues_below_high_water,
            max_tenant_depth=self.max_tenant_depth,
            high_water=self.high_water,
            detail=detail,
        )


protocol.register_dataclass(Health)
protocol.register_dataclass(Readiness)


# --------------------------------------------------------------------------- the server


class AnatidServer:
    """One process owning one or many anatid files, answering verbs over a socket.

    Give it either a :class:`anatid.DatabasePool` (file per tenant, the isolated default) or a
    single :class:`anatid.Anatid` handle (one file, tenants as namespaces).  Not both.

    The public surface is small on purpose::

        server = AnatidServer(pool=pool, config=ServerConfig(socket_path="/run/anatid.sock"))
        await server.start()
        await server.serve_forever()  # until SIGTERM, or shutdown() from elsewhere
        report = await server.shutdown()

    :meth:`call` is the whole dispatcher and it is synchronous, so the tenant check, the verb
    table and the queue can be tested without a socket.
    """

    def __init__(
        self,
        *,
        pool: DatabasePool | None = None,
        database: Anatid | None = None,
        config: ServerConfig | None = None,
        authenticator: Authenticator | None = None,
        idempotency: IdempotencyStore | None = None,
    ) -> None:
        if (pool is None) == (database is None):
            raise ValueError(
                "pass exactly one of pool= (file per tenant, real isolation) or database= "
                "(one shared file, tenants as namespaces)"
            )
        self.config = config or ServerConfig(socket_path="anatid.sock")
        self.pool = pool
        self.database = database
        self.authenticator: Authenticator = authenticator or AllowAllAuthenticator()
        if self.config.http_host is not None:
            check_bind_address(self.config.http_host, self.authenticator)
        self.idempotency = idempotency or (
            IdempotencyStore(ttl=self.config.idempotency_ttl) if self.config.idempotency else None
        )
        self.queue = WriteQueue(
            self._open_db,
            max_depth=self.config.max_depth,
            batch_max=self.config.batch_max,
            workers=self.config.workers,
            idempotency=self.idempotency,
            high_water=self.config.high_water,
        )
        self._started_at = time.monotonic()
        self._status = "stopped"
        self._servers: list[asyncio.AbstractServer] = []
        self._connections: set[asyncio.Task[None]] = set()
        self._reads: Any = None
        self._stopped = asyncio.Event()
        self._shutdown_report: DrainReport | None = None
        self._shutdown_task: asyncio.Future[DrainReport] | None = None
        self._shutdown_budget = float("inf")
        self._signal_count = 0
        #: Frozen at construction so a request cannot be checked against a list it just changed.
        self._configured_tenants = frozenset(int(t) for t in self.config.tenants)

    # -- handles -----------------------------------------------------------------------

    def _open_db(self, tenant_id: int) -> Anatid:
        """The handle for ``tenant_id``, opening it if this is the first call.

        Called on worker and read threads, so it has to be thread-safe.
        ``DatabasePool.get`` is; the single-handle case has nothing to synchronise.

        It does no DDL.  The idempotency table is created by the write queue, before it opens a
        batch's transaction, so no read thread ever runs a CREATE on a file a worker thread is
        in the middle of writing.
        """
        db = self.pool.get(tenant_id) if self.pool is not None else self.database
        if db is None:  # unreachable: the constructor requires one of the two
            raise AnatidError("this server has no database handle")
        return db

    def open_tenants(self, tenants: Iterable[int] | None = None) -> list[int]:
        """Open the handles for ``tenants`` (default: the configured ones) and return them.

        Readiness is false until this has run, so a process that is still migrating a file is
        not sent traffic.
        """
        wanted = list(self.config.tenants if tenants is None else tenants)
        for tenant_id in wanted:
            self._open_db(int(tenant_id))
        return wanted

    def _open_handles(self) -> dict[int, Anatid]:
        if self.pool is not None:
            return {t: db for t, db in self.pool._open.items() if not db.closed}  # noqa: SLF001
        db = self.database
        return {} if db is None or db.closed else {db.namespace.tenant_id: db}

    # -- health and readiness ----------------------------------------------------------

    def health(self) -> Health:
        """Is this process alive.  True while serving AND while draining."""
        return Health(
            ok=self._status in ("serving", "draining"),
            status=self._status,
            pid=os.getpid(),
            uptime_s=time.monotonic() - self._started_at,
            anatid_version=_anatid_version,
            protocol_version=protocol.PROTOCOL_VERSION,
        )

    def readiness(self) -> Readiness:
        """Can this process take traffic: files open, migrations done, queues below high water."""
        handles = self._open_handles()
        wanted = tuple(self.config.tenants) or tuple(sorted(handles))
        versions: dict[int, int] = {}
        missing: list[int] = []
        behind: list[int] = []
        for tenant_id in wanted:
            db = handles.get(int(tenant_id))
            if db is None:
                missing.append(int(tenant_id))
                continue
            found = current_version(db.connection)
            versions[int(tenant_id)] = -1 if found is None else int(found)
            if found is None or int(found) != SCHEMA_VERSION:
                behind.append(int(tenant_id))
        stats = self.queue.stats()
        high_water = max(1, int(self.config.max_depth * self.config.high_water))
        below = not self.queue.above_high_water()
        accepting = self.queue.accepting and self._status == "serving"
        reasons: list[str] = []
        if missing:
            reasons.append(f"no open file for tenants {missing}")
        if behind:
            reasons.append(f"schema below version {SCHEMA_VERSION} for tenants {behind}")
        if not below:
            reasons.append(
                f"a tenant queue is at or above the high-water mark of {high_water} "
                f"({stats.max_tenant_depth} queued)"
            )
        if not accepting:
            reasons.append(f"the server is {self._status} and not accepting writes")
        return Readiness(
            ready=not reasons,
            accepting=accepting,
            open_files=len(handles),
            tenants=tuple(int(t) for t in wanted),
            migrations_done=not missing and not behind,
            schema_versions=versions,
            expected_schema_version=SCHEMA_VERSION,
            queues_below_high_water=below,
            max_tenant_depth=stats.max_tenant_depth,
            high_water=high_water,
            detail="ready" if not reasons else "; ".join(reasons),
        )

    # -- dispatch ----------------------------------------------------------------------

    def _plan(self, request: Request, principal: Principal) -> tuple[VerbSpec, dict[str, Any]]:
        """Validate one request and build the keyword arguments for its verb.

        This is where the tenant boundary is enforced, and it is enforced BEFORE any file is
        opened, so a request for a tenant the principal may not name never touches that tenant's
        file and cannot be told apart from a request for a tenant that does not exist.

        ``create_tenants=False`` is checked in the same place and for the same reason, but it is
        a different kind of refusal and is worded as one.  The principal check hides whether the
        tenant exists; this one is allowed to say the tenant does not, because the caller has
        already been found entitled to name it.
        """
        spec = VERBS.get(request.verb)
        if spec is None:
            raise ProtocolError(
                f"unknown verb {request.verb!r}. This server dispatches from a fixed table; "
                f"the verbs it has are {', '.join(sorted(VERBS))}."
            )
        tenant_id = principal.require(request.tenant)
        if spec.write:
            principal.require_write(spec.name)
        kwargs = dict(request.args)
        if "tenant" in kwargs:
            raise ProtocolError(
                "'tenant' belongs in the request envelope, not in args. The server takes it "
                "from the envelope, checks it against the connection's principal and passes it "
                "to the verb itself; accepting a second one in args would be two answers to the "
                "question of which tenant this call is for."
            )
        if spec.tenant_arg and not spec.server:
            if not self.config.create_tenants:
                self._require_existing_tenant(tenant_id)
            kwargs["tenant"] = tenant_id
        return spec, kwargs

    def _require_existing_tenant(self, tenant_id: int) -> None:
        """Refuse a tenant this server would have to create, when it is configured not to.

        A configured tenant always passes, and so does one whose file is already on disk or
        already open, so restarting a server does not lose the tenants it was serving.  The
        check is a ``stat``, not an open: it must not create the file it is deciding about.
        """
        if tenant_id in self._configured_tenants:
            return
        pool = self.pool
        if pool is None:
            # One shared file.  Every tenant is a namespace inside it, so there is no file to
            # create and the only meaning left for this setting is the configured list.
            raise TenantIsolationError(
                f"tenant {tenant_id} is not one this server serves. It was started on a single "
                f"shared database with create_tenants=False, so the tenants it answers for are "
                f"exactly the ones it was configured with."
            )
        if tenant_id in pool._open:  # noqa: SLF001 - reopening a live handle creates nothing
            return
        try:
            path = pool.path_for(tenant_id)
        except Exception as exc:  # a label this pool cannot turn into a file name
            raise TenantIsolationError(
                f"tenant {tenant_id} has no file on this server: {exc}"
            ) from exc
        if path.exists():
            return
        raise TenantIsolationError(
            f"tenant {tenant_id} has no database file on this server, and it was started with "
            f"create_tenants=False, so it will not make one. Create the tenant out of band, or "
            f"list it in ServerConfig(tenants=...), or start the server with create_tenants=True."
        )

    def _server_verb(self, spec: VerbSpec, principal: Principal) -> Any:
        """The three verbs the server answers about itself rather than about a tenant.

        ``ready`` is scoped to the principal for the reason :meth:`Readiness.scoped_to` gives: the
        full record names every tenant this process holds, and a principal that may name one of
        them must not learn about the others from a probe.  ``health`` says nothing about
        tenants.  ``queue_stats`` is left whole deliberately: it is an operator verb, and a
        principal that may not have it should not have been given it.
        """
        if spec.name == "health":
            return self.health()
        if spec.name == "ready":
            return self.readiness().scoped_to(
                None if principal.tenants is None else sorted(principal.tenants)
            )
        if spec.name == "queue_stats":
            return self.queue.stats()
        raise ProtocolError(f"no server handler for {spec.name!r}")

    def run_read(self, request: Request, principal: Principal) -> Any:
        """Run one read verb on this thread and return its result."""
        spec, kwargs = self._plan(request, principal)
        if spec.server:
            return self._server_verb(spec, principal)
        if spec.write:
            raise ProtocolError(f"{spec.name} is a write; it goes through the queue")
        db = self._open_db(request.tenant)
        return getattr(db, spec.name)(**kwargs)

    def submit_write(self, request: Request, principal: Principal) -> Any:
        """Queue one write verb and return its future.

        Raises :class:`~anatid.server.protocol.BusyError` when the tenant's queue is full.  The
        write is NOT performed in that case, which is what makes the busy answer safe to retry.
        """
        spec, kwargs = self._plan(request, principal)
        if not spec.write:
            raise ProtocolError(f"{spec.name} is a read; it does not go through the queue")
        method = spec.name

        def work(db: Anatid) -> Any:
            return getattr(db, method)(**kwargs)

        digest = None
        if request.idempotency_key is not None and self.idempotency is not None:
            digest = IdempotencyStore.digest_for(spec.name, request.args)
        return self.queue.submit(
            request.tenant,
            spec.name,
            work,
            idempotency_key=request.idempotency_key,
            digest=digest,
            batchable=spec.batchable,
            deadline=self._deadline(request),
        )

    def _deadline(self, request: Request) -> float | None:
        return request.deadline if request.deadline is not None else self.config.default_deadline

    def call(self, request: Request, principal: Principal) -> Response:
        """Dispatch one request and return its response, blocking for a write's turn.

        The synchronous whole of the server's behaviour.  Every transport calls it (the asyncio
        ones through :meth:`acall`, which keeps the blocking part off the event loop), and a test
        can call it with a hand-built :class:`~anatid.server.protocol.Request` and
        :class:`~anatid.server.auth.Principal` and get the same dispatch, the same tenant check
        and the same queue as a real client.
        """
        try:
            spec = VERBS.get(request.verb)
            if spec is not None and spec.write:
                future = self.submit_write(request, principal)
                timeout = self._deadline(request)
                try:
                    result = future.result(timeout=timeout)
                # concurrent.futures.TimeoutError only became an alias of the builtin in 3.11,
                # and anatid supports 3.10, so both names are caught.
                except (TimeoutError, concurrent.futures.TimeoutError) as exc:
                    future.cancel()
                    raise DeadlineExceeded(
                        f"this {request.verb} did not finish within {timeout}s. It may still "
                        f"run: the deadline is checked before the transaction opens, not "
                        f"during it. Retry with the same idempotency key.",
                        deadline=timeout,
                    ) from exc
            else:
                result = self.run_read(request, principal)
        except Exception as exc:  # every error becomes a structured response, not a dropped call
            self._log_failure(request, principal, exc)
            return Response.failure(exc, request_id=request.request_id)
        return Response.ok(result, request_id=request.request_id)

    async def acall(self, request: Request, principal: Principal) -> Response:
        """:meth:`call` without blocking the event loop.

        A read goes to the read thread pool.  A write is queued on this thread -- which is fast
        and never blocks, because :meth:`~anatid.server.queue.WriteQueue.submit` answers busy
        rather than waiting -- and the future is awaited.
        """
        loop = asyncio.get_running_loop()
        try:
            spec = VERBS.get(request.verb)
            if spec is not None and spec.write:
                future = self.submit_write(request, principal)
                timeout = self._deadline(request)
                try:
                    result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout)
                except (TimeoutError, asyncio.TimeoutError) as exc:
                    future.cancel()
                    raise DeadlineExceeded(
                        f"this {request.verb} did not finish within {timeout}s. It may still "
                        f"run: the deadline is checked before the transaction opens, not "
                        f"during it. Retry with the same idempotency key.",
                        deadline=timeout,
                    ) from exc
            elif spec is not None and spec.server:
                result = self.run_read(request, principal)
            else:
                result = await loop.run_in_executor(self._reads, self.run_read, request, principal)
        except Exception as exc:
            self._log_failure(request, principal, exc)
            return Response.failure(exc, request_id=request.request_id)
        return Response.ok(result, request_id=request.request_id)

    def _render(self, response: Response) -> tuple[Response, bytes]:
        """``response`` and its frame body, with an encoding failure turned into a reply.

        A verb can return a value this build's codec has no shape for.  That is a defect in the
        server, not in the request, and the only honest way to report it is a response the client
        can read.  Letting the encoder raise into the connection loop would drop the connection
        with no reply at all, which leaves the client waiting for the answer to a call the server
        has already run: for a write that means a retry it cannot know is a retry.  So the
        failure is encoded instead, and it names the type, because that is the one piece of
        information that turns the report into a fix.
        """
        try:
            return response, protocol.dumps(response.to_wire(), embeddings=self.config.embeddings)
        except Exception as exc:
            log.exception("could not encode the reply to request %s", response.request_id)
            failure = Response.failure(
                ProtocolError(
                    f"the server ran this verb but could not encode what it returned: {exc}"
                ),
                request_id=response.request_id,
            )
            # WireError carries only scalars, so this second encode cannot fail the same way.
            return failure, protocol.dumps(failure.to_wire())

    def _log_failure(self, request: Request, principal: Principal, exc: BaseException) -> None:
        if isinstance(exc, (BusyError, ShuttingDown)):
            level = logging.INFO
        elif isinstance(exc, (AuthorizationError, AuthenticationError, ProtocolError)):
            level = logging.WARNING
        else:
            level = logging.INFO
        log.log(
            level,
            "%s for tenant %s from %s failed: %s: %s",
            request.verb,
            request.tenant,
            principal.name,
            type(exc).__name__,
            exc,
        )

    # -- lifecycle ---------------------------------------------------------------------

    async def start(self) -> "AnatidServer":
        """Open the configured tenants, start the queue and bind the listeners.

        Starting a server that has been shut down works, and the shutdown state is cleared here
        rather than left behind: a finished shutdown task that outlived its server would make the
        next ``shutdown()`` return the previous run's report without stopping anything.
        """
        from concurrent.futures import ThreadPoolExecutor

        if self._status == "serving":
            return self
        self._reads = ThreadPoolExecutor(
            max_workers=self.config.read_workers, thread_name_prefix="anatid-read"
        )
        self.open_tenants()
        self.queue.start()
        self._status = "serving"
        self._stopped = asyncio.Event()
        self._shutdown_task = None
        self._shutdown_report = None
        self._shutdown_budget = float("inf")
        self._signal_count = 0
        if self.config.socket_path is not None:
            self._servers.append(await self._start_unix())
        if self.config.http_host is not None:
            self._servers.append(await self._start_http())
        log.info(
            "anatid server %s serving on %s (pid %d)",
            _anatid_version,
            ", ".join(self.endpoints()),
            os.getpid(),
        )
        return self

    def endpoints(self) -> list[str]:
        """Human-readable addresses this server is listening on."""
        out: list[str] = []
        if self.config.socket_path is not None:
            out.append(f"unix:{self.config.socket_path}")
        if self.config.http_host is not None:
            out.append(f"http://{self.config.http_host}:{self.config.http_port}")
        return out

    def _prepare_socket_path(self) -> str:
        """Make the directory, set 0700 on it, and clear a stale socket.  Filesystem work, so
        it is synchronous and :meth:`_start_unix` calls it rather than doing it in a coroutine.

        Clearing a stale socket is safe because it is only ever the path of a server that is not
        running: two servers on one path is a configuration error, and unlinking the file does
        not disturb a live listener that already has it open, so the second one fails to bind
        loudly rather than silently stealing the address.
        """
        path = Path(os.fspath(self.config.socket_path or "anatid.sock")).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(path.parent, self.config.socket_dir_mode)
        if path.exists() or path.is_symlink():
            with contextlib.suppress(OSError):
                path.unlink()
        return os.fspath(path)

    async def _start_unix(self) -> asyncio.AbstractServer:
        path = self._prepare_socket_path()
        if not hasattr(socket, "AF_UNIX"):
            raise UnsupportedTransport(
                "Unix domain sockets are not available on this platform. Start the server with an HTTP "
                'listener on the loopback interface instead, ServerConfig(http_host="127.0.0.1", '
                "http_port=8787) with a bearer token, and point clients at http://127.0.0.1:8787."
            )
        server = await asyncio.start_unix_server(self._connection_handler("unix"), path=path)
        with contextlib.suppress(OSError):
            os.chmod(path, self.config.socket_mode)
        return server

    async def _start_http(self) -> asyncio.AbstractServer:
        host = self.config.http_host or "127.0.0.1"
        check_bind_address(host, self.authenticator)
        return await asyncio.start_server(self._http_handler, host=host, port=self.config.http_port)

    def _connection_handler(
        self, transport: str
    ) -> Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]:
        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            task = asyncio.current_task()
            if task is not None:
                self._connections.add(task)
            try:
                await self._serve_frames(reader, writer, transport)
            finally:
                if task is not None:
                    self._connections.discard(task)

        return handler

    async def _serve_frames(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, transport: str
    ) -> None:
        """One connection's lifetime: authenticate once, then answer frames until it closes.

        Requests on one connection are answered in order.  A client that wants concurrency opens
        another connection, which is cheap; interleaving replies on one connection would need
        the client to match on ``request_id`` and would buy nothing a second connection does not.
        """
        sock = writer.get_extra_info("socket")
        ctx = ConnectionContext(
            transport=transport, socket=sock, peer=writer.get_extra_info("peername")
        )
        try:
            principal = self.authenticator.authenticate(ctx)
        except Exception as exc:
            log.warning("refused a %s connection: %s", transport, exc)
            with contextlib.suppress(Exception):
                await protocol.write_frame_async(
                    writer, Response.failure(exc), embeddings=self.config.embeddings
                )
            await _close(writer)
            return
        log.debug("%s connection from %s as %s", transport, ctx.peer, principal.name)
        try:
            while True:
                body = await protocol.read_frame_async(
                    reader, max_bytes=self.config.max_frame_bytes
                )
                if body is None:
                    return
                try:
                    request = Request.decode(body)
                except (ProtocolError, FrameError) as exc:
                    await protocol.write_frame_async(
                        writer, Response.failure(exc), embeddings=self.config.embeddings
                    )
                    if isinstance(exc, FrameError):
                        return  # the stream is no longer synchronisable
                    continue
                response = await self.acall(request, principal)
                _, body_out = self._render(response)
                writer.write(protocol.pack_frame(body_out))
                await writer.drain()
        except FrameError as exc:
            log.info("closing a %s connection: %s", transport, exc)
        except (ConnectionResetError, BrokenPipeError):
            log.debug("%s connection reset", transport)
        except asyncio.CancelledError:
            raise
        finally:
            await _close(writer)

    # -- HTTP --------------------------------------------------------------------------

    async def _http_handler(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """A deliberately small HTTP/1.1 surface: POST /rpc, GET /health, GET /ready.

        Written directly against asyncio rather than on a framework so the one rule that matters
        -- no non-loopback bind without a token -- is enforced here and cannot be undone by a
        middleware ordering somewhere else.
        """
        task = asyncio.current_task()
        if task is not None:
            self._connections.add(task)
        try:
            while True:
                head = await _read_headers(reader, self.config.max_header_bytes)
                if head is None:
                    return
                method, target, headers = head
                length = _content_length(headers, self.config.max_frame_bytes)
                body = await reader.readexactly(length) if length else b""
                keep_alive = headers.get("connection", "keep-alive").lower() != "close"
                status, payload = await self._http_route(method, target, headers, body)
                await _write_http(writer, status, payload, keep_alive)
                if not keep_alive:
                    return
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("http connection failed")
        finally:
            if task is not None:
                self._connections.discard(task)
            await _close(writer)

    def _readable_tenants(self, headers: Mapping[str, str]) -> tuple[int, ...] | None:
        """The tenants these headers may be told about: None for all, ``()`` for none.

        Used by the probes, which answer the same verdict to everyone and differ only in how much
        detail they attach.  ``/rpc`` does not go through here: it needs the principal itself and
        it reports the refusal rather than quietly answering less.
        """
        try:
            principal = self.authenticator.authenticate(
                ConnectionContext(transport="http", socket=None, headers=headers)
            )
        except Exception:
            return ()
        allowed = getattr(principal, "tenants", None)
        return None if allowed is None else tuple(sorted(allowed))

    async def _http_route(
        self, method: str, target: str, headers: Mapping[str, str], body: bytes
    ) -> tuple[int, bytes]:
        path = target.split("?", 1)[0]
        if method == "GET" and path == "/health":
            # Open, deliberately.  It carries a pid, an uptime and a version and nothing about
            # tenants, and a supervisor has to be able to ask it without holding a credential.
            health = self.health()
            return (200 if health.ok else 503), protocol.dumps(health)
        if method == "GET" and path == "/ready":
            ready = self.readiness()
            # The verdict is open, for the load balancer that has to read it, and the detail is
            # not: the tenant list and the per-tenant schema versions in a full Readiness say
            # which tenants exist, which is the one thing the tenant boundary refuses to tell a
            # principal.  A caller that authenticates gets the whole record.
            if getattr(self.authenticator, "requires_token", False):
                ready = ready.scoped_to(self._readable_tenants(headers))
            return (200 if ready.ready else 503), protocol.dumps(ready)
        if method != "POST" or path != "/rpc":
            return 404, protocol.dumps({"error": f"no route for {method} {path}"})
        ctx = ConnectionContext(transport="http", socket=None, headers=headers)
        try:
            principal = self.authenticator.authenticate(ctx)
        except Exception as exc:
            return 401, protocol.dumps(Response.failure(exc).to_wire())
        try:
            request = Request.decode(body)
        except (ProtocolError, FrameError) as exc:
            return 400, protocol.dumps(Response.failure(exc).to_wire())
        rendered, body_out = self._render(await self.acall(request, principal))
        return _http_status(rendered), body_out

    # -- shutdown ----------------------------------------------------------------------

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Make SIGTERM and SIGINT start a graceful shutdown.

        A second signal is not ignored and does not repeat the first: it abandons the drain,
        because the operator who sent it has decided that waiting is worse than losing what is
        queued, and pretending to shut down twice gracefully is how a process ends up killed by
        a supervisor mid-transaction instead.
        """
        loop = loop or asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.add_signal_handler(sig, self._on_signal, sig)

    def _on_signal(self, sig: "signal.Signals") -> None:
        self._signal_count += 1
        if self._signal_count == 1:
            log.info("%s: draining", sig.name)
            asyncio.ensure_future(self.shutdown())  # noqa: RUF006 - kept alive by the loop
        else:
            log.warning("%s again: abandoning the drain", sig.name)
            asyncio.ensure_future(self.shutdown(timeout=0.0))  # noqa: RUF006

    async def serve_forever(self) -> DrainReport:
        """Block until :meth:`shutdown` runs, and return what it drained."""
        self.install_signal_handlers()
        await self._stopped.wait()
        return self._shutdown_report or DrainReport(0, 0, False, 0.0)

    async def shutdown(self, *, timeout: float | None = None) -> DrainReport:
        """Stop accepting, drain the queues, release the files, exit, inside ``timeout``.

        The order is the point, and it is not the order it looks like.  The listeners are told
        to close, and then the queue stops accepting IMMEDIATELY, so a request already decoded
        gets a clear :class:`~anatid.server.protocol.ShuttingDown` rather than being committed by
        a server that has announced it is going away.  Then the queues drain under ``timeout``.
        Then the open connections are cancelled.  Only then is the listener waited on, and even
        that wait is bounded.

        What the order avoids is worth naming, because the obvious arrangement has a bug.
        ``asyncio.Server.wait_closed`` does not return while a connection handler is still
        running, and a live deployment always has connections open, so waiting on it first makes
        the whole shutdown last as long as the last client chooses to stay connected.  Measured
        on Python 3.12 before this was reordered: one idle client held a server past 25 seconds
        against a 5 second budget and it exited 0.03 seconds after that socket closed, and 64
        writing clients had about 2,100 further writes accepted and committed AFTER the SIGTERM.

        Then the read pool is joined, so nothing is still touching a file, and every open file is
        released, which folds its write-ahead log in and gives up DuckDB's exclusive lock so the
        next process can open it at all.  Whatever the drain could not finish is completed with
        ``ShuttingDown`` and named per tenant in the report.

        Calling it again while it is running does not start a second one: the second caller
        awaits the first, and a second call with a shorter budget (``timeout=0``, which is what a
        second SIGTERM sends) abandons the drain in progress rather than waiting behind it.
        Calling it after it has run is a no-op that returns the same report.  Starting the server
        again works: the pool reopens each tenant's file on the next call for it.
        """
        if self._status == "stopped":
            return self._shutdown_report or DrainReport(0, 0, False, 0.0)
        budget = self.config.shutdown_timeout if timeout is None else float(timeout)
        running = self._shutdown_task
        if running is not None:
            if budget < self._shutdown_budget:
                # A shorter budget than the one already being spent is an escalation, and the
                # drain it has to shorten is happening on a worker thread with its own deadline
                # fixed.  Telling the queue to give up is the only thing that reaches it.
                self._shutdown_budget = budget
                log.warning("shutdown escalated to a %.3fs budget; abandoning the drain", budget)
                self.queue.abandon()
                self._cancel_connections()
            return await asyncio.shield(running)
        self._shutdown_budget = budget
        self._shutdown_task = asyncio.ensure_future(self._shutdown(budget))
        return await asyncio.shield(self._shutdown_task)

    def _cancel_connections(self) -> None:
        for task in list(self._connections):
            task.cancel()

    async def _shutdown(self, budget: float) -> DrainReport:
        """The body of :meth:`shutdown`, run once as a task so a second caller can join it."""
        self._status = "draining"
        started = time.monotonic()
        for server in self._servers:
            server.close()
        # Before anything is waited on.  This is what makes the ShuttingDown the docstring
        # promises actually reach a client, instead of its write being committed by a server
        # that is supposed to be leaving.
        self.queue.stop_accepting()
        report = await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.queue.stop(timeout=budget, drain=budget > 0)
        )
        self._cancel_connections()
        if self._connections:
            await asyncio.gather(*self._connections, return_exceptions=True)
        self._connections.clear()
        await self._close_listeners(budget - (time.monotonic() - started))
        # The read pool goes down BEFORE the checkpoint.  A checkpoint waits for every other
        # transaction on the file, and a read thread that is still running one would make it
        # wait; joining the pool first bounds the wait to work that was already in flight.
        if self._reads is not None:
            self._reads.shutdown(wait=True, cancel_futures=True)
            self._reads = None
        if self.config.checkpoint_on_shutdown:
            self._checkpoint_all()
        self._remove_socket()
        self._status = "stopped"
        self._shutdown_report = report
        self._stopped.set()
        if report.abandoned:
            log.warning(
                "shutdown abandoned %d queued writes: %s",
                report.abandoned,
                report.abandoned_by_tenant,
            )
        else:
            log.info("shutdown drained %d writes cleanly", report.drained)
        return report

    async def _close_listeners(self, remaining: float) -> None:
        """Let the listeners report themselves closed, under a bound, and let go either way.

        This is the wait that used to be first and unbounded.  It is last because a listener is
        not closed until its connections are, and it is bounded because a client that will not
        let go of a socket is not a reason for a process under SIGTERM to stay alive.  Python
        3.13 grew ``close_clients`` and ``abort_clients`` for exactly this; on 3.12 the
        cancellation in :meth:`_shutdown` is what releases the handlers, and this is the wait for
        their transports to finish closing.
        """
        grace = max(_MIN_LISTENER_GRACE, min(remaining, _MAX_LISTENER_GRACE))
        for server in self._servers:
            closer = getattr(server, "close_clients", None)
            if closer is not None:
                with contextlib.suppress(Exception):
                    closer()
        for server in self._servers:
            try:
                await asyncio.wait_for(server.wait_closed(), timeout=grace)
            except (TimeoutError, asyncio.TimeoutError):
                aborter = getattr(server, "abort_clients", None)
                if aborter is not None:
                    with contextlib.suppress(Exception):
                        aborter()
                log.warning(
                    "a listener still had a connection open %.3fs after every handler was "
                    "cancelled; closing anyway",
                    grace,
                )
            except Exception as exc:  # a listener that fails to close is not worth hanging on
                log.debug("listener close reported %s: %s", type(exc).__name__, exc)
        self._servers = []

    def _checkpoint_all(self) -> None:
        """Fold every open write-ahead log into its file, by releasing the file.

        Closing the handle is the checkpoint here, and it is also the right thing to do on its
        own account: a stopped server that still holds a tenant's file holds an EXCLUSIVE lock on
        it, so the replacement process cannot open that file even read-only.  Releasing it is
        what makes a restart in place work.  ``DatabasePool.get`` reopens on demand, so a server
        that is started again after this picks the files back up.

        It is not ``db.execute("CHECKPOINT")``, and that is measured rather than assumed.  On
        duckdb 1.5.5, a handle that ``Anatid.open`` returned for a file that ALREADY EXISTED
        cannot checkpoint through ``db.execute`` at all: the statement raises
        ``TransactionException: Cannot CHECKPOINT: there are other write transactions active``,
        from any thread, on a handle that has run nothing else.  :meth:`anatid.Anatid.checkpoint`
        exists because of that: it issues the statement on the handle's root connection, where it
        works in both cases.  Even so, releasing is what this method wants, because a stopped
        server that still holds the file holds the lock on it.

        A server built on ``database=`` gets :meth:`anatid.Anatid.checkpoint` and nothing more.
        That handle belongs to the caller, who may still be using it, so this does not close it;
        the caller's own ``close()`` gives up the lock, and the file is not free for another
        process until then either way.
        """
        if self.pool is not None:
            for tenant_id in list(self._open_handles()):
                try:
                    self.pool.close(tenant_id)
                except Exception as exc:
                    log.warning("could not release tenant %s: %s", tenant_id, exc)
            return
        for tenant_id, db in self._open_handles().items():
            try:
                db.checkpoint()
            except Exception as exc:
                log.info(
                    "tenant %s was not checkpointed (%s); this handle belongs to the caller and "
                    "closing it folds the log",
                    tenant_id,
                    exc,
                )

    def _remove_socket(self) -> None:
        if self.config.socket_path is None:
            return
        with contextlib.suppress(OSError):
            Path(os.fspath(self.config.socket_path)).unlink()

    async def backup_tenant(
        self, tenant_id: int, destination: str | os.PathLike[str], *, quiesce: bool = True
    ) -> "BackupReport":
        """Copy one tenant's file and say what the copy promises.

        This is :class:`anatid.server.backup.BackupCoordinator` with one tenant named, and the
        report it returns carries the guarantee, the size, how long the tenant was paused and
        what schema version came back.

        It used to be a drain followed by a copy, and that was weaker than it read.
        ``WriteQueue.drain_tenant`` waits for the tenant's queue to empty and then RETURNS,
        stopping nothing, so a write submitted between the drain returning and the copy starting
        commits into the copy.  The boundary that describes is
        :attr:`~anatid.server.backup.Guarantee.SNAPSHOT`, not the "everything acknowledged so
        far" line the drain implied, and under continuous writes the drain could spend its whole
        budget and still only deliver the weaker one.

        ``quiesce=True`` instead holds a barrier: a non-batchable item takes the tenant's single
        serving slot for the length of the copy, so no other write for that tenant can run while
        it is held.  That is :attr:`~anatid.server.backup.Guarantee.QUIESCED`, and it costs about
        9 ms over the online copy on a 42 MiB file.  Other tenants are not paused at any point.
        ``quiesce=False`` pauses nothing and states SNAPSHOT.

        The copy itself is DuckDB's ``COPY FROM DATABASE`` from inside the handle that owns the
        file, which is already a consistent snapshot of committed data, so there is no checkpoint
        here: it would add nothing, and on a handle opened for a file that already existed
        ``CHECKPOINT`` raises rather than running (see :meth:`_checkpoint_all`), which would turn
        a working backup into a failed one.
        """
        if self.pool is None:
            raise AnatidError(
                "backup_tenant needs a DatabasePool; a single shared file has no per-tenant file "
                "to copy"
            )
        # Imported here and not at module scope: backup.py takes an AnatidServer, so importing
        # it from this module at import time is a cycle.
        from .backup import BackupCoordinator

        coordinator = BackupCoordinator(self, acquire_timeout=self.config.shutdown_timeout)
        result = await coordinator.abackup(
            destination, tenant=int(tenant_id), quiesce=quiesce, verify=False
        )
        return result.one

    async def __aenter__(self) -> "AnatidServer":
        return await self.start()

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.shutdown()


# --------------------------------------------------------------------------- helpers


async def _close(writer: asyncio.StreamWriter) -> None:
    with contextlib.suppress(Exception):
        writer.close()
        await writer.wait_closed()


def _http_status(response: Response) -> int:
    if response.status is Status.OK:
        return 200
    if response.status is Status.BUSY:
        return 429
    error = response.error
    name = "" if error is None else error.error_class
    if name == "AuthenticationError":
        return 401
    if name in ("AuthorizationError", "TenantIsolationError"):
        return 403
    if name == "NotFoundError":
        return 404
    if name in (
        "ProtocolError",
        "ValidationError",
        "RangeError",
        "IdempotencyConflict",
        "BackupDestinationExists",
    ):
        return 400
    if name == "ConflictError":
        return 409
    if name in ("DeadlineExceeded", "ShuttingDown", "QuiesceTimeout", "QuiesceUnavailable"):
        return 503
    if name == "DestinationInUse":
        return 409
    return 500


async def _read_headers(
    reader: asyncio.StreamReader, limit: int
) -> tuple[str, str, dict[str, str]] | None:
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except asyncio.IncompleteReadError:
        return None
    except asyncio.LimitOverrunError as exc:
        raise ProtocolError("HTTP headers are longer than this server accepts") from exc
    if len(head) > limit:
        raise ProtocolError(f"HTTP headers are {len(head)} bytes, over the {limit} byte limit")
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) < 2:
        raise ProtocolError(f"malformed HTTP request line {lines[0]!r}")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return parts[0].upper(), parts[1], headers


def _content_length(headers: Mapping[str, str], limit: int) -> int:
    raw = headers.get("content-length", "0")
    try:
        length = int(raw)
    except ValueError as exc:
        raise ProtocolError(f"content-length {raw!r} is not a number") from exc
    if length < 0 or length > limit:
        raise ProtocolError(f"content-length {length} is outside 0..{limit}")
    return length


async def _write_http(
    writer: asyncio.StreamWriter, status: int, body: bytes, keep_alive: bool
) -> None:
    reason = {
        200: "OK",
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        409: "Conflict",
        429: "Too Many Requests",
        500: "Internal Server Error",
        503: "Service Unavailable",
    }.get(status, "OK")
    head = (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"content-type: application/json\r\n"
        f"content-length: {len(body)}\r\n"
        f"connection: {'keep-alive' if keep_alive else 'close'}\r\n\r\n"
    ).encode("latin-1")
    writer.write(head + body)
    await writer.drain()


async def serve(
    *,
    pool: DatabasePool | None = None,
    database: Anatid | None = None,
    config: ServerConfig | None = None,
    authenticator: Authenticator | None = None,
) -> DrainReport:
    """Start a server, serve until SIGTERM, and return what the shutdown drained.

    The whole of a normal ``python -m`` entry point::

        asyncio.run(serve(pool=pool, config=ServerConfig(socket_path="/run/anatid/anatid.sock")))
    """
    server = AnatidServer(pool=pool, database=database, config=config, authenticator=authenticator)
    await server.start()
    return await server.serve_forever()


def connect_unix(path: str | os.PathLike[str], *, timeout: float | None = 30.0) -> "socket.socket":
    """A blocking client socket connected to a server's Unix socket.

    Here so a test, a shell script or a small synchronous client does not have to reimplement
    the three lines, and so the framing helpers in :mod:`anatid.server.protocol` have something
    obvious to be used with::

        sock = connect_unix("/run/anatid/anatid.sock")
        sock.sendall(Request(verb="get", tenant=1, args={"memory_id": 7}).encode())
        Response.decode(read_frame(sock)).raise_for_status()
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if timeout is not None:
        sock.settimeout(timeout)
    sock.connect(os.fspath(path))
    return sock


__all__.append("connect_unix")
