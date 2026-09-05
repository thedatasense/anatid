"""The anatid server profile: one process owns the files, everybody else talks to it.

Why this exists
---------------
``anatid.Anatid`` is one writer PROCESS with many writer THREADS, and that is not a defect: it
is the deployment profile the embedded database was designed for, it stays exactly as it is, and
nothing here replaces it.  What it cannot do is let two processes write one memory.

DuckDB decides that, and it decides it harder than the documentation implies.  Measured on
duckdb 1.5.5 (macOS arm64), a process holding a database file read-write excludes every other
process from the file, including a process that only wants to READ:

    holder=read-write  second=read-write  -> duckdb.IOException, "Could not set lock on file"
    holder=read-write  second=read-only   -> duckdb.IOException, "Could not set lock on file"
    holder=read-only   second=read-write  -> duckdb.IOException, "Could not set lock on file"
    holder=read-only   second=read-only   -> ok

So there is no arrangement in which clients write through a server and read the file directly.
The server owns the files; every read and every write comes back over the protocol, and the cost
of that lands on every call.  Measured on a 3,001-memory tenant with 384-dimension embeddings,
p50, in process against over a Unix socket:

    get()                        0.49 ms -> 1.01 ms   (+0.52 ms, 2.07x)
    remember()                   2.83 ms -> 4.24 ms   (+1.40 ms, 1.50x)
    recall(), vector arm on     25.38 ms -> 24.99 ms  (no measurable cost)
    recall(), no vector arm     17.09 ms -> 19.12 ms  (+2.03 ms, 1.12x)

So the honest headline is about half a millisecond of fixed cost per call.  That doubles the
cheapest read anatid has and disappears entirely into a recall, which is the call this profile
exists to serve.  A bare Unix-socket echo is 7.5 us, and quoting that number as the cost of the
server would be a lie by a factor of seventy: the round trip is mostly the JSON codec (260 us of
it for a reply carrying a 384-float embedding) and the event loop, not the socket.  Sending
embeddings as base64 float32 (``ServerConfig(embeddings="f32")``) removes most of the codec half.

What it is not
--------------
Not a storage abstraction.  There is one engine, DuckDB, and there is no pluggable backend for
Neo4j or Postgres, because a client over someone else's engine would give up the measured reason
this architecture exists (2-hop recall 2.5x to 3.6x faster than the maintained Kuzu fork).

Not serializable.  DuckDB gives optimistic snapshot isolation with write-write aborts, and
routing writes through one process does not upgrade that.  What the server adds is that the
aborts now happen between threads inside one process, where the queue can serialise per tenant
and batch, instead of between processes, where they could not happen at all because the second
process could not open the file.

The pieces
----------
``protocol``
    The wire: length-prefixed JSON, versioned, with a codec that round-trips every value type
    the verbs return and a structured error that keeps ``ConflictError``'s version fields.
``queue``
    Per-tenant write queues: bounded depth with an explicit busy response, batching into one
    transaction where the verbs allow it, round-robin fairness so a hot tenant cannot starve a
    quiet one, and persisted idempotency keys.
``auth``
    Who a connection is, and which tenants it may name.  Unix sockets authenticate by
    filesystem permissions and peer credentials; HTTP requires a bearer token and refuses to
    bind a non-loopback interface without one.
``server``
    The process: transports, dispatch, tenant enforcement, signal handling, draining, health
    and readiness.
``backup``
    Copying a file the server is holding open, which is the only process that can.  A barrier on
    the tenant's own write queue gives a copy a boundary in that tenant's write order, and every
    path states what it promises rather than leaving the caller to assume.
``client``
    The drop-in replacement for an ``Anatid`` handle, over the wire.
``cli``
    ``anatid-server start|stop|status|backup|restore|doctor``.
``metrics``
    Optional instrumentation: ``attach(server)`` adds ``/metrics``, a ``metrics`` verb and a
    structured per-request log.  Nothing else imports it, so a server that does not want it does
    not pay for it.
"""

from __future__ import annotations

from .auth import (
    AllowAllAuthenticator,
    Authenticator,
    BearerTokenAuthenticator,
    ConnectionContext,
    PeerCredentials,
    Principal,
    UnixPeerAuthenticator,
    check_bind_address,
    is_loopback,
    peer_credentials,
)
from .protocol import (
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    AuthenticationError,
    AuthorizationError,
    BusyError,
    DeadlineExceeded,
    FrameError,
    IdempotencyConflict,
    ProtocolError,
    RemoteError,
    Request,
    Response,
    ShuttingDown,
    Status,
    WireError,
)
from .queue import (
    DrainReport,
    IdempotencyRecord,
    IdempotencyStore,
    QueuedWrite,
    QueueStats,
    WriteQueue,
)
from .backup import (
    BackupCoordinator,
    BackupError,
    BackupInfo,
    BackupReport,
    BackupSet,
    BackupUnreadable,
    DestinationInUse,
    Guarantee,
    PruneReport,
    QuiesceTimeout,
    QuiesceUnavailable,
    RestoreReport,
    in_use,
)
from .backup import inspect as read_backup  # `inspect` is a stdlib module at package level
from .backup import prune as prune_backups  # `prune` is also a verb on Anatid; this is files
from .backup import restore
from .client import AnatidClient, ServerUnavailable, connect
from .server import (
    AnatidServer,
    Health,
    Readiness,
    ServerConfig,
    VerbSpec,
    VERBS,
    serve,
)

__all__ = [
    "PROTOCOL_VERSION",
    "MAX_FRAME_BYTES",
    "Request",
    "Response",
    "Status",
    "WireError",
    "ProtocolError",
    "FrameError",
    "RemoteError",
    "BusyError",
    "DeadlineExceeded",
    "AuthenticationError",
    "AuthorizationError",
    "IdempotencyConflict",
    "ShuttingDown",
    "Principal",
    "PeerCredentials",
    "ConnectionContext",
    "Authenticator",
    "AllowAllAuthenticator",
    "UnixPeerAuthenticator",
    "BearerTokenAuthenticator",
    "peer_credentials",
    "is_loopback",
    "check_bind_address",
    "WriteQueue",
    "QueuedWrite",
    "QueueStats",
    "DrainReport",
    "IdempotencyStore",
    "IdempotencyRecord",
    "AnatidServer",
    "ServerConfig",
    "VerbSpec",
    "VERBS",
    "Health",
    "Readiness",
    "serve",
    "AnatidClient",
    "ServerUnavailable",
    "connect",
    "BackupCoordinator",
    "Guarantee",
    "BackupReport",
    "BackupSet",
    "BackupInfo",
    "RestoreReport",
    "PruneReport",
    "BackupError",
    "QuiesceTimeout",
    "QuiesceUnavailable",
    "DestinationInUse",
    "BackupUnreadable",
    "restore",
    "read_backup",
    "prune_backups",
    "in_use",
]
