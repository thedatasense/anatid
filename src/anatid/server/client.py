"""``AnatidClient``: the verb surface of :class:`anatid.Anatid`, one socket away.

Switching a program from the embedded profile to the server profile is one line::

    db = Anatid.open("memory.anatid", tenant=1)  # one writer PROCESS
    db = AnatidClient.connect("/run/anatid/anatid.sock", tenant=1)  # many

Everything after that line is the same call with the same keywords returning the same objects.
``remember`` returns a :class:`~anatid.types.Memory`, ``recall`` returns
:class:`~anatid.types.RecallHits` with its ``arms`` and its staleness flags, and a
:class:`~anatid.errors.ConflictError` raised inside the server's transaction arrives here as a
``ConflictError`` carrying its ``resource`` and both version numbers.  ``examples/server_demo.py``
runs one workload function against both handles to show that it is the same program.

Why every read comes back over the socket
-----------------------------------------
There is no split where a client writes through the server and reads the file directly.  DuckDB
1.5.5 (measured, macOS arm64) refuses a second process on a file another process holds
read-write, and it refuses it even when the second process only wants to read::

    holder=read-write  second=read-write  -> duckdb.IOException
    holder=read-write  second=read-only   -> duckdb.IOException
    holder=read-only   second=read-write  -> duckdb.IOException
    holder=read-only   second=read-only   -> ok

A server holds its files read-write, so a client cannot open them at all.  This class therefore
has no read-only handle, no local cache and no read-your-writes problem to solve: a write and
the read after it go to the same process, through the same file, in that order.

What it costs, measured
-----------------------
A 3,001-memory tenant with 384-dimension embeddings, p50, in process against over a Unix socket::

    get()                        0.49 ms -> 1.01 ms   (+0.52 ms, 2.07x)
    remember()                   2.83 ms -> 4.24 ms   (+1.40 ms, 1.50x)
    recall(), vector arm on     25.38 ms -> 24.99 ms  (no measurable cost)

About half a millisecond of fixed cost per call.  It doubles the cheapest read anatid has and
disappears into a recall.  Most of it is the JSON codec rather than the socket.

``embeddings="f32"`` is what removes the codec half, and it is NOT symmetric.  The setting on
this object governs only what this object ENCODES, which is the request; the reply is encoded by
the server and governed by ``ServerConfig(embeddings=...)`` there.  So setting it here and not
there does nothing for a call whose cost is in the reply.  Measured, ``get()`` with
``with_embedding=True`` at 384 dimensions, p50 over a Unix socket::

    server="list"  client="list"   0.790 ms
    server="list"  client="f32"    0.771 ms   (the client's setting alone: nothing)
    server="f32"   client="list"   0.584 ms   (the server's setting alone: -26%)
    server="f32"   client="f32"    0.587 ms

Set it on the server for replies, and here as well for a call that ships a wide embedding UP,
which is ``recall(embedding=...)`` and ``remember(embedding=...)``.

Retries, and why the idempotency key is on by default
-----------------------------------------------------
A retryable failure is sent again with the SAME idempotency key and a jittered wait.  A
non-retryable one is raised unchanged, with the class the server raised.  The key is what makes
the first rule safe: if a write commits and the reply is lost, the retry carrying that key gets
the first write's recorded result back instead of writing a second row.  Without a key the
client will only retry a write it knows did not happen (:class:`BusyError` says so in as many
words); a dropped connection or an expired deadline is ambiguous and is raised rather than
guessed at.  Pass ``idempotency=False`` to turn the keys off and accept that narrower retry.

What is not here
----------------
Every verb the server dispatches is here.  What is missing is the part of
:class:`anatid.Anatid` that is about a CONNECTION rather than about memory: ``execute()``,
``transaction()``, ``unsafe_connection()``, the schema DDL, ``attach_read_only()``,
``load_parquet()``, ``register_erasure_hook()``.  None of them has a wire form, and each one is
absent with its own reason attached: :data:`_EMBEDDED_ONLY` holds the sentences and
:meth:`AnatidClient.__getattr__` raises them, so porting a program that calls one gets an
explanation rather than a bare name.  They stay absent rather than becoming methods that raise,
because ``hasattr(db, "execute")`` has to keep answering False.

:meth:`AnatidClient.atomic` is the one exception and is a real method that raises
:class:`NotImplementedError`, because it is the call a caller reaches for by name and it has a
real replacement to be pointed at.  A transaction is a property of one connection inside the
server process; holding one open across a network for the length of a caller's callback would
serialise the tenant behind that caller's think time, which is the thing the write queue exists
to avoid.  The compare-and-swap half of that surface does cross the wire:
:meth:`AnatidClient.update` takes ``expected_version`` and the conflict comes back with both
version numbers.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import random
import socket as _socket
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ..atomic import backoff_delay
from ..errors import AnatidError
from ..types import (
    AsOf,
    DoctorReport,
    Edge,
    Entity,
    Episode,
    ForgetReceipt,
    FtsStatus,
    Memory,
    Namespace,
    Provenance,
    PruneReport,
    RecallHits,
    SchemaInfo,
    to_utc_naive,
)
from . import protocol
from .protocol import (
    MAX_FRAME_BYTES,
    BusyError,
    FrameError,
    ProtocolError,
    Request,
    Response,
    ShuttingDown,
    Status,
)
from .server import VERBS, Health, Readiness

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..derived import HealthReport, MaintenanceReport, MaintenancePolicy
    from .queue import QueueStats

log = logging.getLogger("anatid.client")

__all__ = [
    "AnatidClient",
    "AsOfClient",
    "RetryPolicy",
    "ServerUnavailable",
    "connect",
    "DEFAULT_ATTEMPTS",
    "DEFAULT_DEADLINE",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_CONNECTIONS",
]

#: How many times a retryable failure is sent again before the client gives up.  Three, the same
#: number :func:`anatid.atomic.run` uses for the embedded profile, so a program that moved onto
#: the wire sees the same contention as the same number of attempts.
DEFAULT_ATTEMPTS = 3

#: Seconds of budget put on each request.  The SERVER measures it from the moment it decodes the
#: frame, so it is a relative budget and a wrong client clock cannot expire a request on arrival.
DEFAULT_DEADLINE = 30.0

#: Socket timeout, the backstop under the deadline.  Larger than :data:`DEFAULT_DEADLINE` on
#: purpose: the server's own deadline should fire first and answer with a
#: :class:`~anatid.server.protocol.DeadlineExceeded` that says what happened, rather than the
#: client giving up on a connection that is still going to be answered.
DEFAULT_TIMEOUT = 60.0

#: How many connections one client will open.  One is enough for a single-threaded program and
#: is all it ever opens, because connections are made on demand; a threaded program gets real
#: concurrency out of the server's read pool instead of queueing behind one socket.  Requests on
#: ONE connection are answered strictly in order, which is the server's rule and not a choice
#: made here.
DEFAULT_MAX_CONNECTIONS = 8


class ServerUnavailable(AnatidError):
    """The server could not be reached, or the connection died with the call in flight.

    ``address`` is what the client was talking to and ``stage`` is where it got to:

    ``"connect"``
        Nothing was sent.  The socket does not exist, the connection was refused, or the
        filesystem permissions on it say no.  For a write this is the good case: it certainly
        did not happen.
    ``"exchange"``
        The request was written and the answer did not come back.  Whether the server ran it is
        NOT known here, which is why a write with no idempotency key is not retried through one.

    ``retryable`` is True because a server that is down can come back up, but the client has
    already spent its own attempts by the time this is raised.
    """

    retryable = True

    def __init__(
        self,
        message: str,
        *,
        address: str = "",
        stage: str = "connect",
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.address = address
        self.stage = stage
        self.attempts = attempts


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How the client waits between attempts.

    ``attempts`` counts the FIRST try, so ``attempts=3`` is one call and two retries.

    ``backoff`` and ``max_backoff`` are full-jitter exponential backoff
    (:func:`anatid.atomic.backoff_delay`): the wait is drawn uniformly from
    ``[0, min(max_backoff, backoff * 2**(attempt-1)))``.  Jitter is not decoration -- two clients
    that collide and then sleep the same fixed interval collide again.

    When the server suggested a wait (:class:`~anatid.server.protocol.BusyError` carries one,
    computed from the rate that tenant's queue is actually draining at and scaled by how many
    writes that tenant is currently refusing) that estimate is used instead.  It is first trimmed
    to ``max_retry_after``, then spread over ``[0.5, 1.5)`` of the trimmed value so a crowd does
    not return in lockstep, so the longest single wait this can produce is ``1.5 *
    max_retry_after`` and not ``max_retry_after``.  Ignoring the server's number in favour of a
    local constant would be guessing at something the server measured.

    ``attempts`` is enough for contention and is not enough for sustained overload, and the
    difference is worth knowing.  At the shipped ``--max-queue 256`` a burst of 16 clients writing
    100 memories each never sees a busy answer at all.  A queue deliberately shrunk to 4 refuses
    most of that same burst however patiently each client waits, because the writers are simply
    faster than the disk: no retry policy fixes an arrival rate.  Raise ``attempts`` if you would
    rather wait than fail, and read ``BusyError`` as the signal to send less.

    ``sleep`` and ``rng`` are injectable so a test can assert the waits without taking them.
    """

    attempts: int = DEFAULT_ATTEMPTS
    backoff: float = 0.005
    max_backoff: float = 0.2
    max_retry_after: float = 5.0
    sleep: Callable[[float], Any] = time.sleep
    rng: Callable[[], float] = random.random

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError(
                f"attempts counts the first try, so it is at least 1, got {self.attempts}"
            )

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Seconds to wait before attempt ``attempt + 1``."""
        if retry_after is not None and retry_after > 0:
            wait = min(float(retry_after), self.max_retry_after)
            return wait * (0.5 + self.rng())
        return backoff_delay(attempt, base=self.backoff, cap=self.max_backoff, rng=self.rng)


# --------------------------------------------------------------------------- transports


class _Transport:
    """A pool of connections to one server, and one request-response exchange over one of them.

    The pool exists because the server answers the requests on a single connection strictly in
    order.  A single-threaded caller opens exactly one connection and reuses it for the life of
    the client; a threaded caller gets a connection per concurrent call, up to a limit, and so
    reaches the server's read pool instead of queueing behind a socket.
    """

    address: str = ""

    def __init__(
        self,
        *,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        timeout: float | None = DEFAULT_TIMEOUT,
    ) -> None:
        if max_connections < 1:
            raise ValueError(f"a client needs at least one connection, got {max_connections}")
        self.max_connections = int(max_connections)
        self.timeout = timeout
        self._idle: list[Any] = []
        self._live = 0
        self._cv = threading.Condition()
        self._closed = False

    # -- subclass hooks ----------------------------------------------------------------

    def open_one(self) -> Any:
        raise NotImplementedError

    def close_one(self, conn: Any) -> None:
        raise NotImplementedError

    def round_trip(
        self, conn: Any, request: Request, *, embeddings: str, max_frame_bytes: int
    ) -> Response:
        raise NotImplementedError

    # -- the pool ----------------------------------------------------------------------

    def exchange(self, request: Request, *, embeddings: str, max_frame_bytes: int) -> Response:
        """Send one request and return the response, or raise :class:`ServerUnavailable`.

        A connection that fails is closed rather than returned to the pool.  A half-read stream
        cannot be resynchronised, and a connection that just refused one request is not evidence
        that it will accept the next.  A :class:`~anatid.server.protocol.ProtocolError` is the
        one failure that is NOT turned into a :class:`ServerUnavailable`: the server was reached
        and answered, and what came back was not a message this client understands, which is a
        different problem with a different fix.
        """
        conn, fresh = self._acquire()
        try:
            response = self.round_trip(
                conn, request, embeddings=embeddings, max_frame_bytes=max_frame_bytes
            )
        # FrameError is a ProtocolError, and it is the framing half: the stream is no longer
        # synchronisable, which is a dead connection rather than a message this build cannot
        # read.  It is caught first for that reason.
        except (OSError, EOFError, FrameError) as exc:
            self._discard(conn)
            # A connection taken from the pool may have been closed by the server since it was
            # last used (an idle timeout, a restart).  That is not the same event as a fresh
            # connection dying mid-request, and only the second one is ambiguous for a write.
            stage = "exchange" if fresh else "idle"
            raise ServerUnavailable(
                f"the connection to {self.address} failed while the {request.verb} was in "
                f"flight: {exc}. Whether the server ran it is not known here.",
                address=self.address,
                stage=stage,
            ) from exc
        except ProtocolError:
            self._discard(conn)
            raise
        except BaseException:
            self._discard(conn)
            raise
        self._release(conn)
        return response

    def _acquire(self) -> tuple[Any, bool]:
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        with self._cv:
            while True:
                if self._closed:
                    raise ServerUnavailable(
                        f"this client is closed; it is no longer connected to {self.address}",
                        address=self.address,
                        stage="connect",
                    )
                if self._idle:
                    return self._idle.pop(), False
                if self._live < self.max_connections:
                    self._live += 1
                    break
                waited = self._cv.wait(
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                if not waited and deadline is not None and time.monotonic() >= deadline:
                    raise ServerUnavailable(
                        f"all {self.max_connections} connections to {self.address} were busy "
                        f"for {self.timeout}s. Raise max_connections, or make fewer concurrent "
                        f"calls on one client.",
                        address=self.address,
                        stage="connect",
                    )
        try:
            return self.open_one(), True
        except BaseException as exc:
            # The slot was reserved a few lines up and has to come back however open_one
            # failed, not only when it failed with an OSError.  Decrementing for OSError alone
            # leaks one slot per other failure, and a pool that has leaked max_connections of
            # them answers every later call with "all connections were busy", which is a
            # confident diagnosis of a problem the caller does not have.
            with self._cv:
                self._live -= 1
                self._cv.notify()
            if isinstance(exc, OSError):
                raise self._not_running(exc) from exc
            raise

    def _release(self, conn: Any) -> None:
        with self._cv:
            if self._closed:
                self._live -= 1
                closing = conn
            else:
                self._idle.append(conn)
                closing = None
            self._cv.notify()
        if closing is not None:
            self._shut(closing)

    def _discard(self, conn: Any) -> None:
        with self._cv:
            self._live -= 1
            self._cv.notify()
        self._shut(conn)

    def _shut(self, conn: Any) -> None:
        try:
            self.close_one(conn)
        except OSError:
            pass

    def close(self) -> None:
        with self._cv:
            self._closed = True
            idle = list(self._idle)
            self._idle.clear()
            self._live -= len(idle)
            self._cv.notify_all()
        for conn in idle:
            self._shut(conn)

    def _not_running(self, exc: OSError) -> ServerUnavailable:
        return ServerUnavailable(
            f"could not connect to {self.address}: {exc}", address=self.address, stage="connect"
        )


class _UnixTransport(_Transport):
    """Length-prefixed frames over a Unix domain socket, which is the server's own transport."""

    def __init__(self, path: str | os.PathLike[str], **kw: Any) -> None:
        super().__init__(**kw)
        self.path = os.fspath(path)
        self.address = f"unix:{self.path}"

    def open_one(self) -> Any:
        sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        if self.timeout is not None:
            sock.settimeout(self.timeout)
        try:
            sock.connect(self.path)
        except OSError:
            sock.close()
            raise
        return sock

    def close_one(self, conn: Any) -> None:
        conn.close()

    def round_trip(
        self, conn: Any, request: Request, *, embeddings: str, max_frame_bytes: int
    ) -> Response:
        conn.sendall(request.encode(embeddings=embeddings))
        body = protocol.read_frame(conn, max_bytes=max_frame_bytes)
        if body is None:
            raise EOFError("the server closed the connection without answering")
        return Response.decode(body)

    def _not_running(self, exc: OSError) -> ServerUnavailable:
        if isinstance(exc, FileNotFoundError):
            detail = (
                f"there is no socket at {self.path}. Start a server on it "
                f"(AnatidServer(pool=..., config=ServerConfig(socket_path={self.path!r}))), or "
                f"point this client at the path the running one prints in its endpoints()."
            )
        elif isinstance(exc, ConnectionRefusedError):
            detail = (
                f"{self.path} exists but nothing is listening on it. That is the socket a "
                f"server left behind when it was killed rather than shut down; the next server "
                f"to start on that path clears it."
            )
        elif isinstance(exc, PermissionError):
            detail = (
                f"{self.path} refused this user. A server socket is created 0600 in a 0700 "
                f"directory, which is the whole access control for this transport: talk to it "
                f"as the user the server runs as, or give the server an authenticator and an "
                f"HTTP listener."
            )
        else:
            detail = f"{self.path} could not be connected to: {exc}"
        return ServerUnavailable(detail, address=self.address, stage="connect")


class _HttpTransport(_Transport):
    """``POST /rpc`` over HTTP/1.1 with keep-alive, for the server's optional HTTP listener.

    The body is one request object with no length prefix, which is what
    :meth:`AnatidServer._http_route` reads.  A bearer token is required by any server that binds
    something other than loopback, and is sent on every request because HTTP has no connection
    to remember it on.
    """

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        headers: Mapping[str, str] | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        from urllib.parse import urlsplit

        parts = urlsplit(url if "://" in url else f"http://{url}")
        if parts.scheme not in ("http", "https"):
            raise ValueError(f"an anatid server speaks http or https, not {parts.scheme!r}")
        self.scheme = parts.scheme
        self.host = parts.hostname or "127.0.0.1"
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        self.path = parts.path.rstrip("/") or ""
        self.address = f"{self.scheme}://{self.host}:{self.port}{self.path}"
        self.headers = {"content-type": "application/json"}
        if token:
            self.headers["authorization"] = f"Bearer {token}"
        self.headers.update({k.lower(): v for k, v in (headers or {}).items()})

    def open_one(self) -> Any:
        import http.client

        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        conn = cls(self.host, self.port, timeout=self.timeout)
        conn.connect()
        return conn

    def close_one(self, conn: Any) -> None:
        conn.close()

    def round_trip(
        self, conn: Any, request: Request, *, embeddings: str, max_frame_bytes: int
    ) -> Response:
        body = protocol.dumps(request.to_wire(), embeddings=embeddings)
        if len(body) > max_frame_bytes:
            raise FrameError(
                f"this request is {len(body)} bytes, over the {max_frame_bytes} byte limit"
            )
        import http.client

        try:
            conn.request("POST", f"{self.path}/rpc", body=body, headers=self.headers)
            reply = conn.getresponse()
            payload = reply.read()
        except http.client.HTTPException as exc:
            # http.client raises its own hierarchy rather than OSError for a connection it can
            # no longer use.  The pool treats a dead connection uniformly, so it is translated
            # here instead of being special-cased in every caller.
            raise EOFError(f"the HTTP connection failed: {exc}") from exc
        if len(payload) > max_frame_bytes:
            raise FrameError(f"the reply is {len(payload)} bytes, over the frame limit")
        decoded = protocol.loads(payload)
        if not isinstance(decoded, dict) or "status" not in decoded:
            raise ProtocolError(
                f"{self.address} answered HTTP {reply.status} with a body that is not an anatid "
                f"response: {payload[:200]!r}. Check the URL: /rpc is the only route that "
                f"speaks this protocol."
            )
        return Response.from_wire(decoded)


# --------------------------------------------------------------------------- the client

#: Verbs that go through the server's write queue, taken from the server's own table so the two
#: can never disagree about what a write is.
_WRITES = frozenset(name for name, spec in VERBS.items() if spec.write)

#: The parts of :class:`anatid.Anatid` that are deliberately absent here, and the sentence each
#: absence has earned.  :meth:`AnatidClient.__getattr__` turns a miss into the reason, so a
#: program being ported is told why rather than just told the name again.
#:
#: They stay ABSENT rather than becoming methods that raise.  ``hasattr(db, "execute")`` has to
#: keep answering False, or a caller that checks before it calls ends up worse off than one that
#: does not.  :meth:`AnatidClient.atomic` is the single exception and is a real method, because
#: it is the one a caller reaches for by name with a real replacement to be pointed at.
_EMBEDDED_ONLY: dict[str, str] = {
    "execute": (
        "It runs SQL on one connection inside the process that owns the file, and the server "
        "owns it. Shipping arbitrary SQL over the wire would also hand every client the whole "
        "file, past the tenant boundary the server checks in the request envelope. Use a verb, "
        "or run the statement in the server process."
    ),
    "transaction": (
        "A transaction belongs to one connection inside the server, and holding one open for "
        "the length of a remote block would serialise that tenant behind the network, which is "
        "what the write queue exists to prevent. Use update(expected_version=...) for "
        "compare-and-swap and an idempotency key for a retry that writes once."
    ),
    "unsafe_connection": (
        "There is no DuckDB cursor on this side. This object holds a socket; the connection is "
        "in the server process."
    ),
    "attach_read_only": (
        "ATTACH runs on the server's connection against a path on the SERVER's filesystem, so "
        "a client naming one would be naming a file it cannot see and reaching past the tenant "
        "it authenticated for. Read the second file through a second client."
    ),
    "detach": "It is the other half of attach_read_only, and absent for the same reason.",
    "load_parquet": (
        "It bulk-loads a directory on the machine that holds the file, which is the server's "
        "filesystem and not this one. Copy the directory to the server and load it there, or "
        "write the rows with remember()."
    ),
    "build_csr": (
        "The CSR snapshot is an in-memory structure in the process that owns the file. The "
        "server builds it and its own traversals use it; there is nothing here to hold."
    ),
    "require_csr_extension": (
        "Whether the C++ extension is loaded is a fact about the server process. It is in the "
        "readiness the server reports, not something a client can assert about itself."
    ),
    "create_edge_type": (
        "Schema DDL is file-wide. A client that could add a table would be changing the schema "
        "under every other client and every other tenant in that file. Run it in the server "
        "process before the server starts."
    ),
    "create_node_label": (
        "Schema DDL is file-wide, like create_edge_type, and absent for the same reason."
    ),
    "ensure_schema": (
        "The server migrates every file it opens at startup and reports the outcome in ready(), "
        "which is where to look. Migrating from a client would migrate a file other clients "
        "have open."
    ),
    "register_erasure_hook": (
        "It registers a PYTHON callback that forget(hard=True) calls inside the transaction, "
        "and a callback cannot cross a socket. Register it in the server process."
    ),
    "visibility": (
        "It reports the Visibility a read would apply, computed from the handle's own settings, "
        "and the settings are the server's. Ask the server for the read; as_of() carries the "
        "one part of that a caller chooses."
    ),
    "resolve_tenant": (
        "The server resolves the tenant from the request envelope and checks it against the "
        "connection's principal before it opens anything. On this side the answer is tenant_id."
    ),
}


class AnatidClient:
    """The verbs of :class:`anatid.Anatid`, dispatched to a server over a socket.

    ::

        with AnatidClient.connect("/run/anatid/anatid.sock", tenant=1) as db:
            m = db.remember("Ada prefers tea", entities=["Ada"])
            hits = db.recall("what does Ada drink", k=5)

    Thread safe.  Each concurrent call takes its own connection from a pool that grows to
    ``max_connections`` on demand, so a single-threaded program opens exactly one socket and a
    threaded one reaches the server's read pool instead of queueing behind that socket.

    ``tenant`` is fixed at connect time and travels in the request ENVELOPE, where the server
    checks it against the connection's principal before it resolves a handle.  Every verb still
    takes ``tenant=`` for the calls that name another one; the server refuses a ``tenant`` in the
    argument object outright, so the client moves it into the envelope rather than passing two
    answers to the question of which tenant a call is for.
    """

    def __init__(
        self,
        *,
        socket_path: str | os.PathLike[str] | None = None,
        url: str | None = None,
        token: str | None = None,
        tenant: int | Namespace = 0,
        retry: RetryPolicy | None = None,
        idempotency: bool = True,
        deadline: float | None = DEFAULT_DEADLINE,
        timeout: float | None = DEFAULT_TIMEOUT,
        embeddings: str = "list",
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_frame_bytes: int = MAX_FRAME_BYTES,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if (socket_path is None) == (url is None):
            raise ValueError(
                "pass exactly one of socket_path= (the server's Unix socket, the transport it "
                "is built for) or url= (its optional HTTP listener)"
            )
        if embeddings not in ("list", "f32"):
            raise ValueError(f"embeddings must be 'list' or 'f32', got {embeddings!r}")
        self.namespace = Namespace.coerce(tenant)
        self.retry = retry or RetryPolicy()
        self.idempotency = bool(idempotency)
        self.deadline = deadline
        self.embeddings = embeddings
        self.max_frame_bytes = int(max_frame_bytes)
        self._transport: _Transport = (
            _UnixTransport(socket_path, max_connections=max_connections, timeout=timeout)
            if socket_path is not None
            else _HttpTransport(
                str(url),
                token=token,
                headers=headers,
                max_connections=max_connections,
                timeout=timeout,
            )
        )
        self._closed = False
        #: What the server said about itself when this client connected, or None when the
        #: client was built without a probe.  A snapshot, not a live value: call
        #: :meth:`health` for the current one.
        self.server_info: Health | None = None

    # -- construction ------------------------------------------------------------------

    @classmethod
    def connect(
        cls,
        socket_path: str | os.PathLike[str],
        *,
        tenant: int | Namespace = 0,
        probe: bool = True,
        **kw: Any,
    ) -> "AnatidClient":
        """Connect to a server's Unix socket.  The one-line replacement for ``Anatid.open``.

        ``probe`` sends one ``health`` request before returning, so a server that is not
        running, not speaking this protocol version, or not willing to authenticate this
        connection for ``tenant`` is reported here rather than on the first real call.  That is
        what ``Anatid.open`` does with a file it cannot open, and a constructor that succeeds
        against a dead server would be a worse drop-in.  Pass ``probe=False`` to connect lazily.
        """
        client = cls(socket_path=socket_path, tenant=tenant, **kw)
        if probe:
            client._probe()
        return client

    @classmethod
    def connect_http(
        cls,
        url: str,
        *,
        tenant: int | Namespace = 0,
        token: str | None = None,
        probe: bool = True,
        **kw: Any,
    ) -> "AnatidClient":
        """Connect to a server's HTTP listener (``POST /rpc``).

        The Unix socket is the transport the server is built for and the one with an access
        control the kernel enforces.  Use this when the client is on another host, and give the
        server a :class:`~anatid.server.auth.BearerTokenAuthenticator`: it refuses to bind
        anything outside loopback without one.
        """
        client = cls(url=url, token=token, tenant=tenant, **kw)
        if probe:
            client._probe()
        return client

    def _probe(self) -> Health:
        """One ``health`` round trip, to fail at connect time rather than at the first call.

        It also settles authorization, because the server checks the envelope's tenant against
        the principal before it looks at the verb: a token scoped to another tenant fails here.
        A probe that fails closes the client on its way out, so a refused connection does not
        leave a socket open on an object the caller never received.
        """
        try:
            health = self.health()
        except BaseException:
            self.close()
            raise
        self.server_info = health
        return health

    # -- identity ----------------------------------------------------------------------

    @property
    def tenant_id(self) -> int:
        """The tenant every call carries unless it names another one."""
        return self.namespace.tenant_id

    @property
    def address(self) -> str:
        """What this client is talking to, as it appears in error messages."""
        return self._transport.address

    @property
    def closed(self) -> bool:
        return self._closed

    def __repr__(self) -> str:
        return (
            f"<AnatidClient {self._transport.address} tenant={self.namespace.tenant_id}"
            f"{' closed' if self._closed else ''}>"
        )

    def close(self) -> None:
        """Close every connection.  Idempotent; the server is not affected."""
        self._closed = True
        self._transport.close()

    def __enter__(self) -> "AnatidClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- dispatch ----------------------------------------------------------------------

    def _tenant(self, tenant: int | Namespace | None) -> int:
        if tenant is None:
            return self.namespace.tenant_id
        return Namespace.coerce(tenant, self.namespace).tenant_id

    def call(
        self,
        verb: str,
        *,
        tenant: int | Namespace | None = None,
        idempotency_key: str | None = None,
        deadline: float | None = None,
        **args: Any,
    ) -> Any:
        """Call any verb by name.

        The escape hatch for a server newer than this client: every verb below is this method
        with its arguments spelled out, and a verb this build has never heard of still reaches a
        server that has it.  An unknown verb is treated as a write for retry purposes, which is
        the conservative half of that guess.
        """
        return self._invoke(
            verb,
            dict(args),
            tenant=tenant,
            idempotency_key=idempotency_key,
            deadline=deadline,
        )

    def _invoke(
        self,
        verb: str,
        args: dict[str, Any],
        *,
        tenant: int | Namespace | None = None,
        idempotency_key: str | None = None,
        deadline: float | None = None,
    ) -> Any:
        if self._closed:
            raise ServerUnavailable(
                f"this client is closed; reconnect to {self._transport.address} to use it again",
                address=self._transport.address,
                stage="connect",
            )
        spec = VERBS.get(verb)
        write = spec.write if spec is not None else True
        key = idempotency_key
        if write and key is None and self.idempotency:
            key = protocol.new_request_id()
        request = Request(
            verb=verb,
            tenant=self._tenant(tenant),
            args=args,
            idempotency_key=key,
            deadline=self.deadline if deadline is None else deadline,
        )
        return self._send(request, write=write, keyed=key is not None)

    def _send(self, request: Request, *, write: bool, keyed: bool) -> Any:
        """Send one request, retrying what is worth retrying, and return the verb's value."""
        attempts = self.retry.attempts
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._transport.exchange(
                    request, embeddings=self.embeddings, max_frame_bytes=self.max_frame_bytes
                )
            except ServerUnavailable as exc:
                # "connect" and "idle" both mean nothing was written: the request never left, or
                # it went into a connection the server had already closed.  "exchange" means the
                # request was written to a live connection and the answer never came, which is
                # ambiguous, and only an idempotency key makes sending it again safe.
                ambiguous = write and exc.stage == "exchange" and not keyed
                exc.attempts = attempt
                if ambiguous or attempt >= attempts:
                    raise
                self.retry.sleep(self.retry.delay(attempt))
                request = _renew(request)
                continue
            if response.status is Status.OK:
                return response.result
            error = response.error
            if error is None:
                raise ProtocolError(
                    f"the server answered {response.status.value} for this {request.verb} with "
                    f"no error object, so there is nothing to raise"
                )
            exc = error.to_exception()
            if attempt >= attempts or not _worth_retrying(exc, write=write, keyed=keyed):
                raise exc
            log.debug(
                "retrying %s after %s (attempt %d of %d)",
                request.verb,
                type(exc).__name__,
                attempt,
                attempts,
            )
            self.retry.sleep(self.retry.delay(attempt, response.retry_after))
            request = _renew(request)

    # -- reads -------------------------------------------------------------------------

    def get(
        self,
        memory_id: int,
        *,
        tenant: int | Namespace | None = None,
        as_of: AsOf | _dt.datetime | None = None,
        with_embedding: bool = True,
    ) -> Memory | None:
        """One memory by id, or None when the id is not in this tenant."""
        return self._invoke(
            "get",
            {"memory_id": memory_id, "as_of": as_of, "with_embedding": with_embedding},
            tenant=tenant,
        )

    def versions(
        self,
        memory_id: int,
        *,
        tenant: int | Namespace | None = None,
        with_embedding: bool = False,
    ) -> list[Memory]:
        """Every version of one memory, oldest first."""
        return self._invoke(
            "versions",
            {"memory_id": memory_id, "with_embedding": with_embedding},
            tenant=tenant,
        )

    def recall(
        self,
        query: str | None = None,
        *,
        tenant: int | Namespace | None = None,
        k: int = 10,
        embedding: Sequence[float] | None = None,
        seed_entity: int | str | Entity | None = None,
        hops: int = 2,
        as_of: AsOf | _dt.datetime | None = None,
        kinds: Sequence[str] | None = None,
        candidates: int = 50,
        rrf_k: int = 60,
        with_embedding: bool = False,
        include_about: bool = True,
        on_stale_fts: str = "report",
        allow_slow: bool = False,
    ) -> RecallHits:
        """Fused vector, text and graph retrieval.  The call this profile exists to serve.

        Measured at 25.0 ms over the socket against 25.4 ms in process on a 3,001-memory tenant
        at 384 dimensions: the round trip disappears into the vector arm's scan.
        """
        return self._invoke(
            "recall",
            {
                "query": query,
                "k": k,
                "embedding": embedding,
                "seed_entity": seed_entity,
                "hops": hops,
                "as_of": as_of,
                "kinds": kinds,
                "candidates": candidates,
                "rrf_k": rrf_k,
                "with_embedding": with_embedding,
                "include_about": include_about,
                "on_stale_fts": on_stale_fts,
                "allow_slow": allow_slow,
            },
            tenant=tenant,
        )

    def recall_2hop(
        self,
        seed_entity: int | str | Entity,
        *,
        tenant: int | Namespace | None = None,
        limit: int = 20,
        as_of: AsOf | _dt.datetime | None = None,
        hops: int = 2,
        kinds: Sequence[str] | None = None,
        with_embedding: bool = False,
    ) -> list[Memory]:
        """Memories two hops from a seed entity."""
        return self._invoke(
            "recall_2hop",
            {
                "seed_entity": seed_entity,
                "limit": limit,
                "as_of": as_of,
                "hops": hops,
                "kinds": kinds,
                "with_embedding": with_embedding,
            },
            tenant=tenant,
        )

    def recall_2hop_ids(
        self,
        seed_entity: int | str | Entity,
        *,
        tenant: int | Namespace | None = None,
        limit: int = 20,
        as_of: AsOf | _dt.datetime | None = None,
        hops: int = 2,
        kinds: Sequence[str] | None = None,
    ) -> list[tuple[int, _dt.datetime]]:
        """The same traversal, as ``(memory_id, created_at)`` pairs."""
        return self._invoke(
            "recall_2hop_ids",
            {
                "seed_entity": seed_entity,
                "limit": limit,
                "as_of": as_of,
                "hops": hops,
                "kinds": kinds,
            },
            tenant=tenant,
        )

    def context(
        self,
        entity: int | str | Entity,
        *,
        limit: int = 20,
        as_of: AsOf | _dt.datetime | None = None,
        tenant: int | Namespace | None = None,
        hops: int = 0,
        kinds: Sequence[str] | None = None,
        with_embedding: bool = False,
    ) -> list[Memory]:
        """The memories one entity is about."""
        return self._invoke(
            "context",
            {
                "entity": entity,
                "limit": limit,
                "as_of": as_of,
                "hops": hops,
                "kinds": kinds,
                "with_embedding": with_embedding,
            },
            tenant=tenant,
        )

    def entities_of(
        self,
        memory_id: int,
        *,
        tenant: int | Namespace | None = None,
        as_of: AsOf | _dt.datetime | None = None,
    ) -> list[Entity]:
        """The entities one memory is about."""
        return self._invoke("entities_of", {"memory_id": memory_id, "as_of": as_of}, tenant=tenant)

    def provenance(
        self,
        memory_id: int,
        *,
        tenant: int | Namespace | None = None,
        max_depth: int = 10000,
    ) -> Provenance:
        """The evidence trail behind one memory."""
        return self._invoke(
            "provenance", {"memory_id": memory_id, "max_depth": max_depth}, tenant=tenant
        )

    def get_entity(
        self, entity: int | str | Entity, *, tenant: int | Namespace | None = None
    ) -> Entity | None:
        """One entity by id or name, or None."""
        return self._invoke("get_entity", {"entity": entity}, tenant=tenant)

    def get_episode(
        self, episode_id: int, *, tenant: int | Namespace | None = None
    ) -> Episode | None:
        """One episode by id, or None."""
        return self._invoke("get_episode", {"episode_id": episode_id}, tenant=tenant)

    def memory_version(
        self, memory_id: int, *, tenant: int | Namespace | None = None
    ) -> int | None:
        """The live version number of one memory: the number to hold for :meth:`update`."""
        return self._invoke("memory_version", {"memory_id": memory_id}, tenant=tenant)

    def stats(
        self, *, tenant: int | Namespace | None = None, all_tenants: bool = False
    ) -> dict[str, Any]:
        """Row counts for a tenant."""
        return self._invoke("stats", {"all_tenants": all_tenants}, tenant=tenant)

    def doctor(
        self,
        *,
        tenant: int | Namespace | None = None,
        all_tenants: bool = False,
        deep: bool = True,
        samples: int = 10,
        raise_on_error: bool = False,
    ) -> DoctorReport:
        """The integrity report.

        ``raise_on_error=True`` raises :class:`~anatid.errors.IntegrityError` in the SERVER, and
        it arrives here as an ``IntegrityError`` with its message but with ``report`` unset: the
        report is a value the error carries, not a scalar the wire copies.  Leave it False and
        read the report that comes back, which is the mode this verb is built for.
        """
        return self._invoke(
            "doctor",
            {
                "all_tenants": all_tenants,
                "deep": deep,
                "samples": samples,
                "raise_on_error": raise_on_error,
            },
            tenant=tenant,
        )

    def index_health(
        self,
        *,
        tenant: int | Namespace | None = None,
        policy: "MaintenancePolicy | None" = None,
        as_of: AsOf | _dt.datetime | None = None,
    ) -> "dict[str, HealthReport]":
        """Per-index health and staleness."""
        return self._invoke(
            "index_health",
            {"policy": _wire_policy(policy), "as_of": as_of},
            tenant=tenant,
        )

    def info(self) -> SchemaInfo:
        """The ``anatid_meta`` catalog row for the FILE this tenant lives in.

        File-wide, not tenant-scoped.  With a server on a :class:`anatid.DatabasePool` that is
        this tenant's own file; with a server on a single shared handle it is the one file every
        tenant shares, which is namespaces and not a security boundary.
        """
        return self._invoke("info", {})

    def fts_status(self, *, deep: bool = False) -> FtsStatus:
        """The state of the BM25 index.  File-wide, like :meth:`info`."""
        return self._invoke("fts_status", {"deep": deep})

    # -- writes ------------------------------------------------------------------------

    def remember(
        self,
        content: str,
        *,
        entities: Sequence[int | str | Entity] = (),
        kind: str = "fact",
        embedding: Sequence[float] | None = None,
        writer: str | None = None,
        episode_id: int | None = None,
        episode: str | None = None,
        episode_source: str | None = None,
        confidence: float = 1.0,
        now: _dt.datetime | None = None,
        valid_from: _dt.datetime | None = None,
        created_at: _dt.datetime | None = None,
        tenant: int | Namespace | None = None,
        memory_id: int | None = None,
        entity_kind: str | None = None,
        weight: float = 1.0,
        create_entities: bool = True,
        idempotency_key: str | None = None,
    ) -> Memory:
        """Write a memory.

        Measured at 4.24 ms over the socket against 2.83 ms in process, batching included: the
        server puts writes for one tenant into one transaction where the verbs allow it, which
        was worth 1.30x at sixteen concurrent writers.
        """
        return self._invoke(
            "remember",
            {
                "content": content,
                "entities": list(entities),
                "kind": kind,
                "embedding": embedding,
                "writer": writer,
                "episode_id": episode_id,
                "episode": episode,
                "episode_source": episode_source,
                "confidence": confidence,
                "now": now,
                "valid_from": valid_from,
                "created_at": created_at,
                "memory_id": memory_id,
                "entity_kind": entity_kind,
                "weight": weight,
                "create_entities": create_entities,
            },
            tenant=tenant,
            idempotency_key=idempotency_key,
        )

    def supersede(
        self,
        old_id: int,
        content: str,
        *,
        entities: Sequence[int | str | Entity] | None = None,
        kind: str | None = None,
        embedding: Sequence[float] | None = None,
        writer: str | None = None,
        episode_id: int | None = None,
        episode: str | None = None,
        episode_source: str | None = None,
        confidence: float = 1.0,
        now: _dt.datetime | None = None,
        tenant: int | Namespace | None = None,
        memory_id: int | None = None,
        close_about_edges: bool = False,
        allow_fork: bool = False,
        idempotency_key: str | None = None,
    ) -> Memory:
        """Correct a memory with a newer one, closing the old and linking the two."""
        return self._invoke(
            "supersede",
            {
                "old_id": old_id,
                "content": content,
                "entities": None if entities is None else list(entities),
                "kind": kind,
                "embedding": embedding,
                "writer": writer,
                "episode_id": episode_id,
                "episode": episode,
                "episode_source": episode_source,
                "confidence": confidence,
                "now": now,
                "memory_id": memory_id,
                "close_about_edges": close_about_edges,
                "allow_fork": allow_fork,
            },
            tenant=tenant,
            idempotency_key=idempotency_key,
        )

    def update(
        self,
        memory_id: int,
        content: str,
        *,
        expected_version: int | None = None,
        tenant: int | Namespace | None = None,
        idempotency_key: str | None = None,
        **supersede_kwargs: Any,
    ) -> Memory:
        """Correct a memory, refusing the write if it is not at the version you read.

        The compare-and-swap that survives the wire::

            m = db.get(mid)
            db.update(mid, "Ada drinks tea now", expected_version=m.version)

        A failure raises :class:`~anatid.errors.ConflictError` with ``expected_version`` and
        ``current_version`` filled in and ``retryable`` False, and this client does NOT retry it:
        the version the caller reasoned about is gone, so an identical retry fails identically.
        Re-read, decide whether the change still applies, and write again.
        """
        return self._invoke(
            "update",
            {
                "memory_id": memory_id,
                "content": content,
                "expected_version": expected_version,
                **supersede_kwargs,
            },
            tenant=tenant,
            idempotency_key=idempotency_key,
        )

    def reinforce(
        self,
        memory_id: int,
        *,
        amount: int = 1,
        confidence: float | None = None,
        now: _dt.datetime | None = None,
        tenant: int | Namespace | None = None,
        idempotency_key: str | None = None,
    ) -> Memory:
        """Record an access and optionally adjust confidence."""
        return self._invoke(
            "reinforce",
            {"memory_id": memory_id, "amount": amount, "confidence": confidence, "now": now},
            tenant=tenant,
            idempotency_key=idempotency_key,
        )

    def relate(
        self,
        src: int | str | Entity,
        dst: int | str | Entity,
        *,
        if_current: bool = False,
        tenant: int | Namespace | None = None,
        idempotency_key: str | None = None,
        **kwargs: Any,
    ) -> Edge:
        """An edge between two entities, the edges 2-hop recall traverses.

        Takes everything :meth:`anatid.Anatid.relate` takes: ``rel_kind``, ``writer``,
        ``episode_id``, ``confidence``, ``now``, ``valid_from``, ``create_missing``, ``edge_id``.
        ``if_current=True`` refuses the edge unless both endpoints have a current row, and the
        refusal is a non-retryable ``ConflictError`` naming the endpoint that does not.
        """
        return self._invoke(
            "relate",
            {"src": src, "dst": dst, "if_current": if_current, **kwargs},
            tenant=tenant,
            idempotency_key=idempotency_key,
        )

    def unrelate(
        self,
        src: int | str | Entity,
        dst: int | str | Entity,
        *,
        rel_kind: str | None = None,
        tenant: int | Namespace | None = None,
        now: _dt.datetime | None = None,
        idempotency_key: str | None = None,
    ) -> int:
        """Close the edges between two entities.  Returns how many were closed."""
        return self._invoke(
            "unrelate",
            {"src": src, "dst": dst, "rel_kind": rel_kind, "now": now},
            tenant=tenant,
            idempotency_key=idempotency_key,
        )

    def upsert_entity(
        self,
        name: str,
        *,
        kind: str | None = None,
        tenant: int | Namespace | None = None,
        writer: str | None = None,
        episode_id: int | None = None,
        now: _dt.datetime | None = None,
        idempotency_key: str | None = None,
    ) -> Entity:
        """Create an entity or return the existing one with that name."""
        return self._invoke(
            "upsert_entity",
            {"name": name, "kind": kind, "writer": writer, "episode_id": episode_id, "now": now},
            tenant=tenant,
            idempotency_key=idempotency_key,
        )

    def entity_id(
        self,
        value: int | str | Entity,
        *,
        tenant: int | Namespace | None = None,
        create: bool = False,
        kind: str | None = None,
        now: _dt.datetime | None = None,
        writer: str | None = None,
        episode_id: int | None = None,
        idempotency_key: str | None = None,
    ) -> int:
        """Resolve a name to an entity id, optionally creating it."""
        return self._invoke(
            "entity_id",
            {
                "value": value,
                "create": create,
                "kind": kind,
                "now": now,
                "writer": writer,
                "episode_id": episode_id,
            },
            tenant=tenant,
            idempotency_key=idempotency_key,
        )

    def episode(
        self,
        content: str,
        *,
        source: str | None = None,
        kind: str | None = None,
        writer: str | None = None,
        tenant: int | Namespace | None = None,
        now: _dt.datetime | None = None,
        episode_id: int | None = None,
        idempotency_key: str | None = None,
    ) -> Episode:
        """Write raw source material a memory can point back at."""
        return self._invoke(
            "episode",
            {
                "content": content,
                "source": source,
                "kind": kind,
                "writer": writer,
                "now": now,
                "episode_id": episode_id,
            },
            tenant=tenant,
            idempotency_key=idempotency_key,
        )

    def forget(
        self,
        memory_id: int,
        *,
        hard: bool = False,
        reason: str | None = None,
        writer: str | None = None,
        now: _dt.datetime | None = None,
        tenant: int | Namespace | None = None,
        idempotency_key: str | None = None,
    ) -> ForgetReceipt:
        """Close a memory, or purge it outright with ``hard=True``."""
        return self._invoke(
            "forget",
            {"memory_id": memory_id, "hard": hard, "reason": reason, "writer": writer, "now": now},
            tenant=tenant,
            idempotency_key=idempotency_key,
        )

    def prune(
        self,
        *,
        older_than: _dt.datetime | None = None,
        max_access_count: int | None = None,
        dry_run: bool = True,
        hard: bool = False,
        kinds: Sequence[str] | None = None,
        limit: int | None = None,
        tenant: int | Namespace | None = None,
        writer: str | None = None,
        reason: str | None = "prune",
        now: _dt.datetime | None = None,
        idempotency_key: str | None = None,
    ) -> PruneReport:
        """Close or purge in bulk.  ``dry_run=True`` by default, as in the embedded profile."""
        return self._invoke(
            "prune",
            {
                "older_than": older_than,
                "max_access_count": max_access_count,
                "dry_run": dry_run,
                "hard": hard,
                "kinds": kinds,
                "limit": limit,
                "writer": writer,
                "reason": reason,
                "now": now,
            },
            tenant=tenant,
            idempotency_key=idempotency_key,
        )

    def rebuild_fts_index(
        self,
        *,
        now: _dt.datetime | None = None,
        terms_index: bool = True,
        idempotency_key: str | None = None,
    ) -> FtsStatus:
        """Rebuild the BM25 index.  FILE-wide: it touches every tenant in the file."""
        return self._invoke(
            "rebuild_fts_index",
            {"now": now, "terms_index": terms_index},
            idempotency_key=idempotency_key,
        )

    def maintain_indexes(
        self,
        *,
        tenant: int | Namespace | None = None,
        policy: "MaintenancePolicy | None" = None,
        now: _dt.datetime | None = None,
        idempotency_key: str | None = None,
    ) -> "dict[str, MaintenanceReport]":
        """Build the derived indexes that are due."""
        return self._invoke(
            "maintain_indexes",
            {"policy": _wire_policy(policy), "now": now},
            tenant=tenant,
            idempotency_key=idempotency_key,
        )

    def recluster(
        self, tables: Sequence[str] | None = None, *, idempotency_key: str | None = None
    ) -> dict[str, int]:
        """Rewrite tables by their sort key.  FILE-wide, like :meth:`rebuild_fts_index`."""
        return self._invoke(
            "recluster",
            {"tables": None if tables is None else list(tables)},
            idempotency_key=idempotency_key,
        )

    # -- the server itself -------------------------------------------------------------

    def health(self) -> Health:
        """Is the server process alive.  True while it drains, on purpose."""
        return self._invoke("health", {})

    def ready(self) -> Readiness:
        """Can the server take traffic: files open, migrations done, queues below high water."""
        return self._invoke("ready", {})

    def queue_stats(self) -> "QueueStats":
        """The write queues' depths and counters.

        This one depends on the server process, not on this client, and it can fail where every
        other verb succeeds.  :class:`~anatid.server.queue.QueueStats` is registered with the
        protocol codec in ``anatid.server.cli`` and nowhere else, so a server started through
        the CLI or ``python -m anatid.server`` can encode it and a server built in-process with
        ``AnatidServer(pool=...)`` cannot.  The second one runs the verb and then fails to
        encode the answer, which arrives here as :class:`ProtocolError` naming the class.

        The connection survives that: an unencodable reply is one failed call, not a dead
        socket.  To get it against an embedded server, call
        ``anatid.server.protocol.register_dataclass(QueueStats)`` in BOTH processes.  The codec
        table is a module global consulted in each direction, so the server needs it to encode
        the reply and this process needs it to decode one; with only the server registered the
        call fails here instead, on the unknown wire tag.  Importing ``anatid.server.cli`` has
        the same effect as a side effect, which is why this verb's behaviour can change with an
        import somewhere else in either program.
        """
        return self._invoke("queue_stats", {})

    # -- time travel -------------------------------------------------------------------

    def as_of(
        self, timestamp: _dt.datetime | AsOf, *, tx_time: _dt.datetime | None = None
    ) -> "AsOfClient":
        """Scope reads to a point in time, the same as ``Anatid.as_of``.

        The scope is built here and travels as the ``as_of`` argument of each read; there is no
        round trip in this call and no session state on the server.  Remember what it is:
        anatid's own filter over the bitemporal columns, not a DuckDB feature.
        """
        if isinstance(timestamp, AsOf):
            scope = timestamp
        else:
            ts = to_utc_naive(timestamp)
            scope = AsOf(valid_time=ts, tx_time=to_utc_naive(tx_time) if tx_time else ts)
        return AsOfClient(self, scope)

    # -- the surface that does not cross the wire ---------------------------------------

    def atomic(self, callback: Callable[..., Any], **kw: Any) -> Any:
        """Not available over the wire, and this says so rather than pretending.

        ``db.atomic(fn)`` re-runs ``fn`` inside ONE transaction on ONE connection.  Neither half
        survives the trip: each call here is its own transaction in the server, and holding one
        open across the network for the length of a caller's callback would serialise the
        tenant behind that caller's think time, which is exactly what the write queue exists to
        prevent.  Running the callback anyway and calling it atomic would be a lie about
        isolation.

        What to use instead: one verb is already atomic in the server, ``update(...,
        expected_version=n)`` is the compare-and-swap, and an idempotency key makes a retry
        write once.  A unit of work that genuinely needs several statements in one transaction
        belongs in the server process, either embedded or as a verb.
        """
        raise NotImplementedError(
            "AnatidClient.atomic is not available over the wire: a transaction belongs to one "
            "connection inside the server, and holding one open for a remote callback would "
            "serialise the tenant behind the network. Use update(expected_version=...) for "
            "compare-and-swap, an idempotency key for write-once retries, or run the unit of "
            "work in the server process."
        )

    def __getattr__(self, name: str) -> Any:
        """Answer a missing embedded-only attribute with the reason, not just its name again.

        Python calls this only after normal lookup has already failed, so every real method
        above is untouched and ``hasattr`` still answers False for everything in
        :data:`_EMBEDDED_ONLY`.  A caller that duck-types on ``execute`` therefore still skips
        this object correctly; what changes is the sentence a program being ported gets when it
        reaches for a piece of :class:`anatid.Anatid` that has no wire form.

        It raises :class:`AttributeError` and nothing else, including for a name that is simply
        a typo, because anything else here would break ``hasattr``, ``getattr(x, n, default)``
        and every protocol Python probes for with a plain attribute lookup.
        """
        reason = _EMBEDDED_ONLY.get(name)
        if reason is None:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}", name=name, obj=self
            )
        raise AttributeError(
            f"AnatidClient has no {name}: it is part of the embedded handle that does not cross "
            f"the wire. {reason}",
            name=name,
            obj=self,
        )


class AsOfClient:
    """``client.as_of(t)``: every read on this object carries the same scope.

    The client-side twin of :class:`anatid.verbs.AsOfView`, with the same methods and the same
    rule that ``as_of`` passed explicitly to a call wins over the view's scope.
    """

    __slots__ = ("_client", "scope")

    def __init__(self, client: AnatidClient, scope: AsOf) -> None:
        self._client = client
        self.scope = scope

    def __repr__(self) -> str:
        return f"<AsOfClient valid_time={self.scope.valid_time} tx_time={self.scope.tx_time}>"

    def recall(self, query: str | None = None, **kw: Any) -> RecallHits:
        kw.setdefault("as_of", self.scope)
        return self._client.recall(query, **kw)

    def recall_2hop(self, seed_entity: int | str | Entity, **kw: Any) -> list[Memory]:
        kw.setdefault("as_of", self.scope)
        return self._client.recall_2hop(seed_entity, **kw)

    def recall_2hop_ids(
        self, seed_entity: int | str | Entity, **kw: Any
    ) -> list[tuple[int, _dt.datetime]]:
        kw.setdefault("as_of", self.scope)
        return self._client.recall_2hop_ids(seed_entity, **kw)

    def context(self, entity: int | str | Entity, **kw: Any) -> list[Memory]:
        kw.setdefault("as_of", self.scope)
        return self._client.context(entity, **kw)

    def get(self, memory_id: int, **kw: Any) -> Memory | None:
        kw.setdefault("as_of", self.scope)
        return self._client.get(memory_id, **kw)

    def entities_of(self, memory_id: int, **kw: Any) -> list[Entity]:
        kw.setdefault("as_of", self.scope)
        return self._client.entities_of(memory_id, **kw)

    def provenance(self, memory_id: int, **kw: Any) -> Provenance:
        return self._client.provenance(memory_id, **kw)


# --------------------------------------------------------------------------- helpers


def _renew(request: Request) -> Request:
    """The same call with a fresh request id and the SAME idempotency key.

    Two ids, two jobs.  ``request_id`` identifies one message, and a retry is a new message, so
    the server's log shows two lines rather than one line twice.  ``idempotency_key`` identifies
    one WRITE, and keeping it is what makes the retry return the first attempt's result instead
    of writing a second row.
    """
    return replace(request, request_id=protocol.new_request_id())


def _worth_retrying(exc: BaseException, *, write: bool, keyed: bool) -> bool:
    """Whether sending this call again could do anything but repeat the failure.

    The server decides retryable, because only the server knows whether anything was committed.
    This adds one rule the server cannot: an ambiguous failure on a write with no idempotency
    key is not retried, because a retry that duplicates a memory is worse than an error.
    :class:`~anatid.server.protocol.BusyError` is the exception -- it says in as many words that
    the write was not performed -- so backpressure is always safe to wait out.
    """
    if not getattr(exc, "retryable", False):
        return False
    if isinstance(exc, ShuttingDown):
        # Retryable, the server says, "against another instance, not against this one" -- and
        # this client is bound to this one, which has already removed its socket.  Sending it
        # again here would spend the attempts to arrive at the same sentence.
        return False
    if isinstance(exc, BusyError):
        return True
    return keyed or not write


def _wire_policy(policy: "MaintenancePolicy | None") -> Any:
    """Refuse a ``MaintenancePolicy`` argument with the reason, rather than at the far end.

    :class:`anatid.derived.MaintenancePolicy` is not in the wire types the server registers, so
    this client can encode one that the server then cannot decode.  Failing here names the fix
    and does not spend a round trip on it.
    """
    if policy is None:
        return None
    raise ProtocolError(
        "policy= cannot cross the wire: MaintenancePolicy is not one of the dataclasses "
        "anatid.server.server registers with the protocol, so the server would refuse to "
        "decode it. Leave it out to use the server's default policy, or register the class on "
        "both ends first."
    )


def connect(socket_path: str | os.PathLike[str], **kwargs: Any) -> AnatidClient:
    """Shorthand for :meth:`AnatidClient.connect`, mirroring :func:`anatid.database.connect`."""
    return AnatidClient.connect(socket_path, **kwargs)
