"""``AnatidClient``: the drop-in claim, the retry rules, and what happens when the server is not there.

The claim this file has to defend is that switching a program from ``Anatid.open`` to
``AnatidClient.connect`` is one line and changes nothing else.  Three tests carry it:

* ``test_the_client_takes_every_argument_the_embedded_handle_takes`` compares the two signatures
  parameter by parameter, so a keyword added to a verb and not to the client is a failure here
  rather than a ``TypeError`` in somebody's program;
* ``test_the_same_workload_answers_the_same_embedded_and_over_the_socket`` runs ONE function
  against both handles and compares the answers;
* ``test_the_client_implements_every_verb_the_server_dispatches`` fails when the server grows a
  verb the client cannot call.

Then the parts that only exist because there is a network: a retryable failure is sent again
with the SAME idempotency key, a non-retryable one arrives as the class the server raised, an
ambiguous failure on an unkeyed write is NOT sent again, and a server that is not running is one
error that says which socket and what to do about it.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import inspect
import json
import shutil
import socket as _socket
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from anatid import Anatid, DatabasePool
from anatid.errors import ConflictError, NotFoundError, ValidationError
from anatid.integrations.wire import JS_MAX_SAFE_INTEGER
from anatid.server import auth, protocol, server as S
from anatid.server import client as C
from anatid.server.client import AnatidClient, RetryPolicy, ServerUnavailable
from anatid.server.protocol import (
    AuthenticationError,
    AuthorizationError,
    BusyError,
    ProtocolError,
    Request,
    Response,
    ShuttingDown,
)

pytestmark = pytest.mark.skipif(
    not hasattr(_socket, "AF_UNIX"),
    reason="the server profile's Unix domain socket transport is not available on this platform; use the HTTP transport",
)


DIM = 8

#: A Unix socket path is a fixed-size field in ``sockaddr_un`` (104 bytes on macOS, 108 on
#: Linux) and pytest's ``tmp_path`` is nested deeply enough to overflow it.
SOCKADDR_UN_MAX = 100


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def sock_dir():
    """A directory short enough to hold a Unix socket path."""
    base = Path(tempfile.mkdtemp(prefix="anatid-c-"))
    if len(str(base)) + len("/run/xxxxxxxx.sock") > SOCKADDR_UN_MAX:
        shutil.rmtree(base, ignore_errors=True)
        base = Path(tempfile.mkdtemp(prefix="anatid-c-", dir="/tmp"))
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def pool(tmp_path):
    with DatabasePool(
        str(tmp_path / "tenants" / "t_{tenant}.anatid"), embedding_dim=DIM, max_open=8
    ) as p:
        yield p


@contextmanager
def running_server(**kw):
    """An :class:`AnatidServer` on its own event loop in a background thread.

    A server needs a running loop and the tests need a blocking client, so the loop goes on a
    thread.  It waits on an event rather than on ``serve_forever()`` because that method installs
    signal handlers, and ``add_signal_handler`` only works on the main thread.
    """
    loop = asyncio.new_event_loop()
    holder: dict[str, object] = {}
    ready = threading.Event()

    def run() -> None:
        asyncio.set_event_loop(loop)

        async def main() -> None:
            try:
                server = S.AnatidServer(**kw)
                await server.start()
            except BaseException as exc:  # reported through `ready`, never swallowed
                holder["error"] = exc
                ready.set()
                return
            holder["server"] = server
            ready.set()
            await asyncio.Event().wait()

        try:
            loop.run_until_complete(main())
        except RuntimeError:
            pass  # "Event loop stopped before Future completed", i.e. the shutdown below
        finally:
            loop.close()

    thread = threading.Thread(target=run, name="anatid-test-server", daemon=True)
    thread.start()
    assert ready.wait(30), "the test server never started"
    if "error" in holder:
        raise holder["error"]  # type: ignore[misc]
    server = holder["server"]
    try:
        yield server
    finally:
        future = asyncio.run_coroutine_threadsafe(server.shutdown(), loop)
        future.result(60)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(30)


@pytest.fixture
def served(pool, sock_dir):
    """A running server on a Unix socket, and the socket path."""
    sock_path = sock_dir / "run" / "anatid.sock"
    config = S.ServerConfig(
        socket_path=sock_path, tenants=(1,), workers=1, batch_max=4, shutdown_timeout=10.0
    )
    with running_server(pool=pool, config=config) as server:
        yield server, sock_path


@pytest.fixture
def client(served):
    _server, sock_path = served
    with AnatidClient.connect(sock_path, tenant=1) as handle:
        yield handle


# --------------------------------------------------------------------------- test doubles


class ScriptedTransport(C._Transport):
    """A transport that answers from a list instead of a socket.

    The retry rules are about what the client does with an answer, not about sockets, and a
    scripted answer is the only way to assert "this exact failure was sent again with the same
    idempotency key" without racing a real server into the state that produces it.
    """

    def __init__(self, script, *, max_connections: int = 1) -> None:
        super().__init__(max_connections=max_connections, timeout=5.0)
        self.script = list(script)
        self.sent: list[Request] = []
        self.opened = 0
        self.address = "scripted"

    def open_one(self):
        self.opened += 1
        return object()

    def close_one(self, conn) -> None:
        return None

    def round_trip(self, conn, request, *, embeddings, max_frame_bytes):
        self.sent.append(request)
        if not self.script:
            raise AssertionError(f"the script ran out at request {len(self.sent)}")
        nxt = self.script.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        if callable(nxt):
            return nxt(request)
        return nxt


def scripted(script, **kw) -> AnatidClient:
    """A client wired to :class:`ScriptedTransport`, with waits recorded rather than taken."""
    waits: list[float] = []
    kw.setdefault("retry", RetryPolicy(attempts=3, sleep=waits.append, rng=lambda: 0.5))
    handle = AnatidClient(socket_path="/nonexistent/never-used.sock", tenant=1, **kw)
    handle._transport = ScriptedTransport(script)
    handle.waits = waits  # type: ignore[attr-defined]
    return handle


def ok(result=None) -> Response:
    return Response.ok(result)


def failure(exc: BaseException) -> Response:
    return Response.failure(exc)


# --------------------------------------------------------------------------- the drop-in claim


#: Verbs the server dispatches that the embedded handle has under the same name.  ``health``,
#: ``ready`` and ``queue_stats`` are the server's own and have no embedded twin.
_SERVER_ONLY = {"health", "ready", "queue_stats"}


def test_the_client_implements_every_verb_the_server_dispatches():
    """A verb on the server that the client cannot call is a verb nobody can reach."""
    missing = [name for name in S.VERBS if not hasattr(AnatidClient, name)]
    assert missing == [], f"AnatidClient is missing {missing}"


def test_the_client_takes_every_argument_the_embedded_handle_takes():
    """The drop-in claim, checked parameter by parameter rather than asserted in prose.

    Every parameter of ``Anatid.<verb>`` must exist on ``AnatidClient.<verb>`` with the same
    default.  The client is allowed EXTRA keyword-only parameters (``idempotency_key``), because
    adding one cannot break a call written against the embedded handle.
    """
    problems: list[str] = []
    for name in sorted(set(S.VERBS) - _SERVER_ONLY):
        embedded = inspect.signature(getattr(Anatid, name)).parameters
        remote = inspect.signature(getattr(AnatidClient, name)).parameters
        for param, spec in embedded.items():
            if param == "self" or spec.kind is inspect.Parameter.VAR_KEYWORD:
                continue
            if param not in remote:
                problems.append(f"{name}: client has no {param!r}")
                continue
            got = remote[param]
            if got.default != spec.default:
                problems.append(
                    f"{name}.{param}: embedded default {spec.default!r}, client {got.default!r}"
                )
            if got.kind is not spec.kind:
                problems.append(f"{name}.{param}: embedded {spec.kind}, client {got.kind}")
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in embedded.values()):
            assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in remote.values()), (
                f"{name} takes **kwargs on the embedded handle; the client has to pass them on"
            )
    assert problems == [], "\n".join(problems)


def workload(db, *, label: str) -> dict:
    """One program, run against either handle.  Returns what it learned, for comparison."""
    ada = db.upsert_entity("Ada", kind="person")
    db.upsert_entity("Grace", kind="person")
    db.relate("Ada", "Grace", rel_kind="knows")
    first = db.remember(
        f"{label}: Ada prefers tea", entities=["Ada"], embedding=[0.9, 0.1] + [0.0] * (DIM - 2)
    )
    db.remember(f"{label}: Grace writes compilers", entities=["Grace"], kind="fact")
    corrected = db.update(
        first.memory_id, f"{label}: Ada prefers coffee", expected_version=first.version
    )
    db.reinforce(corrected.memory_id, amount=2)
    hits = db.recall("coffee", k=5)
    return {
        "entity_kind": ada.kind,
        "contents": sorted(m.content for m in db.recall_2hop("Ada")),
        "context": sorted(m.content for m in db.context("Ada")),
        "current": db.get(corrected.memory_id).content,
        "superseded": db.get(first.memory_id) is None
        or db.get(first.memory_id).content.endswith("tea"),
        "versions": [m.version for m in db.versions(corrected.memory_id)],
        "access_count": db.get(corrected.memory_id).access_count,
        "hit_contents": [h.memory.content for h in hits],
        "arms": hits.arms,
        "entities_of": sorted(e.name for e in db.entities_of(corrected.memory_id)),
        "stats": {k: v for k, v in db.stats().items() if isinstance(v, int) and k != "tenant_id"},
        "provenance_len": len(db.provenance(corrected.memory_id).chain),
    }


def test_the_same_workload_answers_the_same_embedded_and_over_the_socket(client, tmp_path):
    """One function, two handles, the same answers.  This is the whole claim in one test.

    The two runs are on two FILES, because a server holding a file read-write excludes every
    other process from it, this one included.  Ids and timestamps differ between the runs and
    are excluded; everything the program actually reasons about is compared.
    """
    with Anatid.open(tmp_path / "embedded.anatid", tenant=1, embedding_dim=DIM) as embedded:
        local = workload(embedded, label="run")
    remote = workload(client, label="run")
    assert remote == local


# --------------------------------------------------------------------------- end to end


def test_the_verbs_come_back_as_the_types_the_embedded_handle_returns(client):
    from anatid.types import Entity, FtsStatus, Memory, RecallHits, SchemaInfo

    memory = client.remember("Ada prefers tea", entities=["Ada"], embedding=[0.5] * DIM)
    assert isinstance(memory, Memory)
    assert memory.embedding is not None and len(memory.embedding) == DIM
    assert isinstance(client.get(memory.memory_id), Memory)
    hits = client.recall("tea")
    assert isinstance(hits, RecallHits)
    assert hits.arms and isinstance(hits.bm25_available, bool)
    assert isinstance(client.get_entity("Ada"), Entity)
    assert isinstance(client.info(), SchemaInfo)
    assert isinstance(client.fts_status(), FtsStatus)
    assert isinstance(client.stats()["memories"], int)
    assert client.doctor().ok is True


def test_a_write_is_visible_to_the_read_after_it(client):
    """No read-your-writes problem to solve: both calls go to the same process.

    DuckDB does not let a client open the file at all while the server holds it, so there is no
    second handle that could lag behind the writer.
    """
    for i in range(20):
        memory = client.remember(f"note {i}")
        got = client.get(memory.memory_id)
        assert got is not None and got.content == f"note {i}"


def test_the_client_reuses_one_connection(client):
    """Twenty calls, one socket.  A client that reconnects per call would pay a connect each time."""
    for i in range(20):
        client.remember(f"reuse {i}")
    assert client._transport._live == 1, "twenty calls, one connection"
    assert len(client._transport._idle) == 1, "and it is back in the pool for the next call"


def test_concurrent_calls_take_concurrent_connections():
    """Four threads reach the server at once, which needs four connections.

    Asserted with a barrier rather than with timing: if the pool handed out fewer than four
    connections the fourth thread could never arrive and the barrier would break.
    """
    barrier = threading.Barrier(4, timeout=10)

    def answer(_request):
        barrier.wait()
        return ok(7)

    handle = AnatidClient(socket_path="/nonexistent/x.sock", tenant=1)
    handle._transport = ScriptedTransport([answer] * 4, max_connections=4)
    results: list[object] = []
    errors: list[BaseException] = []

    def call() -> None:
        try:
            results.append(handle.memory_version(1))
        except BaseException as exc:  # noqa: BLE001 - reported below, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert errors == [], errors
    assert results == [7, 7, 7, 7]
    assert handle._transport.opened == 4


def test_the_context_manager_closes_and_a_closed_client_says_so(served):
    _server, sock_path = served
    with AnatidClient.connect(sock_path, tenant=1) as handle:
        handle.remember("inside the block")
    assert handle.closed
    with pytest.raises(ServerUnavailable) as caught:
        handle.get(1)
    assert "closed" in str(caught.value)


def test_as_of_scopes_every_read_on_the_view(client):
    before = _dt.datetime(2020, 1, 1, 0, 0, 0)
    memory = client.remember("written now", entities=["Ada"])
    view = client.as_of(before)
    assert view.get(memory.memory_id) is None, "a memory written today is not visible in 2020"
    assert view.recall_2hop("Ada") == []
    assert view.context("Ada") == []
    assert client.get(memory.memory_id) is not None
    assert "AsOfClient" in repr(view)
    # An explicit as_of still wins over the view's, as it does on AsOfView.
    assert view.get(memory.memory_id, as_of=None) is not None


def test_the_tenant_travels_in_the_envelope_and_reaches_another_file(client):
    """``tenant=`` on a verb names another tenant, and with a pool that is another FILE.

    The server refuses a ``tenant`` inside the argument object outright, so the client can only
    put it in the envelope; this checks that the envelope is what actually routes the call.
    """
    mine = client.remember("tenant one only")
    assert client.get(mine.memory_id) is not None
    assert client.get(mine.memory_id, tenant=2) is None
    assert client.stats(tenant=2)["memories"] == 0


def test_a_principal_scoped_to_one_tenant_cannot_name_another(pool, sock_dir):
    class OnlyTenantOne:
        requires_token = False

        def authenticate(self, ctx):
            return auth.Principal.for_tenants("only-1", [1], transport=ctx.transport)

    sock_path = sock_dir / "run" / "scoped.sock"
    config = S.ServerConfig(socket_path=sock_path, tenants=(1,), workers=1)
    with running_server(pool=pool, config=config, authenticator=OnlyTenantOne()):
        with AnatidClient.connect(sock_path, tenant=1) as handle:
            handle.remember("mine")
            with pytest.raises(AuthorizationError):
                handle.get(1, tenant=2)
        # The probe is what turns a tenant this connection may not name into an error at
        # connect time rather than at the first call.
        with pytest.raises(AuthorizationError):
            AnatidClient.connect(sock_path, tenant=2)


# --------------------------------------------------------------------------- errors


def test_a_conflict_error_stays_a_conflict_error_with_both_versions(client):
    """The compare-and-swap failure a caller reasons about survives the wire intact."""
    memory = client.remember("Ada prefers tea")
    client.update(memory.memory_id, "Ada prefers coffee", expected_version=memory.version)
    with pytest.raises(ConflictError) as caught:
        client.update(memory.memory_id, "Ada prefers cocoa", expected_version=memory.version)
    error = caught.value
    assert error.expected_version == memory.version
    assert error.current_version is not None and error.current_version != memory.version
    assert error.retryable is False, "a compare-and-swap failure is not retried"
    assert error.resource and str(memory.memory_id) in error.resource


def test_the_error_classes_the_verbs_raise_arrive_as_themselves(client):
    with pytest.raises(NotFoundError):
        client.reinforce(123456789)
    with pytest.raises(ValidationError):
        client.remember("bad confidence", confidence=5.0)
    with pytest.raises(ValidationError):
        client.prune(dry_run=True)


def test_an_unknown_verb_is_refused_with_the_list_of_the_real_ones(client):
    with pytest.raises(ProtocolError) as caught:
        client.call("teleport", memory_id=1)
    assert "teleport" in str(caught.value)
    assert "remember" in str(caught.value), "the refusal names the verbs that do exist"


def test_queue_stats_answers_on_a_server_built_through_the_library(client):
    """A public verb has to work on a server nobody started from the command line.

    It did not, and the reason was an import: ``QueueStats`` was registered with the codec in
    :mod:`anatid.server.cli`, so a server built with ``AnatidServer(pool=...)`` ran the verb and
    then failed to encode the answer, with an error that blamed a version skew that did not
    exist.  ``anatid.server.queue`` registers it now, next to the definition, so every process
    with a write queue can encode it.  The client fixture here never imports the command line.
    """
    from anatid.server.queue import QueueStats

    stats = client.queue_stats()
    assert isinstance(stats, QueueStats)
    assert stats.max_tenant_depth >= 0
    assert client.health().ok is True


def test_the_registration_is_what_makes_it_work_and_a_reply_that_cannot_encode_is_still_an_answer(
    client,
):
    """The mechanism, and the failure mode it used to have, both exercised rather than described.

    Order-independent on purpose: it takes the registration out whatever the starting state,
    asserts the failure, puts it back, asserts the success, and restores what it found.

    Two things at once, and both are worth keeping.  The remedy has to be applied in BOTH
    processes: here they are one process and one registration does both jobs, but across a real
    socket the server needs it to encode the reply and the client needs it to decode one.  And a
    reply the server cannot encode has to come back as a readable error on a connection that
    still works, rather than as a dropped call, because a client waiting for the answer to a
    write it has already run cannot tell a retry from a first attempt.
    """
    from anatid.server.queue import QueueStats

    saved = protocol._DATACLASSES.pop("QueueStats", None)
    try:
        with pytest.raises(ProtocolError) as caught:
            client.queue_stats()
        assert "QueueStats" in str(caught.value)
        assert "could not encode" in str(caught.value)
        assert client.health().ok is True, "the connection survives it"

        protocol.register_dataclass(QueueStats)
        stats = client.queue_stats()
        assert isinstance(stats, QueueStats)
        assert stats.max_tenant_depth >= 0
    finally:
        protocol._DATACLASSES.pop("QueueStats", None)
        if saved is not None:
            protocol._DATACLASSES["QueueStats"] = saved


def test_the_codec_knows_queue_stats_in_a_process_that_never_imports_the_command_line():
    """A fresh interpreter, because in this one another test file has already imported cli.

    This is the assertion the previous shape of these tests could not make.  Anything in this
    suite may import :mod:`anatid.server.cli`, and once it has, a registration that only that
    module performs looks exactly like one performed in the right place.  A subprocess that
    imports only the library is the one place the difference is visible.
    """
    import subprocess
    import sys

    probe = (
        "import anatid.server, anatid.server.server, anatid.server.client, "
        "anatid.server.metrics\n"
        "from anatid.server import protocol\n"
        "assert 'anatid.server.cli' not in __import__('sys').modules, 'the library pulled in cli'\n"
        "print(int('QueueStats' in protocol._DATACLASSES), int('DrainReport' in "
        "protocol._DATACLASSES))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["1", "1"], done.stdout


def test_atomic_says_why_it_is_not_here_instead_of_pretending(client):
    with pytest.raises(NotImplementedError) as caught:
        client.atomic(lambda: None)
    message = str(caught.value)
    assert "transaction" in message
    assert "expected_version" in message, "it names what to use instead"


def test_the_embedded_only_surface_is_absent_and_says_why(client):
    """``execute`` and its neighbours have no wire form, and the AttributeError explains it.

    They stay ABSENT rather than becoming methods that raise, because a caller that duck-types
    on ``hasattr(db, "execute")`` has to keep getting False: one that checks before it calls
    must not end up worse off than one that does not.  What ``__getattr__`` adds is the reason
    on the way out, which is the difference between porting a program and guessing at it.
    """
    assert C._EMBEDDED_ONLY, "the table is the documentation; an empty one documents nothing"
    for name in C._EMBEDDED_ONLY:
        assert not hasattr(client, name), f"{name} must stay absent so duck typing still works"
        assert getattr(client, name, "fallback") == "fallback"
        with pytest.raises(AttributeError) as caught:
            getattr(client, name)
        message = str(caught.value)
        assert name in message
        assert "does not cross the wire" in message
        assert len(message) > 120, f"{name} got a bare AttributeError rather than a reason"


def test_every_embedded_only_name_is_one_the_embedded_handle_really_has():
    """An explanation for a method ``Anatid`` does not have is one nobody will ever read.

    This is the half that rots: a name dropped from ``Anatid`` leaves a sentence here promising
    to explain something that no longer exists.
    """
    missing = sorted(n for n in C._EMBEDDED_ONLY if not hasattr(Anatid, n))
    assert missing == [], f"_EMBEDDED_ONLY explains {missing}, which Anatid does not have"
    overlap = sorted(set(C._EMBEDDED_ONLY) & set(S.VERBS))
    assert overlap == [], f"{overlap} are verbs the client implements; the entries are dead"


def test_a_misspelled_verb_is_still_an_ordinary_attribute_error(client):
    """``__getattr__`` must not turn a typo into something ``hasattr`` cannot swallow."""
    typo = "remmeber"
    assert not hasattr(client, typo)
    with pytest.raises(AttributeError) as caught:
        getattr(client, typo)
    assert typo in str(caught.value)
    assert "AnatidClient" in str(caught.value)


def test_a_maintenance_policy_is_refused_here_rather_than_at_the_far_end(client):
    from anatid.derived import MaintenancePolicy

    with pytest.raises(ProtocolError) as caught:
        client.index_health(policy=MaintenancePolicy())
    assert "MaintenancePolicy" in str(caught.value)
    assert client.index_health() is not None, "no policy is the normal call and still works"


# --------------------------------------------------------------------------- server not running


def test_a_socket_that_does_not_exist_says_so_and_says_what_to_do(sock_dir):
    with pytest.raises(ServerUnavailable) as caught:
        AnatidClient.connect(sock_dir / "nothing-here.sock", tenant=1)
    message = str(caught.value)
    assert "no socket at" in message
    assert "AnatidServer" in message, "the message says how to start one"
    assert caught.value.stage == "connect"


def test_a_socket_left_behind_by_a_dead_server_says_that_instead(sock_dir):
    """A file at the path with nothing listening is the shape a killed server leaves."""
    stale = sock_dir / "stale.sock"
    stale.write_bytes(b"")
    with pytest.raises(ServerUnavailable) as caught:
        AnatidClient.connect(stale, tenant=1)
    assert "could not be connected to" in str(caught.value) or "nothing is listening" in str(
        caught.value
    )


def test_a_refused_socket_names_the_server_that_is_not_there(sock_dir):
    """A real Unix socket that is bound but never accepting still refuses the connection."""
    path = sock_dir / "bound.sock"
    listener = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.close()  # bound, then gone: the inode stays, nothing listens
    with pytest.raises(ServerUnavailable) as caught:
        AnatidClient.connect(path, tenant=1)
    assert caught.value.stage == "connect"
    assert str(path) in str(caught.value)


def test_a_connection_that_fails_to_open_gives_its_slot_back():
    """A leaked pool slot becomes "all connections were busy", a confident wrong diagnosis.

    ``_acquire`` reserves a slot before it calls ``open_one``, so the slot has to come back
    however that call fails.  An OSError was always handled, because that is the ordinary way a
    connection refuses.  Anything else has to give the slot back too: without that, four
    failures against ``max_connections=4`` wedge the client for its whole life and every later
    call reports contention that is not there, on a pool holding no connections at all.
    """

    class Boom(C._Transport):
        address = "test:boom"

        def __init__(self, exc, **kw):
            super().__init__(**kw)
            self.exc = exc

        def open_one(self):
            raise self.exc

        def close_one(self, conn) -> None:
            return None

    for exc in (RuntimeError("not an OSError at all"), ConnectionRefusedError("is an OSError")):
        pool = Boom(exc, max_connections=2, timeout=0.25)
        for attempt in range(4):
            with pytest.raises((RuntimeError, ServerUnavailable)) as caught:
                pool._acquire()
            assert "were busy" not in str(caught.value), (
                f"attempt {attempt + 1} blamed contention on a pool holding nothing"
            )
            assert pool._live == 0, f"{type(exc).__name__} leaked a slot on attempt {attempt + 1}"


def test_the_client_reports_a_server_that_was_killed_under_it(tmp_path, sock_dir):
    """The server dies with the client connected, and the next call says so.

    The server runs in a subprocess and is killed rather than shut down, which is the shape of
    the failure this error exists for: the socket file is still on disk (a killed process never
    unlinks it) and nothing is listening on it.  The client's pooled connection sees the reset,
    reconnects, finds the leftover socket, and reports both facts in one sentence.
    """
    import subprocess
    import sys

    sock_path = sock_dir / "run" / "killed.sock"
    template = str(tmp_path / "tenants" / "t_{tenant}.anatid")
    src = str(Path(__file__).resolve().parent.parent / "src")
    script = f"""
import asyncio
import sys

sys.path.insert(0, {src!r})
from anatid import DatabasePool
from anatid.server import AnatidServer, ServerConfig


async def main():
    pool = DatabasePool({template!r}, embedding_dim=8)
    server = AnatidServer(
        pool=pool, config=ServerConfig(socket_path={str(sock_path)!r}, tenants=(1,), workers=1)
    )
    await server.start()
    print("SERVING", flush=True)
    await asyncio.Event().wait()


asyncio.run(main())
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "SERVING", proc.stderr.read()[-2000:]
        handle = AnatidClient.connect(sock_path, tenant=1)
        assert handle.remember("before the kill").memory_id
    finally:
        proc.kill()
        proc.wait(30)
    with pytest.raises(ServerUnavailable) as caught:
        handle.remember("after the kill")
    assert str(sock_path) in str(caught.value)
    assert caught.value.attempts >= 1
    handle.close()


# --------------------------------------------------------------------------- retries


def test_a_retryable_failure_is_sent_again_with_the_same_idempotency_key():
    """The rule the whole retry design rests on: retry the call, keep the key.

    A new key on the retry would make the server treat the second attempt as a different write,
    which is exactly the duplicate the key exists to prevent.
    """
    handle = scripted([failure(BusyError("full", retry_after=0.01)), ok("written")])
    assert handle.remember("hello") == "written"
    sent = handle._transport.sent
    assert len(sent) == 2
    assert sent[0].idempotency_key == sent[1].idempotency_key is not None
    assert sent[0].request_id != sent[1].request_id, "each attempt is its own message"
    assert sent[0].args == sent[1].args, "the digest the server checks is over the arguments"
    assert handle.waits == [pytest.approx(0.01 * (0.5 + 0.5))]


def test_a_non_retryable_failure_is_raised_unchanged_on_the_first_attempt():
    conflict = ConflictError(
        "memory 7 moved on",
        resource="memory 7",
        expected_version=3,
        current_version=4,
        retryable=False,
    )
    handle = scripted([failure(conflict)])
    with pytest.raises(ConflictError) as caught:
        handle.update(7, "new", expected_version=3)
    assert caught.value.current_version == 4
    assert len(handle._transport.sent) == 1, "a compare-and-swap failure is not sent again"
    assert handle.waits == []


def test_the_client_gives_up_after_its_attempts_and_raises_the_last_error():
    handle = scripted([failure(BusyError("full", retry_after=0.001))] * 3)
    with pytest.raises(BusyError):
        handle.remember("hello")
    assert len(handle._transport.sent) == 3
    assert len(handle.waits) == 2, "three attempts, two waits"


def test_backpressure_is_waited_out_with_the_wait_the_server_measured():
    """``retry_after`` comes from the rate that tenant's queue is actually draining at.

    Replacing it with a local constant would be guessing at something the server measured, so
    the client spreads the server's number rather than ignoring it.
    """
    policy = RetryPolicy(attempts=2, sleep=lambda _s: None, rng=lambda: 0.0)
    assert policy.delay(1, retry_after=2.0) == pytest.approx(1.0)
    policy_high = RetryPolicy(attempts=2, sleep=lambda _s: None, rng=lambda: 1.0)
    assert policy_high.delay(1, retry_after=2.0) == pytest.approx(3.0)
    assert policy_high.delay(1, retry_after=1000.0) == pytest.approx(
        policy_high.max_retry_after * 1.5
    ), "the server's estimate is capped, not trusted without limit"


def test_a_dropped_connection_is_retried_for_a_read_and_for_a_keyed_write():
    """A read can always be sent again.  A keyed write can, because the key makes it write once."""
    handle = scripted([BrokenPipeError("gone"), ok("second try")])
    assert handle.get(7) == "second try"

    keyed = scripted([BrokenPipeError("gone"), ok("written once")])
    assert keyed.remember("hello") == "written once"
    assert keyed._transport.sent[0].idempotency_key == keyed._transport.sent[1].idempotency_key


def test_an_unkeyed_write_is_not_retried_through_an_ambiguous_failure():
    """Without a key, a connection that died in flight might have committed the write.

    Sending it again could write a second memory, and a duplicate memory is worse than an error
    that says the outcome is unknown.  This is the whole reason the keys are on by default.
    """
    handle = scripted(
        [BrokenPipeError("gone"), ok("would have been a duplicate")], idempotency=False
    )
    with pytest.raises(ServerUnavailable) as caught:
        handle.remember("hello")
    assert len(handle._transport.sent) == 1, "it was not sent again"
    assert caught.value.stage == "exchange"
    assert "not known here" in str(caught.value)


def test_an_unkeyed_write_is_still_retried_when_the_server_says_it_did_not_happen():
    """``BusyError`` is the one failure that reports the write was NOT performed."""
    handle = scripted(
        [
            failure(BusyError("queue full, the write was NOT performed", retry_after=0.001)),
            ok("written"),
        ],
        idempotency=False,
    )
    assert handle.remember("hello") == "written"
    assert len(handle._transport.sent) == 2
    assert handle._transport.sent[0].idempotency_key is None


def test_a_draining_server_is_not_retried_against_itself():
    """``ShuttingDown`` is retryable against ANOTHER instance; this client is bound to this one."""
    handle = scripted([failure(ShuttingDown("draining"))])
    with pytest.raises(ShuttingDown):
        handle.remember("hello")
    assert len(handle._transport.sent) == 1
    assert handle.waits == []


def test_a_retry_after_a_lost_reply_writes_once(client):
    """The sequence the idempotency key exists for, run against a real server.

    The reply to the first write is thrown away and the same call is sent again with the same
    key.  One memory, and the second call returns the first one's result.
    """
    key = "the-same-key"
    first = client.call(
        "remember", idempotency_key=key, content="written exactly once", entities=["Ada"]
    )
    again = client.call(
        "remember", idempotency_key=key, content="written exactly once", entities=["Ada"]
    )
    assert again.memory_id == first.memory_id
    assert client.stats()["memories"] == 1


def test_a_key_reused_for_a_different_call_is_refused(client):
    from anatid.server.protocol import IdempotencyConflict

    client.call("remember", idempotency_key="k1", content="first")
    with pytest.raises(IdempotencyConflict) as caught:
        client.call("remember", idempotency_key="k1", content="something else entirely")
    assert caught.value.key == "k1"
    assert client.stats()["memories"] == 1


def test_many_threads_writing_through_one_client_all_land(pool, sock_dir):
    """Eight threads, one client, one tenant.  Every write lands exactly once.

    This is the property the embedded profile cannot give across processes and gives across
    threads: the difference the server makes is that the threads may now be in different
    processes, which ``examples/server_demo.py`` shows.  The retry policy is the default, so a
    busy answer under this load has to be absorbed by it rather than surfaced.
    """
    sock_path = sock_dir / "run" / "threads.sock"
    config = S.ServerConfig(socket_path=sock_path, tenants=(1,), workers=1, batch_max=8)
    with running_server(pool=pool, config=config):
        with AnatidClient.connect(sock_path, tenant=1, max_connections=8) as handle:
            errors: list[BaseException] = []
            written: list[int] = []
            lock = threading.Lock()

            def write(worker: int) -> None:
                try:
                    for i in range(5):
                        memory = handle.remember(f"worker {worker} note {i}", entities=["Ada"])
                        with lock:
                            written.append(memory.memory_id)
                except BaseException as exc:  # noqa: BLE001 - asserted below, not swallowed
                    errors.append(exc)

            threads = [threading.Thread(target=write, args=(w,)) for w in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(120)
            assert errors == [], errors
            assert len(set(written)) == 40
            assert handle.stats()["memories"] == 40
            assert len(handle.recall_2hop("Ada", limit=100)) == 40


def test_two_processes_write_the_same_tenant_through_one_server(tmp_path, sock_dir):
    """The whole point of the server profile, run as two real processes.

    DuckDB allows one read-write process per file, so these two would collide on ``Anatid.open``
    and the second would get an IOException.  Through the server they both write the same tenant
    and both sets of memories are there afterwards.
    """
    import subprocess
    import sys

    sock_path = sock_dir / "run" / "procs.sock"
    template = str(tmp_path / "tenants" / "t_{tenant}.anatid")
    src = str(Path(__file__).resolve().parent.parent / "src")
    worker = f"""
import sys

sys.path.insert(0, {src!r})
from anatid.server.client import AnatidClient


label = sys.argv[1]
with AnatidClient.connect({str(sock_path)!r}, tenant=1) as db:
    for i in range(20):
        db.remember(f"{{label}} wrote {{i}}", entities=[label])
    print("WROTE", label, len(db.recall_2hop(label, limit=100)))
"""

    pool = DatabasePool(template, embedding_dim=DIM, max_open=4)
    config = S.ServerConfig(socket_path=sock_path, tenants=(1,), workers=1, batch_max=8)
    try:
        with running_server(pool=pool, config=config):
            procs = [
                subprocess.Popen(
                    [sys.executable, "-c", worker, name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for name in ("alpha", "beta")
            ]
            outs = [p.communicate(timeout=180) for p in procs]
            for proc, (out, err) in zip(procs, outs):
                assert proc.returncode == 0, err[-3000:]
                assert out.split()[0] == "WROTE", out
                assert out.split()[2] == "20", out
            with AnatidClient.connect(sock_path, tenant=1) as reader:
                assert reader.stats()["memories"] == 40
                assert len(reader.recall_2hop("alpha", limit=100)) == 20
                assert len(reader.recall_2hop("beta", limit=100)) == 20
    finally:
        pool.close_all()


# --------------------------------------------------------------------------- the wire


def test_f32_embeddings_round_trip_within_float32(served):
    """``embeddings="f32"`` is 3 to 5 times smaller and faster; it is float32, and says so."""
    _server, sock_path = served
    values = [i / 100.0 for i in range(DIM)]
    with AnatidClient.connect(sock_path, tenant=1, embeddings="f32") as handle:
        memory = handle.remember("with an embedding", embedding=values)
        back = handle.get(memory.memory_id)
    assert back is not None and back.embedding is not None
    assert back.embedding == pytest.approx(values, abs=1e-6)


def test_the_client_refuses_a_wire_setting_it_cannot_honour():
    with pytest.raises(ValueError, match="embeddings"):
        AnatidClient(socket_path="/x.sock", embeddings="protobuf")
    with pytest.raises(ValueError, match="exactly one"):
        AnatidClient()
    with pytest.raises(ValueError, match="exactly one"):
        AnatidClient(socket_path="/x.sock", url="http://127.0.0.1:1")
    with pytest.raises(ValueError, match="at least 1"):
        RetryPolicy(attempts=0)


def test_a_deadline_travels_as_a_relative_budget():
    handle = scripted([ok(None)], deadline=2.5)
    handle.get(1)
    assert handle._transport.sent[0].deadline == 2.5
    other = scripted([ok(None)], deadline=None)
    other.get(1)
    assert other._transport.sent[0].deadline is None


def test_the_probe_settles_at_connect_time_that_the_server_is_there(client):
    assert client.server_info is not None
    assert client.server_info.protocol_version == protocol.PROTOCOL_VERSION
    assert client.server_info.ok is True
    assert client.tenant_id == 1
    assert "tenant=1" in repr(client)


def test_connecting_without_a_probe_costs_no_round_trip(served):
    _server, sock_path = served
    handle = AnatidClient.connect(sock_path, tenant=1, probe=False)
    try:
        assert handle.server_info is None
        assert handle._transport._live == 0, "no connection is opened until the first call"
        handle.remember("first call opens it")
        assert handle._transport._live == 1
    finally:
        handle.close()


# --------------------------------------------------------------------------- http


def test_http_speaks_the_same_verbs_with_a_bearer_token(pool):
    token = "a-test-token"
    config = S.ServerConfig(
        socket_path=None, http_host="127.0.0.1", http_port=0, tenants=(1,), workers=1
    )
    with running_server(
        pool=pool, config=config, authenticator=auth.BearerTokenAuthenticator({token: [1]})
    ) as server:
        port = server._servers[0].sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        with AnatidClient.connect_http(url, tenant=1, token=token) as handle:
            memory = handle.remember("over http", entities=["Ada"])
            assert handle.get(memory.memory_id).content == "over http"
            assert handle.recall("http").arms
            assert handle.health().ok is True
        with pytest.raises(AuthenticationError):
            AnatidClient.connect_http(url, tenant=1, token="wrong")
        with pytest.raises(AuthenticationError):
            AnatidClient.connect_http(url, tenant=1)


def test_http_says_so_when_the_route_is_not_an_anatid_server(pool):
    """A URL pointing at something that is not ``/rpc`` produces a readable error, not a crash."""
    config = S.ServerConfig(
        socket_path=None, http_host="127.0.0.1", http_port=0, tenants=(1,), workers=1
    )
    with running_server(pool=pool, config=config) as server:
        port = server._servers[0].sockets[0].getsockname()[1]
        with pytest.raises(ProtocolError) as caught:
            AnatidClient.connect_http(f"http://127.0.0.1:{port}/elsewhere", tenant=1)
    assert "/rpc" in str(caught.value), "the message names the one route that speaks anatid"


# --------------------------------------------------------------------------- ids on the wire
#
# The codec's rule is asserted in tests/test_server_protocol.py.  What is asserted here is that
# the rule reaches every REPLY the server actually sends, on a real socket, with real ids: the
# sweep is over the frames, not over the four fields somebody remembered.

NODE = shutil.which("node")
requires_node = pytest.mark.skipif(
    NODE is None,
    reason=(
        "node is not on PATH; the round trip that proves a 63-bit id survives needs a real "
        "JavaScript parser, and a Python one would prove nothing (Python has no 2**53 limit)"
    ),
)

NODE_ROUND_TRIP = """
let raw = "";
process.stdin.on("data", (chunk) => { raw += chunk; });
process.stdin.on("end", () => {
  process.stdout.write(JSON.stringify(JSON.parse(raw)));
});
"""


def through_node(payload: str) -> str:
    done = subprocess.run(
        [str(NODE), "-e", NODE_ROUND_TRIP],
        input=payload,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def unsafe_ints(value, path: str = "$") -> list:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return (
            []
            if value <= JS_MAX_SAFE_INTEGER and value >= -JS_MAX_SAFE_INTEGER
            else [(path, value)]
        )
    if isinstance(value, dict):
        found = []
        for key, item in value.items():
            found.extend(unsafe_ints(item, f"{path}.{key}"))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for index, item in enumerate(value):
            found.extend(unsafe_ints(item, f"{path}[{index}]"))
        return found
    return []


def every_reply(sock_path):
    """One raw reply frame per verb the server dispatches, as ``(verb, bytes)``.

    Raw bytes rather than decoded values, because what is being asserted is what left the
    process.  A decoded reply has already had the tagged ids turned back into Python ints and
    would prove nothing about the JSON a JavaScript client parses.
    """
    from anatid.server.server import connect_unix

    sock = connect_unix(sock_path, timeout=30.0)
    replies: list[tuple[str, bytes]] = []

    def send(verb, **args):
        protocol.write_frame(sock, Request(verb=verb, tenant=1, args=args))
        body = protocol.read_frame(sock)
        assert body is not None, f"{verb} closed the connection"
        replies.append((verb, body))
        return Response.decode(body).raise_for_status()

    try:
        ada = send("upsert_entity", name="Ada", kind="person")
        send("entity_id", value="Grace", create=True)
        send("relate", src="Ada", dst="Grace", rel_kind="knows")
        episode = send("episode", content="the morning session", source="test")
        first = send(
            "remember",
            content="Ada prefers tea",
            entities=["Ada"],
            embedding=[0.5] * DIM,
            episode_id=episode.episode_id,
        )
        second = send("supersede", old_id=first.memory_id, content="Ada prefers coffee")
        moved = send(
            "correct",
            old_id=second.memory_id,
            content="Ada prefers coffee, Grace prefers tea",
            add_relations=[["Ada", "coffee", "drinks"]],
            remove_relations=[["Ada", "Grace"]],
        )
        second = moved.new
        send("update", memory_id=second.memory_id, content="Ada prefers cocoa")
        send("reinforce", memory_id=second.memory_id, amount=2)
        send("get", memory_id=second.memory_id)
        send("versions", memory_id=second.memory_id)
        send("memory_version", memory_id=second.memory_id)
        send("provenance", memory_id=second.memory_id)
        send("entities_of", memory_id=second.memory_id)
        send("get_entity", entity="Ada")
        send("get_episode", episode_id=episode.episode_id)
        send("recall", query="cocoa", k=5)
        send("recall_2hop", seed_entity="Ada")
        send("recall_2hop_ids", seed_entity="Ada")
        send("context", entity="Ada")
        send("stats")
        send("doctor")
        send("index_health")
        send("info")
        send("fts_status")
        send("rebuild_fts_index")
        send("maintain_indexes")
        send("recluster")
        send("prune", dry_run=True, max_access_count=0)
        send("unrelate", src="Ada", dst="Grace")
        send("forget", memory_id=second.memory_id)
        send("health")
        send("ready")
        send("queue_stats")
        assert ada.name == "Ada"
        return replies
    finally:
        sock.close()


@pytest.fixture
def replies(served):
    _server, sock_path = served
    return every_reply(sock_path)


def test_the_sweep_covers_every_verb_the_server_dispatches(replies):
    """A verb nobody sweeps is a verb that can leak an id."""
    swept = {verb for verb, _body in replies}
    assert swept == set(S.VERBS), (
        f"verbs the server has that this sweep never calls: {sorted(set(S.VERBS) - swept)}; "
        f"names swept that the server does not dispatch: {sorted(swept - set(S.VERBS))}"
    )


def test_the_replies_carry_ids_a_json_number_could_not_hold(replies):
    """The control.  Without it the sweep below could pass on a server that returned no ids."""
    tagged = sum(
        body.count(f'"{protocol.TAG}":"{protocol.ID_TAG}"'.encode()) for _v, body in replies
    )
    assert tagged >= 20, f"only {tagged} ids in {len(replies)} replies; the sweep proves nothing"


def test_no_reply_puts_an_id_on_the_wire_as_a_json_number(replies):
    offenders = []
    for verb, body in replies:
        offenders.extend((verb, path, value) for path, value in unsafe_ints(json.loads(body), "$"))
    assert not offenders, (
        f"{len(offenders)} integer(s) went out as JSON numbers that a JavaScript client rounds: "
        + ", ".join(f"{verb} {path} = {value}" for verb, path, value in offenders)
    )


def test_an_error_reply_does_not_leak_an_id_as_a_number_either(client, served):
    """``WireError.details`` carries ``id``, so a failure is the same boundary as a success."""
    from anatid.server.server import connect_unix

    memory = client.remember("for the error path")
    sock = connect_unix(served[1], timeout=30.0)
    try:
        protocol.write_frame(
            sock,
            Request(
                verb="remember", tenant=1, args={"content": "clash", "memory_id": memory.memory_id}
            ),
        )
        body = protocol.read_frame(sock)
    finally:
        sock.close()
    assert body is not None
    payload = json.loads(body)
    assert payload["status"] == "error", payload
    assert not unsafe_ints(payload, "$error"), payload


@requires_node
def test_every_reply_survives_a_real_javascript_parser(replies):
    for verb, body in replies:
        sent = body.decode("utf-8")
        assert json.loads(through_node(sent)) == json.loads(sent), (
            f"the reply to {verb} changed in a JavaScript JSON parser"
        )


@requires_node
def test_an_id_that_has_been_through_node_still_addresses_the_row(client):
    """The whole failure, end to end: server, JavaScript, server again.

    Twelve ids went out as JSON numbers before this was fixed, twelve came back with different
    digits, and twelve follow-up ``get`` calls returned None rather than an error.
    """
    written = [client.remember(f"through node {i}") for i in range(12)]
    sent = protocol.dumps({"ids": [m.memory_id for m in written]}).decode("utf-8")
    returned = protocol.loads(through_node(sent))["ids"]
    assert returned == [m.memory_id for m in written], "an id changed on the way through node"
    for memory_id, original in zip(returned, written):
        got = client.get(memory_id)
        assert got is not None and got.content == original.content
