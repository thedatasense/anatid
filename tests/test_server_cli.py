"""The operator command line: ``anatid-server start|stop|status|backup|restore|doctor``.

The tests that matter here start a real server in a real subprocess on a real Unix socket and
then operate it with the same command an operator would type.  A CLI tested only by calling its
functions in process proves that argparse works; it does not prove that the socket appears, that
SIGTERM drains, that the file is released afterwards, or that the exit code a supervisor reads is
the one the code intended.  Those four are the whole reason this module exists.

The in-process tests cover the parts that must NOT do I/O: ``--check`` refusing a configuration
before anything is created, the token file grammar, and the exit-code contract.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import socket

from anatid import Anatid
from anatid.server import auth, protocol
from anatid.server import cli as C

pytestmark = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="the server profile's Unix domain socket transport is not available on this platform; use the HTTP transport",
)


DIM = 8
REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_SRC = REPO_ROOT / "src"

#: ``sun_path`` is 104 bytes on macOS, and pytest's ``tmp_path`` is nested deeply enough to
#: overflow it, so socket paths come from :func:`sock_dir` rather than from ``tmp_path``.
SOCKADDR_UN_MAX = 100


@pytest.fixture
def sock_dir():
    """A directory short enough to hold a Unix socket path."""
    base = Path(tempfile.mkdtemp(prefix="anatid-c-"))
    if len(str(base)) + len("/xxxxxxxx.sock") > SOCKADDR_UN_MAX:
        shutil.rmtree(base, ignore_errors=True)
        base = Path(tempfile.mkdtemp(prefix="anatid-c-", dir="/tmp"))
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


# --------------------------------------------------------------------------- subprocess glue


def _env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{REPO_SRC}{os.pathsep}{existing}" if existing else str(REPO_SRC)
    return env


def cli(*args: object, timeout: float = 120) -> subprocess.CompletedProcess:
    """Run ``python -m anatid.server ...`` and return the finished process.

    The module entry point rather than the console script, because the console script only
    exists once the package is installed and this has to work from a source checkout too.  They
    are the same function.
    """
    return subprocess.run(
        [sys.executable, "-m", "anatid.server", *[str(a) for a in args]],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_env(),
        check=False,
    )


class ServerProcess:
    """A running ``anatid-server start`` and what it printed when it stopped."""

    def __init__(self, proc: subprocess.Popen, socket_path: Path) -> None:
        self.proc = proc
        self.socket_path = socket_path
        self.returncode: int | None = None
        self.stdout = ""
        self.stderr = ""

    def wait_ready(self, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.finish()
                raise AssertionError(
                    f"the server exited before it listened (rc={self.returncode}):\n"
                    f"{self.stderr[-3000:]}"
                )
            if self.socket_path.exists():
                with contextlib.suppress(OSError):
                    from anatid.server.server import connect_unix

                    connect_unix(self.socket_path, timeout=2.0).close()
                    return
            time.sleep(0.02)
        raise AssertionError(f"{self.socket_path} never appeared")

    def finish(self, timeout: float = 60.0) -> int:
        if self.returncode is None:
            try:
                self.stdout, self.stderr = self.proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.stdout, self.stderr = self.proc.communicate(timeout=30)
                raise
            self.returncode = self.proc.returncode
        return self.returncode

    def terminate(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=30)
        if self.proc.poll() is None:  # pragma: no cover - only if a drain wedges
            self.proc.kill()
        self.finish()


@contextlib.contextmanager
def running(socket_path: Path, template: str, *extra: object, tenant: int = 1):
    """Start a server, wait until it is listening, and make sure it is gone afterwards."""
    argv = [
        sys.executable,
        "-m",
        "anatid.server",
        "start",
        "--socket",
        str(socket_path),
        "--pool",
        template,
        "--tenant",
        str(tenant),
        "--embedding-dim",
        str(DIM),
        *[str(a) for a in extra],
    ]
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_env()
    )
    server = ServerProcess(proc, socket_path)
    try:
        server.wait_ready()
        yield server
    finally:
        server.terminate()


def remember(socket_path: Path, tenant: int, *contents: str) -> None:
    """Write memories through a running server, the way a client would."""
    from anatid.server.server import connect_unix

    sock = connect_unix(socket_path, timeout=30.0)
    try:
        for content in contents:
            protocol.write_frame(
                sock, protocol.Request(verb="remember", tenant=tenant, args={"content": content})
            )
            body = protocol.read_frame(sock)
            assert body is not None
            protocol.Response.decode(body).raise_for_status()
    finally:
        sock.close()


def rows(path: Path) -> int:
    with Anatid.open(path, tenant=1, embedding_dim=DIM) as db:
        row = db.execute("SELECT count(*) FROM memories").fetchone()
        return int(row[0]) if row else 0


# --------------------------------------------------------------------------- --check


def test_check_accepts_a_usable_configuration_and_creates_nothing(sock_dir, tmp_path):
    """The point of ``--check`` is a deploy that fails before it has changed anything."""
    store = tmp_path / "store"
    store.mkdir()
    done = cli(
        "start",
        "--check",
        "--socket",
        sock_dir / "a.sock",
        "--pool",
        str(store / "t_{tenant}.anatid"),
        "--tenant",
        1,
    )
    assert done.returncode == C.EXIT_OK, done.stderr
    assert "the configuration is usable" in done.stdout
    assert "unix:" in done.stdout
    assert list(store.iterdir()) == [], "a check that creates a database file is not a check"
    assert not (sock_dir / "a.sock").exists(), "a check must not bind anything"


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ((), "exactly one of --db"),
        (("--db", "x.anatid", "--pool", "t_{tenant}.anatid"), "exactly one of --db"),
        (("--pool", "no-placeholder.anatid"), "must contain {tenant}"),
        (("--db", "x.anatid", "--max-queue", "0"), "--max-queue must be at least 1"),
        (("--db", "x.anatid", "--workers", "0"), "--workers must be at least 1"),
        (("--db", "x.anatid", "--drain-timeout", "-1"), "--drain-timeout must not be negative"),
        (("--db", "x.anatid", "--embedding-dim", "0"), "--embedding-dim must be positive"),
        (("--db", "x.anatid", "--backup-dir", "b"), "--backup-dir needs --pool"),
    ],
)
def test_check_refuses_a_configuration_that_would_not_work(tmp_path, sock_dir, extra, expected):
    done = cli("start", "--check", "--socket", sock_dir / "a.sock", *extra)
    assert done.returncode == C.EXIT_CONFIG, done.stdout + done.stderr
    assert expected in done.stderr, done.stderr


def test_check_refuses_a_server_with_no_transport(tmp_path):
    done = cli("start", "--check", "--db", str(tmp_path / "x.anatid"))
    assert done.returncode == C.EXIT_CONFIG
    assert "at least one transport" in done.stderr


def test_check_refuses_a_socket_path_longer_than_sockaddr_un(tmp_path):
    long_path = tmp_path / ("d" * 40) / ("e" * 40) / ("f" * 40) / "anatid.sock"
    done = cli("start", "--check", "--socket", long_path, "--db", str(tmp_path / "x.anatid"))
    assert done.returncode == C.EXIT_CONFIG
    assert "sockaddr_un" in done.stderr, done.stderr


def test_http_without_a_token_file_is_refused_even_on_loopback(tmp_path, sock_dir):
    """Loopback is not a boundary on a machine with more than one user on it.

    ``check_bind_address`` only refuses a NON-loopback bind without a token, so the server
    itself accepts ``127.0.0.1`` with no authentication.  That is right for a single-user box
    and wrong on a shared host, where any other local user can reach the port.  The CLI makes
    the operator say so explicitly.
    """
    done = cli(
        "start",
        "--check",
        "--db",
        str(tmp_path / "x.anatid"),
        "--http",
        "127.0.0.1:8787",
    )
    assert done.returncode == C.EXIT_CONFIG
    assert "--http needs --token-file" in done.stderr
    assert "--http-no-auth" in done.stderr


def test_a_public_bind_is_refused_even_with_the_no_auth_flag(tmp_path):
    """``--http-no-auth`` waives the CLI's rule, not the server's."""
    done = cli(
        "start",
        "--check",
        "--db",
        str(tmp_path / "x.anatid"),
        "--http",
        "0.0.0.0:8787",
        "--http-no-auth",
    )
    assert done.returncode == C.EXIT_CONFIG
    assert "refusing to bind HTTP to '0.0.0.0'" in done.stderr


def test_a_public_bind_with_a_token_file_is_accepted(tmp_path, sock_dir):
    tokens = tmp_path / "tokens"
    tokens.write_text("s3cret name=admin\n", encoding="utf-8")
    tokens.chmod(0o600)
    done = cli(
        "start",
        "--check",
        "--db",
        str(tmp_path / "x.anatid"),
        "--http",
        "0.0.0.0:8787",
        "--token-file",
        tokens,
    )
    assert done.returncode == C.EXIT_OK, done.stderr
    assert "http://0.0.0.0:8787" in done.stdout
    assert "s3cret" not in done.stdout, "the CLI must never echo a token"


# --------------------------------------------------------------------------- token files


def test_a_token_file_maps_tokens_to_principals(tmp_path):
    path = tmp_path / "tokens"
    path.write_text(
        "# the admin token\naaa name=admin\n\nbbb name=reader tenants=1,2 read-only\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    tokens = C.read_token_file(path)
    assert set(tokens) == {"aaa", "bbb"}
    assert tokens["aaa"].name == "admin"
    assert tokens["aaa"].tenants is None
    assert tokens["bbb"].tenants == frozenset({1, 2})
    assert tokens["bbb"].read_only is True


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", "holds no tokens"),
        ("# only a comment\n", "holds no tokens"),
        ("aaa tenants=one\n", "not a comma separated list"),
        ("aaa colour=blue\n", "unknown field"),
        ("aaa bare\n", "not 'key=value'"),
        ("aaa\naaa name=twice\n", "repeats an earlier token"),
    ],
)
def test_a_broken_token_file_is_a_configuration_error(tmp_path, text, expected):
    path = tmp_path / "tokens"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(C.ConfigProblem, match=expected):
        C.read_token_file(path)


def test_a_world_readable_token_file_warns_but_still_loads(tmp_path, capsys):
    """It warns rather than refuses: a secret mounted into a container is often 0444."""
    path = tmp_path / "tokens"
    path.write_text("aaa\n", encoding="utf-8")
    path.chmod(0o644)
    assert set(C.read_token_file(path)) == {"aaa"}
    assert "chmod 600" in capsys.readouterr().err


# --------------------------------------------------------------------------- host:port


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("127.0.0.1:8787", ("127.0.0.1", 8787)),
        ("localhost:1", ("localhost", 1)),
        ("0.0.0.0", ("0.0.0.0", 8787)),
        (":9000", ("127.0.0.1", 9000)),
        ("[::1]:9001", ("::1", 9001)),
    ],
)
def test_host_port_forms(spec, expected):
    assert C._parse_host_port(spec) == expected


@pytest.mark.parametrize("spec", ["", "host:notaport", "host:0", "host:70000", "::1:8787"])
def test_a_malformed_host_port_is_a_configuration_error(spec):
    with pytest.raises(C.ConfigProblem):
        C._parse_host_port(spec)


# --------------------------------------------------------------------------- authentication


def test_one_authenticator_serves_both_transports(tmp_path):
    """The trap this class exists for: a token file must not lock out the Unix socket.

    ``AnatidServer`` holds ONE authenticator.  Given a bare ``BearerTokenAuthenticator``, a Unix
    connection carries no ``Authorization`` header and is refused, so
    ``start --socket S --http H --token-file F`` would serve HTTP and nothing else.
    """
    tokens = tmp_path / "tokens"
    tokens.write_text("aaa name=admin\nbbb name=one tenants=1\n", encoding="utf-8")
    tokens.chmod(0o600)
    args = C.build_parser().parse_args(
        [
            "start",
            "--socket",
            str(tmp_path / "s.sock"),
            "--http",
            "127.0.0.1:8787",
            "--token-file",
            str(tokens),
            "--db",
            str(tmp_path / "x.anatid"),
        ]
    )
    authenticator = C._build_authenticator(args)
    assert authenticator.requires_token is True

    unix = authenticator.authenticate(auth.ConnectionContext(transport="unix"))
    assert unix.unrestricted, "a Unix connection is authenticated by the socket's permissions"

    scoped = authenticator.authenticate(
        auth.ConnectionContext(transport="http", headers={"authorization": "Bearer bbb"})
    )
    assert scoped.tenants == frozenset({1})

    with pytest.raises(protocol.AuthenticationError):
        authenticator.authenticate(auth.ConnectionContext(transport="http", headers={}))


def test_allow_uid_restricts_the_unix_socket_to_named_uids(tmp_path):
    args = C.build_parser().parse_args(
        [
            "start",
            "--socket",
            str(tmp_path / "s.sock"),
            "--allow-uid",
            "0",
            "--db",
            str(tmp_path / "x.anatid"),
        ]
    )
    authenticator = C._build_authenticator(args)
    assert isinstance(authenticator.unix, auth.UnixPeerAuthenticator)
    assert authenticator.unix.allow_uids == frozenset({0})


# --------------------------------------------------------------------------- lifecycle


def test_start_serves_then_stop_drains_and_releases_the_file(sock_dir, tmp_path):
    """The whole operator loop, in a subprocess, on a real socket.

    Start it, ask it how it is, write through it, stop it, and then check the three things a
    restart depends on: the socket is gone, the writes are on disk, and the file is no longer
    locked, which it would be if the process had exited without releasing it.
    """
    sock = sock_dir / "life.sock"
    template = str(tmp_path / "t_{tenant}.anatid")
    pid_file = tmp_path / "anatid.pid"
    with running(sock, template, "--pid-file", pid_file) as server:
        assert pid_file.read_text(encoding="utf-8").strip() == str(server.proc.pid)

        status = cli("status", "--socket", sock)
        assert status.returncode == C.EXIT_OK, status.stderr
        assert "status         serving" in status.stdout
        assert "ready          yes" in status.stdout
        assert f"pid {server.proc.pid}" in status.stdout

        remember(sock, 1, "the first note", "the second note")

        stopped = cli("stop", "--socket", sock)
        assert stopped.returncode == C.EXIT_OK, stopped.stderr
        assert "stopped" in stopped.stdout
        assert server.finish() == C.EXIT_OK, server.stderr[-3000:]

    assert not sock.exists(), "a clean shutdown removes its socket"
    assert not pid_file.exists(), "a clean shutdown removes its pid file"
    assert "drained" in server.stdout, server.stdout
    # Opening it at all is the assertion: a process that exited holding the file would make
    # this raise IOException, and DuckDB refuses a second opener even read-only.
    assert rows(tmp_path / "t_1.anatid") == 2


def test_stop_takes_the_pid_from_a_pid_file_when_it_is_given_one(sock_dir, tmp_path):
    """The path for an operator whose socket is gone but whose process is not."""
    sock = sock_dir / "pidf.sock"
    pid_file = tmp_path / "anatid.pid"
    with running(sock, str(tmp_path / "t_{tenant}.anatid"), "--pid-file", pid_file) as server:
        stopped = cli("stop", "--pid-file", pid_file)
        assert stopped.returncode == C.EXIT_OK, stopped.stderr
        assert server.finish() == C.EXIT_OK, server.stderr[-3000:]
    assert not pid_file.exists()


def test_stopping_a_server_that_is_not_running_succeeds_and_says_so(sock_dir):
    """``stop`` is what an init script runs; it has to be idempotent."""
    done = cli("stop", "--socket", sock_dir / "absent.sock")
    assert done.returncode == C.EXIT_OK, done.stderr
    assert "nothing to stop" in done.stdout


def test_status_against_nothing_fails_rather_than_reporting_health(sock_dir):
    done = cli("status", "--socket", sock_dir / "absent.sock")
    assert done.returncode == C.EXIT_FAILED
    assert "no socket at" in done.stderr


def test_sigterm_exits_within_the_drain_timeout_with_a_client_still_connected(sock_dir, tmp_path):
    """The bound a supervisor relies on, with a real signal to a real process.

    A live deployment always has connections open, and this is the case that used not to work:
    ``shutdown`` waited on ``asyncio.Server.wait_closed`` before it stopped the queue or
    cancelled the handlers, and that wait does not return while a handler is running.  Measured
    then, with one idle connection and a 5 second budget: no exit within 25 seconds, and an exit
    0.03 seconds after the client closed its socket.  Under a supervisor with a TERM-then-KILL
    policy that means the server is killed rather than drained.

    The connection here is deliberately left open across the signal.  ``sock`` is closed only in
    the ``finally``, after the process has already gone.
    """
    from anatid.server.server import connect_unix

    sock_path = sock_dir / "term.sock"
    with running(sock_path, str(tmp_path / "t_{tenant}.anatid"), "--drain-timeout", "2") as server:
        held = connect_unix(sock_path, timeout=30.0)
        try:
            protocol.write_frame(
                held, protocol.Request(verb="remember", tenant=1, args={"content": "before"})
            )
            body = protocol.read_frame(held)
            assert body is not None
            protocol.Response.decode(body).raise_for_status()

            started = time.monotonic()
            server.proc.terminate()
            code = server.finish(timeout=60)
            elapsed = time.monotonic() - started
        finally:
            held.close()

    assert code == C.EXIT_OK, server.stderr[-3000:]
    assert elapsed < 15.0, (
        f"SIGTERM to exit took {elapsed:.1f}s with one idle client connected and a 2s drain timeout"
    )
    assert not sock_path.exists(), "a clean shutdown removes its socket"
    assert rows(tmp_path / "t_1.anatid") == 1, "the write acknowledged before the signal is there"


def test_the_queue_counters_cross_the_wire(sock_dir, tmp_path):
    """``queue_stats`` returns a ``QueueStats``, and ``status`` prints the numbers it carries.

    It reaches the operator only if the server can encode it, which is why
    :mod:`anatid.server.queue` registers the type where it is defined rather than leaving it to
    whichever module happens to be imported.
    """
    sock = sock_dir / "stats.sock"
    with running(sock, str(tmp_path / "t_{tenant}.anatid")):
        remember(sock, 1, "one", "two", "three")
        status = cli("status", "--socket", sock)
    assert status.returncode == C.EXIT_OK, status.stderr
    assert "queue totals   submitted=3 completed=3" in status.stdout, status.stdout
    assert "writes per transaction" in status.stdout


def test_status_json_is_machine_readable(sock_dir, tmp_path):
    import json

    sock = sock_dir / "json.sock"
    with running(sock, str(tmp_path / "t_{tenant}.anatid")):
        done = cli("status", "--socket", sock, "--json")
    assert done.returncode == C.EXIT_OK, done.stderr
    payload = json.loads(done.stdout)
    assert payload["health"]["status"] == "serving"
    assert payload["ready"]["ready"] is True
    assert payload["ready"]["expected_schema_version"] >= 4
    assert payload["queue"]["max_depth"] >= 1


# --------------------------------------------------------------------------- backup


@pytest.fixture
def backup_server(sock_dir, tmp_path):
    """A server with a backup directory, an HTTP listener and two tokens."""
    tokens = tmp_path / "tokens"
    tokens.write_text("adm name=admin\nscoped name=one tenants=1\n", encoding="utf-8")
    tokens.chmod(0o600)
    port = _free_port()
    sock = sock_dir / "backup.sock"
    with running(
        sock,
        str(tmp_path / "t_{tenant}.anatid"),
        "--backup-dir",
        tmp_path / "backups",
        "--http",
        f"127.0.0.1:{port}",
        "--token-file",
        tokens,
    ) as server:
        server.port = port  # type: ignore[attr-defined]
        server.tmp_path = tmp_path  # type: ignore[attr-defined]
        server.socket_path = sock
        yield server


def _free_port() -> int:
    import socket as _s

    with _s.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_a_running_server_backs_its_own_tenant_up(backup_server, tmp_path):
    """The only arrangement that works: the process holding the file makes the copy.

    A second process cannot copy the file while the server runs, because DuckDB excludes every
    other opener, read-only ones included.  So the backup verb goes to the server, which drains
    the tenant's queued writes first so that the copy is a moment some client actually observed.
    """
    sock = backup_server.socket_path
    remember(sock, 1, "before the backup")
    done = cli("backup", "--socket", sock, "--tenant", 1, "--name", "t1.anatid")
    assert done.returncode == C.EXIT_OK, done.stderr
    copy = tmp_path / "backups" / "t1.anatid"
    assert copy.exists()
    remember(sock, 1, "after the backup")
    cli("stop", "--socket", sock)
    backup_server.finish()
    assert rows(copy) == 1, "the copy should hold what was committed when it was taken"
    assert rows(tmp_path / "t_1.anatid") == 2


@pytest.mark.parametrize("name", ["../escape.anatid", "sub/dir.anatid", ".hidden", ".."])
def test_a_backup_name_cannot_be_a_path(backup_server, name):
    """The client chooses a name; the server chose the directory.  Both, and the client wins,
    would be a client that can make the server write anywhere it can reach."""
    done = cli("backup", "--socket", backup_server.socket_path, "--tenant", 1, "--name", name)
    assert done.returncode == C.EXIT_FAILED
    assert "plain file name" in done.stderr, done.stderr


def test_a_tenant_scoped_token_cannot_make_the_server_write_a_file(backup_server):
    """Backup is administrative: it writes to the server's disk, not to a tenant's rows."""
    done = cli(
        "backup",
        "--http",
        f"127.0.0.1:{backup_server.port}",
        "--token",
        "scoped",
        "--tenant",
        1,
        "--name",
        "sneaky.anatid",
    )
    assert done.returncode == C.EXIT_FAILED
    assert "administrative verb" in done.stderr, done.stderr


def test_a_server_started_without_a_backup_directory_refuses_backups(sock_dir, tmp_path):
    sock = sock_dir / "nobackup.sock"
    with running(sock, str(tmp_path / "t_{tenant}.anatid")):
        done = cli("backup", "--socket", sock, "--tenant", 1, "--name", "x.anatid")
    assert done.returncode == C.EXIT_FAILED
    assert "--backup-dir" in done.stderr


def test_backup_offline_copies_the_file_when_no_server_holds_it(tmp_path):
    template = str(tmp_path / "t_{tenant}.anatid")
    with Anatid.open(tmp_path / "t_1.anatid", tenant=1, embedding_dim=DIM) as db:
        db.remember("offline")
    done = cli("backup", "--pool", template, "--tenant", 1, "--to", tmp_path / "copy.anatid")
    assert done.returncode == C.EXIT_OK, done.stderr
    assert rows(tmp_path / "copy.anatid") == 1
    again = cli("backup", "--pool", template, "--tenant", 1, "--to", tmp_path / "copy.anatid")
    assert again.returncode != C.EXIT_OK
    assert "exists" in (again.stderr + again.stdout)


def test_backup_of_a_single_shared_file_works_offline(tmp_path):
    path = tmp_path / "one.anatid"
    with Anatid.open(path, tenant=1, embedding_dim=DIM) as db:
        db.remember("shared file")
    done = cli("backup", "--db", path, "--tenant", 1, "--to", tmp_path / "one-copy.anatid")
    assert done.returncode == C.EXIT_OK, done.stderr
    assert rows(tmp_path / "one-copy.anatid") == 1


# ------------------------------------------------------- backup and an existing destination
#
# A backup that would replace an earlier backup is refused, and the refusal has to be the same
# refusal wherever it comes from.  It was not: ``backup --db`` raised the CLI's own
# ConfigProblem and exited 2 naming ``--overwrite``, while ``backup --pool`` and the server's
# backup verb fell through to ``DatabasePool.backup``, which raises FileExistsError naming its
# Python keyword ``overwrite=True``.  An operator cannot type ``overwrite=True``, and an
# OSError reaching ``main`` exits 1, so the recommended arrangement gave both the wrong
# sentence and the wrong exit code.  These tests pin all three paths to one answer.


def test_an_existing_backup_destination_is_refused_the_same_way_by_both_offline_paths(tmp_path):
    with Anatid.open(tmp_path / "t_1.anatid", tenant=1, embedding_dim=DIM) as db:
        db.remember("pool")
    with Anatid.open(tmp_path / "one.anatid", tenant=1, embedding_dim=DIM) as db:
        db.remember("shared")

    results = {}
    for label, storage in (
        ("pool", ("--pool", str(tmp_path / "t_{tenant}.anatid"))),
        ("db", ("--db", tmp_path / "one.anatid")),
    ):
        dest = tmp_path / f"{label}-copy.anatid"
        first = cli("backup", *storage, "--tenant", 1, "--to", dest)
        assert first.returncode == C.EXIT_OK, first.stderr
        before = dest.read_bytes()
        again = cli("backup", *storage, "--tenant", 1, "--to", dest)
        assert dest.read_bytes() == before, f"{label}: the refused backup overwrote the file"
        results[label] = (again.returncode, again.stderr)

    for label, (code, err) in results.items():
        # 2, not 1: nothing was written and the argument is what is wrong, which is exactly
        # what the exit-code table in docs/server.md promises for a 2.
        assert code == C.EXIT_CONFIG, f"{label} exited {code}: {err}"
        assert "--overwrite" in err, f"{label} did not name the flag: {err}"
        assert "overwrite=True" not in err, f"{label} named a Python keyword: {err}"
    assert results["pool"][0] == results["db"][0]


def test_overwrite_replaces_an_existing_backup_on_both_offline_paths(tmp_path):
    """The refusal has to be liftable, or it is just a broken command."""
    template = str(tmp_path / "t_{tenant}.anatid")
    with Anatid.open(tmp_path / "t_1.anatid", tenant=1, embedding_dim=DIM) as db:
        db.remember("one")
    dest = tmp_path / "copy.anatid"
    assert cli("backup", "--pool", template, "--tenant", 1, "--to", dest).returncode == 0
    assert rows(dest) == 1
    with Anatid.open(tmp_path / "t_1.anatid", tenant=1, embedding_dim=DIM) as db:
        db.remember("two")
    done = cli("backup", "--pool", template, "--tenant", 1, "--to", dest, "--overwrite")
    assert done.returncode == C.EXIT_OK, done.stderr
    assert rows(dest) == 2, "--overwrite should have replaced the copy with the newer one"


def test_a_running_server_refuses_to_overwrite_a_backup_and_says_which_flag_lifts_it(
    backup_server, tmp_path
):
    """The online path, where the message crosses the wire before an operator reads it."""
    sock = backup_server.socket_path
    remember(sock, 1, "first")
    assert cli("backup", "--socket", sock, "--tenant", 1, "--name", "d.anatid").returncode == 0
    copy = tmp_path / "backups" / "d.anatid"
    before = copy.read_bytes()

    remember(sock, 1, "second")
    again = cli("backup", "--socket", sock, "--tenant", 1, "--name", "d.anatid")
    assert again.returncode == C.EXIT_FAILED, again.stderr
    assert "--overwrite" in again.stderr, again.stderr
    assert "overwrite=True" not in again.stderr, again.stderr
    assert copy.read_bytes() == before, "the refused backup overwrote the earlier one"

    done = cli("backup", "--socket", sock, "--tenant", 1, "--name", "d.anatid", "--overwrite")
    assert done.returncode == C.EXIT_OK, done.stderr
    cli("stop", "--socket", sock)
    backup_server.finish()
    assert rows(copy) == 2


def test_an_existing_destination_is_a_400_over_http_not_a_500(backup_server, tmp_path):
    """A destination that already exists is a bad argument, like a bad name.

    docs/server.md maps 400 to "malformed request, bad arguments" and 500 to "an error inside a
    verb".  A FileExistsError escaping the verb landed in 500 and told a monitor the server had
    broken when the caller had simply asked twice.
    """
    import http.client

    port = backup_server.port
    remember(backup_server.socket_path, 1, "first")

    def post(name: str) -> int:
        body = protocol.Request(verb="backup", tenant=1, args={"name": name}).encode()[
            protocol.LENGTH_PREFIX_BYTES :
        ]
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        try:
            conn.request(
                "POST",
                "/rpc",
                body=body,
                headers={"content-type": "application/json", "authorization": "Bearer adm"},
            )
            return conn.getresponse().status
        finally:
            conn.close()

    assert post("http.anatid") == 200
    assert post("http.anatid") == 400
    # The bad-name refusal is already a 400; the two now agree.
    assert post("../escape.anatid") == 400


# ------------------------------------------------------- backup and the catalog name
#
# DuckDB names a database's catalog after the file's stem, and ``COPY FROM DATABASE`` has to
# name that catalog.  ``DatabasePool.backup`` quotes it with ``anatid.schema.quote_ident``,
# which allows only ``[A-Za-z_][A-Za-z0-9_]*``, so it raises ValueError for two pool templates
# an operator would reasonably write.  These tests pin the CLI's behaviour for both, and they
# keep passing when ``DatabasePool.backup`` is fixed: they check that the backup happened, not
# which code path made it.


@pytest.mark.parametrize(
    ("template", "filename"),
    [
        ("tenant-{tenant}.anatid", "tenant-1.anatid"),  # catalog "tenant-1": a hyphen
        ("{tenant}.anatid", "1.anatid"),  # catalog "1": starts with a digit
    ],
)
def test_backup_offline_handles_a_pool_whose_file_stem_is_not_an_identifier(
    tmp_path, template, filename
):
    with Anatid.open(tmp_path / filename, tenant=1, embedding_dim=DIM) as db:
        db.remember("awkwardly named file")
    done = cli(
        "backup",
        "--pool",
        str(tmp_path / template),
        "--tenant",
        1,
        "--to",
        tmp_path / "copy.anatid",
    )
    assert done.returncode == C.EXIT_OK, done.stderr
    assert rows(tmp_path / "copy.anatid") == 1


def test_backup_of_a_shared_file_whose_name_is_not_an_identifier(tmp_path):
    path = tmp_path / "anatid-memory.anatid"
    with Anatid.open(path, tenant=1, embedding_dim=DIM) as db:
        db.remember("hyphenated single file")
    done = cli("backup", "--db", path, "--tenant", 1, "--to", tmp_path / "copy.anatid")
    assert done.returncode == C.EXIT_OK, done.stderr
    assert rows(tmp_path / "copy.anatid") == 1


def test_a_running_server_backs_up_a_hyphenated_pool_template(sock_dir, tmp_path):
    """The online path, which is the one an operator uses, on the template the docs show."""
    sock = sock_dir / "hyphen.sock"
    with running(
        sock,
        str(tmp_path / "tenant-{tenant}.anatid"),
        "--backup-dir",
        tmp_path / "backups",
    ) as server:
        remember(sock, 1, "written before the backup")
        done = cli("backup", "--socket", sock, "--tenant", 1, "--name", "t1.anatid")
        assert done.returncode == C.EXIT_OK, done.stderr
        cli("stop", "--socket", sock)
        server.finish()
    assert rows(tmp_path / "backups" / "t1.anatid") == 1


def test_a_catalog_name_is_quoted_rather_than_validated(tmp_path):
    """``_quote_catalog`` accepts what DuckDB accepts and still cannot be escaped out of."""
    assert C._quote_catalog("tenant-1") == '"tenant-1"'
    assert C._quote_catalog("1") == '"1"'
    assert C._quote_catalog('a"b') == '"a""b"'
    for bad in ("", "a\x00b"):
        with pytest.raises(C.ConfigProblem):
            C._quote_catalog(bad)


# --------------------------------------------------------------------------- restore


def test_restore_puts_a_backup_back_and_keeps_the_previous_file(tmp_path):
    template = str(tmp_path / "t_{tenant}.anatid")
    target = tmp_path / "t_1.anatid"
    with Anatid.open(target, tenant=1, embedding_dim=DIM) as db:
        db.remember("in the backup")
    backup = tmp_path / "backup.anatid"
    assert cli("backup", "--pool", template, "--tenant", 1, "--to", backup).returncode == 0
    with Anatid.open(target, tenant=1, embedding_dim=DIM) as db:
        db.remember("written after the backup")
    assert rows(target) == 2

    done = cli("restore", "--pool", template, "--tenant", 1, backup)
    assert done.returncode == C.EXIT_OK, done.stderr
    assert rows(target) == 1, "the restore did not replace the file"
    kept = [p for p in tmp_path.iterdir() if ".replaced-" in p.name]
    assert kept, "the previous file must be kept aside, not deleted"
    assert "the previous file is at" in done.stdout


def test_restore_moves_the_write_ahead_log_aside_with_its_file(tmp_path):
    """A restored file beside the previous file's WAL is a database DuckDB will try to finish."""
    template = str(tmp_path / "t_{tenant}.anatid")
    target = tmp_path / "t_1.anatid"
    with Anatid.open(target, tenant=1, embedding_dim=DIM) as db:
        db.remember("original")
    backup = tmp_path / "backup.anatid"
    assert cli("backup", "--pool", template, "--tenant", 1, "--to", backup).returncode == 0
    stale = Path(f"{target}.wal")
    stale.write_bytes(b"not a real write-ahead log")

    done = cli("restore", "--pool", template, "--tenant", 1, backup)
    assert done.returncode == C.EXIT_OK, done.stderr
    assert not stale.exists(), "the old write-ahead log was left beside the restored file"
    assert any(".wal.replaced-" in p.name for p in tmp_path.iterdir())


def test_restore_refuses_a_file_that_is_not_an_anatid_database(tmp_path):
    not_a_db = tmp_path / "notes.txt"
    not_a_db.write_text("hello", encoding="utf-8")
    done = cli("restore", "--db", tmp_path / "t.anatid", "--tenant", 1, not_a_db)
    assert done.returncode == C.EXIT_CONFIG
    assert "DuckDB database" in done.stderr


def test_restore_refuses_a_duckdb_file_with_no_anatid_schema_in_it(tmp_path):
    import duckdb

    stranger = tmp_path / "stranger.duckdb"
    con = duckdb.connect(str(stranger))
    con.execute("CREATE TABLE t (x INTEGER)")
    con.close()
    done = cli("restore", "--db", tmp_path / "t.anatid", "--tenant", 1, stranger)
    assert done.returncode == C.EXIT_CONFIG
    assert "no anatid schema" in done.stderr


def test_restore_refuses_a_backup_from_a_newer_anatid(tmp_path):
    """A file from a future schema is not something this build can be trusted to open."""
    source = tmp_path / "future.anatid"
    with Anatid.open(source, tenant=1, embedding_dim=DIM) as db:
        db.remember("from the future")
        db.execute("UPDATE anatid_meta SET schema_version = 9999")
    done = cli("restore", "--db", tmp_path / "t.anatid", "--tenant", 1, source)
    assert done.returncode == C.EXIT_CONFIG
    assert "9999" in done.stderr


def test_restore_refuses_while_a_server_holds_the_file(sock_dir, tmp_path):
    """Restore is offline, and this is why: the file cannot even be opened while a server runs."""
    template = str(tmp_path / "t_{tenant}.anatid")
    sock = sock_dir / "held.sock"
    backup = tmp_path / "backup.anatid"
    with running(sock, template) as server:
        remember(sock, 1, "held")
        assert cli("backup", "--socket", sock, "--tenant", 1).returncode != C.EXIT_OK
        # Take the backup the honest way, by stopping nothing: copy an unrelated file in as the
        # source, since the point here is the refusal, not the contents.
        with Anatid.open(tmp_path / "source.anatid", tenant=1, embedding_dim=DIM) as db:
            db.remember("a source to restore from")
        shutil.copy2(tmp_path / "source.anatid", backup)

        done = cli("restore", "--pool", template, "--tenant", 1, "--socket", sock, backup)
        assert done.returncode == C.EXIT_CONFIG
        assert "still running" in done.stderr or "holds" in done.stderr

        without_the_hint = cli("restore", "--pool", template, "--tenant", 1, backup)
        assert without_the_hint.returncode == C.EXIT_CONFIG
        assert "exclusive use" in without_the_hint.stderr, without_the_hint.stderr
        cli("stop", "--socket", sock)
        server.finish()


# --------------------------------------------------------------------------- doctor


def test_doctor_reports_a_clean_file_and_exits_zero(tmp_path):
    path = tmp_path / "clean.anatid"
    with Anatid.open(path, tenant=1, embedding_dim=DIM) as db:
        db.remember("nothing wrong here", entities=["Ada"])
    done = cli("doctor", "--db", path, "--tenant", 1)
    assert done.returncode == C.EXIT_OK, done.stderr
    assert "clean: nothing to report" in done.stdout


def test_doctor_exits_three_when_it_finds_an_error(tmp_path):
    """A monitor reads the exit code, so a fault has to change it."""
    path = tmp_path / "broken.anatid"
    with Anatid.open(path, tenant=1, embedding_dim=DIM) as db:
        db.remember("one")
        db.execute("INSERT INTO memories SELECT * FROM memories LIMIT 1")
    done = cli("doctor", "--db", path, "--tenant", 1)
    assert done.returncode == C.EXIT_UNHEALTHY, done.stdout + done.stderr
    assert "duplicate_memory_ids" in done.stdout
    assert "ERROR" in done.stdout


def test_doctor_runs_over_the_wire_against_a_running_server(sock_dir, tmp_path):
    sock = sock_dir / "doc.sock"
    with running(sock, str(tmp_path / "t_{tenant}.anatid")):
        remember(sock, 1, "a memory to check")
        done = cli("doctor", "--socket", sock, "--tenant", 1)
    assert done.returncode == C.EXIT_OK, done.stderr
    assert "schema v" in done.stdout
    assert "memories=1" in done.stdout


def test_doctor_needs_somewhere_to_look(tmp_path):
    done = cli("doctor", "--tenant", 1)
    assert done.returncode == C.EXIT_CONFIG
    assert "--socket" in done.stderr


# --------------------------------------------------------------------------- exit codes


class _StubClient:
    """A client that answers with whatever the test hands it, for the exit-code contract."""

    def __init__(self, answers):
        self.answers = answers
        self.address = "unix:/stub"

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def call(self, verb, **_kw):
        return self.answers[verb]


def test_status_exits_three_when_the_server_is_healthy_but_not_ready(monkeypatch, capsys):
    """Healthy and not ready is the state a load balancer must act on and a supervisor must not.

    A busy server answers true on health and false on readiness; restarting it would throw away
    the work it is busy with, and sending it traffic would queue behind a full queue.
    """
    from anatid.server.server import Health, Readiness

    answers = {
        "health": Health(
            ok=True,
            status="serving",
            pid=1,
            uptime_s=1.0,
            anatid_version="0.2.0",
            protocol_version=1,
        ),
        "ready": Readiness(
            ready=False,
            accepting=True,
            open_files=1,
            tenants=(1,),
            migrations_done=True,
            detail="a tenant queue is at or above the high-water mark of 8 (9 queued)",
        ),
        "queue_stats": None,
    }
    monkeypatch.setattr(C, "_client_from", lambda args: _StubClient(answers))
    args = C.build_parser().parse_args(["status", "--socket", "/stub"])
    assert C.cmd_status(args) == C.EXIT_UNHEALTHY
    out = capsys.readouterr().out
    assert "ready          no" in out
    assert "high-water mark" in out


def test_status_survives_a_server_that_cannot_encode_its_queue_stats(monkeypatch, capsys):
    """An older server answers ProtocolError for ``queue_stats``; status still reports the rest."""
    from anatid.server.server import Health, Readiness

    class Refusing(_StubClient):
        def call(self, verb, **kw):
            if verb == "queue_stats":
                raise protocol.ProtocolError(
                    "QueueStats is a dataclass this protocol does not carry"
                )
            return super().call(verb, **kw)

    answers = {
        "health": Health(
            ok=True,
            status="serving",
            pid=1,
            uptime_s=1.0,
            anatid_version="0.2.0",
            protocol_version=1,
        ),
        "ready": Readiness(
            ready=True, accepting=True, open_files=1, tenants=(1,), migrations_done=True
        ),
    }
    monkeypatch.setattr(C, "_client_from", lambda args: Refusing(answers))
    args = C.build_parser().parse_args(["status", "--socket", "/stub"])
    assert C.cmd_status(args) == C.EXIT_OK
    out = capsys.readouterr().out
    assert "ready          yes" in out
    assert "queue totals" not in out


def test_main_turns_a_configuration_problem_into_exit_two_and_one_sentence(capsys):
    assert C.main(["start", "--check", "--socket", "/tmp/x.sock"]) == C.EXIT_CONFIG
    captured = capsys.readouterr()
    assert captured.err.startswith("anatid-server: the configuration is not usable")
    assert "Traceback" not in captured.err


def test_an_unknown_subcommand_is_a_usage_error(sock_dir):
    done = cli("teleport")
    assert done.returncode == 2
    assert "invalid choice" in done.stderr


# --------------------------------------------------------------------------- packaging


def test_the_console_script_is_declared():
    """``anatid-server`` has to be an entry point, or none of the above is reachable as typed."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'anatid-server = "anatid.server.cli:main"' in text


def test_the_module_entry_point_runs():
    done = cli("--version")
    assert done.returncode == C.EXIT_OK, done.stderr
    assert done.stdout.startswith("anatid-server ")
