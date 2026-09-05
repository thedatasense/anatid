"""``anatid-mcp`` against a running ``anatid-server``: the socket backend.

Until now ``anatid-mcp`` always opened ``ANATID_DB`` itself.  DuckDB gives one process
exclusive use of a database file and refuses a second opener even for reading, so Claude Desktop
and Claude Code pointed at one file could not both be running, and the second one to start died
on a raw ``IOException`` with nothing to say about what to do instead.

:func:`anatid.integrations.mcp.backend.open_backend` is the fix, and this file holds its three
claims:

* two MCP servers on one socket share one memory: a write through one is read through the other;
* every MCP tool answers the same over the socket as it does embedded, which is the drop-in
  claim of ``AnatidClient`` checked at the tool layer rather than the verb layer;
* the direct-file path on a file another process holds fails with a message that names the two
  commands to run instead, and those two commands work.

The ``AnatidServer`` for most tests runs on its own event loop in a thread, the way
``tests/test_server_client.py`` does it.  The held-file test needs a second PROCESS, because the
lock is per process, so it starts a real ``anatid-server`` as a subprocess.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
import socket

pytest.importorskip("mcp.client", reason="the MCP server needs 'mcp>=2.1'")

from mcp import StdioServerParameters
from mcp.client import Client

from anatid import Anatid
from anatid.integrations.mcp import backend as B
from anatid.integrations.mcp.backend import (
    BackendConfigError,
    DatabaseLocked,
    ServerHandle,
    open_backend,
    resolve_token,
    sharing_recipe,
)
from anatid.integrations.mcp.server import ServerConfig, build_server, main
from anatid.server import server as S
from anatid.server.client import AnatidClient
from anatid.server.server import connect_unix

pytestmark = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="the server profile's Unix domain socket transport is not available on this platform; use the HTTP transport",
)


DIM = 8
TENANT = 1
REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_SRC = REPO_ROOT / "src"

#: ``sun_path`` is 104 bytes on macOS and pytest's ``tmp_path`` is nested deeply enough to
#: overflow it, so socket paths come from :func:`sock_dir` rather than from ``tmp_path``.
SOCKADDR_UN_MAX = 100


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def sock_dir():
    """A directory short enough to hold a Unix socket path."""
    base = Path(tempfile.mkdtemp(prefix="anatid-m-"))
    if len(str(base)) + len("/xxxxxxxx.sock") > SOCKADDR_UN_MAX:
        shutil.rmtree(base, ignore_errors=True)
        base = Path(tempfile.mkdtemp(prefix="anatid-m-", dir="/tmp"))
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def clean_env(monkeypatch):
    """No ``ANATID_*`` variable from the developer's shell reaches :func:`main`."""
    for name in list(os.environ):
        if name.startswith("ANATID_"):
            monkeypatch.delenv(name, raising=False)


@contextlib.contextmanager
def running_server(**kw):
    """An :class:`AnatidServer` on its own event loop in a background thread."""
    loop = asyncio.new_event_loop()
    holder: dict[str, object] = {}
    ready = threading.Event()

    def run() -> None:
        asyncio.set_event_loop(loop)

        async def go() -> None:
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
            loop.run_until_complete(go())
        except RuntimeError:
            pass  # the loop was stopped below
        finally:
            loop.close()

    thread = threading.Thread(target=run, name="anatid-mcp-test-server", daemon=True)
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
def shared(tmp_path, sock_dir):
    """One file, held by an ``AnatidServer`` on a Unix socket: ``(socket_path, file_handle)``.

    The handle is the server's own, which is the only process that can read the file while the
    server holds it, so a test uses it to check what really landed on disk.
    """
    sock = sock_dir / "anatid.sock"
    with Anatid.open(tmp_path / "shared.anatid", tenant=TENANT, embedding_dim=DIM) as handle:
        config = S.ServerConfig(
            socket_path=sock, tenants=(TENANT,), workers=1, batch_max=4, shutdown_timeout=10.0
        )
        with running_server(database=handle, config=config):
            yield sock, handle


def socket_config(sock: Path, **kw) -> ServerConfig:
    """An ``anatid-mcp`` configuration pointed at ``sock``, with the environment ignored."""
    kw.setdefault("tenant", TENANT)
    return ServerConfig(socket=str(sock), env={}, **kw)


# --------------------------------------------------------------------------- MCP glue


def drive(server, body):
    async def go():
        async with Client(server) as client:
            return await body(client)

    return asyncio.run(go())


def call(server, name: str, arguments: dict | None = None):
    async def body(client):
        return await client.call_tool(name, arguments or {})

    return drive(server, body)


def text_of(result) -> str:
    return " ".join(getattr(c, "text", "") for c in result.content)


def ok(result):
    assert result.is_error is False, text_of(result)
    assert result.structured_content is not None, "tool returned no structured content"
    return result.structured_content


def tool_names(server) -> set[str]:
    return {t.name for t in drive(server, lambda c: c.list_tools()).tools}


# --------------------------------------------------------------------------- open_backend


def test_a_named_file_opens_an_embedded_handle(tmp_path):
    cfg = ServerConfig(db=str(tmp_path / "e.anatid"), tenant=TENANT, embedding_dim=DIM, env={})
    assert cfg.remote is False
    assert cfg.db_explicit is True
    with open_backend(cfg) as db:
        assert isinstance(db, Anatid)
        assert not isinstance(db, AnatidClient)
        assert db.path == str(tmp_path / "e.anatid")
        assert db.namespace.tenant_id == TENANT


def test_a_named_socket_connects_a_client(shared):
    sock, _ = shared
    cfg = socket_config(sock)
    assert cfg.remote is True
    assert cfg.db_explicit is False, "the default path is not a NAMED file"
    with open_backend(cfg) as db:
        assert isinstance(db, ServerHandle)
        assert isinstance(db, AnatidClient)
        assert db.tenant_id == TENANT
        assert db.server_info is not None, "connect probes the server"
        # The three handle attributes the tool layer reads, filled in.
        assert db.read_only is False
        assert db.path == f"unix:{sock}"
        assert db.expand_path == "sql"
        assert "ServerHandle" in repr(db)


def test_the_socket_comes_from_the_environment_too(shared):
    """A client config block can only set environment variables, so the contract is there."""
    sock, _ = shared
    cfg = ServerConfig(env={"ANATID_SOCKET": str(sock), "ANATID_TENANT": str(TENANT)})
    assert cfg.socket == str(sock)
    assert cfg.remote is True
    assert cfg.target == str(sock)
    with open_backend(cfg) as db:
        assert isinstance(db, ServerHandle)
    # An empty value is unset, so a block that clears the variable falls back to the file.
    assert ServerConfig(env={"ANATID_SOCKET": ""}).socket is None


# --------------------------------------------------------------------------- two clients


def test_two_mcp_servers_on_one_socket_share_one_memory(shared):
    """Remember through one MCP server, recall through the other.  The point of the feature."""
    sock, handle = shared
    with open_backend(socket_config(sock)) as a, open_backend(socket_config(sock)) as b:
        server_a = build_server(a, socket_config(sock))
        server_b = build_server(b, socket_config(sock))

        written = ok(
            call(
                server_a,
                "remember",
                {"content": "Ada leads Kestrel", "entities": ["Ada", "Kestrel"], "writer": "a"},
            )
        )
        memory_id = written["memory"]["memory_id"]
        assert isinstance(memory_id, str), "an id crosses the wire as a decimal string"
        assert written["memory"]["tenant_id"] == str(TENANT)

        found = ok(call(server_b, "recall", {"query": "who leads Kestrel", "k": 5}))
        assert "text" in found["arms"]
        assert [h["memory_id"] for h in found["hits"]] == [memory_id]

        ctx = ok(call(server_b, "context", {"entity": "Ada"}))
        assert [m["content"] for m in ctx["memories"]] == ["Ada leads Kestrel"]

        got = ok(call(server_b, "get", {"memory_id": memory_id}))
        assert got["memory"]["writer"] == "a"
        assert {e["name"] for e in got["about"]} == {"Ada", "Kestrel"}

        # Both report the same database, and say it is reached through the socket.
        stats_a, stats_b = ok(call(server_a, "stats")), ok(call(server_b, "stats"))
        assert stats_a["counts"] == stats_b["counts"]
        assert stats_a["counts"]["memories"] == 1
        assert stats_a["path"] == f"unix:{sock}"
        assert stats_a["tenant_id"] == str(TENANT)
        assert stats_a["sql_tool"]["enabled"] is False

    # And the row is in the file the server holds, not just in the replies.
    assert handle.get(int(memory_id)) is not None


def test_an_edge_related_through_one_client_widens_recall_in_the_other(shared):
    """Three facts stored by entity name alone reach one hop; two RELATES_TO edges reach all
    three.  Written through one client and read through the other, so the graph is shared too."""
    sock, _ = shared
    with open_backend(socket_config(sock)) as a, open_backend(socket_config(sock)) as b:
        writer, reader = build_server(a, socket_config(sock)), build_server(b, socket_config(sock))
        for content, entities in [
            ("Ada leads Kestrel", ["Ada", "Kestrel"]),
            ("Kestrel owns the ingest service", ["Kestrel", "ingest service"]),
            ("Bo maintains the ingest service", ["Bo", "ingest service"]),
        ]:
            ok(call(writer, "remember", {"content": content, "entities": entities}))

        def reached() -> set[str]:
            hits = ok(call(reader, "recall", {"seed_entity": "Ada", "hops": 2, "k": 10}))
            assert hits["arms"] == ["graph"]
            return {h["content"] for h in hits["hits"]}

        assert reached() == {"Ada leads Kestrel"}

        ok(call(writer, "relate", {"src": "Ada", "dst": "Kestrel", "rel_kind": "leads"}))
        ok(call(writer, "relate", {"src": "Kestrel", "dst": "ingest service", "rel_kind": "owns"}))
        assert reached() == {
            "Ada leads Kestrel",
            "Kestrel owns the ingest service",
            "Bo maintains the ingest service",
        }


def test_read_only_over_a_socket_registers_no_write_tools(shared):
    """``--read-only`` is this MCP server's own restriction; the server is not asked to enforce it."""
    sock, handle = shared
    handle.remember("written by the server's owner", entities=["thing"])
    with open_backend(socket_config(sock, read_only=True)) as db:
        assert db.read_only is True
        server = build_server(db, socket_config(sock, read_only=True))
        names = tool_names(server)
        assert names == {"recall", "context", "get", "provenance", "stats"}
        ctx = ok(call(server, "context", {"entity": "thing"}))
        assert ctx["memories"][0]["content"] == "written by the server's owner"
        assert ok(call(server, "stats"))["read_only"] is True


# --------------------------------------------------------------------------- the drop-in claim


ID_RE = re.compile(r"^\d{16,}$")
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
SET_LIKE = {"about", "sources", "writers", "arms"}


def normalise(value, ids: dict[str, str], key: str | None = None):
    """Ids by order of first appearance, timestamps as one token, set-like lists sorted."""
    if isinstance(value, dict):
        return {k: normalise(v, ids, k) for k, v in value.items()}
    if isinstance(value, list):
        out = [normalise(v, ids) for v in value]
        if key in SET_LIKE and all(isinstance(v, str) for v in out):
            return sorted(out)
        return out
    if isinstance(value, str):
        if ID_RE.match(value):
            return ids.setdefault(value, f"id#{len(ids)}")
        if TS_RE.match(value):
            return "<timestamp>"
    return value


def tool_workload(server) -> dict:
    """Every MCP tool both backends register, in one fixed order, with the answers keyed by step.

    ``sql`` is left out because a socket backend refuses it (tested on its own).  ``correct`` is
    in: since 0.4.0 the server profile carries the verb, so both backends register the tool and
    the diff test checks that the two tool lists are equal.
    """
    out: dict = {}
    first = ok(
        call(
            server,
            "remember",
            {
                "content": "Ada leads Kestrel",
                "entities": ["Ada", "Kestrel"],
                "kind": "fact",
                "writer": "standup",
                "episode": "standup 2026-09-01: Ada said she now leads Kestrel",
                "episode_source": "standup",
                "confidence": 0.9,
                "valid_from": "2026-09-01T09:00:00Z",
            },
        )
    )
    second = ok(
        call(
            server,
            "remember",
            {
                "content": "Kestrel owns the ingest service",
                "entities": ["Kestrel", "ingest service"],
            },
        )
    )
    third = ok(
        call(
            server,
            "remember",
            {"content": "Bo maintains the ingest service", "entities": ["Bo", "ingest service"]},
        )
    )
    out["remember"] = [first, second, third]
    out["relate"] = [
        ok(call(server, "relate", {"src": "Ada", "dst": "Kestrel", "rel_kind": "leads"})),
        ok(
            call(
                server,
                "relate",
                {"src": "Kestrel", "dst": "ingest service", "rel_kind": "owns", "confidence": 0.8},
            )
        ),
    ]
    out["recall_text"] = ok(
        call(server, "recall", {"query": "who maintains the ingest service", "k": 5})
    )
    out["recall_graph"] = ok(call(server, "recall", {"seed_entity": "Ada", "hops": 2, "k": 10}))
    out["recall_both"] = ok(
        call(server, "recall", {"query": "ingest", "seed_entity": "Kestrel", "hops": 1, "k": 10})
    )
    out["context"] = [
        ok(call(server, "context", {"entity": "ingest service"})),
        ok(call(server, "context", {"entity": "Ada", "hops": 2, "limit": 5})),
    ]
    out["unrelate"] = ok(
        call(server, "unrelate", {"src": "Kestrel", "dst": "ingest service", "rel_kind": "owns"})
    )
    out["recall_graph_after_unrelate"] = ok(
        call(server, "recall", {"seed_entity": "Ada", "hops": 2, "k": 10})
    )
    third_id = third["memory"]["memory_id"]
    replaced = ok(
        call(
            server,
            "supersede",
            {
                "old_id": third_id,
                "content": "Cy maintains the ingest service",
                "entities": ["Cy", "ingest service"],
                "writer": "oncall",
                "episode": "oncall handover: Cy took the ingest service from Bo",
            },
        )
    )
    out["supersede"] = replaced
    out["correct"] = ok(
        call(
            server,
            "correct",
            {
                "old_id": replaced["memory"]["memory_id"],
                "content": "Dee maintains the ingest service",
                "entities": ["Dee", "ingest service"],
                "add_relations": [{"src": "Dee", "dst": "ingest service", "rel_kind": "maintains"}],
                "remove_relations": [{"src": "Cy", "dst": "ingest service"}],
            },
        )
    )
    out["recall_graph_after_correct"] = ok(
        call(server, "recall", {"seed_entity": "Dee", "hops": 2, "k": 10})
    )
    out["get_old"] = ok(call(server, "get", {"memory_id": third_id}))
    out["get_new"] = ok(call(server, "get", {"memory_id": replaced["memory"]["memory_id"]}))
    out["provenance"] = ok(
        call(server, "provenance", {"memory_id": replaced["memory"]["memory_id"]})
    )
    first_id = first["memory"]["memory_id"]
    out["reinforce"] = ok(
        call(server, "reinforce", {"memory_id": first_id, "amount": 2, "confidence": 0.95})
    )
    second_id = second["memory"]["memory_id"]
    out["forget_soft"] = ok(call(server, "forget", {"memory_id": second_id, "reason": "moved"}))
    out["context_after"] = ok(call(server, "context", {"entity": "Kestrel", "hops": 1}))
    out["prune_dry"] = ok(call(server, "prune", {"max_access_count": 0}))
    out["rebuild"] = ok(call(server, "rebuild_fts_index"))
    out["stats"] = ok(call(server, "stats"))
    out["forget_hard"] = ok(call(server, "forget", {"memory_id": second_id, "hard": True}))
    out["get_gone"] = ok(call(server, "get", {"memory_id": second_id}))
    # Two anticipated failures: the message the model reads has to be the same too.
    out["bad_timestamp"] = text_of(call(server, "recall", {"query": "x", "as_of": "yesterday"}))
    out["unknown_id"] = text_of(call(server, "reinforce", {"memory_id": "12345"}))
    return out


def test_every_tool_answers_the_same_embedded_and_over_the_socket(shared, tmp_path):
    """One workload through every tool, against an embedded handle and against the socket.

    Two files, because the server's file cannot be opened by anyone else.  Ids and timestamps
    differ between the runs and are normalised away; everything else is compared as the client
    would see it.  ``stats.path`` is the one field that says which backend answered, so it is
    compared on its own.
    """
    sock, _ = shared
    embedded_path = tmp_path / "embedded.anatid"
    with Anatid.open(embedded_path, tenant=TENANT, embedding_dim=DIM) as embedded:
        cfg = ServerConfig(db=str(embedded_path), tenant=TENANT, env={})
        embedded_server = build_server(embedded, cfg)
        local_tools = tool_names(embedded_server)
        local = tool_workload(embedded_server)
    with open_backend(socket_config(sock)) as client:
        remote_server = build_server(client, socket_config(sock))
        remote_tools = tool_names(remote_server)
        remote = tool_workload(remote_server)

    # The tool lists: the socket registers every tool the file does.  A tool missing on either
    # side is a regression.
    assert remote_tools == local_tools, remote_tools ^ local_tools
    assert {
        "remember",
        "relate",
        "unrelate",
        "supersede",
        "correct",
        "recall",
        "stats",
    } <= remote_tools

    assert local["stats"].pop("path") == str(embedded_path)
    assert remote["stats"].pop("path") == f"unix:{sock}"
    assert local["stats"]["expand_path"] == remote["stats"]["expand_path"] == "sql"

    local_n = normalise(local, {})
    remote_n = normalise(remote, {})
    for step in local:
        assert remote_n[step] == local_n[step], step
    assert remote_n == local_n

    # The workload did what it says, and the answers carry substance.
    assert len(local["recall_graph"]["hits"]) == 3
    assert local["unrelate"]["edges_closed"] == 1
    # Ada to Kestrel still stands, so the memory filed under Kestrel is still one hop away; the
    # one filed under the ingest service alone is what the closed edge took out of reach.
    assert {h["content"] for h in local["recall_graph_after_unrelate"]["hits"]} == {
        "Ada leads Kestrel",
        "Kestrel owns the ingest service",
    }
    assert local["provenance"]["depth"] == 1
    assert local["get_gone"]["memory"] is None
    assert "ISO-8601" in local["bad_timestamp"]
    assert "12345" in local["unknown_id"]


# --------------------------------------------------------------------------- refusals


def test_a_file_and_a_socket_together_are_refused(shared, tmp_path, capsys, clean_env):
    """Two backends named is a misconfiguration, and it is refused before anything is opened."""
    sock, _ = shared
    never = tmp_path / "never-created.anatid"
    cfg = ServerConfig(db=str(never), socket=str(sock), tenant=TENANT, env={})
    with pytest.raises(BackendConfigError) as info:
        open_backend(cfg)
    message = str(info.value)
    assert str(never) in message and str(sock) in message
    assert "anatid-server start" in message, "the refusal says how to share the file instead"
    assert not never.exists()

    # Through the environment as well: a config block with both variables set.
    with pytest.raises(BackendConfigError):
        open_backend(ServerConfig(env={"ANATID_DB": str(never), "ANATID_SOCKET": str(sock)}))

    # And from the command line, as the operator sees it.
    code = main(["--db", str(never), "--socket", str(sock)])
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("anatid-mcp: ")
    assert "two different backends" in err
    assert not never.exists()


def test_the_sql_escape_hatch_is_refused_over_a_socket(shared):
    """The escape hatch runs on the file's connection, and the file is in the server process."""
    sock, _ = shared
    with pytest.raises(BackendConfigError, match="enable-sql"):
        open_backend(socket_config(sock, sql_tool=True))
    # build_server checks as well, for a program that connected its own client.
    with open_backend(socket_config(sock)) as client:
        with pytest.raises(BackendConfigError, match="embedded"):
            build_server(client, socket_config(sock, sql_tool=True))
        assert "sql" not in tool_names(build_server(client, socket_config(sock)))


def test_an_http_url_needs_a_token(tmp_path):
    cfg = ServerConfig(http_url="http://127.0.0.1:1/", tenant=TENANT, env={})
    with pytest.raises(BackendConfigError, match="bearer token"):
        open_backend(cfg)
    # A token file in anatid-server's own format: comments, blank lines, fields after the token.
    token_file = tmp_path / "tokens"
    token_file.write_text("# tokens\n\n7f3c9e01a4b2  name=claude-desktop  tenants=1\n")
    cfg = ServerConfig(
        http_url="http://127.0.0.1:1/", token_file=str(token_file), tenant=TENANT, env={}
    )
    assert resolve_token(cfg) == "7f3c9e01a4b2"
    assert resolve_token(ServerConfig(token="abc", env={})) == "abc"
    with pytest.raises(BackendConfigError, match="cannot read the token file"):
        resolve_token(ServerConfig(token_file=str(tmp_path / "missing"), env={}))
    (tmp_path / "empty").write_text("# nothing here\n")
    with pytest.raises(BackendConfigError, match="holds no token"):
        resolve_token(ServerConfig(token_file=str(tmp_path / "empty"), env={}))
    # Both a socket and a URL is two servers.
    with pytest.raises(BackendConfigError, match="two servers"):
        open_backend(ServerConfig(socket="/tmp/x.sock", http_url="http://h/", env={}))


def test_a_socket_nobody_listens_on_is_one_readable_error(sock_dir, capsys, clean_env):
    """The client's probe fails at connect time, and main reports it rather than tracebacking."""
    absent = sock_dir / "absent.sock"
    code = main(["--socket", str(absent), "--tenant", str(TENANT)])
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith(f"anatid-mcp: cannot open {str(absent)!r}")
    assert "no socket at" in err
    assert f"anatid-server start --socket {absent}" in err, "the error says what to start"
    assert f"--tenant {TENANT}" in err


# --------------------------------------------------------------------------- the held file


class ServerProcess:
    """A real ``anatid-server start`` in a subprocess: the other PROCESS the lock test needs."""

    def __init__(self, sock: Path, db: Path, tenant: int) -> None:
        env = dict(os.environ)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{REPO_SRC}{os.pathsep}{existing}" if existing else str(REPO_SRC)
        self.sock = sock
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "anatid.server",
                "start",
                "--socket",
                str(sock),
                "--db",
                str(db),
                "--tenant",
                str(tenant),
                "--embedding-dim",
                str(DIM),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

    def wait_ready(self, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                _, err = self.proc.communicate(timeout=10)
                raise AssertionError(f"anatid-server exited before it listened:\n{err[-3000:]}")
            if self.sock.exists():
                with contextlib.suppress(OSError):
                    connect_unix(self.sock, timeout=2.0).close()
                    return
            time.sleep(0.02)
        raise AssertionError(f"{self.sock} never appeared")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=30)
        if self.proc.poll() is None:  # pragma: no cover - only if a drain wedges
            self.proc.kill()
        self.proc.communicate(timeout=30)


def test_a_file_another_process_holds_is_refused_with_the_recipe(
    sock_dir, tmp_path, capsys, clean_env
):
    """The G4 reproduction, and its exit.

    A real ``anatid-server`` holds the file.  ``anatid-mcp --db`` on that file used to die on a
    raw ``IOException``; now it exits 2 with the lock rule in one paragraph and the two commands
    to run instead.  Then the recipe is followed: the server the message describes is the one
    already running, and an ``anatid-mcp --socket`` against it writes and reads the same file.
    """
    db_path = tmp_path / "held.anatid"
    sock = sock_dir / "held.sock"
    server = ServerProcess(sock, db_path, TENANT)
    try:
        server.wait_ready()

        # The direct-file path, as `anatid-mcp --db` runs it.
        code = main(["--db", str(db_path), "--tenant", str(TENANT)])
        assert code == 2
        err = capsys.readouterr().err
        assert err.startswith(f"anatid-mcp: another process holds {db_path}")
        assert "exclusive use of a database file" in err
        assert "even for reading" in err
        assert "anatid-server start --socket" in err and f"--db {db_path}" in err
        assert f"--tenant {TENANT}" in err
        assert "anatid-mcp --socket" in err
        assert "ANATID_SOCKET=" in err
        assert "Could not set lock on file" in err, "DuckDB's own message is kept"
        assert "Traceback" not in err

        # The same refusal as an exception, for a program calling open_backend itself.
        with pytest.raises(DatabaseLocked) as info:
            open_backend(ServerConfig(db=str(db_path), tenant=TENANT, env={}))
        assert info.value.path == str(db_path)
        assert info.value.tenant == TENANT
        start, connect = sharing_recipe(
            str(db_path), tenant=TENANT, socket_path=info.value.socket_path
        )
        assert start in str(info.value) and connect in str(info.value)

        # The recipe works: the socket the running server listens on, in place of the file.
        with open_backend(ServerConfig(socket=str(sock), tenant=TENANT, env={})) as client:
            mcp = build_server(client, ServerConfig(socket=str(sock), tenant=TENANT, env={}))
            written = ok(call(mcp, "remember", {"content": "shared through the socket"}))
            found = ok(call(mcp, "recall", {"query": "shared socket", "k": 3}))
            assert [h["memory_id"] for h in found["hits"]] == [written["memory"]["memory_id"]]
    finally:
        server.stop()

    # The server released the file on the way out, and the write is in it.
    with Anatid.open(db_path, tenant=TENANT, embedding_dim=DIM, read_only=True) as check:
        assert check.stats()["memories"] == 1


def test_a_lock_error_that_is_not_a_lock_is_not_rewritten(tmp_path):
    """Only the lock gets the recipe.  Any other DuckDB refusal keeps DuckDB's own words."""
    import duckdb

    missing = tmp_path / "does-not-exist" / "x.anatid"
    with pytest.raises(duckdb.Error) as info:
        open_backend(ServerConfig(db=str(missing), tenant=TENANT, read_only=True, env={}))
    assert not isinstance(info.value, DatabaseLocked)
    assert B.LOCK_MARKER not in str(info.value)


# --------------------------------------------------------------------------- over stdio


def test_the_stdio_server_takes_its_socket_from_the_environment(shared):
    """Launch ``anatid-mcp`` the way a client config block does, with ``ANATID_SOCKET`` set.

    The subprocess connects to the server running in this process's thread, so this is a
    second process on the same memory, which is the thing a file could not give.
    """
    sock, handle = shared
    env = dict(os.environ)
    for name in list(env):
        if name.startswith("ANATID_"):
            del env[name]
    env.update(
        {
            "ANATID_SOCKET": str(sock),
            "ANATID_TENANT": str(TENANT),
            "PYTHONPATH": str(REPO_SRC),
        }
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "anatid.integrations.mcp.server"],
        env=env,
        cwd=str(sock.parent),
    )

    async def go():
        async with Client(params) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            written = await client.call_tool(
                "remember", {"content": "written over stdio, stored over the socket"}
            )
            stats = await client.call_tool("stats", {})
            return tools, written, stats

    tools, written, stats = asyncio.run(go())
    assert "remember" in tools and "sql" not in tools
    assert written.is_error is False, text_of(written)
    assert stats.structured_content["path"] == f"unix:{sock}", "ANATID_SOCKET was not honoured"
    assert stats.structured_content["tenant_id"] == str(TENANT)
    memory_id = written.structured_content["memory"]["memory_id"]
    assert handle.get(int(memory_id)) is not None, "the write reached the server's file"


# --------------------------------------------------------------------------- surface


def test_the_recipe_quotes_paths_and_names_the_tenant():
    start, connect = sharing_recipe(
        "/Users/me/Library/Application Support/Claude/memory.anatid",
        tenant=2,
        socket_path="/tmp/anatid/anatid.sock",
    )
    assert start == (
        "anatid-server start --socket /tmp/anatid/anatid.sock "
        "--db '/Users/me/Library/Application Support/Claude/memory.anatid' --tenant 2"
    )
    assert connect == "anatid-mcp --socket /tmp/anatid/anatid.sock --tenant 2"
    assert B.suggested_socket_path().endswith("anatid/anatid.sock")


def test_the_package_reexports_the_backend_names():
    import anatid.integrations.mcp as pkg

    assert pkg.open_backend is open_backend
    assert pkg.ServerHandle is ServerHandle
    assert pkg.DatabaseLocked is DatabaseLocked
    assert pkg.BackendConfigError is BackendConfigError
    assert issubclass(pkg.DatabaseLocked, pkg.BackendError)
