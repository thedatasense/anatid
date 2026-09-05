"""The product review's gaps, closed where an agent meets them: through the integrations.

The core closed each gap first; these tests hold the tool layers to the same behaviour.

* G2, recall by default.  ``recall(query)`` with no seed through the Agents SDK tool, through
  the MCP tool, and through the MCP tool over a socket runs the graph arm from the entities the
  query names and reports them as ``seeds``.  ``anatid-mcp`` builds an embedder from
  ``ANATID_EMBED_*`` / ``--embed-hash``, so a stock server runs the vector arm from
  configuration alone, and a configuration that cannot be honoured is refused with the reason.
* G1 over the server profile.  ``correct`` is a client verb and an MCP tool over a socket, so
  ``AnatidClient`` is a drop-in again and both backends list the same tools.
* Blank entity names, newly reachable through the graph tools, are refused by the core and
  reported by both tool layers, and nothing half-lands.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json

import pytest
import socket

from anatid import AUTO_SEED, Anatid, CorrectionReceipt, HashEmbedder
from anatid.errors import ValidationError

pytest.importorskip("mcp.client", reason="the MCP server needs 'mcp>=2.1'")

from anatid.integrations.mcp.backend import (
    BackendConfigError,
    check_backend_config,
    open_backend,
)
from anatid.integrations.mcp.server import ServerConfig, _parse_args, build_server, main
from conftest import DIM, T0
from test_mcp_over_server import (
    TENANT,
    call,
    clean_env,  # noqa: F401  (registers the fixture)
    ok,
    shared,  # noqa: F401  (registers the fixture)
    sock_dir,  # noqa: F401  (registers the fixture)
    socket_config,
    text_of,
    tool_names,
)

pytest.importorskip("agents", reason="the Agents SDK tools need 'openai-agents'")

from agents.tool_context import ToolContext

from anatid.integrations.openai_agents import create_memory_tools

posix_only = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="needs Unix domain sockets or POSIX file semantics; not available on this platform",
)


MINUTE = _dt.timedelta(minutes=1)

#: The on-call story, stored by entity name alone and with no ``relate`` call, which is how an
#: agent writes through ``remember``.
FACTS = (
    ("Ada leads Kestrel", ("Ada", "Kestrel")),
    ("Kestrel owns the ingest service", ("Kestrel", "ingest service")),
    ("Bo maintains the ingest service", ("Bo", "ingest service")),
)
QUESTION = "who maintains the ingest service"


def store(db) -> None:
    for i, (content, entities) in enumerate(FACTS):
        db.remember(content, entities=list(entities), now=T0 + i * MINUTE)


def invoke(tool, **arguments) -> dict:
    """Run an Agents SDK tool body the way the Runner does once a write is approved."""
    payload = json.dumps(arguments)
    ctx = ToolContext(None, tool_name=tool.name, tool_call_id="call-1", tool_arguments=payload)
    return json.loads(asyncio.run(tool.on_invoke_tool(ctx, payload)))


@pytest.fixture
def mcp_db(tmp_path):
    with Anatid.open(tmp_path / "review.anatid", tenant=3, embedding_dim=DIM) as handle:
        yield handle


@pytest.fixture
def server(mcp_db):
    return build_server(mcp_db, ServerConfig(db=mcp_db.path, tenant=3, env={}))


# =========================================================================== G2: recall by default


def test_the_agents_recall_tool_seeds_the_graph_arm_from_the_query(db):
    store(db)
    tools = {t.name: t for t in create_memory_tools(db)}
    out = invoke(tools["anatid_recall"], query=QUESTION, k=None, seed_entity=None, hops=None)
    assert "error" not in out, out
    assert "graph" in out["arms"] and "text" in out["arms"]
    assert out["seeds"] == ["ingest service"]
    contents = {h["content"] for h in out["hits"]}
    assert "Bo maintains the ingest service" in contents
    assert "Kestrel owns the ingest service" in contents


def test_the_agents_recall_tool_still_honours_an_explicit_seed(db):
    store(db)
    tools = {t.name: t for t in create_memory_tools(db)}
    out = invoke(tools["anatid_recall"], query=QUESTION, k=None, seed_entity="Ada", hops=None)
    assert out["seeds"] == ["Ada"]
    assert "graph" in out["arms"]
    missing = invoke(
        tools["anatid_recall"], query=QUESTION, k=None, seed_entity="Nobody", hops=None
    )
    assert any("Nobody" in note for note in missing["notes"])
    assert missing["seeds"] == ["ingest service"], "an unknown seed falls back to the query's"


def test_the_mcp_recall_tool_seeds_the_graph_arm_from_the_query(server, mcp_db):
    store(mcp_db)
    res = ok(call(server, "recall", {"query": QUESTION}))
    assert "graph" in res["arms"] and "text" in res["arms"]
    assert res["seeds"] == ["ingest service"]
    assert "Bo maintains the ingest service" in {h["memory"]["content"] for h in res["hits"]}

    seeded = ok(call(server, "recall", {"query": QUESTION, "seed_entity": "Ada"}))
    assert seeded["seeds"] == ["Ada"]


@posix_only
def test_recall_over_a_socket_seeds_the_graph_arm_and_carries_the_seeds(request):
    """The client's default, the protocol's ``seeds`` field, and the MCP tool on top of both."""
    sock, handle = request.getfixturevalue("shared")
    store(handle)
    with open_backend(socket_config(sock)) as client:
        hits = client.recall(QUESTION, k=5)
        assert "graph" in hits.arms
        assert hits.seeds == ("ingest service",)
        assert client.recall(QUESTION, k=5, seed_entity=None).arms == ("text",)
        assert client.recall(QUESTION, k=5, seed_entity=AUTO_SEED).seeds == ("ingest service",)

        remote = build_server(client, socket_config(sock))
        res = ok(call(remote, "recall", {"query": QUESTION}))
        assert "graph" in res["arms"]
        assert res["seeds"] == ["ingest service"]


# =========================================================================== G2: the embedder


def test_anatid_mcp_builds_an_embedder_from_the_environment(tmp_path):
    """``ANATID_EMBED_HASH=1`` and nothing else: ``remember`` stores a vector, ``recall`` runs
    all three arms, and ``stats`` says which embedder did it."""
    cfg = ServerConfig(
        db=str(tmp_path / "e.anatid"), tenant=1, embedding_dim=DIM, env={"ANATID_EMBED_HASH": "1"}
    )
    with open_backend(cfg) as db:
        assert isinstance(db, Anatid)
        assert isinstance(db.embedder, HashEmbedder)
        server = build_server(db, cfg)
        written = ok(
            call(
                server,
                "remember",
                {
                    "content": "Bo maintains the ingest service",
                    "entities": ["Bo", "ingest service"],
                },
            )
        )
        stored = db.get(int(written["memory"]["memory_id"]), with_embedding=True)
        assert stored is not None and stored.embedding is not None
        assert len(stored.embedding) == DIM

        res = ok(call(server, "recall", {"query": QUESTION}))
        assert set(res["arms"]) == {"vector", "text", "graph"}
        assert res["seeds"] == ["ingest service"]

        stats = ok(call(server, "stats"))
        assert stats["embedder"]["kind"] == "hash"
        assert stats["embedder"]["dim"] == DIM
        assert stats["extractor"] is None
        assert stats["pending_patches"] == 0


def test_the_embed_hash_flag_reaches_the_configuration():
    assert _parse_args(["--db", "x", "--embed-hash"]).embed_hash is True
    assert _parse_args(["--db", "x"]).embed_hash is None
    cfg = ServerConfig(db="x", embed_hash=True, env={})
    assert cfg.embed_hash is True
    assert ServerConfig(db="x", env={}).embed_hash is None


def test_a_stock_server_without_embedding_settings_has_no_embedder(tmp_path):
    cfg = ServerConfig(db=str(tmp_path / "plain.anatid"), tenant=1, embedding_dim=DIM, env={})
    with open_backend(cfg) as db:
        assert db.embedder is None
        server = build_server(db, cfg)
        store(db)
        res = ok(call(server, "recall", {"query": QUESTION}))
        assert set(res["arms"]) == {"text", "graph"}
        assert ok(call(server, "stats"))["embedder"] is None


def test_half_an_embedding_endpoint_is_refused_before_the_file_is_opened(tmp_path):
    path = tmp_path / "half.anatid"
    cfg = ServerConfig(db=str(path), tenant=1, env={"ANATID_EMBED_MODEL": "text-embedding-3-small"})
    with pytest.raises(BackendConfigError, match="ANATID_EMBED_BASE_URL"):
        open_backend(cfg)
    assert not path.exists()


def test_embedding_and_extraction_settings_are_refused_with_a_socket():
    for env in (
        {"ANATID_EMBED_HASH": "1"},
        {"ANATID_EMBED_BASE_URL": "http://127.0.0.1:1/v1", "ANATID_EMBED_MODEL": "m"},
    ):
        with pytest.raises(BackendConfigError, match="embedding happens in the process"):
            check_backend_config(ServerConfig(socket="/tmp/anatid/x.sock", env=env))
    with pytest.raises(BackendConfigError, match="--embed-hash"):
        check_backend_config(ServerConfig(socket="/tmp/anatid/x.sock", embed_hash=True, env={}))
    with pytest.raises(BackendConfigError, match="ingest tools"):
        check_backend_config(
            ServerConfig(
                socket="/tmp/anatid/x.sock",
                env={
                    "ANATID_EXTRACT_BASE_URL": "http://127.0.0.1:1/v1",
                    "ANATID_EXTRACT_MODEL": "m",
                },
            )
        )
    # A plain socket configuration is still fine.
    check_backend_config(ServerConfig(socket="/tmp/anatid/x.sock", env={}))


def test_main_reports_a_bad_embedding_configuration_and_exits_2(
    tmp_path, request, monkeypatch, capsys
):
    request.getfixturevalue("clean_env")
    monkeypatch.setenv("ANATID_EMBED_MODEL", "text-embedding-3-small")
    code = main(["--db", str(tmp_path / "m.anatid"), "--tenant", "1"])
    assert code == 2
    err = capsys.readouterr().err
    assert "anatid-mcp:" in err
    assert "ANATID_EMBED_BASE_URL" in err
    assert "Traceback" not in err


# =========================================================================== G1 over the server


@posix_only
def test_correct_is_a_client_verb_and_an_mcp_tool_over_the_socket(request):
    sock, handle = request.getfixturevalue("shared")
    with open_backend(socket_config(sock)) as client:
        assert hasattr(client, "correct")
        m = client.remember("Bo maintains the ingest service", entities=["Bo", "ingest service"])
        client.relate("Bo", "ingest service", rel_kind="maintains")
        receipt = client.correct(
            m.memory_id,
            "Cy maintains the ingest service",
            entities=["Cy", "ingest service"],
            add_relations=[("Cy", "ingest service", "maintains")],
            remove_relations=[("Bo", "ingest service")],
        )
        assert isinstance(receipt, CorrectionReceipt)
        assert receipt.edges_closed == 1
        assert receipt.closed == (("Bo", "ingest service", None),)
        assert len(receipt.opened) == 1 and receipt.opened[0].rel_kind == "maintains"
        assert receipt.old.is_current is False and receipt.new.content.startswith("Cy")
        # Verified in the server's own handle, the only process that can read the file.
        assert [x.content for x in handle.recall_2hop("Cy")] == ["Cy maintains the ingest service"]
        assert handle.recall_2hop("Bo") == []

        remote = build_server(client, socket_config(sock))
        assert "correct" in tool_names(remote)
        res = ok(
            call(
                remote,
                "correct",
                {
                    "old_id": str(receipt.new.memory_id),
                    "content": "Dee maintains the ingest service",
                    "entities": ["Dee", "ingest service"],
                    "add_relations": [
                        {"src": "Dee", "dst": "ingest service", "rel_kind": "maintains"}
                    ],
                    "remove_relations": [{"src": "Cy", "dst": "ingest service"}],
                },
            )
        )
        assert res["edges_closed"] == 1
        assert res["memory"]["content"] == "Dee maintains the ingest service"
        assert res["closed"] == [{"src": "Cy", "dst": "ingest service", "rel_kind": None}]
        assert handle.recall_2hop("Cy") == []


@posix_only
def test_both_backends_list_the_same_tools(request, tmp_path):
    sock, _ = request.getfixturevalue("shared")
    with Anatid.open(tmp_path / "local.anatid", tenant=TENANT, embedding_dim=DIM) as local:
        local_tools = tool_names(
            build_server(local, ServerConfig(db=local.path, tenant=TENANT, env={}))
        )
    with open_backend(socket_config(sock)) as client:
        remote_tools = tool_names(build_server(client, socket_config(sock)))
    assert local_tools == remote_tools
    assert "correct" in remote_tools


# =========================================================================== blank entity names


@pytest.mark.parametrize("blank", ["", " ", "\t\n"])
def test_blank_entity_names_are_refused_by_the_core_and_nothing_half_lands(db, blank):
    with pytest.raises(ValidationError, match="empty or whitespace"):
        db.upsert_entity(blank)
    with pytest.raises(ValidationError):
        db.entity_id(blank, create=True)
    with pytest.raises(ValidationError):
        db.relate("Cy", blank)
    assert db.get_entity("Cy") is None, (
        "the relate's first endpoint was created, then not rolled back"
    )
    with pytest.raises(ValidationError):
        db.remember("Bo moved", entities=["Bo", blank])
    assert db.stats()["memories"] == 0 and db.stats()["entities"] == 0

    m = db.remember("Bo maintains the ingest service", entities=["Bo", "ingest service"], now=T0)
    with pytest.raises(ValidationError):
        db.correct(m.memory_id, "Cy maintains the ingest service", add_relations=[("Cy", blank)])
    current = db.get(m.memory_id)
    assert current is not None and current.is_current
    assert db.stats()["entities"] == 2
    assert db.stats()["edges_relates"] == 0


def test_blank_entity_names_are_readable_errors_through_both_tool_layers(db, server, mcp_db):
    tools = {t.name: t for t in create_memory_tools(db)}
    out = invoke(tools["anatid_relate"], src="Cy", dst="", rel_kind=None)
    assert out["error"] == "ValidationError"
    assert "empty or whitespace" in out["message"]
    saved = invoke(tools["anatid_remember"], content="Bo maintains it", entities=["Bo"], kind=None)
    out = invoke(
        tools["anatid_correct"],
        memory_id=saved["memory_id"],
        content="Cy maintains it",
        entities=None,
        kind=None,
        add_relations=[{"src": "Cy", "dst": " ", "rel_kind": None}],
        remove_relations=None,
    )
    assert out["error"] == "ValidationError"
    assert db.get(int(saved["memory_id"])).is_current

    res = call(server, "relate", {"src": "Cy", "dst": ""})
    assert res.is_error is True
    assert "empty or whitespace" in text_of(res)
    res = call(server, "remember", {"content": "x", "entities": [" "]})
    assert res.is_error is True
    assert mcp_db.stats()["entities"] == 0
