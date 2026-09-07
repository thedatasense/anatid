"""The ingestion pipeline through the two integrations: ``anatid_ingest`` and ``ingest``/``apply_patch``.

:mod:`anatid.ingest` takes text and applies one reviewed patch in one transaction.  These tests
hold the tool layers to that contract:

* the Agents SDK tool ``anatid_ingest`` is built only when an extractor is given, waits for
  approval like every write, and a ``dry_run`` proposes the patch, returns its diff and never
  waits, because it writes nothing;
* a patch a dry run returned can be handed back, edited or not, and is applied exactly, without
  asking the model again;
* the MCP tools split the same flow in two calls: ``ingest`` proposes and returns a
  ``patch_id`` with the diff, ``apply_patch`` commits, and a failed apply keeps the proposal;
* a proposal is applied at most once: a second ``apply_patch`` for the same ``patch_id``,
  concurrent with the first or a retry after it, is answered with the first one's receipt;
* every row the patch writes cites the episode, and every id in every result is a decimal
  string that survives a real JavaScript parser.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json

import pytest
import socket
import threading

from anatid import Anatid, utcnow
from anatid.errors import NotFoundError
from anatid.ids import new_id
from anatid.ingest import MemoryPatch, ScriptedExtractor

pytest.importorskip("mcp.client", reason="the MCP server needs 'mcp>=2.1'")

from anatid.integrations.mcp.backend import BackendConfigError, open_backend
from anatid.integrations.mcp.ingest import (
    AppliedPatch,
    ExtractorConfigError,
    PendingPatches,
    describe_extractor,
    extractor_from_config,
)
from anatid.integrations.mcp.server import ServerConfig, build_server
from conftest import DIM
from test_ingest import NOTE1, NOTE2, patch1, patch2
from test_mcp_over_server import (
    call,
    ok,
    shared,  # noqa: F401  (registers the fixture)
    sock_dir,  # noqa: F401  (registers the fixture)
    socket_config,
    text_of,
    tool_names,
)
from test_wire_ids import NODE, assert_wire_safe, through_node

pytest.importorskip("agents", reason="the Agents SDK tools need 'openai-agents'")

from agents.tool_context import ToolContext
from mcp.client import Client

from anatid.integrations.openai_agents import (
    INGEST_TOOL,
    TOOL_NAMES,
    ApprovalRequest,
    approve_low_risk,
    create_memory_tools,
)

posix_only = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="needs Unix domain sockets or POSIX file semantics; not available on this platform",
)


requires_node = pytest.mark.skipif(NODE is None, reason="node is not on PATH")

APRIL = _dt.datetime(2026, 4, 1)


def run(coro):
    return asyncio.run(coro)


def invoke(tool, **arguments) -> dict:
    payload = json.dumps(arguments)
    ctx = ToolContext(None, tool_name=tool.name, tool_call_id="call-1", tool_arguments=payload)
    return json.loads(run(tool.on_invoke_tool(ctx, payload)))


def ingest_call(tool, text, **rest) -> dict:
    """``anatid_ingest`` with every optional argument spelled out, the way the SDK sends them."""
    arguments = {"text": text, "source": None, "dry_run": None, "patch": None}
    arguments.update(rest)
    return invoke(tool, **arguments)


def listed(server) -> dict:
    async def go():
        async with Client(server) as client:
            return {t.name: t for t in (await client.list_tools()).tools}

    return run(go())


def instructions_of(server) -> str:
    async def go():
        async with Client(server) as client:
            return client.instructions or ""

    return run(go())


# =========================================================================== the Agents SDK tool


def test_anatid_ingest_is_built_only_with_an_extractor(db):
    assert INGEST_TOOL not in {t.name for t in create_memory_tools(db)}
    assert INGEST_TOOL not in TOOL_NAMES
    with pytest.raises(ValueError, match="extractor="):
        create_memory_tools(db, include=[INGEST_TOOL])

    tools = create_memory_tools(db, extractor=ScriptedExtractor([]))
    assert [t.name for t in tools] == list(TOOL_NAMES) + [INGEST_TOOL]
    ingest = tools[-1]
    assert callable(ingest.needs_approval), "anatid_ingest is a write and has to be gated"
    only = create_memory_tools(db, extractor=ScriptedExtractor([]), include=[INGEST_TOOL])
    assert [t.name for t in only] == [INGEST_TOOL]
    without = create_memory_tools(db, extractor=ScriptedExtractor([]), exclude=[INGEST_TOOL])
    assert INGEST_TOOL not in {t.name for t in without}


def test_a_dry_run_proposes_without_writing_and_without_waiting(db):
    extractor = ScriptedExtractor([patch1()])
    tools = {t.name: t for t in create_memory_tools(db, extractor=extractor)}
    gate = tools[INGEST_TOOL].needs_approval
    assert run(gate(None, {"text": NOTE1, "dry_run": True}, "c")) is False
    assert run(gate(None, {"text": NOTE1}, "c")) is True
    assert run(gate(None, {"text": NOTE1, "dry_run": False}, "c")) is True

    out = ingest_call(tools[INGEST_TOOL], NOTE1, source="notes/2026-03-02.md", dry_run=True)
    assert out["applied"] is False and out["dry_run"] is True
    assert out["operations"] == 6 and out["is_empty"] is False
    assert "+ fact" in out["diff"] and "+ relation" in out["diff"]
    assert out["how"].startswith("proposed by the extractor")
    assert db.stats()["memories"] == 0 and db.stats()["episodes"] == 0
    assert extractor.remaining == 0 and extractor.calls[0].text == NOTE1
    proposed = MemoryPatch.from_json(out["patch_json"])
    assert proposed.operations == 6 and proposed.source_text == NOTE1


def test_anatid_ingest_applies_the_patch_with_the_note_as_its_episode(db):
    tools = {
        t.name: t
        for t in create_memory_tools(
            db, extractor=ScriptedExtractor([patch1(), patch2()]), writer="notes-bot"
        )
    }
    first = ingest_call(tools[INGEST_TOOL], NOTE1, source="notes/2026-03-02.md")
    assert first["applied"] is True
    assert len(first["memories_created"]) == 3 and len(first["relations_opened"]) == 3
    assert first["changes"] == 6
    assert "episode" in first["summary"] and first["writer"] == "notes-bot"
    episode = db.get_episode(int(first["episode_id"]))
    assert episode is not None
    assert episode.content == NOTE1 and episode.source == "notes/2026-03-02.md"
    for mid in first["memories_created"]:
        memory = db.get(int(mid))
        assert memory is not None and memory.episode_id == int(first["episode_id"])
        assert memory.writer == "notes-bot"
    assert {m.content for m in db.recall_2hop("Ada")} == {
        "Ada leads Kestrel",
        "Kestrel owns the ingest service",
        "Bo maintains the ingest service",
    }

    second = ingest_call(tools[INGEST_TOOL], NOTE2, source="notes/2026-06-15.md")
    assert second["applied"] is True
    assert len(second["corrections"]) == 1 and len(second["relations_closed"]) == 1
    assert "~ correction" in second["diff"]
    old_id = int(second["corrections"][0]["old_id"])
    old = db.get(old_id)
    assert old is not None and old.content == "Bo maintains the ingest service"
    assert old.is_current is False
    reachable = {m.content for m in db.recall_2hop("Ada")}
    assert "Cy maintains the ingest service" in reachable
    assert "Bo maintains the ingest service" not in reachable
    assert db.recall_2hop("Bo") == [] or all(
        "ingest service" not in m.content for m in db.recall_2hop("Bo")
    )


def test_a_patch_from_a_dry_run_is_applied_as_given_without_the_model(db):
    extractor = ScriptedExtractor([patch1()])
    tools = {t.name: t for t in create_memory_tools(db, extractor=extractor)}
    dry = ingest_call(tools[INGEST_TOOL], NOTE1, dry_run=True)
    edited = json.loads(dry["patch_json"])
    edited["add_facts"] = edited["add_facts"][:1]
    edited["add_relations"] = edited["add_relations"][:1]
    out = ingest_call(tools[INGEST_TOOL], NOTE1, patch=json.dumps(edited))
    assert out["applied"] is True
    assert out["how"] == "patch supplied by the caller"
    assert len(out["memories_created"]) == 1 and len(out["relations_opened"]) == 1
    assert extractor.remaining == 0 and len(extractor.calls) == 1, "the model was asked once"
    assert db.get(int(out["memories_created"][0])).content == "Ada leads Kestrel"


def test_the_review_hook_can_decline_or_edit(db):
    declined = {
        t.name: t
        for t in create_memory_tools(
            db, extractor=ScriptedExtractor([patch1()]), ingest_review=lambda patch: None
        )
    }
    out = ingest_call(declined[INGEST_TOOL], NOTE1)
    assert out["applied"] is False and out["declined"] is True
    assert "nothing was written" in out["note"]
    assert db.stats()["memories"] == 0 and db.stats()["episodes"] == 0

    edited = {
        t.name: t
        for t in create_memory_tools(
            db,
            extractor=ScriptedExtractor([patch1()]),
            ingest_review=lambda patch: patch.replace(add_relations=()),
        )
    }
    out = ingest_call(edited[INGEST_TOOL], NOTE1)
    assert out["applied"] is True
    assert len(out["memories_created"]) == 3 and out["relations_opened"] == []
    assert "+ relation" not in out["diff"]


def test_empty_text_is_an_error_not_a_write(db):
    tools = {t.name: t for t in create_memory_tools(db, extractor=ScriptedExtractor([patch1()]))}
    out = ingest_call(tools[INGEST_TOOL], "   ")
    assert out["error"] == "ValueError"
    assert db.stats()["episodes"] == 0


def test_approve_low_risk_stops_ingest_unless_told_otherwise():
    request = ApprovalRequest(INGEST_TOOL, {"text": "Cy took over the ingest service."}, "c")
    assert approve_low_risk()(request) is True
    assert approve_low_risk(ingest=True)(request) is False
    long_note = ApprovalRequest(INGEST_TOOL, {"text": "x" * 6}, "c")
    assert approve_low_risk(ingest=True, max_content_chars=5)(long_note) is True
    assert request.is_dry_run is False
    assert ApprovalRequest(INGEST_TOOL, {"text": "x", "dry_run": True}, "c").is_dry_run is True


def test_anatid_ingest_results_carry_no_integer_a_json_number_would_round(db):
    extractor = ScriptedExtractor([patch1(), patch1(), patch2()])  # the dry run consumes one
    tools = {t.name: t for t in create_memory_tools(db, extractor=extractor)}
    dry = ingest_call(tools[INGEST_TOOL], NOTE1, dry_run=True)
    first = ingest_call(tools[INGEST_TOOL], NOTE1)
    second = ingest_call(tools[INGEST_TOOL], NOTE2)
    for label, payload in (("dry", dry), ("first", first), ("second", second)):
        assert_wire_safe(f"anatid_ingest {label}", payload)
    assert int(first["episode_id"]) > 2**53
    assert int(second["corrections"][0]["old_id"]) > 2**53
    if NODE is not None:
        text = json.dumps(second)
        assert json.loads(through_node(text)) == json.loads(text)


# =========================================================================== the MCP tools


@pytest.fixture
def mcp_db(tmp_path):
    with Anatid.open(tmp_path / "ingest.anatid", tenant=3, embedding_dim=DIM) as handle:
        yield handle


def ingest_server(db, *patches):
    return build_server(
        db, ServerConfig(db=db.path, tenant=3, env={}), extractor=ScriptedExtractor(list(patches))
    )


def test_the_ingest_tools_are_registered_only_with_an_extractor(mcp_db):
    plain = build_server(mcp_db, ServerConfig(db=mcp_db.path, tenant=3, env={}))
    assert not {"ingest", "apply_patch"} & tool_names(plain)
    assert "apply_patch" not in instructions_of(plain)

    with_extractor = ingest_server(mcp_db)
    tools = listed(with_extractor)
    assert {"ingest", "apply_patch"} <= set(tools)
    assert tools["ingest"].annotations.read_only_hint is True
    assert tools["ingest"].annotations.open_world_hint is True
    assert tools["apply_patch"].annotations.destructive_hint is True
    assert tools["apply_patch"].annotations.read_only_hint is False
    assert "WITHOUT writing" in tools["ingest"].description
    assert "apply_patch(patch_id)" in instructions_of(with_extractor)
    prop = tools["apply_patch"].input_schema["properties"]["patch_id"]
    assert prop["type"] == "string" and "2**53" in prop["description"]


def test_the_extractor_comes_from_the_environment(mcp_db):
    env = {
        "ANATID_EXTRACT_BASE_URL": "http://127.0.0.1:9/v1",
        "ANATID_EXTRACT_MODEL": "z-ai/glm-5.3-flash",
        "ANATID_EXTRACT_API_KEY": "not-printed",
        "ANATID_EXTRACT_REASONING": "1",
    }
    extractor = extractor_from_config(env=env)
    assert extractor is not None
    assert extractor.model == "z-ai/glm-5.3-flash"
    assert extractor.extra_body == {"reasoning": {"enabled": True}}
    described = describe_extractor(extractor)
    assert described["kind"] == "openai_compatible" and described["model"] == "z-ai/glm-5.3-flash"
    assert "not-printed" not in json.dumps(described)

    server = build_server(mcp_db, ServerConfig(db=mcp_db.path, tenant=3, env=env))
    assert {"ingest", "apply_patch"} <= tool_names(server)
    stats = ok(call(server, "stats"))
    assert stats["extractor"]["model"] == "z-ai/glm-5.3-flash"
    assert stats["pending_patches"] == 0

    assert extractor_from_config(env={}) is None
    with pytest.raises(ExtractorConfigError, match="ANATID_EXTRACT_BASE_URL"):
        extractor_from_config(env={"ANATID_EXTRACT_MODEL": "m"})
    with pytest.raises(ExtractorConfigError, match="ANATID_EXTRACT_MODEL"):
        build_server(
            mcp_db,
            ServerConfig(db=mcp_db.path, tenant=3, env={"ANATID_EXTRACT_BASE_URL": "http://x/v1"}),
        )


def test_a_read_only_server_registers_no_ingest_tools(tmp_path):
    path = tmp_path / "ro.anatid"
    with Anatid.open(path, tenant=3, embedding_dim=DIM):
        pass
    with Anatid.open(path, tenant=3, read_only=True) as ro:
        server = build_server(
            ro,
            ServerConfig(db=str(path), tenant=3, read_only=True, env={}),
            extractor=ScriptedExtractor([]),
        )
        assert not {"ingest", "apply_patch", "remember"} & tool_names(server)


@posix_only
def test_ingest_over_a_socket_is_refused(request):
    sock, _ = request.getfixturevalue("shared")
    with open_backend(socket_config(sock)) as client:
        with pytest.raises(BackendConfigError, match=r"embedded anatid\.Anatid handle"):
            build_server(client, socket_config(sock), extractor=ScriptedExtractor([]))
        with pytest.raises(BackendConfigError, match="ingest tools"):
            open_backend(
                ServerConfig(
                    socket=str(sock),
                    tenant=3,
                    env={"ANATID_EXTRACT_BASE_URL": "http://x/v1", "ANATID_EXTRACT_MODEL": "m"},
                )
            )


def test_ingest_proposes_and_apply_patch_commits(mcp_db):
    server = ingest_server(mcp_db, patch1())
    proposed = ok(
        call(
            server,
            "ingest",
            {"text": NOTE1, "source": "notes/2026-03-02.md", "writer": "notes-bot"},
        )
    )
    assert proposed["applied"] is False
    assert proposed["patch_id"].isdigit() and int(proposed["patch_id"]) > 2**53
    assert proposed["operations"] == 6 and proposed["is_empty"] is False
    assert proposed["diff"].startswith("memory patch: 3 facts, 3 relation(s) added")
    assert proposed["patch"]["source_text"] == NOTE1
    assert "apply_patch" in proposed["next"]
    assert mcp_db.stats()["memories"] == 0 and mcp_db.stats()["episodes"] == 0
    assert ok(call(server, "stats"))["pending_patches"] == 1

    applied = ok(call(server, "apply_patch", {"patch_id": proposed["patch_id"]}))
    assert applied["applied"] is True and applied["patch_id"] == proposed["patch_id"]
    assert len(applied["memories_created"]) == 3 and len(applied["relations_opened"]) == 3
    assert applied["changes"] == 6 and applied["writer"] == "notes-bot"
    episode = mcp_db.get_episode(int(applied["episode_id"]))
    assert episode is not None and episode.content == NOTE1
    assert episode.source == "notes/2026-03-02.md"
    for mid in applied["memories_created"]:
        assert mcp_db.get(int(mid)).episode_id == int(applied["episode_id"])
    assert ok(call(server, "stats"))["pending_patches"] == 0
    assert applied["already_applied"] is False and "note" not in applied

    # A retry, say after the reply was lost, is the same receipt again and writes nothing.
    again = ok(call(server, "apply_patch", {"patch_id": proposed["patch_id"]}))
    assert again["applied"] is True and again["already_applied"] is True
    assert again["episode_id"] == applied["episode_id"]
    assert again["memories_created"] == applied["memories_created"]
    assert again["relations_opened"] == applied["relations_opened"]
    assert again["summary"] == applied["summary"] and again["diff"] == applied["diff"]
    assert "nothing was written" in again["note"] and "call ingest again" in again["note"]
    assert mcp_db.stats()["memories"] == 3 and mcp_db.stats()["episodes"] == 1
    edited = dict(proposed["patch"])
    edited["add_facts"] = []
    third = ok(call(server, "apply_patch", {"patch_id": proposed["patch_id"], "patch": edited}))
    assert (
        third["already_applied"] is True
        and third["memories_created"] == applied["memories_created"]
    )
    assert mcp_db.stats()["memories"] == 3, "an edited retry does not apply either"

    unknown = call(server, "apply_patch", {"patch_id": str(new_id())})
    assert unknown.is_error is True and "call ingest again" in text_of(unknown).lower()
    assert "no pending patch" in text_of(unknown)


def test_apply_patch_takes_an_edited_patch_and_keeps_a_failed_proposal(mcp_db):
    server = ingest_server(mcp_db, patch1())
    proposed = ok(call(server, "ingest", {"text": NOTE1}))

    broken = dict(proposed["patch"])
    broken["corrections"] = [{"old_id": str(new_id()), "new_content": "nothing to correct"}]
    failed = call(server, "apply_patch", {"patch_id": proposed["patch_id"], "patch": broken})
    assert failed.is_error is True
    assert mcp_db.stats()["memories"] == 0 and mcp_db.stats()["episodes"] == 0, (
        "nothing half-landed"
    )
    assert ok(call(server, "stats"))["pending_patches"] == 1, "a failed apply keeps the proposal"

    edited = dict(proposed["patch"])
    edited["add_facts"] = edited["add_facts"][:1]
    edited["add_relations"] = []
    applied = ok(
        call(
            server,
            "apply_patch",
            {"patch_id": proposed["patch_id"], "patch": edited, "writer": "me"},
        )
    )
    assert len(applied["memories_created"]) == 1 and applied["relations_opened"] == []
    assert applied["writer"] == "me"
    assert mcp_db.get(int(applied["memories_created"][0])).content == "Ada leads Kestrel"
    assert ok(call(server, "stats"))["pending_patches"] == 0


def test_two_apply_patch_calls_for_one_proposal_write_it_once(mcp_db, monkeypatch):
    """The race a product review forced: two ``apply_patch`` calls for one ``patch_id``.

    Before the fix both read the proposal, both committed, and the graph held two copies of
    every memory and edge.  The first call is held inside ``MemoryPatch.apply`` while the second
    arrives; the second has to wait for the first's outcome and be handed its receipt.
    """
    server = ingest_server(mcp_db, patch1())
    proposed = ok(call(server, "ingest", {"text": NOTE1}))
    started, release = threading.Event(), threading.Event()
    original = MemoryPatch.apply

    def slow_apply(self, *args, **kwargs):
        started.set()
        assert release.wait(30), "the test never released the first apply"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MemoryPatch, "apply", slow_apply)
    results = {}

    def go(name):
        results[name] = call(server, "apply_patch", {"patch_id": proposed["patch_id"]})

    first = threading.Thread(target=go, args=("first",))
    first.start()
    assert started.wait(30), "the first call never reached apply"
    second = threading.Thread(target=go, args=("second",))
    second.start()
    second.join(0.5)
    assert second.is_alive(), "the second call must wait for the first, not fail or run ahead"
    release.set()
    first.join(60)
    second.join(60)
    assert not first.is_alive() and not second.is_alive()

    a, b = ok(results["first"]), ok(results["second"])
    assert a["already_applied"] is False and b["already_applied"] is True
    assert a["episode_id"] == b["episode_id"]
    assert a["memories_created"] == b["memories_created"] and len(a["memories_created"]) == 3
    assert mcp_db.stats()["memories"] == 3 and mcp_db.stats()["episodes"] == 1
    assert mcp_db.stats()["entities"] == 4, "one Ada, one Kestrel, one Bo, one ingest service"
    assert ok(call(server, "stats"))["pending_patches"] == 0


def test_the_second_note_corrects_the_owner_and_history_keeps_the_first(mcp_db):
    server = ingest_server(mcp_db, patch1(), patch2())
    first = ok(call(server, "ingest", {"text": NOTE1, "source": "notes/2026-03-02.md"}))
    ok(call(server, "apply_patch", {"patch_id": first["patch_id"]}))
    before = utcnow()

    second = ok(call(server, "ingest", {"text": NOTE2, "source": "notes/2026-06-15.md"}))
    assert "~ correction  memory " in second["diff"], (
        "the text-named correction was resolved to an id"
    )
    applied = ok(call(server, "apply_patch", {"patch_id": second["patch_id"]}))
    assert len(applied["corrections"]) == 1 and len(applied["relations_closed"]) == 1

    now = ok(call(server, "recall", {"query": "who maintains the ingest service"}))
    contents = {h["memory"]["content"] for h in now["hits"]}
    assert "Cy maintains the ingest service" in contents
    assert "Bo maintains the ingest service" not in contents
    then = ok(
        call(
            server,
            "recall",
            {"query": "who maintains the ingest service", "as_of": before.isoformat()},
        )
    )
    assert "Bo maintains the ingest service" in {h["memory"]["content"] for h in then["hits"]}
    prov = ok(call(server, "provenance", {"memory_id": applied["corrections"][0]["new_id"]}))
    assert prov["source_text"] == NOTE1


def test_mcp_ingest_payloads_survive_a_javascript_parser(mcp_db):
    server = ingest_server(mcp_db, patch1(), patch2())
    proposed = ok(call(server, "ingest", {"text": NOTE1}))
    applied = ok(call(server, "apply_patch", {"patch_id": proposed["patch_id"]}))
    second = ok(call(server, "ingest", {"text": NOTE2}))
    for label, payload in (("ingest", proposed), ("apply_patch", applied), ("ingest(2)", second)):
        assert_wire_safe(f"mcp {label}", payload)
    if NODE is not None:
        for payload in (proposed, applied, second):
            text = json.dumps(payload)
            assert json.loads(through_node(text)) == json.loads(text)


# =========================================================================== the pending table


def test_pending_patches_is_bounded_and_addressed_by_id():
    table = PendingPatches(limit=2)
    a = table.put(patch1(), text=NOTE1)
    b = table.put(patch2(), text=NOTE2)
    assert len(table) == 2 and a.patch_id in table and b.patch_id in table
    c = table.put(patch1(), text=NOTE1, source="s", writer="w")
    assert len(table) == 2 and a.patch_id not in table, "the oldest is dropped when full"
    assert table.get(c.patch_id).source == "s" and table.get(c.patch_id).writer == "w"
    with pytest.raises(Exception, match="no pending patch"):
        table.get(a.patch_id)
    assert table.discard(b.patch_id) is True and table.discard(b.patch_id) is False
    assert len(table) == 1
    with pytest.raises(ValueError):
        PendingPatches(limit=0)


def test_pending_patches_hands_a_proposal_to_one_caller_and_keeps_its_receipt(db):
    table = PendingPatches(limit=2)
    entry = table.put(patch1(), text=NOTE1)

    claimed = table.claim(entry.patch_id)
    assert claimed is entry and entry.patch_id not in table and len(table) == 0

    # The apply failed: the proposal goes back and can be claimed again.
    table.restore(entry)
    assert entry.patch_id in table and table.claim(entry.patch_id) is entry

    # The apply landed: from now on the id means the receipt, and only the receipt.
    receipt = patch1().replace(source_text=NOTE1).apply(db, writer="w")
    applied = table.settle(entry, receipt)
    assert isinstance(applied, AppliedPatch)
    assert applied.patch_id == entry.patch_id and applied.receipt is receipt
    assert table.claim(entry.patch_id) is applied and table.claim(entry.patch_id) is applied
    assert entry.patch_id not in table and len(table) == 0
    with pytest.raises(NotFoundError, match="no pending patch"):
        table.get(entry.patch_id)

    # Receipts are bounded the way proposals are, oldest first.
    for _ in range(2):
        later = table.put(patch2(), text=NOTE2)
        table.settle(table.claim(later.patch_id), receipt)
    with pytest.raises(NotFoundError, match="has been forgotten"):
        table.claim(entry.patch_id)

    # A claim for an id another caller holds waits for that caller's outcome.
    held = table.put(patch2(), text=NOTE2)
    assert table.claim(held.patch_id) is held
    got: list = []
    waiter = threading.Thread(target=lambda: got.append(table.claim(held.patch_id)))
    waiter.start()
    waiter.join(0.3)
    assert waiter.is_alive() and got == [], "in flight: the second claim blocks"
    table.restore(held)
    waiter.join(10)
    assert got == [held], "after a failure the waiter gets the proposal itself"
    outcome = table.settle(held, receipt)
    assert table.claim(held.patch_id) is outcome
