"""The graph tools: an agent maintains the edges recall walks, through both integrations.

The product review's reproduction, held here as a test.  Three connected facts remembered by
entity name alone, "Ada leads Kestrel", "Kestrel owns the ingest service" and "Bo maintains the
ingest service", leave ``recall_2hop("Ada")`` with one of them, because memories are filed under
the entities they name and nothing links those entities to each other.  Before this file the
Agents SDK tools had no way to add that link (no ``relate``) and neither integration had a way
to retire one (no ``unrelate``) or to correct a fact together with its edges.  Now each has
``relate``, ``unrelate`` and ``correct``, and the same reproduction ends with all three facts
reachable from Ada.

``correct`` is :meth:`anatid.Anatid.correct`, one verb that runs ``supersede``, ``unrelate`` and
``relate`` inside one transaction.  The atomicity tests force a failure after the memory has been
superseded and check that the memory is still current and every edge is where it was.  The node
tests hand every id the new tools return to a real JavaScript JSON parser and check it comes
back unchanged and still addresses the row; they skip with a reason when ``node`` is not on PATH.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import shutil
import subprocess

import pytest

from anatid import Anatid, CorrectionReceipt
from anatid.errors import NotFoundError, ValidationError
from anatid.integrations.wire import JS_MAX_SAFE_INTEGER
from anatid.visibility import current_row_sql

agents = pytest.importorskip("agents", reason='needs pip install "anatid[agents]"')
pytest.importorskip("mcp.client", reason="the MCP server needs 'mcp>=2.1'")

from agents import Agent, Runner  # noqa: E402
from agents.run_config import RunConfig  # noqa: E402
from agents.testing import ScriptedModel, assistant_message, function_call  # noqa: E402
from agents.tool_context import ToolContext  # noqa: E402
from mcp.client import Client  # noqa: E402

from anatid.integrations.mcp.server import ServerConfig, build_server  # noqa: E402
from anatid.integrations.openai_agents import (  # noqa: E402
    ApprovalRequest,
    approve_low_risk,
    create_memory_tools,
)

agents.set_tracing_disabled(True)
NO_TRACE = RunConfig(tracing_disabled=True)

DIM = 8
T0 = dt.datetime(2026, 1, 1, 12, 0, 0)
T1 = dt.datetime(2026, 2, 1, 12, 0, 0)

#: The review's three facts, each filed under the entities its sentence names.
FACTS = [
    ("Ada leads Kestrel", ["Ada", "Kestrel"]),
    ("Kestrel owns the ingest service", ["Kestrel", "ingest service"]),
    ("Bo maintains the ingest service", ["Bo", "ingest service"]),
]
ALL_THREE = {content for content, _ in FACTS}

NODE = shutil.which("node")
requires_node = pytest.mark.skipif(
    NODE is None,
    reason="node is not on PATH; the round trip needs a real JavaScript JSON parser",
)
NODE_ROUND_TRIP = """
let raw = "";
process.stdin.on("data", (chunk) => { raw += chunk; });
process.stdin.on("end", () => {
  process.stdout.write(JSON.stringify(JSON.parse(raw)));
});
"""


# --------------------------------------------------------------------------- helpers


def through_node(payload: str) -> str:
    """Hand ``payload`` to a real node, and return what its JSON parser gives back."""
    done = subprocess.run(
        [str(NODE), "-e", NODE_ROUND_TRIP],
        input=payload,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def unsafe_ints(value, path: str = "$") -> list[str]:
    """Every JSON path in ``value`` holding an integer a JSON number cannot carry exactly."""
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [] if -JS_MAX_SAFE_INTEGER <= value <= JS_MAX_SAFE_INTEGER else [path]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in unsafe_ints(v, f"{path}.{k}")]
    if isinstance(value, (list, tuple)):
        return [p for i, v in enumerate(value) for p in unsafe_ints(v, f"{path}[{i}]")]
    return []


def current_relations(db) -> set[tuple[str, str, str | None]]:
    """Every RELATES_TO edge the database still believes, as (src name, dst name, rel_kind)."""
    rows = db.execute(
        f"SELECT s.name, d.name, r.rel_kind FROM edges_relates r "
        f"JOIN entities s ON s.entity_id = r.src "
        f"JOIN entities d ON d.entity_id = r.dst "
        f"WHERE r.tenant_id = ? AND {current_row_sql('r')}",
        [db.namespace.tenant_id],
    ).fetchall()
    return {(src, dst, kind) for src, dst, kind in rows}


def current_contents(db) -> set[str]:
    rows = db.execute(
        f"SELECT content FROM memories WHERE tenant_id = ? AND {current_row_sql()}",
        [db.namespace.tenant_id],
    ).fetchall()
    return {row[0] for row in rows}


def reachable_from(db, seed: str) -> set[str]:
    return {m.content for m in db.recall_2hop(seed)}


def invoke(tool, **arguments) -> dict:
    """Run an Agents SDK tool body the way the Runner does once a write is approved."""
    payload = json.dumps(arguments)
    ctx = ToolContext(None, tool_name=tool.name, tool_call_id="call-1", tool_arguments=payload)
    return json.loads(asyncio.run(tool.on_invoke_tool(ctx, payload)))


def call(server, name: str, arguments: dict | None = None):
    """One MCP ``tools/call`` round trip; returns the ``CallToolResult``, errors included."""

    async def go():
        async with Client(server) as client:
            return await client.call_tool(name, arguments or {})

    return asyncio.run(go())


def list_tools(server) -> dict:
    async def go():
        async with Client(server) as client:
            return {t.name: t for t in (await client.list_tools()).tools}

    return asyncio.run(go())


def ok(result) -> dict:
    assert result.is_error is False, error_text(result)
    assert result.structured_content is not None
    return result.structured_content


def error_text(result) -> str:
    return " ".join(getattr(c, "text", "") for c in result.content)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def db():
    with Anatid.open(":memory:", tenant=1, embedding_dim=DIM) as handle:
        yield handle


@pytest.fixture
def agent_tools(db):
    return {t.name: t for t in create_memory_tools(db)}


@pytest.fixture
def mcp_db(tmp_path):
    with Anatid.open(tmp_path / "graph.anatid", tenant=3, embedding_dim=DIM) as handle:
        yield handle


@pytest.fixture
def server(mcp_db):
    return build_server(mcp_db, ServerConfig(db=mcp_db.path, tenant=3, env={}))


def bo_scene(db):
    """Bo maintains the ingest service, with the edge that says so.  Returns Bo's memory."""
    memory = db.remember(
        "Bo maintains the ingest service", entities=["Bo", "ingest service"], now=T0
    )
    db.relate("Bo", "ingest service", rel_kind="maintains", now=T0)
    return memory


# =========================================================================== the core verb


def test_correct_moves_the_memory_and_its_edges_in_one_transaction(db):
    old = bo_scene(db)

    receipt = db.correct(
        old.memory_id,
        "Cy maintains the ingest service",
        entities=["Cy", "ingest service"],
        add_relations=[("Cy", "ingest service", "maintains")],
        remove_relations=[("Bo", "ingest service", "maintains")],
        writer="handover",
        now=T1,
    )

    assert isinstance(receipt, CorrectionReceipt)
    assert receipt.at == T1
    assert receipt.new.content == "Cy maintains the ingest service"
    assert receipt.new.is_current and receipt.new.valid_from == T1
    assert receipt.new.writer == "handover"
    assert receipt.old.memory_id == old.memory_id
    assert receipt.old.is_current is False and receipt.old.valid_to == T1
    assert receipt.closed == (("Bo", "ingest service", "maintains"),)
    assert receipt.edges_closed == 1
    assert [(e.rel_kind, e.valid_from, e.writer) for e in receipt.opened] == [
        ("maintains", T1, "handover")
    ]

    assert current_relations(db) == {("Cy", "ingest service", "maintains")}
    assert current_contents(db) == {"Cy maintains the ingest service"}
    assert reachable_from(db, "ingest service") == {"Cy maintains the ingest service"}

    # Time travel sees the whole correction or none of it: one timestamp on every row.
    before = db.as_of(T1 - dt.timedelta(seconds=1))
    assert {m.content for m in before.recall_2hop("Bo")} == {"Bo maintains the ingest service"}
    assert {m.content for m in before.recall_2hop("ingest service")} == {
        "Bo maintains the ingest service"
    }
    assert db.provenance(receipt.new.memory_id).depth == 1
    assert [m.memory_id for m in db.provenance(receipt.new.memory_id).chain] == [
        receipt.new.memory_id,
        old.memory_id,
    ]


def test_correct_inherits_the_old_entities_and_accepts_two_element_relations(db):
    old = db.remember("Ada leads Kestrel", entities=["Ada", "Kestrel"], now=T0)
    receipt = db.correct(
        old.memory_id, "Ada leads Kestrel and Heron", add_relations=[("Ada", "Heron")]
    )
    assert {e.name for e in db.entities_of(receipt.new.memory_id)} == {"Ada", "Kestrel"}
    assert current_relations(db) == {("Ada", "Heron", None)}
    assert receipt.closed == () and receipt.edges_closed == 0


def test_correct_rolls_back_when_a_relation_names_an_unknown_entity(db):
    """The supersede lands first, the unrelate fails: nothing may survive."""
    old = bo_scene(db)
    edges, contents, entities = current_relations(db), current_contents(db), db.stats()["entities"]

    with pytest.raises(NotFoundError, match="Nobody"):
        db.correct(
            old.memory_id,
            "Cy maintains the ingest service",
            entities=["Cy", "ingest service"],
            add_relations=[("Cy", "ingest service", "maintains")],
            remove_relations=[("Nobody", "ingest service")],
            now=T1,
        )

    reread = db.get(old.memory_id)
    assert reread is not None and reread.is_current
    assert current_contents(db) == contents
    assert current_relations(db) == edges
    assert db.stats()["entities"] == entities, "an entity from the rolled-back half survived"
    assert db.get_entity("Cy") is None
    assert db.versions(old.memory_id) == [reread], "no closed version was written"


def test_correct_rolls_back_when_the_last_step_fails(db, monkeypatch):
    """Forced at the final step, after the memory was superseded and the old edge closed."""
    old = bo_scene(db)
    edges, contents = current_relations(db), current_contents(db)

    def boom(*args, **kwargs):
        raise RuntimeError("the edge write failed")

    monkeypatch.setattr(Anatid, "relate", boom)
    with pytest.raises(RuntimeError, match="the edge write failed"):
        db.correct(
            old.memory_id,
            "Cy maintains the ingest service",
            add_relations=[("Cy", "ingest service", "maintains")],
            remove_relations=[("Bo", "ingest service", "maintains")],
            now=T1,
        )

    assert db.get(old.memory_id).is_current
    assert current_contents(db) == contents
    assert current_relations(db) == edges
    assert (
        db.execute("SELECT count(*) FROM edges_relates WHERE valid_to IS NOT NULL").fetchone()[0]
        == 0
    ), "a closed edge version survived the rollback"

    # Nothing was consumed by the failure: the same correction applies cleanly afterwards.
    monkeypatch.undo()
    receipt = db.correct(
        old.memory_id,
        "Cy maintains the ingest service",
        add_relations=[("Cy", "ingest service", "maintains")],
        remove_relations=[("Bo", "ingest service", "maintains")],
        now=T1,
    )
    assert receipt.edges_closed == 1
    assert current_relations(db) == {("Cy", "ingest service", "maintains")}


def test_correct_refuses_a_malformed_relation_before_writing_anything(db):
    old = bo_scene(db)
    for bad in ([("Cy",)], [("a", "b", "c", "d")], ["Cy"], [None]):
        with pytest.raises(ValidationError, match="add_relations"):
            db.correct(old.memory_id, "x", add_relations=bad)
    with pytest.raises(ValidationError, match="remove_relations"):
        db.correct(old.memory_id, "x", remove_relations=[("only one",)])
    assert db.get(old.memory_id).is_current
    assert current_contents(db) == {"Bo maintains the ingest service"}


def test_correct_is_also_a_module_function_and_the_receipt_is_exported(db):
    import anatid
    from anatid import verbs

    old = db.remember("Ada leads Kestrel", entities=["Ada"], now=T0)
    receipt = verbs.correct(
        db, old.memory_id, "Ada leads Kestrel", add_relations=[("Ada", "Kestrel")]
    )
    assert isinstance(receipt, anatid.CorrectionReceipt)
    assert "CorrectionReceipt" in anatid.__all__
    assert "correct" in verbs.__all__


# =========================================================================== the Agents SDK


def test_facts_by_name_alone_are_islands_until_the_agent_relates_them(db, agent_tools):
    """The review's reproduction, through the Agents SDK tools."""
    for content, entities in FACTS:
        saved = invoke(
            agent_tools["anatid_remember"], content=content, entities=entities, kind=None
        )
        assert saved["saved"] is True

    # Before: one fact, because nothing links Ada's entities to Kestrel's or the service's.
    assert reachable_from(db, "Ada") == {"Ada leads Kestrel"}
    seeded = invoke(agent_tools["anatid_recall"], query="ingest", k=10, seed_entity="Ada", hops=2)
    assert {h["content"] for h in seeded["hits"] if "graph" in h["sources"]} == {
        "Ada leads Kestrel"
    }
    around = invoke(agent_tools["anatid_context"], entity="Ada", limit=10, hops=2)
    assert {m["content"] for m in around["memories"]} == {"Ada leads Kestrel"}

    # The agent adds the two edges the world has, through the new tool.
    first = invoke(agent_tools["anatid_relate"], src="Ada", dst="Kestrel", rel_kind="leads")
    second = invoke(
        agent_tools["anatid_relate"], src="Kestrel", dst="ingest service", rel_kind="owns"
    )
    assert first["related"] is True and second["related"] is True
    assert (first["src"], first["dst"], first["rel_kind"]) == ("Ada", "Kestrel", "leads")
    assert isinstance(first["edge_id"], str) and int(first["edge_id"]) > JS_MAX_SAFE_INTEGER

    # After: all three, from the core verb and from both read tools.
    assert reachable_from(db, "Ada") == ALL_THREE
    seeded = invoke(agent_tools["anatid_recall"], query="ingest", k=10, seed_entity="Ada", hops=2)
    assert {h["content"] for h in seeded["hits"] if "graph" in h["sources"]} == ALL_THREE
    around = invoke(agent_tools["anatid_context"], entity="Ada", limit=10, hops=2)
    assert {m["content"] for m in around["memories"]} == ALL_THREE
    assert current_relations(db) == {
        ("Ada", "Kestrel", "leads"),
        ("Kestrel", "ingest service", "owns"),
    }


def test_the_graph_tools_are_approval_gated_like_every_other_write(db):
    tools = {t.name: t for t in create_memory_tools(db)}
    for name in ("anatid_relate", "anatid_unrelate", "anatid_correct"):
        assert callable(tools[name].needs_approval), f"{name} must be approval-gated"
        assert asyncio.run(tools[name].needs_approval(None, {"src": "a", "dst": "b"}, "id")) is True

    # approve_low_risk keeps all three for a human unless relate is opted in explicitly.
    cautious = approve_low_risk()
    assert cautious(ApprovalRequest("anatid_relate", {"src": "Ada", "dst": "Kestrel"}, "i")) is True
    assert (
        cautious(ApprovalRequest("anatid_unrelate", {"src": "Ada", "dst": "Kestrel"}, "i")) is True
    )
    assert cautious(ApprovalRequest("anatid_correct", {"content": "short"}, "i")) is True
    relaxed = approve_low_risk(relate=True)
    assert relaxed(ApprovalRequest("anatid_relate", {"src": "Ada", "dst": "Kestrel"}, "i")) is False
    assert (
        relaxed(ApprovalRequest("anatid_unrelate", {"src": "Ada", "dst": "Kestrel"}, "i")) is True
    )
    assert relaxed(ApprovalRequest("anatid_correct", {"content": "short"}, "i")) is True

    # exclude works on the new names as on the old ones
    names = [t.name for t in create_memory_tools(db, exclude=["anatid_unrelate", "anatid_correct"])]
    assert "anatid_unrelate" not in names and "anatid_correct" not in names
    assert "anatid_relate" in names


def test_a_scripted_run_stops_for_approval_before_relating(db):
    """End to end in the SDK's own Runner: the edge is written only once a human approves."""
    model = ScriptedModel(
        [
            [
                function_call(
                    "anatid_relate",
                    {"src": "Ada", "dst": "Kestrel", "rel_kind": "leads"},
                    call_id="call-1",
                )
            ],
            [assistant_message("Linked.")],
        ]
    )
    agent = Agent(
        name="assistant", model=model, tools=create_memory_tools(db, include=["anatid_relate"])
    )

    result = asyncio.run(Runner.run(agent, "Ada leads Kestrel, link them", run_config=NO_TRACE))
    assert [i.tool_name for i in result.interruptions] == ["anatid_relate"]
    assert current_relations(db) == set(), "the edge must not exist before approval"

    state = result.to_state()
    for item in result.interruptions:
        state.approve(item)
    final = asyncio.run(Runner.run(agent, state, run_config=NO_TRACE))
    assert final.final_output == "Linked."
    assert current_relations(db) == {("Ada", "Kestrel", "leads")}
    edges = db.execute("SELECT writer FROM edges_relates").fetchall()
    assert edges == [("agent:assistant",)], "the writer stamp names the agent"


def test_agent_correct_moves_a_fact_and_its_edges_together(db, agent_tools):
    for content, entities in FACTS:
        invoke(agent_tools["anatid_remember"], content=content, entities=entities, kind=None)
    invoke(agent_tools["anatid_relate"], src="Ada", dst="Kestrel", rel_kind="leads")
    invoke(agent_tools["anatid_relate"], src="Kestrel", dst="ingest service", rel_kind="owns")
    invoke(agent_tools["anatid_relate"], src="Bo", dst="ingest service", rel_kind="maintains")
    bo = next(m for m in db.recall_2hop("Bo") if m.content.startswith("Bo"))

    result = invoke(
        agent_tools["anatid_correct"],
        memory_id=str(bo.memory_id),
        content="Cy maintains the ingest service",
        entities=["Cy", "ingest service"],
        kind=None,
        add_relations=[{"src": "Cy", "dst": "ingest service", "rel_kind": "maintains"}],
        remove_relations=[
            {"src": "Bo", "dst": "ingest service", "rel_kind": "maintains"},
            {"src": "Bo", "dst": "Kestrel", "rel_kind": None},  # never existed
        ],
    )

    assert result["corrected"] is True
    assert result["superseded"] == str(bo.memory_id)
    assert result["content"] == "Cy maintains the ingest service"
    assert set(result["about"]) == {"Cy", "ingest service"}
    assert [(e["src"], e["dst"], e["rel_kind"]) for e in result["opened"]] == [
        ("Cy", "ingest service", "maintains")
    ]
    assert result["closed"] == [{"src": "Bo", "dst": "ingest service", "rel_kind": "maintains"}]
    assert result["not_closed"] == [{"src": "Bo", "dst": "Kestrel", "rel_kind": None}]
    assert result["edges_closed"] == 1
    assert unsafe_ints(result) == []

    assert reachable_from(db, "Ada") == {
        "Ada leads Kestrel",
        "Kestrel owns the ingest service",
        "Cy maintains the ingest service",
    }
    assert reachable_from(db, "Bo") == set(), "Bo is an island again, as the world says"
    assert db.get(bo.memory_id).is_current is False
    chain = invoke(agent_tools["anatid_provenance"], memory_id=result["memory_id"])
    assert [m["memory_id"] for m in chain["chain"]] == [result["memory_id"], str(bo.memory_id)]


def test_agent_correct_that_fails_is_an_error_payload_and_changes_nothing(db, agent_tools):
    old = bo_scene(db)
    edges, contents = current_relations(db), current_contents(db)

    result = invoke(
        agent_tools["anatid_correct"],
        memory_id=str(old.memory_id),
        content="Cy maintains the ingest service",
        entities=None,
        kind=None,
        add_relations=[{"src": "Cy", "dst": "ingest service", "rel_kind": "maintains"}],
        remove_relations=[{"src": "Nobody", "dst": "ingest service", "rel_kind": None}],
    )

    assert result["error"] == "NotFoundError" and "Nobody" in result["message"]
    assert db.get(old.memory_id).is_current
    assert current_contents(db) == contents
    assert current_relations(db) == edges
    assert db.get_entity("Cy") is None


def test_agent_unrelate_retires_the_edge_and_says_when_nothing_matched(db, agent_tools):
    bo_scene(db)
    db.relate("Bo", "ingest service", rel_kind="pages", now=T0)

    narrowed = invoke(
        agent_tools["anatid_unrelate"], src="ingest service", dst="Bo", rel_kind="pages"
    )
    assert narrowed["edges_closed"] == 1, "undirected: the reversed pair closes the edge"
    assert current_relations(db) == {("Bo", "ingest service", "maintains")}

    everything = invoke(
        agent_tools["anatid_unrelate"], src="Bo", dst="ingest service", rel_kind=None
    )
    assert everything["edges_closed"] == 1
    assert current_relations(db) == set()

    again = invoke(agent_tools["anatid_unrelate"], src="Bo", dst="ingest service", rel_kind=None)
    assert again["edges_closed"] == 0 and "nothing changed" in again["note"]

    unknown = invoke(agent_tools["anatid_unrelate"], src="Nobody", dst="Bo", rel_kind=None)
    assert unknown["error"] == "NotFoundError"

    # Closed, not deleted: a read before the close still walks the edge.
    assert {m.content for m in db.as_of(T0).recall_2hop("ingest service")} == {
        "Bo maintains the ingest service"
    }


# =========================================================================== MCP


def test_mcp_facts_by_name_alone_are_islands_until_related(server, mcp_db):
    """The review's reproduction, through the MCP tools."""
    for content, entities in FACTS:
        ok(call(server, "remember", {"content": content, "entities": entities}))

    assert reachable_from(mcp_db, "Ada") == {"Ada leads Kestrel"}
    seeded = ok(call(server, "recall", {"seed_entity": "Ada", "hops": 2, "k": 10}))
    assert {h["content"] for h in seeded["hits"]} == {"Ada leads Kestrel"}

    ok(call(server, "relate", {"src": "Ada", "dst": "Kestrel", "rel_kind": "leads"}))
    ok(call(server, "relate", {"src": "Kestrel", "dst": "ingest service", "rel_kind": "owns"}))

    assert reachable_from(mcp_db, "Ada") == ALL_THREE
    seeded = ok(call(server, "recall", {"seed_entity": "Ada", "hops": 2, "k": 10}))
    assert {h["content"] for h in seeded["hits"]} == ALL_THREE
    around = ok(call(server, "context", {"entity": "Ada", "hops": 2}))
    assert {m["content"] for m in around["memories"]} == ALL_THREE


def test_mcp_registers_unrelate_and_correct_as_destructive_with_string_ids(server):
    tools = list_tools(server)
    assert {"relate", "unrelate", "correct"} <= set(tools)
    for name in ("unrelate", "correct"):
        assert tools[name].annotations.destructive_hint is True, name
        assert tools[name].annotations.read_only_hint is False, name
        assert "one transaction" in tools["correct"].description
    assert "deletes nothing" in tools["unrelate"].description

    unrelate = tools["unrelate"].input_schema
    assert unrelate["required"] == ["src", "dst"]
    for argument in ("src", "dst"):
        assert unrelate["properties"][argument]["type"] == "string"
        assert "2**53" in unrelate["properties"][argument]["description"]

    correct = tools["correct"].input_schema
    assert correct["required"] == ["old_id", "content"]
    assert correct["properties"]["old_id"]["type"] == "string"
    assert "2**53" in correct["properties"]["old_id"]["description"]
    relation = correct["$defs"]["Relation"]
    assert set(relation["properties"]) == {"src", "dst", "rel_kind"}
    assert relation["required"] == ["src", "dst"]
    assert relation["properties"]["src"]["type"] == "string"
    assert "2**53" in relation["properties"]["src"]["description"]
    for argument in ("add_relations", "remove_relations"):
        options = correct["properties"][argument]["anyOf"]
        assert {"items": {"$ref": "#/$defs/Relation"}, "type": "array"} in options


def test_mcp_correct_moves_a_fact_and_its_edges_together(server, mcp_db):
    for content, entities in FACTS:
        written = ok(call(server, "remember", {"content": content, "entities": entities}))
    bo_id = written["memory"]["memory_id"]
    ok(call(server, "relate", {"src": "Ada", "dst": "Kestrel", "rel_kind": "leads"}))
    ok(call(server, "relate", {"src": "Kestrel", "dst": "ingest service", "rel_kind": "owns"}))
    ok(call(server, "relate", {"src": "Bo", "dst": "ingest service", "rel_kind": "maintains"}))
    service_id = next(e["entity_id"] for e in written["about"] if e["name"] == "ingest service")

    result = ok(
        call(
            server,
            "correct",
            {
                "old_id": bo_id,
                "content": "Cy maintains the ingest service",
                "entities": ["Cy", service_id],  # a name and an id string
                "writer": "handover",
                "episode": "Bo handed the pager to Cy on Monday",
                "add_relations": [{"src": "Cy", "dst": service_id, "rel_kind": "maintains"}],
                "remove_relations": [{"src": "Bo", "dst": "ingest service"}],
            },
        )
    )

    assert result["superseded"] == bo_id
    assert result["memory"]["content"] == "Cy maintains the ingest service"
    assert result["memory"]["is_current"] is True and result["memory"]["writer"] == "handover"
    assert result["old"]["memory_id"] == bo_id and result["old"]["is_current"] is False
    assert len(result["opened"]) == 1
    assert result["opened"][0]["rel_kind"] == "maintains"
    assert result["opened"][0]["dst"] == service_id
    # closed relations come back as they were passed: the name here, not the id
    assert result["closed"] == [{"src": "Bo", "dst": "ingest service", "rel_kind": None}]
    assert result["edges_closed"] == 1
    assert unsafe_ints(result) == []

    assert reachable_from(mcp_db, "Ada") == {
        "Ada leads Kestrel",
        "Kestrel owns the ingest service",
        "Cy maintains the ingest service",
    }
    assert current_relations(mcp_db) == {
        ("Ada", "Kestrel", "leads"),
        ("Kestrel", "ingest service", "owns"),
        ("Cy", "ingest service", "maintains"),
    }
    prov = ok(call(server, "provenance", {"memory_id": result["memory"]["memory_id"]}))
    assert [m["memory_id"] for m in prov["chain"]] == [result["memory"]["memory_id"], bo_id]
    assert prov["source_text"] == "Bo handed the pager to Cy on Monday"


def test_mcp_correct_that_fails_is_a_tool_error_and_changes_nothing(server, mcp_db):
    old = bo_scene(mcp_db)
    edges, contents = current_relations(mcp_db), current_contents(mcp_db)

    failed = call(
        server,
        "correct",
        {
            "old_id": str(old.memory_id),
            "content": "Cy maintains the ingest service",
            "add_relations": [{"src": "Cy", "dst": "ingest service", "rel_kind": "maintains"}],
            "remove_relations": [{"src": "Nobody", "dst": "ingest service"}],
        },
    )
    assert failed.is_error is True
    assert "Nobody" in error_text(failed)

    assert mcp_db.get(old.memory_id).is_current
    assert current_contents(mcp_db) == contents
    assert current_relations(mcp_db) == edges
    assert mcp_db.get_entity("Cy") is None

    # An unknown memory is the same kind of readable error, before anything is written.
    missing = call(server, "correct", {"old_id": "12345", "content": "x"})
    assert missing.is_error is True and "12345" in error_text(missing)


def test_mcp_unrelate_closes_edges_and_reports_zero_when_nothing_is_current(server, mcp_db):
    bo_scene(mcp_db)
    mcp_db.relate("Bo", "ingest service", rel_kind="pages", now=T0)

    narrowed = ok(
        call(server, "unrelate", {"src": "ingest service", "dst": "Bo", "rel_kind": "pages"})
    )
    assert narrowed["edges_closed"] == 1
    assert current_relations(mcp_db) == {("Bo", "ingest service", "maintains")}

    rest = ok(call(server, "unrelate", {"src": "Bo", "dst": "ingest service"}))
    assert rest["edges_closed"] == 1 and rest["rel_kind"] is None
    assert current_relations(mcp_db) == set()

    assert ok(call(server, "unrelate", {"src": "Bo", "dst": "ingest service"}))["edges_closed"] == 0
    unknown = call(server, "unrelate", {"src": "Nobody", "dst": "Bo"})
    assert unknown.is_error is True and "Nobody" in error_text(unknown)

    # closed, not deleted
    assert mcp_db.execute("SELECT count(*) FROM edges_relates").fetchone()[0] == 4
    assert {m.content for m in mcp_db.as_of(T0).recall_2hop("ingest service")} == {
        "Bo maintains the ingest service"
    }


def test_mcp_read_only_server_has_no_graph_write_tools(tmp_path):
    path = tmp_path / "ro.anatid"
    with Anatid.open(path, tenant=3, embedding_dim=DIM) as writable:
        writable.remember("written before", entities=["thing"])
    with Anatid.open(path, tenant=3, embedding_dim=DIM, read_only=True) as ro:
        names = set(
            list_tools(build_server(ro, ServerConfig(db=str(path), read_only=True, env={})))
        )
    assert names.isdisjoint({"relate", "unrelate", "correct", "supersede", "remember"})
    assert {"recall", "context"} <= names


class _WithoutCorrect:
    """A backend that answers every verb but ``correct``, the way a client of an
    ``anatid-server`` that does not carry the verb yet does."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        if name == "correct":
            raise AttributeError(name)
        return getattr(self._inner, name)


def test_mcp_registers_correct_only_when_the_backend_has_the_verb(mcp_db):
    partial = build_server(_WithoutCorrect(mcp_db), ServerConfig(db=mcp_db.path, tenant=3, env={}))
    names = set(list_tools(partial))
    assert "correct" not in names
    assert {"relate", "unrelate", "supersede"} <= names
    full = build_server(mcp_db, ServerConfig(db=mcp_db.path, tenant=3, env={}))
    assert "correct" in set(list_tools(full))


# =========================================================================== ids through node


def test_every_new_tool_result_carries_no_integer_a_json_number_would_round(
    db, agent_tools, server, mcp_db
):
    old = bo_scene(db)
    payloads = [
        invoke(agent_tools["anatid_relate"], src="Ada", dst="Kestrel", rel_kind="leads"),
        invoke(
            agent_tools["anatid_correct"],
            memory_id=str(old.memory_id),
            content="Cy maintains the ingest service",
            entities=None,
            kind=None,
            add_relations=[{"src": "Cy", "dst": "ingest service", "rel_kind": "maintains"}],
            remove_relations=[{"src": "Bo", "dst": "ingest service", "rel_kind": None}],
        ),
        invoke(agent_tools["anatid_unrelate"], src="Ada", dst="Kestrel", rel_kind=None),
    ]
    mcp_old = bo_scene(mcp_db)
    payloads += [
        call(server, "relate", {"src": "Ada", "dst": "Kestrel"}).model_dump(mode="json"),
        call(
            server,
            "correct",
            {
                "old_id": str(mcp_old.memory_id),
                "content": "Cy maintains the ingest service",
                "add_relations": [{"src": "Cy", "dst": "ingest service", "rel_kind": "maintains"}],
                "remove_relations": [{"src": "Bo", "dst": "ingest service"}],
            },
        ).model_dump(mode="json"),
        call(server, "unrelate", {"src": "Ada", "dst": "Kestrel"}).model_dump(mode="json"),
    ]
    for payload in payloads:
        assert unsafe_ints(payload) == [], payload


@requires_node
def test_agent_graph_tool_ids_survive_node_and_still_address_the_rows(db, agent_tools):
    old = bo_scene(db)
    related = invoke(agent_tools["anatid_relate"], src="Ada", dst="Kestrel", rel_kind="leads")
    corrected = invoke(
        agent_tools["anatid_correct"],
        memory_id=str(old.memory_id),
        content="Cy maintains the ingest service",
        entities=None,
        kind=None,
        add_relations=[{"src": "Cy", "dst": "ingest service", "rel_kind": "maintains"}],
        remove_relations=[{"src": "Bo", "dst": "ingest service", "rel_kind": "maintains"}],
    )

    for payload in (related, corrected):
        back = through_node(json.dumps(payload))
        assert json.loads(back) == payload, "a JavaScript JSON parser changed the payload"

    # The ids node handed back are the ids the database answers to.
    edge_id = json.loads(through_node(json.dumps(related)))["edge_id"]
    assert f'"{edge_id}"' in through_node(json.dumps(related)), "not byte for byte"
    assert int(edge_id) > JS_MAX_SAFE_INTEGER
    assert db.execute(
        "SELECT rel_kind FROM edges_relates WHERE edge_id = ?", [int(edge_id)]
    ).fetchone() == ("leads",)

    returned = json.loads(through_node(json.dumps(corrected)))
    chain = invoke(agent_tools["anatid_provenance"], memory_id=returned["memory_id"])
    assert [m["memory_id"] for m in chain["chain"]] == [
        returned["memory_id"],
        returned["superseded"],
    ]
    assert db.get(int(returned["superseded"])).is_current is False
    opened = int(returned["opened"][0]["edge_id"])
    assert db.execute(
        "SELECT rel_kind FROM edges_relates WHERE edge_id = ?", [opened]
    ).fetchone() == ("maintains",)
    assert int(returned["opened"][0]["src_entity_id"]) == db.entity_id("Cy")


@requires_node
def test_mcp_graph_tool_ids_survive_node_and_still_address_the_rows(server, mcp_db):
    old = bo_scene(mcp_db)
    related = ok(call(server, "relate", {"src": "Ada", "dst": "Kestrel", "rel_kind": "leads"}))
    corrected = ok(
        call(
            server,
            "correct",
            {
                "old_id": str(old.memory_id),
                "content": "Cy maintains the ingest service",
                "entities": ["Cy", "ingest service"],
                "add_relations": [{"src": "Cy", "dst": "ingest service", "rel_kind": "maintains"}],
                "remove_relations": [{"src": "Bo", "dst": "ingest service"}],
            },
        )
    )
    unrelated = ok(call(server, "unrelate", {"src": "Ada", "dst": "Kestrel"}))

    for payload in (related, corrected, unrelated):
        assert json.loads(through_node(json.dumps(payload))) == payload

    returned = json.loads(through_node(json.dumps(corrected)))
    assert f'"{returned["memory"]["memory_id"]}"' in through_node(json.dumps(corrected))
    fetched = ok(call(server, "get", {"memory_id": returned["memory"]["memory_id"]}))
    assert fetched["memory"]["content"] == "Cy maintains the ingest service"
    assert (
        ok(call(server, "get", {"memory_id": returned["superseded"]}))["memory"]["is_current"]
        is False
    )
    opened = returned["opened"][0]
    assert mcp_db.execute(
        "SELECT rel_kind FROM edges_relates WHERE edge_id = ?", [int(opened["edge_id"])]
    ).fetchone() == ("maintains",)
    assert int(opened["src"]) == mcp_db.entity_id("Cy")
    # and an id that came back from node is accepted as an argument, string or int
    by_string = ok(call(server, "context", {"entity": opened["src"]}))
    by_int = ok(call(server, "context", {"entity": int(opened["src"])}))
    assert [m["memory_id"] for m in by_string["memories"]] == [
        m["memory_id"] for m in by_int["memories"]
    ]
    assert by_string["memories"][0]["content"] == "Cy maintains the ingest service"
