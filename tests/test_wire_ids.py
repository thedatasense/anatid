"""Ids on the wire: what the MCP server and the Agents SDK tools actually put in JSON.

anatid mints 63-bit ids.  A JSON *number* is a double in JavaScript, whose largest exact integer
is ``2**53 - 1``, so an id sent as a number comes back rounded and nothing anywhere raises --
``883768514279557120`` becomes ``883768514279557100``, which is not a row.  Every id at an
integration boundary is therefore a decimal string, and every id a tool accepts takes a decimal
string or an integer (:mod:`anatid.integrations.wire`).

Three things are checked here, and the first is what makes the other two mean something:

* :func:`test_node_rounds_a_63_bit_json_number` is the **negative control**.  It runs the real
  ``node`` on the id-as-a-number form and shows it corrupted.  Without it the round-trip tests
  below could pass because node is lenient rather than because anatid is correct.
* every response of every tool -- both boundaries, success and failure -- is walked, and no value
  anywhere in the tree may be an integer above ``2**53``.  This is a sweep over the whole tree
  rather than an assertion about the four fields somebody thought of.
* the same responses are parsed and re-serialised by a real ``node`` subprocess, and must come
  back byte for byte.

The node half skips with a reason when node is not installed; the sweep and the schema checks do
not need it and always run.

``is_loopback_host`` is here too, because it is the other thing a client's configuration hands
the server as text: "LOCALHOST" and "localhost" have to be one question.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import socket
import subprocess

import pytest

from anatid import Anatid
from anatid.ids import new_id
from anatid.integrations.wire import (
    JS_MAX_SAFE_INTEGER,
    WireEntityRef,
    WireId,
    coerce_entity_ref,
    coerce_id,
    is_wire_safe_int,
    wire_id,
    wire_ids,
    wire_unsafe_ints,
)

mcp_client = pytest.importorskip("mcp.client", reason="the MCP server needs 'mcp>=2.1'")

from mcp.client import Client  # noqa: E402

from anatid.integrations.mcp.server import (  # noqa: E402
    InsecureTransport,
    ServerConfig,
    build_server,
    check_transport_security,
    is_loopback_host,
    normalise_host,
)
from anatid.integrations.openai_agents import TOOL_NAMES, create_memory_tools  # noqa: E402

DIM = 8

NODE = shutil.which("node")
requires_node = pytest.mark.skipif(
    NODE is None,
    reason=(
        "node is not on PATH; the JSON round trip that proves a 63-bit id survives needs a real "
        "JavaScript parser, and a Python one would prove nothing (Python has no 2**53 limit)"
    ),
)

HAVE_AGENTS = importlib.util.find_spec("agents") is not None
requires_agents = pytest.mark.skipif(
    not HAVE_AGENTS, reason='the Agents SDK tools need pip install "anatid[agents]"'
)

#: Parse stdin as JSON and write it straight back out.  Any integer JavaScript cannot hold
#: exactly is silently rounded in between, which is the whole point.
NODE_ROUND_TRIP = """
let raw = "";
process.stdin.on("data", (chunk) => { raw += chunk; });
process.stdin.on("end", () => {
  process.stdout.write(JSON.stringify(JSON.parse(raw)));
});
"""


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


def unsafe_ints(value, path: str = "$") -> list[tuple[str, int]]:
    """Every ``(json path, value)`` in ``value`` that a JSON number could not carry exactly."""
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [] if is_wire_safe_int(value) else [(path, value)]
    if isinstance(value, dict):
        found: list[tuple[str, int]] = []
        for key, item in value.items():
            found.extend(unsafe_ints(item, f"{path}.{key}"))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for index, item in enumerate(value):
            found.extend(unsafe_ints(item, f"{path}[{index}]"))
        return found
    return []


def assert_wire_safe(label: str, payload) -> None:
    """Fail with the path of every id that went out as a JSON number."""
    offenders = unsafe_ints(payload, f"{label}")
    assert not offenders, (
        f"{label} returned {len(offenders)} integer(s) a JSON number cannot carry exactly; "
        f"each of these is rounded by any JavaScript client: "
        + ", ".join(f"{path} = {value}" for path, value in offenders)
    )


# --------------------------------------------------------------------------- the wire contract


def test_ids_are_bigger_than_a_json_number_can_hold():
    """The premise.  If this ever fails, the rest of this file is about nothing."""
    minted = new_id()
    assert minted > JS_MAX_SAFE_INTEGER, minted
    assert JS_MAX_SAFE_INTEGER == 2**53 - 1


@requires_node
def test_node_rounds_a_63_bit_json_number():
    """The negative control: the corruption this whole file exists to prevent, reproduced."""
    memory_id = 883768514279557120
    assert memory_id > JS_MAX_SAFE_INTEGER

    as_number = through_node(json.dumps({"memory_id": memory_id}))
    assert json.loads(as_number)["memory_id"] == 883768514279557100
    assert json.loads(as_number)["memory_id"] != memory_id

    as_string = through_node(json.dumps({"memory_id": str(memory_id)}))
    assert json.loads(as_string)["memory_id"] == "883768514279557120"


def test_encoders_and_decoders_are_inverses():
    memory_id = new_id()
    assert wire_id(memory_id) == str(memory_id)
    assert coerce_id(wire_id(memory_id)) == memory_id
    assert wire_id(None) is None
    assert wire_ids([1, 2, 3]) == ["1", "2", "3"]


def test_a_decoder_takes_a_string_or_an_int():
    assert coerce_id("883768514279557120") == 883768514279557120
    assert coerce_id(883768514279557120) == 883768514279557120
    assert coerce_id("  883768514279557120  ") == 883768514279557120
    assert coerce_id("-7") == -7
    assert coerce_entity_ref(883768514279557120) == "883768514279557120"
    assert coerce_entity_ref("Ada Lovelace") == "Ada Lovelace"


@pytest.mark.parametrize("bad", ["", "abc", "12.0", "0x10", None, 12.0, True, [1]])
def test_a_decoder_refuses_what_is_not_an_id(bad):
    with pytest.raises(ValueError):
        coerce_id(bad)


def test_the_unknown_shape_sweep_only_touches_what_it_must():
    payload = {
        "memory_id": 883768514279557120,
        "count": 3,
        "flag": True,
        "score": 0.5,
        "nested": [{"entity_id": -883768514279557120}, {"rank": 1}],
        "text": "883768514279557120",
    }
    assert wire_unsafe_ints(payload) == {
        "memory_id": "883768514279557120",
        "count": 3,
        "flag": True,
        "score": 0.5,
        "nested": [{"entity_id": "-883768514279557120"}, {"rank": 1}],
        "text": "883768514279557120",
    }


def test_wire_annotations_subclass_what_python_sees():
    # The annotations exist so a checker still sees an int (or a str) where the argument is
    # handed on to an anatid verb.
    assert issubclass(WireId, int)
    assert issubclass(WireEntityRef, str)


# --------------------------------------------------------------------------- the MCP boundary


@pytest.fixture
def mcp_db(tmp_path):
    with Anatid.open(tmp_path / "wire.anatid", tenant=3, embedding_dim=DIM) as handle:
        yield handle


@pytest.fixture
def server(mcp_db):
    return build_server(mcp_db, ServerConfig(db=mcp_db.path, tenant=3, sql_tool=True, env={}))


def call(server, name: str, arguments: dict | None = None):
    async def body(client):
        async with Client(server) as c:
            return await c.call_tool(name, arguments or {})

    return asyncio.run(body(server))


def ok(result):
    assert result.is_error is False, " ".join(getattr(c, "text", "") for c in result.content)
    assert result.structured_content is not None
    return result.structured_content


def every_mcp_response(server) -> list[tuple[str, str, object]]:
    """Call every tool the server registers; return ``(label, tool, whole CallToolResult)``.

    The whole result, not just its structured content: the failure cases carry their payload in
    the content blocks instead, and an id must not be a JSON number there either.

    :func:`test_every_mcp_tool_is_covered_by_the_sweep` holds this list to the tools the server
    actually registers, so a tool added later cannot quietly go unswept.
    """
    written = ok(
        call(
            server,
            "remember",
            {
                "content": "Ada Lovelace wrote the first published algorithm.",
                "entities": ["Ada Lovelace", "Analytical Engine"],
                "episode": "From the notes on the Analytical Engine.",
            },
        )
    )
    memory_id = written["memory"]["memory_id"]
    entity_id = written["about"][0]["entity_id"]
    replaced = ok(
        call(server, "supersede", {"old_id": memory_id, "content": "Ada Lovelace wrote note G."})
    )
    new_memory_id = replaced["memory"]["memory_id"]
    doomed = ok(call(server, "remember", {"content": "to be forgotten"}))["memory"]["memory_id"]

    calls = [
        ("remember", "remember", {"content": "a second fact", "entities": [entity_id]}),
        ("relate", "relate", {"src": "Ada Lovelace", "dst": entity_id, "rel_kind": "built"}),
        ("supersede", "supersede", {"old_id": new_memory_id, "content": "note G, again"}),
        ("reinforce", "reinforce", {"memory_id": memory_id}),
        ("forget", "forget", {"memory_id": doomed, "reason": "test"}),
        ("prune", "prune", {"max_access_count": 0}),
        ("rebuild_fts_index", "rebuild_fts_index", {}),
        ("recall", "recall", {"query": "algorithm", "k": 5, "seed_entity": entity_id}),
        ("context", "context", {"entity": entity_id, "hops": 1}),
        ("get", "get", {"memory_id": memory_id}),
        ("provenance", "provenance", {"memory_id": memory_id}),
        ("stats", "stats", {}),
        ("sql", "sql", {"query": "SELECT memory_id, tenant_id, content FROM memories"}),
        ("sql_counts", "sql", {"query": "SELECT count(*) AS n FROM memories"}),
        # Failures: an unknown id, and an id that is not one.
        ("get_missing", "get", {"memory_id": str(new_id())}),
        ("reinforce_missing", "reinforce", {"memory_id": str(new_id())}),
        ("provenance_missing", "provenance", {"memory_id": str(new_id())}),
        ("forget_missing", "forget", {"memory_id": str(new_id())}),
        ("get_not_an_id", "get", {"memory_id": "not-an-id"}),
    ]
    responses = [
        ("remember(first)", "remember", written),
        ("supersede(first)", "supersede", replaced),
    ]
    responses.extend(
        (label, tool, call(server, tool, args).model_dump(mode="json"))
        for label, tool, args in calls
    )
    return responses


def test_a_real_memory_gets_an_id_no_json_number_could_carry(server):
    written = ok(call(server, "remember", {"content": "Ada wrote note G", "entities": ["Ada"]}))
    memory_id = written["memory"]["memory_id"]
    assert isinstance(memory_id, str), "an id leaves as a decimal string"
    assert int(memory_id) > JS_MAX_SAFE_INTEGER, "this id would have fitted in a JSON number"
    assert memory_id == str(int(memory_id)), "a decimal string, not hex and not padded"


def test_no_mcp_tool_returns_an_int_a_json_number_would_round(server):
    for label, _tool, payload in every_mcp_response(server):
        assert_wire_safe(f"mcp {label}", payload)


def test_every_mcp_tool_is_covered_by_the_sweep(server):
    async def go():
        async with Client(server) as client:
            return await client.list_tools()

    registered = {t.name for t in asyncio.run(go()).tools}
    swept = {tool for _label, tool, _payload in every_mcp_response(server)}
    assert swept == registered, "a tool nobody sweeps is a tool that can leak an id"


@requires_node
def test_every_mcp_response_survives_a_real_javascript_parser(server):
    for label, _tool, payload in every_mcp_response(server):
        sent = json.dumps(payload)
        back = through_node(sent)
        assert json.loads(back) == payload, f"mcp {label} changed in a JavaScript JSON parser"


@requires_node
def test_an_mcp_id_survives_node_byte_for_byte_and_still_addresses_the_row(server, mcp_db):
    written = ok(call(server, "remember", {"content": "Ada wrote note G", "entities": ["Ada"]}))
    memory_id = written["memory"]["memory_id"]

    returned = json.loads(through_node(json.dumps(written)))["memory"]["memory_id"]
    assert returned == memory_id, "the id changed on the way through node"
    assert f'"{memory_id}"' in through_node(json.dumps(written)), "not byte for byte"

    # And the id node handed back is still the id the server answers to.
    fetched = ok(call(server, "get", {"memory_id": returned}))
    assert fetched["memory"]["content"] == "Ada wrote note G"
    assert mcp_db.get(int(returned)) is not None


def test_mcp_id_arguments_take_a_string_or_an_int(server):
    memory_id = ok(call(server, "remember", {"content": "either form"}))["memory"]["memory_id"]

    as_string = ok(call(server, "get", {"memory_id": memory_id}))
    as_int = ok(call(server, "get", {"memory_id": int(memory_id)}))
    assert as_string == as_int
    assert as_string["memory"]["content"] == "either form"


def test_mcp_entity_arguments_take_a_name_an_id_string_or_an_id_int(server):
    written = ok(call(server, "remember", {"content": "about Ada", "entities": ["Ada"]}))
    entity_id = written["about"][0]["entity_id"]

    by_name = ok(call(server, "context", {"entity": "Ada"}))
    by_id_string = ok(call(server, "context", {"entity": entity_id}))
    by_id_int = ok(call(server, "context", {"entity": int(entity_id)}))
    assert [m["memory_id"] for m in by_name["memories"]] == [
        m["memory_id"] for m in by_id_string["memories"]
    ]
    assert [m["memory_id"] for m in by_id_int["memories"]] == [
        m["memory_id"] for m in by_name["memories"]
    ]


def test_an_mcp_id_argument_that_is_not_an_id_is_a_readable_error(server):
    bad = call(server, "get", {"memory_id": "not-an-id"})
    assert bad.is_error is True
    text = " ".join(getattr(c, "text", "") for c in bad.content)
    assert "decimal string" in text, text


ID_ARGUMENTS = [
    ("supersede", "old_id"),
    ("reinforce", "memory_id"),
    ("forget", "memory_id"),
    ("get", "memory_id"),
    ("provenance", "memory_id"),
]
ENTITY_ARGUMENTS = [
    ("relate", "src"),
    ("relate", "dst"),
    ("context", "entity"),
    ("recall", "seed_entity"),
]


def test_mcp_tool_schemas_declare_ids_as_strings_and_say_why(server):
    async def body(client):
        return await client.list_tools()

    async def go():
        async with Client(server) as client:
            return await body(client)

    listed = {t.name: t.input_schema for t in asyncio.run(go()).tools}

    for tool, argument in ID_ARGUMENTS:
        prop = listed[tool]["properties"][argument]
        assert prop["type"] == "string", f"{tool}.{argument} is declared {prop}"
        assert "2**53" in prop["description"], f"{tool}.{argument} does not say why"

    for tool, argument in ENTITY_ARGUMENTS:
        prop = listed[tool]["properties"][argument]
        types = [prop["type"]] if "type" in prop else [o.get("type") for o in prop["anyOf"]]
        assert "string" in types, f"{tool}.{argument} is declared {prop}"
        description = prop.get("description") or prop["anyOf"][0]["description"]
        assert "2**53" in description, f"{tool}.{argument} does not say why"

    items = listed["remember"]["properties"]["entities"]
    assert "2**53" in json.dumps(items), items


def test_the_sql_escape_hatch_hands_back_ids_that_still_address_rows(server, mcp_db):
    written = ok(call(server, "remember", {"content": "found by sql"}))
    memory_id = written["memory"]["memory_id"]

    rows = ok(call(server, "sql", {"query": "SELECT memory_id, content FROM memories"}))["rows"]
    assert rows == [[memory_id, "found by sql"]], (
        "an id SELECTed through the escape hatch must be the same decimal string the verbs use"
    )

    # A count is small, so it stays a number: only what could not survive changes type.
    counted = ok(call(server, "sql", {"query": "SELECT count(*) AS n FROM memories"}))
    assert counted["rows"] == [[1]]
    assert ok(call(server, "get", {"memory_id": rows[0][0]}))["memory"] is not None


# ------------------------------------------------------------------ the Agents SDK boundary


@pytest.fixture
def agents_db(tmp_path):
    with Anatid.open(tmp_path / "agents.anatid", tenant=1, embedding_dim=DIM) as handle:
        yield handle


def invoke(tool, **arguments):
    from agents.tool_context import ToolContext

    payload = json.dumps(arguments)
    ctx = ToolContext(None, tool_name=tool.name, tool_call_id="call-1", tool_arguments=payload)
    return json.loads(asyncio.run(tool.on_invoke_tool(ctx, payload)))


def every_agents_response(db) -> list[tuple[str, str, dict]]:
    """Every tool in :data:`TOOL_NAMES`, with the label the failure message uses."""
    tools = {t.name: t for t in create_memory_tools(db)}
    saved = invoke(
        tools["anatid_remember"], content="Ada prefers DuckDB", entities=["Ada"], kind="preference"
    )
    memory_id = saved["memory_id"]
    corrected = invoke(
        tools["anatid_supersede"],
        memory_id=memory_id,
        content="Ada prefers DuckDB over SQLite",
        entities=None,
        kind=None,
    )
    doomed = invoke(tools["anatid_remember"], content="to be forgotten", entities=None, kind=None)

    return [
        ("remember", "anatid_remember", saved),
        ("supersede", "anatid_supersede", corrected),
        (
            "recall",
            "anatid_recall",
            invoke(tools["anatid_recall"], query="DuckDB", k=5, seed_entity="Ada", hops=2),
        ),
        (
            "context",
            "anatid_context",
            invoke(tools["anatid_context"], entity="Ada", limit=5, hops=0),
        ),
        (
            "provenance",
            "anatid_provenance",
            invoke(tools["anatid_provenance"], memory_id=corrected["memory_id"]),
        ),
        (
            "forget",
            "anatid_forget",
            invoke(tools["anatid_forget"], memory_id=doomed["memory_id"], hard=True, reason="test"),
        ),
        (
            "forget(missing)",
            "anatid_forget",
            invoke(tools["anatid_forget"], memory_id=str(new_id()), hard=None, reason=None),
        ),
        (
            "provenance(missing)",
            "anatid_provenance",
            invoke(tools["anatid_provenance"], memory_id=str(new_id())),
        ),
    ]


@requires_agents
def test_no_agents_tool_returns_an_int_a_json_number_would_round(agents_db):
    for label, _tool, payload in every_agents_response(agents_db):
        assert_wire_safe(f"agents {label}", payload)


@requires_agents
def test_every_agents_tool_is_covered_by_the_sweep(agents_db):
    swept = {tool for _label, tool, _payload in every_agents_response(agents_db)}
    assert swept == set(TOOL_NAMES), "a tool nobody sweeps is a tool that can leak an id"


@requires_agents
def test_an_agents_tool_writes_an_id_no_json_number_could_carry(agents_db):
    tools = {t.name: t for t in create_memory_tools(agents_db)}
    saved = invoke(
        tools["anatid_remember"], content="Ada prefers DuckDB", entities=["Ada"], kind=None
    )
    assert isinstance(saved["memory_id"], str)
    assert int(saved["memory_id"]) > JS_MAX_SAFE_INTEGER
    assert agents_db.get(int(saved["memory_id"])) is not None


@requires_agents
@requires_node
def test_every_agents_response_survives_a_real_javascript_parser(agents_db):
    for label, _tool, payload in every_agents_response(agents_db):
        sent = json.dumps(payload)
        back = through_node(sent)
        assert json.loads(back) == payload, f"agents {label} changed in a JavaScript parser"
        if "memory_id" in payload:
            assert f'"{payload["memory_id"]}"' in back, f"agents {label} id is not byte for byte"


@requires_agents
@requires_node
def test_an_agents_id_survives_node_and_still_addresses_the_row(agents_db):
    tools = {t.name: t for t in create_memory_tools(agents_db)}
    saved = invoke(tools["anatid_remember"], content="round trip me", entities=None, kind=None)

    returned = json.loads(through_node(json.dumps(saved)))["memory_id"]
    assert returned == saved["memory_id"]

    chain = invoke(tools["anatid_provenance"], memory_id=returned)
    assert chain["memory_id"] == returned
    assert [m["content"] for m in chain["chain"]] == ["round trip me"]


@requires_agents
def test_agents_id_arguments_take_a_string_or_an_int(agents_db):
    tools = {t.name: t for t in create_memory_tools(agents_db)}
    saved = invoke(tools["anatid_remember"], content="either form", entities=None, kind=None)

    as_string = invoke(tools["anatid_provenance"], memory_id=saved["memory_id"])
    as_int = invoke(tools["anatid_provenance"], memory_id=int(saved["memory_id"]))
    assert as_string == as_int
    assert as_string["memory_id"] == saved["memory_id"]


@requires_agents
def test_agents_tool_schemas_declare_ids_as_strings_and_say_why(agents_db):
    tools = {t.name: t for t in create_memory_tools(agents_db)}
    for name in ("anatid_supersede", "anatid_forget", "anatid_provenance"):
        prop = tools[name].params_json_schema["properties"]["memory_id"]
        assert prop["type"] == "string", f"{name}.memory_id is declared {prop}"
        assert "2**53" in prop["description"], f"{name}.memory_id does not say why"


# --------------------------------------------------------------------------- loopback hosts


@pytest.mark.parametrize(
    "host,loopback",
    [
        ("localhost", True),
        ("LOCALHOST", True),
        ("Localhost", True),
        ("localhost.", True),
        ("LOCALHOST.", True),
        ("  localhost  ", True),
        ("127.0.0.1", True),
        ("127.0.0.1.", True),
        ("127.0.0.53", True),  # 127.0.0.0/8 is loopback, not just .1
        ("127.255.255.254", True),
        ("::1", True),
        ("[::1]", True),
        ("[::1].", True),
        ("::1%lo0", True),
        ("0.0.0.0", False),
        ("::", False),
        ("[::]", False),
        ("", False),
        ("   ", False),
        ("*", False),
        ("192.0.2.7", False),  # TEST-NET-1, RFC 5737
        ("example.com", False),
        (None, False),
    ],
)
def test_is_loopback_host(host, loopback):
    assert is_loopback_host(host) is loopback


def test_normalise_host_reduces_the_spellings_of_one_address():
    assert normalise_host("LOCALHOST.") == "localhost"
    assert normalise_host(" [::1]. ") == "::1"
    assert normalise_host("Example.COM") == "example.com"
    assert normalise_host(".") == "."


def test_a_case_folding_resolver_is_not_what_makes_uppercase_work(monkeypatch):
    """The reported bug: "localhost" passed and "LOCALHOST" failed.

    glibc folds case when it reads /etc/hosts; musl (Alpine) does not, and neither does a DNS
    server asked for a name that only /etc/hosts has.  So the old code, which handed the name
    straight to ``getaddrinfo``, gave a different answer on Debian and on Alpine for the same
    configuration.  This pins the case fold to anatid by giving it a resolver that refuses
    anything but exact lower case.
    """
    real = socket.getaddrinfo

    def case_sensitive(host, *args, **kwargs):
        if host != host.lower():
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", case_sensitive)

    assert is_loopback_host("localhost") is True
    assert is_loopback_host("LOCALHOST") is True
    assert is_loopback_host("Localhost") is True
    assert is_loopback_host("LocalHost.") is True


def test_a_name_that_also_resolves_off_this_machine_is_not_loopback(monkeypatch):
    """Normalising case must not turn the resolver check into a rubber stamp."""

    def treacherous(host, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.7", 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", treacherous)
    assert is_loopback_host("localhost") is False
    assert is_loopback_host("LOCALHOST") is False


def test_an_uppercase_loopback_host_is_allowed_to_serve():
    for host in ("LOCALHOST", "Localhost", "localhost.", "[::1]", "127.0.0.1"):
        cfg = ServerConfig(transport="streamable-http", host=host, env={})
        check_transport_security(cfg)  # must not raise

    public = ServerConfig(transport="streamable-http", host="0.0.0.0", env={})
    with pytest.raises(InsecureTransport):
        check_transport_security(public)
