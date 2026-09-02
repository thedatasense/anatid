"""Tests for anatid's MCP server (``anatid.integrations.mcp``).

Two ways of driving it, both real:

* **In-process** -- ``mcp.client.Client(server)`` runs the actual
  :class:`~mcp.server.mcpserver.MCPServer` and dispatches real ``tools/list`` and ``tools/call``
  requests against it.  Nothing is stubbed; the tool bodies write to a real DuckDB file.
* **Over stdio** -- one test launches ``python -m anatid.integrations.mcp.server`` as a
  subprocess and talks JSON-RPC over its pipes, which is exactly how Claude Desktop / Claude
  Code / Cursor run it.  That test also proves the ``ANATID_DB`` / ``ANATID_TENANT``
  environment contract, since a client config block has no other way to configure the server.

``anyio``'s pytest plugin is not depended on: each test drives the async client through
:func:`drive`, which is a plain ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import duckdb
import pytest

mcp_client = pytest.importorskip("mcp.client", reason="the MCP server needs 'mcp>=2.1'")

from mcp import StdioServerParameters                       # noqa: E402
from mcp.client import Client                               # noqa: E402

from anatid import Anatid                                   # noqa: E402
from anatid.integrations.mcp.server import ServerConfig, build_server   # noqa: E402
from anatid.integrations.mcp.sqlgate import SqlGateway, SqlNotAllowed   # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DIM = 8


# --------------------------------------------------------------------------- harness


def drive(server, body):
    """Run ``body(client)`` against ``server`` over a real MCP session, and return its result."""

    async def go():
        async with Client(server) as client:
            return await body(client)

    return asyncio.run(go())


def call(server, name: str, arguments: dict | None = None):
    """One ``tools/call`` round trip.  Returns the ``CallToolResult`` (errors included)."""

    async def body(client):
        return await client.call_tool(name, arguments or {})

    return drive(server, body)


def ok(result):
    """Assert the call succeeded and return its structured content."""
    assert result.is_error is False, _text(result)
    assert result.structured_content is not None, "tool returned no structured content"
    return result.structured_content


def _text(result) -> str:
    return " ".join(getattr(c, "text", "") for c in result.content)


@pytest.fixture
def mcp_db(tmp_path):
    """A file-backed anatid database for the MCP tests (tenant 3, FLOAT[8])."""
    with Anatid.open(tmp_path / "mcp.anatid", tenant=3, embedding_dim=DIM) as handle:
        yield handle


@pytest.fixture
def server(mcp_db):
    """The MCP server under test, with the environment ignored so tests are hermetic."""
    return build_server(mcp_db, ServerConfig(db=mcp_db.path, tenant=3, env={}))


# --------------------------------------------------------------------------- tools/list


EXPECTED_TOOLS = {
    "remember", "relate", "supersede", "reinforce", "forget", "prune", "rebuild_fts_index",
    "recall", "context", "get", "provenance", "stats", "sql",
}
READ_ONLY_TOOLS = {"recall", "context", "get", "provenance", "stats", "sql"}
DESTRUCTIVE_TOOLS = {"forget", "prune"}


def test_lists_tools_with_schemas_and_annotations(server):
    listed = drive(server, lambda c: c.list_tools())
    by_name = {t.name: t for t in listed.tools}
    assert set(by_name) == EXPECTED_TOOLS

    for name, tool in by_name.items():
        assert tool.description, f"{name} has no description"
        assert tool.input_schema.get("type") == "object", name
        assert tool.annotations is not None, f"{name} has no annotations"

    for name in READ_ONLY_TOOLS:
        assert by_name[name].annotations.read_only_hint is True, name
        assert by_name[name].annotations.destructive_hint is False, name

    # The brief's requirement: a client must be able to prompt before anything is removed.
    for name in DESTRUCTIVE_TOOLS:
        assert by_name[name].annotations.destructive_hint is True, name
        assert by_name[name].annotations.read_only_hint is False, name

    for name in EXPECTED_TOOLS - READ_ONLY_TOOLS - DESTRUCTIVE_TOOLS:
        assert by_name[name].annotations.destructive_hint is False, name
        assert by_name[name].annotations.read_only_hint is False, name

    # remember's schema must actually describe its arguments, not accept anything.
    remember = by_name["remember"].input_schema
    assert remember["required"] == ["content"]
    assert set(remember["properties"]) >= {"content", "entities", "kind", "writer", "episode"}


# --------------------------------------------------------------------------- the round trip


def test_remember_then_recall_returns_the_memory(server, mcp_db):
    written = ok(call(server, "remember", {
        "content": "Ada Lovelace wrote the first published algorithm, for the Analytical Engine.",
        "entities": ["Ada Lovelace", "Analytical Engine"],
        "kind": "fact",
        "writer": "test",
    }))
    memory_id = written["memory"]["memory_id"]
    assert written["memory"]["is_current"] is True
    assert written["memory"]["tenant_id"] == 3
    assert {e["name"] for e in written["about"]} == {"Ada Lovelace", "Analytical Engine"}

    # DuckDB's fts index is not incremental, so the text arm cannot see the row yet.
    ok(call(server, "rebuild_fts_index"))

    hits = ok(call(server, "recall", {"query": "Analytical Engine algorithm", "k": 5}))
    assert "text" in hits["arms"]
    assert memory_id in [h["memory_id"] for h in hits["hits"]], hits
    top = next(h for h in hits["hits"] if h["memory_id"] == memory_id)
    assert top["content"] == written["memory"]["content"]
    assert set(top["about"]) == {"Ada Lovelace", "Analytical Engine"}

    # And the write really is in the database, not just in the response.
    assert mcp_db.get(memory_id) is not None


def test_recall_reports_bm25_staleness_rather_than_hiding_it(server):
    ok(call(server, "remember", {"content": "first fact about ducks"}))
    ok(call(server, "rebuild_fts_index"))
    fresh = ok(call(server, "recall", {"query": "ducks"}))
    assert fresh["bm25_stale"] is False
    assert fresh["pending_fts_rows"] == 0

    ok(call(server, "remember", {"content": "second fact about ducks"}))
    stale = ok(call(server, "recall", {"query": "ducks"}))
    assert stale["bm25_stale"] is True
    assert stale["pending_fts_rows"] == 1


def test_recall_from_a_graph_seed_widens_one_entity_per_hop(server):
    """``hops`` counts RELATES_TO edges traversed from the seed, so build a 3-entity chain."""
    ok(call(server, "remember", {"content": "Ada corresponded with Babbage",
                                 "entities": ["Ada Lovelace"]}))
    ok(call(server, "remember", {"content": "The Analytical Engine had a mill and a store",
                                 "entities": ["Analytical Engine"]}))
    ok(call(server, "remember", {"content": "Babbage designed the Difference Engine",
                                 "entities": ["Charles Babbage"]}))
    ok(call(server, "relate", {"src": "Ada Lovelace", "dst": "Analytical Engine",
                               "rel_kind": "worked_on"}))
    # written the other way round: traversal is undirected, so hop 2 still reaches it
    ok(call(server, "relate", {"src": "Charles Babbage", "dst": "Analytical Engine",
                               "rel_kind": "designed"}))

    def contents(hops):
        hits = ok(call(server, "recall", {"seed_entity": "Ada Lovelace", "hops": hops, "k": 10}))
        assert hits["arms"] == ["graph"], hits["arms"]
        return {h["content"] for h in hits["hits"]}

    assert contents(0) == {"Ada corresponded with Babbage"}
    assert contents(1) == {"Ada corresponded with Babbage",
                           "The Analytical Engine had a mill and a store"}
    assert contents(2) == {"Ada corresponded with Babbage",
                           "The Analytical Engine had a mill and a store",
                           "Babbage designed the Difference Engine"}


def test_context_get_and_provenance_follow_a_supersession(server):
    first = ok(call(server, "remember", {
        "content": "The office is in Berlin.",
        "entities": ["office"],
        "episode": "Slack #general, 2026-01-04: 'we're in Berlin now'",
    }))
    old_id = first["memory"]["memory_id"]

    replaced = ok(call(server, "supersede", {"old_id": old_id, "content": "The office is in Lisbon."}))
    new_id = replaced["memory"]["memory_id"]
    assert replaced["superseded"] == old_id

    # context returns only what is currently believed
    ctx = ok(call(server, "context", {"entity": "office"}))
    assert [m["content"] for m in ctx["memories"]] == ["The office is in Lisbon."]

    # the superseded row is still there, closed
    old = ok(call(server, "get", {"memory_id": old_id}))
    assert old["memory"]["is_current"] is False
    assert old["memory"]["valid_to"] is not None

    prov = ok(call(server, "provenance", {"memory_id": new_id}))
    assert prov["depth"] == 1
    assert [m["memory_id"] for m in prov["chain"]] == [new_id, old_id]
    assert prov["source_text"] == "Slack #general, 2026-01-04: 'we're in Berlin now'"


def test_forget_is_soft_by_default_and_hard_on_request(server, mcp_db):
    written = ok(call(server, "remember", {"content": "temporary note", "entities": ["scratch"]}))
    memory_id = written["memory"]["memory_id"]

    soft = ok(call(server, "forget", {"memory_id": memory_id, "reason": "obsolete"}))
    assert soft["receipt"]["hard"] is False
    assert soft["receipt"]["memories_deleted"] == 0
    assert soft["receipt"]["audit_rows_written"] == 1
    assert ok(call(server, "get", {"memory_id": memory_id}))["memory"]["is_current"] is False

    hard = ok(call(server, "forget", {"memory_id": memory_id, "hard": True}))
    assert hard["receipt"]["hard"] is True
    assert hard["receipt"]["memories_deleted"] == 1
    assert hard["receipt"]["about_edges_deleted"] == 1
    # right to erasure: nothing anywhere references the id any more
    assert ok(call(server, "get", {"memory_id": memory_id}))["memory"] is None
    assert mcp_db.execute(
        "SELECT count(*) FROM memories WHERE memory_id = ?", [memory_id]).fetchone()[0] == 0
    assert mcp_db.execute(
        "SELECT count(*) FROM anatid_audit WHERE memory_id = ?", [memory_id]).fetchone()[0] == 0


def test_prune_defaults_to_a_dry_run_and_refuses_a_missing_policy(server, mcp_db):
    ok(call(server, "remember", {"content": "unused note"}))

    dry = ok(call(server, "prune", {"max_access_count": 0}))
    assert dry["dry_run"] is True
    assert dry["count"] == 1
    assert mcp_db.stats()["current_memories"] == 1, "a dry run must change nothing"

    wet = ok(call(server, "prune", {"max_access_count": 0, "dry_run": False}))
    assert wet["dry_run"] is False
    assert wet["count"] == 1
    assert mcp_db.stats()["current_memories"] == 0

    # anatid refuses to delete a whole tenant because an argument was forgotten
    refused = call(server, "prune", {"dry_run": False})
    assert refused.is_error is True
    assert "older_than" in _text(refused)


def test_stats_reports_the_database_and_the_sql_policy(server):
    ok(call(server, "remember", {"content": "one", "entities": ["x"]}))
    got = ok(call(server, "stats"))
    assert got["tenant_id"] == 3
    assert got["counts"]["memories"] == 1
    assert got["counts"]["entities"] == 1
    assert got["expand_path"] == "sql"
    assert got["embedding_dim"] == DIM
    assert got["sql_tool"]["enabled"] is True
    assert "BEGIN TRANSACTION READ ONLY" in got["sql_tool"]["enforcement"]


def test_a_bad_timestamp_is_a_readable_tool_error(server):
    bad = call(server, "recall", {"query": "x", "as_of": "last tuesday"})
    assert bad.is_error is True
    assert "ISO-8601" in _text(bad)


def test_unknown_memory_id_is_a_readable_tool_error(server):
    missing = call(server, "reinforce", {"memory_id": 12345})
    assert missing.is_error is True
    assert "12345" in _text(missing)


# --------------------------------------------------------------------------- the escape hatch


def test_sql_tool_reads(server):
    written = ok(call(server, "remember", {"content": "duck typing", "entities": ["ducks"]}))
    memory_id = written["memory"]["memory_id"]

    got = ok(call(server, "sql", {
        "query": "SELECT memory_id, content, kind FROM memories WHERE tenant_id = 3"}))
    assert got["columns"] == ["memory_id", "content", "kind"]
    assert got["rows"] == [[memory_id, "duck typing", "fact"]]
    assert got["statement_types"] == ["SELECT"]
    assert got["truncated"] is False

    # joins across the graph tables work, which is the point of the escape hatch
    joined = ok(call(server, "sql", {"query": (
        "SELECT e.name, count(*) FROM edges_about a "
        "JOIN entities e ON e.entity_id = a.dst GROUP BY 1 ORDER BY 1")}))
    assert joined["rows"] == [["ducks", 1]]

    explained = ok(call(server, "sql", {"query": "EXPLAIN SELECT 1"}))
    assert explained["statement_types"] == ["EXPLAIN"]


def test_sql_tool_caps_rows(mcp_db):
    for i in range(10):
        mcp_db.remember(f"row {i}")
    small = build_server(mcp_db, ServerConfig(db=mcp_db.path, max_rows=4, env={}))
    got = ok(call(small, "sql", {"query": "SELECT memory_id FROM memories"}))
    assert got["row_count"] == 4
    assert got["truncated"] is True


#: Every one of these must be refused, and must leave the database untouched.
REJECTED_SQL = [
    ("insert", "INSERT INTO memories (memory_id, tenant_id, content) VALUES (1, 3, 'evil')"),
    ("update", "UPDATE memories SET content = 'tampered'"),
    ("delete", "DELETE FROM memories"),
    ("attach", "ATTACH ':memory:' AS evil"),
    ("copy_to", "COPY (SELECT * FROM memories) TO '/tmp/anatid-mcp-should-not-exist.csv'"),
    ("copy_from", "COPY memories FROM '/tmp/nope.csv'"),
    ("create", "CREATE TABLE evil (a INTEGER)"),
    ("drop", "DROP TABLE memories"),
    ("multi_statement", "SELECT 1; DELETE FROM memories"),
    ("multi_statement_comment", "SELECT 1;\n-- sneak\nUPDATE memories SET content = 'x'"),
    ("explain_analyze_delete", "EXPLAIN ANALYZE DELETE FROM memories"),
    ("explain_analyze_insert",
     "EXPLAIN ANALYZE INSERT INTO memories (memory_id, tenant_id) VALUES (2, 3)"),
    ("explain_options_update", "EXPLAIN (ANALYZE) UPDATE memories SET content = 'x'"),
    ("pragma_create_index", "PRAGMA create_fts_index('memories', 'memory_id', 'content')"),
    ("install", "INSTALL httpfs"),
    ("load", "LOAD httpfs"),
    ("set", "SET threads = 1"),
    ("begin", "BEGIN TRANSACTION"),
    ("vacuum", "VACUUM"),
    ("prepare", "PREPARE p AS SELECT 1"),
    ("call", "CALL pragma_table_info('memories')"),
    ("merge", "MERGE INTO memories USING memories m2 ON 1=1 WHEN MATCHED THEN DELETE"),
    ("read_csv", "SELECT * FROM read_csv('/etc/passwd')"),
    ("read_text_union",
     "SELECT content FROM memories UNION ALL SELECT * FROM read_text('/etc/passwd')"),
    ("glob", "SELECT * FROM glob('/etc/*')"),
    ("secrets", "SELECT * FROM duckdb_secrets()"),
    # DuckDB's replacement scan: these are classified SELECT and contain NO function at all --
    # the path IS the table name -- so only the base-table check in layer 2 stops them.
    ("replacement_scan_csv", "SELECT * FROM '/etc/passwd.csv'"),
    ("replacement_scan_parquet", "SELECT * FROM '/tmp/anatid-does-not-exist.parquet'"),
    ("replacement_scan_glob", "SELECT * FROM '/etc/*.csv'"),
    ("replacement_scan_url", "SELECT * FROM 'https://example.com/x.csv'"),
    ("replacement_scan_in_join",
     "SELECT m.memory_id FROM memories m JOIN '/etc/passwd.csv' f ON true"),
    ("replacement_scan_in_cte",
     "WITH leaked AS (SELECT * FROM '/etc/passwd.csv') SELECT * FROM leaked"),
    # query()/query_table() take SQL as a STRING, so a denied function inside one is a plain
    # constant in the AST and invisible to the function scan.
    ("query_wrapping_read_csv", "SELECT * FROM query('SELECT * FROM read_csv(''/etc/passwd'')')"),
    ("query_table", "SELECT * FROM query_table('memories')"),
    ("comment_only", "-- nothing here"),
    ("garbage", "SELECT nonsense syntax (("),
]


@pytest.mark.parametrize("label,statement", REJECTED_SQL, ids=[c[0] for c in REJECTED_SQL])
def test_sql_tool_rejects_everything_that_is_not_a_select(server, mcp_db, label, statement):
    baseline = ok(call(server, "remember", {"content": "canary", "entities": ["canary"]}))
    canary_id = baseline["memory"]["memory_id"]

    result = call(server, "sql", {"query": statement})
    assert result.is_error is True, f"{label} was ALLOWED: {result.structured_content}"

    # the database is exactly as it was
    rows = mcp_db.execute("SELECT memory_id, content FROM memories").fetchall()
    assert rows == [(canary_id, "canary")], f"{label} mutated the database"
    tables = {r[0] for r in mcp_db.execute(
        "SELECT table_name FROM duckdb_tables()").fetchall()}
    assert "evil" not in tables
    assert not Path("/tmp/anatid-mcp-should-not-exist.csv").exists(), \
        f"{label} wrote a file outside the database"


def test_the_sql_tool_cannot_read_a_file_the_server_process_can(server, mcp_db, tmp_path):
    """Layer 2 is not only about functions: DuckDB's replacement scan hides a file read as a
    perfectly ordinary ``SELECT`` whose AST contains no function name at all."""
    secret = tmp_path / "secret.csv"
    secret.write_text("name,secret\nalice,TOPSECRET-123\n")
    # the server process really can read it -- the gate is the only thing in the way
    assert mcp_db.execute("SELECT secret FROM read_csv(?)",
                          [str(secret)]).fetchone() == ("TOPSECRET-123",)

    for statement in (
        f"SELECT * FROM '{secret}'",
        f"SELECT * FROM '{tmp_path}/*.csv'",
        f"SELECT * FROM query('SELECT * FROM read_csv(''{secret}'')')",
        f"SELECT * FROM read_csv('{secret}')",
    ):
        result = call(server, "sql", {"query": statement})
        assert result.is_error is True, f"LEAKED via {statement}"
        assert "TOPSECRET" not in _text(result)

    # and the legitimate shapes still run
    assert ok(call(server, "sql", {"query": "SELECT count(*) FROM memories"}))["row_count"] == 1
    assert ok(call(server, "sql", {
        "query": "WITH x AS (SELECT 1 AS a) SELECT a FROM x"}))["rows"] == [[1]]


def test_a_select_duckdb_will_not_run_returns_duckdbs_own_message(server):
    """A query that passes the gate but fails to bind is the caller's SQL to fix."""
    broken = call(server, "sql", {"query": "SELECT no_such_column FROM memories"})
    assert broken.is_error is True
    message = _text(broken)
    assert "no_such_column" in message
    assert "DuckDB rejected this query" in message


def test_rejection_message_says_what_duckdb_classified_it_as(server):
    result = call(server, "sql", {"query": "DELETE FROM memories"})
    assert result.is_error is True
    message = _text(result)
    assert "DELETE" in message
    assert "read-only" in message


def test_gateway_rejects_before_running_anything(mcp_db):
    """The gate is a *pre*-check: a rejected statement never reaches the engine."""
    mcp_db.remember("untouched")
    gate = SqlGateway(lambda: mcp_db.connection)

    decision = gate.inspect("DELETE FROM memories")
    assert decision.allowed is False
    assert decision.statement_types == ("DELETE",)

    with pytest.raises(SqlNotAllowed):
        gate.run("DELETE FROM memories")
    assert mcp_db.execute("SELECT count(*) FROM memories").fetchone()[0] == 1


def test_layer_three_is_duckdb_itself_not_our_parser(mcp_db):
    """The read-only *transaction* blocks writes even when the parser gate is bypassed.

    This is the claim the ``sql`` tool's docstring makes about layer 3, asserted directly:
    open the same ``BEGIN TRANSACTION READ ONLY`` the gateway uses and hand DuckDB an INSERT
    it never saw through :meth:`SqlGateway.inspect`.
    """
    mcp_db.remember("untouched")
    con = mcp_db.connection.cursor()
    con.execute("BEGIN TRANSACTION READ ONLY")
    try:
        with pytest.raises(duckdb.Error) as caught:
            con.execute("INSERT INTO memories (memory_id, tenant_id) VALUES (1, 3)")
        assert "read-only" in str(caught.value).lower()
    finally:
        con.execute("ROLLBACK")
        con.close()
    assert mcp_db.execute("SELECT count(*) FROM memories").fetchone()[0] == 1


def test_sql_tool_can_be_switched_off(mcp_db):
    off = build_server(mcp_db, ServerConfig(db=mcp_db.path, sql_tool=False, env={}))
    names = {t.name for t in drive(off, lambda c: c.list_tools()).tools}
    assert "sql" not in names
    assert "recall" in names


def test_sql_tool_off_via_environment():
    cfg = ServerConfig(env={"ANATID_SQL_TOOL": "off"})
    assert cfg.sql_tool is False
    assert ServerConfig(env={}).sql_tool is True


# --------------------------------------------------------------------------- read-only server


def test_read_only_database_registers_no_write_tools(tmp_path):
    path = tmp_path / "ro.anatid"
    with Anatid.open(path, tenant=3, embedding_dim=DIM) as writable:
        writable.remember("written before the server opened", entities=["thing"])

    with Anatid.open(path, tenant=3, embedding_dim=DIM, read_only=True) as ro:
        srv = build_server(ro, ServerConfig(db=str(path), read_only=True, env={}))
        names = {t.name for t in drive(srv, lambda c: c.list_tools()).tools}
        assert names == READ_ONLY_TOOLS
        assert "remember" not in names
        ctx = ok(call(srv, "context", {"entity": "thing"}))
        assert ctx["memories"][0]["content"] == "written before the server opened"


# --------------------------------------------------------------------------- over stdio


def test_server_runs_over_stdio_in_a_subprocess(tmp_path):
    """Launch the server the way an MCP client does and talk JSON-RPC over its pipes.

    This is the only test that exercises :func:`anatid.integrations.mcp.server.main`, the
    ``ANATID_DB`` / ``ANATID_TENANT`` environment contract, and real stdio framing.
    """
    db_path = tmp_path / "stdio.anatid"
    env = dict(os.environ)
    env.update({
        "ANATID_DB": str(db_path),
        "ANATID_TENANT": "11",
        "ANATID_EMBEDDING_DIM": str(DIM),
        "PYTHONPATH": str(REPO_ROOT / "src"),
    })
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "anatid.integrations.mcp.server"],
        env=env,
        cwd=str(tmp_path),
    )

    async def body(client):
        tools = {t.name for t in (await client.list_tools()).tools}
        written = await client.call_tool("remember", {
            "content": "anatid is an embedded graph database built on DuckDB",
            "entities": ["anatid", "DuckDB"],
        })
        await client.call_tool("rebuild_fts_index", {})
        found = await client.call_tool("recall", {"query": "embedded graph database", "k": 5})
        stats = await client.call_tool("stats", {})
        blocked = await client.call_tool("sql", {"query": "DELETE FROM memories"})
        allowed = await client.call_tool("sql", {"query": "SELECT count(*) AS n FROM memories"})
        return tools, written, found, stats, blocked, allowed

    async def go():
        async with Client(params) as client:
            return await body(client)

    tools, written, found, stats, blocked, allowed = asyncio.run(go())

    assert tools == EXPECTED_TOOLS
    assert written.is_error is False, _text(written)
    memory_id = written.structured_content["memory"]["memory_id"]

    assert found.is_error is False, _text(found)
    assert memory_id in [h["memory_id"] for h in found.structured_content["hits"]]

    assert stats.structured_content["tenant_id"] == 11, "ANATID_TENANT was not honoured"
    assert stats.structured_content["path"] == str(db_path), "ANATID_DB was not honoured"

    assert blocked.is_error is True, "the sql escape hatch ran a DELETE over stdio"
    assert allowed.structured_content["rows"] == [[1]]

    # the subprocess really wrote the file, and the DELETE really did not land
    assert db_path.is_file()
    with Anatid.open(db_path, tenant=11, embedding_dim=DIM, read_only=True) as check:
        assert check.get(memory_id) is not None
        assert check.stats()["memories"] == 1


def test_module_is_importable_and_reexports(server):
    import anatid.integrations.mcp as pkg

    assert pkg.build_server is build_server
    assert callable(pkg.main)
    assert "READ ONLY" in pkg.ENFORCEMENT
    # the console script declared in pyproject.toml points at a real callable
    from anatid.integrations.mcp.server import main

    assert callable(main)
    assert json.dumps(ok(call(server, "stats"))), "stats must be JSON-serialisable"
