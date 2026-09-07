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
import time
from pathlib import Path

import duckdb
import pytest

mcp_client = pytest.importorskip("mcp.client", reason="the MCP server needs 'mcp>=2.1'")

from mcp import StdioServerParameters                       # noqa: E402
from mcp.client import Client                               # noqa: E402

from anatid import Anatid                                   # noqa: E402
from anatid.integrations.mcp.server import (                # noqa: E402
    InsecureTransport,
    ServerConfig,
    build_server,
    check_transport_security,
    is_loopback_host,
    main,
)
from anatid.integrations.mcp.sqlgate import (               # noqa: E402
    SqlGateway,
    SqlNotAllowed,
    SqlTimeout,
)

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
    """The MCP server under test, with the environment ignored so tests are hermetic.

    ``sql_tool=True`` on purpose: the escape hatch is **opt-in** from 0.1.1 on, and a fixture
    that took the default would turn the whole rejection battery below into a battery of
    "Unknown tool: sql" -- green, and testing nothing.  :func:`test_the_sql_tool_is_off_unless_
    the_operator_opts_in` is the test that the default really is off.
    """
    return build_server(mcp_db, ServerConfig(db=mcp_db.path, tenant=3, sql_tool=True, env={}))


# --------------------------------------------------------------------------- tools/list


MEMORY_TOOLS = {
    "remember", "relate", "unrelate", "supersede", "correct", "reinforce", "forget", "prune",
    "rebuild_fts_index", "recall", "context", "get", "provenance", "stats",
}
#: What a server with the escape hatch opted in exposes.
EXPECTED_TOOLS = MEMORY_TOOLS | {"sql"}
READ_ONLY_TOOLS = {"recall", "context", "get", "provenance", "stats", "sql"}
#: `unrelate` and `correct` close edges; the server module's docstring says why that is marked.
DESTRUCTIVE_TOOLS = {"forget", "prune", "unrelate", "correct"}


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
    assert isinstance(memory_id, str), "an id crosses the wire as a decimal string"
    assert written["memory"]["is_current"] is True
    assert written["memory"]["tenant_id"] == "3"
    assert {e["name"] for e in written["about"]} == {"Ada Lovelace", "Analytical Engine"}

    # Not needed for the row to be found: this server's handle journals the write inside the
    # writing transaction, so the recall below would match it either way (see
    # tests/test_fts_visibility.py).  Called here to exercise the tool on a real corpus.
    ok(call(server, "rebuild_fts_index"))

    hits = ok(call(server, "recall", {"query": "Analytical Engine algorithm", "k": 5}))
    assert set(hits["weights"]) == set(hits["arms"]) and hits["weights"]["text"] == 1.0
    quiet = ok(call(server, "recall", {"query": "Analytical Engine algorithm", "k": 5,
                                       "arm_weights": {"text": 0}}))
    assert quiet["weights"]["text"] == 0.0 and all(h.get("text_rank") is None for h in quiet["hits"])
    bad = call(server, "recall", {"query": "x", "arm_weights": {"bm25": 1}})
    assert bad.is_error is True and "unknown arm" in _text(bad)
    assert "text" in hits["arms"]
    assert memory_id in [h["memory_id"] for h in hits["hits"]], hits
    top = next(h for h in hits["hits"] if h["memory_id"] == memory_id)
    assert top["content"] == written["memory"]["content"]
    assert set(top["about"]) == {"Ada Lovelace", "Analytical Engine"}

    # And the write really is in the database, not just in the response.  int() here is the
    # decode half of the wire contract: the tool speaks decimal strings, anatid speaks ints.
    assert mcp_db.get(int(memory_id)) is not None


def test_recall_reports_bm25_staleness_rather_than_hiding_it(tmp_path):
    """0.1.1's index cannot see a row written after its rebuild, and the tool says so.

    ``accelerators=False`` because that is the configuration the claim is about; the test below
    is the same sequence on the default handle, where the row is findable at once and the tool
    reports it is not stale.
    """
    with Anatid.open(
        tmp_path / "legacy.anatid", tenant=3, embedding_dim=DIM, accelerators=False
    ) as db:
        server = build_server(db, ServerConfig(db=db.path, tenant=3, sql_tool=True, env={}))
        ok(call(server, "remember", {"content": "first fact about ducks"}))
        ok(call(server, "rebuild_fts_index"))
        fresh = ok(call(server, "recall", {"query": "ducks"}))
        assert fresh["bm25_stale"] is False
        assert fresh["pending_fts_rows"] == 0

        ok(call(server, "remember", {"content": "second fact about ducks"}))
        stale = ok(call(server, "recall", {"query": "ducks"}))
        assert stale["bm25_stale"] is True
        assert stale["pending_fts_rows"] == 1
        assert len(ok(call(server, "recall", {"query": "ducks"}))["hits"]) == 1


def test_on_the_default_handle_a_new_row_is_searchable_and_nothing_is_reported_stale(server):
    """The same sequence on a default handle: the derived index journals the write inside the
    writing transaction, so the second fact is in the very next recall and ``bm25_stale`` is
    False because nothing is hidden."""
    ok(call(server, "remember", {"content": "first fact about ducks"}))
    ok(call(server, "rebuild_fts_index"))
    ok(call(server, "remember", {"content": "second fact about ducks"}))
    after = ok(call(server, "recall", {"query": "ducks"}))
    assert after["bm25_stale"] is False
    assert after["pending_fts_rows"] == 1        # one document a search rescans, not one hidden
    assert len(after["hits"]) == 2


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
    # the derived-index half of the erasure is reported too: the memory's two journal rows (the
    # insert and the soft forget's close) go with it, and no generation had to be invalidated
    # because nothing has been built on this database.
    assert hard["receipt"]["derived_rows_deleted"] == 2
    assert hard["receipt"]["invalidated_generations"] == 0
    assert mcp_db.execute(
        "SELECT count(*) FROM anatid_index_journal WHERE doc_id = ?", [memory_id]
    ).fetchone()[0] == 0
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
    assert got["tenant_id"] == "3"
    assert got["counts"]["memories"] == 1
    assert got["counts"]["entities"] == 1
    assert got["expand_path"] == "sql"
    assert got["embedding_dim"] == DIM
    assert got["sql_tool"]["enabled"] is True
    assert got["sql_tool"]["opt_in"] is True
    assert "BEGIN TRANSACTION READ ONLY" in got["sql_tool"]["enforcement"]
    # the limits are reported, not just claimed in prose
    limits = got["sql_tool"]["limits"]
    assert limits["external_access_disabled"] is True
    assert limits["memory_limit"] == "1GB"
    assert limits["timeout_seconds"] == 30.0


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
    small = build_server(mcp_db, ServerConfig(db=mcp_db.path, max_rows=4, sql_tool=True, env={}))
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
    # host metadata: read-only, but the answer is the host's configuration and filesystem
    # layout (home_directory, temp_directory, every attached database's path, spill files),
    # not memory data.  Found by the 0.1.1 verifier; refused by name in layer 2.
    ("host_settings", "SELECT * FROM duckdb_settings() WHERE name LIKE '%dir%'"),
    ("host_home_directory", "SELECT current_setting('home_directory')"),
    ("host_databases", "SELECT * FROM duckdb_databases()"),
    ("host_temp_files", "SELECT * FROM duckdb_temporary_files()"),
    ("host_storage_info", "SELECT * FROM pragma_storage_info('memories')"),
    ("host_database_size", "SELECT * FROM pragma_database_size()"),
    ("host_platform", "SELECT * FROM pragma_platform()"),
    ("comment_only", "-- nothing here"),
    ("garbage", "SELECT nonsense syntax (("),
    # DuckDB cannot serialize this one's parse tree ("Only SELECT statements can be serialized
    # to json!"), because it expands PIVOT into a CREATE plus a SELECT. Layer 1 rejects it on
    # the CREATE; test_a_statement_whose_ast_cannot_be_serialized_is_refused covers the case
    # where the serializer is the only thing that failed.
    ("pivot_unserializable_ast", "SELECT * FROM (PIVOT memories ON kind USING count(*))"),
]


@pytest.mark.parametrize("label,statement", REJECTED_SQL, ids=[c[0] for c in REJECTED_SQL])
def test_sql_tool_rejects_everything_that_is_not_a_select(server, mcp_db, label, statement):
    baseline = ok(call(server, "remember", {"content": "canary", "entities": ["canary"]}))
    canary_id = baseline["memory"]["memory_id"]

    result = call(server, "sql", {"query": statement})
    assert result.is_error is True, f"{label} was ALLOWED: {result.structured_content}"

    # the database is exactly as it was
    rows = mcp_db.execute("SELECT memory_id, content FROM memories").fetchall()
    assert rows == [(int(canary_id), "canary")], f"{label} mutated the database"
    tables = {r[0] for r in mcp_db.execute(
        "SELECT table_name FROM duckdb_tables()").fetchall()}
    assert "evil" not in tables
    assert not Path("/tmp/anatid-mcp-should-not-exist.csv").exists(), \
        f"{label} wrote a file outside the database"


def test_the_sql_tool_cannot_read_a_file_the_server_process_can(mcp_db, tmp_path):
    """Layer 2 is not only about functions: DuckDB's replacement scan hides a file read as a
    perfectly ordinary ``SELECT`` whose AST contains no function name at all.

    Takes ``mcp_db`` rather than the ``server`` fixture because the order matters: the process
    can read the file **until** a gateway is built, and building one is what applies layer 4's
    ``SET enable_external_access=false``.  Both halves are asserted.
    """
    secret = tmp_path / "secret.csv"
    secret.write_text("name,secret\nalice,TOPSECRET-123\n")
    # the server process really can read it -- until the gateway exists, the gate is the only
    # thing in the way
    assert mcp_db.execute("SELECT secret FROM read_csv(?)",
                          [str(secret)]).fetchone() == ("TOPSECRET-123",)

    srv = build_server(mcp_db, ServerConfig(db=mcp_db.path, tenant=3, sql_tool=True, env={}))
    for statement in (
        f"SELECT * FROM '{secret}'",
        f"SELECT * FROM '{tmp_path}/*.csv'",
        f"SELECT * FROM query('SELECT * FROM read_csv(''{secret}'')')",
        f"SELECT * FROM read_csv('{secret}')",
    ):
        result = call(srv, "sql", {"query": statement})
        assert result.is_error is True, f"LEAKED via {statement}"
        assert "TOPSECRET" not in _text(result)

    # and the legitimate shapes still run
    assert ok(call(srv, "sql", {"query": "SELECT count(*) FROM memories"}))["row_count"] == 1
    assert ok(call(srv, "sql", {
        "query": "WITH x AS (SELECT 1 AS a) SELECT a FROM x"}))["rows"] == [[1]]


def test_enabling_the_sql_tool_disables_duckdbs_external_access(mcp_db, tmp_path):
    """Layer 4, asserted on DuckDB rather than on our own gate.

    ``enable_external_access=false`` is DuckDB's own recommendation for an untrusted SQL surface,
    and it is what stands behind layer 2: a replacement scan that somehow got past the AST check
    still could not reach the filesystem.  It is GLOBAL scope in DuckDB, so this also documents
    the cost -- the *handle* loses file access, which is why the escape hatch is opt-in.
    """
    secret = tmp_path / "secret.csv"
    secret.write_text("name,secret\nalice,TOPSECRET-123\n")
    assert mcp_db.execute("SELECT secret FROM read_csv(?)", [str(secret)]).fetchone()[0] \
        == "TOPSECRET-123"
    assert mcp_db.execute("SELECT current_setting('enable_external_access')").fetchone()[0] is True

    gateway = SqlGateway(lambda: mcp_db.connection)
    assert gateway.hardening["external_access_disabled"] is True
    assert gateway.hardening["memory_limit"] == "1GB"
    assert gateway.hardening["errors"] == []
    assert mcp_db.execute("SELECT current_setting('enable_external_access')").fetchone()[0] is False

    # DuckDB itself now refuses, with no help from the gate
    with pytest.raises(duckdb.Error) as caught:
        mcp_db.execute("SELECT secret FROM read_csv(?)", [str(secret)]).fetchall()
    assert "file system operations are disabled" in str(caught.value).lower()

    # ... and the database still works: memory verbs are untouched by the hardening
    written = mcp_db.remember("still writable after hardening", entities=["hardening"])
    assert mcp_db.get(written.memory_id) is not None
    # would raise "Cannot access directory .../extensions" if fts had to autoload now;
    # anatid loads it in Anatid.open(), before any gateway exists
    assert mcp_db.rebuild_fts_index() is not None


def test_a_runaway_query_is_interrupted_rather_than_run_forever(mcp_db):
    """DuckDB has no statement_timeout, so the gateway interrupts the query itself."""
    gateway = SqlGateway(lambda: mcp_db.connection, timeout=0.25)
    assert gateway.timeout == 0.25

    started = time.monotonic()
    with pytest.raises(SqlTimeout) as caught:
        gateway.run("SELECT count(*) FROM range(1000000000000) t(i) WHERE i % 7 = 0")
    elapsed = time.monotonic() - started
    assert elapsed < 20, "the watchdog did not fire"
    assert "0.25s" in str(caught.value)

    # the connection is still usable afterwards: an interrupt is not a poisoned connection
    assert gateway.run("SELECT 1 AS n")["rows"] == [[1]]

    # and the timeout reaches the MCP client as a readable tool error, not a crash
    srv = build_server(mcp_db, ServerConfig(db=mcp_db.path, tenant=3, sql_tool=True,
                                            sql_timeout=0.25, env={}))
    slow = call(srv, "sql", {"query": "SELECT count(*) FROM range(1000000000000) t(i)"})
    assert slow.is_error is True
    assert "0.25s" in _text(slow)


def test_the_limits_are_reported_with_every_result(mcp_db):
    gateway = SqlGateway(lambda: mcp_db.connection, timeout=5, memory_limit="512MB")
    result = gateway.run("SELECT 1")
    assert result["limits"] == {
        "max_rows": 200,
        "timeout_seconds": 5.0,
        "memory_limit": "512MB",
        "external_access_disabled": True,
    }
    # DuckDB reports what it actually applied, in its own units: 512MB -> "488.2 MiB"
    assert mcp_db.execute("SELECT current_setting('memory_limit')").fetchone()[0] == "488.2 MiB"


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


def test_the_sql_tool_is_off_unless_the_operator_opts_in(mcp_db, tmp_path):
    """The defect: `sql` used to be registered by default.

    An MCP server is driven by a model, and a model is driven by whatever text lands in its
    context, so a default-on arbitrary-SQL tool hands every prompt injection a primitive the
    memory verbs deliberately do not offer. It is now opt-in, and this is the test of that --
    the ``server`` fixture opts in, so nothing else in this file would notice a regression.
    """
    default = build_server(mcp_db, ServerConfig(db=mcp_db.path, tenant=3, env={}))
    names = {t.name for t in drive(default, lambda c: c.list_tools()).tools}
    assert names == MEMORY_TOOLS
    assert "sql" not in names

    # calling it is an error, not a silent success
    assert call(default, "sql", {"query": "SELECT 1"}).is_error is True

    # ... and the model is not told about a tool that is not there
    assert "`sql`" not in (default.instructions or "")

    # stats says so, so an operator can check without reading the tool list
    stats = ok(call(default, "stats"))
    assert stats["sql_tool"]["enabled"] is False
    assert stats["sql_tool"]["opt_in"] is True
    assert "limits" not in stats["sql_tool"]

    # and building the default server does NOT harden the handle: no gateway, no side effect
    secret = tmp_path / "readable.csv"
    secret.write_text("a\n1\n")
    assert mcp_db.execute("SELECT * FROM read_csv(?)", [str(secret)]).fetchone() == (1,)

    # opting in registers it, and says so in the instructions the client sees
    on = build_server(mcp_db, ServerConfig(db=mcp_db.path, tenant=3, sql_tool=True, env={}))
    assert "sql" in {t.name for t in drive(on, lambda c: c.list_tools()).tools}
    assert "`sql`" in (on.instructions or "")
    assert "OFF BY DEFAULT" in next(
        t.description for t in drive(on, lambda c: c.list_tools()).tools if t.name == "sql")


def test_sql_tool_opt_in_from_the_environment():
    """``ANATID_ENABLE_SQL`` is the switch; ``ANATID_SQL_TOOL=off`` keeps its old meaning."""
    assert ServerConfig(env={}).sql_tool is False                       # the new default
    assert ServerConfig(env={"ANATID_ENABLE_SQL": "1"}).sql_tool is True
    assert ServerConfig(env={"ANATID_ENABLE_SQL": "true"}).sql_tool is True
    assert ServerConfig(env={"ANATID_ENABLE_SQL": "0"}).sql_tool is False
    # the 0.1.0 variable still works for turning it ON ...
    assert ServerConfig(env={"ANATID_SQL_TOOL": "on"}).sql_tool is True
    # ... and an explicit OFF wins over anything that would enable it, which is the direction a
    # config block written against 0.1.0 must never break in
    assert ServerConfig(env={"ANATID_SQL_TOOL": "off"}).sql_tool is False
    assert ServerConfig(
        env={"ANATID_SQL_TOOL": "off", "ANATID_ENABLE_SQL": "1"}).sql_tool is False
    # an explicit argument beats the environment either way
    assert ServerConfig(sql_tool=True, env={"ANATID_SQL_TOOL": "off"}).sql_tool is True
    assert ServerConfig(sql_tool=False, env={"ANATID_ENABLE_SQL": "1"}).sql_tool is False


def test_the_sql_limits_are_configurable_from_the_environment():
    assert ServerConfig(env={}).sql_timeout == 30.0
    assert ServerConfig(env={}).sql_memory_limit == "1GB"
    cfg = ServerConfig(env={"ANATID_SQL_TIMEOUT": "2.5", "ANATID_SQL_MEMORY_LIMIT": "256MB"})
    assert cfg.sql_timeout == 2.5
    assert cfg.sql_memory_limit == "256MB"
    assert ServerConfig(sql_timeout=0, env={}).sql_timeout == 0.0        # 0 disables the watchdog


# ------------------------------------------------------------------- layer 2 fails CLOSED


#: What DuckDB really returns from ``json_serialize_sql`` when it cannot serialize a statement
#: (captured verbatim from ``SELECT json_serialize_sql('COPY (SELECT 1) TO ''/dev/null''')`` on
#: duckdb 1.5.5).  The gate must treat this as a refusal, not as "no functions found".
DUCKDB_SERIALIZER_ERROR = json.dumps({
    "error": True,
    "error_type": "not implemented",
    "error_message": "Only SELECT statements can be serialized to json!",
})


class _OneRow:
    def __init__(self, value):
        self._value = value

    def fetchone(self):
        return (self._value,)


class _SerializerFails:
    """A DuckDB connection whose ``json_serialize_sql`` fails.  Everything else is the real thing.

    DuckDB fails at serialization in two ways -- it raises, or it returns a JSON object with
    ``error: true`` -- and sqlgate handled both by *skipping the scan and running the query
    anyway*.  Injecting the failure is the only way to reach that branch on demand: on duckdb
    1.5.5 every statement that fails to serialize also fails layer 1, so a fixture that relied on
    finding one would stop testing this the moment DuckDB's serializer changed.
    """

    def __init__(self, con, mode: str):
        self._con = con
        self._mode = mode

    def __getattr__(self, name):
        return getattr(self._con, name)

    def execute(self, sql, params=None):
        if "json_serialize_sql" in sql:
            if self._mode == "raise":
                raise duckdb.Error("Not implemented Error: cannot serialize this statement")
            return _OneRow(DUCKDB_SERIALIZER_ERROR)
        return self._con.execute(sql) if params is None else self._con.execute(sql, params)


@pytest.mark.parametrize("mode", ["raise", "error_json"])
def test_a_statement_whose_ast_cannot_be_serialized_is_refused(mcp_db, tmp_path, mode):
    """THE DEFECT: an unscannable statement used to set ``ast_scanned=False`` and run anyway.

    Layers 1 and 3 do not cover what layer 2 covers. ``SELECT * FROM '/etc/passwd.csv'`` is
    classified SELECT (layer 1 passes it) and writes nothing (layer 3 passes it); the *only*
    thing that stops DuckDB's replacement scan from reading the file is the base-table check in
    the AST scan. So a gate that shrugs when the AST is unavailable is a gate that reads files.
    """
    secret = tmp_path / "secret.csv"
    secret.write_text("name,secret\nalice,TOPSECRET-123\n")

    working = SqlGateway(lambda: mcp_db.connection, harden=False)
    allowed = working.inspect("SELECT count(*) FROM memories")
    assert allowed.allowed is True and allowed.ast_scanned is True      # the contrast case

    broken = SqlGateway(lambda: _SerializerFails(mcp_db.connection, mode), harden=False)

    decision = broken.inspect("SELECT count(*) FROM memories")
    assert decision.allowed is False, "an unscannable statement was ALLOWED"
    assert decision.ast_scanned is False
    assert "could not serialize" in (decision.reason or "")

    with pytest.raises(SqlNotAllowed):
        broken.run("SELECT count(*) FROM memories")

    # and the case that makes it matter: the file read is refused instead of served
    with pytest.raises(SqlNotAllowed) as caught:
        broken.run(f"SELECT * FROM '{secret}'")
    assert "TOPSECRET" not in str(caught.value)


def test_every_allowed_statement_was_actually_scanned(server):
    """``ast_scanned`` is now an invariant of acceptance, not a field that can come back False."""
    for query in ("SELECT 1",
                  "SELECT count(*) FROM memories",
                  "EXPLAIN SELECT memory_id FROM memories",
                  "WITH x AS (SELECT 1 AS a) SELECT a FROM x"):
        got = ok(call(server, "sql", {"query": query}))
        assert got["ast_scanned"] is True, query


# --------------------------------------------------------------- transport security (defect 3)


def test_is_loopback_host_decides_what_can_be_reached_from_outside():
    for host in ("127.0.0.1", "127.0.0.53", "::1", "[::1]", "localhost", "LOCALHOST"):
        assert is_loopback_host(host) is True, host
    for host in ("0.0.0.0", "::", "192.168.1.10", "10.0.0.1", "example.com", "", "  ", None,
                 "no-such-host.invalid"):
        assert is_loopback_host(host) is False, host


def test_a_non_loopback_http_bind_without_authentication_is_refused():
    """anatid ships no authentication, so binding a network interface must be a decision.

    The MCP authorization spec puts authorization on the HTTP transports; a memory server on
    0.0.0.0 with none of it hands every memory in the file, and every write tool, to whoever can
    route to the port.
    """
    for transport in ("streamable-http", "sse"):
        cfg = ServerConfig(transport=transport, host="0.0.0.0", env={})
        with pytest.raises(InsecureTransport) as caught:
            check_transport_security(cfg)
        message = str(caught.value)
        assert "0.0.0.0" in message
        assert "authentication" in message
        assert "ANATID_MCP_AUTH" in message                 # says how to proceed on purpose

    # loopback is fine, and so is stdio whatever the host says (it opens no socket)
    check_transport_security(ServerConfig(transport="streamable-http", host="127.0.0.1", env={}))
    check_transport_security(ServerConfig(transport="streamable-http", host="localhost", env={}))
    check_transport_security(ServerConfig(transport="stdio", host="0.0.0.0", env={}))

    # and an operator who really does have an authenticating proxy in front says so
    check_transport_security(ServerConfig(transport="streamable-http", host="0.0.0.0",
                                          auth="oauth2 proxy", env={}))
    check_transport_security(ServerConfig(
        transport="streamable-http", host="0.0.0.0",
        env={"ANATID_MCP_AUTH": "behind an authenticating gateway"}))
    # ... but not by writing "off" in it
    with pytest.raises(InsecureTransport):
        check_transport_security(ServerConfig(transport="streamable-http", host="0.0.0.0",
                                              env={"ANATID_MCP_AUTH": "off"}))


def test_build_server_refuses_an_insecure_transport(mcp_db):
    with pytest.raises(InsecureTransport):
        build_server(mcp_db, ServerConfig(db=mcp_db.path, tenant=3, transport="streamable-http",
                                          host="0.0.0.0", env={}))
    # the safe configurations still build
    assert build_server(mcp_db, ServerConfig(db=mcp_db.path, tenant=3,
                                             transport="streamable-http", host="127.0.0.1",
                                             env={})) is not None


def test_main_refuses_to_start_and_creates_nothing(tmp_path, capsys, monkeypatch):
    """The refusal happens before the database is opened: a refused start leaves no file."""
    monkeypatch.delenv("ANATID_MCP_AUTH", raising=False)
    db_path = tmp_path / "never-created.anatid"
    code = main(["--db", str(db_path), "--transport", "streamable-http", "--host", "0.0.0.0",
                 "--port", "8899"])
    assert code == 2
    err = capsys.readouterr().err
    assert "refusing to start" in err
    assert "0.0.0.0" in err
    assert not db_path.exists(), "a refused start still created the database file"


# --------------------------------------------------------------------------- read-only server


def test_read_only_database_registers_no_write_tools(tmp_path):
    path = tmp_path / "ro.anatid"
    with Anatid.open(path, tenant=3, embedding_dim=DIM) as writable:
        writable.remember("written before the server opened", entities=["thing"])

    with Anatid.open(path, tenant=3, embedding_dim=DIM, read_only=True) as ro:
        # sql_tool=True so READ_ONLY_TOOLS is the whole list; it also proves the hardening
        # (SET memory_limit / enable_external_access) works on a read-only connection
        srv = build_server(ro, ServerConfig(db=str(path), read_only=True, sql_tool=True, env={}))
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
        # the escape hatch is opt-in, and this test calls it: the client's config block is
        # exactly where an operator would turn it on, so turn it on the same way
        "ANATID_ENABLE_SQL": "1",
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

    assert stats.structured_content["tenant_id"] == "11", "ANATID_TENANT was not honoured"
    assert stats.structured_content["path"] == str(db_path), "ANATID_DB was not honoured"

    assert blocked.is_error is True, "the sql escape hatch ran a DELETE over stdio"
    assert allowed.structured_content["rows"] == [[1]]

    # the subprocess really wrote the file, and the DELETE really did not land
    assert db_path.is_file()
    with Anatid.open(db_path, tenant=11, embedding_dim=DIM, read_only=True) as check:
        assert check.get(int(memory_id)) is not None
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
