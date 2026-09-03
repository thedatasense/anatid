"""Tests for ``anatid.integrations.openai_agents``.

**No network, no API key.**  The full human-in-the-loop flow -- model asks for a write, the run is
interrupted, the state is parked in DuckDB, another "process" approves it, the run resumes and the
tool finally executes -- is driven by the SDK's own ``agents.testing.ScriptedModel``, which is a
real ``Model`` implementation the ``Runner`` drives exactly like a live one.  So nothing here is
faked and nothing here is skipped for want of credentials.  There is deliberately no live-model
test: every behaviour these modules own is reachable without one.

The session is additionally checked *against the SDK's own SQLite session* -- identical
operations, identical output -- so "matches the protocol" is a measurement, not a claim.
"""

from __future__ import annotations

import asyncio
import json
import types

import pytest

from anatid import Anatid

agents = pytest.importorskip("agents", reason='needs pip install "anatid[agents]"')

from agents import Agent, Runner, RunState  # noqa: E402
from agents.memory import Session, SQLiteSession  # noqa: E402
from agents.run_config import RunConfig  # noqa: E402
from agents.testing import ScriptedModel, assistant_message, function_call  # noqa: E402
from agents.tool_context import ToolContext  # noqa: E402

from anatid.integrations.openai_agents import (  # noqa: E402
    AnatidSession,
    ApprovalRequest,
    RunStateStore,
    always_require_approval,
    approve_low_risk,
    create_memory_tools,
    never_require_approval,
)

agents.set_tracing_disabled(True)
NO_TRACE = RunConfig(tracing_disabled=True)

DIM = 8


def run(coro):
    """``asyncio.run`` under a plain sync test (pytest-asyncio is not a dependency)."""
    return asyncio.run(coro)


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "output_text", "text": text}]}


@pytest.fixture
def db():
    with Anatid.open(":memory:", tenant=1, embedding_dim=DIM) as handle:
        yield handle


@pytest.fixture
def session(db):
    return AnatidSession("conv-1", db)


# --------------------------------------------------------------------------- protocol shape

def test_session_satisfies_the_sdk_session_protocol(session):
    assert isinstance(session, Session)
    assert session.session_id == "conv-1"
    assert hasattr(session, "session_settings")
    for name in ("get_items", "add_items", "pop_item", "clear_session"):
        method = getattr(session, name)
        assert asyncio.iscoroutinefunction(method), f"{name} must be async"


def test_add_get_pop_clear_ordering_and_limits(session):
    async def scenario():
        assert await session.get_items() == []
        assert await session.pop_item() is None

        items = [user("one"), assistant("two"), user("three"), assistant("four")]
        await session.add_items(items)

        assert await session.get_items() == items                 # oldest first
        assert await session.get_items(2) == items[-2:]           # newest N, chronological
        assert await session.get_items(0) == []                   # non-positive -> nothing
        assert await session.get_items(-1) == []
        assert await session.get_items(99) == items               # limit past the end

        assert await session.pop_item() == items[-1]
        assert await session.get_items() == items[:-1]

        await session.add_items([])                               # no-op, must not raise
        assert await session.get_items() == items[:-1]

        await session.clear_session()
        assert await session.get_items() == []
        assert await session.pop_item() is None

    run(scenario())


def test_matches_the_sdk_sqlite_session_item_for_item(session):
    """The same calls against ``SQLiteSession`` must produce the same answers."""
    sqlite = SQLiteSession("conv-1", ":memory:")
    script = [user("a"), assistant("b"), user("c"), assistant("d"), user("e")]

    async def scenario():
        for pair in (script[:2], script[2:4], script[4:]):
            await session.add_items(pair)
            await sqlite.add_items(pair)
        for limit in (None, 1, 3, 5, 10, 0):
            assert await session.get_items(limit) == await sqlite.get_items(limit), limit
        assert await session.pop_item() == await sqlite.pop_item()
        assert await session.get_items() == await sqlite.get_items()
        await session.clear_session()
        await sqlite.clear_session()
        assert await session.get_items() == await sqlite.get_items() == []

    try:
        run(scenario())
    finally:
        sqlite.close()


def test_session_settings_limit_is_honoured(db):
    settings = agents.memory.SessionSettings(limit=2)
    scoped = AnatidSession("conv-limit", db, session_settings=settings)
    items = [user("1"), user("2"), user("3")]

    async def scenario():
        await scoped.add_items(items)
        assert await scoped.get_items() == items[-2:]     # settings default applies
        assert await scoped.get_items(3) == items         # explicit limit wins

    run(scenario())


def test_sessions_in_one_file_do_not_leak_into_each_other(db):
    a, b = AnatidSession("a", db), AnatidSession("b", db)

    async def scenario():
        await a.add_items([user("only a")])
        await b.add_items([user("only b"), user("also b")])
        assert await a.get_items() == [user("only a")]
        assert len(await b.get_items()) == 2
        await a.clear_session()
        assert await a.get_items() == []
        assert len(await b.get_items()) == 2

    run(scenario())


def test_unreadable_rows_are_skipped_not_fatal(session, db):
    """A hand-corrupted ``item_json`` must not break history retrieval."""
    async def scenario():
        await session.add_items([user("good one"), user("good two")])
        db.execute("UPDATE agent_messages SET item_json = 'not json' WHERE seq = 2")
        assert await session.get_items() == [user("good one")]
        assert await session.get_items(1) == [user("good one")]   # window widens past the bad row
        assert await session.pop_item() == user("good one")       # bad row dropped on the way

    run(scenario())


# --------------------------------------------------------------------------- queryable history

def test_history_is_queryable_rows_not_a_blob(session, db):
    async def scenario():
        await session.add_items([
            user("what do you know about Ada?"),
            {"type": "function_call", "name": "anatid_recall", "call_id": "c1",
             "arguments": '{"query": "Ada"}'},
            assistant("She wrote the first algorithm."),
        ])

    run(scenario())
    rows = db.execute(
        "SELECT seq, turn, item_type, role, tool_name, item_text, writer, valid_from, tx_from "
        "FROM agent_messages ORDER BY seq").fetchall()
    assert [r[0] for r in rows] == [1, 2, 3]
    assert [r[1] for r in rows] == [1, 1, 1]                       # one user turn
    assert [r[2] for r in rows] == ["message", "function_call", "message"]
    assert [r[3] for r in rows] == ["user", None, "assistant"]
    assert [r[4] for r in rows] == [None, "anatid_recall", None]
    assert rows[0][5] == "what do you know about Ada?"             # text extracted for SQL
    assert all(r[6] == "session:conv-1" for r in rows)             # writer provenance stamped
    assert all(r[7] is not None and r[8] is not None for r in rows)  # system columns filled

    # ...and the SDK still gets its dicts back verbatim.
    assert run(session.get_items())[2] == assistant("She wrote the first algorithm.")


def test_close_mode_keeps_the_audit_trail(db):
    kept = AnatidSession("audit", db, clear_mode="close")

    async def scenario():
        await kept.add_items([user("said once"), user("said twice")])
        await kept.pop_item()
        assert await kept.get_items() == [user("said once")]
        await kept.clear_session()
        assert await kept.get_items() == []

    run(scenario())
    total, closed = db.execute(
        "SELECT count(*), count(*) FILTER (WHERE valid_to IS NOT NULL) FROM agent_messages"
    ).fetchone()
    assert total == 2 and closed == 2      # nothing deleted, everything closed


def test_analytics_rollups(session):
    usage = types.SimpleNamespace(
        requests=2, input_tokens=120, output_tokens=30, total_tokens=150,
        input_tokens_details=types.SimpleNamespace(cached_tokens=100),
        output_tokens_details=types.SimpleNamespace(reasoning_tokens=5),
    )
    result = types.SimpleNamespace(context_wrapper=types.SimpleNamespace(usage=usage))

    async def scenario():
        await session.add_items([user("turn one"), assistant("ok")])
        await session.store_run_usage(result)
        await session.add_items([
            user("turn two"),
            {"type": "function_call", "name": "anatid_remember", "call_id": "c1",
             "arguments": "{}"},
        ])
        await session.store_run_usage(result)

        totals = await session.usage_totals()
        assert totals == {"requests": 4, "input_tokens": 240, "output_tokens": 60,
                          "total_tokens": 300, "cached_tokens": 200, "reasoning_tokens": 10,
                          "rows": 2}
        by_day = await session.usage_by_day()
        assert len(by_day) == 1 and by_day[0]["total_tokens"] == 300
        turns = await session.turn_counts_by_day()
        assert len(turns) == 1 and turns[0]["turns"] == 2 and turns[0]["items"] == 4
        assert await session.tool_usage() == [
            {"tool_name": "anatid_remember", "calls": 1, "turns": 1}]
        types_seen = {row["item_type"] for row in await session.item_type_counts()}
        assert types_seen == {"message", "function_call"}
        # No usage on the result -> no row, no exception.
        assert await session.store_run_usage(types.SimpleNamespace()) is False

    run(scenario())


def test_session_and_graph_are_joinable_in_one_query(session, db):
    """The reason this integration exists: history JOIN knowledge, one file, one statement."""
    db.remember("Ada Lovelace wrote the first algorithm", entities=["Ada Lovelace"],
                writer=session.writer)
    db.remember("Grace Hopper built the first compiler", entities=["Grace Hopper"],
                writer="someone-else")

    async def scenario():
        await session.add_items([
            user("tell me about Ada Lovelace"),
            assistant("Ada Lovelace wrote the first algorithm."),
            user("and who else?"),
        ])
        mentioned = await session.entities_mentioned()
        assert [(m["name"], m["mentions"]) for m in mentioned] == [("Ada Lovelace", 2)]
        assert "Grace Hopper" not in {m["name"] for m in mentioned}

        written = await session.memories_written_here()
        assert [m.content for m in written] == ["Ada Lovelace wrote the first algorithm"]

    run(scenario())

    # The same join, written by hand, to show it is ordinary SQL over ordinary tables.
    rows = db.execute(
        "SELECT e.name, count(*) FROM agent_messages m "
        "JOIN entities e ON e.tenant_id = m.tenant_id "
        " AND contains(lower(m.item_text), lower(e.name)) "
        "WHERE m.session_id = 'conv-1' GROUP BY e.name").fetchall()
    assert rows == [("Ada Lovelace", 2)]


def test_hard_forget_erases_the_memory_from_the_transcript_too(session, db):
    """The whole point of one file is that history and knowledge are together -- which makes a
    transcript a copy of the memory, and therefore part of a right-to-erasure request.

    Before this was fixed, ``forget(hard=True)`` deleted the memory row while the tool call and
    the tool result (which quote the id AND the content verbatim) stayed in ``agent_messages``,
    so ``get_items()`` replayed the erased content into the model's context on the next turn.
    """
    m = db.remember("Ada's home address is 12 Elm Street", entities=["Ada"],
                    writer=session.writer)
    keep = db.remember("Ada likes DuckDB", entities=["Ada"], writer=session.writer)

    async def scenario():
        await session.add_items([
            user("remember my address"),
            {"type": "function_call", "call_id": "c1", "name": "anatid_remember",
             "arguments": json.dumps({"content": "Ada's home address is 12 Elm Street"})},
            {"type": "function_call_output", "call_id": "c1",
             "output": json.dumps({"memory_id": m.memory_id,
                                   "content": "Ada's home address is 12 Elm Street"})},
            assistant("Saved."),
            user("what else do you know?"),
        ])
        assert len(await session.get_items()) == 5

    run(scenario())

    receipt = db.forget(m.memory_id, hard=True)
    assert receipt.extra_rows_deleted == 2          # the call and its output
    assert receipt.rows_removed >= 3

    assert db.execute("SELECT count(*) FROM agent_messages WHERE contains(item_json, ?)",
                      [str(m.memory_id)]).fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM agent_messages "
                      "WHERE contains(item_json, 'Elm Street')").fetchone()[0] == 0
    remaining = run(session.get_items())
    assert len(remaining) == 3
    assert all("Elm Street" not in json.dumps(item) for item in remaining)

    # a SOFT forget is not an erasure and must leave the transcript alone
    before = db.execute("SELECT count(*) FROM agent_messages").fetchone()[0]
    db.forget(keep.memory_id)
    assert db.execute("SELECT count(*) FROM agent_messages").fetchone()[0] == before


# ------------------------------------------------------- erasure across EVERY bundled table


def _every_column(db) -> dict[str, list[str]]:
    """Every table in the database and its columns, **from the catalog**, schema-qualified.

    Deliberately not a hard-coded list: the point of the test below is to catch a table nobody
    remembered to cover, and a list written by hand can only ever contain the tables somebody
    remembered.  ``duckdb_columns()`` sees the memory graph, the four anatid_fts_* tables, the
    fts extension's own ``fts_main_anatid_fts_documents`` schema and whatever an integration
    created, all the same way.
    """
    rows = db.execute(
        "SELECT schema_name, table_name, column_name FROM duckdb_columns() "
        "WHERE NOT internal ORDER BY schema_name, table_name, column_index").fetchall()
    out: dict[str, list[str]] = {}
    for schema_name, table_name, column_name in rows:
        out.setdefault(f'"{schema_name}"."{table_name}"', []).append(column_name)
    return out


def _bare_names(names) -> set[str]:
    """``{'"main"."memories"': 1}`` -> ``{"memories"}`` for readable assertions."""
    return {n.split(".")[-1].strip('"') for n in names}


def _tables_containing(db, needles: list[str]) -> dict[str, int]:
    """``{table: matching rows}`` for every table with a row that mentions any needle anywhere.

    Every column is cast to VARCHAR and substring-matched, so this looks inside numeric id
    columns, JSON blobs, the FLOAT[] embedding and the BM25 source text alike -- an erased id
    hiding in ``fts_doc_id`` as ``'1:883...'`` is still a hit.
    """
    found: dict[str, int] = {}
    for table, columns in _every_column(db).items():
        if not columns:
            continue
        tests = " OR ".join(
            f"contains(coalesce(CAST(\"{c}\" AS VARCHAR), ''), ?)" for _ in needles
            for c in columns)
        params = [n for n in needles for _ in columns]
        count = db.execute(f'SELECT count(*) FROM {table} WHERE {tests}', params).fetchone()[0]
        if count:
            found[table] = int(count)
    return found


def test_a_hard_forget_leaves_no_trace_of_the_memory_in_any_table(tmp_path):
    """THE DEFECT: ``forget(hard=True)`` left the erased id *and* its verbatim text behind.

    ``AnatidSession`` covered ``agent_messages``.  Nothing covered ``agent_run_states``, which
    holds a whole serialised ``RunState`` -- the conversation so far, the pending tool calls and
    their arguments -- so a purge returned a receipt saying the memory was erased while
    ``store.load(run_id)`` still handed the id and the text back, ready to be replayed into a
    model days later.

    The test the review asked for: after a hard forget, scan **every table in the database**,
    enumerated from the catalog rather than from a list in this file, for the memory id and the
    exact content string, and assert zero rows anywhere.
    """
    secret = "Ada's passport number is X9981-DO-NOT-KEEP"
    path = tmp_path / "erasure.anatid"
    with Anatid.open(path, tenant=1, embedding_dim=DIM) as db:
        session = AnatidSession("conv-erasure", db)
        store = RunStateStore(db)

        # No `entities=` on the target on purpose, and it is not incidental: `remember` stamps
        # the episode_id onto the entity rows it creates, so an entity that outlives the memory
        # still cites the episode and `forget(hard=True)` -- which deletes an episode only when
        # nothing cites it -- leaves the raw source text in `episodes`. Measured, and reported
        # to the orchestrator: it is a hole in verbs.forget, not in the integration tables this
        # test owns, and it is not this test's to paper over. With the episode orphaned, the
        # purge takes it, which is what the scan below then proves.
        target = db.remember(secret, episode=f"user said: {secret}", writer=session.writer)
        # `keeper` is written AFTER `target` on purpose, so that the BM25 watermark
        # `anatid_meta.fts_indexed_max_id` (the largest memory_id at index time) is keeper's id,
        # not the erased one.  When the erased memory IS the newest indexed row that watermark
        # keeps its id -- a core-table hole this file does not own, pinned by
        # test_erasing_the_newest_indexed_memory_leaves_its_id_in_the_fts_watermark below.
        keeper = db.remember("Ada likes DuckDB", entities=["Ada"], writer=session.writer)
        mid = target.memory_id

        # a transcript that quotes the content (the call) and the id (the result)
        run(session.add_items([
            user("remember my passport number"),
            {"type": "function_call", "call_id": "c1", "name": "anatid_remember",
             "arguments": json.dumps({"content": secret, "entities": ["Ada"]})},
            {"type": "function_call_output", "call_id": "c1",
             "output": json.dumps({"memory_id": mid, "content": secret})},
            assistant("Saved."),
        ]))

        # a parked approval whose serialised state quotes both -- the table with no hook
        state = json.dumps({
            "$schemaVersion": "test", "currentTurn": 2,
            "generatedItems": [
                {"type": "function_call_output", "call_id": "c1",
                 "output": {"memory_id": mid, "content": secret}},
            ],
        })
        run_id = store.save(state, session_id="conv-erasure", agent_name="assistant",
                            pending_tools=["anatid_forget"], metadata={"quoted": secret})

        # and the BM25 index, whose source table keeps the content verbatim
        db.rebuild_fts_index()

        needles = [str(mid), secret]
        before = _bare_names(_tables_containing(db, needles))
        # the copies are real, and in more than one place, before the purge
        assert {"memories", "agent_messages", "anatid_fts_documents"} <= before, before
        assert "agent_run_states" in before, "the fixture did not reproduce the defect"
        assert store.load(run_id) is not None

        receipt = db.forget(mid, hard=True)
        assert receipt.hard is True
        assert receipt.memories_deleted == 1
        assert receipt.extra_rows_deleted >= 3        # 2 transcript rows + the parked run

        after = _tables_containing(db, needles)
        assert after == {}, (
            f"hard forget left the erased id or its content in {sorted(after)}; "
            f"every table in the file was scanned: {sorted(_every_column(db))}")

        # the erasure is not a table drop: everything else is still there and still works
        assert store.load(run_id) is None
        assert db.get(keeper.memory_id) is not None
        assert run(session.get_items()) != []
        assert db.execute("SELECT count(*) FROM agent_messages").fetchone()[0] == 2
        assert db.stats()["memories"] == 1


def test_erasing_the_newest_indexed_memory_also_clears_the_fts_watermark(tmp_path):
    """The same catalog-wide scan, with the erased memory as the NEWEST indexed row.

    Found while writing the test above: every integration table was clean, but the core
    ``anatid_meta.fts_indexed_max_id`` watermark -- "the largest memory_id the BM25 index has
    seen" -- is by construction equal to the newest memory's id, and an early 0.1.1 draft did
    not clamp it when that memory was erased.  ``forget(hard=True)`` now clamps it to the
    largest id still in the index, and this test holds the whole file to zero leftovers.
    """
    secret = "Ada's passport number is X9981-DO-NOT-KEEP"
    with Anatid.open(tmp_path / "erasure-newest.anatid", tenant=1, embedding_dim=DIM) as db:
        session = AnatidSession("conv-newest", db)
        store = RunStateStore(db)
        db.remember("Ada likes DuckDB", entities=["Ada"], writer=session.writer)
        target = db.remember(secret, episode=f"user said: {secret}", writer=session.writer)
        mid = target.memory_id
        run(session.add_items([
            {"type": "function_call_output", "call_id": "c1",
             "output": json.dumps({"memory_id": mid, "content": secret})},
        ]))
        store.save(json.dumps({"quoted": {"memory_id": mid, "content": secret}}),
                   session_id="conv-newest")
        db.rebuild_fts_index()
        needles = [str(mid), secret]
        assert {"memories", "agent_messages", "agent_run_states", "anatid_fts_documents"} <= \
            _bare_names(_tables_containing(db, needles))

        db.forget(mid, hard=True)

        after = _tables_containing(db, needles)
        integration_leftovers = {t for t in _bare_names(after) if t.startswith("agent_")}
        assert integration_leftovers == set(), (
            f"an integration table kept the erased id or content: {sorted(after)}")

        watermark = db.execute(
            "SELECT fts_indexed_max_id FROM anatid_meta").fetchone()[0]
        assert watermark != mid
        assert after == {}, f"hard forget left the erased id or its content in {sorted(after)}"


def test_a_hard_forget_from_another_handle_still_erases_the_integration_tables(tmp_path):
    """Erasure hooks live on a handle; the file does not care which handle asks.

    The break the verifier found in the first 0.1.1 draft: build the session and the store in
    one process (hooks registered on *that* handle), then issue ``forget(hard=True)`` from a
    plain ``Anatid.open`` of the same file -- what a maintenance script, a second process or the
    ``anatid-mcp`` ``forget`` tool does.  The draft deleted nothing from ``agent_messages`` or
    ``agent_run_states`` and reported ``extra_rows_deleted=0``.  Now ``forget`` finds the bundled
    integration tables in the catalog itself (``anatid.erasure.BUNDLED_INTEGRATION_TABLES``) and
    the receipt counts what it removed from them.
    """
    from anatid.erasure import BUNDLED_INTEGRATION_TABLES

    secret = "Bee's door code is 4471-KEEP-OUT"
    path = tmp_path / "cross-handle.anatid"
    with Anatid.open(path, tenant=1, embedding_dim=DIM) as writer:
        session = AnatidSession("conv-cross", writer)
        store = RunStateStore(writer)
        target = writer.remember(secret, episode=f"user said: {secret}", writer=session.writer)
        mid = target.memory_id
        run(session.add_items([
            {"type": "function_call", "call_id": "c1", "name": "anatid_remember",
             "arguments": json.dumps({"content": secret})},
            {"type": "function_call_output", "call_id": "c1",
             "output": json.dumps({"memory_id": mid, "content": secret})},
            assistant("Saved."),
        ]))
        run_id = store.save(json.dumps({"quoted": {"memory_id": mid, "content": secret}}),
                            session_id="conv-cross")
        store.save(json.dumps({"quoted": "nothing to do with it"}), session_id="conv-cross")
        writer.rebuild_fts_index()
        assert len(writer.erasure_hooks) == 4

    needles = [str(mid), secret]
    with Anatid.open(path, tenant=1, embedding_dim=DIM) as other:
        assert other.erasure_hooks == []                       # nothing registered here
        before = _bare_names(_tables_containing(other, needles))
        assert {"memories", "agent_messages", "agent_run_states"} <= before, before

        receipt = other.forget(mid, hard=True)

        assert receipt.extra_rows_deleted == 3                 # 2 transcript rows + 1 run state
        after = _tables_containing(other, needles)
        assert after == {}, (
            f"a hard forget from a handle without hooks left the erased id or its content in "
            f"{sorted(after)}")
        # the pass is a purge of matching rows, not a table drop
        assert other.execute("SELECT count(*) FROM agent_messages").fetchone()[0] == 1
        assert other.execute("SELECT count(*) FROM agent_run_states").fetchone()[0] == 1
        assert set(BUNDLED_INTEGRATION_TABLES) <= _bare_names(_every_column(other))

    # and the handle that DID register hooks does not count the same rows twice
    with Anatid.open(path, tenant=1, embedding_dim=DIM) as again:
        session = AnatidSession("conv-cross", again)
        store = RunStateStore(again)
        m2 = again.remember("Bee's second secret", writer=session.writer)
        run(session.add_items([
            {"type": "function_call_output", "call_id": "c2",
             "output": json.dumps({"memory_id": m2.memory_id, "content": "Bee's second secret"})},
        ]))
        store.save(json.dumps({"memory_id": m2.memory_id}), session_id="conv-cross")
        receipt = again.forget(m2.memory_id, hard=True)
        assert receipt.extra_rows_deleted == 2
        assert store.load(run_id) is None
        assert _tables_containing(again, [str(m2.memory_id), "Bee's second secret"]) == {}


def test_every_bundled_integration_table_has_an_erasure_hook(db):
    """A hook per table, registered once however many sessions and stores share the handle."""
    from anatid.integrations.erasure import TableErasureHook

    AnatidSession("conv-a", db)
    AnatidSession("conv-b", db)                       # a second session must not double-register
    RunStateStore(db)
    RunStateStore(db, tenant=2)

    hooked = {h.table for h in db.erasure_hooks if isinstance(h, TableErasureHook)}
    integration_tables = {t for t in _bare_names(_every_column(db)) if t.startswith("agent_")}
    assert integration_tables == {"agent_messages", "agent_sessions", "agent_turn_usage",
                                  "agent_run_states"}
    assert integration_tables <= hooked, f"no erasure hook for {integration_tables - hooked}"
    assert len(db.erasure_hooks) == len(hooked), "a hook was registered twice"
    # and the list core purges by name on every handle is exactly these tables
    from anatid.erasure import BUNDLED_INTEGRATION_TABLES

    assert set(BUNDLED_INTEGRATION_TABLES) == integration_tables


def test_the_run_state_hook_is_scoped_to_the_tenant_and_leaves_other_runs_alone(db):
    store = RunStateStore(db, tenant=1)
    other_tenant = RunStateStore(db, tenant=2)

    m = db.remember("Bee keeps bees in Bristol", tenant=1)
    quoting = store.save(json.dumps({"note": "Bee keeps bees in Bristol"}))
    unrelated = store.save(json.dumps({"note": "nothing to do with it"}))
    elsewhere = other_tenant.save(json.dumps({"note": "Bee keeps bees in Bristol"}))

    db.forget(m.memory_id, hard=True, tenant=1)

    assert store.load(quoting) is None                # erased
    assert store.load(unrelated) is not None          # untouched
    assert other_tenant.load(elsewhere) is not None   # a different tenant is a different file's
    #                                                   worth of data; scoping is by tenant_id


def test_session_can_own_its_own_database(tmp_path):
    path = tmp_path / "owned.anatid"
    with AnatidSession("owned", path=str(path), tenant=4, embedding_dim=DIM) as owned:
        run(owned.add_items([user("hello")]))
        assert owned.tenant_id == 4
    assert path.exists()
    with Anatid.open(path, tenant=4, embedding_dim=DIM) as reopened:
        assert reopened.execute("SELECT count(*) FROM agent_messages").fetchone()[0] == 1
    with pytest.raises(ValueError):
        AnatidSession("bad", Anatid.open(":memory:"), path="x")


# --------------------------------------------------------------------------- tools & approvals

def test_writes_are_approval_gated_and_reads_are_not(db):
    tools = {t.name: t for t in create_memory_tools(db)}
    assert set(tools) == {"anatid_remember", "anatid_recall", "anatid_context",
                          "anatid_supersede", "anatid_forget", "anatid_provenance"}
    for name in ("anatid_remember", "anatid_supersede", "anatid_forget"):
        assert callable(tools[name].needs_approval), f"{name} must be approval-gated"
    for name in ("anatid_recall", "anatid_context", "anatid_provenance"):
        assert tools[name].needs_approval is False, f"{name} must not be gated"


def test_the_approval_policy_is_consulted_with_the_call(db):
    seen: list[ApprovalRequest] = []

    def policy(request: ApprovalRequest) -> bool:
        seen.append(request)
        return request.tool_name != "anatid_remember"

    tools = {t.name: t for t in create_memory_tools(db, approval_policy=policy)}
    verdict = run(tools["anatid_remember"].needs_approval(
        "ctx-sentinel", {"content": "Ada likes DuckDB"}, "call-42"))

    assert verdict is False                                  # policy let it through
    assert len(seen) == 1
    assert seen[0].tool_name == "anatid_remember"
    assert seen[0].arguments == {"content": "Ada likes DuckDB"}
    assert seen[0].call_id == "call-42"
    assert seen[0].context == "ctx-sentinel"

    assert run(tools["anatid_supersede"].needs_approval("c", {}, "id")) is True


def test_default_policy_requires_approval_for_every_write(db):
    tools = {t.name: t for t in create_memory_tools(db)}
    for name in ("anatid_remember", "anatid_supersede", "anatid_forget"):
        assert run(tools[name].needs_approval(None, {}, "id")) is True
    assert always_require_approval(ApprovalRequest("anatid_remember", {}, "id")) is True


def test_hard_forget_always_needs_a_human(db):
    """Even a policy that approves everything cannot auto-approve an erasure."""
    tools = {t.name: t for t in create_memory_tools(db, approval_policy=never_require_approval)}
    forget = tools["anatid_forget"]
    assert run(forget.needs_approval(None, {"memory_id": 1, "hard": False}, "i")) is False
    assert run(forget.needs_approval(None, {"memory_id": 1, "hard": True}, "i")) is True

    opted_out = {t.name: t for t in create_memory_tools(
        db, approval_policy=never_require_approval, force_approval_for_hard_forget=False)}
    assert run(opted_out["anatid_forget"].needs_approval(
        None, {"memory_id": 1, "hard": True}, "i")) is False


def test_approve_low_risk_policy():
    policy = approve_low_risk()
    assert policy(ApprovalRequest("anatid_remember", {"content": "small"}, "i")) is False
    assert policy(ApprovalRequest("anatid_remember", {"content": "x" * 5000}, "i")) is True
    assert policy(ApprovalRequest("anatid_supersede", {"content": "c"}, "i")) is True
    assert policy(ApprovalRequest("anatid_forget", {"hard": False}, "i")) is True
    assert policy(ApprovalRequest("anatid_forget", {"hard": True}, "i")) is True
    assert approve_low_risk(soft_forget=True)(
        ApprovalRequest("anatid_forget", {"hard": False}, "i")) is False


def test_tool_selection_and_bad_names(db):
    names = [t.name for t in create_memory_tools(db, exclude=["anatid_forget"])]
    assert "anatid_forget" not in names and "anatid_remember" in names
    assert [t.name for t in create_memory_tools(db, include=["anatid_recall"])] == \
        ["anatid_recall"]
    with pytest.raises(ValueError, match="unknown tool"):
        create_memory_tools(db, include=["anatid_teleport"])


def _invoke(tool, **arguments) -> dict:
    ctx = ToolContext(None, tool_name=tool.name, tool_call_id="call-1",
                      tool_arguments=json.dumps(arguments))
    return json.loads(run(tool.on_invoke_tool(ctx, json.dumps(arguments))))


def test_tool_bodies_read_and_write_the_database(db, session):
    tools = {t.name: t for t in create_memory_tools(db, session=session)}

    saved = _invoke(tools["anatid_remember"], content="Ada prefers DuckDB",
                    entities=["Ada"], kind="preference")
    assert saved["saved"] is True and saved["about"] == ["Ada"]
    memory = db.get(saved["memory_id"])
    assert memory is not None and memory.writer == "session:conv-1"

    found = _invoke(tools["anatid_recall"], query="DuckDB", k=5, seed_entity="Ada", hops=2)
    assert saved["memory_id"] in [hit["memory_id"] for hit in found["hits"]]
    assert "graph" in found["arms"]

    about = _invoke(tools["anatid_context"], entity="Ada", limit=5, hops=0)
    assert [m["memory_id"] for m in about["memories"]] == [saved["memory_id"]]

    corrected = _invoke(tools["anatid_supersede"], memory_id=saved["memory_id"],
                        content="Ada prefers DuckDB over SQLite", entities=None, kind=None)
    chain = _invoke(tools["anatid_provenance"], memory_id=corrected["memory_id"])
    assert [m["memory_id"] for m in chain["chain"]] == [corrected["memory_id"],
                                                        saved["memory_id"]]

    receipt = _invoke(tools["anatid_forget"], memory_id=corrected["memory_id"], hard=True,
                      reason="test erasure")
    assert receipt["hard"] is True and receipt["rows_removed"] > 0
    assert db.get(corrected["memory_id"]) is None


def test_tool_errors_come_back_as_json_not_exceptions(db):
    tools = {t.name: t for t in create_memory_tools(db)}
    answer = _invoke(tools["anatid_context"], entity="Nobody At All", limit=5, hops=0)
    assert answer["error"] == "NotFoundError"
    unknown_seed = _invoke(tools["anatid_recall"], query="anything", k=3,
                           seed_entity="Nobody At All", hops=2)
    assert unknown_seed["hits"] == []
    assert any("unknown entity" in note for note in unknown_seed["notes"])


def test_tools_use_the_agent_name_as_writer_when_no_session(db):
    tools = {t.name: t for t in create_memory_tools(db)}
    agent = Agent(name="archivist", model=ScriptedModel([]))
    ctx = ToolContext(None, tool_name="anatid_remember", tool_call_id="c", tool_arguments="{}",
                      agent=agent)
    payload = json.dumps({"content": "written by an agent", "entities": None, "kind": None})
    saved = json.loads(run(tools["anatid_remember"].on_invoke_tool(ctx, payload)))
    assert db.get(saved["memory_id"]).writer == "agent:archivist"


# --------------------------------------------------------------------------- run-state store

FAKE_STATE = json.dumps({"$schemaVersion": "test", "currentTurn": 1, "note": "not a real state"})


def test_run_state_round_trip_persistence(db):
    store = RunStateStore(db)
    run_id = store.save(FAKE_STATE, session_id="conv-1", agent_name="assistant",
                        pending_tools=["anatid_remember"], metadata={"build": "abc123"})

    assert store.load(run_id) == FAKE_STATE
    stored = store.get(run_id)
    assert stored.status == "pending_approval"
    assert stored.is_pending
    assert stored.agent_name == "assistant"
    assert stored.session_id == "conv-1"
    assert stored.pending_tools == ("anatid_remember",)
    assert stored.metadata == {"build": "abc123"}

    assert [r.run_id for r in store.pending()] == [run_id]
    assert [r.run_id for r in store.pending(session_id="other")] == []

    assert store.mark_resolved(run_id, status="approved") is True
    assert store.get(run_id).status == "approved"
    assert store.pending() == []
    assert store.mark_resolved("run_nope") is False


def test_run_state_store_accepts_a_runstate_object(db):
    store = RunStateStore(db)
    fake = types.SimpleNamespace(to_string=lambda: FAKE_STATE,
                                 _current_agent=types.SimpleNamespace(name="assistant"))
    run_id = store.save(fake)
    assert store.load(run_id) == FAKE_STATE
    assert store.get(run_id).agent_name == "assistant"
    with pytest.raises(TypeError):
        store.save(object())


def test_saving_again_keeps_the_previous_version(db):
    store = RunStateStore(db)
    run_id = store.save(FAKE_STATE)
    store.save('{"second": true}', run_id=run_id, status="re-asked")

    assert store.load(run_id) == '{"second": true}'
    history = store.history(run_id)
    assert len(history) == 2                                    # nothing was overwritten
    assert [h.status for h in history] == ["pending_approval", "re-asked"]
    live = db.execute("SELECT count(*) FROM agent_run_states "
                      "WHERE valid_to IS NULL AND tx_to IS NULL").fetchone()[0]
    assert live == 1                                            # exactly one current row


def test_delete_soft_then_hard(db):
    store = RunStateStore(db)
    run_id = store.save(FAKE_STATE)
    assert store.delete(run_id) == 1
    assert store.load(run_id) is None
    assert db.execute("SELECT count(*) FROM agent_run_states").fetchone()[0] == 1  # still there
    assert store.delete(run_id, hard=True) == 1
    assert db.execute("SELECT count(*) FROM agent_run_states").fetchone()[0] == 0


def test_resume_of_an_unknown_run_raises(db):
    from anatid.errors import NotFoundError

    store = RunStateStore(db)
    agent = Agent(name="assistant", model=ScriptedModel([]))
    with pytest.raises(NotFoundError):
        run(store.resume(agent, "run_does_not_exist"))


def test_two_tenants_do_not_see_each_others_runs(db):
    a, b = RunStateStore(db, tenant=1), RunStateStore(db, tenant=2)
    run_id = a.save(FAKE_STATE)
    assert a.load(run_id) == FAKE_STATE
    assert b.load(run_id) is None
    assert b.pending() == []


# --------------------------------------------------------------------------- end to end (HITL)

def build_agent(db, *, session=None, policy=None):
    """An agent whose only tool is the approval-gated ``anatid_remember``."""
    model = ScriptedModel([
        [function_call("anatid_remember",
                       {"content": "Ada prefers DuckDB", "entities": ["Ada"], "kind": "preference"},
                       call_id="call-1")],
        [assistant_message("Noted.")],
    ])
    tools = create_memory_tools(db, session=session, approval_policy=policy,
                               include=["anatid_remember"])
    return Agent(name="assistant", instructions="Remember what the user tells you.",
                 model=model, tools=tools)


def test_interrupted_run_is_parked_in_duckdb_and_resumed_by_another_process(db, session):
    """The headline path, offline: interrupt -> persist -> approve later -> resume -> write."""
    agent = build_agent(db, session=session)
    store = RunStateStore(db)

    result = run(Runner.run(agent, "remember that Ada prefers DuckDB", session=session,
                            run_config=NO_TRACE))
    assert [i.tool_name for i in result.interruptions] == ["anatid_remember"]
    assert db.execute("SELECT count(*) FROM memories").fetchone()[0] == 0, \
        "the tool body must not run before approval"

    run_id = store.save_result(result, session_id=session.session_id)
    assert store.get(run_id).pending_tools == ("anatid_remember",)

    # --- a different process, later: it has only the anatid file and the run id.
    reloaded = RunStateStore(db)
    resumed_state = run(reloaded.resume(agent, run_id))
    assert isinstance(resumed_state, RunState)
    interruptions = resumed_state.get_interruptions()
    assert [i.tool_name for i in interruptions] == ["anatid_remember"]
    for item in interruptions:
        resumed_state.approve(item)

    final = run(Runner.run(agent, resumed_state, session=session, run_config=NO_TRACE))
    assert final.final_output == "Noted."
    reloaded.mark_resolved(run_id, status="approved")

    rows = db.execute("SELECT content, writer FROM memories").fetchall()
    assert rows == [("Ada prefers DuckDB", "session:conv-1")]
    assert reloaded.get(run_id).status == "approved"
    assert reloaded.pending() == []

    # the conversation went into the same file, and joins to what it produced
    assert len(run(session.get_items())) > 0
    assert [m.content for m in run(session.memories_written_here())] == ["Ada prefers DuckDB"]


def test_rejecting_the_interruption_writes_nothing(db, session):
    agent = build_agent(db, session=session)
    store = RunStateStore(db)

    result = run(Runner.run(agent, "remember that Ada prefers DuckDB", session=session,
                            run_config=NO_TRACE))
    run_id = store.save_result(result)
    state = run(store.resume(agent, run_id))
    for item in state.get_interruptions():
        state.reject(item)

    run(Runner.run(agent, state, session=session, run_config=NO_TRACE))
    store.mark_resolved(run_id, status="rejected")

    assert db.execute("SELECT count(*) FROM memories").fetchone()[0] == 0
    assert store.get(run_id).status == "rejected"


def test_an_auto_approved_write_runs_without_interruption(db, session):
    """``approve_low_risk`` lets a plain remember through -- no human, no interruption."""
    agent = build_agent(db, session=session, policy=approve_low_risk())
    result = run(Runner.run(agent, "remember that Ada prefers DuckDB", session=session,
                            run_config=NO_TRACE))
    assert result.interruptions == []
    assert result.final_output == "Noted."
    assert db.execute("SELECT content FROM memories").fetchone() == ("Ada prefers DuckDB",)


def test_state_string_survives_a_real_serialise_deserialise_cycle(db, session):
    """``RunState.to_string()`` -> DuckDB VARCHAR -> ``RunState.from_string()``, byte for byte."""
    agent = build_agent(db, session=session)
    store = RunStateStore(db)
    result = run(Runner.run(agent, "remember that", session=session, run_config=NO_TRACE))

    original = result.to_state().to_string()
    run_id = store.save(original)
    assert store.load(run_id) == original
    assert json.loads(store.load(run_id))["$schemaVersion"]

    rebuilt = run(RunState.from_string(agent, store.load(run_id)))
    assert [i.tool_name for i in rebuilt.get_interruptions()] == ["anatid_remember"]
