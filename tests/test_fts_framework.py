"""Full-text search on the derived-index framework.

Every property the design document asks of an accelerator, checked for the BM25 arm against the
oracle that :mod:`anatid.fts` itself falls back to: :func:`anatid.fts.scan`, an exact BM25 over
the tenant's visible documents with no index at all.  The merged answer (published generation +
journal) has to reproduce the oracle's ROWS exactly and its ORDER wherever the scores are not
tied, after arbitrary sequences of writes, corrections, purges and rebuilds.

The three things this file is really about:

* **immediacy** -- a write, a supersession or a forget is in the next search with no rebuild, on
  the handle that wrote it and on a second handle that has never heard of the index;
* **identity** -- the document key is ``(tenant_id, memory_id)`` everywhere, so the 0.1.1
  cross-tenant leak (memory 4242 in two tenants) stays fixed and one tenant's journal never
  touches another's documents;
* **honesty** -- when a generation cannot be used the search says which
  :class:`~anatid.derived.HealthReason` applies and still answers correctly.
"""

from __future__ import annotations

import datetime as _dt
import random
import threading

import duckdb
import pytest

from anatid import Anatid, AsOf, IndexValidationError, MaintenancePolicy
from anatid import fts
from anatid import schema as S
from anatid.derived import HealthReason, NullIndex, maintain
from anatid.errors import StaleIndexError

from conftest import DIM, T0

MINUTE = _dt.timedelta(minutes=1)
HOUR = _dt.timedelta(hours=1)


# --------------------------------------------------------------------------- helpers


@pytest.fixture
def file_db(legacy_file_db):
    """A handle that has NOT attached the derived index.

    ``Anatid.open(accelerators=False)``, so this file decides for itself which half is in play:
    the :func:`idx` fixture attaches, and the tests of the 0.1.1 half (and of what ``attach``
    itself changes) need a handle where nothing has.
    """
    return legacy_file_db


@pytest.fixture
def idx(file_db):
    """An on-disk database with the derived full-text index attached and nothing built."""
    return fts.attach(file_db)


def ids(hits) -> list[int]:
    return [int(m) for m, _score in hits]


def oracle(db, query: str, *, tenant: int = 1, topn: int = 50, as_of=None) -> list[int]:
    """The SQL path: exact BM25 over the tenant's visible documents, no index at all."""
    return ids(
        fts.scan(
            db.connection,
            tenant_id=tenant,
            query_text=query,
            topn=topn,
            as_of=as_of or fts.CURRENT,
        )
    )


def found(db, query: str, *, tenant: int = 1, topn: int = 50, as_of=None, **kw):
    return fts.search(
        db.connection,
        tenant_id=tenant,
        query_text=query,
        topn=topn,
        as_of=as_of or fts.CURRENT,
        **kw,
    )


def tables(db) -> set[str]:
    return S.table_names(db.connection)


def journal(db) -> list[tuple]:
    return [
        (int(r[0]), int(r[1]), str(r[2]), r[3])
        for r in db.execute(
            f"SELECT tenant_id, doc_id, op, absorbed_by FROM {S.INDEX_JOURNAL_TABLE} "
            f"WHERE index_name = ? ORDER BY change_seq",
            [fts.FTS_INDEX_NAME],
        ).fetchall()
    ]


def find_in_file(db, *needles: str) -> list[tuple[str, str, str]]:
    """Every ``(schema.table, column, needle)`` in the whole file whose text holds a needle.

    Enumerated from the catalog rather than from a list anatid keeps, so a generation's storage
    is scanned even though no fixed name knows about it.
    """
    hits: list[tuple[str, str, str]] = []
    objects = db.execute(
        "SELECT table_schema, table_name FROM information_schema.tables ORDER BY 1, 2"
    ).fetchall()
    for schema_name, table_name in objects:
        if schema_name in ("information_schema", "pg_catalog"):
            continue
        qualified = f'"{schema_name}"."{table_name}"'
        try:
            columns = [r[0] for r in db.execute(f"DESCRIBE {qualified}").fetchall()]
        except duckdb.Error:  # pragma: no cover - a view over something dropped
            continue
        for column in columns:
            for needle in needles:
                sql = (
                    f"SELECT count(*) FROM {qualified} "
                    f'WHERE contains(CAST("{column}" AS VARCHAR), ?)'
                )
                try:
                    n = db.execute(sql, [needle]).fetchone()[0]
                except duckdb.Error:  # pragma: no cover - a type with no VARCHAR cast
                    continue
                if n:
                    hits.append((f"{schema_name}.{table_name}", column, needle))
    return hits


# --------------------------------------------------------------------------- attachment


def test_attach_replaces_the_placeholder_and_records_the_definition_in_the_file(file_db):
    db = file_db
    assert isinstance(db.indexes["fts"], NullIndex)
    index = fts.attach(db)
    assert db.indexes["fts"] is index
    definition = db.indexes.definitions()["fts"]
    assert definition.source_table == "memories"
    assert definition.source_id_column == "memory_id"
    assert definition.per_tenant is False  # one generation covers the file
    assert definition.delta_mode == "table"  # exact for explicit ids, not only minted ones
    assert definition.supports_delta is True
    assert definition.enabled
    assert definition.params["statistics"] == "per tenant"
    assert definition.params["doc_key"] == "tenant_id:memory_id"
    # idempotent
    assert fts.attach(db) is index
    assert fts.index_of(db) is index


def test_every_write_verb_journals_inside_its_own_transaction(idx, file_db):
    db = file_db
    a = db.remember("Ada drinks espresso", now=T0).memory_id
    b = db.remember("Bob drinks tea", now=T0).memory_id
    c = db.supersede(a, "Ada drinks decaf", now=T0 + MINUTE).memory_id
    db.forget(b, now=T0 + 2 * MINUTE)
    assert journal(db) == [
        (1, a, "insert", None),
        (1, b, "insert", None),
        (1, c, "insert", None),
        (1, a, "close", None),
        (1, b, "close", None),
    ]
    # and nothing is journalled for a table no index derives from
    db.relate("Ada", "Bob", now=T0)
    assert [row for row in journal(db) if row[1] not in (a, b, c)] == []


def test_a_write_is_journalled_inside_the_callers_transaction_and_rolls_back_with_it(idx, file_db):
    db = file_db
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.remember("Ada drinks espresso", now=T0)
            assert len(journal(db)) == 1
            raise RuntimeError("boom")
    assert journal(db) == []
    assert found(db, "espresso").hits == []


# --------------------------------------------------------------------------- immediacy


def test_a_write_is_searchable_by_the_very_next_recall_with_no_rebuild(idx, file_db):
    """The property the whole framework exists for.  No rebuild anywhere in this test."""
    db = file_db
    a = db.remember("Ada drinks espresso", now=T0).memory_id
    assert ids(found(db, "espresso").hits) == [a]
    assert [h.memory.memory_id for h in db.recall("espresso", k=5)] == [a]

    # ... and with a generation published, the next write is still immediate
    fts.rebuild(idx, now=T0)
    b = db.remember("decaf espresso, still espresso", now=T0 + MINUTE).memory_id
    result = found(db, "espresso")
    assert set(ids(result.hits)) == {a, b}
    assert result.backend == "generation" and result.exact
    assert result.journal_rows == 1
    assert next(h.memory.memory_id for h in db.recall("espresso", k=5)) == b


def test_a_write_from_a_second_handle_is_searchable_on_both_with_no_rebuild(idx, file_db):
    """P0-1 for the text arm: the definition is in the FILE, so a handle holding no full-text
    code journals for it, and a handle that never attached can still search the generation."""
    db = file_db
    a = db.remember("Ada drinks espresso", now=T0).memory_id
    fts.rebuild(idx, now=T0)

    other = Anatid.open(db.path, tenant=1, embedding_dim=DIM, ensure=False, accelerators=False)
    try:
        assert isinstance(other.indexes["fts"], NullIndex)  # no implementation here
        assert other.indexes.definitions()["fts"].enabled  # but the file has the definition
        b = other.remember("espresso machine descaling", now=T0 + MINUTE).memory_id
        # visible on the writing handle, which holds no FtsIndex at all
        assert set(ids(found(other, "espresso").hits)) == {a, b}
        assert found(other, "espresso").backend == "generation"
    finally:
        other.close()
    # and on the handle that built the generation
    assert set(ids(found(db, "espresso").hits)) == {a, b}


def test_a_superseded_memory_disappears_from_text_results_immediately(idx, file_db):
    db = file_db
    a = db.remember("Ada drinks espresso", now=T0).memory_id
    fts.rebuild(idx, now=T0)
    assert ids(found(db, "espresso").hits) == [a]

    b = db.supersede(a, "Ada drinks tea now", now=T0 + MINUTE).memory_id
    assert ids(found(db, "espresso").hits) == []  # gone at once, no rebuild
    assert ids(found(db, "tea").hits) == [b]  # and the successor is findable
    assert oracle(db, "espresso") == [] and oracle(db, "tea") == [b]


def test_a_forgotten_memory_disappears_immediately_and_prune_does_too(idx, file_db):
    db = file_db
    a = db.remember("Ada drinks espresso", now=T0).memory_id
    b = db.remember("Bob drinks espresso", now=T0).memory_id
    fts.rebuild(idx, now=T0)

    db.forget(a, now=T0 + MINUTE)
    assert ids(found(db, "espresso").hits) == [b]
    db.prune(older_than=T0 + 2 * MINUTE, dry_run=False, now=T0 + 3 * MINUTE)
    assert ids(found(db, "espresso").hits) == []
    assert oracle(db, "espresso") == []


def test_a_correction_that_changes_no_content_costs_the_index_nothing(idx, file_db):
    """``reinforce`` writes a new version (a confidence change is a correction) but no journal
    row, because every version of a memory carries the same ``content`` and the document the
    index holds is unchanged.  The memory therefore stays in the base and stays findable."""
    db = file_db
    a = db.remember("Ada drinks espresso", now=T0).memory_id
    fts.rebuild(idx, now=T0)
    assert journal(db) == []  # absorbed by the build

    db.reinforce(a, confidence=0.9, now=T0 + MINUTE)
    assert journal(db) == []
    assert len(db.versions(a)) == 2
    assert ids(found(db, "espresso").hits) == [a] == oracle(db, "espresso")
    assert db.get(a).confidence == pytest.approx(0.9)


def test_a_tombstone_narrows_the_base_and_never_the_answer(idx, file_db):
    """The rule the merge turns on.  A journalled ``close`` means "the base's view of this
    document is out of date", NOT "this document is gone": whether it survives is the
    visibility predicate's decision, on the canonical row, and for an ``as_of`` read a closed
    document is a perfectly good candidate.  Subtracting tombstones from the ANSWER would lose
    those rows; subtracting them from the BASE and rescoring from the canonical row is exact
    either way.

    A journal row is written by hand here on purpose: it is the merge's contract that is under
    test, not the verb that happens to produce such a row today.
    """
    db = file_db
    a = db.remember("Ada drinks espresso", now=T0).memory_id
    fts.rebuild(idx, now=T0)
    db.execute(
        f"INSERT INTO {S.INDEX_JOURNAL_TABLE} (index_name, tenant_id, doc_id, change_seq, op, "
        f"written_at, reason, absorbed_by) "
        f"VALUES (?, 1, ?, nextval('{S.INDEX_CHANGE_SEQUENCE}'), 'close', ?, 'by hand', NULL)",
        [fts.FTS_INDEX_NAME, a, T0 + MINUTE],
    )
    assert [(t, d, op) for t, d, op, _ in journal(db)] == [(1, a, "close")]
    plan = fts.resolve(db.connection, tenant_id=1)
    assert plan.usable and plan.journal_rows == 1
    # the memory is still current, so it is still the answer
    assert ids(found(db, "espresso").hits) == [a] == oracle(db, "espresso")


def test_a_purged_and_reused_document_id_comes_back(idx, file_db):
    """P0-2 for the text arm.  Two independent id sets cannot represent this: the id would be
    in both the delta and the tombstones and the merge would drop a row the SQL path returns."""
    db = file_db
    a = db.remember("Ada drinks espresso", memory_id=4242, now=T0).memory_id
    fts.rebuild(idx, now=T0)
    db.forget(a, hard=True, now=T0 + MINUTE)
    assert ids(found(db, "espresso").hits) == []
    again = db.remember("a different espresso entirely", memory_id=4242, now=T0 + 2 * MINUTE)
    assert again.memory_id == 4242
    assert ids(found(db, "espresso").hits) == [4242] == oracle(db, "espresso")


# --------------------------------------------------------------------------- the oracle


def test_the_merged_answer_matches_the_oracle_and_a_rebuild_after_random_mutations(file_db):
    """The primary acceptance test, and it checks two different things.

    **Rows**: base + journal returns exactly what the SQL path returns, for a random sequence of
    writes, supersessions, forgets, hard purges and reinforcements with rebuilds interleaved.
    The oracle is :func:`anatid.fts.scan`, which uses no index at all.

    **Scores**: the answer with a journal is the answer a rebuild would give, to the last bit.
    That is the property that makes the journal a performance question rather than a ranking
    question: the corpus statistics a search reconstructs from base + journal are the ones the
    rebuild computes, so absorbing the journal moves nothing.
    """
    db = file_db
    index = fts.attach(db)
    rng = random.Random(11)
    words = [
        "espresso",
        "coffee",
        "tea",
        "decaf",
        "latte",
        "ada",
        "bob",
        "carol",
        "grind",
        "roast",
        "bean",
        "cup",
    ]
    live: list[int] = []
    now = T0
    rebuild_checks = 0

    def text() -> str:
        return " ".join(rng.choice(words) for _ in range(rng.randint(3, 8)))

    for step in range(120):
        now += MINUTE
        move = rng.random()
        if move < 0.45 or not live:
            live.append(db.remember(text(), now=now).memory_id)
        elif move < 0.60:
            victim = rng.choice(live)
            live.remove(victim)
            live.append(db.supersede(victim, text(), now=now).memory_id)
        elif move < 0.72:
            live.remove(db.forget(rng.choice(live), now=now).memory_id)
        elif move < 0.80:
            live.remove(db.forget(rng.choice(live), hard=True, now=now).memory_id)
        elif move < 0.88:
            db.reinforce(rng.choice(live), confidence=rng.random(), now=now)
        else:
            fts.rebuild(index, now=now)

        query = " ".join(rng.sample(words, 2))
        want = oracle(db, query)
        merged = found(db, query)
        assert merged.exact
        assert set(ids(merged.hits)) == set(want), (
            f"step {step} query {query!r}: merged {sorted(ids(merged.hits))} != oracle "
            f"{sorted(want)}"
        )

        if step % 12 == 11 and index.current_generation() is not None:
            journalled = merged
            fts.rebuild(index, now=now)
            absorbed = found(db, query)
            assert absorbed.journal_rows == 0
            assert ids(absorbed.hits) == ids(journalled.hits), (
                f"step {step}: absorbing {journalled.journal_rows} journal row(s) reordered "
                f"the answer for {query!r}"
            )
            for (_, before), (_, after) in zip(journalled.hits, absorbed.hits):
                assert before == pytest.approx(after, rel=1e-12), (
                    f"step {step}: absorbing the journal changed a score for {query!r}"
                )
            rebuild_checks += 1

    assert rebuild_checks >= 5


def test_the_base_arm_scores_exactly_what_the_oracle_scores_when_every_document_is_visible(
    idx, file_db
):
    """The two corpora coincide when no document has been superseded, and then the generation's
    scores must be bit-for-bit the ones the index-free scan computes."""
    db = file_db
    for text in ("Ada drinks espresso", "Bob drinks tea and espresso", "Carol grinds beans"):
        db.remember(text, now=T0)
    fts.rebuild(idx, now=T0)
    merged = fts.search(db.connection, tenant_id=1, query_text="espresso drinks")
    exact = fts.scan(db.connection, tenant_id=1, query_text="espresso drinks")
    assert merged.journal_rows == 0
    assert merged.hits == exact


def test_the_sql_tokenizer_agrees_with_the_lengths_the_index_recorded(idx, file_db):
    """Both arms divide by a document length: the base's comes from the extension, the journal
    arm's from :func:`anatid.fts.tokens_sql`.  A disagreement would silently skew every score."""
    db = file_db
    for text in ("Ada drinks espresso.", "a b c d e", "Punctuation; and, CAPITALS -- 123"):
        db.remember(text, now=T0)
    fts.rebuild(idx, now=T0)
    generation = idx.current_generation()
    storage = idx.storage(generation)
    rows = db.execute(
        f"SELECT dm.memory_id, dm.len, "
        f"       len(list_filter({fts.tokens_sql('m.content')}, x -> x <> '')) "
        f"FROM {storage.docmap} dm JOIN memories m ON m.memory_id = dm.memory_id "
        f"AND m.tenant_id = dm.tenant_id"
    ).fetchall()
    assert rows and all(int(r[1]) == int(r[2]) for r in rows), rows


def test_a_generation_indexes_the_documents_the_legacy_rebuild_would(idx, file_db):
    """Two independent statements build the document set (``anatid.schema`` for the legacy index,
    :meth:`FtsIndex._build` for a generation).  They must not drift apart."""
    db = file_db
    a = db.remember("Ada drinks espresso", now=T0).memory_id
    db.supersede(a, "Ada drinks decaf", now=T0 + MINUTE)
    db.remember("Bob drinks tea", tenant=1, now=T0)
    fts.rebuild(idx, now=T0 + 2 * MINUTE)
    generation = fts.FtsStorage.of(idx.current_generation())
    fts.legacy_rebuild(db.connection, now=T0 + 2 * MINUTE)
    mine = set(db.execute(f"SELECT fts_doc_id, content FROM {generation.source}").fetchall())
    theirs = set(db.execute(f"SELECT fts_doc_id, content FROM {S.FTS_SOURCE_TABLE}").fetchall())
    assert mine == theirs


# --------------------------------------------------------------------------- generations


def test_rebuild_is_build_next_validate_publish(idx, file_db):
    db = file_db
    db.remember("Ada drinks espresso", now=T0)
    status = fts.rebuild(idx, now=T0)
    first = idx.current_generation()
    assert first.generation == 1 and first.validated and first.published
    assert first.stats["rows"] == 1
    assert status.indexed_rows == 1 and status.pending_rows == 0 and not status.stale

    db.remember("Bob drinks tea", now=T0 + MINUTE)
    fts.rebuild(idx, now=T0 + MINUTE)
    second = idx.current_generation()
    assert second.generation == 2 and second.published
    # built beside the first, then the first is retired by the build after it
    assert [g.generation for g in idx.generations()] == [1, 2]
    assert fts.FtsStorage.of(second).source in tables(db)
    db.remember("Carol grinds beans", now=T0 + 2 * MINUTE)
    fts.rebuild(idx, now=T0 + 2 * MINUTE)
    assert fts.FtsStorage.of(first).source not in tables(db)
    assert [g.generation for g in idx.generations()] == [2, 3]


def test_generations_are_built_beside_the_live_one(idx, file_db):
    """Publication is a metadata switch: the new generation's storage exists before the switch
    and the old generation's storage survives it."""
    db = file_db
    db.remember("Ada drinks espresso", now=T0)
    fts.rebuild(idx, now=T0)
    first = idx.current_generation()

    db.remember("Bob drinks espresso", now=T0 + MINUTE)
    second = idx.build_next(now=T0 + MINUTE)
    assert fts.FtsStorage.of(first).source in tables(db)
    assert fts.FtsStorage.of(second).source in tables(db)
    assert idx.current_generation().generation == first.generation  # not published yet
    idx.publish(idx.validate(second).generation)
    assert idx.current_generation().generation == second.generation
    assert fts.FtsStorage.of(first).source in tables(db)  # still there for readers


def test_publishing_a_new_generation_mid_read_does_not_change_a_resolved_read(idx, file_db):
    """A read resolves one generation and keeps it.  A publication that lands afterwards is
    invisible to that read, which is what "each read pins one generation" buys."""
    db = file_db
    a = db.remember("Ada drinks espresso", now=T0).memory_id
    fts.rebuild(idx, now=T0)
    plan = fts.resolve(db.connection, tenant_id=1)
    assert plan.usable and plan.generation is not None and plan.generation.generation == 1

    b = db.remember("Bob drinks espresso", now=T0 + MINUTE).memory_id
    fts.rebuild(idx, now=T0 + MINUTE)  # generation 2 is now published
    republished = fts.resolve(db.connection, tenant_id=1).generation
    assert republished is not None and republished.generation == 2

    pinned = fts.search(db.connection, tenant_id=1, query_text="espresso", plan=plan)
    assert pinned.generation == 1
    # generation 1's base knows only `a`; `b` reaches this read through the journal, which the
    # publication did not prune because generation 1 is still in the catalog.
    assert set(ids(pinned.hits)) == {a, b}
    fresh = fts.search(db.connection, tenant_id=1, query_text="espresso")
    assert fresh.generation == 2 and set(ids(fresh.hits)) == {a, b}


def test_a_pinned_generation_is_not_dropped_by_a_rebuild(idx, file_db):
    db = file_db
    db.remember("Ada drinks espresso", now=T0)
    fts.rebuild(idx, now=T0)
    first = idx.current_generation()
    with idx.pin(1) as pin:
        assert pin.usable and pin.generation.generation == 1
        db.remember("Bob drinks espresso", now=T0 + MINUTE)
        fts.rebuild(idx, now=T0 + MINUTE)
        db.remember("Carol drinks espresso", now=T0 + 2 * MINUTE)
        fts.rebuild(idx, now=T0 + 2 * MINUTE)
        # the build after next would normally retire generation 1; the pin refuses
        assert fts.FtsStorage.of(first).source in tables(db)
        assert idx.pinned(first) == 1
        assert idx.retire(first) is False
    assert idx.retire(first) is True
    assert fts.FtsStorage.of(first).source not in tables(db)


def test_validation_refuses_a_generation_that_is_missing_the_oracles_documents(
    idx, file_db, monkeypatch
):
    """Validation is a superset check against the oracle for a sample of queries.  Delete a
    document out of a built generation and it must fail, and the live generation must stay."""
    db = file_db
    for word in ("espresso", "ristretto", "cortado", "macchiato", "affogato"):
        db.remember(f"Ada drinks {word} every morning", now=T0)
    fts.rebuild(idx, now=T0)
    good = idx.current_generation()

    db.remember("Bob drinks lungo", now=T0 + MINUTE)
    broken = idx.build_next(now=T0 + MINUTE)
    storage = fts.FtsStorage.of(broken)
    db.execute(f"DELETE FROM {storage.index_schema}.terms")  # the postings, not the map
    report = idx.validate(broken)
    assert not report.ok
    assert "missing from generation" in report.detail
    assert idx.current_generation().generation == good.generation

    idx.retire(report.generation)

    # and the same failure through rebuild(), which retires the generation and raises rather
    # than publishing a base the oracle disagrees with
    build = idx._build

    def half_built(generation):
        stats = build(generation)
        db.execute(f"DELETE FROM {fts.FtsStorage.of(generation).index_schema}.terms")
        return stats

    monkeypatch.setattr(idx, "_build", half_built)
    db.remember("Carol drinks flat white", now=T0 + 2 * MINUTE)
    with pytest.raises(IndexValidationError) as raised:
        fts.rebuild(idx, now=T0 + 2 * MINUTE)
    assert "failed validation" in str(raised.value)
    assert idx.current_generation().generation == good.generation
    assert ids(found(db, "espresso").hits) == oracle(db, "espresso")


def test_validation_catches_a_half_built_generation(idx, file_db):
    db = file_db
    db.remember("Ada drinks espresso", now=T0)
    fts.rebuild(idx, now=T0)
    db.remember("Bob drinks tea", now=T0 + MINUTE)
    generation = idx.build_next(now=T0 + MINUTE)
    db.execute(f"DELETE FROM {fts.FtsStorage.of(generation).docmap}")
    report = idx.validate(generation)
    assert not report.ok and "did not finish" in report.detail


# --------------------------------------------------------------------------- health


def test_health_reasons(idx, file_db):
    db = file_db
    db.remember("Ada drinks espresso", now=T0)

    absent = idx.health(1)
    assert absent.reason is HealthReason.ABSENT and not absent.usable
    assert fts.resolve(db.connection, tenant_id=1).reason is HealthReason.ABSENT

    fts.rebuild(idx, now=T0)
    fresh = idx.health(1, now=T0)
    assert fresh.reason is HealthReason.FRESH and fresh.usable
    assert fts.resolve(db.connection, tenant_id=1).reason is HealthReason.FRESH

    # a bulk load bypasses the journal, so the generation is invalidated
    db.indexes.invalidate("memories", reason="a bulk load bypassed the journal")
    stale = idx.health(1, now=T0)
    assert stale.reason is HealthReason.STALE_GENERATION and not stale.usable
    declined = fts.resolve(db.connection, tenant_id=1)
    assert declined.reason is HealthReason.STALE_GENERATION and not declined.usable
    assert "bulk load" in declined.detail

    # a load failure is reported through the index object the caller holds
    idx.load_error = "the extension could not be loaded"
    assert idx.health(1).reason is HealthReason.LOAD_FAILURE
    assert fts.resolve(db.connection, tenant_id=1, index=idx).reason is HealthReason.LOAD_FAILURE
    idx.load_error = None


def test_a_rebuild_in_progress_is_reported_not_guessed(idx, file_db):
    db = file_db
    db.remember("Ada drinks espresso", now=T0)
    db.execute(
        f"INSERT INTO {S.INDEX_GENERATIONS_TABLE} (index_name, generation, tenant_id, "
        f"built_at, validated, published, published_unvalidated, stats, notes) "
        f"VALUES (?, 1, NULL, ?, FALSE, FALSE, FALSE, '{{}}', 'building')",
        [fts.FTS_INDEX_NAME, T0],
    )
    plan = fts.resolve(db.connection, tenant_id=1)
    assert plan.reason is HealthReason.REBUILD_IN_PROGRESS and not plan.usable
    assert idx.health(1).reason is HealthReason.REBUILD_IN_PROGRESS
    # and the read is still right
    assert ids(found(db, "espresso").hits) == oracle(db, "espresso")


def test_a_missing_storage_table_is_a_load_failure_not_an_exception(idx, file_db):
    db = file_db
    db.remember("Ada drinks espresso", now=T0)
    fts.rebuild(idx, now=T0)
    storage = fts.FtsStorage.of(idx.current_generation())
    db.execute(f"DROP SCHEMA IF EXISTS {storage.index_schema} CASCADE")
    plan = fts.resolve(db.connection, tenant_id=1)
    assert plan.reason is HealthReason.LOAD_FAILURE and not plan.usable
    result = found(db, "espresso")
    assert result.backend == "scan" and result.exact
    assert ids(result.hits) == oracle(db, "espresso")


def test_an_unvalidated_generation_is_usable_and_says_so(idx, file_db):
    """P2-9: ``publish(force=True)`` must not create a generation nothing can read."""
    db = file_db
    db.remember("Ada drinks espresso", now=T0)
    status = fts.rebuild(idx, now=T0, validate=False)
    generation = idx.current_generation()
    assert generation.published and generation.published_unvalidated
    plan = fts.resolve(db.connection, tenant_id=1)
    assert plan.usable and plan.reason is HealthReason.UNVALIDATED
    assert idx.health(1).reason is HealthReason.UNVALIDATED
    assert ids(found(db, "espresso").hits) == oracle(db, "espresso")
    assert not status.stale


def test_the_scan_ceiling_is_enforced_and_reported(idx, file_db, monkeypatch):
    db = file_db
    for word in ("espresso", "tea", "cortado"):
        db.remember(f"Ada drinks {word}", now=T0)
    monkeypatch.setattr(fts, "SCAN_CEILING", 2)
    result = found(db, "espresso")
    assert result.hits == [] and not result.exact
    assert result.backend == "none" and result.scanned_rows == 3
    assert "SCAN_CEILING" in result.detail and "maintain_indexes" in result.detail
    reported = fts.status(db.connection, tenant_id=1)
    assert reported.available and reported.stale
    assert "SCAN_CEILING" in fts.staleness_message(reported)
    # publishing a generation removes the ceiling from the picture entirely
    fts.rebuild(idx, now=T0)
    assert ids(found(db, "espresso").hits) == oracle(db, "espresso")
    assert not fts.status(db.connection, tenant_id=1).stale


def test_an_overflowing_journal_falls_back_to_the_scan_and_says_so(idx, file_db, monkeypatch):
    db = file_db
    db.remember("Ada drinks espresso", now=T0)
    fts.rebuild(idx, now=T0)
    monkeypatch.setattr(fts, "MAX_MERGED_JOURNAL_ROWS", 1)
    db.remember("Bob drinks espresso", now=T0 + MINUTE)
    db.remember("Carol drinks espresso", now=T0 + 2 * MINUTE)
    plan = fts.resolve(db.connection, tenant_id=1)
    assert plan.overflowed and not plan.usable
    assert plan.reason is HealthReason.STALE_GENERATION
    assert "MAX_MERGED_JOURNAL_ROWS" in plan.detail
    result = found(db, "espresso")
    assert result.backend == "scan" and result.exact
    assert set(ids(result.hits)) == set(oracle(db, "espresso"))


def test_on_unusable_empty_and_error(idx, file_db):
    db = file_db
    db.remember("Ada drinks espresso", now=T0)
    assert found(db, "espresso", on_unusable="empty").hits == []
    assert not found(db, "espresso", on_unusable="empty").exact
    with pytest.raises(StaleIndexError):
        found(db, "espresso", on_unusable="error")


def test_a_load_failure_falls_back_to_the_scan_and_recall_can_be_told_to_raise(
    idx, file_db, monkeypatch
):
    """``load_error`` is how an accelerator says "my storage or extension is not there".  The
    read still happens, exactly, and the reason reaches the caller."""
    db = file_db
    a = db.remember("Ada drinks espresso", now=T0).memory_id
    fts.rebuild(idx, now=T0)
    idx.load_error = "the fts extension could not be loaded"

    plan = fts.resolve(db.connection, tenant_id=1, index=idx)
    assert plan.reason is HealthReason.LOAD_FAILURE and not plan.usable
    result = fts.search(db.connection, tenant_id=1, query_text="espresso", index=idx)
    assert result.backend == "scan" and result.exact and ids(result.hits) == [a]

    # ... and the ceiling turns the same state into a reported, raisable staleness
    monkeypatch.setattr(fts, "SCAN_CEILING", 0)
    monkeypatch.setattr(fts, "MAX_MERGED_JOURNAL_ROWS", -1)
    assert fts.status(db.connection, tenant_id=1).stale
    with pytest.raises(StaleIndexError):
        db.recall("espresso", k=5, on_stale_fts="error")
    hits = db.recall("espresso", k=5)
    assert hits.bm25_stale and "SCAN_CEILING" in " ".join(hits.notes)


def test_a_damaged_base_is_detected_on_the_read_and_the_scan_answers(idx, file_db):
    """The design says an index may be CORRUPT without making a query wrong.

    Damage that raises is caught for free.  This is the other kind: a base that is present,
    queryable and quietly incomplete.  Deleting a third of a generation's document map leaves
    every table readable and every count plausible, and the merged search then scores a
    fraction of the corpus and reports ``fresh``.  The invariant the read checks is the same
    one ``_validate`` checks structurally -- the source table and the document map hold the
    same documents -- and it costs two metadata counts.
    """
    db = file_db
    words = ["espresso", "cortado", "ristretto"]
    for i in range(60):
        db.remember(f"{words[i % 3]} number {i}", now=T0 + i * MINUTE)
    fts.rebuild(idx, now=T0)
    complete = found(db, "espresso", topn=100)
    assert complete.backend == "generation"
    assert ids(complete.hits) == oracle(db, "espresso", topn=100)

    storage = fts.FtsStorage.of(idx.current_generation())
    db.execute(f"DELETE FROM {storage.docmap} WHERE docid % 3 = 0")

    plan = fts.resolve(db.connection, tenant_id=1)
    assert plan.reason is HealthReason.DAMAGED_BASE and not plan.usable
    assert "document(s)" in plan.detail
    damaged = found(db, "espresso", topn=100)
    assert damaged.backend == "scan" and damaged.exact
    assert ids(damaged.hits) == oracle(db, "espresso", topn=100)
    assert ids(damaged.hits) == ids(complete.hits)

    # The health report agrees with the read, and a rebuild is due because of it.
    health = idx.health(1)
    assert health.reason is HealthReason.DAMAGED_BASE and not health.usable
    assert "damaged base" in (MaintenancePolicy().due(health) or "")
    assert maintain(idx, 1, MaintenancePolicy()).action == "published"
    assert fts.resolve(db.connection, tenant_id=1).reason is HealthReason.FRESH
    assert ids(found(db, "espresso", topn=100).hits) == oracle(db, "espresso", topn=100)


def test_detach_from_a_handle_that_never_attached_leaves_the_storage_alone(idx, file_db):
    """Dropping a generation's storage needs the implementation.  A handle that holds none
    disables the definition and leaves the storage to the handle that built it, rather than
    reporting a removal that did not happen."""
    db = file_db
    db.remember("Ada drinks espresso", now=T0)
    fts.rebuild(idx, now=T0)
    storage = fts.FtsStorage.of(idx.current_generation())

    other = Anatid.open(db.path, tenant=1, embedding_dim=DIM, ensure=False, accelerators=False)
    try:
        assert fts.index_of(other) is None
        fts.detach(other)
        assert other.indexes.definitions()["fts"].enabled is False
    finally:
        other.close()
    assert storage.source in tables(db)


# --------------------------------------------------------------------------- maintenance


def test_maintain_honours_the_default_policy(idx, file_db):
    """The framework defaults: 10,000 rows, 5% of the base, or 900 s with something pending."""
    db = file_db
    default = MaintenancePolicy()
    assert (default.rebuild_after_rows, default.rebuild_after_ratio) == (10_000, 0.05)
    assert default.rebuild_after_seconds == 900.0

    for i in range(40):
        db.remember(f"document number {i} about espresso", now=T0)
    first = maintain(idx, 1, default, now=T0)
    assert first.action == "published" and "no generation published" in first.reason
    assert idx.current_generation().generation == 1

    # one write in forty is 2.5%: under both triggers, and the age trigger has not fired
    db.remember("one more espresso", now=T0 + MINUTE)
    quiet = maintain(idx, 1, default, now=T0 + MINUTE)
    assert quiet.action == "none", quiet.reason
    assert idx.current_generation().generation == 1

    # 5% of the base is due
    for i in range(2):
        db.remember(f"another espresso {i}", now=T0 + MINUTE)
    due = maintain(idx, 1, default, now=T0 + MINUTE)
    assert due.action == "published" and "rebuild_after_ratio" in due.reason
    assert idx.current_generation().generation == 2

    # and the seconds trigger, with something pending
    db.remember("a late espresso", now=T0 + HOUR)
    aged = maintain(idx, 1, default, now=T0 + HOUR)
    assert aged.action == "published" and "rebuild_after_seconds" in aged.reason
    assert idx.current_generation().generation == 3
    # the whole time, every search was right
    assert set(ids(found(db, "espresso").hits)) == set(oracle(db, "espresso"))


def test_maintain_indexes_covers_the_text_index(idx, file_db):
    db = file_db
    db.remember("Ada drinks espresso", now=T0)
    reports = db.maintain_indexes(now=T0)
    assert reports["fts"].action == "published"
    assert db.index_health()["fts"].reason is HealthReason.FRESH
    assert idx.current_generation().generation == 1


def test_health_counts_the_journal_over_every_tenant_because_the_generation_is_file_wide(
    idx, file_db
):
    db = file_db
    db.remember("Ada drinks espresso", now=T0)
    fts.rebuild(idx, now=T0)
    db.remember("tenant two writes", tenant=2, now=T0 + MINUTE)
    assert idx.health(1).pending_rows == 1  # a rebuild would absorb tenant 2's row too
    assert fts.status(db.connection, tenant_id=1).pending_rows == 0  # tenant 1 rescans nothing
    assert fts.status(db.connection, tenant_id=2).pending_rows == 1


def test_the_index_object_pins_the_generation_around_its_own_search(idx, file_db):
    """:meth:`FtsIndex.search` is the module function with a pin held around it, for a caller
    that holds the object: a rebuild on another thread cannot retire the generation it chose
    between resolving it and reading it."""
    db = file_db
    a = db.remember("Ada drinks espresso", now=T0).memory_id
    fts.rebuild(idx, now=T0)
    result = idx.search(tenant_id=1, query_text="espresso")
    assert ids(result.hits) == [a] and result.generation == 1
    assert result.backend == "generation" and result.exact

    plan = idx.plan(1)
    assert plan.usable and plan.generation is not None and plan.generation.generation == 1
    reported = idx.status(tenant_id=1)
    assert reported.available and reported.indexed_rows == 1 and not reported.stale

    # the file-wide report counts every tenant's journal, the per-tenant one only this tenant's
    db.remember("tenant two writes espresso", tenant=2, now=T0 + MINUTE)
    assert idx.status().pending_rows == 1
    assert idx.status(tenant_id=1).pending_rows == 0
    assert fts.status(db.connection).pending_rows == 1


# --------------------------------------------------------------------------- tenancy


def test_the_cross_tenant_repro_from_0_1_1_stays_fixed(idx, file_db):
    """memory_id 4242 exists in two tenants with different text.  In schema v2 a BM25 hit on
    tenant 2's word returned tenant 1's row.  Checked on every path: base, journal and scan."""
    db = file_db
    db.remember("apples and pears", memory_id=4242, tenant=1, now=T0)
    db.remember("swordfish and tuna", memory_id=4242, tenant=2, now=T0)

    for stage in ("no generation", "generation", "generation + journal"):
        if stage == "generation":
            fts.rebuild(idx, now=T0)
        if stage == "generation + journal":
            db.remember("more apples", tenant=1, now=T0 + MINUTE)
            db.remember("more swordfish", tenant=2, now=T0 + MINUTE)
        assert ids(found(db, "swordfish", tenant=1).hits) == [], stage
        assert ids(found(db, "apples", tenant=2).hits) == [], stage
        assert 4242 in ids(found(db, "apples", tenant=1).hits), stage
        assert 4242 in ids(found(db, "swordfish", tenant=2).hits), stage
        assert ids(found(db, "apples", tenant=1).hits) == oracle(db, "apples", tenant=1), stage


def test_one_tenants_journal_never_touches_anothers_document(idx, file_db):
    """P0-3 for the text arm: the journal key is ``(tenant_id, doc_id)``, so forgetting tenant
    1's document 4242 must not remove tenant 2's document 4242 from the answer."""
    db = file_db
    db.remember("shared identifier, tenant one", memory_id=4242, tenant=1, now=T0)
    db.remember("shared identifier, tenant two", memory_id=4242, tenant=2, now=T0)
    fts.rebuild(idx, now=T0)
    db.forget(4242, tenant=1, now=T0 + MINUTE)
    assert ids(found(db, "identifier", tenant=1).hits) == []
    assert ids(found(db, "identifier", tenant=2).hits) == [4242]
    assert ids(found(db, "identifier", tenant=2).hits) == oracle(db, "identifier", tenant=2)


@pytest.mark.slow
def test_ten_thousand_documents_in_another_tenant_change_nothing_for_this_one(idx, file_db):
    """Corpus statistics are PER TENANT, on both arms.  Another tenant's writes must change
    neither this tenant's rows, nor its SCORES, nor its reported staleness -- the last one
    because a neighbour's activity is not this tenant's business and leaks that it wrote."""
    db = file_db
    for text in ("espresso for ada", "espresso and tea for bob", "tea for carol"):
        db.remember(text, tenant=1, now=T0)
    fts.rebuild(idx, now=T0)
    before = fts.search(db.connection, tenant_id=1, query_text="espresso tea")
    before_status = fts.status(db.connection, tenant_id=1)
    assert len(before.hits) == 3

    with db.transaction():
        for i in range(10_000):
            db.remember(f"espresso tea document {i}", tenant=2, now=T0 + MINUTE)

    after = fts.search(db.connection, tenant_id=1, query_text="espresso tea")
    assert after.hits == before.hits  # same rows AND the same scores
    after_status = fts.status(db.connection, tenant_id=1)
    assert after_status.pending_rows == 0 and not after_status.stale
    assert after_status.indexed_rows == before_status.indexed_rows

    # and again after a rebuild that absorbs all ten thousand of the neighbour's documents
    fts.rebuild(idx, now=T0 + 2 * MINUTE)
    rebuilt = fts.search(db.connection, tenant_id=1, query_text="espresso tea")
    assert ids(rebuilt.hits) == ids(before.hits)
    assert rebuilt.hits == pytest.approx(before.hits)


def test_a_tenant_newer_than_the_watermark_is_scored_from_its_own_statistics(idx, file_db):
    """A generation with no statistics row for a tenant: every one of that tenant's documents is
    in the journal, so the rescanned set IS its corpus and the statistics are exact."""
    db = file_db
    db.remember("tenant one only", tenant=1, now=T0)
    fts.rebuild(idx, now=T0)
    for text in ("espresso for ada", "espresso and tea", "tea alone"):
        db.remember(text, tenant=2, now=T0 + MINUTE)
    merged = fts.search(db.connection, tenant_id=2, query_text="espresso tea")
    exact = fts.scan(db.connection, tenant_id=2, query_text="espresso tea")
    assert merged.hits == exact


# --------------------------------------------------------------------------- erasure


def test_a_hard_forget_erases_the_text_from_every_generation(idx, file_db):
    """``forget(hard=True)`` is destructive to the accelerator too.  The generation's source
    table holds the content verbatim and its postings hold the tokens, so a purge that only
    tombstoned would leave the erased text in the file.  Scanned from the catalog, not from a
    list of names anatid keeps."""
    db = file_db
    secret = "xylophonic quokkas"
    victim = db.remember(f"Ada wrote {secret} in her notes", now=T0).memory_id
    db.remember("Bob wrote something else", now=T0)
    fts.rebuild(idx, now=T0)
    # two generations alive, both holding the text
    db.remember("Carol wrote a third thing", now=T0 + MINUTE)
    fts.rebuild(idx, now=T0 + MINUTE)
    assert find_in_file(db, "xylophonic"), "the fixture must actually store the text"

    receipt = db.forget(victim, hard=True, now=T0 + 2 * MINUTE)
    assert receipt.hard and receipt.memories_deleted == 1
    assert receipt.derived_rows_deleted > 0
    assert receipt.invalidated_generations == 0
    assert find_in_file(db, "xylophonic", "quokkas") == []
    # erased, not tombstoned: a tombstone would leave the id in the file
    assert [row for row in journal(db) if row[1] == victim] == []
    assert ids(found(db, "xylophonic").hits) == []
    assert fts.resolve(db.connection, tenant_id=1).usable  # the generation stayed in service


def test_detach_drops_every_generation_and_the_search_falls_back(idx, file_db):
    db = file_db
    db.remember("Ada wrote xylophonic quokkas", now=T0)
    fts.rebuild(idx, now=T0)
    storage = fts.FtsStorage.of(idx.current_generation())
    assert storage.source in tables(db)

    fts.detach(db)
    assert storage.source not in tables(db)
    assert find_in_file(db, "xylophonic") == [("main.memories", "content", "xylophonic")]
    assert db.indexes.definitions()["fts"].enabled is False
    # and the text arm reverts to the legacy half, which has nothing built
    assert fts.search(db.connection, tenant_id=1, query_text="xylophonic").backend == "legacy"


def test_a_rebuild_keeps_the_catalog_watermark_in_step(idx, file_db):
    """``db.info()`` and the MCP health tool report ``anatid_meta``'s three full-text columns.
    A generation carries its own watermark; the meta row is kept in step so the two reports do
    not contradict each other."""
    db = file_db
    a = db.remember("Ada drinks espresso", now=T0).memory_id
    assert db.info().fts_indexed_at is None
    fts.rebuild(idx, now=T0)
    info = db.info()
    assert info.fts_indexed_at == T0 and info.fts_indexed_rows == 1
    row = db.execute("SELECT fts_indexed_max_id FROM anatid_meta").fetchone()
    assert int(row[0]) == a == idx.current_generation().watermark_id


def test_the_doctor_is_clean_with_the_framework_index_and_no_legacy_one(idx, file_db):
    """``doctor()`` reads :func:`anatid.fts.status`, which on the framework reports a fresh
    index rather than "N rows written since the last rebuild": there is no such state."""
    db = file_db
    db.remember("Ada drinks espresso", now=T0)
    fts.rebuild(idx, now=T0)
    db.remember("Bob drinks espresso", now=T0 + MINUTE)  # pending, and searchable
    report = db.doctor(deep=True)
    assert "stale_fts_index" in report.checks_run
    assert [f for f in report.findings if "fts" in f.check] == []
    assert report.ok


def test_attach_and_a_bulk_load_style_invalidation_never_lose_a_row(idx, file_db):
    """Anything that bypasses the journal (a bulk load) invalidates the generation rather than
    being missed, and the search falls back to the exact scan until maintenance rebuilds."""
    db = file_db
    db.remember("Ada drinks espresso", now=T0)
    fts.rebuild(idx, now=T0)
    db.execute(
        "INSERT INTO memories (memory_id, tenant_id, content, kind, created_at, valid_from, "
        "tx_from, version) VALUES (99, 1, 'a smuggled espresso row', 'note', ?, ?, ?, 1)",
        [T0, T0, T0],
    )
    assert db.indexes.invalidate("memories", reason="bulk load") >= 1
    result = found(db, "espresso")
    assert result.backend == "scan" and result.exact
    assert 99 in ids(result.hits)
    db.maintain_indexes(now=T0 + MINUTE)
    assert found(db, "espresso").backend == "generation"
    assert 99 in ids(found(db, "espresso").hits)


# --------------------------------------------------------------------------- as_of


def test_an_as_of_search_uses_the_generation_and_agrees_with_the_oracle(idx, file_db):
    """A current-state index cannot answer an as_of read; this one is not a current-state index.
    Its base holds document identity and content, and the time predicate is applied to the
    canonical rows, so an as_of read uses a generation and gets the exact answer."""
    db = file_db
    a = db.remember("Ada drinks espresso", now=T0).memory_id
    fts.rebuild(idx, now=T0)
    b = db.supersede(a, "Ada drinks tea now", now=T0 + HOUR).memory_id

    then = AsOf(valid_time=T0 + MINUTE)
    now = fts.CURRENT
    assert ids(found(db, "espresso", as_of=then).hits) == [a]
    assert ids(found(db, "espresso", as_of=now).hits) == []
    assert ids(found(db, "tea", as_of=then).hits) == []
    assert ids(found(db, "tea", as_of=now).hits) == [b]
    assert found(db, "espresso", as_of=then).backend == "generation"
    assert ids(found(db, "espresso", as_of=then).hits) == oracle(db, "espresso", as_of=then)

    # transaction time: what the database BELIEVED before the supersession
    believed = AsOf(valid_time=T0 + 2 * HOUR, tx_time=T0 + MINUTE)
    assert ids(found(db, "espresso", as_of=believed).hits) == [a]
    assert idx.health(1, as_of=then).usable
    assert "as_of read uses this generation" in idx.health(1, as_of=then).detail
    assert db.as_of(T0 + MINUTE).recall("espresso", k=5)[0].memory.memory_id == a


def test_kinds_filter_both_arms(idx, file_db):
    db = file_db
    a = db.remember("espresso note", kind="note", now=T0).memory_id
    b = db.remember("espresso fact", kind="fact", now=T0).memory_id
    fts.rebuild(idx, now=T0)
    c = db.remember("espresso journal note", kind="note", now=T0 + MINUTE).memory_id
    d = db.remember("espresso journal fact", kind="fact", now=T0 + MINUTE).memory_id
    notes = set(ids(found(db, "espresso", kinds=["note"]).hits))
    assert notes == {a, c}
    assert set(ids(found(db, "espresso", kinds=["fact"]).hits)) == {b, d}
    assert notes == set(
        ids(fts.scan(db.connection, tenant_id=1, query_text="espresso", kinds=["note"]))
    )


# --------------------------------------------------------------------------- the legacy half


def test_a_database_with_no_derived_index_behaves_as_0_1_1(file_db):
    """Nothing about the legacy path changes until someone attaches: the index is not
    incremental, staleness is reported, and a rebuild is the caller's decision."""
    db = file_db
    assert fts.index_of(db) is None
    a = db.remember("Ada drinks espresso", now=T0).memory_id
    assert not fts.fts_index_present(db.connection)
    assert fts.search(db.connection, tenant_id=1, query_text="espresso").hits == []
    assert db.fts_status().stale and not db.fts_status().available

    db.rebuild_fts_index(now=T0)
    assert fts.fts_index_present(db.connection)
    assert ids(fts.search(db.connection, tenant_id=1, query_text="espresso").hits) == [a]
    assert not db.fts_status().stale

    b = db.remember("Bob drinks espresso", now=T0 + MINUTE).memory_id
    assert ids(fts.search(db.connection, tenant_id=1, query_text="espresso").hits) == [a]
    status = db.fts_status()
    assert status.stale and status.pending_rows == 1
    assert "not incremental" in fts.staleness_message(status)
    db.rebuild_fts_index(now=T0 + MINUTE)
    assert set(ids(fts.search(db.connection, tenant_id=1, query_text="espresso").hits)) == {a, b}


def test_attaching_over_a_legacy_index_takes_over_the_text_arm(file_db):
    db = file_db
    a = db.remember("Ada drinks espresso", now=T0).memory_id
    db.rebuild_fts_index(now=T0)
    index = fts.attach(db)
    b = db.remember("Bob drinks espresso", now=T0 + MINUTE).memory_id
    # no generation yet, so the scan answers -- and it already sees the new row
    result = fts.search(db.connection, tenant_id=1, query_text="espresso")
    assert result.backend == "scan" and set(ids(result.hits)) == {a, b}
    # rebuild_fts_index() now builds a generation rather than the legacy index
    db.rebuild_fts_index(now=T0 + MINUTE)
    assert index.current_generation() is not None
    assert fts.search(db.connection, tenant_id=1, query_text="espresso").backend == "generation"


# --------------------------------------------------------------------------- concurrency


def test_a_rebuild_on_another_thread_does_not_break_a_running_search(idx, file_db):
    """Reads and rebuilds overlap: the build happens beside the live generation and publication
    is one metadata row, so a search running throughout is never wrong and never raises."""
    db = file_db
    for i in range(30):
        db.remember(f"document {i} about espresso and tea", now=T0)
    fts.rebuild(idx, now=T0)
    stop = threading.Event()
    errors: list[BaseException] = []
    seen: list[int] = []

    def reader():
        try:
            while not stop.is_set():
                hits = fts.search(db.connection, tenant_id=1, query_text="espresso")
                seen.append(len(hits.hits))
                assert hits.exact
        except BaseException as exc:  # noqa: BLE001 - reported on the main thread
            errors.append(exc)

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for i in range(4):
            db.remember(f"another espresso {i}", now=T0 + (i + 1) * MINUTE)
            fts.rebuild(idx, now=T0 + (i + 1) * MINUTE)
    finally:
        stop.set()
        thread.join(timeout=30)
    assert errors == []
    assert seen and min(seen) >= 30
    assert set(ids(found(db, "espresso").hits)) == set(oracle(db, "espresso"))
