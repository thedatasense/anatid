"""The default ``recall(query)``: text plus graph, with the graph arm seeded from the query.

Before this, ``recall("who maintains the ingest service")`` with no seed and no embedding ran
the text arm alone, so the graph -- the thing the engine was chosen for -- only helped a caller
who already knew which entity to name.  Now ``seed_entity="auto"`` is the default: the query's
words are matched against the tenant's entity names, longest name first, at most three, and
the graph arm expands from each.  ``seed_entity=None`` switches it off; an explicit entity is
used as it always was.  ``RecallHits.arms`` and ``RecallHits.seeds`` say what happened.

The fixture is the on-call story from the product review, stored by entity name alone and
with no ``relate()`` calls, which is how an agent writes through the tools.
"""

from __future__ import annotations

import datetime as _dt
import statistics
import time

import pytest

from anatid import (
    AUTO_SEED,
    AUTO_SEED_LIMIT,
    Anatid,
    BruteForceCeilingError,
    HashEmbedder,
    RecallHits,
)
from anatid import recall as recall_mod
from conftest import DIM, T0, vec

MINUTE = _dt.timedelta(minutes=1)

FACTS = (
    ("Ada leads Kestrel", ("Ada", "Kestrel")),
    ("Kestrel owns the ingest service", ("Kestrel", "ingest service")),
    ("Bo maintains the ingest service", ("Bo", "ingest service")),
    ("postgres-primary has a nightly vacuum window at 02:00 UTC", ("postgres-primary",)),
)


def store(db, *, tenant=None):
    out = {}
    for i, (content, entities) in enumerate(FACTS):
        m = db.remember(content, entities=list(entities), now=T0 + i * MINUTE, tenant=tenant)
        out[content] = m.memory_id
    return out


@pytest.fixture
def oncall(db):
    return store(db)


def contents(hits):
    return [h.content for h in hits]


# ============================================================================ the default


def test_the_default_recall_runs_text_and_graph_seeded_from_the_query(db, oncall):
    hits = db.recall("who maintains the ingest service")
    assert hits.arms == ("text", "graph")
    assert hits.seeds == ("ingest service",)
    # Both facts filed under the ingest service came back, and the graph arm is what reached the
    # one whose words do not answer the question.
    by_content = {h.content: h for h in hits}
    assert "Bo maintains the ingest service" in by_content
    owner = by_content["Kestrel owns the ingest service"]
    assert owner.graph_rank is not None and "graph" in owner.sources


def test_seed_entity_none_restores_text_only(db, oncall):
    hits = db.recall("who maintains the ingest service", seed_entity=None)
    assert hits.arms == ("text",)
    assert hits.seeds == ()
    assert all(h.graph_rank is None for h in hits)


def test_auto_is_spelled_out_and_is_the_default(db, oncall):
    assert AUTO_SEED == "auto"
    explicit = db.recall("who maintains the ingest service", seed_entity=AUTO_SEED)
    default = db.recall("who maintains the ingest service")
    assert explicit.arms == default.arms == ("text", "graph")
    assert explicit.seeds == default.seeds == ("ingest service",)
    assert explicit.memory_ids == default.memory_ids


def test_an_explicit_seed_wins_over_auto(db, oncall):
    hits = db.recall("who maintains the ingest service", seed_entity="Ada")
    assert hits.arms == ("text", "graph")
    assert hits.seeds == ("Ada",)  # the query names the ingest service, and it was not used
    assert "Ada leads Kestrel" in contents(hits)

    ada = db.get_entity("Ada")
    assert ada is not None
    by_id = db.recall("who maintains the ingest service", seed_entity=ada.entity_id)
    by_obj = db.recall("who maintains the ingest service", seed_entity=ada)
    assert by_id.seeds == by_obj.seeds == ("Ada",)
    assert by_id.memory_ids == by_obj.memory_ids == hits.memory_ids


def test_with_an_embedder_all_three_arms_run():
    with Anatid.open(":memory:", tenant=1, embedding_dim=DIM, embedder=HashEmbedder(DIM)) as db:
        ids = store(db)
        hits = db.recall("who maintains the ingest service")
        assert hits.arms == ("vector", "text", "graph")
        assert hits.seeds == ("ingest service",)
        top = hits[0]
        assert top.content == "Bo maintains the ingest service"
        assert top.sources == ("vector", "text", "graph")
        stored = db.get(ids["Bo maintains the ingest service"], with_embedding=True)
        assert stored is not None and stored.embedding is not None
        assert len(stored.embedding) == DIM


def test_recall_with_no_query_runs_no_arm_even_under_auto(db, oncall):
    hits = db.recall()
    assert list(hits) == [] and hits.arms == () and hits.seeds == ()


def test_a_query_naming_no_entity_runs_the_text_arm_alone(db, oncall):
    hits = db.recall("nightly vacuum window")
    assert hits.arms == ("text",) and hits.seeds == ()
    assert contents(hits) == ["postgres-primary has a nightly vacuum window at 02:00 UTC"]


def test_seeds_are_reported_in_the_result_and_default_to_empty():
    assert RecallHits().seeds == ()
    assert RecallHits(seeds=("Ada",)).seeds == ("Ada",)


# ============================================================================ the matcher


def test_seeds_are_longest_name_first_and_capped_at_three(db, oncall):
    assert AUTO_SEED_LIMIT == 3
    hits = db.recall("Ada Bo Kestrel ingest service postgres-primary")
    # Two two-word names first, left to right, then the earliest one-word name; Bo and Kestrel
    # are past the cap.
    assert hits.seeds == ("ingest service", "postgres-primary", "Ada")
    assert len(hits.seeds) == AUTO_SEED_LIMIT


def test_matching_is_case_insensitive_and_whole_word(db, oncall):
    assert db.recall("WHO MAINTAINS THE INGEST SERVICE").seeds == ("ingest service",)
    assert db.recall("Who Maintains The Ingest Service").seeds == ("ingest service",)
    # Substrings and inflections are not names.
    assert db.recall("the ingestion services").seeds == ()
    assert db.recall("Adam and Bob").seeds == ()


def test_a_longer_name_shadows_the_names_inside_it(db):
    db.remember("Kestrel is a bird", entities=["Kestrel"], now=T0)
    db.remember("Project Kestrel ships on Fridays", entities=["Project Kestrel"], now=T0)
    hits = db.recall("when does Project Kestrel ship")
    assert hits.seeds == ("Project Kestrel",)
    assert db.recall("what is a kestrel").seeds == ("Kestrel",)


def test_hyphenated_and_spaced_spellings_find_each_other(db, oncall):
    assert db.recall("vacuum window for postgres primary").seeds == ("postgres-primary",)
    assert db.recall("vacuum window for postgres_primary").seeds == ("postgres-primary",)
    assert db.recall("who owns the ingest-service").seeds == ("ingest service",)


def test_possessives_and_punctuation_do_not_hide_a_name(db, oncall):
    hits = db.recall("Ada's project is paging. Who should I wake up, and why?")
    assert hits.seeds == ("Ada",)
    assert "Ada leads Kestrel" in contents(hits)
    assert db.recall("(Bo) maintains what?").seeds == ("Bo",)
    assert db.recall("“Kestrel”?").seeds == ("Kestrel",)


def test_seed_candidates_are_ngrams_longest_first_with_three_spellings():
    cands = recall_mod.seed_candidates("who owns the ingest-service?")
    assert list(cands)[:3] == [
        "who owns the ingest service",
        "who-owns-the-ingest-service",
        "who_owns_the_ingest_service",
    ]
    assert cands["ingest service"] == (3, 5)
    assert cands["ingest-service"] == (3, 5)
    assert cands["who"] == (0, 1)
    assert "service" in cands and cands["service"] == (4, 5)
    assert recall_mod.seed_candidates("") == {}
    assert recall_mod.seed_candidates("...") == {}
    # A word carrying the record separator is dropped rather than escaped.
    assert "a\x1fb" not in recall_mod.seed_candidates("a\x1fb c")
    # Only the first AUTO_SEED_QUERY_WORDS words are considered.
    long_query = " ".join(f"w{i}" for i in range(recall_mod.AUTO_SEED_QUERY_WORDS + 10))
    spans = recall_mod.seed_candidates(long_query).values()
    assert max(end for _s, end in spans) == recall_mod.AUTO_SEED_QUERY_WORDS


def test_auto_seeds_returns_ids_and_names_in_seed_order(db, oncall):
    seeds = recall_mod.auto_seeds(
        db.connection, tenant_id=1, query="Bo maintains the ingest service"
    )
    assert [name for _id, name in seeds] == ["ingest service", "Bo"]
    ingest = db.get_entity("ingest service")
    assert ingest is not None and seeds[0][0] == ingest.entity_id
    assert recall_mod.auto_seeds(db.connection, tenant_id=1, query="nothing here") == []
    assert recall_mod.auto_seeds(db.connection, tenant_id=1, query="Bo", limit=0) == []


def test_auto_seeds_stay_inside_the_tenant(db):
    store(db, tenant=2)
    mine = db.recall("who maintains the ingest service", tenant=1)
    assert mine.arms == ("text",) and mine.seeds == () and list(mine) == []
    theirs = db.recall("who maintains the ingest service", tenant=2)
    assert theirs.arms == ("text", "graph") and theirs.seeds == ("ingest service",)
    assert all(h.memory.tenant_id == 2 for h in theirs)


def test_auto_seeds_through_the_as_of_view(db, oncall):
    then = db.as_of(T0 + 1 * MINUTE + _dt.timedelta(seconds=30)).recall("the ingest service")
    assert then.arms == ("text", "graph") and then.seeds == ("ingest service",)
    assert contents(then) == ["Kestrel owns the ingest service"]  # Bo's fact is a minute later
    before = db.as_of(T0 - MINUTE).recall("the ingest service")
    assert before.seeds == ("ingest service",) and list(before) == []


# ============================================================================ hybrid_recall


def test_hybrid_recall_takes_ids_a_sequence_auto_or_none(db, oncall):
    con = db.connection
    ada = db.entity_id("Ada")
    bo = db.entity_id("Bo")
    one = recall_mod.hybrid_recall(con, tenant_id=1, dim=DIM, seed_entity=ada)
    assert one.arms == ("graph",) and one.seeds == ("Ada",)
    both = recall_mod.hybrid_recall(con, tenant_id=1, dim=DIM, seed_entity=[ada, bo, ada])
    assert both.arms == ("graph",) and both.seeds == ("Ada", "Bo")
    # The union of the two expansions, newest first, each memory once.  Nothing here called
    # relate(), so each seed reaches only the memories filed directly under it.
    assert contents(both) == ["Bo maintains the ingest service", "Ada leads Kestrel"]
    assert [h.graph_rank for h in both] == [1, 2]
    assert contents(one) == ["Ada leads Kestrel"]
    none = recall_mod.hybrid_recall(
        con, tenant_id=1, dim=DIM, seed_entity=None, query="ingest service"
    )
    assert none.arms == ("text",) and none.seeds == ()
    auto = recall_mod.hybrid_recall(con, tenant_id=1, dim=DIM, query="ingest service")
    assert auto.arms == ("text", "graph") and auto.seeds == ("ingest service",)
    with pytest.raises(TypeError):
        recall_mod.hybrid_recall(con, tenant_id=1, dim=DIM, seed_entity="Ada")
    with pytest.raises(TypeError):
        recall_mod.hybrid_recall(con, tenant_id=1, dim=DIM, seed_entity=True)
    with pytest.raises(ValueError, match="on_ceiling"):
        recall_mod.hybrid_recall(con, tenant_id=1, dim=DIM, on_ceiling="maybe")


def test_an_unknown_explicit_seed_id_is_reported_by_its_id(db, oncall):
    hits = recall_mod.hybrid_recall(db.connection, tenant_id=1, dim=DIM, seed_entity=424242)
    assert hits.arms == ("graph",) and hits.seeds == ("424242",) and list(hits) == []


def test_an_implicit_embedding_past_the_ceiling_skips_the_vector_arm_and_says_so(monkeypatch):
    """The handle's embedder must never turn a recall that answered into one that raises."""
    monkeypatch.setattr(recall_mod, "BRUTE_FORCE_CEILING", 2)
    with Anatid.open(":memory:", tenant=1, embedding_dim=DIM, embedder=HashEmbedder(DIM)) as db:
        store(db)  # four embedded rows, ceiling two
        hits = db.recall("who maintains the ingest service")
        assert hits.arms == ("text", "graph")
        assert any("vector arm skipped" in n and "BRUTE_FORCE_CEILING=2" in n for n in hits.notes)
        assert "Bo maintains the ingest service" in contents(hits)
        # An embedding the caller passed keeps the documented refusal, and the escape hatch.
        with pytest.raises(BruteForceCeilingError):
            db.recall("who maintains the ingest service", embedding=vec(1, 0))
        slow = db.recall("who maintains the ingest service", allow_slow=True)
        assert slow.arms == ("vector", "text", "graph") and slow.notes == ()


@pytest.mark.slow
def test_the_entity_match_costs_a_few_milliseconds_at_100k_entities(db):
    """The measured figure on the development machine is 1.9 ms p50 at 100,000 entities in the
    tenant (200,000 in the file); the bound here is loose because CI hardware is not that
    machine.  The shape being guarded is that the cost does not grow with the candidate count."""
    con = db.connection
    con.execute(
        "INSERT INTO entities (entity_id, tenant_id, kind, name, valid_from, valid_to, tx_from, "
        "tx_to, writer, episode_id, confidence) "
        "SELECT 1000000 + i, (i % 2) + 1, 'thing', "
        "CASE WHEN i = 0 THEN 'ingest service' ELSE 'entity ' || i::VARCHAR END, "
        "now()::TIMESTAMP, NULL, now()::TIMESTAMP, NULL, NULL, NULL, 1.0 FROM range(200000) t(i)"
    )
    short = "who maintains the ingest service"
    long_query = " ".join([short] * 12)

    def p50(query):
        samples = []
        for _ in range(9):
            t = time.perf_counter()
            recall_mod.auto_seeds(con, tenant_id=1, query=query)
            samples.append((time.perf_counter() - t) * 1000)
        return statistics.median(samples)

    assert recall_mod.auto_seeds(con, tenant_id=1, query=short)[0][1] == "ingest service"
    assert p50(short) < 10.0
    assert p50(long_query) < 10.0
