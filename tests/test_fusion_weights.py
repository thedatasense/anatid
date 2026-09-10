"""Weighted fusion, and a graph arm that votes for what the question is about.

Before this, ``recall()`` fused its arms with equal votes and the graph arm handed the fusion
its neighbourhood newest first.  On the answer-quality benchmark (docs/quality.md) that cost
the fusion questions the vector arm alone had right: a BM25 arm over short extracted
sentences ranked the wrong memories first, and the graph arm voted for whatever was written
last.  Now :func:`anatid.recall.default_arm_weights` lets the vector arm lead when it ran and
the text arm otherwise, :func:`anatid.recall.rank_graph_candidates` orders the graph arm's
candidates by cosine or BM25 before they vote, ``arm_weights=`` overrides a weight by name, and
``RecallHits.weights`` says what was used.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from anatid import Anatid, HashEmbedder, RecallHits
from anatid import recall as recall_mod
from anatid.recall import (
    ARM_WEIGHTS_WITH_VECTOR,
    ARM_WEIGHTS_WITHOUT_VECTOR,
    check_arm_weights,
    default_arm_weights,
    rank_graph_candidates,
    rrf_fuse,
)
from conftest import DIM, T0, vec

MINUTE = _dt.timedelta(minutes=1)


# ============================================================================ the pure functions


def test_rrf_fuse_weights_each_arm_and_skips_an_arm_weighted_zero():
    arms = {"vector": [(1, 0.9), (2, 0.8)], "text": [(2, 5.0), (3, 4.0)]}
    equal = rrf_fuse(arms, k=60)
    assert [m for m, *_ in equal] == [2, 1, 3], "2 is in both lists, then ties by memory id"
    assert equal[0][1] == pytest.approx(1 / 62 + 1 / 61)

    weighted = rrf_fuse(arms, k=60, weights={"vector": 1.0, "text": 0.25})
    assert [m for m, *_ in weighted] == [2, 1, 3], "two votes still beat one, at a quarter"
    assert weighted[0][1] == pytest.approx(1 / 62 + 0.25 / 61)
    assert weighted[1][1] == pytest.approx(1 / 61)
    assert weighted[2][1] == pytest.approx(0.25 / 62), "the text arm's lone vote is scaled"
    assert weighted[0][2] == {"vector": 2, "text": 1} and weighted[1][2] == {"vector": 1}
    heavy = rrf_fuse(arms, k=60, weights={"vector": 1.0, "text": 3.0})
    assert [m for m, *_ in heavy] == [2, 3, 1], "a heavy text arm lifts its lone candidate"

    silenced = rrf_fuse(arms, k=60, weights={"text": 0})
    assert [m for m, *_ in silenced] == [1, 2], "an arm weighted 0 neither votes nor ranks"
    assert silenced[1][2] == {"vector": 2}
    assert rrf_fuse(arms, k=60, weights={"graph": 3.0}) == equal, "an unnamed arm weighs 1.0"


def test_default_weights_follow_whether_the_vector_arm_ran():
    assert default_arm_weights(["vector", "text", "graph"]) == ARM_WEIGHTS_WITH_VECTOR
    assert default_arm_weights(["text", "graph"]) == ARM_WEIGHTS_WITHOUT_VECTOR
    assert default_arm_weights(["graph"]) == {"graph": 0.5}
    assert default_arm_weights(["vector"]) == {"vector": 1.0}
    assert default_arm_weights([]) == {}
    assert (
        ARM_WEIGHTS_WITH_VECTOR["vector"]
        > ARM_WEIGHTS_WITH_VECTOR["graph"]
        > (ARM_WEIGHTS_WITH_VECTOR["text"])
    )


def test_arm_weights_are_checked():
    assert check_arm_weights({"text": 0, "graph": 2}) == {"text": 0.0, "graph": 2.0}
    for bad in ({"bm25": 1.0}, {"text": -1}, {"text": float("inf")}, {"text": "1"}, {"text": True}):
        with pytest.raises(ValueError):
            check_arm_weights(bad)
    with pytest.raises(TypeError):
        check_arm_weights([("text", 1.0)])  # type: ignore[arg-type]


# ============================================================================ the graph arm's order


@pytest.fixture
def town():
    """A seed entity with four memories: two about the question, two recent but off-topic."""
    embedder = HashEmbedder(dim=DIM)
    db = Anatid.open(":memory:", tenant=1, embedding_dim=DIM, embedder=embedder)
    ids = {}
    facts = [
        ("Ada leads the Kestrel team", ("Ada", "Kestrel")),
        ("Kestrel owns the ingest service", ("Kestrel", "ingest service")),
        ("The Kestrel offsite is in Lisbon in May", ("Kestrel",)),
        ("Kestrel ordered new laptops for the team", ("Kestrel",)),
    ]
    for i, (content, entities) in enumerate(facts):
        ids[content] = db.remember(content, entities=list(entities), now=T0 + i * MINUTE).memory_id
    yield db, ids, embedder
    db.close()


def test_the_graph_arm_is_ranked_by_cosine_when_the_query_has_an_embedding(town):
    db, ids, embedder = town
    con = db.connection
    kestrel = db.entity_id("Kestrel", create=False)
    newest_first = recall_mod.graph_arm(con, tenant_id=1, seed_entity_id=kestrel, hops=2, topn=50)
    assert [m for m, _ in newest_first][:2] == [
        ids["Kestrel ordered new laptops for the team"],
        ids["The Kestrel offsite is in Lisbon in May"],
    ], "the raw graph arm is newest first"

    query = "who owns the ingest service"
    ranked = rank_graph_candidates(
        con, newest_first, tenant_id=1, embedding=embedder.embed_one(query), dim=DIM
    )
    assert ranked[0][0] == ids["Kestrel owns the ingest service"]
    assert {m for m, _ in ranked} == {m for m, _ in newest_first}, "an order, not a filter"
    assert all(-1.0 <= s <= 1.0 for _m, s in ranked), "scores are cosines now"

    # through the verb: the graph arm's rank of the on-topic memory is 1, and the fusion put
    # it first even though the text arm and the vector arm both like it too
    hits = db.recall(query, seed_entity="Kestrel", k=4)
    assert hits.arms == ("vector", "text", "graph")
    top = hits[0]
    assert top.memory_id == ids["Kestrel owns the ingest service"] and top.graph_rank == 1
    assert hits.weights == ARM_WEIGHTS_WITH_VECTOR


def test_the_graph_arm_is_ranked_by_bm25_without_an_embedding(town):
    db, ids, _embedder = town
    con = db.connection
    kestrel = db.entity_id("Kestrel", create=False)
    newest_first = recall_mod.graph_arm(con, tenant_id=1, seed_entity_id=kestrel, hops=2, topn=50)
    text = recall_mod.bm25_arm(con, tenant_id=1, query_text="offsite Lisbon", topn=50)
    ranked = rank_graph_candidates(con, newest_first, tenant_id=1, text_hits=dict(text))
    assert ranked[0][0] == ids["The Kestrel offsite is in Lisbon in May"]
    assert [m for m, _ in ranked][1:] == [
        m for m, _ in newest_first if m != ids["The Kestrel offsite is in Lisbon in May"]
    ], "the candidates the text arm did not score follow, newest first"
    assert rank_graph_candidates(con, newest_first, tenant_id=1) == list(newest_first)
    assert rank_graph_candidates(con, [], tenant_id=1, text_hits={1: 2.0}) == []

    hits = db.recall("offsite Lisbon", seed_entity="Kestrel", embedding=None, k=4, arm_weights=None)
    # the handle has an embedder, so the vector arm ran: silence it to see the text-led default
    plain = Anatid.open(":memory:", tenant=1, embedding_dim=DIM)
    try:
        for content in ids:
            plain.remember(content, entities=["Kestrel"])
        got = plain.recall("offsite Lisbon", seed_entity="Kestrel", k=4)
        assert got.arms == ("text", "graph") and got.weights == ARM_WEIGHTS_WITHOUT_VECTOR
        assert got[0].content == "The Kestrel offsite is in Lisbon in May"
    finally:
        plain.close()
    assert hits.weights == ARM_WEIGHTS_WITH_VECTOR


# ============================================================================ arm_weights=


def test_arm_weights_override_by_name_and_are_reported(town):
    db, _ids, _embedder = town
    query = "who owns the ingest service"
    default = db.recall(query, seed_entity="Kestrel", k=4)
    assert default.weights == {"vector": 1.0, "graph": 0.5, "text": 0.25}

    only_text = db.recall(query, seed_entity="Kestrel", k=4, arm_weights={"vector": 0, "graph": 0})
    assert only_text.weights == {"vector": 0.0, "graph": 0.0, "text": 0.25}
    assert only_text.arms == ("vector", "text", "graph"), "the arms still ran"
    assert all(h.vector_rank is None and h.graph_rank is None for h in only_text)
    assert all(h.text_rank is not None for h in only_text)

    boosted = db.recall(query, seed_entity="Kestrel", k=4, arm_weights={"text": 4.0})
    assert boosted.weights == {"vector": 1.0, "graph": 0.5, "text": 4.0}
    with pytest.raises(ValueError, match="unknown arm"):
        db.recall(query, arm_weights={"bm25": 1.0})
    with pytest.raises(ValueError):
        db.recall(query, arm_weights={"text": -0.5})

    empty = RecallHits()
    assert empty.weights == {} and empty.seeds == () and empty.arms == ()


# ============================================================================ an arm that found nothing


def test_an_empty_vector_arm_casts_no_vote_and_leaves_the_text_arm_leading():
    """Three memories with no embeddings.  Passing a query embedding makes the vector arm run and
    find nothing; before this it still quartered the text arm's weight and switched the graph
    arm from BM25 to a cosine order it could not compute, and a picnic note outranked the
    ownership fact the text arm had first."""
    db = Anatid.open(":memory:", tenant=1, embedding_dim=DIM)
    try:
        db.remember("Atlas owns the ledger", entities=["Atlas", "ledger"], now=T0)
        db.relate("Atlas", "ledger", rel_kind="owns", now=T0)
        db.remember(
            "The Atlas team picnic is on Friday", entities=["Atlas", "picnic"], now=T0 + MINUTE
        )
        db.remember(
            "Atlas will bring a kite to the picnic",
            entities=["Atlas", "picnic"],
            now=T0 + 2 * MINUTE,
        )
        query = "who owns the ledger"

        plain = db.recall(query, k=3)
        assert plain.arms == ("text", "graph") and plain[0].content == "Atlas owns the ledger"

        with_vector = db.recall(query, k=3, embedding=vec(1, 0))
        assert with_vector.arms == ("vector", "text", "graph"), "the vector arm ran"
        assert with_vector.weights == {"text": 1.0, "graph": 0.5}, "and cast no vote"
        assert with_vector[0].content == "Atlas owns the ledger"
        assert [h.content for h in with_vector] == [h.content for h in plain]
        assert all(h.vector_rank is None for h in with_vector)
        assert any("vector arm scored no memory" in n for n in with_vector.notes)

        # a caller's weight for the empty arm is ignored, not applied to nothing
        forced = db.recall(
            query, k=3, embedding=vec(1, 0), arm_weights={"vector": 5.0, "text": 2.0}
        )
        assert forced.weights == {"text": 2.0, "graph": 0.5}
    finally:
        db.close()
