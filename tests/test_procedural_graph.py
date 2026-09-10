"""The procedural demo must preserve direction, evidence, isolation and atomic history."""

from dataclasses import replace
from datetime import timedelta
import json

import pytest

from anatid import Anatid
from examples.procedural_graph import (
    CHECKS,
    INITIAL,
    REPAIR,
    T0,
    T1,
    T2,
    TEST,
    TRAIN,
    Graph,
    ProcedureStore,
    evaluate,
    evolve,
    rollout,
    score,
)


@pytest.fixture
def store(db):
    result = ProcedureStore(db)
    result.seed(INITIAL)
    return result


def test_directed_neighborhood_retains_order_and_fallback():
    graph = Graph((INITIAL.transitions[0], INITIAL.transitions[1], REPAIR, *CHECKS))
    assert [(hop, t.target) for hop, t in graph.neighborhood("read")] == [
        (1, "check"),
        (2, "answer"),
        (2, "abstain"),
    ]
    assert graph.neighborhood("answer") == ()  # Never walk an incoming prerequisite backward.
    assert graph.neighborhood("read", hops=0) == ()
    assert {t for _, t in graph.neighborhood("unrecognized_tool")} == set(graph.transitions)
    with pytest.raises(ValueError, match="nonnegative"):
        graph.neighborhood("read", hops=-1)


def test_repair_changes_execution_and_keeps_history_and_evidence(store):
    frozen = store.snapshot()
    result = evolve(store, REPAIR, CHECKS, now=T1)
    assert result["accepted"]
    assert (result["before"], result["candidate"]) == (1, 4)
    current = store.snapshot()
    assert score(evaluate(frozen, TEST)) == 1
    assert score(evaluate(current, TEST)) == 4
    assert rollout(current, TRAIN[1]).actions == ("start", "search", "read", "check", "answer")
    assert rollout(current, TRAIN[1]).citation == "note-c"
    assert store.snapshot(T0 + timedelta(hours=1)) == frozen
    assert store.snapshot(T1) == current  # Half-open temporal boundary.
    assert rollout(frozen, TRAIN[1]).answer == "Bo"  # In-flight checkpoint is unchanged.
    mid = next(mid for mid, t in store.rows() if t.key == "finish")
    provenance = store.db.provenance(mid)
    assert len(provenance.chain) == 2
    assert any("validation_traces" in ep.content for ep in provenance.episodes)
    assert provenance.source_text.startswith("Synthetic initial runbook")


def test_rejected_shortcut_keeps_graph_and_records_negative_evidence(store):
    evolve(store, REPAIR, CHECKS, now=T1)
    retained = store.snapshot()
    result = evolve(store, INITIAL.transitions[-1], (), now=T2)
    assert not result["accepted"]
    assert result["candidate"] == 1
    assert store.snapshot() == retained
    rejected = store.db.context(
        store.entity("evolution"), kinds=[f"procedure_rejected:{store.name}"]
    )
    assert len(rejected) == 1
    payload = json.loads(rejected[0].content)
    assert payload["replacement"]["target"] == "answer"
    assert any(not t["success"] for t in payload["validation_traces"])
    assert payload["training_traces"]


@pytest.mark.parametrize(
    "invalid", [replace(REPAIR, target="dead_end"), replace(REPAIR, guidance="")]
)
def test_invalid_proposal_is_rejected_before_candidate_rollout(store, invalid):
    before = store.snapshot()
    result = evolve(store, invalid, (), now=T1)
    assert not result["accepted"]
    assert result["structural_error"]
    assert result["candidate"] is None
    assert result["validation_traces"] == []
    assert store.snapshot() == before


def test_equal_validation_score_is_accepted(store):
    result = evolve(
        store, replace(INITIAL.transitions[-1], guidance="Use the first passage."), (), now=T1
    )
    assert result["accepted"]
    assert result["before"] == result["candidate"] == 1


def test_failed_commit_rolls_back_memory_edges_and_audit_record(store, monkeypatch):
    before = store.rows()
    old_stats = store.db.stats()

    def fail(*args, **kwargs):
        raise RuntimeError("simulated add failure")

    monkeypatch.setattr(store, "add", fail)  # Fails after correct has moved the first edge.
    with pytest.raises(RuntimeError, match="simulated add failure"):
        evolve(store, REPAIR, CHECKS, now=T1)
    assert store.rows() == before
    assert store.db.stats() == old_stats
    assert store.db.get_entity(store.entity("evolution")) is None


def test_graph_names_are_scoped_within_a_tenant(store):
    other = ProcedureStore(store.db, "other_graph")
    other.seed(Graph((replace(INITIAL.transitions[-1], source="start"),)))
    assert len(other.snapshot().transitions) == 1
    assert len(store.snapshot().transitions) == 3
    evolve(store, REPAIR, CHECKS, now=T1)
    assert other.snapshot().transitions[0].target == "answer"


def test_reopen_preserves_current_and_historical_graphs(tmp_path):
    path = tmp_path / "procedures.anatid"
    with Anatid.open(path, tenant=1) as db:
        store = ProcedureStore(db)
        store.seed(INITIAL)
        before = store.snapshot()
        evolve(store, REPAIR, CHECKS, now=T1)
        after = store.snapshot()
    with Anatid.open(path, tenant=1) as db:
        store = ProcedureStore(db)
        assert store.snapshot() == after
        assert store.snapshot(T0 + timedelta(hours=1)) == before
    with Anatid.open(path, tenant=2) as db:
        other = ProcedureStore(db)
        assert other.rows() == ()
        other.seed(INITIAL)
        assert len(other.snapshot().transitions) == 3
    with Anatid.open(path, tenant=1) as db:
        assert ProcedureStore(db).snapshot() == after


def test_repeated_seed_does_not_duplicate_rules(store):
    before = store.rows()
    with pytest.raises(ValueError, match="already exists"):
        store.seed(INITIAL)
    assert store.rows() == before
