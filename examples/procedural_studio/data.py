"""Build the visual's data by executing Anatid's real storage and evolution operations."""

from dataclasses import asdict
from datetime import timedelta
import json

from anatid import Anatid
from examples.procedural_graph import (
    T0,
    T1,
    T2,
    ProcedureStore,
    evolve,
    score,
)
from .manufacturing import (
    CHECKS,
    INITIAL,
    REPAIR,
    TEST,
    TRAIN,
    VALIDATION,
    POLICY,
    evaluate,
    rollout,
    public_case,
)


def build_demo() -> dict:
    """Produce immutable replay data in a fresh in-memory database, without a model or key."""
    with Anatid.open(":memory:", tenant=1) as db:
        store = ProcedureStore(db, "cedar_manufacturing")
        store.seed(
            INITIAL,
            "Synthetic Cedar manufacturing runbook R1: summarize the first test document and advance the packet to human review.",
        )
        original = store.snapshot()
        accepted = evolve(
            store, REPAIR, CHECKS, now=T1, training=TRAIN, validation=VALIDATION, evaluator=evaluate
        )
        repaired = store.snapshot()
        rejected = evolve(
            store,
            INITIAL.transitions[-1],
            (),
            now=T2,
            training=TRAIN,
            validation=VALIDATION,
            evaluator=evaluate,
        )
        retained = store.snapshot()
        historical = store.snapshot(T0 + timedelta(hours=1))
        checkpoints = {
            "original": original,
            "repaired": repaired,
            "retained": retained,
            "historical": historical,
        }
        cases = (TRAIN[0], *TEST)
        records = {}
        with db.transaction():
            for case in cases:
                for record in case.records:
                    content = json.dumps(asdict(record), sort_keys=True)
                    db.remember(
                        content,
                        kind="manufacturing_record",
                        entities=[case.lot, record.record_id],
                        episode=content,
                        writer="synthetic-manufacturing-export",
                        now=T0,
                    )
                    for reference in record.references:
                        db.relate(record.record_id, reference, rel_kind="references", now=T0)
                fetched = {
                    json.loads(m.content)["record_id"]: json.loads(m.content)
                    for m in db.context(case.lot, kinds=["manufacturing_record"])
                }
                records[case.name] = [fetched[r.record_id] for r in case.records]
        mid = next(mid for mid, rule in store.rows() if rule.key == "finish")
        provenance = db.provenance(mid)
        flat = db.recall("review summary", seed_entity=None, kinds=[store.kind], k=2)
        negative = db.context(store.entity("evolution"), kinds=[f"procedure_rejected:{store.name}"])
        return {
            "policy": POLICY,
            "domain": "Fictional Cedar infusion-pump manufacturing",
            "test_labels": [case.label for case in TEST],
            "llm": {"enabled": False, "model": None},
            "graphs": {
                key: [asdict(t) for t in graph.transitions] for key, graph in checkpoints.items()
            },
            "cases": [
                {
                    "label": case.label,
                    **public_case(case),
                    "records": records[case.name],
                    "expected": case.expected,
                    "runs": {
                        key: asdict(rollout(graph, case)) for key, graph in checkpoints.items()
                    },
                }
                for case in cases
            ],
            "accepted": accepted,
            "rejected": rejected,
            "negative_memories": [json.loads(m.content) for m in negative],
            "flat": [json.loads(hit.content) for hit in flat],
            "evaluation": {
                key: {
                    "passed": score(evaluate(graph, TEST)),
                    "total": len(TEST),
                    "outcomes": [asdict(o) for o in evaluate(graph, TEST)],
                }
                for key, graph in checkpoints.items()
            },
            "validation_size": len(VALIDATION),
            "provenance": [
                {
                    "id": str(m.memory_id),
                    "rule": json.loads(m.content),
                    "writer": m.writer,
                    "time": m.valid_from.isoformat() + "Z",
                    "evidence": next(
                        (ep.content for ep in provenance.episodes if ep.episode_id == m.episode_id),
                        None,
                    ),
                }
                for m in provenance.chain
            ],
        }
