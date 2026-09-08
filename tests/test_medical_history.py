"""Guard the synthetic example's truth, time boundaries and baseline fairness."""

from dataclasses import replace
import json

import pytest

from anatid import AsOf, HashEmbedder
from examples.medical_history.__main__ import main, run
from examples.medical_history.evaluate import assess, interpret
from examples.medical_history.systems import RetrievalSystems, pack, timestamp, token_estimate
from examples.medical_history.world import build_world


@pytest.fixture
def world():
    return build_world(7, documents=30, questions=10)


@pytest.fixture
def systems(world, tmp_path):
    result = RetrievalSystems(world.records, HashEmbedder(dim=64), tmp_path / "history.anatid")
    try:
        yield result
    finally:
        result.close()


@pytest.mark.parametrize("seed", [7, 11, 23])
def test_gold_agrees_with_source_only_interpreter(seed):
    world = build_world(seed)
    assert len(world.records) == 500
    assert len(world.questions) == 100
    for question in world.questions:
        q = question.query
        visible = [r for r in world.records if r.visible(q.effective_on, q.known_on)]
        answer = interpret(q, visible)
        assert answer.value == question.expected, q.qid
        assert assess(question, visible, answer)["supported_correct"], q.qid


def test_seed_reproducibility_and_noise_only_growth():
    small = build_world(7, documents=30, questions=10)
    assert small == build_world(7, documents=30, questions=10)
    large = build_world(7, documents=100, questions=10)
    assert small.questions == large.questions
    assert {r.record_id for r in small.records} <= {r.record_id for r in large.records}
    larger = {r.record_id: r for r in large.records}
    assert all(r == larger[r.record_id] for r in small.records)
    assert small != build_world(11, documents=30, questions=10)


@pytest.mark.parametrize(("documents", "questions"), [(10, 10), (500, 0), (-1, 10)])
def test_invalid_world_sizes(documents, questions):
    with pytest.raises(ValueError):
        build_world(documents=documents, questions=questions)


def test_source_export_contains_every_interpreted_field(world):
    for record in world.records:
        exported = json.loads(
            record.render().split("Structured source export (available to every system):\n")[1]
        )
        assert exported["fields"] == record.fields
        assert exported["references"] == list(record.references)


def test_retrospective_withdrawal_and_late_result(world):
    contexts = list(world.records)  # Even accidental future context is rejected by the interpreter.
    values = [interpret(question.query, contexts).value for question in world.questions[:6]]
    assert values[0] == {"status": "supported", "eligible_reports": ["TR-103"]}
    assert (
        values[1] == values[2] == values[3] == {"status": "not_established", "eligible_reports": []}
    )
    assert values[4] == values[5] == {"status": "supported", "eligible_reports": ["TR-104"]}


def test_different_config_draft_and_plan_are_not_evidence(world):
    answer = interpret(world.questions[2].query, list(world.records))
    assert answer.exclusions == {
        "TR-88": "wrong requirement revision",
        "TR-103": "withdrawn by COR-12",
        "TR-DRAFT-0": "not approved",
        "TR-VARIANT-0": "wrong hardware/firmware configuration",
        "TP-104": "planned execution is not a passing report",
    }


def test_gold_is_not_recomputed_from_retrieved_fields(world):
    original = world.questions[0].expected.copy()
    modified = [
        replace(r, fields={**r.fields, "result": "fail"}) if r.record_id == "TR-103" else r
        for r in world.records
    ]
    answer = interpret(world.questions[0].query, modified)
    assert answer.value != original
    assert world.questions[0].expected == original


def test_accidental_gap_answer_is_not_supported_correct(world):
    question = world.questions[2]
    context = [r for r in world.records if r.record_id == "R-44@B"]
    score = assess(question, context, interpret(question.query, context))
    assert score["exact_answer"]
    assert not score["supported_correct"]
    context.append(next(r for r in world.records if r.record_id == "TR-103"))
    score = assess(question, context, interpret(question.query, context))
    assert score["false_support"]
    assert "COR-12" in score["missing_evidence"]


def test_pack_whole_record_budget_and_deduplication(world):
    first, second = world.records[:2]
    assert pack([first], token_estimate(first) - 1) == []
    assert pack([first, first, second], token_estimate(first)) == [first]
    assert pack([first, second], token_estimate(first) + token_estimate(second)) == [first, second]


def test_every_system_obeys_visibility_and_context_budget(world, systems):
    for question in world.questions:
        q = question.query
        for context in systems.retrieve(q, budget=700).values():
            assert sum(token_estimate(r) for r in context) <= 700
            assert len({r.record_id for r in context}) == len(context)
            assert all(
                r.product == q.product and r.visible(q.effective_on, q.known_on) for r in context
            )


def test_two_hop_reference_resolves_correction_for_both_trace_systems(world, systems):
    q = world.questions[2].query
    assert "COR-12" not in systems.sql_neighborhood(q, hops=1)
    assert "COR-12" in systems.sql_neighborhood(q, hops=2)
    scope = AsOf(timestamp(q.effective_on), timestamp(q.known_on))
    one_hop = systems.db.recall_2hop(q.anchor, hops=1, limit=100, as_of=scope)
    two_hop = systems.db.recall_2hop(q.anchor, hops=2, limit=100, as_of=scope)
    assert "COR-12" not in {systems.memory_to_record[m.memory_id] for m in one_hop}
    assert "COR-12" in {systems.memory_to_record[m.memory_id] for m in two_hop}
    for question in world.questions:
        contexts = systems.retrieve(question.query)
        assert contexts["trace-sql"] == contexts["anatid-trace"]
        assert assess(
            question, contexts["anatid-trace"], interpret(question.query, contexts["anatid-trace"])
        )["supported_correct"]


def test_anatid_native_dates_hide_future_knowledge(world, systems):
    report_mid = next(mid for mid, rid in systems.memory_to_record.items() if rid == "TR-104")
    correction_mid = next(mid for mid, rid in systems.memory_to_record.items() if rid == "COR-12")
    assert (
        systems.db.get(report_mid, as_of=AsOf(timestamp("2025-05-03"), timestamp("2025-05-04")))
        is None
    )
    assert (
        systems.db.get(report_mid, as_of=AsOf(timestamp("2025-05-03"), timestamp("2025-05-06")))
        is not None
    )
    assert (
        systems.db.get(correction_mid, as_of=AsOf(timestamp("2025-03-20"), timestamp("2025-03-20")))
        is None
    )
    correction = systems.db.get(
        correction_mid, as_of=AsOf(timestamp("2025-03-20"), timestamp("2025-04-15"))
    )
    assert correction is not None
    provenance = systems.db.provenance(correction.memory_id)
    assert any("wrong firmware image" in episode.content for episode in provenance.episodes)


def test_raw_and_anatid_receive_identical_embeddings_and_source(systems):
    for mid, rid in systems.memory_to_record.items():
        memory = systems.db.get(mid)
        assert memory.content == systems.records[rid].render()
        assert memory.embedding == pytest.approx(systems.vectors[rid])


def test_existing_database_refused_before_embedding(world, tmp_path):
    path = tmp_path / "existing.anatid"
    path.write_text("user-owned data")

    class NoCalls:
        dim = 64

        def embed(self, texts):
            pytest.fail("must refuse before embedding")

    with pytest.raises(FileExistsError):
        RetrievalSystems(world.records, NoCalls(), path)
    assert path.read_text() == "user-owned data"


def test_cli_offline_artifacts_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("ANATID_EMBEDDING_KEY", "must-not-be-read-or-printed")
    output = tmp_path / "run"
    main(["--seeds", "7", "--documents", "30", "--questions", "10", "--out", str(output)])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["metadata"]["documents"] == 30
    assert "NOT semantic" in summary["metadata"]["embedding"]
    assert len((output / "seed-7" / "results.jsonl").read_text().splitlines()) == 50
    assert len((output / "seed-7" / "documents.jsonl").read_text().splitlines()) == 30
    report = (output / "REPORT.md").read_text()
    assert "withdrawn by COR-12" in report
    with pytest.raises(FileExistsError):
        run(
            output,
            seeds=[7],
            documents=30,
            questions=10,
            budget=2000,
            candidates=100,
            embedder=HashEmbedder(dim=64),
            embedding_label="offline",
        )
    assert (output / "REPORT.md").read_text() == report


@pytest.mark.parametrize(
    "args",
    [
        ["--embedding-model", "test"],
        ["--embedding-url", "https://example.com/v1"],
        ["--embedding-url", "https://secret@example.com/v1", "--embedding-model", "test"],
        ["--embedding-dim", "0"],
    ],
)
def test_cli_rejects_invalid_endpoint_configuration_before_network(args):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
