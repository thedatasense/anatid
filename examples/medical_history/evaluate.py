"""One transparent source-only interpreter shared by every retrieval system.

This measures evidence retrieval under explicit simulated rules, not LLM answer quality.
The interpreter never receives a World, Case, expected answer, or support set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .world import Query, Question, Record


@dataclass
class Answer:
    value: dict[str, Any]
    citations: list[str] = field(default_factory=list)
    exclusions: dict[str, str] = field(default_factory=dict)


def interpret(query: Query, context: list[Record]) -> Answer:
    records = {
        r.record_id: r
        for r in context
        if r.product == query.product and r.visible(query.effective_on, query.known_on)
    }
    unknown = Answer({"status": "unknown"})
    if query.operation == "impact":
        change = records.get(query.anchor.removeprefix("doc:"))
        if (
            change is None
            or change.kind != "change_assessment"
            or not change.fields.get("approved")
        ):
            return unknown
        requirements = sorted(change.fields["requirements"])
        protocols = sorted(change.fields["protocols"])
        return Answer(
            {"status": "found", "requirements": requirements, "protocols": protocols},
            sorted(
                {change.record_id, *(rid for rid in requirements + protocols if rid in records)}
            ),
        )
    if query.operation == "rationale":
        decisions = [
            r
            for r in records.values()
            if r.kind == "decision"
            and r.fields.get("requirement") == query.requirement
            and r.fields.get("approved")
        ]
        if not decisions:
            return unknown
        decision = max(decisions, key=lambda r: (r.effective_on, r.recorded_on, r.record_id))
        citations = [decision.record_id]
        if decision.fields["change"] in records:
            citations.append(decision.fields["change"])
        return Answer(
            {"status": "found", "rationale": decision.fields["rationale"]}, sorted(citations)
        )

    requirement = records.get(f"{query.requirement}@{query.revision}")
    if (
        requirement is None
        or requirement.kind != "requirement"
        or not requirement.fields.get("approved")
        or requirement.fields.get("configuration") != query.configuration
    ):
        return unknown
    if query.operation == "owner":
        return Answer(
            {"status": "found", "owner": requirement.fields["owner"]}, [requirement.record_id]
        )
    if query.operation != "coverage":
        raise ValueError(f"unknown operation {query.operation}")

    citations = {requirement.record_id}
    exclusions: dict[str, str] = {}
    eligible = []
    withdrawals = {
        r.fields["target"]: r
        for r in records.values()
        if r.kind == "withdrawal" and r.fields.get("approved")
    }
    for record in records.values():
        fields = record.fields
        if record.kind == "change_assessment" and requirement.record_id in fields.get(
            "requirements", []
        ):
            citations.add(record.record_id)
        if fields.get("requirement") != query.requirement:
            continue
        if record.kind == "test_plan":
            citations.add(record.record_id)
            exclusions[record.record_id] = "planned execution is not a passing report"
        if record.kind != "test_report":
            continue
        citations.add(record.record_id)
        if record.record_id in withdrawals:
            withdrawal = withdrawals[record.record_id]
            citations.add(withdrawal.record_id)
            exclusions[record.record_id] = f"withdrawn by {withdrawal.record_id}"
        elif fields.get("revision") != query.revision:
            exclusions[record.record_id] = "wrong requirement revision"
        elif fields.get("configuration") != query.configuration:
            exclusions[record.record_id] = "wrong hardware/firmware configuration"
        elif not fields.get("approved"):
            exclusions[record.record_id] = "not approved"
        elif fields.get("result") != "pass":
            exclusions[record.record_id] = "not a passing result"
        else:
            eligible.append(record.record_id)
    return Answer(
        {
            "status": "supported" if eligible else "not_established",
            "eligible_reports": sorted(eligible),
        },
        sorted(citations),
        exclusions,
    )


def assess(question: Question, context: list[Record], answer: Answer) -> dict[str, Any]:
    """Only the scorer can see gold. Exact matches alone can reward accidental abstention."""
    retrieved = {r.record_id for r in context}
    support = set(question.support)
    found = support & retrieved
    exact = answer.value == question.expected
    complete = support <= retrieved
    grounded = support <= set(answer.citations)
    q = question.query
    return {
        "qid": q.qid,
        "category": q.category,
        "expected": question.expected,
        "answer": asdict(answer),
        "retrieved": [r.record_id for r in context],
        "required_evidence": sorted(support),
        "missing_evidence": sorted(support - retrieved),
        "exact_answer": exact,
        "evidence_recall": len(found) / len(support) if support else 1.0,
        "complete_evidence": complete,
        "supported_correct": exact and complete and grounded,
        "false_support": bool(
            set(answer.value.get("eligible_reports", []))
            - set(question.expected.get("eligible_reports", []))
        ),
        "visibility_violations": sum(
            r.product != q.product or not r.visible(q.effective_on, q.known_on) for r in context
        ),
    }
