"""An independent, seeded history and answer key for a fictional device program.

No anatid imports: expected answers come from the event schedule, not the database under test.
The structured fields and references are public source exports, rendered into every document.
They are not secret gold patches. Dates, rules and results are invented for this simulation.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

DISCLAIMER = (
    "SYNTHETIC RESEARCH ONLY. No Medtronic records, patient data, clinical acceptance criteria "
    "or regulatory determinations. Missing evidence does not establish that a device is unsafe."
)


@dataclass(frozen=True)
class Record:
    record_id: str
    kind: str
    product: str
    effective_on: str
    recorded_on: str
    narrative: str
    fields: dict[str, Any] = field(default_factory=dict)
    references: tuple[str, ...] = ()

    def visible(self, effective_on: str, known_on: str) -> bool:
        return self.effective_on <= effective_on and self.recorded_on <= known_on

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        import json

        return (
            f"# {self.record_id} — {self.kind}\n"
            f"Product: {self.product}; effective: {self.effective_on}; "
            f"recorded: {self.recorded_on}\n\n{self.narrative}\n\n"
            "Structured source export (available to every system):\n"
            + json.dumps({"fields": self.fields, "references": self.references}, sort_keys=True)
        )


@dataclass(frozen=True)
class Query:
    qid: str
    category: str
    operation: str
    question: str
    product: str
    requirement: str
    revision: str
    configuration: str
    effective_on: str
    known_on: str
    anchor: str


@dataclass(frozen=True)
class Question:
    query: Query
    expected: dict[str, Any]
    support: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self.query), "expected": self.expected, "support": self.support}


@dataclass(frozen=True)
class Case:
    """The simulator's facts. Neither a retrieval system nor its answerer receives this object."""

    requirement: str
    component: str
    change: str
    old_report: str
    claimed_report: str
    correction: str
    plan: str
    retest: str
    draft: str
    other_variant: str
    decision: str
    outcome: str
    rationale: str
    owner: str

    def expectation(self, query: Query) -> tuple[dict[str, Any], tuple[str, ...]]:
        """Gold from the independently scheduled events, without reading a rendered record."""
        req = f"{self.requirement}@B"
        if query.category == "unanswerable":
            return {"status": "unknown"}, ()
        if query.operation == "owner":
            return {"status": "found", "owner": self.owner}, (req,)
        if query.operation == "rationale":
            return {"status": "found", "rationale": self.rationale}, (self.decision, self.change)
        if query.operation == "impact":
            return {
                "status": "found",
                "requirements": [req],
                "protocols": [self.plan],
            }, (self.change, req, self.plan)

        # The initial test claims a pass on March 10. On April 1 that claim is withdrawn,
        # effective back to March 10. The replacement is approved May 3, recorded May 5.
        eligible: list[str] = []
        support = [
            req,
            self.change,
            self.old_report,
            self.claimed_report,
            self.draft,
            self.other_variant,
        ]
        if query.known_on < "2025-04-01":
            eligible.append(self.claimed_report)
        else:
            support.append(self.correction)
        if query.effective_on >= "2025-04-02" and query.known_on >= "2025-04-02":
            support.append(self.plan)
        if query.effective_on >= "2025-05-03" and query.known_on >= "2025-05-05":
            support.append(self.retest)
            if self.outcome == "pass":
                eligible.append(self.retest)
        return {
            "status": "supported" if eligible else "not_established",
            "eligible_reports": sorted(eligible),
        }, tuple(sorted(support))


@dataclass(frozen=True)
class World:
    seed: int
    records: tuple[Record, ...]
    questions: tuple[Question, ...]
    cases: tuple[Case, ...]

    def verify(self) -> None:
        """Check source integrity and prevent future evidence from entering question gold."""
        records = {r.record_id: r for r in self.records}
        if len(records) != len(self.records):
            raise ValueError("duplicate source record identifier")
        for record in self.records:
            date.fromisoformat(record.effective_on)
            date.fromisoformat(record.recorded_on)
            for ref in record.references:
                if ref.startswith("doc:") and ref[4:] not in records:
                    raise ValueError(f"dangling reference {ref} in {record.record_id}")
        for question in self.questions:
            q = question.query
            for rid in question.support:
                if rid not in records or not records[rid].visible(q.effective_on, q.known_on):
                    raise ValueError(f"missing or future support {rid} for {q.qid}")


COMPONENTS = (
    "pressure sensor",
    "battery monitor",
    "door interlock",
    "alarm speaker",
    "air detector",
    "display controller",
    "motor controller",
    "power connector",
    "wireless module",
    "event logger",
)


def _case(index: int, rng: random.Random) -> Case:
    n = index * 1000
    component = COMPONENTS[index % len(COMPONENTS)]
    return Case(
        requirement=f"R-{44 + n}",
        component=component,
        change=f"ECO-{23 + n}",
        old_report=f"TR-{88 + n}",
        claimed_report=f"TR-{103 + n}",
        correction=f"COR-{12 + n}",
        plan=f"TP-{104 + n}",
        retest=f"TR-{104 + n}",
        draft=f"TR-DRAFT-{n}",
        other_variant=f"TR-VARIANT-{n}",
        decision=f"DEC-{23 + n}",
        outcome="pass" if index == 0 else rng.choice(("pass", "pass", "fail", "draft")),
        rationale=f"The replacement {component} changed the interface assumptions; "
        "the review required new configuration-specific evidence instead of reusing "
        "the earlier report.",
        owner=rng.choice(("Systems Engineering", "Verification Engineering", "Device Software")),
    )


def _records(case: Case, product: str, rng: random.Random) -> list[Record]:
    out: list[Record] = []
    req = case.requirement
    family = f"requirement:{req}"

    def add(rid, kind, effective, recorded, narrative, fields, refs=()):
        out.append(Record(rid, kind, product, effective, recorded, narrative, fields, tuple(refs)))

    for rev, config, effective in (
        ("A", "HW-A/FW-1.0", "2024-01-15"),
        ("B", "HW-B/FW-1.1", "2025-02-20"),
    ):
        add(
            f"{req}@{rev}",
            "requirement",
            effective,
            effective,
            f"Controlled requirement revision {rev} for the fictional {case.component}. "
            f"This record applies to {config}, not all releases of {product}. "
            "Approval of this requirement does not itself establish successful verification. "
            "The simulated program requires approved passing results for the exact requirement "
            "revision and hardware/firmware configuration. A withdrawn claim cannot be used.",
            {
                "requirement": req,
                "revision": rev,
                "configuration": config,
                "approved": True,
                "owner": case.owner,
                "evidence_reuse": "exact configuration only",
            },
            (family,),
        )

    add(
        case.change,
        "change_assessment",
        "2025-02-20",
        "2025-02-20",
        f"The {case.component} supplier change creates a new configuration. "
        "The change board did not approve reuse of earlier verification. "
        "The listed requirement revision and protocol require reassessment. "
        "Unlisted subsystems are not automatically invalidated by this decision.",
        {"approved": True, "requirements": [f"{req}@B"], "protocols": [case.plan]},
        (family, f"doc:{req}@B", f"doc:{case.plan}"),
    )
    add(
        case.decision,
        "decision",
        "2025-02-20",
        "2025-02-21",
        case.rationale,
        {"requirement": req, "rationale": case.rationale, "approved": True, "change": case.change},
        (family, f"doc:{case.change}"),
    )

    def report(rid, rev, config, effective, recorded, result, approved):
        add(
            rid,
            "test_report",
            effective,
            recorded,
            rng.choice(
                (
                    (
                        f"Verification report for the {case.component}. The executed configuration "
                        "and the approved scope are listed below. A matching report title is not "
                        "evidence that a different firmware image was tested."
                    ),
                    (
                        f"The team evaluated the {case.component} using the controlled protocol. "
                        "Use the configuration manifest and requirement revision below when assessing "
                        "applicability. This is a synthetic result, not a clinical performance claim."
                    ),
                )
            ),
            {
                "requirement": req,
                "revision": rev,
                "configuration": config,
                "result": result,
                "approved": approved,
            },
            (family, f"doc:{req}@{rev}"),
        )

    report(case.old_report, "A", "HW-A/FW-1.0", "2024-07-10", "2024-07-12", "pass", True)
    report(case.claimed_report, "B", "HW-B/FW-1.1", "2025-03-10", "2025-03-12", "pass", True)
    report(case.draft, "B", "HW-B/FW-1.1", "2025-03-18", "2025-03-18", "pass", False)
    report(case.other_variant, "B", "HW-C/FW-2.0", "2025-03-18", "2025-03-18", "pass", True)
    # Deliberately no requirement identifier in this correction: the report reference is the
    # association. All systems receive that reference, and can exploit it if implemented.
    add(
        case.correction,
        "withdrawal",
        "2025-03-10",
        "2025-04-01",
        f"Review of the firmware-image manifest for {case.claimed_report} found the wrong "
        "image was executed. Its claimed applicability is withdrawn from the original test "
        "date. This correction became known only when recorded; it must not leak into an "
        "earlier reconstruction of what the review team knew.",
        {
            "target": case.claimed_report,
            "approved": True,
            "reason": "wrong firmware image; applicability claim withdrawn",
        },
        (f"doc:{case.claimed_report}",),
    )
    add(
        case.plan,
        "test_plan",
        "2025-04-02",
        "2025-04-02",
        "A replacement execution is scheduled. This protocol is approved for execution; "
        "it is not a test report and records neither completion nor a passing result.",
        {
            "requirement": req,
            "revision": "B",
            "configuration": "HW-B/FW-1.1",
            "approved": True,
            "execution": "planned",
        },
        (family, f"doc:{req}@B"),
    )
    report(
        case.retest,
        "B",
        "HW-B/FW-1.1",
        "2025-05-03",
        "2025-05-05",
        "fail" if case.outcome == "fail" else "pass",
        case.outcome != "draft",
    )
    return out


def _questions(case: Case, product: str, first: int) -> list[Question]:
    scenarios = (
        ("historical_claim", "coverage", "2025-03-20", "2025-03-20"),
        ("retrospective_correction", "coverage", "2025-03-20", "2025-04-15"),
        ("evidence_gap", "coverage", "2025-04-15", "2025-04-15"),
        ("not_yet_recorded", "coverage", "2025-05-03", "2025-05-04"),
        ("late_record_available", "coverage", "2025-05-03", "2025-05-06"),
        ("current_coverage", "coverage", "2025-06-01", "2025-06-01"),
        ("decision_rationale", "rationale", "2025-04-15", "2025-04-15"),
        ("change_impact", "impact", "2025-04-15", "2025-04-15"),
        ("simple_lookup", "owner", "2025-04-15", "2025-04-15"),
        ("unanswerable", "owner", "2025-04-15", "2025-04-15"),
    )
    out = []
    for i, (category, operation, effective, known) in enumerate(scenarios):
        req = case.requirement if category != "unanswerable" else f"UNRECORDED-{case.requirement}"
        subject = f"{req} revision B on HW-B/FW-1.1"
        if operation == "coverage":
            question = f"Which approved passing reports establish evidence for {subject}?"
        elif operation == "rationale":
            question = f"Why did {case.change} reject reuse of earlier evidence for {subject}?"
        elif operation == "impact":
            question = f"Which requirement revisions and protocols did {case.change} flag for reassessment?"
        else:
            question = f"Which team owns {subject}?"
        query = Query(
            qid=f"Q{first + i:04d}",
            category=category,
            operation=operation,
            question=f"{question} Effective date {effective}; use only records known by {known}.",
            product=product,
            requirement=req,
            revision="B",
            configuration="HW-B/FW-1.1",
            effective_on=effective,
            known_on=known,
            anchor=f"doc:{case.change}" if operation == "impact" else f"requirement:{req}",
        )
        expected, support = case.expectation(query)
        out.append(Question(query, expected, support))
    return out


def build_world(seed: int = 7, *, documents: int = 500, questions: int = 100) -> World:
    if questions < 1:
        raise ValueError("questions must be positive")
    rng = random.Random(seed)
    product = f"Cedar-{seed}: fictional infusion pump"
    cases = tuple(_case(i, rng) for i in range(math.ceil(questions / 10)))
    records = [r for case in cases for r in _records(case, product, rng)]
    if documents < len(records):
        raise ValueError(f"{questions} questions require at least {len(records)} source documents")
    for i in range(documents - len(records)):
        component = rng.choice(COMPONENTS)
        day = date(2024, rng.randrange(1, 13), rng.randrange(1, 28)).isoformat()
        records.append(
            Record(
                f"BACKGROUND-{i:06d}",
                "meeting_note",
                product,
                day,
                day,
                f"The {component} working group discussed verification evidence, supplier changes, "
                "alarm requirements and firmware release history. This meeting concerns another "
                "work package. Participants proposed reviewing the test protocol at the next "
                "design review. No execution result, controlled requirement change or release "
                "approval was recorded. Earlier discussion text was copied into the agenda for "
                "context; the minutes do not supersede any controlled report.",
                {"work_package": f"OTHER-{i}", "approved": False},
                (f"work-package:OTHER-{i}",),
            )
        )
    generated = [q for i, case in enumerate(cases) for q in _questions(case, product, i * 10 + 1)]
    world = World(
        seed,
        tuple(sorted(records, key=lambda r: (r.recorded_on, r.record_id))),
        tuple(generated[:questions]),
        cases,
    )
    world.verify()
    return world
