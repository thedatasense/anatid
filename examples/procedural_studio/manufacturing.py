"""Fictional Cedar manufacturing records and a declared packet-review policy.

These rules are simulation rules, not regulatory requirements or device acceptance criteria.
Neither terminal action releases a lot: both produce a packet for an authorized human reviewer.
"""

from dataclasses import asdict, dataclass

from examples.procedural_graph import Graph, Transition


POLICY = (
    "Fictional Cedar packet-review policy: ready_for_review requires an approved passing final-test "
    "report for this exact lot, configuration, and work-instruction revision, an approved traveler, "
    "no withdrawal referencing that report, and no open nonconformance for the lot. A test plan is "
    "not a completed result. Otherwise hold_for_review and cite the gap. Both statuses are advisory "
    "packet routing to an authorized reviewer; neither releases a device or establishes safety."
)

INITIAL = Graph(
    (
        Transition(
            "search",
            "start",
            "search",
            "always",
            "Retrieve the lot's manufacturing packet.",
            "Preserve document identifiers and references.",
        ),
        Transition(
            "read",
            "search",
            "read",
            "always",
            "Read the final-test report and traveler.",
            "A green test result may be incomplete evidence.",
        ),
        Transition(
            "finish",
            "read",
            "answer",
            "always",
            "Summarize the passing test report and mark the packet ready for review.",
            "Keep the review summary concise.",
        ),
    )
)
REPAIR = Transition(
    "finish",
    "read",
    "check",
    "always",
    "Reconcile the lot, configuration, work-instruction revision, approved traveler, report withdrawals, and open nonconformances.",
    "A passing result from another configuration, a withdrawn result, or a test plan cannot establish packet completeness.",
    "REQUIRES",
)
CHECKS = (
    Transition(
        "verified",
        "check",
        "answer",
        "supported",
        "Prepare a ready-for-human-review summary with the supporting record IDs.",
        "Do not release the lot or claim device safety.",
    ),
    Transition(
        "missing",
        "check",
        "abstain",
        "unsupported",
        "Hold the packet for human review; cite each missing or conflicting record.",
        "Do not treat a retest plan as a passing result or ignore an open nonconformance.",
    ),
)


@dataclass(frozen=True)
class Record:
    record_id: str
    kind: str
    title: str
    narrative: str
    lot: str
    configuration: str = "HW-B / FW-1.1"
    instruction: str = "WI-17 revision C"
    status: str = "approved"
    result: str = ""
    references: tuple[str, ...] = ()


@dataclass(frozen=True)
class LotCase:
    name: str
    label: str
    lot: str
    records: tuple[Record, ...]
    expected: str
    configuration: str = "HW-B / FW-1.1"
    instruction: str = "WI-17 revision C"

    @property
    def question(self):
        return (
            f"Is the manufacturing evidence packet for Cedar lot {self.lot} ready for human review?"
        )


@dataclass(frozen=True)
class Finding:
    status: str
    reasons: tuple[str, ...]
    citations: tuple[str, ...]


@dataclass(frozen=True)
class Outcome:
    name: str
    actions: tuple[str, ...]
    answer: str
    citation: str | None
    success: bool
    reasons: tuple[str, ...]
    citations: tuple[str, ...]


def make_case(prefix: str, pattern: str) -> LotCase:
    """Gold is chosen from the scenario event, before the evidence interpreter runs."""
    lot = f"CED-{prefix}"
    report_id, traveler_id = f"FT-{prefix}", f"DHR-{prefix}"
    report = Record(
        report_id,
        "final_test",
        "Final functional test",
        "Final functional test recorded PASS for this lot.",
        lot,
        result="pass",
    )
    traveler = Record(
        traveler_id,
        "traveler",
        "Approved assembly traveler",
        "Assembly traveler reviewed and approved for the specified configuration and work instruction.",
        lot,
    )
    extra = ()
    if pattern == "wrong_configuration":
        report = Record(
            report_id,
            "final_test",
            "Passing test · old configuration",
            "PASS was recorded on HW-A / FW-1.0 using WI-17 revision B, before the sensor change.",
            lot,
            "HW-A / FW-1.0",
            "WI-17 revision B",
            result="pass",
        )
        extra = (
            Record(
                f"ECO-{prefix}",
                "change",
                "Sensor change implemented",
                "This lot uses HW-B / FW-1.1 and WI-17 revision C. The earlier configuration does not meet this packet's declared evidence scope.",
                lot,
                references=(report_id,),
            ),
        )
    elif pattern == "withdrawn":
        extra = (
            Record(
                f"COR-{prefix}",
                "withdrawal",
                "Test report withdrawn",
                "The test fixture loaded the wrong firmware image. The referenced passing report was withdrawn.",
                lot,
                references=(report_id,),
            ),
            Record(
                f"TP-{prefix}",
                "test_plan",
                "Replacement test planned",
                "A retest is scheduled; no replacement result is included in this packet.",
                lot,
                references=(report_id,),
            ),
        )
    elif pattern == "open_nonconformance":
        extra = (
            Record(
                f"NCR-{prefix}",
                "nonconformance",
                "Sensor rework unresolved",
                "Rework remains open after a sensor assembly deviation. Closure and post-rework verification are not recorded.",
                lot,
                status="open",
                references=(report_id,),
            ),
        )
    elif pattern == "plan_only":
        report = Record(
            f"TP-{prefix}",
            "test_plan",
            "Approved final-test plan",
            "The final-test plan is approved. Execution and a result are not yet recorded.",
            lot,
        )
    elif pattern != "complete":
        raise ValueError(f"unknown scenario {pattern}")
    labels = {
        "complete": "Complete packet",
        "wrong_configuration": "Wrong configuration / instruction",
        "withdrawn": "Withdrawn pass + planned retest",
        "open_nonconformance": "Open nonconformance",
        "plan_only": "Plan without a result",
    }
    return LotCase(
        f"{prefix}-{pattern}",
        labels[pattern],
        lot,
        (report, traveler, *extra),
        "ready_for_review" if pattern == "complete" else "hold_for_review",
    )


PATTERNS = ("complete", "wrong_configuration", "withdrawn", "open_nonconformance", "plan_only")
TRAIN = (make_case("2409", "withdrawn"), make_case("2410", "complete"))
VALIDATION = tuple(make_case(f"V{i + 1:02}", pattern) for i, pattern in enumerate(PATTERNS))
TEST = tuple(make_case(f"T{i + 1:02}", pattern) for i, pattern in enumerate(PATTERNS))


def inspect_packet(case: LotCase) -> Finding:
    """Interpret source fields under POLICY; never inspect case.expected."""
    rows = tuple(r for r in case.records if r.lot == case.lot)
    withdrawn = {ref for r in rows if r.kind == "withdrawal" for ref in r.references}
    eligible = tuple(
        r
        for r in rows
        if r.kind == "final_test"
        and r.status == "approved"
        and r.result == "pass"
        and r.configuration == case.configuration
        and r.instruction == case.instruction
        and r.record_id not in withdrawn
    )
    travelers = tuple(
        r
        for r in rows
        if r.kind == "traveler"
        and r.status == "approved"
        and r.configuration == case.configuration
        and r.instruction == case.instruction
    )
    open_ncr = tuple(r for r in rows if r.kind == "nonconformance" and r.status == "open")
    reasons = []
    for r in rows:
        if r.kind == "final_test":
            if r.configuration != case.configuration or r.instruction != case.instruction:
                reasons.append(
                    f"{r.record_id}: configuration or work-instruction revision does not match the lot."
                )
            if r.record_id in withdrawn:
                reasons.append(
                    f"{r.record_id}: its passing claim was withdrawn by a linked correction."
                )
            if r.status != "approved" or r.result != "pass":
                reasons.append(f"{r.record_id}: not an approved passing result.")
        elif r.kind == "test_plan":
            reasons.append(f"{r.record_id}: an approved plan is not an executed passing test.")
    reasons.extend(f"{r.record_id}: nonconformance remains open." for r in open_ncr)
    if not eligible:
        reasons.append(
            "No applicable, unwithdrawn, approved passing final-test result is available."
        )
    if not travelers:
        reasons.append("An approved traveler for this lot and configuration is missing.")
    supported = bool(eligible and travelers and not open_ncr)
    if supported:
        reasons = [
            "An applicable approved passing test and approved traveler are present; no withdrawal or open nonconformance is recorded."
        ]
    return Finding(
        "ready_for_review" if supported else "hold_for_review",
        tuple(reasons),
        tuple(r.record_id for r in rows),
    )


def rollout(graph: Graph, case: LotCase) -> Outcome:
    """Scripted replay of the stored procedure; it does not call a model or release a lot."""
    active, actions, finding = "start", ["start"], None
    for _ in range(12):
        applies = {
            "always",
            "supported" if finding and finding.status == "ready_for_review" else "unsupported",
        }
        options = [
            t for hop, t in graph.neighborhood(active) if hop == 1 and t.condition in applies
        ]
        if not options:
            break
        active = options[0].target
        actions.append(active)
        if active == "check":
            finding = inspect_packet(case)
        if active in {"answer", "abstain"}:
            break
    status = "ready_for_review" if active == "answer" else "hold_for_review"
    citations = finding.citations if finding else tuple(r.record_id for r in case.records[:1])
    reasons = (
        finding.reasons
        if finding
        else (
            "The original shortcut treated the first document as sufficient; reconciliation was skipped.",
        )
    )
    return Outcome(
        case.name,
        tuple(actions),
        status,
        citations[0] if citations else None,
        active in {"answer", "abstain"} and status == case.expected,
        reasons,
        citations,
    )


def evaluate(graph: Graph, cases) -> tuple[Outcome, ...]:
    return tuple(rollout(graph, case) for case in cases)


def public_case(case: LotCase) -> dict:
    """Source-only context for the model: excludes gold, scores, and scripted answers."""
    return {
        "name": case.name,
        "lot": case.lot,
        "configuration": case.configuration,
        "instruction": case.instruction,
        "question": case.question,
        "records": [asdict(r) for r in case.records],
    }
