"""Procedural graph mechanics on Anatid; offline, with a scripted solver and refiner.

Run: python examples/procedural_graph.py
Inspired by Lu et al., arXiv:2609.09153v1, sections 3.1-3.3. This is an original
synthetic demonstration, not a reproduction of the paper's LLM experiments.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from anatid import Anatid
from anatid.visibility import Visibility


T0 = datetime(2026, 9, 1)
T1 = T0 + timedelta(days=1)
T2 = T1 + timedelta(days=1)


@dataclass(frozen=True)
class Transition:
    key: str
    source: str
    target: str
    condition: str
    guidance: str
    pitfalls: str
    relation: str = "LEADS_TO"

    def content(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass(frozen=True)
class Graph:
    """An immutable checkpoint, frozen for the duration of a solver rollout."""

    transitions: tuple[Transition, ...]

    def neighborhood(self, active: str, hops: int = 2) -> tuple[tuple[int, Transition], ...]:
        """Outgoing edges only, by hop; unknown actions fall back to the full graph."""
        if hops < 0:
            raise ValueError("hops must be nonnegative")
        nodes = {n for t in self.transitions for n in (t.source, t.target)}
        if active not in nodes:
            return tuple((0, t) for t in self.transitions)
        frontier = {active}
        visited = {active}
        result = []
        for hop in range(1, hops + 1):
            edges = [t for t in self.transitions if t.source in frontier]
            result.extend((hop, t) for t in edges)
            frontier = {t.target for t in edges} - visited
            visited |= frontier
        return tuple(result)

    def guidance(self, active: str) -> str:
        """Serialize connected context for a guidance model; no condition is executed here."""
        lines = [f"Active procedure: {active}"]
        for hop, t in self.neighborhood(active):
            label = f"hop {hop}" if hop else "full-graph fallback"
            lines.append(
                f"[{label}] {t.source} -{t.relation}-> {t.target}\n"
                f"  condition: {t.condition}\n  guidance: {t.guidance}\n"
                f"  pitfalls: {t.pitfalls}"
            )
        return "\n".join(lines)

    def validate(self) -> None:
        """Reject duplicate rules and reachable dead ends; recovery cycles are allowed."""
        keys = [t.key for t in self.transitions]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate transition key")
        if any(not all(asdict(t).values()) for t in self.transitions):
            raise ValueError("transition fields must be nonempty")
        if any(t.source in {"answer", "abstain"} for t in self.transitions):
            raise ValueError("terminal procedures cannot have outgoing transitions")
        reachable = {"start"}
        can_finish = {"answer", "abstain"}
        for _ in range(len(self.transitions) + 1):
            reachable |= {t.target for t in self.transitions if t.source in reachable}
            can_finish |= {t.source for t in self.transitions if t.target in can_finish}
        if not reachable <= can_finish:
            raise ValueError("a reachable procedure has no path to a terminal")


class ProcedureStore:
    """Example adapter over existing tables, scoped to one graph and the handle's tenant.

    Each transition is a memory with evidence, ABOUT its two procedure entities.
    A uniquely named RELATES_TO edge stores direction; the memory stores attributes.
    All writes use existing verbs, including their index journaling and transactions.
    Loading scans this small procedure graph; this is not a large-graph query engine.
    """

    def __init__(self, db: Anatid, name: str = "evidence_qa"):
        self.db = db
        self.name = name
        self.kind = f"procedure:{name}"

    def entity(self, node: str) -> str:
        return f"{self.name}/{node}"

    def relation(self, t: Transition) -> tuple[str, str, str]:
        # Unique per rule and orientation: unrelate otherwise closes both directions.
        return self.entity(t.source), self.entity(t.target), f"{self.kind}:{t.key}"

    def add(self, t: Transition, evidence: str, now: datetime) -> int:
        src, dst, kind = self.relation(t)
        with self.db.transaction():
            memory = self.db.remember(
                t.content(),
                entities=[src, dst],
                kind=self.kind,
                entity_kind="procedure",
                episode=evidence,
                writer="procedural-demo",
                now=now,
            )
            self.db.relate(src, dst, rel_kind=kind, episode_id=memory.episode_id, now=now)
        return memory.memory_id

    def rows(self, as_of: datetime | None = None) -> tuple[tuple[int, Transition], ...]:
        visibility = Visibility.at(self.db.resolve_tenant(None).tenant_id, as_of)
        predicates, params = [], []
        for alias in ("m", "r", "s", "d"):
            predicate, values = visibility.predicate(alias)
            predicates.append(predicate)
            params.extend(values)
        rows = self.db.execute(
            "SELECT m.memory_id, m.content, s.name, d.name FROM memories m "
            "JOIN edges_relates r ON r.rel_kind = "
            "m.kind || ':' || json_extract_string(try_cast(m.content AS JSON), '$.key') "
            "JOIN entities s ON s.entity_id = r.src "
            "JOIN entities d ON d.entity_id = r.dst "
            f"WHERE {' AND '.join(predicates)} AND m.kind = ? ORDER BY r.rel_kind",
            [*params, self.kind],
        ).fetchall()
        result = []
        for memory_id, content, source, target in rows:
            t = Transition(**json.loads(content))
            if (source, target) != (self.entity(t.source), self.entity(t.target)):
                raise ValueError("transition attributes disagree with stored edge direction")
            result.append((memory_id, t))
        return tuple(result)

    def snapshot(self, as_of: datetime | None = None) -> Graph:
        return Graph(tuple(t for _, t in self.rows(as_of)))

    def seed(self, graph: Graph) -> None:
        graph.validate()
        with self.db.transaction():
            if self.rows():
                raise ValueError("procedure graph already exists")
            for t in graph.transitions:
                self.add(t, "Synthetic initial runbook, revision 1: trust the first passage.", T0)


INITIAL = Graph(
    (
        Transition(
            "search", "start", "search", "always", "Find source passages.", "Keep source IDs."
        ),
        Transition(
            "read",
            "search",
            "read",
            "always",
            "Read the retrieved passages.",
            "Do not use snippets alone.",
        ),
        Transition(
            "finish",
            "read",
            "answer",
            "always",
            "Submit the final answer from the first passage.",
            "Keep the response concise.",
        ),
    )
)
REPAIR = Transition(
    "finish",
    "read",
    "check",
    "always",
    "Verify the draft against the newest source revision.",
    "Search order does not establish which source is current.",
    "REQUIRES",
)
CHECKS = (
    Transition(
        "verified",
        "check",
        "answer",
        "supported",
        "Submit the final answer with its source ID.",
        "Do not submit an unverified draft.",
    ),
    Transition(
        "missing",
        "check",
        "abstain",
        "unsupported",
        "State that evidence is missing.",
        "Do not invent an answer.",
    ),
)


@dataclass(frozen=True)
class Case:
    name: str
    # Synthetic search results, in retrieval order: (revision, value, source ID).
    passages: tuple[tuple[int, str, str], ...]
    expected: str | None


@dataclass(frozen=True)
class Outcome:
    name: str
    actions: tuple[str, ...]
    answer: str | None
    citation: str | None
    success: bool


TRAIN = (
    Case("train-current", ((1, "Ada", "note-a"),), "Ada"),
    Case("train-stale", ((1, "Bo", "note-b"), (2, "Cy", "note-c")), "Cy"),
)
VALIDATION = (
    Case("val-current", ((3, "Dee", "note-d"),), "Dee"),
    Case("val-stale", ((1, "Eli", "note-e"), (4, "Flo", "note-f")), "Flo"),
    Case("val-shuffled", ((2, "Gus", "note-g"), (1, "Hal", "note-h"), (5, "Ira", "note-i")), "Ira"),
    Case("val-missing", (), None),
)
TEST = (
    Case("test-current", ((2, "Jay", "note-j"),), "Jay"),
    Case("test-stale", ((2, "Kai", "note-k"), (7, "Lou", "note-l")), "Lou"),
    Case(
        "test-shuffled", ((3, "Moe", "note-m"), (8, "Nia", "note-n"), (1, "Ori", "note-o")), "Nia"
    ),
    Case("test-missing", (), None),
)


def rollout(graph: Graph, case: Case) -> Outcome:
    """Deterministic test double, not an LLM: follow the first applicable outgoing edge.

    It knows three condition labels and four mock tools. The check tool reads source
    revisions, never the expected answer. Expected answers are used only for scoring.
    A real solver receives graph.guidance(active) as advice and chooses its own actions.
    """
    active, answer, citation = "start", None, None
    actions = [active]
    for _ in range(12):
        applicable = {"always", "supported" if case.passages else "unsupported"}
        options = [
            t for hop, t in graph.neighborhood(active) if hop == 1 and t.condition in applicable
        ]
        if not options:
            break
        active = options[0].target
        actions.append(active)
        if active == "read":
            _, answer, citation = case.passages[0] if case.passages else (0, "unknown", None)
        elif active == "check" and case.passages:
            _, answer, citation = max(case.passages, key=lambda p: p[0])
        elif active in {"answer", "abstain"}:
            if active == "abstain":
                answer, citation = None, None
            break
    success = (
        active in {"answer", "abstain"}
        and answer == case.expected
        and (active == "abstain" if case.expected is None else citation is not None)
    )
    return Outcome(case.name, tuple(actions), answer, citation, success)


def evaluate(graph: Graph, cases: tuple[Case, ...]) -> tuple[Outcome, ...]:
    return tuple(rollout(graph, case) for case in cases)


def score(outcomes: tuple[Outcome, ...]) -> int:
    return sum(o.success for o in outcomes)


def evolve(
    store: ProcedureStore,
    replacement: Transition,
    additions: tuple[Transition, ...],
    *,
    now: datetime,
) -> dict:
    """Evaluate a copied candidate, then atomically retain it or record its rejection.

    This example replaces one rule and optionally adds rules. Proposals are scripted.
    Validation cases are separate from diagnostic training cases and final test cases.
    Evaluation is cheap and local, so the transaction spans the read and commit; an
    LLM-backed implementation should evaluate outside it and compare checkpoint versions.
    """
    with store.db.transaction():
        rows = store.rows()
        before = Graph(tuple(t for _, t in rows))
        old_id, old = next((mid, t) for mid, t in rows if t.key == replacement.key)
        candidate = Graph(
            tuple(replacement if t.key == old.key else t for _, t in rows) + additions
        )
        baseline = evaluate(before, VALIDATION)
        error = None
        try:
            candidate.validate()
        except ValueError as exc:
            error = str(exc)
        outcomes = () if error else evaluate(candidate, VALIDATION)
        accepted = error is None and score(outcomes) >= score(baseline)
        record = {
            "accepted": accepted,
            "structural_error": error,
            "before": score(baseline),
            "candidate": None if error else score(outcomes),
            "validation_size": len(VALIDATION),
            "replacement": asdict(replacement),
            "additions": [asdict(t) for t in additions],
            "training_traces": [asdict(o) for o in evaluate(before, TRAIN)],
            "validation_traces": [asdict(o) for o in outcomes],
        }
        evidence = json.dumps(record, sort_keys=True)
        if accepted:
            store.db.correct(
                old_id,
                replacement.content(),
                entities=[store.entity(replacement.source), store.entity(replacement.target)],
                remove_relations=[store.relation(old)],
                add_relations=[store.relation(replacement)],
                episode=evidence,
                writer="scripted-refiner",
                now=now,
            )
            for t in additions:
                store.add(t, evidence, now)
        store.db.remember(
            evidence,
            entities=[store.entity("evolution")],
            kind=f"procedure_{'accepted' if accepted else 'rejected'}:{store.name}",
            episode=evidence,
            writer="validation-gate",
            now=now,
        )
    return record


def demo(db: Anatid) -> None:
    store = ProcedureStore(db)
    store.seed(INITIAL)
    frozen = store.snapshot()
    print("Procedural graphs on Anatid — deterministic mechanics demo, no LLM\n")
    before = rollout(frozen, TRAIN[1])
    print(
        f"Flawed prior: {' -> '.join(before.actions)}; answer={before.answer}; pass={before.success}"
    )
    accepted = evolve(store, REPAIR, CHECKS, now=T1)
    print(
        f"Repair validation: {accepted['before']}/4 -> {accepted['candidate']}/4; accepted={accepted['accepted']}"
    )
    current = store.snapshot()
    print("\nConnected guidance after reading:\n" + current.guidance("read"))
    flat = db.recall("submit final answer", seed_entity=None, kinds=[store.kind], k=2)
    print("\nFlat BM25 top-2 for 'submit final answer':")
    for hit in flat:
        t = Transition(**json.loads(hit.content))
        print(f"  {t.source} -> {t.target}")
    print("Flat hits are relevant rules; they do not guarantee a connected route from 'read'.")
    print("\nUntouched synthetic test cases (same solver, tools and data):")
    for label, graph in (("initial graph", frozen), ("repaired graph", current)):
        outcomes = evaluate(graph, TEST)
        print(f"  {label}: {score(outcomes)}/{len(outcomes)} passed")
    repaired = rollout(current, TRAIN[1])
    print(
        f"Repaired trace: {' -> '.join(repaired.actions)}; answer={repaired.answer}; source={repaired.citation}"
    )
    rejected = evolve(
        store,
        replace(INITIAL.transitions[-1], guidance="Save a step; submit immediately."),
        (),
        now=T2,
    )
    print(
        f"\nProposed shortcut: {rejected['before']}/4 -> {rejected['candidate']}/4; accepted={rejected['accepted']}"
    )
    print("Rejected proposal and validation traces retained as procedure_rejected memory.")
    historical = store.snapshot(T0 + timedelta(hours=1))
    print(f"Historical replay: {' -> '.join(rollout(historical, TRAIN[1]).actions)}")
    mid = next(mid for mid, t in store.rows() if t.key == "finish")
    provenance = db.provenance(mid)
    print(
        f"Repair provenance: {len(provenance.chain)} linked memories; evidence includes validation traces."
    )
    print("\nThese fixture scores demonstrate mechanics, not measured LLM accuracy gains.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, help="Optional NEW database file; default is in-memory.")
    args = parser.parse_args()
    if args.db and args.db.exists():
        parser.error("--db must name a new file; existing files are never reset")
    with Anatid.open(str(args.db) if args.db else ":memory:", tenant=1) as db:
        demo(db)


if __name__ == "__main__":
    main()
