"""Render the world into notes, gold memory patches and a question set.

::

    python -m bench.quality.gen_corpus            # write bench/quality/data/*.jsonl
    python -m bench.quality.gen_corpus --verify   # also ingest the gold through anatid and check it

The world (:mod:`bench.quality.world`) is the source of truth.  This module only renders:

* ``notes.jsonl``: one short note per line, in arrival order.  ``text`` is the body every
  system reads; ``rendered`` is the body with its date and source in front, the exact string
  the harness should hand to every system so provenance questions are answerable by all of
  them.  A note carries one to three events; phrasing varies so keyword overlap between a
  question and its supporting note is not guaranteed.
* ``gold_patches.jsonl``: for each note, the :class:`anatid.ingest.MemoryPatch` a perfect
  extractor would propose (``MemoryPatch.to_dict()`` shape), with ``source_text`` set to the
  note's ``rendered`` string.  Corrections name the fact they replace by ``old_text``; the
  pipeline resolves it.  This feeds the S5 oracle.
* ``questions.jsonl``: about 150 questions in six categories with the gold answer, accepted
  aliases, the ids of the notes that support the answer, a judge rubric and, where the
  question has a stale wrong answer, the distractors.

``--verify`` ingests the gold patches into an in-memory anatid database with the
:class:`~anatid.ingest.ScriptedExtractor`, then checks that no correction was downgraded, that
the only dedupe drops are the reminder notes, that the final graph matches the world's final
state, and that every temporal question is answered by an ``as_of`` read.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import DATA_DIR
from .world import (
    DEFAULT_SEED,
    DEPENDENCIES,
    NOW,
    WINDOW_CHANGES,
    WINDOWED,
    Event,
    World,
    content_dep,
    content_oncall,
    content_owner,
    content_window,
)

QUESTIONS_PER_CATEGORY = 25
CATEGORIES = (
    "single_fact",
    "knowledge_update",
    "temporal",
    "multi_hop",
    "provenance",
    "abstention",
)
IDK = "I don't know"

# --------------------------------------------------------------------------- notes


@dataclass
class Note:
    note_id: str
    seq: int
    date: _dt.date
    source: str
    source_kind: str
    team: str
    author: str
    text: str
    events: list[Event] = field(default_factory=list)

    @property
    def rendered(self) -> str:
        return f"[{self.date.isoformat()}] {self.source}: {self.text}"

    def to_dict(self) -> dict:
        return {
            "note_id": self.note_id,
            "seq": self.seq,
            "date": self.date.isoformat(),
            "source": self.source,
            "source_kind": self.source_kind,
            "team": self.team,
            "author": self.author,
            "text": self.text,
            "rendered": self.rendered,
            "event_ids": [e.event_id for e in self.events],
            "event_kinds": [e.kind for e in self.events],
        }


def _cap(text: str) -> str:
    """Sentence-initial form: capitalise the first letter ("the ledger" -> "The ledger")."""
    return text[0].upper() + text[1:]


def _poss(text: str) -> str:
    """Possessive of a surface form: "the ledger's", "webhooks'"."""
    return text + "'" if text.endswith("s") else text + "'s"


class Renderer:
    """Turn one event into one or two sentences, varying the phrasing and the names used."""

    def __init__(self, world: World, rng: random.Random) -> None:
        self.w = world
        self.rng = rng

    # surface forms
    def s(self, name: str) -> str:
        return self.rng.choice(self.w.svc(name).forms)

    def t(self, team: str) -> str:
        return self.rng.choice(self.w.teams[team].forms)

    def p(self, person: str) -> str:
        return self.w.people[person].full_name if self.rng.random() < 0.25 else person

    def pick(self, templates: list[str], **kw) -> str:
        return self.rng.choice(templates).format(**kw)

    def render(self, e: Event) -> str:  # one branch per event kind
        pl = e.payload
        k = e.kind
        if k == "membership_init":
            team, t0 = self.t(pl["team"]), _cap(self.t(pl["team"]))
            names = [self.p(m) for m in pl["members"]]
            if len(names) == 2:
                return self.pick(
                    [
                        "{T0} this year is {p1} and {p2}.",
                        "{p1} and {p2} make up {T}.",
                        "For the org chart: {p1} and {p2} are on {T}.",
                        "{T0} roster: {p1} and {p2}.",
                    ],
                    T=team,
                    T0=t0,
                    p1=names[0],
                    p2=names[1],
                )
            return self.pick(
                [
                    "{p1} is the only engineer on {T} for now.",
                    "{T0} is a one-person team at the moment: {p1}.",
                    "{p1} runs {T} solo until the next hire lands.",
                ],
                T=team,
                T0=t0,
                p1=names[0],
            )
        if k == "move":
            return self.pick(
                [
                    "{p} moves from {old} to {new} today.",
                    "As of today {p} is on {new}; {old} loses a pair of hands.",
                    "Team change: {p} leaves {old} and joins {new}, effective today.",
                    "{p} has transferred to {new}. {old0} continues without them.",
                ],
                p=self.p(pl["person"]),
                old=self.t(pl["old"]),
                old0=_cap(self.t(pl["old"])),
                new=self.t(pl["new"]),
            )
        if k == "oncall_switch":
            return self.pick(
                [
                    "{p} takes the {Tn} pager from {prev}, who is leaving the team.",
                    "With {prev} moving on, {p} is on call for {T} from today.",
                    "On-call for {T} passes from {prev} to {p} with the move.",
                ],
                p=self.p(pl["person"]),
                prev=self.p(pl["prev"]),
                T=self.t(pl["team"]),
                Tn=pl["team"],
            )
        if k == "oncall_init":
            return self.pick(
                [
                    "{p} is on call for {T} to start the year.",
                    "{Tn} pager: {p} has it first.",
                    "{p} takes the first {T} on-call shift of the year.",
                    "First on-call rotation for {T} goes to {p}.",
                ],
                p=self.p(pl["person"]),
                T=self.t(pl["team"]),
                T0=_cap(self.t(pl["team"])),
                Tn=pl["team"],
            )
        if k == "oncall":
            return self.pick(
                [
                    "{p} has the {Tn} pager this cycle, taking over from {prev}.",
                    "On-call rotation: {p} is primary for {T} starting today; {prev} is off rotation.",
                    "{T0} on-call swapped this morning. {p} carries the phone until the next rotation.",
                    "Reminder from {T}: page {p} for anything this cycle, not {prev}.",
                    "{p} picked up the {Tn} on-call shift from {prev} today.",
                    "Rotation day for {T}: {prev} hands the pager to {p}.",
                ],
                p=self.p(pl["person"]),
                prev=self.p(pl["prev"]),
                T=self.t(pl["team"]),
                T0=_cap(self.t(pl["team"])),
                Tn=pl["team"],
            )
        if k == "ownership_init":
            s = self.s(pl["service"])
            return self.pick(
                [
                    "{T0} owns {s}.",
                    "For the record, {s} is owned by {T}.",
                    "{S0} lives with {T}.",
                    "{S0} is a {T} service.",
                    "{T0} looks after {s}.",
                ],
                s=s,
                S0=_cap(s),
                T=self.t(pl["team"]),
                T0=_cap(self.t(pl["team"])),
            )
        if k in ("handover", "handover_wrong"):
            s = self.s(pl["service"])
            return self.pick(
                [
                    "Handover: as of today {s} moves from {old} to {new}. {oldp} walked {newp} through the runbook this morning.",
                    "{new0} is taking over {s} from {old}, effective today. Dashboards and alerts are being repointed.",
                    "Ownership change: {s} now sits with {new}. {old0} no longer owns it.",
                    "From today the pager for {s} rings {new}, not {old}. Ownership transferred.",
                    "{old0} handed {s} to {new} today; {oldp} wrote up the handover doc for {newp}.",
                    "Quick one: {s} belongs to {new} starting today. Thanks to {old} for running it until now.",
                    "{S0} has a new home: {new}. {old0} kept it until today.",
                ],
                s=s,
                S0=_cap(s),
                old=self.t(pl["old"]),
                old0=_cap(self.t(pl["old"])),
                new=self.t(pl["new"]),
                new0=_cap(self.t(pl["new"])),
                oldp=self.p(pl["old_person"]),
                newp=self.p(pl["new_person"]),
            )
        if k == "handover_fix":
            s = self.s(pl["service"])
            return self.pick(
                [
                    "Correction to the handover note of {d}: {s} went to {truth}, not {wrong}. {wrong0} never owned it; the note was written against the wrong team.",
                    "Fixing an error in the {d} handover: the new owner of {s} is {truth}. {wrong0} was recorded by mistake and never held it.",
                    "The {d} note about {s} named the wrong team. {truth0} took it over on that date; {wrong} did not.",
                ],
                d=pl["wrong_date"].isoformat(),
                s=s,
                truth=self.t(pl["truth"]),
                truth0=_cap(self.t(pl["truth"])),
                wrong=self.t(pl["wrong"]),
                wrong0=_cap(self.t(pl["wrong"])),
            )
        if k == "dependency":
            s, d = self.s(pl["service"]), self.s(pl["dep"])
            return self.pick(
                [
                    "{S0} depends on {d} for {purpose}.",
                    "{S0} calls {d} for {purpose}, so treat {d} as a hard dependency of {s}.",
                    "Dependency map update: {s} needs {d} ({purpose}).",
                    "If {d} is down, {s} is down: it relies on it for {purpose}.",
                    "{S0} pulls {purpose} from {d} on every run.",
                ],
                s=s,
                S0=_cap(s),
                d=d,
                purpose=pl["purpose"],
            )
        if k == "dependency_rewire":
            s = self.s(pl["service"])
            return self.pick(
                [
                    "Decision: {what}, because {reason}. {S0} no longer talks to {olds}.",
                    "We decided that {what}. The reason: {reason}. That removes the dependency on {olds} from {s}.",
                    "{S0} switches from {olds} to {news} as its source. Why: {reason}. {S0} no longer depends on {olds}.",
                ],
                s=s,
                S0=_cap(s),
                what=pl["what"],
                reason=pl["reason"],
                olds=self.s(pl["old_dep"]),
                news=self.s(pl["new_dep"]),
            )
        if k == "dependency_new":
            s = self.s(pl["service"])
            return self.pick(
                [
                    "Decision: {what}, because {reason}. {S0} now depends on {d}.",
                    "{S0} takes on a new dependency: {d}. {what_cap}, because {reason}.",
                    "New dependency for {s}: {d}. {what_cap}; the reasoning is that {reason}.",
                ],
                s=s,
                S0=_cap(s),
                d=self.s(pl["dep"]),
                what=pl["what"],
                what_cap=pl["what"][0].upper() + pl["what"][1:],
                reason=pl["reason"],
            )
        if k == "window":
            s = self.s(pl["service"])
            return self.pick(
                [
                    "{S0} deploy window: {window}.",
                    "Reminder that {s} only ships {window}.",
                    "Per the change policy, {s} deploys {window}.",
                    "Deploys to {s} happen {window} and at no other time.",
                    "{S0} can be released {window}; anything else needs an exception.",
                ],
                s=s,
                S0=_cap(s),
                window=pl["window"],
            )
        if k == "window_change":
            s = self.s(pl["service"])
            return self.pick(
                [
                    "{S0p} deploy window changes today: it is now {new}. The old slot ({old_short}) is retired.",
                    "New deploy window for {s}: {new}. Forget the previous {old_short} slot.",
                    "From today {s} deploys {new} instead of {old}.",
                ],
                s=s,
                S0p=_cap(_poss(s)),
                new=pl["new"],
                old=pl["old"],
                old_short=pl["old_short"],
            )
        if k == "pin":
            s = self.s(pl["service"])
            return self.pick(
                [
                    "{S0} must stay on {version} until {condition}.",
                    "Do not upgrade {s} past {version} until {condition}.",
                    "{S0} is pinned to {version}; the pin lifts when {condition}.",
                ],
                s=s,
                S0=_cap(s),
                version=pl["version"],
                condition=pl["condition"],
            )
        if k == "approval":
            s = self.s(pl["service"])
            return self.pick(
                [
                    "Changes to {s} need {rule}.",
                    "Policy: every {s} change requires {rule}.",
                    "Before merging anything into {s}, make sure it has {rule}.",
                ],
                s=s,
                rule=pl["rule"],
            )
        if k == "freeze":
            s = self.s(pl["service"])
            return self.pick(
                [
                    "No deploys to {s} from {start} to {end} because of {reason}.",
                    "{S0} is frozen {start} through {end} for {reason}. Plan releases around it.",
                    "Deploy freeze for {s}: {start} to {end} ({reason}).",
                ],
                s=s,
                S0=_cap(s),
                start=pl["start"].isoformat(),
                end=pl["end"].isoformat(),
                reason=pl["reason"],
            )
        if k == "decision":
            what = pl["what"]
            return self.pick(
                [
                    "Decision: {what}, because {reason}.",
                    "We agreed today that {what}. The reason is that {reason}.",
                    "{what_cap}. Rationale: {reason}.",
                    "Decided in today's standup: {what}, since {reason}.",
                ],
                what=what,
                what_cap=what[0].upper() + what[1:],
                reason=pl["reason"],
            )
        if k == "incident":
            s = self.s(pl["service"])
            return self.pick(
                [
                    "Incident review {inc}. {S0} {symptom} on {d}. Root cause: {cause}. Decision: {what}, because {reason}.",
                    "{inc} review. What happened: {s} {symptom} on {d}. Why: {cause}. Follow-up decision: {what}, because {reason}.",
                    "Review of {inc} ({s}, {d}). The root cause was {cause}. We decided that {what}, because {reason}.",
                ],
                inc=pl["inc"],
                s=s,
                S0=_cap(s),
                symptom=pl["symptom"],
                d=pl["incident_date"].isoformat(),
                cause=pl["cause"],
                what=pl["what"],
                reason=pl["reason"],
            )
        if k == "observation":
            s = self.s(pl["service"])
            return self.pick(
                [
                    "{T0} noticed that {sp} {metric} {change} after {trigger} and pinged the owners.",
                    "Heads-up from {T}: {sp} {metric} {change} after {trigger}.",
                    "{T0} saw that {sp} {metric} {change} following {trigger}; not their service, so they flagged it.",
                    "Observation from {T}: after {trigger}, {sp} {metric} {change}.",
                ],
                T=self.t(pl["team"]),
                T0=_cap(self.t(pl["team"])),
                sp=_poss(s),
                metric=pl["metric"],
                change=pl["change"],
                trigger=pl["trigger"],
            )
        if k == "filler":
            fk = pl["kind"]
            if fk == "planning":
                return self.pick(
                    [
                        "{T0} holds planning on {v}s.",
                        "{T0} planning meeting is every {v}.",
                        "{T0} runs its planning on {v}s.",
                    ],
                    T0=_cap(self.t(pl["team"])),
                    v=pl["value"],
                )
            if fk == "practice":
                return self.pick(
                    ["{T0} adopted {v}.", "{T0} now does {v}.", "{T0} has switched to {v}."],
                    T0=_cap(self.t(pl["team"])),
                    v=pl["value"],
                )
            if fk == "captain":
                return self.pick(
                    ["{p} is the release captain for {v}.", "Release captain for {v}: {p}."],
                    p=self.p(pl["person"]),
                    v=pl["value"],
                )
            if fk == "review":
                return self.pick(
                    ["{p} runs {v}.", "{v_cap} is run by {p}."],
                    p=self.p(pl["person"]),
                    v=pl["value"],
                    v_cap=_cap(pl["value"]),
                )
            return self.pick(
                ["{T0} sits on the {v} floor.", "{T0} has moved to the {v} floor."],
                T0=_cap(self.t(pl["team"])),
                v=pl["value"],
            )
        if k == "reminder":
            what = pl["what"]
            if what == "window":
                s = self.s(pl["service"])
                return self.pick(
                    [
                        "Reminder: {s} deploys only {window}.",
                        "For newcomers: the deploy window for {s} is still {window}.",
                        "{S0} deploy window (unchanged): {window}.",
                    ],
                    s=s,
                    S0=_cap(s),
                    window=pl["window"],
                )
            if what == "dep":
                s = self.s(pl["service"])
                return self.pick(
                    [
                        "Reminder that {s} depends on {d}.",
                        "{S0} still relies on {d}; keep that in mind for the maintenance.",
                        "As before, {s} needs {d} to work.",
                    ],
                    s=s,
                    S0=_cap(s),
                    d=self.s(pl["dep"]),
                )
            if what == "owner":
                s = self.s(pl["service"])
                return self.pick(
                    [
                        "Reminder: {s} is owned by {T}.",
                        "{T0} still owns {s}, in case anyone asks.",
                        "{S0} remains a {T} service.",
                    ],
                    s=s,
                    S0=_cap(s),
                    T=self.t(pl["team"]),
                    T0=_cap(self.t(pl["team"])),
                )
            return self.pick(
                [
                    "Reminder: {p} is on call for {T} this cycle.",
                    "{p} still has the {Tn} pager.",
                    "For anything {Tn}-related, page {p}.",
                ],
                p=self.p(pl["person"]),
                T=self.t(pl["team"]),
                Tn=pl["team"],
            )
        raise ValueError(f"no template for event kind {k!r}")


def build_notes(world: World, rng: random.Random) -> list[Note]:
    """Bundle events into notes (one to three standup events per note) and render them."""
    renderer = Renderer(world, rng)
    drafts: list[
        tuple[_dt.date, int, str, str, list[Event]]
    ] = []  # date, first event, kind, team, events

    standups: dict[tuple[_dt.date, str], list[Event]] = defaultdict(list)
    for e in world.events:
        if e.source_kind == "standup":
            standups[(e.date, e.team)].append(e)
        else:
            drafts.append((e.date, e.event_id, e.source_kind, e.team, [e]))

    for (date, team), events in sorted(standups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        # bundled events first, as one chunk each; then the rest packed one to three per note
        chunks: list[list[Event]] = []
        bundles: dict[int, list[Event]] = defaultdict(list)
        loose: list[Event] = []
        for e in events:
            (bundles[e.bundle] if e.bundle is not None else loose).append(e)
        chunks.extend(bundles.values())
        current: list[Event] = []
        for e in loose:
            current.append(e)
            close = {1: 0.55, 2: 0.8}.get(len(current), 1.0)
            if rng.random() < close:
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)
        for chunk in chunks:
            drafts.append((date, min(x.event_id for x in chunk), "standup", team, chunk))

    drafts.sort(key=lambda d: (d[0], d[1]))
    per_label: Counter[str] = Counter()
    notes: list[Note] = []
    for seq, (date, _first, kind, team, events) in enumerate(drafts, start=1):
        if kind == "standup":
            base = f"standup/{team.lower()}/{date.isoformat()}"
            per_label[base] += 1
            label = base if per_label[base] == 1 else f"{base}-{per_label[base]}"
            members = world.members_at(team, date)
            author = rng.choice(members) if members else team
        else:
            label = events[0].label or f"{kind}/{date.isoformat()}"
            pl = events[0].payload
            if events[0].kind in ("handover", "handover_wrong"):
                author = pl["old_person"]
            else:
                members = world.members_at(team, date)
                author = rng.choice(members) if members else team
        text = " ".join(renderer.render(e) for e in events)
        notes.append(
            Note(
                note_id=f"n{seq:03d}",
                seq=seq,
                date=date,
                source=label,
                source_kind=kind,
                team=team,
                author=author,
                text=text,
                events=list(events),
            )
        )
    return notes


# --------------------------------------------------------------------------- gold patches


def gold_patch(note: Note) -> dict:
    """The patch a perfect extractor would propose for this note, in ``MemoryPatch.to_dict`` shape."""
    from anatid.ingest import AddFact, Correction, MemoryPatch, Relation

    facts, corrections, adds, removes = [], [], [], []
    for e in note.events:
        for f in e.facts:
            facts.append(AddFact(f["content"], tuple(f["entities"]), f.get("kind") or "fact"))
        for c in e.corrections:
            corrections.append(
                Correction(
                    c["new_content"],
                    old_text=c["old_text"],
                    entities=tuple(c["entities"]),
                    kind=c.get("kind"),
                )
            )
        for src, dst, rel in e.remove_relations:
            removes.append(Relation(src, dst, rel))
        for src, dst, rel in e.add_relations:
            adds.append(Relation(src, dst, rel))
    patch = MemoryPatch(
        source_text=note.rendered,
        add_facts=tuple(facts),
        corrections=tuple(corrections),
        add_relations=tuple(adds),
        remove_relations=tuple(removes),
    )
    return patch.to_dict()


# --------------------------------------------------------------------------- questions


def _date_forms(d: _dt.date) -> list[str]:
    return [
        d.isoformat(),
        f"{d.day} {d.strftime('%B %Y')}",
        f"{d.strftime('%B')} {d.day}, {d.year}",
    ]


RUBRICS = {
    "single_fact": (
        "Correct if the answer states the gold fact or an accepted alias; extra correct context "
        "is fine. A different value, or 'I don't know', is wrong."
    ),
    "knowledge_update": (
        "Correct only if the answer gives the CURRENT value (the gold). Naming an earlier value "
        "listed under distractors is wrong, even if it was once true. 'I don't know' is wrong."
    ),
    "temporal": (
        "Correct only if the answer gives the value that held at the date or moment the question "
        "names (the gold). The current value, if different, is wrong. 'I don't know' is wrong."
    ),
    "multi_hop": (
        "Correct if the final answer names the gold; the intermediate hops need not be stated. "
        "Where the gold is a list, every item must be named and no extra item. 'I don't know' is wrong."
    ),
    "provenance": (
        "Correct if the answer identifies the source label of the gold note, or its kind together "
        "with the gold date. The date alone with the right kind of note is acceptable; a "
        "different note or date is wrong."
    ),
    "abstention": (
        "Correct only if the answer says it does not know, or that the memory does not contain "
        "the answer. Any specific answer is wrong, however plausible."
    ),
}


class QuestionBank:
    def __init__(self, world: World, notes: list[Note], rng: random.Random) -> None:
        self.w = world
        self.rng = rng
        self.note_of: dict[int, Note] = {}
        for n in notes:
            for e in n.events:
                self.note_of[e.event_id] = n
        self.candidates: dict[str, list[dict]] = {c: [] for c in CATEGORIES}

    # aliases
    def team_aliases(self, t: str) -> list[str]:
        return list(self.w.teams[t].forms)

    def person_aliases(self, p: str) -> list[str]:
        return list(self.w.people[p].forms)

    def svc_aliases(self, s: str) -> list[str]:
        svc = self.w.svc(s)
        out = [svc.name, svc.display, *svc.forms]
        return list(dict.fromkeys(out))

    def nid(self, *event_ids: int) -> list[str]:
        return list(dict.fromkeys(self.note_of[e].note_id for e in event_ids))

    def add(
        self,
        category: str,
        subtype: str,
        question: str,
        gold: str,
        aliases: list[str],
        support: list[str],
        **extra,
    ) -> None:
        self.candidates[category].append(
            {
                "category": category,
                "subtype": subtype,
                "question": question,
                "gold": gold,
                "aliases": [a for a in dict.fromkeys(aliases) if a != gold],
                "support": support,
                "as_of": None,
                "hops": None,
                "distractors": [],
                "rubric": RUBRICS[category],
                **extra,
            }
        )
        q = self.candidates[category][-1]
        q["distractors"] = [d for d in dict.fromkeys(q["distractors"]) if d != gold]

    def label_gold(self, note: Note) -> tuple[str, list[str]]:
        kind_name = {
            "standup": "standup note",
            "handover": "handover message",
            "incident_review": "incident review",
        }[note.source_kind]
        gold = f"{note.source} ({note.date.isoformat()})"
        aliases = [
            note.source,
            *(_date_forms(note.date)),
            *(f"the {kind_name} of {d}" for d in _date_forms(note.date)),
        ]
        return gold, aliases

    # ---------------------------------------------------------------- categories

    def build(self) -> None:
        self.single_fact()
        self.knowledge_update()
        self.temporal()
        self.multi_hop()
        self.provenance()
        self.abstention()

    def single_fact(self) -> None:
        w = self.w
        cat = "single_fact"
        for name in WINDOWED:
            if name in WINDOW_CHANGES:
                continue
            ch = w.window[name].first
            ev = w.by_id[ch.event_id]
            self.add(
                cat,
                "deploy_window",
                self.rng.choice(
                    [
                        "What is the deploy window for {s}?",
                        "When can {s} be deployed?",
                        "Which deploy window applies to {s}?",
                    ]
                ).format(s=w.svc(name).display),
                ch.value,
                [ev.payload["short"]],
                self.nid(ch.event_id),
            )
        for pin in w.pins:
            s = w.svc(pin["service"]).display
            self.add(
                cat,
                "version_pin",
                f"Which version is {s} pinned to, and until when?",
                f"{pin['version']}, until {pin['condition']}",
                [pin["version"]],
                self.nid(pin["event"]),
                rubric=RUBRICS[cat] + " Both the version and the condition must be given.",
            )
        for ap in w.approvals:
            s = w.svc(ap["service"]).display
            self.add(
                cat,
                "approval_rule",
                f"What does a change to {s} need before it can merge?",
                ap["rule"],
                [],
                self.nid(ap["event"]),
            )
        for fr in w.freezes:
            s = w.svc(fr["service"]).display
            self.add(
                cat,
                "deploy_freeze",
                f"When is the deploy freeze for {s}, and why?",
                f"{fr['start'].isoformat()} to {fr['end'].isoformat()}, because of {fr['reason']}",
                [f"{fr['start'].isoformat()} to {fr['end'].isoformat()}", fr["reason"]],
                self.nid(fr["event"]),
            )
        for name in w.services:
            ivs = w.deps[name]
            if len(ivs) == 1 and ivs[0].end is None:
                dep = ivs[0].dep
                self.add(
                    cat,
                    "dependency",
                    self.rng.choice(
                        ["Which service does {s} depend on?", "What is {s}'s dependency?"]
                    ).format(s=w.svc(name).display),
                    dep,
                    self.svc_aliases(dep),
                    self.nid(ivs[0].add_event),
                )
        for dec in w.decisions:
            self.add(
                cat,
                "decision_reason",
                dec["q"],
                dec["reason"],
                [dec["short"]],
                self.nid(dec["event"]),
            )
        for inc in w.incidents:
            self.add(cat, "decision_reason", inc["q"], inc["reason"], [], self.nid(inc["event"]))
            self.add(
                cat,
                "incident_cause",
                f"What was the root cause of {inc['inc']}?",
                inc["cause"],
                [],
                self.nid(inc["event"]),
            )
            self.add(
                cat,
                "incident_date",
                f"On what date did {inc['inc']} happen?",
                inc["incident_date"].isoformat(),
                _date_forms(inc["incident_date"]),
                self.nid(inc["event"]),
            )
        movers = {m["person"] for m in w.moves}
        for p in w.people:
            if p in movers:
                continue
            ch = w.member[p].first
            self.add(
                cat,
                "team_membership",
                f"Which team is {p} on?",
                ch.value,
                self.team_aliases(ch.value),
                self.nid(ch.event_id),
            )
        for f in w.filler:
            if f["kind"] == "planning":
                self.add(
                    cat,
                    "filler",
                    f"On which day does {f['team']} hold its planning meeting?",
                    f"{f['value']}s",
                    [f["value"]],
                    self.nid(f["event"]),
                )
            elif f["kind"] == "practice":
                self.add(
                    cat,
                    "filler",
                    f"Which engineering practice did {f['team']} adopt?",
                    f["value"],
                    [],
                    self.nid(f["event"]),
                )
            elif f["kind"] == "captain":
                self.add(
                    cat,
                    "filler",
                    f"Who is the release captain for {f['value']}?",
                    f["person"],
                    self.person_aliases(f["person"]),
                    self.nid(f["event"]),
                )
            elif f["kind"] == "review":
                self.add(
                    cat,
                    "filler",
                    f"Who runs {f['value']}?",
                    f["person"],
                    self.person_aliases(f["person"]),
                    self.nid(f["event"]),
                )
            else:
                self.add(
                    cat,
                    "filler",
                    f"Which floor does {f['team']} sit on?",
                    f"the {f['value']} floor",
                    [f["value"]],
                    self.nid(f["event"]),
                )
        for wr in w.wrong_records:
            s = w.svc(wr["service"]).display
            self.add(
                cat,
                "wrong_record",
                f"Which team was wrongly recorded as the owner of {s} before the record was corrected?",
                wr["wrong"],
                self.team_aliases(wr["wrong"]),
                self.nid(wr["fix_event"], wr["wrong_event"]),
                distractors=[wr["truth"], wr["old"]],
            )

    def knowledge_update(self) -> None:
        w = self.w
        cat = "knowledge_update"
        for name in w.services:
            hist = w.owner[name]
            cur = hist.current
            support = [cur.event_id]
            for wr in w.wrong_records:
                if wr["service"] == name and wr["wrong_event"] == cur.event_id:
                    support.append(wr["fix_event"])
            previous = [c.value for c in hist.changes[:-1]]
            for wr in w.wrong_records:
                if wr["service"] == name:
                    previous.append(wr["wrong"])
            self.add(
                cat,
                "owner_now",
                self.rng.choice(
                    [
                        "Which team owns {s} now?",
                        "Who owns {s} today?",
                        "As of now, which team is responsible for {s}?",
                    ]
                ).format(s=w.svc(name).display),
                cur.value,
                self.team_aliases(cur.value),
                self.nid(*support),
                distractors=list(dict.fromkeys(t for t in previous if t != cur.value)),
            )
        for m in w.moves:
            self.add(
                cat,
                "team_now",
                f"Which team is {m['person']} on now?",
                m["new"],
                self.team_aliases(m["new"]),
                self.nid(m["event"]),
                distractors=[m["old"]],
            )
        for team in w.teams:
            hist = w.oncall[team]
            if len(hist.changes) < 2:
                continue
            cur = hist.current
            self.add(
                cat,
                "oncall_now",
                self.rng.choice(
                    ["Who is on call for {t} right now?", "Who currently holds the {t} pager?"]
                ).format(t=team),
                cur.value,
                self.person_aliases(cur.value),
                self.nid(cur.event_id),
                distractors=list(
                    dict.fromkeys(c.value for c in hist.changes if c.value != cur.value)
                ),
            )
        for name in WINDOW_CHANGES:
            hist = w.window[name]
            cur = hist.current
            ev = w.by_id[cur.event_id]
            self.add(
                cat,
                "window_now",
                f"What is the current deploy window for {w.svc(name).display}?",
                cur.value,
                [ev.payload["short"]],
                self.nid(cur.event_id),
                distractors=[hist.first.value],
            )
        for name in w.services:
            ivs = w.deps[name]
            if len(ivs) < 2:
                continue
            active = w.deps_at(name, NOW)
            gone = [d for d in ivs if d.end is not None]
            if len(active) == 1:
                self.add(
                    cat,
                    "dependency_now",
                    f"Which service does {w.svc(name).display} depend on now?",
                    active[0].dep,
                    self.svc_aliases(active[0].dep),
                    self.nid(active[0].add_event),
                    distractors=[d.dep for d in gone],
                )
            else:
                names = [d.dep for d in active]
                self.add(
                    cat,
                    "dependency_now",
                    f"Which services does {w.svc(name).display} depend on now?",
                    " and ".join(names),
                    [],
                    self.nid(*(d.add_event for d in active)),
                    rubric=RUBRICS[cat] + " Every listed service must be named.",
                )

    def _date_inside(
        self, start: _dt.date, end: _dt.date | None, name: str | None = None
    ) -> _dt.date | None:
        hi = (end or NOW) - _dt.timedelta(days=7)
        lo = start + _dt.timedelta(days=7)
        if hi < lo:
            return None
        for _ in range(50):
            d = lo + _dt.timedelta(days=self.rng.randint(0, (hi - lo).days))
            if name is None or not self.w.ambiguous(name, d):
                return d
        return None

    def temporal(self) -> None:
        w = self.w
        cat = "temporal"
        for name in w.services:
            hist = w.owner[name]
            ivs = hist.intervals()
            for start, end, team, eid in ivs[:-1]:
                d = self._date_inside(start, end, name)
                if d is None:
                    continue
                self.add(
                    cat,
                    "owner_at",
                    self.rng.choice(
                        ["Which team owned {s} on {d}?", "Who was responsible for {s} on {d}?"]
                    ).format(s=w.svc(name).display, d=d.isoformat()),
                    team,
                    self.team_aliases(team),
                    self.nid(eid),
                    as_of=d.isoformat(),
                    distractors=[t for _s, _e, t, _i in ivs if t != team],
                )
            takers = Counter(c.value for c in hist.changes[1:])
            for i, ch in enumerate(hist.changes[1:], start=1):
                if takers[ch.value] != 1:
                    continue
                prev = hist.changes[i - 1].value
                support = [ch.event_id]
                for wr in w.wrong_records:
                    if wr["wrong_event"] == ch.event_id:
                        support.append(wr["fix_event"])
                self.add(
                    cat,
                    "owner_before",
                    f"Which team owned {w.svc(name).display} before {ch.value} took it over?",
                    prev,
                    self.team_aliases(prev),
                    self.nid(*support),
                    distractors=[ch.value],
                )
                self.add(
                    cat,
                    "handover_date",
                    f"When did {ch.value} take over {w.svc(name).display}?",
                    ch.date.isoformat(),
                    _date_forms(ch.date),
                    self.nid(*support),
                )
        for team in w.teams:
            ivs = w.oncall[team].intervals()
            if len(ivs) < 2:
                continue
            chosen = self.rng.sample(ivs[:-1], k=min(3, len(ivs) - 1))
            for start, end, person, eid in chosen:
                d = self._date_inside(start, end)
                if d is None:
                    continue
                self.add(
                    cat,
                    "oncall_at",
                    f"Who was on call for {team} on {d.isoformat()}?",
                    person,
                    self.person_aliases(person),
                    self.nid(eid),
                    as_of=d.isoformat(),
                    distractors=[p for _s, _e, p, _i in ivs if p != person],
                )
        for m in w.moves:
            self.add(
                cat,
                "team_before",
                f"Which team was {m['person']} on before moving to {m['new']}?",
                m["old"],
                self.team_aliases(m["old"]),
                self.nid(m["event"], w.member[m["person"]].first.event_id),
                distractors=[m["new"]],
            )
        for name in WINDOW_CHANGES:
            hist = w.window[name]
            first, cur = hist.first, hist.current
            ev = w.by_id[cur.event_id]
            self.add(
                cat,
                "window_before",
                f"What was the deploy window for {w.svc(name).display} before it changed?",
                first.value,
                [ev.payload["old_short"]],
                self.nid(cur.event_id, first.event_id),
                distractors=[cur.value],
            )
        for name in w.services:
            for iv in w.deps[name]:
                if iv.end is not None:
                    new = [d for d in w.deps[name] if d.start == iv.end]
                    if new:
                        self.add(
                            cat,
                            "dependency_before",
                            f"What did {w.svc(name).display} depend on before it switched to {w.svc(new[0].dep).display}?",
                            iv.dep,
                            self.svc_aliases(iv.dep),
                            self.nid(iv.remove_event or iv.add_event, iv.add_event),
                            distractors=[new[0].dep],
                        )

    def _oncall_support(self, team: str) -> int:
        return self.w.oncall[team].current.event_id

    def _owner_support(self, name: str) -> list[int]:
        cur = self.w.owner[name].current
        out = [cur.event_id]
        for wr in self.w.wrong_records:
            if wr["wrong_event"] == cur.event_id:
                out.append(wr["fix_event"])
        return out

    def multi_hop(self) -> None:
        w = self.w
        cat = "multi_hop"
        for name in w.services:
            team = w.owner_at(name, NOW)
            person = w.oncall_at(team, NOW)
            assert team and person
            self.add(
                cat,
                "oncall_of_owner",
                f"Who is on call for the team that owns {w.svc(name).display}?",
                person,
                self.person_aliases(person),
                self.nid(*self._owner_support(name), self._oncall_support(team)),
                hops=2,
                chain=[f"{name} is owned by {team}", f"{person} is on call for {team}"],
            )
            active = w.deps_at(name, NOW)
            if len(active) == 1:
                dep = active[0].dep
                dep_team = w.owner_at(dep, NOW)
                dep_person = w.oncall_at(dep_team, NOW)
                assert dep_team and dep_person
                self.add(
                    cat,
                    "owner_of_dependency",
                    f"Which team owns the service that {w.svc(name).display} depends on?",
                    dep_team,
                    self.team_aliases(dep_team),
                    self.nid(active[0].add_event, *self._owner_support(dep)),
                    hops=2,
                    chain=[f"{name} depends on {dep}", f"{dep} is owned by {dep_team}"],
                )
                self.add(
                    cat,
                    "oncall_of_dependency_owner",
                    f"Who is on call for the team that owns the service {w.svc(name).display} depends on?",
                    dep_person,
                    self.person_aliases(dep_person),
                    self.nid(
                        active[0].add_event,
                        *self._owner_support(dep),
                        self._oncall_support(dep_team),
                    ),
                    hops=3,
                    chain=[
                        f"{name} depends on {dep}",
                        f"{dep} is owned by {dep_team}",
                        f"{dep_person} is on call for {dep_team}",
                    ],
                )
                window = w.window_at(dep, NOW)
                if window:
                    wev = w.by_id[w.window[dep].current.event_id]
                    self.add(
                        cat,
                        "window_of_dependency",
                        f"What is the deploy window of the service that {w.svc(name).display} depends on?",
                        window,
                        [wev.payload["short"]],
                        self.nid(active[0].add_event, w.window[dep].current.event_id),
                        hops=2,
                        chain=[f"{name} depends on {dep}", f"{dep} deploys only {window}"],
                    )
        for p in w.people:
            team = w.team_at(p, NOW)
            person = w.oncall_at(team, NOW)
            assert team and person
            if person == p:
                continue
            self.add(
                cat,
                "oncall_of_persons_team",
                f"Who is on call for {p}'s team right now?",
                person,
                self.person_aliases(person),
                self.nid(w.member[p].current.event_id, self._oncall_support(team)),
                hops=2,
                chain=[f"{p} is a member of {team}", f"{person} is on call for {team}"],
            )
        for team in w.teams:
            owned = w.services_of(team, NOW)
            if not owned:
                continue
            gold = ", ".join(owned[:-1]) + (" and " if len(owned) > 1 else "") + owned[-1]
            self.add(
                cat,
                "services_of_team",
                f"Which services does {team} own now?",
                gold,
                [],
                self.nid(*(e for s in owned for e in self._owner_support(s))),
                hops=len(owned),
                chain=[f"{team} owns {s}" for s in owned],
                rubric=RUBRICS[cat] + f" The gold has {len(owned)} services.",
            )

    def provenance(self) -> None:
        w = self.w
        cat = "provenance"
        for name in w.services:
            hist = w.owner[name]
            takers = Counter(c.value for c in hist.changes[1:])
            for ch in hist.changes[1:]:
                ev = w.by_id[ch.event_id]
                if ev.kind != "handover" or takers[ch.value] != 1:
                    continue
                note = self.note_of[ch.event_id]
                gold, aliases = self.label_gold(note)
                self.add(
                    cat,
                    "handover_note",
                    f"Which note recorded the handover of {w.svc(name).display} to {ch.value}, and on what date?",
                    gold,
                    aliases,
                    [note.note_id],
                )
        for wr in w.wrong_records:
            note = self.note_of[wr["fix_event"]]
            gold, aliases = self.label_gold(note)
            self.add(
                cat,
                "correction_note",
                f"Which note corrected the recorded owner of {w.svc(wr['service']).display}, and when?",
                gold,
                aliases,
                [note.note_id],
            )
        for name in WINDOWED:
            note = self.note_of[w.window[name].first.event_id]
            gold, aliases = self.label_gold(note)
            self.add(
                cat,
                "window_first_note",
                f"Which note first recorded the deploy window for {w.svc(name).display}, and when?",
                gold,
                aliases,
                [note.note_id],
            )
        for dec in w.decisions:
            note = self.note_of[dec["event"]]
            gold, aliases = self.label_gold(note)
            self.add(
                cat,
                "decision_note",
                f"Which note records the decision that {dec['what']}, and when was it written?",
                gold,
                aliases,
                [note.note_id],
            )
        for inc in w.incidents:
            note = self.note_of[inc["event"]]
            gold, aliases = self.label_gold(note)
            self.add(
                cat,
                "decision_note",
                f"Which note records the decision that {inc['what']}, and when was it written?",
                gold,
                aliases,
                [note.note_id],
            )
            self.add(
                cat,
                "review_date",
                f"On what date was the review of {inc['inc']} written?",
                note.date.isoformat(),
                _date_forms(note.date),
                [note.note_id],
            )
        for name, dep, _purpose in DEPENDENCIES:
            iv = next(d for d in w.deps[name] if d.dep == dep)
            note = self.note_of[iv.add_event]
            gold, aliases = self.label_gold(note)
            self.add(
                cat,
                "dependency_first_note",
                f"Which note first recorded that {w.svc(name).display} depends on {w.svc(dep).display}, and when?",
                gold,
                aliases,
                [note.note_id],
            )
        for m in w.moves:
            note = self.note_of[m["event"]]
            gold, aliases = self.label_gold(note)
            self.add(
                cat,
                "move_note",
                f"Which note recorded {m['person']}'s move to {m['new']}, and when?",
                gold,
                aliases,
                [note.note_id],
            )

    def abstention(self) -> None:
        w = self.w
        cat = "abstention"

        def idk(subtype: str, question: str, **extra) -> None:
            self.add(cat, subtype, question, IDK, [], [], **extra)

        for name in w.services:
            if name not in WINDOWED:
                idk(
                    "no_window",
                    self.rng.choice(
                        ["What is the deploy window for {s}?", "When can {s} be deployed?"]
                    ).format(s=w.svc(name).display),
                )
            if not w.deps[name]:
                idk("no_dependency", f"Which service does {w.svc(name).display} depend on?")
            if not w.dependants_at(name, NOW) and not any(
                d.dep == name for s in w.services for d in w.deps[s]
            ):
                idk("no_dependants", f"Which service depends on {w.svc(name).display}?")
        for name in list(w.services)[::2]:
            first = w.owner[name].first.value
            idk(
                "owner_before_first",
                f"Which team owned {w.svc(name).display} before {first}?",
                distractors=[c.value for c in w.owner[name].changes[1:]],
            )
        handovers = [e for e in w.events if e.kind == "handover"]
        for e in self.rng.sample(handovers, k=min(6, len(handovers))):
            pl = e.payload
            idk(
                "no_reason_recorded",
                f"Why was {w.svc(pl['service']).display} moved from {pl['old']} to {pl['new']}?",
            )
        movers = {m["person"] for m in w.moves}
        for p in w.people:
            if p not in movers:
                idk(
                    "no_previous_team",
                    f"Which team was {p} on before joining {w.member[p].first.value}?",
                )
        for name in list(w.services)[1::4]:
            idk("no_slo_recorded", f"What is the availability target for {w.svc(name).display}?")
        for fake in ("INC-2025-008", "INC-2025-031", "INC-2026-014"):
            idk("unknown_incident", f"What was the root cause of {fake}?")
        for team in list(w.teams)[::2]:
            idk("no_manager_recorded", f"Who is the manager of {team}?")
        for team in list(w.teams)[1::2]:
            idk("no_secondary_recorded", f"Who is the secondary on-call for {team}?")

    # ---------------------------------------------------------------- selection

    def select(self, per_category: int) -> list[dict]:
        out: list[dict] = []
        for cat in CATEGORIES:
            by_sub: dict[str, list[dict]] = defaultdict(list)
            for q in self.candidates[cat]:
                by_sub[q["subtype"]].append(q)
            for items in by_sub.values():
                self.rng.shuffle(items)
            order = sorted(by_sub)
            self.rng.shuffle(order)
            chosen: list[dict] = []
            seen_questions: set[str] = set()
            while len(chosen) < per_category and any(by_sub[s] for s in order):
                for sub in order:
                    if len(chosen) >= per_category:
                        break
                    while by_sub[sub]:
                        q = by_sub[sub].pop()
                        if q["question"] not in seen_questions:
                            seen_questions.add(q["question"])
                            chosen.append(q)
                            break
            chosen.sort(key=lambda q: (q["subtype"], q["question"]))
            out.extend(chosen)
        for i, q in enumerate(out, start=1):
            q["qid"] = f"q{i:03d}"
        return [{"qid": q["qid"], **{k: v for k, v in q.items() if k != "qid"}} for q in out]


# --------------------------------------------------------------------------- verification


def verify(
    world: World, notes: list[Note], patches: list[dict], questions: list[dict]
) -> list[str]:
    """Ingest the gold through anatid and compare the graph with the world.  Returns problems."""
    from anatid import Anatid
    from anatid.ingest import MemoryPatch, ScriptedExtractor, current_relation_ids, ingest

    problems: list[str] = []
    extractor = ScriptedExtractor([MemoryPatch.from_dict(p["patch"]) for p in patches])
    by_note = {n.note_id: n for n in notes}
    with Anatid.open(":memory:", tenant=1, embedding_dim=8) as db:
        for note in notes:
            when = _dt.datetime.combine(
                note.date, _dt.time(9, 0, tzinfo=_dt.timezone.utc)
            ) + _dt.timedelta(minutes=note.seq)
            receipt = ingest(
                db,
                note.rendered,
                extractor=extractor,
                writer=note.source,
                source=note.source,
                now=when,
            )
            assert receipt is not None
            # A reminder restates a current fact, so the pipeline must drop exactly its
            # content and its relation as duplicates; every other pipeline note is a gold bug.
            restated: set[str] = set()
            for e in note.events:
                if e.kind == "reminder":
                    restated.update(repr(f["content"]) for f in e.facts)
                    restated.update(f"{src} -{rel}-> {dst}" for src, dst, rel in e.add_relations)
            for line in receipt.patch.notes:
                if line.startswith("dedupe:") and any(item in line for item in restated):
                    continue
                problems.append(f"{note.note_id} ({note.source}): unexpected pipeline note: {line}")
            for item in restated:
                if not any(item in line for line in receipt.patch.notes):
                    problems.append(f"{note.note_id}: reminder of {item} was not deduplicated")

        def current_about(name: str, as_of: _dt.datetime | None = None) -> list[str]:
            view = db if as_of is None else db.as_of(as_of)
            return [m.content for m in view.context(name, limit=200)]

        def has_edge(src: str, dst: str, rel: str) -> bool:
            s, d = db.get_entity(src), db.get_entity(dst)
            if s is None or d is None:
                return False
            return bool(current_relation_ids(db, s.entity_id, d.entity_id, rel_kind=rel))

        for name, svc in world.services.items():
            team = world.owner_at(name, NOW)
            owns = [c for c in current_about(name) if " owns " in c]
            if owns != [content_owner(team, svc)]:
                problems.append(f"{name}: current owner facts {owns!r}, world says {team}")
            for t in world.teams:
                if has_edge(t, name, "owns") != (t == team):
                    problems.append(
                        f"{name}: owns edge from {t} is {'present' if t != team else 'missing'}"
                    )
            deps = sorted(d.dep for d in world.deps_at(name, NOW))
            got = sorted(c for c in current_about(name) if c.startswith(f"{svc.cap} depends on "))
            want = sorted(content_dep(svc, world.svc(d)) for d in deps)
            if got != want:
                problems.append(f"{name}: dependency facts {got!r}, world says {want!r}")
            window = world.window_at(name, NOW)
            got_w = [c for c in current_about(name) if " deploys only " in c]
            want_w = [content_window(svc, window)] if window else []
            if got_w != want_w:
                problems.append(f"{name}: window facts {got_w!r}, world says {want_w!r}")
        for team in world.teams:
            person = world.oncall_at(team, NOW)
            got = [c for c in current_about(team) if " is on call for " in c]
            if got != [content_oncall(person, team)]:
                problems.append(f"{team}: on-call facts {got!r}, world says {person}")
            members = sorted(world.members_at(team, NOW))
            got_m = sorted(
                c.split(" is a member of ")[0]
                for c in current_about(team)
                if " is a member of " in c
            )
            if got_m != members:
                problems.append(f"{team}: members {got_m!r}, world says {members!r}")

        for q in questions:
            if q["category"] != "temporal" or not q.get("as_of"):
                continue
            d = _dt.date.fromisoformat(q["as_of"])
            at = _dt.datetime.combine(d, _dt.time(23, 59, tzinfo=_dt.timezone.utc))
            if q["subtype"] == "owner_at":
                name = next(s for s in world.services if world.svc(s).display in q["question"])
                facts = [c for c in current_about(name, at) if " owns " in c]
                want = content_owner(q["gold"], world.svc(name))
                if facts != [want]:
                    problems.append(f"{q['qid']}: as_of {d} owner facts {facts!r}, gold {want!r}")
            elif q["subtype"] == "oncall_at":
                team = next(t for t in world.teams if f"for {t} on" in q["question"])
                facts = [c for c in current_about(team, at) if " is on call for " in c]
                want = content_oncall(q["gold"], team)
                if facts != [want]:
                    problems.append(f"{q['qid']}: as_of {d} on-call facts {facts!r}, gold {want!r}")

        # every supporting note exists and every non-abstention question has one
        for q in questions:
            for nid in q["support"]:
                if nid not in by_note:
                    problems.append(f"{q['qid']}: unknown support note {nid}")
            if q["category"] != "abstention" and not q["support"]:
                problems.append(f"{q['qid']}: no supporting note")
            if q["category"] == "abstention" and q["support"]:
                problems.append(f"{q['qid']}: abstention question with support")
    return problems


# --------------------------------------------------------------------------- main


def generate(
    seed: int = DEFAULT_SEED,
) -> tuple[World, list[Note], list[dict], list[dict], QuestionBank]:
    world = World(seed)
    rng = random.Random(seed ^ 0x5EED)
    notes = build_notes(world, rng)
    patches = [
        {
            "note_id": n.note_id,
            "source": n.source,
            "date": n.date.isoformat(),
            "patch": gold_patch(n),
        }
        for n in notes
    ]
    bank = QuestionBank(world, notes, random.Random(seed ^ 0xC0FFEE))
    bank.build()
    questions = bank.select(QUESTIONS_PER_CATEGORY)
    return world, notes, patches, questions, bank


def say(text: str) -> None:
    """The command's report goes to stdout; this is a command-line tool and that is its output."""
    sys.stdout.write(text + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--verify", action="store_true", help="ingest the gold through anatid and check it"
    )
    args = parser.parse_args(argv)

    world, notes, patches, questions, bank = generate(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "notes.jsonl", [n.to_dict() for n in notes])
    write_jsonl(args.out / "gold_patches.jsonl", patches)
    write_jsonl(args.out / "questions.jsonl", questions)

    per_category = Counter(q["category"] for q in questions)
    per_subtype = Counter(f"{q['category']}/{q['subtype']}" for q in questions)
    words = [len(n.text.split()) for n in notes]
    counts = {
        "seed": args.seed,
        "events": len(world.events),
        "notes": len(notes),
        "notes_by_source_kind": dict(Counter(n.source_kind for n in notes)),
        "events_per_note": dict(Counter(len(n.events) for n in notes)),
        "note_words_mean": round(sum(words) / len(words), 1),
        "note_words_max": max(words),
        "gold_facts": sum(len(p["patch"]["add_facts"]) for p in patches),
        "gold_corrections": sum(len(p["patch"]["corrections"]) for p in patches),
        "gold_relations_added": sum(len(p["patch"]["add_relations"]) for p in patches),
        "gold_relations_removed": sum(len(p["patch"]["remove_relations"]) for p in patches),
        "questions": len(questions),
        "questions_by_category": dict(per_category),
        "questions_by_subtype": dict(sorted(per_subtype.items())),
        "candidates_by_category": {c: len(bank.candidates[c]) for c in CATEGORIES},
    }
    (args.out / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    (args.out / "world.json").write_text(json.dumps(world.summary(), indent=2, default=str) + "\n")

    say(f"events {counts['events']}, notes {counts['notes']}, questions {counts['questions']}")
    for cat in CATEGORIES:
        say(f"  {cat:17s} {per_category.get(cat, 0)}")
    if args.verify:
        problems = verify(world, notes, patches, questions)
        if problems:
            say(f"VERIFY: {len(problems)} problem(s)")
            for p in problems[:50]:
                say("  " + p)
            return 1
        say("VERIFY: gold patches reproduce the world; temporal questions agree with as_of reads")
    return 0


if __name__ == "__main__":
    sys.exit(main())
