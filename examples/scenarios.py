"""The demo scenarios, held as data so a script and the studio can share one definition.

A scenario is everything needed to stage the same demonstration twice: the entities and
the edges between them, the facts a person would have told an assistant over months, the
question whose answer no single stored sentence contains, the belief that later changes,
and the sentences that explain what is happening.

Nothing here imports anything but the standard library and anatid.

    from anatid import Anatid
    import scenarios

    db = Anatid.open(":memory:", tenant=1, embedding_dim=64)
    scenario = scenarios.SCENARIOS[scenarios.DEFAULT]
    scenarios.build(db, scenario)
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "Fact",
    "Supersede",
    "Correction",
    "Scenario",
    "SCENARIOS",
    "DEFAULT",
    "build",
    "find_fact",
    "apply_supersede",
    "parse_time",
    "format_day",
]


@dataclass(frozen=True)
class Fact:
    """One durable sentence, and the raw episode it came from.

    ``when`` is the moment the database was told this, as an ISO 8601 string. It is
    written into the row, so a later as_of read replays the world as it stood then.
    """

    content: str
    entities: tuple[str, ...]
    writer: str
    episode: str
    when: str | None = None


@dataclass(frozen=True)
class Supersede:
    """A belief that changes later. The old row is closed, never deleted.

    A correction is rarely only a sentence. Priya's allergy moves from pine nuts to prawns
    and the graph has to move with it, because the graph is what recall walks: the old edge
    stops being true at the same instant the new one starts. ``removed_relations`` holds
    the edges the change closes and ``extra_relations`` the edges it opens, both as
    (source, target, rel_kind). :func:`apply_supersede` writes the memory and both sets of
    edges in one transaction.
    """

    match_text: str
    new_content: str
    entities: tuple[str, ...]
    writer: str
    episode: str
    when: str | None = None
    extra_relations: tuple[tuple[str, str, str], ...] = ()
    removed_relations: tuple[tuple[str, str, str], ...] = ()
    extra_entities: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Correction:
    """What applying a scenario's supersede did, as the database now has it.

    ``old`` is the memory that was closed and ``new`` the one written in its place.
    ``closed_relations`` are the edges the same transaction stopped believing, and
    ``opened_relations`` the ones it added.
    """

    old: Any
    new: Any
    closed_relations: tuple[tuple[str, str, str], ...] = ()
    opened_relations: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    one_liner: str
    entities: tuple[tuple[str, str], ...]
    relations: tuple[tuple[str, str, str], ...]
    facts: tuple[Fact, ...]
    question: str
    followup_question: str
    supersede: Supersede
    write_request: str
    seed_entity: str
    system_prompt: str
    base_time: str
    asof_time: str
    asof_match: str
    asof_label_past: str
    asof_label_now: str
    explain: Mapping[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# The dinner party. A question that names no guest and no ingredient.
# --------------------------------------------------------------------------------------

DINNER = Scenario(
    key="dinner",
    title="A dinner on Friday, and an allergy the question never mentions",
    one_liner=(
        "Ordinary household memory: who is coming, what is in each dish, who reacts to "
        "what. The cook asks about the menu and never mentions a guest."
    ),
    entities=(
        ("Priya", "person"),
        ("Tom", "person"),
        ("Maya", "person"),
        ("Sam", "person"),
        ("Nora", "person"),
        ("Friday dinner", "event"),
        ("book club", "event"),
        ("pesto pasta", "dish"),
        ("pavlova", "dish"),
        ("pine nuts", "ingredient"),
        ("parmesan", "ingredient"),
        ("eggs", "ingredient"),
        ("cream", "ingredient"),
        ("prawns", "ingredient"),
        ("meat", "ingredient"),
        ("basil", "ingredient"),
        ("olive oil", "ingredient"),
        ("berries", "ingredient"),
        ("bread", "ingredient"),
        ("dairy", "category"),
        ("the oven", "thing"),
        ("the big pot", "thing"),
        ("the boiler", "thing"),
        ("the car", "thing"),
        ("the street", "place"),
        ("the market", "place"),
        ("the garden", "place"),
        ("the flat", "place"),
    ),
    # Edges are the standing shape of the household. They are written once, at the
    # scenario's base time, so the shape is in place for every reading. The dates that
    # carry the story sit on the facts. Plenty of the entities above carry no edge at
    # all, which is what most household memory looks like.
    relations=(
        ("Friday dinner", "Priya", "invites"),
        ("Friday dinner", "Tom", "invites"),
        ("Friday dinner", "Maya", "invites"),
        ("Friday dinner", "Sam", "invites"),
        ("pesto pasta", "pine nuts", "contains"),
        ("pesto pasta", "parmesan", "contains"),
        ("pesto pasta", "basil", "contains"),
        ("pesto pasta", "olive oil", "contains"),
        ("pavlova", "eggs", "contains"),
        ("pavlova", "cream", "contains"),
        ("pavlova", "berries", "contains"),
        ("parmesan", "dairy", "is_a"),
        ("cream", "dairy", "is_a"),
        ("Priya", "pine nuts", "reacts_to"),
        ("Maya", "prawns", "reacts_to"),
        ("Tom", "meat", "avoids"),
    ),
    # Twenty-six facts, in the order they were told. Six of them carry the question:
    # the guest list, the two recipe cards, the two allergies and Tom going vegetarian.
    # One more, the March risotto note, is what the time travel step reads back. The
    # rest is household noise the answer has to be found inside: the oven, the car, the
    # boiler, who walks home. Nothing here was reworded to make the demonstration
    # easier, and the fact that carries the answer is the oldest one in the file.
    facts=(
        Fact(
            content="Sam always brings wine and forgets the corkscrew.",
            entities=("Sam",),
            writer="you",
            episode="Note, 12 September. Sam arrived with wine and no corkscrew again.",
            when="2025-09-12T20:00:00",
        ),
        Fact(
            content="Priya prefers still water to sparkling.",
            entities=("Priya",),
            writer="you",
            episode="Note, 4 October. Priya asked for still water again.",
            when="2025-10-04T19:30:00",
        ),
        Fact(
            content="Tom's birthday is in November.",
            entities=("Tom",),
            writer="you",
            episode="Note, 3 November. Tom's birthday. Book a table next year.",
            when="2025-11-03T09:00:00",
        ),
        Fact(
            content="Maya lives two streets away and walks home.",
            entities=("Maya",),
            writer="you",
            episode="Note, 20 November. Maya walked home. No taxi needed.",
            when="2025-11-20T23:00:00",
        ),
        # The fact the whole demonstration turns on. It is the oldest fact in the
        # database and it shares no word with the question that needs it.
        Fact(
            content="Priya reacts to pine nuts. She carries an adrenaline pen.",
            entities=("Priya", "pine nuts"),
            writer="Sam",
            episode=(
                "Message from Sam, 26 December. Priya had to leave the Christmas dinner "
                "early, her lips swelled up. We think it was the pesto."
            ),
            when="2025-12-26T21:40:00",
        ),
        Fact(
            content="The oven runs about ten degrees hot, so set it lower than the card says.",
            entities=("the oven",),
            writer="kitchen notes",
            episode="Kitchen notes, 3 January. Checked the oven with a thermometer.",
            when="2026-01-03T10:00:00",
        ),
        Fact(
            content="The pesto is basil, pine nuts, parmesan and olive oil.",
            entities=("pesto pasta", "pine nuts", "parmesan"),
            writer="kitchen notes",
            episode="Recipe card copied into the kitchen notes, 11 January.",
            when="2026-01-11T11:00:00",
        ),
        Fact(
            content="The pesto tastes better if the basil goes in last.",
            entities=("pesto pasta", "basil"),
            writer="kitchen notes",
            episode="Kitchen notes, 11 January. Basil in last, it keeps the colour.",
            when="2026-01-11T11:02:00",
        ),
        Fact(
            content="The pavlova is eggs, cream and berries.",
            entities=("pavlova", "eggs", "cream"),
            writer="kitchen notes",
            episode="Recipe card copied into the kitchen notes, 11 January.",
            when="2026-01-11T11:05:00",
        ),
        Fact(
            content="The meringue base needs a low oven for two hours and a slow cool after.",
            entities=("pavlova", "the oven"),
            writer="kitchen notes",
            episode="Kitchen notes, 11 January. Low oven, two hours, leave it in to cool.",
            when="2026-01-11T11:06:00",
        ),
        Fact(
            content="The berries come from the market on Thursday.",
            entities=("berries", "the market"),
            writer="you",
            episode="Note, 15 January. Market on Thursday for the berries.",
            when="2026-01-15T08:00:00",
        ),
        Fact(
            content="Parmesan and cream are both dairy.",
            entities=("parmesan", "cream", "dairy"),
            writer="kitchen notes",
            episode="Kitchen notes, 20 January. Wrote down which things count as dairy.",
            when="2026-01-20T09:00:00",
        ),
        Fact(
            content="The good olive oil is the one in the green tin.",
            entities=("olive oil",),
            writer="kitchen notes",
            episode="Kitchen notes, 2 February. Green tin is the good oil.",
            when="2026-02-02T10:00:00",
        ),
        Fact(
            content="Maya cannot have shellfish.",
            entities=("Maya", "prawns"),
            writer="Maya",
            episode=(
                "Maya, 2 February. No shellfish for me, it is the one that puts me in hospital."
            ),
            when="2026-02-02T18:30:00",
        ),
        Fact(
            content="The big pot lives in the cupboard under the window.",
            entities=("the big pot",),
            writer="you",
            episode="Note, 14 February. Moved the big pot under the window.",
            when="2026-02-14T12:00:00",
        ),
        Fact(
            content="Parking on our street is hard after six.",
            entities=("the street",),
            writer="you",
            episode="Note, 6 March. Tell people to come early, parking is hopeless.",
            when="2026-03-06T18:00:00",
        ),
        Fact(
            content="Book club is the second Tuesday of the month.",
            entities=("book club",),
            writer="you",
            episode="Note, 10 March. Book club moved to the second Tuesday.",
            when="2026-03-10T21:00:00",
        ),
        # The artefact the as_of step reads back. It was written under the December
        # belief, and it is still on record after that belief closes.
        Fact(
            content="The Sunday lunch in March is risotto with no pine nuts, for Priya.",
            entities=("Priya", "pine nuts"),
            writer="you",
            episode=(
                "Note to the assistant, 12 March. Risotto on Sunday. Leaving the pine "
                "nuts out because Priya is coming."
            ),
            when="2026-03-12T09:00:00",
        ),
        Fact(
            content="The boiler is serviced every October.",
            entities=("the boiler",),
            writer="you",
            episode="Note, 1 April. Boiler service booked for October.",
            when="2026-04-01T09:00:00",
        ),
        Fact(
            content="Tom stopped eating meat in the spring.",
            entities=("Tom", "meat"),
            writer="Tom",
            episode="Tom, 4 April. I have gone vegetarian. Fish is still fine.",
            when="2026-04-04T13:15:00",
        ),
        Fact(
            content="The car goes in for its MOT in May.",
            entities=("the car",),
            writer="you",
            episode="Note, 12 April. MOT booked for May.",
            when="2026-04-12T09:00:00",
        ),
        Fact(
            content="Nora next door waters the garden when we are away.",
            entities=("Nora", "the garden"),
            writer="you",
            episode="Note, 2 May. Nora has the spare key for the watering.",
            when="2026-05-02T09:00:00",
        ),
        Fact(
            content="The bread from the corner shop is better than the supermarket one.",
            entities=("bread",),
            writer="you",
            episode="Note, 9 May. Corner shop bread, get it in the morning.",
            when="2026-05-09T09:00:00",
        ),
        Fact(
            content="The flat gets cold in the evening, so put the heating on an hour early.",
            entities=("the flat",),
            writer="you",
            episode="Note, 20 May. Heating on early, the flat takes an hour to warm.",
            when="2026-05-20T18:00:00",
        ),
        Fact(
            content="The garden table seats six if the bench comes out.",
            entities=("the garden",),
            writer="you",
            episode="Note, 1 June. Bench out, the table seats six.",
            when="2026-06-01T11:00:00",
        ),
        Fact(
            content="Friday dinner is Priya, Tom, Maya and Sam, at seven.",
            entities=("Friday dinner", "Priya", "Tom", "Maya", "Sam"),
            writer="you",
            episode=(
                "Note to the assistant, 28 August. Dinner on Friday, the four of them, "
                "seven o'clock."
            ),
            when="2026-08-28T20:10:00",
        ),
    ),
    question="I'm making pesto pasta and pavlova for Friday. Any problems?",
    followup_question=(
        "I have added a prawn cocktail to start, so the menu is prawn cocktail, then "
        "pesto pasta, then pavlova. Any problems now?"
    ),
    supersede=Supersede(
        match_text="reacts to pine nuts",
        new_content=(
            "Priya was cleared for pine nuts by her allergist in August. She reacted to "
            "prawns at a work lunch in July."
        ),
        entities=("Priya", "pine nuts", "prawns"),
        writer="Priya",
        episode=(
            "Priya, 20 August. The allergist retested me and pine nuts are fine now. The "
            "thing that got me at the work lunch in July was the prawns."
        ),
        when="2026-08-20T17:00:00",
        extra_relations=(
            ("Priya", "prawns", "reacts_to"),
            ("prawn cocktail", "prawns", "contains"),
        ),
        removed_relations=(("Priya", "pine nuts", "reacts_to"),),
        extra_entities=(("prawn cocktail", "dish"),),
    ),
    write_request=(
        "Sam just told me he is bringing his partner on Friday and she is dairy-free. "
        "Remember that."
    ),
    seed_entity="Friday dinner",
    system_prompt=(
        "You are a household assistant with a graph memory of one person's life: who is "
        "coming to what, what goes into each dish, and who reacts to what. "
        "Always call recall before you answer. Pass seed_entity naming the event, person "
        "or dish the question is about, so the search can walk the graph outward from it "
        "and reach facts that do not contain any of the question's words. "
        "Answer only from what recall returns. If part of your answer rests on something "
        "you know rather than on something recall returned, say so in a short clause. "
        "Do not write sentences of the form 'not X, it is Y'. "
        "Before anything else, check every guest at the event against every ingredient in "
        "every dish being served, following the recalled facts from guest to ingredient to "
        "dish. If a guest reacts to something that is in something on the menu, say that "
        "first, and name the guest, the ingredient and the dish. Then mention anything "
        "smaller. Be brief and plain. "
        "Write plain prose in short sentences. Do not use markdown, asterisks, bold, "
        "bullet characters, headings or dashes for emphasis."
    ),
    base_time="2025-09-01T09:00:00",
    asof_time="2026-03-15T12:00:00",
    asof_match="pine nuts",
    asof_label_past="what the database said about pine nuts on 15 March 2026",
    asof_label_now="the same read today",
    explain=MappingProxyType(
        {
            "setup": (
                "Nothing below was written for this demo. It is the sort of thing a person "
                "tells an assistant over eleven months and never thinks about again. Most "
                "of it has nothing to do with Friday."
            ),
            "two_hop": (
                "The question names two dishes and a day. It names no guest and no "
                "ingredient. No stored sentence contains both 'pesto' and 'Priya', so word "
                "search returns the recipe cards and the guest list, and none of them joins "
                "a guest to an ingredient. Any index that scores sentences by their words "
                "has the same problem. The joins live in the edges. A search agent could "
                "still get there in two passes, by reading the recipe card and then "
                "searching each of its four ingredients in turn, and that only works if you "
                "already know that ingredients are the thing to chase. The graph does it in "
                "one call, because the edges are already there."
            ),
            "stakes": (
                "Priya carries an adrenaline pen. An assistant that answers 'looks fine' "
                "here sends someone to hospital."
            ),
            "supersede": (
                "Allergies change. The old belief is closed rather than deleted, and a new "
                "one is written in its place, so the same question gets a different answer "
                "without losing the reason the first answer was given."
            ),
            "approval": (
                "The assistant wants to add something to its own memory. A person decides "
                "whether it lands. Every write goes through the gate. Reads run straight "
                "through. The consequence the model draws here, that parmesan and cream are "
                "dairy, is on record in the database as its own line, so it can be checked "
                "rather than taken on trust. The connection it could never have guessed is "
                "that Priya was coming."
            ),
            "as_of": (
                "On 12 March you planned a Sunday risotto and left the pine nuts out, "
                "because of what the database said about Priya then. That note is still on "
                "record, and so is the belief it was written under. A database that quietly "
                "rewrote history would leave the note looking like a mistake."
            ),
            "provenance": (
                "Why did we ever think pine nuts? The belief traces back to the message Sam "
                "sent after the Christmas dinner, with his name and the date on it. A memory "
                "you cannot audit is a rumour."
            ),
        }
    ),
)


# --------------------------------------------------------------------------------------
# The on-call rotation. The original engineering story, kept intact.
# --------------------------------------------------------------------------------------

ONCALL = Scenario(
    key="oncall",
    title="An on-call page, and the maintainer the question never mentions",
    one_liner=(
        "The team told the assistant who leads what and who owns what. The page names a "
        "person, and the answer is a different person two hops away."
    ),
    entities=(
        ("Ada", "person"),
        ("Bo", "person"),
        ("Project Kestrel", "project"),
        ("ingest-service", "service"),
        ("postgres-primary", "service"),
    ),
    relations=(
        ("Ada", "Project Kestrel", "leads"),
        ("Project Kestrel", "ingest-service", "owns"),
        ("Bo", "ingest-service", "maintains"),
        ("Project Kestrel", "postgres-primary", "depends_on"),
    ),
    facts=(
        Fact(
            content="Ada leads Project Kestrel.",
            entities=("Ada", "Project Kestrel"),
            writer="onboarding",
            episode="Team onboarding doc, 2026-03-01.",
        ),
        Fact(
            content="Bo maintains the ingest-service and is the person to page for it.",
            entities=("Bo", "ingest-service"),
            writer="onboarding",
            episode="Team onboarding doc, 2026-03-01.",
        ),
        Fact(
            content="The ingest-service is owned by Project Kestrel.",
            entities=("ingest-service", "Project Kestrel"),
            writer="onboarding",
            episode="Team onboarding doc, 2026-03-01.",
        ),
        Fact(
            content="postgres-primary has a nightly vacuum window at 02:00 UTC.",
            entities=("postgres-primary",),
            writer="onboarding",
            episode="Team onboarding doc, 2026-03-01.",
        ),
        Fact(
            content="Project Kestrel ships on Fridays.",
            entities=("Project Kestrel",),
            writer="onboarding",
            episode="Team onboarding doc, 2026-03-01.",
        ),
    ),
    question="Ada's project is paging. Who should I wake up, and why?",
    followup_question="Same question again. Who do I page for Ada's project?",
    supersede=Supersede(
        match_text="Bo maintains",
        new_content=("Cy maintains the ingest-service. Bo moved to Project Harrier on 2026-04-15."),
        entities=("Cy", "ingest-service", "Bo"),
        writer="handover-notes",
        episode=("Handover notes, 2026-04-15. Bo moves to Harrier, Cy takes the ingest-service."),
        when="2026-04-15T09:00:00",
        extra_relations=(("Cy", "ingest-service", "maintains"),),
        removed_relations=(("Bo", "ingest-service", "maintains"),),
        extra_entities=(("Cy", "person"),),
    ),
    write_request=(
        "Remember this: the ingest-service must not be deployed during the "
        "postgres-primary vacuum window."
    ),
    seed_entity="Ada",
    system_prompt=(
        "You are an engineering-team assistant with a graph memory. "
        "Always call recall before answering a question about the team, passing "
        "seed_entity when the question names a person or project. "
        "Answer only from what recall returns. Be brief. "
        "Write plain prose in short sentences. Do not use markdown, asterisks, bold, "
        "bullet characters, headings or dashes for emphasis."
    ),
    base_time="2026-03-01T09:00:00",
    asof_time="2026-03-01T10:00:00",
    asof_match="maintains",
    asof_label_past="as the database saw it on 1 March 2026",
    asof_label_now="as it stands today",
    explain=MappingProxyType(
        {
            "setup": "What the team told the assistant on the first day.",
            "two_hop": (
                "No stored sentence contains both 'Ada' and 'page'. Bo is never mentioned "
                "in the question. The answer needs Ada, then Kestrel, then the "
                "ingest-service, then Bo."
            ),
            "stakes": "The service is down while you work out who to call.",
            "supersede": (
                "People move teams. The old row is closed rather than deleted, so the "
                "answer changes without the history going missing."
            ),
            "approval": (
                "The assistant wants to add something to its own memory. A person decides "
                "whether it lands. Every write goes through the gate. Reads run straight "
                "through."
            ),
            "as_of": (
                "A runbook written in March pointed at Bo. Replaying the March read is how "
                "you find out that the runbook was right when it was written."
            ),
            "provenance": (
                "Every belief traces back to the document it came from and the writer who "
                "put it there. A memory you cannot audit is a rumour."
            ),
        }
    ),
)


SCENARIOS: dict[str, Scenario] = {"dinner": DINNER, "oncall": ONCALL}
DEFAULT = "dinner"


# --------------------------------------------------------------------------------------
# Helpers. One place that writes a scenario into a database, one that finds it again.
# --------------------------------------------------------------------------------------


def parse_time(value: str) -> _dt.datetime:
    return _dt.datetime.fromisoformat(value)


def format_day(value: str | _dt.datetime) -> str:
    """'2025-12-26T21:40:00' becomes '26 December 2025'."""
    moment = parse_time(value) if isinstance(value, str) else value
    return f"{moment.day} {moment:%B %Y}"


def build(db: Any, scenario: Scenario, now: _dt.datetime | None = None) -> list[Any]:
    """Write the scenario's entities, edges and facts into ``db``.

    Every row is written with the timestamp the story says it happened, so as_of reads
    later in the demo replay a world that really was that way. Returns the stored
    memories, in scenario order.
    """
    base = now if now is not None else parse_time(scenario.base_time)

    for name, kind in scenario.entities:
        db.upsert_entity(name, kind=kind, now=base)

    for src, dst, rel_kind in scenario.relations:
        db.relate(src, dst, rel_kind=rel_kind, now=base)

    stored = []
    for fact in scenario.facts:
        when = parse_time(fact.when) if fact.when else base
        stored.append(
            db.remember(
                fact.content,
                entities=list(fact.entities),
                writer=fact.writer,
                episode=fact.episode,
                now=when,
            )
        )

    db.rebuild_fts_index()
    return stored


def find_fact(db: Any, scenario: Scenario) -> Any:
    """Find the memory the scenario's supersede targets, by walking out from the seed."""
    needle = scenario.supersede.match_text.lower()
    for memory in db.recall_2hop(scenario.seed_entity, limit=100):
        if needle in memory.content.lower():
            return memory
    raise LookupError(
        f"no memory matching {scenario.supersede.match_text!r} near {scenario.seed_entity!r}"
    )


def apply_supersede(db: Any, scenario: Scenario) -> Correction:
    """Close the old belief, write the new one, and move the edges the change moves.

    The whole correction is one transaction: the memory, the entities the change
    introduces, the edges it closes and the edges it opens. Half of it landing would leave
    the graph saying two contradictory things at once, so if any step raises, the old
    memory is still current and every edge is where it was.

    Returns a :class:`Correction`. The old memory is still readable afterwards and reports
    is_current False.
    """
    change = scenario.supersede
    when = parse_time(change.when) if change.when else None
    stamp: dict[str, Any] = {"now": when} if when is not None else {}
    old = find_fact(db, scenario)

    with db.transaction():
        replacement = db.supersede(
            old.memory_id,
            change.new_content,
            entities=list(change.entities),
            writer=change.writer,
            episode=change.episode,
            **stamp,
        )
        for name, kind in change.extra_entities:
            db.upsert_entity(name, kind=kind, **stamp)
        # unrelate returns how many edge versions it closed, so what is recorded here is
        # what the database did rather than what the scenario asked for; an edge that was
        # already closed returns zero and is left out.
        closed = tuple(
            (src, dst, rel_kind)
            for src, dst, rel_kind in change.removed_relations
            if db.unrelate(src, dst, rel_kind=rel_kind, **stamp)
        )
        for src, dst, rel_kind in change.extra_relations:
            db.relate(src, dst, rel_kind=rel_kind, **stamp)

    db.rebuild_fts_index()
    return Correction(old, replacement, closed, tuple(change.extra_relations))


def entity_kinds(scenario: Scenario) -> dict[str, str]:
    kinds = dict(scenario.entities)
    kinds.update(dict(scenario.supersede.extra_entities))
    return kinds


def all_relations(scenario: Scenario) -> tuple[tuple[str, str, str], ...]:
    """Every edge the scenario ever writes, including the ones the correction later closes."""
    return tuple(scenario.relations) + tuple(scenario.supersede.extra_relations)


def current_relations(scenario: Scenario) -> tuple[tuple[str, str, str], ...]:
    """The edges that stand once the correction has been applied."""
    removed = set(scenario.supersede.removed_relations)
    return tuple(edge for edge in all_relations(scenario) if edge not in removed)


def timeline(scenario: Scenario) -> list[tuple[str, Fact]]:
    """The facts in the order they were told, each with a human date."""
    rows = []
    for fact in scenario.facts:
        when = fact.when or scenario.base_time
        rows.append((when, fact))
    rows.sort(key=lambda row: row[0])
    return [(format_day(when), fact) for when, fact in rows]
