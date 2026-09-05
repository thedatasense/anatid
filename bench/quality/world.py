"""A small engineering organisation, simulated over eighteen months, with exact state at any date.

The benchmark needs gold answers that are right by construction rather than by hand.  This
module is the source of truth: 12 services, 8 people, 5 teams, and a seeded sequence of
dated events (ownership handovers, on-call rotations, team moves, dependency changes, deploy
constraints, decisions with reasons, incident reviews, observations, and a few records that
were written down wrong and later corrected).  Every event carries the canonical facts,
corrections and relations it establishes, so :mod:`bench.quality.gen_corpus` can render it as
a note in prose and as the gold memory patch for that note, and can ask questions whose
answers come from :meth:`World.owner_at`, :meth:`World.oncall_at` and friends.

Two histories are kept for service ownership: ``owner`` is what was true, ``recorded_owner``
is what the notes said.  They differ inside the intervals listed in :attr:`World.wrong_records`,
where a handover note named the wrong team until a correction note fixed it.  Gold patches
follow the record (an extractor can only extract what the note says); question gold follows
the truth, and temporal questions avoid the ambiguous intervals.

Everything is deterministic for a given seed.  Dates are ``datetime.date`` values; the harness
turns them into timestamps.
"""

from __future__ import annotations

import datetime as _dt
import random
from dataclasses import dataclass, field
from itertools import pairwise

__all__ = [
    "START",
    "END",
    "NOW",
    "DEFAULT_SEED",
    "Person",
    "Team",
    "Service",
    "Change",
    "History",
    "DepInterval",
    "Event",
    "World",
    "content_owner",
    "content_member",
    "content_oncall",
    "content_dep",
    "content_window",
]

START = _dt.date(2025, 1, 6)  # a Monday
END = _dt.date(2026, 6, 26)  # last possible note date
NOW = _dt.date(2026, 6, 30)  # the date every "now" question is asked on
DEFAULT_SEED = 20260905

# --------------------------------------------------------------------------- the cast


@dataclass(frozen=True)
class Person:
    name: str  # canonical entity name (the first name; unique in this organisation)
    full_name: str

    @property
    def forms(self) -> tuple[str, ...]:
        return (self.name, self.full_name)


@dataclass(frozen=True)
class Team:
    name: str

    @property
    def forms(self) -> tuple[str, ...]:
        return (self.name, f"the {self.name} team", f"team {self.name}")


@dataclass(frozen=True)
class Service:
    name: str  # canonical entity name
    display: str  # how the canonical fact sentences refer to it ("the ledger", "billing-api")
    forms: tuple[str, ...]  # surface forms the notes use

    @property
    def cap(self) -> str:
        return self.display[0].upper() + self.display[1:]


PEOPLE = (
    Person("Priya", "Priya Raman"),
    Person("Tomasz", "Tomasz Nowak"),
    Person("Lena", "Lena Fischer"),
    Person("Marcus", "Marcus Oyelaran"),
    Person("Yuki", "Yuki Tanaka"),
    Person("Diego", "Diego Alvarez"),
    Person("Farah", "Farah Haddad"),
    Person("Oskar", "Oskar Lind"),
)

TEAMS = (Team("Atlas"), Team("Boreal"), Team("Cinder"), Team("Dune"), Team("Ember"))

SERVICES = (
    Service("ledger", "the ledger", ("the ledger", "the ledger service", "Ledger")),
    Service(
        "billing-api", "billing-api", ("billing-api", "the billing API", "the billing service")
    ),
    Service("notifier", "the notifier", ("the notifier", "the notification service", "notifier")),
    Service(
        "search-indexer", "search-indexer", ("search-indexer", "the search indexer", "the indexer")
    ),
    Service(
        "auth-gateway",
        "auth-gateway",
        ("auth-gateway", "the auth gateway", "the authentication gateway"),
    ),
    Service(
        "catalog", "the catalog", ("the catalog", "the catalog service", "the product catalog")
    ),
    Service(
        "image-resizer", "image-resizer", ("image-resizer", "the image resizer", "the resizer")
    ),
    Service("event-bus", "the event-bus", ("the event-bus", "the event bus", "the bus")),
    Service(
        "reporting", "reporting", ("reporting", "the reporting service", "the reports pipeline")
    ),
    Service("scheduler", "the scheduler", ("the scheduler", "the job scheduler", "scheduler")),
    Service("webhooks", "webhooks", ("webhooks", "the webhooks service", "the webhook dispatcher")),
    Service(
        "feature-flags",
        "feature-flags",
        ("feature-flags", "the flags service", "the feature flag service"),
    ),
)

INITIAL_MEMBERS: dict[str, tuple[str, ...]] = {
    "Atlas": ("Priya", "Tomasz"),
    "Boreal": ("Lena", "Marcus"),
    "Cinder": ("Yuki", "Diego"),
    "Dune": ("Farah",),
    "Ember": ("Oskar",),
}

#: (person, from team, to team, earliest date, latest date) for the three team moves.
MOVES = (
    ("Tomasz", "Atlas", "Dune", _dt.date(2025, 6, 23), _dt.date(2025, 8, 4)),
    ("Diego", "Cinder", "Ember", _dt.date(2025, 11, 10), _dt.date(2025, 12, 15)),
    ("Marcus", "Boreal", "Atlas", _dt.date(2026, 2, 16), _dt.date(2026, 3, 23)),
)

INITIAL_OWNERS: dict[str, str] = {
    "ledger": "Atlas",
    "billing-api": "Atlas",
    "feature-flags": "Atlas",
    "event-bus": "Boreal",
    "notifier": "Boreal",
    "webhooks": "Boreal",
    "catalog": "Cinder",
    "search-indexer": "Cinder",
    "image-resizer": "Dune",
    "reporting": "Dune",
    "scheduler": "Ember",
    "auth-gateway": "Ember",
}

#: (service, dependency, what it is used for).  Four services have no dependency at all and
#: six are depended on by nothing; both gaps feed the abstention questions.
DEPENDENCIES = (
    ("billing-api", "ledger", "invoice postings"),
    ("reporting", "ledger", "the nightly balance export"),
    ("notifier", "event-bus", "delivery events"),
    ("webhooks", "event-bus", "outbound events"),
    ("search-indexer", "catalog", "product documents"),
    ("catalog", "image-resizer", "thumbnails"),
    ("scheduler", "feature-flags", "per-job kill switches"),
    ("ledger", "auth-gateway", "service tokens"),
)

#: Deploy windows as (long form used in the canonical fact, short form accepted as an alias).
WINDOW_POOL = (
    ("on Tuesdays and Thursdays between 10:00 and 12:00 UTC", "Tue/Thu 10:00-12:00 UTC"),
    ("on weekday mornings before 11:00 UTC", "weekday mornings before 11:00 UTC"),
    ("on Mondays and Wednesdays between 14:00 and 16:00 UTC", "Mon/Wed 14:00-16:00 UTC"),
    ("on Wednesdays between 09:00 and 11:00 UTC", "Wednesdays 09:00-11:00 UTC"),
    (
        "on weekdays between 08:00 and 10:00 UTC, never on Fridays",
        "weekdays 08:00-10:00 UTC except Fridays",
    ),
    ("on Tuesdays between 13:00 and 15:00 UTC", "Tuesdays 13:00-15:00 UTC"),
    ("on Mondays to Thursdays between 09:00 and 17:00 UTC", "Mon-Thu 09:00-17:00 UTC"),
    ("on Thursdays between 07:00 and 09:00 UTC", "Thursdays 07:00-09:00 UTC"),
    ("on weekdays after 15:00 UTC", "weekdays after 15:00 UTC"),
    ("on Mondays between 10:00 and 12:00 UTC", "Mondays 10:00-12:00 UTC"),
    ("on Tuesdays and Fridays between 09:00 and 10:30 UTC", "Tue/Fri 09:00-10:30 UTC"),
)

#: Services with a deploy window; the other four have none (abstention targets).
WINDOWED = (
    "ledger",
    "billing-api",
    "auth-gateway",
    "event-bus",
    "catalog",
    "reporting",
    "scheduler",
    "webhooks",
)
#: Services whose window changes once during the period.
WINDOW_CHANGES = ("ledger", "catalog", "scheduler")

PINS = (
    ("reporting", "Python 3.11", "the pandas 3 upgrade lands"),
    ("ledger", "Postgres 14", "the partitioning work finishes"),
    ("search-indexer", "OpenSearch 2.11", "the full reindex completes"),
)

APPROVALS = (
    ("auth-gateway", "two reviewer approvals and a security sign-off"),
    ("billing-api", "a rollback plan attached to the pull request"),
)

#: (service, freeze start, freeze end, reason)
FREEZES = (
    ("billing-api", _dt.date(2025, 11, 24), _dt.date(2025, 12, 2), "the peak season"),
    ("ledger", _dt.date(2026, 3, 30), _dt.date(2026, 4, 3), "the fiscal year close"),
)

#: Decisions with reasons.  ``what`` is the canonical statement, ``q`` its question form.
DECISIONS = (
    {
        "service": "notifier",
        "what": "the notifier will retry failed pushes three times with exponential backoff",
        "q": "Why does the notifier retry failed pushes three times with exponential backoff?",
        "reason": "the flat one-second retry was hammering the push provider during its outages",
        "short": "the reason is the flat one-second retry hammering the push provider during outages",
    },
    {
        "service": "catalog",
        "what": "the catalog will keep image metadata in a separate table",
        "q": "Why did the catalog move image metadata into a separate table?",
        "reason": "the joined writes were locking product updates for seconds at a time",
        "short": "joined writes were locking product updates",
    },
    {
        "service": "scheduler",
        "what": "the scheduler will use leader election instead of a single fixed node",
        "q": "Why did the scheduler switch to leader election instead of a single fixed node?",
        "reason": "the fixed node turned every restart into a missed run",
        "short": "the fixed node made every restart a missed run",
    },
    {
        "service": "feature-flags",
        "what": "clients will cache feature-flags evaluations for 30 seconds",
        "q": "Why do clients cache feature-flags evaluations for 30 seconds?",
        "reason": "feature-flags saw 40,000 requests per second at peak and most of them were identical",
        "short": "the flags service saw 40k identical requests per second at peak",
    },
    {
        "service": "webhooks",
        "what": "webhooks will sign payloads with rotating keys",
        "q": "Why does webhooks sign payloads with rotating keys?",
        "reason": "a partner asked for key rotation in their security review",
        "short": "a partner's security review asked for key rotation",
    },
    {
        "service": "search-indexer",
        "what": "search-indexer will rebuild the index nightly instead of on every catalog write",
        "q": "Why does search-indexer rebuild the index nightly instead of on every catalog write?",
        "reason": "incremental updates left stale entries behind after failed writes",
        "short": "incremental updates left stale entries after failed writes",
    },
    {
        "service": "auth-gateway",
        "what": "auth-gateway tokens will expire after 15 minutes instead of an hour",
        "q": "Why were auth-gateway tokens shortened to 15 minutes?",
        "reason": "the security audit flagged long-lived tokens",
        "short": "the security audit flagged long-lived tokens",
    },
    {
        "service": "image-resizer",
        "what": "image-resizer will keep original uploads for 90 days only",
        "q": "Why does image-resizer keep original uploads for only 90 days?",
        "reason": "storage for originals doubled in six months",
        "short": "storage for originals doubled in six months",
    },
    {
        "service": "event-bus",
        "what": "the event-bus will retain messages for seven days",
        "q": "Why does the event-bus retain messages for seven days?",
        "reason": "replaying a full week has covered every incident recovery so far",
        "short": "a week of replay covered every incident recovery so far",
    },
)

#: The dependency rewire and the new dependency are decisions too, with their own notes.
DEP_REWIRE = {
    "service": "reporting",
    "old_dep": "ledger",
    "new_dep": "event-bus",
    "what": "reporting will read from the event-bus instead of querying the ledger directly",
    "q": "Why did reporting stop querying the ledger directly and read from the event-bus instead?",
    "reason": "the nightly ledger scans were blocking billing runs",
    "short": "nightly ledger scans were blocking billing runs",
}
DEP_NEW = {
    "service": "billing-api",
    "dep": "feature-flags",
    "what": "billing-api will gate the new invoice flow behind feature-flags",
    "q": "Why is the new billing-api invoice flow gated behind feature-flags?",
    "reason": "the rollout needs a per-tenant kill switch",
    "short": "the rollout needs a per-tenant kill switch",
}

INCIDENTS = (
    {
        "inc": "INC-2025-004",
        "service": "notifier",
        "symptom": "missed alerts for 40 minutes",
        "cause": "a consumer offset stuck after a broker restart",
        "what": "notifier consumers will commit offsets every five seconds",
        "q": "Why do notifier consumers commit offsets every five seconds?",
        "reason": "a stuck offset went unnoticed for 40 minutes during INC-2025-004",
        "lo": _dt.date(2025, 2, 3),
        "hi": _dt.date(2025, 3, 14),
    },
    {
        "inc": "INC-2025-011",
        "service": "ledger",
        "symptom": "saw write latency spike tenfold for an hour",
        "cause": "an index missing after the schema migration",
        "what": "ledger migrations will ship with a reviewed query plan",
        "q": "Why do ledger migrations ship with a reviewed query plan?",
        "reason": "a missing index after a migration caused INC-2025-011",
        "lo": _dt.date(2025, 4, 14),
        "hi": _dt.date(2025, 5, 30),
    },
    {
        "inc": "INC-2025-017",
        "service": "auth-gateway",
        "symptom": "rejected every login for 12 minutes",
        "cause": "an expired intermediate certificate",
        "what": "certificate expiry will alert 30 days ahead",
        "q": "Why does certificate expiry alert 30 days ahead?",
        "reason": "an expired intermediate certificate took logins down in INC-2025-017",
        "lo": _dt.date(2025, 7, 7),
        "hi": _dt.date(2025, 8, 22),
    },
    {
        "inc": "INC-2025-023",
        "service": "search-indexer",
        "symptom": "served stale results for two days",
        "cause": "a reindex job that failed silently on a full disk",
        "what": "reindex jobs will page on failure",
        "q": "Why do reindex jobs page on failure?",
        "reason": "a silent reindex failure left stale results for two days in INC-2025-023",
        "lo": _dt.date(2025, 10, 6),
        "hi": _dt.date(2025, 11, 14),
    },
    {
        "inc": "INC-2026-003",
        "service": "webhooks",
        "symptom": "delivered some events twice",
        "cause": "a retry loop that ignored idempotency keys",
        "what": "webhooks will require an idempotency key on every delivery",
        "q": "Why does webhooks require an idempotency key on every delivery?",
        "reason": "duplicate deliveries in INC-2026-003 came from retries without one",
        "lo": _dt.date(2026, 1, 12),
        "hi": _dt.date(2026, 2, 27),
    },
    {
        "inc": "INC-2026-009",
        "service": "scheduler",
        "symptom": "ran the nightly jobs twice",
        "cause": "two leaders elected during a network partition",
        "what": "the scheduler leader lease will be shortened to ten seconds",
        "q": "Why was the scheduler leader lease shortened to ten seconds?",
        "reason": "two leaders ran the nightly jobs twice in INC-2026-009",
        "lo": _dt.date(2026, 4, 6),
        "hi": _dt.date(2026, 5, 22),
    },
)

OBSERVATIONS = (
    ("p99 latency", "doubled", "Tuesday's deploy"),
    ("error rate", "crept up to 2 percent", "the dependency bump"),
    ("memory use", "grew steadily for a week", "the cache change"),
    ("queue depth", "hit 50,000", "the marketing send"),
    ("CPU", "sat at 90 percent", "the traffic shift"),
    ("disk usage", "jumped 30 percent", "the retention change"),
    ("cold start time", "tripled", "the runtime upgrade"),
    ("request volume", "halved", "the client rollout"),
)

OBSERVATION_COUNT = 30
#: How many notes restate a fact that is already recorded and still true.  They are what a
#: real notes stream looks like, and the gold pipeline drops them as duplicates.
REMINDER_COUNT = 20

#: Facts about teams and people that are true, dated, and mostly never asked about.
FILLER = (
    {"kind": "planning", "team": "Atlas", "value": "Tuesday"},
    {"kind": "planning", "team": "Boreal", "value": "Thursday"},
    {"kind": "planning", "team": "Cinder", "value": "Monday"},
    {"kind": "planning", "team": "Dune", "value": "Wednesday"},
    {"kind": "planning", "team": "Ember", "value": "Friday"},
    {"kind": "practice", "team": "Cinder", "value": "trunk-based development"},
    {"kind": "practice", "team": "Boreal", "value": "a weekly dependency update day"},
    {"kind": "practice", "team": "Ember", "value": "pairing on every production change"},
    {"kind": "captain", "person": "Farah", "value": "Q3 2025"},
    {"kind": "captain", "person": "Lena", "value": "Q1 2026"},
    {"kind": "review", "person": "Oskar", "value": "the weekly architecture review"},
    {"kind": "review", "person": "Yuki", "value": "the monthly cost review"},
    {"kind": "floor", "team": "Dune", "value": "fourth"},
    {"kind": "floor", "team": "Atlas", "value": "second"},
    {"kind": "floor", "team": "Cinder", "value": "third"},
    {"kind": "practice", "team": "Dune", "value": "a monthly game day"},
    {"kind": "practice", "team": "Atlas", "value": "a no-deploy Friday"},
    {"kind": "review", "person": "Priya", "value": "the quarterly capacity review"},
)


# --------------------------------------------------------------------------- canonical facts


def content_owner(team: str, svc: Service) -> str:
    return f"{team} owns {svc.display}"


def content_member(person: str, team: str) -> str:
    return f"{person} is a member of {team}"


def content_oncall(person: str, team: str) -> str:
    return f"{person} is on call for {team}"


def content_dep(svc: Service, dep: Service) -> str:
    return f"{svc.cap} depends on {dep.display}"


def content_window(svc: Service, window: str) -> str:
    return f"{svc.cap} deploys only {window}"


# --------------------------------------------------------------------------- histories


@dataclass(frozen=True)
class Change:
    date: _dt.date
    value: str
    event_id: int


class History:
    """The dated values one attribute took, and the value at any date."""

    def __init__(self) -> None:
        self.changes: list[Change] = []

    def add(self, date: _dt.date, value: str, event_id: int) -> None:
        self.changes.append(Change(date, value, event_id))
        self.changes.sort(key=lambda c: (c.date, c.event_id))

    def at(self, date: _dt.date) -> Change | None:
        out = None
        for change in self.changes:
            if change.date <= date:
                out = change
            else:
                break
        return out

    def value_at(self, date: _dt.date) -> str | None:
        change = self.at(date)
        return None if change is None else change.value

    @property
    def current(self) -> Change:
        return self.changes[-1]

    @property
    def first(self) -> Change:
        return self.changes[0]

    def intervals(self) -> list[tuple[_dt.date, _dt.date | None, str, int]]:
        """``(start, end or None, value, event_id)`` for each value the attribute held."""
        out = []
        for i, change in enumerate(self.changes):
            end = self.changes[i + 1].date if i + 1 < len(self.changes) else None
            out.append((change.date, end, change.value, change.event_id))
        return out


@dataclass
class DepInterval:
    dep: str
    start: _dt.date
    end: _dt.date | None
    add_event: int
    remove_event: int | None = None


# --------------------------------------------------------------------------- events


@dataclass
class Event:
    """One dated thing that happened, with the gold operations it establishes.

    ``facts`` are ``{"content", "entities", "kind"}``; ``corrections`` are
    ``{"new_content", "old_text", "entities", "kind"}``; relations are ``(src, dst, rel_kind)``.
    ``team`` is the team whose note reports the event (it names the standup source).  Events
    sharing a ``bundle`` id are rendered into the same note.
    """

    event_id: int
    date: _dt.date
    kind: str
    source_kind: str
    team: str
    payload: dict
    facts: list[dict] = field(default_factory=list)
    corrections: list[dict] = field(default_factory=list)
    add_relations: list[tuple[str, str, str]] = field(default_factory=list)
    remove_relations: list[tuple[str, str, str]] = field(default_factory=list)
    label: str | None = None
    bundle: int | None = None


# --------------------------------------------------------------------------- the world


class World:
    """Build the organisation's history from a seed and answer questions about it."""

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.people = {p.name: p for p in PEOPLE}
        self.teams = {t.name: t for t in TEAMS}
        self.services = {s.name: s for s in SERVICES}
        self.events: list[Event] = []
        self.owner: dict[str, History] = {s: History() for s in self.services}
        self.recorded_owner: dict[str, History] = {s: History() for s in self.services}
        self.member: dict[str, History] = {p: History() for p in self.people}
        self.oncall: dict[str, History] = {t: History() for t in self.teams}
        self.deps: dict[str, list[DepInterval]] = {s: [] for s in self.services}
        self.window: dict[str, History] = {}
        self.pins: list[dict] = []
        self.approvals: list[dict] = []
        self.freezes: list[dict] = []
        self.decisions: list[dict] = []
        self.incidents: list[dict] = []
        self.observations: list[dict] = []
        self.filler: list[dict] = []
        self.moves: list[dict] = []
        self.wrong_records: list[dict] = []
        self._build()
        self.events.sort(key=lambda e: (e.date, e.event_id))
        self.by_id = {e.event_id: e for e in self.events}
        for e in self.events:
            if e.team not in self.teams:
                raise AssertionError(
                    f"event {e.event_id} ({e.kind}, {e.date}) has no team: {e.team!r}"
                )
        contents = [f["content"] for e in self.events if e.kind != "reminder" for f in e.facts]
        duplicates = sorted({c for c in contents if contents.count(c) > 1})
        if duplicates:
            raise AssertionError(f"world produced the same fact twice: {duplicates}")

    # ------------------------------------------------------------------ helpers

    def _workday(self, lo: _dt.date, hi: _dt.date) -> _dt.date:
        """A uniformly random Monday-to-Friday date in ``[lo, hi]``."""
        if hi < lo:
            hi = lo
        for _ in range(1000):
            d = lo + _dt.timedelta(days=self.rng.randint(0, (hi - lo).days))
            if d.weekday() < 5:
                return d
        while lo.weekday() >= 5:
            lo += _dt.timedelta(days=1)
        return lo

    def _add(self, **kw) -> Event:
        event = Event(event_id=len(self.events) + 1, **kw)
        self.events.append(event)
        return event

    def _team_day(self, team: str, lo: _dt.date, hi: _dt.date) -> _dt.date:
        """A date for a low-stakes note: half the time a day this team already wrote a standup
        note on (so notes carry more than one item, as real ones do), otherwise any workday."""
        days = sorted(
            {
                e.date
                for e in self.events
                if e.source_kind == "standup" and e.team == team and lo <= e.date <= hi
            }
        )
        if days and self.rng.random() < 0.5:
            return self.rng.choice(days)
        return self._workday(lo, hi)

    def _once_owned(self, service: str, lo: _dt.date) -> _dt.date:
        """``lo``, or the day after the service's first ownership note if that is later: nothing
        is said about a service before the notes have introduced it."""
        first = self.owner[service].first.date + _dt.timedelta(days=1)
        return max(lo, first)

    def members_at(self, team: str, date: _dt.date) -> list[str]:
        return [p for p in self.people if self.member[p].value_at(date) == team]

    def team_at(self, person: str, date: _dt.date) -> str | None:
        return self.member[person].value_at(date)

    def owner_at(self, service: str, date: _dt.date) -> str | None:
        return self.owner[service].value_at(date)

    def recorded_owner_at(self, service: str, date: _dt.date) -> str | None:
        return self.recorded_owner[service].value_at(date)

    def oncall_at(self, team: str, date: _dt.date) -> str | None:
        return self.oncall[team].value_at(date)

    def deps_at(self, service: str, date: _dt.date) -> list[DepInterval]:
        return [
            d for d in self.deps[service] if d.start <= date and (d.end is None or date < d.end)
        ]

    def dependants_at(self, service: str, date: _dt.date) -> list[str]:
        return [s for s in self.services if any(d.dep == service for d in self.deps_at(s, date))]

    def window_at(self, service: str, date: _dt.date) -> str | None:
        history = self.window.get(service)
        return None if history is None else history.value_at(date)

    def services_of(self, team: str, date: _dt.date) -> list[str]:
        return [s for s in self.services if self.owner_at(s, date) == team]

    def ambiguous(self, service: str, date: _dt.date) -> bool:
        """True while the notes recorded a different owner than the truth."""
        return self.owner_at(service, date) != self.recorded_owner_at(service, date)

    def svc(self, name: str) -> Service:
        return self.services[name]

    # ------------------------------------------------------------------ build

    def _build(self) -> None:
        self._build_membership()
        self._build_ownership()
        self._build_oncall()
        self._build_dependencies()
        self._build_constraints()
        self._build_decisions()
        self._build_incidents()
        self._build_observations()
        self._build_filler()
        self._build_reminders()

    def _build_membership(self) -> None:
        for i, team in enumerate(self.teams):
            members = INITIAL_MEMBERS[team]
            date = START + _dt.timedelta(days=i)
            event = self._add(
                date=date,
                kind="membership_init",
                source_kind="standup",
                team=team,
                payload={"team": team, "members": list(members)},
                facts=[
                    {"content": content_member(p, team), "entities": [p, team], "kind": "fact"}
                    for p in members
                ],
                add_relations=[(p, team, "member_of") for p in members],
            )
            for p in members:
                self.member[p].add(date, team, event.event_id)
        for person, old, new, lo, hi in MOVES:
            date = self._workday(lo, hi)
            event = self._add(
                date=date,
                kind="move",
                source_kind="standup",
                team=old,
                payload={"person": person, "old": old, "new": new, "oncall_switch_to": None},
                corrections=[
                    {
                        "new_content": content_member(person, new),
                        "old_text": content_member(person, old),
                        "entities": [person, new],
                        "kind": "fact",
                    }
                ],
                remove_relations=[(person, old, "member_of")],
                add_relations=[(person, new, "member_of")],
            )
            self.member[person].add(date, new, event.event_id)
            self.moves.append(
                {"person": person, "old": old, "new": new, "date": date, "event": event.event_id}
            )

    def _build_ownership(self) -> None:
        first_dates: dict[str, _dt.date] = {}
        for i, team in enumerate(self.teams):
            date = START + _dt.timedelta(days=i + (7 if i >= 2 else 0) + 2)
            while date.weekday() >= 5:
                date += _dt.timedelta(days=1)
            for name in INITIAL_OWNERS:
                if INITIAL_OWNERS[name] != team:
                    continue
                svc = self.svc(name)
                event = self._add(
                    date=date,
                    kind="ownership_init",
                    source_kind="standup",
                    team=team,
                    payload={"service": name, "team": team},
                    facts=[
                        {
                            "content": content_owner(team, svc),
                            "entities": [team, name],
                            "kind": "fact",
                        }
                    ],
                    add_relations=[(team, name, "owns")],
                )
                self.owner[name].add(date, team, event.event_id)
                self.recorded_owner[name].add(date, team, event.event_id)
                first_dates[name] = date

        specs: list[tuple[_dt.date, str]] = []
        per_service: dict[str, list[_dt.date]] = {}
        for name in self.services:
            n = self.rng.choices([1, 2, 3], weights=[4, 5, 3])[0]
            lo = first_dates[name] + _dt.timedelta(days=35)
            hi = END - _dt.timedelta(days=21)
            dates: list[_dt.date] = []
            for _attempt in range(500):
                dates = sorted(
                    lo + _dt.timedelta(days=self.rng.randint(0, (hi - lo).days)) for _ in range(n)
                )
                if all((b - a).days >= 45 for a, b in pairwise(dates)):
                    break
            dates = [self._workday(d, d + _dt.timedelta(days=4)) for d in dates]
            per_service[name] = dates
            specs.extend((dd, name) for dd in dates)
        specs.sort()

        # three handovers are recorded against the wrong team and corrected later
        candidates = []
        for name, dates in per_service.items():
            for i, d in enumerate(dates):
                nxt = dates[i + 1] if i + 1 < len(dates) else END - _dt.timedelta(days=7)
                if (nxt - d).days >= 45:
                    candidates.append((name, d))
        self.rng.shuffle(candidates)
        wrong: dict[str, _dt.date] = {}
        for name, d in candidates:
            if name not in wrong:
                wrong[name] = d
            if len(wrong) == 3:
                break

        for d, name in specs:
            svc = self.svc(name)
            old = self.owner_at(name, d)
            assert old is not None
            new = self.rng.choice([t for t in self.teams if t != old])
            old_person = self.rng.choice(self.members_at(old, d))
            new_person = self.rng.choice(self.members_at(new, d))
            if wrong.get(name) == d:
                wrong_team = self.rng.choice([t for t in self.teams if t not in (old, new)])
                wrong_person = self.rng.choice(self.members_at(wrong_team, d))
                event = self._add(
                    date=d,
                    kind="handover_wrong",
                    source_kind="handover",
                    team=old,
                    label=f"handover/{d.isoformat()}-{name}",
                    payload={
                        "service": name,
                        "old": old,
                        "new": wrong_team,
                        "truth": new,
                        "old_person": old_person,
                        "new_person": wrong_person,
                    },
                    corrections=[
                        {
                            "new_content": content_owner(wrong_team, svc),
                            "old_text": content_owner(old, svc),
                            "entities": [wrong_team, name],
                            "kind": "fact",
                        }
                    ],
                    remove_relations=[(old, name, "owns")],
                    add_relations=[(wrong_team, name, "owns")],
                )
                self.owner[name].add(d, new, event.event_id)
                self.recorded_owner[name].add(d, wrong_team, event.event_id)
                fix = self._workday(d + _dt.timedelta(days=7), d + _dt.timedelta(days=28))
                fix_event = self._add(
                    date=fix,
                    kind="handover_fix",
                    source_kind="handover",
                    team=new,
                    label=f"handover/{fix.isoformat()}-{name}-correction",
                    payload={
                        "service": name,
                        "wrong": wrong_team,
                        "truth": new,
                        "old": old,
                        "wrong_date": d,
                    },
                    facts=[
                        {
                            "content": (
                                f"The handover note of {d.isoformat()} wrongly recorded "
                                f"{wrong_team} as the owner of {svc.display}"
                            ),
                            "entities": [name, wrong_team],
                            "kind": "fact",
                        }
                    ],
                    corrections=[
                        {
                            "new_content": content_owner(new, svc),
                            "old_text": content_owner(wrong_team, svc),
                            "entities": [new, name],
                            "kind": "fact",
                        }
                    ],
                    remove_relations=[(wrong_team, name, "owns")],
                    add_relations=[(new, name, "owns")],
                )
                self.recorded_owner[name].add(fix, new, fix_event.event_id)
                self.wrong_records.append(
                    {
                        "service": name,
                        "old": old,
                        "wrong": wrong_team,
                        "truth": new,
                        "wrong_date": d,
                        "fix_date": fix,
                        "wrong_event": event.event_id,
                        "fix_event": fix_event.event_id,
                    }
                )
            else:
                event = self._add(
                    date=d,
                    kind="handover",
                    source_kind="handover",
                    team=old,
                    label=f"handover/{d.isoformat()}-{name}",
                    payload={
                        "service": name,
                        "old": old,
                        "new": new,
                        "old_person": old_person,
                        "new_person": new_person,
                    },
                    corrections=[
                        {
                            "new_content": content_owner(new, svc),
                            "old_text": content_owner(old, svc),
                            "entities": [new, name],
                            "kind": "fact",
                        }
                    ],
                    remove_relations=[(old, name, "owns")],
                    add_relations=[(new, name, "owns")],
                )
                self.owner[name].add(d, new, event.event_id)
                self.recorded_owner[name].add(d, new, event.event_id)

    def _build_oncall(self) -> None:
        move_events = [e for e in self.events if e.kind == "move"]
        for team in self.teams:
            changes = sorted(
                (e for e in move_events if team in (e.payload["old"], e.payload["new"])),
                key=lambda e: e.date,
            )
            joined = min(self.member[p].first.date for p in INITIAL_MEMBERS[team])
            d0 = self._workday(joined + _dt.timedelta(days=1), joined + _dt.timedelta(days=9))
            members = self.members_at(team, d0)
            cur = self.rng.choice(members)
            event = self._add(
                date=d0,
                kind="oncall_init",
                source_kind="standup",
                team=team,
                payload={"team": team, "person": cur},
                facts=[
                    {"content": content_oncall(cur, team), "entities": [cur, team], "kind": "fact"}
                ],
                add_relations=[(cur, team, "on_call_for")],
            )
            self.oncall[team].add(d0, cur, event.event_id)
            next_rot = d0 + _dt.timedelta(days=self.rng.randint(30, 56))
            i = 0
            while True:
                change = changes[i] if i < len(changes) else None
                if change is not None and change.date <= next_rot:
                    i += 1
                    if change.payload["old"] == team:
                        if change.payload["person"] == cur:
                            remaining = self.members_at(team, change.date)
                            new = self.rng.choice(remaining)
                            switch = self._add(
                                date=change.date,
                                kind="oncall_switch",
                                source_kind="standup",
                                team=team,
                                payload={"team": team, "person": new, "prev": cur, "leaving": cur},
                                corrections=[
                                    {
                                        "new_content": content_oncall(new, team),
                                        "old_text": content_oncall(cur, team),
                                        "entities": [new, team],
                                        "kind": "fact",
                                    }
                                ],
                                remove_relations=[(cur, team, "on_call_for")],
                                add_relations=[(new, team, "on_call_for")],
                                bundle=change.event_id,
                            )
                            change.bundle = change.event_id
                            change.payload["oncall_switch_to"] = new
                            self.oncall[team].add(change.date, new, switch.event_id)
                            cur = new
                    else:
                        next_rot = max(
                            next_rot, change.date + _dt.timedelta(days=self.rng.randint(14, 42))
                        )
                    continue
                if next_rot > END - _dt.timedelta(days=7):
                    break
                others = [m for m in self.members_at(team, next_rot) if m != cur]
                if others:
                    new = self.rng.choice(others)
                    rot = self._add(
                        date=self._workday(next_rot, next_rot + _dt.timedelta(days=4)),
                        kind="oncall",
                        source_kind="standup",
                        team=team,
                        payload={"team": team, "person": new, "prev": cur},
                        corrections=[
                            {
                                "new_content": content_oncall(new, team),
                                "old_text": content_oncall(cur, team),
                                "entities": [new, team],
                                "kind": "fact",
                            }
                        ],
                        remove_relations=[(cur, team, "on_call_for")],
                        add_relations=[(new, team, "on_call_for")],
                    )
                    self.oncall[team].add(rot.date, new, rot.event_id)
                    cur = new
                    next_rot = rot.date + _dt.timedelta(days=self.rng.randint(30, 56))
                else:
                    next_rot += _dt.timedelta(days=self.rng.randint(30, 56))

    def _build_dependencies(self) -> None:
        pair_dates: list[_dt.date] = []
        for i, (name, dep, purpose) in enumerate(DEPENDENCIES):
            # two pairs of dependency notes share a date, so some notes carry two of them
            if i in (1, 3):
                date = max(pair_dates[-1], self._once_owned(name, START))
            else:
                date = self._workday(
                    self._once_owned(name, START + _dt.timedelta(days=3)),
                    START + _dt.timedelta(days=110),
                )
                pair_dates.append(date)
            svc, dsvc = self.svc(name), self.svc(dep)
            team = self.owner_at(name, date)
            assert team is not None
            event = self._add(
                date=date,
                kind="dependency",
                source_kind="standup",
                team=team,
                payload={"service": name, "dep": dep, "purpose": purpose},
                facts=[
                    {"content": content_dep(svc, dsvc), "entities": [name, dep], "kind": "fact"}
                ],
                add_relations=[(name, dep, "depends_on")],
            )
            self.deps[name].append(DepInterval(dep, date, None, event.event_id))

        # the rewire: a decision that changes a dependency
        r = DEP_REWIRE
        date = self._workday(_dt.date(2025, 8, 11), _dt.date(2025, 10, 10))
        svc, old_dep, new_dep = (
            self.svc(r["service"]),
            self.svc(r["old_dep"]),
            self.svc(r["new_dep"]),
        )
        team = self.owner_at(r["service"], date)
        event = self._add(
            date=date,
            kind="dependency_rewire",
            source_kind="standup",
            team=team,
            payload=dict(r),
            facts=[
                {
                    "content": f"Decision: {r['what']}, because {r['reason']}",
                    "entities": [r["service"], r["old_dep"], r["new_dep"]],
                    "kind": "decision",
                }
            ],
            corrections=[
                {
                    "new_content": content_dep(svc, new_dep),
                    "old_text": content_dep(svc, old_dep),
                    "entities": [r["service"], r["new_dep"]],
                    "kind": "fact",
                }
            ],
            remove_relations=[(r["service"], r["old_dep"], "depends_on")],
            add_relations=[(r["service"], r["new_dep"], "depends_on")],
        )
        for interval in self.deps[r["service"]]:
            if interval.dep == r["old_dep"] and interval.end is None:
                interval.end = date
                interval.remove_event = event.event_id
        self.deps[r["service"]].append(DepInterval(r["new_dep"], date, None, event.event_id))
        self.decisions.append({**r, "date": date, "event": event.event_id, "note_kind": "rewire"})

        n = DEP_NEW
        date = self._workday(_dt.date(2025, 9, 1), _dt.date(2025, 11, 7))
        svc, dsvc = self.svc(n["service"]), self.svc(n["dep"])
        team = self.owner_at(n["service"], date)
        event = self._add(
            date=date,
            kind="dependency_new",
            source_kind="standup",
            team=team,
            payload=dict(n),
            facts=[
                {
                    "content": f"Decision: {n['what']}, because {n['reason']}",
                    "entities": [n["service"], n["dep"]],
                    "kind": "decision",
                },
                {
                    "content": content_dep(svc, dsvc),
                    "entities": [n["service"], n["dep"]],
                    "kind": "fact",
                },
            ],
            add_relations=[(n["service"], n["dep"], "depends_on")],
        )
        self.deps[n["service"]].append(DepInterval(n["dep"], date, None, event.event_id))
        self.decisions.append({**n, "date": date, "event": event.event_id, "note_kind": "new_dep"})

    def _build_constraints(self) -> None:
        pool = list(WINDOW_POOL)
        self.rng.shuffle(pool)
        assigned: dict[str, tuple[str, str]] = {}
        for name in WINDOWED:
            assigned[name] = pool.pop()
        for name in WINDOWED:
            svc = self.svc(name)
            date = self._workday(
                self._once_owned(name, START + _dt.timedelta(days=5)),
                START + _dt.timedelta(days=120),
            )
            team = self.owner_at(name, date)
            long, short = assigned[name]
            event = self._add(
                date=date,
                kind="window",
                source_kind="standup",
                team=team,
                payload={"service": name, "window": long, "short": short},
                facts=[
                    {"content": content_window(svc, long), "entities": [name], "kind": "constraint"}
                ],
            )
            self.window[name] = History()
            self.window[name].add(date, long, event.event_id)
        for name in WINDOW_CHANGES:
            svc = self.svc(name)
            old_long, old_short = assigned[name]
            new_long, new_short = pool.pop()
            first = self.window[name].first.date
            date = self._workday(
                max(first + _dt.timedelta(days=90), _dt.date(2025, 6, 2)), _dt.date(2026, 5, 1)
            )
            team = self.owner_at(name, date)
            event = self._add(
                date=date,
                kind="window_change",
                source_kind="standup",
                team=team,
                payload={
                    "service": name,
                    "old": old_long,
                    "old_short": old_short,
                    "new": new_long,
                    "short": new_short,
                },
                corrections=[
                    {
                        "new_content": content_window(svc, new_long),
                        "old_text": content_window(svc, old_long),
                        "entities": [name],
                        "kind": "constraint",
                    }
                ],
            )
            self.window[name].add(date, new_long, event.event_id)
        for name, version, condition in PINS:
            svc = self.svc(name)
            date = self._workday(
                self._once_owned(name, START + _dt.timedelta(days=20)), _dt.date(2025, 12, 12)
            )
            team = self.owner_at(name, date)
            event = self._add(
                date=date,
                kind="pin",
                source_kind="standup",
                team=team,
                payload={"service": name, "version": version, "condition": condition},
                facts=[
                    {
                        "content": f"{svc.cap} must stay on {version} until {condition}",
                        "entities": [name],
                        "kind": "constraint",
                    }
                ],
            )
            self.pins.append(
                {
                    "service": name,
                    "version": version,
                    "condition": condition,
                    "date": date,
                    "event": event.event_id,
                }
            )
        for name, rule in APPROVALS:
            svc = self.svc(name)
            date = self._workday(
                self._once_owned(name, START + _dt.timedelta(days=20)), _dt.date(2026, 2, 27)
            )
            team = self.owner_at(name, date)
            event = self._add(
                date=date,
                kind="approval",
                source_kind="standup",
                team=team,
                payload={"service": name, "rule": rule},
                facts=[
                    {
                        "content": f"Changes to {svc.display} need {rule}",
                        "entities": [name],
                        "kind": "constraint",
                    }
                ],
            )
            self.approvals.append(
                {"service": name, "rule": rule, "date": date, "event": event.event_id}
            )
        for name, f_start, f_end, reason in FREEZES:
            svc = self.svc(name)
            date = self._workday(f_start - _dt.timedelta(days=21), f_start - _dt.timedelta(days=10))
            team = self.owner_at(name, date)
            event = self._add(
                date=date,
                kind="freeze",
                source_kind="standup",
                team=team,
                payload={"service": name, "start": f_start, "end": f_end, "reason": reason},
                facts=[
                    {
                        "content": (
                            f"No deploys to {svc.display} from {f_start.isoformat()} to "
                            f"{f_end.isoformat()} because of {reason}"
                        ),
                        "entities": [name],
                        "kind": "constraint",
                    }
                ],
            )
            self.freezes.append(
                {
                    "service": name,
                    "start": f_start,
                    "end": f_end,
                    "reason": reason,
                    "date": date,
                    "event": event.event_id,
                }
            )

    def _build_decisions(self) -> None:
        span = (END - _dt.timedelta(days=14) - (START + _dt.timedelta(days=30))).days
        for i, dec in enumerate(DECISIONS):
            lo = START + _dt.timedelta(days=30 + (span * i) // len(DECISIONS))
            hi = START + _dt.timedelta(days=30 + (span * (i + 1)) // len(DECISIONS))
            date = self._workday(lo, hi)
            team = self.owner_at(dec["service"], date)
            event = self._add(
                date=date,
                kind="decision",
                source_kind="standup",
                team=team,
                payload=dict(dec),
                facts=[
                    {
                        "content": f"Decision: {dec['what']}, because {dec['reason']}",
                        "entities": [dec["service"]],
                        "kind": "decision",
                    }
                ],
            )
            self.decisions.append(
                {**dec, "date": date, "event": event.event_id, "note_kind": "decision"}
            )

    def _build_incidents(self) -> None:
        for inc in INCIDENTS:
            date = self._workday(inc["lo"], inc["hi"])
            incident_date = date - _dt.timedelta(days=self.rng.randint(2, 6))
            svc = self.svc(inc["service"])
            team = self.owner_at(inc["service"], date)
            event = self._add(
                date=date,
                kind="incident",
                source_kind="incident_review",
                team=team,
                label=f"incident-review/{inc['inc']}",
                payload={**inc, "incident_date": incident_date},
                facts=[
                    {
                        "content": (
                            f"{inc['inc']}: {svc.display} {inc['symptom']} on {incident_date.isoformat()}; "
                            f"the root cause was {inc['cause']}"
                        ),
                        "entities": [inc["service"], inc["inc"]],
                        "kind": "fact",
                    },
                    {
                        "content": f"Decision: {inc['what']}, because {inc['reason']}",
                        "entities": [inc["service"], inc["inc"]],
                        "kind": "decision",
                    },
                ],
            )
            self.incidents.append(
                {**inc, "incident_date": incident_date, "date": date, "event": event.event_id}
            )

    def _build_observations(self) -> None:
        names = list(self.services)
        for i in range(OBSERVATION_COUNT):
            name = names[i % len(names)]
            svc = self.svc(name)
            team = self.rng.choice(list(self.teams))
            for _attempt in range(100):
                date = self._team_day(
                    team, START + _dt.timedelta(days=40), END - _dt.timedelta(days=5)
                )
                if self.owner_at(name, date) != team:
                    break
            else:
                continue
            # the phrase index steps by a value coprime with the phrase count per service cycle,
            # so no (service, phrase) pair repeats within OBSERVATION_COUNT
            metric, change, trigger = OBSERVATIONS[(i + i // len(names)) % len(OBSERVATIONS)]
            content = f"{team} noticed that {svc.display}'s {metric} {change} after {trigger}"
            event = self._add(
                date=date,
                kind="observation",
                source_kind="standup",
                team=team,
                payload={
                    "team": team,
                    "service": name,
                    "metric": metric,
                    "change": change,
                    "trigger": trigger,
                },
                facts=[{"content": content, "entities": [name, team], "kind": "observation"}],
            )
            self.observations.append(
                {
                    "team": team,
                    "service": name,
                    "date": date,
                    "event": event.event_id,
                    "content": content,
                }
            )

    def _build_filler(self) -> None:
        for item in FILLER:
            anchor = item.get("team") or self.team_at(
                item["person"], START + _dt.timedelta(days=10)
            )
            date = self._team_day(
                anchor, START + _dt.timedelta(days=10), END - _dt.timedelta(days=10)
            )
            if item["kind"] == "planning":
                team = item["team"]
                content = f"{team} holds its planning meeting on {item['value']}s"
                entities = [team]
            elif item["kind"] == "practice":
                team = item["team"]
                content = f"{team} adopted {item['value']}"
                entities = [team]
            elif item["kind"] == "captain":
                team = self.team_at(item["person"], date)
                content = f"{item['person']} is the release captain for {item['value']}"
                entities = [item["person"]]
            elif item["kind"] == "review":
                team = self.team_at(item["person"], date)
                content = f"{item['person']} runs {item['value']}"
                entities = [item["person"]]
            else:
                team = item["team"]
                content = f"{team} sits on the {item['value']} floor"
                entities = [team]
            event = self._add(
                date=date,
                kind="filler",
                source_kind="standup",
                team=team,
                payload={**item, "content": content},
                facts=[{"content": content, "entities": entities, "kind": "fact"}],
            )
            self.filler.append({**item, "date": date, "event": event.event_id, "content": content})

    def _build_reminders(self) -> None:
        """Notes that restate something already recorded and still true at that date."""
        kinds = ["window", "dep", "owner", "oncall"]
        made = 0
        attempts = 0
        while made < REMINDER_COUNT and attempts < 2000:
            attempts += 1
            what = kinds[made % len(kinds)]
            lo, hi = START + _dt.timedelta(days=45), END - _dt.timedelta(days=3)
            if what == "window":
                name = self.rng.choice(list(self.window))
                date = self._team_day(self.owner_at(name, hi) or "Atlas", lo, hi)
                change = self.window[name].at(date)
                if change is None or (date - change.date).days < 20:
                    continue
                svc = self.svc(name)
                team = self.owner_at(name, date)
                facts = [
                    {
                        "content": content_window(svc, change.value),
                        "entities": [name],
                        "kind": "constraint",
                    }
                ]
                payload = {
                    "what": what,
                    "service": name,
                    "window": change.value,
                    "since": change.date,
                }
                rels: list[tuple[str, str, str]] = []
            elif what == "dep":
                name = self.rng.choice(list(self.services))
                date = self._team_day(self.owner_at(name, hi) or "Atlas", lo, hi)
                active = self.deps_at(name, date)
                if not active or (date - active[0].start).days < 20:
                    continue
                dep = active[0]
                svc, dsvc = self.svc(name), self.svc(dep.dep)
                team = self.owner_at(name, date)
                facts = [
                    {"content": content_dep(svc, dsvc), "entities": [name, dep.dep], "kind": "fact"}
                ]
                payload = {"what": what, "service": name, "dep": dep.dep, "since": dep.start}
                rels = [(name, dep.dep, "depends_on")]
            elif what == "owner":
                name = self.rng.choice(list(self.services))
                date = self._team_day(self.owner_at(name, hi) or "Atlas", lo, hi)
                change = self.owner[name].at(date)
                if change is None or (date - change.date).days < 20 or self.ambiguous(name, date):
                    continue
                svc = self.svc(name)
                team = change.value
                facts = [
                    {"content": content_owner(team, svc), "entities": [team, name], "kind": "fact"}
                ]
                payload = {"what": what, "service": name, "team": team, "since": change.date}
                rels = [(team, name, "owns")]
            else:
                team = self.rng.choice(list(self.teams))
                date = self._team_day(team, lo, hi)
                change = self.oncall[team].at(date)
                if change is None or (date - change.date).days < 10:
                    continue
                nxt = [c for c in self.oncall[team].changes if c.date > date]
                if nxt and (nxt[0].date - date).days < 5:
                    continue
                facts = [
                    {
                        "content": content_oncall(change.value, team),
                        "entities": [change.value, team],
                        "kind": "fact",
                    }
                ]
                payload = {"what": what, "team": team, "person": change.value, "since": change.date}
                rels = [(change.value, team, "on_call_for")]
            assert team is not None
            self._add(
                date=date,
                kind="reminder",
                source_kind="standup",
                team=team,
                payload=payload,
                facts=facts,
                add_relations=rels,
            )
            made += 1

    # ------------------------------------------------------------------ summary

    def summary(self) -> dict:
        """The final state and the histories, JSON-ready, for the README and for inspection."""

        def hist(h: History) -> list[dict]:
            return [
                {
                    "from": s.isoformat(),
                    "to": None if e is None else e.isoformat(),
                    "value": v,
                    "event": eid,
                }
                for s, e, v, eid in h.intervals()
            ]

        return {
            "seed": self.seed,
            "start": START.isoformat(),
            "end": END.isoformat(),
            "now": NOW.isoformat(),
            "people": {p.name: p.full_name for p in PEOPLE},
            "teams": [t.name for t in TEAMS],
            "services": {s.name: {"display": s.display, "forms": list(s.forms)} for s in SERVICES},
            "owner": {s: hist(h) for s, h in self.owner.items()},
            "recorded_owner": {s: hist(h) for s, h in self.recorded_owner.items()},
            "member": {p: hist(h) for p, h in self.member.items()},
            "oncall": {t: hist(h) for t, h in self.oncall.items()},
            "deps": {
                s: [
                    {
                        "dep": d.dep,
                        "from": d.start.isoformat(),
                        "to": None if d.end is None else d.end.isoformat(),
                    }
                    for d in ds
                ]
                for s, ds in self.deps.items()
            },
            "window": {s: hist(h) for s, h in self.window.items()},
            "wrong_records": [
                {
                    **w,
                    "wrong_date": w["wrong_date"].isoformat(),
                    "fix_date": w["fix_date"].isoformat(),
                }
                for w in self.wrong_records
            ],
            "events": len(self.events),
        }
