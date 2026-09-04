"""Value types returned by anatid.

Every row type is a frozen dataclass carrying the full system-column set, so a caller can always
answer "when was this true, when was it recorded, who wrote it, what evidence backs it" without a
second query.

System columns (bitemporal, on every node and edge table by default)
-------------------------------------------------------------------
``valid_from`` / ``valid_to``
    *Valid time*: when the fact was true in the world.  ``valid_to IS NULL`` means "still true".
``tx_from`` / ``tx_to``
    *Transaction time*: when anatid recorded / retired the row.  ``tx_to IS NULL`` means "live".
``writer``
    Identity of the agent, tool or human that wrote the row.
``episode_id``
    The :class:`Episode` (raw source material) this row was derived from -- "evidence before
    belief": the source text is stored first and derived facts point back at it.
``confidence``
    Caller-supplied belief strength in ``[0, 1]``.

Intervals are half-open ``[from, to)``.  A row is visible "as of T" when
``valid_from <= T < valid_to`` (NULL ``valid_to`` = open) and ``tx_from <= T < tx_to``.

``version``
    Rows of ``memories``, ``edges_about`` and ``edges_relates`` are immutable versions of one
    logical id (schema v4).  A correction closes the current version's ``tx_to`` and inserts
    version ``n + 1`` with the corrected valid interval; ``memory_id`` / ``edge_id`` never
    change.  :attr:`Memory.version` says which physical row an object was read from.

Timestamps are naive ``datetime`` objects in UTC, because the underlying DuckDB columns are
``TIMESTAMP`` (no zone).  Use :func:`utcnow` and :func:`to_utc_naive` to stay consistent; every
verb accepts an explicit ``now=`` so a run can be made deterministic.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

__all__ = [
    "utcnow",
    "to_utc_naive",
    "Isolation",
    "Namespace",
    "AsOf",
    "CURRENT",
    "Memory",
    "Entity",
    "Edge",
    "Episode",
    "RecallHit",
    "RecallHits",
    "Provenance",
    "ForgetReceipt",
    "PruneReport",
    "FtsStatus",
    "EdgeType",
    "SchemaInfo",
    "DoctorFinding",
    "DoctorReport",
    "Severity",
    "DOCTOR_SAMPLE_LIMIT",
    "ABOUT",
    "RELATES_TO",
    "SUPERSEDES",
]


# --------------------------------------------------------------------------- time helpers

def utcnow() -> _dt.datetime:
    """Current UTC time as a naive ``datetime`` (what anatid stores in TIMESTAMP columns)."""
    return _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)


def to_utc_naive(value: _dt.datetime | None) -> _dt.datetime | None:
    """Normalise a datetime to naive UTC.  Aware inputs are converted; naive ones pass through."""
    if value is None:
        return None
    if not isinstance(value, _dt.datetime):
        raise TypeError(f"expected datetime, got {type(value).__name__}")
    if value.tzinfo is not None:
        return value.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    return value


def _tuple_or_none(v: Any) -> tuple[float, ...] | None:
    if v is None:
        return None
    return tuple(float(x) for x in v)


# --------------------------------------------------------------------------- isolation

class Isolation(str, Enum):
    """How a :class:`Namespace` is separated from other namespaces.

    ``FILE_PER_TENANT``
        Real isolation: the tenant lives in its own DuckDB file and a handle opened for that
        tenant refuses to touch any other tenant's rows (:class:`anatid.errors.TenantIsolationError`).
        The enforcement is anatid's wrapper plus the filesystem -- DuckDB has no access control
        of its own.
    ``SCOPED``
        NOT isolation.  Several tenants share one file and are separated only by the
        ``tenant_id`` column that anatid puts in every predicate.  Anything with a connection to
        the file can read every tenant in it.  Use this for a single-user or single-trust-domain
        deployment, never as a security boundary between mutually distrusting tenants.
    """

    FILE_PER_TENANT = "file_per_tenant"
    SCOPED = "scoped"


@dataclass(frozen=True, slots=True)
class Namespace:
    """A tenant / namespace identity plus the isolation level anatid will enforce for it.

    ``tenant_id`` is the integer written into every row's ``tenant_id`` column (an INTEGER, so it
    stays cheap in zone maps and joins).  ``label`` is a human name for logs and path templates.
    """

    tenant_id: int
    label: str | None = None
    isolation: Isolation = Isolation.SCOPED

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", int(self.tenant_id))
        object.__setattr__(self, "isolation", Isolation(self.isolation))

    @property
    def is_isolated(self) -> bool:
        """True when this namespace is a real boundary (its own file), not just a column filter."""
        return self.isolation is Isolation.FILE_PER_TENANT

    @property
    def name(self) -> str:
        return self.label if self.label is not None else f"tenant-{self.tenant_id}"

    @classmethod
    def coerce(cls, value: "int | Namespace | None", default: "Namespace | None" = None) -> "Namespace":
        """Accept an int, a Namespace or None (-> ``default``) and return a Namespace."""
        if value is None:
            if default is None:
                raise ValueError("no tenant given and no default namespace available")
            return default
        if isinstance(value, Namespace):
            return value
        if isinstance(value, bool):  # bool is an int; almost certainly a caller mistake
            raise TypeError("tenant must be an int or Namespace, not bool")
        if isinstance(value, int):
            iso = default.isolation if default is not None else Isolation.SCOPED
            return cls(tenant_id=value, isolation=iso)
        raise TypeError(f"tenant must be int | Namespace | None, got {type(value).__name__}")


# --------------------------------------------------------------------------- time travel

@dataclass(frozen=True, slots=True)
class AsOf:
    """A point-in-time scope for reads.

    IMPORTANT: DuckDB has **no** ``AS OF SYSTEM TIME`` clause.  This is anatid's own filter over
    the ``valid_from``/``valid_to`` and ``tx_from``/``tx_to`` columns, compiled into the WHERE
    clause of every query it scopes.  Nothing in the engine rewinds; rows that were hard-purged
    are gone from every as-of view too.

    ``valid_time``
        See the world as it was believed to be true at this instant.
    ``tx_time``
        See only rows the database had already recorded (and not yet retired) at this instant.

    ``AsOf(None, None)`` is :data:`CURRENT`: ``valid_to IS NULL AND tx_to IS NULL``.
    """

    valid_time: _dt.datetime | None = None
    tx_time: _dt.datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "valid_time", to_utc_naive(self.valid_time))
        object.__setattr__(self, "tx_time", to_utc_naive(self.tx_time))

    @property
    def is_current(self) -> bool:
        """True when this scope means "the current state" (no historical filter)."""
        return self.valid_time is None and self.tx_time is None

    @classmethod
    def coerce(cls, value: "AsOf | _dt.datetime | None") -> "AsOf":
        if value is None:
            return CURRENT
        if isinstance(value, AsOf):
            return value
        if isinstance(value, _dt.datetime):
            ts = to_utc_naive(value)
            return cls(valid_time=ts, tx_time=ts)
        raise TypeError(f"as_of must be AsOf | datetime | None, got {type(value).__name__}")


CURRENT = AsOf()
"""The default read scope: current valid time and current transaction time."""


# --------------------------------------------------------------------------- edge types

class EdgeType(str, Enum):
    """The three built-in edge types.  Each is its own table (edge type -> table)."""

    ABOUT = "ABOUT"                # memory -> entity
    RELATES_TO = "RELATES_TO"      # entity -> entity
    SUPERSEDES = "SUPERSEDES"      # newer memory -> older memory


ABOUT = EdgeType.ABOUT
RELATES_TO = EdgeType.RELATES_TO
SUPERSEDES = EdgeType.SUPERSEDES


# --------------------------------------------------------------------------- rows

@dataclass(frozen=True, slots=True)
class Memory:
    """One memory (node label ``memory`` -> table ``memories``).

    ``embedding`` is ``None`` when the row was read without hydrating the vector column.
    ``version`` is the physical row this object was read from: 1 for a memory that has never
    been corrected, higher after each ``supersede`` / soft ``forget`` / confidence change.
    ``memory_id`` is the same across every version.
    """

    memory_id: int
    tenant_id: int
    content: str
    kind: str | None = None
    embedding: tuple[float, ...] | None = None
    created_at: _dt.datetime | None = None
    valid_from: _dt.datetime | None = None
    valid_to: _dt.datetime | None = None
    tx_from: _dt.datetime | None = None
    tx_to: _dt.datetime | None = None
    writer: str | None = None
    episode_id: int | None = None
    confidence: float | None = None
    access_count: int = 0
    last_access_at: _dt.datetime | None = None
    version: int = 1

    @property
    def is_current(self) -> bool:
        """True when the row is neither superseded/forgotten (valid) nor retired (tx)."""
        return self.valid_to is None and self.tx_to is None

    @property
    def is_live(self) -> bool:
        """True when no correction has closed this version (``tx_to`` is ``NULL``).

        A superseded memory's live version is not current: it carries the ``valid_to`` the
        correction gave it.  ``get()`` by id returns the live version.
        """
        return self.tx_to is None

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> "Memory":
        """Build from a row selected with :data:`anatid.schema.MEMORY_COLUMNS` order.

        A 15-column row (the v3 order, without ``version``) is accepted and read as version 1,
        so a caller that selects the old column list keeps working.
        """
        return cls(
            memory_id=int(row[0]),
            tenant_id=int(row[1]),
            content=row[2],
            kind=row[3],
            embedding=_tuple_or_none(row[4]),
            created_at=row[5],
            valid_from=row[6],
            valid_to=row[7],
            tx_from=row[8],
            tx_to=row[9],
            writer=row[10],
            episode_id=None if row[11] is None else int(row[11]),
            confidence=None if row[12] is None else float(row[12]),
            access_count=0 if row[13] is None else int(row[13]),
            last_access_at=row[14],
            version=1 if len(row) < 16 or row[15] is None else int(row[15]),
        )


@dataclass(frozen=True, slots=True)
class Entity:
    """One entity (node label ``entity`` -> table ``entities``)."""

    entity_id: int
    tenant_id: int
    name: str
    kind: str | None = None
    valid_from: _dt.datetime | None = None
    valid_to: _dt.datetime | None = None
    tx_from: _dt.datetime | None = None
    tx_to: _dt.datetime | None = None
    writer: str | None = None
    episode_id: int | None = None
    confidence: float | None = None

    @property
    def is_current(self) -> bool:
        return self.valid_to is None and self.tx_to is None

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> "Entity":
        """Build from a row selected with :data:`anatid.schema.ENTITY_COLUMNS` order."""
        return cls(
            entity_id=int(row[0]),
            tenant_id=int(row[1]),
            kind=row[2],
            name=row[3],
            valid_from=row[4],
            valid_to=row[5],
            tx_from=row[6],
            tx_to=row[7],
            writer=row[8],
            episode_id=None if row[9] is None else int(row[9]),
            confidence=None if row[10] is None else float(row[10]),
        )


@dataclass(frozen=True, slots=True)
class Edge:
    """One edge from any of the edge tables.

    ``weight`` is only meaningful for ABOUT edges and ``rel_kind`` only for RELATES_TO edges;
    the other is ``None``.  SUPERSEDES edges carry only ``tx_from`` (they are a record of a
    write, never re-dated) and are always version 1.  ABOUT and RELATES_TO edges are versioned
    like memories: closing one inserts version ``n + 1`` with ``valid_to`` set.
    """

    edge_id: int
    edge_type: EdgeType
    src: int
    dst: int
    tenant_id: int
    weight: float | None = None
    rel_kind: str | None = None
    valid_from: _dt.datetime | None = None
    valid_to: _dt.datetime | None = None
    tx_from: _dt.datetime | None = None
    tx_to: _dt.datetime | None = None
    writer: str | None = None
    episode_id: int | None = None
    confidence: float | None = None
    version: int = 1

    @property
    def is_current(self) -> bool:
        return self.valid_to is None and self.tx_to is None


@dataclass(frozen=True, slots=True)
class Episode:
    """Raw source material a memory was derived from ("evidence before belief").

    An episode is written BEFORE the facts extracted from it, and every derived row carries its
    ``episode_id``, so :meth:`anatid.Anatid.provenance` can always walk back to the text a belief
    came from and the writer that produced it.
    """

    episode_id: int
    tenant_id: int
    content: str
    source: str | None = None
    kind: str | None = None
    created_at: _dt.datetime | None = None
    valid_from: _dt.datetime | None = None
    valid_to: _dt.datetime | None = None
    tx_from: _dt.datetime | None = None
    tx_to: _dt.datetime | None = None
    writer: str | None = None

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> "Episode":
        """Build from a row selected with :data:`anatid.schema.EPISODE_COLUMNS` order."""
        return cls(
            episode_id=int(row[0]),
            tenant_id=int(row[1]),
            source=row[2],
            content=row[3],
            kind=row[4],
            created_at=row[5],
            valid_from=row[6],
            valid_to=row[7],
            tx_from=row[8],
            tx_to=row[9],
            writer=row[10],
        )


# --------------------------------------------------------------------------- recall results

@dataclass(frozen=True, slots=True)
class RecallHit:
    """One fused retrieval result.

    ``score`` is the RRF score (sum of ``1 / (rrf_k + rank)`` over the lists that produced it).
    ``vector_rank`` / ``text_rank`` / ``graph_rank`` are 1-based ranks in the contributing lists,
    or ``None`` when that list did not return the memory.
    """

    memory: Memory
    score: float
    rank: int
    vector_rank: int | None = None
    text_rank: int | None = None
    graph_rank: int | None = None
    vector_score: float | None = None
    text_score: float | None = None
    about: tuple[str, ...] = ()

    @property
    def memory_id(self) -> int:
        return self.memory.memory_id

    @property
    def content(self) -> str:
        return self.memory.content

    @property
    def sources(self) -> tuple[str, ...]:
        """Which retrieval arms produced this hit: any of ``"vector"``, ``"text"``, ``"graph"``."""
        out = []
        if self.vector_rank is not None:
            out.append("vector")
        if self.text_rank is not None:
            out.append("text")
        if self.graph_rank is not None:
            out.append("graph")
        return tuple(out)


class RecallHits(list):
    """``list[RecallHit]`` that also carries how the search was answered.

    It *is* a plain list, so callers can ignore the extra attributes entirely.  They exist so
    :meth:`anatid.Anatid.recall` can state -- rather than hide -- how the BM25 arm answered:

    ``bm25_available``
        False when this database has no full-text index at all.
    ``bm25_stale``
        True when the BM25 arm could not answer exactly.  On the derived text index that
        :meth:`anatid.Anatid.open` attaches by default a write is searchable by the very next
        recall with nothing rebuilt, so this means no generation was usable AND the tenant's
        corpus is above :data:`anatid.fts.SCAN_CEILING`, which made the exact fallback scan too
        expensive to run.  On 0.1.1's file-wide index (``accelerators=False``) it keeps its old
        meaning: rows have been written since the last ``PRAGMA create_fts_index`` and the arm
        cannot find them until :meth:`anatid.Anatid.rebuild_fts_index` runs.
    ``pending_fts_rows``
        On the derived index, how many documents the search re-read from ``memories`` because
        the journal had touched them since the last build.  They were searched.  On 0.1.1's
        index, how many rows are missing from it (``count(*) - fts_indexed_rows``), which are
        the rows it cannot see.
    ``arms``
        The arms that actually ran, e.g. ``("vector", "text", "graph")``.
    """

    __slots__ = ("bm25_available", "bm25_stale", "pending_fts_rows", "arms", "as_of", "notes")

    def __init__(
        self,
        hits: Iterable[RecallHit] = (),
        *,
        bm25_available: bool = False,
        bm25_stale: bool = False,
        pending_fts_rows: int = 0,
        arms: tuple[str, ...] = (),
        as_of: AsOf = CURRENT,
        notes: tuple[str, ...] = (),
    ) -> None:
        super().__init__(hits)
        self.bm25_available = bm25_available
        self.bm25_stale = bm25_stale
        self.pending_fts_rows = pending_fts_rows
        self.arms = arms
        self.as_of = as_of
        self.notes = notes

    @property
    def memory_ids(self) -> list[int]:
        return [h.memory.memory_id for h in self]


# --------------------------------------------------------------------------- provenance

@dataclass(frozen=True, slots=True)
class Provenance:
    """The evidence trail behind one memory.

    ``chain`` is the SUPERSEDES chain newest-first: ``chain[0]`` is the memory asked about and
    ``chain[-1]`` is the original assertion nothing supersedes.  ``episodes`` holds the source
    material for the chain in the same order (an entry is absent when a link has no episode).
    ``writers`` is every distinct writer in the chain, newest-first.

    ``versions`` is the other axis: every physical version of the memory asked about, oldest
    first (``versions[0].version == 1``, ``versions[-1]`` is the live one).  The chain links
    logical ids; the versions are the corrections one logical id went through, each with the
    transaction interval during which the database believed it.
    """

    memory_id: int
    chain: tuple[Memory, ...] = ()
    episodes: tuple[Episode, ...] = ()
    edges: tuple[Edge, ...] = ()
    writers: tuple[str, ...] = ()
    versions: tuple[Memory, ...] = ()

    @property
    def root(self) -> Memory | None:
        """The oldest memory in the chain -- the original assertion."""
        return self.chain[-1] if self.chain else None

    @property
    def source_text(self) -> str | None:
        """Raw text of the oldest episode in the chain, if any."""
        return self.episodes[-1].content if self.episodes else None

    @property
    def depth(self) -> int:
        """How many supersessions deep the memory is (0 = never superseded anything)."""
        return max(0, len(self.chain) - 1)


# --------------------------------------------------------------------------- write receipts

@dataclass(frozen=True, slots=True)
class ForgetReceipt:
    """What :meth:`anatid.Anatid.forget` actually did.

    A *hard* purge leaves no row **in the memory graph** referencing the memory -- not in
    ``memories``, not in any edge table, not the embedding (it is a column of the purged row),
    not the episode when no other memory cites it (entities and edges that carried its id as a
    provenance stamp keep existing with the stamp cleared), not the BM25 watermark in
    ``anatid_meta``, not in the derived indexes (every generation's storage and the change
    journal, see :attr:`derived_rows_deleted`), and not in ``anatid_audit`` (neither as
    ``memory_id`` nor as the ``related_memory_id`` of some other memory's supersede).  That is
    the point of erasure, so the receipt is returned to the caller to log outside the database
    if they need a record.

    Beyond the memory graph it reaches the tables anatid's bundled integrations create -- the
    Agents SDK transcript, session, usage and run-state tables, which quote memory content and
    ids verbatim -- by name from the catalog, on every handle
    (:data:`anatid.erasure.BUNDLED_INTEGRATION_TABLES`).  Anything else you write into the file
    is covered only if you register it: see :meth:`anatid.Anatid.register_erasure_hook`.  Rows
    removed from integration tables, by hook or by the bundled-table pass, are counted in
    :attr:`extra_rows_deleted`.
    """

    memory_id: int
    tenant_id: int
    hard: bool
    at: _dt.datetime
    #: Logical memories removed: 1 when the memory existed, 0 otherwise.  Every version of it
    #: goes; :attr:`memory_versions_deleted` counts the physical rows.
    memories_deleted: int = 0
    #: Distinct ABOUT edges removed; :attr:`about_edge_versions_deleted` counts their rows.
    about_edges_deleted: int = 0
    supersedes_edges_deleted: int = 0
    episodes_deleted: int = 0
    audit_rows_deleted: int = 0
    audit_rows_written: int = 0
    #: Rows removed from the BM25 index tables (schema v3).  ``anatid_fts_documents`` holds the
    #: memory's ``content`` **verbatim**, so a purge that skipped it would leave the erased text
    #: in the file; :func:`anatid.schema.fts_purge` is part of the purge transaction.
    fts_rows_deleted: int = 0
    #: Rows removed by :meth:`anatid.Anatid.register_erasure_hook` hooks (transcripts, etc).
    extra_rows_deleted: int = 0
    #: Physical rows removed from ``memories`` and ``edges_about``: every version of the memory
    #: and of each of its ABOUT edges (schema v4).  At least the logical counts above.
    memory_versions_deleted: int = 0
    about_edge_versions_deleted: int = 0
    #: Rows removed from the derived indexes (schema v4): each generation's storage and the
    #: change journal.  An accelerator's built generation holds a copy of what it indexed, so
    #: an erasure that stopped at the canonical tables would leave the document findable
    #: through the index.
    derived_rows_deleted: int = 0
    #: Generations this purge could not clean, because their storage cannot delete one document
    #: (an index that only rebuilds) or because the erasure came from a handle holding no code
    #: for that index.  They are taken out of service, report ``stale_generation`` until a
    #: rebuild, and reads fall back to the SQL path, which is correct but slower.  A non-zero
    #: count means the document is still inside that generation's storage until it is rebuilt
    #: or its storage is dropped.
    invalidated_generations: int = 0
    reason: str | None = None

    @property
    def rows_removed(self) -> int:
        """Every physical row the purge removed, versions included."""
        return (
            max(self.memories_deleted, self.memory_versions_deleted)
            + max(self.about_edges_deleted, self.about_edge_versions_deleted)
            + self.supersedes_edges_deleted
            + self.episodes_deleted
            + self.audit_rows_deleted
            + self.fts_rows_deleted
            + self.extra_rows_deleted
            + self.derived_rows_deleted
        )


@dataclass(frozen=True, slots=True)
class PruneReport:
    """What :meth:`anatid.Anatid.prune` did, or would do when ``dry_run=True``."""

    dry_run: bool
    hard: bool
    at: _dt.datetime
    memory_ids: tuple[int, ...] = ()
    receipts: tuple[ForgetReceipt, ...] = ()
    older_than: _dt.datetime | None = None
    max_access_count: int | None = None

    @property
    def count(self) -> int:
        return len(self.memory_ids)


@dataclass(frozen=True, slots=True)
class FtsStatus:
    """How the text arm on ``memories`` will answer, on either half of the library.

    On the derived text index that :meth:`anatid.Anatid.open` attaches by default, a write is
    searchable by the very next :meth:`anatid.Anatid.recall` with nothing rebuilt, ``stale``
    means the answer would not be exact rather than that rows are missing, and ``pending_rows``
    counts documents a search rescans from the canonical rows, all of which it searched.  On
    0.1.1's file-wide index (``accelerators=False``) the index is not incremental, ``stale``
    means rows have been written since the last build, and ``pending_rows`` is how many rows the
    index cannot see.  ``policy`` says which half produced this status and is the one field to
    read when a caller has to tell them apart.
    """

    available: bool
    stale: bool
    indexed_rows: int | None
    current_rows: int
    pending_rows: int
    indexed_at: _dt.datetime | None
    newest_row_at: _dt.datetime | None = None
    policy: str = ""
    #: Largest ``memory_id`` present when the index was last built (schema v2).  Row *count*
    #: alone cannot see an insert that is cancelled out by a hard purge; ids are time-ordered,
    #: so this watermark can.
    indexed_max_id: int | None = None
    #: Largest ``memory_id`` in ``memories`` now.
    current_max_id: int | None = None


# --------------------------------------------------------------------------- misc

@dataclass(frozen=True, slots=True)
class SchemaInfo:
    """Contents of the ``anatid_meta`` catalog row."""

    schema_version: int
    created_at: _dt.datetime
    embedding_dim: int
    anatid_version: str
    duckdb_version: str
    fts_indexed_at: _dt.datetime | None
    fts_indexed_rows: int | None
    contract: str
    extras: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- integrity report

class Severity(str, Enum):
    """How bad a :class:`DoctorFinding` is.

    ``ERROR``
        The database contradicts something anatid's verbs assume.  Reads can already be wrong:
        a duplicate ``(tenant_id, memory_id)`` makes ``get()`` return an arbitrary row, a
        dangling edge makes ``recall_2hop`` skip a hop, a NaN embedding poisons every cosine
        comparison against it.
    ``WARNING``
        A fault that costs correctness of *ranking* or freshness, not of content: a stale BM25
        index, a missing performance index, per-tenant fts statistics that have drifted.
    """

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class DoctorFinding:
    """One fault :meth:`anatid.Anatid.doctor` found.

    ``check``
        Stable machine-readable name (``"duplicate_memory_ids"``, ``"dangling_edges"``, ...).
        Never localised, never reworded: it is what a caller matches on.
    ``count``
        How many rows are affected.  Always ``>= 1`` -- a check with nothing to report produces
        no finding at all.
    ``samples``
        Up to :data:`DOCTOR_SAMPLE_LIMIT` example rows, each a tuple, for the operator to chase.
        Ids only; ``doctor()`` never copies memory *content* into its report.
    ``detail``
        One sentence naming the fault and its consequence.  For humans.
    """

    check: str
    severity: Severity
    count: int
    detail: str
    table: str | None = None
    samples: tuple[tuple[Any, ...], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", Severity(self.severity))

    @property
    def is_error(self) -> bool:
        return self.severity is Severity.ERROR

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready mapping (the MCP/agent surface serialises this)."""
        return {
            "check": self.check,
            "severity": self.severity.value,
            "count": self.count,
            "detail": self.detail,
            "table": self.table,
            "samples": [list(s) for s in self.samples],
        }


#: How many example rows :class:`DoctorFinding` carries per check.
DOCTOR_SAMPLE_LIMIT = 10


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """The structured result of :meth:`anatid.Anatid.doctor` -- a health check, not a repair.

    ``doctor()`` reads; it never writes.  Nothing here is a promise that the database *was*
    consistent at any single instant either: the checks are separate statements, so a concurrent
    writer can land between two of them.  Wrap the call in ``with db.transaction():`` for one
    snapshot.

    ``ok`` is True exactly when no finding has ``severity == ERROR``; warnings do not clear it
    on their own -- read :attr:`clean` for "nothing at all to report".
    """

    checked_at: _dt.datetime
    schema_version: int | None
    expected_schema_version: int
    tenant_id: int | None
    all_tenants: bool = False
    findings: tuple[DoctorFinding, ...] = ()
    #: Row counts per table, for the tenants the report covers.
    counts: dict[str, int] = field(default_factory=dict)
    #: Checks that ran and found nothing.  Present so a caller can tell "clean" from "skipped".
    checks_run: tuple[str, ...] = ()
    #: Checks that could not run, and why (e.g. no BM25 index built yet).
    checks_skipped: dict[str, str] = field(default_factory=dict)
    duration_ms: float | None = None

    @property
    def errors(self) -> tuple[DoctorFinding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[DoctorFinding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.WARNING)

    @property
    def ok(self) -> bool:
        """True when nothing of ``severity == ERROR`` was found."""
        return not self.errors

    @property
    def clean(self) -> bool:
        """True when there is no finding at all, not even a warning."""
        return not self.findings

    def find(self, check: str) -> DoctorFinding | None:
        """The finding for ``check``, or None when that check passed (or did not run)."""
        for f in self.findings:
            if f.check == check:
                return f
        return None

    def __bool__(self) -> bool:      # `if db.doctor():` reads as "is it healthy"
        return self.ok

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready mapping.  This is the machine-readable form; ``str()`` is the human one."""
        return {
            "ok": self.ok,
            "clean": self.clean,
            "checked_at": self.checked_at.isoformat(),
            "schema_version": self.schema_version,
            "expected_schema_version": self.expected_schema_version,
            "tenant_id": self.tenant_id,
            "all_tenants": self.all_tenants,
            "counts": dict(self.counts),
            "checks_run": list(self.checks_run),
            "checks_skipped": dict(self.checks_skipped),
            "duration_ms": self.duration_ms,
            "findings": [f.as_dict() for f in self.findings],
        }

    def __str__(self) -> str:
        scope = "all tenants" if self.all_tenants else f"tenant {self.tenant_id}"
        head = (f"doctor({scope}) schema v{self.schema_version}: "
                f"{'ok' if self.ok else 'FAULTS'} "
                f"({len(self.errors)} error(s), {len(self.warnings)} warning(s), "
                f"{len(self.checks_run)} check(s) run)")
        lines = [head]
        for f in self.findings:
            lines.append(f"  [{f.severity.value}] {f.check}: {f.count} -- {f.detail}")
        return "\n".join(lines)
