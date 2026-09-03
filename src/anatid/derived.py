"""Derived indexes: base generations, an ordered change journal, atomic publication.

This module implements ``docs/design/derived-index-framework.md`` generically, so the full-text
index, the CSR adjacency structure and a vector index all sit on one mechanism instead of three
staleness stories.  Nothing here knows how to build any particular index; it knows how a
derived index is versioned, how it stays complete between rebuilds, and how a read finds out
whether it may use one.

The shape
---------
A derived index is:

* a **definition**: one row of :data:`~anatid.schema.INDEX_REGISTRY_TABLE` saying that this
  file has an index of this name over this source table.  The definition lives in the FILE, not
  on the handle that created it, which is what makes the journal complete: a second handle
  writing into the same file journals for every enabled definition whether or not it holds that
  accelerator's code (:class:`IndexRegistry`);
* a **base generation**: storage the index built from a snapshot of its source table, recorded
  as one row of :data:`~anatid.schema.INDEX_GENERATIONS_TABLE` with the snapshot's watermark,
  a ``validated`` flag and a ``published`` flag;
* a **journal**: the ordered record of what happened to the source rows since that snapshot.
  The verbs append one row of :data:`~anatid.schema.INDEX_JOURNAL_TABLE` per change **inside
  the writing transaction**, so a read in the next statement already sees it.

Every journal row is keyed by ``(index_name, tenant_id, doc_id)`` and stamped with a
``change_seq`` from :data:`~anatid.schema.INDEX_CHANGE_SEQUENCE`; the row with the largest
``change_seq`` for a key is that document's current state.  ``op = "insert"`` makes it pending,
anything else makes it a tombstone.  One ordered journal rather than a delta set and a
tombstone set, because two independent id sets cannot represent a document id that was purged
and then reused: the id would be in both sets and ``(base | delta) - tombstones`` would drop a
row the SQL path returns.  The tenant is part of the key even for an index whose generations
cover the whole file, because ``memory_id`` is unique only within a tenant: without it,
tombstoning tenant 1's document 42 would suppress tenant 2's document 42.

A read pins the published generation (:meth:`DerivedIndex.pin`), takes candidates from the
base, adds :meth:`DerivedIndex.pending`, subtracts :meth:`DerivedIndex.tombstones`, and only
then applies the tenant and time predicate from :mod:`anatid.visibility` to the canonical rows.
The predicate runs after candidate generation because an index over current state cannot
answer what was visible last Tuesday; historical reads therefore never use an index at all and
:class:`HealthReason.HISTORICAL_QUERY` says so.

Erasure
-------
``forget(hard=True)`` is destructive to the accelerators too.  :meth:`DerivedIndex.erase`
deletes the document from **every** generation's storage through :meth:`DerivedIndex._erase`,
deletes its journal rows, clamps any generation watermark that IS the erased id
(:meth:`DerivedIndex._clamp_watermarks`), and invalidates any generation whose storage cannot
delete (the default, so an accelerator that has not implemented ``_erase`` fails closed into
the SQL path rather than keeping the erased text in a built index).  The counts reach the
caller on :class:`~anatid.types.ForgetReceipt`.

Erasure follows the STORAGE, not the journal: a definition that has been disabled
(:meth:`IndexRegistry.unregister`) still has generations holding whatever the index put in
them, so it is erased and invalidated like any other.  What stops when a definition is disabled
is journalling, and because a write in that window reaches no delta, disabling and re-enabling
both take every generation of that index out of service.

A build in flight cannot be reached by any of that: its storage is inside a transaction the
purge cannot see, and the purge's own snapshot goes on calling it ``building`` until the purge
commits.  The hand-off is a process-wide counter, bumped by
:meth:`IndexRegistry.announce_erasure` BEFORE the purge opens its transaction and compared by
:meth:`DerivedIndex.build_next` around its snapshot.  Bumping it later would let a build finish,
find the counter unchanged, compare itself with canonical rows the purge had not deleted yet,
and publish a base holding the erased document.  :meth:`DerivedIndex.validate` refuses a
generation whose note says it was invalidated for the same reason: the oracle can agree with an
invalidated base, because what makes it wrong has not committed yet.

Lifecycle locking
-----------------
Generations are created, pinned, retired and dropped from several threads and several handles
on one file.  :func:`lifecycle_lock` is one re-entrant lock per canonical database path, held
across build ownership, pin acquisition, retirement and storage removal, so a generation cannot
be dropped between a reader choosing it and the reader registering its pin.  It is NOT held
while a build runs or while a pinned read runs: only the transitions are serialised.

Why this is exact under DuckDB's MVCC
-------------------------------------
The canonical row and its journal row are written in one transaction, so a build's snapshot
sees both or neither.  :meth:`DerivedIndex.build_next` stamps every journal row visible in its
snapshot with ``absorbed_by = <new generation>`` inside that same transaction; the base it built
contains exactly those rows.  A reader pinned to generation N takes the rows with
``absorbed_by IS NULL OR absorbed_by > N`` as its journal, whichever builds are running, and
:meth:`DerivedIndex.publish` deletes the rows a generation ``<= N`` absorbed only when N goes
live.  No timestamp comparison is involved, so a writer that read the clock before a build
started and committed after it cannot slip through.  A base generation can therefore be stale,
half-built, corrupt or absent without a read being wrong: the journal covers what the base does
not, and the canonical SQL path is always available and always the oracle.

Delta modes
-----------
``delta_mode = "table"`` (the default) journals every insert on the source table.  It is exact
for any id.  ``delta_mode = "watermark"`` journals nothing for an insert whose id is above the
published generation's ``watermark_id`` and treats ``id > watermark_id`` on the source table as
the delta, which is what anatid's time-ordered ids make cheap; an insert with an explicit id at
or below the watermark is journalled so it is not lost.  The watermark mode reads the catalog
through a per-process cache that is invalidated by every publish in this process, which is the
only process that can write a DuckDB file.

Publication and pinning
-----------------------
:meth:`DerivedIndex.publish` flips the ``published`` flag of one (index, tenant) with one
``UPDATE`` statement inside a transaction: the old generation goes off and the new one on
atomically, and a reader that already pinned the old one keeps using it.  Pins are counted in
process; :meth:`DerivedIndex.retire` refuses to drop a pinned generation, and the next
:meth:`DerivedIndex.build_next` retires whatever is neither published nor pinned.

Maintenance
-----------
:class:`MaintenancePolicy` says when a generation is due for a rebuild; :func:`maintain`
decides and acts: build beside the current generation, validate against the oracle, publish.
It is explicitly callable and runs no thread of its own.

What it costs
-------------
Measured on this build (duckdb 1.5.5, macOS arm64), so an accelerator author can decide where
to spend.  On the WRITE path, :meth:`IndexRegistry.emit` for a table no index in the file
derives from is 0.42 us: one dict lookup against the per-epoch definition cache, which is why
a database with no accelerators pays nothing.  Journalling one document costs 706 us, of which
586 us is the bare ``INSERT`` and about 120 us is ``nextval``; both are dominated by DuckDB's
fixed per-statement cost, and one statement covers however many documents an event carries.

On the READ path, resolving the newest op per document costs 1.5 ms over 1,000 journal rows,
2.9 ms over 10,000 and 4.4 ms over 20,000, against 0.4 / 1.2 / 2.0 ms for the same scan without
the ordering.  That factor of roughly 2.2 is the price of representing a purged and reused id,
and :class:`MaintenancePolicy`'s ``rebuild_after_rows = 10_000`` is what keeps the journal in
the range where it is a few milliseconds: a build absorbs the rows and a publish prunes them.

What an accelerator implements
------------------------------
Subclass :class:`DerivedIndex`, set ``name``, ``source_table`` and ``source_id_column``, and
implement :meth:`~DerivedIndex._build`, :meth:`~DerivedIndex._validate` and
:meth:`~DerivedIndex._drop`.  Implement :meth:`~DerivedIndex._erase` too unless the storage
genuinely cannot delete one document.  Register the instance with
:attr:`anatid.Anatid.indexes`, which writes the definition into the file; the verbs then
journal every write on the source table, on every handle.  The read side is the accelerator's
own (it knows its storage); it pins a generation, merges, and applies
:class:`~anatid.visibility.Visibility`.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar, Iterator, Sequence

from .errors import IndexGenerationError, IndexValidationError
from .schema import (
    INDEX_CHANGE_SEQUENCE,
    INDEX_GENERATIONS_TABLE,
    INDEX_JOURNAL_TABLE,
    INDEX_REGISTRY_TABLE,
    quote_ident,
)
from .types import AsOf, Namespace, to_utc_naive, utcnow
from .visibility import Visibility

log = logging.getLogger("anatid.derived")

__all__ = [
    "HealthReason",
    "Watermark",
    "Generation",
    "ValidationReport",
    "HealthReport",
    "Pin",
    "MaintenancePolicy",
    "MaintenanceReport",
    "IndexEvent",
    "IndexDefinition",
    "ErasureResult",
    "INSERT",
    "CLOSE",
    "PURGE",
    "DerivedIndex",
    "NullIndex",
    "JournalOnlyIndex",
    "IndexRegistry",
    "database_key",
    "lifecycle_lock",
    "maintain",
]

#: Event kinds the verbs emit.  ``INSERT``: a new source row.  ``CLOSE``: a row whose valid
#: interval was closed (supersede, soft forget).  ``PURGE``: a row deleted outright, which is
#: an erasure and not a journal entry (see :meth:`DerivedIndex.erase`).
INSERT = "insert"
CLOSE = "close"
PURGE = "purge"

#: The journal ops.  ``INSERT`` makes a document pending; anything else makes it a tombstone.
JOURNAL_OPS: tuple[str, ...] = (INSERT, CLOSE, PURGE)

_GEN = INDEX_GENERATIONS_TABLE
_JOURNAL = INDEX_JOURNAL_TABLE
_REGISTRY = INDEX_REGISTRY_TABLE
_SEQ = quote_ident(INDEX_CHANGE_SEQUENCE)


# --------------------------------------------------------------------------- value types


class HealthReason(str, Enum):
    """Why a read may or may not use a derived index.  Machine-readable; match on it.

    ``FRESH``
        A validated generation is published and its delta is within policy.
    ``STALE_GENERATION``
        A generation is published but is due for a rebuild under the policy, or was
        invalidated (a bulk load bypassed the journal, or a hard erasure could not reach the
        generation's storage).  Usable when the index merges deltas and the generation is still
        validated; not usable otherwise.
    ``UNVALIDATED``
        A generation was published with ``force=True`` (``MaintenancePolicy(validate=False)``),
        so it was never compared with the oracle.  Usable and flagged: candidate generation is
        only ever a narrowing, the canonical rows decide the answer, and refusing such a
        generation would leave the mode able to publish generations nothing could read.
    ``HISTORICAL_QUERY``
        The read carries an ``as_of``.  A current-state index cannot answer it; the SQL path
        does.
    ``REBUILD_IN_PROGRESS``
        No generation is published and one is being built.
    ``LOAD_FAILURE``
        The index's storage or extension could not be loaded (``DerivedIndex.load_error``), or
        a statement against a generation's storage raised.
    ``DAMAGED_BASE``
        A generation's storage is present and queryable but no longer holds what its build
        recorded, so the candidates it offers are not the candidates it was validated on.  This
        is the reason a read reports when a cheap invariant fails: the accelerator is skipped
        and the oracle answers.  ``LOAD_FAILURE`` covers damage that raises; this covers damage
        that does not.  Every derived index checks at least one such invariant on every read,
        because the design's promise is that a corrupt index does not make a query WRONG, and
        an index that quietly returns half its candidates would.  The invariants are cheap and
        therefore partial: :meth:`DerivedIndex.validate` is the deep comparison with the oracle
        and is what a rebuild runs.
    ``ABSENT``
        No generation has ever been published for this tenant.
    """

    FRESH = "fresh"
    STALE_GENERATION = "stale_generation"
    UNVALIDATED = "unvalidated"
    HISTORICAL_QUERY = "historical_query"
    REBUILD_IN_PROGRESS = "rebuild_in_progress"
    LOAD_FAILURE = "load_failure"
    DAMAGED_BASE = "damaged_base"
    ABSENT = "absent"


@dataclass(frozen=True, slots=True)
class Watermark:
    """Where a generation's snapshot of its source table ended.

    ``id`` is the largest source id in the snapshot and ``ts`` the largest ``tx_from``.  Both
    are ``None`` for an empty source or an index with no source table.
    """

    id: int | None = None
    ts: _dt.datetime | None = None


@dataclass(frozen=True, slots=True)
class Generation:
    """One row of :data:`~anatid.schema.INDEX_GENERATIONS_TABLE`."""

    index_name: str
    generation: int
    tenant_id: int | None
    built_at: _dt.datetime
    watermark_id: int | None = None
    watermark_ts: _dt.datetime | None = None
    validated: bool = False
    published: bool = False
    #: Set by :meth:`DerivedIndex.publish` with ``force=True``: this generation went live
    #: without being compared with the oracle.  Reads may use it; :class:`HealthReason` reports
    #: ``UNVALIDATED`` so nobody mistakes it for a validated one.  Cleared by a successful
    #: :meth:`DerivedIndex.validate` and by :meth:`IndexRegistry.invalidate`.
    published_unvalidated: bool = False
    stats: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    @property
    def usable_without_validation(self) -> bool:
        """True when a read may use this generation although ``validated`` is False.

        Exactly the force-published case.  An INVALIDATED generation also has
        ``validated = False``, and must not be read: whatever bypassed the journal is not in
        the base and is not in the journal either.
        """
        return bool(self.published_unvalidated) and not self.validated

    @property
    def key(self) -> tuple[str, int | None, int]:
        return (self.index_name, self.tenant_id, self.generation)

    @property
    def watermark(self) -> Watermark:
        return Watermark(id=self.watermark_id, ts=self.watermark_ts)

    @property
    def building(self) -> bool:
        """True while :meth:`DerivedIndex.build_next` has announced but not finished it."""
        return self.notes == "building"

    @property
    def storage_suffix(self) -> str:
        """A bare identifier fragment unique to this generation: ``<index>_t<tenant>_g<n>``.

        Accelerators append it to a table or schema name so a generation is built beside the
        current one.  Passes :func:`anatid.schema.quote_ident` when the index name does.

        A NEGATIVE tenant id renders its sign as ``n`` (``_tn7``) rather than as ``-``, which is
        not an identifier character: nothing in anatid mints a negative tenant id, but
        :class:`~anatid.types.Namespace` accepts one, and a name that fails to quote would fail
        the build rather than the input.  ``n`` cannot collide with a positive id, because a
        positive one renders digits only.
        """
        if self.tenant_id is None:
            tenant = ""
        elif int(self.tenant_id) < 0:
            tenant = f"_tn{abs(int(self.tenant_id))}"
        else:
            tenant = f"_t{int(self.tenant_id)}"
        return f"{self.index_name}{tenant}_g{self.generation}"

    def storage_name(self, prefix: str = "anatid_idx") -> str:
        """``<prefix>_<storage_suffix>``, validated as an identifier."""
        name = f"{prefix}_{self.storage_suffix}"
        quote_ident(name)
        return name

    def as_dict(self) -> dict[str, Any]:
        return {
            "index_name": self.index_name,
            "generation": self.generation,
            "tenant_id": self.tenant_id,
            "built_at": self.built_at.isoformat() if self.built_at else None,
            "watermark_id": self.watermark_id,
            "watermark_ts": self.watermark_ts.isoformat() if self.watermark_ts else None,
            "validated": self.validated,
            "published": self.published,
            "published_unvalidated": self.published_unvalidated,
            "stats": dict(self.stats),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class IndexDefinition:
    """One row of :data:`~anatid.schema.INDEX_REGISTRY_TABLE`: an index this FILE has.

    Written when an index is registered and read by every handle that opens the file, which is
    what lets a handle holding no accelerator code journal that accelerator's writes correctly.
    ``source_table`` and ``delta_mode`` are therefore part of the definition rather than of the
    Python object: they are exactly what such a handle needs in order to journal.
    """

    index_name: str
    kind: str = ""
    per_tenant: bool = True
    source_table: str | None = None
    source_id_column: str = "memory_id"
    delta_mode: str = "table"
    supports_delta: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    created_at: _dt.datetime | None = None
    enabled: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "index_name": self.index_name,
            "kind": self.kind,
            "per_tenant": self.per_tenant,
            "source_table": self.source_table,
            "source_id_column": self.source_id_column,
            "delta_mode": self.delta_mode,
            "supports_delta": self.supports_delta,
            "params": dict(self.params),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "enabled": self.enabled,
        }


@dataclass(frozen=True, slots=True)
class ErasureResult:
    """What :meth:`DerivedIndex.erase` removed from the derived side of a hard purge.

    ``rows_deleted`` counts rows removed from generation storage and from the journal;
    ``generations_invalidated`` counts generations whose storage could not delete the document
    and were therefore marked unusable until they are rebuilt.  A generation that was already
    out of service counts too: it still holds the document, so the number is what the purge
    failed to clean rather than what it changed.
    """

    rows_deleted: int = 0
    generations_invalidated: int = 0

    def __add__(self, other: "ErasureResult") -> "ErasureResult":
        return ErasureResult(
            self.rows_deleted + other.rows_deleted,
            self.generations_invalidated + other.generations_invalidated,
        )

    def __bool__(self) -> bool:
        return bool(self.rows_deleted or self.generations_invalidated)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """What :meth:`DerivedIndex.validate` found when it compared a generation with the oracle."""

    ok: bool
    generation: Generation
    checked: int = 0
    mismatches: tuple[Any, ...] = ()
    detail: str = ""

    def __bool__(self) -> bool:
        return self.ok


@dataclass(frozen=True, slots=True)
class HealthReport:
    """The state of one derived index for one tenant, and whether a read may use it.

    ``usable`` is the decision; ``reason`` is why.  ``pending_rows`` and ``tombstone_rows``
    are the delta a read would merge, ``pending_ratio`` is that against ``base_rows`` (the
    generation's ``stats["rows"]`` when the index recorded one), ``age_seconds`` is how old the
    generation is.
    """

    index_name: str
    tenant_id: int | None
    reason: HealthReason
    usable: bool
    generation: Generation | None = None
    pending_rows: int = 0
    tombstone_rows: int = 0
    pending_ratio: float = 0.0
    base_rows: int | None = None
    age_seconds: float | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "index_name": self.index_name,
            "tenant_id": self.tenant_id,
            "reason": self.reason.value,
            "usable": self.usable,
            "generation": None if self.generation is None else self.generation.as_dict(),
            "pending_rows": self.pending_rows,
            "tombstone_rows": self.tombstone_rows,
            "pending_ratio": self.pending_ratio,
            "base_rows": self.base_rows,
            "age_seconds": self.age_seconds,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class Pin:
    """What :meth:`DerivedIndex.pin` yields: the generation fixed for this read, or why none.

    ``bool(pin)`` is ``pin.usable``.  ``generation`` is set whenever one is published, even
    when it is not usable, so a caller can log which one it declined.
    """

    generation: Generation | None
    reason: HealthReason
    usable: bool
    detail: str = ""

    def __bool__(self) -> bool:
        return self.usable


@dataclass(frozen=True)
class MaintenancePolicy:
    """When a published generation is due for a rebuild.

    Any trigger fires.  ``rebuild_after_rows``: the delta plus tombstones reach this many rows.
    ``rebuild_after_ratio``: they reach this fraction of the generation's base rows (only when
    the generation recorded ``stats["rows"]``).  ``rebuild_after_seconds``: the generation is
    this old and at least one row is pending.  ``None`` disables a trigger.  An absent or
    invalidated generation is always due; a generation published with ``validate=False`` is
    not, or the same policy that published it would rebuild it on every call.
    """

    rebuild_after_rows: int | None = 10_000
    rebuild_after_ratio: float | None = 0.05
    rebuild_after_seconds: float | None = 900.0
    validate: bool = True

    def due(self, health: HealthReport) -> str | None:
        """The reason a rebuild is due under this policy, or ``None``."""
        gen = health.generation
        if gen is None:
            if health.reason is HealthReason.ABSENT:
                return "no generation published"
            return None
        if not gen.validated and not gen.published_unvalidated:
            return f"generation {gen.generation} is not validated ({gen.notes or 'no note'})"
        if health.reason is HealthReason.DAMAGED_BASE:
            return f"generation {gen.generation} has a damaged base ({health.detail})"
        changes = int(health.pending_rows) + int(health.tombstone_rows)
        if self.rebuild_after_rows is not None and changes >= int(self.rebuild_after_rows):
            return f"{changes} pending row(s) >= rebuild_after_rows={self.rebuild_after_rows}"
        if (
            self.rebuild_after_ratio is not None
            and health.base_rows
            and health.pending_ratio >= float(self.rebuild_after_ratio)
        ):
            return (
                f"pending ratio {health.pending_ratio:.4f} >= "
                f"rebuild_after_ratio={self.rebuild_after_ratio}"
            )
        if (
            self.rebuild_after_seconds is not None
            and health.age_seconds is not None
            and changes > 0
            and health.age_seconds >= float(self.rebuild_after_seconds)
        ):
            return (
                f"generation is {health.age_seconds:.0f}s old with {changes} pending "
                f"row(s) >= rebuild_after_seconds={self.rebuild_after_seconds}"
            )
        return None


@dataclass(frozen=True, slots=True)
class MaintenanceReport:
    """What :func:`maintain` decided and did.

    ``action`` is one of ``"none"`` (nothing due), ``"skipped"`` (a rebuild is in progress or
    the index cannot load), ``"published"`` (built, validated, published),
    ``"validation_failed"`` (built, failed validation, retired; the previous generation stays
    published) and ``"published_unvalidated"`` (built and published with ``validate=False``).
    """

    index_name: str
    tenant_id: int | None
    action: str
    reason: str
    before: HealthReport
    after: HealthReport | None = None
    generation: Generation | None = None
    validation: ValidationReport | None = None


@dataclass(frozen=True, slots=True)
class IndexEvent:
    """One write the verbs report to the registered indexes, inside the writing transaction."""

    kind: str
    table: str
    tenant_id: int
    doc_ids: tuple[int, ...]
    at: _dt.datetime
    reason: str | None = None


# --------------------------------------------------------------------------- process state

_EPOCH_LOCK = threading.Lock()
#: Per database path, bumped by every publish / retire / build / registration in this process.
#: Watermark and definition caches compare against it.  One process is the only writer of a
#: DuckDB file, so this is the whole story for invalidation.
_EPOCH: dict[Any, int] = {}
#: Pins, process-wide: ``(database key, index name, tenant, generation) -> readers``.  Kept
#: outside the index object because two handles on one file each have their own registry,
#: and a build on one handle must not drop a generation a read on the other handle pinned.
_PINS: dict[tuple[Any, str, int | None, int], int] = {}
#: The last generation number handed out per ``(database key, index name, tenant)``, so a
#: number is never reused in this process even after the generation's catalog row is gone.
_LAST_NUMBER: dict[tuple[Any, str, int | None], int] = {}
#: Builds THIS PROCESS owns: ``(database key, index name, tenant) -> thread ident``.  Process
#: wide rather than per instance, because ``abandon_builds`` on a second handle would otherwise
#: mistake a build running on the first handle for the leftovers of a dead process.
_BUILDING: dict[tuple[Any, str, int | None], int] = {}
#: Hard erasures per ``(database key, index name)``.  A build compares the counter before and
#: after :meth:`DerivedIndex._build`: a purge that commits while the build's snapshot is open is
#: invisible to that snapshot, so the generation it produces may contain the erased document and
#: must not go live validated.
_ERASURES: dict[tuple[Any, str], int] = {}

_LIFECYCLE_GUARD = threading.Lock()
#: One re-entrant lock per database, keyed the same way as the epoch.  See the module docstring.
_LIFECYCLE_LOCKS: dict[Any, threading.RLock] = {}


def database_key(db: Any) -> Any:
    """The process-wide identity of the database ``db`` talks to.

    A file path is canonicalised (``realpath``) so two handles opened through a symlink, a
    relative path and an absolute path share one key, which is what makes the lifecycle lock and
    the pin table cover every handle on one file.  ``":memory:"`` databases are distinct per
    connection, so they key on the handle itself.
    """
    path = getattr(db, "path", None)
    if path and str(path) != ":memory:":
        return os.path.realpath(os.path.expanduser(str(path)))
    return id(db)


def lifecycle_lock(key: Any) -> threading.RLock:
    """The lock serialising generation lifecycle transitions for one database.

    Held across build ownership, pin acquisition, retirement and storage removal, and never
    across a build body or a pinned read.  Re-entrant, because ``build_next`` retires while
    holding it.
    """
    with _LIFECYCLE_GUARD:
        lock = _LIFECYCLE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LIFECYCLE_LOCKS[key] = lock
        return lock


def _epoch(key: Any) -> int:
    with _EPOCH_LOCK:
        return _EPOCH.get(key, 0)


def _bump_epoch(key: Any) -> None:
    with _EPOCH_LOCK:
        _EPOCH[key] = _EPOCH.get(key, 0) + 1


def _erasure_count(key: Any, index_name: str) -> int:
    with _EPOCH_LOCK:
        return _ERASURES.get((key, index_name), 0)


def _bump_erasures(key: Any, index_name: str) -> None:
    with _EPOCH_LOCK:
        _ERASURES[(key, index_name)] = _ERASURES.get((key, index_name), 0) + 1


def _json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        out = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def _tenant_of(tenant: Any) -> int:
    if isinstance(tenant, Namespace):
        return int(tenant.tenant_id)
    if isinstance(tenant, bool) or not isinstance(tenant, int):
        raise TypeError(f"tenant must be an int or Namespace, got {type(tenant).__name__}")
    return int(tenant)


# --------------------------------------------------------------------------- the journal
#
# Module-level primitives rather than methods, because a handle that holds no code for an index
# still has to journal for it: :class:`IndexRegistry` drives these straight from the persisted
# :class:`IndexDefinition`.  :class:`DerivedIndex` delegates to them so there is one
# implementation of "what a write does to the journal".


def journal_append(
    db: Any,
    index_name: str,
    tenant_id: int,
    doc_ids: Sequence[int],
    op: str,
    *,
    at: _dt.datetime,
    reason: str | None = None,
) -> int:
    """Append one journal row per document, ordered by :data:`INDEX_CHANGE_SEQUENCE`.

    One statement whatever the number of documents: ``nextval`` is evaluated per row of the
    ``unnest``, so the ids of one event are ordered among themselves and after everything
    already journalled.  Call inside the transaction that made the write; a rollback takes the
    journal rows with it (and leaves a gap in the sequence, which is harmless because only the
    order is read).
    """
    ids = [int(d) for d in doc_ids]
    if not ids:
        return 0
    if op not in JOURNAL_OPS:
        raise ValueError(f"journal op must be one of {JOURNAL_OPS}, got {op!r}")
    db.execute(
        f"INSERT INTO {_JOURNAL} (index_name, tenant_id, doc_id, change_seq, op, written_at, "
        f"reason, absorbed_by) "
        f"SELECT ?, ?, unnest(?::BIGINT[]), nextval('{_SEQ}'), ?, ?, ?, NULL",
        [str(index_name), int(tenant_id), ids, str(op), to_utc_naive(at) or utcnow(), reason],
    )
    return len(ids)


def journal_erase(db: Any, index_name: str, tenant_id: int, doc_ids: Sequence[int]) -> int:
    """Delete every journal row for these documents.  Part of a hard erasure, not of a close.

    A tombstone would leave the erased id in the file, which is the one thing
    ``forget(hard=True)`` promises not to do.  Correctness does not depend on the tombstone:
    the document is deleted from every generation's storage in the same transaction, and any
    generation that could not delete is invalidated.
    """
    ids = [int(d) for d in doc_ids]
    if not ids:
        return 0
    row = db.execute(
        f"DELETE FROM {_JOURNAL} WHERE index_name = ? AND tenant_id = ? "
        f"AND doc_id = ANY(?::BIGINT[])",
        [str(index_name), int(tenant_id), ids],
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def journal_latest_sql(index_name: str, tenant_id: int | None, generation: int) -> tuple[str, list]:
    """SQL for the current journal state of each document, as ``(tenant_id, doc_id, op)``.

    The newest ``change_seq`` for a ``(tenant_id, doc_id)`` wins, over the rows generation
    ``generation`` has not absorbed.  This is what makes a purged-and-reused id right: the reuse
    is a later ``insert`` and it is the one that survives.
    """
    where = "index_name = ?"
    params: list[Any] = [str(index_name)]
    if tenant_id is not None:
        where += " AND tenant_id = ?"
        params.append(int(tenant_id))
    sql = (
        f"SELECT tenant_id, doc_id, op FROM {_JOURNAL} "
        f"WHERE {where} AND (absorbed_by IS NULL OR absorbed_by > ?) "
        f"QUALIFY row_number() OVER (PARTITION BY tenant_id, doc_id "
        f"ORDER BY change_seq DESC) = 1"
    )
    return sql, params + [int(generation)]


def record_event(db: Any, definition: IndexDefinition, event: IndexEvent) -> int:
    """Journal ``event`` for ``definition``.  The generic write path, used by any handle.

    :meth:`DerivedIndex.on_write` is this function plus whatever private storage an accelerator
    keeps; a handle with no implementation for the index runs exactly this.  ``PURGE`` writes
    nothing: an erasure deletes journal rows rather than adding one
    (:func:`journal_erase`).
    """
    if definition.source_table is None or event.table != definition.source_table:
        return 0
    if not event.doc_ids:
        return 0
    if event.kind == INSERT:
        ids: Sequence[int] = event.doc_ids
        if definition.delta_mode == "watermark":
            key = None if not definition.per_tenant else int(event.tenant_id)
            gen = published_generation(db, definition.index_name, key)
            if gen is None or gen.watermark_id is None:
                return 0
            ids = [d for d in event.doc_ids if int(d) <= gen.watermark_id]
        return journal_append(db, definition.index_name, event.tenant_id, ids, INSERT, at=event.at)
    if event.kind == CLOSE:
        return journal_append(
            db,
            definition.index_name,
            event.tenant_id,
            event.doc_ids,
            CLOSE,
            at=event.at,
            reason=event.reason or event.kind,
        )
    return 0


# --------------------------------------------------------------------------- catalog reads

_GEN_COLUMNS = (
    "index_name, generation, tenant_id, watermark_id, watermark_ts, built_at, validated, "
    "published, coalesce(published_unvalidated, FALSE), stats, notes"
)


def _generation_from_row(row: Sequence[Any]) -> Generation:
    return Generation(
        index_name=str(row[0]),
        generation=int(row[1]),
        tenant_id=None if row[2] is None else int(row[2]),
        watermark_id=None if row[3] is None else int(row[3]),
        watermark_ts=row[4],
        built_at=row[5],
        validated=bool(row[6]),
        published=bool(row[7]),
        published_unvalidated=bool(row[8]),
        stats=_json(row[9]),
        notes=row[10],
    )


#: ``(database key, index name, tenant) -> (epoch, generation)``.  Process wide so two handles
#: on one file share it, and invalidated by the epoch rather than by a timer.
_PUBLISHED_CACHE: dict[tuple[Any, str, int | None], tuple[int, Generation | None]] = {}


def published_generation(db: Any, index_name: str, key: int | None) -> Generation | None:
    """The published generation of ``(index_name, key)``, through the per-epoch cache."""
    dbkey = database_key(db)
    epoch = _epoch(dbkey)
    cache_key = (dbkey, str(index_name), key)
    with _EPOCH_LOCK:
        hit = _PUBLISHED_CACHE.get(cache_key)
    if hit is not None and hit[0] == epoch:
        return hit[1]
    row = db.execute(
        f"SELECT {_GEN_COLUMNS} FROM {_GEN} WHERE index_name = ? "
        f"AND tenant_id IS NOT DISTINCT FROM ? AND published ORDER BY generation",
        [str(index_name), key],
    ).fetchall()
    gen = _generation_from_row(row[-1]) if row else None
    with _EPOCH_LOCK:
        _PUBLISHED_CACHE[cache_key] = (epoch, gen)
    return gen


# --------------------------------------------------------------------------- the index


class DerivedIndex(ABC):
    """One derived index: definition, catalog, journal and lifecycle.  Subclass per accelerator.

    Class attributes a subclass sets:

    ``name``
        The ``index_name`` in the catalog.  A bare identifier.
    ``kind``
        A short label recorded in the persisted definition so an operator reading the file can
        tell what built a generation.  Defaults to the class name.
    ``source_table`` / ``source_id_column`` / ``source_ts_column``
        The canonical table the index derives from, its id column (the ``doc_id`` of journal
        rows) and its ``tx_from`` column (the timestamp watermark).  ``source_table`` of
        ``None`` means the index has no source and records nothing (:class:`NullIndex`).
    ``per_tenant``
        True when a generation covers one tenant (CSR: dense ids are tenant-local), False when
        one generation covers the file (the full-text index is built over every tenant's rows,
        with visibility applied per query).  It is the GENERATION that is file-wide, never the
        journal: journal rows always carry the writing tenant.
    ``delta_mode``
        ``"table"`` or ``"watermark"``; see the module docstring.
    ``supports_delta``
        True when the read side merges base + delta.  False means any pending row makes the
        published generation unusable (the CSR extension as it stood in 0.1: a snapshot with
        no delta path), which :meth:`health` reports as ``STALE_GENERATION``.

    Instance state: ``load_error`` (a string when the index's storage or extension failed to
    load; :meth:`health` then reports ``LOAD_FAILURE``).
    """

    name: str = "index"
    kind: str = ""
    source_table: str | None = None
    source_id_column: str = "memory_id"
    source_ts_column: str = "tx_from"
    per_tenant: bool = True
    delta_mode: str = "table"
    supports_delta: bool = True

    def __init__(self, db: Any, *, name: str | None = None) -> None:
        if name is not None:
            self.name = str(name)
        quote_ident(self.name)
        if self.delta_mode not in ("table", "watermark"):
            raise ValueError(f"delta_mode must be 'table' or 'watermark', got {self.delta_mode!r}")
        if self.source_table is not None:
            quote_ident(self.source_table)
            quote_ident(self.source_id_column)
            quote_ident(self.source_ts_column)
        self.db = db
        self.load_error: str | None = None
        self._lock = threading.RLock()
        self._epoch_key = database_key(db)
        self._lifecycle = lifecycle_lock(self._epoch_key)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r} source={self.source_table!r}>"

    # ------------------------------------------------------------------ keys and scopes

    def definition(self) -> IndexDefinition:
        """This index as a row of :data:`~anatid.schema.INDEX_REGISTRY_TABLE`.

        :meth:`IndexRegistry.register` writes it, and every other handle on the file reads it
        instead of needing this class.  Override ``params`` in a subclass to record whatever an
        operator would want to see (the backend an accelerator chose, its dimensions).
        """
        return IndexDefinition(
            index_name=self.name,
            kind=self.kind or type(self).__name__,
            per_tenant=bool(self.per_tenant),
            source_table=self.source_table,
            source_id_column=self.source_id_column,
            delta_mode=self.delta_mode,
            supports_delta=bool(self.supports_delta),
        )

    def tenant_key(self, tenant: Any) -> int | None:
        """The catalog ``tenant_id`` for a verb's ``tenant`` argument: ``None`` file-wide."""
        if not self.per_tenant:
            return None
        if tenant is None:
            raise ValueError(f"index {self.name!r} is per tenant; a tenant is required")
        return _tenant_of(tenant)

    def journal_tenant(self, tenant: Any) -> int | None:
        """The tenant whose journal rows a call is about, or ``None`` for every tenant.

        Different from :meth:`tenant_key`, and the difference is the whole of the cross-tenant
        fix: a file-wide index has ``tenant_key() is None`` (one generation for the file) but
        its journal rows still belong to a tenant, so ``pending(2)`` must return tenant 2's
        documents and ``tombstones(1)`` must not suppress tenant 2's document with the same id.
        """
        if tenant is None:
            return None
        return _tenant_of(tenant)

    def _catalog_scope(self, key: int | None) -> tuple[str, list]:
        return "index_name = ? AND tenant_id IS NOT DISTINCT FROM ?", [self.name, key]

    def _row_scope(self, tenant_id: int | None) -> tuple[str, list]:
        """Predicate over the journal for one index, narrowed to one tenant when given."""
        if tenant_id is None:
            return "index_name = ?", [self.name]
        return "index_name = ? AND tenant_id = ?", [self.name, int(tenant_id)]

    def _source_scope(self, tenant_id: int | None, alias: str | None = None) -> tuple[str, list]:
        if tenant_id is None:
            return "TRUE", []
        return Visibility(int(tenant_id)).tenant(alias)

    # ------------------------------------------------------------------ catalog reads

    def _load_rows(self, key: int | None, generation: int | None = None) -> list[Generation]:
        where, params = self._catalog_scope(key)
        if generation is not None:
            where += " AND generation = ?"
            params = params + [int(generation)]
        rows = self.db.execute(
            f"SELECT {_GEN_COLUMNS} FROM {_GEN} WHERE {where} ORDER BY generation", params
        ).fetchall()
        return [_generation_from_row(r) for r in rows]

    def all_generations(self) -> list[Generation]:
        """Every generation of this index, for every tenant scope.  Used by :meth:`erase`."""
        rows = self.db.execute(
            f"SELECT {_GEN_COLUMNS} FROM {_GEN} WHERE index_name = ? ORDER BY tenant_id, "
            f"generation",
            [self.name],
        ).fetchall()
        return [_generation_from_row(r) for r in rows]

    def generations(self, tenant: Any = None) -> list[Generation]:
        """Every generation recorded for ``tenant``, oldest first."""
        return self._load_rows(self.tenant_key(tenant))

    def generation(self, tenant: Any, number: int) -> Generation | None:
        """One generation by number, or ``None``."""
        rows = self._load_rows(self.tenant_key(tenant), int(number))
        return rows[0] if rows else None

    def current_generation(self, tenant: Any = None) -> Generation | None:
        """The published generation for ``tenant``, or ``None``."""
        key = self.tenant_key(tenant)
        for g in self._load_rows(key):
            if g.published:
                return g
        return None

    def _cached_current(self, key: int | None) -> Generation | None:
        """The published generation, through the process-wide per-epoch cache."""
        return published_generation(self.db, self.name, key)

    def watermark(self, tenant: Any = None) -> Watermark:
        """The published generation's watermark, or an empty one."""
        gen = self.current_generation(tenant)
        return gen.watermark if gen is not None else Watermark()

    def owns_build(self, key: int | None) -> bool:
        """True when a build of ``(this index, key)`` is running in THIS process."""
        with _EPOCH_LOCK:
            return (self._epoch_key, self.name, key) in _BUILDING

    def is_building(self, tenant: Any = None) -> bool:
        """True while a build for ``tenant`` is in progress in this process or announced in
        the catalog by another handle."""
        key = self.tenant_key(tenant)
        if self.owns_build(key):
            return True
        return any(g.building for g in self._load_rows(key))

    # ------------------------------------------------------------------ the journal

    def _source_rows_after(
        self, tenant_id: int | None, watermark_id: int | None
    ) -> tuple[str, list]:
        """SQL selecting ``(tenant_id, source id)`` above ``watermark_id`` (all rows when None)."""
        assert self.source_table is not None
        idc = quote_ident(self.source_id_column)
        tbl = quote_ident(self.source_table)
        scope, params = self._source_scope(tenant_id)
        sql = f"SELECT tenant_id, {idc} FROM {tbl} WHERE {scope}"
        if watermark_id is None:
            return sql, params
        return f"{sql} AND {idc} > ?", params + [int(watermark_id)]

    def _latest_sql(self, tenant_id: int | None, generation: Generation | None) -> tuple[str, list]:
        number = 0 if generation is None else int(generation.generation)
        return journal_latest_sql(self.name, tenant_id, number)

    def _pending_sql(
        self, tenant_id: int | None, generation: Generation | None
    ) -> tuple[str, list]:
        """``(tenant_id, doc_id)`` a read must merge with the base of ``generation``."""
        latest, params = self._latest_sql(tenant_id, generation)
        sql = f"SELECT tenant_id, doc_id FROM ({latest}) WHERE op = '{INSERT}'"
        if self.source_table is not None and (self.delta_mode == "watermark" or generation is None):
            src_sql, src_params = self._source_rows_after(
                tenant_id, None if generation is None else generation.watermark_id
            )
            sql = f"{sql} UNION {src_sql}"
            params = params + src_params
        return sql, params

    def _tombstone_sql(
        self, tenant_id: int | None, generation: Generation | None
    ) -> tuple[str, list]:
        latest, params = self._latest_sql(tenant_id, generation)
        return f"SELECT tenant_id, doc_id FROM ({latest}) WHERE op <> '{INSERT}'", params

    def _scope_for(
        self, tenant: Any, generation: Generation | None
    ) -> tuple[int | None, Generation | None]:
        key = self.tenant_key(tenant)
        gen = generation if generation is not None else self._cached_current(key)
        return self.journal_tenant(tenant), gen

    def pending_keys(
        self, tenant: Any = None, generation: Generation | None = None
    ) -> list[tuple[int, int]]:
        """``(tenant_id, doc_id)`` written since ``generation`` (default: the published one).

        The pair, not the bare id: ``memory_id`` is unique within a tenant only, so an index
        whose generations cover the file must key its candidates by both or one tenant's write
        lands on another tenant's document.
        """
        jt, gen = self._scope_for(tenant, generation)
        sql, params = self._pending_sql(jt, gen)
        return sorted({(int(r[0]), int(r[1])) for r in self.db.execute(sql, params).fetchall()})

    def pending(self, tenant: Any = None, generation: Generation | None = None) -> list[int]:
        """Source ids written since ``generation``, for ``tenant``, sorted.

        With no generation at all every source row is pending.  The list is the delta a read
        merges with the base before it applies visibility; it is not itself filtered.  Pass the
        tenant even for a file-wide index: without it this returns every tenant's ids and the
        caller has lost the identity it needs (:meth:`pending_keys` keeps it).
        """
        return sorted({doc for _t, doc in self.pending_keys(tenant, generation)})

    def pending_count(self, tenant: Any = None, generation: Generation | None = None) -> int:
        jt, gen = self._scope_for(tenant, generation)
        sql, params = self._pending_sql(jt, gen)
        row = self.db.execute(f"SELECT count(*) FROM ({sql})", params).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def tombstone_keys(
        self, tenant: Any = None, generation: Generation | None = None
    ) -> list[tuple[int, int]]:
        """``(tenant_id, doc_id)`` closed since ``generation``.  Subtract them from the merge."""
        jt, gen = self._scope_for(tenant, generation)
        sql, params = self._tombstone_sql(jt, gen)
        return sorted({(int(r[0]), int(r[1])) for r in self.db.execute(sql, params).fetchall()})

    def tombstones(self, tenant: Any = None, generation: Generation | None = None) -> list[int]:
        """Source ids closed since ``generation``, for ``tenant``, sorted.  Subtract them."""
        return sorted({doc for _t, doc in self.tombstone_keys(tenant, generation)})

    def tombstone_count(self, tenant: Any = None, generation: Generation | None = None) -> int:
        jt, gen = self._scope_for(tenant, generation)
        sql, params = self._tombstone_sql(jt, gen)
        row = self.db.execute(f"SELECT count(*) FROM ({sql})", params).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def record_delta(self, tenant_id: int, doc_ids: Sequence[int], *, at: _dt.datetime) -> int:
        """Journal ``doc_ids`` as written.  Call inside the transaction that wrote them."""
        return journal_append(self.db, self.name, tenant_id, doc_ids, INSERT, at=at)

    def tombstone(
        self, tenant_id: int, doc_ids: Sequence[int], *, at: _dt.datetime, reason: str | None = None
    ) -> int:
        """Journal ``doc_ids`` as closed.  Call inside the transaction that closed them."""
        return journal_append(self.db, self.name, tenant_id, doc_ids, CLOSE, at=at, reason=reason)

    def on_write(self, event: IndexEvent) -> None:
        """Journal ``event``.  Runs inside the verb's transaction.

        The default is :func:`record_event` against this index's :meth:`definition`, which is
        exactly what a handle holding no implementation runs, so the journal does not depend on
        which handle wrote.  An index that keeps its own delta storage overrides this and may
        call ``super().on_write(event)`` to keep the shared journal as well.  ``PURGE`` writes
        nothing here: an erasure goes through :meth:`erase`.
        """
        record_event(self.db, self.definition(), event)

    # ------------------------------------------------------------------ erasure

    def erase(
        self, tenant_id: int, doc_ids: Sequence[int], *, reason: str | None = None
    ) -> ErasureResult:
        """Remove ``doc_ids`` from every generation's storage and from the journal.

        Called by ``forget(hard=True)`` inside the purge transaction.  Each generation gets
        :meth:`_erase`; a generation that returns ``None`` (the default, meaning "this storage
        cannot delete one document") or raises is INVALIDATED instead, which takes it out of
        service until it is rebuilt rather than leaving the erased document in a live index.
        Journal rows for the documents are deleted, not tombstoned: a tombstone would keep the
        erased id in the file.
        """
        ids = [int(d) for d in doc_ids]
        if not ids:
            return ErasureResult()
        # Under the lifecycle lock, so no generation is retired and dropped while this is
        # deleting from its storage.  It does not close the window entirely: a retire that
        # committed after the purge's transaction opened is invisible to the purge's snapshot,
        # and the purge's commit then fails with a retryable ConflictError, which is DuckDB's
        # optimistic contract and anatid's documented one.  Taking a process lock inside a
        # transaction is safe here because DuckDB never blocks a transaction on another: the
        # thread holding this lock always makes progress.
        with self._lifecycle:
            return self._erase_all(ids, int(tenant_id), reason)

    def _erase_all(self, ids: list[int], tenant_id: int, reason: str | None) -> ErasureResult:
        # Bumped BEFORE the generations are read, which is what makes the hand-off with a
        # concurrent build total: either the build's check sees this bump, or this loop sees the
        # build's committed generation.  A rolled-back purge leaves a bump behind and costs one
        # rebuild, which is the safe direction.
        _bump_erasures(self._epoch_key, self.name)
        deleted = 0
        invalidated = 0
        recorded = self.all_generations()
        for gen in recorded:
            if gen.tenant_id is not None and gen.tenant_id != int(tenant_id):
                continue
            if gen.building:
                # A build in flight owns that catalog row, and its storage is inside a
                # transaction this one cannot see.  The erasure counter is what reaches it:
                # build_next compares it before and after and marks its own result invalid.
                # A 'building' row left by a dead process is orphaned storage that
                # abandon_builds drops.
                continue
            try:
                removed = self._erase(gen, int(tenant_id), ids)
            except Exception as exc:  # noqa: BLE001 - any storage failure fails closed
                log.warning(
                    "index %s: generation %d could not erase %d document(s): %s",
                    self.name,
                    gen.generation,
                    len(ids),
                    exc,
                )
                removed = None
            if removed is None:
                detail = "a hard erasure could not reach this generation's storage"
                self._invalidate_generation(gen, f"{detail} ({reason})" if reason else detail)
                # Counted whether or not the generation was still in service.  The number on
                # the receipt answers "what did this purge fail to clean", and a generation
                # that was already invalid still holds the document; reporting 0 for it would
                # read as a complete erasure.
                invalidated += 1
            else:
                deleted += int(removed)
        deleted += journal_erase(self.db, self.name, tenant_id, ids)
        self._clamp_watermarks(ids, recorded)
        if invalidated:
            _bump_epoch(self._epoch_key)
        return ErasureResult(deleted, invalidated)

    def _clamp_watermarks(
        self, ids: list[int], recorded: Sequence[Generation] | None = None
    ) -> int:
        """Lower any generation watermark that IS one of the erased ids.  Returns rows changed.

        ``watermark_id`` is the largest source id in a generation's snapshot, so purging the row
        that set it leaves the erased id sitting in the catalog: the same leak
        ``anatid_meta.fts_indexed_max_id`` had, in the table this framework added.  The new value
        is the largest SURVIVING source id at or below the old one (``NULL`` when nothing is
        left), which can only widen the set a read treats as pending and never narrow it, so the
        clamp cannot lose a row.  ``watermark_ts`` moves with it so the pair stays one snapshot.

        Rows are matched by ``watermark_id`` alone: a generation of any tenant scope whose
        watermark happens to be this id is rewritten, because the id is what must leave the file.

        ``recorded`` is the generation list the caller has already loaded.  Given it, the usual
        purge -- of a document that is nobody's watermark -- costs no statement at all: the
        ``UPDATE`` scans the source table and was measured at 1.2 ms against 2,000 rows, which
        is a sixth of a hard forget.
        """
        if self.source_table is None or not ids:
            return 0
        if recorded is not None:
            wanted = {int(d) for d in ids}
            if not any(g.watermark_id in wanted for g in recorded):
                return 0
        idc = quote_ident(self.source_id_column)
        tsc = quote_ident(self.source_ts_column)
        src = quote_ident(self.source_table)
        surviving = (
            f"FROM {src} s WHERE s.{idc} <= g.watermark_id "
            f"AND (g.tenant_id IS NULL OR s.tenant_id = g.tenant_id)"
        )
        row = self.db.execute(
            f"UPDATE {_GEN} AS g SET "
            f"watermark_id = (SELECT max(s.{idc}) {surviving}), "
            f"watermark_ts = (SELECT max(s.{tsc}) {surviving}) "
            f"WHERE g.index_name = ? AND g.watermark_id = ANY(?::BIGINT[])",
            [self.name, [int(d) for d in ids]],
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def _invalidate_generation(self, generation: Generation, reason: str) -> int:
        """Take one generation out of service.  Returns 1 when it was still in service.

        The note is written whatever the generation's state, so a later :meth:`validate` cannot
        quietly bless storage that still holds an erased document; only a generation that was
        usable is counted, so a receipt does not claim to have taken out something already out.
        """
        row = self.db.execute(
            f"SELECT validated, coalesce(published_unvalidated, FALSE) FROM {_GEN} "
            f"WHERE index_name = ? AND tenant_id IS NOT DISTINCT FROM ? AND generation = ?",
            [self.name, generation.tenant_id, generation.generation],
        ).fetchone()
        if row is None:
            return 0
        self.db.execute(
            f"UPDATE {_GEN} SET validated = FALSE, published_unvalidated = FALSE, notes = ? "
            f"WHERE index_name = ? AND tenant_id IS NOT DISTINCT FROM ? AND generation = ?",
            [
                f"invalidated: {reason}"[:500],
                self.name,
                generation.tenant_id,
                generation.generation,
            ],
        )
        return 1 if (bool(row[0]) or bool(row[1])) else 0

    # ------------------------------------------------------------------ lifecycle

    def _next_number(self, key: int | None) -> int:
        """The next generation number: above every number the catalog, the absorbed journal
        rows, and this process have seen, so a retired number is never reused."""
        where, params = self._catalog_scope(key)
        row = self.db.execute(
            f"SELECT max(generation) FROM {_GEN} WHERE {where}", params
        ).fetchone()
        top = int(row[0]) if row and row[0] is not None else 0
        scope, sparams = self._row_scope(key)
        r = self.db.execute(
            f"SELECT max(absorbed_by) FROM {_JOURNAL} WHERE {scope}", sparams
        ).fetchone()
        if r and r[0] is not None:
            top = max(top, int(r[0]))
        with _EPOCH_LOCK:
            top = max(top, _LAST_NUMBER.get((self._epoch_key, self.name, key), 0))
            _LAST_NUMBER[(self._epoch_key, self.name, key)] = top + 1
        return top + 1

    def _prune_absorbed(self, key: int | None) -> int:
        """Delete journal rows no generation still alive needs.

        A reader pinned to generation N takes the rows with ``absorbed_by > N``, so a row may
        only go once every generation older than the one that absorbed it is gone from the
        catalog.  ``floor`` is the oldest generation still recorded (published, pinned or
        building); rows absorbed by ``floor`` or older are in every alive base.
        """
        where, params = self._catalog_scope(key)
        row = self.db.execute(
            f"SELECT min(generation) FROM {_GEN} WHERE {where}", params
        ).fetchone()
        if row is None or row[0] is None:
            return 0
        floor = int(row[0])
        scope, sparams = self._row_scope(key)
        r = self.db.execute(
            f"DELETE FROM {_JOURNAL} WHERE {scope} AND absorbed_by IS NOT NULL "
            f"AND absorbed_by <= ?",
            sparams + [floor],
        ).fetchone()
        return int(r[0]) if r and r[0] is not None else 0

    def _source_watermark(self, key: int | None) -> Watermark:
        if self.source_table is None:
            return Watermark()
        idc = quote_ident(self.source_id_column)
        tsc = quote_ident(self.source_ts_column)
        scope, params = self._source_scope(key)
        row = self.db.execute(
            f"SELECT max({idc}), max({tsc}) FROM {quote_ident(self.source_table)} WHERE {scope}",
            params,
        ).fetchone()
        if row is None:
            return Watermark()
        return Watermark(id=None if row[0] is None else int(row[0]), ts=row[1])

    def _absorb(self, key: int | None, number: int) -> int:
        """Stamp the journal rows visible in this transaction with ``number``."""
        scope, params = self._row_scope(key)
        d = self.db.execute(
            f"UPDATE {_JOURNAL} SET absorbed_by = ? WHERE {scope} AND absorbed_by IS NULL",
            [number] + params,
        ).fetchone()
        return int(d[0]) if d and d[0] is not None else 0

    def _claim_build(self, key: int | None) -> None:
        """Take process-wide ownership of a build, under the lifecycle lock."""
        ident = threading.get_ident()
        with _EPOCH_LOCK:
            owner = _BUILDING.get((self._epoch_key, self.name, key))
            if owner is not None:
                raise IndexGenerationError(
                    f"index {self.name!r}: a build for tenant {key} is already in progress "
                    f"in this process (thread {owner})",
                    index=self.name,
                )
            _BUILDING[(self._epoch_key, self.name, key)] = ident

    def _release_build(self, key: int | None) -> None:
        with _EPOCH_LOCK:
            _BUILDING.pop((self._epoch_key, self.name, key), None)

    def _announce_build(
        self, key: int | None, now: _dt.datetime | None
    ) -> tuple[int, _dt.datetime]:
        """Take build ownership and write the ``building`` catalog row, under the lifecycle lock.

        Returns the generation number and the start time.  Ownership is released by
        :meth:`build_next`, which is the only caller: the lock covers claiming, the retire sweep
        and the announcement, and is dropped before the build itself runs.
        """
        with self._lifecycle:
            self._claim_build(key)
            try:
                stale = [g for g in self._load_rows(key) if g.building]
                if stale:
                    raise IndexGenerationError(
                        f"index {self.name!r}: generation {stale[-1].generation} for tenant "
                        f"{key} is marked building (announced {stale[-1].built_at}); another "
                        f"build owns it, or its process died. Call abandon_builds() to clear a "
                        f"dead one.",
                        index=self.name,
                        generation=stale[-1].generation,
                    )
                self._retire_unpinned(key)
                number = self._next_number(key)
                started = to_utc_naive(now) or utcnow()
                with self.db.transaction():
                    self.db.execute(
                        f"INSERT INTO {_GEN} (index_name, generation, tenant_id, watermark_id, "
                        f"watermark_ts, built_at, validated, published, published_unvalidated, "
                        f"stats, notes) "
                        f"VALUES (?, ?, ?, NULL, NULL, ?, FALSE, FALSE, FALSE, '{{}}', "
                        f"'building')",
                        [self.name, number, key, started],
                    )
                _bump_epoch(self._epoch_key)
                return number, started
            except BaseException:
                self._release_build(key)
                raise

    def build_next(self, tenant: Any = None, *, now: _dt.datetime | None = None) -> Generation:
        """Build the next generation beside the current one.  Returns it, unpublished.

        Announces the generation in the catalog first (``notes = 'building'``), then in one
        transaction takes the source watermark, calls :meth:`_build`, stamps the journal rows in
        its snapshot as absorbed, and records the watermark and stats.  On any failure the
        storage is dropped, the catalog row removed and the error re-raised.  Leftover
        generations that are neither published, pinned nor building are retired first, which is
        when the journal rows only they still needed are pruned.

        Build ownership, the announcement and the retire sweep are taken under the process-wide
        lifecycle lock; the build itself is not, so a rebuild does not block reads.  A hard
        erasure that commits while the build's snapshot is open leaves the new generation
        unvalidated with a note, because the snapshot cannot see the purge and the storage it
        built may still hold the erased document.

        Best called outside a transaction: inside one the announcement is not visible to other
        threads until the caller commits.
        """
        key = self.tenant_key(tenant)
        erasures_before = _erasure_count(self._epoch_key, self.name)
        number, started = self._announce_build(key, now)
        try:
            gen = Generation(
                index_name=self.name,
                generation=number,
                tenant_id=key,
                built_at=started,
                notes="building",
            )
            try:
                with self.db.transaction():
                    wm = self._source_watermark(key)
                    gen = replace(gen, watermark_id=wm.id, watermark_ts=wm.ts)
                    stats = dict(self._build(gen) or {})
                    absorbed = self._absorb(key, number)
                    stats.setdefault("absorbed_journal_rows", absorbed)
                    note = None
                    if _erasure_count(self._epoch_key, self.name) != erasures_before:
                        note = (
                            "invalidated: a hard erasure landed while this generation was "
                            "being built, so its snapshot may still hold the erased "
                            "document"
                        )
                    self.db.execute(
                        f"UPDATE {_GEN} SET watermark_id = ?, watermark_ts = ?, stats = ?, "
                        f"notes = ? WHERE index_name = ? AND tenant_id IS NOT DISTINCT FROM ? "
                        f"AND generation = ?",
                        [
                            wm.id,
                            wm.ts,
                            json.dumps(stats, default=str),
                            note,
                            self.name,
                            key,
                            number,
                        ],
                    )
            except BaseException:
                with contextlib.suppress(Exception):
                    self._drop(gen)
                with contextlib.suppress(Exception):
                    self.db.execute(
                        f"DELETE FROM {_GEN} WHERE index_name = ? AND tenant_id IS NOT DISTINCT "
                        f"FROM ? AND generation = ?",
                        [self.name, key, number],
                    )
                raise
        finally:
            self._release_build(key)
            _bump_epoch(self._epoch_key)
        if _erasure_count(self._epoch_key, self.name) != erasures_before:
            # A purge landed between the in-transaction check and the commit.  The check inside
            # the transaction cannot see that, and the purge skipped this generation because its
            # row still said 'building', so the note goes on here.
            self._invalidate_generation(
                Generation(
                    index_name=self.name, generation=number, tenant_id=key, built_at=started
                ),
                "a hard erasure landed while this generation was being built, so its snapshot "
                "may still hold the erased document",
            )
            _bump_epoch(self._epoch_key)
        built = self.generation(tenant, number)
        assert built is not None
        log.info(
            "index %s: built generation %d for tenant %s (watermark id=%s)",
            self.name,
            number,
            key,
            built.watermark_id,
        )
        return built

    def validate(self, generation: Generation) -> ValidationReport:
        """Run :meth:`_validate` and record the verdict on the catalog row.

        A generation whose ``notes`` say it was INVALIDATED is refused here, whatever the
        oracle says, and the report comes back not ok so :func:`maintain` retires it and keeps
        the generation that is live.

        The refusal is a check on the catalog row and not an appeal to the oracle, because the
        oracle can agree with an invalidated base.  A hard erasure that is still uncommitted is
        invisible to this transaction, so a base that already holds the erased document matches
        the canonical rows this sees exactly; validating it would clear the note the purge's
        hand-off wrote and publish a base with the erased document in it.  The same holds for a
        bulk load: the rows it added are in neither the base nor the journal, and a check run
        before it commits cannot see the difference.
        """
        recorded = self.generation(generation.tenant_id, generation.generation)
        if recorded is not None and recorded.notes and recorded.notes.startswith("invalidated:"):
            return ValidationReport(
                ok=False,
                generation=recorded,
                detail=(
                    f"generation {recorded.generation} was invalidated and cannot be "
                    f"revalidated: {recorded.notes}. Build a new one."
                ),
            )
        report = self._validate(generation)
        self.db.execute(
            f"UPDATE {_GEN} SET validated = ?, published_unvalidated = FALSE, notes = ? "
            f"WHERE index_name = ? AND tenant_id IS NOT DISTINCT FROM ? AND generation = ?",
            [
                bool(report.ok),
                None if report.ok else f"validation failed: {report.detail}"[:500],
                self.name,
                generation.tenant_id,
                generation.generation,
            ],
        )
        _bump_epoch(self._epoch_key)
        refreshed = self.generation(generation.tenant_id, generation.generation)
        return replace(report, generation=refreshed or generation)

    def publish(self, generation: Generation, *, force: bool = False) -> Generation:
        """Make ``generation`` the one reads pin.  One ``UPDATE`` inside a transaction.

        Requires ``validated`` unless ``force=True``, which records
        ``published_unvalidated`` so :meth:`health` and :meth:`pin` report
        :class:`HealthReason.UNVALIDATED` rather than pretending the oracle check happened.
        Such a generation IS usable: candidate generation only narrows, the canonical rows
        decide the answer, and a mode that published generations no read would accept would
        just loop.  It is refused for a generation that was INVALIDATED, where the base is
        missing rows the journal does not carry either.

        The same statement takes the flag off the previous generation, so no two are published
        at once and no instant has none.  The previous generation stays in the catalog, with its
        storage, until it is retired (by the next :meth:`build_next`, or explicitly): a reader
        may still be pinned to it, and it reads the old base, so the journal rows that
        generation still needs are kept too.  Rows are pruned only when no alive generation
        needs them (:meth:`_prune_absorbed`).
        """
        key = generation.tenant_id if self.per_tenant else None
        current = self.generation(key, generation.generation)
        if current is None:
            raise IndexGenerationError(
                f"index {self.name!r}: generation {generation.generation} for tenant {key} "
                f"does not exist",
                index=self.name,
                generation=generation.generation,
            )
        if current.building:
            raise IndexGenerationError(
                f"index {self.name!r}: generation {generation.generation} is still building",
                index=self.name,
                generation=generation.generation,
            )
        invalidated = bool(current.notes and current.notes.startswith("invalidated:"))
        if not current.validated and not force:
            raise IndexGenerationError(
                f"index {self.name!r}: generation {generation.generation} has not been "
                f"validated; validate() it or pass force=True",
                index=self.name,
                generation=generation.generation,
            )
        if not current.validated and force and invalidated:
            raise IndexGenerationError(
                f"index {self.name!r}: generation {generation.generation} was invalidated "
                f"({current.notes}); force cannot publish it, because the rows it is missing "
                f"are not in the journal either. Build a new generation.",
                index=self.name,
                generation=generation.generation,
            )
        number = int(generation.generation)
        unvalidated = not current.validated
        with self._lifecycle, self.db.transaction():
            self.db.execute(
                f"UPDATE {_GEN} SET published = (generation = ?) WHERE index_name = ? "
                f"AND tenant_id IS NOT DISTINCT FROM ?",
                [number, self.name, key],
            )
            if unvalidated:
                self.db.execute(
                    f"UPDATE {_GEN} SET published_unvalidated = TRUE, notes = ? WHERE "
                    f"index_name = ? AND tenant_id IS NOT DISTINCT FROM ? AND generation = ?",
                    ["published without validation (force)", self.name, key, number],
                )
            self._prune_absorbed(key)
        _bump_epoch(self._epoch_key)
        published = self.generation(key, number)
        assert published is not None
        log.info(
            "index %s: published generation %d for tenant %s%s",
            self.name,
            number,
            key,
            " without validation" if unvalidated else "",
        )
        return published

    def _pin_key(self, generation: Generation) -> tuple[Any, str, int | None, int]:
        return (self._epoch_key, generation.index_name, generation.tenant_id, generation.generation)

    def pinned(self, generation: Generation) -> int:
        """How many reads in this process currently pin ``generation`` (on any handle)."""
        with _EPOCH_LOCK:
            return _PINS.get(self._pin_key(generation), 0)

    def retire(self, generation: Generation) -> bool:
        """Drop a generation's storage and catalog row.  Refuses a published one; returns
        False (and keeps it) when a read in this process still pins it.

        The pin check, the storage drop and the catalog delete are one critical section under
        the lifecycle lock, so a read that is choosing this generation right now either
        registers its pin first (and this returns False) or finds the generation already gone.
        """
        with self._lifecycle:
            key = generation.tenant_id if self.per_tenant else None
            current = self.generation(key, generation.generation)
            if current is None:
                return False
            if current.published:
                raise IndexGenerationError(
                    f"index {self.name!r}: generation {generation.generation} is published; "
                    f"publish another one first",
                    index=self.name,
                    generation=generation.generation,
                )
            if self.pinned(current):
                return False
            self._drop(current)
            with self.db.transaction():
                self.db.execute(
                    f"DELETE FROM {_GEN} WHERE index_name = ? AND tenant_id IS NOT DISTINCT "
                    f"FROM ? AND generation = ?",
                    [self.name, key, generation.generation],
                )
                self._prune_absorbed(key)
            _bump_epoch(self._epoch_key)
            return True

    def drop_all(self) -> int:
        """Drop every generation of this index -- storage, catalog rows and journal.  Returns
        how many generations went.

        The PUBLISHED one included, which :meth:`retire` refuses on its own.  Publication is
        the promise that a read has a generation to pin; the promise ends when the index does,
        and without this there is no supported way to reclaim a published generation's storage
        at all, so unregistering an index left a table holding a copy of everything it had
        indexed, out of reach of ``forget(hard=True)``.

        Refuses while a generation is pinned or building: neither its storage nor its catalog
        row is this method's to remove.  Whole critical section under the lifecycle lock.
        """
        with self._lifecycle:
            rows = self.all_generations()
            for g in rows:
                if g.building:
                    raise IndexGenerationError(
                        f"index {self.name!r}: generation {g.generation} is being built; "
                        f"wait for it or call abandon_builds()",
                        index=self.name,
                        generation=g.generation,
                    )
                if self.pinned(g):
                    raise IndexGenerationError(
                        f"index {self.name!r}: generation {g.generation} is pinned by a read "
                        f"in this process",
                        index=self.name,
                        generation=g.generation,
                    )
            for g in rows:
                self._drop(g)
            with self.db.transaction():
                self.db.execute(f"DELETE FROM {_GEN} WHERE index_name = ?", [self.name])
                self.db.execute(f"DELETE FROM {_JOURNAL} WHERE index_name = ?", [self.name])
            if rows:
                _bump_epoch(self._epoch_key)
            return len(rows)

    def _retire_unpinned(self, key: int | None) -> int:
        """Retire every generation for ``key`` that is neither published, pinned nor building.

        A ``building`` row belongs to a build in progress somewhere else, or to a build that
        died; this cannot tell the two apart, so it leaves them to :meth:`abandon_builds`.
        """
        n = 0
        with self._lifecycle:
            for g in self._load_rows(key):
                if g.published or g.building or self.pinned(g):
                    continue
                with contextlib.suppress(Exception):
                    if self.retire(g):
                        n += 1
        return n

    def abandon_builds(self, tenant: Any = None) -> int:
        """Retire the ``building`` generations of ``tenant`` left behind by a build that died.

        DuckDB admits one writer process per file, so a ``building`` row that no build in this
        PROCESS owns can only be a leftover from a process that no longer exists.  Ownership is
        tracked process-wide, so this never takes a build running on another handle or another
        thread of this process.  Returns how many were retired.  :meth:`build_next` refuses to
        start while such a row exists, so an operator (or :func:`maintain` with
        ``abandon_stale=True``) calls this first.
        """
        key = self.tenant_key(tenant)
        n = 0
        with self._lifecycle:
            if self.owns_build(key):
                return 0
            for g in self._load_rows(key):
                if g.building and not g.published and not self.pinned(g):
                    self._drop(g)
                    with self.db.transaction():
                        self.db.execute(
                            f"DELETE FROM {_GEN} WHERE index_name = ? AND tenant_id IS NOT "
                            f"DISTINCT FROM ? AND generation = ?",
                            [self.name, key, g.generation],
                        )
                        self._prune_absorbed(key)
                    n += 1
            if n:
                _bump_epoch(self._epoch_key)
        return n

    @contextlib.contextmanager
    def pin(self, tenant: Any = None, *, as_of: AsOf | _dt.datetime | None = None) -> Iterator[Pin]:
        """Fix the published generation for the duration of one read.

        Yields a :class:`Pin`.  While the block runs the generation cannot be retired by this
        process, so a publish that lands mid-read leaves the reader on the generation it
        started with.  Choosing the generation and registering the pin happen together under
        the lifecycle lock, so nothing can retire it in between; the lock is released before
        the read runs.  Yields an unusable pin, with the reason, for a historical ``as_of``, a
        load failure, an absent or building generation, an invalidated generation, and (for an
        index that does not merge deltas) a generation with pending rows or tombstones.  A
        force-published generation yields a usable pin with
        :attr:`HealthReason.UNVALIDATED`.
        """
        key = self.tenant_key(tenant)
        scope = AsOf.coerce(as_of)
        pk: tuple[Any, str, int | None, int] | None = None
        with self._lifecycle:
            gen: Generation | None = None
            if self.load_error:
                pin = Pin(None, HealthReason.LOAD_FAILURE, False, self.load_error)
            elif not scope.is_current:
                pin = Pin(
                    None,
                    HealthReason.HISTORICAL_QUERY,
                    False,
                    "a current-state index cannot answer an as_of read; use the SQL path",
                )
            else:
                gen = self._cached_current(key)
                if gen is None:
                    if self.is_building(tenant):
                        pin = Pin(
                            None,
                            HealthReason.REBUILD_IN_PROGRESS,
                            False,
                            "no generation is published and one is being built",
                        )
                    else:
                        pin = Pin(
                            None, HealthReason.ABSENT, False, "no generation has been published"
                        )
                elif not gen.validated and not gen.usable_without_validation:
                    pin = Pin(
                        gen,
                        HealthReason.STALE_GENERATION,
                        False,
                        gen.notes or "the published generation is not validated",
                    )
                elif not self.supports_delta and (
                    self.pending_count(tenant, gen) or self.tombstone_count(tenant, gen)
                ):
                    pin = Pin(
                        gen,
                        HealthReason.STALE_GENERATION,
                        False,
                        "the index does not merge deltas and rows were written since the build",
                    )
                elif gen.usable_without_validation:
                    pin = Pin(
                        gen,
                        HealthReason.UNVALIDATED,
                        True,
                        "published with force: this generation was never compared with the oracle",
                    )
                else:
                    pin = Pin(gen, HealthReason.FRESH, True, "")
            if gen is not None:
                pk = self._pin_key(gen)
                with _EPOCH_LOCK:
                    _PINS[pk] = _PINS.get(pk, 0) + 1
        try:
            yield pin
        finally:
            if pk is not None:
                with _EPOCH_LOCK:
                    left = _PINS.get(pk, 0) - 1
                    if left <= 0:
                        _PINS.pop(pk, None)
                    else:
                        _PINS[pk] = left

    def health(
        self,
        tenant: Any = None,
        *,
        as_of: AsOf | _dt.datetime | None = None,
        policy: MaintenancePolicy | None = None,
        now: _dt.datetime | None = None,
    ) -> HealthReport:
        """The state of this index for ``tenant``: a :class:`HealthReport` with a reason."""
        key = self.tenant_key(tenant)
        scope = AsOf.coerce(as_of)
        pol = policy or MaintenancePolicy()
        at = to_utc_naive(now) or utcnow()
        gen = self.current_generation(tenant)
        building = self.is_building(tenant)

        def report(
            reason: HealthReason, usable: bool, detail: str, *, pending: int = 0, tombs: int = 0
        ) -> HealthReport:
            base = None
            ratio = 0.0
            age = None
            if gen is not None:
                base_val = gen.stats.get("rows")
                base = int(base_val) if isinstance(base_val, (int, float)) else None
                changes = pending + tombs
                ratio = (changes / base) if base else (1.0 if changes else 0.0)
                if gen.built_at is not None:
                    age = max(0.0, (at - gen.built_at).total_seconds())
            return HealthReport(
                index_name=self.name,
                tenant_id=key,
                reason=reason,
                usable=usable,
                generation=gen,
                pending_rows=pending,
                tombstone_rows=tombs,
                pending_ratio=ratio,
                base_rows=base,
                age_seconds=age,
                detail=detail,
            )

        if self.load_error:
            return report(HealthReason.LOAD_FAILURE, False, self.load_error)
        if not scope.is_current:
            return report(
                HealthReason.HISTORICAL_QUERY,
                False,
                "a current-state index cannot answer an as_of read; the SQL path does",
            )
        if gen is None:
            pending = self.pending_count(tenant, None) if self.source_table else 0
            if building:
                return report(
                    HealthReason.REBUILD_IN_PROGRESS,
                    False,
                    "no generation is published and one is being built",
                    pending=pending,
                )
            return report(
                HealthReason.ABSENT, False, "no generation has been published", pending=pending
            )
        pending = self.pending_count(tenant, gen)
        tombs = self.tombstone_count(tenant, gen)
        if not gen.validated and not gen.usable_without_validation:
            return report(
                HealthReason.STALE_GENERATION,
                False,
                gen.notes or "the published generation is not validated",
                pending=pending,
                tombs=tombs,
            )
        damage = self.base_damage(gen)
        if damage is not None:
            return report(HealthReason.DAMAGED_BASE, False, damage, pending=pending, tombs=tombs)
        probe = report(HealthReason.FRESH, True, "", pending=pending, tombs=tombs)
        due = pol.due(probe)
        suffix = " (a rebuild is in progress)" if building else ""
        if due is not None:
            usable = bool(self.supports_delta)
            return report(
                HealthReason.STALE_GENERATION,
                usable,
                f"{due}{suffix}",
                pending=pending,
                tombs=tombs,
            )
        if not self.supports_delta and (pending or tombs):
            return report(
                HealthReason.STALE_GENERATION,
                False,
                f"the index does not merge deltas and {pending + tombs} row(s) were "
                f"written since the build{suffix}",
                pending=pending,
                tombs=tombs,
            )
        if gen.usable_without_validation:
            return report(
                HealthReason.UNVALIDATED,
                True,
                f"generation {gen.generation} was published with force and never compared "
                f"with the oracle{suffix}",
                pending=pending,
                tombs=tombs,
            )
        return report(
            HealthReason.FRESH,
            True,
            f"generation {gen.generation} is current{suffix}",
            pending=pending,
            tombs=tombs,
        )

    def base_damage(self, generation: Generation) -> str | None:
        """Why ``generation``'s storage cannot be trusted, in one sentence, or ``None``.

        A cheap invariant an index checks WITHOUT scanning its base: the row counts the build
        recorded, the agreement between two of its own tables, whatever that index can settle
        for the price of a metadata read.  It exists because the framework's promise is that a
        corrupt index does not make a query wrong, and only damage that RAISES gets caught for
        free; an accelerator that quietly returns half its candidates is the case that needs
        looking for.  :meth:`health` reports :attr:`HealthReason.DAMAGED_BASE` when this
        answers, :meth:`MaintenancePolicy.due` treats it as due, so ``maintain()`` rebuilds.

        The read paths call their own version of this rather than this method, because they
        hold a bare connection and no index object, and because they can usually fold the
        counts into the statement they were already running.

        The default returns ``None``: an index that cannot check anything cheaply says so by
        not overriding.  It must never scan the corpus -- that is :meth:`validate`.
        """
        return None

    # ------------------------------------------------------------------ storage hooks

    @abstractmethod
    def _build(self, generation: Generation) -> dict[str, Any] | None:
        """Materialise ``generation``'s storage beside the current one, from a snapshot of the
        source table taken in the calling transaction.  Returns stats to record (``"rows"`` is
        read by the maintenance policy).  ``generation.watermark`` is already set."""

    @abstractmethod
    def _validate(self, generation: Generation) -> ValidationReport:
        """Compare the generation's storage with the oracle and report."""

    @abstractmethod
    def _drop(self, generation: Generation) -> None:
        """Remove the generation's storage.  Must tolerate storage that is already gone."""

    def _erase(self, generation: Generation, tenant_id: int, doc_ids: Sequence[int]) -> int | None:
        """Delete these documents from ``generation``'s storage; the number of rows removed.

        Called by :meth:`erase` inside ``forget(hard=True)``'s transaction, once per generation
        including the unpublished and the still-building ones.  Return ``None`` when the
        storage cannot delete one document (an HNSW index that only rebuilds, for instance);
        the generation is then invalidated, which costs recall until the next rebuild and is
        the only honest alternative to leaving erased content in a live index.  The default
        returns ``None``, so an accelerator that has not thought about erasure fails closed.

        ``tenant_id`` is part of the key: a file-wide generation holds every tenant's documents
        and only this tenant's are being erased.
        """
        return None


class NullIndex(DerivedIndex):
    """A placeholder for an index no accelerator has registered yet.

    It records nothing, reports :class:`HealthReason.ABSENT` and refuses to build, so the
    verbs' hooks and the health surface exist from the first open and an accelerator only has
    to :meth:`IndexRegistry.register` its implementation under the same name.
    """

    source_table = None

    #: How to turn this index on, by name, for the health report to point at.  A placeholder
    #: that says only "absent" leaves an operator with nothing to do about it.
    HINTS: ClassVar[dict[str, str]] = {
        "fts": "anatid.fts.attach(db), or Anatid.open(accelerators=True)",
        "csr": "anatid.csr.attach_csr_index(db), or Anatid.open(accelerators=True)",
        "vector": 'anatid.vector.attach(db), or Anatid.open(vector_backend="duckdb_vss")',
    }

    def __init__(self, db: Any, name: str) -> None:
        super().__init__(db, name=name)

    def _build(self, generation: Generation) -> dict[str, Any] | None:
        raise IndexGenerationError(
            f"index {self.name!r} has no implementation registered; register a DerivedIndex "
            f"under that name before building",
            index=self.name,
        )

    def _validate(self, generation: Generation) -> ValidationReport:
        return ValidationReport(ok=False, generation=generation, detail="no implementation")

    def _drop(self, generation: Generation) -> None:
        return None

    def _erase(self, generation: Generation, tenant_id: int, doc_ids: Sequence[int]) -> int:
        return 0

    def health(
        self,
        tenant: Any = None,
        *,
        as_of: AsOf | _dt.datetime | None = None,
        policy: MaintenancePolicy | None = None,
        now: _dt.datetime | None = None,
    ) -> HealthReport:
        key = self.tenant_key(tenant) if tenant is not None else None
        return HealthReport(
            index_name=self.name,
            tenant_id=key,
            reason=HealthReason.ABSENT,
            usable=False,
            detail=(
                f"no implementation registered; enable it with {hint}"
                if (hint := self.HINTS.get(self.name))
                else "no implementation registered"
            ),
        )


class JournalOnlyIndex(DerivedIndex):
    """An index this handle holds no code for, reconstructed from its persisted definition.

    It can do everything that does not need the accelerator's storage: journal a write, report
    health, and invalidate generations a hard erasure could not reach.  It cannot build,
    validate or drop, and :meth:`_erase` returns ``None`` because it does not know where the
    storage is, so an erasure through this object takes the generations out of service rather
    than pretending they are clean.
    """

    def __init__(self, db: Any, definition: IndexDefinition) -> None:
        self.per_tenant = bool(definition.per_tenant)
        self.delta_mode = definition.delta_mode
        self.supports_delta = bool(definition.supports_delta)
        self.source_table = definition.source_table
        self.source_id_column = definition.source_id_column
        self.kind = definition.kind
        super().__init__(db, name=definition.index_name)
        self._definition = definition

    def definition(self) -> IndexDefinition:
        return self._definition

    def _build(self, generation: Generation) -> dict[str, Any] | None:
        raise IndexGenerationError(
            f"index {self.name!r} is defined in this file but this handle holds no "
            f"implementation for it; register one before building",
            index=self.name,
        )

    def _validate(self, generation: Generation) -> ValidationReport:
        return ValidationReport(
            ok=False, generation=generation, detail="no implementation on this handle"
        )

    def _drop(self, generation: Generation) -> None:
        return None

    def _erase(self, generation: Generation, tenant_id: int, doc_ids: Sequence[int]) -> int | None:
        return None


# --------------------------------------------------------------------------- maintenance


def maintain(
    index: DerivedIndex,
    tenant: Any = None,
    policy: MaintenancePolicy | None = None,
    *,
    now: _dt.datetime | None = None,
    raise_on_failure: bool = False,
    abandon_stale: bool = False,
) -> MaintenanceReport:
    """Decide whether ``index`` needs a new generation for ``tenant`` under ``policy``, and act.

    Explicitly callable; anatid runs no background thread.  The sequence when a rebuild is
    due: :meth:`DerivedIndex.build_next` (beside the current generation, reads keep using the
    old one), :meth:`DerivedIndex.validate` unless ``policy.validate`` is False, then
    :meth:`DerivedIndex.publish`.  A generation that fails validation is retired and the
    previous one stays published; ``raise_on_failure=True`` raises
    :class:`~anatid.errors.IndexValidationError` instead of reporting it.  A build in progress
    is skipped; ``abandon_stale=True`` first clears ``building`` rows no build in this process
    owns (:meth:`DerivedIndex.abandon_builds`).
    """
    pol = policy or MaintenancePolicy()
    unimplemented = isinstance(index, (NullIndex, JournalOnlyIndex))
    if abandon_stale and not unimplemented:
        index.abandon_builds(tenant)
    before = index.health(tenant, policy=pol, now=now)
    key = before.tenant_id
    if unimplemented:
        return MaintenanceReport(
            index.name, key, "skipped", "no implementation registered on this handle", before
        )
    if before.reason is HealthReason.LOAD_FAILURE:
        return MaintenanceReport(index.name, key, "skipped", before.detail, before)
    if before.reason is HealthReason.REBUILD_IN_PROGRESS or index.is_building(tenant):
        return MaintenanceReport(index.name, key, "skipped", "a rebuild is in progress", before)
    due = pol.due(before)
    if due is None:
        return MaintenanceReport(index.name, key, "none", before.detail or "not due", before)
    gen = index.build_next(tenant, now=now)
    verdict: ValidationReport | None = None
    if pol.validate:
        verdict = index.validate(gen)
        if not verdict.ok:
            index.retire(verdict.generation)
            after = index.health(tenant, policy=pol, now=now)
            if raise_on_failure:
                raise IndexValidationError(
                    f"index {index.name!r}: generation {gen.generation} failed validation: "
                    f"{verdict.detail}",
                    report=verdict,
                    index=index.name,
                    generation=gen.generation,
                )
            return MaintenanceReport(
                index.name,
                key,
                "validation_failed",
                due,
                before,
                after,
                verdict.generation,
                verdict,
            )
        published = index.publish(verdict.generation)
        action = "published"
    else:
        published = index.publish(gen, force=True)
        action = "published_unvalidated"
    after = index.health(tenant, policy=pol, now=now)
    return MaintenanceReport(index.name, key, action, due, before, after, published, verdict)


# --------------------------------------------------------------------------- registry


class IndexRegistry:
    """The derived indexes of one database: the definitions in the FILE, the objects on this
    handle.

    The two are deliberately separate.  :meth:`register` writes an :class:`IndexDefinition` into
    :data:`~anatid.schema.INDEX_REGISTRY_TABLE`, and every handle opened on that file reads it,
    so a write through a handle that holds no code for the index still journals for it.  Before
    that, index registries were per handle: a second handle had only :class:`NullIndex`
    placeholders, its writes journalled nothing, and the first handle's next read silently
    missed them.

    A :class:`NullIndex` placeholder is created for each of :data:`DEFAULT_NAMES` and is NOT
    persisted: it says "the framework knows this accelerator by name", not "this file has such
    an index".  The verbs call :meth:`emit` inside their transactions; it costs one dict lookup
    against a per-epoch cache of the definitions when nothing derives from the written table.
    """

    DEFAULT_NAMES: tuple[str, ...] = ("fts", "csr", "vector")

    _COLUMNS = (
        "index_name, kind, per_tenant, source_table, source_id_column, delta_mode, "
        "supports_delta, params, created_at, enabled"
    )

    def __init__(self, db: Any, *, placeholders: Sequence[str] | None = None) -> None:
        self.db = db
        self.enabled = True
        self._indexes: dict[str, DerivedIndex] = {}
        self._lock = threading.RLock()
        self._key = database_key(db)
        #: ``(epoch, by name, enabled by source table, every definition by source table)``.
        #: The enabled map is what the write path reads, so a verb writing into a table no
        #: index derives from costs one dict lookup.  The second map keeps the DISABLED
        #: definitions too, because erasure and invalidation follow the storage rather than the
        #: journal (:meth:`_defined_over`).
        self._defs: tuple[
            int,
            dict[str, IndexDefinition],
            dict[str, tuple[IndexDefinition, ...]],
            dict[str, tuple[IndexDefinition, ...]],
        ]
        self._defs = (-1, {}, {}, {})
        for name in self.DEFAULT_NAMES if placeholders is None else placeholders:
            self._indexes[str(name)] = NullIndex(db, str(name))

    def __repr__(self) -> str:
        return f"<IndexRegistry {sorted(self._indexes)}>"

    # ------------------------------------------------------------------ definitions

    def _read_definitions(self) -> dict[str, IndexDefinition]:
        try:
            rows = self.db.execute(
                f"SELECT {self._COLUMNS} FROM {_REGISTRY} ORDER BY index_name"
            ).fetchall()
        except Exception:  # noqa: BLE001 - a file older than the registry table has no indexes
            return {}
        out: dict[str, IndexDefinition] = {}
        for r in rows:
            out[str(r[0])] = IndexDefinition(
                index_name=str(r[0]),
                kind="" if r[1] is None else str(r[1]),
                per_tenant=bool(r[2]),
                source_table=None if r[3] is None else str(r[3]),
                source_id_column="memory_id" if r[4] is None else str(r[4]),
                delta_mode="table" if r[5] is None else str(r[5]),
                supports_delta=bool(r[6]),
                params=_json(r[7]),
                created_at=r[8],
                enabled=bool(r[9]),
            )
        return out

    def _cached(
        self,
    ) -> tuple[
        dict[str, IndexDefinition],
        dict[str, tuple[IndexDefinition, ...]],
        dict[str, tuple[IndexDefinition, ...]],
    ]:
        epoch = _epoch(self._key)
        with self._lock:
            cached_epoch, by_name, by_table, every = self._defs
            if cached_epoch == epoch:
                return by_name, by_table, every
        by_name = self._read_definitions()
        grouped: dict[str, list[IndexDefinition]] = {}
        all_grouped: dict[str, list[IndexDefinition]] = {}
        for d in by_name.values():
            if d.source_table is None:
                continue
            all_grouped.setdefault(d.source_table, []).append(d)
            if d.enabled:
                grouped.setdefault(d.source_table, []).append(d)
        frozen = {t: tuple(v) for t, v in grouped.items()}
        frozen_all = {t: tuple(v) for t, v in all_grouped.items()}
        with self._lock:
            self._defs = (epoch, by_name, frozen, frozen_all)
        return by_name, frozen, frozen_all

    def definitions(self, *, enabled_only: bool = False) -> dict[str, IndexDefinition]:
        """The index definitions recorded in the file, by name, through the per-epoch cache."""
        by_name, _by_table, _every = self._cached()
        if enabled_only:
            return {n: d for n, d in by_name.items() if d.enabled}
        return dict(by_name)

    def _persist(self, definition: IndexDefinition) -> None:
        """Write (or replace) one definition.  No constraint to lean on, so delete then insert.

        A read-only handle defines nothing: it cannot write, so its journal is never the one at
        risk, and the definition it would have written is the writer's to make.  The object
        stays on the handle and can still pin and read a generation.

        Turning ``enabled`` off or on takes every generation of the index out of service, in the
        same transaction.  Journalling stops at the moment it goes off, so writes after that are
        in no delta; switching it back on cannot know what was written in between, and a
        generation that stayed marked validated across that window would be read as complete
        when it is not.  A first definition invalidates nothing: there is nothing to be stale.
        """
        if getattr(self.db, "read_only", False):
            return
        previous = self.definitions().get(definition.index_name)
        with self.db.transaction():
            self.db.execute(
                f"DELETE FROM {_REGISTRY} WHERE index_name = ?", [definition.index_name]
            )
            self.db.execute(
                f"INSERT INTO {_REGISTRY} ({self._COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    definition.index_name,
                    definition.kind,
                    bool(definition.per_tenant),
                    definition.source_table,
                    definition.source_id_column,
                    definition.delta_mode,
                    bool(definition.supports_delta),
                    json.dumps(definition.params or {}, default=str),
                    to_utc_naive(definition.created_at) or utcnow(),
                    bool(definition.enabled),
                ],
            )
            if previous is not None and bool(previous.enabled) != bool(definition.enabled):
                switched = "on" if definition.enabled else "off"
                self._invalidate_index(
                    definition.index_name,
                    f"journalling for this index was switched {switched}, so writes outside "
                    f"that window are in no delta",
                )
        _bump_epoch(self._key)

    def define(self, definition: IndexDefinition) -> IndexDefinition:
        """Record an index in the file without holding an implementation for it.

        The declarative half of :meth:`register`, for a process that maintains the journal for
        an accelerator another process builds.
        """
        self._persist(definition)
        return definition

    # ------------------------------------------------------------------ objects

    def register(self, index: DerivedIndex) -> DerivedIndex:
        """Add ``index`` to this handle and record its definition in the file.

        A :class:`NullIndex` is a placeholder rather than an index, so it is kept on the handle
        and not written to the file.
        """
        if not isinstance(index, DerivedIndex):
            raise TypeError("register() takes a DerivedIndex")
        with self._lock:
            self._indexes[index.name] = index
        if not isinstance(index, NullIndex) and index.source_table is not None:
            self._persist(index.definition())
        return index

    def unregister(
        self, name: str, *, forget: bool = False, retire: bool = False
    ) -> DerivedIndex | None:
        """Drop the object from this handle and stop the file journalling for it.

        The definition is DISABLED rather than deleted, so the row remains as a record that the
        file once had this index and an operator can see why generations are lying about.
        ``forget=True`` deletes the definition and its journal rows outright.

        Either way every generation of the index is INVALIDATED in the same transaction: from
        this moment nothing journals for it, so no generation can be merged into a complete
        answer again until it is rebuilt.  A hard erasure still reaches its storage, because
        :meth:`erase` follows the storage rather than the ``enabled`` flag.

        ``retire=True`` also drops that storage (:meth:`DerivedIndex.drop_all`), which is what
        an index holding indexed CONTENT needs: left behind, its base is a copy of everything
        the index had seen, and once this handle has dropped the object nothing can delete one
        document from it any more, so a later ``forget(hard=True)`` can only report the
        generation as unclean.  It needs this handle to hold the implementation, and raises
        when it does not, rather than reporting a removal that did not happen.
        """
        # Storage first, and BEFORE the object leaves the handle: a refusal here has to leave
        # the registry exactly as it was, or a caller that retries has already lost the object
        # that could have dropped the storage.
        if retire:
            with self._lock:
                out = self._indexes.get(str(name))
            if out is None or isinstance(out, NullIndex):
                raise IndexGenerationError(
                    f"index {name!r}: this handle holds no implementation for it, so its "
                    f"generation storage cannot be dropped from here; unregister it from the "
                    f"handle that does",
                    index=str(name),
                )
            out.drop_all()
        with self._lock:
            out = self._indexes.pop(str(name), None)
        if str(name) in self.definitions():
            with self.db.transaction():
                if forget:
                    self.db.execute(f"DELETE FROM {_REGISTRY} WHERE index_name = ?", [str(name)])
                    self.db.execute(f"DELETE FROM {_JOURNAL} WHERE index_name = ?", [str(name)])
                else:
                    self.db.execute(
                        f"UPDATE {_REGISTRY} SET enabled = FALSE WHERE index_name = ?",
                        [str(name)],
                    )
                self._invalidate_index(
                    str(name),
                    "the index was unregistered, so nothing journals for it any more",
                )
            _bump_epoch(self._key)
        return out

    def get(self, name: str) -> DerivedIndex | None:
        return self._indexes.get(str(name))

    def __getitem__(self, name: str) -> DerivedIndex:
        return self._indexes[str(name)]

    def __contains__(self, name: object) -> bool:
        return name in self._indexes

    def __iter__(self) -> Iterator[DerivedIndex]:
        return iter(list(self._indexes.values()))

    def __len__(self) -> int:
        return len(self._indexes)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._indexes)

    def _implementation(self, definition: IndexDefinition) -> DerivedIndex | None:
        """This handle's object for a definition, or ``None`` when it holds only a placeholder."""
        idx = self._indexes.get(definition.index_name)
        if idx is None or isinstance(idx, NullIndex):
            return None
        return idx

    def _for_table(self, table: str) -> tuple[IndexDefinition, ...]:
        """The enabled definitions over ``table``: one dict lookup on the cached map.

        The write path.  A disabled definition is deliberately absent: nothing journals for it.
        """
        return self._cached()[1].get(table, ())

    def _defined_over(self, table: str) -> tuple[IndexDefinition, ...]:
        """Every definition over ``table``, the DISABLED ones included.

        Erasure and invalidation follow the storage, not the journal.  A disabled definition
        still has generations, and their storage still holds whatever the index put there, so a
        hard purge has to reach it and a bulk load has to take it out of service.  Using the
        enabled map here would leave an unregistered index holding erased content in a base a
        later re-registration would trust.
        """
        return self._cached()[2].get(table, ())

    def _invalidate_index(self, index_name: str, reason: str) -> int:
        """Clear ``validated`` on every generation of one index.  Returns how many rows changed.

        Every generation, not only the published one: an unpublished generation that keeps its
        ``validated`` flag through a journal gap can be published later and read as complete.
        A ``building`` row is left alone; it belongs to a build that has not recorded its
        verdict yet, and the erasure counter is what reaches that one
        (:meth:`DerivedIndex.build_next`).
        """
        row = self.db.execute(
            f"UPDATE {_GEN} SET validated = FALSE, published_unvalidated = FALSE, notes = ? "
            f"WHERE index_name = ? AND notes IS DISTINCT FROM 'building'",
            [f"invalidated: {reason}"[:500], str(index_name)],
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def wants(self, table: str) -> bool:
        """True when some index DEFINED IN THIS FILE derives from ``table``."""
        if not self.enabled:
            return False
        return bool(self._for_table(str(table)))

    def emit(
        self,
        kind: str,
        table: str,
        tenant_id: int,
        doc_ids: Sequence[int],
        *,
        at: _dt.datetime,
        reason: str | None = None,
    ) -> IndexEvent | None:
        """Report a write on ``table`` to every index the FILE defines over it.

        Call inside the transaction that made the write.  An index this handle implements gets
        :meth:`DerivedIndex.on_write`; one it does not gets :func:`record_event` against the
        persisted definition, which is the same journal row.  Returns the event dispatched, or
        ``None`` when no index wanted it.
        """
        if not self.enabled:
            return None
        defs = self._for_table(str(table))
        if not defs:
            return None
        ids = tuple(int(d) for d in doc_ids)
        if not ids:
            return None
        event = IndexEvent(
            kind=str(kind),
            table=str(table),
            tenant_id=int(tenant_id),
            doc_ids=ids,
            at=to_utc_naive(at) or utcnow(),
            reason=reason,
        )
        for definition in defs:
            idx = self._implementation(definition)
            if idx is not None:
                idx.on_write(event)
            else:
                record_event(self.db, definition, event)
        return event

    def erase(
        self,
        table: str,
        tenant_id: int,
        doc_ids: Sequence[int],
        *,
        at: _dt.datetime,
        reason: str | None = None,
    ) -> ErasureResult:
        """Erase documents from every index the file defines over ``table``.

        Called by ``forget(hard=True)`` inside the purge transaction, after the canonical rows
        are gone.  Every index gets the ``PURGE`` event first (an accelerator with private
        storage reacts to it), then its generations are cleaned or invalidated and its journal
        rows deleted.  An index this handle does not implement cannot delete from a built
        generation, so every generation of it is invalidated: correct, and visible in
        :meth:`health` as an invalidated generation rather than as silent retention.

        DISABLED definitions are erased too (:meth:`_defined_over`).  Unregistering an index
        stops the journal, not the storage: its generations still hold whatever it built, and a
        purge that skipped them would leave the erased document in the file, which is the one
        thing ``forget(hard=True)`` promises not to do.
        """
        ids = tuple(int(d) for d in doc_ids)
        if not self.enabled or not ids:
            return ErasureResult()
        defs = self._defined_over(str(table))
        if not defs:
            return ErasureResult()
        event = IndexEvent(
            kind=PURGE,
            table=str(table),
            tenant_id=int(tenant_id),
            doc_ids=ids,
            at=to_utc_naive(at) or utcnow(),
            reason=reason,
        )
        out = ErasureResult()
        for definition in defs:
            idx = self._implementation(definition)
            if idx is not None:
                idx.on_write(event)
                out = out + idx.erase(int(tenant_id), ids, reason=reason)
            else:
                out = out + JournalOnlyIndex(self.db, definition).erase(
                    int(tenant_id),
                    ids,
                    reason=(
                        f"{reason}, from a handle with no implementation for this index"
                        if reason
                        else "from a handle with no implementation for this index"
                    ),
                )
        return out

    def invalidate(self, table: str, *, reason: str) -> int:
        """Clear ``validated`` on every generation derived from ``table``.

        For writes that bypass the verbs (a bulk load): the journal cannot know about them, so
        the published generation stops being usable until :func:`maintain` rebuilds it.  Driven
        by the persisted definitions, so it covers indexes this handle holds no code for and
        indexes whose definition is disabled (:meth:`_defined_over`) -- their storage is still
        in the file and a later re-registration would otherwise trust it.  Unpublished
        generations are invalidated as well, because one of them may be published later.
        Returns the number of generations affected.
        """
        n = 0
        for definition in self._defined_over(str(table)):
            n += self._invalidate_index(definition.index_name, reason)
        if n:
            _bump_epoch(self._key)
        return n

    def announce_erasure(self, table: str | None = None) -> int:
        """Tell the builds running right now that a hard erasure is starting.  Returns how many
        indexes were told.

        The counter :meth:`DerivedIndex.build_next` compares before and after its snapshot.  It
        is bumped here, before the purge opens its transaction, rather than only when the
        erasure reaches an index: a build that commits while the purge's snapshot still calls it
        ``building`` is skipped by the purge, and a bump that arrived after that build had
        checked would leave the erased document in a base marked validated.  ``table`` narrows
        it to the indexes over one source; the default tells every index the file defines,
        which is what a purge that touches several tables wants.
        """
        names = (
            [d.index_name for d in self._defined_over(str(table))]
            if table is not None
            else list(self.definitions())
        )
        for name in names:
            _bump_erasures(self._key, name)
        return len(names)

    def lifecycle(self) -> threading.RLock:
        """The per-file generation lifecycle lock (:func:`lifecycle_lock`), as a context manager.

        ``forget(hard=True)`` holds it across its whole transaction.  Without that, a purge
        transaction could open before a build announced its generation (so the purge skips that
        generation) and commit after the build's snapshot was taken (so the build's base still
        holds the erased document), and nothing would ever notice: the base is not wrong, but it
        keeps content an erasure was supposed to remove.  Held across the purge, the build's
        snapshot is either taken after the purge committed, or the build starts first and the
        erasure counter catches it.
        """
        return lifecycle_lock(self._key)

    def journal_only(self) -> dict[str, JournalOnlyIndex]:
        """One :class:`JournalOnlyIndex` per definition this handle holds no code for.

        The file knows about more indexes than any one handle implements, and an operator on
        the handle that does not implement one still wants to see whether it is fresh.
        """
        out: dict[str, JournalOnlyIndex] = {}
        for name, definition in self.definitions(enabled_only=True).items():
            if self._implementation(definition) is not None:
                continue
            with contextlib.suppress(Exception):
                out[name] = JournalOnlyIndex(self.db, definition)
        return out

    def health(
        self,
        tenant: Any = None,
        *,
        as_of: AsOf | _dt.datetime | None = None,
        policy: MaintenancePolicy | None = None,
    ) -> dict[str, HealthReport]:
        """A :class:`HealthReport` per index: this handle's objects, plus every definition in
        the file this handle holds no code for."""
        out = {
            name: idx.health(tenant, as_of=as_of, policy=policy)
            for name, idx in self.journal_only().items()
        }
        out.update(
            {
                name: idx.health(tenant, as_of=as_of, policy=policy)
                for name, idx in self._indexes.items()
            }
        )
        return out

    def maintain(
        self,
        tenant: Any = None,
        policy: MaintenancePolicy | None = None,
        *,
        now: _dt.datetime | None = None,
    ) -> dict[str, MaintenanceReport]:
        """:func:`maintain` every index REGISTERED ON THIS HANDLE, for ``tenant``.

        Not every definition in the file, unlike :meth:`health`: building a generation needs the
        accelerator's code, so a definition this handle holds no implementation for is left to
        the handle that does.
        """
        return {name: maintain(idx, tenant, policy, now=now) for name, idx in self._indexes.items()}
