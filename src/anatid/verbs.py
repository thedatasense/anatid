"""The memory verbs.

Each **write** verb (``remember``, ``supersede``, ``forget``, ``relate``, ``unrelate``,
``reinforce``, ``episode``, ``entity_id``) is one transaction.  ``prune`` is not -- it is a query
plus one transaction per memory, and says so.  **Read** verbs (``recall``, ``recall_2hop``,
``context``, ``get``, ``versions``, ``provenance``, ``stats``) open no transaction: ``recall``
runs its staleness probe, its arms, its hydration and its ABOUT lookup as separate statements,
so another thread's commit can land between them.  Wrap the call in ``with db.transaction():``
when you need one snapshot.

Corrections are versions
------------------------
No verb rewrites the valid interval of a row that a read can see.  ``supersede``, soft
``forget``, ``unrelate`` and a ``reinforce`` that changes ``confidence`` close the current
version on the transaction axis (``tx_to = now``) and insert the next version of the same
logical id, with ``tx_from = now`` and the corrected columns, in the same transaction
(:meth:`MemoryVerbs._close_memory`, :meth:`MemoryVerbs._close_edges`).  The transaction-time
half of every ``as_of`` therefore has something to select: ``as_of(valid_time=t4, tx_time=t2)``
after a forget at ``t3`` returns the version the database held at ``t2``, open-ended.  The one
in-place update left is the usage counters ``access_count`` / ``last_access_at``, which
``reinforce`` bumps on the live version; they are not bitemporal and the docstring says so.
The ``UPDATE`` that closes a version is also the compare-and-swap: a concurrent correction of
the same version aborts with :class:`~anatid.errors.ConflictError`, and a correction that
arrives after the version is already closed matches nothing and is reported rather than
applied twice.

Every value a verb writes or filters on travels as a bound parameter -- no statement is ever
built by concatenating caller data.  (Two narrow exceptions, both documented where they occur:
integer id lists in an ``IN`` clause and the C++ extension's bind-time arguments, where a bound
parameter defeats the index or the function.  Both go through ``int()`` first, so only digits
can reach the SQL.)  DDL is the other half of that promise: identifiers go through
:func:`anatid.schema.quote_ident` and column *types* through :func:`anatid.schema.check_type`.

Every verb takes an explicit ``now=`` (and reads take ``as_of=``) defaulting to UTC now, so a run
can be made bit-for-bit deterministic by passing timestamps in.

The verbs live on a mixin so they are methods of :class:`anatid.Anatid`; the module-level
functions of the same name are thin wrappers for callers who prefer ``verbs.remember(db, ...)``.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import logging
import math
import numbers
from typing import Any, Callable, Sequence, TypeVar

import duckdb

from . import derived as _derived
from . import erasure as _erasure
from . import recall as _recall
from . import schema as _schema
from ._typing import VerbHostMixin
from .errors import (
    ConflictError,
    DuplicateIdError,
    EmbeddingDimensionError,
    EmbeddingValueError,
    NotFoundError,
    RangeError,
    ValidationError,
)
from .ids import new_id
from .schema import (
    EDGE_ABOUT_COLUMNS,
    EDGE_RELATES_COLUMNS,
    ENTITY_COLUMNS,
    EPISODE_COLUMNS,
    MEMORY_COLUMNS,
    VERSION_COLUMN,
    VERSIONED_TABLES,
    entity_key_sql,
    memory_select,
    version_expr,
)
from .types import (
    RELATES_TO,
    SUPERSEDES,
    AsOf,
    Edge,
    Entity,
    Episode,
    ForgetReceipt,
    Memory,
    Namespace,
    Provenance,
    PruneReport,
    RecallHits,
    utcnow,
    to_utc_naive,
)
from .visibility import Visibility, current_row_sql, live_row_sql, tenant_sql

log = logging.getLogger("anatid.verbs")

__all__ = [
    "MemoryVerbs",
    "AsOfView",
    "MAX_CONFIDENCE",
    "MIN_CONFIDENCE",
    "remember",
    "recall",
    "recall_2hop",
    "context",
    "supersede",
    "reinforce",
    "forget",
    "unrelate",
    "prune",
    "as_of",
    "provenance",
    "versions",
]

def _mem_select(con, *, embedding: bool = True) -> str:
    """The select list for one memory row, adapted to what this file actually has.

    A handle that could not run the migration ladder (``read_only=True`` or ``ensure=False``)
    can be looking at a schema-v3 file, whose ``memories`` has no ``version`` column.  Naming
    it would fail to bind, so :func:`anatid.schema.memory_select` renders ``1 AS version``
    there instead: every row of a pre-v4 table is version 1.
    """
    return memory_select(con, embedding=embedding)

_ENT_SELECT = ", ".join(ENTITY_COLUMNS)
_EPI_SELECT = ", ".join(EPISODE_COLUMNS)

_INSERT_MEMORY_TPL = (
    "INSERT INTO memories (memory_id, tenant_id, content, kind, embedding, created_at, "
    "valid_from, valid_to, tx_from, tx_to, writer, episode_id, confidence, access_count, "
    "last_access_at) VALUES (?, ?, ?, ?, ?::FLOAT[{dim}], ?, ?, NULL, ?, NULL, ?, ?, ?, 0, NULL)"
)

_INSERT_ABOUT = (
    "INSERT INTO edges_about (edge_id, src, dst, tenant_id, weight, valid_from, valid_to, "
    "tx_from, tx_to, writer, episode_id, confidence) "
    "SELECT eid, ?, dst, ?, ?, ?, NULL, ?, NULL, ?, ?, ? "
    "FROM (SELECT unnest(?::BIGINT[]) AS eid, unnest(?::BIGINT[]) AS dst)"
)

#: Entity lookup by the canonical key the UNIQUE index is built on.  The raw name is BOUND and
#: DuckDB canonicalises it: comparing against a key computed in Python would be comparing two
#: different Unicode case-folding implementations, and a miss there ends in a constraint error.
#: No time predicate on purpose: the uniqueness constraint spans history (see ``entity_id``).
_ENTITY_BY_KEY_SQL = (
    f"SELECT entity_id FROM entities WHERE {tenant_sql()} AND entity_key = "
    + entity_key_sql("?")
    + " ORDER BY entity_id LIMIT 1"
)

#: The write-side compare-and-swap guard: the version of a memory or edge that is current on
#: both axes, which is the only version a correction may close.  From :mod:`anatid.visibility`,
#: like every other predicate over the bitemporal columns.
_CURRENT = current_row_sql()

#: The live version of a logical id (no correction has closed it), whatever its valid interval.
#: Id lookups and the usage counters address this one.
_LIVE = live_row_sql()

#: ``coalesce(version, 1)``: a table the 3->4 migration altered has a nullable version column
#: (DuckDB cannot add a NOT NULL column), and a NULL written there by raw SQL means version 1.
_VERSION = f"coalesce({VERSION_COLUMN}, 1)"


class _Expr:
    """A SQL expression with its own bound parameters, for :func:`_successor` overrides."""

    __slots__ = ("sql", "params")

    def __init__(self, sql: str, *params: Any) -> None:
        self.sql = sql
        self.params = list(params)


def _successor(columns: Sequence[str], **values: Any) -> tuple[str, str, list]:
    """The column list, SELECT list and parameters that copy one version into the next.

    Used as ``INSERT INTO t (<columns>) SELECT <select> FROM t WHERE <the version closed>``.
    Every column keeps its value except the ones named in ``values`` (a bound value, or an
    :class:`_Expr`) and the two the rule fixes: ``version`` becomes ``coalesce(version, 1) + 1``
    and ``tx_to`` becomes ``NULL``.  Parameters come back in column order, so the caller
    appends its WHERE parameters after them.
    """
    parts: list[str] = []
    params: list = []
    for col in columns:
        if col == VERSION_COLUMN:
            parts.append(f"{_VERSION} + 1")
        elif col == "tx_to":
            parts.append("NULL")
        elif col in values:
            value = values[col]
            if isinstance(value, _Expr):
                parts.append(value.sql)
                params += value.params
            else:
                parts.append("?")
                params.append(value)
        else:
            parts.append(col)
    return ", ".join(columns), ", ".join(parts), params


_INSERT_ENTITY = (
    "INSERT INTO entities (entity_id, tenant_id, kind, name, valid_from, valid_to, "
    "tx_from, tx_to, writer, episode_id, confidence) "
    "VALUES (?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?, ?)"
)

_MAX_CHAIN = 10_000  # cycle guard for the SUPERSEDES walk

#: Inclusive bounds for every ``confidence`` and ABOUT-edge ``weight`` anatid writes.
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0

#: How many times a verb re-runs its whole transaction after losing an entity-creation race.
#: Each attempt re-reads, so the loser finds the winner; 8 is far past what 100 racing writers
#: need (the test in tests/test_integrity.py converges in 1-2).
ENTITY_RACE_RETRIES = 8

_T = TypeVar("_T")


def _count(result) -> int:
    """Row count reported by a DuckDB DML statement."""
    row = result.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _ids(result) -> list[int]:
    """The ids a ``... RETURNING <id>`` DML statement produced, in statement order."""
    return [int(r[0]) for r in result.fetchall()]


def _distinct(ids: Sequence[int]) -> list[int]:
    """``ids`` with repeats removed, first occurrence kept.  A statement over a versioned table
    returns one row per version, and an index event or a receipt counts logical ids."""
    return list(dict.fromkeys(int(i) for i in ids))


# --------------------------------------------------------------------------- argument checks
#
# Every one of these runs BEFORE any statement does, and raises a subclass of
# anatid.errors.ValidationError -- which is also a ValueError, so 0.1.0 callers that catch
# ValueError keep working.  anatid 0.1.0 accepted confidence=-1, confidence=2, NaN embeddings
# and k=0 and wrote them; the reviewer confirmed all three.


def _check_number(field: str, value: Any) -> float:
    """Coerce to a finite float or raise :class:`~anatid.errors.ValidationError`."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a number, got {type(value).__name__}") from None
    if not math.isfinite(out):
        raise ValidationError(f"{field} must be a finite number, got {value!r}")
    return out


def _check_unit(field: str, value: Any) -> float:
    """A confidence / weight: a finite number in ``[0, 1]``."""
    out = _check_number(field, value)
    if not (MIN_CONFIDENCE <= out <= MAX_CONFIDENCE):
        raise RangeError(
            f"{field} must be in [{MIN_CONFIDENCE}, {MAX_CONFIDENCE}], got {out!r}",
            field=field,
            value=out,
            low=MIN_CONFIDENCE,
            high=MAX_CONFIDENCE,
        )
    return out


def _check_int(field: str, value: Any) -> int:
    """An integer argument.  ``bool`` is refused (``k=True`` is a mistake, not a 1), and any
    :class:`numbers.Integral` is accepted so a ``numpy.int64`` from an index lookup works."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValidationError(f"{field} must be an int, got {type(value).__name__}")
    return int(value)


def _check_positive(field: str, value: Any) -> int:
    """A row limit / k / candidate count: an integer >= 1.

    Zero and negatives are rejected rather than clamped: ``k=0`` silently returns nothing, and
    ``LIMIT -1`` is a DuckDB error thrown from deep inside a query the caller did not write.
    """
    out = _check_int(field, value)
    if out < 1:
        raise RangeError(f"{field} must be >= 1, got {out}", field=field, value=out, low=1)
    return out


def _check_non_negative(field: str, value: Any) -> int:
    """A hop count: an integer >= 0."""
    out = _check_int(field, value)
    if out < 0:
        raise RangeError(f"{field} must be >= 0, got {out}", field=field, value=out, low=0)
    return out


def _check_row_id(field: str, value: Any) -> int:
    """An explicit id: a positive integer (the allocator's ids are 63-bit and positive)."""
    out = _check_int(field, value)
    if out <= 0:
        raise RangeError(
            f"{field} must be a positive integer, got {out}", field=field, value=out, low=1
        )
    return out


def _check_closes_after_open(
    verb: str, memory_id: int, valid_from: _dt.datetime | None, at: _dt.datetime
) -> None:
    """Refuse to close a memory's valid interval before it opened.

    ``supersede`` and soft ``forget`` both write ``valid_to = now``.  With ``now`` earlier than
    the row's ``valid_from`` the interval becomes ``[from, to)`` with ``to < from`` -- one that
    was never open, which no ``as_of`` query can return and which
    :meth:`~anatid.Anatid.doctor` reports as ``timestamp_order``.  anatid 0.1.0 wrote it
    without comment; this is the verb-boundary half of that check.
    """
    if valid_from is not None and at < valid_from:
        raise RangeError(
            f"{verb}: now={at.isoformat()} is before memory {memory_id}'s "
            f"valid_from={valid_from.isoformat()}, which would close its interval before it "
            f"opened. Pass now >= valid_from, or forget(hard=True) to erase the row instead.",
            field="now",
            value=at,
        )


def _check_records_after(
    verb: str, what: str, tx_from: _dt.datetime | None, at: _dt.datetime
) -> None:
    """Refuse to record a correction before the version it corrects was recorded.

    A correction closes the current version's transaction interval at ``now`` and opens the
    next one there.  With ``now`` earlier than that version's ``tx_from`` the closed interval
    is ``[tx_from, now)`` with ``now < tx_from``: never live, invisible to every ``as_of``, and
    the successor would claim to have been recorded before the row it corrects.  Transaction
    time is when the database recorded something, so it does not run backwards.
    """
    if tx_from is not None and at < tx_from:
        raise RangeError(
            f"{verb}: now={at.isoformat()} is before {what}'s tx_from={tx_from.isoformat()}, "
            f"which would record the correction before the version it corrects. Pass "
            f"now >= tx_from; transaction time does not run backwards.",
            field="now",
            value=at,
        )


# --------------------------------------------------------------------------- entity races


def _is_entity_race(exc: BaseException) -> bool:
    """True when ``exc`` is a lost race to create an entity, and re-running will resolve it.

    Two shapes, both from the ``UNIQUE (tenant_id, entity_key)`` index:

    * ``ConstraintException: Duplicate key "tenant_id: 1, entity_key: ada" violates unique
      constraint`` -- the winner's row was already visible when we inserted;
    * ``TransactionException: Failed to commit: PRIMARY KEY or UNIQUE constraint violation`` --
      the winner committed while we held an older snapshot, so our INSERT saw nothing to
      conflict with and the check happened at COMMIT.

    Both are re-runnable *because the unit of work re-reads*: the next attempt's SELECT finds
    the winner and inserts nothing.  Nothing else is treated as retryable here -- a
    write-write ``ConflictError`` on ``supersede`` is a different race with a different answer.
    """
    cause = getattr(exc, "cause", None)
    if cause is not None:
        exc = cause
    if not isinstance(exc, (duckdb.ConstraintException, duckdb.TransactionException)):
        return False
    msg = str(exc).lower()
    return "duplicate key" in msg or "unique constraint" in msg


def _in_transaction(db: Any) -> bool:
    """True when the host already has a transaction open on this thread.

    ``getattr`` with a default on purpose: the verbs only *require* the six members of
    :class:`anatid._typing.VerbHost`, and a test double that does not offer ``in_transaction``
    simply never gets the retry (it also never nests, so that is the right answer).
    """
    return bool(getattr(db, "in_transaction", False))


class AsOfView:
    """Read-only view of a database at one point in time (``db.as_of(t)``).

    Every read on this object carries the same :class:`~anatid.types.AsOf` scope.  There is no
    engine support behind it: DuckDB has no ``AS OF SYSTEM TIME``, so this simply passes the
    scope to the verbs, which compile it into a WHERE clause over the bitemporal columns.

    ::

        before = db.as_of(t0)
        before.recall_2hop(seed)  # the answer as of t0
        before.get(mid).content  # the content believed at t0
    """

    __slots__ = ("_db", "scope")

    def __init__(self, db: "MemoryVerbs", scope: AsOf) -> None:
        self._db = db
        self.scope = scope

    def __repr__(self) -> str:
        return f"<AsOfView valid_time={self.scope.valid_time} tx_time={self.scope.tx_time}>"

    def recall(self, query=None, **kw) -> RecallHits:
        kw.setdefault("as_of", self.scope)
        return self._db.recall(query, **kw)

    def recall_2hop(self, seed_entity, **kw) -> list[Memory]:
        kw.setdefault("as_of", self.scope)
        return self._db.recall_2hop(seed_entity, **kw)

    def recall_2hop_ids(self, seed_entity, **kw):
        kw.setdefault("as_of", self.scope)
        return self._db.recall_2hop_ids(seed_entity, **kw)

    def context(self, entity, **kw) -> list[Memory]:
        kw.setdefault("as_of", self.scope)
        return self._db.context(entity, **kw)

    def get(self, memory_id, **kw) -> Memory | None:
        kw.setdefault("as_of", self.scope)
        return self._db.get(memory_id, **kw)

    def entities_of(self, memory_id, **kw) -> list[Entity]:
        kw.setdefault("as_of", self.scope)
        return self._db.entities_of(memory_id, **kw)

    def provenance(self, memory_id, **kw) -> Provenance:
        return self._db.provenance(memory_id, **kw)

    def visibility(self, tenant: int | Namespace | None = None) -> Visibility:
        """The :class:`~anatid.visibility.Visibility` every read on this view applies."""
        return self._db.visibility(tenant=tenant, as_of=self.scope)


class MemoryVerbs(VerbHostMixin):
    """The verb surface of :class:`anatid.Anatid`.

    The mixin expects its host to provide ``execute()``, ``transaction()``, ``connection``,
    ``resolve_tenant()``, ``config`` and ``csr`` -- the contract is
    :class:`anatid._typing.VerbHost`, and :class:`anatid.database.Anatid` is the implementation.
    :data:`~anatid._typing.VerbHostMixin` is that Protocol to a type checker and plain ``object``
    at run time, so this class's MRO is unchanged; it replaces the bodyless ``def execute(...):
    ...`` placeholders that used to live here and that a checker read as "returns ``None``".
    """

    # ------------------------------------------------------------------ internals

    def visibility(
        self, tenant: int | Namespace | None = None, as_of: AsOf | _dt.datetime | None = None
    ) -> Visibility:
        """The :class:`~anatid.visibility.Visibility` a read with these arguments applies.

        The tenant goes through :meth:`resolve_tenant`, so the file-per-tenant boundary is
        enforced here exactly as in every verb.  Accelerators use it to filter the candidates
        they generate.
        """
        ns = self.resolve_tenant(tenant)
        return Visibility.at(ns.tenant_id, as_of)

    def _index_event(
        self,
        kind: str,
        table: str,
        tenant_id: int,
        doc_ids: Sequence[int],
        *,
        at: _dt.datetime,
        reason: str | None = None,
    ) -> None:
        """Report a write to the derived indexes, inside the transaction that made it.

        ``getattr`` with a default on purpose, like ``in_transaction``: a test double without
        an :attr:`~anatid.Anatid.indexes` registry simply has no indexes to keep current.
        """
        registry = getattr(self, "indexes", None)
        if registry is None:
            return
        registry.emit(kind, table, tenant_id, doc_ids, at=at, reason=reason)

    @contextlib.contextmanager
    def _index_lifecycle(self):
        """Hold the derived indexes' per-file lifecycle lock across a purge, and announce it.

        ``forget(hard=True)`` enters this BEFORE opening its transaction, and holds it to the
        commit, so a generation cannot be announced from a snapshot that still contains the
        document this purge is erasing (:meth:`anatid.derived.IndexRegistry.lifecycle`).

        Announcing is the other half, and it has to happen before the purge's snapshot is
        taken.  A build already running when the purge starts commits its own catalog row while
        the purge's snapshot still says ``building``, so the purge skips that generation; the
        erasure counter is what tells the build its base may hold the erased document, and a
        counter bumped inside the purge's transaction could arrive after the build had already
        checked it and gone live.  Bumped here, the two orders are exhaustive: either the build
        sees the bump and marks itself invalid, or the purge started after the build committed
        and its snapshot sees a finished generation whose storage it can clean.

        A no-op without a registry, and a spurious announcement (a purge that then raises) costs
        one rebuild, which is the safe direction.
        """
        registry = getattr(self, "indexes", None)
        if registry is None:
            yield
            return
        with registry.lifecycle():
            registry.announce_erasure()
            yield

    def _index_erase(
        self,
        table: str,
        tenant_id: int,
        doc_ids: Sequence[int],
        *,
        at: _dt.datetime,
        reason: str | None = None,
    ):
        """Erase purged documents from every derived index, inside the purge transaction.

        The destructive counterpart of :meth:`_index_event`: a tombstone would leave the erased
        id in the file, so the journal rows go and each generation either deletes the document
        from its storage or is invalidated.  Returns an
        :class:`~anatid.derived.ErasureResult` (an empty one for a test double with no
        registry).
        """
        registry = getattr(self, "indexes", None)
        if registry is None:
            return _derived.ErasureResult()
        return registry.erase(table, tenant_id, doc_ids, at=at, reason=reason)

    # ------------------------------------------------------------------ versions

    def _close_memory(
        self, tenant_id: int, memory_id: int, version: int, at: _dt.datetime, *, reason: str
    ) -> bool:
        """Close the current version of a memory at ``at`` by writing its successor.

        Two statements, in the caller's transaction.  The current version's ``tx_to`` is set
        to ``at``: this is the compare-and-swap, guarded by ``version`` and
        :data:`~anatid.visibility.CURRENT_ROW_SQL`, so a concurrent correction of the same
        version aborts with :class:`~anatid.errors.ConflictError` and a version that is no
        longer current matches nothing.  Then a copy of that version is inserted as version
        ``n + 1`` with ``valid_to = at``, ``tx_from = at`` and ``tx_to`` open, so history
        keeps both what was believed and when the belief changed.  Returns ``False`` when
        nothing was current to close.  Reports the close to the derived indexes.
        """
        closed = _count(
            self.execute(
                f"UPDATE memories SET tx_to = ? WHERE memory_id = ? AND tenant_id = ? "
                f"AND {_VERSION} = ? AND {_CURRENT}",
                [at, int(memory_id), int(tenant_id), int(version)],
            )
        )
        if not closed:
            return False
        cols, select, params = _successor(MEMORY_COLUMNS, valid_to=at, tx_from=at)
        self.execute(
            f"INSERT INTO memories ({cols}) SELECT {select} FROM memories "
            f"WHERE memory_id = ? AND tenant_id = ? AND {_VERSION} = ?",
            params + [int(memory_id), int(tenant_id), int(version)],
        )
        self._index_event(
            _derived.CLOSE, "memories", int(tenant_id), [int(memory_id)], at=at, reason=reason
        )
        return True

    def _close_edges(
        self,
        table: str,
        columns: Sequence[str],
        where: str,
        params: Sequence[Any],
        at: _dt.datetime,
        *,
        tenant_id: int,
        reason: str,
    ) -> list[int]:
        """Close every current edge of ``table`` matching ``where`` at ``at``; the edge ids.

        The successors go in first (``valid_to = at``, ``tx_from = at``, version ``n + 1``),
        selected by :data:`~anatid.visibility.CURRENT_ROW_SQL` and returned by id; then the
        originals' ``tx_to`` is set with the same predicate, which the successors no longer
        satisfy because their ``valid_to`` is set.  Both statements are set-based, so the
        ABOUT edges of a memory close in two statements however many there are.  ``where``
        must carry the tenant predicate.  Refuses, with :class:`~anatid.errors.RangeError`,
        an ``at`` before any matched edge's ``valid_from`` or ``tx_from``.
        """
        if table not in VERSIONED_TABLES:
            raise ValueError(f"{table} is not a versioned table")
        bounds = self.execute(
            f"SELECT max(valid_from), max(tx_from) FROM {table} WHERE {where} AND {_CURRENT}",
            list(params),
        ).fetchone()
        if bounds is not None:
            if bounds[0] is not None and at < bounds[0]:
                raise RangeError(
                    f"{reason}: now={at.isoformat()} is before an edge's "
                    f"valid_from={bounds[0].isoformat()} in {table}, which would close its "
                    f"interval before it opened. Pass now >= valid_from.",
                    field="now",
                    value=at,
                )
            _check_records_after(reason, f"an edge in {table}", bounds[1], at)
        cols, select, sparams = _successor(columns, valid_to=at, tx_from=at)
        closed = _distinct(
            _ids(
                self.execute(
                    f"INSERT INTO {table} ({cols}) SELECT {select} FROM {table} "
                    f"WHERE {where} AND {_CURRENT} RETURNING edge_id",
                    sparams + list(params),
                )
            )
        )
        if closed:
            self.execute(
                f"UPDATE {table} SET tx_to = ? WHERE {where} AND {_CURRENT}", [at] + list(params)
            )
            self._index_event(_derived.CLOSE, table, int(tenant_id), closed, at=at, reason=reason)
        return closed

    def _check_embedding(self, embedding: Sequence[float] | None) -> str | None:
        """Validate an embedding and render it as a SQL literal, or return None.

        Two checks, both of which anatid 0.1.0 skipped for the second one:

        * **length** must equal the database's ``FLOAT[N]`` -- already enforced in 0.1.0;
        * **every value must be finite.**  DuckDB stores NaN and +/-inf without complaint, and
          ``array_cosine_similarity`` then returns NaN against that row for *every* query, which
          orders arbitrarily and can push real hits out of the vector arm's candidate list.  A
          NaN embedding is not a bad score, it is a corrupt index entry, so it is refused at the
          boundary rather than written and reported later by :meth:`~anatid.Anatid.doctor`.
        """
        if embedding is None:
            return None
        dim = self.config.embedding_dim
        if len(embedding) != dim:
            raise EmbeddingDimensionError(
                f"embedding has {len(embedding)} dimensions, database is FLOAT[{dim}]",
                expected=dim,
                got=len(embedding),
            )
        for i, x in enumerate(embedding):
            try:
                value = float(x)
            except (TypeError, ValueError):
                raise EmbeddingValueError(
                    f"embedding[{i}] is {type(x).__name__}, not a number", index=i, value=x
                ) from None
            if not math.isfinite(value):
                raise EmbeddingValueError(
                    f"embedding[{i}] is {value!r}; every value must be finite (NaN and inf "
                    f"poison array_cosine_similarity for this row against every query)",
                    index=i,
                    value=value,
                )
        return _recall.embedding_literal(embedding)

    def _atomic(self, work: Callable[[], _T], *, retries: int = ENTITY_RACE_RETRIES) -> _T:
        """Run ``work()`` -- which opens its own transaction -- retrying a lost entity race.

        This is the application half of the fix for the entity-creation race.  The database half
        is ``UNIQUE (tenant_id, entity_key)``: with it, two writers naming the same new entity
        can no longer both succeed.  One of them therefore has to lose, and losing has to be
        *invisible* to the caller, because both of them asked for the same thing and the answer
        exists.  So the loser re-runs its whole unit of work; its second attempt reads the
        winner's row and inserts nothing.

        Re-running the *whole* transaction is not an implementation detail, it is forced:
        DuckDB marks a transaction aborted the moment a constraint fires inside it ("Current
        transaction is aborted (please ROLLBACK)"), and the write-write form is not detected
        until COMMIT.  There is no way to recover inside the transaction that failed.

        When the caller opened the transaction (``with db.transaction(): db.remember(...)``)
        this steps aside: only the caller can decide to re-run the caller's transaction, and
        the error propagates so they can.
        """
        if _in_transaction(self):
            return work()
        last: BaseException | None = None
        for attempt in range(retries):
            try:
                return work()
            except Exception as exc:
                if not _is_entity_race(exc):
                    raise
                last = exc
                log.debug(
                    "lost an entity-creation race (attempt %d/%d), re-running: %s",
                    attempt + 1,
                    retries,
                    exc,
                )
        assert last is not None
        raise last

    def _reject_duplicate_id(self, table: str, column: str, value: int, tenant_id: int) -> None:
        """Raise :class:`~anatid.errors.DuplicateIdError` if ``value`` is already used here.

        Only ever called for an id the *caller* supplied -- minted ids cannot collide (see
        :mod:`anatid.ids`), and a lookup on every write would be a needless index probe.

        Honest about its reach: this is a check, not a constraint.  It runs inside the writing
        transaction, so it catches the serialized case that actually happens -- a retried
        request, two agents acting on the same stale read -- but two writers holding separate
        snapshots can both pass it, because ``memories`` carries no UNIQUE index on
        ``(tenant_id, memory_id)`` (adding one would be a schema change and would rewrite the
        table's ART on every append).  :meth:`~anatid.Anatid.doctor` reports duplicates that
        get through.
        """
        row = self.execute(
            f"SELECT 1 FROM {_schema.quote_ident(table)} WHERE {_schema.quote_ident(column)} = ? "
            f"AND {tenant_sql()} LIMIT 1",
            [int(value), int(tenant_id)],
        ).fetchone()
        if row is not None:
            raise DuplicateIdError(
                f"{column} {int(value)} already exists in tenant {int(tenant_id)}; "
                f"ids are unique per tenant. Let anatid mint one, or supersede/forget the "
                f"existing row first.",
                table=table,
                id=int(value),
                tenant_id=int(tenant_id),
            )

    def entity_id(
        self,
        value: "int | str | Entity",
        *,
        tenant: int | Namespace | None = None,
        create: bool = False,
        kind: str | None = None,
        now: _dt.datetime | None = None,
        writer: str | None = None,
        episode_id: int | None = None,
    ) -> int:
        r"""Resolve an entity reference to an ``entity_id``.

        ``int`` is taken as an id verbatim (no existence check -- that is a scan anatid will not
        do on every write).  ``Entity`` uses its id.  ``str`` is looked up by **canonical name**
        and, with ``create=True``, inserted if missing.

        The canonical name is ``entities.entity_key``, the generated column
        ``trim(regexp_replace(lower(name), '\s+', ' ', 'g'))`` that schema v3 put a
        ``UNIQUE (tenant_id, entity_key)`` index on.  Three consequences worth stating:

        * ``"Ada Lovelace"``, ``"ada lovelace"`` and ``"  Ada   Lovelace "`` are one entity.
          Schema v2 made them three, which is how the same agent naming the same person twice
          fractured its own graph.
        * The lookup binds the **raw** name and lets DuckDB canonicalise it.  Canonicalising in
          Python and comparing keys would be comparing ``str.lower()`` against DuckDB's
          ``lower()`` -- two Unicode implementations, and a disagreement there means the SELECT
          misses and the INSERT then hits the constraint.
        * There is no ``valid_to IS NULL`` filter any more.  The uniqueness constraint spans
          history, so a soft-forgotten entity is still the entity of that name; re-creating it
          is not something the database will allow, and silently returning a second row is not
          something it can do.

        Concurrency: the lookup-then-insert is still two statements, so two writers can both
        find nothing.  They can no longer both insert -- one gets a constraint error (or a
        commit-time conflict) which :meth:`_atomic` turns into a re-run of the whole verb, and
        the re-run reads the winner.  See :meth:`_atomic` for why recovery cannot happen inside
        the failed transaction.
        """
        ns = self.resolve_tenant(tenant)
        if isinstance(value, Entity):
            return int(value.entity_id)
        if isinstance(value, bool):
            raise TypeError("entity reference must be int | str | Entity, not bool")
        if isinstance(value, int):
            return int(value)
        if not isinstance(value, str):
            raise TypeError(
                f"entity reference must be int | str | Entity, got {type(value).__name__}"
            )
        row = self.execute(_ENTITY_BY_KEY_SQL, [ns.tenant_id, value]).fetchone()
        if row is not None:
            return int(row[0])
        if not create:
            raise NotFoundError(f"no entity named {value!r} in tenant {ns.tenant_id}")
        at = to_utc_naive(now) or utcnow()
        eid = new_id()
        try:
            self.execute(
                _INSERT_ENTITY, [eid, ns.tenant_id, kind, value, at, at, writer, episode_id, 1.0]
            )
            self._index_event(_derived.INSERT, "entities", ns.tenant_id, [eid], at=at)
        except Exception as exc:
            if not _is_entity_race(exc):
                raise
            if _in_transaction(self):
                # The constraint has already aborted the enclosing transaction; nothing can be
                # read or written on it now.  Raise the documented retryable error -- with the
                # DuckDB exception as `cause`, which is what _atomic re-runs the whole verb on,
                # and what a caller who opened the transaction themselves retries on
                # (ConflictError.retryable): the loser's next attempt reads the winner.
                raise ConflictError(
                    f"lost the race to create entity {value!r} in tenant {ns.tenant_id}: another "
                    f"transaction committed it first and the enclosing transaction is aborted; "
                    f"roll back and re-run the unit of work (its lookup will find the winner)",
                    cause=exc,
                ) from exc
            # Autocommit: this statement was its own transaction, so the connection is healthy
            # and the winner is committed and visible.  Read it and return it.
            row = self.execute(_ENTITY_BY_KEY_SQL, [ns.tenant_id, value]).fetchone()
            if row is None:  # pragma: no cover - winner purged in between
                raise
            return int(row[0])
        return eid

    # ------------------------------------------------------------------ episodes

    def episode(
        self,
        content: str,
        *,
        source: str | None = None,
        kind: str | None = None,
        writer: str | None = None,
        tenant: int | Namespace | None = None,
        now: _dt.datetime | None = None,
        episode_id: int | None = None,
    ) -> Episode:
        """Record raw source material before anything is derived from it.

        "Evidence before belief": write the episode, then pass its ``episode_id`` to
        :meth:`remember` so every derived fact points back at the text it came from and
        :meth:`provenance` can walk there.
        """
        ns = self.resolve_tenant(tenant)
        at = to_utc_naive(now) or utcnow()
        explicit_id = episode_id is not None
        eid = _check_row_id("episode_id", episode_id) if explicit_id else new_id()
        with self.transaction():
            if explicit_id:
                self._reject_duplicate_id("episodes", "episode_id", eid, ns.tenant_id)
            self.execute(
                "INSERT INTO episodes (episode_id, tenant_id, source, content, kind, created_at, "
                "valid_from, valid_to, tx_from, tx_to, writer) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?)",
                [eid, ns.tenant_id, source, content, kind, at, at, at, writer],
            )
        return Episode(
            episode_id=eid,
            tenant_id=ns.tenant_id,
            content=content,
            source=source,
            kind=kind,
            created_at=at,
            valid_from=at,
            tx_from=at,
            writer=writer,
        )

    def get_episode(
        self, episode_id: int, *, tenant: int | Namespace | None = None
    ) -> Episode | None:
        """Fetch one episode, or None."""
        ns = self.resolve_tenant(tenant)
        row = self.execute(
            f"SELECT {_EPI_SELECT} FROM episodes WHERE episode_id = ? AND {tenant_sql()}",
            [int(episode_id), ns.tenant_id],
        ).fetchone()
        return None if row is None else Episode.from_row(row)

    # ------------------------------------------------------------------ entities & edges

    def upsert_entity(
        self,
        name: str,
        *,
        kind: str | None = None,
        tenant: int | Namespace | None = None,
        writer: str | None = None,
        episode_id: int | None = None,
        now: _dt.datetime | None = None,
    ) -> Entity:
        """Find or create an entity by name and return it.

        Idempotent under concurrency: 100 threads calling this with one new name end with one
        entity row and all 100 holding its id.  See :meth:`entity_id` and :meth:`_atomic`.
        """
        ns = self.resolve_tenant(tenant)

        def _work() -> int:
            with self.transaction():
                return self.entity_id(
                    name,
                    tenant=ns,
                    create=True,
                    kind=kind,
                    now=now,
                    writer=writer,
                    episode_id=episode_id,
                )

        eid = self._atomic(_work)
        got = self.get_entity(eid, tenant=ns)
        if got is None:  # pragma: no cover - only if another writer purged it in between
            raise NotFoundError(f"entity {eid} vanished during upsert")
        return got

    def get_entity(
        self,
        entity: "int | str | Entity",
        *,
        tenant: int | Namespace | None = None,
    ) -> Entity | None:
        """Fetch one entity by id or name, or None."""
        ns = self.resolve_tenant(tenant)
        if isinstance(entity, str):
            # By canonical key, exactly as entity_id() resolves it, so get_entity(name) and
            # entity_id(name) can never disagree about which row a name means.
            row = self.execute(
                f"SELECT {_ENT_SELECT} FROM entities WHERE {tenant_sql()} AND entity_key = "
                + entity_key_sql("?")
                + " ORDER BY entity_id LIMIT 1",
                [ns.tenant_id, entity],
            ).fetchone()
        else:
            eid = entity.entity_id if isinstance(entity, Entity) else int(entity)
            row = self.execute(
                f"SELECT {_ENT_SELECT} FROM entities WHERE entity_id = ? AND {tenant_sql()}",
                [int(eid), ns.tenant_id],
            ).fetchone()
        return None if row is None else Entity.from_row(row)

    def relate(
        self,
        src: "int | str | Entity",
        dst: "int | str | Entity",
        *,
        rel_kind: str | None = None,
        tenant: int | Namespace | None = None,
        writer: str | None = None,
        episode_id: int | None = None,
        confidence: float = 1.0,
        now: _dt.datetime | None = None,
        valid_from: _dt.datetime | None = None,
        create_missing: bool = True,
        edge_id: int | None = None,
    ) -> Edge:
        """Add a ``RELATES_TO`` edge between two entities -- the edges 2-hop recall traverses.

        Traversal is undirected (both directions are expanded), so ``relate(a, b)`` makes ``b``
        reachable from ``a`` and vice versa.  Marks the optional CSR snapshot stale.
        """
        ns = self.resolve_tenant(tenant)
        at = to_utc_naive(now) or utcnow()
        vf = to_utc_naive(valid_from) or at
        confidence = _check_unit("confidence", confidence)
        explicit_id = edge_id is not None
        eid = _check_row_id("edge_id", edge_id) if explicit_id else new_id()

        def _work() -> tuple[int, int]:
            with self.transaction():
                if explicit_id:
                    self._reject_duplicate_id("edges_relates", "edge_id", eid, ns.tenant_id)
                s_id = self.entity_id(src, tenant=ns, create=create_missing, now=at, writer=writer)
                d_id = self.entity_id(dst, tenant=ns, create=create_missing, now=at, writer=writer)
                self.execute(
                    "INSERT INTO edges_relates (edge_id, src, dst, tenant_id, rel_kind, "
                    "valid_from, valid_to, tx_from, tx_to, writer, episode_id, confidence) "
                    "VALUES (?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?, ?)",
                    [
                        eid,
                        s_id,
                        d_id,
                        ns.tenant_id,
                        rel_kind,
                        vf,
                        at,
                        writer,
                        episode_id,
                        confidence,
                    ],
                )
                self._index_event(_derived.INSERT, "edges_relates", ns.tenant_id, [eid], at=at)
            return s_id, d_id

        s, d = self._atomic(_work)
        self.csr.note_edge_write()
        return Edge(
            edge_id=eid,
            edge_type=RELATES_TO,
            src=s,
            dst=d,
            tenant_id=ns.tenant_id,
            rel_kind=rel_kind,
            valid_from=vf,
            tx_from=at,
            writer=writer,
            episode_id=episode_id,
            confidence=confidence,
        )

    def entities_of(
        self,
        memory_id: int,
        *,
        tenant: int | Namespace | None = None,
        as_of: AsOf | _dt.datetime | None = None,
    ) -> list[Entity]:
        """Entities a memory is ABOUT, ordered by entity_id."""
        aw, ap = self.visibility(tenant, as_of).predicate("a")
        rows = self.execute(
            f"SELECT {', '.join('e.' + c for c in ENTITY_COLUMNS)} FROM edges_about a "
            f"JOIN entities e ON e.entity_id = a.dst AND e.tenant_id = a.tenant_id "
            f"WHERE a.src = ? AND {aw} ORDER BY e.entity_id",
            [int(memory_id)] + ap,
        ).fetchall()
        return [Entity.from_row(r) for r in rows]

    # ------------------------------------------------------------------ remember

    def remember(
        self,
        content: str,
        *,
        entities: Sequence["int | str | Entity"] = (),
        kind: str = "fact",
        embedding: Sequence[float] | None = None,
        writer: str | None = None,
        episode_id: int | None = None,
        episode: str | None = None,
        episode_source: str | None = None,
        confidence: float = 1.0,
        now: _dt.datetime | None = None,
        valid_from: _dt.datetime | None = None,
        created_at: _dt.datetime | None = None,
        tenant: int | Namespace | None = None,
        memory_id: int | None = None,
        entity_kind: str | None = None,
        weight: float = 1.0,
        create_entities: bool = True,
    ) -> Memory:
        """Write one memory and its ABOUT edges, in a single transaction.

        ``entities``
            Names (created on demand), ``entity_id`` ints, or :class:`~anatid.types.Entity`
            objects.  Each becomes one ABOUT edge, which is what makes the memory reachable by
            graph recall.
        ``episode`` / ``episode_source``
            Raw source text to record first and attach as this memory's provenance.  Mutually
            exclusive with an explicit ``episode_id``.
        ``now``
            Defaults to UTC now and sets ``valid_from``, ``tx_from`` and ``created_at`` unless
            those are given.  Pass it to make a run deterministic.

        Appends never conflict under DuckDB's optimistic MVCC, so concurrent ``remember()`` from
        many writers all commit.  Naming the same *new* entity from several of them is the one
        place where they can contend, and it is resolved without the caller seeing it: the
        losers re-run and read the winner's entity row (:meth:`_atomic`).

        Arguments are checked before anything is written: ``confidence`` and ``weight`` must be
        finite and in ``[0, 1]``, every embedding value must be finite and the vector must have
        the database's dimension, and an explicit ``memory_id`` must not already exist in this
        tenant.  Each raises a subclass of :class:`~anatid.errors.ValidationError`.
        """
        ns = self.resolve_tenant(tenant)
        at = to_utc_naive(now) or utcnow()
        vf = to_utc_naive(valid_from) or at
        ca = to_utc_naive(created_at) or at
        confidence = _check_unit("confidence", confidence)
        weight = _check_unit("weight", weight)
        emb = self._check_embedding(embedding)
        explicit_id = memory_id is not None
        mid = _check_row_id("memory_id", memory_id) if explicit_id else new_id()
        if episode is not None and episode_id is not None:
            raise ValidationError(
                "pass either episode (raw text to record) or episode_id, not both"
            )

        def _work() -> Memory:
            ep = None if episode_id is None else int(episode_id)
            with self.transaction():
                if explicit_id:
                    # memory_id is unique per TENANT, not per file (the same id in two tenants
                    # is legal and tested).  Two rows with one id inside one tenant are not:
                    # get() would return an arbitrary one, supersede would close both, and the
                    # BM25 source table would have to de-duplicate them.
                    self._reject_duplicate_id("memories", "memory_id", mid, ns.tenant_id)
                if episode is not None:
                    ep = new_id()
                    self.execute(
                        "INSERT INTO episodes (episode_id, tenant_id, source, content, kind, "
                        "created_at, valid_from, valid_to, tx_from, tx_to, writer) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?)",
                        [ep, ns.tenant_id, episode_source, episode, "source", ca, vf, at, writer],
                    )

                self.execute(
                    _INSERT_MEMORY_TPL.format(dim=self.config.embedding_dim),
                    [mid, ns.tenant_id, content, kind, emb, ca, vf, at, writer, ep, confidence],
                )
                self._index_event(_derived.INSERT, "memories", ns.tenant_id, [mid], at=at)

                dsts = [
                    self.entity_id(
                        e,
                        tenant=ns,
                        create=create_entities,
                        kind=entity_kind,
                        now=at,
                        writer=writer,
                        episode_id=ep,
                    )
                    for e in entities
                ]
                if dsts:
                    edge_ids = [new_id() for _ in dsts]
                    self.execute(
                        _INSERT_ABOUT,
                        [mid, ns.tenant_id, weight, vf, at, writer, ep, confidence, edge_ids, dsts],
                    )
                    self._index_event(_derived.INSERT, "edges_about", ns.tenant_id, edge_ids, at=at)

            return Memory(
                memory_id=mid,
                tenant_id=ns.tenant_id,
                content=content,
                kind=kind,
                embedding=None if embedding is None else tuple(float(x) for x in embedding),
                created_at=ca,
                valid_from=vf,
                valid_to=None,
                tx_from=at,
                tx_to=None,
                writer=writer,
                episode_id=ep,
                confidence=confidence,
                access_count=0,
                last_access_at=None,
            )

        return self._atomic(_work)

    # ------------------------------------------------------------------ reads

    def get(
        self,
        memory_id: int,
        *,
        tenant: int | Namespace | None = None,
        as_of: AsOf | _dt.datetime | None = None,
        with_embedding: bool = True,
    ) -> Memory | None:
        """Fetch one memory by id under the given time scope, or None.

        Without ``as_of`` this returns the memory's **live** version: the physical row no
        correction has closed, which is a superseded or forgotten memory's last version with
        its ``valid_to`` set, so the row is returned and ``Memory.is_current`` says so.  With
        ``as_of`` it returns the version visible at that instant on both axes, or ``None``.
        """
        vis = self.visibility(tenant, as_of)
        cols = _mem_select(self.connection, embedding=with_embedding)
        if vis.is_current:
            # No validity filter on purpose: the as-of form below is the one that hides history.
            # The transaction-time half is what selects one version of the id.
            l_sql, l_p = vis.live()
            row = self.execute(
                f"SELECT {cols} FROM memories WHERE memory_id = ? AND {l_sql}",
                [int(memory_id)] + l_p,
            ).fetchone()
        else:
            w, wp = vis.predicate("m")
            row = self.execute(
                f"SELECT {cols} FROM memories m WHERE m.memory_id = ? AND {w}",
                [int(memory_id)] + wp,
            ).fetchone()
        return None if row is None else Memory.from_row(row)

    def versions(
        self,
        memory_id: int,
        *,
        tenant: int | Namespace | None = None,
        with_embedding: bool = False,
    ) -> list[Memory]:
        """Every physical version of one memory, oldest first (``[]`` for an unknown id).

        ``versions[0]`` is version 1, the row ``remember`` wrote; each later entry is the
        successor a correction inserted, with ``tx_from`` at the correction and the valid
        interval it decided.  Every entry but the last has ``tx_to`` set.  The list is the
        transaction-time history of one logical id; :meth:`provenance` carries it as
        ``Provenance.versions`` next to the SUPERSEDES chain between logical ids.
        """
        ns = self.resolve_tenant(tenant)
        con = self.connection
        cols = _mem_select(con, embedding=with_embedding)
        t_sql, t_p = Visibility(ns.tenant_id).tenant()
        rows = self.execute(
            f"SELECT {cols} FROM memories WHERE memory_id = ? AND {t_sql} "
            f"ORDER BY {version_expr(con)}, tx_from",
            [int(memory_id)] + t_p,
        ).fetchall()
        return [Memory.from_row(r) for r in rows]

    def recall_2hop_ids(
        self,
        seed_entity: "int | str | Entity",
        *,
        tenant: int | Namespace | None = None,
        limit: int = 20,
        as_of: AsOf | _dt.datetime | None = None,
        hops: int = 2,
        kinds: Sequence[str] | None = None,
    ) -> list[tuple[int, _dt.datetime]]:
        """The benchmarked 2-hop recall exactly as measured: ``[(memory_id, created_at)]``.

        Two columns only, because hydrating full rows would drag the ``FLOAT[N]`` embedding into
        the TOP_N.  Use :meth:`recall_2hop` for :class:`~anatid.types.Memory` objects.
        """
        ns = self.resolve_tenant(tenant)
        scope = AsOf.coerce(as_of)
        limit = _check_positive("limit", limit)
        hops = _check_non_negative("hops", hops)
        seed = (
            self.entity_id(seed_entity, tenant=ns, create=False)
            if not isinstance(seed_entity, int)
            else int(seed_entity)
        )
        return _recall.recall_2hop_ids(
            self.connection,
            tenant_id=ns.tenant_id,
            seed_entity_id=seed,
            limit=limit,
            hops=hops,
            as_of=scope,
            backend=self.csr,
            kinds=kinds,
        )

    def recall_2hop(
        self,
        seed_entity: "int | str | Entity",
        *,
        tenant: int | Namespace | None = None,
        limit: int = 20,
        as_of: AsOf | _dt.datetime | None = None,
        hops: int = 2,
        kinds: Sequence[str] | None = None,
        with_embedding: bool = False,
    ) -> list[Memory]:
        """Memories ABOUT any entity within ``hops`` of the seed, newest first.

        The graph query the whole engine choice was decided on: at 1M memories / 2.3M edges /
        10 tenants it runs in 2.88 ms p50 on plain DuckDB SQL and 2.04 ms with the optional C++
        CSR extension, against LadybugDB's tuned 7.35 ms.

        Traversal is undirected over currently-valid same-tenant ``RELATES_TO`` edges; results are
        ordered ``created_at DESC, memory_id DESC``.
        """
        pairs = self.recall_2hop_ids(
            seed_entity, tenant=tenant, limit=limit, as_of=as_of, hops=hops, kinds=kinds
        )
        ns = self.resolve_tenant(tenant)
        rows = _recall.hydrate(
            self.connection,
            [m for m, _ in pairs],
            tenant_id=ns.tenant_id,
            with_embedding=with_embedding,
            as_of=AsOf.coerce(as_of),
        )
        return [rows[m] for m, _ in pairs if m in rows]

    def context(
        self,
        entity: "int | str | Entity",
        *,
        limit: int = 20,
        as_of: AsOf | _dt.datetime | None = None,
        tenant: int | Namespace | None = None,
        hops: int = 0,
        kinds: Sequence[str] | None = None,
        with_embedding: bool = False,
    ) -> list[Memory]:
        """Memories directly ABOUT one entity, newest first.

        ``hops=0`` (the default) is the entity's own memories.  ``hops=1`` widens to its
        neighbours, ``hops=2`` is :meth:`recall_2hop`.
        """
        return self.recall_2hop(
            entity,
            tenant=tenant,
            limit=limit,
            as_of=as_of,
            hops=hops,
            kinds=kinds,
            with_embedding=with_embedding,
        )

    def recall(
        self,
        query: str | None = None,
        *,
        tenant: int | Namespace | None = None,
        k: int = 10,
        embedding: Sequence[float] | None = None,
        seed_entity: "int | str | Entity | None" = None,
        hops: int = 2,
        as_of: AsOf | _dt.datetime | None = None,
        kinds: Sequence[str] | None = None,
        candidates: int = _recall.DEFAULT_CANDIDATES,
        rrf_k: int = _recall.RRF_K,
        with_embedding: bool = False,
        include_about: bool = True,
        on_stale_fts: str = "report",
        allow_slow: bool = False,
    ) -> RecallHits:
        """Hybrid retrieval: cosine + BM25 + graph expansion, fused with RRF (k=60).

        Arms run when their input exists -- ``embedding`` for the vector arm, ``query`` plus an
        fts index for BM25, ``seed_entity`` for the graph arm -- and are fused by Reciprocal Rank
        Fusion.  The result is a ``list[RecallHit]`` that also reports how it was answered:

        * ``hits.bm25_stale`` -- rows written since the last :meth:`~anatid.Anatid.rebuild_fts_index`
          are invisible to the text arm (DuckDB's fts index is not incremental).  A warning is
          also logged on ``anatid.recall``.  ``on_stale_fts="error"`` raises instead.
        * ``hits.pending_fts_rows`` -- how many rows that is.
        * ``hits.arms`` -- which arms actually ran.

        The vector arm is a brute-force scan of the tenant's embeddings: comfortable to roughly
        1e5 memories per tenant, linear beyond that.  That limit is
        :data:`anatid.BRUTE_FORCE_CEILING`, and it is enforced: when ``embedding`` is given and
        the rows the arm would scan exceed it, this raises
        :class:`~anatid.errors.BruteForceCeilingError` before any arm runs, unless
        ``allow_slow=True``.  The text and graph arms do not scale with the tenant's size and
        are always available.
        """
        ns = self.resolve_tenant(tenant)
        scope = AsOf.coerce(as_of)
        k = _check_positive("k", k)
        candidates = _check_positive("candidates", candidates)
        rrf_k = _check_positive("rrf_k", rrf_k)
        hops = _check_non_negative("hops", hops)
        seed = None
        if seed_entity is not None:
            seed = (
                int(seed_entity)
                if isinstance(seed_entity, int) and not isinstance(seed_entity, bool)
                else self.entity_id(seed_entity, tenant=ns, create=False)
            )
        if embedding is not None:
            self._check_embedding(embedding)
        return _recall.hybrid_recall(
            self.connection,
            tenant_id=ns.tenant_id,
            query=query,
            embedding=embedding,
            dim=self.config.embedding_dim,
            k=k,
            seed_entity=seed,
            hops=hops,
            as_of=scope,
            kinds=kinds,
            candidates=candidates,
            rrf_k=rrf_k,
            backend=self.csr,
            with_embedding=with_embedding,
            include_about=include_about,
            on_stale_fts=on_stale_fts,
            allow_slow=allow_slow,
        )

    def as_of(
        self, timestamp: _dt.datetime | AsOf, *, tx_time: _dt.datetime | None = None
    ) -> AsOfView:
        """Scope reads to a point in time.

        ``db.as_of(t).recall_2hop(seed)`` answers as the database believed at ``t``.

        This is anatid's own filter over ``valid_from``/``valid_to`` and ``tx_from``/``tx_to``.
        **DuckDB has no ``AS OF SYSTEM TIME``**; nothing rewinds, and rows removed by a hard
        purge are absent from every as-of view too.  Pass ``tx_time`` to separate the two axes
        (what was true then vs. what the database knew then).
        """
        if isinstance(timestamp, AsOf):
            scope = timestamp
        else:
            ts = to_utc_naive(timestamp)
            scope = AsOf(valid_time=ts, tx_time=to_utc_naive(tx_time) if tx_time else ts)
        return AsOfView(self, scope)

    # ------------------------------------------------------------------ mutations

    def supersede(
        self,
        old_id: int,
        content: str,
        *,
        entities: Sequence["int | str | Entity"] | None = None,
        kind: str | None = None,
        embedding: Sequence[float] | None = None,
        writer: str | None = None,
        episode_id: int | None = None,
        episode: str | None = None,
        episode_source: str | None = None,
        confidence: float = 1.0,
        now: _dt.datetime | None = None,
        tenant: int | Namespace | None = None,
        memory_id: int | None = None,
        close_about_edges: bool = False,
        allow_fork: bool = False,
    ) -> Memory:
        """Replace a memory with a newer one, in one transaction.

        Inserts the new memory, closes the old one at ``now``, and records a ``SUPERSEDES``
        edge (new -> old) so :meth:`provenance` can walk the chain.  Closing is a correction:
        the old memory's current version gets ``tx_to = now`` and a successor version with
        ``valid_to = now`` is inserted (:meth:`_close_memory`); nothing is rewritten.  Reads at
        the current time see only the new memory; ``db.as_of(t)`` for ``t`` before ``now``
        still sees the old one, and ``as_of(valid_time=later, tx_time=t)`` sees it open-ended,
        because at ``t`` the database had not yet recorded the replacement.

        ``entities=None`` (default) inherits the old memory's ABOUT entities; pass a sequence to
        replace them, or ``()`` for none.  ``kind=None`` inherits the old kind.

        The ``UPDATE`` on the old version is what can lose a write-write race: if another
        transaction is *concurrently* superseding the same memory, this one raises
        :class:`~anatid.errors.ConflictError` and is safe to retry.

        The *serialized* version of that race -- a retry after a client timeout, or two agents
        acting on the same stale read -- does not conflict in the engine at all: the ``UPDATE``
        simply matches nothing, because the memory has no current version any more.  anatid
        checks the row count and raises :class:`~anatid.errors.ConflictError` naming the memory
        that already superseded it, rather than committing a second current memory and leaving
        two heads on one chain.  Pass ``allow_fork=True`` if branching really is what you want.

        A lost entity-creation race (two writers naming the same *new* entity in ``entities=``)
        is re-run transparently, exactly as in :meth:`remember` -- the first attempt rolled back,
        so the retry's ``UPDATE`` still finds the current version and the fork check is not
        confused by its own replay.  Only that specific constraint failure is retried; the
        write-write ``ConflictError`` above is the caller's to handle.

        ``now`` may not precede the old memory's ``valid_from`` (that would close its interval
        before it opened, a row no ``as_of`` can ever return) nor its ``tx_from`` (that would
        record the correction before the version it corrects).
        :class:`~anatid.errors.RangeError` either way.
        """
        ns = self.resolve_tenant(tenant)
        at = to_utc_naive(now) or utcnow()
        old = self.get(int(old_id), tenant=ns, with_embedding=False)
        if old is None:
            raise NotFoundError(f"memory {old_id} not found in tenant {ns.tenant_id}")
        _check_closes_after_open("supersede", int(old_id), old.valid_from, at)
        _check_records_after("supersede", f"memory {int(old_id)}", old.tx_from, at)
        if entities is None:
            entities = [e.entity_id for e in self.entities_of(int(old_id), tenant=ns)]
        confidence = _check_unit("confidence", confidence)
        emb = self._check_embedding(embedding)
        explicit_id = memory_id is not None
        new_mid = _check_row_id("memory_id", memory_id) if explicit_id else new_id()
        if episode is not None and episode_id is not None:
            raise ValidationError(
                "pass either episode (raw text to record) or episode_id, not both"
            )

        def _work() -> Memory:
            ep = None if episode_id is None else int(episode_id)
            with self.transaction():
                if explicit_id:
                    self._reject_duplicate_id("memories", "memory_id", new_mid, ns.tenant_id)
                if episode is not None:
                    ep = new_id()
                    self.execute(
                        "INSERT INTO episodes (episode_id, tenant_id, source, content, kind, "
                        "created_at, valid_from, valid_to, tx_from, tx_to, writer) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?)",
                        [ep, ns.tenant_id, episode_source, episode, "source", at, at, at, writer],
                    )

                self.execute(
                    _INSERT_MEMORY_TPL.format(dim=self.config.embedding_dim),
                    [
                        new_mid,
                        ns.tenant_id,
                        content,
                        kind if kind is not None else old.kind,
                        emb,
                        at,
                        at,
                        at,
                        writer,
                        ep,
                        confidence,
                    ],
                )
                self._index_event(_derived.INSERT, "memories", ns.tenant_id, [new_mid], at=at)

                dsts = [
                    self.entity_id(e, tenant=ns, create=True, now=at, writer=writer, episode_id=ep)
                    for e in entities
                ]
                if dsts:
                    edge_ids = [new_id() for _ in dsts]
                    self.execute(
                        _INSERT_ABOUT,
                        [
                            new_mid,
                            ns.tenant_id,
                            1.0,
                            at,
                            at,
                            writer,
                            ep,
                            confidence,
                            edge_ids,
                            dsts,
                        ],
                    )
                    self._index_event(_derived.INSERT, "edges_about", ns.tenant_id, edge_ids, at=at)

                closed = self._close_memory(
                    ns.tenant_id, int(old_id), old.version, at, reason="supersede"
                )
                if not closed and not allow_fork:
                    head = self.execute(
                        f"SELECT src FROM edges_supersedes WHERE dst = ? AND {tenant_sql()} "
                        f"ORDER BY tx_from, edge_id LIMIT 1",
                        [int(old_id), ns.tenant_id],
                    ).fetchone()
                    by = f" (memory {int(head[0])} already superseded it)" if head else ""
                    raise ConflictError(
                        f"memory {int(old_id)} is not current in tenant {ns.tenant_id}, so this "
                        f"supersede would fork the chain into two heads{by}. Re-read the current "
                        f"memory and supersede that one, or pass allow_fork=True."
                    )
                if close_about_edges:
                    self._close_edges(
                        "edges_about",
                        EDGE_ABOUT_COLUMNS,
                        "src = ? AND tenant_id = ?",
                        [int(old_id), ns.tenant_id],
                        at,
                        tenant_id=ns.tenant_id,
                        reason="supersede",
                    )
                self.execute(
                    "INSERT INTO edges_supersedes (edge_id, src, dst, tenant_id, tx_from, writer) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [new_id(), new_mid, int(old_id), ns.tenant_id, at, writer],
                )
                # The counterpart id goes in a COLUMN, never into `reason`: forget(hard=True) has to
                # be able to find and delete every audit row that names an erased memory, and it
                # cannot search free text for it.
                self.execute(
                    "INSERT INTO anatid_audit (audit_id, tenant_id, memory_id, related_memory_id, "
                    "action, reason, writer, happened_at) "
                    "VALUES (?, ?, ?, ?, 'supersede', ?, ?, ?)",
                    [new_id(), ns.tenant_id, int(old_id), new_mid, "superseded", writer, at],
                )

            return Memory(
                memory_id=new_mid,
                tenant_id=ns.tenant_id,
                content=content,
                kind=kind if kind is not None else old.kind,
                embedding=None if embedding is None else tuple(float(x) for x in embedding),
                created_at=at,
                valid_from=at,
                valid_to=None,
                tx_from=at,
                tx_to=None,
                writer=writer,
                episode_id=ep,
                confidence=confidence,
            )

        return self._atomic(_work)

    def reinforce(
        self,
        memory_id: int,
        *,
        amount: int = 1,
        confidence: float | None = None,
        now: _dt.datetime | None = None,
        tenant: int | Namespace | None = None,
    ) -> Memory:
        """Record that a memory was used: bump ``access_count`` and ``last_access_at``.

        The counters are **not bitemporal**.  They are updated in place on the memory's live
        version, so an ``as_of`` read on the transaction axis returns whatever count that
        version carried when it was closed, and a memory's history of use is not something
        this database records.  That is the one in-place update the verbs make, and it is
        made here because a version per access would double the table for no belief change.

        ``confidence`` is a belief, so changing it is a correction like any other: the live
        version is closed at ``now`` and a successor with the new confidence (and the bumped
        counters) is inserted, exactly as ``supersede`` does.  Either way this touches one
        row, so two concurrent reinforcements of the same memory race and the loser gets
        :class:`~anatid.errors.ConflictError` -- retry it.
        """
        ns = self.resolve_tenant(tenant)
        at = to_utc_naive(now) or utcnow()
        mid = int(memory_id)
        amount = _check_int("amount", amount)
        if confidence is not None:
            confidence = _check_unit("confidence", confidence)
        l_sql, l_p = Visibility(ns.tenant_id).live()
        with self.transaction():
            row = self.execute(
                f"SELECT {_VERSION}, tx_from FROM memories WHERE memory_id = ? AND {l_sql}",
                [mid] + l_p,
            ).fetchone()
            if row is None:
                raise NotFoundError(f"memory {memory_id} not found in tenant {ns.tenant_id}")
            version, tx_from = int(row[0]), row[1]
            if confidence is None:
                n = _count(
                    self.execute(
                        f"UPDATE memories SET access_count = coalesce(access_count, 0) + ?, "
                        f"last_access_at = ? WHERE memory_id = ? AND tenant_id = ? "
                        f"AND {_VERSION} = ? AND {_LIVE}",
                        [int(amount), at, mid, ns.tenant_id, version],
                    )
                )
            else:
                _check_records_after("reinforce", f"memory {mid}", tx_from, at)
                n = _count(
                    self.execute(
                        f"UPDATE memories SET tx_to = ? WHERE memory_id = ? AND tenant_id = ? "
                        f"AND {_VERSION} = ? AND {_LIVE}",
                        [at, mid, ns.tenant_id, version],
                    )
                )
                if n:
                    cols, select, params = _successor(
                        MEMORY_COLUMNS,
                        tx_from=at,
                        confidence=float(confidence),
                        access_count=_Expr("coalesce(access_count, 0) + ?", int(amount)),
                        last_access_at=at,
                    )
                    self.execute(
                        f"INSERT INTO memories ({cols}) SELECT {select} FROM memories "
                        f"WHERE memory_id = ? AND tenant_id = ? AND {_VERSION} = ?",
                        params + [mid, ns.tenant_id, version],
                    )
            if n == 0:  # pragma: no cover - a purge committed between the SELECT and the UPDATE
                raise NotFoundError(f"memory {memory_id} vanished during reinforce")
        got = self.get(mid, tenant=ns, with_embedding=False)
        if got is None:  # pragma: no cover - concurrent purge
            raise NotFoundError(f"memory {memory_id} vanished during reinforce")
        return got

    def forget(
        self,
        memory_id: int,
        *,
        hard: bool = False,
        reason: str | None = None,
        writer: str | None = None,
        now: _dt.datetime | None = None,
        tenant: int | Namespace | None = None,
    ) -> ForgetReceipt:
        """Stop believing a memory (soft), or erase it (hard).

        **Soft** (default): closes the memory and its ABOUT edges at ``now`` and writes an
        ``anatid_audit`` row.  Closing is a correction, not a rewrite: the current version of
        the memory (and of each current ABOUT edge) gets ``tx_to = now`` and a successor with
        ``valid_to = now`` is inserted, so ``db.as_of(t)`` before ``now`` still returns it,
        ``as_of(valid_time=after, tx_time=before)`` still returns the open-ended belief the
        database held before the forget, and :meth:`provenance` still walks through it.  A
        memory that is already closed gets no new version; the audit row is still written.
        ``now`` may not precede the memory's ``valid_from`` or ``tx_from``
        (:class:`~anatid.errors.RangeError`): a fact cannot stop being true before it started,
        a correction cannot be recorded before what it corrects, and the inverted intervals
        that would write are invisible to every ``as_of``.  Erase such a row with ``hard=True``
        instead.

        **Hard** (``hard=True``): a right-to-erasure purge.  Deletes every version of the
        ``memories`` row (the embedding is a column of it), every version of every ABOUT edge,
        every SUPERSEDES edge in either direction, the episode if no other *memory* cites it
        (entity and edge rows stamped with
        that episode's id keep existing, with the stamp cleared -- the raw source text usually
        quotes the erased content, and an entity that outlives the memory may not keep it in
        the file), and every ``anatid_audit`` row that names the memory -- as its own
        ``memory_id`` *or* as the ``related_memory_id`` of some other memory's supersede, which
        is why that id is a column and not free text.  It also clears the memory out of the BM25
        index tables, because schema v3's ``anatid_fts_documents`` stores ``content`` verbatim
        -- an erasure that left the text in a search index would not be an erasure -- and clamps
        the ``anatid_meta.fts_indexed_max_id`` watermark when the erased row was the newest
        indexed one.  The same argument covers every derived index (schema v4): the document is
        deleted from each generation's storage and from the change journal, and any generation
        whose storage cannot delete one document is invalidated so no read uses it again until
        it is rebuilt (``derived_rows_deleted`` / ``invalidated_generations`` on the receipt).
        When it returns, **no row in the memory graph references that memory_id**
        -- including in every as-of view, because there is nothing left to find.  That is the
        point: an audit trail that retained the id would defeat the erasure.

        The purge covers the tables anatid owns, and the tables anatid's bundled integrations
        create (:data:`anatid.erasure.BUNDLED_INTEGRATION_TABLES`: the Agents SDK transcript,
        session and usage tables and the run-state store), which it finds in the catalog and
        purges on **every** handle -- a hard forget issued from another process, or from the
        ``anatid-mcp`` ``forget`` tool, reaches them too.  It reaches anything else in the file
        only through the hooks you register with :meth:`~anatid.Anatid.register_erasure_hook`,
        which run inside this transaction.  Rows removed from integration tables, by hook or by
        the bundled-table pass, are reported as ``extra_rows_deleted``.  The returned
        :class:`~anatid.types.ForgetReceipt` is the record; log it outside the database if you
        need one.
        """
        ns = self.resolve_tenant(tenant)
        at = to_utc_naive(now) or utcnow()
        mid = int(memory_id)

        if not hard:
            l_sql, l_p = Visibility(ns.tenant_id).live()
            with self.transaction():
                row = self.execute(
                    f"SELECT {_VERSION}, valid_from, valid_to, tx_from FROM memories "
                    f"WHERE memory_id = ? AND {l_sql}",
                    [mid] + l_p,
                ).fetchone()
                if row is None:
                    raise NotFoundError(f"memory {mid} not found in tenant {ns.tenant_id}")
                version, valid_from, valid_to, tx_from = int(row[0]), row[1], row[2], row[3]
                if valid_to is None:
                    _check_closes_after_open("forget", mid, valid_from, at)
                    _check_records_after("forget", f"memory {mid}", tx_from, at)
                    self._close_memory(ns.tenant_id, mid, version, at, reason="forget")
                self._close_edges(
                    "edges_about",
                    EDGE_ABOUT_COLUMNS,
                    "src = ? AND tenant_id = ?",
                    [mid, ns.tenant_id],
                    at,
                    tenant_id=ns.tenant_id,
                    reason="forget",
                )
                self.execute(
                    "INSERT INTO anatid_audit (audit_id, tenant_id, memory_id, action, reason, "
                    "writer, happened_at) VALUES (?, ?, ?, 'forget_soft', ?, ?, ?)",
                    [new_id(), ns.tenant_id, mid, reason, writer, at],
                )
            return ForgetReceipt(
                memory_id=mid,
                tenant_id=ns.tenant_id,
                hard=False,
                at=at,
                memories_deleted=0,
                audit_rows_written=1,
                reason=reason,
            )

        with self._index_lifecycle(), self.transaction():
            row = self.execute(
                f"SELECT episode_id, content FROM memories WHERE memory_id = ? AND {tenant_sql()}",
                [mid, ns.tenant_id],
            ).fetchone()
            if row is None:
                raise NotFoundError(f"memory {mid} not found in tenant {ns.tenant_id}")
            ep = None if row[0] is None else int(row[0])
            # Read the content BEFORE the delete: erasure hooks need it to find verbatim copies
            # of the text (a tool-result row quotes the content, not just the id).
            purged_content = row[1]

            # One row per VERSION comes back from each versioned table; the receipt and the
            # index events count logical ids, the receipt also keeps the physical counts.
            about_rows = _ids(
                self.execute(
                    "DELETE FROM edges_about WHERE src = ? AND tenant_id = ? RETURNING edge_id",
                    [mid, ns.tenant_id],
                )
            )
            about_ids = _distinct(about_rows)
            about = len(about_ids)
            derived = self._index_erase(
                "edges_about", ns.tenant_id, about_ids, at=at, reason="forget_hard"
            )
            sup_ids = _ids(
                self.execute(
                    "DELETE FROM edges_supersedes WHERE (src = ? OR dst = ?) AND tenant_id = ? "
                    "RETURNING edge_id",
                    [mid, mid, ns.tenant_id],
                )
            )
            sup = len(sup_ids)
            derived = derived + self._index_erase(
                "edges_supersedes", ns.tenant_id, sup_ids, at=at, reason="forget_hard"
            )
            audit = _count(
                self.execute(
                    "DELETE FROM anatid_audit WHERE (memory_id = ? OR related_memory_id = ?) "
                    "AND tenant_id = ?",
                    [mid, mid, ns.tenant_id],
                )
            )
            mem_rows = _count(
                self.execute(
                    "DELETE FROM memories WHERE memory_id = ? AND tenant_id = ?",
                    [mid, ns.tenant_id],
                )
            )
            mems = 1 if mem_rows else 0
            if mems:
                derived = derived + self._index_erase(
                    "memories", ns.tenant_id, [mid], at=at, reason="forget_hard"
                )

            episodes = 0
            if ep is not None:
                # An episode is evidence for MEMORIES: it stays while any other memory cites it
                # and goes with the last one.  Entity and edge rows that carry its id are
                # provenance STAMPS -- remember() puts the memory's episode on every entity it
                # mints -- and a stamp may not keep the raw source text (which usually quotes
                # the erased content verbatim) in the file: with "Ada" outliving the memory, the
                # old rule kept "user said: <secret>" in `episodes` for as long as Ada existed.
                # The stamps are cleared so nothing dangles (doctor() would report it).
                still = _count(
                    self.execute(
                        f"SELECT count(*) FROM memories WHERE episode_id = ? AND {tenant_sql()}",
                        [ep, ns.tenant_id],
                    )
                )
                if still == 0:
                    for table in ("entities", "edges_about", "edges_relates"):
                        self.execute(
                            f"UPDATE {table} SET episode_id = NULL "
                            f"WHERE episode_id = ? AND tenant_id = ?",
                            [ep, ns.tenant_id],
                        )
                    episodes = _count(
                        self.execute(
                            "DELETE FROM episodes WHERE episode_id = ? AND tenant_id = ?",
                            [ep, ns.tenant_id],
                        )
                    )

            # The BM25 index tables hold the memory's content VERBATIM (anatid_fts_documents)
            # and its tokens (fts_main_*.terms).  A purge that skipped them would leave the
            # erased text in the file and findable by recall(). Safe when no index exists yet.
            fts = _schema.fts_purge(self.connection, ns.tenant_id, mid)
            # The BM25 watermark anatid_meta.fts_indexed_max_id is "the largest memory_id present
            # at the last rebuild".  When that is the memory being erased, the erased id would
            # survive in anatid_meta.  Clamp it to the largest id still in the index (NULL when
            # the index is now empty); fts_status() keeps reporting the index stale either way.
            self.execute(
                f"UPDATE anatid_meta SET fts_indexed_max_id = "
                f"(SELECT max(memory_id) FROM {_schema.FTS_DOCS_TABLE}) "
                f"WHERE fts_indexed_max_id = ?",
                [mid],
            )

            # Tables anatid does not own (conversation transcripts, caller-defined labels).
            # Inside the transaction on purpose: a hook that raises aborts the whole purge
            # rather than committing a half-erased file.
            extra = 0
            hooks = tuple(getattr(self, "erasure_hooks", ()))
            for hook in hooks:
                extra += int(hook(self, mid, ns.tenant_id, purged_content) or 0)
            # The bundled integrations' tables, by name from the catalog, whether or not THIS
            # handle constructed the session/store that registered a hook for them.  Hooks are
            # per handle; the file is not, and a purge from a second process, a maintenance
            # script or the MCP server's `forget` tool must reach the same rows.
            extra += _erasure.purge_bundled_tables(
                self,
                mid,
                ns.tenant_id,
                purged_content,
                skip=[t for t in (getattr(h, "table", None) for h in hooks) if t],
            )

        return ForgetReceipt(
            memory_id=mid,
            tenant_id=ns.tenant_id,
            hard=True,
            at=at,
            memories_deleted=mems,
            about_edges_deleted=about,
            supersedes_edges_deleted=sup,
            episodes_deleted=episodes,
            audit_rows_deleted=audit,
            fts_rows_deleted=fts,
            extra_rows_deleted=extra,
            memory_versions_deleted=mem_rows,
            about_edge_versions_deleted=len(about_rows),
            derived_rows_deleted=derived.rows_deleted,
            invalidated_generations=derived.generations_invalidated,
            reason=reason,
        )

    def unrelate(
        self,
        src: "int | str | Entity",
        dst: "int | str | Entity",
        *,
        rel_kind: str | None = None,
        tenant: int | Namespace | None = None,
        now: _dt.datetime | None = None,
    ) -> int:
        """Stop believing the ``RELATES_TO`` edges between two entities; returns how many closed.

        The inverse of :meth:`relate`, and undirected like the traversal: edges in either
        direction between ``src`` and ``dst`` are closed, all of them unless ``rel_kind``
        narrows it.  Closing follows the same rule as a memory: each current edge version gets
        ``tx_to = now`` and a successor with ``valid_to = now`` is inserted, so
        ``db.as_of(t)`` before ``now`` still traverses the edge and a transaction-time read
        before ``now`` still sees it open-ended.  Nothing is deleted; ``forget(hard=True)`` on
        a memory is the only verb that erases.  Marks the optional CSR snapshot stale.
        Unknown entity names raise :class:`~anatid.errors.NotFoundError`.
        """
        ns = self.resolve_tenant(tenant)
        at = to_utc_naive(now) or utcnow()
        s_id = self.entity_id(src, tenant=ns, create=False)
        d_id = self.entity_id(dst, tenant=ns, create=False)
        where = "((src = ? AND dst = ?) OR (src = ? AND dst = ?)) AND tenant_id = ?"
        params: list[Any] = [s_id, d_id, d_id, s_id, ns.tenant_id]
        if rel_kind is not None:
            where += " AND rel_kind = ?"
            params.append(str(rel_kind))
        with self.transaction():
            closed = self._close_edges(
                "edges_relates",
                EDGE_RELATES_COLUMNS,
                where,
                params,
                at,
                tenant_id=ns.tenant_id,
                reason="unrelate",
            )
        if closed:
            self.csr.note_edge_write()
        return len(closed)

    def prune(
        self,
        *,
        older_than: _dt.datetime | None = None,
        max_access_count: int | None = None,
        dry_run: bool = True,
        hard: bool = False,
        kinds: Sequence[str] | None = None,
        limit: int | None = None,
        tenant: int | Namespace | None = None,
        writer: str | None = None,
        reason: str | None = "prune",
        now: _dt.datetime | None = None,
    ) -> PruneReport:
        """Forget memories matching an age and/or usage policy.

        ``dry_run=True`` (the default) only reports what would go -- read
        :attr:`~anatid.types.PruneReport.memory_ids` before running it for real.  At least one of
        ``older_than`` / ``max_access_count`` must be given: anatid will not delete a whole
        tenant because an argument was forgotten.

        ``hard=True`` purges instead of closing validity; see :meth:`forget`.

        **Not atomic.**  Unlike every write verb, ``prune`` is one ``SELECT`` plus one
        transaction per memory it forgets.  A failure part-way (a ``ConflictError`` on one row, a
        crash) leaves the earlier ones committed and does not return a report at all -- run a
        ``dry_run`` first and keep its ``memory_ids`` if you need to resume.
        """
        ns = self.resolve_tenant(tenant)
        at = to_utc_naive(now) or utcnow()
        if older_than is None and max_access_count is None:
            raise ValidationError("prune needs older_than and/or max_access_count")
        if limit is not None:
            limit = _check_positive("limit", limit)
        if max_access_count is not None:
            max_access_count = _check_non_negative("max_access_count", max_access_count)

        visible, params = Visibility(ns.tenant_id).predicate()
        where = [visible]
        if older_than is not None:
            where.append("created_at < ?")
            params.append(to_utc_naive(older_than))
        if max_access_count is not None:
            where.append("coalesce(access_count, 0) <= ?")
            params.append(int(max_access_count))
        if kinds:
            where.append(f"kind IN ({', '.join('?' for _ in kinds)})")
            params += [str(k) for k in kinds]
        sql = (
            f"SELECT memory_id FROM memories WHERE {' AND '.join(where)} "
            f"ORDER BY created_at ASC, memory_id ASC"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        ids = [int(r[0]) for r in self.execute(sql, params).fetchall()]

        if dry_run:
            return PruneReport(
                dry_run=True,
                hard=hard,
                at=at,
                memory_ids=tuple(ids),
                older_than=to_utc_naive(older_than),
                max_access_count=max_access_count,
            )
        receipts = [
            self.forget(m, hard=hard, reason=reason, writer=writer, now=at, tenant=ns) for m in ids
        ]
        return PruneReport(
            dry_run=False,
            hard=hard,
            at=at,
            memory_ids=tuple(ids),
            receipts=tuple(receipts),
            older_than=to_utc_naive(older_than),
            max_access_count=max_access_count,
        )

    # ------------------------------------------------------------------ provenance

    def provenance(
        self,
        memory_id: int,
        *,
        tenant: int | Namespace | None = None,
        max_depth: int = _MAX_CHAIN,
    ) -> Provenance:
        """Walk a memory's SUPERSEDES chain back to the original assertion and its evidence.

        Returns the chain newest-first, the episodes behind it, the SUPERSEDES edges traversed
        and every distinct writer involved -- the answer to "where did this belief come from and
        who put it there".  ``Provenance.versions`` adds the other axis for the memory asked
        about: every physical version of it, oldest first, which is the record of when the
        database changed its mind about that one memory (:meth:`versions`).

        A hard purge anywhere in the chain truncates it: the erased link is genuinely gone.
        """
        ns = self.resolve_tenant(tenant)
        chain: list[Memory] = []
        edges: list[Edge] = []
        seen: set[int] = set()
        cur: int | None = int(memory_id)
        while cur is not None and cur not in seen and len(chain) < max_depth:
            seen.add(cur)
            mem = self.get(cur, tenant=ns, with_embedding=False)
            if mem is None:
                break
            chain.append(mem)
            row = self.execute(
                f"SELECT edge_id, dst, tx_from, writer FROM edges_supersedes "
                f"WHERE src = ? AND {tenant_sql()} ORDER BY tx_from, edge_id LIMIT 1",
                [cur, ns.tenant_id],
            ).fetchone()
            if row is None:
                break
            edges.append(
                Edge(
                    edge_id=int(row[0]),
                    edge_type=SUPERSEDES,
                    src=cur,
                    dst=int(row[1]),
                    tenant_id=ns.tenant_id,
                    tx_from=row[2],
                    writer=row[3],
                )
            )
            cur = int(row[1])
        if not chain:
            raise NotFoundError(f"memory {memory_id} not found in tenant {ns.tenant_id}")

        ep_ids = [m.episode_id for m in chain if m.episode_id is not None]
        episodes: list[Episode] = []
        if ep_ids:
            marks = ", ".join("?" for _ in ep_ids)
            rows = {
                int(r[0]): Episode.from_row(r)
                for r in self.execute(
                    f"SELECT {_EPI_SELECT} FROM episodes WHERE episode_id IN ({marks}) "
                    f"AND {tenant_sql()}",
                    [*ep_ids, ns.tenant_id],
                ).fetchall()
            }
            episodes = [rows[e] for e in ep_ids if e in rows]

        writers: list[str] = []
        for m in chain:
            if m.writer and m.writer not in writers:
                writers.append(m.writer)
        return Provenance(
            memory_id=int(memory_id),
            chain=tuple(chain),
            episodes=tuple(episodes),
            edges=tuple(edges),
            writers=tuple(writers),
            versions=tuple(self.versions(int(memory_id), tenant=ns)),
        )

    # ------------------------------------------------------------------ misc

    def stats(self, *, tenant: int | Namespace | None = None, all_tenants: bool = False) -> dict:
        """Row counts for this tenant (or the whole file), plus the active expansion path.

        A versioned table (``memories``, ``edges_about``, ``edges_relates``) is counted by
        logical row, i.e. live versions, so a correction does not change ``memories``;
        ``memory_versions`` is the physical row count of ``memories``.
        """
        ns = self.resolve_tenant(tenant)
        out: dict[str, Any] = {}
        tables = [
            "memories",
            "entities",
            "episodes",
            "edges_about",
            "edges_relates",
            "edges_supersedes",
            "anatid_audit",
        ]
        vis = Visibility(ns.tenant_id)
        t_sql, t_p = vis.tenant()
        l_sql, l_p = vis.live()
        for t in tables:
            if t in VERSIONED_TABLES:
                where, params = (_LIVE, []) if all_tenants else (l_sql, l_p)
            else:
                where, params = ("TRUE", []) if all_tenants else (t_sql, t_p)
            out[t] = _count(self.execute(f"SELECT count(*) FROM {t} WHERE {where}", params))
        out["memory_versions"] = _count(
            self.execute(
                "SELECT count(*) FROM memories" + ("" if all_tenants else f" WHERE {t_sql}"),
                [] if all_tenants else t_p,
            )
        )
        visible, vp = vis.predicate()
        out["current_memories"] = _count(
            self.execute(f"SELECT count(*) FROM memories WHERE {visible}", vp)
        )
        out["tenant_id"] = ns.tenant_id
        out["expand_path"] = self.csr.active
        return out


# --------------------------------------------------------------------------- function forms


def remember(db: MemoryVerbs, content: str, **kw) -> Memory:
    """Function form of :meth:`MemoryVerbs.remember`."""
    return db.remember(content, **kw)


def recall(db: MemoryVerbs, query: str | None = None, **kw) -> RecallHits:
    """Function form of :meth:`MemoryVerbs.recall`."""
    return db.recall(query, **kw)


def recall_2hop(db: MemoryVerbs, seed_entity, **kw) -> list[Memory]:
    """Function form of :meth:`MemoryVerbs.recall_2hop`."""
    return db.recall_2hop(seed_entity, **kw)


def context(db: MemoryVerbs, entity, **kw) -> list[Memory]:
    """Function form of :meth:`MemoryVerbs.context`."""
    return db.context(entity, **kw)


def supersede(db: MemoryVerbs, old_id: int, content: str, **kw) -> Memory:
    """Function form of :meth:`MemoryVerbs.supersede`."""
    return db.supersede(old_id, content, **kw)


def reinforce(db: MemoryVerbs, memory_id: int, **kw) -> Memory:
    """Function form of :meth:`MemoryVerbs.reinforce`."""
    return db.reinforce(memory_id, **kw)


def forget(db: MemoryVerbs, memory_id: int, **kw) -> ForgetReceipt:
    """Function form of :meth:`MemoryVerbs.forget`."""
    return db.forget(memory_id, **kw)


def unrelate(db: MemoryVerbs, src, dst, **kw) -> int:
    """Function form of :meth:`MemoryVerbs.unrelate`."""
    return db.unrelate(src, dst, **kw)


def versions(db: MemoryVerbs, memory_id: int, **kw) -> list[Memory]:
    """Function form of :meth:`MemoryVerbs.versions`."""
    return db.versions(memory_id, **kw)


def prune(db: MemoryVerbs, **kw) -> PruneReport:
    """Function form of :meth:`MemoryVerbs.prune`."""
    return db.prune(**kw)


def as_of(timestamp: _dt.datetime, *, tx_time: _dt.datetime | None = None) -> AsOf:
    """Build an :class:`~anatid.types.AsOf` scope for a timestamp.

    ``db.as_of(t)`` returns a bound view instead; this is the plain value, for passing as
    ``as_of=`` to any read.  Remember what it is: anatid's WHERE filter over the bitemporal
    columns, not a DuckDB feature.
    """
    ts = to_utc_naive(timestamp)
    return AsOf(valid_time=ts, tx_time=to_utc_naive(tx_time) if tx_time is not None else ts)


def provenance(db: MemoryVerbs, memory_id: int, **kw) -> Provenance:
    """Function form of :meth:`MemoryVerbs.provenance`."""
    return db.provenance(memory_id, **kw)
