"""Vector search backends: the exact oracle, and DuckDB's experimental HNSW.

anatid answers the vector arm of :func:`anatid.recall.hybrid_recall` from one of three backends.

``exact``
    A brute-force ``array_cosine_similarity`` scan of the tenant's visible embeddings.  It is
    the DEFAULT, it is the correctness oracle, and it is what every fallback path runs.  Its
    cost is linear in the tenant's rows, which is what :data:`anatid.recall.BRUTE_FORCE_CEILING`
    guards.
``duckdb_vss``
    Opt in.  An HNSW index from DuckDB's ``vss`` extension over a COLD generation of the
    derived-index framework (:mod:`anatid.derived`), merged with an exact scan of everything
    written since that generation's watermark.  The HNSW structure only ever chooses which rows
    are considered; every row that survives is scored with the same cosine expression the exact
    path uses, so a ranking never depends on an approximation.
``owned_hnsw``
    Not implemented.  :func:`attach` raises :class:`NotImplementedError` naming it, rather than
    silently falling back to something else.

Turning it on::

    from anatid import Anatid, vector

    db = Anatid.open("memory.anatid", tenant=1, embedding_dim=768)
    vector.attach(db, backend="duckdb_vss")  # records the index in the FILE
    db.maintain_indexes()  # builds, validates and publishes a generation

After that every ``db.recall(embedding=...)`` goes through the index, on this handle and on
every other handle that opens the file, and :meth:`anatid.Anatid.index_health` reports it.
Nothing else changes: the same rows come back, in the same order, with the same scores.

Why a generation rather than an index on ``memories``
----------------------------------------------------

An HNSW index over the live table would have to answer historical reads, and it cannot: a
structure built over current state does not know what was visible last Tuesday.  It would also
have to be correct after every write, and DuckDB's HNSW is not incrementally correct under
deletes (its persistence is documented as experimental).  So the structure is built over a
frozen, tenant-scoped copy of the cold rows, published through the framework, and the read
merges:

    HNSW top-N over the cold generation
      UNION exact scan of the journal's pending rows and anything past the watermark
      MINUS journal tombstones
      then tenant and temporal visibility on the CANONICAL rows
      then exact cosine rescoring
      then the top k

Past :func:`merge_cap` changed documents the merge stops being cheaper than the scan it
replaces, and the read falls back to the exact scan and reports ``stale_generation``.

Every step after the first reads ``memories``, so a stale, corrupt, absent or deliberately
sabotaged generation costs recall and latency, never correctness.  Any ``as_of`` read skips the
structure entirely and reports :attr:`~anatid.derived.HealthReason.HISTORICAL_QUERY`.

What the search reports
-----------------------

:func:`search` returns a :class:`VectorSearch`, which names the backend that actually answered
and the :class:`~anatid.derived.HealthReason` for it: ``fresh``, ``unvalidated``,
``stale_generation``, ``historical_query``, ``rebuild_in_progress``, ``load_failure`` or
``absent``.  "The index was not used" is not an actionable thing to tell an operator; "the vss
extension failed to load" is.

What it costs, and where it starts paying
-----------------------------------------

Measured on this repository's spike dataset (``spike/data/small``, 64 dimensions, macOS arm64,
10 threads, DuckDB 1.5.5), 100 queries per row, defaults throughout.  ``recall@k`` is against
the exact oracle over the same visible rows:

    rows/tenant   build   validate   k    depth   recall@k   p50 vss    p50 exact
    9,500         0.59 s  0.11 s     10   246     1.0000     2.8 ms     2.5 ms
    9,500         0.59 s  0.11 s     50   246     0.9982     2.6 ms     2.6 ms
    95,000        6.74 s  0.52 s     10   779     1.0000     6.3 ms     15.2 ms
    95,000        6.74 s  0.52 s     50   779     0.9984     5.6 ms     15.9 ms

``tests/test_vector_backends.py`` measures the same numbers on every run and logs them, so
they can be checked rather than believed; its latencies come out a few tenths of a millisecond
higher because it interleaves the oracle call with the measured one.

Read the last two columns of that table before turning the backend on.  The exact scan is linear in
the tenant's rows and the graph search is not, but the graph search carries a fixed cost of
about 0.6 ms of catalog and journal reads plus a search depth that has to grow with the corpus,
so the two cross somewhere around 15,000 rows per tenant: below that the exact scan is as fast
or faster and there is nothing to gain, and by 95,000 rows the index is 2.4 to 2.8 times faster.
That is why ``exact`` is still the default and this is opt in.

The recall figures hold at the design's promotion criterion of 0.98 with the defaults, at
``k = 50`` as well as ``k = 10``, and ``k = 50`` is the size the arm actually runs at
(:data:`anatid.recall.DEFAULT_CANDIDATES`).  The spike's embeddings are independent Gaussian
vectors, which is close to the worst case for a proximity graph: real embeddings cluster and
need less depth for the same recall.  Reaching 0.98 on them needs a depth well above the
extension's default of 64, which is why :data:`DEFAULT_EF_SEARCH` is 800.

Three DuckDB facts this module is built around
----------------------------------------------

``hnsw_enable_experimental_persistence`` must be set before ``CREATE INDEX`` on a file-backed
database, and it is a per-CONNECTION setting, so the build sets it on the connection it builds
with rather than once at open time.

``CREATE INDEX ... USING HNSW`` inside a transaction is deferred to the COMMIT and costs about
five times what it costs outside one (47 s against 9.5 s on 95,000 vectors here).  So
:meth:`VectorIndex._build` copies the rows inside the framework's build transaction, which is
what makes the copy, the watermark and the absorbed journal rows one snapshot, and
:meth:`VectorIndex.build_next` builds the graph over that frozen table immediately after the
commit.

An HNSW index does NOT keep the ``ef_search`` it was created with across a reopen: DuckDB
records it as ``CREATE INDEX ... USING HNSW (embedding)`` with the options dropped, and recall
silently falls back to the extension's default of 64.  Depth therefore has to be set per query
with ``SET hnsw_ef_search``, and a build-time ``ef_search`` option would be a lie after the
first restart.  :func:`search` sets it every time, scaled to the generation's size by
:func:`search_depth`.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import logging
import math
import threading
import time
import weakref
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Sequence

import duckdb

from .derived import (
    INSERT,
    DerivedIndex,
    Generation,
    HealthReason,
    IndexDefinition,
    ValidationReport,
    journal_latest_sql,
)
from .errors import EmbeddingDimensionError, IndexGenerationError
from .schema import INDEX_GENERATIONS_TABLE, INDEX_REGISTRY_TABLE, quote_ident
from .schema import embedding_dim as _recorded_dim
from .types import CURRENT, AsOf, Namespace
from .visibility import Visibility, tenant_sql

log = logging.getLogger("anatid.vector")

__all__ = [
    "VectorBackend",
    "BACKENDS",
    "VECTOR_INDEX_NAME",
    "DEFAULT_EF_CONSTRUCTION",
    "DEFAULT_EF_SEARCH",
    "DEFAULT_M",
    "DEFAULT_METRIC",
    "DEFAULT_OVERFETCH",
    "EF_REFERENCE_ROWS",
    "MAX_MERGED_JOURNAL_ROWS",
    "MERGE_FLOOR",
    "MERGE_RATIO",
    "merge_cap",
    "VectorSearch",
    "VectorPlan",
    "VectorIndex",
    "attach",
    "detach",
    "vss_error",
    "load_vss",
    "index_of",
    "resolve",
    "search",
    "search_depth",
    "exact_search",
    "scan_rows",
]

#: The backends :func:`attach` accepts.  ``exact`` registers nothing (it is the default path),
#: ``duckdb_vss`` registers a :class:`VectorIndex`, ``owned_hnsw`` raises.
VectorBackend = Literal["exact", "duckdb_vss", "owned_hnsw"]

BACKENDS: tuple[str, ...] = ("exact", "duckdb_vss", "owned_hnsw")

#: The index name in :data:`~anatid.schema.INDEX_REGISTRY_TABLE`.  Matches the placeholder
#: :class:`~anatid.derived.NullIndex` every :class:`~anatid.derived.IndexRegistry` starts with,
#: so registering a real one replaces the placeholder.
VECTOR_INDEX_NAME = "vector"

#: The DuckDB extension providing HNSW.
VSS_EXTENSION = "vss"

#: Prefix for a generation's storage table.  ``anatid_idx_vector_t<tenant>_g<n>``.
STORAGE_PREFIX = "anatid_idx"

#: Prefix for the HNSW index on that table.
HNSW_PREFIX = "anatid_hnsw"

DEFAULT_METRIC = "cosine"
#: Neighbours per node in the proximity graph.  Higher is a better graph, a slower build and a
#: bigger index.
DEFAULT_M = 32
#: Candidate list size during the build.
DEFAULT_EF_CONSTRUCTION = 200
#: Candidate list size during a search, set per query (see the module docstring).  800 is what
#: 64-dimensional Gaussian embeddings need for recall at k of 0.98 and above at 100k rows.
#: It is the depth AT :data:`EF_REFERENCE_ROWS`; a smaller generation scales it down.
DEFAULT_EF_SEARCH = 800

#: The base size :data:`DEFAULT_EF_SEARCH` is calibrated for.  A generation with fewer rows
#: searches ``ef_search * sqrt(rows / EF_REFERENCE_ROWS)`` deep, because the depth a proximity
#: graph needs for a given recall grows with the corpus and paying 100k depth on a 10k base is
#: how an approximate search ends up slower than the scan it replaced.  Measured on the spike
#: data: 9,500 rows reach recall at 10 of 0.999 at depth 200, 95,000 rows need 800.  Above the
#: reference size the configured value is the ceiling; raise ``ef_search`` for a bigger corpus.
EF_REFERENCE_ROWS = 100_000
#: The HNSW is asked for this multiple of the requested top-N, so that candidates dropped by
#: the visibility and kind filters do not eat the answer.  It also raises the effective search
#: depth, which is ``max(ef_search, limit)`` in the extension.
DEFAULT_OVERFETCH = 4

#: The hard ceiling on documents one read will merge.  It sits at the default
#: :attr:`~anatid.derived.MaintenancePolicy.rebuild_after_rows`, so a database whose maintenance
#: is running never reaches it.
MAX_MERGED_JOURNAL_ROWS = 10_000

#: The merge is also capped at this fraction of the generation's rows, because the exact arm
#: over the pending documents is a random-access read of the ``FLOAT[N]`` column and DuckDB
#: decompresses a whole vector of rows to reach one of them.  Measured at 95,000 rows: 0.9 ms
#: for 10 pending documents, 3.3 ms for 1,000, 11.2 ms for 5,000, 33 ms for 10,000, against
#: 16.2 ms for the full scan the merge is meant to beat.  Break-even is near 5% of the base,
#: which is also :attr:`~anatid.derived.MaintenancePolicy.rebuild_after_ratio`: past it the
#: read falls back to the scan and reports ``stale_generation``, and the same policy that set
#: the number has already called for a rebuild.
MERGE_RATIO = 0.05

#: The cap never goes below this, so a small generation does not fall back over a handful of
#: writes.  At that size the whole scan is a couple of milliseconds either way.
MERGE_FLOOR = 1_000

#: Probes :meth:`VectorIndex._validate` runs against the oracle, and the recall they must reach.
DEFAULT_VALIDATE_PROBES = 24
DEFAULT_VALIDATE_RECALL = 0.95
#: Top-N each validation probe compares.
VALIDATE_TOPN = 10


# --------------------------------------------------------------------------- helpers


def _literal(vec: Sequence[float]) -> str:
    """An embedding as the text DuckDB casts to ``FLOAT[N]``.

    Delegates to :func:`anatid.recall.embedding_literal` so there is one serialisation.  The
    import is deferred because :mod:`anatid.recall` imports this module.
    """
    from .recall import embedding_literal

    return embedding_literal(vec)


def _kinds(alias: str, kinds: Sequence[str] | None) -> tuple[str, list]:
    from .recall import _kind_filter

    return _kind_filter(alias, kinds)


def _ids(ids: Sequence[int]) -> str:
    from .recall import _int_list

    return _int_list(ids)


def _tenant_of(tenant: Any) -> int:
    if isinstance(tenant, Namespace):
        return int(tenant.tenant_id)
    return int(tenant)


# --------------------------------------------------------------------------- the extension

_VSS_LOCK = threading.Lock()
#: Per connection: ``None`` once ``vss`` is loaded, the error text when it could not be.
#: Keyed weakly so a closed connection does not keep an entry alive.
_VSS_STATE: "weakref.WeakKeyDictionary[Any, str | None]" = weakref.WeakKeyDictionary()
#: Per connection: the ``hnsw_ef_search`` this module last set, so a query that wants the same
#: value costs no statement.  The setting is per connection, like the persistence flag.
_EF_STATE: "weakref.WeakKeyDictionary[Any, int]" = weakref.WeakKeyDictionary()
#: Connections on which ``hnsw_enable_experimental_persistence`` has been set.
_PERSIST_STATE: "weakref.WeakKeyDictionary[Any, bool]" = weakref.WeakKeyDictionary()


def load_vss(con) -> str | None:
    """Load the ``vss`` extension on ``con``.  ``None`` on success, the error text otherwise.

    Cached per connection, failures included: an air-gapped machine must not pay an
    ``INSTALL`` timeout on every recall.  ``LOAD`` is tried before ``INSTALL`` so an already
    installed extension needs no repository at all.
    """
    with _VSS_LOCK, contextlib.suppress(TypeError):
        if con in _VSS_STATE:
            return _VSS_STATE[con]
    error: str | None = None
    try:
        con.execute(f"LOAD {VSS_EXTENSION}")
    except duckdb.Error:
        try:
            con.execute(f"INSTALL {VSS_EXTENSION}")
            con.execute(f"LOAD {VSS_EXTENSION}")
        except duckdb.Error as exc:
            error = f"the {VSS_EXTENSION} extension could not be loaded: {exc}"
    with _VSS_LOCK, contextlib.suppress(TypeError):
        _VSS_STATE[con] = error
    return error


def vss_error(con) -> str | None:
    """:func:`load_vss` without loading twice.  ``None`` when HNSW is available on ``con``."""
    return load_vss(con)


def _enable_persistence(con) -> None:
    """Allow ``CREATE INDEX ... USING HNSW`` on a file-backed database.

    DuckDB refuses one otherwise, and the flag is per connection, so it is set on the
    connection that builds rather than once when the database is opened.
    """
    with contextlib.suppress(TypeError):
        if _PERSIST_STATE.get(con):
            return
    con.execute("SET hnsw_enable_experimental_persistence = true")
    with contextlib.suppress(TypeError):
        _PERSIST_STATE[con] = True


def _set_ef_search(con, value: int) -> None:
    """Set ``hnsw_ef_search`` for the next query on ``con``.

    A persisted HNSW index loses the ``ef_search`` it was created with, so this is the only
    lever that survives a restart.  Skipped when the connection already carries the value.
    """
    want = int(value)
    with contextlib.suppress(TypeError):
        if _EF_STATE.get(con) == want:
            return
    con.execute(f"SET hnsw_ef_search = {want}")
    with contextlib.suppress(TypeError):
        _EF_STATE[con] = want


# --------------------------------------------------------------------------- value types


@dataclass(frozen=True, slots=True)
class VectorSearch:
    """The vector arm's answer, and how it was reached.

    ``hits`` is ``[(memory_id, cosine similarity)]`` newest score first, exactly what
    :func:`anatid.recall.vector_arm` returns.  ``backend`` is the backend that ANSWERED, which
    is ``"exact"`` whenever the approximate path was declined or failed, and ``reason`` says
    why (``fresh`` when the HNSW answered).
    """

    hits: list[tuple[int, float]]
    backend: str
    reason: HealthReason
    detail: str = ""
    generation: int | None = None
    #: Documents the exact arm scored ON TOP OF the base generation: the journal's pending rows
    #: plus anything past the watermark.  Zero on the exact path, where the scan is the whole
    #: visible set and :func:`scan_rows` is what reports its size.
    scanned: int = 0
    #: Candidates the HNSW produced before visibility was applied.
    candidates: int = 0

    @property
    def approximate(self) -> bool:
        """True when an approximate structure chose the candidates.  Never affects the score."""
        return self.backend != "exact"

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "reason": self.reason.value,
            "detail": self.detail,
            "generation": self.generation,
            "scanned": self.scanned,
            "candidates": self.candidates,
            "hits": len(self.hits),
        }


@dataclass(frozen=True, slots=True)
class VectorPlan:
    """How one vector read will be answered, decided before any scoring runs.

    ``usable`` is the decision.  When it is False the read is an exact scan and ``reason`` is
    the fallback reason.  When it is True, ``storage`` is the generation's table, ``pending``
    the documents the exact arm must add and ``tombstones`` the ones it must subtract.
    """

    usable: bool
    reason: HealthReason
    detail: str = ""
    generation: Generation | None = None
    storage: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    pending: tuple[int, ...] = ()
    tombstones: tuple[int, ...] = ()

    @property
    def backend(self) -> str:
        return "duckdb_vss" if self.usable else "exact"


# --------------------------------------------------------------------------- catalog reads
#
# Read straight from the catalog tables rather than through a DerivedIndex, because the read
# path holds a bare DuckDB connection: anatid.recall takes a connection, not a database handle,
# and a second handle on the file may hold no vector code at all.  The catalog is the file's,
# so both see the same generation.

_REGISTRY_COLUMNS = "kind, per_tenant, source_table, source_id_column, delta_mode, params, enabled"
_GENERATION_COLUMNS = (
    "generation, tenant_id, watermark_id, watermark_ts, built_at, validated, published, "
    "coalesce(published_unvalidated, FALSE), stats, notes"
)

#: Bumped whenever a vector index is attached to or detached from any database in this process.
#: The per-connection definition cache compares against it, so the common case (no vector index
#: in the file) costs no statement per query after the first.
_DEFINITIONS_EPOCH = 0
_DEFINITION_CACHE: "weakref.WeakKeyDictionary[Any, tuple[int, dict[str, Any] | None]]"
_DEFINITION_CACHE = weakref.WeakKeyDictionary()


def _bump_definitions() -> None:
    """Invalidate every connection's cached definition.  The epoch does it; no entry is touched."""
    global _DEFINITIONS_EPOCH
    with _VSS_LOCK:
        _DEFINITIONS_EPOCH += 1


def _json_params(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        out = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def _inline(sql: str, params: Sequence[Any]) -> str:
    """Render a parameterised catalog query with its parameters as literals.

    Binding a parameter costs about 0.35 ms per statement in duckdb-python 1.5.5, which is more
    than these catalog reads themselves cost, and the vector arm runs two of them per query.
    The same reasoning as :func:`anatid.recall._int_list`.  Only an integer and a string that
    passes :func:`anatid.schema.quote_ident` are accepted, so nothing that could close a quote
    reaches the statement; anything else raises rather than being interpolated.
    """
    out = sql
    for value in params:
        if isinstance(value, bool) or value is None:
            raise ValueError(f"cannot inline {value!r} into a catalog query")
        if isinstance(value, int):
            literal = str(int(value))
        else:
            quote_ident(str(value))
            literal = f"'{value}'"
        out = out.replace("?", literal, 1)
    if "?" in out:
        raise ValueError("not every parameter of the catalog query was inlined")
    return out


def _definition(con) -> dict[str, Any] | None:
    """The persisted vector definition as a plain dict, or ``None`` when the file has none.

    Cached per connection against :data:`_DEFINITIONS_EPOCH`.  A file with no vector index is
    the default, and it must not pay a catalog read on every recall.
    """
    with _VSS_LOCK:
        epoch = _DEFINITIONS_EPOCH
    with contextlib.suppress(TypeError):
        hit = _DEFINITION_CACHE.get(con)
        if hit is not None and hit[0] == epoch:
            return hit[1]
    out: dict[str, Any] | None = None
    try:
        row = con.execute(
            f"SELECT {_REGISTRY_COLUMNS} FROM {INDEX_REGISTRY_TABLE} "
            f"WHERE index_name = '{VECTOR_INDEX_NAME}'"
        ).fetchone()
    except duckdb.Error:  # a file older than the registry table has no derived indexes
        row = None
    if row is not None and bool(row[6]):
        out = {
            "kind": row[0],
            "per_tenant": bool(row[1]),
            "source_table": row[2],
            "source_id_column": row[3],
            "delta_mode": row[4],
            "params": _json_params(row[5]),
        }
    with contextlib.suppress(TypeError):
        _DEFINITION_CACHE[con] = (epoch, out)
    return out


def _generation_row(con, tenant_id: int) -> tuple[Generation | None, bool]:
    """The published generation for one tenant, and whether one is being built.  One statement.

    Read from the catalog rather than through a :class:`~anatid.derived.DerivedIndex`, because
    the read path holds a bare DuckDB connection: :mod:`anatid.recall` takes a connection, not a
    database handle, and a second handle on the file may hold no vector code at all.  The
    catalog belongs to the file, so both see the same generation.
    """
    rows = con.execute(
        _inline(
            f"SELECT {_GENERATION_COLUMNS} FROM {INDEX_GENERATIONS_TABLE} "
            f"WHERE index_name = ? AND {tenant_sql()} "
            f"AND (published OR notes = 'building') ORDER BY generation",
            [VECTOR_INDEX_NAME, int(tenant_id)],
        )
    ).fetchall()
    building = any(r[9] == "building" for r in rows)
    published = [r for r in rows if bool(r[6])]
    if not published:
        return None, building
    r = published[-1]
    gen = Generation(
        index_name=VECTOR_INDEX_NAME,
        generation=int(r[0]),
        tenant_id=None if r[1] is None else int(r[1]),
        watermark_id=None if r[2] is None else int(r[2]),
        watermark_ts=r[3],
        built_at=r[4],
        validated=bool(r[5]),
        published=True,
        published_unvalidated=bool(r[7]),
        stats=_json_params(r[8]),
        notes=r[9],
    )
    return gen, building


def merge_cap(base_rows: Any) -> int:
    """How many changed documents a read will merge before it falls back to the exact scan.

    See :data:`MERGE_RATIO`.  The merge is only worth doing while it is cheaper than the scan
    it replaces, and that stops being true at a few percent of the base.
    """
    rows = int(base_rows) if isinstance(base_rows, (int, float)) else 0
    return int(min(MAX_MERGED_JOURNAL_ROWS, max(MERGE_FLOOR, rows * MERGE_RATIO)))


def _journal(con, tenant_id: int, generation: int, cap: int) -> tuple[list[int], list[int], bool]:
    """``(pending, tombstoned, overflowed)`` for one tenant against one generation.

    The journal's latest operation per document wins, which is what makes an id that was
    purged and then reused come back as pending rather than as a tombstone.  The read is
    capped at ``cap``: past that the merge is no longer cheaper than the exact scan and the
    caller falls back and says so.
    """
    latest, params = journal_latest_sql(VECTOR_INDEX_NAME, int(tenant_id), int(generation))
    rows = con.execute(
        f"SELECT op, doc_id FROM ({_inline(latest, params)}) LIMIT {int(cap) + 1}"
    ).fetchall()
    if len(rows) > int(cap):
        return [], [], True
    pending = [int(r[1]) for r in rows if r[0] == INSERT]
    tombs = [int(r[1]) for r in rows if r[0] != INSERT]
    return pending, tombs, False


def _above_watermark(
    con, tenant_id: int, watermark_id: int | None, budget: int
) -> tuple[list[int], bool]:
    """Visible rows past the generation's watermark, whatever the journal says.

    The journal covers every write that went through a verb, which is every write anatid makes.
    This covers the rest: a row inserted by raw SQL through :attr:`anatid.Anatid.connection`
    journals nothing, and would otherwise be invisible to this arm while the exact scan
    returned it, which is exactly the kind of quiet disagreement between two paths that this
    framework exists to remove.  It is also, in those words, what the design document says the
    read merges.

    Cheap because it is a range predicate on the id column: 0.12 ms against 95,000 rows, where
    the zone maps prune every block below the watermark.  ``budget`` is what is left of
    :func:`merge_cap` after the journal.
    """
    w, wp = Visibility.at(int(tenant_id), CURRENT).predicate("m", inline_tenant=True)
    sql = f"SELECT m.memory_id FROM memories m WHERE {w} AND m.embedding IS NOT NULL"
    if watermark_id is not None:
        sql += f" AND m.memory_id > {int(watermark_id)}"
    rows = con.execute(f"{sql} LIMIT {max(0, int(budget)) + 1}", wp).fetchall()
    if len(rows) > max(0, int(budget)):
        return [], True
    return [int(r[0]) for r in rows], False


# --------------------------------------------------------------------------- planning


def _storage_name(generation: Generation) -> str:
    return generation.storage_name(STORAGE_PREFIX)


def _hnsw_name(generation: Generation) -> str:
    name = f"{HNSW_PREFIX}_{generation.storage_suffix}"
    quote_ident(name)
    return name


def _exact_plan(reason: HealthReason, detail: str) -> VectorPlan:
    return VectorPlan(usable=False, reason=reason, detail=detail)


def resolve(
    con,
    *,
    tenant_id: int,
    as_of: AsOf = CURRENT,
    index: "VectorIndex | None" = None,
) -> VectorPlan:
    """Decide how a vector read for ``tenant_id`` will be answered.

    Every branch that declines the approximate path names the reason, and every one of them
    still answers the query: the exact scan is always available and always right.  The order is
    the order in which a reason makes the structure unusable, not the order that is cheapest to
    check, so the reported reason is the FIRST thing wrong rather than the last.

    ``index`` is the accelerator object when the caller holds one.  It is used for its
    ``load_error`` only; the generation itself is read from the catalog either way, so a handle
    with no vector code plans the same read as the handle that built the index.
    """
    scope = AsOf.coerce(as_of)
    if index is not None and index.load_error:
        return _exact_plan(HealthReason.LOAD_FAILURE, index.load_error)
    if not scope.is_current:
        return _exact_plan(
            HealthReason.HISTORICAL_QUERY,
            "a current-state index cannot answer an as_of read; the exact scan does",
        )
    definition = _definition(con)
    if definition is None:
        return _exact_plan(
            HealthReason.ABSENT,
            "this database defines no vector index; exact search is the default",
        )
    params = dict(definition.get("params") or {})
    backend = str(params.get("backend", "duckdb_vss"))
    if backend != "duckdb_vss":
        return _exact_plan(
            HealthReason.ABSENT, f"the vector index is configured with the {backend!r} backend"
        )
    error = load_vss(con)
    if error:
        return _exact_plan(HealthReason.LOAD_FAILURE, error)
    try:
        gen, building = _generation_row(con, int(tenant_id))
    except (duckdb.Error, ValueError) as exc:
        # ValueError: a catalog row whose tenant or generation does not make an identifier.
        return _exact_plan(HealthReason.LOAD_FAILURE, f"the index catalog could not be read: {exc}")
    if gen is None:
        if building:
            return _exact_plan(
                HealthReason.REBUILD_IN_PROGRESS,
                "no generation is published and one is being built",
            )
        return _exact_plan(HealthReason.ABSENT, "no generation has been published")
    if not gen.validated and not gen.usable_without_validation:
        return _exact_plan(
            HealthReason.STALE_GENERATION,
            gen.notes or f"generation {gen.generation} is not validated",
        )
    cap = merge_cap(gen.stats.get("rows"))
    try:
        pending, tombs, overflowed = _journal(con, int(tenant_id), gen.generation, cap)
        if not overflowed:
            extra, overflowed = _above_watermark(
                con, int(tenant_id), gen.watermark_id, cap - len(pending)
            )
            if extra:
                pending = sorted(set(pending) | set(extra))
    except (duckdb.Error, ValueError) as exc:
        return _exact_plan(HealthReason.LOAD_FAILURE, f"the index journal could not be read: {exc}")
    if overflowed:
        return _exact_plan(
            HealthReason.STALE_GENERATION,
            f"more than {cap} document(s) have changed or arrived since generation "
            f"{gen.generation}, which is where merging them stops being cheaper than the "
            f"exact scan; rebuild it (maintain_indexes)",
        )
    reason = HealthReason.UNVALIDATED if gen.usable_without_validation else HealthReason.FRESH
    detail = (
        f"generation {gen.generation} was published without validation"
        if gen.usable_without_validation
        else f"generation {gen.generation}, {len(pending)} pending, {len(tombs)} tombstoned"
    )
    return VectorPlan(
        usable=True,
        reason=reason,
        detail=detail,
        generation=gen,
        storage=_storage_name(gen),
        params=params,
        pending=tuple(pending),
        tombstones=tuple(tombs),
    )


# --------------------------------------------------------------------------- the read


def exact_search(
    con,
    *,
    tenant_id: int,
    embedding: Sequence[float],
    dim: int,
    topn: int,
    as_of: AsOf = CURRENT,
    kinds: Sequence[str] | None = None,
) -> list[tuple[int, float]]:
    """Brute-force cosine top-N over the tenant's visible memories.  The oracle.

    Identical to what :func:`anatid.recall.vector_arm` ran in 0.1.1, and the answer every
    other path is measured against.  Ties break on ``memory_id`` ascending.
    """
    w, wp = Visibility.at(int(tenant_id), as_of).predicate("m")
    kf, kp = _kinds("m", kinds)
    sql = (
        f"SELECT m.memory_id, array_cosine_similarity(m.embedding, ?::FLOAT[{int(dim)}]) AS score "
        f"FROM memories m WHERE {w} AND m.embedding IS NOT NULL{kf} "
        f"ORDER BY score DESC, m.memory_id ASC LIMIT ?"
    )
    params = [_literal(embedding)] + wp + kp + [int(topn)]
    return [(int(r[0]), float(r[1])) for r in con.execute(sql, params).fetchall()]


def search_depth(plan: VectorPlan, limit: int) -> int:
    """How deep the HNSW search goes for one query: ``hnsw_ef_search``.

    Scaled to the generation's size (:data:`EF_REFERENCE_ROWS`) and never below the number of
    candidates asked for, because the extension searches ``max(ef_search, LIMIT)`` deep and a
    depth under the limit would be silently raised anyway.
    """
    configured = max(1, int(plan.params.get("ef_search", DEFAULT_EF_SEARCH)))
    rows = 0
    if plan.generation is not None:
        value = plan.generation.stats.get("rows")
        rows = int(value) if isinstance(value, (int, float)) else 0
    if rows and rows < EF_REFERENCE_ROWS:
        scaled = int(configured * math.sqrt(rows / float(EF_REFERENCE_ROWS)))
        configured = max(1, min(configured, scaled))
    return max(configured, int(limit))


@dataclass(frozen=True, slots=True)
class ColdArm:
    """What the cold generation returned, and the two numbers that say whether to trust it.

    ``candidates`` is how many rows the approximate scan produced BEFORE the visibility join,
    and ``base_rows`` how many rows the generation's table holds now.  A healthy scan produces
    exactly ``min(limit, base_rows)`` of them: the candidate subquery carries no predicate, so
    nothing but the structure itself can make it short.  See :meth:`damage`.
    """

    hits: list[tuple[int, float]]
    candidates: int
    base_rows: int
    limit: int

    def damage(self, recorded_rows: int | None) -> str | None:
        """Why this generation's base cannot be trusted, or ``None``.

        Two invariants, both free (the numbers come out of the read's own statement):

        * the base holds what the build recorded.  A generation's table is frozen after its
          build; only ``_erase`` touches it, and that drops the HNSW index and invalidates the
          generation, so a usable generation whose row count has moved has been changed by
          something anatid did not do.
        * the approximate scan returned as many candidates as there are rows to return.  This
          is what catches a structure that is present, queryable and wrong: a persisted HNSW
          graph that lost its edges answers with a fraction of the neighbours and reports no
          error (DuckDB documents HNSW persistence as experimental, with WAL and crash-recovery
          caveats, which is why this backend is opt in).

        Neither is a full validation, which is :meth:`VectorIndex._validate`'s job and costs a
        scan of the base.  They are the two checks a read can afford, and between them a
        damaged base becomes a reported fallback rather than a short answer.
        """
        if recorded_rows is not None and int(recorded_rows) != self.base_rows:
            return (
                f"the base holds {self.base_rows} row(s) against the {int(recorded_rows)} its "
                f"build recorded"
            )
        expected = min(self.limit, self.base_rows)
        if self.candidates < expected:
            return (
                f"the approximate scan returned {self.candidates} candidate(s) where "
                f"{expected} were available; the HNSW structure is damaged"
            )
        return None


def _cold_arm(
    con,
    plan: VectorPlan,
    *,
    tenant_id: int,
    literal: str,
    dim: int,
    topn: int,
    kinds: Sequence[str] | None,
) -> ColdArm:
    """HNSW candidates from the cold generation, filtered and exactly scored.

    Two things make this fast enough to be worth having.  The candidate subquery projects the
    cosine SIMILARITY alongside the distance it orders by, so the score comes out of the same
    scan that chose the row and is bit-identical to the oracle's; and the join back to
    ``memories`` touches only the narrow visibility columns, never the ``FLOAT[N]`` column.
    Adding ``m.embedding IS NOT NULL`` to that join would read the embedding column of every
    touched row group and cost 9 ms instead of 2 ms at 95,000 rows.  It would also be
    redundant: a row is in the base only because it had an embedding, and losing one is a new
    version, which is a journal entry.

    The statement carries one extra row, marked by a third column, holding the candidate count
    and the base's row count.  They are what :meth:`ColdArm.damage` reads, and they are in the
    same statement so a read pays one aggregate over an already materialised CTE and one
    metadata count rather than a second round trip.
    """
    over = max(1, int(plan.params.get("overfetch", DEFAULT_OVERFETCH)))
    limit = max(int(topn) * over, int(topn))
    _set_ef_search(con, search_depth(plan, limit))
    # The tenant and the limits are rendered rather than bound, for the reason
    # :func:`anatid.recall.recall_2hop_ids` renders them: a bound parameter costs about 0.35 ms
    # per statement in duckdb-python 1.5.5, which is a sixth of this query.  Both go through
    # int() first, and the tenant through Visibility, so nothing but a number is rendered.
    w, wp = Visibility.at(int(tenant_id), CURRENT).predicate("m", inline_tenant=True)
    kf, kp = _kinds("m", kinds)
    storage = quote_ident(str(plan.storage))
    tomb = ""
    if plan.tombstones:
        tomb = f" AND c.memory_id NOT IN ({_ids(plan.tombstones)})"
    sql = (
        f"WITH cand AS MATERIALIZED ("
        f"SELECT memory_id, array_cosine_similarity(embedding, ?::FLOAT[{int(dim)}]) AS score "
        f"FROM {storage} "
        f"ORDER BY array_cosine_distance(embedding, ?::FLOAT[{int(dim)}]) LIMIT {int(limit)}), "
        f"hits AS ("
        f"SELECT c.memory_id AS memory_id, c.score AS score FROM cand c "
        f"JOIN memories m ON m.memory_id = c.memory_id "
        f"WHERE {w}{kf}{tomb} "
        f"ORDER BY c.score DESC, c.memory_id ASC LIMIT {int(topn)}) "
        f"SELECT memory_id, score, FALSE AS census FROM hits "
        f"UNION ALL SELECT (SELECT count(*) FROM cand), "
        f"(SELECT count(*) FROM {storage}), TRUE"
    )
    params = [literal, literal] + wp + kp
    hits: list[tuple[int, float]] = []
    candidates = base_rows = 0
    for row in con.execute(sql, params).fetchall():
        if row[2]:
            candidates, base_rows = int(row[0]), int(row[1])
        else:
            hits.append((int(row[0]), float(row[1])))
    return ColdArm(hits=hits, candidates=candidates, base_rows=base_rows, limit=limit)


def _hot_arm(
    con,
    plan: VectorPlan,
    *,
    tenant_id: int,
    literal: str,
    dim: int,
    topn: int,
    kinds: Sequence[str] | None,
) -> list[tuple[int, float]]:
    """Exact cosine over the documents the base generation cannot know about.

    The journal's pending documents plus anything visible past the watermark, bounded by
    :func:`merge_cap`.  This is what makes a write visible to the very next read, and it is
    exact, so a pending document is scored the same way a cold one is.
    """
    if not plan.pending:
        return []
    w, wp = Visibility.at(int(tenant_id), CURRENT).predicate("m", inline_tenant=True)
    kf, kp = _kinds("m", kinds)
    sql = (
        f"SELECT m.memory_id, array_cosine_similarity(m.embedding, ?::FLOAT[{int(dim)}]) AS score "
        f"FROM memories m WHERE m.memory_id IN ({_ids(plan.pending)}) AND {w} "
        f"AND m.embedding IS NOT NULL{kf} ORDER BY score DESC, m.memory_id ASC "
        f"LIMIT {int(topn)}"
    )
    params = [literal] + wp + kp
    return [(int(r[0]), float(r[1])) for r in con.execute(sql, params).fetchall()]


def search(
    con,
    *,
    tenant_id: int,
    embedding: Sequence[float],
    dim: int,
    topn: int = 50,
    as_of: AsOf = CURRENT,
    kinds: Sequence[str] | None = None,
    index: "VectorIndex | None" = None,
    plan: VectorPlan | None = None,
) -> VectorSearch:
    """The vector arm: cosine top-N for one tenant, from whichever backend can answer it.

    The result always carries the same scores the exact scan would have produced for the rows
    it returns, because the merged path scores with the same expression over the same vectors.
    What an approximate structure changes is which rows were considered, and that is reported
    rather than hidden: :attr:`VectorSearch.backend` and :attr:`VectorSearch.reason`.

    A failure inside the approximate path is caught and answered exactly.  DuckDB does not
    abort a transaction on a failed statement, so a caller that wrapped this in
    ``db.transaction()`` keeps its transaction.
    """
    if len(embedding) != int(dim):
        raise EmbeddingDimensionError(
            f"embedding has {len(embedding)} dimensions, database is FLOAT[{dim}]",
            expected=int(dim),
            got=len(embedding),
        )
    literal = _literal(embedding)
    p = plan if plan is not None else resolve(con, tenant_id=tenant_id, as_of=as_of, index=index)
    if p.usable and not AsOf.coerce(as_of).is_current:
        # A caller that resolved the plan itself and then asked for a historical read.  The
        # generation cannot answer it whatever the plan says, and silently answering with
        # current-state candidates is the one mistake this module must not make.
        p = _exact_plan(
            HealthReason.HISTORICAL_QUERY,
            "a current-state index cannot answer an as_of read; the exact scan does",
        )
    if not p.usable:
        hits = exact_search(
            con,
            tenant_id=tenant_id,
            embedding=embedding,
            dim=dim,
            topn=topn,
            as_of=as_of,
            kinds=kinds,
        )
        return VectorSearch(
            hits=hits,
            backend="exact",
            reason=p.reason,
            detail=p.detail,
            generation=None if p.generation is None else p.generation.generation,
        )
    try:
        cold = _cold_arm(
            con, p, tenant_id=tenant_id, literal=literal, dim=dim, topn=topn, kinds=kinds
        )
        hot = _hot_arm(
            con, p, tenant_id=tenant_id, literal=literal, dim=dim, topn=topn, kinds=kinds
        )
    except (duckdb.Error, ValueError) as exc:
        number = p.generation.generation if p.generation else "?"
        detail = f"generation {number} could not be searched: {exc}"
        log.warning("vector index: %s; answering exactly", detail)
        hits = exact_search(
            con,
            tenant_id=tenant_id,
            embedding=embedding,
            dim=dim,
            topn=topn,
            as_of=as_of,
            kinds=kinds,
        )
        return VectorSearch(
            hits=hits,
            backend="exact",
            reason=HealthReason.LOAD_FAILURE,
            detail=detail,
            generation=None if p.generation is None else p.generation.generation,
        )
    recorded = None
    if p.generation is not None:
        value = p.generation.stats.get("rows")
        recorded = int(value) if isinstance(value, (int, float)) else None
    damage = cold.damage(recorded)
    if damage is not None:
        number = p.generation.generation if p.generation else "?"
        detail = f"generation {number}: {damage}; answering exactly"
        log.warning("vector index: %s", detail)
        return VectorSearch(
            hits=exact_search(
                con,
                tenant_id=tenant_id,
                embedding=embedding,
                dim=dim,
                topn=topn,
                as_of=as_of,
                kinds=kinds,
            ),
            backend="exact",
            reason=HealthReason.DAMAGED_BASE,
            detail=detail,
            generation=None if p.generation is None else p.generation.generation,
        )
    merged: dict[int, float] = dict(cold.hits)
    merged.update(dict(hot))  # the canonical row wins
    hits = sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))[: int(topn)]
    return VectorSearch(
        hits=[(int(m), float(s)) for m, s in hits],
        backend="duckdb_vss",
        reason=p.reason,
        detail=p.detail,
        generation=None if p.generation is None else p.generation.generation,
        scanned=len(p.pending),
        candidates=len(cold.hits) + len(hot),
    )


def scan_rows(
    con,
    *,
    tenant_id: int,
    as_of: AsOf = CURRENT,
    kinds: Sequence[str] | None = None,
    index: "VectorIndex | None" = None,
    plan: VectorPlan | None = None,
) -> int:
    """How many rows :func:`search` will scan exactly for this request.

    This is the number :data:`anatid.recall.BRUTE_FORCE_CEILING` is compared against, and it is
    the whole point of having a backend: with a usable generation the exact scan covers the
    journal's pending documents only, so a tenant past the ceiling can be served instead of
    refused.  With no usable generation it is the tenant's visible embeddings, exactly as in
    0.1.1.
    """
    p = plan if plan is not None else resolve(con, tenant_id=tenant_id, as_of=as_of, index=index)
    if p.usable and AsOf.coerce(as_of).is_current:
        return len(p.pending)
    w, wp = Visibility.at(int(tenant_id), as_of).predicate("m")
    kf, kp = _kinds("m", kinds)
    row = con.execute(
        f"SELECT count(*) FROM memories m WHERE {w} AND m.embedding IS NOT NULL{kf}", wp + kp
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


# --------------------------------------------------------------------------- the index


class VectorIndex(DerivedIndex):
    """An HNSW accelerator for the vector arm, on the derived-index framework.

    One generation per tenant, because dense vector storage per tenant is also the isolation
    story: a tenant's embeddings never enter another tenant's structure, so a bug in the
    approximate search cannot leak across the boundary the way a shared document key did in
    schema v2.

    A generation's storage is one table, ``anatid_idx_vector_t<tenant>_g<n>``, holding
    ``(memory_id, embedding)`` for the tenant's CURRENTLY VISIBLE rows at or below the
    generation watermark, with an HNSW index on the embedding.  Everything else the read needs
    is in the shared journal.

    The class attributes the framework reads are the defaults for a memory-derived index:
    source ``memories`` keyed by ``memory_id``, ``delta_mode="table"`` (exact for any id,
    including an explicit one below the watermark) and ``supports_delta=True`` (the read merges,
    so pending rows cost recall time, not correctness).
    """

    name = VECTOR_INDEX_NAME
    kind = "duckdb_vss"
    source_table = "memories"
    source_id_column = "memory_id"
    source_ts_column = "tx_from"
    per_tenant = True
    delta_mode = "table"
    supports_delta = True

    def __init__(
        self,
        db: Any,
        *,
        backend: str = "duckdb_vss",
        dim: int | None = None,
        metric: str = DEFAULT_METRIC,
        m: int = DEFAULT_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
        overfetch: int = DEFAULT_OVERFETCH,
        validate_probes: int = DEFAULT_VALIDATE_PROBES,
        validate_recall: float = DEFAULT_VALIDATE_RECALL,
        name: str | None = None,
    ) -> None:
        if str(backend) == "owned_hnsw":
            raise NotImplementedError(
                "the 'owned_hnsw' vector backend is not implemented. anatid's own graph index "
                "is a later phase; use 'duckdb_vss' for the DuckDB HNSW index or 'exact' for "
                "the brute-force oracle"
            )
        if str(backend) != "duckdb_vss":
            raise ValueError(
                f"VectorIndex is the {'duckdb_vss'!r} backend; got backend={backend!r}. "
                f"The 'exact' backend registers no index (it is the default read path)"
            )
        if str(metric) not in ("cosine", "l2sq", "ip"):
            raise ValueError(f"metric must be one of 'cosine', 'l2sq', 'ip', got {metric!r}")
        super().__init__(db, name=name)
        self.backend = "duckdb_vss"
        self.metric = str(metric)
        self.m = int(m)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.overfetch = max(1, int(overfetch))
        self.validate_probes = max(0, int(validate_probes))
        self.validate_recall = float(validate_recall)
        self.dim = int(dim) if dim is not None else self._discover_dim()
        error = load_vss(self._con())
        if error:
            self.load_error = error
            log.warning("vector index %r: %s; reads will use the exact scan", self.name, error)
        _bump_definitions()

    # ------------------------------------------------------------------ plumbing

    def _con(self):
        """The connection to run storage statements on.

        ``self.db`` is an :class:`~anatid.Anatid` handle for everything the framework does, and
        that is what ``self.db.execute`` uses.  A few statements (``LOAD``, ``SET``) are
        connection state rather than data, so they need the connection itself.
        """
        con = getattr(self.db, "connection", None)
        return con if con is not None else self.db

    def _discover_dim(self) -> int:
        cfg = getattr(self.db, "config", None)
        if cfg is not None and getattr(cfg, "embedding_dim", None):
            return int(cfg.embedding_dim)
        recorded = _recorded_dim(self._con())
        if recorded is None:
            raise IndexGenerationError(
                "the vector index needs the embedding dimension and this database records "
                "none; pass dim= or open the database with a schema",
                index=self.name,
            )
        return int(recorded)

    @property
    def params(self) -> dict[str, Any]:
        """What goes into the persisted definition, and what :func:`resolve` reads back."""
        return {
            "backend": self.backend,
            "dim": self.dim,
            "metric": self.metric,
            "M": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "overfetch": self.overfetch,
        }

    def definition(self) -> IndexDefinition:
        return IndexDefinition(
            index_name=self.name,
            kind=self.kind,
            per_tenant=True,
            source_table=self.source_table,
            source_id_column=self.source_id_column,
            delta_mode=self.delta_mode,
            supports_delta=True,
            params=self.params,
        )

    def storage(self, generation: Generation) -> str:
        """The table one generation's vectors live in."""
        return _storage_name(generation)

    def _table_exists(self, name: str) -> bool:
        row = self.db.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name = ?", [str(name)]
        ).fetchone()
        return bool(row and row[0])

    # ------------------------------------------------------------------ storage hooks

    def build_next(self, tenant: Any = None, *, now: _dt.datetime | None = None) -> Generation:
        """Build the next generation: the frozen copy in the transaction, the graph after it.

        The framework runs :meth:`_build` inside the build transaction, which is what makes the
        copy, the watermark and the absorbed journal rows one snapshot.  ``CREATE INDEX ...
        USING HNSW`` inside a transaction is DEFERRED to the commit by DuckDB and costs about
        five times what it costs outside one: measured on 95,000 64-dimensional vectors here,
        47 s of commit against 9.5 s.  So the copy happens in the transaction and the graph is
        built over it immediately afterwards.

        Nothing is lost by the split.  The base table is frozen: nothing but a hard erasure
        writes to it, the generation is not published until it has been validated, and a
        failure while indexing retires the generation instead of leaving a published one with
        no graph.
        """
        generation = super().build_next(tenant, now=now)
        return self._index_generation(generation)

    def _index_generation(self, generation: Generation) -> Generation:
        """Build the HNSW structure over an already-committed generation table."""
        con = self._con()
        _enable_persistence(con)
        storage = quote_ident(self.storage(generation))
        hnsw = quote_ident(_hnsw_name(generation))
        started = time.perf_counter()
        try:
            self.db.execute(
                f"CREATE INDEX {hnsw} ON {storage} USING HNSW (embedding) "
                f"WITH (metric = '{self.metric}', ef_construction = {self.ef_construction}, "
                f"M = {self.m})"
            )
        except BaseException:
            with contextlib.suppress(Exception):
                self.retire(generation)
            raise
        seconds = time.perf_counter() - started
        stats = dict(generation.stats)
        stats["index_seconds"] = round(seconds, 4)
        self.db.execute(
            f"UPDATE {INDEX_GENERATIONS_TABLE} SET stats = ? WHERE index_name = ? "
            f"AND tenant_id IS NOT DISTINCT FROM ? AND generation = ?",
            [
                json.dumps(stats, default=str),
                self.name,
                generation.tenant_id,
                generation.generation,
            ],
        )
        log.info(
            "vector index: generation %d for tenant %s indexed %s row(s) in %.2fs",
            generation.generation,
            generation.tenant_id,
            stats.get("rows"),
            seconds,
        )
        return replace(generation, stats=stats)

    def _build(self, generation: Generation) -> dict[str, Any] | None:
        """Copy the tenant's cold, visible embeddings into this generation's own table.

        The copy is taken with the visibility predicate applied, so the base holds one row per
        live memory rather than one per version, and with ``memory_id <= watermark_id`` so the
        boundary between "in the base" and "in the journal" is the generation watermark the
        framework recorded in the same transaction.  The graph over it is built by
        :meth:`build_next` once this transaction has committed.
        """
        if self.load_error:
            raise IndexGenerationError(
                f"index {self.name!r}: {self.load_error}",
                index=self.name,
                generation=generation.generation,
            )
        key = generation.tenant_id
        if key is None:  # pragma: no cover - per_tenant is True
            raise IndexGenerationError(
                f"index {self.name!r}: a vector generation is per tenant", index=self.name
            )
        storage = quote_ident(self.storage(generation))
        started = time.perf_counter()
        self.db.execute(f"DROP TABLE IF EXISTS {storage}")
        w, wp = Visibility.at(int(key), CURRENT).predicate("m")
        sql = (
            f"CREATE TABLE {storage} AS SELECT m.memory_id, m.embedding FROM memories m "
            f"WHERE {w} AND m.embedding IS NOT NULL"
        )
        params = list(wp)
        if generation.watermark_id is not None:
            sql += " AND m.memory_id <= ?"
            params.append(int(generation.watermark_id))
        self.db.execute(sql, params)
        copied = time.perf_counter()
        row = self.db.execute(f"SELECT count(*) FROM {storage}").fetchone()
        rows = int(row[0]) if row and row[0] is not None else 0
        return {
            "rows": rows,
            "backend": self.backend,
            "dim": self.dim,
            "metric": self.metric,
            "M": self.m,
            "ef_construction": self.ef_construction,
            "storage": self.storage(generation),
            "hnsw_index": _hnsw_name(generation),
            "copy_seconds": round(copied - started, 4),
        }

    def _validate(self, generation: Generation) -> ValidationReport:
        """Compare the generation with the oracle on membership, fidelity and recall.

        Membership and fidelity are exact set comparisons against ``memories``: the base must
        hold exactly the tenant's visible cold documents, with byte-identical embeddings.  They
        are what lets the read take a candidate's score straight out of the base scan instead of
        going back to the ``FLOAT[N]`` column, which is the difference between a 2 ms and an
        11 ms query.

        Recall is measured against the base itself: probes are averages of pairs of stored
        vectors (so they are inside the distribution but are not rows), the oracle is an exact
        scan of the same table ordered by similarity (which the HNSW optimiser does not rewrite,
        because it only matches distance ascending), and the structure must find at least
        ``validate_recall`` of the true top ``VALIDATE_TOPN``.  An approximate index that has
        become a bad graph is a real failure mode, and a row-count check would not see it.
        """
        key = generation.tenant_id
        storage_name = self.storage(generation)
        if not self._table_exists(storage_name):
            return ValidationReport(
                ok=False,
                generation=generation,
                detail=f"storage table {storage_name!r} does not exist",
            )
        storage = quote_ident(storage_name)
        w, wp = Visibility.at(int(key or 0), CURRENT).predicate("m")
        cold = " AND m.memory_id <= ?" if generation.watermark_id is not None else ""
        cold_params = [int(generation.watermark_id)] if generation.watermark_id is not None else []
        oracle = (
            f"SELECT m.memory_id, m.embedding FROM memories m "
            f"WHERE {w} AND m.embedding IS NOT NULL{cold}"
        )
        params = list(wp) + cold_params
        row = self.db.execute(
            f"SELECT "
            f"(SELECT count(*) FROM ({oracle})), "
            f"(SELECT count(*) FROM {storage}), "
            f"(SELECT count(*) FROM ({oracle}) o "
            f" LEFT JOIN {storage} b ON b.memory_id = o.memory_id WHERE b.memory_id IS NULL), "
            f"(SELECT count(*) FROM {storage} b "
            f" LEFT JOIN ({oracle}) o ON o.memory_id = b.memory_id WHERE o.memory_id IS NULL), "
            f"(SELECT count(*) FROM {storage} b JOIN ({oracle}) o ON o.memory_id = b.memory_id "
            f" WHERE b.embedding IS DISTINCT FROM o.embedding)",
            params * 4,
        ).fetchone()
        expected, held, missing, extra, differing = (int(x) for x in row)
        if missing or extra or differing:
            return ValidationReport(
                ok=False,
                generation=generation,
                checked=expected,
                mismatches=(("missing", missing), ("extra", extra), ("differing", differing)),
                detail=(
                    f"the base holds {held} row(s) against {expected} in the oracle: "
                    f"{missing} missing, {extra} not visible, {differing} with a different "
                    f"embedding"
                ),
            )
        recall, probes = self._probe_recall(storage, held)
        if probes and recall < self.validate_recall:
            return ValidationReport(
                ok=False,
                generation=generation,
                checked=expected,
                mismatches=(("recall", recall),),
                detail=(
                    f"the HNSW structure found {recall:.3f} of the true top {VALIDATE_TOPN} "
                    f"over {probes} probe(s), below validate_recall={self.validate_recall}"
                ),
            )
        return ValidationReport(
            ok=True,
            generation=generation,
            checked=expected,
            detail=(
                f"{expected} row(s) match the oracle exactly; recall@{VALIDATE_TOPN} "
                f"{recall:.3f} over {probes} probe(s)"
            ),
        )

    def _probe_recall(self, storage: str, rows: int) -> tuple[float, int]:
        """Measured recall of the structure against an exact scan of the same table."""
        wanted = min(self.validate_probes, max(0, rows // 2))
        if not wanted or rows < VALIDATE_TOPN:
            return 1.0, 0
        con = self._con()
        sample = self.db.execute(
            f"SELECT embedding FROM {storage} USING SAMPLE {int(wanted) * 2} ROWS (reservoir, 7)"
        ).fetchall()
        vectors = [list(r[0]) for r in sample if r[0] is not None]
        probes = []
        for i in range(0, len(vectors) - 1, 2):
            a, b = vectors[i], vectors[i + 1]
            probes.append([(x + y) / 2.0 for x, y in zip(a, b, strict=False)])
        if not probes:
            return 1.0, 0
        _set_ef_search(con, max(self.ef_search, VALIDATE_TOPN * self.overfetch))
        total = 0.0
        for probe in probes:
            lit = _literal(probe)
            truth = {
                int(r[0])
                for r in self.db.execute(
                    f"SELECT memory_id FROM {storage} "
                    f"ORDER BY array_cosine_similarity(embedding, ?::FLOAT[{self.dim}]) DESC, "
                    f"memory_id ASC LIMIT {VALIDATE_TOPN}",
                    [lit],
                ).fetchall()
            }
            got = {
                int(r[0])
                for r in self.db.execute(
                    f"SELECT memory_id FROM {storage} "
                    f"ORDER BY array_cosine_distance(embedding, ?::FLOAT[{self.dim}]) "
                    f"LIMIT {VALIDATE_TOPN * self.overfetch}",
                    [lit],
                ).fetchall()
            }
            total += len(truth & got) / float(len(truth) or 1)
        return total / len(probes), len(probes)

    def _drop(self, generation: Generation) -> None:
        """Remove a generation's table, and with it the HNSW structure over it."""
        storage = self.storage(generation)
        with contextlib.suppress(Exception):
            self.db.execute(f"DROP INDEX IF EXISTS {quote_ident(_hnsw_name(generation))}")
        self.db.execute(f"DROP TABLE IF EXISTS {quote_ident(storage)}")

    def _erase(self, generation: Generation, tenant_id: int, doc_ids: Sequence[int]) -> int | None:
        """Delete purged documents from a generation, and take it out of service if it held any.

        A ``DELETE`` removes the row from the table, but DuckDB's HNSW keeps the vector inside
        the graph it built: the entry is marked deleted, not overwritten.  ``forget(hard=True)``
        promises the content leaves the file, so the index is dropped as well, and the
        generation then cannot serve approximate search.  Returning ``None`` is how the
        framework is told that: it invalidates the generation, the reads fall back to the exact
        scan (which is right, just slower), and the next
        :func:`~anatid.derived.maintain` builds a clean one.

        A generation that did not hold the document returns 0 and is left alone, so an ordinary
        purge does not cost a rebuild.
        """
        ids = [int(d) for d in doc_ids]
        if not ids:
            return 0
        if generation.tenant_id is not None and int(generation.tenant_id) != int(tenant_id):
            return 0
        storage_name = self.storage(generation)
        if not self._table_exists(storage_name):
            return 0
        storage = quote_ident(storage_name)
        row = self.db.execute(f"DELETE FROM {storage} WHERE memory_id IN ({_ids(ids)})").fetchone()
        deleted = int(row[0]) if row and row[0] is not None else 0
        if not deleted:
            return 0
        self.db.execute(f"DROP INDEX IF EXISTS {quote_ident(_hnsw_name(generation))}")
        log.info(
            "vector index: erased %d document(s) from generation %d and dropped its HNSW "
            "structure, which holds a copy of the vector",
            deleted,
            generation.generation,
        )
        return None

    # ------------------------------------------------------------------ reads

    def base_damage(self, generation: Generation) -> str | None:
        """Whether the generation's table still holds what its build recorded.

        The read path also checks that the HNSW scan returns as many candidates as there are
        rows (:meth:`ColdArm.damage`), which needs a query vector and so cannot be checked
        here; this is the half a health report can answer.
        """
        value = generation.stats.get("rows")
        recorded = int(value) if isinstance(value, (int, float)) else None
        if recorded is None:
            return None
        name = self.storage(generation)
        try:
            row = self._con().execute(f"SELECT count(*) FROM {quote_ident(name)}").fetchone()
        except duckdb.Error as exc:
            return f"the base at {name} could not be counted: {exc}"
        held = int(row[0]) if row and row[0] is not None else 0
        if held != recorded:
            return f"the base holds {held} row(s) against the {recorded} its build recorded"
        return None

    def plan(self, tenant: Any = None, *, as_of: AsOf | None = None) -> VectorPlan:
        """How a read for ``tenant`` would be answered right now.  See :func:`resolve`."""
        return resolve(
            self._con(),
            tenant_id=self.tenant_key(tenant) or 0,
            as_of=AsOf.coerce(as_of),
            index=self,
        )

    def search(
        self,
        tenant: Any,
        embedding: Sequence[float],
        *,
        topn: int = 50,
        as_of: AsOf | None = None,
        kinds: Sequence[str] | None = None,
    ) -> VectorSearch:
        """Run the vector arm for ``tenant`` through this index."""
        return search(
            self._con(),
            tenant_id=_tenant_of(tenant),
            embedding=embedding,
            dim=self.dim,
            topn=topn,
            as_of=AsOf.coerce(as_of),
            kinds=kinds,
            index=self,
        )


# --------------------------------------------------------------------------- wiring


def attach(db: Any, *, backend: str = "duckdb_vss", **params: Any) -> VectorIndex | None:
    """Give ``db`` a vector accelerator and record it in the FILE, not just on this handle.

    ``backend="exact"`` registers nothing and returns ``None``: the exact scan is the default
    read path and needs no index.  ``backend="owned_hnsw"`` raises
    :class:`NotImplementedError`.  ``backend="duckdb_vss"`` builds a :class:`VectorIndex`,
    registers it (which persists the definition, so every handle on the file journals for it),
    and returns it.  Nothing is built yet: call
    :meth:`~anatid.Anatid.maintain_indexes` or :meth:`VectorIndex.build_next` for that.

    This is what ``Anatid.open(vector_backend=...)`` should call.  Wiring it into
    :meth:`anatid.Anatid.open` is one branch after ``ensure_schema``; until that lands, call
    this directly.
    """
    name = str(backend)
    if name == "exact":
        return None
    if name == "owned_hnsw":
        raise NotImplementedError(
            "the 'owned_hnsw' vector backend is not implemented. anatid's own graph index is a "
            "later phase; use 'duckdb_vss' or the default 'exact'"
        )
    if name != "duckdb_vss":
        raise ValueError(f"vector backend must be one of {BACKENDS}, got {backend!r}")
    index = VectorIndex(db, backend="duckdb_vss", **params)
    db.indexes.register(index)
    _bump_definitions()
    return index


def detach(db: Any, *, retire: bool = True) -> None:
    """Take the vector accelerator off ``db`` and stop the file journalling for it.

    ``retire=True`` (the default) also drops every generation's storage, which is what an index
    holding a copy of the embeddings needs: left behind, those tables are a copy of everything
    the index ever saw, and once no handle holds the implementation nothing can delete one
    document from them.  It needs this handle to hold the implementation, and
    :meth:`~anatid.derived.IndexRegistry.unregister` raises when it does not rather than
    reporting a removal that did not happen.
    """
    registry = getattr(db, "indexes", None)
    if registry is None:
        return
    registry.unregister(VECTOR_INDEX_NAME, retire=retire)
    _bump_definitions()


def index_of(db: Any) -> VectorIndex | None:
    """This handle's :class:`VectorIndex`, or ``None`` when it holds none."""
    registry = getattr(db, "indexes", None)
    if registry is None:
        return None
    found = registry.get(VECTOR_INDEX_NAME)
    return found if isinstance(found, VectorIndex) else None
