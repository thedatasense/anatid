"""Graph expansion: a derived CSR index, an optional C++ expander, and a pure-SQL oracle.

The k-hop frontier of a seed entity over same-tenant ``RELATES_TO`` edges, computed three ways
that return the **same entity set**.  Only the cost and the staleness story differ.

``"sql"`` (always available, always correct)
    Two explicit self-join levels over the ``relates_undirected`` view, UNION ALL'ed and *not*
    deduplicated -- the frontier is consumed through an ``IN (...)`` semi join where duplicates
    are harmless, and each DISTINCT costs a HASH_GROUP_BY.  This is the formulation the spike
    A/B-tested against a recursive CTE with ``USING KEY`` and found faster.  Hops > 2 fall back
    to the recursive CTE, which handles any depth.  It reads the canonical tables, so it is the
    oracle: every other path is checked against it.

``"csr"`` (a published generation of :class:`CsrIndex`, merged in SQL)
    One statement over the generation's materialised adjacency plus the change journal:
    ``base - tombstoned + pending``, expanded level by level.  No extension needed.

``"extension"`` (the same generation, expanded in C++)
    ``graph_expand(tenant, seed, hops, key := ..., add_src := ..., drop_src := ...)`` from the
    optional ``anatid`` DuckDB extension: a BFS over the generation's in-memory CSR with the
    journal applied inside the traversal.

The index, not the snapshot
---------------------------
In 0.1 the CSR was a process-wide snapshot that anatid abandoned after any RELATES_TO write, so
a single ``relate()`` cost the extension for the rest of the session.  It is now a derived index
in the sense of ``docs/design/derived-index-framework.md`` (:mod:`anatid.derived`):

* a **base generation** is one snapshot of the tenant's current edges, materialised as two
  tables (a dense vertex mapping and an edge list) and, when the extension is loaded, an
  in-memory CSR named after that generation;
* the **journal** is what the verbs wrote since, inside their own transactions, so a write is
  visible to the next read;
* a read **pins** one generation, merges base + pending - tombstoned level by level, and only
  then applies :class:`anatid.visibility.Visibility` to the memories and ABOUT edges it selects.
  Merging by levels rather than expanding the base and filtering afterwards is what makes it
  exact: retiring an edge changes reachability, not just membership.

Dense vertex ids are an internal detail of a generation.  The extension indexes adjacency by
``vertex - min_vertex``, so it needs ids dense per tenant, and anatid's 63-bit time-ordered
entity ids are anything but: two entities minted 10 ms apart are ~42,000,000 apart, which used
to make the extension unusable on a real database.  A generation therefore owns an explicit
``entity_id <-> vertex_id`` mapping, numbered 0..n-1 over the entities its snapshot has an edge
for, and hands it to the extension as ``labels``.  Seeds and results are external entity ids on
every path; nothing outside this module sees a vertex id.

Why a read may not use it
-------------------------
Never silently.  :meth:`CsrIndex.expand` reports one of
:class:`anatid.derived.HealthReason` every time, and :attr:`CsrBackend.active` (what
``Anatid.expand_path`` returns) is a string that carries it:

``historical_query``   an ``as_of`` read; a current-state index cannot answer it
``absent``             no generation has been published for this tenant yet
``rebuild_in_progress``   none is published and one is being built
``stale_generation``   the published one was invalidated (a bulk load, or an erasure that could
                       not reach its storage), or the policy says it is due for a rebuild
``load_failure``       the extension or the generation's storage could not be loaded
``unvalidated``        published with ``force``; usable, and never mistaken for validated
``fresh``              a validated generation, within policy

What it costs
-------------
Measured on the Phase 0 spike ``small`` dataset (100k memories, 27.8k RELATES_TO rows, 10
tenants; tenant 0 is 2,471 current edges over 998 entities), duckdb 1.5.5 on macOS arm64, p50 of
600 2-hop expansions from 200 benchmark seeds.  ``ext/`` and this module were A/B'd on the one
question the design doc leaves open, whether the journal should be applied outside the C++
expander or inside it:

=========================================  ========  =========  ==========
2-hop frontier                             no delta  ~50 rows   ~550 rows
=========================================  ========  =========  ==========
pure SQL over the canonical tables          0.84 ms   0.87 ms    0.88 ms
the generation, journal merged in SQL       0.95 ms   1.64 ms    2.27 ms
the generation, journal merged in C++       0.41 ms   1.06 ms    1.50 ms
=========================================  ========  =========  ==========

The C++ merge wins at every journal size, so ``strategy="auto"`` takes it whenever the extension
is loaded and the generation's snapshot can be built.  End to end, ``recall_2hop_ids`` is 1.80 ms
through the index against 2.05 ms through pure SQL with an empty journal.  Building, validating
and publishing all ten tenants' generations costs 279 ms.

Two things those numbers say out loud.  A journal of a few hundred rows costs more than the whole
expansion saves, which is what :class:`~anatid.derived.MaintenancePolicy` exists to prevent:
``rebuild_after_ratio = 0.05`` fires at 124 rows on this graph, well before the crossover.  And
the floor is the 0.09 ms the 0.1 snapshot took: the framework's 0.3 ms over that is one query for
the journal plus the pin, and it is what buys an answer that stays right after a write.

Turning it on
-------------
The framework half needs no extension::

    from anatid.csr import attach_csr_index
    index = attach_csr_index(db)          # registers it in the file
    db.maintain_indexes()                 # builds, validates and publishes a generation

The C++ half needs ``allow_unsigned_extensions`` in the **connect config**, which can only be set
when the database is opened: ``Anatid.open(..., use_csr_extension=True)`` does that and it cannot
be turned on afterwards.  Failure to load is not an error; the index stays on its SQL merge, and
the SQL merge is checked against the oracle by the same test that checks the extension.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .derived import INSERT, Generation, HealthReason, ValidationReport
from .derived import DerivedIndex as _DerivedIndex
from .errors import ExtensionUnavailable
from .schema import INDEX_JOURNAL_TABLE, quote_ident
from .types import AsOf, CURRENT
from .visibility import LIVE_ROW_SQL, Visibility

log = logging.getLogger("anatid.csr")

__all__ = [
    "CsrInfo",
    "CsrBackend",
    "CsrIndex",
    "CsrDelta",
    "CsrExpansion",
    "ExpandPath",
    "attach_csr_index",
    "discover_extension_path",
    "extension_unsupported",
    "frontier_sql",
]

#: Environment variable that overrides extension discovery.
EXTENSION_PATH_ENV = "ANATID_EXTENSION_PATH"

_REL_NAME = "anatid.duckdb_extension"


#: Where a built extension is looked for, relative to a repository root, in priority order.
_BUILD_TREES: tuple[tuple[str, ...], ...] = (
    # the productionized extension: `cd ext && GEN=ninja make release`
    ("ext", "build", "release", "extension", "anatid"),
    # the Phase 0 spike tree, kept as a fallback so an old checkout still works
    ("spike", "extension", "build", "release", "extension", "anatid"),
)


def discover_extension_path() -> Path | None:
    """Locate the compiled ``anatid`` DuckDB extension, or return None.

    Search order:

    1. ``$ANATID_EXTENSION_PATH`` (if set, it is the answer -- no fallback, so a typo is visible)
    2. ``<package>/_ext/anatid.duckdb_extension`` (a wheel that ships the binary)
    3. ``<repo>/ext/build/release/extension/anatid/`` -- what ``cd ext && GEN=ninja make release``
       produces
    4. ``<repo>/spike/extension/build/release/extension/anatid/`` -- the Phase 0 spike build

    The walk goes all the way up from this file, *not* stopping at ``src``: in an editable
    install the package sits at ``<repo>/src/anatid/``, so stopping there put both build trees
    permanently out of reach and discovery silently returned ``None`` for every checkout of this
    repository.  Returning ``None`` is not an error -- the caller falls back to the pure-SQL
    expansion, which returns identical results -- but it did mean the documented
    ``use_csr_extension=True`` enable path was a silent no-op.
    """
    env = os.environ.get(EXTENSION_PATH_ENV)
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "_ext" / _REL_NAME,
    ]
    # walk up looking for a repository root holding ext/ or spike/ (source / editable install)
    for parent in here.parents:
        for tree in _BUILD_TREES:
            candidates.append(parent.joinpath(*tree) / _REL_NAME)
    for c in candidates:
        if c.is_file():
            return c
    return None


# --------------------------------------------------------------------------- the path a read took


class ExpandPath(str):
    """Which expansion path a read took, and WHY, in one value.

    It **is** the string ``"sql"``, ``"csr"`` or ``"extension"``, so
    ``db.expand_path == "sql"`` and ``json.dumps(db.stats())`` behave exactly as before.  It also
    carries the :class:`~anatid.derived.HealthReason` and a sentence: reporting only that the CSR
    was not used tells an operator nothing actionable, which is the whole complaint the derived
    index design opens with.
    """

    __slots__ = ("reason", "detail", "generation", "tenant_id")

    def __new__(
        cls,
        value: str,
        *,
        reason: HealthReason = HealthReason.ABSENT,
        detail: str = "",
        generation: int | None = None,
        tenant_id: int | None = None,
    ) -> "ExpandPath":
        self = super().__new__(cls, value)
        self.reason = reason
        self.detail = detail
        self.generation = generation
        self.tenant_id = tenant_id
        return self

    def explain(self) -> str:
        """``"sql (historical_query): a current-state index cannot answer an as_of read"``."""
        head = f"{self!s} ({self.reason.value})"
        return f"{head}: {self.detail}" if self.detail else head

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self),
            "reason": self.reason.value,
            "detail": self.detail,
            "generation": self.generation,
            "tenant_id": self.tenant_id,
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"ExpandPath({str(self)!r}, reason={self.reason.value!r})"


@dataclass(frozen=True, slots=True)
class CsrDelta:
    """The change journal of one generation, resolved into undirected entity-id pairs.

    ``added`` are pairs a currently live edge connects that the base snapshot does not have.
    ``removed`` are pairs the base has and **every** one of whose base edges has been closed or
    purged since: a pair with two edges, one of them retired, is still connected and is not here.
    ``pending`` and ``tombstones`` are the journal row counts behind them, for reporting.
    """

    added: tuple[tuple[int, int], ...] = ()
    removed: tuple[tuple[int, int], ...] = ()
    pending: int = 0
    tombstones: int = 0

    @property
    def empty(self) -> bool:
        return not self.added and not self.removed


@dataclass(frozen=True, slots=True)
class CsrExpansion:
    """What :meth:`CsrIndex.expand` produced.

    ``entity_ids`` is ``None`` when no generation could serve the read; the caller falls back to
    the SQL oracle and ``path`` says why.  Otherwise it is the deduplicated frontier including
    the seed, and ``path`` is ``"csr"`` or ``"extension"``.
    """

    entity_ids: tuple[int, ...] | None
    path: ExpandPath
    #: The journal as pairs, when the expansion had to resolve it (the C++ expander needs it;
    #: the SQL merge reads the journal inline and never builds it).  ``None`` means not resolved,
    #: which is not the same as empty.
    delta: CsrDelta | None = None
    generation: Generation | None = None

    def __bool__(self) -> bool:
        return self.entity_ids is not None


@dataclass(frozen=True)
class CsrInfo:
    """Result of one ``anatid_build_csr()`` call."""

    tenants: int
    vertices: int
    edges: int
    build_ms: float
    wall_ms: float
    edge_table: str
    built_at: float


# --------------------------------------------------------------------------- the backend


#: The edge filter a build has to apply for its answers to match the SQL path.  Schema v4 made
#: this reachable: a correction closes the OLD version row on the transaction axis and leaves its
#: ``valid_to`` alone, so a binary filtering on ``valid_to IS NULL`` alone traverses a belief the
#: database has replaced and returns rows the SQL path does not. The Phase 0 spike binary
#: (``anatid 0.0.0-spike``) does exactly that.  From :mod:`anatid.visibility`, like every other
#: predicate over the bitemporal columns.
REQUIRED_EDGE_FILTER = LIVE_ROW_SQL


def extension_unsupported(path: str | os.PathLike) -> str | None:
    """Why the extension binary at ``path`` cannot be used, or ``None`` when it can.

    Loads it on a throwaway in-memory connection and asks it what edge filter it applies, which
    is the one property a binary older than the schema it is reading can get wrong silently.
    Useful before pointing a handle at a binary of unknown vintage, and used by the test suite
    to skip rather than fail on the Phase 0 spike build.
    """
    import duckdb

    try:
        con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    except Exception as exc:  # noqa: BLE001 - report, do not raise, from a probe
        return f"{type(exc).__name__}: {exc}"
    try:
        con.execute(f"LOAD '{Path(path).as_posix()}'")
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    else:
        return _unsupported_binary(con)
    finally:
        with contextlib.suppress(Exception):
            con.close()


def _unsupported_binary(con) -> str | None:
    """Why the loaded extension must not be used, or ``None``.

    Called once per connection, straight after ``LOAD``. A binary too old to report its own edge
    filter is refused rather than trusted: the alternative is a graph expansion that silently
    disagrees with the oracle, which is the one thing an accelerator may not do.
    """
    try:
        columns = [
            d[0] for d in con.execute("SELECT * FROM anatid_csr_stats() LIMIT 0").description
        ]
    except Exception as exc:  # noqa: BLE001 - a missing function is the answer, not a failure
        return (
            f"this build of the anatid extension does not report its edge filter "
            f"(anatid_csr_stats: {exc}); it predates the transaction-time predicate and would "
            f"traverse edges a correction closed. Rebuild it from ext/."
        )
    if "current_filter" not in columns:
        return (
            "this build of the anatid extension does not report current_filter, so its edge "
            "filter cannot be checked against schema v4. Rebuild it from ext/."
        )
    return None


class CsrBackend:
    """Chooses and drives the graph-expansion path for one :class:`anatid.Anatid` handle.

    Construct it with ``enabled=True`` only if the database connection was opened with
    ``allow_unsigned_extensions``; otherwise ``LOAD`` fails and the backend stays on ``"sql"``.

    Two modes.  With no :class:`CsrIndex` attached it is the 0.1 backend: one unnamed in-memory
    snapshot, marked stale by every RELATES_TO write, used only while fresh.  With one attached
    (:func:`attach_csr_index`) it delegates to the index, which pins a generation and merges the
    journal, and :attr:`active` reports that decision.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        extension_path: str | os.PathLike | None = None,
        require: bool = False,
        edge_table: str = "edges_relates",
    ) -> None:
        self.enabled = bool(enabled)
        self.require = bool(require)
        self.edge_table = edge_table
        self._path: Path | None = None
        if self.enabled or self.require:
            p = Path(extension_path) if extension_path is not None else discover_extension_path()
            if p is not None and p.is_file():
                self._path = p
            elif self.require:
                raise ExtensionUnavailable(
                    f"no anatid DuckDB extension found (set ${EXTENSION_PATH_ENV} to its path)")
        self._loaded = False
        self._info: CsrInfo | None = None
        self._stale = True
        self.load_error: str | None = None
        #: The derived index this backend drives, when one has been attached.
        self.index: "CsrIndex | None" = None
        #: The tenant :attr:`active` reports on.  Set by :func:`attach_csr_index`.
        self.default_tenant: int | None = None
        self._last: ExpandPath | None = None

    # ---------------------------------------------------------------- state

    @property
    def extension_path(self) -> Path | None:
        """Path of the extension binary anatid would load, or None."""
        return self._path

    @property
    def loaded(self) -> bool:
        """True when the extension is loaded on this handle's connection."""
        return self._loaded

    @property
    def available(self) -> bool:
        """True when the extension is loaded on the connection and a CSR has been built."""
        return self._loaded and self._info is not None

    @property
    def fresh(self) -> bool:
        """True when the unnamed 0.1 snapshot still reflects ``edges_relates``, as far as anatid
        knows.  Meaningless once a :class:`CsrIndex` is attached: staleness is then a property of
        a generation and its journal, not of this flag."""
        return self.available and not self._stale

    @property
    def active(self) -> ExpandPath:
        """Which expansion path a current-state read would take right now, and why.

        With an index attached this pins the published generation for
        :attr:`default_tenant` and reports the decision without running the read; a read for
        another tenant makes its own.  Never raises: a closed handle or an unreadable catalog
        reports the last decision this backend saw, or ``sql`` with the failure as its detail.
        """
        idx = self.index
        if idx is None:
            return self.legacy_path()
        try:
            chosen = idx.explain(self.default_tenant)
            if str(chosen) == "sql" and self.fresh:
                # The index has nothing to offer yet and the 0.1 snapshot is current: that is
                # the path frontier_sql would take, so it is the path to report.
                return self.legacy_path()
            return chosen
        except Exception as exc:  # noqa: BLE001 - describe() must never be the thing that raises
            if self._last is not None:
                return self._last
            return ExpandPath(
                "sql", reason=HealthReason.LOAD_FAILURE, detail=f"{type(exc).__name__}: {exc}")

    def legacy_path(self) -> ExpandPath:
        """The 0.1 decision: the unnamed snapshot while it is fresh, otherwise SQL."""
        if self.fresh:
            return ExpandPath(
                "extension", reason=HealthReason.FRESH,
                detail="the in-memory snapshot is current as far as this handle knows")
        if self.load_error:
            return ExpandPath("sql", reason=HealthReason.LOAD_FAILURE, detail=self.load_error)
        if not self._loaded:
            return ExpandPath(
                "sql", reason=HealthReason.ABSENT,
                detail="the csr extension is not loaded on this connection")
        if self._info is None:
            return ExpandPath(
                "sql", reason=HealthReason.ABSENT,
                detail="no CSR snapshot has been built; call build_csr()")
        return ExpandPath(
            "sql", reason=HealthReason.STALE_GENERATION,
            detail="a RELATES_TO row was written since the snapshot was built")

    def note_expansion(self, path: ExpandPath) -> ExpandPath:
        """Record the decision one expansion made.  Called by :class:`CsrIndex`."""
        self._last = path
        return path

    @property
    def last_expansion(self) -> ExpandPath | None:
        """The decision the most recent expansion made, or None."""
        return self._last

    @property
    def info(self) -> CsrInfo | None:
        """Stats from the last :meth:`build`."""
        return self._info

    def describe(self) -> dict:
        """A dict for logs / health endpoints."""
        path = self.active
        return {
            "active": str(path),
            "reason": path.reason.value,
            "detail": path.detail,
            "generation": path.generation,
            "enabled": self.enabled,
            "loaded": self._loaded,
            "fresh": self.fresh,
            "indexed": self.index is not None,
            "extension_path": str(self._path) if self._path else None,
            "load_error": self.load_error,
            "csr": None if self._info is None else {
                "tenants": self._info.tenants, "vertices": self._info.vertices,
                "edges": self._info.edges, "build_ms": self._info.build_ms},
        }

    # ---------------------------------------------------------------- lifecycle

    def load(self, con) -> bool:
        """``LOAD`` the extension on ``con``.  Returns True on success, False (and stays on the
        SQL path) on any failure -- unless the backend was constructed with ``require=True``."""
        if self._loaded:
            return True
        if not self.enabled or self._path is None:
            if self.require:
                raise ExtensionUnavailable("csr extension not enabled for this connection")
            return False
        try:
            con.execute(f"LOAD '{self._path.as_posix()}'")
        except Exception as exc:  # duckdb.IOException / InvalidInputException / ...
            self.load_error = f"{type(exc).__name__}: {exc}"
            if self.require:
                raise ExtensionUnavailable(
                    f"could not load {self._path}: {self.load_error}") from exc
            return False
        unsupported = _unsupported_binary(con)
        if unsupported is not None:
            self.load_error = unsupported
            if self.require:
                raise ExtensionUnavailable(f"{self._path}: {unsupported}")
            log.warning("csr extension not used: %s", unsupported)
            return False
        self._loaded = True
        return True

    def build(self, con, edge_table: str | None = None) -> CsrInfo | None:
        """(Re)build the unnamed in-memory CSR snapshot.  Returns None on the SQL path.

        The 0.1 entry point, kept because it is public API and because a database whose entity
        ids are already dense can still use it directly.  With a :class:`CsrIndex` attached this
        is not what recall reads: the index builds one named snapshot per generation
        (:meth:`CsrIndex.build_next`), and this unnamed one is left alone.
        """
        table = edge_table or self.edge_table
        if not self.load(con):
            return None
        t0 = time.perf_counter()
        # anatid_build_csr reads its argument at BIND time, so it must be a literal, not a bound
        # parameter. quote_ident validates the name is a bare identifier before it is formatted in.
        quote_ident(table)
        row = con.execute(f"SELECT * FROM anatid_build_csr('{table}')").fetchone()
        applied = con.execute("SELECT current_filter FROM anatid_csr_stats()").fetchone()
        if applied is None or REQUIRED_EDGE_FILTER not in str(applied[0]):
            # The build succeeded and its answers would not match the SQL path.  Refuse the
            # snapshot rather than serving from it; reads stay on SQL and say why.
            self._info = None
            self._stale = True
            self.load_error = (
                f"the extension built a CSR filtered on {applied and applied[0]!r}, which does "
                f"not include {REQUIRED_EDGE_FILTER}: it would traverse edges a correction "
                f"closed. Rebuild the extension from ext/."
            )
            if self.require:
                raise ExtensionUnavailable(self.load_error)
            log.warning("csr snapshot refused: %s", self.load_error)
            return None
        self._info = CsrInfo(
            tenants=int(row[0]), vertices=int(row[1]), edges=int(row[2]),
            build_ms=float(row[3]), wall_ms=(time.perf_counter() - t0) * 1000.0,
            edge_table=table, built_at=time.time())
        self._stale = False
        return self._info

    def note_edge_write(self) -> None:
        """Mark the unnamed 0.1 snapshot stale (called by every RELATES_TO write).

        The index does not need it: a write reaches the journal in its own transaction, so the
        next read merges it whatever this flag says.
        """
        self._stale = True


# --------------------------------------------------------------------------- the derived index

_SNAPSHOT_LOCK = threading.Lock()
#: ``(database key, generation storage name) -> how many times its storage was erased from``.
#: A hard erasure deletes rows from a generation's edge table; the in-memory CSR built from it
#: is process memory that knows nothing about that, so it is rebuilt on the next read.  Keyed
#: process-wide because a purge on one handle must reach the snapshot another handle built.
_SNAPSHOT_EPOCH: dict[tuple[Any, str], int] = {}


def _snapshot_epoch(key: tuple[Any, str]) -> int:
    with _SNAPSHOT_LOCK:
        return _SNAPSHOT_EPOCH.get(key, 0)


def _bump_snapshot(key: tuple[Any, str]) -> None:
    with _SNAPSHOT_LOCK:
        _SNAPSHOT_EPOCH[key] = _SNAPSHOT_EPOCH.get(key, 0) + 1


def _ids(values: Iterable[int]) -> str:
    """``1, 2, 3`` for an ``IN`` list.  Every element goes through ``int()`` first."""
    return ", ".join(str(int(v)) for v in values)


def _pairs(values: Iterable[tuple[int, int]]) -> str:
    """``VALUES (1::BIGINT, 2::BIGINT), (3, 4)`` for a literal two-column relation."""
    rows = [(int(a), int(b)) for a, b in values]
    first = f"({rows[0][0]}::BIGINT, {rows[0][1]}::BIGINT)"
    rest = "".join(f", ({a}, {b})" for a, b in rows[1:])
    return f"VALUES {first}{rest}"


def _id_array(values: Iterable[int]) -> str:
    """``[1,2,3]::BIGINT[]``.  Every element goes through ``int()`` first.

    Inlining rather than binding for the same reason :func:`_num` does it: ``graph_expand`` reads
    its arguments at BIND time, so they cannot be bound parameters at all.  Only integers ever
    take this path.
    """
    return "[" + ",".join(str(int(v)) for v in values) + "]::BIGINT[]"


class CsrIndex(_DerivedIndex):
    """The CSR adjacency structure as a derived index (:mod:`anatid.derived`).

    One generation per tenant, because the dense vertex numbering is tenant-local.  Its storage
    is two tables:

    ``<generation>_vertices(tenant_id, vertex_id, entity_id)``
        The generation's mapping.  ``vertex_id`` runs 0..n-1 over the entities the snapshot has a
        current edge for, ordered by entity id.  This is what makes the C++ expander usable with
        anatid's sparse 63-bit ids, and it is owned by the generation: a rebuild renumbers, and
        two generations of one tenant have unrelated numberings, which is exactly why a read pins
        one generation and stays on it.

    ``<generation>_edges(edge_id, tenant_id, src, dst, a, b)``
        One row per current edge in the snapshot.  ``src``/``dst`` are dense vertex ids (what the
        extension builds from); ``a``/``b`` are the external entity ids, normalised so ``a <= b``
        (what the SQL merge traverses and what resolves a tombstoned ``edge_id`` back to the pair
        it used to connect).  ``edge_id`` is kept so a hard erasure can delete one document.

    Reads never touch either table directly: :meth:`expand` pins a generation, resolves the
    journal into :class:`CsrDelta`, and expands base + added - removed level by level.
    """

    name = "csr"
    kind = "csr"
    source_table = "edges_relates"
    source_id_column = "edge_id"
    source_ts_column = "tx_from"
    #: Per tenant: the dense numbering is tenant-local, and a busy tenant should not force a
    #: rebuild of a quiet one.
    per_tenant = True
    #: Every RELATES_TO write is journalled, whichever handle made it.  Exact for any edge id,
    #: including one supplied explicitly below the watermark.
    delta_mode = "table"
    #: The read side merges: this is the whole point of the rewrite.
    supports_delta = True

    #: Strategies :meth:`expand` may pick.  ``"auto"`` uses the extension when it is loaded and
    #: a snapshot of the pinned generation can be built, and the SQL merge otherwise.
    STRATEGIES = ("auto", "extension", "sql")

    def __init__(
        self,
        db: Any,
        *,
        backend: CsrBackend | None = None,
        name: str | None = None,
        strategy: str = "auto",
    ) -> None:
        super().__init__(db, name=name)
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}, got {strategy!r}")
        self.strategy = strategy
        self.backend = backend
        if backend is not None:
            backend.index = self
        #: ``storage name -> the erasure epoch the in-memory snapshot was built at``.  Per
        #: instance, because the extension's object cache belongs to one DuckDB database
        #: instance, which is one handle.
        self._snapshots: dict[str, int] = {}
        self._snapshot_lock = threading.RLock()

    # ------------------------------------------------------------------ storage names

    def _vertices_table(self, generation: Generation) -> str:
        return f"{generation.storage_name()}_vertices"

    def _edges_table(self, generation: Generation) -> str:
        return f"{generation.storage_name()}_edges"

    def storage_tables(self, generation: Generation) -> tuple[str, str]:
        """``(vertices table, edges table)`` of one generation.  Both bare identifiers."""
        return self._vertices_table(generation), self._edges_table(generation)

    def _snapshot_key(self, generation: Generation) -> str:
        """The name the extension holds this generation's in-memory CSR under."""
        return generation.storage_name()

    def _epoch_slot(self, generation: Generation) -> tuple[Any, str]:
        return (self._epoch_key, self._snapshot_key(generation))

    # ------------------------------------------------------------------ building

    def _build(self, generation: Generation) -> dict[str, Any] | None:
        """Materialise the generation: the dense mapping, then the edge list against it.

        Runs inside ``build_next``'s transaction, so the two tables and the ``absorbed_by`` stamp
        on the journal rows commit together with the snapshot they describe.  The in-memory CSR
        is NOT built here: it is rebuilt lazily by the first read that wants it, because the
        extension reads through a nested connection that cannot see this transaction.
        """
        tenant_id = int(generation.tenant_id) if generation.tenant_id is not None else None
        if tenant_id is None:  # pragma: no cover - per_tenant is True
            raise ValueError("the csr index builds one generation per tenant")
        vertices, edges = self.storage_tables(generation)
        v_tbl, e_tbl = quote_ident(vertices), quote_ident(edges)
        vis = Visibility(tenant_id)
        where, params = vis.predicate("r")
        self.db.execute(f"DROP TABLE IF EXISTS {e_tbl}")
        self.db.execute(f"DROP TABLE IF EXISTS {v_tbl}")
        # row_number() over the distinct endpoints: dense 0..n-1, stable under a rebuild of the
        # same edge set, and independent of the sparse entity ids themselves.
        self.db.execute(
            f"CREATE TABLE {v_tbl} AS SELECT ?::INTEGER AS tenant_id, "
            f"(row_number() OVER (ORDER BY entity_id) - 1)::BIGINT AS vertex_id, entity_id FROM ("
            f"  SELECT r.src AS entity_id FROM edges_relates r WHERE {where}"
            f"  UNION SELECT r.dst AS entity_id FROM edges_relates r WHERE {where})",
            [tenant_id] + list(params) + list(params),
        )
        self.db.execute(
            f"CREATE TABLE {e_tbl} AS SELECT r.edge_id, ?::INTEGER AS tenant_id, "
            f"vs.vertex_id AS src, vd.vertex_id AS dst, "
            f"least(r.src, r.dst) AS a, greatest(r.src, r.dst) AS b "
            f"FROM edges_relates r "
            f"JOIN {v_tbl} vs ON vs.entity_id = r.src "
            f"JOIN {v_tbl} vd ON vd.entity_id = r.dst "
            f"WHERE {where}",
            [tenant_id] + list(params),
        )
        n_edges = self._count(e_tbl)
        n_vertices = self._count(v_tbl)
        # A fresh generation has a fresh name, so no snapshot can be holding its storage.
        with self._snapshot_lock:
            self._snapshots.pop(self._snapshot_key(generation), None)
        return {
            "rows": n_edges,
            "edges": n_edges,
            "vertices": n_vertices,
            "vertex_table": vertices,
            "edge_table": edges,
        }

    def _count(self, quoted_table: str) -> int:
        row = self.db.execute(f"SELECT count(*) FROM {quoted_table}").fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def base_damage(self, generation: Generation) -> str | None:
        """Whether the generation's edge list has grown past what its build recorded.

        Only ``>`` is damage.  ``forget(hard=True)`` deletes edges from a generation's storage
        (and any vertex left with none), so a base SMALLER than the recorded count is the
        expected state after an erasure; a bigger one means something outside anatid wrote to
        the table, and the dense vertex ids in it no longer mean what the mapping says.
        """
        value = generation.stats.get("edges")
        recorded = int(value) if isinstance(value, (int, float)) else None
        if recorded is None:
            return None
        _vertices, edges = self.storage_tables(generation)
        try:
            held = self._count(quote_ident(edges))
        except Exception as exc:  # noqa: BLE001 - a missing or unreadable table is damage too
            return f"the edge list at {edges} could not be counted: {exc}"
        if held > recorded:
            return f"the edge list holds {held} row(s) against the {recorded} its build recorded"
        return None

    def _drop(self, generation: Generation) -> None:
        """Remove the generation's two tables and free its in-memory snapshot."""
        vertices, edges = self.storage_tables(generation)
        for table in (edges, vertices):
            with contextlib.suppress(Exception):
                self.db.execute(f"DROP TABLE IF EXISTS {quote_ident(table)}")
        self._forget_snapshot(generation)

    def _forget_snapshot(self, generation: Generation) -> None:
        key = self._snapshot_key(generation)
        with self._snapshot_lock:
            self._snapshots.pop(key, None)
        if self.backend is not None and self.backend.loaded:
            with contextlib.suppress(Exception):
                self.db.execute(f"SELECT * FROM anatid_drop_csr('{key}')").fetchall()

    # ------------------------------------------------------------------ erasure

    def _erase(self, generation: Generation, tenant_id: int, doc_ids: Sequence[int]) -> int | None:
        """Delete these edges from the generation's storage.  Runs inside the purge transaction.

        A vertex left with no edge at all goes too: after ``forget(hard=True)`` on the last edge
        of an entity, nothing anatid built may still name it.  The in-memory CSR is process
        memory built from these tables, so it is marked for rebuild rather than patched; the
        rebuild reads the committed tables, after this transaction lands.
        """
        ids = [int(d) for d in doc_ids]
        if not ids:
            return 0
        if generation.tenant_id is not None and int(generation.tenant_id) != int(tenant_id):
            return 0
        vertices, edges = self.storage_tables(generation)
        v_tbl, e_tbl = quote_ident(vertices), quote_ident(edges)
        try:
            row = self.db.execute(
                f"DELETE FROM {e_tbl} WHERE edge_id = ANY(?::BIGINT[])", [ids]
            ).fetchone()
        except Exception:  # noqa: BLE001 - storage a build never got to, or already dropped
            if self._storage_missing(edges):
                return 0
            raise
        removed = int(row[0]) if row and row[0] is not None else 0
        orphan = self.db.execute(
            f"DELETE FROM {v_tbl} WHERE vertex_id NOT IN ("
            f"  SELECT src FROM {e_tbl} UNION SELECT dst FROM {e_tbl})"
        ).fetchone()
        removed += int(orphan[0]) if orphan and orphan[0] is not None else 0
        _bump_snapshot(self._epoch_slot(generation))
        return removed

    def _storage_missing(self, table: str) -> bool:
        row = self.db.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name = ?", [table]
        ).fetchone()
        return not (row and row[0])

    # ------------------------------------------------------------------ validation

    def _validate(self, generation: Generation) -> ValidationReport:
        """Compare the MERGE with the oracle, not the base with the oracle.

        What the index promises a read is ``base - tombstoned + pending``, and that is what has
        to equal the current-state adjacency in ``edges_relates``.  Comparing the base alone
        would fail every time a write landed between the build and the check, which is most of
        the time, and would not test the thing that can actually be wrong.  Both sides are read
        in one transaction so a concurrent write cannot fake a mismatch.
        """
        tenant_id = int(generation.tenant_id) if generation.tenant_id is not None else None
        if tenant_id is None:  # pragma: no cover - per_tenant is True
            return ValidationReport(False, generation, detail="no tenant scope")
        vertices, edges = self.storage_tables(generation)
        v_tbl, e_tbl = quote_ident(vertices), quote_ident(edges)
        latest, lparams = self._latest_sql(tenant_id, generation)
        vis = Visibility(tenant_id)
        where, wparams = vis.predicate("r")
        with self.db.transaction():
            mismatches: list[tuple] = []
            rows = self.db.execute(
                f"WITH j AS ({latest}), "
                f"tomb AS (SELECT doc_id FROM j WHERE op <> 'insert'), "
                f"pend AS (SELECT doc_id FROM j WHERE op = 'insert'), "
                f"base AS (SELECT e.a, e.b FROM {e_tbl} e "
                f"         WHERE e.edge_id NOT IN (SELECT doc_id FROM tomb)), "
                f"added AS (SELECT least(r.src, r.dst) AS a, greatest(r.src, r.dst) AS b "
                f"          FROM edges_relates r JOIN pend ON pend.doc_id = r.edge_id "
                f"          WHERE {where}), "
                f"merged AS (SELECT a, b FROM base UNION SELECT a, b FROM added), "
                f"oracle AS (SELECT DISTINCT least(r.src, r.dst) AS a, "
                f"           greatest(r.src, r.dst) AS b FROM edges_relates r WHERE {where}) "
                f"SELECT 'missing' AS side, a, b FROM (SELECT * FROM oracle EXCEPT "
                f"                                     SELECT * FROM merged) "
                f"UNION ALL "
                f"SELECT 'extra', a, b FROM (SELECT * FROM merged EXCEPT SELECT * FROM oracle) "
                f"LIMIT 20",
                list(lparams) + list(wparams) + list(wparams),
            ).fetchall()
            mismatches.extend((str(r[0]), int(r[1]), int(r[2])) for r in rows)
            checked = self._count_expr(
                f"SELECT DISTINCT least(r.src, r.dst), greatest(r.src, r.dst) "
                f"FROM edges_relates r WHERE {where}",
                list(wparams),
            )
            mismatches.extend(self._validate_mapping(v_tbl, e_tbl))
        ok = not mismatches
        detail = (
            f"{checked} current pair(s) agree with the merged generation"
            if ok
            else f"{len(mismatches)} disagreement(s): {mismatches[:5]}"
        )
        return ValidationReport(
            ok=ok,
            generation=generation,
            checked=checked,
            mismatches=tuple(mismatches),
            detail=detail,
        )

    def _count_expr(self, sql: str, params: list) -> int:
        row = self.db.execute(f"SELECT count(*) FROM ({sql})", params).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def _validate_mapping(self, v_tbl: str, e_tbl: str) -> list[tuple]:
        """The dense mapping is a bijection over 0..n-1 covering every edge endpoint.

        A duplicate or a gap would not show up as a wrong answer on the SQL path (which never
        reads it) and would quietly mean a different entity on the extension path, which is the
        one failure mode a dense numbering has.
        """
        out: list[tuple] = []
        row = self.db.execute(
            f"SELECT count(*), count(DISTINCT vertex_id), count(DISTINCT entity_id), "
            f"min(vertex_id), max(vertex_id) FROM {v_tbl}"
        ).fetchone()
        n = int(row[0]) if row and row[0] is not None else 0
        if n:
            if int(row[1]) != n or int(row[2]) != n:
                out.append(("mapping", "not one to one", n))
            if int(row[3]) != 0 or int(row[4]) != n - 1:
                out.append(("mapping", "not dense 0..n-1", (int(row[3]), int(row[4]), n)))
        gap = self.db.execute(
            f"SELECT count(*) FROM {e_tbl} e WHERE e.src NOT IN (SELECT vertex_id FROM {v_tbl}) "
            f"OR e.dst NOT IN (SELECT vertex_id FROM {v_tbl})"
        ).fetchone()
        if gap and gap[0]:
            out.append(("mapping", "edge endpoint with no vertex row", int(gap[0])))
        crossed = self.db.execute(
            f"SELECT count(*) FROM {e_tbl} e "
            f"JOIN {v_tbl} vs ON vs.vertex_id = e.src JOIN {v_tbl} vd ON vd.vertex_id = e.dst "
            f"WHERE least(vs.entity_id, vd.entity_id) IS DISTINCT FROM e.a "
            f"OR greatest(vs.entity_id, vd.entity_id) IS DISTINCT FROM e.b"
        ).fetchone()
        if crossed and crossed[0]:
            out.append(("mapping", "dense pair disagrees with the entity pair", int(crossed[0])))
        return out

    # ------------------------------------------------------------------ the journal, as pairs

    def _journal_sql(self, tenant_id: int, generation: Generation) -> str:
        """The newest journal op per edge for one generation, with every value inlined.

        Same rule as :func:`anatid.derived.journal_latest_sql` -- the largest ``change_seq`` for a
        document wins, over the rows this generation has not absorbed -- rendered with literals
        instead of bind parameters.  ``index_name`` went through
        :func:`anatid.schema.quote_ident` when the index was constructed, so it is a bare
        identifier and cannot terminate the quotes; the other two are integers.  It is worth
        doing: duckdb-python 1.5.5 spends about 0.25 ms per statement marshalling parameters, and
        this query costs 0.10 ms without them against 0.35 ms with them, on a read whose whole
        budget is under a millisecond.
        """
        t_sql, _t_params = Visibility(int(tenant_id)).tenant(inline=True)
        return (
            f"SELECT doc_id, op FROM {INDEX_JOURNAL_TABLE} "
            f"WHERE index_name = '{self.name}' AND {t_sql} "
            f"AND (absorbed_by IS NULL OR absorbed_by > {int(generation.generation)}) "
            f"QUALIFY row_number() OVER (PARTITION BY doc_id ORDER BY change_seq DESC) = 1"
        )

    def delta(self, generation: Generation) -> CsrDelta:
        """Resolve this generation's journal into undirected entity pairs to add and to skip.

        The journal itself is one small query; the two that turn ids into pairs run only when
        there is something to turn.  ``added`` joins the pending edge ids back to
        ``edges_relates``, so an edge that was written and then closed contributes nothing.
        ``removed`` is the harder half: neither expander holds edge ids in its adjacency, so a
        tombstone has to become the PAIR it used to connect, and only when no other base edge
        still connects the same two entities.  Dropping a pair because one of two parallel edges
        was retired would remove an edge the SQL oracle still traverses.
        """
        tenant_id = int(generation.tenant_id) if generation.tenant_id is not None else None
        if tenant_id is None:  # pragma: no cover - per_tenant is True
            return CsrDelta()
        rows = self.db.execute(self._journal_sql(tenant_id, generation)).fetchall()
        if not rows:
            return CsrDelta()
        pending = [int(r[0]) for r in rows if r[1] == INSERT]
        tombstoned = [int(r[0]) for r in rows if r[1] != INSERT]
        added: tuple[tuple[int, int], ...] = ()
        removed: tuple[tuple[int, int], ...] = ()
        if pending:
            where, _p = Visibility(tenant_id).predicate("r", inline_tenant=True)
            found = self.db.execute(
                f"SELECT DISTINCT least(r.src, r.dst) AS a, greatest(r.src, r.dst) AS b "
                f"FROM edges_relates r WHERE {where} "
                f"AND r.edge_id IN ({_ids(pending)})"
            ).fetchall()
            added = tuple((int(a), int(b)) for a, b in found)
        if tombstoned:
            _v, edges = self.storage_tables(generation)
            e_tbl = quote_ident(edges)
            marks = _ids(tombstoned)
            found = self.db.execute(
                f"SELECT a, b FROM {e_tbl} WHERE edge_id IN ({marks}) "
                f"EXCEPT SELECT a, b FROM {e_tbl} WHERE edge_id NOT IN ({marks})"
            ).fetchall()
            removed = tuple((int(a), int(b)) for a, b in found)
        return CsrDelta(added, removed, len(pending), len(tombstoned))

    # ------------------------------------------------------------------ the in-memory snapshot

    def _extension_ready(self) -> bool:
        backend = self.backend
        if backend is None or not backend.enabled:
            return False
        if backend.loaded:
            return True
        with contextlib.suppress(Exception):
            return bool(backend.load(self.db.connection))
        return False

    def ensure_snapshot(self, generation: Generation) -> bool:
        """Make sure the extension holds this generation's CSR under its own name.

        Idempotent and lazy.  Built from the generation's two tables through the extension's
        nested connection, so it must run outside the transaction that created them; a failure
        (no extension, uncommitted storage, a budget refusal) is not an error, it just leaves
        :meth:`expand` on the SQL merge with a reason.
        """
        if not self._extension_ready():
            return False
        key = self._snapshot_key(generation)
        want = _snapshot_epoch(self._epoch_slot(generation))
        with self._snapshot_lock:
            if self._snapshots.get(key) == want:
                return True
        vertices, edges = self.storage_tables(generation)
        quote_ident(vertices)
        quote_ident(edges)
        quote_ident(key)
        try:
            self.db.execute(
                f"SELECT * FROM anatid_build_csr('{edges}', labels := '{vertices}', "
                f"key := '{key}')"
            ).fetchall()
        except Exception as exc:  # noqa: BLE001 - the SQL merge answers the same question
            # Not a permanent failure and not load_error: the storage may simply be
            # uncommitted, and the next read tries again.
            self._last_snapshot_error = f"{type(exc).__name__}: {exc}"
            return False
        with self._snapshot_lock:
            self._snapshots[key] = want
        self._last_snapshot_error = None
        return True

    _last_snapshot_error: str | None = None

    # ------------------------------------------------------------------ expansion

    def expand(
        self,
        tenant: Any,
        seed_entity_id: int,
        hops: int,
        *,
        as_of: AsOf | _dt.datetime | None = None,
        strategy: str | None = None,
    ) -> CsrExpansion:
        """The ``hops``-hop frontier of ``seed_entity_id``, or a reason why not.

        Pins one generation for the whole computation, so a publish that lands mid-read leaves
        this read on the generation it started with and a retire cannot drop the storage it is
        reading.  The frontier comes back as entity ids rather than as SQL over the generation's
        tables on purpose: the statement the caller then runs holds no reference to a generation
        that could be retired between building it and executing it.

        ``entity_ids`` includes the seed and is deduplicated.  ``hops = 0`` is the seed alone and
        needs no index.
        """
        seed = int(seed_entity_id)
        hops = int(hops)
        if hops < 0:
            raise ValueError("hops must be >= 0")
        scope = AsOf.coerce(as_of)
        want = strategy or self.strategy
        if want not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}, got {want!r}")
        with self.pin(tenant, as_of=scope) as pin:
            if not pin:
                return self._decline(pin.reason, pin.detail, pin.generation, tenant)
            generation = pin.generation
            assert generation is not None
            try:
                ids, used, note, delta = self.expand_generation(
                    generation, seed, hops, strategy=want
                )
            except Exception as exc:  # noqa: BLE001 - the oracle answers the same question
                return self._decline(
                    HealthReason.LOAD_FAILURE,
                    f"the generation's storage could not be read: "
                    f"{type(exc).__name__}: {exc}",
                    generation,
                    tenant,
                )
            path = ExpandPath(
                used,
                reason=pin.reason,
                detail=note or pin.detail,
                generation=generation.generation,
                tenant_id=generation.tenant_id,
            )
            if self.backend is not None:
                self.backend.note_expansion(path)
            return CsrExpansion(ids, path, delta, generation)

    def expand_generation(
        self,
        generation: Generation,
        seed_entity_id: int,
        hops: int,
        *,
        strategy: str | None = None,
        delta: CsrDelta | None = None,
    ) -> tuple[tuple[int, ...], str, str, CsrDelta | None]:
        """Merge one SPECIFIC generation with its journal.  ``(ids, path, note, delta)``.

        The pin is the caller's business here, which is what :meth:`expand` adds.  Any generation
        still in the catalog can be merged, not only the published one: a reader pinned to the
        previous generation goes on getting a right answer from it while the next one is
        published, because the journal covers exactly what that generation's base does not.
        """
        seed, hops = int(seed_entity_id), int(hops)
        want = strategy or self.strategy
        if want not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}, got {want!r}")
        note = ""
        # One resolution of the journal, used by whichever expander runs.  Both need the same
        # thing -- neither adjacency holds edge ids -- and reading it twice could see two
        # different journals.
        resolved = delta if delta is not None else self.delta(generation)
        if want in ("auto", "extension") and self.ensure_snapshot(generation):
            try:
                ids = self._expand_extension(generation, seed, hops, resolved)
                return tuple(ids), "extension", note, resolved
            except Exception as exc:  # noqa: BLE001 - fall back, never fail the read
                note = (
                    f"the extension declined this expansion "
                    f"({type(exc).__name__}: {exc}); merged in SQL instead"
                )
        elif want == "extension":
            note = self._last_snapshot_error or "the csr extension is not loaded on this handle"
        return tuple(self._expand_sql(generation, seed, hops, resolved)), "csr", note, resolved

    def _decline(
        self,
        reason: HealthReason,
        detail: str,
        generation: Generation | None,
        tenant: Any,
    ) -> CsrExpansion:
        path = ExpandPath(
            "sql",
            reason=reason,
            detail=detail,
            generation=None if generation is None else generation.generation,
            tenant_id=self._tenant_or_none(tenant),
        )
        if self.backend is not None:
            self.backend.note_expansion(path)
        return CsrExpansion(None, path, None, generation)

    def _tenant_or_none(self, tenant: Any) -> int | None:
        try:
            return self.tenant_key(tenant)
        except (TypeError, ValueError):
            return None

    def explain(self, tenant: Any = None) -> ExpandPath:
        """Which path a current-state read for ``tenant`` would take right now, and why.

        Costs one pin: no SQL at all while the catalog cache is warm.  It does not run the read,
        so it does not know whether the extension would decline this particular expansion.
        """
        if tenant is None:
            return ExpandPath(
                "sql",
                reason=HealthReason.ABSENT,
                detail="no tenant given; the csr index has one generation per tenant",
            )
        with self.pin(tenant) as pin:
            if not pin:
                return ExpandPath(
                    "sql",
                    reason=pin.reason,
                    detail=pin.detail,
                    generation=None if pin.generation is None else pin.generation.generation,
                    tenant_id=self._tenant_or_none(tenant),
                )
            generation = pin.generation
            assert generation is not None
            ready = self.strategy != "sql" and self._extension_ready()
            with self._snapshot_lock:
                resident = self._snapshots.get(self._snapshot_key(generation)) == _snapshot_epoch(
                    self._epoch_slot(generation)
                )
            return ExpandPath(
                "extension" if (ready and resident) else "csr",
                reason=pin.reason,
                detail=pin.detail,
                generation=generation.generation,
                tenant_id=generation.tenant_id,
            )

    # ------------------------------------------------------------------ the two expanders

    def _expand_extension(
        self, generation: Generation, seed: int, hops: int, delta: CsrDelta
    ) -> list[int]:
        """One ``graph_expand`` call: the BFS and the journal merge both happen in C++.

        Every argument is an integer or a validated identifier, which is what
        ``graph_expand`` requires: it reads its arguments at BIND time, so none of them can be a
        bound parameter.
        """
        key = self._snapshot_key(generation)
        quote_ident(key)
        tenant_id = int(generation.tenant_id or 0)
        args = [str(tenant_id), str(seed), str(hops), f"key := '{key}'"]
        args.append(f"max_hops := {max(int(hops), 30)}")
        if delta.added:
            args.append(f"add_src := {_id_array(a for a, _b in delta.added)}")
            args.append(f"add_dst := {_id_array(b for _a, b in delta.added)}")
        if delta.removed:
            args.append(f"drop_src := {_id_array(a for a, _b in delta.removed)}")
            args.append(f"drop_dst := {_id_array(b for _a, b in delta.removed)}")
        rows = self.db.execute(
            f"SELECT entity_id FROM graph_expand({', '.join(args)})"
        ).fetchall()
        return [int(r[0]) for r in rows]

    def _expand_sql(
        self, generation: Generation, seed: int, hops: int, delta: CsrDelta
    ) -> list[int]:
        """BFS by levels over base - removed + added, in one statement.

        Level k + 1 is the merged neighbours of level k.  Two explicit levels for the benchmarked
        1- and 2-hop shapes; a recursive CTE with ``USING KEY`` beyond that, correct at any depth
        and slower.  The delta arrives as pairs rather than as a join back to the journal so this
        is one scan of the generation's edge list and nothing else: at 2,471 edges it costs
        0.46 ms, against 0.77 ms for the same frontier over the canonical tables.
        """
        if hops <= 0:
            return [int(seed)]
        _vertices, edges = self.storage_tables(generation)
        e_tbl = quote_ident(edges)
        heads: list[str] = []
        keep = ""
        if delta.removed:
            heads.append(f"anatid_dead(a, b) AS ({_pairs(delta.removed)})")
            keep = (
                " WHERE NOT EXISTS (SELECT 1 FROM anatid_dead d "
                "WHERE d.a = e.a AND d.b = e.b)"
            )
        adj = (
            f"SELECT e.a AS x, e.b AS y FROM {e_tbl} e{keep} "
            f"UNION ALL SELECT e.b AS x, e.a AS y FROM {e_tbl} e{keep}"
        )
        if delta.added:
            both = tuple(delta.added) + tuple((b, a) for a, b in delta.added)
            heads.append(f"anatid_new(x, y) AS ({_pairs(both)})")
            adj = f"{adj} UNION ALL SELECT x, y FROM anatid_new"
        heads.append(f"anatid_adj AS ({adj})")
        prefix = "WITH " + ", ".join(heads)
        if hops <= 2:
            sql = (
                f"{prefix}, h1 AS (SELECT y AS entity_id FROM anatid_adj WHERE x = {int(seed)})"
            )
            if hops >= 2:
                sql += (
                    ", h2 AS (SELECT a.y AS entity_id FROM h1 "
                    "JOIN anatid_adj a ON a.x = h1.entity_id)"
                )
            sql += f" SELECT {int(seed)}::BIGINT AS entity_id UNION ALL SELECT entity_id FROM h1"
            if hops >= 2:
                sql += " UNION ALL SELECT entity_id FROM h2"
        else:
            sql = (
                f"WITH RECURSIVE {prefix[len('WITH '):]}, "
                f"bfs USING KEY (entity_id) AS ("
                f"  SELECT {int(seed)}::BIGINT AS entity_id, 0 AS depth"
                f"  UNION"
                f"  SELECT a.y AS entity_id, f.depth + 1 AS depth FROM bfs f "
                f"  JOIN anatid_adj a ON a.x = f.entity_id WHERE f.depth < {int(hops)})"
                f" SELECT entity_id FROM bfs"
            )
        rows = self.db.execute(sql).fetchall()
        return sorted({int(r[0]) for r in rows})



def attach_csr_index(
    db: Any,
    *,
    backend: CsrBackend | None = None,
    strategy: str = "auto",
    register: bool = True,
) -> CsrIndex:
    """Put the CSR on the derived-index framework for one handle.

    Constructs a :class:`CsrIndex`, links it to the handle's :class:`CsrBackend` so
    :func:`frontier_sql` and ``Anatid.expand_path`` go through it, and records the definition in
    the FILE (``register=False`` skips that, for a read-only handle or a dry run).  Recording it
    is what makes the journal complete: from then on every handle that opens the file journals
    RELATES_TO writes for this index whether or not it holds this class.

    Nothing is built yet.  ``db.maintain_indexes()`` (or ``index.build_next()`` /
    ``validate()`` / ``publish()``) produces the first generation; until then
    ``expand_path`` reports ``absent`` and reads take the SQL oracle.
    """
    csr_backend = backend if backend is not None else getattr(db, "csr", None)
    index = CsrIndex(db, backend=csr_backend, strategy=strategy)
    if csr_backend is not None:
        namespace = getattr(db, "namespace", None)
        if namespace is not None:
            csr_backend.default_tenant = int(namespace.tenant_id)
    if register:
        db.indexes.register(index)
    return index


# --------------------------------------------------------------------------- frontier SQL

def _num(value: int, inline: bool) -> tuple[str, list]:
    """Render an integer as either a bound parameter or a literal.

    Inlining is safe by construction -- the value has already been through ``int()``, and a
    Python int can only render as digits and a leading minus -- and it is measurably worth it on
    the hot path: duckdb-python 1.5.5 spends about 0.6 ms per call marshalling bound parameters
    (much of it in a repeated, failing ``import pandas``), which is half the cost of the 2-hop
    recall query itself at 100k memories.  Only integers ever take this path; strings, timestamps
    and embeddings are always bound.
    """
    return (str(int(value)), []) if inline else ("?", [int(value)])


def _frontier_values(entity_ids: Sequence[int]) -> str:
    """The frontier as a literal relation: ``SELECT unnest([...]::BIGINT[]) AS entity_id``.

    The ids have already been through ``int()`` (see :func:`_num`), and handing the consumer a
    literal rather than a query over the generation's tables is deliberate: the statement the
    caller executes next holds no reference to a generation, so retiring one between building
    the statement and running it cannot break the read.
    """
    return f"SELECT unnest({_id_array(entity_ids)}) AS entity_id"


def frontier_sql(
    tenant_id: int,
    seed_entity_id: int,
    hops: int,
    *,
    as_of: AsOf = CURRENT,
    backend: CsrBackend | None = None,
    inline_ints: bool = False,
) -> tuple[str, list, ExpandPath]:
    """Build the SQL that yields the seed's ``hops``-hop frontier as one ``entity_id`` column.

    Returns ``(sql, params, path)`` where ``sql`` is a complete SELECT usable as a subquery or CTE
    body, and ``path`` is an :class:`ExpandPath`: ``"extension"``, ``"csr"`` or ``"sql"``, and the
    reason.  Rows are NOT deduplicated on the SQL path: the frontier is consumed by a semi join.

    ``hops=0`` is the seed alone.  ``hops in (1, 2)`` uses the benchmarked two-self-join form.
    ``hops > 2`` uses a recursive CTE (correct at any depth, slower).

    When ``backend`` carries a :class:`CsrIndex` (:func:`attach_csr_index`) the index gets first
    refusal: it pins a generation, merges the journal, and this returns the frontier it computed
    as a literal.  When it declines -- a historical read, no published generation, a rebuild in
    flight, a stale or unloadable one -- the returned ``path`` says which, and the SQL below
    answers the query instead.  Every tenant and time predicate comes from
    :class:`anatid.visibility.Visibility`.

    ``inline_ints=True`` renders the (already ``int()``-coerced) tenant and seed as literals
    instead of bound parameters -- see :func:`_num` for why that is both safe and worth 0.6 ms.
    It only applies to the current-state forms; as-of forms always bind their timestamps.
    """
    tenant_id, seed_entity_id, hops = int(tenant_id), int(seed_entity_id), int(hops)
    if hops < 0:
        raise ValueError("hops must be >= 0")
    vis = Visibility.at(tenant_id, as_of)

    if hops == 0:
        seed_sql, seed_p = _num(seed_entity_id, inline_ints)
        return (
            f"SELECT {seed_sql}::BIGINT AS entity_id",
            seed_p,
            ExpandPath("sql", reason=HealthReason.FRESH, detail="hops = 0 is the seed alone"),
        )

    # --- the derived index: pins one generation, merges base + pending - tombstoned, and hands
    # back entity ids.  It declines with a reason for anything a current-state index cannot
    # answer, and the reason travels back to the caller on `path`.
    index = getattr(backend, "index", None) if backend is not None else None
    declined: ExpandPath | None = None
    if index is not None:
        expansion = index.expand(tenant_id, seed_entity_id, hops, as_of=vis.as_of)
        if expansion.entity_ids is not None:
            return _frontier_values(expansion.entity_ids), [], expansion.path
        declined = expansion.path

    # --- the 0.1 snapshot path: a BFS over the unnamed CSR, current state only (the snapshot has
    # no notion of valid/tx time), and only while it is fresh.  Tried after the derived index
    # and not instead of it: a handle can hold both (Anatid.open attaches the index, and
    # build_csr() loads the snapshot), and until a generation is published the snapshot is the
    # faster of the two things that can answer.  `fresh` is what makes it safe -- it is False
    # from the first RELATES_TO write after the build.
    if backend is not None and backend.fresh and vis.is_current:
        return (
            f"SELECT entity_id FROM graph_expand({tenant_id}, {seed_entity_id}, {hops})",
            [],
            backend.note_expansion(backend.legacy_path()),
        )

    if declined is None:
        if not vis.is_current:
            # Whatever the backend holds, it is a current-state structure: the as_of forms below
            # are the only thing that can answer this, and that is the reason to report.
            declined = ExpandPath(
                "sql",
                reason=HealthReason.HISTORICAL_QUERY,
                detail="a current-state index cannot answer an as_of read; the SQL path does",
            )
        elif backend is not None:
            declined = backend.legacy_path()
        else:
            declined = ExpandPath(
                "sql",
                reason=HealthReason.ABSENT,
                detail="no csr index is registered on this handle",
            )

    if hops <= 2 and vis.is_current:
        # The exact formulation the spike benchmarked, over the relates_undirected view (one row
        # per direction, so each hop is an equality lookup rather than an OR-join).  The view
        # carries the current-state time predicate; the tenant predicate is applied here.
        t_sql, t_p = vis.tenant("r", inline=inline_ints)
        s_sql, s_p = _num(seed_entity_id, inline_ints)
        sql = (
            "WITH h1 AS ("
            "  SELECT r.b AS entity_id FROM relates_undirected r"
            f"  WHERE {t_sql} AND r.a = {s_sql})"
        )
        params: list = list(t_p) + list(s_p)
        if hops >= 2:
            sql += (
                ", h2 AS ("
                "  SELECT r.b AS entity_id FROM h1"
                "  JOIN relates_undirected r ON r.a = h1.entity_id"
                f"  WHERE {t_sql})"
            )
            params += list(t_p)
        sql += f" SELECT {s_sql}::BIGINT AS entity_id UNION ALL SELECT entity_id FROM h1"
        params += list(s_p)
        if hops >= 2:
            sql += " UNION ALL SELECT entity_id FROM h2"
        return sql, params, declined

    if hops <= 2:
        # As-of expansion: the same two levels, but with the full visibility predicate on
        # edges_relates in both directions (the view is hard-wired to the current state).
        w, wp = vis.predicate("e")
        h1 = (f"SELECT e.dst AS entity_id FROM edges_relates e WHERE {w} AND e.src = ?"
              f" UNION ALL "
              f"SELECT e.src AS entity_id FROM edges_relates e WHERE {w} AND e.dst = ?")
        params = wp + [seed_entity_id] + wp + [seed_entity_id]
        sql = f"WITH h1 AS ({h1})"
        if hops >= 2:
            h2 = (f"SELECT e.dst AS entity_id FROM h1 JOIN edges_relates e ON e.src = h1.entity_id"
                  f" WHERE {w}"
                  f" UNION ALL "
                  f"SELECT e.src AS entity_id FROM h1 JOIN edges_relates e ON e.dst = h1.entity_id"
                  f" WHERE {w}")
            sql += f", h2 AS ({h2})"
            params += wp + wp
        sql += " SELECT ?::BIGINT AS entity_id UNION ALL SELECT entity_id FROM h1"
        params.append(seed_entity_id)
        if hops >= 2:
            sql += " UNION ALL SELECT entity_id FROM h2"
        return sql, params, declined

    # --- hops > 2: recursive CTE with USING KEY (DuckDB >= 1.3). The key makes the recurring
    # table a set of entities; depth stops the recursion.  ``rel`` holds only the edges visible
    # to this tenant at this scope, so the recursion needs no predicate of its own.
    w, wp = vis.predicate("e")
    sql = (
        "WITH RECURSIVE rel AS ("
        f" SELECT e.src AS a, e.dst AS b FROM edges_relates e WHERE {w}"
        " UNION ALL"
        f" SELECT e.dst AS a, e.src AS b FROM edges_relates e WHERE {w}),"
        " bfs USING KEY (entity_id) AS ("
        "   SELECT ?::BIGINT AS entity_id, 0 AS depth"
        "   UNION"
        "   SELECT r.b AS entity_id, f.depth + 1 AS depth"
        "   FROM bfs f JOIN rel r ON r.a = f.entity_id"
        "   WHERE f.depth < ?)"
        " SELECT entity_id FROM bfs"
    )
    params = wp + wp + [seed_entity_id, hops]
    return sql, params, declined
