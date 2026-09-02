"""Graph expansion: optional C++ CSR extension, with an identical pure-SQL fallback.

Two ways to compute the k-hop frontier of a seed entity over current same-tenant ``RELATES_TO``
edges.  They return the **same entity set**; only the cost differs (Phase 0, 1M memories / 2.3M
edges / 10 tenants: 2-hop recall p50 2.88 ms pure SQL, 2.04 ms with the extension).

``"sql"`` (always available)
    Two explicit self-join levels over the ``relates_undirected`` view, UNION ALL'ed and *not*
    deduplicated -- the frontier is consumed through an ``IN (...)`` semi join where duplicates
    are harmless, and each DISTINCT costs a HASH_GROUP_BY.  This is the formulation the spike
    A/B-tested against a recursive CTE with ``USING KEY`` and found faster.  Hops > 2 fall back to
    the recursive CTE, which handles any depth.

``"extension"`` (optional)
    The ``anatid`` DuckDB extension's ``graph_expand(tenant, seed, hops)`` table function, a BFS
    over an in-memory per-tenant undirected CSR built by ``anatid_build_csr('edges_relates')``.

    Three things to know before turning it on:

    1. It needs ``allow_unsigned_extensions`` in the **connect config**, which can only be set
       when the database is opened.  ``Anatid.open(..., use_csr_extension=True)`` does that; it
       cannot be turned on afterwards.
    2. The CSR is a **snapshot**.  Edges written after ``anatid_build_csr()`` are invisible to
       ``graph_expand`` until :meth:`CsrBackend.build` runs again.  anatid therefore uses the
       extension only for ``as_of``-free reads and only while the backend reports itself fresh;
       :meth:`CsrBackend.note_edge_write` marks it stale on every RELATES_TO write.
    3. ``graph_expand``'s arguments are read at bind time, so they must be literals, not bound
       parameters.  anatid coerces them with ``int()`` before formatting -- an int cannot carry
       SQL.  No caller-supplied text ever reaches a string-built statement.

Which path is active is public: :attr:`CsrBackend.active`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from .errors import ExtensionUnavailable
from .schema import quote_ident, temporal_predicate
from .types import AsOf, CURRENT

__all__ = ["CsrInfo", "CsrBackend", "discover_extension_path", "frontier_sql"]

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


class CsrBackend:
    """Chooses and drives the graph-expansion path for one :class:`anatid.Anatid` handle.

    Construct it with ``enabled=True`` only if the database connection was opened with
    ``allow_unsigned_extensions``; otherwise ``LOAD`` fails and the backend stays on ``"sql"``.
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

    # ---------------------------------------------------------------- state

    @property
    def extension_path(self) -> Path | None:
        """Path of the extension binary anatid would load, or None."""
        return self._path

    @property
    def available(self) -> bool:
        """True when the extension is loaded on the connection and a CSR has been built."""
        return self._loaded and self._info is not None

    @property
    def fresh(self) -> bool:
        """True when the in-memory CSR still reflects ``edges_relates`` as far as anatid knows."""
        return self.available and not self._stale

    @property
    def active(self) -> str:
        """Which expansion path a current-state read would take right now: ``"extension"`` or ``"sql"``."""
        return "extension" if self.fresh else "sql"

    @property
    def info(self) -> CsrInfo | None:
        """Stats from the last :meth:`build`."""
        return self._info

    def describe(self) -> dict:
        """A dict for logs / health endpoints."""
        return {
            "active": self.active,
            "enabled": self.enabled,
            "loaded": self._loaded,
            "fresh": self.fresh,
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
        self._loaded = True
        return True

    def build(self, con, edge_table: str | None = None) -> CsrInfo | None:
        """(Re)build the in-memory CSR snapshot.  Returns None when running on the SQL path.

        Call after bulk-loading or after a batch of RELATES_TO writes.  Until it is called again,
        edges written since the last build are invisible to the extension path -- which is why
        anatid marks the backend stale on every RELATES_TO write and silently falls back to SQL.
        """
        table = edge_table or self.edge_table
        if not self.load(con):
            return None
        t0 = time.perf_counter()
        # anatid_build_csr reads its argument at BIND time, so it must be a literal, not a bound
        # parameter. quote_ident validates the name is a bare identifier before it is formatted in.
        quote_ident(table)
        row = con.execute(f"SELECT * FROM anatid_build_csr('{table}')").fetchone()
        self._info = CsrInfo(
            tenants=int(row[0]), vertices=int(row[1]), edges=int(row[2]),
            build_ms=float(row[3]), wall_ms=(time.perf_counter() - t0) * 1000.0,
            edge_table=table, built_at=time.time())
        self._stale = False
        return self._info

    def note_edge_write(self) -> None:
        """Mark the CSR snapshot stale (called by every RELATES_TO write)."""
        self._stale = True


# --------------------------------------------------------------------------- frontier SQL

_temporal = temporal_predicate


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


def frontier_sql(
    tenant_id: int,
    seed_entity_id: int,
    hops: int,
    *,
    as_of: AsOf = CURRENT,
    backend: CsrBackend | None = None,
    inline_ints: bool = False,
) -> tuple[str, list, str]:
    """Build the SQL that yields the seed's ``hops``-hop frontier as one ``entity_id`` column.

    Returns ``(sql, params, path)`` where ``sql`` is a complete SELECT usable as a subquery or CTE
    body, and ``path`` is ``"extension"`` or ``"sql"``.  Rows are NOT deduplicated: the frontier is
    consumed by a semi join.

    ``hops=0`` is the seed alone.  ``hops in (1, 2)`` uses the benchmarked two-self-join form.
    ``hops > 2`` uses a recursive CTE (correct at any depth, slower).

    ``inline_ints=True`` renders the (already ``int()``-coerced) tenant and seed as literals
    instead of bound parameters -- see :func:`_num` for why that is both safe and worth 0.6 ms.
    It only applies to the current-state forms; as-of forms always bind their timestamps.
    """
    tenant_id, seed_entity_id, hops = int(tenant_id), int(seed_entity_id), int(hops)
    if hops < 0:
        raise ValueError("hops must be >= 0")

    if hops == 0:
        seed_sql, seed_p = _num(seed_entity_id, inline_ints)
        return f"SELECT {seed_sql}::BIGINT AS entity_id", seed_p, "sql"

    # --- extension path: BFS over the CSR snapshot. Current state only (the CSR has no notion of
    # valid/tx time), and only while the snapshot is fresh.
    if backend is not None and backend.fresh and as_of.is_current:
        return (f"SELECT entity_id FROM graph_expand({tenant_id}, {seed_entity_id}, {hops})",
                [], "extension")

    if hops <= 2 and as_of.is_current:
        # The exact formulation the spike benchmarked, over the relates_undirected view (one row
        # per direction, so each hop is an equality lookup rather than an OR-join).
        t_sql, t_p = _num(tenant_id, inline_ints)
        s_sql, s_p = _num(seed_entity_id, inline_ints)
        sql = (
            "WITH h1 AS ("
            "  SELECT r.b AS entity_id FROM relates_undirected r"
            f"  WHERE r.tenant_id = {t_sql} AND r.a = {s_sql})"
        )
        params: list = list(t_p) + list(s_p)
        if hops >= 2:
            sql += (
                ", h2 AS ("
                "  SELECT r.b AS entity_id FROM h1"
                "  JOIN relates_undirected r ON r.a = h1.entity_id"
                f"  WHERE r.tenant_id = {t_sql})"
            )
            params += list(t_p)
        sql += f" SELECT {s_sql}::BIGINT AS entity_id UNION ALL SELECT entity_id FROM h1"
        params += list(s_p)
        if hops >= 2:
            sql += " UNION ALL SELECT entity_id FROM h2"
        return sql, params, "sql"

    if hops <= 2:
        # As-of expansion: the same two levels, but with the temporal predicate inlined on
        # edges_relates in both directions (the view is hard-wired to the current state).
        w, wp = _temporal("e", as_of)
        h1 = (f"SELECT e.dst AS entity_id FROM edges_relates e WHERE e.tenant_id = ? AND e.src = ? AND {w}"
              f" UNION ALL "
              f"SELECT e.src AS entity_id FROM edges_relates e WHERE e.tenant_id = ? AND e.dst = ? AND {w}")
        params = [tenant_id, seed_entity_id] + wp + [tenant_id, seed_entity_id] + wp
        sql = f"WITH h1 AS ({h1})"
        if hops >= 2:
            h2 = (f"SELECT e.dst AS entity_id FROM h1 JOIN edges_relates e ON e.src = h1.entity_id"
                  f" WHERE e.tenant_id = ? AND {w}"
                  f" UNION ALL "
                  f"SELECT e.src AS entity_id FROM h1 JOIN edges_relates e ON e.dst = h1.entity_id"
                  f" WHERE e.tenant_id = ? AND {w}")
            sql += f", h2 AS ({h2})"
            params += [tenant_id] + wp + [tenant_id] + wp
        sql += " SELECT ?::BIGINT AS entity_id UNION ALL SELECT entity_id FROM h1"
        params.append(seed_entity_id)
        if hops >= 2:
            sql += " UNION ALL SELECT entity_id FROM h2"
        return sql, params, "sql"

    # --- hops > 2: recursive CTE with USING KEY (DuckDB >= 1.3). The key makes the recurring
    # table a set of entities; depth stops the recursion.
    w, wp = _temporal("e", as_of)
    sql = (
        "WITH RECURSIVE rel AS ("
        f" SELECT e.tenant_id, e.src AS a, e.dst AS b FROM edges_relates e WHERE {w}"
        " UNION ALL"
        f" SELECT e.tenant_id, e.dst AS a, e.src AS b FROM edges_relates e WHERE {w}),"
        " bfs USING KEY (entity_id) AS ("
        "   SELECT ?::BIGINT AS entity_id, 0 AS depth"
        "   UNION"
        "   SELECT r.b AS entity_id, f.depth + 1 AS depth"
        "   FROM bfs f JOIN rel r ON r.a = f.entity_id AND r.tenant_id = ?"
        "   WHERE f.depth < ?)"
        " SELECT entity_id FROM bfs"
    )
    params = wp + wp + [seed_entity_id, tenant_id, hops]
    return sql, params, "sql"
