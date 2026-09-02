"""The optional C++ CSR extension, checked against the pure-SQL expansion it replaces.

The whole point of ``ext/`` is that it is a *drop-in* for :func:`anatid.csr.frontier_sql`'s SQL
path: same frontier, less time.  So the load-bearing test here is not "does the extension work",
it is "do the two paths return the SAME entity ids" -- on the Phase 0 spike's 100k-memory dataset,
for hundreds of real benchmark seeds, at 1 and 2 hops.  A third arm compares both against
``spike/bench/common.py``'s pure-numpy oracle.

Everything skips with a reason when the extension has not been built.  Build it with::

    cd ext && PATH=<venv>/bin:$PATH GEN=ninja make release

or point ``$ANATID_EXTENSION_PATH`` at a binary built somewhere else.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILT_EXTENSION = REPO_ROOT / "ext" / "build" / "release" / "extension" / "anatid" / "anatid.duckdb_extension"
SPIKE_SMALL = REPO_ROOT / "spike" / "data" / "small"

#: Benchmark seeds compared on both paths.  The brief asks for at least 100.
SEED_COUNT = 200

#: The version banner the hardened extension reports.  An older binary (the Phase 0 spike build)
#: still passes the parity tests but has no strict mode, hop limit or stats functions.
HARDENED_PREFIX = "anatid 0.1."


def _extension_path() -> Path | None:
    """The extension binary to test: ``$ANATID_EXTENSION_PATH`` if set, else ``ext/``'s build."""
    env = os.environ.get("ANATID_EXTENSION_PATH")
    if env:
        candidate = Path(env).expanduser()
        return candidate if candidate.is_file() else None
    return BUILT_EXTENSION if BUILT_EXTENSION.is_file() else None


@pytest.fixture(scope="module")
def extension_path() -> Path:
    path = _extension_path()
    if path is None:
        pytest.skip(
            "the anatid DuckDB extension is not built. Expected it at "
            f"{BUILT_EXTENSION} -- build it with `cd ext && GEN=ninja make release` -- "
            "or set $ANATID_EXTENSION_PATH to a binary built elsewhere.")
    return path


@pytest.fixture(scope="module")
def ext_db(extension_path, tmp_path_factory):
    """The spike ``small`` dataset in a real anatid database with the extension loaded."""
    from anatid import Anatid
    from anatid.errors import ExtensionUnavailable

    if not (SPIKE_SMALL / "edges_relates.parquet").is_file():
        pytest.skip(f"spike small dataset not present at {SPIKE_SMALL}")
    path = tmp_path_factory.mktemp("csr_ext") / "ext.anatid"
    try:
        handle = Anatid.open(path, tenant=0, embedding_dim=64, fts=False,
                             use_csr_extension=True, extension_path=extension_path,
                             require_extension=True)
    except ExtensionUnavailable as exc:  # wrong platform, wrong DuckDB version, unreadable file
        pytest.skip(f"could not load {extension_path}: {exc}")
    try:
        counts = handle.load_parquet(SPIKE_SMALL, rebuild_fts=False, build_csr=False)
        assert counts.get("memories") == 100_000, counts
        info = handle.build_csr()
        assert info is not None, "build_csr() returned None -- the extension did not load"
        yield handle
    finally:
        handle.close()


@pytest.fixture(scope="module")
def seeds(ext_db) -> list[tuple[int, int]]:
    """(tenant_id, seed_entity_id) for the first SEED_COUNT benchmark queries."""
    rows = ext_db.connection.execute(
        "SELECT tenant_id, seed_entity_id FROM read_parquet(?) ORDER BY query_id LIMIT ?",
        [str(SPIKE_SMALL / "queries.parquet"), SEED_COUNT]).fetchall()
    assert len(rows) >= 100, f"need >= 100 seeds, got {len(rows)}"
    return [(int(t), int(s)) for t, s in rows]


@pytest.fixture(scope="module")
def hardened(ext_db, extension_path) -> str:
    """Skip the hardening tests when the binary under test predates them."""
    try:
        banner = ext_db.connection.execute("SELECT anatid_version()").fetchone()[0]
    except Exception as exc:  # the spike build has no zero-argument overload
        pytest.skip(f"{extension_path} predates the hardened extension: {exc}")
    if not banner.startswith(HARDENED_PREFIX):
        pytest.skip(f"{extension_path} reports {banner!r}, not {HARDENED_PREFIX}x")
    return banner


def _frontier(db, tenant: int, seed: int, hops: int, *, backend) -> tuple[list[int], str]:
    """Run :func:`anatid.csr.frontier_sql` and return (sorted unique ids, which path ran)."""
    from anatid.csr import frontier_sql

    sql, params, path = frontier_sql(tenant, seed, hops, backend=backend)
    rows = db.connection.execute(sql, params).fetchall()
    return sorted({int(r[0]) for r in rows}), path


# --------------------------------------------------------------------------- the extension loads

def test_extension_loads_and_builds_a_csr(ext_db):
    """The extension is live, the CSR covers the dataset, and reads take the C++ path."""
    info = ext_db.build_csr()
    con = ext_db.connection
    expected_edges = con.execute(
        "SELECT count(*) FROM edges_relates WHERE valid_to IS NULL AND tx_to IS NULL").fetchone()[0]
    expected_tenants = con.execute(
        "SELECT count(DISTINCT tenant_id) FROM edges_relates "
        "WHERE valid_to IS NULL AND tx_to IS NULL").fetchone()[0]
    # For the record: the spike small dataset is 10 tenants / 9,963 vertices / 26,479 current edges.
    assert info.edges == expected_edges
    assert info.tenants == expected_tenants
    assert info.vertices > 0 and info.build_ms >= 0.0
    assert ext_db.csr.fresh is True
    assert ext_db.expand_path == "extension"
    assert ext_db.stats()["expand_path"] == "extension"


def test_csr_stats_functions_report_the_snapshot(ext_db, hardened):
    """anatid_csr_stats() / anatid_csr_tenants() describe what is actually in memory."""
    ext_db.build_csr()
    con = ext_db.connection
    row = con.execute(
        "SELECT edge_table, current_filter, tenants, vertices, edges, bytes, build_ms, built_at "
        "FROM anatid_csr_stats()").fetchone()
    edge_table, current_filter, tenants, vertices, edges, nbytes, build_ms, built_at = row
    assert edge_table == "edges_relates"
    # The CSR applies anatid's full current-state predicate, not just valid_to.
    assert current_filter == "valid_to IS NULL AND tx_to IS NULL"
    assert (tenants, vertices, edges) == (ext_db.csr.info.tenants, ext_db.csr.info.vertices,
                                          ext_db.csr.info.edges)
    assert nbytes > 0 and build_ms >= 0.0 and built_at is not None

    per_tenant = con.execute(
        "SELECT tenant_id, vertices, edges, min_entity_id, max_entity_id, bytes "
        "FROM anatid_csr_tenants()").fetchall()
    assert len(per_tenant) == tenants
    assert [r[0] for r in per_tenant] == sorted(r[0] for r in per_tenant), "not ordered by tenant"
    assert sum(r[1] for r in per_tenant) == vertices
    assert sum(r[2] for r in per_tenant) == edges
    assert sum(r[5] for r in per_tenant) <= nbytes
    for tenant_id, _v, _e, lo, hi, _b in per_tenant:
        assert lo <= hi
        real_lo, real_hi = con.execute(
            "SELECT min(least(src, dst)), max(greatest(src, dst)) FROM edges_relates "
            "WHERE tenant_id = ? AND valid_to IS NULL AND tx_to IS NULL", [tenant_id]).fetchone()
        assert (lo, hi) == (real_lo, real_hi)


# --------------------------------------------------------------------------- the load-bearing test

@pytest.mark.slow
@pytest.mark.parametrize("hops", [1, 2])
def test_extension_and_sql_frontiers_are_identical(ext_db, seeds, hops):
    """The C++ BFS and the pure-SQL expansion return the SAME entity ids, seed by seed.

    This is the contract that lets :attr:`anatid.CsrBackend.active` flip between the two without
    changing a single answer -- 200 benchmark seeds from the spike's 100k-memory dataset.
    """
    ext_db.build_csr()
    assert ext_db.csr.fresh, "the CSR must be fresh or the extension arm silently falls back to SQL"
    mismatches = []
    reached = 0
    for tenant, seed in seeds:
        got_ext, path_ext = _frontier(ext_db, tenant, seed, hops, backend=ext_db.csr)
        got_sql, path_sql = _frontier(ext_db, tenant, seed, hops, backend=None)
        assert path_ext == "extension" and path_sql == "sql", (path_ext, path_sql)
        reached += len(got_ext)
        if got_ext != got_sql:
            mismatches.append((tenant, seed, len(got_ext), len(got_sql)))
    assert not mismatches, (
        f"{len(mismatches)}/{len(seeds)} seeds disagree at {hops} hops "
        f"(tenant, seed, n_extension, n_sql): {mismatches[:5]}")
    # Guard against a vacuous pass: seed-only frontiers on both sides would agree about nothing.
    assert reached > 3 * len(seeds), (
        f"only {reached} entities reached from {len(seeds)} seeds at {hops} hops -- "
        "the frontiers are trivial and this comparison proves nothing")


@pytest.mark.slow
@pytest.mark.oracle
def test_frontiers_match_the_numpy_oracle(ext_db, seeds, spike_common):
    """Both paths agree with ``spike/bench/common.py``'s pure-numpy reference frontier."""
    ext_db.build_csr()
    reached = 0
    for tenant, seed in seeds:
        expected = spike_common.reference_frontier("small", tenant, seed)
        got_ext, _ = _frontier(ext_db, tenant, seed, 2, backend=ext_db.csr)
        got_sql, _ = _frontier(ext_db, tenant, seed, 2, backend=None)
        assert got_ext == expected, f"extension frontier differs for tenant {tenant} seed {seed}"
        assert got_sql == expected, f"sql frontier differs for tenant {tenant} seed {seed}"
        reached += len(expected)
    assert reached > 3 * len(seeds), f"trivial frontiers ({reached} entities from {len(seeds)} seeds)"


@pytest.mark.slow
def test_recall_2hop_ids_identical_on_both_paths(ext_db, seeds):
    """End to end: anatid's own verb returns the same rows whichever path it takes."""
    ext_db.build_csr()
    assert ext_db.expand_path == "extension"
    with_ext = [ext_db.recall_2hop_ids(seed, tenant=tenant) for tenant, seed in seeds]
    ext_db.csr.note_edge_write()            # what a RELATES_TO write does
    assert ext_db.expand_path == "sql"
    with_sql = [ext_db.recall_2hop_ids(seed, tenant=tenant) for tenant, seed in seeds]
    ext_db.build_csr()
    differing = [seeds[i] for i, (a, b) in enumerate(zip(with_ext, with_sql)) if a != b]
    assert not differing, f"{len(differing)} seeds differ: {differing[:5]}"
    assert any(rows for rows in with_ext), "every seed returned nothing -- the test proved nothing"


# --------------------------------------------------------------------------- the hardening

def test_unknown_tenant_returns_the_seed_and_strict_mode_explains(ext_db, hardened):
    """An unknown tenant is not an error by default -- that is what the SQL path does too."""
    ext_db.build_csr()
    con = ext_db.connection
    unknown = 987_654
    assert con.execute(f"SELECT entity_id, depth FROM graph_expand({unknown}, 1, 2)").fetchall() == [(1, 0)]
    got_sql, _ = _frontier(ext_db, unknown, 1, 2, backend=None)
    assert got_sql == [1]

    with pytest.raises(Exception) as excinfo:
        con.execute(f"SELECT * FROM graph_expand({unknown}, 1, 2, strict := true)").fetchall()
    message = str(excinfo.value)
    assert f"tenant {unknown} has no edges in the CSR" in message
    assert "edges_relates" in message, message

    # A seed with no current RELATES_TO edge, in a tenant that does exist.
    with pytest.raises(Exception) as excinfo:
        con.execute("SELECT * FROM graph_expand(0, 999999999, 2, strict := true)").fetchall()
    assert "has no current edge in tenant 0" in str(excinfo.value)


def test_hop_limit_defaults_to_30_and_can_be_raised(ext_db, hardened):
    """A guard rail with a clear error, not a wall."""
    ext_db.build_csr()
    con = ext_db.connection
    seed = con.execute(
        "SELECT src FROM edges_relates WHERE tenant_id = 0 AND valid_to IS NULL LIMIT 1").fetchone()[0]
    assert con.execute(f"SELECT count(*) FROM graph_expand(0, {seed}, 30)").fetchone()[0] > 0

    with pytest.raises(Exception) as excinfo:
        con.execute(f"SELECT * FROM graph_expand(0, {seed}, 31)").fetchall()
    message = str(excinfo.value)
    assert "hops=31 exceeds the limit of 30" in message
    assert "max_hops := 31" in message, message

    deep = con.execute(f"SELECT count(*) FROM graph_expand(0, {seed}, 31, max_hops := 31)").fetchone()[0]
    assert deep > 0

    with pytest.raises(Exception) as excinfo:
        con.execute(f"SELECT * FROM graph_expand(0, {seed}, -1)").fetchall()
    assert "hops must be >= 0" in str(excinfo.value)


def test_graph_expand_without_a_csr_says_how_to_build_one(extension_path, hardened):
    """No CSR is a clear instruction, not a crash and not an empty result."""
    from anatid import Anatid
    from anatid.errors import ExtensionUnavailable

    try:
        handle = Anatid.open(":memory:", tenant=0, embedding_dim=8, fts=False,
                             use_csr_extension=True, extension_path=extension_path,
                             require_extension=True)
    except ExtensionUnavailable as exc:
        pytest.skip(f"could not load {extension_path}: {exc}")
    with handle:
        # No anatid_build_csr() has run on this database.
        assert handle.connection.execute("SELECT count(*) FROM anatid_csr_stats()").fetchone()[0] == 0
        with pytest.raises(Exception) as excinfo:
            handle.connection.execute("SELECT * FROM graph_expand(0, 1, 2)").fetchall()
        message = str(excinfo.value)
        assert "no CSR has been built" in message
        assert "anatid_build_csr" in message, message

        # A table that is not an edge table is named, with the columns it does have.
        handle.connection.execute("CREATE TABLE not_edges(a BIGINT, b BIGINT)")
        with pytest.raises(Exception) as excinfo:
            handle.connection.execute("SELECT * FROM anatid_build_csr('not_edges')").fetchall()
        assert "has no column 'tenant_id'" in str(excinfo.value)
        with pytest.raises(Exception) as excinfo:
            handle.connection.execute("SELECT * FROM anatid_build_csr('nope')").fetchall()
        assert "cannot read edge table 'nope'" in str(excinfo.value)


def test_a_relates_write_makes_the_snapshot_stale(extension_path):
    """The CSR is a snapshot: anatid drops back to SQL until it is rebuilt."""
    from anatid import Anatid
    from anatid.errors import ExtensionUnavailable

    try:
        handle = Anatid.open(":memory:", tenant=0, embedding_dim=8, fts=False,
                             use_csr_extension=True, extension_path=extension_path,
                             require_extension=True)
    except ExtensionUnavailable as exc:
        pytest.skip(f"could not load {extension_path}: {exc}")
    with handle:
        # Dense, caller-supplied entity ids: the CSR indexes by (entity_id - min_id).
        for entity_id in range(4):
            handle.execute("INSERT INTO entities (entity_id, tenant_id, name) VALUES (?, 0, ?)",
                           [entity_id, f"e{entity_id}"])
        handle.relate(0, 1)
        handle.relate(1, 2)
        assert handle.build_csr() is not None
        assert handle.expand_path == "extension"
        before, _ = _frontier(handle, 0, 0, 2, backend=handle.csr)
        assert before == [0, 1, 2]

        handle.relate(2, 3)                        # a write the snapshot cannot see
        assert handle.csr.fresh is False
        assert handle.expand_path == "sql"
        # The stale snapshot is still the old answer -- which is exactly why anatid stops using it.
        stale, path = _frontier(handle, 0, 0, 2, backend=None)
        assert path == "sql"
        assert stale == [0, 1, 2]
        deeper, _ = _frontier(handle, 0, 1, 2, backend=None)
        assert deeper == [0, 1, 2, 3]

        handle.build_csr()
        assert handle.expand_path == "extension"
        fresh, path = _frontier(handle, 0, 1, 2, backend=handle.csr)
        assert path == "extension"
        assert fresh == [0, 1, 2, 3]


def test_anatid_default_ids_are_too_sparse_for_the_csr(extension_path, hardened):
    """anatid's own 63-bit time-ordered ids cannot back a dense CSR, and the build says so.

    The offsets array costs 8 bytes per id in ``[min_id, max_id]``, occupied or not.  Two entities
    minted 10 ms apart by :func:`anatid.new_id` are ~42,000,000 ids apart -- a third of a gigabyte
    of offsets for a single edge.  Refusing is the hardening; silently allocating it was the bug.
    """
    import time

    from anatid import Anatid, new_id
    from anatid.errors import ExtensionUnavailable

    try:
        handle = Anatid.open(":memory:", tenant=0, embedding_dim=8, fts=False,
                             use_csr_extension=True, extension_path=extension_path,
                             require_extension=True)
    except ExtensionUnavailable as exc:
        pytest.skip(f"could not load {extension_path}: {exc}")
    with handle:
        first = new_id()
        time.sleep(0.01)                       # ~42M ids of drift at 2**22 ids per millisecond
        second = new_id()
        assert second - first > 4096, (first, second)
        handle.relate(first, second)           # int ids pass through entity_id() verbatim
        with pytest.raises(Exception) as excinfo:
            handle.build_csr()
        message = str(excinfo.value)
        assert "current edge(s) but its entity ids span" in message, message
        assert "max_span_factor := 0" in message, message
        # anatid never silently serves a stale or wrong answer because of it: the SQL path stands.
        assert handle.expand_path == "sql"
        frontier, path = _frontier(handle, 0, first, 2, backend=handle.csr)
        assert path == "sql"
        assert frontier == sorted([first, second])
