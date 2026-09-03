"""The CSR as a derived index: base generation + change journal, checked against the SQL oracle.

The load-bearing test here is :func:`test_random_mutations_keep_csr_and_sql_identical`: seed a
random graph, apply 500 random mutations (relate, expire an edge, hard-forget an entity, supersede
a memory) interleaved with rebuilds and 2-hop recalls, and after EVERY step assert that the CSR
plus its journal returns exactly the rows the pure-SQL path returns.  It runs twice: once with the
merge done in SQL over the generation's tables, and once with it done in C++ inside
``graph_expand``.  The C++ half skips cleanly when the extension has not been built.

Everything else in this file is a property the design doc names: the dense vertex numbering never
leaves the library, a publish does not move a read that pinned the previous generation, a hard
erasure leaves nothing of the erased ids in any generation's storage, a second handle's writes
reach the journal, and every refusal to use the index reports WHY.
"""

from __future__ import annotations

import datetime as _dt
import os
import random
from pathlib import Path

import pytest

from anatid import Anatid
from anatid.csr import CsrIndex, attach_csr_index, frontier_sql
from anatid.derived import HealthReason, MaintenancePolicy, maintain
from anatid.errors import ExtensionUnavailable
from anatid.recall import recall_2hop_ids as sql_recall_2hop_ids
from anatid.schema import quote_ident
from anatid.types import AsOf

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILT_EXTENSION = (
    REPO_ROOT / "ext" / "build" / "release" / "extension" / "anatid" / "anatid.duckdb_extension"
)

DIM = 8
TENANT = 1

#: The brief asks for 500.  Each one is followed by a full comparison against the oracle.
MUTATIONS = 500

#: How often a rebuild is attempted, and how aggressive the policy is when it runs.
ALWAYS = MaintenancePolicy(
    rebuild_after_rows=1, rebuild_after_ratio=None, rebuild_after_seconds=None
)


def _extension_path() -> Path | None:
    env = os.environ.get("ANATID_EXTENSION_PATH")
    if env:
        candidate = Path(env).expanduser()
        return candidate if candidate.is_file() else None
    return BUILT_EXTENSION if BUILT_EXTENSION.is_file() else None


# --------------------------------------------------------------------------- the oracle


def oracle_frontier(db, seed: int, hops: int, *, tenant: int = TENANT, as_of=None) -> list[int]:
    """The frontier the pure-SQL path returns: no index, no journal, canonical tables only."""
    sql, params, path = frontier_sql(tenant, seed, hops, as_of=AsOf.coerce(as_of), backend=None)
    assert path == "sql", path
    rows = (
        db.connection.execute(sql, params).fetchall()
        if params
        else db.connection.execute(sql).fetchall()
    )
    return sorted({int(r[0]) for r in rows})


def oracle_recall(db, seed: int, *, hops: int = 2, limit: int = 50, tenant: int = TENANT):
    """``recall_2hop_ids`` with the index taken out of the loop."""
    return sql_recall_2hop_ids(
        db.connection, tenant_id=tenant, seed_entity_id=seed, limit=limit, hops=hops, backend=None
    )


# --------------------------------------------------------------------------- fixtures


def _open(path: Path, extension: Path | None) -> Anatid:
    kwargs = {}
    if extension is not None:
        kwargs = {"use_csr_extension": True, "extension_path": extension, "require_extension": True}
    return Anatid.open(path, tenant=TENANT, embedding_dim=DIM, fts=False, **kwargs)


@pytest.fixture(params=["csr", "extension"])
def csr_db(request, tmp_path):
    """An anatid handle with the CSR index attached, on each merge strategy.

    ``"csr"``: the merge runs in SQL over the generation's tables, no extension involved.
    ``"extension"``: the same generation, expanded in C++ with the journal passed in.  The
    strategy is pinned rather than left on ``"auto"`` so a silent downgrade to SQL shows up as a
    failed assertion on ``expansion.path`` instead of passing quietly.
    """
    extension = _extension_path() if request.param == "extension" else None
    if request.param == "extension" and extension is None:
        pytest.skip(
            "the anatid DuckDB extension is not built. Expected it at "
            f"{BUILT_EXTENSION} -- build it with `cd ext && GEN=ninja make release` -- "
            "or set $ANATID_EXTENSION_PATH to a binary built elsewhere."
        )
    try:
        db = _open(tmp_path / f"csr-{request.param}.anatid", extension)
    except ExtensionUnavailable as exc:  # wrong platform / wrong DuckDB version
        pytest.skip(f"could not load {extension}: {exc}")
    # The fixture parameter is the PATH a read should report; the strategy that produces it is
    # "sql" for the SQL merge, which reports itself as "csr" (a generation answered), and
    # "extension" for the C++ one.  Pinning it rather than leaving it on "auto" is what makes a
    # silent downgrade to SQL a failed assertion instead of a quiet pass.
    index = attach_csr_index(db, strategy="sql" if request.param == "csr" else "extension")
    try:
        yield db, index, request.param
    finally:
        db.close()


@pytest.fixture
def plain_db(tmp_path):
    """A handle with the index attached and no extension: the SQL merge only."""
    db = _open(tmp_path / "plain.anatid", None)
    index = attach_csr_index(db, strategy="sql")
    try:
        yield db, index
    finally:
        db.close()


# --------------------------------------------------------------------------- graph helpers


def seed_graph(db, *, entities: int, edges: int, rng: random.Random) -> list[int]:
    """A random connected-ish graph: one memory per entity, then random relations."""
    names = [f"n{i}" for i in range(entities)]
    for name in names:
        db.remember(f"a memory about {name}", entities=[name])
    ids = [db.entity_id(name, create=False) for name in names]
    for i in range(1, entities):  # a spanning path, so nothing is isolated
        db.relate(ids[i - 1], ids[i])
    for _ in range(max(0, edges - entities + 1)):
        a, b = rng.sample(ids, 2)
        db.relate(a, b)
    return ids


def live_edges(db, *, tenant: int = TENANT) -> list[tuple[int, int]]:
    rows = db.execute(
        "SELECT src, dst FROM edges_relates WHERE tenant_id = ? "
        "AND valid_to IS NULL AND tx_to IS NULL",
        [tenant],
    ).fetchall()
    return [(int(a), int(b)) for a, b in rows]


def live_entities(db, *, tenant: int = TENANT) -> list[int]:
    rows = db.execute(
        "SELECT entity_id FROM entities WHERE tenant_id = ? ORDER BY entity_id", [tenant]
    ).fetchall()
    return [int(r[0]) for r in rows]


def hard_forget_entity(
    db, entity_id: int, *, tenant: int = TENANT, now: _dt.datetime | None = None
) -> int:
    """Purge an entity and every RELATES_TO edge it touches, the way ``forget(hard=True)`` does.

    There is no ``forget`` verb for an entity, so this is the shape a purge takes: announce the
    erasure BEFORE the transaction opens (so a build in flight is told), delete the canonical
    rows, and hand the ids to the registry, which deletes them from every generation's storage
    and from the journal.
    """
    at = now or _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)
    db.indexes.announce_erasure("edges_relates")
    with db.indexes.lifecycle():
        with db.transaction():
            rows = db.execute(
                "DELETE FROM edges_relates WHERE (src = ? OR dst = ?) AND tenant_id = ? "
                "RETURNING edge_id",
                [entity_id, entity_id, tenant],
            ).fetchall()
            edge_ids = sorted({int(r[0]) for r in rows})
            if edge_ids:
                db.indexes.erase("edges_relates", tenant, edge_ids, at=at, reason="forget_hard")
            db.execute(
                "DELETE FROM edges_about WHERE dst = ? AND tenant_id = ?", [entity_id, tenant]
            )
            db.execute(
                "DELETE FROM entities WHERE entity_id = ? AND tenant_id = ?", [entity_id, tenant]
            )
    return len(edge_ids)


def generation_tables(db, index: CsrIndex) -> list[str]:
    """Every storage table of every generation this index still has."""
    out = []
    for gen in index.all_generations():
        out.extend(index.storage_tables(gen))
    present = {r[0] for r in db.execute("SELECT table_name FROM duckdb_tables()").fetchall()}
    return [t for t in out if t in present]


# --------------------------------------------------------------------------- THE acceptance test


def test_random_mutations_keep_csr_and_sql_identical(csr_db):
    """500 random mutations, and after every one the CSR agrees with the SQL path exactly.

    The mutations are the four that can make a snapshot wrong: a new relation (the base does not
    have it), an expired relation (the base has an edge that is no longer current), a hard-forgotten
    entity (the base has edges whose rows no longer exist at all) and a superseded memory (the
    frontier is unchanged but the rows behind it are not).  Rebuilds are interleaved so the base
    generation moves under the reads, and every published generation is validated against the
    oracle before it goes live.
    """
    db, index, strategy = csr_db
    rng = random.Random(20260903)
    ids = seed_graph(db, entities=40, edges=70, rng=rng)
    report = maintain(index, TENANT, ALWAYS, raise_on_failure=True)
    assert report.action == "published", report

    seeds = rng.sample(ids, 3)
    counts = {"relate": 0, "unrelate": 0, "purge": 0, "supersede": 0, "rebuild": 0}
    reached = 0
    for step in range(MUTATIONS):
        entities = live_entities(db)
        roll = rng.random()
        if roll < 0.36 and len(entities) >= 2:
            a, b = rng.sample(entities, 2)
            db.relate(a, b)
            counts["relate"] += 1
        elif roll < 0.58:
            edges = live_edges(db)
            if edges:
                a, b = rng.choice(edges)
                db.unrelate(a, b)
                counts["unrelate"] += 1
        elif roll < 0.68 and len(entities) > 12:
            victim = rng.choice([e for e in entities if e not in seeds])
            hard_forget_entity(db, victim)
            counts["purge"] += 1
        elif roll < 0.88:
            rows = db.execute(
                "SELECT memory_id FROM memories WHERE tenant_id = ? AND valid_to IS NULL "
                "AND tx_to IS NULL LIMIT 40",
                [TENANT],
            ).fetchall()
            if rows:
                db.supersede(int(rng.choice(rows)[0]), f"corrected at step {step}")
                counts["supersede"] += 1
        else:
            maintain(index, TENANT, ALWAYS, raise_on_failure=True)
            counts["rebuild"] += 1

        for seed in seeds:
            for hops in (1, 2):
                expansion = index.expand(TENANT, seed, hops)
                assert expansion.entity_ids is not None, expansion.path.explain()
                assert expansion.path == strategy, (
                    f"step {step}: expected the {strategy!r} path, got {expansion.path.explain()}"
                )
                expected = oracle_frontier(db, seed, hops)
                assert sorted(expansion.entity_ids) == expected, (
                    f"step {step} ({counts}): {hops}-hop frontier of {seed} differs"
                )
                reached += len(expected)
            assert db.recall_2hop_ids(seed, limit=50) == oracle_recall(db, seed, limit=50), (
                f"step {step} ({counts}): recall_2hop_ids differs for seed {seed}"
            )

    assert all(counts.values()), f"some mutation never happened: {counts}"
    # Guard against a vacuous pass: frontiers of one entity would agree about nothing.
    assert reached > 6 * MUTATIONS, f"frontiers were trivial ({reached} entities over {MUTATIONS})"


def test_both_merge_strategies_agree_with_each_other(plain_db):
    """The SQL merge and, when it is available, the C++ one are interchangeable.

    The acceptance test above runs each strategy against the oracle separately; this runs them
    against each other on one database, which is what a caller flipping ``strategy`` gets.
    """
    extension = _extension_path()
    if extension is None:
        pytest.skip("the anatid DuckDB extension is not built")
    db, _index = plain_db
    db.close()
    path = Path(db.path).with_name("both.anatid")
    try:
        handle = _open(path, extension)
    except ExtensionUnavailable as exc:
        pytest.skip(f"could not load {extension}: {exc}")
    with handle:
        index = attach_csr_index(handle, strategy="auto")
        rng = random.Random(7)
        ids = seed_graph(handle, entities=25, edges=45, rng=rng)
        maintain(index, TENANT, ALWAYS, raise_on_failure=True)
        for _ in range(12):
            a, b = rng.sample(ids, 2)
            handle.relate(a, b)
        for _ in range(8):
            edges = live_edges(handle)
            a, b = rng.choice(edges)
            handle.unrelate(a, b)
        differing = []
        live = index.current_generation(TENANT)
        assert live is not None
        for seed in ids:
            for hops in (1, 2, 3):
                in_sql, sql_path, _n, _d = index.expand_generation(live, seed, hops, strategy="sql")
                in_ext, ext_path, _n2, _d2 = index.expand_generation(
                    live, seed, hops, strategy="extension"
                )
                assert (sql_path, ext_path) == ("csr", "extension")
                expected = oracle_frontier(handle, seed, hops)
                if sorted(in_sql) != expected or sorted(in_ext) != expected:
                    differing.append((seed, hops, len(in_sql), len(in_ext), len(expected)))
        assert not differing, differing[:5]


# --------------------------------------------------------------------------- dense ids


def test_dense_vertex_ids_are_internal_and_owned_by_the_generation(plain_db):
    """anatid's own sparse 63-bit entity ids go in and come out; the CSR sees 0..n-1.

    This is the fix for the extension's dense-id assumption.  Before it, a real anatid database
    could not build a CSR at all: two entities minted 10 ms apart are ~42,000,000 ids apart, and
    the offsets array costs 8 bytes per id in the range.
    """
    db, index = plain_db
    rng = random.Random(3)
    ids = seed_graph(db, entities=12, edges=18, rng=rng)
    assert max(ids) - min(ids) > 4096, "these ids are not the sparse ones this test is about"
    gen = index.build_next(TENANT)
    vertices, edges = index.storage_tables(gen)

    rows = db.execute(
        f"SELECT vertex_id, entity_id FROM {quote_ident(vertices)} ORDER BY vertex_id"
    ).fetchall()
    dense = [int(r[0]) for r in rows]
    external = [int(r[1]) for r in rows]
    assert dense == list(range(len(rows))), "the mapping is not dense 0..n-1"
    assert external == sorted(external) and len(set(external)) == len(external)
    assert set(external) <= set(ids)

    # the edge table is stored in the dense space, and its entity pair agrees with the mapping
    crossed = db.execute(
        f"SELECT count(*) FROM {quote_ident(edges)} e "
        f"JOIN {quote_ident(vertices)} vs ON vs.vertex_id = e.src "
        f"JOIN {quote_ident(vertices)} vd ON vd.vertex_id = e.dst "
        f"WHERE least(vs.entity_id, vd.entity_id) IS DISTINCT FROM e.a "
        f"OR greatest(vs.entity_id, vd.entity_id) IS DISTINCT FROM e.b"
    ).fetchone()[0]
    assert crossed == 0
    top = db.execute(f"SELECT max(src), max(dst) FROM {quote_ident(edges)}").fetchone()
    assert max(top) < len(rows), "a dense vertex id is outside 0..n-1"

    # and nothing a caller ever sees is a vertex id
    index.publish(index.validate(gen).generation)
    frontier = index.expand(TENANT, ids[0], 2)
    assert set(frontier.entity_ids) <= set(ids)
    assert sorted(frontier.entity_ids) == oracle_frontier(db, ids[0], 2)


def test_hops_zero_is_the_seed_alone_on_every_path(plain_db):
    """A zero-hop expansion is the seed, whatever merged it: the same thing the SQL path returns."""
    db, index = plain_db
    ids = seed_graph(db, entities=8, edges=12, rng=random.Random(41))
    maintain(index, TENANT, ALWAYS, raise_on_failure=True)
    for seed in ids[:3]:
        assert index.expand(TENANT, seed, 0).entity_ids == (seed,)
        assert oracle_frontier(db, seed, 0) == [seed]
        # frontier_sql answers hops = 0 without consulting the index at all
        sql, params, path = frontier_sql(TENANT, seed, 0, backend=db.csr)
        assert path == "sql" and "seed alone" in path.detail
        rows = db.execute(sql, params).fetchall()
        assert [int(r[0]) for r in rows] == [seed]


def test_two_generations_number_their_vertices_independently(plain_db):
    """A rebuild renumbers, which is exactly why a read pins one generation and stays on it."""
    db, index = plain_db
    rng = random.Random(11)
    ids = seed_graph(db, entities=10, edges=14, rng=rng)
    first = index.build_next(TENANT)
    index.publish(index.validate(first).generation)
    hard_forget_entity(db, ids[0])
    second = index.build_next(TENANT)
    index.publish(index.validate(second).generation)

    def mapping(gen):
        v, _e = index.storage_tables(gen)
        return {
            int(a): int(b)
            for a, b in db.execute(f"SELECT entity_id, vertex_id FROM {quote_ident(v)}").fetchall()
        }

    one, two = mapping(first), mapping(second)
    assert one and two and one != two
    shared = set(one) & set(two)
    assert shared, "the two generations share no entity, so this proves nothing"
    assert any(one[e] != two[e] for e in shared), "the renumbering did not move anything"


# --------------------------------------------------------------------------- pinning


def test_a_publish_mid_read_does_not_alter_a_pinned_read(plain_db):
    """A read that pinned generation N keeps N's base, its storage and its answer."""
    db, index = plain_db
    rng = random.Random(5)
    ids = seed_graph(db, entities=20, edges=30, rng=rng)
    first = index.build_next(TENANT)
    index.publish(index.validate(first).generation)
    seed = ids[0]

    for _ in range(6):
        a, b = rng.sample(ids, 2)
        db.relate(a, b)
    db.unrelate(*live_edges(db)[0])

    with index.pin(TENANT) as pin:
        assert pin and pin.generation is not None and pin.generation.generation == first.generation
        before, path, _note, _delta = index.expand_generation(pin.generation, seed, 2)
        assert sorted(before) == oracle_frontier(db, seed, 2)

        second = index.build_next(TENANT)
        published = index.publish(index.validate(second).generation)
        assert published.generation != first.generation
        # the pinned generation is still there, with its storage, and still refuses to go
        assert index.retire(pin.generation) is False
        assert index.pinned(pin.generation) == 1
        after, path_again, _n, _d = index.expand_generation(pin.generation, seed, 2)
        assert (after, path_again) == (before, path)
        # ... and the new one answers the same question the same way
        fresh = index.expand(TENANT, seed, 2)
        assert fresh.generation is not None
        assert fresh.generation.generation == published.generation
        assert sorted(fresh.entity_ids) == sorted(before)

    assert index.retire(first) is True
    assert [g.generation for g in index.generations(TENANT)] == [published.generation]


# --------------------------------------------------------------------------- health reasons


def test_expand_path_says_why_it_is_on_the_sql_path(plain_db):
    """Every refusal is a named reason, not a shrug.  ``expand_path`` carries it."""
    db, index = plain_db
    ids = seed_graph(db, entities=8, edges=12, rng=random.Random(2))
    seed = ids[0]

    # 1. nothing published yet
    path = frontier_sql(TENANT, seed, 2, backend=db.csr)[2]
    assert (path, path.reason) == ("sql", HealthReason.ABSENT)
    assert db.expand_path.reason is HealthReason.ABSENT
    assert "no generation" in db.expand_path.detail

    gen = index.build_next(TENANT)
    index.publish(index.validate(gen).generation)

    # 2. published, validated, nothing pending
    path = frontier_sql(TENANT, seed, 2, backend=db.csr)[2]
    assert (path, path.reason) == ("csr", HealthReason.FRESH)
    assert path.generation == gen.generation
    assert db.expand_path == "csr"

    # 3. a historical read cannot use a current-state index, whatever its health
    past = AsOf.coerce(_dt.datetime(2020, 1, 1))
    path = frontier_sql(TENANT, seed, 2, as_of=past, backend=db.csr)[2]
    assert (path, path.reason) == ("sql", HealthReason.HISTORICAL_QUERY)
    assert "as_of" in path.detail
    assert db.recall_2hop_ids(seed, as_of=_dt.datetime(2020, 1, 1)) == []

    # 4. a bulk load bypasses the journal, so the generation is invalidated
    db.indexes.invalidate("edges_relates", reason="a bulk load bypassed the journal")
    path = frontier_sql(TENANT, seed, 2, backend=db.csr)[2]
    assert (path, path.reason) == ("sql", HealthReason.STALE_GENERATION)
    assert "bulk load" in path.detail
    assert index.health(TENANT).usable is False

    # 5. a rebuild puts it back
    maintain(index, TENANT, ALWAYS, raise_on_failure=True)
    assert frontier_sql(TENANT, seed, 2, backend=db.csr)[2].reason is HealthReason.FRESH

    # 6. a load failure keeps the answers coming from SQL
    index.load_error = "the extension segfaulted on load"
    path = frontier_sql(TENANT, seed, 2, backend=db.csr)[2]
    assert (path, path.reason) == ("sql", HealthReason.LOAD_FAILURE)
    assert path.detail == "the extension segfaulted on load"
    assert db.recall_2hop_ids(seed) == oracle_recall(db, seed, limit=20)
    index.load_error = None

    # 7. a build announced but not finished
    index._announce_build(TENANT, None)
    try:
        assert index.health(TENANT).generation is not None  # the previous one is still published
        assert index.is_building(TENANT) is True
    finally:
        index._release_build(TENANT)
        index.abandon_builds(TENANT)


def test_a_stale_policy_keeps_the_index_usable_and_says_so(plain_db):
    """A generation due for a rebuild is still exact: the journal covers what the base misses."""
    db, index = plain_db
    ids = seed_graph(db, entities=15, edges=25, rng=random.Random(9))
    maintain(index, TENANT, ALWAYS, raise_on_failure=True)
    for _ in range(5):
        db.relate(*random.Random(4).sample(ids, 2))
    health = index.health(TENANT, policy=MaintenancePolicy(rebuild_after_rows=1))
    assert health.reason is HealthReason.STALE_GENERATION
    assert health.usable is True and health.pending_rows >= 1
    expansion = index.expand(TENANT, ids[0], 2)
    assert sorted(expansion.entity_ids) == oracle_frontier(db, ids[0], 2)


def test_an_unvalidated_generation_is_usable_and_flagged(plain_db):
    """``publish(force=True)`` is honest rather than unreadable: candidates only narrow."""
    db, index = plain_db
    ids = seed_graph(db, entities=10, edges=15, rng=random.Random(6))
    gen = index.build_next(TENANT)
    index.publish(gen, force=True)
    expansion = index.expand(TENANT, ids[0], 2)
    assert expansion.path.reason is HealthReason.UNVALIDATED
    assert expansion.path == "csr"
    assert sorted(expansion.entity_ids) == oracle_frontier(db, ids[0], 2)


# --------------------------------------------------------------------------- erasure


def test_a_hard_erasure_leaves_nothing_of_the_edge_in_any_generation(plain_db):
    """After a purge, a scan of every generation's storage finds neither the ids nor the pair."""
    db, index = plain_db
    ids = seed_graph(db, entities=14, edges=22, rng=random.Random(13))
    index.publish(index.validate(index.build_next(TENANT)).generation)
    older = index.current_generation(TENANT)
    # a second generation, so the purge has to reach an unpublished one too
    index.build_next(TENANT)
    victim = ids[3]
    doomed = {int(a) for a, b in live_edges(db) if victim in (a, b)}
    doomed |= {int(b) for a, b in live_edges(db) if victim in (a, b)}
    edge_ids = [
        int(r[0])
        for r in db.execute(
            "SELECT edge_id FROM edges_relates WHERE (src = ? OR dst = ?) AND tenant_id = ?",
            [victim, victim, TENANT],
        ).fetchall()
    ]
    assert edge_ids, "the victim had no edges, so this proves nothing"

    removed = hard_forget_entity(db, victim)
    assert removed == len(set(edge_ids))

    for table in generation_tables(db, index):
        cols = {
            r[0]
            for r in db.execute(
                "SELECT column_name FROM duckdb_columns() WHERE table_name = ?", [table]
            ).fetchall()
        }
        if "edge_id" in cols:
            left = db.execute(
                f"SELECT count(*) FROM {quote_ident(table)} WHERE edge_id = ANY(?::BIGINT[])",
                [edge_ids],
            ).fetchone()[0]
            assert left == 0, f"{table} still holds a purged edge id"
            touching = db.execute(
                f"SELECT count(*) FROM {quote_ident(table)} WHERE a = ? OR b = ?", [victim, victim]
            ).fetchone()[0]
            assert touching == 0, f"{table} still holds an edge of the purged entity"
        else:
            left = db.execute(
                f"SELECT count(*) FROM {quote_ident(table)} WHERE entity_id = ?", [victim]
            ).fetchone()[0]
            assert left == 0, f"{table} still maps the purged entity"
    journal = db.execute(
        "SELECT count(*) FROM anatid_index_journal WHERE index_name = 'csr' "
        "AND doc_id = ANY(?::BIGINT[])",
        [edge_ids],
    ).fetchone()[0]
    assert journal == 0, "the journal still names a purged edge"

    # and the answers stay right
    assert index.current_generation(TENANT).generation == older.generation
    for seed in ids[:5]:
        if seed == victim:
            continue
        expansion = index.expand(TENANT, seed, 2)
        assert sorted(expansion.entity_ids) == oracle_frontier(db, seed, 2)


def test_a_purged_edge_id_can_be_reused(plain_db):
    """The ordered journal represents insert -> purge -> insert of the same id.

    Two independent sets cannot: the id would be in both, and ``(base | delta) - tombstones``
    would drop a row the SQL path returns.
    """
    db, index = plain_db
    seed_graph(db, entities=6, edges=8, rng=random.Random(17))
    # three entities of their own, so the only thing connecting them is the edge under test
    for name in ("px", "py", "pz"):
        db.remember(f"a memory about {name}", entities=[name])
    a, b, c = (db.entity_id(n, create=False) for n in ("px", "py", "pz"))
    edge = db.relate(a, b, edge_id=999_001)
    index.publish(index.validate(index.build_next(TENANT)).generation)
    assert b in index.expand(TENANT, a, 1).entity_ids

    db.indexes.announce_erasure("edges_relates")
    at = _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)
    with db.indexes.lifecycle(), db.transaction():
        db.execute(
            "DELETE FROM edges_relates WHERE edge_id = ? AND tenant_id = ?", [edge.edge_id, TENANT]
        )
        db.indexes.erase("edges_relates", TENANT, [edge.edge_id], at=at, reason="forget_hard")
    assert b not in index.expand(TENANT, a, 1).entity_ids

    db.relate(a, c, edge_id=999_001)  # the same id, a different pair
    expansion = index.expand(TENANT, a, 1)
    assert sorted(expansion.entity_ids) == oracle_frontier(db, a, 1)
    assert c in expansion.entity_ids


# --------------------------------------------------------------------------- across handles


def test_a_second_handle_journals_for_an_index_it_has_no_code_for(tmp_path):
    """The definition lives in the FILE, so a write through any handle reaches the journal."""
    path = tmp_path / "shared.anatid"
    writer = _open(path, None)
    index = attach_csr_index(writer, strategy="sql")
    ids = seed_graph(writer, entities=10, edges=14, rng=random.Random(23))
    maintain(index, TENANT, ALWAYS, raise_on_failure=True)
    writer.close()

    first = _open(path, None)
    owner = attach_csr_index(first, strategy="sql")
    second = _open(path, None)  # holds no CsrIndex: only the persisted definition
    try:
        assert "csr" in second.indexes.definitions()
        assert second.indexes.get("csr") is not None
        before = owner.pending_count(TENANT)
        second.relate(ids[0], ids[9])
        assert owner.pending_count(TENANT) == before + 1
        expansion = owner.expand(TENANT, ids[0], 1)
        assert expansion.entity_ids is not None, expansion.path.explain()
        assert ids[9] in expansion.entity_ids
        assert sorted(expansion.entity_ids) == oracle_frontier(first, ids[0], 1)
    finally:
        second.close()
        first.close()


# --------------------------------------------------------------------------- registry surface


def test_the_index_registers_itself_in_the_file(plain_db):
    db, _index = plain_db
    definition = db.indexes.definitions()["csr"]
    assert definition.source_table == "edges_relates"
    assert definition.source_id_column == "edge_id"
    assert definition.per_tenant is True
    assert definition.delta_mode == "table"
    assert definition.supports_delta is True
    assert db.index_health()["csr"].reason is HealthReason.ABSENT
    seed_graph(db, entities=6, edges=8, rng=random.Random(31))
    reports = db.maintain_indexes(policy=ALWAYS)
    assert reports["csr"].action == "published"
    assert db.index_health()["csr"].reason is HealthReason.FRESH


# --------------------------------------------------------------------------- the spike oracle


@pytest.mark.slow
@pytest.mark.oracle
def test_the_index_matches_the_spike_reference_on_both_merges(
    tmp_path_factory, spike_common, spike_queries
):
    """200 spike queries through the index, zero mismatches against the pure-numpy reference.

    The same bar the SQL path and the 0.1 snapshot are held to in ``tests/test_core.py``, applied
    to the merged index: ``spike/bench/common.py``'s ``reference_r1`` computed straight from the
    Parquet files, ids and timestamps compared, on each merge strategy.
    """
    small = REPO_ROOT / "spike" / "data" / "small"
    if not (small / "edges_relates.parquet").is_file():
        pytest.skip(f"spike small dataset not present at {small}")
    extension = _extension_path()
    path = tmp_path_factory.mktemp("csr-index-oracle") / "oracle.anatid"
    kwargs = (
        {"use_csr_extension": True, "extension_path": extension, "require_extension": True}
        if extension
        else {}
    )
    try:
        db = Anatid.open(path, tenant=0, embedding_dim=64, fts=False, **kwargs)
    except ExtensionUnavailable as exc:
        pytest.skip(f"could not load {extension}: {exc}")
    with db:
        counts = db.load_parquet(small, rebuild_fts=False, build_csr=False)
        assert counts.get("memories") == 100_000, counts
        index = attach_csr_index(db, strategy="sql")
        tenants = [
            int(r[0])
            for r in db.execute(
                "SELECT DISTINCT tenant_id FROM edges_relates ORDER BY tenant_id"
            ).fetchall()
        ]
        for tenant in tenants:
            assert maintain(index, tenant, ALWAYS, raise_on_failure=True).action == "published"

        strategies = ["sql"] + (["extension"] if extension else [])
        for strategy in strategies:
            index.strategy = strategy
            expected_path = "csr" if strategy == "sql" else "extension"
            mismatches = []
            for qid in range(200):
                q = spike_queries[qid]
                tenant, seed = int(q["tenant_id"]), int(q["seed_entity_id"])
                used = frontier_sql(tenant, seed, 2, backend=db.csr)[2]
                assert used == expected_path, used.explain()
                got = db.recall_2hop_ids(seed, tenant=tenant, limit=spike_common.R1_LIMIT)
                reference = spike_common.reference_r1(
                    "small", tenant, seed, limit=spike_common.R1_LIMIT
                )
                ok, message = spike_common.compare_r1(got, reference, check_created_at=True)
                if not ok:
                    mismatches.append((qid, strategy, message))
            assert mismatches == [], (
                f"{len(mismatches)}/200 queries differ on the {strategy!r} merge: {mismatches[:3]}"
            )
