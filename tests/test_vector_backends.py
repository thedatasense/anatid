"""The vector backends: the exact oracle, and the DuckDB HNSW one built on the framework.

The promotion criterion in ``docs/design/derived-index-framework.md`` is what this file checks,
item by item: recall at k of at least 0.98 against brute force, immediate visibility of new
inserts and supersessions, filtered recall, and fallback with a reported reason on a corrupted
index.  The measured numbers are printed (run with ``-s``) as well as asserted, because a recall
figure that is only ever compared with a threshold tells nobody how much headroom there is.

Every test that needs the ``vss`` extension goes through the ``vss_ready`` fixture, which SKIPS
with the loader's own error message when the extension cannot be installed or loaded.  Nothing
here fakes an HNSW index: an environment without the extension reports that it did not run these
checks rather than passing them.
"""

from __future__ import annotations

import datetime as _dt
import logging
import statistics
import time
from pathlib import Path

import duckdb
import pytest

import anatid
from anatid import Anatid, HealthReason, MaintenancePolicy, maintain
from anatid import vector as V
from anatid.types import AsOf

REPO_ROOT = Path(__file__).resolve().parent.parent
SPIKE_SMALL = REPO_ROOT / "spike" / "data" / "small"

DIM = 8
T0 = _dt.datetime(2026, 1, 1, 0, 0, 0)
MINUTE = _dt.timedelta(minutes=1)

#: Queries and top-N for the oracle comparison on the spike dataset.
RECALL_QUERIES = 100
RECALL_K = 10
RECALL_FLOOR = 0.98

#: Every measurement this module took, in order.  Logged as it happens (run with
#: ``--log-cli-level=INFO`` to watch it) and kept so a failing assertion can quote it.
MEASUREMENTS: list[str] = []

log = logging.getLogger("anatid.tests.vector")


def note(text: str) -> None:
    MEASUREMENTS.append(text)
    log.info("%s", text)


def vec(*values: float) -> list[float]:
    """Pad or truncate to :data:`DIM`."""
    out = list(values) + [0.0] * DIM
    return [float(x) for x in out[:DIM]]


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def vss_ready() -> str:
    """The ``vss`` extension, or a skip carrying the reason it is unavailable."""
    con = duckdb.connect()
    try:
        error = V.load_vss(con)
    finally:
        con.close()
    if error:
        pytest.skip(f"DuckDB vss extension unavailable: {error}")
    return "duckdb_vss"


@pytest.fixture
def vdb(vss_ready):
    """A small on-disk database with the ``duckdb_vss`` backend attached but nothing built.

    On disk rather than in memory because ``hnsw_enable_experimental_persistence`` only matters
    for a file, and that is the configuration a user runs.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        with Anatid.open(Path(tmp) / "v.anatid", tenant=1, embedding_dim=DIM) as db:
            V.attach(db, backend="duckdb_vss")
            yield db


@pytest.fixture
def plain_db():
    """A database with no vector index at all: the default exact backend."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        with Anatid.open(Path(tmp) / "p.anatid", tenant=1, embedding_dim=DIM) as db:
            yield db


def seed(db, count: int = 24, *, tenant: int = 1, kind: str = "note") -> list[int]:
    """``count`` memories spread over a circle, so cosine order is not the insertion order."""
    import math

    out = []
    for i in range(count):
        angle = (i * 7 % count) / count * math.pi / 2
        out.append(
            db.remember(
                f"memory {i}",
                kind=kind if i % 2 else "fact",
                embedding=vec(math.cos(angle), math.sin(angle), i / 100.0),
                tenant=tenant,
                now=T0 + i * MINUTE,
            ).memory_id
        )
    return out


def publish(index, tenant: int):
    """Build, validate and publish one generation, failing loudly if validation does."""
    gen = index.build_next(tenant)
    report = index.validate(gen)
    assert report.ok, report.detail
    return index.publish(report.generation)


@pytest.fixture(scope="module")
def spike_vss(tmp_path_factory, vss_ready):
    """The spike ``small`` dataset with a published HNSW generation for every tenant.

    100,000 memories over 10 tenants, 9,500 currently visible each, 64 dimensions.  Module
    scoped: the load and the ten builds cost a few seconds and nothing in this file mutates it.
    """
    if not (SPIKE_SMALL / "memories.parquet").is_file():
        pytest.skip(f"spike small dataset not present at {SPIKE_SMALL}")
    path = tmp_path_factory.mktemp("vector") / "spike.anatid"
    db = Anatid.open(path, tenant=0, embedding_dim=64)
    started = time.perf_counter()
    counts = db.load_parquet(SPIKE_SMALL, rebuild_fts=False, build_csr=False)
    loaded = time.perf_counter() - started
    assert counts.get("memories") == 100_000, counts
    index = V.attach(db, backend="duckdb_vss")
    build_seconds = {}
    for tenant in range(10):
        started = time.perf_counter()
        publish(index, tenant)
        build_seconds[tenant] = time.perf_counter() - started
    total = sum(build_seconds.values())
    rows = db.connection.execute(
        "SELECT count(*) FROM memories WHERE tenant_id = 0 AND valid_to IS NULL AND tx_to IS NULL"
    ).fetchone()[0]
    note(
        f"spike small: loaded 100,000 memories in {loaded:.2f}s; built, validated and published "
        f"10 generations of {rows} rows in {total:.2f}s "
        f"({total / 10:.2f}s per tenant, validation included)"
    )
    try:
        yield db, index
    finally:
        db.close()


@pytest.fixture(scope="module")
def spike_probes(spike_vss):
    """The spike benchmark's query embeddings, by query id."""
    import importlib.util
    import sys

    common = REPO_ROOT / "spike" / "bench" / "common.py"
    if not common.is_file():
        pytest.skip(f"spike harness not present at {common}")
    spec = importlib.util.spec_from_file_location("anatid_spike_common_vector", common)
    module = importlib.util.module_from_spec(spec)
    sys.modules["anatid_spike_common_vector"] = module
    spec.loader.exec_module(module)
    return module.load_queries("small")


# --------------------------------------------------------------------------- backends


def test_the_default_backend_is_exact_and_costs_nothing_to_have(plain_db):
    """A database with no vector index answers exactly and says why, with no extra work."""
    ids = seed(plain_db)
    plan = V.resolve(plain_db.connection, tenant_id=1)
    assert plan.usable is False
    assert plan.reason is HealthReason.ABSENT
    assert "defines no vector index" in plan.detail
    assert plan.backend == "exact"

    got = V.search(plain_db.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=5)
    assert got.backend == "exact"
    assert got.approximate is False
    assert got.reason is HealthReason.ABSENT
    assert [m for m, _ in got.hits] == [
        m
        for m, _ in V.exact_search(
            plain_db.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=5
        )
    ]
    assert {m for m, _ in got.hits} <= set(ids)


def test_owned_hnsw_raises_rather_than_falling_back_silently(plain_db):
    """``owned_hnsw`` is named in the design and not implemented; that is said out loud."""
    with pytest.raises(NotImplementedError) as exc:
        V.attach(plain_db, backend="owned_hnsw")
    assert "owned_hnsw" in str(exc.value)
    assert "duckdb_vss" in str(exc.value)
    with pytest.raises(NotImplementedError):
        V.VectorIndex(plain_db, backend="owned_hnsw")
    with pytest.raises(ValueError):
        V.attach(plain_db, backend="faiss")
    assert V.attach(plain_db, backend="exact") is None
    assert V.index_of(plain_db) is None
    assert "vector" not in plain_db.indexes.definitions()


def test_attach_records_the_backend_in_the_file_not_on_the_handle(vdb):
    """A second handle journals for the vector index although it holds no vector code."""
    definition = vdb.indexes.definitions()["vector"]
    assert definition.enabled and definition.source_table == "memories"
    assert definition.params["backend"] == "duckdb_vss"
    assert definition.params["dim"] == DIM
    assert definition.params["ef_search"] == V.DEFAULT_EF_SEARCH

    index = V.index_of(vdb)
    assert isinstance(index, V.VectorIndex)
    seed(vdb, 12)
    publish(index, 1)

    with Anatid.open(vdb.path, tenant=1, embedding_dim=DIM, ensure=False) as other:
        assert V.index_of(other) is None  # no vector code on this handle
        new = other.remember("written elsewhere", embedding=vec(1, 0), now=T0 + 100 * MINUTE)
        assert new.memory_id in index.pending(1)
        # and the read on the ORIGINAL handle sees it, because the journal is in the file
        got = V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=5)
        assert got.backend == "duckdb_vss"
        assert new.memory_id in [m for m, _ in got.hits]


# --------------------------------------------------------------------------- lifecycle


def test_a_generation_is_built_validated_and_published(vdb):
    index = V.index_of(vdb)
    seed(vdb, 24)
    assert index.health(1).reason is HealthReason.ABSENT
    assert index.pending_count(1) == 24

    gen = index.build_next(1)
    assert gen.stats["rows"] == 24
    assert gen.stats["backend"] == "duckdb_vss"
    assert gen.stats["storage"] == "anatid_idx_vector_t1_g1"
    assert gen.watermark_id is not None
    with index.pin(1) as pin:
        assert pin.usable is False  # not published yet
        assert pin.reason is HealthReason.ABSENT

    report = index.validate(gen)
    assert report.ok, report.detail
    assert "match the oracle exactly" in report.detail

    published = index.publish(report.generation)
    assert published.published and published.validated
    health = index.health(1)
    assert health.reason is HealthReason.FRESH and health.usable
    assert health.pending_rows == 0 and health.base_rows == 24
    assert vdb.index_health()["vector"].reason is HealthReason.FRESH


def test_validation_fails_when_the_base_does_not_match_the_oracle(vdb):
    """The oracle check is a real comparison, not a row count that always agrees."""
    index = V.index_of(vdb)
    ids = seed(vdb, 24)
    gen = index.build_next(1)
    storage = index.storage(gen)
    vdb.execute(f"DELETE FROM {storage} WHERE memory_id = ?", [ids[0]])
    report = index.validate(gen)
    assert report.ok is False
    assert "1 missing" in report.detail
    assert index.generation(1, gen.generation).validated is False

    with pytest.raises(anatid.errors.IndexGenerationError):
        index.publish(gen)


def test_maintenance_builds_publishes_and_rebuilds_under_the_policy(vdb):
    index = V.index_of(vdb)
    seed(vdb, 24)
    policy = MaintenancePolicy(
        rebuild_after_rows=5, rebuild_after_ratio=None, rebuild_after_seconds=None
    )
    first = maintain(index, 1, policy)
    assert first.action == "published", first.reason
    assert first.after.reason is HealthReason.FRESH

    assert maintain(index, 1, policy).action == "none"

    for i in range(5):
        vdb.remember(f"later {i}", embedding=vec(0, 1), now=T0 + (200 + i) * MINUTE)
    due = index.health(1, policy=policy)
    assert due.reason is HealthReason.STALE_GENERATION and due.usable  # merges, so still usable
    second = maintain(index, 1, policy)
    assert second.action == "published"
    assert second.generation.generation == first.generation.generation + 1
    assert index.health(1, policy=policy).pending_rows == 0


# --------------------------------------------------------------------------- correctness


def test_a_memory_written_after_the_watermark_is_found_immediately(vdb):
    """The point of the journal: no rebuild between a write and the read that must see it."""
    index = V.index_of(vdb)
    seed(vdb, 24)
    publish(index, 1)

    fresh = vdb.remember("the newest thing", embedding=vec(1, 0, 0), now=T0 + 500 * MINUTE)
    got = V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0, 0), dim=DIM, topn=5)
    assert got.backend == "duckdb_vss"
    assert got.scanned == 1  # exactly one row scanned exactly
    scores = dict(got.hits)
    assert scores[fresh.memory_id] == pytest.approx(1.0)
    exact = V.exact_search(vdb.connection, tenant_id=1, embedding=vec(1, 0, 0), dim=DIM, topn=5)
    assert got.hits == exact  # same rows, same order, same float bits as the oracle

    # and inside the writing transaction, before the commit
    with vdb.transaction():
        inner = vdb.remember("inside", embedding=vec(1, 0, 0), now=T0 + 501 * MINUTE)
        during = V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0, 0), dim=DIM, topn=5)
        assert inner.memory_id in [m for m, _ in during.hits]


def test_a_row_inserted_by_raw_sql_is_still_found(vdb):
    """The journal covers the verbs; the generation watermark covers everything else.

    A raw ``INSERT`` through the connection journals nothing, so without the watermark half of
    the merge the exact scan would return the row and the approximate path would not.  Two read
    paths that disagree is the failure this framework exists to remove.
    """
    index = V.index_of(vdb)
    seed(vdb, 24)
    gen = publish(index, 1)
    assert index.pending_count(1) == 0

    raw_id = int(gen.watermark_id) + 1_000
    vdb.execute(
        "INSERT INTO memories (memory_id, tenant_id, content, kind, embedding, created_at, "
        "valid_from, tx_from, confidence, access_count, version) "
        f"VALUES (?, 1, 'raw insert', 'fact', ?::FLOAT[{DIM}], ?, ?, ?, 1.0, 0, 1)",
        [raw_id, V._literal(vec(1, 0, 0)), T0, T0, T0],
    )
    assert index.pending_count(1) == 0  # the journal never heard of it

    plan = V.resolve(vdb.connection, tenant_id=1)
    assert plan.usable is True
    assert raw_id in plan.pending
    got = V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0, 0), dim=DIM, topn=5)
    assert raw_id in [m for m, _ in got.hits]
    assert got.hits == V.exact_search(
        vdb.connection, tenant_id=1, embedding=vec(1, 0, 0), dim=DIM, topn=5
    )


def test_a_superseded_or_forgotten_memory_is_not_returned(vdb):
    index = V.index_of(vdb)
    ids = seed(vdb, 24)
    publish(index, 1)
    target = ids[0]
    before = V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=30)
    assert target in [m for m, _ in before.hits]

    vdb.supersede(target, "corrected", embedding=vec(0, 1), now=T0 + 600 * MINUTE)
    after = V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=30)
    assert target not in [m for m, _ in after.hits]
    assert target in index.tombstones(1)
    assert after.hits == V.exact_search(
        vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=30
    )

    forgotten = ids[1]
    vdb.forget(forgotten, now=T0 + 601 * MINUTE)
    gone = V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=30)
    assert forgotten not in [m for m, _ in gone.hits]
    assert gone.hits == V.exact_search(
        vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=30
    )


def test_the_scores_are_the_exact_cosine_not_the_index_distance(vdb):
    """The approximate structure chooses rows; it never contributes a number to the ranking."""
    index = V.index_of(vdb)
    seed(vdb, 24)
    publish(index, 1)
    probe = vec(0.6, 0.8)
    merged = V.search(vdb.connection, tenant_id=1, embedding=probe, dim=DIM, topn=24)
    exact = V.exact_search(vdb.connection, tenant_id=1, embedding=probe, dim=DIM, topn=24)
    assert merged.backend == "duckdb_vss"
    assert merged.hits == exact  # ids, order and float bits


def test_the_public_recall_returns_the_same_hits_with_and_without_the_index(vdb):
    """The end-to-end promise: turning the backend on changes speed, not answers."""
    index = V.index_of(vdb)
    seed(vdb, 24)
    before = vdb.recall(embedding=vec(0.6, 0.8), k=8)
    assert len(before) == 8
    publish(index, 1)
    after = vdb.recall(embedding=vec(0.6, 0.8), k=8)
    assert [h.memory.memory_id for h in after] == [h.memory.memory_id for h in before]
    assert [h.vector_score for h in after] == [h.vector_score for h in before]
    assert [h.vector_rank for h in after] == [h.vector_rank for h in before]
    assert after.arms == before.arms == ("vector",)


def test_another_tenants_rows_never_enter_the_answer(vdb):
    index = V.index_of(vdb)
    mine = seed(vdb, 12, tenant=1)
    theirs = seed(vdb, 12, tenant=2)
    publish(index, 1)
    publish(index, 2)

    ours = V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=20)
    assert {m for m, _ in ours.hits} <= set(mine)
    assert not {m for m, _ in ours.hits} & set(theirs)
    assert ours.hits == V.exact_search(
        vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=20
    )
    # a generation is per tenant, and each holds only its own rows
    for tenant, expected in ((1, mine), (2, theirs)):
        gen = index.current_generation(tenant)
        held = {
            int(r[0]) for r in vdb.execute(f"SELECT memory_id FROM {index.storage(gen)}").fetchall()
        }
        assert held == set(expected)


def test_a_kind_filter_is_applied_to_the_candidates(vdb):
    index = V.index_of(vdb)
    seed(vdb, 24)
    publish(index, 1)
    vdb.remember("late note", kind="note", embedding=vec(1, 0), now=T0 + 700 * MINUTE)
    for kinds in (["note"], ["fact"], ["note", "fact"], ["nothing"]):
        got = V.search(
            vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=30, kinds=kinds
        )
        exact = V.exact_search(
            vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=30, kinds=kinds
        )
        assert got.hits == exact, kinds
        rows = vdb.execute(
            "SELECT DISTINCT kind FROM memories WHERE memory_id IN "
            f"({','.join(str(int(m)) for m, _ in got.hits) or 'NULL'})"
        ).fetchall()
        assert {r[0] for r in rows} <= set(kinds)


# --------------------------------------------------------------------------- fallback


def test_as_of_forces_the_exact_path_and_says_so(vdb):
    index = V.index_of(vdb)
    ids = seed(vdb, 24)
    publish(index, 1)
    vdb.supersede(ids[0], "corrected", embedding=vec(0, 1), now=T0 + 800 * MINUTE)

    scope = AsOf(T0 + 700 * MINUTE, T0 + 700 * MINUTE)
    plan = V.resolve(vdb.connection, tenant_id=1, as_of=scope)
    assert plan.usable is False and plan.reason is HealthReason.HISTORICAL_QUERY
    assert "as_of" in plan.detail

    got = V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=30, as_of=scope)
    assert got.backend == "exact" and got.reason is HealthReason.HISTORICAL_QUERY
    # the historical answer still holds the memory that was corrected afterwards
    assert ids[0] in [m for m, _ in got.hits]
    assert got.hits == V.exact_search(
        vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=30, as_of=scope
    )
    assert index.health(1, as_of=scope).reason is HealthReason.HISTORICAL_QUERY


def test_a_plan_resolved_for_a_current_read_never_answers_a_historical_one(vdb):
    """A caller that resolves once and then asks for an as_of read still gets the exact path."""
    index = V.index_of(vdb)
    ids = seed(vdb, 24)
    publish(index, 1)
    plan = V.resolve(vdb.connection, tenant_id=1)
    assert plan.usable is True
    vdb.supersede(ids[0], "corrected", embedding=vec(0, 1), now=T0 + 900 * MINUTE)

    scope = AsOf(T0 + 800 * MINUTE, T0 + 800 * MINUTE)
    got = V.search(
        vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=30, as_of=scope, plan=plan
    )
    assert got.backend == "exact" and got.reason is HealthReason.HISTORICAL_QUERY
    assert got.hits == V.exact_search(
        vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=30, as_of=scope
    )
    assert ids[0] in [m for m, _ in got.hits]


def test_the_search_depth_scales_with_the_generation_and_never_falls_below_the_limit():
    """``ef_search`` is the depth at 100,000 rows; a smaller base searches proportionally less."""
    from anatid.derived import Generation

    def plan_of(rows):
        gen = Generation(
            index_name="vector",
            generation=1,
            tenant_id=1,
            built_at=T0,
            stats={"rows": rows},
        )
        return V.VectorPlan(
            usable=True, reason=HealthReason.FRESH, generation=gen, params={"ef_search": 800}
        )

    assert V.search_depth(plan_of(100_000), 40) == 800
    assert V.search_depth(plan_of(1_000_000), 40) == 800  # the configured value is the ceiling
    assert V.search_depth(plan_of(95_000), 40) == 779
    assert V.search_depth(plan_of(9_500), 40) == 246
    assert V.search_depth(plan_of(100), 40) == 40  # never below the candidates asked for
    assert V.search_depth(plan_of(9_500), 4000) == 4000  # the extension searches max(ef, LIMIT)
    # no recorded row count: the configured depth, unscaled
    assert (
        V.search_depth(V.VectorPlan(True, HealthReason.FRESH, params={"ef_search": 300}), 10) == 300
    )


def test_a_missing_storage_table_falls_back_to_exact_with_load_failure(vdb):
    """The storage is gone and the catalog still publishes the generation.

    The plan does not probe for the table: that probe cost 0.4 ms of a 2 ms query, which is
    more than the branch is worth when DuckDB reports the missing table itself and a failed
    statement does not abort the caller's transaction.  So the fallback happens at the query
    and carries DuckDB's own message.
    """
    index = V.index_of(vdb)
    seed(vdb, 24)
    gen = publish(index, 1)
    vdb.execute(f"DROP TABLE {index.storage(gen)}")

    got = V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=10)
    assert got.backend == "exact" and got.reason is HealthReason.LOAD_FAILURE
    assert index.storage(gen) in got.detail
    assert got.hits == V.exact_search(
        vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=10
    )
    # and the generation is still the published one, so maintenance can see the state
    assert index.current_generation(1).generation == gen.generation


def test_a_corrupted_storage_table_falls_back_without_failing_the_query(vdb, caplog):
    """The table is there and unusable: wrong dimension, no HNSW index, garbage rows.

    This is the crash-recovery shape the design asks about.  The query must still be answered,
    correctly, and the reason must name the failure rather than being "the index was not used".
    """
    index = V.index_of(vdb)
    seed(vdb, 24)
    gen = publish(index, 1)
    storage = index.storage(gen)
    vdb.execute(f"DROP TABLE {storage}")
    vdb.execute(f"CREATE TABLE {storage} (memory_id BIGINT, embedding FLOAT[3])")
    vdb.execute(f"INSERT INTO {storage} VALUES (1, [0.0, 0.0, 1.0])")

    plan = V.resolve(vdb.connection, tenant_id=1)
    assert plan.usable is True  # the catalog and the table both look fine from outside
    with caplog.at_level("WARNING", logger="anatid.vector"):
        got = V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=10)
    assert got.backend == "exact"
    assert got.reason is HealthReason.LOAD_FAILURE
    assert "could not be searched" in got.detail
    assert any("answering exactly" in r.message for r in caplog.records)
    assert got.hits == V.exact_search(
        vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=10
    )
    # and the public read path is unaffected
    hits = vdb.recall(embedding=vec(1, 0), k=3)
    assert len(hits) == 3


def test_a_base_that_lost_rows_is_detected_and_the_exact_scan_answers(vdb, caplog):
    """The damage that does NOT raise: a base that is present, queryable and incomplete.

    A generation's table is frozen after its build (only ``_erase`` touches it, and that
    invalidates the generation), so a row count that has moved means something outside anatid
    wrote to it.  The count comes back in the read's own statement, so noticing costs nothing.
    """
    index = V.index_of(vdb)
    seed(vdb, 24)
    gen = publish(index, 1)
    storage = index.storage(gen)
    complete = V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=5)
    assert complete.backend == "duckdb_vss"

    vdb.execute(f"DELETE FROM {storage} WHERE memory_id % 3 = 0")
    with caplog.at_level("WARNING", logger="anatid.vector"):
        got = V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=5)
    assert got.backend == "exact" and got.reason is HealthReason.DAMAGED_BASE
    assert "its build recorded" in got.detail
    assert got.hits == V.exact_search(
        vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=5
    )
    assert got.hits == complete.hits
    assert index.health(1).reason is HealthReason.DAMAGED_BASE
    assert maintain(index, 1, MaintenancePolicy()).action == "published"
    assert (
        V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=5).backend
        == "duckdb_vss"
    )


def test_an_hnsw_structure_that_returns_too_few_candidates_is_not_trusted(vdb):
    """The other half of the same invariant, and the one a row count cannot see.

    The candidate subquery carries no predicate, so a healthy approximate scan returns exactly
    ``min(limit, base_rows)`` rows.  A persisted HNSW graph that lost its edges answers with a
    fraction of them and reports no error, which is the crash-recovery hazard DuckDB documents
    for experimental HNSW persistence.  Simulated here through the arm's own accounting, so the
    test does not depend on being able to corrupt a graph on this machine.
    """
    index = V.index_of(vdb)
    seed(vdb, 24)
    gen = publish(index, 1)
    plan = V.resolve(vdb.connection, tenant_id=1)
    assert plan.usable

    healthy = V._cold_arm(
        vdb.connection,
        plan,
        tenant_id=1,
        literal=V._literal(vec(1, 0)),
        dim=DIM,
        topn=5,
        kinds=None,
    )
    recorded = int(gen.stats["rows"])
    assert healthy.base_rows == recorded == 24
    assert healthy.candidates == min(healthy.limit, recorded)
    assert healthy.damage(recorded) is None

    starved = V.ColdArm(
        hits=healthy.hits[:1],
        candidates=healthy.candidates // 2,
        base_rows=healthy.base_rows,
        limit=healthy.limit,
    )
    assert "HNSW structure is damaged" in (starved.damage(recorded) or "")
    grown = V.ColdArm(hits=[], candidates=0, base_rows=0, limit=healthy.limit)
    assert "its build recorded" in (grown.damage(recorded) or "")


def test_a_load_failure_on_the_extension_is_reported_and_answered_exactly(vdb):
    index = V.index_of(vdb)
    seed(vdb, 24)
    publish(index, 1)
    index.load_error = "the vss extension could not be loaded: simulated"

    plan = V.resolve(vdb.connection, tenant_id=1, index=index)
    assert plan.usable is False and plan.reason is HealthReason.LOAD_FAILURE
    assert "simulated" in plan.detail
    got = V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=10, index=index)
    assert got.backend == "exact" and got.reason is HealthReason.LOAD_FAILURE
    assert index.health(1).reason is HealthReason.LOAD_FAILURE
    assert maintain(index, 1).action == "skipped"


def test_an_unvalidated_generation_is_usable_and_flagged(vdb):
    """``publish(force=True)`` is readable and reported as never having met the oracle."""
    index = V.index_of(vdb)
    seed(vdb, 24)
    gen = index.build_next(1)
    index.publish(gen, force=True)
    plan = V.resolve(vdb.connection, tenant_id=1)
    assert plan.usable is True and plan.reason is HealthReason.UNVALIDATED
    got = V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=10)
    assert got.backend == "duckdb_vss" and got.reason is HealthReason.UNVALIDATED
    assert got.hits == V.exact_search(
        vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=10
    )


def test_a_rebuild_in_progress_and_an_invalidated_generation_report_their_reasons(vdb):
    index = V.index_of(vdb)
    seed(vdb, 24)
    # announced but not finished: another handle's build, seen through the catalog
    vdb.execute(
        "INSERT INTO anatid_index_generations (index_name, generation, tenant_id, built_at, "
        "validated, published, published_unvalidated, stats, notes) "
        "VALUES ('vector', 99, 1, ?, FALSE, FALSE, FALSE, '{}', 'building')",
        [T0],
    )
    plan = V.resolve(vdb.connection, tenant_id=1)
    assert plan.reason is HealthReason.REBUILD_IN_PROGRESS and plan.usable is False
    vdb.execute("DELETE FROM anatid_index_generations WHERE generation = 99")

    gen = publish(index, 1)
    vdb.indexes.invalidate("memories", reason="a bulk load bypassed the journal")
    stale = V.resolve(vdb.connection, tenant_id=1)
    assert stale.usable is False and stale.reason is HealthReason.STALE_GENERATION
    assert "bulk load" in stale.detail
    got = V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=10)
    assert got.backend == "exact" and got.reason is HealthReason.STALE_GENERATION
    assert got.hits == V.exact_search(
        vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=10
    )
    assert gen.generation == 1


def test_the_merge_cap_is_a_fraction_of_the_base_not_a_flat_number():
    """Merging is only worth doing while it is cheaper than the scan it replaces."""
    assert V.merge_cap(None) == V.MERGE_FLOOR  # no recorded base
    assert V.merge_cap(0) == V.MERGE_FLOOR
    assert V.merge_cap(10_000) == V.MERGE_FLOOR  # 5% of 10,000 is under the floor
    assert V.merge_cap(95_000) == int(95_000 * V.MERGE_RATIO)
    assert V.merge_cap(100_000_000) == V.MAX_MERGED_JOURNAL_ROWS  # the hard ceiling


def test_too_much_journal_falls_back_rather_than_merging_a_huge_list(vdb, monkeypatch):
    index = V.index_of(vdb)
    seed(vdb, 24)
    publish(index, 1)
    monkeypatch.setattr(V, "MAX_MERGED_JOURNAL_ROWS", 2)
    assert V.merge_cap(24) == 2
    for i in range(3):
        vdb.remember(f"overflow {i}", embedding=vec(0, 1), now=T0 + (900 + i) * MINUTE)
    plan = V.resolve(vdb.connection, tenant_id=1)
    assert plan.usable is False and plan.reason is HealthReason.STALE_GENERATION
    assert "maintain_indexes" in plan.detail
    got = V.search(vdb.connection, tenant_id=1, embedding=vec(0, 1), dim=DIM, topn=5)
    assert got.backend == "exact"
    assert got.hits == V.exact_search(
        vdb.connection, tenant_id=1, embedding=vec(0, 1), dim=DIM, topn=5
    )


# --------------------------------------------------------------------------- erasure


def test_a_hard_forget_leaves_no_copy_in_the_generation(vdb):
    """P0-4: scan every generation's storage after a purge.

    A ``DELETE`` takes the row out of the table, but DuckDB's HNSW keeps the vector inside the
    graph, so the index is dropped too and the generation is taken out of service until it is
    rebuilt.  The receipt says so.
    """
    index = V.index_of(vdb)
    ids = seed(vdb, 24)
    gen = publish(index, 1)
    storage = index.storage(gen)
    target = ids[3]
    assert (
        vdb.execute(f"SELECT count(*) FROM {storage} WHERE memory_id = ?", [target]).fetchone()[0]
        == 1
    )

    receipt = vdb.forget(target, hard=True, now=T0 + 1000 * MINUTE)
    assert receipt.hard is True
    assert receipt.invalidated_generations >= 1

    for row in vdb.execute(
        "SELECT table_name FROM duckdb_tables() WHERE table_name LIKE 'anatid_idx_vector%'"
    ).fetchall():
        left = vdb.execute(
            f"SELECT count(*) FROM {row[0]} WHERE memory_id = ?", [target]
        ).fetchone()[0]
        assert left == 0, f"{target} survives in {row[0]}"
    hnsw = vdb.execute(
        "SELECT count(*) FROM duckdb_indexes() WHERE index_name LIKE 'anatid_hnsw_vector%'"
    ).fetchone()[0]
    assert hnsw == 0

    plan = V.resolve(vdb.connection, tenant_id=1)
    assert plan.usable is False and plan.reason is HealthReason.STALE_GENERATION
    got = V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=30)
    assert got.backend == "exact"
    assert target not in [m for m, _ in got.hits]

    # the next maintenance builds a clean one, and it validates
    report = maintain(index, 1)
    assert report.action == "published", report.reason
    assert V.resolve(vdb.connection, tenant_id=1).usable is True


def test_dropping_the_index_takes_the_generation_storage_with_it(vdb):
    index = V.index_of(vdb)
    seed(vdb, 12)
    publish(index, 1)
    assert (
        vdb.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name LIKE 'anatid_idx_vector%'"
        ).fetchone()[0]
        == 1
    )
    V.detach(vdb, retire=True)
    assert (
        vdb.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name LIKE 'anatid_idx_vector%'"
        ).fetchone()[0]
        == 0
    )
    assert V.index_of(vdb) is None
    got = V.search(vdb.connection, tenant_id=1, embedding=vec(1, 0), dim=DIM, topn=5)
    assert got.backend == "exact"


# --------------------------------------------------------------------------- the ceiling


def test_a_usable_generation_lifts_the_brute_force_ceiling(vdb, monkeypatch):
    """The ceiling counts what the arm will actually scan, so the index buys headroom."""
    from anatid import recall as recall_mod

    index = V.index_of(vdb)
    seed(vdb, 24)
    monkeypatch.setattr(recall_mod, "BRUTE_FORCE_CEILING", 5)
    with pytest.raises(anatid.errors.BruteForceCeilingError) as exc:
        vdb.recall(embedding=vec(1, 0))
    assert "duckdb_vss" in str(exc.value) or "anatid.vector.attach" in str(exc.value)
    assert exc.value.rows == 24

    publish(index, 1)
    hits = vdb.recall(embedding=vec(1, 0), k=3)  # no longer refused
    assert len(hits) == 3
    assert recall_mod.vector_scan_rows(vdb.connection, tenant_id=1) == 0

    for i in range(6):
        vdb.remember(f"pending {i}", embedding=vec(0, 1), now=T0 + (1100 + i) * MINUTE)
    assert recall_mod.vector_scan_rows(vdb.connection, tenant_id=1) == 6
    with pytest.raises(anatid.errors.BruteForceCeilingError):
        vdb.recall(embedding=vec(1, 0))


# --------------------------------------------------------------------------- the oracle


@pytest.mark.slow
@pytest.mark.oracle
@pytest.mark.parametrize("k", [10, 50])
@pytest.mark.parametrize("kinds", [None, ["semantic", "preference"]])
def test_recall_at_k_against_the_exact_oracle_on_the_spike_dataset(
    spike_vss, spike_probes, k, kinds
):
    """The promotion criterion: recall at k of at least 0.98 against brute force.

    100 spike queries over 9,500 visible memories per tenant, 64 dimensions, compared arm to
    arm: the same tenant, the same top-N, the same visibility, the same kind filter.  The
    queries span all ten tenants, so this also checks that the per-tenant generations are the
    ones being read, and ``k=50`` is the size the arm actually runs at
    (:data:`anatid.recall.DEFAULT_CANDIDATES`).
    """
    db, _index = spike_vss
    con = db.connection
    recalls: list[float] = []
    ann_ms: list[float] = []
    exact_ms: list[float] = []
    for qid in range(RECALL_QUERIES):
        probe = spike_probes[qid]
        tenant = int(probe["tenant_id"])
        embedding = list(probe["query_embedding"])
        started = time.perf_counter()
        got = V.search(con, tenant_id=tenant, embedding=embedding, dim=64, topn=k, kinds=kinds)
        ann_ms.append((time.perf_counter() - started) * 1000)
        assert got.backend == "duckdb_vss", got.detail
        started = time.perf_counter()
        truth = V.exact_search(
            con, tenant_id=tenant, embedding=embedding, dim=64, topn=k, kinds=kinds
        )
        exact_ms.append((time.perf_counter() - started) * 1000)
        assert len(truth) == k
        hit = {m for m, _ in got.hits} & {m for m, _ in truth}
        recalls.append(len(hit) / k)
        # every score the approximate path returns is the exact one for that row
        scores = dict(truth)
        for mid, score in got.hits:
            if mid in scores:
                assert score == scores[mid]

    mean = statistics.mean(recalls)
    label = "unfiltered" if kinds is None else f"kinds={kinds}"
    note(
        f"recall@{k} ({label}) duckdb_vss vs exact over {RECALL_QUERIES} spike queries at "
        f"9,500 rows/tenant: mean {mean:.4f}, min {min(recalls):.2f}, "
        f"{sum(1 for r in recalls if r == 1.0)} of {RECALL_QUERIES} perfect; "
        f"p50 {statistics.median(ann_ms):.2f} ms vs exact {statistics.median(exact_ms):.2f} ms "
        f"(ef_search={V.DEFAULT_EF_SEARCH}, overfetch={V.DEFAULT_OVERFETCH}, "
        f"M={V.DEFAULT_M}, ef_construction={V.DEFAULT_EF_CONSTRUCTION})"
    )
    assert mean >= RECALL_FLOOR, f"recall@{k} ({label}) was {mean:.4f}, floor is {RECALL_FLOOR}"


@pytest.mark.slow
def test_recall_and_latency_with_every_row_in_one_tenant(tmp_path, vss_ready, spike_probes):
    """The 100k-per-tenant point the design asks for, where the exact scan starts to hurt."""
    path = tmp_path / "one.anatid"
    db = Anatid.open(path, tenant=0, embedding_dim=64)
    try:
        db.execute(
            "INSERT INTO memories (memory_id, tenant_id, content, kind, embedding, created_at, "
            "valid_from, valid_to, tx_from, tx_to, writer, episode_id, confidence, "
            "access_count, last_access_at, version) "
            "SELECT memory_id, 0, content, kind, embedding, created_at, valid_from, valid_to, "
            "tx_from, tx_to, writer, episode_id, confidence, 0, NULL, 1 "
            f"FROM read_parquet('{(SPIKE_SMALL / 'memories.parquet').as_posix()}')"
        )
        rows = db.execute(
            "SELECT count(*) FROM memories WHERE tenant_id = 0 AND valid_to IS NULL "
            "AND tx_to IS NULL"
        ).fetchone()[0]
        index = V.attach(db, backend="duckdb_vss")
        started = time.perf_counter()
        gen = index.build_next(0)
        build = time.perf_counter() - started
        started = time.perf_counter()
        report = index.validate(gen)
        validate = time.perf_counter() - started
        assert report.ok, report.detail
        index.publish(report.generation)

        con = db.connection
        recalls, ann_ms, exact_ms = [], [], []
        for qid in range(RECALL_QUERIES):
            embedding = list(spike_probes[qid]["query_embedding"])
            started = time.perf_counter()
            got = V.search(con, tenant_id=0, embedding=embedding, dim=64, topn=RECALL_K)
            ann_ms.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            truth = V.exact_search(con, tenant_id=0, embedding=embedding, dim=64, topn=RECALL_K)
            exact_ms.append((time.perf_counter() - started) * 1000)
            assert got.backend == "duckdb_vss"
            recalls.append(len({m for m, _ in got.hits} & {m for m, _ in truth}) / float(RECALL_K))
        mean = statistics.mean(recalls)
        note(
            f"{rows} rows in ONE tenant: build {build:.2f}s (HNSW "
            f"{gen.stats['index_seconds']:.2f}s of it), validate {validate:.2f}s, "
            f"recall@{RECALL_K} {mean:.4f} (min {min(recalls):.2f}), "
            f"duckdb_vss p50 {statistics.median(ann_ms):.2f} ms vs exact p50 "
            f"{statistics.median(exact_ms):.2f} ms "
            f"({statistics.median(exact_ms) / statistics.median(ann_ms):.1f}x)"
        )
        assert mean >= RECALL_FLOOR, f"recall@{RECALL_K} was {mean:.4f}"
        assert statistics.median(ann_ms) < statistics.median(exact_ms)
    finally:
        db.close()


@pytest.mark.slow
def test_the_spike_generations_validate_and_report_their_health(spike_vss):
    db, index = spike_vss
    for tenant in range(10):
        health = index.health(tenant)
        assert health.usable and health.reason is HealthReason.FRESH, health.detail
        assert health.base_rows == 9500
        assert health.pending_rows == 0 and health.tombstone_rows == 0
        gen = index.current_generation(tenant)
        assert gen.validated and gen.stats["rows"] == 9500
    assert db.index_health(tenant=0)["vector"].usable
