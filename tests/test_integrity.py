"""Defect 2 (the entity-creation race) and defect 5 (no input validation, no health check).

What is being proved here, in the reviewer's own terms:

* 100 concurrent ``remember()`` calls naming one new entity produced **four** entity rows in
  0.1.0.  ``test_one_hundred_racing_writers_create_exactly_one_entity`` runs that race for real
  -- 100 threads, one barrier, the public verb -- and requires exactly one row.
* ``confidence=-1``, ``confidence=2`` and NaN embeddings were all accepted and written.  Each
  now raises a typed :class:`anatid.errors.ValidationError` at the verb boundary, *before* any
  statement runs, and the tests assert that nothing was written.
* a simulated clock rollback made :func:`anatid.ids.new_id` repeat after 1,024 ids.  The
  rollback is simulated here the same way and the ids are required to stay unique and
  increasing.
* nothing could tell you a file had gone bad.  ``db.doctor()`` now can, and every fault it
  reports is seeded here by hand first.

The tests that need real concurrency use the on-disk ``file_db`` fixture and their own
``cursor()`` per thread, which is anatid's documented threading model.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import math
import threading

import duckdb
import pytest

from anatid import Anatid
from anatid import schema as S
from anatid.errors import (
    ConflictError,
    DuplicateIdError,
    EmbeddingDimensionError,
    EmbeddingValueError,
    IntegrityError,
    NotFoundError,
    RangeError,
    ValidationError,
)
from anatid.ids import EPOCH_MS, IdAllocator
from anatid.types import Severity

from conftest import DIM, T0, vec

NAME = "Ada Lovelace"


# ==================================================================== defect 2: the race


def test_one_hundred_racing_writers_create_exactly_one_entity(file_db):
    """The reviewer's reproduction, through the public verb.

    100 threads, released together, each calling ``remember(entities=["Ada Lovelace"])`` for a
    name that does not exist yet.  0.1.0's unguarded SELECT-then-INSERT produced four entity
    rows; the graph then had four "Ada Lovelace" nodes and no query could see all of her
    memories at once.

    Two things have to hold for one row to come out, and both are asserted:

    * the ``UNIQUE (tenant_id, entity_key)`` index makes a second row unrepresentable, and
    * the writer that loses re-runs its whole transaction and reads the winner -- so all 100
      memories point at the *same* entity, rather than 99 of them failing.
    """
    started = threading.Barrier(100)
    failures: list[BaseException] = []
    entity_ids: list[int] = []
    lock = threading.Lock()

    def writer(i: int) -> None:
        started.wait()
        try:
            m = file_db.remember(f"memory {i}", entities=[NAME], now=T0)
            eid = file_db.entity_id(NAME)
        except BaseException as exc:  # noqa: BLE001 - reported, then asserted on
            with lock:
                failures.append(exc)
            return
        with lock:
            entity_ids.append(eid)
        assert m.memory_id > 0

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert failures == [], f"{len(failures)} writer(s) failed, first: {failures[0]!r}"
    con = file_db.connection
    rows = con.execute("SELECT entity_id, name FROM entities WHERE tenant_id = 1").fetchall()
    assert len(rows) == 1, rows
    assert rows[0][1] == NAME
    # every writer came away with the same id, and every memory is ABOUT that one entity
    assert set(entity_ids) == {rows[0][0]}
    assert con.execute("SELECT count(*) FROM memories WHERE tenant_id = 1").fetchone()[0] == 100
    assert (
        con.execute("SELECT count(DISTINCT dst) FROM edges_about WHERE tenant_id = 1").fetchone()[0]
        == 1
    )
    assert con.execute("SELECT count(*) FROM edges_about WHERE tenant_id = 1").fetchone()[0] == 100
    assert file_db.doctor().find("duplicate_entity_names") is None


def test_one_hundred_racing_upserts_create_exactly_one_entity(file_db):
    """Same race through ``upsert_entity``, which returns the row rather than an id."""
    started = threading.Barrier(50)
    got: list[int] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def writer() -> None:
        started.wait()
        try:
            e = file_db.upsert_entity("Grace Hopper", kind="person", now=T0)
        except BaseException as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)
            return
        with lock:
            got.append(e.entity_id)

    threads = [threading.Thread(target=writer) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"first failure: {errors[0]!r}"
    assert len(set(got)) == 1
    assert (
        file_db.connection.execute("SELECT count(*) FROM entities WHERE tenant_id = 1").fetchone()[
            0
        ]
        == 1
    )


def test_names_that_canonicalise_together_are_one_entity(db):
    """Case and whitespace are not identity.

    ``"Ada Lovelace"``, ``"ada lovelace"`` and ``"  ADA   Lovelace "`` are one person.  v2 made
    them three entities in one tenant -- the same fracture as the race, reached by one writer
    being inconsistent instead of two writers colliding.
    """
    a = db.entity_id(NAME, create=True, now=T0)
    b = db.entity_id("ada lovelace", create=True, now=T0)
    c = db.entity_id("  ADA   Lovelace ", create=True, now=T0)
    assert a == b == c
    assert db.connection.execute("SELECT count(*) FROM entities").fetchone()[0] == 1
    # the row keeps the name as first written; only the KEY is canonical
    assert db.get_entity(a).name == NAME
    assert db.get_entity("ada    LOVELACE").entity_id == a


def test_one_remember_naming_a_name_two_ways_writes_one_entity(db):
    """The same collision inside a single statement list -- own writes are visible, so the
    second lookup finds the first insert and no constraint is ever reached."""
    m = db.remember("she wrote the first program", entities=[NAME, "ada lovelace"], now=T0)
    assert db.connection.execute("SELECT count(*) FROM entities").fetchone()[0] == 1
    about = db.entities_of(m.memory_id)
    assert {e.entity_id for e in about} == {db.entity_id(NAME)}


def test_the_entity_lookup_spans_history_so_a_forgotten_name_is_still_that_entity(db):
    """``entity_id`` no longer filters on ``valid_to IS NULL``.

    It cannot: the uniqueness constraint spans history, so a soft-closed entity row is still
    the only row that name can have.  v2's validity filter would miss it and then hit the
    constraint on the insert that followed.
    """
    eid = db.entity_id(NAME, create=True, now=T0)
    db.execute("UPDATE entities SET valid_to = ? WHERE entity_id = ?", [T0, eid])
    assert db.entity_id(NAME, create=True, now=T0) == eid
    assert db.connection.execute("SELECT count(*) FROM entities").fetchone()[0] == 1


def _rival_writer(db, name: str, entity_id: int):
    """Commit ``name`` from a second connection, i.e. from outside the caller's transaction."""
    rival = db._root.cursor()
    rival.execute(
        "INSERT INTO entities (entity_id, tenant_id, kind, name, valid_from, tx_from, "
        "confidence) VALUES (?, 1, NULL, ?, ?, ?, 1.0)",
        [entity_id, name, T0, T0],
    )
    rival.close()


def test_a_writer_that_loses_the_race_re_runs_and_reads_the_winner(file_db, monkeypatch):
    """The race, made deterministic, so the *recovery* is what is under test.

    A rival commits the entity from another connection in the window between this writer's
    lookup (which found nothing) and its COMMIT.  DuckDB does not notice at INSERT time -- the
    rival's row is not in this transaction's snapshot -- so the whole transaction fails at
    COMMIT with ``Failed to commit: PRIMARY KEY or UNIQUE constraint violation``.  There is no
    recovering inside it; the verb re-runs, and the second attempt reads the rival's row.
    """
    real_execute = file_db.execute
    fired: list[int] = []

    def sneaky(sql, params=None, *, con=None):
        if sql.startswith("INSERT INTO entities") and not fired:
            fired.append(1)
            _rival_writer(file_db, NAME, 777)
        return real_execute(sql, params, con=con)

    monkeypatch.setattr(file_db, "execute", sneaky)
    m = file_db.remember("lost the race", entities=[NAME], now=T0)

    assert fired == [1], "the rival never got in; the test proved nothing"
    monkeypatch.undo()
    con = file_db.connection
    assert con.execute("SELECT entity_id FROM entities").fetchall() == [(777,)]
    assert [
        r[0]
        for r in con.execute("SELECT dst FROM edges_about WHERE src = ?", [m.memory_id]).fetchall()
    ] == [777]
    # the first attempt's rows are not half-committed anywhere
    assert con.execute("SELECT count(*) FROM memories").fetchone()[0] == 1


def test_inside_a_caller_transaction_the_race_is_the_callers_to_retry(file_db, monkeypatch):
    """anatid will not silently re-run a transaction the caller opened.

    Only the caller knows what else went into it, so the failure propagates and they decide.
    (Re-running it here would replay their other statements too.)  It propagates as the
    documented retryable :class:`ConflictError` -- ``retryable`` is what a caller's retry loop
    keys on -- carrying DuckDB's constraint exception as ``cause``.
    """
    real_execute = file_db.execute
    fired: list[int] = []

    def sneaky(sql, params=None, *, con=None):
        if sql.startswith("INSERT INTO entities") and not fired:
            fired.append(1)
            _rival_writer(file_db, NAME, 778)
        return real_execute(sql, params, con=con)

    monkeypatch.setattr(file_db, "execute", sneaky)
    with pytest.raises(ConflictError) as exc:
        with file_db.transaction():
            assert file_db.in_transaction is True
            file_db.remember("inside the caller's transaction", entities=[NAME], now=T0)
    assert exc.value.retryable is True
    assert isinstance(exc.value.cause, duckdb.Error)
    assert "re-run" in str(exc.value)

    assert fired == [1]
    monkeypatch.undo()
    assert file_db.in_transaction is False
    con = file_db.connection
    assert con.execute("SELECT entity_id FROM entities").fetchall() == [(778,)]
    assert con.execute("SELECT count(*) FROM memories").fetchone()[0] == 0


def test_in_transaction_tracks_the_nesting(db):
    assert db.in_transaction is False
    with db.transaction():
        assert db.in_transaction is True
        with db.transaction():
            assert db.in_transaction is True
        assert db.in_transaction is True
    assert db.in_transaction is False


def test_entity_id_without_create_still_raises_not_found(db):
    with pytest.raises(NotFoundError):
        db.entity_id("nobody here", create=False)


# ==================================================================== defect 5: validation

NAN = float("nan")
INF = float("inf")


def _nothing_was_written(db) -> None:
    """Every validation check runs before any statement does, so the file is untouched."""
    con = db.connection
    for table in (
        "memories",
        "entities",
        "episodes",
        "edges_about",
        "edges_relates",
        "anatid_audit",
    ):
        assert con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0, table


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_a_non_finite_embedding_is_refused(db, bad):
    """The reviewer's third accepted-garbage case.

    DuckDB stores NaN in a ``FLOAT[N]`` without a murmur, and then
    ``array_cosine_similarity`` returns NaN for that row against *every* query vector -- so the
    row does not score badly, it scores unorderably, and can displace real hits from the
    candidate list depending on how the sort breaks ties.
    """
    v = vec(1.0, 2.0)
    v[3] = bad
    with pytest.raises(EmbeddingValueError) as exc:
        db.remember("poisoned", embedding=v, now=T0)
    assert exc.value.index == 3
    assert isinstance(exc.value, ValidationError) and isinstance(exc.value, ValueError)
    _nothing_was_written(db)


def test_a_non_finite_query_embedding_is_refused_too(db):
    db.remember("fine", embedding=vec(1.0), now=T0)
    with pytest.raises(EmbeddingValueError):
        db.recall("fine", embedding=[NAN] * DIM)


def test_a_wrong_length_embedding_is_refused(db):
    with pytest.raises(EmbeddingDimensionError) as exc:
        db.remember("too short", embedding=[0.1, 0.2], now=T0)
    assert (exc.value.expected, exc.value.got) == (DIM, 2)
    assert isinstance(exc.value, ValidationError)
    _nothing_was_written(db)


@pytest.mark.parametrize("bad", [-1.0, 2.0, -0.0001, 1.0001, NAN, INF])
def test_confidence_outside_the_unit_interval_is_refused(db, bad):
    """``confidence=-1`` and ``confidence=2`` were both accepted and written by 0.1.0."""
    with pytest.raises(ValidationError) as exc:
        db.remember("badly believed", confidence=bad, now=T0)
    if math.isfinite(bad):
        assert isinstance(exc.value, RangeError)
        assert (exc.value.low, exc.value.high) == (0.0, 1.0)
        assert exc.value.field == "confidence"
    _nothing_was_written(db)


@pytest.mark.parametrize("good", [0.0, 0.5, 1.0])
def test_the_endpoints_of_the_unit_interval_are_accepted(db, good):
    m = db.remember("believed exactly this much", confidence=good, now=T0)
    assert db.get(m.memory_id).confidence == pytest.approx(good)


@pytest.mark.parametrize("bad", [-1.0, 2.0, NAN])
def test_weight_outside_the_unit_interval_is_refused(db, bad):
    with pytest.raises(ValidationError):
        db.remember("heavy", entities=["Ada"], weight=bad, now=T0)
    _nothing_was_written(db)


def test_confidence_is_checked_on_supersede_relate_and_reinforce(db):
    m = db.remember("first", now=T0)
    with pytest.raises(RangeError):
        db.supersede(m.memory_id, "second", confidence=1.5, now=T0)
    with pytest.raises(RangeError):
        db.relate("Ada", "coffee", confidence=-0.5, now=T0)
    with pytest.raises(RangeError):
        db.reinforce(m.memory_id, confidence=99.0, now=T0)
    # none of them wrote
    con = db.connection
    assert con.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM entities").fetchone()[0] == 0
    assert con.execute("SELECT confidence FROM memories").fetchone()[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs", [{"k": 0}, {"k": -1}, {"candidates": 0}, {"rrf_k": 0}, {"hops": -1}]
)
def test_non_positive_recall_arguments_are_refused(db, kwargs):
    """``k=0`` silently returned nothing and ``LIMIT -1`` failed deep inside a generated query.

    Both are caller errors, and both now say so by name.
    """
    db.remember("something", now=T0)
    with pytest.raises(RangeError) as exc:
        db.recall("something", **kwargs)
    assert exc.value.field in kwargs


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": -5}, {"hops": -1}])
def test_non_positive_graph_arguments_are_refused(db, kwargs):
    db.remember("something", entities=["Ada"], now=T0)
    with pytest.raises(RangeError):
        db.recall_2hop("Ada", **kwargs)


def test_a_non_positive_prune_limit_is_refused(db):
    with pytest.raises(RangeError):
        db.prune(older_than=T0, limit=0)


def test_prune_with_no_policy_still_raises_and_is_still_a_value_error(db):
    with pytest.raises(ValidationError) as exc:
        db.prune()
    assert isinstance(exc.value, ValueError)  # 0.1.0 raised a bare ValueError here


def test_a_duplicate_memory_id_in_one_tenant_is_refused(db):
    """0.1.0 accepted it, and then ``get()`` returned an arbitrary one of the two rows."""
    db.remember("first", memory_id=4242, now=T0)
    with pytest.raises(DuplicateIdError) as exc:
        db.remember("second", memory_id=4242, now=T0)
    assert (exc.value.id, exc.value.tenant_id, exc.value.table) == (4242, 1, "memories")
    assert db.connection.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
    assert db.get(4242).content == "first"


def test_the_same_memory_id_in_another_tenant_is_still_fine(db):
    """ids are unique per TENANT.  Rejecting a cross-tenant reuse would break the documented
    (and tested) shape where two tenants both hold memory 4242."""
    db.remember("TENANT-1 PUBLIC", memory_id=4242, tenant=1, now=T0)
    db.remember("TENANT-2 SECRET", memory_id=4242, tenant=2, now=T0)
    assert db.get(4242, tenant=1).content == "TENANT-1 PUBLIC"
    assert db.get(4242, tenant=2).content == "TENANT-2 SECRET"


def test_a_duplicate_memory_id_is_refused_on_supersede_too(db):
    a = db.remember("first", now=T0)
    with pytest.raises(DuplicateIdError):
        db.supersede(a.memory_id, "second", memory_id=a.memory_id, now=T0)
    assert db.get(a.memory_id).valid_to is None  # the supersede rolled back entirely


def test_a_duplicate_episode_id_is_refused(db):
    db.episode("raw text", episode_id=99, now=T0)
    with pytest.raises(DuplicateIdError):
        db.episode("other text", episode_id=99, now=T0)


@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_explicit_id_is_refused(db, bad):
    with pytest.raises(RangeError):
        db.remember("negative id", memory_id=bad, now=T0)


def test_validation_errors_are_value_errors_for_0_1_0_callers(db):
    """Backward compatibility: everything raised here is still a ``ValueError``."""
    for call in (
        lambda: db.remember("x", confidence=5.0),
        lambda: db.remember("x", embedding=[NAN] * DIM),
        lambda: db.recall("x", k=0),
        lambda: db.remember("x", memory_id=-3),
    ):
        with pytest.raises(ValueError):
            call()


# ==================================================================== defect 5: the allocator


class FakeClock:
    """A settable ``time.time`` replacement, in seconds since the unix epoch."""

    def __init__(self, ms: int) -> None:
        self.ms = ms

    def __call__(self) -> float:
        return self.ms / 1000.0


def test_a_clock_rollback_cannot_make_the_allocator_repeat_an_id(monkeypatch):
    """The reviewer's reproduction: roll the clock back, then mint more than 1,024 ids.

    0.1.0 pinned the timestamp and advanced the sequence with ``(seq + 1) & 0x3FF``, so the
    1,025th id after a rollback was bit-for-bit the 1st.  Two memories would then share an id
    inside a tenant -- exactly the fault ``doctor()`` reports below, arriving without anyone
    doing anything wrong.
    """
    clock = FakeClock(EPOCH_MS + 10_000_000)
    monkeypatch.setattr("anatid.ids.time.time", clock)
    alloc = IdAllocator(worker_id=7)

    before = [alloc.next_id() for _ in range(2000)]
    clock.ms -= 3_600_000  # an hour backwards, mid-run
    after = [alloc.next_id() for _ in range(3000)]

    ids = before + after
    assert len(set(ids)) == len(ids), "the allocator repeated an id after a clock rollback"
    assert all(b > a for a, b in itertools.pairwise(ids)), "ids stopped increasing"
    # the specific 0.1.0 failure: the id 1,024 places after the rollback equalled the first
    assert after[1024] != after[0]
    assert alloc.drift_ms > 0  # it is holding its own clock, not following


def test_the_allocator_never_repeats_across_sequence_exhaustion(monkeypatch):
    """A frozen clock plus more than 1,024 ids in it: borrow the next millisecond, never wrap."""
    clock = FakeClock(EPOCH_MS + 5_000)
    monkeypatch.setattr("anatid.ids.time.time", clock)
    alloc = IdAllocator(worker_id=1)

    ids = [alloc.next_id() for _ in range(5_000)]
    assert len(set(ids)) == 5_000
    assert ids == sorted(ids)
    assert alloc.last_ms > 5_000  # it walked forward through logical time
    clock.ms += 60_000  # the wall clock catches up and overtakes
    assert alloc.next_id() > ids[-1]
    assert alloc.drift_ms == 0


def test_ids_stay_positive_when_the_clock_predates_the_epoch(monkeypatch):
    """A clock set before 2020 would otherwise shift a negative number into the sign bit."""
    monkeypatch.setattr("anatid.ids.time.time", FakeClock(EPOCH_MS - 10_000_000_000))
    alloc = IdAllocator(worker_id=3)
    ids = [alloc.next_id() for _ in range(2000)]
    assert min(ids) > 0
    assert len(set(ids)) == len(ids)


def test_the_allocator_is_thread_safe():
    alloc = IdAllocator(worker_id=11)
    out: list[list[int]] = []
    lock = threading.Lock()
    started = threading.Barrier(8)

    def mint() -> None:
        started.wait()
        got = [alloc.next_id() for _ in range(2000)]
        with lock:
            out.append(got)

    threads = [threading.Thread(target=mint) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    flat = [i for chunk in out for i in chunk]
    assert len(set(flat)) == len(flat) == 16_000


# ==================================================================== defect 5: doctor()


def test_doctor_on_a_healthy_database_reports_no_errors(db):
    db.remember("Ada likes coffee", entities=["Ada", "coffee"], embedding=vec(1.0), now=T0)
    db.relate("Ada", "coffee", now=T0)
    db.rebuild_fts_index(now=T0)
    report = db.doctor()
    assert report.ok is True
    assert report.clean is True, report
    assert bool(report) is True
    assert report.schema_version == S.SCHEMA_VERSION == report.expected_schema_version
    assert report.counts["memories"] == 1 and report.counts["entities"] == 2
    assert "duplicate_memory_ids" in report.checks_run
    assert report.as_dict()["ok"] is True
    assert "ok" in str(report)


def test_doctor_reports_duplicate_memory_ids(db):
    """Seeded the only way it can now happen: raw SQL, a bulk load, or an id allocator that
    repeated -- which is what defect 5's clock rollback did."""
    db.remember("first", memory_id=4242, now=T0)
    db.execute(
        "INSERT INTO memories (memory_id, tenant_id, content, kind, created_at, "
        "valid_from, tx_from, confidence, access_count) "
        "VALUES (4242, 1, 'second', 'fact', ?, ?, ?, 1.0, 0)",
        [T0, T0, T0],
    )
    report = db.doctor()
    f = report.find("duplicate_memory_ids")
    assert f is not None and f.severity is Severity.ERROR
    assert f.count == 1 and f.table == "memories"
    assert f.samples[0][:2] == (1, 4242)
    assert report.ok is False and bool(report) is False


def test_doctor_reports_duplicate_entities(db):
    """A pre-v3 fracture: two rows whose names canonicalise the same inside one tenant.

    The unique index forbids it, so the fault has to be seeded by dropping the index first --
    which is itself the point: ``doctor()`` reports the missing index too.
    """
    db.execute("DROP INDEX ux_entities_tenant_key")
    for eid, name in ((1, NAME), (2, "ada   lovelace")):
        db.execute(
            "INSERT INTO entities (entity_id, tenant_id, name, valid_from, tx_from) "
            "VALUES (?, 1, ?, ?, ?)",
            [eid, name, T0, T0],
        )
    report = db.doctor()
    dup = report.find("duplicate_entity_names")
    assert dup is not None and dup.count == 1
    assert dup.samples[0][1] == "ada lovelace"
    drift = report.find("schema_drift")
    assert drift is not None
    assert ("missing_required_index", "ux_entities_tenant_key") in drift.samples


def test_doctor_reports_duplicate_entity_ids(db):
    db.execute(
        "INSERT INTO entities (entity_id, tenant_id, name, valid_from, tx_from) "
        "VALUES (5, 1, 'Ada', ?, ?), (5, 1, 'Grace', ?, ?)",
        [T0, T0, T0, T0],
    )
    f = db.doctor().find("duplicate_entity_ids")
    assert f is not None and f.count == 1 and f.samples[0][:2] == (1, 5)


def test_doctor_reports_duplicate_live_edges(db):
    """Two current edges saying the same thing: nothing is lost, but a 2-hop weight doubles."""
    db.relate("Ada", "Kestrel", rel_kind="leads", now=T0)
    m = db.remember("Ada leads Kestrel", entities=["Ada"], now=T0)
    assert db.doctor().find("duplicate_live_edges") is None
    ada, kestrel = db.entity_id("Ada"), db.entity_id("Kestrel")
    # raw copies, as a bulk load would write them
    db.execute(
        "INSERT INTO edges_relates (edge_id, src, dst, tenant_id, rel_kind, valid_from, "
        "tx_from) VALUES (800, ?, ?, 1, 'leads', ?, ?)",
        [ada, kestrel, T0, T0],
    )
    db.execute(
        "INSERT INTO edges_about (edge_id, src, dst, tenant_id, weight, valid_from, tx_from) "
        "VALUES (801, ?, ?, 1, 1.0, ?, ?)",
        [m.memory_id, ada, T0, T0],
    )
    # a CLOSED copy is history, not a duplicate
    db.execute(
        "INSERT INTO edges_relates (edge_id, src, dst, tenant_id, rel_kind, valid_from, "
        "valid_to, tx_from) VALUES (802, ?, ?, 1, 'leads', ?, ?, ?)",
        [ada, kestrel, T0, T0, T0],
    )
    report = db.doctor()
    f = report.find("duplicate_live_edges")
    assert f is not None and f.severity is Severity.WARNING
    assert f.count == 2, f.samples
    assert {(s[0], s[5]) for s in f.samples} == {("edges_about", 2), ("edges_relates", 2)}
    assert report.ok is True  # a warning, not a break
    assert "duplicate_live_edges" in report.checks_run


def test_doctor_reports_dangling_edges(db):
    m = db.remember("Ada likes coffee", entities=["Ada"], now=T0)
    db.execute(
        "INSERT INTO edges_about (edge_id, src, dst, tenant_id, weight, valid_from, "
        "tx_from) VALUES (900, ?, 123456, 1, 1.0, ?, ?)",
        [m.memory_id, T0, T0],
    )
    db.execute(
        "INSERT INTO edges_relates (edge_id, src, dst, tenant_id, valid_from, tx_from) "
        "VALUES (901, 123456, 654321, 1, ?, ?)",
        [T0, T0],
    )
    f = db.doctor().find("dangling_edges")
    assert f is not None and f.severity is Severity.ERROR
    # one ABOUT dst plus two RELATES_TO endpoints
    assert f.count == 3, f.samples
    assert {s[0] for s in f.samples} == {"edges_about", "edges_relates"}


def test_doctor_reports_embedding_dimension_mismatches(db):
    """A vector of the wrong length, reachable through a bulk load of a file written for another
    model.  The verbs refuse it; ``load_parquet`` casts, and a cast that silently truncated
    would be worse."""
    db.remember("fine", embedding=vec(1.0), now=T0)
    db.execute("DROP INDEX idx_memories_id")  # the column type cannot change under an index
    db.execute("ALTER TABLE memories ALTER COLUMN embedding SET DATA TYPE FLOAT[]")
    db.execute(
        "INSERT INTO memories (memory_id, tenant_id, content, kind, created_at, "
        "valid_from, tx_from, confidence, access_count, embedding) "
        "VALUES (77, 1, 'wrong width', 'fact', ?, ?, ?, 1.0, 0, [1.0, 2.0])",
        [T0, T0, T0],
    )
    report = db.doctor()
    f = report.find("embedding_dimension_mismatch")
    assert f is not None and f.count == 1
    assert f.samples[0] == (1, 77, 2)
    # the column's declared type is schema drift in its own right: it now accepts what the
    # verbs refuse
    drift = report.find("schema_drift")
    assert drift is not None
    assert ("embedding_column_type", "FLOAT[]", "FLOAT[8]") in drift.samples


def test_doctor_reports_non_finite_embeddings(db):
    """Seeded in SQL, because DuckDB's Python client turns a NaN inside a bound list into NULL
    -- which is its own flavour of the same fault, so both are checked."""
    db.execute(
        "INSERT INTO memories (memory_id, tenant_id, content, kind, created_at, "
        "valid_from, tx_from, confidence, access_count, embedding) VALUES "
        "(78, 1, 'poisoned', 'fact', ?, ?, ?, 1.0, 0, "
        " ['nan'::FLOAT, 0, 0, 0, 0, 0, 0, 0]::FLOAT[8]), "
        "(79, 1, 'holed', 'fact', ?, ?, ?, 1.0, 0, "
        " [NULL, 0, 0, 0, 0, 0, 0, 0]::FLOAT[8])",
        [T0, T0, T0, T0, T0, T0],
    )
    f = db.doctor().find("non_finite_embeddings")
    assert f is not None and f.count == 2
    assert f.samples[0] == (1, 78) and f.samples[1] == (1, 79)
    # and it is the one check that can be skipped, because it reads every vector
    shallow = db.doctor(deep=False)
    assert shallow.find("non_finite_embeddings") is None
    assert "non_finite_embeddings" in shallow.checks_skipped


def test_doctor_reports_confidence_and_weight_out_of_range(db):
    db.execute(
        "INSERT INTO memories (memory_id, tenant_id, content, kind, created_at, "
        "valid_from, tx_from, confidence, access_count) "
        "VALUES (80, 1, 'over-believed', 'fact', ?, ?, ?, 2.0, 0)",
        [T0, T0, T0],
    )
    db.execute(
        "INSERT INTO edges_about (edge_id, src, dst, tenant_id, weight, valid_from, "
        "tx_from) VALUES (902, 80, 1, 1, -1.0, ?, ?)",
        [T0, T0],
    )
    report = db.doctor()
    conf = report.find("confidence_out_of_range")
    assert conf is not None and conf.count == 1 and conf.samples[0][0] == "memories"
    weight = report.find("weight_out_of_range")
    assert weight is not None and weight.count == 1


def test_doctor_reports_intervals_that_close_before_they_open(db):
    db.execute(
        "INSERT INTO memories (memory_id, tenant_id, content, kind, created_at, "
        "valid_from, valid_to, tx_from, confidence, access_count) "
        "VALUES (81, 1, 'never true', 'fact', ?, ?, ?, ?, 1.0, 0)",
        [T0, T0, T0 - _dt.timedelta(days=1), T0],
    )
    f = db.doctor().find("timestamp_order")
    assert f is not None and f.count == 1 and f.samples[0][0] == "memories"


def test_doctor_reports_a_derived_index_that_cannot_serve_reads(db):
    """The upkeep signal for a database on the framework.

    ``stale_fts_index`` is about 0.1.1's non-incremental index and is silent here, correctly:
    nothing is invisible, because the journal covers every write.  What an operator needs
    instead is which generation reads cannot use and why, and doctor has to say so without
    claiming the answers are wrong -- they are not, the SQL path is the oracle.
    """
    from anatid import fts as _fts

    for i in range(20):
        db.remember(f"note {i} about ducks", now=T0)
    db.maintain_indexes()
    assert db.doctor().find("unusable_derived_index") is None
    assert db.doctor().find("stale_fts_index") is None
    assert "unusable_derived_index" in db.doctor().checks_run

    storage = _fts.FtsStorage.of(_fts.index_of(db).current_generation())
    db.execute(f"DELETE FROM {storage.docmap} WHERE docid % 3 = 0")

    finding = db.doctor().find("unusable_derived_index")
    assert finding is not None and finding.severity is Severity.WARNING
    assert finding.samples[0][0] == "fts" and finding.samples[0][1] == "damaged_base"
    assert db.doctor().ok is True  # a fallback is a warning, not an integrity error
    # ... and the answer is still the right one, from the oracle
    assert len(db.recall("ducks", k=20)) == 20

    db.maintain_indexes()
    assert db.doctor().find("unusable_derived_index") is None


def test_doctor_reports_a_stale_full_text_index(legacy_db):
    # 0.1.1's file-wide BM25 index, whose tables these checks are about: with the derived
    # index attached, rebuild_fts_index() builds a generation instead and leaves them empty.
    db = legacy_db
    db.remember("indexed", now=T0)
    db.rebuild_fts_index(now=T0)
    assert db.doctor().find("stale_fts_index") is None
    db.remember("written after the rebuild", now=T0)
    f = db.doctor().find("stale_fts_index")
    assert f is not None and f.severity is Severity.WARNING and f.count == 1
    assert db.doctor().ok is True  # staleness is a warning, not an error


def test_doctor_reports_a_bm25_document_whose_memory_is_gone(legacy_db):
    """The orphan check earns its severity: ``anatid_fts_documents.content`` is verbatim text,
    so an orphan is a copy of a deleted memory still sitting in the file."""
    # 0.1.1's file-wide BM25 index, whose tables these checks are about: with the derived
    # index attached, rebuild_fts_index() builds a generation instead and leaves them empty.
    db = legacy_db
    m = db.remember("swordfish are extremely secret", now=T0)
    db.rebuild_fts_index(now=T0)
    db.execute("DELETE FROM memories WHERE memory_id = ?", [m.memory_id])  # raw, not forget()
    f = db.doctor().find("orphaned_fts_documents")
    assert f is not None and f.severity is Severity.ERROR
    assert f.samples[0] == (1, m.memory_id)


def test_a_hard_forget_leaves_no_orphaned_bm25_document(legacy_db):
    """The supported path does not create that orphan: ``forget(hard=True)`` purges the index
    rows in the same transaction, and says how many on the receipt."""
    # 0.1.1's file-wide BM25 index, whose tables these checks are about: with the derived
    # index attached, rebuild_fts_index() builds a generation instead and leaves them empty.
    db = legacy_db
    m = db.remember("swordfish are extremely secret", now=T0)
    db.rebuild_fts_index(now=T0)
    receipt = db.forget(m.memory_id, hard=True, now=T0)
    assert receipt.fts_rows_deleted > 0
    assert receipt.rows_removed >= receipt.fts_rows_deleted
    assert (
        db.connection.execute(
            f"SELECT count(*) FROM {S.FTS_SOURCE_TABLE} WHERE content LIKE '%extremely secret%'"
        ).fetchone()[0]
        == 0
    )
    assert db.doctor().find("orphaned_fts_documents") is None


def test_doctor_reports_per_tenant_fts_statistics_drift(legacy_db):
    # 0.1.1's file-wide BM25 index, whose tables these checks are about: with the derived
    # index attached, rebuild_fts_index() builds a generation instead and leaves them empty.
    db = legacy_db
    db.remember("indexed", now=T0)
    db.rebuild_fts_index(now=T0)
    db.execute(f"DELETE FROM {S.FTS_STATS_TABLE} WHERE tenant_id = 1")
    f = db.doctor().find("fts_statistics_drift")
    assert f is not None and f.severity is Severity.WARNING and f.samples[0] == (1,)


def test_doctor_reports_schema_drift_when_a_table_is_missing(db):
    db.execute("DROP TABLE edges_supersedes")
    f = db.doctor().find("schema_drift")
    assert f is not None and ("missing_table", "edges_supersedes") in f.samples


def test_doctor_is_scoped_to_one_tenant_by_default(db):
    db.execute(
        "INSERT INTO memories (memory_id, tenant_id, content, kind, created_at, "
        "valid_from, tx_from, confidence, access_count) VALUES "
        "(90, 2, 'a', 'fact', ?, ?, ?, 1.0, 0), (90, 2, 'b', 'fact', ?, ?, ?, 1.0, 0)",
        [T0, T0, T0, T0, T0, T0],
    )
    assert db.doctor(tenant=1).find("duplicate_memory_ids") is None
    assert db.doctor(tenant=2).find("duplicate_memory_ids") is not None
    everything = db.doctor(all_tenants=True)
    assert everything.find("duplicate_memory_ids") is not None
    assert everything.tenant_id is None and everything.all_tenants is True


def test_doctor_can_raise_instead_of_reporting(db):
    db.execute(
        "INSERT INTO entities (entity_id, tenant_id, name, valid_from, tx_from) "
        "VALUES (5, 1, 'Ada', ?, ?), (5, 1, 'Grace', ?, ?)",
        [T0, T0, T0, T0],
    )
    with pytest.raises(IntegrityError) as exc:
        db.doctor(raise_on_error=True)
    assert "duplicate_entity_ids" in str(exc.value)
    assert exc.value.report.find("duplicate_entity_ids") is not None
    assert all(f.is_error for f in exc.value.findings)


def test_doctor_reads_only(db):
    """It is a health check.  Every repair it might make has more than one right answer."""
    db.remember("one", entities=["Ada"], now=T0)
    con = db.connection
    before = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in S.ALL_TABLES}
    db.doctor(all_tenants=True)
    after = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in S.ALL_TABLES}
    assert before == after


# ==================================================================== against a v2 file

#: The schema-v2 entity table: no ``entity_key``, no unique index.  Kept here rather than
#: imported so this module stands alone.
V2_DDL = [
    """CREATE TABLE anatid_meta (
    schema_version   INTEGER   NOT NULL,
    created_at       TIMESTAMP NOT NULL,
    embedding_dim    INTEGER   NOT NULL,
    anatid_version   VARCHAR   NOT NULL,
    duckdb_version   VARCHAR   NOT NULL,
    system_columns   BOOLEAN   NOT NULL,
    fts_indexed_at   TIMESTAMP,
    fts_indexed_rows BIGINT,
    fts_indexed_max_id BIGINT,
    contract         VARCHAR   NOT NULL)""",
    """CREATE TABLE anatid_audit (
    audit_id BIGINT NOT NULL, tenant_id INTEGER NOT NULL, memory_id BIGINT,
    related_memory_id BIGINT, action VARCHAR NOT NULL, reason VARCHAR, writer VARCHAR,
    happened_at TIMESTAMP NOT NULL)""",
    """CREATE TABLE entities (
    entity_id BIGINT NOT NULL, tenant_id INTEGER NOT NULL, kind VARCHAR, name VARCHAR,
    valid_from TIMESTAMP, valid_to TIMESTAMP, tx_from TIMESTAMP, tx_to TIMESTAMP,
    writer VARCHAR, episode_id BIGINT, confidence FLOAT)""",
    """CREATE TABLE memories (
    memory_id BIGINT NOT NULL, tenant_id INTEGER NOT NULL, content VARCHAR, kind VARCHAR,
    embedding FLOAT[8], created_at TIMESTAMP NOT NULL,
    valid_from TIMESTAMP, valid_to TIMESTAMP, tx_from TIMESTAMP, tx_to TIMESTAMP,
    writer VARCHAR, episode_id BIGINT, confidence FLOAT,
    access_count INTEGER DEFAULT 0, last_access_at TIMESTAMP)""",
    """CREATE TABLE episodes (
    episode_id BIGINT NOT NULL, tenant_id INTEGER NOT NULL, source VARCHAR, content VARCHAR,
    kind VARCHAR, created_at TIMESTAMP NOT NULL, valid_from TIMESTAMP, valid_to TIMESTAMP,
    tx_from TIMESTAMP, tx_to TIMESTAMP, writer VARCHAR)""",
    """CREATE TABLE edges_about (
    edge_id BIGINT NOT NULL, src BIGINT NOT NULL, dst BIGINT NOT NULL, tenant_id INTEGER NOT NULL,
    weight FLOAT, valid_from TIMESTAMP, valid_to TIMESTAMP, tx_from TIMESTAMP, tx_to TIMESTAMP,
    writer VARCHAR, episode_id BIGINT, confidence FLOAT)""",
    """CREATE TABLE edges_relates (
    edge_id BIGINT NOT NULL, src BIGINT NOT NULL, dst BIGINT NOT NULL, tenant_id INTEGER NOT NULL,
    rel_kind VARCHAR, valid_from TIMESTAMP, valid_to TIMESTAMP, tx_from TIMESTAMP,
    tx_to TIMESTAMP, writer VARCHAR, episode_id BIGINT, confidence FLOAT)""",
    """CREATE TABLE edges_supersedes (
    edge_id BIGINT NOT NULL, src BIGINT NOT NULL, dst BIGINT NOT NULL, tenant_id INTEGER NOT NULL,
    tx_from TIMESTAMP, writer VARCHAR)""",
]


def write_fractured_v2_file(path) -> None:
    """A v2 file holding the fracture the race produced: four rows for one person.

    Four spellings, one tenant, plus a memory ABOUT two of them and a RELATES_TO between two
    more -- so the merge has to repoint edges, not just delete rows.
    """
    con = duckdb.connect(str(path))
    try:
        for stmt in V2_DDL:
            con.execute(stmt)
        con.execute(
            "INSERT INTO anatid_meta VALUES (2, ?, 8, '0.1.0', ?, TRUE, NULL, NULL, NULL, 'v2')",
            [T0, duckdb.__version__],
        )
        for eid, name in (
            (10, NAME),
            (11, "ada lovelace"),
            (12, "  ADA   LOVELACE "),
            (13, "Ada  Lovelace"),
        ):
            con.execute(
                "INSERT INTO entities (entity_id, tenant_id, kind, name, valid_from, "
                "tx_from, confidence) VALUES (?, 1, 'person', ?, ?, ?, 1.0)",
                [eid, name, T0, T0],
            )
        # a different tenant may legitimately hold the same name
        con.execute(
            "INSERT INTO entities (entity_id, tenant_id, kind, name, valid_from, "
            "tx_from, confidence) VALUES (20, 2, 'person', ?, ?, ?, 1.0)",
            [NAME, T0, T0],
        )
        con.execute(
            "INSERT INTO memories (memory_id, tenant_id, content, kind, created_at, "
            "valid_from, tx_from, confidence, access_count) "
            "VALUES (100, 1, 'she wrote the first program', 'fact', ?, ?, ?, 1.0, 0)",
            [T0, T0, T0],
        )
        for edge_id, dst in ((1, 10), (2, 12)):
            con.execute(
                "INSERT INTO edges_about (edge_id, src, dst, tenant_id, weight, "
                "valid_from, tx_from, confidence) VALUES (?, 100, ?, 1, 1.0, ?, ?, 1.0)",
                [edge_id, dst, T0, T0],
            )
        con.execute(
            "INSERT INTO edges_relates (edge_id, src, dst, tenant_id, valid_from, "
            "tx_from, confidence) VALUES (3, 11, 13, 1, ?, ?, 1.0)",
            [T0, T0],
        )
        assert con.execute("SELECT count(*) FROM entities WHERE tenant_id = 1").fetchone()[0] == 4
    finally:
        con.close()


def test_opening_a_fractured_v2_file_merges_it_and_the_verbs_then_hold_the_line(tmp_path):
    """The migration repairs the damage the race already did, and the verbs keep it repaired.

    This is the whole fix end to end: the file arrives with four rows for one person (what the
    reviewer's 100 concurrent writers produced), the 2->3 migration merges them and repoints
    every edge, and afterwards ``entity_id`` resolves every spelling to the survivor and cannot
    create a fifth.
    """
    path = tmp_path / "fractured.anatid"
    write_fractured_v2_file(path)

    with Anatid.open(path, tenant=1, embedding_dim=DIM) as db:
        con = db.connection
        assert db.info().schema_version == S.SCHEMA_VERSION == 4

        rows = con.execute("SELECT entity_id, name FROM entities WHERE tenant_id = 1").fetchall()
        assert len(rows) == 1, rows
        survivor = rows[0][0]
        assert survivor == 10  # the lowest id wins, deterministically

        # tenant 2's identically-named entity is untouched: the constraint is per tenant
        assert con.execute("SELECT entity_id FROM entities WHERE tenant_id = 2").fetchall() == [
            (20,)
        ]

        # every spelling now resolves to the survivor, and none of them creates a new row
        for spelling in (NAME, "ada lovelace", "  ADA   LOVELACE ", "Ada  Lovelace"):
            assert db.entity_id(spelling, create=True, now=T0) == survivor
        assert con.execute("SELECT count(*) FROM entities WHERE tenant_id = 1").fetchone()[0] == 1

        # the edges were repointed, not dropped, and the duplicate ABOUT collapsed to one
        assert con.execute("SELECT dst FROM edges_about WHERE tenant_id = 1").fetchall() == [
            (survivor,)
        ]
        assert con.execute("SELECT src, dst FROM edges_relates WHERE tenant_id = 1").fetchall() == [
            (survivor, survivor)
        ]

        # and the graph still answers
        assert [e.entity_id for e in db.entities_of(100)] == [survivor]
        m = db.remember("she also wrote the notes", entities=["ADA LOVELACE"], now=T0)
        assert [e.entity_id for e in db.entities_of(m.memory_id)] == [survivor]

        report = db.doctor()
        assert report.find("duplicate_entity_names") is None
        assert report.find("schema_drift") is None
        assert report.ok is True, report


def test_a_migrated_v2_file_survives_the_hundred_writer_race(tmp_path):
    """The constraint the migration installs is the one that holds under load."""
    path = tmp_path / "fractured-race.anatid"
    write_fractured_v2_file(path)
    with Anatid.open(path, tenant=1, embedding_dim=DIM) as db:
        started = threading.Barrier(60)
        errors: list[BaseException] = []
        lock = threading.Lock()

        def writer(i: int) -> None:
            started.wait()
            try:
                db.remember(f"note {i}", entities=["Charles Babbage"], now=T0)
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(60)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"first failure: {errors[0]!r}"
        assert (
            db.connection.execute("SELECT count(*) FROM entities WHERE tenant_id = 1").fetchone()[0]
            == 2
        )
        assert db.doctor().ok is True


# ==================================================================== rebuild paths


def test_recluster_with_no_arguments_rebuilds_every_clustered_table(file_db):
    """Regression: ``entities.entity_key`` is a GENERATED column, and ``recluster()`` built its
    INSERT column list from ``PRAGMA table_info``, which lists generated columns like any other.

    DuckDB then refused the whole statement with "Binder Error: Cannot insert into a generated
    column", so the no-argument form -- the documented way to compact a file after a long run of
    appends -- raised.  It goes through ``schema.insertable_columns`` now.
    """
    for i in range(5):
        file_db.remember(
            f"memory {i}", entities=[f"person {i}", "shared"], embedding=vec(float(i)), now=T0
        )
    file_db.relate("person 1", "person 2", now=T0)
    file_db.episode("raw source", now=T0)

    counts = file_db.recluster()

    assert counts["entities"] == 6  # 5 people plus "shared"
    assert counts["memories"] == 5
    assert set(counts) == set(S.CLUSTER_ORDER)
    # the generated column survived the rebuild and still enforces uniqueness
    assert (
        file_db.connection.execute(
            "SELECT count(*) FROM entities WHERE entity_key = 'shared'"
        ).fetchone()[0]
        == 1
    )
    with pytest.raises(duckdb.ConstraintException):
        file_db.connection.execute(
            "INSERT INTO entities (entity_id, tenant_id, name) VALUES (999, 1, 'SHARED')"
        )
    assert file_db.doctor().ok is True
    # and the verbs still resolve names against the rebuilt table
    assert file_db.entity_id("Shared") == file_db.entity_id("shared")


# ==================================================================== defect 5: more boundary checks


def test_numpy_integers_are_accepted_where_ints_are(db):
    """``isinstance(numpy.int64(5), int)`` is False.  A caller that pulls ``k`` or an id out of a
    numpy array must not be told it passed the wrong type: any ``numbers.Integral`` will do."""
    np = pytest.importorskip("numpy")
    m = db.remember("counted", memory_id=np.int64(4243), entities=["Ada"], now=T0)
    assert m.memory_id == 4243 and isinstance(m.memory_id, int)
    assert db.recall("counted", k=np.int32(3), candidates=np.int64(7)) is not None
    assert db.recall_2hop("Ada", limit=np.int64(2), hops=np.int64(1)) is not None
    assert db.reinforce(4243, amount=np.int64(2), now=T0).access_count == 2
    with pytest.raises(ValidationError):
        db.recall("counted", k=True)  # a bool is still not a count


def test_a_duplicate_explicit_edge_id_is_refused(db):
    db.relate("Ada", "coffee", edge_id=55, now=T0)
    with pytest.raises(DuplicateIdError) as exc:
        db.relate("Ada", "tea", edge_id=55, now=T0)
    assert (exc.value.table, exc.value.id) == ("edges_relates", 55)
    assert db.connection.execute("SELECT count(*) FROM edges_relates").fetchone()[0] == 1
    # the refused relate created no entity either: the whole transaction rolled back
    assert db.get_entity("tea") is None


def test_supersede_cannot_close_a_memory_before_it_opened(db):
    """Timestamp ordering at the verb boundary.

    ``supersede(old, now=t)`` writes ``old.valid_to = t``.  With ``t`` before ``old.valid_from``
    the interval is ``[from, to)`` with ``to < from`` -- never open, invisible to every
    ``as_of``, and exactly what ``doctor()`` reports as ``timestamp_order``.  0.1.0 wrote it.
    """
    m = db.remember("true from the first", valid_from=T0, now=T0)
    with pytest.raises(RangeError) as exc:
        db.supersede(m.memory_id, "replaced before it began", now=T0 - _dt.timedelta(days=1))
    assert exc.value.field == "now"
    assert db.get(m.memory_id).valid_to is None
    assert db.connection.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
    assert db.doctor().find("timestamp_order") is None
    # closing exactly at valid_from is allowed: [t, t) is empty, not inverted
    db.supersede(m.memory_id, "replaced at once", now=T0)


def test_soft_forget_cannot_close_a_memory_before_it_opened(db):
    m = db.remember("scheduled fact", valid_from=T0 + _dt.timedelta(days=30), now=T0)
    with pytest.raises(RangeError):
        db.forget(m.memory_id, now=T0)
    assert db.get(m.memory_id).valid_to is None
    assert db.connection.execute("SELECT count(*) FROM anatid_audit").fetchone()[0] == 0
    # the escape hatch the error message names still works
    receipt = db.forget(m.memory_id, hard=True, now=T0)
    assert receipt.memories_deleted == 1
    assert db.doctor().find("timestamp_order") is None


# ==================================================================== defect 5: doctor() edges


def test_doctor_with_zero_samples_still_reports_the_fault(db):
    """Regression: ``probe`` fetched ``LIMIT {samples}`` and treated an empty page as a clean
    check, so ``doctor(samples=0)`` reported every database as healthy."""
    db.execute(
        "INSERT INTO entities (entity_id, tenant_id, name, valid_from, tx_from) "
        "VALUES (5, 1, 'Ada', ?, ?), (5, 1, 'Grace', ?, ?)",
        [T0, T0, T0, T0],
    )
    report = db.doctor(samples=0)
    f = report.find("duplicate_entity_ids")
    assert f is not None and f.count == 1 and f.samples == ()
    assert report.ok is False
    # and samples=1 caps the examples without touching the count
    db.execute(
        "INSERT INTO entities (entity_id, tenant_id, name, valid_from, tx_from) "
        "VALUES (6, 1, 'Alan', ?, ?), (6, 1, 'Turing', ?, ?)",
        [T0, T0, T0, T0],
    )
    f = db.doctor(samples=1).find("duplicate_entity_ids")
    assert f is not None and f.count == 2 and len(f.samples) == 1


def test_doctor_reports_a_schema_version_behind_this_build(db):
    """Seeded by hand: ensure_schema() would have migrated a real one on open."""
    db.execute("UPDATE anatid_meta SET schema_version = 2")
    f = db.doctor().find("schema_drift")
    assert f is not None and f.severity is Severity.ERROR
    assert ("schema_version", 2, S.SCHEMA_VERSION) in f.samples
    assert db.doctor().schema_version == 2


def test_doctor_reports_dangling_episode_references(db):
    m = db.remember("derived from evidence", episode="the raw text", now=T0)
    assert m.episode_id is not None
    db.execute("DELETE FROM episodes WHERE episode_id = ?", [m.episode_id])  # raw, not forget()
    f = db.doctor().find("dangling_episode_references")
    assert f is not None and f.severity is Severity.WARNING
    assert f.samples[0] == ("memories", 1, m.memory_id, m.episode_id)
    assert db.doctor().ok is True  # evidence missing is a warning, not a break
    # the supported path does not create it: an episode goes only when nothing cites it
    n = db.remember("also derived", episode="shared evidence", now=T0)
    db.remember("cites the same evidence", episode_id=n.episode_id, now=T0)
    db.forget(n.memory_id, hard=True, now=T0)
    assert db.doctor().find("dangling_episode_references").count == 1  # still just the raw one


def test_doctor_on_an_unmigrated_v2_file_reports_drift_instead_of_crashing(tmp_path):
    """``read_only=True`` skips ``ensure_schema()``, so a v2 file stays v2 under this handle.

    ``doctor()`` has to work there -- it is the tool you reach for BEFORE letting a migration
    rewrite a file -- so no check may assume a v3-only object such as the generated
    ``entity_key`` column or the fts sidecar tables.
    """
    path = tmp_path / "unmigrated.anatid"
    write_fractured_v2_file(path)
    with Anatid.open(path, tenant=1, embedding_dim=DIM, read_only=True) as db:
        report = db.doctor()
        assert report.schema_version == 2
        drift = report.find("schema_drift")
        assert drift is not None
        assert ("schema_version", 2, S.SCHEMA_VERSION) in drift.samples
        assert ("missing_table", S.FTS_SOURCE_TABLE) in drift.samples
        assert ("missing_required_index", "ux_entities_tenant_key") in drift.samples
        dup = report.find("duplicate_entity_names")
        assert dup is not None and dup.count == 1 and dup.samples[0] == (1, "ada lovelace", 4)
        assert report.ok is False
        # the file is untouched: still v2, still four rows
        assert (
            db.connection.execute("SELECT count(*) FROM entities WHERE tenant_id = 1").fetchone()[0]
            == 4
        )


def test_doctor_report_serialises_and_prints(db):
    db.execute(
        "INSERT INTO entities (entity_id, tenant_id, name, valid_from, tx_from) "
        "VALUES (5, 1, 'Ada', ?, ?), (5, 1, 'Grace', ?, ?)",
        [T0, T0, T0, T0],
    )
    report = db.doctor()
    d = report.as_dict()
    assert d["ok"] is False and d["expected_schema_version"] == S.SCHEMA_VERSION
    assert [f["check"] for f in d["findings"]] == ["duplicate_entity_ids"]
    assert d["findings"][0]["samples"] == [[1, 5, 2]]
    import json

    json.dumps(d)  # machine-readable means JSON-ready
    text = str(report)
    assert "FAULTS" in text and "duplicate_entity_ids" in text


# ==================================================================== erasure holes in verbs.forget


def _every_column(db) -> dict[str, list[str]]:
    """Every table and its columns, from the catalog -- so a table nobody remembered is scanned."""
    rows = db.execute(
        "SELECT schema_name, table_name, column_name FROM duckdb_columns() "
        "WHERE NOT internal ORDER BY schema_name, table_name, column_index"
    ).fetchall()
    out: dict[str, list[str]] = {}
    for schema_name, table_name, column_name in rows:
        out.setdefault(f'"{schema_name}"."{table_name}"', []).append(column_name)
    return out


def _tables_containing(db, needles: list[str]) -> dict[str, int]:
    """``{table: rows}`` for every table whose rows mention any needle in any column."""
    found: dict[str, int] = {}
    for table, columns in _every_column(db).items():
        tests = " OR ".join(
            f"contains(coalesce(CAST(\"{c}\" AS VARCHAR), ''), ?)" for _ in needles for c in columns
        )
        params = [n for n in needles for _ in columns]
        count = db.execute(f"SELECT count(*) FROM {table} WHERE {tests}", params).fetchone()[0]
        if count:
            found[table] = int(count)
    return found


def test_erasing_the_newest_indexed_memory_clamps_the_fts_watermark(legacy_db):
    """``anatid_meta.fts_indexed_max_id`` is "the largest memory_id at the last rebuild".

    By construction that is the newest memory's id, so erasing the newest indexed memory left
    its id in ``anatid_meta`` -- a number, but the erased id, and a catalog-wide scan for it
    found it.  ``forget(hard=True)`` now clamps the watermark to the largest id still indexed.
    """
    # 0.1.1's file-wide BM25 index, whose tables these checks are about: with the derived
    # index attached, rebuild_fts_index() builds a generation instead and leaves them empty.
    db = legacy_db
    older = db.remember("Ada likes DuckDB", entities=["Ada"], now=T0)
    target = db.remember("Ada's passport number is X9981", now=T0)
    db.rebuild_fts_index(now=T0)
    con = db.connection
    assert (
        con.execute("SELECT fts_indexed_max_id FROM anatid_meta").fetchone()[0] == target.memory_id
    )

    db.forget(target.memory_id, hard=True, now=T0)

    assert (
        con.execute("SELECT fts_indexed_max_id FROM anatid_meta").fetchone()[0] == older.memory_id
    )
    assert _tables_containing(db, [str(target.memory_id), "X9981"]) == {}
    # honesty about freshness survives the clamp: a document is gone, so the index is stale
    assert db.fts_status().stale is True
    db.rebuild_fts_index(now=T0)
    assert db.fts_status().stale is False


def test_erasing_the_only_indexed_memory_nulls_the_watermark(db):
    m = db.remember("alone", now=T0)
    db.rebuild_fts_index(now=T0)
    db.forget(m.memory_id, hard=True, now=T0)
    assert db.connection.execute("SELECT fts_indexed_max_id FROM anatid_meta").fetchone()[0] is None
    assert _tables_containing(db, [str(m.memory_id)]) == {}


def test_erasing_a_memory_whose_entity_outlives_it_still_erases_its_episode(db):
    """``remember(content, entities=[...], episode=raw)`` -- the documented evidence-first shape.

    ``remember`` stamps the episode onto every entity it mints, so under the old rule ("the
    episode goes when no other ROW cites it") the raw text -- which quotes the secret -- stayed
    in ``episodes`` for as long as the entity existed, i.e. forever.  An episode is evidence for
    memories: it goes with the last memory that cites it, and the stamps are cleared.
    """
    secret = "Ada's passport number is X9981-DO-NOT-KEEP"
    target = db.remember(secret, entities=["Ada"], episode=f"user said: {secret}", now=T0)
    db.remember("Ada likes DuckDB", entities=["Ada"], now=T0)  # Ada outlives the target
    db.rebuild_fts_index(now=T0)
    con = db.connection
    assert (
        con.execute("SELECT episode_id FROM entities WHERE name = 'Ada'").fetchone()[0]
        == target.episode_id
    )
    assert "episodes" in {t.split(".")[-1].strip('"') for t in _tables_containing(db, [secret])}

    receipt = db.forget(target.memory_id, hard=True, now=T0)

    assert receipt.episodes_deleted == 1
    assert _tables_containing(db, [str(target.memory_id), secret]) == {}, _tables_containing(
        db, [str(target.memory_id), secret]
    )
    # the entity is still there, minus the stamp; nothing dangles
    assert db.get_entity("Ada") is not None
    assert con.execute("SELECT episode_id FROM entities WHERE name = 'Ada'").fetchone()[0] is None
    report = db.doctor()
    assert report.find("dangling_episode_references") is None
    assert report.ok is True, report


def test_an_episode_another_memory_cites_survives_a_hard_forget_and_keeps_its_stamps(db):
    ep = db.episode("shared evidence", now=T0)
    a = db.remember("a", entities=["Ada"], episode_id=ep.episode_id, now=T0)
    db.remember("b", episode_id=ep.episode_id, now=T0)
    receipt = db.forget(a.memory_id, hard=True, now=T0)
    assert receipt.episodes_deleted == 0
    assert db.get_episode(ep.episode_id) is not None
    # the stamp on Ada is untouched: the evidence it points at still exists
    assert (
        db.connection.execute("SELECT episode_id FROM entities WHERE name = 'Ada'").fetchone()[0]
        == ep.episode_id
    )
