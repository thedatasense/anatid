"""Concurrency primitives: ``db.atomic``, compare-and-swap, and what is retryable.

The design's position, which these tests hold anatid to: DuckDB gives optimistic snapshot
isolation and aborts a write-write race; it does not give serializable application invariants.
So anatid offers three narrow things instead of pretending otherwise.

* :meth:`anatid.Anatid.atomic` re-runs the WHOLE callback in a fresh transaction after an
  engine-level conflict, with jittered backoff, and never re-runs anything else.
* :meth:`anatid.Anatid.update` writes only while the memory is at the version the caller read,
  and says which version it found when it is not.
* ``relate(..., if_current=True)`` refuses to attach an edge to an endpoint that is not there.

The distinction the whole module turns on is ``ConflictError.retryable``.  An engine abort is
retryable: nothing was committed and the next attempt re-reads.  A compare-and-swap failure is
not: the version the caller reasoned about is gone, so an identical retry fails identically.
"""

from __future__ import annotations

import datetime as _dt
import threading

import pytest

from anatid import Anatid, ConflictError, NotFoundError
from anatid import atomic as atomic_mod
from anatid.atomic import Attempt

from conftest import DIM, T0

NAME = "Ada Lovelace"


# ============================================================================ the retry loop


def test_atomic_returns_the_callbacks_result_without_retrying_a_success(db):
    calls = []

    def work():
        calls.append(1)
        return db.remember("Ada likes coffee", entities=[NAME], now=T0)

    m = db.atomic(work)
    assert calls == [1]
    assert db.get(m.memory_id).content == "Ada likes coffee"


def test_atomic_reruns_the_whole_callback_after_a_retryable_conflict(db):
    """The callback runs again from the top, not the failed statement.

    A transaction DuckDB aborted cannot be written to at all, so re-running one statement in it
    is meaningless.  Everything the callback did in the failed attempt was rolled back with it,
    which is why the second attempt sees a clean database.
    """
    seen = []

    def work(attempt: Attempt):
        seen.append(attempt.number)
        db.remember(f"attempt {attempt.number}", now=T0)
        if attempt.number < 3:
            raise ConflictError("engine says no")
        return attempt.number

    got = db.atomic(work, sleep=lambda _d: None)
    assert got == 3
    assert seen == [1, 2, 3]
    # the two rolled-back attempts left nothing behind
    assert db.stats()["memories"] == 1
    assert db.recall_2hop_ids(db.entity_id(NAME, create=True, now=T0)) == []


def test_atomic_sleeps_a_jittered_exponential_backoff_between_attempts(db):
    sleeps: list[float] = []

    def work(attempt: Attempt):
        if attempt.number < 3:
            raise ConflictError("nope")
        return "done"

    got = db.atomic(work, sleep=sleeps.append, rng=lambda: 1.0, backoff=0.01, max_backoff=1.0)
    assert got == "done"
    assert sleeps == [0.01, 0.02]


def test_backoff_is_full_jitter_and_capped():
    """Uniform in [0, step), doubling each attempt, never past the cap.

    Fixed backoff makes two writers that collided wake together and collide again; the jitter
    is what spreads them.
    """
    assert atomic_mod.backoff_delay(1, base=0.01, cap=1.0, rng=lambda: 0.5) == pytest.approx(0.005)
    assert atomic_mod.backoff_delay(2, base=0.01, cap=1.0, rng=lambda: 0.5) == pytest.approx(0.01)
    assert atomic_mod.backoff_delay(9, base=0.01, cap=0.02, rng=lambda: 1.0) == pytest.approx(0.02)
    assert atomic_mod.backoff_delay(1, base=0.0, cap=0.0, rng=lambda: 1.0) == 0.0
    draws = {atomic_mod.backoff_delay(4, base=0.01, cap=1.0) for _ in range(50)}
    assert len(draws) > 1 and max(draws) < 0.08 + 1e-9
    with pytest.raises(ValueError):
        atomic_mod.backoff_delay(0)


def test_atomic_never_retries_something_that_is_not_a_conflict(db):
    calls = []

    def work():
        calls.append(1)
        raise ValueError("the unit of work is wrong")

    with pytest.raises(ValueError):
        db.atomic(work)
    assert calls == [1]


def test_atomic_never_retries_a_compare_and_swap_failure(db):
    """A conflict the caller has to see.

    Re-running a callback that asks for version 7 when the row is at version 8 asks for the
    same impossible thing again.  Only the caller knows whether the change still applies to
    what is there now.
    """
    calls = []

    def work():
        calls.append(1)
        raise atomic_mod.version_conflict(
            "memory 5", expected_version=7, current_version=8, action="update"
        )

    with pytest.raises(ConflictError) as excinfo:
        db.atomic(work)
    assert calls == [1]
    assert excinfo.value.retryable is False
    assert excinfo.value.attempt == 1


def test_atomic_gives_up_after_max_attempts_and_records_which_one(db):
    calls = []

    def work():
        calls.append(1)
        raise ConflictError("always")

    with pytest.raises(ConflictError) as excinfo:
        db.atomic(work, max_attempts=4, sleep=lambda _d: None)
    assert len(calls) == 4
    assert excinfo.value.attempt == 4
    assert excinfo.value.retryable is True


def test_atomic_steps_aside_inside_a_transaction_the_caller_opened(db):
    """Only the caller can re-run the caller's transaction.

    anatid does not know what else went into it, and DuckDB has no savepoint to roll back to,
    so the conflict propagates and the decision stays where the knowledge is.
    """
    calls = []

    def work():
        calls.append(1)
        raise ConflictError("engine says no")

    with pytest.raises(ConflictError):
        with db.transaction():
            db.atomic(work, max_attempts=5)
    assert calls == [1]


def test_the_callback_may_take_the_attempt_or_no_arguments(db):
    seen: list[Attempt] = []

    def wants_it(attempt):
        seen.append(attempt)
        if attempt.first:
            raise ConflictError("first")
        return attempt.number

    assert db.atomic(wants_it, sleep=lambda _d: None) == 2
    assert [a.number for a in seen] == [1, 2]
    assert seen[0].last_error is None and str(seen[1].last_error) == "first"
    assert seen[0].first and not seen[0].final
    assert seen[1].final is False and seen[1].max_attempts == 3
    assert db.atomic(lambda: 42) == 42


def test_run_reports_the_conflicts_it_swallowed(db):
    def work(attempt):
        if attempt.number < 3:
            raise ConflictError(f"conflict {attempt.number}")
        return "ok"

    outcome = atomic_mod.run(db, work, sleep=lambda _d: None)
    assert outcome.result == "ok"
    assert outcome.attempts == 3 and outcome.retried
    assert [str(c) for c in outcome.conflicts] == ["conflict 1", "conflict 2"]


def test_atomic_rejects_a_nonsense_retry_budget(db):
    with pytest.raises(ValueError):
        db.atomic(lambda: 1, max_attempts=0)
    with pytest.raises(TypeError):
        db.atomic("not callable")


# ============================================================================ the entity race


def test_a_hundred_threads_racing_on_one_entity_name_all_win(file_db):
    """The design's acceptance test for :meth:`~anatid.Anatid.atomic`.

    100 threads ask for the same new entity at once.  ``UNIQUE (tenant_id, entity_key)`` means
    only one insert can survive, so 99 of them lose; losing has to be invisible, because all
    100 asked for the same thing and the answer exists.  Every thread re-runs its whole unit of
    work through ``atomic`` and reads the winner's row.
    """
    got: list[int] = []
    errors: list[BaseException] = []
    lock = threading.Lock()
    start = threading.Barrier(100)

    def writer():
        start.wait()
        try:
            entity = file_db.atomic(lambda: file_db.upsert_entity(NAME, now=T0), max_attempts=8)
        except BaseException as exc:  # noqa: BLE001 - the test reports whatever came out
            with lock:
                errors.append(exc)
            return
        with lock:
            got.append(entity.entity_id)

    threads = [threading.Thread(target=writer) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"{len(errors)} threads failed; first: {errors[0]!r}"
    assert len(got) == 100
    assert len(set(got)) == 1
    rows = (
        file_db.unsafe_connection(reason="test assertion")
        .execute("SELECT count(*) FROM entities WHERE tenant_id = 1")
        .fetchone()
    )
    assert rows[0] == 1


def test_threads_writing_memories_about_one_new_entity_all_commit(file_db):
    """The same race with real work attached, so a retry has something to undo.

    Each thread remembers its own memory about a shared new entity.  All 25 memories must
    exist exactly once: a re-run that duplicated its own rolled-back insert would show up here.
    """
    errors: list[BaseException] = []
    lock = threading.Lock()
    start = threading.Barrier(25)

    def writer(n: int):
        start.wait()
        try:
            file_db.atomic(
                lambda: file_db.remember(f"fact {n}", entities=[NAME], now=T0), max_attempts=8
            )
        except BaseException as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"first failure: {errors[0]!r}"
    contents = [m.content for m in file_db.recall_2hop(NAME, limit=100)]
    assert sorted(contents) == sorted(f"fact {n}" for n in range(25))


# ============================================================================ compare-and-swap


def test_the_second_writer_of_one_memory_is_told_which_version_it_missed(db):
    """Two writers act on the same read.  The second must not commit over the first.

    This is the serialized shape of the race, which is the one that actually happens: a retry
    after a client timeout, or two agents that read the same row a second apart.  DuckDB sees
    no conflict at all -- the first writer committed long before -- so the guard has to be
    anatid's.
    """
    m = db.remember("Ada drinks coffee", entities=[NAME], now=T0)
    read_version = m.version
    assert read_version == 1 and db.memory_version(m.memory_id) == 1

    first = db.update(m.memory_id, "Ada drinks tea", expected_version=read_version, now=T0)
    assert first.memory_id != m.memory_id
    assert db.memory_version(m.memory_id) == 2

    with pytest.raises(ConflictError) as excinfo:
        db.update(m.memory_id, "Ada drinks cocoa", expected_version=read_version, now=T0)
    exc = excinfo.value
    assert exc.resource == f"memory {m.memory_id}"
    assert exc.expected_version == 1
    assert exc.current_version == 2
    assert exc.retryable is False
    # and nothing was written by the loser
    assert [mm.content for mm in db.recall_2hop(NAME)] == ["Ada drinks tea"]


def test_update_without_an_expected_version_is_plain_supersede(db):
    m = db.remember("Ada drinks coffee", entities=[NAME], now=T0)
    n = db.update(m.memory_id, "Ada drinks tea", now=T0)
    assert n.content == "Ada drinks tea"
    assert db.get(m.memory_id).is_current is False
    assert [x.memory_id for x in db.provenance(n.memory_id).chain] == [n.memory_id, m.memory_id]


def test_update_passes_supersede_keywords_through(db):
    m = db.remember("Ada drinks coffee", entities=[NAME], now=T0, writer="agent-1")
    n = db.update(
        m.memory_id,
        "Ada drinks tea",
        expected_version=1,
        entities=["tea"],
        kind="preference",
        confidence=0.5,
        writer="agent-2",
        now=T0,
    )
    assert n.kind == "preference" and n.confidence == 0.5 and n.writer == "agent-2"
    assert [e.name for e in db.entities_of(n.memory_id)] == ["tea"]


def test_update_on_an_id_this_tenant_does_not_have_is_not_found(db):
    with pytest.raises(NotFoundError):
        db.update(404, "nothing here", expected_version=1)


def test_memory_version_counts_physical_versions_and_agrees_with_the_row(db):
    m = db.remember("Ada drinks coffee", now=T0)
    assert db.memory_version(m.memory_id) == db.get(m.memory_id).version == 1
    db.update(m.memory_id, "Ada drinks tea", now=T0)
    assert db.memory_version(m.memory_id) == db.get(m.memory_id).version == 2
    assert [v.version for v in db.versions(m.memory_id)] == [1, 2]
    assert db.memory_version(404) is None


def _rival_holding(db, memory_id: int, *, tenant_id: int = 1):
    """A second connection whose open transaction has already written the memory's ``tx_to``.

    The engine-level race, made deterministic.  The column matters: DuckDB detects a
    write-write conflict per COLUMN of a row, not per row (two transactions updating different
    columns of one row both commit), and ``tx_to`` is the column every correction writes, so a
    correction against this rival is exactly the race anatid promises to lose safely.
    """
    rival = db._root.cursor()
    rival.execute("BEGIN")
    rival.execute(
        "UPDATE memories SET tx_to = ? WHERE memory_id = ? AND tenant_id = ? AND tx_to IS NULL",
        [_dt.datetime(2026, 6, 1), int(memory_id), int(tenant_id)],
    )
    return rival


def test_a_concurrent_correction_of_the_same_row_is_a_retryable_conflict(file_db):
    m = file_db.remember("Ada drinks coffee", now=T0)
    rival = _rival_holding(file_db, m.memory_id)
    try:
        with pytest.raises(ConflictError) as excinfo:
            file_db.update(m.memory_id, "Ada drinks tea", expected_version=1, now=T0)
        assert excinfo.value.retryable is True
        assert "onflict" in str(excinfo.value)
    finally:
        rival.execute("ROLLBACK")
        rival.close()
    # the loser wrote nothing: the whole transaction rolled back with it
    assert file_db.get(m.memory_id).content == "Ada drinks coffee"
    assert file_db.stats()["memories"] == 1


def test_atomic_wins_the_engine_race_on_the_next_attempt(file_db):
    """The retry loop against real MVCC rather than a raised stand-in.

    Attempt 1 collides with a transaction that holds the row.  The rival lets go, attempt 2
    re-reads and commits, and the caller never sees the conflict.
    """
    m = file_db.remember("Ada drinks coffee", now=T0)
    rival = _rival_holding(file_db, m.memory_id)

    def work(attempt: Attempt):
        if attempt.number == 2:
            rival.execute("ROLLBACK")
        return file_db.update(m.memory_id, "Ada drinks tea", now=T0)

    try:
        new = file_db.atomic(work, sleep=lambda _d: None)
    finally:
        rival.close()
    assert new.content == "Ada drinks tea"
    assert file_db.get(m.memory_id).is_current is False
    # one correction, not two: the first attempt rolled back before the second ran
    assert file_db.stats()["memories"] == 2
    assert [x.content for x in file_db.recall_2hop_ids(0)] == []
    assert len(file_db.versions(m.memory_id)) == 2


# ============================================================================ relate(if_current)


def test_relate_if_current_refuses_an_endpoint_that_is_not_there(db):
    """``relate`` takes an int endpoint verbatim and never checks it, by design: a lookup on
    every edge write is a scan anatid will not do.  ``if_current=True`` is how a caller asks
    for the check when it matters."""
    ada = db.entity_id(NAME, create=True, now=T0)
    ghost = 999_999

    edge = db.relate(ada, ghost, now=T0)  # the unguarded verb writes it
    assert edge.dst == ghost

    with pytest.raises(ConflictError) as excinfo:
        db.relate(ada, ghost, if_current=True, now=T0)
    assert excinfo.value.resource == f"entity {ghost}"
    assert excinfo.value.retryable is False


def test_relate_if_current_refuses_a_closed_endpoint(db):
    ada = db.entity_id(NAME, create=True, now=T0)
    bob = db.entity_id("Bob", create=True, now=T0)
    db.execute("UPDATE entities SET valid_to = ? WHERE entity_id = ?", [T0, bob])

    with pytest.raises(ConflictError):
        db.relate(ada, bob, if_current=True, now=T0)
    assert db.relate(ada, bob, now=T0).dst == bob  # unguarded still writes


def test_relate_if_current_writes_the_same_edge_when_the_endpoints_are_there(db):
    edge = db.relate(NAME, "Bob", if_current=True, now=T0, rel_kind="knows", confidence=0.5)
    assert edge.rel_kind == "knows" and edge.confidence == 0.5
    assert edge.src == db.entity_id(NAME) and edge.dst == db.entity_id("Bob")
    # an entity created by this very call counts as current: the guard sees its own transaction
    assert db.get_entity("Bob") is not None
    m = db.remember("they corresponded", entities=[NAME], now=T0)
    assert [x.memory_id for x in db.recall_2hop("Bob", hops=2)] == [m.memory_id]


def test_relate_if_current_honours_create_missing(db):
    with pytest.raises(NotFoundError):
        db.relate("nobody", "no one else", if_current=True, create_missing=False, now=T0)


# ============================================================================ the error itself


def test_conflict_error_carries_the_fields_the_design_asks_for():
    plain = ConflictError("engine abort")
    assert plain.retryable is True
    assert plain.resource is None and plain.attempt is None
    assert plain.expected_version is None and plain.current_version is None
    assert ConflictError.retryable is True  # answerable without an instance

    cas = atomic_mod.version_conflict(
        "memory 9", expected_version=3, current_version=4, action="update"
    )
    assert (cas.resource, cas.expected_version, cas.current_version) == ("memory 9", 3, 4)
    assert cas.retryable is False
    assert "version 4" in str(cas) and "version 3" in str(cas)
    assert atomic_mod.is_retryable(cas) is False
    assert atomic_mod.is_retryable(plain) is True
    assert atomic_mod.is_retryable(ValueError("x")) is False


def test_a_cause_is_kept_so_the_duckdb_message_is_not_lost(file_db):
    m = file_db.remember("Ada drinks coffee", now=T0)
    rival = _rival_holding(file_db, m.memory_id)
    try:
        with pytest.raises(ConflictError) as excinfo:
            file_db.update(m.memory_id, "Ada drinks tea", now=T0)
        assert excinfo.value.cause is not None
    finally:
        rival.execute("ROLLBACK")
        rival.close()


def test_version_of_reads_the_live_row_not_a_closed_one(file_db):
    m = file_db.remember("Ada drinks coffee", now=T0)
    file_db.update(m.memory_id, "Ada drinks tea", now=T0)
    assert atomic_mod.version_of(file_db, "memories", "memory_id", m.memory_id, tenant_id=1) == 2
    assert atomic_mod.version_of(file_db, "memories", "memory_id", m.memory_id, tenant_id=2) is None


def test_current_ids_sees_only_this_tenants_current_rows(tmp_path):
    with Anatid.open(tmp_path / "two.anatid", tenant=1, embedding_dim=DIM) as db:
        one = db.entity_id(NAME, create=True, tenant=1, now=T0)
        two = db.entity_id(NAME, create=True, tenant=2, now=T0)
        assert atomic_mod.current_ids(db, "entities", "entity_id", [one, two], tenant_id=1) == {one}
        assert atomic_mod.current_ids(db, "entities", "entity_id", [one, two], tenant_id=2) == {two}
        assert atomic_mod.current_ids(db, "entities", "entity_id", [], tenant_id=1) == set()
        closed = _dt.datetime(2026, 2, 1)
        db.execute("UPDATE entities SET valid_to = ? WHERE entity_id = ?", [closed, one])
        assert atomic_mod.current_ids(db, "entities", "entity_id", [one], tenant_id=1) == set()
        assert atomic_mod.missing_or_closed(set(), [one, one, two]) == [one, two]
