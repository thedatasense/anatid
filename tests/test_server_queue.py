"""Per-tenant write queues, backpressure, batching, fairness, idempotency, and the dispatcher.

The three properties the queue exists for are each asserted against behaviour rather than
against a counter the queue keeps about itself:

* backpressure is a typed, retryable answer with a suggested wait, and the write did NOT happen;
* a batch that fails re-runs its members one at a time, so a neighbour is neither lost nor given
  someone else's error;
* a tenant with a thousand queued writes cannot starve a tenant with five.

Then idempotency, the whole point of which is the sequence in
``test_a_retry_after_a_lost_response_writes_once``: send a write, lose the response, retry with
the same key, and find exactly one row and the original result.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
import os
import shutil
import signal
import socket
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from anatid import Anatid, DatabasePool
from anatid.server import auth, protocol, queue as Q, server as S

DIM = 8
T0 = _dt.datetime(2026, 1, 1, 0, 0, 0)

#: A Unix socket path is a fixed-size field in ``sockaddr_un``: 104 bytes on macOS, 108 on
#: Linux.  pytest's ``tmp_path`` is nested deeply enough on macOS to overflow it, so socket
#: paths come from :func:`sock_dir` instead of ``tmp_path``.
SOCKADDR_UN_MAX = 100


@pytest.fixture
def sock_dir():
    """A directory short enough to hold a Unix socket path."""
    base = Path(tempfile.mkdtemp(prefix="anatid-t-"))
    if len(str(base)) + len("/run/xxxxxxxx.sock") > SOCKADDR_UN_MAX:
        shutil.rmtree(base, ignore_errors=True)
        base = Path(tempfile.mkdtemp(prefix="anatid-t-", dir="/tmp"))
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


# --------------------------------------------------------------------------- fakes


class FakeDb:
    """A database handle with just enough behaviour for the scheduler tests.

    ``transaction()`` is re-entrant like the real one and rolls back by truncating what the
    outermost block wrote, which is what makes the batch-isolation test mean anything.
    """

    def __init__(self, tenant_id: int = 0) -> None:
        self.tenant_id = tenant_id
        self.rows: list[object] = []
        self.transactions = 0
        self.rollbacks = 0
        self._depth = 0

    @contextmanager
    def transaction(self):
        self._depth += 1
        opened = self._depth == 1
        if opened:
            self.transactions += 1
        mark = len(self.rows)
        try:
            yield self
        except Exception:
            if opened:
                del self.rows[mark:]
                self.rollbacks += 1
            self._depth -= 1
            raise
        else:
            self._depth -= 1

    def write(self, value: object) -> object:
        self.rows.append(value)
        return value


@pytest.fixture
def fake_dbs():
    dbs: dict[int, FakeDb] = {}

    def open_db(tenant_id: int) -> FakeDb:
        return dbs.setdefault(int(tenant_id), FakeDb(int(tenant_id)))

    open_db.dbs = dbs  # type: ignore[attr-defined]
    return open_db


def _writer(value):
    def work(db):
        return db.write(value)

    return work


# --------------------------------------------------------------------------- backpressure


def test_a_full_tenant_queue_answers_busy_and_does_not_write(fake_dbs):
    """Backpressure is an answer, not a block and not a drop."""
    q = Q.WriteQueue(fake_dbs, max_depth=3, batch_max=1, workers=1)
    # Not started: nothing drains, so the queue fills deterministically.
    for i in range(3):
        q.submit(1, "remember", _writer(i))
    with pytest.raises(protocol.BusyError) as excinfo:
        q.submit(1, "remember", _writer("overflow"))
    busy = excinfo.value
    assert busy.retryable is True
    assert busy.tenant_id == 1
    assert busy.depth == 3
    assert busy.max_depth == 3
    assert busy.retry_after > 0
    assert q.depth(1) == 3
    assert q.stats().rejected == 1
    assert fake_dbs.dbs == {}, "a busy answer must not have opened a file or written anything"


def test_a_busy_answer_never_blocks_the_caller(fake_dbs):
    q = Q.WriteQueue(fake_dbs, max_depth=1, batch_max=1, workers=1)
    q.submit(1, "remember", _writer(0))
    started = time.monotonic()
    with pytest.raises(protocol.BusyError):
        q.submit(1, "remember", _writer(1))
    assert time.monotonic() - started < 0.5


def test_one_tenants_full_queue_does_not_refuse_another_tenant(fake_dbs):
    q = Q.WriteQueue(fake_dbs, max_depth=2, batch_max=1, workers=1)
    q.submit(1, "remember", _writer("a"))
    q.submit(1, "remember", _writer("b"))
    with pytest.raises(protocol.BusyError):
        q.submit(1, "remember", _writer("c"))
    q.submit(2, "remember", _writer("d"))  # a different tenant, a different queue
    assert q.depth(2) == 1


def test_the_suggested_wait_is_bounded_and_grows_with_the_backlog(fake_dbs):
    q = Q.WriteQueue(
        fake_dbs, max_depth=8, batch_max=2, workers=1, min_retry_after=0.001, max_retry_after=1.0
    )
    for i in range(8):
        q.submit(1, "remember", _writer(i))
    with pytest.raises(protocol.BusyError) as excinfo:
        q.submit(1, "remember", _writer(99))
    assert 0.001 <= excinfo.value.retry_after <= 1.0


def test_above_high_water_is_what_readiness_reads(fake_dbs):
    q = Q.WriteQueue(fake_dbs, max_depth=10, batch_max=1, workers=1, high_water=0.5)
    assert q.above_high_water() is False
    for i in range(5):
        q.submit(1, "remember", _writer(i))
    assert q.above_high_water() is True


def test_a_stopped_queue_refuses_new_work_rather_than_accepting_it_forever(fake_dbs):
    q = Q.WriteQueue(fake_dbs, workers=1).start()
    q.stop_accepting()
    with pytest.raises(protocol.ShuttingDown):
        q.submit(1, "remember", _writer(0))
    q.stop(timeout=2.0)


# --------------------------------------------------------------------------- batching


def test_writes_queued_together_share_one_transaction(fake_dbs):
    q = Q.WriteQueue(fake_dbs, max_depth=64, batch_max=16, workers=1)
    futures = [q.submit(1, "remember", _writer(i)) for i in range(16)]
    q.start()
    assert Q.wait_all(futures, timeout=10) == list(range(16))
    db = fake_dbs.dbs[1]
    assert db.rows == list(range(16))
    assert db.transactions == 1, f"expected one transaction for the batch, got {db.transactions}"
    assert q.stats().mean_batch == 16.0
    q.stop(timeout=5.0)


def test_batch_max_bounds_how_much_shares_a_transaction(fake_dbs):
    q = Q.WriteQueue(fake_dbs, max_depth=64, batch_max=4, workers=1)
    futures = [q.submit(1, "remember", _writer(i)) for i in range(16)]
    q.start()
    Q.wait_all(futures, timeout=10)
    assert fake_dbs.dbs[1].transactions == 4
    q.stop(timeout=5.0)


def test_a_non_batchable_write_gets_a_transaction_of_its_own(fake_dbs):
    q = Q.WriteQueue(fake_dbs, max_depth=64, batch_max=16, workers=1)
    futures = [q.submit(1, "remember", _writer(i)) for i in range(4)]
    futures.append(q.submit(1, "forget", _writer("purge"), batchable=False))
    futures += [q.submit(1, "remember", _writer(i)) for i in range(4, 8)]
    q.start()
    Q.wait_all(futures, timeout=10)
    db = fake_dbs.dbs[1]
    # The four before it, the purge alone, the four after it.
    assert db.transactions == 3
    assert db.rows == [0, 1, 2, 3, "purge", 4, 5, 6, 7]
    q.stop(timeout=5.0)


def test_a_failing_write_in_a_batch_does_not_lose_or_misattribute_its_neighbours(fake_dbs):
    def boom(db):
        raise ValueError("this one write is bad")

    q = Q.WriteQueue(fake_dbs, max_depth=64, batch_max=8, workers=1)
    before = [q.submit(1, "remember", _writer(f"before-{i}")) for i in range(3)]
    bad = q.submit(1, "remember", boom)
    after = [q.submit(1, "remember", _writer(f"after-{i}")) for i in range(3)]
    q.start()
    assert [f.result(timeout=10) for f in before] == ["before-0", "before-1", "before-2"]
    assert [f.result(timeout=10) for f in after] == ["after-0", "after-1", "after-2"]
    with pytest.raises(ValueError, match="this one write is bad"):
        bad.result(timeout=10)
    db = fake_dbs.dbs[1]
    assert db.rows == ["before-0", "before-1", "before-2", "after-0", "after-1", "after-2"]
    assert db.rollbacks >= 1, "the batch has to have rolled back before it was split"
    stats = q.stats()
    assert stats.split_batches == 1
    assert stats.failed == 1
    assert stats.completed == 6
    q.stop(timeout=5.0)


def test_the_split_never_writes_a_neighbour_twice(fake_dbs):
    def boom(db):
        db.write("partial")
        raise RuntimeError("after writing")

    q = Q.WriteQueue(fake_dbs, max_depth=64, batch_max=8, workers=1)
    good = [q.submit(1, "remember", _writer(i)) for i in range(3)]
    bad = q.submit(1, "remember", boom)
    q.start()
    Q.wait_all(good, timeout=10)
    with pytest.raises(RuntimeError):
        bad.result(timeout=10)
    assert fake_dbs.dbs[1].rows == [0, 1, 2], "the aborted batch must not leave its rows behind"
    q.stop(timeout=5.0)


# --------------------------------------------------------------------------- fairness


def test_a_hot_tenant_cannot_starve_a_quiet_one(fake_dbs):
    """The deterministic proof: one worker, a flooded tenant and a small one.

    A thousand queued writes for tenant 1 must not make tenant 2 wait for all thousand.  With
    round-robin scheduling and ``batch_max`` of 4, tenant 2's five writes finish inside the
    first handful of tenant 1's batches.
    """
    order: list[tuple[int, int]] = []
    lock = threading.Lock()

    def record(tenant_id, i):
        def work(db):
            with lock:
                order.append((tenant_id, i))
            return i

        return work

    q = Q.WriteQueue(fake_dbs, max_depth=2048, batch_max=4, workers=1)
    hot = [q.submit(1, "remember", record(1, i)) for i in range(1000)]
    quiet = [q.submit(2, "remember", record(2, i)) for i in range(5)]
    q.start()
    try:
        Q.wait_all(quiet, timeout=30)
        Q.wait_all(hot, timeout=60)
    finally:
        q.stop(timeout=10.0)
    # Count what the hot tenant got through BEFORE the quiet tenant's last write ran, not what
    # it got through by the time the assertion runs.
    last_quiet = max(i for i, (t, _) in enumerate(order) if t == 2)
    hot_before = sum(1 for t, _ in order[:last_quiet] if t == 1)
    assert hot_before < 40, (
        f"the quiet tenant waited behind {hot_before} of the hot tenant's 1000 writes; "
        f"round-robin should have let it through in a handful"
    )
    assert [i for t, i in order if t == 2] == [0, 1, 2, 3, 4]
    assert [i for t, i in order if t == 1] == list(range(1000))


def test_two_tenants_at_very_different_rates_both_make_progress(fake_dbs):
    """The same property under real concurrency: a flood on one thread, a trickle on another."""
    stop = threading.Event()
    hot_done = []
    slow_latencies: list[float] = []

    def flood():
        while not stop.is_set():
            try:
                fut = q.submit(1, "remember", _writer("hot"))
            except protocol.BusyError as busy:
                time.sleep(min(0.01, busy.retry_after))
                continue
            hot_done.append(fut)

    q = Q.WriteQueue(fake_dbs, max_depth=64, batch_max=8, workers=2).start()
    thread = threading.Thread(target=flood, daemon=True)
    thread.start()
    try:
        for _ in range(20):
            started = time.monotonic()
            q.submit(2, "remember", _writer("slow")).result(timeout=15)
            slow_latencies.append(time.monotonic() - started)
            time.sleep(0.005)
    finally:
        stop.set()
        thread.join(timeout=10)
        q.stop(timeout=15.0)
    assert len(slow_latencies) == 20
    assert max(slow_latencies) < 2.0, (
        f"the quiet tenant's slowest write took {max(slow_latencies):.3f}s while the other "
        f"tenant flooded; it should never wait for more than a batch per tenant ahead of it"
    )
    assert len(hot_done) > 20, "the hot tenant should have got far more writes through"
    assert len(fake_dbs.dbs[2].rows) == 20


# --------------------------------------------------------------------------- deadlines


def test_a_write_whose_deadline_passes_in_the_queue_is_dropped_and_says_so(fake_dbs):
    q = Q.WriteQueue(fake_dbs, max_depth=64, batch_max=8, workers=1)
    doomed = q.submit(1, "remember", _writer("late"), deadline=0.01)
    time.sleep(0.05)
    fine = q.submit(1, "remember", _writer("ok"), deadline=30.0)
    q.start()
    with pytest.raises(protocol.DeadlineExceeded) as excinfo:
        doomed.result(timeout=10)
    assert excinfo.value.retryable is True
    assert fine.result(timeout=10) == "ok"
    assert fake_dbs.dbs[1].rows == ["ok"], "an expired write must not be written"
    assert q.stats().expired == 1
    q.stop(timeout=5.0)


# --------------------------------------------------------------------------- draining


def test_stop_drains_what_is_queued(fake_dbs):
    q = Q.WriteQueue(fake_dbs, max_depth=512, batch_max=4, workers=2)
    futures = [q.submit(1, "remember", _writer(i)) for i in range(100)]
    q.start()
    report = q.stop(timeout=30.0)
    assert report.clean, report
    assert report.abandoned == 0
    assert all(f.done() for f in futures)
    assert len(fake_dbs.dbs[1].rows) == 100


def test_a_drain_report_counts_what_that_drain_drained_and_not_the_queue_lifetime(fake_dbs):
    """``drained`` is a delta across the call, not the lifetime completed+failed counter.

    This was wrong: the report quoted ``self._completed + self._failed``, so a server that had
    served two hundred writes and drained three on the way out logged "shutdown drained 203
    writes cleanly" and the CLI printed the same number.  Nothing was lost, but the operational
    report was a lifetime total wearing a shutdown's name.
    """
    q = Q.WriteQueue(fake_dbs, max_depth=512, batch_max=1, workers=1).start()
    done = [q.submit(1, "remember", _writer(i)) for i in range(200)]
    for f in done:
        f.result(timeout=10)
    stats = q.stats()
    assert stats.completed + stats.failed == 200, "the lifetime counter is still the lifetime"

    gate = threading.Event()
    q.submit(1, "remember", lambda db: gate.wait(10))
    left = [q.submit(1, "remember", _writer(i)) for i in range(3)]
    threading.Timer(0.1, gate.set).start()
    report = q.stop(timeout=10.0)
    assert report.clean, report
    assert report.drained == 4, (
        f"the barrier plus three writes is four; the report said {report.drained}"
    )
    assert all(f.done() for f in left)
    assert q.stats().completed + q.stats().failed == 204, "the lifetime counter still counts all"


def test_a_stop_partitions_what_was_outstanding_into_drained_and_abandoned(fake_dbs):
    """The two numbers have to add up, or the report cannot be used to decide anything.

    Measured before the fix on a run of 400 writes: ``drained=44`` alongside 45 futures that
    completed and 355 that were abandoned, so 44 + 355 = 399 against 400 submitted.  The
    discrepancy came from sampling the counter a moment before the abandon.
    """

    def slow(db):
        time.sleep(0.004)
        return db.write("x")

    q = Q.WriteQueue(fake_dbs, max_depth=2048, batch_max=4, workers=2).start()
    # Prior traffic first, so the lifetime counter is not zero when the stop begins.  Without
    # it a fresh queue's lifetime total happens to equal the delta and the test proves nothing.
    for f in [q.submit(9, "remember", _writer(i)) for i in range(50)]:
        f.result(timeout=10)
    futures = [q.submit(1 + (i % 3), "remember", slow) for i in range(400)]
    finished_before = q.stats().completed + q.stats().failed - 50
    report = q.stop(timeout=0.2)
    assert report.abandoned > 0, "the point of this test is a stop that gives up on something"

    settled = sum(1 for f in futures if f.done() and f.exception() is None)
    abandoned = sum(1 for f in futures if isinstance(f.exception(), protocol.ShuttingDown))
    assert settled + abandoned == 400, "every submission got exactly one answer"
    assert report.abandoned == abandoned, "the report names every future it gave up on"
    assert report.drained + report.abandoned + finished_before == 400, (
        f"drained {report.drained} + abandoned {report.abandoned} + "
        f"{finished_before} already finished should account for all 400"
    )


def test_an_abandoned_write_completes_with_shutting_down_rather_than_hanging(fake_dbs):
    def slow(db):
        time.sleep(0.2)
        return db.write("slow")

    q = Q.WriteQueue(fake_dbs, max_depth=512, batch_max=1, workers=1)
    futures = [q.submit(7, "remember", slow) for _ in range(40)]
    q.start()
    report = q.stop(timeout=0.3)
    assert report.abandoned > 0
    assert report.abandoned_by_tenant.get(7) == report.abandoned
    assert all(f.done() for f in futures)
    abandoned = [f for f in futures if isinstance(f.exception(), protocol.ShuttingDown)]
    assert len(abandoned) == report.abandoned


# --------------------------------------------------------------------------- idempotency


@pytest.fixture
def real_db(tmp_path):
    with Anatid.open(tmp_path / "one.anatid", tenant=1, embedding_dim=DIM) as db:
        yield db


def _store(**kw):
    return Q.IdempotencyStore(**kw)


def test_the_table_is_created_on_demand_and_the_creation_is_idempotent(real_db):
    store = _store()
    store.ensure_table(real_db)
    store.ensure_table(real_db)
    names = {r[0] for r in real_db.execute("SHOW TABLES").fetchall()}
    assert Q.IDEMPOTENCY_TABLE in names
    assert store.count(real_db) == 0


def test_a_key_records_and_replays_the_original_result(real_db):
    store = _store()
    store.ensure_table(real_db)
    digest = store.digest_for("remember", {"content": "hello"})
    memory = real_db.remember("hello")
    store.record(real_db, 1, "k1", verb="remember", digest=digest, value=memory)
    found = store.lookup(real_db, 1, "k1", verb="remember", digest=digest)
    assert found is not None
    assert found.value() == memory
    assert found.verb == "remember"


def test_a_key_is_scoped_to_its_tenant(real_db):
    store = _store()
    store.ensure_table(real_db)
    digest = store.digest_for("remember", {"content": "hello"})
    store.record(real_db, 1, "shared", verb="remember", digest=digest, value="tenant 1 result")
    assert store.lookup(real_db, 2, "shared", verb="remember", digest=digest) is None
    assert store.lookup(real_db, 1, "shared", verb="remember", digest=digest) is not None


def test_reusing_a_key_for_a_different_request_is_refused(real_db):
    store = _store()
    store.ensure_table(real_db)
    first = store.digest_for("remember", {"content": "hello"})
    second = store.digest_for("remember", {"content": "something else"})
    store.record(real_db, 1, "k1", verb="remember", digest=first, value="r")
    with pytest.raises(protocol.IdempotencyConflict) as excinfo:
        store.lookup(real_db, 1, "k1", verb="remember", digest=second)
    assert excinfo.value.key == "k1"
    assert excinfo.value.recorded_verb == "remember"
    assert excinfo.value.retryable is False


def test_the_digest_ignores_keyword_order_and_sees_through_datetimes(real_db):
    store = _store()
    a = store.digest_for("remember", {"content": "x", "now": T0, "kind": "fact"})
    b = store.digest_for("remember", {"kind": "fact", "now": T0, "content": "x"})
    c = store.digest_for("remember", {"kind": "fact", "now": T0 + _dt.timedelta(1), "content": "x"})
    assert a == b
    assert a != c


def test_an_expired_record_is_ignored_and_then_purged(real_db):
    clock = {"now": T0}
    store = _store(ttl=60.0, now=lambda: clock["now"])
    store.ensure_table(real_db)
    digest = store.digest_for("remember", {"content": "hello"})
    store.record(real_db, 1, "k1", verb="remember", digest=digest, value="r")
    assert store.lookup(real_db, 1, "k1", verb="remember", digest=digest) is not None
    clock["now"] = T0 + _dt.timedelta(seconds=61)
    assert store.lookup(real_db, 1, "k1", verb="remember", digest=digest) is None
    assert store.count(real_db) == 1, "still on disk until it is purged"
    assert store.purge_expired(real_db) == 1
    assert store.count(real_db) == 0


def test_a_key_and_its_write_commit_together(real_db):
    """The record is written in the write's transaction, so a rollback takes both."""
    store = _store()
    store.ensure_table(real_db)
    digest = store.digest_for("remember", {"content": "doomed"})
    before = real_db.execute("SELECT count(*) FROM memories").fetchone()[0]
    with pytest.raises(RuntimeError), real_db.transaction():
        memory = real_db.remember("doomed")
        store.record(real_db, 1, "k1", verb="remember", digest=digest, value=memory)
        raise RuntimeError("something went wrong after both writes")
    assert real_db.execute("SELECT count(*) FROM memories").fetchone()[0] == before
    assert store.count(real_db) == 0


def test_the_queue_replays_a_key_instead_of_writing_twice(real_db):
    store = _store()
    store.ensure_table(real_db)
    q = Q.WriteQueue(lambda _t: real_db, max_depth=64, batch_max=1, workers=1, idempotency=store)
    digest = store.digest_for("remember", {"content": "once"})

    def work(db):
        return db.remember("once")

    q.start()
    try:
        first = q.submit(1, "remember", work, idempotency_key="k1", digest=digest).result(10)
        second = q.submit(1, "remember", work, idempotency_key="k1", digest=digest).result(10)
    finally:
        q.stop(timeout=10.0)
    assert first.memory_id == second.memory_id
    assert second == first
    rows = real_db.execute("SELECT count(*) FROM memories WHERE content = 'once'").fetchone()[0]
    assert rows == 1
    assert q.stats().replayed == 1


def test_a_failed_write_is_not_recorded_so_its_retry_runs(real_db):
    store = _store()
    store.ensure_table(real_db)
    q = Q.WriteQueue(lambda _t: real_db, max_depth=64, batch_max=1, workers=1, idempotency=store)
    attempts = {"n": 0}

    def flaky(db):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ValueError("the first attempt fails")
        return db.remember("eventually")

    digest = store.digest_for("remember", {"content": "eventually"})
    q.start()
    try:
        with pytest.raises(ValueError):
            q.submit(1, "remember", flaky, idempotency_key="k1", digest=digest).result(10)
        memory = q.submit(1, "remember", flaky, idempotency_key="k1", digest=digest).result(10)
    finally:
        q.stop(timeout=10.0)
    assert memory.content == "eventually"
    assert attempts["n"] == 2
    assert store.count(real_db) == 1


# --------------------------------------------------------------------------- the server


@pytest.fixture
def pool(tmp_path):
    with DatabasePool(
        str(tmp_path / "tenants" / "t_{tenant}.anatid"), embedding_dim=DIM, max_open=8
    ) as p:
        yield p


@pytest.fixture
def server(pool, sock_dir):
    config = S.ServerConfig(
        socket_path=sock_dir / "run" / "anatid.sock",
        max_depth=32,
        batch_max=8,
        workers=2,
        read_workers=2,
        tenants=(1, 2),
        shutdown_timeout=10.0,
    )
    srv = S.AnatidServer(pool=pool, config=config)
    srv.queue.start()
    srv.open_tenants()
    srv._status = "serving"  # start() also binds sockets; these tests do not need one
    try:
        yield srv
    finally:
        srv.queue.stop(timeout=10.0)
        srv._status = "stopped"


ANY_TENANT = auth.Principal(name="test", tenants=None)
ONLY_TENANT_1 = auth.Principal.for_tenants("only-1", [1])


def test_a_write_and_a_read_go_through_the_dispatcher(server):
    written = server.call(
        protocol.Request(verb="remember", tenant=1, args={"content": "Ada likes coffee"}),
        ANY_TENANT,
    ).raise_for_status()
    read = server.call(
        protocol.Request(verb="get", tenant=1, args={"memory_id": written.memory_id}), ANY_TENANT
    ).raise_for_status()
    assert read.content == "Ada likes coffee"
    assert read.memory_id == written.memory_id


def test_a_principal_cannot_touch_a_tenant_it_is_not_authorized_for(server, pool):
    response = server.call(
        protocol.Request(verb="remember", tenant=2, args={"content": "not yours"}), ONLY_TENANT_1
    )
    assert response.status is protocol.Status.ERROR
    assert response.error is not None
    assert response.error.error_class == "AuthorizationError"
    assert response.retryable is False
    with pytest.raises(protocol.AuthorizationError):
        response.raise_for_status()
    # Tenant 2's file must be untouched: the check runs before a handle is resolved.
    tenant_2 = pool.get(2)
    assert tenant_2.execute("SELECT count(*) FROM memories").fetchone()[0] == 0


def test_the_refusal_does_not_reveal_whether_the_tenant_exists(server):
    exists = server.call(
        protocol.Request(verb="get", tenant=2, args={"memory_id": 1}), ONLY_TENANT_1
    )
    absent = server.call(
        protocol.Request(verb="get", tenant=9999, args={"memory_id": 1}), ONLY_TENANT_1
    )
    assert exists.error is not None and absent.error is not None
    assert exists.error.error_class == absent.error.error_class == "AuthorizationError"
    assert exists.error.message.replace("2", "N") == absent.error.message.replace("9999", "N")


def test_a_read_is_refused_for_an_unauthorized_tenant_too(server):
    response = server.call(
        protocol.Request(verb="recall", tenant=2, args={"query": "coffee"}), ONLY_TENANT_1
    )
    assert response.error is not None
    assert response.error.error_class == "AuthorizationError"


def test_a_tenant_smuggled_into_args_is_refused(server):
    response = server.call(
        protocol.Request(verb="get", tenant=1, args={"memory_id": 1, "tenant": 2}), ANY_TENANT
    )
    assert response.error is not None
    assert response.error.error_class == "ProtocolError"
    assert "envelope" in response.error.message


def test_an_unknown_verb_is_refused_and_names_what_is_available(server):
    response = server.call(protocol.Request(verb="unsafe_connection", tenant=1), ANY_TENANT)
    assert response.error is not None
    assert response.error.error_class == "ProtocolError"
    assert "unsafe_connection" in response.error.message
    assert "remember" in response.error.message


def test_a_read_only_principal_cannot_write(server):
    read_only = auth.Principal(name="ro", tenants=None, read_only=True)
    assert (
        server.call(protocol.Request(verb="get", tenant=1, args={"memory_id": 1}), read_only).status
        is protocol.Status.OK
    )
    response = server.call(
        protocol.Request(verb="remember", tenant=1, args={"content": "nope"}), read_only
    )
    assert response.error is not None
    assert response.error.error_class == "AuthorizationError"


def test_a_verb_error_comes_back_structured_not_as_a_dropped_call(server):
    response = server.call(
        protocol.Request(verb="remember", tenant=1, args={"content": "x", "confidence": 5.0}),
        ANY_TENANT,
    )
    assert response.status is protocol.Status.ERROR
    assert response.error is not None
    assert response.error.retryable is False
    with pytest.raises(Exception) as excinfo:
        response.raise_for_status()
    assert "confidence" in str(excinfo.value).lower()


def test_a_conflict_survives_the_dispatcher_with_its_versions(server):
    memory = server.call(
        protocol.Request(verb="remember", tenant=1, args={"content": "v1"}), ANY_TENANT
    ).raise_for_status()
    response = server.call(
        protocol.Request(
            verb="update",
            tenant=1,
            args={"memory_id": memory.memory_id, "content": "v2", "expected_version": 99},
        ),
        ANY_TENANT,
    )
    assert response.error is not None
    assert response.error.error_class == "ConflictError"
    assert response.error.expected_version == 99
    assert response.error.current_version == 1
    assert response.retryable is False


def test_a_retry_after_a_lost_response_writes_once(server):
    """The sequence the whole idempotency mechanism exists for.

    Send a write.  Lose the response before the client sees it.  Retry with the same key.  There
    must be exactly one row, and the retry must return the original result.
    """
    request = protocol.Request(
        verb="remember",
        tenant=1,
        args={"content": "the write that was retried"},
        idempotency_key="order-42",
    )
    first = server.call(request, ANY_TENANT)
    original = first.raise_for_status()
    del first  # the client never saw it

    retry = protocol.Request(
        verb="remember",
        tenant=1,
        args=dict(request.args),
        idempotency_key="order-42",
    )
    replayed = server.call(retry, ANY_TENANT).raise_for_status()

    db = server._open_db(1)
    rows = db.execute(
        "SELECT count(*) FROM memories WHERE content = 'the write that was retried'"
    ).fetchone()[0]
    assert rows == 1, f"the retry wrote a second row: {rows} rows exist"
    assert replayed.memory_id == original.memory_id
    assert replayed == original
    assert server.queue.stats().replayed == 1


def test_a_retry_after_a_server_restart_still_writes_once(pool, sock_dir):
    """Persistence is the point: an in-memory key table would let a restart duplicate the write."""
    config = S.ServerConfig(socket_path=sock_dir / "run" / "a.sock", tenants=(1,), workers=1)

    def build():
        srv = S.AnatidServer(pool=pool, config=config)
        srv.queue.start()
        srv.open_tenants()
        srv._status = "serving"
        return srv

    request = protocol.Request(
        verb="remember",
        tenant=1,
        args={"content": "survives a restart"},
        idempotency_key="order-43",
    )
    first_server = build()
    try:
        original = first_server.call(request, ANY_TENANT).raise_for_status()
    finally:
        first_server.queue.stop(timeout=10.0)

    second_server = build()
    try:
        replayed = second_server.call(
            protocol.Request(
                verb="remember", tenant=1, args=dict(request.args), idempotency_key="order-43"
            ),
            ANY_TENANT,
        ).raise_for_status()
    finally:
        second_server.queue.stop(timeout=10.0)

    db = pool.get(1)
    rows = db.execute(
        "SELECT count(*) FROM memories WHERE content = 'survives a restart'"
    ).fetchone()[0]
    assert rows == 1
    assert replayed == original


def test_a_key_reused_with_different_arguments_is_refused_by_the_dispatcher(server):
    server.call(
        protocol.Request(verb="remember", tenant=1, args={"content": "first"}, idempotency_key="k"),
        ANY_TENANT,
    ).raise_for_status()
    response = server.call(
        protocol.Request(
            verb="remember", tenant=1, args={"content": "second"}, idempotency_key="k"
        ),
        ANY_TENANT,
    )
    assert response.error is not None
    assert response.error.error_class == "IdempotencyConflict"
    db = server._open_db(1)
    assert db.execute("SELECT count(*) FROM memories WHERE content = 'second'").fetchone()[0] == 0


def test_keys_do_not_leak_between_tenants_over_the_dispatcher(server):
    one = server.call(
        protocol.Request(verb="remember", tenant=1, args={"content": "same"}, idempotency_key="k"),
        ANY_TENANT,
    ).raise_for_status()
    two = server.call(
        protocol.Request(verb="remember", tenant=2, args={"content": "same"}, idempotency_key="k"),
        ANY_TENANT,
    ).raise_for_status()
    assert one.tenant_id == 1
    assert two.tenant_id == 2
    for tenant_id in (1, 2):
        db = server._open_db(tenant_id)
        assert db.execute("SELECT count(*) FROM memories").fetchone()[0] == 1


# --------------------------------------------------------------------------- health


def test_health_and_readiness_are_different_questions(server):
    health = server.health()
    assert health.ok is True
    assert health.status == "serving"
    assert health.pid == os.getpid()
    assert health.protocol_version == protocol.PROTOCOL_VERSION
    ready = server.readiness()
    assert ready.ready is True
    assert ready.migrations_done is True
    assert set(ready.tenants) == {1, 2}
    assert ready.open_files >= 2


def test_a_server_under_backpressure_is_healthy_and_not_ready(pool, sock_dir):
    """A busy server must not be restarted, and must not be sent more traffic.

    Health and readiness answering the same question would force one of those two mistakes.
    """
    config = S.ServerConfig(
        socket_path=sock_dir / "bp.sock", tenants=(1,), max_depth=8, high_water=0.5, workers=1
    )
    srv = S.AnatidServer(pool=pool, config=config)
    srv.open_tenants()
    srv._status = "serving"
    # The queue accepts from construction and the workers are not started, so the writes pile up
    # exactly as they would behind a server that cannot keep up.
    for i in range(5):
        srv.submit_write(
            protocol.Request(verb="remember", tenant=1, args={"content": f"backlog {i}"}),
            ANY_TENANT,
        )
    assert srv.queue.depth(1) == 5
    assert srv.health().ok is True, "a busy server is alive; restarting it would lose the backlog"
    ready = srv.readiness()
    assert ready.ready is False
    assert ready.queues_below_high_water is False
    assert ready.max_tenant_depth == 5
    assert "high-water" in ready.detail
    srv.queue.stop(timeout=5.0, drain=False)


def test_readiness_is_false_while_a_tenants_file_is_not_open(pool, sock_dir):
    config = S.ServerConfig(socket_path=sock_dir / "x.sock", tenants=(1, 2, 3))
    srv = S.AnatidServer(pool=pool, config=config)
    assert srv.readiness().ready is False
    assert "no open file" in srv.readiness().detail
    srv.open_tenants()
    srv.queue.start()
    srv._status = "serving"
    try:
        assert srv.readiness().ready is True
    finally:
        srv.queue.stop(timeout=5.0)


def test_health_and_ready_are_reachable_as_verbs(server):
    health = server.call(protocol.Request(verb="health", tenant=1), ANY_TENANT).raise_for_status()
    ready = server.call(protocol.Request(verb="ready", tenant=1), ANY_TENANT).raise_for_status()
    stats = server.call(
        protocol.Request(verb="queue_stats", tenant=1), ANY_TENANT
    ).raise_for_status()
    assert isinstance(health, S.Health) and health.ok
    assert isinstance(ready, S.Readiness)
    assert isinstance(stats, Q.QueueStats)
    # and they survive the wire
    assert protocol.loads(protocol.dumps(health)) == health
    assert protocol.loads(protocol.dumps(ready)) == ready


# --------------------------------------------------------------------------- auth


def test_a_non_loopback_http_bind_without_a_token_is_refused():
    with pytest.raises(ValueError) as excinfo:
        auth.check_bind_address("0.0.0.0", auth.AllowAllAuthenticator())
    assert "0.0.0.0" in str(excinfo.value)
    assert "BearerTokenAuthenticator" in str(excinfo.value)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.5"])
def test_loopback_binds_are_allowed_without_a_token(host):
    assert auth.is_loopback(host) is True
    auth.check_bind_address(host, auth.AllowAllAuthenticator())


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "", "192.168.1.10", "example.com"])
def test_non_loopback_hosts_are_recognised_as_such(host):
    assert auth.is_loopback(host) is False


def test_a_non_loopback_http_bind_with_a_token_is_allowed():
    authenticator = auth.BearerTokenAuthenticator({"s3cret": [1]})
    auth.check_bind_address("0.0.0.0", authenticator)


def test_the_server_refuses_to_construct_with_a_public_bind_and_no_token(pool, sock_dir):
    with pytest.raises(ValueError, match="refusing to bind"):
        S.AnatidServer(
            pool=pool,
            config=S.ServerConfig(socket_path=sock_dir / "s.sock", http_host="0.0.0.0"),
        )


def test_bearer_tokens_map_to_principals_with_their_own_tenants():
    authenticator = auth.BearerTokenAuthenticator({"tok-a": [1], "tok-b": [2, 3]})
    ctx = auth.ConnectionContext(transport="http", headers={"authorization": "Bearer tok-a"})
    principal = authenticator.authenticate(ctx)
    assert principal.may(1) is True
    assert principal.may(2) is False
    with pytest.raises(protocol.AuthorizationError):
        principal.require(2)


@pytest.mark.parametrize("header", [None, "", "Basic tok-a", "Bearer", "Bearer wrong", "tok-a"])
def test_a_missing_or_wrong_bearer_token_is_refused(header):
    authenticator = auth.BearerTokenAuthenticator({"tok-a": None})
    headers = {} if header is None else {"authorization": header}
    with pytest.raises(protocol.AuthenticationError):
        authenticator.authenticate(auth.ConnectionContext(transport="http", headers=headers))


def test_the_failure_never_echoes_the_token():
    authenticator = auth.BearerTokenAuthenticator({"tok-a": None})
    with pytest.raises(protocol.AuthenticationError) as excinfo:
        authenticator.authenticate(
            auth.ConnectionContext(transport="http", headers={"authorization": "Bearer guessed"})
        )
    assert "guessed" not in str(excinfo.value)


def test_an_empty_token_map_is_a_startup_error_not_a_runtime_one():
    with pytest.raises(ValueError, match="at least one token"):
        auth.BearerTokenAuthenticator({})


@pytest.mark.parametrize("token", [" s3cret", "s3cret ", " s3cret ", "s3cret\n", "\ts3cret"])
def test_a_token_with_surrounding_whitespace_is_refused_at_construction(token):
    """It could never authenticate, so it is a startup error and not a mystery at runtime.

    ``authenticate`` strips the value it takes off the header, because a client that appends a
    newline is routine.  A configured token that is not already stripped therefore matches
    nothing, and the server used to start happily and answer every request with "the bearer
    token was not accepted", which tells the operator nothing about the cause.
    """
    with pytest.raises(ValueError, match="whitespace"):
        auth.BearerTokenAuthenticator({token: None})


def test_a_token_with_a_space_inside_it_is_refused_at_construction():
    """A space ends the value in an ``Authorization`` header, so the token cannot survive it."""
    with pytest.raises(ValueError, match="whitespace or control characters"):
        auth.BearerTokenAuthenticator({"two words": None})


def test_a_token_offered_with_stray_whitespace_still_authenticates():
    """The stripping that made the above a defect is deliberate and stays.

    Shells, config files and client libraries all add a trailing newline, and refusing a token
    over one would be a worse failure than accepting it.
    """
    authenticator = auth.BearerTokenAuthenticator({"s3cret": [1]})
    for offered in ("Bearer s3cret", "Bearer s3cret\n", "Bearer  s3cret  "):
        principal = authenticator.authenticate(
            auth.ConnectionContext(transport="http", headers={"authorization": offered})
        )
        assert principal.tenants == frozenset({1})


def test_unix_peer_credentials_identify_the_connecting_process(sock_dir):
    path = sock_dir / "peer.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)

    def connect():
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(path))
        client.recv(1)
        client.close()

    thread = threading.Thread(target=connect, daemon=True)
    thread.start()
    conn, _ = listener.accept()
    try:
        creds = auth.peer_credentials(conn)
        if creds is None:
            pytest.skip("this platform does not report Unix peer credentials")
        assert creds.uid == os.getuid()
        principal = auth.UnixPeerAuthenticator(default_tenants=[1]).authenticate(
            auth.ConnectionContext(transport="unix", socket=conn)
        )
        assert principal.may(1) is True
        assert principal.may(2) is False
        assert principal.peer is not None and principal.peer.uid == os.getuid()
    finally:
        conn.sendall(b"x")
        conn.close()
        listener.close()
        thread.join(timeout=5)


def test_a_uid_that_is_not_allowed_cannot_connect(sock_dir):
    path = sock_dir / "peer2.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)

    def connect():
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(path))
        client.recv(1)
        client.close()

    thread = threading.Thread(target=connect, daemon=True)
    thread.start()
    conn, _ = listener.accept()
    try:
        if auth.peer_credentials(conn) is None:
            pytest.skip("this platform does not report Unix peer credentials")
        authenticator = auth.UnixPeerAuthenticator(allow_uids=[os.getuid() + 10_000])
        with pytest.raises(protocol.AuthenticationError):
            authenticator.authenticate(auth.ConnectionContext(transport="unix", socket=conn))
    finally:
        conn.sendall(b"x")
        conn.close()
        listener.close()
        thread.join(timeout=5)


def test_a_server_needs_exactly_one_of_pool_or_database(pool, tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        S.AnatidServer(config=S.ServerConfig(socket_path=tmp_path / "s.sock"))
    with (
        Anatid.open(tmp_path / "one.anatid", tenant=1, embedding_dim=DIM) as db,
        pytest.raises(ValueError, match="exactly one"),
    ):
        S.AnatidServer(
            pool=pool, database=db, config=S.ServerConfig(socket_path=tmp_path / "s.sock")
        )


def test_a_config_with_no_transport_is_refused():
    with pytest.raises(ValueError, match="at least one transport"):
        S.ServerConfig()


# --------------------------------------------------------------------------- end to end


def _run(coro):
    return asyncio.run(coro)


def test_a_client_talks_to_a_real_unix_socket(pool, sock_dir):
    sock_path = sock_dir / "run" / "e2e.sock"

    async def scenario():
        server = S.AnatidServer(
            pool=pool,
            config=S.ServerConfig(socket_path=sock_path, tenants=(1,), workers=1),
        )
        await server.start()
        try:
            assert oct(os.stat(sock_path).st_mode & 0o777) == oct(auth.SOCKET_MODE)
            assert oct(os.stat(sock_path.parent).st_mode & 0o777) == oct(auth.SOCKET_DIR_MODE)
            reader, writer = await asyncio.open_unix_connection(str(sock_path))
            try:
                await protocol.write_frame_async(
                    writer,
                    protocol.Request(
                        verb="remember", tenant=1, args={"content": "over the socket"}
                    ),
                )
                body = await protocol.read_frame_async(reader)
                assert body is not None
                memory = protocol.Response.decode(body).raise_for_status()
                assert memory.content == "over the socket"

                await protocol.write_frame_async(
                    writer,
                    protocol.Request(verb="get", tenant=1, args={"memory_id": memory.memory_id}),
                )
                body = await protocol.read_frame_async(reader)
                assert body is not None
                assert protocol.Response.decode(body).raise_for_status().content == (
                    "over the socket"
                )

                await protocol.write_frame_async(writer, protocol.Request(verb="health", tenant=1))
                body = await protocol.read_frame_async(reader)
                assert body is not None
                assert protocol.Response.decode(body).raise_for_status().ok is True
            finally:
                writer.close()
                await writer.wait_closed()
        finally:
            report = await server.shutdown()
        assert report.abandoned == 0
        assert not sock_path.exists(), "shutdown removes its socket"

    _run(scenario())


def test_the_socket_client_helper_speaks_the_same_frames(pool, sock_dir):
    sock_path = sock_dir / "run" / "sync.sock"
    result: dict[str, object] = {}

    async def scenario():
        server = S.AnatidServer(
            pool=pool, config=S.ServerConfig(socket_path=sock_path, tenants=(1,), workers=1)
        )
        await server.start()

        def client():
            sock = S.connect_unix(sock_path)
            try:
                sock.sendall(
                    protocol.Request(
                        verb="remember", tenant=1, args={"content": "from a blocking client"}
                    ).encode()
                )
                body = protocol.read_frame(sock)
                assert body is not None
                result["memory"] = protocol.Response.decode(body).raise_for_status()
            finally:
                sock.close()

        await asyncio.get_running_loop().run_in_executor(None, client)
        await server.shutdown()

    _run(scenario())
    memory = result["memory"]
    assert memory.content == "from a blocking client"  # type: ignore[union-attr]


def test_http_serves_health_ready_and_rpc_on_loopback(pool):
    token = "test-token"
    body: dict[str, object] = {}

    async def scenario():
        server = S.AnatidServer(
            pool=pool,
            config=S.ServerConfig(
                socket_path=None, http_host="127.0.0.1", http_port=0, tenants=(1,), workers=1
            ),
            authenticator=auth.BearerTokenAuthenticator({token: [1]}),
        )
        # port 0 asks the OS for a free port; read it back off the bound socket
        await server.start()
        sockets = server._servers[0].sockets
        port = sockets[0].getsockname()[1]
        try:
            body["health"] = await _http(port, "GET", "/health")
            body["ready"] = await _http(port, "GET", "/ready")
            body["unauthorized"] = await _http(
                port,
                "POST",
                "/rpc",
                protocol.dumps(
                    protocol.Request(verb="remember", tenant=1, args={"content": "x"}).to_wire()
                ),
            )
            body["ok"] = await _http(
                port,
                "POST",
                "/rpc",
                protocol.dumps(
                    protocol.Request(
                        verb="remember", tenant=1, args={"content": "over http"}
                    ).to_wire()
                ),
                token=token,
            )
            body["forbidden"] = await _http(
                port,
                "POST",
                "/rpc",
                protocol.dumps(
                    protocol.Request(verb="remember", tenant=2, args={"content": "x"}).to_wire()
                ),
                token=token,
            )
            body["missing"] = await _http(port, "GET", "/nope")
        finally:
            await server.shutdown()

    _run(scenario())
    assert body["health"][0] == 200
    assert body["ready"][0] == 200
    assert body["unauthorized"][0] == 401
    assert body["ok"][0] == 200
    assert (
        protocol.Response.from_wire(protocol.loads(body["ok"][1])).raise_for_status().content
        == "over http"
    )
    assert body["forbidden"][0] == 403
    assert body["missing"][0] == 404


async def _http(port, method, path, payload=b"", token=None):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        head = f"{method} {path} HTTP/1.1\r\nhost: localhost\r\ncontent-length: {len(payload)}\r\n"
        if token:
            head += f"authorization: Bearer {token}\r\n"
        head += "connection: close\r\n\r\n"
        writer.write(head.encode("latin-1") + payload)
        await writer.drain()
        raw = await reader.read()
    finally:
        writer.close()
        await writer.wait_closed()
    header, _, body = raw.partition(b"\r\n\r\n")
    status = int(header.split(b" ")[1])
    return status, body


def test_graceful_shutdown_drains_queued_writes(pool, sock_dir):
    sock_path = sock_dir / "run" / "drain.sock"

    async def scenario():
        server = S.AnatidServer(
            pool=pool,
            config=S.ServerConfig(
                socket_path=sock_path,
                tenants=(1,),
                workers=1,
                batch_max=4,
                max_depth=256,
                shutdown_timeout=30.0,
            ),
        )
        await server.start()
        futures = [
            server.submit_write(
                protocol.Request(verb="remember", tenant=1, args={"content": f"m{i}"}),
                ANY_TENANT,
            )
            for i in range(50)
        ]
        report = await server.shutdown()
        assert report.abandoned == 0, report
        assert all(f.done() and f.exception() is None for f in futures)
        return server

    server = _run(scenario())
    assert server.health().status == "stopped"
    assert pool.get(1).execute("SELECT count(*) FROM memories").fetchone()[0] == 50


def test_a_write_arriving_during_shutdown_is_refused_clearly(pool, sock_dir):
    async def scenario():
        server = S.AnatidServer(
            pool=pool,
            config=S.ServerConfig(socket_path=sock_dir / "run" / "late.sock", tenants=(1,)),
        )
        await server.start()
        await server.shutdown()
        return server.call(
            protocol.Request(verb="remember", tenant=1, args={"content": "too late"}), ANY_TENANT
        )

    response = _run(scenario())
    assert response.error is not None
    assert response.error.error_class == "ShuttingDown"
    assert response.retryable is True


def test_shutdown_checkpoints_so_the_next_process_has_no_wal_to_replay(pool, sock_dir):
    async def scenario():
        server = S.AnatidServer(
            pool=pool,
            config=S.ServerConfig(socket_path=sock_dir / "run" / "cp.sock", tenants=(1,)),
        )
        await server.start()
        server.call(
            protocol.Request(verb="remember", tenant=1, args={"content": "checkpoint me"}),
            ANY_TENANT,
        ).raise_for_status()
        await server.shutdown()

    _run(scenario())
    path = pool.path_for(1)
    wal = path.with_name(path.name + ".wal")
    assert not wal.exists() or wal.stat().st_size == 0


# --------------------------------------------------------------------------- the constraint


def test_a_second_process_cannot_open_a_file_the_server_holds(pool):
    """The measurement the whole server profile rests on, as an executable fact.

    DuckDB takes an exclusive lock on a file opened read-write.  Another process cannot open it
    read-write, and -- the part that decides the design -- cannot open it READ-ONLY either.  So
    there is no arrangement where clients write through the server and read the file directly:
    every read comes back over the protocol too.  If this test ever starts failing because
    DuckDB relaxed the lock, the direct-read fast path becomes available and this module's
    reasoning should be revisited.
    """
    import subprocess
    import sys

    db = pool.get(1)
    db.remember("held by this process")
    path = str(pool.path_for(1))
    script = (
        "import duckdb, sys\n"
        f"try:\n"
        f"    duckdb.connect({path!r}, read_only=True).execute('SELECT 1').fetchone()\n"
        "    print('OPENED')\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__ + '|' + str(exc).splitlines()[0])\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    ).stdout.strip()
    assert out != "OPENED", (
        "a second process opened read-only a file this process holds read-write; DuckDB's "
        "lock has changed and the server's 'every read comes over the wire' rule can be relaxed"
    )
    name, _, message = out.partition("|")
    assert name == "IOException", out
    assert "Could not set lock on file" in message, out


def test_install_migration_registers_the_ddl_and_refuses_a_collision():
    """The hook for the release that raises ``SCHEMA_VERSION``.

    The table is created by idempotent DDL today, because ``SCHEMA_VERSION`` is 4 and a step
    registered for 5 would never run.  This asserts the hook works, so the release that does
    bump the constant only has to call it.
    """
    from anatid import schema as schema_mod

    version = 9999
    assert version not in schema_mod.MIGRATIONS
    try:
        Q.install_migration(version)
        assert version in schema_mod.MIGRATIONS
        Q.install_migration(version)  # calling twice is a no-op, not a collision

        with Anatid.open(":memory:", tenant=1, embedding_dim=DIM) as db:
            names = {r[0] for r in db.execute("SHOW TABLES").fetchall()}
            assert Q.IDEMPOTENCY_TABLE not in names
            schema_mod.MIGRATIONS[version](db.connection)
            names = {r[0] for r in db.execute("SHOW TABLES").fetchall()}
            assert Q.IDEMPOTENCY_TABLE in names

        def someone_else(_con):
            pass

        schema_mod.MIGRATIONS[version] = someone_else
        with pytest.raises(ValueError, match="already registered"):
            Q.install_migration(version)
    finally:
        schema_mod.MIGRATIONS.pop(version, None)


def test_the_idempotency_ddl_is_not_registered_at_import_time():
    """A migration step that never runs would let a file claim a version it does not have."""
    from anatid import schema as schema_mod

    for version, step in schema_mod.MIGRATIONS.items():
        assert not getattr(step, "anatid_idempotency_migration", False), (
            f"the idempotency DDL is registered as the step to schema version {version}; it "
            f"must not be until SCHEMA_VERSION moves to that number in the same change"
        )


def test_sigterm_drains_and_exits(sock_dir, tmp_path):
    """A real process, a real SIGTERM, and the writes that were queued when it arrived.

    In a subprocess rather than in-process, because the thing under test is a signal reaching a
    running event loop and the process exiting afterwards, and because a signal handler
    installed inside the test runner is a hazard to the test runner.
    """
    import subprocess
    import sys

    sock_path = sock_dir / "sigterm.sock"
    template = str(tmp_path / "tenants" / "t_{tenant}.anatid")
    script = f"""
import asyncio, os, signal, sys
sys.path.insert(0, {str(Path(__file__).resolve().parent.parent / "src")!r})
from anatid import DatabasePool
from anatid.server import AnatidServer, ServerConfig
from anatid.server.protocol import Request


async def main():
    pool = DatabasePool({template!r}, embedding_dim=8)
    server = AnatidServer(
        pool=pool,
        config=ServerConfig(
            socket_path={str(sock_path)!r}, tenants=(1,), workers=1, batch_max=4,
            max_depth=512, shutdown_timeout=30.0,
        ),
    )
    await server.start()
    server.install_signal_handlers()
    from anatid.server.auth import Principal

    everyone = Principal(name="local", tenants=None)
    for i in range(60):
        server.submit_write(
            Request(verb="remember", tenant=1, args={{"content": "queued %d" % i}}), everyone
        )
    print("QUEUED", server.queue.stats().queued, flush=True)
    os.kill(os.getpid(), signal.SIGTERM)
    report = await server.serve_forever()
    print("DRAINED", report.drained, report.abandoned, flush=True)
    rows = pool.get(1).execute("SELECT count(*) FROM memories").fetchone()[0]
    print("ROWS", rows, flush=True)
    pool.close_all()


asyncio.run(main())
"""
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert done.returncode == 0, done.stderr[-3000:]
    lines = dict(
        (parts[0], parts[1:]) for parts in (ln.split() for ln in done.stdout.splitlines()) if parts
    )
    assert "DRAINED" in lines, done.stdout
    assert lines["DRAINED"][1] == "0", f"SIGTERM abandoned writes it should have drained: {lines}"
    assert lines["ROWS"] == ["60"], f"the drain did not write every queued memory: {lines}"
    assert not sock_path.exists(), "a clean shutdown removes its socket"


# --------------------------------------------------------------------------- reply encoding


def test_stats_and_info_cross_the_wire(server):
    """Both return a mapping with an ``ExpandPath`` in it, which is not a dataclass.

    These were the two verbs the dispatch table exposed that could not be answered: the verb ran,
    the encoder raised on the value it returned, and the connection died without a reply.
    """
    server.call(
        protocol.Request(verb="remember", tenant=1, args={"content": "counted"}), ANY_TENANT
    ).raise_for_status()
    stats = server.call(protocol.Request(verb="stats", tenant=1), ANY_TENANT)
    body = stats.encode()
    back = protocol.Response.decode(body[4:]).raise_for_status()
    assert back["memories"] == 1
    assert back["expand_path"] == stats.result["expand_path"]
    assert back["expand_path"].reason is stats.result["expand_path"].reason

    info = server.call(protocol.Request(verb="info", tenant=1), ANY_TENANT)
    again = protocol.Response.decode(info.encode()[4:]).raise_for_status()
    assert again == info.result


def test_a_reply_the_codec_cannot_encode_becomes_an_error_not_a_dead_connection(pool, sock_dir):
    """A verb that returns something unencodable is a bug in the server, not in the request.

    The client has to be told, because the verb has already run: for a write, silence means a
    retry that cannot know it is a retry.  So the failure comes back as a response and the
    connection stays open for the next request.
    """
    sock_path = sock_dir / "run" / "badenc.sock"

    class Unencodable:
        pass

    async def scenario():
        server = S.AnatidServer(
            pool=pool, config=S.ServerConfig(socket_path=sock_path, tenants=(1,), workers=1)
        )
        S.VERBS["_test_unencodable"] = S.VerbSpec(
            name="_test_unencodable", server=True, tenant_arg=False, summary="test only"
        )
        original = server._server_verb
        server._server_verb = lambda spec, principal: (  # type: ignore[method-assign]
            Unencodable() if spec.name == "_test_unencodable" else original(spec, principal)
        )
        await server.start()
        try:
            reader, writer = await asyncio.open_unix_connection(str(sock_path))
            try:
                await protocol.write_frame_async(
                    writer, protocol.Request(verb="_test_unencodable", tenant=1)
                )
                body = await protocol.read_frame_async(reader)
                assert body is not None, "the connection was dropped instead of answering"
                response = protocol.Response.decode(body)
                assert response.status is protocol.Status.ERROR
                assert response.error is not None
                assert "could not encode" in response.error.message
                assert "Unencodable" in response.error.message

                # and the connection is still usable
                await protocol.write_frame_async(writer, protocol.Request(verb="health", tenant=1))
                body = await protocol.read_frame_async(reader)
                assert body is not None
                assert protocol.Response.decode(body).raise_for_status().ok is True
            finally:
                writer.close()
                await writer.wait_closed()
        finally:
            S.VERBS.pop("_test_unencodable", None)
            await server.shutdown()

    _run(scenario())


# --------------------------------------------------------------------------- file release


def test_shutdown_folds_the_wal_on_a_file_that_already_existed(pool, sock_dir):
    """The path a restart actually takes, which a fresh file does not exercise.

    ``Anatid.open`` on an EXISTING file leaves a transaction behind that makes
    ``db.execute("CHECKPOINT")`` raise ``TransactionException`` from any thread, so a shutdown
    that checkpointed that way logged a warning and left the log for the next process.  Closing
    the handle folds it.
    """
    # Create the file and close it, so the server's open is a REOPEN.
    first = pool.get(1)
    first.remember("written before the server started")
    assert pool.close(1)

    async def scenario():
        server = S.AnatidServer(
            pool=pool,
            config=S.ServerConfig(socket_path=sock_dir / "run" / "wal.sock", tenants=(1,)),
        )
        await server.start()
        try:
            reader, writer = await asyncio.open_unix_connection(str(sock_dir / "run" / "wal.sock"))
            try:
                for verb, args in (("remember", {"content": "during"}), ("stats", {})):
                    await protocol.write_frame_async(
                        writer, protocol.Request(verb=verb, tenant=1, args=args)
                    )
                    body = await protocol.read_frame_async(reader)
                    assert body is not None
                    protocol.Response.decode(body).raise_for_status()
            finally:
                writer.close()
                await writer.wait_closed()
        finally:
            await server.shutdown()

    _run(scenario())
    path = pool.path_for(1)
    wal = path.with_name(path.name + ".wal")
    assert not wal.exists() or wal.stat().st_size == 0, (
        f"shutdown left a {wal.stat().st_size} byte write-ahead log for the next process"
    )


def test_shutdown_releases_the_files_so_a_replacement_process_can_open_them(pool, sock_dir):
    """A stopped server that still held the file would make an in-place restart impossible.

    DuckDB's lock is exclusive, so a successor cannot open the file even read-only while the
    predecessor holds it.  Shutting down has to give it back.
    """
    import subprocess
    import sys

    async def scenario():
        server = S.AnatidServer(
            pool=pool,
            config=S.ServerConfig(socket_path=sock_dir / "run" / "rel.sock", tenants=(1,)),
        )
        await server.start()
        server.call(
            protocol.Request(verb="remember", tenant=1, args={"content": "held"}), ANY_TENANT
        ).raise_for_status()
        await server.shutdown()

    _run(scenario())
    path = str(pool.path_for(1))
    script = (
        "import duckdb\n"
        f"con = duckdb.connect({path!r})\n"
        "print(con.execute('SELECT count(*) FROM memories').fetchone()[0])\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "1", done.stdout


def test_a_server_can_be_started_again_after_it_shut_down(pool, sock_dir):
    """Releasing the files must not make the pool unusable: ``get`` reopens on demand."""

    async def scenario():
        config = S.ServerConfig(socket_path=sock_dir / "run" / "again.sock", tenants=(1,))
        server = S.AnatidServer(pool=pool, config=config)
        await server.start()
        server.call(
            protocol.Request(verb="remember", tenant=1, args={"content": "first run"}), ANY_TENANT
        ).raise_for_status()
        await server.shutdown()

        again = S.AnatidServer(pool=pool, config=config)
        await again.start()
        try:
            assert again.readiness().ready is True
            stats = again.call(protocol.Request(verb="stats", tenant=1), ANY_TENANT)
            assert stats.raise_for_status()["memories"] == 1
        finally:
            await again.shutdown()

    _run(scenario())


def test_backup_tenant_copies_a_file_the_server_reopened(pool, sock_dir, tmp_path):
    """Backup goes through ``COPY FROM DATABASE``, which needs no checkpoint.

    It used to checkpoint first, which raises on a handle opened for a file that already
    existed, so a backup of any tenant this process had opened once before failed outright.

    It also used to drain and then copy, which is a weaker boundary than it read: the drain
    returns without stopping anything, so a write submitted after it returned could land in the
    copy.  The report now names the guarantee the copy actually has.
    """
    first = pool.get(1)
    first.remember("before the server")
    assert pool.close(1)

    async def scenario():
        server = S.AnatidServer(
            pool=pool,
            config=S.ServerConfig(socket_path=sock_dir / "run" / "bk.sock", tenants=(1,)),
        )
        await server.start()
        try:
            server.call(
                protocol.Request(verb="remember", tenant=1, args={"content": "after"}), ANY_TENANT
            ).raise_for_status()
            return await server.backup_tenant(1, tmp_path / "backup.anatid")
        finally:
            await server.shutdown()

    report = _run(scenario())
    dest = report.path
    assert dest.exists()
    assert report.quiesced, "an idle tenant takes the barrier at once, so the boundary is real"
    assert report.guarantee.value == "quiesced"
    assert report.bytes == dest.stat().st_size
    copy = Anatid.open(dest, tenant=1, embedding_dim=DIM)
    try:
        assert copy.stats()["memories"] == 2
    finally:
        copy.close()


# --------------------------------------------------------------------------- the drain scope


def _blocking(gate: threading.Event):
    def work(db):
        gate.wait(20.0)
        return db.write("slow")

    return work


def test_draining_one_tenant_does_not_wait_for_another_tenants_traffic(fake_dbs):
    """The property a backup depends on, and the one the global drain does not have.

    Measured before this existed: backing up an untouched tenant cost 0.16 s with nothing
    running and 3.90 s while a DIFFERENT tenant was being written, because the wait summed every
    tenant's queue.  A per-tenant backup that pauses on another tenant's traffic is not the
    "only this tenant pauses" its docstring claimed.
    """
    q = Q.WriteQueue(fake_dbs, max_depth=64, batch_max=1, workers=1)
    gate = threading.Event()
    q.start()
    try:
        q.submit(1, "remember", _blocking(gate))
        deadline = time.monotonic() + 5.0
        while not q._serving and time.monotonic() < deadline:
            time.sleep(0.01)
        assert q._serving == {1}, "tenant 1's slow write never started"

        started = time.monotonic()
        quiet = q.drain_tenant(2, timeout=5.0)
        assert quiet.clean, quiet
        assert time.monotonic() - started < 1.0, "an idle tenant waited for a busy one"

        whole = q.drain(timeout=0.2)
        assert whole.timed_out, "the process-wide drain does wait, which is why it is not used"

        busy = q.drain_tenant(1, timeout=0.2)
        assert busy.timed_out, "and this tenant's own drain waits for this tenant"
    finally:
        gate.set()
        q.stop(timeout=10.0)


def test_a_tenant_drain_that_runs_out_of_budget_says_how_much_was_left(fake_dbs):
    q = Q.WriteQueue(fake_dbs, max_depth=64, batch_max=1, workers=1)
    for i in range(7):
        q.submit(1, "remember", _writer(i))
    report = q.drain_tenant(1, timeout=0.05)
    assert report.timed_out is True
    assert report.clean is False
    assert report.abandoned == 7
    assert report.abandoned_by_tenant == {1: 7}


def test_abandon_ends_a_drain_that_is_already_waiting(fake_dbs):
    """What a second SIGTERM has to reach: a drain waiting on another thread with its deadline
    already fixed."""
    q = Q.WriteQueue(fake_dbs, max_depth=64, batch_max=1, workers=1)
    for i in range(5):
        q.submit(1, "remember", _writer(i))
    done: list[Q.DrainReport] = []
    waiter = threading.Thread(target=lambda: done.append(q.drain(timeout=30.0)), daemon=True)
    waiter.start()
    time.sleep(0.05)
    assert not done, "the drain returned before anything drained it"
    started = time.monotonic()
    q.abandon()
    waiter.join(5.0)
    assert done and done[0].timed_out is True
    assert done[0].abandoned == 5
    assert time.monotonic() - started < 2.0
    with pytest.raises(protocol.ShuttingDown):
        q.submit(1, "remember", _writer(99))


# --------------------------------------------------------------------------- the wait hint


def test_the_suggested_wait_grows_with_the_crowd_and_not_only_with_the_backlog(fake_dbs):
    """A hint a crowd can act on.

    The wait used to be computed from queue depth alone, so sixteen clients refused at the same
    instant were each told to come back in about five milliseconds, came back together, and
    fifteen were refused again.  Measured then: 231 of 640 writes got through with the shipped
    three attempts.  The number of writes this tenant has refused since it was last under its
    high-water mark is the closest thing the queue has to a count of who is waiting, so the hint
    scales on it.
    """
    q = Q.WriteQueue(
        fake_dbs,
        max_depth=8,
        batch_max=2,
        workers=1,
        min_retry_after=1e-6,
        max_retry_after=1000.0,
    )
    for i in range(8):
        q.submit(1, "remember", _writer(i))
    hints = []
    for _ in range(20):
        with pytest.raises(protocol.BusyError) as excinfo:
            q.submit(1, "remember", _writer("over"))
        hints.append(excinfo.value.retry_after)
    assert hints[-1] > hints[0] * 5, (
        f"the twentieth refusal was told {hints[-1]:.6f}s and the first {hints[0]:.6f}s; "
        f"a hint that does not grow with the crowd sends the crowd back together"
    )
    assert len(set(hints)) > 1, "identical hints make a crowd retry in step"
    assert all(h > 0 for h in hints)


def test_the_wait_hint_is_still_bounded_by_its_configured_ceiling(fake_dbs):
    q = Q.WriteQueue(
        fake_dbs, max_depth=4, batch_max=2, workers=1, min_retry_after=0.001, max_retry_after=0.05
    )
    for i in range(4):
        q.submit(1, "remember", _writer(i))
    for _ in range(500):
        with pytest.raises(protocol.BusyError) as excinfo:
            q.submit(1, "remember", _writer("over"))
        assert 0.001 <= excinfo.value.retry_after <= 0.05


def test_the_refusal_count_resets_once_the_queue_is_under_the_mark_again(fake_dbs):
    q = Q.WriteQueue(fake_dbs, max_depth=4, batch_max=1, workers=1, high_water=0.5)
    for i in range(4):
        q.submit(1, "remember", _writer(i))
    for _ in range(10):
        with pytest.raises(protocol.BusyError):
            q.submit(1, "remember", _writer("over"))
    assert q._refusals[1] == 10
    q.start()
    try:
        deadline = time.monotonic() + 5.0
        while q.depth(1) and time.monotonic() < deadline:
            time.sleep(0.01)
        q.submit(1, "remember", _writer("after the queue emptied"))
        assert 1 not in q._refusals, "a queue back under its mark is not still refusing a crowd"
    finally:
        q.stop(timeout=10.0)


# --------------------------------------------------------------------------- bounded shutdown


def test_shutdown_stops_accepting_before_it_waits_for_anything(pool, sock_dir):
    """A write that arrives after SIGTERM is refused, not committed.

    This is the ordering bug, as behaviour.  ``shutdown`` used to wait on
    ``asyncio.Server.wait_closed`` before it told the queue to stop accepting, and that wait does
    not return while a connection handler is running, so a server under SIGTERM kept taking and
    committing writes on connections that were already open: about 2,100 of them in the run that
    found this, eight seconds after the signal.
    """
    accepted: list[bool] = []

    async def scenario():
        server = S.AnatidServer(
            pool=pool,
            config=S.ServerConfig(
                socket_path=sock_dir / "run" / "order.sock",
                tenants=(1,),
                workers=1,
                shutdown_timeout=10.0,
            ),
        )
        await server.start()
        gate = threading.Event()

        def slow(db):
            gate.wait(20.0)

        server.queue.submit(1, "remember", slow)
        shutting = asyncio.ensure_future(server.shutdown())
        await asyncio.sleep(0.1)
        assert not shutting.done(), "the drain finished before the test could look at it"
        accepted.append(server.queue.accepting)
        response = server.call(
            protocol.Request(verb="remember", tenant=1, args={"content": "after the signal"}),
            ANY_TENANT,
        )
        gate.set()
        await shutting
        return response

    response = _run(scenario())
    assert accepted == [False], "the queue was still accepting while the server was draining"
    assert response.error is not None
    assert response.error.error_class == "ShuttingDown"
    assert (
        pool.get(1)
        .execute("SELECT count(*) FROM memories WHERE content = 'after the signal'")
        .fetchone()[0]
        == 0
    ), "a write refused as not performed was performed"


def test_a_shutdown_is_not_held_open_by_a_client_that_stays_connected(pool, sock_dir):
    """The time bound, in process.  A connected but idle client used to hold this open forever.

    Measured before the reorder: one idle connection kept the server alive past 25 seconds
    against a 5 second budget, and it exited 0.03 seconds after that socket closed.
    """
    sock_path = sock_dir / "run" / "held.sock"
    elapsed: list[float] = []

    async def scenario():
        server = S.AnatidServer(
            pool=pool,
            config=S.ServerConfig(
                socket_path=sock_path, tenants=(1,), workers=1, shutdown_timeout=2.0
            ),
        )
        await server.start()
        reader, writer = await asyncio.open_unix_connection(str(sock_path))
        try:
            await protocol.write_frame_async(
                writer, protocol.Request(verb="remember", tenant=1, args={"content": "one"})
            )
            assert await protocol.read_frame_async(reader) is not None
            started = time.monotonic()
            report = await server.shutdown()  # the connection is open and stays open
            elapsed.append(time.monotonic() - started)
            return report
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    report = _run(scenario())
    assert report.clean, report
    assert elapsed[0] < 5.0, (
        f"shutdown took {elapsed[0]:.2f}s with one idle client connected and a 2s budget"
    )


def test_a_second_signal_abandons_a_drain_the_first_one_started(pool, sock_dir):
    """``install_signal_handlers`` documents this, and it did not work.

    The second signal used to call ``shutdown(timeout=0.0)``, which blocked at the same await as
    the first; the log showed both completing at the instant the last client disconnected.  It
    now reaches the queue, which is where the waiting actually happens.
    """
    timings: list[float] = []

    async def scenario():
        server = S.AnatidServer(
            pool=pool,
            config=S.ServerConfig(
                socket_path=sock_dir / "run" / "twice.sock",
                tenants=(1,),
                workers=1,
                batch_max=1,
                max_depth=256,
                shutdown_timeout=60.0,
            ),
        )
        await server.start()
        gate = threading.Event()

        def slow(db):
            gate.wait(20.0)

        server.queue.submit(1, "remember", slow)
        futures = [
            server.submit_write(
                protocol.Request(verb="remember", tenant=1, args={"content": f"m{i}"}),
                ANY_TENANT,
            )
            for i in range(30)
        ]
        server._on_signal(signal.SIGTERM)
        await asyncio.sleep(0.1)
        started = time.monotonic()
        server._on_signal(signal.SIGTERM)
        asyncio.get_running_loop().call_later(0.1, gate.set)
        await server._stopped.wait()
        timings.append(time.monotonic() - started)
        return server._shutdown_report, futures

    report, futures = _run(scenario())
    assert timings[0] < 15.0, f"the second signal took {timings[0]:.1f}s to end a 60s drain"
    assert report is not None and report.abandoned > 0, report
    abandoned = [f for f in futures if isinstance(f.exception(), protocol.ShuttingDown)]
    assert abandoned, "an abandoned write must complete with ShuttingDown, not hang"
    assert all(f.done() for f in futures), "every caller got an answer"


def test_a_second_shutdown_call_joins_the_first_instead_of_starting_another(pool, sock_dir):
    async def scenario():
        server = S.AnatidServer(
            pool=pool,
            config=S.ServerConfig(socket_path=sock_dir / "run" / "join.sock", tenants=(1,)),
        )
        await server.start()
        first, second = await asyncio.gather(server.shutdown(), server.shutdown())
        third = await server.shutdown()
        return first, second, third

    first, second, third = _run(scenario())
    assert first is second is third, "three calls, one shutdown, one report"


def test_ready_does_not_hand_an_anonymous_caller_the_tenant_list(pool):
    """``/ready`` is open so a load balancer can read the verdict; the detail is not.

    A full ``Readiness`` carries the tenant list, the per-tenant schema versions, how many files
    are open and the busiest queue depth.  On a token-protected listener that is exactly what the
    tenant boundary refuses to tell a principal: ``anatid.server.auth`` says the refusal must not
    let a client enumerate tenants, and ``/metrics`` already honours it.  ``/ready`` handed the
    same class of information to a caller with no credential at all.
    """
    token = "test-token"
    seen: dict[str, object] = {}

    async def scenario():
        server = S.AnatidServer(
            pool=pool,
            config=S.ServerConfig(
                socket_path=None, http_host="127.0.0.1", http_port=0, tenants=(1, 2), workers=1
            ),
            authenticator=auth.BearerTokenAuthenticator({token: None}),
        )
        await server.start()
        port = server._servers[0].sockets[0].getsockname()[1]
        try:
            seen["anonymous"] = await _http(port, "GET", "/ready")
            seen["authenticated"] = await _http(port, "GET", "/ready", token=token)
            seen["health"] = await _http(port, "GET", "/health")
        finally:
            await server.shutdown()

    _run(scenario())
    status, body = seen["anonymous"]
    anonymous = protocol.loads(body)
    assert status == 200, "the verdict stays readable without a credential"
    assert anonymous.ready is True
    assert anonymous.tenants == (), "an anonymous caller learned which tenants exist"
    assert anonymous.schema_versions == {}
    assert anonymous.open_files == 0
    assert b'"tenants"' in body and b"[1,2]" not in body

    status, body = seen["authenticated"]
    full = protocol.loads(body)
    assert status == 200
    assert full.tenants == (1, 2), "a caller with the token gets the whole record"
    assert full.schema_versions and full.open_files == 2

    assert seen["health"][0] == 200, "health stays open: a supervisor has no token"


def test_ready_is_not_redacted_on_a_listener_that_has_no_token_at_all(pool):
    """A loopback or Unix-socket deployment has no credential to present, and nothing to hide
    behind one: the reduction applies exactly where ``/rpc`` needs a token."""
    seen: dict[str, object] = {}

    async def scenario():
        server = S.AnatidServer(
            pool=pool,
            config=S.ServerConfig(
                socket_path=None, http_host="127.0.0.1", http_port=0, tenants=(1,), workers=1
            ),
        )
        await server.start()
        port = server._servers[0].sockets[0].getsockname()[1]
        try:
            seen["ready"] = await _http(port, "GET", "/ready")
        finally:
            await server.shutdown()

    _run(scenario())
    status, body = seen["ready"]
    assert status == 200
    assert protocol.loads(body).tenants == (1,)


def test_a_server_creates_a_tenant_on_demand_by_default(pool, sock_dir, tmp_path):
    """The pool's own behaviour, unchanged.  A service that provisions from traffic needs it."""
    server = S.AnatidServer(
        pool=pool, config=S.ServerConfig(socket_path=sock_dir / "make.sock", tenants=(1,))
    )
    response = server.call(protocol.Request(verb="stats", tenant=8801), ANY_TENANT)
    assert response.status is protocol.Status.OK
    assert pool.path_for(8801).exists()


def test_create_tenants_false_stops_a_client_turning_traffic_into_files(pool, sock_dir):
    """An unrestricted principal could name any integer and get a database file for it.

    That is not a tenant boundary problem, because the caller was entitled to name the tenant.
    It is unbounded resource use: a loop over ``remember(tenant=i)`` fills the pool directory.
    ``create_tenants=False`` confines the server to its configured tenants plus the files that
    already exist.
    """
    pool.get(2).remember("tenant two already has a file")  # created out of band
    pool.close(2)
    server = S.AnatidServer(
        pool=pool,
        config=S.ServerConfig(
            socket_path=sock_dir / "fixed.sock", tenants=(1,), create_tenants=False
        ),
    )
    server.open_tenants()

    response = server.call(protocol.Request(verb="stats", tenant=8802), ANY_TENANT)
    assert response.error is not None
    assert response.error.error_class == "TenantIsolationError"
    assert not pool.path_for(8802).exists(), "the refusal created the file it refused"

    write = server.call(
        protocol.Request(verb="remember", tenant=8803, args={"content": "no"}), ANY_TENANT
    )
    assert write.error is not None and write.error.error_class == "TenantIsolationError"
    assert not pool.path_for(8803).exists()

    assert server.call(protocol.Request(verb="stats", tenant=1), ANY_TENANT).status is (
        protocol.Status.OK
    ), "a configured tenant is still served"
    assert server.call(protocol.Request(verb="stats", tenant=2), ANY_TENANT).status is (
        protocol.Status.OK
    ), "a tenant whose file already exists is still served"


def test_the_ready_verb_is_scoped_to_the_principal_that_asked(server):
    """The probe is a second door onto the same question the tenant boundary answers.

    An unrestricted principal sees every tenant this process holds.  One scoped to tenant 1 sees
    tenant 1, and cannot tell from the answer whether tenant 2 exists.
    """
    whole = server.call(protocol.Request(verb="ready", tenant=1), ANY_TENANT).raise_for_status()
    assert whole.tenants == (1, 2)
    assert set(whole.schema_versions) == {1, 2}

    scoped = server.call(protocol.Request(verb="ready", tenant=1), ONLY_TENANT_1).raise_for_status()
    assert scoped.ready == whole.ready, "the verdict is the same server either way"
    assert scoped.tenants == (1,)
    assert set(scoped.schema_versions) == {1}
    assert "2" not in scoped.detail


def test_a_server_started_again_after_a_shutdown_can_be_shut_down_again(pool, sock_dir):
    """``start`` clears the previous shutdown, or the second one would be a no-op.

    The docstring on :meth:`shutdown` promises a restart in place works, and a joined shutdown
    task that outlived its server would break it quietly: the second shutdown would return the
    first one's report while the queue kept accepting.
    """
    sock_path = sock_dir / "run" / "again.sock"
    config = S.ServerConfig(socket_path=sock_path, tenants=(1,), workers=1, shutdown_timeout=5.0)

    async def scenario():
        server = S.AnatidServer(pool=pool, config=config)
        await server.start()
        server.call(
            protocol.Request(verb="remember", tenant=1, args={"content": "first life"}), ANY_TENANT
        ).raise_for_status()
        first = await server.shutdown()
        await server.start()
        assert server.queue.accepting is True, "a restarted server accepts writes again"
        server.call(
            protocol.Request(verb="remember", tenant=1, args={"content": "second life"}),
            ANY_TENANT,
        ).raise_for_status()
        second = await server.shutdown()
        return first, second, server

    first, second, server = _run(scenario())
    assert second is not first, "the second shutdown returned the first one's report"
    assert server.queue.accepting is False
    assert server.health().status == "stopped"
    assert pool.get(1).execute("SELECT count(*) FROM memories").fetchone()[0] == 2
