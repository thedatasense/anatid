"""Backups taken from a running server, and what each of them is actually worth.

The test this file exists for is
:func:`test_a_backup_taken_under_four_writers_is_a_clean_prefix`.  Four threads write through
the server's own dispatcher without pausing, a backup is taken in the middle of that, and the
copy is then held to three separate claims:

* it opens, and ``doctor()`` finds no errors in it;
* every write acknowledged before the backup call is in it;
* what is in it is a PREFIX of each writer's sequence, with no gap -- which is the shape a
  quiesced boundary has and a torn copy does not.

The third one is the interesting assertion.  Each ``remember`` writes across three tables
(``memories``, ``entities``, ``edges_about``) in one transaction, so a copy that caught a
transaction half way would show a memory with no ABOUT edge; the test counts them and they
match.  And each writer waits for its own acknowledgement before sending the next write, so its
memories are totally ordered; a copy that was a snapshot of no single moment would show a hole.

The rest of the file is the surrounding claims: which guarantee each path gives and that the
weaker ones say so, the barrier's two failure modes (it cannot be acquired, and it must not
leave a worker holding it), restore refusing to write over a live database, and a retention
helper that will not delete the last good copy.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import duckdb
import pytest

from anatid import Anatid, DatabasePool
from anatid.schema import SCHEMA_VERSION
from anatid.server import auth, protocol
from anatid.server.backup import (
    BackupCoordinator,
    BackupError,
    BackupUnreadable,
    DestinationInUse,
    Guarantee,
    QuiesceTimeout,
    QuiesceUnavailable,
    in_use,
    inspect,
    prune,
    restore,
    sidecars,
)
from anatid.server.server import AnatidServer, ServerConfig
from anatid.types import Isolation

DIM = 8

#: A Unix socket path is a fixed-size field in ``sockaddr_un``, and these servers never bind
#: one -- the tests drive :meth:`AnatidServer.call` directly, which is the whole dispatcher --
#: but ``ServerConfig`` requires a transport, so they are given a path they do not listen on.
UNUSED_SOCKET = "anatid-test.sock"

ADMIN = auth.Principal(name="test", tenants=None, peer=None)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def workspace(tmp_path):
    """A directory for databases and backups, with the pool's files under ``db/``."""
    (tmp_path / "db").mkdir()
    (tmp_path / "backups").mkdir()
    return tmp_path


def make_pool_server(workspace, *, tenants=(1,), **config) -> tuple[AnatidServer, DatabasePool]:
    """A server over a file-per-tenant pool, with the queue running and no socket bound."""
    pool = DatabasePool(str(workspace / "db" / "t_{tenant}.anatid"), embedding_dim=DIM)
    server = AnatidServer(
        pool=pool,
        config=ServerConfig(
            socket_path=str(workspace / UNUSED_SOCKET), tenants=tuple(tenants), **config
        ),
    )
    server.open_tenants()
    server.queue.start()
    return server, pool


def stop(server: AnatidServer, pool: DatabasePool | None = None) -> None:
    server.queue.stop(timeout=10.0)
    if pool is not None:
        pool.close_all()


def remember(server: AnatidServer, tenant: int, content: str, *, entities=()) -> protocol.Response:
    return server.call(
        protocol.Request(
            "remember",
            tenant=tenant,
            args={"content": content, "embedding": [1.0] * DIM, "entities": list(entities)},
        ),
        ADMIN,
    )


def rows(path: Path, sql: str) -> list[tuple]:
    """Read a backup with a plain DuckDB connection, so the assertion does not go through anatid."""
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def rows_via_handle(pool: DatabasePool, tenant: int) -> int:
    """Memories in the LIVE file, read through the handle the server itself holds.

    A second connection cannot be opened while the server owns the file, so a test that wants
    to know what the source still contains has to ask the owner.
    """
    row = pool.get(tenant).execute("SELECT count(*) FROM memories").fetchone()
    assert row is not None
    return int(row[0])


# --------------------------------------------------------------------------- the headline


def test_a_backup_taken_under_four_writers_is_a_clean_prefix(workspace):
    """The whole point of the module, asserted against the copy rather than against a counter.

    Four threads write continuously through the server's dispatcher.  A backup is taken while
    they run.  The copy has to open, pass ``doctor()``, hold every write acknowledged before the
    call, and hold a contiguous prefix of each writer's sequence with no gaps and no half-written
    transactions.
    """
    server, pool = make_pool_server(workspace, workers=4)
    stop_writing = threading.Event()
    acked: dict[int, list[int]] = {w: [] for w in range(4)}
    failures: list[object] = []

    def writer(w: int) -> None:
        n = 0
        while not stop_writing.is_set():
            response = remember(server, 1, f"w{w}-{n}", entities=[f"entity-{w}"])
            if response.status is protocol.Status.OK:
                acked[w].append(n)
            else:
                failures.append(response.error)
            n += 1

    threads = [threading.Thread(target=writer, args=(w,), daemon=True) for w in range(4)]
    for t in threads:
        t.start()
    try:
        # Long enough that every writer has acknowledged writes before the backup and keeps
        # going through it, which is the situation the copy has to survive.
        while min(len(v) for v in acked.values()) < 20:
            time.sleep(0.01)
        before = {w: len(v) for w, v in acked.items()}
        report = (
            BackupCoordinator(server)
            .backup(workspace / "backups" / "live.anatid", tenant=1, doctor=True)
            .one
        )
        after = {w: len(v) for w, v in acked.items()}
    finally:
        stop_writing.set()
        for t in threads:
            t.join(timeout=10.0)

    assert not failures, failures[:3]
    assert report.guarantee is Guarantee.QUIESCED
    assert report.quiesced_tenants == (1,)
    assert report.schema_version == SCHEMA_VERSION

    # 1. it opens, and doctor finds nothing wrong with it
    assert report.doctor is not None
    assert report.doctor.ok, [f.check for f in report.doctor.errors]
    assert not report.doctor.errors

    # 2. every write acknowledged before the call is in the copy, and the writers kept going
    contents = [r[0] for r in rows(report.path, "SELECT content FROM memories")]
    seen: dict[int, list[int]] = {w: [] for w in range(4)}
    for content in contents:
        w, n = content.split("-")
        seen[int(w[1:])].append(int(n))
    for w in range(4):
        got = sorted(seen[w])
        # 3. a contiguous prefix: 0..k with no hole, and at least everything acknowledged first
        assert got == list(range(len(got))), (w, got[:5], got[-5:])
        assert len(got) >= before[w], (w, len(got), before[w])
        assert after[w] > before[w], "the writers stopped, so this proves nothing"

    # no torn rows: remember() writes memories, entities and edges_about in ONE transaction, so
    # a copy that caught one half way would have a memory with no ABOUT edge
    memories = rows(report.path, "SELECT count(*) FROM memories")[0][0]
    edges = rows(report.path, "SELECT count(*) FROM edges_about")[0][0]
    entities = rows(report.path, "SELECT count(*) FROM entities")[0][0]
    assert memories == len(contents)
    assert edges == memories, "every memory named exactly one entity"
    assert entities == 4, "four writers, four entity names"
    orphans = rows(
        report.path,
        "SELECT count(*) FROM edges_about e "
        "LEFT JOIN memories m ON m.memory_id = e.src AND m.tenant_id = e.tenant_id "
        "WHERE m.memory_id IS NULL",
    )[0][0]
    assert orphans == 0
    stop(server, pool)


def test_the_source_keeps_taking_writes_after_the_barrier_releases(workspace):
    """A quiesced backup pauses the tenant; it must also un-pause it."""
    server, pool = make_pool_server(workspace)
    for i in range(5):
        assert remember(server, 1, f"before-{i}").status is protocol.Status.OK
    BackupCoordinator(server).backup(workspace / "backups" / "b.anatid", tenant=1)
    for i in range(5):
        assert remember(server, 1, f"after-{i}").status is protocol.Status.OK
    assert server.queue.depth(1) == 0
    assert rows_via_handle(pool, 1) == 10
    stop(server, pool)


# --------------------------------------------------------------------------- guarantees


def test_the_quiesced_boundary_holds_writes_queued_after_the_call(workspace):
    """A write submitted while the barrier is closed lands after the copy, never inside it.

    The barrier is queued behind everything already there and ahead of everything sent later, so
    this is the boundary stated as a test: a write submitted during the copy is acknowledged
    after it and is not in it.
    """
    server, pool = make_pool_server(workspace, workers=2)
    for i in range(3):
        remember(server, 1, f"early-{i}")

    later: list[object] = []
    gate = threading.Event()

    def send_during_the_copy() -> None:
        gate.wait(5.0)
        later.append(remember(server, 1, "during").status)

    thread = threading.Thread(target=send_during_the_copy, daemon=True)
    thread.start()

    # The copy itself is the seam: by the time the pool is asked for one, the barrier is already
    # holding tenant 1, so a write submitted here is certainly behind it.
    real_copy = pool.backup

    def copy_while_a_write_arrives(*args, **kwargs):
        gate.set()
        time.sleep(0.2)
        return real_copy(*args, **kwargs)

    pool.backup = copy_while_a_write_arrives  # type: ignore[method-assign]
    report = (
        BackupCoordinator(server).backup(workspace / "backups" / "boundary.anatid", tenant=1).one
    )
    pool.backup = real_copy  # type: ignore[method-assign]
    thread.join(timeout=10.0)

    assert later == [protocol.Status.OK]
    contents = {r[0] for r in rows(report.path, "SELECT content FROM memories")}
    assert contents == {"early-0", "early-1", "early-2"}
    assert rows_via_handle(pool, 1) == 4
    stop(server, pool)


def test_an_unquiesced_backup_says_so_rather_than_claiming_a_boundary(workspace):
    server, pool = make_pool_server(workspace)
    remember(server, 1, "one")
    report = (
        BackupCoordinator(server)
        .backup(workspace / "backups" / "online.anatid", tenant=1, quiesce=False)
        .one
    )
    assert report.guarantee is Guarantee.SNAPSHOT
    assert report.quiesced_tenants == ()
    assert report.quiesced_for == 0.0
    assert "may or may not be" in report.detail
    assert rows(report.path, "SELECT count(*) FROM memories")[0][0] == 1
    stop(server, pool)


def test_an_unquiesced_copy_never_holds_an_uncommitted_transaction(workspace):
    """DuckDB's own snapshot, asserted rather than assumed.

    ``COPY FROM DATABASE`` runs in its own transaction, so a row another connection has inserted
    and not committed is not in the copy.  That is the floor under the snapshot guarantee: no
    path in this module can produce a torn copy, only a copy whose upper boundary is vague.
    """
    server, pool = make_pool_server(workspace)
    remember(server, 1, "committed")
    db = pool.get(1)
    holding = threading.Event()
    release = threading.Event()

    def hold_an_open_write() -> None:
        with db.transaction():
            db.remember("uncommitted", embedding=[2.0] * DIM)
            holding.set()
            release.wait(10.0)

    thread = threading.Thread(target=hold_an_open_write, daemon=True)
    thread.start()
    assert holding.wait(5.0)
    try:
        report = (
            BackupCoordinator(server)
            .backup(workspace / "backups" / "mvcc.anatid", tenant=1, quiesce=False)
            .one
        )
    finally:
        release.set()
        thread.join(timeout=10.0)
    contents = {r[0] for r in rows(report.path, "SELECT content FROM memories")}
    assert contents == {"committed"}
    stop(server, pool)


def test_an_export_round_trips_through_restore(workspace):
    """``EXPORT DATABASE`` out, ``IMPORT DATABASE`` back, same rows and a clean doctor.

    The path that survives a DuckDB storage-format change: the export is Parquet plus DDL, so
    the restore rebuilds the file at the reading build's format rather than trusting the bytes.
    """
    server, pool = make_pool_server(workspace)
    for i in range(20):
        remember(server, 1, f"m{i}", entities=["Ada"])
    report = BackupCoordinator(server).export(workspace / "backups" / "exp", tenant=1).one
    assert report.guarantee is Guarantee.EXPORT
    assert (report.path / "schema.sql").is_file()
    assert (report.path / "memories.parquet").is_file()

    back = restore(report.path, workspace / "backups" / "from-export.anatid")
    assert back.kind == "export"
    assert back.schema_version == SCHEMA_VERSION
    assert back.doctor is not None and back.doctor.ok
    assert rows(back.destination, "SELECT count(*) FROM memories")[0][0] == 20
    assert rows(back.destination, "SELECT count(*) FROM edges_about")[0][0] == 20
    stop(server, pool)


def test_a_shared_file_reports_the_snapshot_guarantee_and_names_what_it_paused(workspace):
    """One file for every tenant means the copy holds every tenant, so pausing one is not enough.

    This is the honest half of ``database=``: the barrier does pin the named tenant's rows, and
    the report says which tenants that was, but the copy as a whole cannot claim the quiesced
    boundary because a tenant nobody named could have committed into the same file.
    """
    db = Anatid.open(workspace / "db" / "shared.anatid", tenant=0, embedding_dim=DIM)
    server = AnatidServer(
        database=db, config=ServerConfig(socket_path=str(workspace / UNUSED_SOCKET), workers=2)
    )
    server.queue.start()
    for tenant in (1, 2):
        remember(server, tenant, f"t{tenant}")
    report = BackupCoordinator(server).backup(workspace / "backups" / "shared.anatid", tenant=1).one
    assert report.guarantee is Guarantee.SNAPSHOT
    assert report.quiesced_tenants == (1,)
    assert "one shared file" in report.detail
    # the copy is the whole file: both tenants are in it
    assert rows(report.path, "SELECT count(*) FROM memories")[0][0] == 2
    stop(server)
    db.close()


def test_a_file_per_tenant_handle_can_claim_the_quiesced_guarantee(workspace):
    """``database=`` with FILE_PER_TENANT holds exactly one tenant, so pausing it covers the file."""
    db = Anatid.open(
        workspace / "db" / "solo.anatid",
        tenant=7,
        embedding_dim=DIM,
        isolation=Isolation.FILE_PER_TENANT,
    )
    server = AnatidServer(
        database=db,
        config=ServerConfig(socket_path=str(workspace / UNUSED_SOCKET), tenants=(7,)),
    )
    server.queue.start()
    remember(server, 7, "only")
    report = BackupCoordinator(server).backup(workspace / "backups" / "solo.anatid").one
    assert report.guarantee is Guarantee.QUIESCED
    assert report.quiesced_tenants == (7,)
    stop(server)
    db.close()


# --------------------------------------------------------------------------- the barrier


def test_a_barrier_that_cannot_be_acquired_copies_nothing(workspace):
    """A queue too deep to reach in time is a refusal, not a silently weaker backup."""
    server, pool = make_pool_server(workspace, workers=1, batch_max=1)
    blocked = threading.Event()
    for _ in range(20):
        server.queue.submit(1, "slow", lambda db: blocked.wait(5.0), batchable=False)
    destination = workspace / "backups" / "never.anatid"
    try:
        with pytest.raises(QuiesceTimeout) as caught:
            BackupCoordinator(server, acquire_timeout=0.1).backup(destination, tenant=1)
    finally:
        blocked.set()
    assert "did not reach the front" in str(caught.value)
    assert "quiesce=False" in str(caught.value)
    assert not destination.exists(), "a refused backup must not leave a partial file behind"
    stop(server, pool)


def test_an_abandoned_barrier_does_not_keep_holding_the_worker(workspace):
    """The failure mode a barrier must not have: a timed-out backup leaving the tenant paused.

    One worker, a queue full of slow writes, and a barrier that gives up before it reaches the
    front.  When the slow writes finish, the barrier is next -- and it has to return immediately
    rather than wait for a release nobody is going to send, or this tenant never writes again.
    """
    server, pool = make_pool_server(workspace, workers=1, batch_max=1)
    blocked = threading.Event()
    for _ in range(5):
        server.queue.submit(1, "slow", lambda db: blocked.wait(5.0), batchable=False)
    with pytest.raises(QuiesceTimeout):
        BackupCoordinator(server, acquire_timeout=0.05, hold_timeout=300.0).backup(
            workspace / "backups" / "abandoned.anatid", tenant=1
        )
    blocked.set()
    started = time.monotonic()
    response = remember(server, 1, "after the abandoned barrier")
    assert response.status is protocol.Status.OK
    assert time.monotonic() - started < 10.0
    stop(server, pool)


def test_quiescing_more_tenants_than_workers_is_refused_rather_than_deadlocked(workspace):
    """N barriers occupy N workers, so asking for more than there are would wait on itself."""
    db = Anatid.open(workspace / "db" / "many.anatid", tenant=0, embedding_dim=DIM)
    server = AnatidServer(
        database=db, config=ServerConfig(socket_path=str(workspace / UNUSED_SOCKET), workers=2)
    )
    server.queue.start()
    for tenant in (1, 2, 3):
        remember(server, tenant, f"t{tenant}")
    with pytest.raises(QuiesceUnavailable) as caught:
        BackupCoordinator(server).backup(workspace / "backups" / "many.anatid", tenant=[1, 2, 3])
    assert "worker threads and it has 2" in str(caught.value)
    stop(server)
    db.close()


def test_a_backup_on_a_stopped_queue_is_quiesced_because_nothing_can_commit(workspace):
    """No workers means no queued write can commit, which is the boundary a barrier would give."""
    pool = DatabasePool(str(workspace / "db" / "t_{tenant}.anatid"), embedding_dim=DIM)
    pool.get(1).remember("written before the server started", embedding=[1.0] * DIM)
    server = AnatidServer(
        pool=pool,
        config=ServerConfig(socket_path=str(workspace / UNUSED_SOCKET), tenants=(1,)),
    )
    assert not server.queue.running
    report = BackupCoordinator(server).backup(workspace / "backups" / "cold.anatid", tenant=1).one
    assert report.guarantee is Guarantee.QUIESCED
    assert "the write queue was not running" in report.detail
    assert rows(report.path, "SELECT count(*) FROM memories")[0][0] == 1
    pool.close_all()


# --------------------------------------------------------------------------- coverage of the surface


def test_every_tenant_gets_its_own_file_when_none_is_named(workspace):
    server, pool = make_pool_server(workspace, tenants=(1, 2, 3), workers=2)
    for tenant in (1, 2, 3):
        for i in range(tenant):
            remember(server, tenant, f"t{tenant}-{i}")
    reports = BackupCoordinator(server).backup(workspace / "backups" / "all")
    assert [r.tenant for r in reports] == [1, 2, 3]
    assert [r.path.name for r in reports] == [
        "tenant-1.anatid",
        "tenant-2.anatid",
        "tenant-3.anatid",
    ]
    assert all(r.guarantee is Guarantee.QUIESCED for r in reports)
    for report in reports:
        assert report.counts["memories"] == report.tenant
    assert reports.bytes == sum(r.bytes for r in reports)
    with pytest.raises(BackupError, match="covered 3 tenants"):
        reports.one
    stop(server, pool)


def test_a_backup_will_not_silently_replace_the_previous_one(workspace):
    server, pool = make_pool_server(workspace)
    remember(server, 1, "one")
    coordinator = BackupCoordinator(server)
    destination = workspace / "backups" / "once.anatid"
    coordinator.backup(destination, tenant=1)
    with pytest.raises(BackupError, match="neither copy"):
        coordinator.backup(destination, tenant=1)
    remember(server, 1, "two")
    coordinator.backup(destination, tenant=1, overwrite=True)
    assert rows(destination, "SELECT count(*) FROM memories")[0][0] == 2
    stop(server, pool)


def test_the_awaitable_wrapper_runs_the_same_backup(workspace):
    server, pool = make_pool_server(workspace)
    remember(server, 1, "one")
    coordinator = BackupCoordinator(server)
    report = asyncio.run(coordinator.abackup(workspace / "backups" / "async.anatid", tenant=1)).one
    assert report.guarantee is Guarantee.QUIESCED
    assert report.counts["memories"] == 1
    exported = asyncio.run(coordinator.aexport(workspace / "backups" / "async-exp", tenant=1)).one
    assert exported.guarantee is Guarantee.EXPORT
    stop(server, pool)


def test_in_memory_databases_are_refused_with_a_reason(workspace):
    db = Anatid.open(":memory:", tenant=1, embedding_dim=DIM)
    server = AnatidServer(
        database=db, config=ServerConfig(socket_path=str(workspace / UNUSED_SOCKET))
    )
    with pytest.raises(BackupError, match="nothing on disk"):
        BackupCoordinator(server).backup(workspace / "backups" / "nope.anatid", tenant=1)
    db.close()


# --------------------------------------------------------------------------- inspect and in_use


def test_in_use_tells_a_held_file_from_a_broken_one(workspace):
    live = workspace / "db" / "live.anatid"
    handle = Anatid.open(live, tenant=1, embedding_dim=DIM)
    handle.remember("held", embedding=[1.0] * DIM)
    held = in_use(live)
    assert held is not None
    assert "configuration" in held or "lock" in held

    broken = workspace / "backups" / "broken.anatid"
    broken.write_bytes(b"this is not a duckdb file")
    assert in_use(broken) is None, "a broken file is unreadable, not in use"
    assert not inspect(broken).ok

    assert in_use(workspace / "backups" / "absent.anatid") is None
    assert inspect(workspace / "backups" / "absent.anatid").kind == "missing"
    handle.close()
    assert in_use(live) is None


def test_inspect_reads_a_backup_without_writing_to_it(workspace):
    server, pool = make_pool_server(workspace)
    for i in range(4):
        remember(server, 1, f"m{i}", entities=["Ada"])
    report = BackupCoordinator(server).backup(workspace / "backups" / "look.anatid", tenant=1).one
    stop(server, pool)

    before = report.path.stat().st_mtime_ns
    info = inspect(report.path, doctor=True)
    assert info.ok
    assert info.kind == "database"
    assert info.schema_version == SCHEMA_VERSION
    assert not info.needs_migration and not info.too_new
    assert info.counts["memories"] == 4
    assert info.doctor is not None and info.doctor.ok
    assert report.path.stat().st_mtime_ns == before
    assert sidecars(report.path) == [], "a COPY FROM DATABASE result is folded and self-contained"


def test_a_copy_that_will_not_open_is_raised_rather_than_returned(workspace, monkeypatch):
    """``verify=True`` exists so a broken copy is an exception, not a file in the backup rota.

    The copy is replaced with rubbish the moment it is written, which is the closest a test gets
    to a disk that stored something other than what DuckDB handed it.  The call has to fail, and
    it has to leave the evidence on disk rather than tidy it away.
    """
    server, pool = make_pool_server(workspace)
    remember(server, 1, "one")
    coordinator = BackupCoordinator(server)
    destination = workspace / "backups" / "corrupt.anatid"
    real_copy = pool.backup

    def copy_then_corrupt(*args, **kwargs):
        out = real_copy(*args, **kwargs)
        Path(out).write_bytes(b"truncated")
        return out

    monkeypatch.setattr(pool, "backup", copy_then_corrupt)
    with pytest.raises(BackupUnreadable) as caught:
        coordinator.backup(destination, tenant=1)
    assert "do not treat it as a backup" in str(caught.value)
    assert destination.exists(), "the failed copy is left in place for inspection"
    # verify=False skips the check, which is exactly why it is not the default
    monkeypatch.setattr(pool, "backup", copy_then_corrupt)
    report = coordinator.backup(destination, tenant=1, overwrite=True, verify=False).one
    assert report.schema_version is None
    assert not inspect(report.path).ok
    stop(server, pool)


# --------------------------------------------------------------------------- restore


def test_restore_reports_the_schema_version_and_leaves_the_file_alone(workspace):
    server, pool = make_pool_server(workspace)
    for i in range(6):
        remember(server, 1, f"m{i}", entities=["Ada"])
    report = BackupCoordinator(server).backup(workspace / "backups" / "src.anatid", tenant=1).one
    stop(server, pool)

    destination = workspace / "db" / "restored.anatid"
    back = restore(report.path, destination)
    assert back.kind == "database"
    assert back.schema_version == SCHEMA_VERSION
    assert back.expected_schema_version == SCHEMA_VERSION
    assert not back.needs_migration
    assert back.doctor is not None and back.doctor.ok
    assert rows(destination, "SELECT count(*) FROM memories")[0][0] == 6
    # the source is untouched and still opens
    assert inspect(report.path).ok


def test_restore_refuses_a_destination_a_server_currently_owns(workspace):
    server, pool = make_pool_server(workspace)
    remember(server, 1, "live")
    report = BackupCoordinator(server).backup(workspace / "backups" / "src.anatid", tenant=1).one
    live = pool.path_for(1)
    with pytest.raises(DestinationInUse) as caught:
        BackupCoordinator(server).restore(report.path, live)
    assert "open in this process" in str(caught.value)
    # and the plain function catches it too, through the filesystem rather than the server
    with pytest.raises(DestinationInUse):
        restore(report.path, live)
    assert rows_via_handle(pool, 1) == 1
    stop(server, pool)


def test_restore_refuses_a_source_that_is_not_a_backup(workspace):
    junk = workspace / "backups" / "junk.anatid"
    junk.write_bytes(b"\x00" * 4096)
    with pytest.raises(BackupUnreadable):
        restore(junk, workspace / "db" / "out.anatid")
    assert not (workspace / "db" / "out.anatid").exists()

    with pytest.raises(BackupUnreadable, match="no backup at"):
        restore(workspace / "backups" / "absent.anatid", workspace / "db" / "out.anatid")


def test_restore_will_not_overwrite_without_being_told_to(workspace):
    server, pool = make_pool_server(workspace)
    remember(server, 1, "one")
    report = BackupCoordinator(server).backup(workspace / "backups" / "src.anatid", tenant=1).one
    stop(server, pool)
    destination = workspace / "db" / "target.anatid"
    restore(report.path, destination)
    with pytest.raises(BackupError, match="no undo"):
        restore(report.path, destination)
    again = restore(report.path, destination, overwrite=True)
    assert again.destination == destination


def test_restore_onto_the_backup_itself_is_refused_before_anything_is_deleted(workspace):
    """The one restore that must never run: source and destination the same file.

    ``overwrite=True`` unlinks the destination and then copies the source onto it, so if they
    are the same file the unlink deletes the backup and the copy then has nothing to copy.  That
    leaves neither file, which is worse than any failure a restore is allowed to have, so it is
    refused on the arguments before a byte is touched.
    """
    server, pool = make_pool_server(workspace)
    remember(server, 1, "one")
    report = BackupCoordinator(server).backup(workspace / "backups" / "self.anatid", tenant=1).one
    stop(server, pool)
    size = report.path.stat().st_size

    with pytest.raises(BackupError, match="both the source and the destination"):
        restore(report.path, report.path, overwrite=True)
    # the same file reached by a different spelling of the path, which resolve() has to catch
    with pytest.raises(BackupError, match="both the source and the destination"):
        restore(report.path, workspace / "backups" / "." / "self.anatid", overwrite=True)

    assert report.path.exists(), "the backup is still there"
    assert report.path.stat().st_size == size
    assert inspect(report.path).ok


def test_restore_into_a_directory_says_to_name_the_file(workspace):
    """``restore(backup, /var/lib/anatid/)`` is an operator typo, not an overwrite decision.

    Before this was checked the directory was reported as an existing destination, so the advice
    was "pass overwrite=True", and following that advice raised ``PermissionError`` from
    ``unlink`` on a directory.
    """
    server, pool = make_pool_server(workspace)
    remember(server, 1, "one")
    report = BackupCoordinator(server).backup(workspace / "backups" / "d.anatid", tenant=1).one
    stop(server, pool)
    target = workspace / "db"
    assert target.is_dir()

    for overwrite in (False, True):
        with pytest.raises(BackupError, match="is a directory"):
            restore(report.path, target, overwrite=overwrite)
    assert sorted(p.name for p in target.iterdir()) == ["t_1.anatid"], "nothing was written"


def test_restore_carries_the_write_ahead_log_with_the_file(workspace):
    """A copy of the database file alone loses every committed write DuckDB has not folded in.

    Not the path :meth:`BackupCoordinator.backup` produces -- its copies are folded and have no
    sidecar -- but the path a file lands on when an operator copies one by hand, which is exactly
    the input ``restore`` has to handle correctly.
    """
    source = workspace / "backups" / "handmade.anatid"
    handle = Anatid.open(source, tenant=1, embedding_dim=DIM)
    for i in range(30):
        handle.remember(f"m{i}", embedding=[1.0] * DIM)
    wal = Path(f"{source}.wal")
    assert wal.exists() and wal.stat().st_size > 0, "a live file keeps a write-ahead log"
    # snapshot both files the way a hand-rolled backup would, then let the handle go
    staged = workspace / "backups" / "staged.anatid"
    shutil.copy2(source, staged)
    shutil.copy2(wal, Path(f"{staged}.wal"))
    handle.close()

    assert sidecars(staged) == [Path(f"{staged}.wal")]
    back = restore(staged, workspace / "db" / "replayed.anatid")
    assert back.sidecars == (Path(f"{workspace / 'db' / 'replayed.anatid'}.wal"),)
    assert rows(back.destination, "SELECT count(*) FROM memories")[0][0] == 30


# --------------------------------------------------------------------------- retention


def make_backups(workspace, server, count: int, *, name: str = "b") -> list[Path]:
    out = []
    coordinator = BackupCoordinator(server)
    for i in range(count):
        remember(server, 1, f"m{i}")
        path = coordinator.backup(workspace / "backups" / f"{name}{i}.anatid", tenant=1).one.path
        # mtime is the ordering key and a fast machine writes several inside one tick
        os.utime(path, (time.time() + i, time.time() + i))
        out.append(path)
    return out


def test_prune_keeps_the_newest_n_and_removes_the_rest(workspace):
    server, pool = make_pool_server(workspace)
    made = make_backups(workspace, server, 5)
    stop(server, pool)
    report = prune(workspace / "backups", keep=2)
    assert [p.name for p in report.kept] == ["b3.anatid", "b4.anatid"]
    assert [p.name for p in report.pruned] == ["b0.anatid", "b1.anatid", "b2.anatid"]
    assert report.freed_bytes > 0
    assert all(not p.exists() for p in made[:3])
    assert all(p.exists() for p in made[3:])


def test_prune_never_deletes_the_only_good_backup(workspace):
    """Seven corrupt copies must not push the last readable one out of the retention window."""
    server, pool = make_pool_server(workspace)
    good = make_backups(workspace, server, 1, name="good")[0]
    stop(server, pool)
    for i in range(4):
        broken = workspace / "backups" / f"bad{i}.anatid"
        broken.write_bytes(b"not a database")
        os.utime(broken, (time.time() + 100 + i, time.time() + 100 + i))

    report = prune(workspace / "backups", keep=2)
    assert good.exists(), "the last good backup was deleted"
    assert good in report.kept
    assert set(report.bad) == {"bad0.anatid", "bad1.anatid", "bad2.anatid", "bad3.anatid"}
    # the two newest (both bad) are kept by the window, and the good one by the rule
    assert [p.name for p in report.kept] == ["bad2.anatid", "bad3.anatid", "good0.anatid"]
    assert [p.name for p in report.pruned] == ["bad0.anatid", "bad1.anatid"]
    assert "would not open" in report.detail


def test_prune_with_keep_zero_still_leaves_one_good_backup(workspace):
    """``keep=0`` is the sharpest form of the rule: the last good copy is not prunable.

    A retention job that can be configured down to zero is a retention job that can delete
    everything, and the point of this helper is that it cannot.  A directory with nothing good
    in it does empty out, because there is no good copy to save.
    """
    server, pool = make_pool_server(workspace)
    made = make_backups(workspace, server, 3)
    stop(server, pool)

    report = prune(workspace / "backups", keep=0)
    assert [p.name for p in report.kept] == ["b2.anatid"], "the newest good one, and only it"
    assert [p.name for p in report.pruned] == ["b0.anatid", "b1.anatid"]
    assert made[2].exists()

    made[2].write_bytes(b"corrupted after the fact")
    empty = prune(workspace / "backups", keep=0)
    assert empty.kept == ()
    assert [p.name for p in empty.pruned] == ["b2.anatid"], "nothing good left to protect"


def test_prune_leaves_a_file_a_server_is_using_alone(workspace):
    """A live database that happens to match the pattern is not a backup and is never deleted."""
    server, pool = make_pool_server(workspace)
    make_backups(workspace, server, 3)
    live = workspace / "backups" / "zzz-live.anatid"
    handle = Anatid.open(live, tenant=9, embedding_dim=DIM)
    handle.remember("open right now", embedding=[1.0] * DIM)
    try:
        report = prune(workspace / "backups", keep=1)
    finally:
        handle.close()
    assert live.exists()
    assert "zzz-live.anatid" in report.held
    assert live not in report.pruned
    assert [p.name for p in report.kept] == ["b2.anatid"]
    stop(server, pool)


def test_prune_dry_run_removes_nothing(workspace):
    server, pool = make_pool_server(workspace)
    made = make_backups(workspace, server, 4)
    stop(server, pool)
    report = prune(workspace / "backups", keep=1, dry_run=True)
    assert report.dry_run
    assert len(report.pruned) == 3
    assert all(p.exists() for p in made)
    assert report.freed_bytes > 0


def test_prune_takes_the_sidecars_with_the_file(workspace):
    server, pool = make_pool_server(workspace)
    made = make_backups(workspace, server, 3)
    stop(server, pool)
    stray = Path(f"{made[0]}.wal")
    stray.write_bytes(b"\x00" * 128)
    report = prune(workspace / "backups", keep=1)
    assert made[0] in report.pruned
    assert not stray.exists(), "a write-ahead log left behind belongs to a file that is gone"
    assert "b0.anatid.wal" not in {p.name for p in report.pruned}


def test_prune_refuses_a_directory_that_is_not_one(workspace):
    with pytest.raises(BackupError, match="not a directory"):
        prune(workspace / "backups" / "nothing-here")


# --------------------------------------------------------------------------- measurement


@pytest.mark.slow
def test_backup_cost_on_the_hundred_thousand_memory_file(spike_db):
    """Time and size for the spike's 100k-memory file, every path, printed for the record.

    Run with ``-s`` to see the table.  The numbers are asserted only loosely: a bound loose
    enough to survive a slower machine is a smoke test, not a benchmark, and the point of the
    test is the printed table plus the round trip at the end, which is the assertion that the
    42 MiB the copy wrote are the 42 MiB the source held.

    The staging step is deliberate.  ``spike_db`` is one shared handle, so a copy of it can only
    ever carry the snapshot guarantee (the file holds every tenant and the server cannot pause a
    tenant it has never heard of).  Staging it into a pool file and measuring there is the shape
    a real deployment has, and it is the only one where the quiesced number means anything.
    """
    source = Path(spike_db.path)
    assert spike_db.execute("SELECT count(*) FROM memories").fetchone()[0] == 100_000

    with tempfile.TemporaryDirectory(prefix="anatid-backup-bench-") as raw:
        out = Path(raw)
        (out / "db").mkdir()

        shared = AnatidServer(
            database=spike_db, config=ServerConfig(socket_path=str(out / UNUSED_SOCKET))
        )
        shared.queue.start()
        try:
            staged = (
                BackupCoordinator(shared)
                .backup(out / "db" / "t_0.anatid", tenant=0, quiesce=False, verify=False)
                .one
            )
            exported = BackupCoordinator(shared).export(out / "spike-export", tenant=0).one
        finally:
            shared.queue.stop(timeout=30.0)

        pool = DatabasePool(str(out / "db" / "t_{tenant}.anatid"), embedding_dim=64)
        server = AnatidServer(
            pool=pool,
            config=ServerConfig(socket_path=str(out / UNUSED_SOCKET), tenants=(0,)),
        )
        server.open_tenants()
        server.queue.start()
        try:
            copy = BackupCoordinator(server).backup(out / "spike.anatid", tenant=0, doctor=True).one
        finally:
            server.queue.stop(timeout=30.0)

        started = time.monotonic()
        shutil.copy2(source, out / "unsafe.anatid")
        unsafe_seconds = time.monotonic() - started

        assert copy.doctor is not None, "doctor=True was asked for, so the report must be here"
        print(  # noqa: T201 - the measurement IS the output of this test; run it with -s
            f"\nbackup cost, {source.stat().st_size / 1048576:.1f} MiB source, "
            f"100,000 memories, 10,000 entities, 196,135 ABOUT edges\n"
            f"  COPY FROM DATABASE, quiesced  {copy.seconds * 1e3:8.1f} ms  "
            f"{copy.megabytes:6.1f} MiB  paused tenant {list(copy.quiesced_tenants)} for "
            f"{copy.quiesced_for * 1e3:.0f} ms\n"
            f"  COPY FROM DATABASE, online    {staged.seconds * 1e3:8.1f} ms  "
            f"{staged.megabytes:6.1f} MiB  paused nothing\n"
            f"  EXPORT DATABASE (parquet)     {exported.seconds * 1e3:8.1f} ms  "
            f"{exported.megabytes:6.1f} MiB  paused nothing, restores through IMPORT DATABASE\n"
            f"  shutil.copy2                  {unsafe_seconds * 1e3:8.1f} ms  "
            f"{(out / 'unsafe.anatid').stat().st_size / 1048576:6.1f} MiB  "
            f"no coordination at all, no guarantee at all\n"
            f"  doctor() on the copy          "
            f"{(copy.doctor.duration_ms or 0.0):8.1f} ms  ok={copy.doctor.ok}"
        )

        assert copy.guarantee is Guarantee.QUIESCED
        assert copy.doctor.ok
        assert copy.counts["memories"] == 100_000
        assert copy.bytes > 20 * 1024 * 1024
        assert copy.seconds < 60.0
        assert exported.guarantee is Guarantee.EXPORT
        assert exported.bytes > 20 * 1024 * 1024

        restored = restore(copy.path, out / "restored.anatid", doctor=True)
        imported = restore(exported.path, out / "imported.anatid", doctor=False)
        print(  # noqa: T201 - restore is half of what this module ships, so it is timed too
            f"  restore, file copy + doctor    {restored.seconds * 1e3:8.1f} ms  "
            f"{restored.bytes / 1048576:6.1f} MiB  schema v{restored.schema_version}\n"
            f"  restore, IMPORT DATABASE       {imported.seconds * 1e3:8.1f} ms  "
            f"{imported.bytes / 1048576:6.1f} MiB  rebuilt at this build's storage format"
        )
        assert restored.schema_version == SCHEMA_VERSION
        assert not restored.needs_migration
        assert restored.doctor is not None and restored.doctor.ok
        for put_back in (restored.destination, imported.destination):
            assert rows(put_back, "SELECT count(*) FROM memories")[0][0] == 100_000
            assert rows(put_back, "SELECT count(*) FROM edges_about")[0][0] == 196_135
            assert rows(put_back, "SELECT count(*) FROM entities")[0][0] == 10_000
        pool.close_all()
