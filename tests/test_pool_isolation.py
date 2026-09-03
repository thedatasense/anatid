"""Adversarial tenant isolation, and the hardening of :class:`anatid.DatabasePool`.

Two tenants are set up to collide as hard as the data model allows: the same memory ids, the
same entity names, the same words in every memory, the same query, the same embedding.  The
only difference is one marker word.  Every public read path is then run for tenant 1 and must
return nothing that belongs to tenant 2, twice over:

* **one file**, two ``tenant_id`` values, which is a NAMESPACE.  The predicate anatid compiles
  into every read is the whole boundary; the raw cursor still sees both tenants, and one test
  proves that rather than hiding it.
* **two files** from a :class:`~anatid.DatabasePool`, which is the SECURITY boundary.  The
  handle cannot reach the other tenant at all, and asking it to raises.

The rest of the module covers what the pool does with the filesystem, where the tenant boundary
stops being a predicate and becomes a path: a tenant label may not escape the pool's directory,
files are created 0600 inside a 0700 directory, deletion and backup are per tenant and audited,
and a raw DuckDB cursor has to be asked for by its real name.
"""

from __future__ import annotations

import datetime as _dt
import os
import stat

import duckdb
import pytest

from anatid import (
    Anatid,
    AnatidError,
    DatabasePool,
    Isolation,
    Namespace,
    NotFoundError,
    TenantIsolationError,
)
from anatid.database import PoolEvent

from conftest import DIM, T0, vec

#: The two tenants write the same content except for this word.
MARK = {1: "alpha", 2: "beta"}

#: Deliberately the same ids in both tenants.  ``memory_id`` is unique per tenant, not per file,
#: so this is legal, and it is the shape that broke the BM25 arm before 0.1.1: a document key of
#: ``memory_id`` alone made tenant 2's row answer tenant 1's search.
IDS = (9001, 9002, 9003)

NAME = "Ada Lovelace"
OTHER = "Charles Babbage"
LATER = _dt.datetime(2026, 3, 1)


def _seed(db: Anatid, tenant: int) -> dict:
    """Write one tenant's world.  Identical in both tenants but for :data:`MARK`."""
    mark = MARK[tenant]
    ep = db.episode(f"{mark} transcript: she wrote the notes", tenant=tenant, now=T0)
    made = []
    for n, mid in enumerate(IDS):
        made.append(
            db.remember(
                f"{mark} secret note {n} about the analytical engine",
                entities=[NAME, OTHER],
                tenant=tenant,
                memory_id=mid,
                embedding=vec(1.0, float(n), 0.5),
                episode_id=ep.episode_id,
                writer=f"{mark}-writer",
                now=T0,
            )
        )
    db.relate(NAME, OTHER, tenant=tenant, now=T0)
    newer = db.supersede(
        IDS[0], f"{mark} secret note 0 corrected", tenant=tenant, now=LATER, memory_id=9100
    )
    return {"episode": ep, "memories": made, "newer": newer, "mark": mark}


def _texts(rows) -> list[str]:
    """Content of anything a read path returns: memories, hits, or (id, ts) pairs."""
    out = []
    for row in rows:
        content = getattr(row, "content", None)
        if content is None and hasattr(row, "memory"):
            content = row.memory.content
        out.append(content if content is not None else repr(row))
    return out


def _tenants_of(rows) -> set[int]:
    out = set()
    for row in rows:
        mem = getattr(row, "memory", row)
        tid = getattr(mem, "tenant_id", None)
        if tid is not None:
            out.add(int(tid))
    return out


#: Every public read path, as ``name -> callable(db, tenant) -> rows``.  A row is anything that
#: carries content and a tenant_id, so one assertion covers them all.
READS = {
    "recall_text": lambda db, t: db.recall("secret note", tenant=t, k=20),
    "recall_vector": lambda db, t: db.recall(embedding=vec(1.0, 1.0, 0.5), tenant=t, k=20),
    "recall_graph": lambda db, t: db.recall(seed_entity=NAME, tenant=t, hops=2, k=20),
    "recall_all_arms": lambda db, t: db.recall(
        "secret note", embedding=vec(1.0, 1.0, 0.5), seed_entity=NAME, tenant=t, hops=2, k=20
    ),
    "recall_2hop": lambda db, t: db.recall_2hop(NAME, tenant=t, limit=50),
    "recall_2hop_hop1": lambda db, t: db.recall_2hop(OTHER, tenant=t, limit=50, hops=1),
    "context": lambda db, t: db.context(NAME, tenant=t, limit=50),
    "context_2hop": lambda db, t: db.context(NAME, tenant=t, limit=50, hops=2),
    "get": lambda db, t: [m for m in (db.get(i, tenant=t) for i in IDS) if m is not None],
    "versions": lambda db, t: [v for i in IDS for v in db.versions(i, tenant=t)],
    "provenance_chain": lambda db, t: list(db.provenance(9100, tenant=t).chain),
    "provenance_versions": lambda db, t: list(db.provenance(9100, tenant=t).versions),
    "as_of_recall_2hop": lambda db, t: db.as_of(T0 + _dt.timedelta(hours=1)).recall_2hop(
        NAME, tenant=t, limit=50
    ),
    "as_of_get": lambda db, t: [
        m
        for m in (db.as_of(T0 + _dt.timedelta(hours=1)).get(i, tenant=t) for i in IDS)
        if m is not None
    ],
}


def _assert_only(rows, tenant: int, path: str) -> None:
    """No row from another tenant, by tenant_id and by the marker word in the content."""
    assert rows, f"{path} returned nothing for tenant {tenant}; the test would prove nothing"
    foreign = _tenants_of(rows) - {tenant}
    assert not foreign, f"{path} leaked rows of tenant(s) {foreign} into tenant {tenant}"
    theirs = [t for t in MARK.values() if t != MARK[tenant]]
    for text in _texts(rows):
        for word in theirs:
            assert word not in text, f"{path} leaked {word!r} content into tenant {tenant}: {text}"
        assert MARK[tenant] in text, f"{path} returned something unmarked: {text}"


# ============================================================================ one shared file


@pytest.fixture
def shared(tmp_path):
    """One file holding two tenants.  A namespace, not a security boundary."""
    with Anatid.open(tmp_path / "shared.anatid", tenant=1, embedding_dim=DIM) as db:
        _seed(db, 1)
        _seed(db, 2)
        db.rebuild_fts_index()
        yield db


@pytest.mark.parametrize("path", sorted(READS))
@pytest.mark.parametrize("tenant", (1, 2))
def test_a_shared_file_read_returns_only_the_asking_tenants_rows(shared, path, tenant):
    """The adversarial case: same ids, same names, same words, same query, same vector.

    Both tenants hold memory 9001.  Every read has to pick the right one, and the only thing
    that makes it pick is the predicate :mod:`anatid.visibility` compiles in.
    """
    _assert_only(READS[path](shared, tenant), tenant, path)


def test_a_shared_file_read_for_an_id_the_tenant_does_not_have_finds_nothing(shared):
    lonely = shared.remember("gamma only", tenant=3, memory_id=9500, now=T0)
    assert shared.get(9500, tenant=1) is None
    assert shared.versions(9500, tenant=1) == []
    assert shared.entities_of(9500, tenant=1) == []
    with pytest.raises(NotFoundError):
        shared.provenance(9500, tenant=1)
    assert shared.get(lonely.memory_id, tenant=3) is not None


def test_a_shared_file_entity_lookup_is_per_tenant(shared):
    one = shared.entity_id(NAME, tenant=1)
    two = shared.entity_id(NAME, tenant=2)
    assert one != two
    assert shared.get_entity(one, tenant=2) is None
    assert shared.get_entity(NAME, tenant=1).entity_id == one
    # a foreign entity id is not a seed: the traversal is scoped, so it finds nothing
    assert shared.recall_2hop(two, tenant=1) == []


def test_a_shared_file_episode_lookup_is_per_tenant(shared):
    ep_one = shared.provenance(9100, tenant=1).episodes[0]
    assert shared.get_episode(ep_one.episode_id, tenant=2) is None
    assert shared.get_episode(ep_one.episode_id, tenant=1) is not None


def test_stats_and_fts_status_count_one_tenant(shared):
    one = shared.stats(tenant=1)
    two = shared.stats(tenant=2)
    assert one["memories"] == two["memories"] == 4  # 3 written, 1 correction, 1 closed
    assert shared.stats(all_tenants=True)["memories"] == one["memories"] + two["memories"]
    status = shared.fts_status()
    assert status.indexed_rows >= 8


def test_index_health_is_reported_per_tenant(shared):
    """Each report names the tenant it is about, or ``None`` for a file-wide index.

    The full-text index builds ONE generation for the whole file (``per_tenant=False``: PRAGMA
    create_fts_index has a fixed per-call cost, so per-tenant scoping is a predicate rather than
    a structure), and a file-wide report carries ``tenant_id=None`` on purpose.  Everything the
    report says about it -- pending rows, staleness -- is therefore file-wide too, which is the
    number an operator deciding when to rebuild needs.  The per-tenant accelerators must never
    report another tenant's number.
    """
    for tenant in (1, 2):
        health = shared.index_health(tenant=tenant)
        assert set(health) == {"fts", "csr", "vector"}
        assert health["fts"].tenant_id is None
        assert {r.tenant_id for r in health.values()} == {tenant, None}
        assert {name for name, r in health.items() if r.tenant_id is None} == {"fts"}
        reports = shared.maintain_indexes(tenant=tenant)
        assert reports["fts"].tenant_id is None
        assert {r.tenant_id for r in reports.values()} == {tenant, None}


def test_the_raw_cursor_of_a_shared_file_sees_every_tenant(shared):
    """Stated, not hidden.  A shared file is a namespace; DuckDB has no row-level security, so
    anything that skips anatid's predicate sees the whole file.  That is exactly why
    :class:`anatid.DatabasePool` exists and why the pool guards the cursor."""
    rows = (
        shared.unsafe_connection(reason="test assertion")
        .execute("SELECT DISTINCT tenant_id FROM memories ORDER BY tenant_id")
        .fetchall()
    )
    assert [r[0] for r in rows] == [1, 2]


# ============================================================================ file per tenant


@pytest.fixture
def pool(tmp_path):
    with DatabasePool(
        str(tmp_path / "pool" / "t_{tenant}.anatid"), embedding_dim=DIM, max_open=4
    ) as p:
        for tenant in (1, 2):
            db = p.get(tenant)
            _seed(db, tenant)
            db.rebuild_fts_index()
        yield p


@pytest.mark.parametrize("path", sorted(READS))
@pytest.mark.parametrize("tenant", (1, 2))
def test_a_pooled_read_returns_only_its_own_files_rows(pool, path, tenant):
    """The same reads against the real boundary.  The other tenant is not in the file at all."""
    _assert_only(READS[path](pool.get(tenant), tenant), tenant, path)


@pytest.mark.parametrize("path", sorted(READS))
def test_a_pooled_handle_refuses_to_read_another_tenant(pool, path):
    with pytest.raises(TenantIsolationError):
        READS[path](pool.get(1), 2)


def test_a_pooled_handle_refuses_to_write_another_tenant(pool):
    db = pool.get(1)
    for call in (
        lambda: db.remember("theirs", tenant=2, now=T0),
        lambda: db.relate(NAME, OTHER, tenant=2, now=T0),
        lambda: db.forget(IDS[1], tenant=2, now=T0),
        lambda: db.update(IDS[1], "theirs", tenant=2, now=T0),
        lambda: db.upsert_entity("Grace", tenant=2, now=T0),
        lambda: db.maintain_indexes(tenant=2),
    ):
        with pytest.raises(TenantIsolationError):
            call()
    assert db.resolve_tenant(1).tenant_id == 1


def test_a_pooled_files_raw_cursor_holds_one_tenant(pool):
    """The property a shared file cannot have: even the escape hatch sees only this tenant."""
    for tenant in (1, 2):
        rows = (
            pool.unsafe_connection(tenant, reason="test assertion")
            .execute("SELECT DISTINCT tenant_id FROM memories")
            .fetchall()
        )
        assert [r[0] for r in rows] == [tenant]
        contents = pool.unsafe_connection(tenant).execute("SELECT content FROM memories").fetchall()
        assert all(MARK[3 - tenant] not in c[0] for c in contents)


def test_the_doctor_of_a_pooled_file_sees_one_tenant(pool):
    """The operator's tool over one file.  In a pool that file is one tenant, so a count it
    reports is that tenant's; in a shared file it is the file's, which is the difference the
    two isolation levels are."""
    counts = [pool.get(t).doctor().counts["memories"] for t in (1, 2)]
    assert counts[0] == counts[1] == 5  # 4 logical memories, one of them with two versions
    for tenant in (1, 2):
        report = pool.get(tenant).doctor()
        assert [f for f in report.findings if f.severity.value == "error"] == []


def test_a_backup_holds_one_tenant_and_nothing_else(pool, tmp_path):
    dest = pool.backup(1, tmp_path / "backups" / "one.anatid")
    assert dest.is_file()
    assert stat.S_IMODE(os.stat(dest).st_mode) == 0o600
    con = duckdb.connect(str(dest), read_only=True)
    try:
        tenants = [r[0] for r in con.execute("SELECT DISTINCT tenant_id FROM memories").fetchall()]
        contents = [r[0] for r in con.execute("SELECT content FROM memories").fetchall()]
    finally:
        con.close()
    assert tenants == [1]
    assert contents and all("beta" not in c for c in contents)
    assert [e.action for e in pool.events(tenant=1)][-1] == "backup"
    with pytest.raises(FileExistsError):
        pool.backup(1, dest)
    assert pool.backup(1, dest, overwrite=True) == dest


def test_a_backup_is_a_committed_snapshot(pool, tmp_path):
    """``COPY FROM DATABASE`` runs inside the handle, so uncommitted work is not in the copy.

    Copying the file with ``cp`` while a writer is running promises nothing of the sort, which
    is why the pool does not do that.
    """
    db = pool.get(1)
    writer = db._root.cursor()  # a second connection, so the backup does not join its transaction
    writer.execute("BEGIN")
    writer.execute(
        "INSERT INTO memories (memory_id, tenant_id, content, created_at, valid_from, tx_from) "
        "VALUES (77, 1, 'alpha uncommitted', ?, ?, ?)",
        [T0, T0, T0],
    )
    try:
        dest = pool.backup(1, tmp_path / "snap.anatid")
    finally:
        writer.execute("ROLLBACK")
        writer.close()
    other = duckdb.connect(str(dest), read_only=True)
    try:
        assert (
            other.execute("SELECT count(*) FROM memories WHERE memory_id = 77").fetchone()[0] == 0
        )
    finally:
        other.close()


def test_a_backup_is_complete_even_before_a_checkpoint(pool, tmp_path):
    """Why the backup goes through DuckDB rather than through ``cp``.

    Most of a freshly written database lives in the write-ahead log, not in the ``.anatid``
    file, so copying that file alone yields a database with no tables in it.  ``COPY FROM
    DATABASE`` copies what the handle can see, which is everything committed, plus the schemas
    the BM25 index lives in.
    """
    src = pool.path_for(1)
    dest = pool.backup(1, tmp_path / "complete.anatid")
    con = duckdb.connect(str(dest), read_only=True)
    try:
        assert con.execute("SELECT count(*) FROM memories").fetchone()[0] == 5
        indexes = con.execute(
            "SELECT count(*) FROM information_schema.schemata WHERE schema_name LIKE 'fts_main%'"
        ).fetchone()[0]
        assert indexes == 1
    finally:
        con.close()
    assert dest.stat().st_size > src.stat().st_size


def test_deleting_a_tenant_removes_its_file_and_leaves_the_others(pool):
    one, two = pool.path_for(1), pool.path_for(2)
    assert one.is_file() and two.is_file()
    assert pool.delete(1) is True
    assert not one.exists() and two.is_file()
    assert pool.events(action="delete", tenant=1) != []
    assert pool.get(2).stats()["memories"] == 4
    # gone means gone: the next get() opens an empty file
    assert pool.get(1).stats()["memories"] == 0
    pool.close(1)
    pool.delete(1)
    assert pool.delete(1) is False
    with pytest.raises(FileNotFoundError):
        pool.delete(1, missing_ok=False)


# ============================================================================ paths


@pytest.mark.parametrize(
    "label",
    ["../escape", "..", ".", "", "a/b", "~root", "x" * 200, f"a{os.sep}b"],
)
def test_a_tenant_label_can_never_become_a_path(tmp_path, label):
    """The pool turns caller data into a filesystem path, so caller data is checked.

    Refused, not sanitised: rewriting ``"../x"`` to ``"x"`` would map two tenants onto one
    file, which is the same leak by another route.
    """
    with DatabasePool(str(tmp_path / "root" / "{label}.anatid"), embedding_dim=DIM) as pool:
        with pytest.raises(TenantIsolationError):
            pool.path_for(Namespace(7, label))
        with pytest.raises(TenantIsolationError):
            pool.get(Namespace(7, label))
        assert list((tmp_path / "root").glob("*")) == []  # nothing was created on the way out


def test_a_traversal_attempt_is_audited(tmp_path):
    seen: list[PoolEvent] = []
    with DatabasePool(
        str(tmp_path / "root" / "{label}.anatid"), embedding_dim=DIM, audit=seen.append
    ) as pool:
        with pytest.raises(TenantIsolationError):
            pool.path_for(Namespace(7, "../../etc/passwd"))
    assert [e.action for e in seen] == ["rejected"]
    assert seen[0].tenant_id == 7


def test_a_label_that_is_a_plain_name_still_works(tmp_path):
    with DatabasePool(str(tmp_path / "root" / "{label}.anatid"), embedding_dim=DIM) as pool:
        db = pool.get(Namespace(7, "acme"))
        db.remember("hello", now=T0)
        assert (tmp_path / "root" / "acme.anatid").is_file()
        assert pool.registry() == {7: str(tmp_path / "root" / "acme.anatid")}


def test_opaque_naming_keeps_tenant_names_out_of_the_directory(tmp_path):
    with DatabasePool(
        str(tmp_path / "root" / "{label}.anatid"), opaque=True, embedding_dim=DIM
    ) as pool:
        for tenant in (1, 2):
            pool.get(Namespace(tenant, f"customer-{tenant}")).remember(f"{MARK[tenant]}", now=T0)
        files = sorted((tmp_path / "root").glob("*.anatid"))
        assert len(files) == 2
        assert not any("customer" in f.name or f.stem in ("1", "2") for f in files)
        # one tenant is one file however it is addressed, and the map back is kept
        assert pool.path_for(1) == pool.path_for(Namespace(1, "renamed since"))
        assert set(pool.registry()) == {1, 2}
        assert {os.path.basename(p) for p in pool.registry().values()} == {f.name for f in files}


def test_an_opaque_secret_changes_every_file_name(tmp_path):
    plain = DatabasePool(str(tmp_path / "r" / "{tenant}.anatid"), opaque=True)
    keyed = DatabasePool(str(tmp_path / "r" / "{tenant}.anatid"), opaque=True, secret=b"pepper")
    assert plain.path_for(1) != keyed.path_for(1)
    assert plain.path_for(1) == DatabasePool(
        str(tmp_path / "r" / "{tenant}.anatid"), opaque=True
    ).path_for(1)


def test_a_template_without_a_tenant_field_is_refused(tmp_path):
    with pytest.raises(ValueError):
        DatabasePool(str(tmp_path / "one-file.anatid"))
    with pytest.raises(ValueError):
        DatabasePool(str(tmp_path / "{nope}_{tenant}.anatid")).path_for(1)
    with pytest.raises(ValueError):
        DatabasePool(str(tmp_path / "t_{tenant}.anatid"), max_open=0)
    with pytest.raises(ValueError):
        DatabasePool(str(tmp_path / "t_{tenant}.anatid"), raw_access="sometimes")


def test_the_pool_root_is_the_fixed_prefix_of_the_template(tmp_path):
    pool = DatabasePool(str(tmp_path / "a" / "t_{tenant}.anatid"))
    assert pool.root == tmp_path / "a"
    nested = DatabasePool(str(tmp_path / "b" / "{label}" / "db.anatid"))
    assert nested.root == tmp_path / "b"
    explicit = DatabasePool(str(tmp_path / "c" / "t_{tenant}.anatid"), root=tmp_path)
    assert explicit.root == tmp_path


# ============================================================================ permissions


def test_files_and_directories_the_pool_creates_are_private(tmp_path):
    root = tmp_path / "private"
    with DatabasePool(str(root / "t_{tenant}.anatid"), embedding_dim=DIM) as pool:
        db = pool.get(1)
        db.remember("alpha", now=T0)
        path = pool.path_for(1)
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        # the directory is what protects the write-ahead log, which DuckDB creates on its own
        assert stat.S_IMODE(os.stat(root).st_mode) == 0o700
        hardened = pool.harden(1)
        assert str(path) in hardened
        for extra in hardened:
            assert stat.S_IMODE(os.stat(extra).st_mode) == 0o600


def test_permission_hardening_can_be_turned_off(tmp_path):
    """A deployment that manages modes itself (an ACL, a group-readable backup mount) says so
    once instead of fighting the pool on every open."""
    root = tmp_path / "loose"
    with DatabasePool(
        str(root / "t_{tenant}.anatid"), embedding_dim=DIM, file_mode=None, dir_mode=None
    ) as pool:
        pool.get(1).remember("alpha", now=T0)
        assert pool.harden(1) == []
        assert pool.path_for(1).is_file()


# ============================================================================ raw cursors


def test_a_pooled_handle_warns_once_about_the_raw_cursor(tmp_path, caplog):
    with DatabasePool(str(tmp_path / "p" / "t_{tenant}.anatid"), embedding_dim=DIM) as pool:
        db = pool.get(1)
        with caplog.at_level("WARNING", logger="anatid"):
            first = db.connection
            second = db.connection
        assert first is second
        assert sum("raw DuckDB cursor" in r.message for r in caplog.records) == 1
        assert [e.action for e in pool.events(action="raw_access")] != []
        # the verbs are not the ones being warned about: they are anatid
        db.remember("alpha", now=T0)
        assert sum("raw DuckDB cursor" in r.message for r in caplog.records) == 1


def test_a_denying_pool_hands_out_a_cursor_only_through_unsafe_connection(tmp_path):
    with DatabasePool(
        str(tmp_path / "p" / "t_{tenant}.anatid"), embedding_dim=DIM, raw_access="deny"
    ) as pool:
        db = pool.get(1)
        with pytest.raises(TenantIsolationError) as excinfo:
            db.connection
        assert "unsafe_connection" in str(excinfo.value)
        assert [e.action for e in pool.events(action="raw_access_denied")] != []
        # the verbs keep working: they are inside anatid, which is the point of the check
        db.remember("alpha", entities=[NAME], now=T0)
        assert db.recall_2hop(NAME)[0].content == "alpha"
        # and the named way in still works, with a reason on the audit event
        con = db.unsafe_connection(reason="incident 4")
        assert con.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
        event = pool.events(action="unsafe_connection")[-1]
        assert event.detail == "incident 4" and event.tenant_id == 1


def test_an_allowing_pool_is_the_unguarded_old_behaviour(tmp_path):
    with DatabasePool(
        str(tmp_path / "p" / "t_{tenant}.anatid"), embedding_dim=DIM, raw_access="allow"
    ) as pool:
        db = pool.get(1)
        assert db.connection.execute("SELECT 1").fetchone() == (1,)
        assert pool.events(action="raw_access") == []


def test_a_directly_opened_handle_is_not_guarded(db):
    """The guard belongs to the pool.  A handle you opened yourself is yours."""
    assert db.connection.execute("SELECT 1").fetchone() == (1,)
    assert db.unsafe_connection(reason="direct") is db.connection


# ============================================================================ pool mechanics


def test_eviction_closes_the_least_recently_used_file_and_says_so(tmp_path):
    seen: list[PoolEvent] = []
    with DatabasePool(
        str(tmp_path / "p" / "t_{tenant}.anatid"), max_open=2, embedding_dim=DIM, audit=seen.append
    ) as pool:
        one, two = pool.get(1), pool.get(2)
        one.remember("alpha", now=T0)
        two.remember("beta", now=T0)
        three = pool.get(3)
        assert len(pool) == 2 and one.closed and pool.known_tenants() == [2, 3]
        three.remember("gamma", now=T0)
        evictions = [e for e in seen if e.action == "evict"]
        assert [e.tenant_id for e in evictions] == [1]
        again = pool.get(1)
        assert not again.closed and again.stats()["memories"] == 1
        assert again is not one
    assert again.closed
    assert [e.action for e in seen if e.action == "close"]


def test_a_closed_pool_hands_out_nothing(tmp_path):
    pool = DatabasePool(str(tmp_path / "p" / "t_{tenant}.anatid"), embedding_dim=DIM)
    pool.get(1)
    pool.close_all()
    with pytest.raises(AnatidError):
        pool.get(1)


def test_an_audit_hook_that_fails_does_not_break_the_operation(tmp_path):
    def angry(_event):
        raise RuntimeError("the audit sink is down")

    with DatabasePool(
        str(tmp_path / "p" / "t_{tenant}.anatid"), embedding_dim=DIM, audit=angry
    ) as pool:
        db = pool.get(1)
        db.remember("alpha", now=T0)
        assert db.stats()["memories"] == 1
        assert [e.action for e in pool.events()] == ["open"]


def test_events_can_be_filtered_and_are_bounded(tmp_path):
    with DatabasePool(
        str(tmp_path / "p" / "t_{tenant}.anatid"), embedding_dim=DIM, audit_size=3
    ) as pool:
        for tenant in range(6):
            pool.get(tenant)
        assert len(pool.events()) == 3
        assert pool.events(tenant=5)[0].as_dict()["action"] == "open"
        assert pool.events(action="nothing-of-the-sort") == []


def test_cross_tenant_reads_still_need_an_explicit_read_only_attach(tmp_path):
    with DatabasePool(str(tmp_path / "p" / "t_{tenant}.anatid"), embedding_dim=DIM) as pool:
        pool.get(1).remember("alpha secret", now=T0)
        pool.get(2).remember("beta secret", now=T0)
        alias = pool.attach_read_only(1, 2)
        host = pool.get(1)
        con = host.unsafe_connection(reason="support request")
        assert con.execute(f"SELECT content FROM {alias}.memories").fetchall() == [("beta secret",)]
        with pytest.raises(duckdb.Error):
            con.execute(f"DELETE FROM {alias}.memories")
        assert [e.action for e in pool.events(action="attach")] == ["attach"]
        # anatid's own verbs never look outside main, attached or not
        assert host.stats()["memories"] == 1
        assert [m.content for m in host.recall_2hop(0, limit=10)] == []
        host.detach(alias)
        assert host.attached == {}


def test_pooled_handles_are_file_per_tenant(tmp_path):
    with DatabasePool(str(tmp_path / "p" / "t_{tenant}.anatid"), embedding_dim=DIM) as pool:
        db = pool.get(1)
        assert db.namespace.isolation is Isolation.FILE_PER_TENANT
        assert db is pool[1]
        assert len(pool) == 1 and pool.known_tenants() == [1]
