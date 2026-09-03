"""Immutable versions (schema v4): a correction closes a version and inserts its successor.

anatid 0.1.1 made the bitemporal claim and did not keep it: ``supersede`` and soft ``forget``
rewrote ``valid_to`` in place and never closed ``tx_to``, so asking what the database believed
on Jan 2 about Jan 4, after a forget on Jan 3, returned nothing instead of the belief it held
on Jan 2.  The first test here is that exact scenario.  The rest pin the version model on
every verb that corrects a row, on both edge tables, on the receipts, on the doctor and on the
3->4 migration.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from anatid import Anatid, AsOf, ConflictError, NotFoundError, RangeError
from anatid import schema as S

from conftest import DIM, T0, vec
from test_schema_v3 import add_memory, v2_file


def jan(day: int) -> _dt.datetime:
    return _dt.datetime(2026, 1, day)


# --------------------------------------------------------------------------- the reviewer's case


def test_soft_forget_keeps_the_belief_the_database_held_before_it(db):
    """Created Jan 1, forgotten Jan 3.

    ``as_of(valid_at=Jan 4, tx_at=Jan 2)`` returns the old open-ended belief: on Jan 2 the
    database had not yet recorded the forget.  ``as_of(valid_at=Jan 4, tx_at=Jan 4)`` returns
    nothing: by Jan 4 it knew the fact stopped being true on Jan 3.
    """
    db.relate("Ada", "coffee", now=jan(1))
    m = db.remember("Ada drinks coffee", entities=["Ada"], embedding=vec(1, 0), now=jan(1))
    db.forget(m.memory_id, now=jan(3))

    believed_on_jan2 = AsOf(valid_time=jan(4), tx_time=jan(2))
    believed_on_jan4 = AsOf(valid_time=jan(4), tx_time=jan(4))

    old = db.get(m.memory_id, as_of=believed_on_jan2)
    assert old is not None
    assert old.valid_to is None and old.version == 1 and old.tx_to == jan(3)
    assert db.get(m.memory_id, as_of=believed_on_jan4) is None

    # the same answer from every read verb
    assert [x.memory_id for x in db.recall_2hop("Ada", as_of=believed_on_jan2)] == [m.memory_id]
    assert db.recall_2hop("Ada", as_of=believed_on_jan4) == []
    assert [e.name for e in db.entities_of(m.memory_id, as_of=believed_on_jan2)] == ["Ada"]
    assert db.entities_of(m.memory_id, as_of=believed_on_jan4) == []
    hits = db.recall(embedding=vec(1, 0), seed_entity="Ada", as_of=believed_on_jan2, k=5)
    assert [h.memory_id for h in hits] == [m.memory_id]
    assert hits[0].memory.valid_to is None and hits[0].memory.version == 1
    later = db.recall(embedding=vec(1, 0), seed_entity="Ada", as_of=believed_on_jan4, k=5)
    assert list(later) == []
    view = db.as_of(believed_on_jan2)
    assert [x.memory_id for x in view.recall_2hop("Ada")] == [m.memory_id]
    assert view.get(m.memory_id).valid_to is None

    # the current state: nothing is believed, and the live version says when belief ended
    assert db.recall_2hop("Ada") == []
    live = db.get(m.memory_id)
    assert live is not None
    assert (live.version, live.valid_to, live.tx_from, live.tx_to) == (2, jan(3), jan(3), None)
    assert not live.is_current and live.is_live
    # a valid-time-only as_of goes through the live version's corrected interval
    assert db.get(m.memory_id, as_of=AsOf(valid_time=jan(2))).version == 2
    assert db.get(m.memory_id, as_of=AsOf(valid_time=jan(3))) is None

    versions = db.versions(m.memory_id)
    assert [(v.version, v.valid_from, v.valid_to, v.tx_from, v.tx_to) for v in versions] == [
        (1, jan(1), None, jan(1), jan(3)),
        (2, jan(1), jan(3), jan(3), None),
    ]
    assert all(v.memory_id == m.memory_id and v.content == "Ada drinks coffee" for v in versions)
    assert versions[0].is_live is False and versions[1].is_live is True


# --------------------------------------------------------------------------- supersede


def test_supersede_versions_the_old_memory_and_never_rewrites_it(db):
    old = db.remember("dark roast", entities=["Ada"], now=jan(1))
    new = db.supersede(old.memory_id, "decaf", now=jan(3))

    versions = db.versions(old.memory_id)
    assert [(v.version, v.valid_from, v.valid_to, v.tx_from, v.tx_to) for v in versions] == [
        (1, jan(1), None, jan(1), jan(3)),
        (2, jan(1), jan(3), jan(3), None),
    ]
    assert all(v.content == "dark roast" for v in versions)
    assert db.get(old.memory_id).version == 2 and not db.get(old.memory_id).is_current

    # on Jan 2 the database believed the old memory open-ended and knew nothing of the new one
    jan2 = AsOf(valid_time=jan(4), tx_time=jan(2))
    assert db.get(old.memory_id, as_of=jan2).valid_to is None
    assert db.get(new.memory_id, as_of=jan2) is None
    assert [x.memory_id for x in db.recall_2hop("Ada", as_of=jan2)] == [old.memory_id]
    jan4 = AsOf(valid_time=jan(4), tx_time=jan(4))
    assert [x.memory_id for x in db.recall_2hop("Ada", as_of=jan4)] == [new.memory_id]

    # provenance: the SUPERSEDES chain links logical ids, versions belong to the id asked about
    prov = db.provenance(old.memory_id)
    assert [m.memory_id for m in prov.chain] == [old.memory_id]
    assert [v.version for v in prov.versions] == [1, 2]
    prov_new = db.provenance(new.memory_id)
    assert [m.memory_id for m in prov_new.chain] == [new.memory_id, old.memory_id]
    assert [v.version for v in prov_new.versions] == [1]
    assert prov_new.chain[1].version == 2  # the chain reads live versions

    # a second supersede finds no current version: reported, not applied as a third version
    with pytest.raises(ConflictError):
        db.supersede(old.memory_id, "again", now=jan(5))
    assert len(db.versions(old.memory_id)) == 2


def test_an_already_closed_memory_gets_no_new_version_from_a_second_forget(db):
    m = db.remember("once", now=jan(1))
    first = db.forget(m.memory_id, now=jan(2), reason="first")
    second = db.forget(m.memory_id, now=jan(4), reason="second")
    assert first.audit_rows_written == second.audit_rows_written == 1
    assert [(v.version, v.valid_to) for v in db.versions(m.memory_id)] == [(1, None), (2, jan(2))]
    assert db.get(m.memory_id).valid_to == jan(2)  # the first forget's answer stands


# --------------------------------------------------------------------------- edges


def test_about_and_relates_edges_are_versioned_when_closed(db):
    db.relate("Ada", "coffee", rel_kind="likes", now=jan(1))
    db.relate("Ada", "tea", now=jan(1))
    m = db.remember("Ada drinks coffee", entities=["Ada", "coffee"], now=jan(1))
    db.supersede(
        m.memory_id, "Ada drinks tea", entities=["Ada"], now=jan(3), close_about_edges=True
    )

    about = db.execute(
        "SELECT edge_id, version, valid_to, tx_to FROM edges_about WHERE src = ? "
        "ORDER BY edge_id, version",
        [m.memory_id],
    ).fetchall()
    edge_ids = sorted({r[0] for r in about})
    assert len(edge_ids) == 2 and len(about) == 4
    for eid in edge_ids:
        rows = [(r[1], r[2], r[3]) for r in about if r[0] == eid]
        assert rows == [(1, None, jan(3)), (2, jan(3), None)]
    jan2 = AsOf(valid_time=jan(4), tx_time=jan(2))
    assert sorted(e.name for e in db.entities_of(m.memory_id, as_of=jan2)) == ["Ada", "coffee"]
    assert db.entities_of(m.memory_id) == []

    # RELATES_TO: unrelate is undirected, closes by version, and history stays traversable
    c = db.remember("coffee is roasted", entities=["coffee"], now=jan(1))
    assert c.memory_id in [x.memory_id for x in db.recall_2hop("Ada")]
    assert db.unrelate("coffee", "Ada", now=jan(5)) == 1
    relates = db.execute(
        "SELECT version, valid_to, tx_to FROM edges_relates WHERE rel_kind = 'likes' "
        "ORDER BY version"
    ).fetchall()
    assert relates == [(1, None, jan(5)), (2, jan(5), None)]
    assert c.memory_id not in [x.memory_id for x in db.recall_2hop("Ada")]
    assert c.memory_id in [x.memory_id for x in db.recall_2hop("Ada", as_of=jan(4))]
    still_believed = AsOf(valid_time=jan(6), tx_time=jan(4))
    assert c.memory_id in [x.memory_id for x in db.recall_2hop("Ada", as_of=still_believed)]
    known_closed = AsOf(valid_time=jan(6), tx_time=jan(6))
    assert c.memory_id not in [x.memory_id for x in db.recall_2hop("Ada", as_of=known_closed)]
    # nothing current to close, a rel_kind that matches nothing, then the last edge
    assert db.unrelate("Ada", "coffee", now=jan(6)) == 0
    assert db.unrelate("Ada", "tea", rel_kind="nope", now=jan(6)) == 0
    assert db.unrelate("Ada", "tea", now=jan(6)) == 1
    assert db.expand_path == "sql"
    with pytest.raises(NotFoundError):
        db.unrelate("Ada", "nobody", now=jan(6))
    assert db.doctor().ok


# --------------------------------------------------------------------------- reinforce


def test_reinforce_counters_are_in_place_and_a_confidence_change_is_a_version(db):
    m = db.remember("x", now=jan(1))
    db.reinforce(m.memory_id, now=jan(2))
    assert [(v.version, v.access_count) for v in db.versions(m.memory_id)] == [(1, 1)]

    got = db.reinforce(m.memory_id, amount=2, confidence=0.4, now=jan(3))
    assert got.version == 2 and got.access_count == 3
    assert got.confidence == pytest.approx(0.4)
    versions = db.versions(m.memory_id)
    assert [(v.version, v.access_count, v.tx_from, v.tx_to) for v in versions] == [
        (1, 1, jan(1), jan(3)),
        (2, 3, jan(3), None),
    ]
    assert versions[0].confidence == pytest.approx(1.0)
    # on the transaction axis the old confidence is what the database believed
    assert db.get(m.memory_id, as_of=AsOf(tx_time=jan(2))).confidence == pytest.approx(1.0)
    assert db.get(m.memory_id, as_of=AsOf(tx_time=jan(4))).confidence == pytest.approx(0.4)
    # counters are not bitemporal: version 1 froze at the count it had when it was closed
    db.reinforce(m.memory_id, amount=5, now=jan(5))
    assert db.versions(m.memory_id)[0].access_count == 1
    assert db.get(m.memory_id).access_count == 8
    assert len(db.versions(m.memory_id)) == 2


# --------------------------------------------------------------------------- guards


def test_a_correction_cannot_be_recorded_before_what_it_corrects(db):
    m = db.remember("back-dated", valid_from=jan(1), now=jan(5))
    with pytest.raises(RangeError) as exc:
        db.forget(m.memory_id, now=jan(3))  # valid_from is fine, tx_from is not
    assert exc.value.field == "now" and "tx_from" in str(exc.value)
    with pytest.raises(RangeError):
        db.supersede(m.memory_id, "x", now=jan(3))
    with pytest.raises(RangeError):
        db.reinforce(m.memory_id, confidence=0.5, now=jan(3))
    assert len(db.versions(m.memory_id)) == 1
    assert db.execute("SELECT count(*) FROM anatid_audit").fetchone()[0] == 0
    db.relate("a", "b", now=jan(5))
    with pytest.raises(RangeError):
        db.unrelate("a", "b", now=jan(3))
    assert db.doctor().find("timestamp_order") is None
    db.forget(m.memory_id, now=jan(5))  # closing at the recording instant is allowed
    assert [(v.version, v.tx_from, v.tx_to) for v in db.versions(m.memory_id)] == [
        (1, jan(5), jan(5)),
        (2, jan(5), None),
    ]


def test_doctor_tells_versions_from_duplicates(db):
    m = db.remember("v", now=jan(1))
    db.forget(m.memory_id, now=jan(2))
    report = db.doctor()
    assert report.find("duplicate_memory_ids") is None and report.ok
    # two LIVE versions of one id is the fault
    db.execute(
        "UPDATE memories SET tx_to = NULL WHERE memory_id = ? AND version = 1", [m.memory_id]
    )
    f = db.doctor().find("duplicate_memory_ids")
    assert f is not None and f.samples[0][:2] == (1, m.memory_id)
    # so is a repeated version number
    db.execute(
        "UPDATE memories SET tx_to = ? WHERE memory_id = ? AND version = 1", [jan(2), m.memory_id]
    )
    assert db.doctor().find("duplicate_memory_ids") is None
    db.execute("UPDATE memories SET version = 2 WHERE memory_id = ? AND version = 1", [m.memory_id])
    assert db.doctor().find("duplicate_memory_ids") is not None


# --------------------------------------------------------------------------- erasure


def test_hard_forget_erases_every_version_and_the_receipt_counts_them(db):
    m = db.remember("secret", entities=["Ada"], now=jan(1))
    db.forget(m.memory_id, now=jan(2))
    r = db.forget(m.memory_id, hard=True, now=jan(3))
    assert (r.memories_deleted, r.memory_versions_deleted) == (1, 2)
    assert (r.about_edges_deleted, r.about_edge_versions_deleted) == (1, 2)
    assert r.rows_removed >= 4
    assert db.versions(m.memory_id) == []
    assert db.get(m.memory_id, as_of=AsOf(valid_time=jan(1), tx_time=jan(1))) is None
    for table in ("memories", "edges_about"):
        col = "memory_id" if table == "memories" else "src"
        assert (
            db.execute(f"SELECT count(*) FROM {table} WHERE {col} = ?", [m.memory_id]).fetchone()[0]
            == 0
        )


def test_a_correction_is_not_a_new_document_for_the_index_or_the_counts(db):
    m = db.remember("alpha alpha", now=jan(1))
    db.remember("beta", now=jan(1))
    db.rebuild_fts_index(now=jan(1))
    db.forget(m.memory_id, now=jan(2))
    status = db.fts_status()
    assert (status.stale, status.current_rows, status.indexed_rows) == (False, 2, 2)
    stats = db.stats()
    assert (stats["memories"], stats["memory_versions"], stats["current_memories"]) == (2, 3, 1)
    # the BM25 arm narrows candidates; visibility decides, on both axes
    assert [h.memory_id for h in db.recall("alpha", k=5)] == []
    believed = AsOf(valid_time=jan(3), tx_time=jan(1))
    hits = db.recall("alpha", k=5, as_of=believed)
    assert [h.memory_id for h in hits] == [m.memory_id] and hits[0].memory.version == 1


# --------------------------------------------------------------------------- migration


def test_migration_carries_existing_rows_over_as_version_1(tmp_path):
    """A v2 file (what 0.1.1 wrote, with valid_to rewritten in place) migrates without copying
    a row: every row is version 1 with ``tx_to`` untouched, and corrections made afterwards
    version it."""
    path = tmp_path / "v2.anatid"
    con = v2_file(path)
    add_memory(con, 1, 1, "kept")
    add_memory(con, 2, 1, "closed in place")
    con.execute("UPDATE memories SET valid_to = ? WHERE memory_id = 2", [T0])
    assert S.ensure_schema(con) == S.SCHEMA_VERSION
    for table in S.VERSIONED_TABLES:
        cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
        assert S.VERSION_COLUMN in cols, table
    assert con.execute(
        "SELECT memory_id, version, valid_to IS NULL, tx_to FROM memories ORDER BY memory_id"
    ).fetchall() == [(1, 1, True, None), (2, 1, False, None)]
    assert S.ensure_version_columns(con) == []  # idempotent
    con.close()

    with Anatid.open(path, tenant=1, embedding_dim=DIM) as db:
        assert db.get(2).valid_to == T0 and db.get(2).version == 1
        assert db.doctor(all_tenants=True).find("schema_drift") is None
        db.supersede(1, "replaced", now=T0 + _dt.timedelta(days=1))
        assert [v.version for v in db.versions(1)] == [1, 2]
        # a raw insert on the migrated table gets version 1 from the DEFAULT as well
        db.execute(
            "INSERT INTO memories (memory_id, tenant_id, content, created_at, valid_from, "
            "tx_from) VALUES (3, 1, 'raw', ?, ?, ?)",
            [T0, T0, T0],
        )
        assert db.get(3).version == 1
        db.forget(3, now=T0 + _dt.timedelta(days=2))
        assert [v.version for v in db.versions(3)] == [1, 2]


def test_a_v4_file_without_the_version_column_is_repaired_on_open(tmp_path):
    """A file written by a v4 build that predates the column: doctor names the drift, and the
    next open adds the column, so the verbs can version its rows."""
    path = tmp_path / "early-v4.anatid"
    with Anatid.open(path, tenant=1, embedding_dim=DIM) as db:
        m = db.remember("x", entities=["Ada"], now=T0)
        db.execute("DROP INDEX idx_memories_id")  # DuckDB refuses to alter an indexed table
        for table in S.VERSIONED_TABLES:
            db.execute(f"ALTER TABLE {table} DROP COLUMN {S.VERSION_COLUMN}")
        drift = db.doctor().find("schema_drift")
        assert drift is not None
        assert ("missing_column", "memories", S.VERSION_COLUMN) in drift.samples
    with Anatid.open(path, tenant=1, embedding_dim=DIM) as db:
        assert db.doctor().find("schema_drift") is None
        assert db.get(m.memory_id).version == 1
        db.forget(m.memory_id, now=T0 + _dt.timedelta(days=1))
        assert [v.version for v in db.versions(m.memory_id)] == [1, 2]
        assert [e.name for e in db.entities_of(m.memory_id, as_of=T0)] == ["Ada"]
        assert db.execute(
            "SELECT list(version ORDER BY version) FROM edges_about WHERE src = ?",
            [m.memory_id],
        ).fetchone()[0] == [1, 2]
