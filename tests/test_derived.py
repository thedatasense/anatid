"""The derived-index framework: generations, the journal, publication, health, erasure.

A small concrete index (:class:`TableIndex`) stands in for the accelerators: its base
generation is one table of ``(tenant_id, memory_id)`` per generation, and its read merges
base + pending - tombstones and then applies :class:`anatid.visibility.Visibility` to the
canonical rows.  Every property the design document promises is checked here against the SQL
oracle, including under a random sequence of mutations with rebuilds interleaved.

The candidate key is ``(tenant_id, doc_id)`` throughout, including for an index whose
generations cover the file: ``memory_id`` is unique within a tenant only, so a bare id lets one
tenant's change land on another tenant's document.
"""

from __future__ import annotations

import datetime as _dt
import random
import threading
import time
from dataclasses import replace

import duckdb
import pytest

from anatid import (
    Anatid,
    AsOf,
    ConflictError,
    DerivedIndex,
    Generation,
    HealthReason,
    IndexDefinition,
    IndexGenerationError,
    IndexRegistry,
    IndexValidationError,
    MaintenancePolicy,
    ValidationReport,
    Visibility,
    maintain,
)
from anatid import derived as derived_mod
from anatid import schema as S
from anatid.derived import NullIndex

from conftest import DIM, T0

MINUTE = _dt.timedelta(minutes=1)


# --------------------------------------------------------------------------- a concrete index


class TableIndex(DerivedIndex):
    """Base generation = a table of ``(tenant_id, memory_id)``, built beside the last one."""

    name = "tidx"
    kind = "test_table"
    source_table = "memories"
    source_id_column = "memory_id"

    def __init__(self, db, *, name=None, fail_validation_from: int | None = None):
        super().__init__(db, name=name)
        self.fail_validation_from = fail_validation_from
        self.built: list[int] = []

    def table(self, gen: Generation) -> str:
        return gen.storage_name()

    def _scope(self, gen: Generation) -> tuple[str, list]:
        if gen.tenant_id is None:
            return "TRUE", []
        return "tenant_id = ?", [gen.tenant_id]

    def _build(self, gen: Generation):
        tbl = self.table(gen)
        where, params = self._scope(gen)
        self.db.execute(f"DROP TABLE IF EXISTS {tbl}")
        self.db.execute(
            f"CREATE TABLE {tbl} AS SELECT tenant_id, memory_id FROM memories WHERE {where}",
            params,
        )
        n = self.db.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
        self.built.append(gen.generation)
        return {"rows": int(n)}

    def _validate(self, gen: Generation) -> ValidationReport:
        if self.fail_validation_from is not None and gen.generation >= self.fail_validation_from:
            return ValidationReport(ok=False, generation=gen, detail="told to fail")
        where, params = self._scope(gen)
        canonical = {
            (int(r[0]), int(r[1]))
            for r in self.db.execute(
                f"SELECT tenant_id, memory_id FROM memories WHERE {where}", params
            ).fetchall()
        }
        base = self.base_keys(gen)
        pending = set(self.pending_keys(gen.tenant_id, gen))
        tombs = set(self.tombstone_keys(gen.tenant_id, gen))
        # The contract the merge has to meet: every canonical document is in base + pending
        # UNLESS the journal has tombstoned it, and every document in the base is canonical
        # unless the journal has tombstoned it.  A tombstone is written on close, so a memory
        # that was superseded or soft-forgotten after the build is legitimately in neither the
        # base nor the pending set: the merge subtracts it and the visibility predicate would
        # have dropped it anyway.  Not subtracting it here made the check report a mismatch
        # for a correct merge.
        missing = canonical - base - pending - tombs
        extra = (base - canonical) - tombs
        ok = not missing and not extra
        return ValidationReport(
            ok=ok,
            generation=gen,
            checked=len(canonical),
            mismatches=tuple(sorted(missing | extra)),
            detail="" if ok else f"missing={sorted(missing)} extra={sorted(extra)}",
        )

    def _drop(self, gen: Generation) -> None:
        self.db.execute(f"DROP TABLE IF EXISTS {self.table(gen)}")

    def _erase(self, gen: Generation, tenant_id: int, doc_ids) -> int | None:
        """The storage can delete one document, so an erasure does not cost the generation."""
        if self.table(gen) not in tables(self.db):
            return 0
        marks = ", ".join(str(int(d)) for d in doc_ids)
        row = self.db.execute(
            f"DELETE FROM {self.table(gen)} WHERE tenant_id = ? AND memory_id IN ({marks})",
            [int(tenant_id)],
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    # ---- the read side an accelerator writes
    def base_keys(self, gen: Generation) -> set[tuple[int, int]]:
        return {
            (int(r[0]), int(r[1]))
            for r in self.db.execute(
                f"SELECT tenant_id, memory_id FROM {self.table(gen)}"
            ).fetchall()
        }

    def base(self, gen: Generation, tenant: int | None = None) -> set[int]:
        return {doc for t, doc in self.base_keys(gen) if tenant is None or t == int(tenant)}

    def candidates(self, tenant: int, as_of=None) -> tuple[set[int] | None, str]:
        with self.pin(tenant, as_of=as_of) as pin:
            if not pin:
                return None, pin.reason.value
            gen = pin.generation
            merged = (self.base_keys(gen) | set(self.pending_keys(tenant, gen))) - set(
                self.tombstone_keys(tenant, gen)
            )
            ids = {doc for t, doc in merged if t == int(tenant)}
            if not ids:
                return set(), pin.reason.value
            w, wp = Visibility.at(tenant, as_of).predicate("m")
            marks = ", ".join(str(int(i)) for i in ids)
            rows = self.db.execute(
                f"SELECT m.memory_id FROM memories m WHERE m.memory_id IN ({marks}) AND {w}", wp
            ).fetchall()
            return {int(r[0]) for r in rows}, pin.reason.value


class FileWide(TableIndex):
    """One generation for the whole file, as the BM25 index has.  The journal is still keyed
    by tenant, which is the only thing keeping two tenants' document 42 apart."""

    name = "wide"
    per_tenant = False


class ContentIndex(TableIndex):
    """A base generation that stores the CONTENT, as a real full-text index does.

    The id-only :class:`TableIndex` cannot show what an erasure that misses a generation costs:
    an accelerator keeps a copy of what it indexed, and that copy is the erased text.
    """

    name = "tidx_content"

    def _build(self, gen: Generation):
        tbl = self.table(gen)
        where, params = self._scope(gen)
        self.db.execute(f"DROP TABLE IF EXISTS {tbl}")
        self.db.execute(
            f"CREATE TABLE {tbl} AS SELECT tenant_id, memory_id, content FROM memories "
            f"WHERE {where}",
            params,
        )
        n = self.db.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
        self.built.append(gen.generation)
        return {"rows": int(n)}


def sql_oracle(db, tenant: int, as_of=None) -> set[int]:
    w, wp = Visibility.at(tenant, as_of).predicate("m")
    return {
        int(r[0])
        for r in db.execute(f"SELECT m.memory_id FROM memories m WHERE {w}", wp).fetchall()
    }


def tables(db) -> set[str]:
    return S.table_names(db.connection)


def journal(db, index_name: str = "tidx") -> list[tuple]:
    """``(tenant_id, doc_id, op, absorbed_by)`` in journal order."""
    return [
        (int(r[0]), int(r[1]), str(r[2]), r[3])
        for r in db.execute(
            f"SELECT tenant_id, doc_id, op, absorbed_by FROM {S.INDEX_JOURNAL_TABLE} "
            f"WHERE index_name = ? ORDER BY change_seq",
            [index_name],
        ).fetchall()
    ]


def find_in_file(db, *needles: str) -> list[tuple[str, str, str]]:
    """Every ``(schema.table, column, needle)`` in the whole file whose text holds a needle.

    Enumerated from the catalog rather than from a list anatid keeps, so a table added by any
    part of the system is scanned.  Values are cast to VARCHAR, which covers timestamps, JSON,
    embeddings and ids alike.
    """
    hits: list[tuple[str, str, str]] = []
    objects = db.execute(
        "SELECT table_schema, table_name FROM information_schema.tables ORDER BY 1, 2"
    ).fetchall()
    for schema_name, table_name in objects:
        if schema_name in ("information_schema", "pg_catalog"):
            continue
        qualified = f'"{schema_name}"."{table_name}"'
        try:
            columns = [r[0] for r in db.execute(f"DESCRIBE {qualified}").fetchall()]
        except duckdb.Error:  # pragma: no cover - a view over something dropped
            continue
        for column in columns:
            for needle in needles:
                sql = (
                    f"SELECT count(*) FROM {qualified} "
                    f'WHERE contains(CAST("{column}" AS VARCHAR), ?)'
                )
                try:
                    n = db.execute(sql, [needle]).fetchone()[0]
                except duckdb.Error:  # pragma: no cover - a type with no VARCHAR cast
                    continue
                if n:
                    hits.append((f"{schema_name}.{table_name}", column, needle))
    return hits


@pytest.fixture
def db(legacy_db):
    """The framework, not the accelerators built on it.

    ``Anatid.open(accelerators=False)``: this file registers its own :class:`TableIndex` over
    the placeholder registry and counts the journal rows it produces, so the shipped full-text
    and CSR definitions would be noise in every assertion here.  What they do with the same
    machinery is ``tests/test_fts_framework.py`` and ``tests/test_csr_framework.py``.
    """
    return legacy_db


@pytest.fixture
def file_db(legacy_file_db):
    """:func:`db` on disk.  Same reason."""
    return legacy_file_db


@pytest.fixture
def idx(file_db):
    return file_db.indexes.register(TableIndex(file_db))


# --------------------------------------------------------------------------- placeholders


def test_placeholders_exist_report_absent_and_refuse_to_build(db):
    assert db.indexes.names == ("fts", "csr", "vector")
    assert all(isinstance(db.indexes[n], NullIndex) for n in db.indexes.names)
    health = db.index_health()
    assert {n: h.reason for n, h in health.items()} == dict.fromkeys(health, HealthReason.ABSENT)
    assert not any(h.usable for h in health.values())
    reports = db.maintain_indexes()
    assert {n: r.action for n, r in reports.items()} == dict.fromkeys(reports, "skipped")
    with pytest.raises(IndexGenerationError) as exc:
        db.indexes["fts"].build_next(1)
    assert exc.value.index == "fts"
    # a placeholder is not a definition: it is not written to the file and writes cost nothing
    assert db.indexes.definitions() == {}
    assert db.execute(f"SELECT count(*) FROM {S.INDEX_REGISTRY_TABLE}").fetchone()[0] == 0
    db.remember("x", now=T0)
    assert db.execute(f"SELECT count(*) FROM {S.INDEX_JOURNAL_TABLE}").fetchone()[0] == 0
    assert not db.indexes.wants("memories")


def test_registry_register_replaces_by_name_and_records_the_definition_in_the_file(db):
    a = db.indexes.register(TableIndex(db, name="fts"))
    assert db.indexes["fts"] is a and len(db.indexes) == 3
    assert db.indexes.wants("memories") and not db.indexes.wants("edges_relates")
    assert db.indexes.emit("insert", "edges_relates", 1, [1], at=T0) is None
    # the definition, not the object, is what the file records
    defn = db.indexes.definitions()["fts"]
    assert defn.source_table == "memories" and defn.kind == "test_table" and defn.enabled
    assert defn.per_tenant and defn.delta_mode == "table" and defn.supports_delta
    assert set(defn.as_dict()) >= {"index_name", "kind", "source_table", "enabled"}
    b = db.indexes.register(TableIndex(db, name="fts"))
    assert db.indexes["fts"] is b
    assert (
        db.execute(
            f"SELECT count(*) FROM {S.INDEX_REGISTRY_TABLE} WHERE index_name = 'fts'"
        ).fetchone()[0]
        == 1
    )
    # unregister disables the definition, so no handle journals for it any more
    assert db.indexes.unregister("fts") is b and "fts" not in db.indexes
    assert not db.indexes.wants("memories")
    assert not db.indexes.definitions()["fts"].enabled
    assert db.indexes.definitions(enabled_only=True) == {}
    # ... and forget=True removes the row and its journal outright
    db.indexes.register(TableIndex(db, name="fts"))
    db.remember("x", now=T0)
    assert journal(db, "fts")
    db.indexes.unregister("fts", forget=True)
    assert db.indexes.definitions() == {} and journal(db, "fts") == []
    with pytest.raises(TypeError):
        db.indexes.register(object())
    empty = IndexRegistry(db, placeholders=())
    assert len(empty) == 0


# --------------------------------------------------------------------------- lifecycle


def test_generation_lifecycle_build_beside_validate_publish_retire(file_db, idx):
    db = file_db
    ids = [db.remember(f"m{i}", now=T0 + i * MINUTE).memory_id for i in range(3)]
    assert idx.current_generation(1) is None
    assert idx.health(1).reason is HealthReason.ABSENT
    assert idx.pending(1) == sorted(ids)  # no generation: everything is pending

    g1 = idx.build_next(1)
    assert g1.generation == 1 and not g1.published and not g1.validated and not g1.building
    assert g1.watermark_id == max(ids) and g1.stats["rows"] == 3
    assert g1.stats["absorbed_journal_rows"] == 3
    assert idx.table(g1) in tables(db)
    assert idx.current_generation(1) is None  # built beside, not yet live
    assert idx.pending(1, g1) == []
    with pytest.raises(IndexGenerationError):
        idx.publish(g1)  # not validated
    report = idx.validate(g1)
    assert report.ok and report.generation.validated
    pub = idx.publish(report.generation)
    assert pub.published and idx.current_generation(1) == pub
    assert idx.health(1).reason is HealthReason.FRESH and idx.health(1).usable
    # the absorbed journal rows were pruned by the publish
    assert journal(db) == []

    later = db.remember("m3", now=T0 + 10 * MINUTE).memory_id
    assert idx.pending(1) == [later]
    g2 = idx.build_next(1)
    assert {idx.table(g1), idx.table(g2)} <= tables(db)  # beside each other
    assert idx.current_generation(1).generation == 1  # still the old one
    idx.publish(idx.validate(g2).generation)
    assert idx.current_generation(1).generation == 2
    assert idx.pending(1) == []
    # exactly one published row for this (index, tenant)
    assert (
        db.execute(
            f"SELECT count(*) FROM {S.INDEX_GENERATIONS_TABLE} WHERE index_name = 'tidx' AND published"
        ).fetchone()[0]
        == 1
    )
    # generation 1 is kept until the next build retires it (a reader may still pin it)
    assert idx.table(g1) in tables(db)
    g3 = idx.build_next(1)
    assert idx.table(g1) not in tables(db) and idx.table(g2) in tables(db)
    assert [g.generation for g in idx.generations(1)] == [2, 3]
    with pytest.raises(IndexGenerationError):
        idx.retire(idx.current_generation(1))
    assert idx.retire(g3) is True and idx.table(g3) not in tables(db)
    assert idx.retire(g3) is False


def test_publish_of_an_unknown_generation_is_refused(file_db, idx):
    file_db.remember("m", now=T0)
    g = idx.build_next(1)
    with pytest.raises(IndexGenerationError):
        idx.publish(g)  # not validated and not forced
    ghost = Generation(index_name="tidx", generation=99, tenant_id=1, built_at=T0)
    with pytest.raises(IndexGenerationError):
        idx.publish(ghost)


def test_publish_is_atomic_and_a_pinned_reader_keeps_its_generation(file_db, idx):
    """A read pinned to generation N is untouched by the publication of N + 1."""
    db = file_db
    first = [db.remember(f"m{i}", now=T0 + i * MINUTE).memory_id for i in range(3)]
    g1 = idx.publish(idx.validate(idx.build_next(1)).generation)
    published_in_thread: list[Generation] = []
    errors: list[BaseException] = []

    with idx.pin(1) as pin:
        assert pin and pin.generation == g1 and pin.reason is HealthReason.FRESH
        base_before = idx.base(g1, 1)

        def rebuild():
            try:
                other = Anatid.open(
                    db.path, tenant=1, embedding_dim=DIM, ensure=False, accelerators=False
                )
                try:
                    o_idx = other.indexes.register(TableIndex(other))
                    other.remember("late", now=T0 + 20 * MINUTE)
                    g2 = o_idx.build_next(1)
                    published_in_thread.append(o_idx.publish(o_idx.validate(g2).generation))
                finally:
                    other.close()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=rebuild)
        t.start()
        t.join()
        assert errors == []
        assert published_in_thread and published_in_thread[0].generation == 2
        # publication is visible to a NEW read ...
        assert idx.current_generation(1).generation == 2
        # ... but this read is still on generation 1: same object, same base, storage intact
        assert pin.generation == g1
        assert idx.base(g1, 1) == base_before == set(first)
        assert idx.table(g1) in tables(db)
        assert idx.pinned(g1) == 1
        assert idx.retire(g1) is False  # refused while pinned
        # and the merge on the pinned generation still sees the late write through the delta
        merged = (idx.base(g1, 1) | set(idx.pending(1, g1))) - set(idx.tombstones(1, g1))
        assert merged == sql_oracle(db, 1)
    assert idx.pinned(g1) == 0
    assert idx.retire(g1) is True


def test_pin_reports_why_it_declined(file_db, idx):
    db = file_db
    db.remember("m", now=T0)
    with idx.pin(1) as pin:
        assert not pin and pin.reason is HealthReason.ABSENT and pin.generation is None
    g = idx.publish(idx.validate(idx.build_next(1)).generation)
    with idx.pin(1, as_of=T0) as pin:
        assert not pin and pin.reason is HealthReason.HISTORICAL_QUERY
    with idx.pin(1, as_of=AsOf(valid_time=None, tx_time=T0)) as pin:
        assert pin.reason is HealthReason.HISTORICAL_QUERY
    db.indexes.invalidate("memories", reason="test")
    with idx.pin(1) as pin:
        assert not pin and pin.reason is HealthReason.STALE_GENERATION
        assert pin.generation is not None and pin.generation.generation == g.generation
    idx.load_error = "extension refused to load"
    with idx.pin(1) as pin:
        assert pin.reason is HealthReason.LOAD_FAILURE and pin.detail == idx.load_error


def test_an_index_that_cannot_merge_deltas_is_unusable_once_rows_are_pending(file_db):
    class Snapshot(TableIndex):
        name = "snap"
        supports_delta = False

    db = file_db
    snap = db.indexes.register(Snapshot(db))
    db.remember("m", now=T0)
    snap.publish(snap.validate(snap.build_next(1)).generation)
    assert snap.health(1).usable
    db.remember("n", now=T0 + MINUTE)
    h = snap.health(1)
    assert h.reason is HealthReason.STALE_GENERATION and not h.usable and h.pending_rows == 1
    with snap.pin(1) as pin:
        assert not pin and pin.reason is HealthReason.STALE_GENERATION


# --------------------------------------------------------------------------- delta / tombstones


def test_the_journal_is_written_in_the_verbs_transaction(file_db, idx):
    db = file_db
    a = db.remember("a", now=T0).memory_id
    idx.publish(idx.validate(idx.build_next(1)).generation)

    # a rolled-back write leaves no journal row
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.remember("rolled back", now=T0 + MINUTE)
            raise RuntimeError("boom")
    assert journal(db) == []
    b = db.remember("b", now=T0 + MINUTE).memory_id
    assert journal(db) == [(1, b, "insert", None)]
    assert idx.pending(1) == [b]
    # visible to the merge in the same transaction as the write
    with db.transaction():
        c = db.remember("c", now=T0 + 2 * MINUTE).memory_id
        assert c in idx.pending(1)
    # supersede: an insert for the new row, a close for the old one
    a2 = db.supersede(a, "a2", now=T0 + 3 * MINUTE).memory_id
    assert journal(db) == [
        (1, b, "insert", None),
        (1, c, "insert", None),
        (1, a2, "insert", None),
        (1, a, "close", None),
    ]
    # a soft forget closes; a HARD forget erases, so nothing about c is left in the journal
    db.forget(b, now=T0 + 4 * MINUTE)
    db.forget(c, hard=True, now=T0 + 5 * MINUTE)
    assert journal(db) == [
        (1, b, "insert", None),
        (1, a2, "insert", None),
        (1, a, "close", None),
        (1, b, "close", None),
    ]
    assert idx.tombstones(1) == sorted([a, b])
    assert idx.pending(1) == [a2]
    assert idx.candidates(1)[0] == sql_oracle(db, 1) == {a2}
    # the reasons are on the journal rows, for an operator reading the file
    reasons = {
        (int(r[0]), str(r[1])): r[2]
        for r in db.execute(f"SELECT doc_id, op, reason FROM {S.INDEX_JOURNAL_TABLE}").fetchall()
    }
    assert reasons[(a, "close")] == "supersede" and reasons[(b, "close")] == "forget"
    # prune goes through forget
    db.prune(older_than=T0 + 10 * MINUTE, dry_run=False, now=T0 + 6 * MINUTE)
    assert a2 in idx.tombstones(1)
    assert idx.candidates(1)[0] == sql_oracle(db, 1) == set()


def test_tombstones_are_absorbed_by_the_next_build_and_pruned_on_publish(file_db, idx):
    db = file_db
    a = db.remember("a", now=T0).memory_id
    b = db.remember("b", now=T0).memory_id
    g1 = idx.publish(idx.validate(idx.build_next(1)).generation)
    db.forget(a, now=T0 + MINUTE)
    assert idx.tombstones(1, g1) == [a]
    g2 = idx.build_next(1)
    # stamped, not deleted: a reader on g1 still needs it
    assert journal(db) == [(1, a, "close", 2)]
    assert idx.tombstones(1, g1) == [a]
    assert idx.tombstones(1, g2) == []
    assert g2.stats["absorbed_journal_rows"] == 1
    idx.publish(idx.validate(g2).generation)
    # still there: generation 1 is alive (a reader may be pinned to it) and reads it
    assert len(journal(db)) == 1
    assert idx.candidates(1)[0] == {b} == sql_oracle(db, 1)
    # pruned as soon as no alive generation needs it
    assert idx.retire(g1) is True
    assert journal(db) == []
    assert idx.candidates(1)[0] == {b} == sql_oracle(db, 1)


def test_watermark_delta_mode_records_only_out_of_order_ids(file_db):
    class WmIndex(TableIndex):
        name = "wm"
        delta_mode = "watermark"

    db = file_db
    wm = db.indexes.register(WmIndex(db))
    big = db.remember("big", memory_id=1_000_000, now=T0).memory_id
    assert journal(db, "wm") == []  # no generation yet: the source table is the delta
    g1 = wm.publish(wm.validate(wm.build_next(1)).generation)
    assert g1.watermark_id == big
    above = db.remember("above", memory_id=2_000_000, now=T0 + MINUTE).memory_id
    assert journal(db, "wm") == []
    below = db.remember("below", memory_id=5, now=T0 + 2 * MINUTE).memory_id
    assert journal(db, "wm") == [(1, below, "insert", None)]
    assert wm.pending(1) == sorted([below, above])
    assert wm.candidates(1)[0] == sql_oracle(db, 1)
    with pytest.raises(ValueError):
        type("Bad", (TableIndex,), {"delta_mode": "nope"})(db)


# --------------------------------------------------------------------------- health / policy


def test_health_reasons_cover_the_whole_enum(file_db, idx):
    db = file_db
    assert idx.health(1).reason is HealthReason.ABSENT
    db.remember("m", now=T0)
    assert idx.health(1, as_of=T0).reason is HealthReason.HISTORICAL_QUERY
    # rebuild in progress: a build announced in the catalog and owned by nobody here
    db.execute(
        f"INSERT INTO {S.INDEX_GENERATIONS_TABLE} (index_name, generation, tenant_id, built_at, "
        f"validated, published, stats, notes) VALUES ('tidx', 1, 1, ?, FALSE, FALSE, '{{}}', "
        f"'building')",
        [T0],
    )
    h = idx.health(1)
    assert h.reason is HealthReason.REBUILD_IN_PROGRESS and not h.usable
    assert maintain(idx, 1).action == "skipped"
    with pytest.raises(IndexGenerationError):
        idx.build_next(1)
    assert idx.abandon_builds(1) == 1
    g = idx.publish(idx.validate(idx.build_next(1)).generation)
    h = idx.health(1)
    assert h.reason is HealthReason.FRESH and h.usable and h.generation == g
    assert h.pending_rows == 0 and h.tombstone_rows == 0 and h.base_rows == 1
    db.remember("n", now=T0 + MINUTE)
    h = idx.health(1, policy=MaintenancePolicy(rebuild_after_rows=1))
    assert h.reason is HealthReason.STALE_GENERATION and h.usable  # merges deltas
    assert h.pending_rows == 1 and h.pending_ratio == 1.0
    idx.load_error = "cannot load"
    h = idx.health(1)
    assert h.reason is HealthReason.LOAD_FAILURE and not h.usable and h.detail == "cannot load"
    idx.load_error = None
    # A base that is present and queryable but no longer holds what the build recorded.  The
    # index answers that question itself (each one has a different cheap invariant); the
    # framework's job is to report it and to make a rebuild due.
    idx.base_damage = lambda _generation: "the base holds 0 row(s) against the 2 recorded"
    h = idx.health(1)
    assert h.reason is HealthReason.DAMAGED_BASE and not h.usable
    assert "0 row(s)" in h.detail
    assert "damaged base" in (MaintenancePolicy().due(h) or "")
    del idx.base_damage
    assert idx.health(1).reason is not HealthReason.DAMAGED_BASE
    assert {r.value for r in HealthReason} == {
        "fresh",
        "stale_generation",
        "unvalidated",
        "historical_query",
        "rebuild_in_progress",
        "load_failure",
        "damaged_base",
        "absent",
    }
    assert set(h.as_dict()) >= {"reason", "usable", "pending_rows", "generation"}


def test_maintenance_policy_triggers(file_db, idx):
    db = file_db
    policy = MaintenancePolicy(
        rebuild_after_rows=3, rebuild_after_ratio=None, rebuild_after_seconds=None
    )
    db.remember("m0", now=T0)
    r = maintain(idx, 1, policy)
    assert r.action == "published" and r.reason == "no generation published"
    assert r.generation.published and r.after.reason is HealthReason.FRESH
    db.remember("m1", now=T0 + MINUTE)
    db.remember("m2", now=T0 + 2 * MINUTE)
    r = maintain(idx, 1, policy)
    assert r.action == "none" and idx.current_generation(1).generation == 1
    db.remember("m3", now=T0 + 3 * MINUTE)
    r = maintain(idx, 1, policy)
    assert r.action == "published" and "pending row(s) >= rebuild_after_rows=3" in r.reason
    assert idx.current_generation(1).generation == 2 and r.after.pending_rows == 0
    # ratio: 4 base rows, 1 pending = 0.25
    ratio = MaintenancePolicy(
        rebuild_after_rows=None, rebuild_after_ratio=0.25, rebuild_after_seconds=None
    )
    db.remember("m4", now=T0 + 4 * MINUTE)
    assert maintain(idx, 1, ratio).action == "published"
    assert maintain(idx, 1, ratio).action == "none"
    # seconds: only with something pending
    age = MaintenancePolicy(
        rebuild_after_rows=None, rebuild_after_ratio=None, rebuild_after_seconds=60
    )
    built_at = idx.current_generation(1).built_at
    assert maintain(idx, 1, age, now=built_at + 2 * MINUTE).action == "none"
    db.remember("m5", now=T0 + 5 * MINUTE)
    assert maintain(idx, 1, age, now=built_at + 30 * _dt.timedelta(seconds=1)).action == "none"
    r = maintain(idx, 1, age, now=built_at + 2 * MINUTE)
    assert r.action == "published" and "rebuild_after_seconds" in r.reason
    # an invalidated generation is always due
    db.indexes.invalidate("memories", reason="bulk load")
    h = idx.health(1)
    assert h.reason is HealthReason.STALE_GENERATION and not h.usable
    assert "invalidated: bulk load" in h.detail
    r = maintain(
        idx,
        1,
        MaintenancePolicy(
            rebuild_after_rows=None, rebuild_after_ratio=None, rebuild_after_seconds=None
        ),
    )
    assert r.action == "published" and r.after.usable
    # policy.validate=False publishes without the oracle check, and says so
    db.remember("m6", now=T0 + 6 * MINUTE)
    r = maintain(idx, 1, MaintenancePolicy(rebuild_after_rows=1, validate=False))
    assert r.action == "published_unvalidated" and not r.generation.validated
    assert r.generation.published_unvalidated
    assert r.after.reason is HealthReason.UNVALIDATED and r.after.usable
    # the handle-level entry point runs every registered index
    out = db.maintain_indexes(policy=policy)
    assert set(out) == {"fts", "csr", "vector", "tidx"}
    assert out["fts"].action == "skipped"


def test_validation_failure_retires_the_candidate_and_keeps_the_old_generation(file_db):
    db = file_db
    idx = db.indexes.register(TableIndex(db, fail_validation_from=2))
    db.remember("m0", now=T0)
    policy = MaintenancePolicy(rebuild_after_rows=1)
    assert maintain(idx, 1, policy).action == "published"
    db.remember("m1", now=T0 + MINUTE)
    r = maintain(idx, 1, policy)
    assert r.action == "validation_failed" and r.validation is not None and not r.validation.ok
    assert idx.current_generation(1).generation == 1
    assert [g.generation for g in idx.generations(1)] == [1]  # candidate retired
    assert r.generation.generation == 2 and idx.table(r.generation) not in tables(db)
    with pytest.raises(IndexValidationError) as exc:
        maintain(idx, 1, policy, raise_on_failure=True)
    assert exc.value.report is not None and exc.value.generation == 3
    # reads keep merging on generation 1 and stay right
    assert idx.candidates(1)[0] == sql_oracle(db, 1)


def test_a_failed_build_leaves_no_generation_behind(file_db):
    class Broken(TableIndex):
        name = "broken"

        def _build(self, gen):
            self.db.execute(f"CREATE TABLE {self.table(gen)} (x INTEGER)")
            raise RuntimeError("disk full")

    db = file_db
    broken = db.indexes.register(Broken(db))
    db.remember("m", now=T0)
    with pytest.raises(RuntimeError):
        broken.build_next(1)
    assert broken.generations(1) == []
    assert not any(t.startswith("anatid_idx_broken") for t in tables(db))
    assert not broken.is_building(1)
    assert broken.health(1).reason is HealthReason.ABSENT
    # the journal row was not absorbed by the failed build
    assert [r[3] for r in journal(db, "broken")] == [None]


def test_bulk_load_invalidates_published_generations(file_db, tmp_path, idx):
    db = file_db
    db.remember("m0", now=T0)
    idx.publish(idx.validate(idx.build_next(1)).generation)
    con = duckdb.connect()
    con.execute(
        f"CREATE TABLE memories (memory_id BIGINT, tenant_id INTEGER, content VARCHAR, "
        f"kind VARCHAR, embedding FLOAT[{DIM}], created_at TIMESTAMP, valid_from TIMESTAMP, "
        f"tx_from TIMESTAMP)"
    )
    con.execute(
        "INSERT INTO memories VALUES (777, 1, 'loaded', 'fact', NULL, ?, ?, ?)", [T0, T0, T0]
    )
    con.execute(f"COPY memories TO '{tmp_path / 'memories.parquet'}' (FORMAT PARQUET)")
    con.close()
    assert db.load_parquet(tmp_path, rebuild_fts=False)["memories"] == 1
    h = idx.health(1)
    assert h.reason is HealthReason.STALE_GENERATION and not h.usable
    assert "bulk load of 1 row(s) into memories" in h.detail
    assert idx.candidates(1) == (None, "stale_generation")
    assert maintain(idx, 1).action == "published"
    assert idx.candidates(1)[0] == sql_oracle(db, 1) and 777 in sql_oracle(db, 1)


def test_an_invalidated_generation_cannot_be_validated_back_into_service(file_db, idx):
    """The oracle can AGREE with an invalidated base, so the refusal is a check on the note.

    Whatever bypassed the journal is uncommitted or invisible from here (a bulk load in another
    transaction, a hard erasure that has not committed), which is exactly why the base and the
    canonical rows can match while the base is wrong.  Only a new generation clears it.
    """
    db = file_db
    db.remember("m", now=T0)
    gen = idx.publish(idx.validate(idx.build_next(1)).generation)
    assert idx.validate(gen).ok  # it does compare equal, before and after

    db.indexes.invalidate("memories", reason="bulk load")
    report = idx.validate(idx.current_generation(1))
    assert not report.ok
    assert "cannot be revalidated" in report.detail and "bulk load" in report.detail
    assert not idx.current_generation(1).validated
    assert idx.health(1).reason is HealthReason.STALE_GENERATION and not idx.health(1).usable
    # and a new generation is not tarred with it
    assert maintain(idx, 1).action == "published"
    assert idx.health(1).reason is HealthReason.FRESH


def test_invalidation_reaches_the_generations_that_are_not_published_yet(file_db, idx):
    """A bulk load is invisible to every generation, not only to the live one: an unpublished
    generation that kept ``validated`` could be published afterwards and read as complete."""
    db = file_db
    db.remember("m0", now=T0)
    live = idx.publish(idx.validate(idx.build_next(1)).generation)
    db.remember("m1", now=T0 + MINUTE)
    spare = idx.validate(idx.build_next(1)).generation  # built, checked, not published
    assert spare.validated and not spare.published

    assert db.indexes.invalidate("memories", reason="bulk load") == 2
    assert not idx.generation(1, live.generation).validated
    after = idx.generation(1, spare.generation)
    assert not after.validated and "bulk load" in after.notes
    with pytest.raises(IndexGenerationError):
        idx.publish(after, force=True)


# ------------------------------------------------------- the definition lives in the file


def test_a_second_handle_journals_for_an_index_it_holds_no_code_for(file_db):
    """P0-1.  Registries used to be per handle, so a second handle wrote nothing to the delta
    and the first handle's next read silently missed every row it had written."""
    db = file_db
    idx = db.indexes.register(TableIndex(db))
    a = db.remember("a", now=T0).memory_id
    idx.publish(idx.validate(idx.build_next(1)).generation)

    other = Anatid.open(db.path, tenant=1, embedding_dim=DIM, ensure=False, accelerators=False)
    try:
        # this handle has only the placeholders; it has never heard of TableIndex
        assert "tidx" not in other.indexes.names
        assert all(isinstance(other.indexes[n], NullIndex) for n in other.indexes.names)
        # but the FILE says the index exists, so this handle journals for it
        defn = other.indexes.definitions()["tidx"]
        assert defn.source_table == "memories" and defn.enabled
        assert other.indexes.wants("memories")
        b = other.remember("b", now=T0 + MINUTE).memory_id
        c = other.supersede(b, "c", now=T0 + 2 * MINUTE).memory_id
        other.forget(a, now=T0 + 3 * MINUTE)
    finally:
        other.close()

    assert journal(db) == [
        (1, b, "insert", None),
        (1, c, "insert", None),
        (1, b, "close", None),
        (1, a, "close", None),
    ]
    assert idx.pending(1) == [c] and idx.tombstones(1) == sorted([a, b])
    h = idx.health(1)
    assert h.pending_rows == 1 and h.tombstone_rows == 2
    got, reason = idx.candidates(1)
    assert got == sql_oracle(db, 1) == {c}, reason
    # and the definition survives the handle that made it
    reopened = Anatid.open(db.path, tenant=1, embedding_dim=DIM, ensure=False, accelerators=False)
    try:
        assert set(reopened.indexes.definitions()) == {"tidx"}
    finally:
        reopened.close()


def test_an_index_can_be_defined_without_an_implementation(file_db):
    """The declarative half: a process that only keeps the journal current for someone else."""
    db = file_db
    defn = db.indexes.define(
        IndexDefinition(
            index_name="remote",
            kind="elsewhere",
            source_table="memories",
            source_id_column="memory_id",
            params={"backend": "hnsw"},
        )
    )
    assert defn.index_name == "remote"
    assert db.indexes.definitions()["remote"].params == {"backend": "hnsw"}
    m = db.remember("m", now=T0).memory_id
    assert journal(db, "remote") == [(1, m, "insert", None)]
    db.forget(m, now=T0 + MINUTE)
    assert journal(db, "remote")[-1] == (1, m, "close", None)


def test_a_handle_without_the_code_still_reports_the_index_health(file_db):
    """An operator on any handle can see the state of every index the file defines."""
    db = file_db
    idx = db.indexes.register(TableIndex(db))
    db.remember("a", now=T0)
    idx.publish(idx.validate(idx.build_next(1)).generation)
    other = Anatid.open(db.path, tenant=1, embedding_dim=DIM, ensure=False, accelerators=False)
    try:
        other.remember("b", now=T0 + MINUTE)
        lenient = MaintenancePolicy(
            rebuild_after_rows=None, rebuild_after_ratio=None, rebuild_after_seconds=None
        )
        health = other.index_health(policy=lenient)
        assert set(health) == {"fts", "csr", "vector", "tidx"}
        assert health["tidx"].reason is HealthReason.FRESH and health["tidx"].usable
        assert health["tidx"].pending_rows == 1 and health["tidx"].base_rows == 1
        assert health["fts"].reason is HealthReason.ABSENT
        # the default policy sees the same numbers and calls it due, still usable
        due = other.index_health()["tidx"]
        assert due.reason is HealthReason.STALE_GENERATION and due.usable
        journal_only = other.indexes.journal_only()["tidx"]
        assert journal_only.definition().kind == "test_table"
        # it can journal and report, it cannot build
        with pytest.raises(IndexGenerationError):
            journal_only.build_next(1)
        assert not journal_only.validate(journal_only.current_generation(1)).ok
    finally:
        other.close()


# --------------------------------------------------------------------------- ordered journal


def test_the_newest_journal_op_for_a_document_wins(file_db, idx):
    """P0-2, at the journal level: one ordered log, not a delta set and a tombstone set."""
    db = file_db
    idx.record_delta(1, [42], at=T0)
    assert idx.pending(1) == [42] and idx.tombstones(1) == []
    idx.tombstone(1, [42], at=T0 + MINUTE, reason="close")
    assert idx.pending(1) == [] and idx.tombstones(1) == [42]
    idx.record_delta(1, [42], at=T0 + 2 * MINUTE)
    assert idx.pending(1) == [42] and idx.tombstones(1) == []
    seqs = [
        int(r[0])
        for r in db.execute(
            f"SELECT change_seq FROM {S.INDEX_JOURNAL_TABLE} ORDER BY change_seq"
        ).fetchall()
    ]
    assert len(seqs) == 3 and seqs == sorted(set(seqs))
    with pytest.raises(ValueError):
        derived_mod.journal_append(db, "tidx", 1, [1], "nonsense", at=T0)


def test_a_document_id_that_is_purged_and_then_reused(file_db, idx):
    """P0-2, the reviewer's sequence: insert 5, build, hard purge 5, insert a new 5.

    With independent delta and tombstone sets, 5 was in both and ``(base | delta) - tombstones``
    returned nothing while the SQL path returned the row.
    """
    db = file_db
    db.remember("first five", memory_id=5, now=T0)
    g1 = idx.publish(idx.validate(idx.build_next(1)).generation)
    assert idx.base(g1, 1) == {5}
    db.forget(5, hard=True, now=T0 + MINUTE)
    reused = db.remember("second five", memory_id=5, now=T0 + 2 * MINUTE)
    assert reused.memory_id == 5 and reused.content == "second five"

    assert journal(db) == [(1, 5, "insert", None)]  # the purge took the old rows with it
    assert idx.pending(1) == [5] and idx.tombstones(1) == []
    got, reason = idx.candidates(1)
    assert got == sql_oracle(db, 1) == {5}, reason
    # the same holds with the removal still in the journal, which is what a generation whose
    # storage could not delete the old document would look like
    idx.tombstone(1, [5], at=T0 + 3 * MINUTE, reason="forget_hard")
    idx.record_delta(1, [5], at=T0 + 4 * MINUTE)
    assert idx.pending(1) == [5] and idx.tombstones(1) == []
    assert idx.candidates(1)[0] == sql_oracle(db, 1) == {5}


# --------------------------------------------------------------------------- tenant keying


def test_one_tenants_change_never_touches_another_tenants_document(file_db):
    """P0-3.  A file-wide index used to scope its delta and tombstones by index name only, so
    tombstoning tenant 1's document 42 removed tenant 2's candidate 42."""
    db = file_db
    wide = db.indexes.register(FileWide(db))
    a = db.remember("tenant one", tenant=1, memory_id=42, now=T0).memory_id
    b = db.remember("tenant two", tenant=2, memory_id=42, now=T0).memory_id
    assert a == b == 42
    g = wide.publish(wide.validate(wide.build_next()).generation)
    assert wide.base_keys(g) == {(1, 42), (2, 42)}

    db.forget(42, tenant=1, now=T0 + MINUTE)
    assert wide.tombstone_keys() == [(1, 42)]
    assert wide.tombstones(1) == [42] and wide.tombstones(2) == []
    assert wide.candidates(2)[0] == sql_oracle(db, 2) == {42}
    assert wide.candidates(1)[0] == sql_oracle(db, 1) == set()

    # and the other way round: tenant 2 writing a new document 7 is not tenant 1's pending row
    db.remember("tenant two again", tenant=2, memory_id=7, now=T0 + 2 * MINUTE)
    assert wide.pending(2) == [7] and wide.pending(1) == []
    assert wide.pending_keys() == [(2, 7)]
    for tenant in (1, 2):
        assert wide.candidates(tenant)[0] == sql_oracle(db, tenant)

    # a per-tenant index keys its journal the same way, from the same rows
    db.indexes.register(TableIndex(db))
    db.remember("t1", tenant=1, memory_id=99, now=T0 + 3 * MINUTE)
    db.remember("t2", tenant=2, memory_id=99, now=T0 + 3 * MINUTE)
    assert journal(db)[-2:] == [(1, 99, "insert", None), (2, 99, "insert", None)]


# --------------------------------------------------------------------------- erasure


def test_a_hard_forget_erases_the_document_from_every_generation_and_the_journal(file_db, idx):
    """P0-4.  A purge that stopped at the canonical tables left the id, and an accelerator that
    stores content would have left the text, inside every generation it had already built."""
    db = file_db
    secret = "erase-me-4d1f9c22"
    mid = 987654321
    db.remember(secret, memory_id=mid, entities=["Ada"], episode="user said: " + secret, now=T0)
    db.remember("kept", now=T0 + MINUTE)
    g1 = idx.publish(idx.validate(idx.build_next(1)).generation)
    db.remember("later", now=T0 + 2 * MINUTE)
    g2 = idx.build_next(1)  # a second, unpublished generation that also holds the document
    db.rebuild_fts_index(now=T0 + 3 * MINUTE)
    assert (1, mid) in idx.base_keys(g1) and (1, mid) in idx.base_keys(g2)
    # the scan is worth something: before the purge it finds the id and the text in several
    # places, the generation storage among them
    before = find_in_file(db, str(mid), secret)
    assert {table for table, _c, _n in before} >= {
        "main.memories",
        "main.anatid_fts_documents",
        f"main.{idx.table(g1)}",
        f"main.{idx.table(g2)}",
    }

    receipt = db.forget(mid, hard=True, now=T0 + 4 * MINUTE)
    assert receipt.derived_rows_deleted >= 2  # one row per generation storage, plus the journal
    assert receipt.invalidated_generations == 0
    assert receipt.rows_removed >= receipt.derived_rows_deleted
    assert (1, mid) not in idx.base_keys(g1) and (1, mid) not in idx.base_keys(g2)
    assert all(doc != mid for _t, doc, _op, _a in journal(db))

    # the whole file, enumerated from the catalog: neither the id nor the text is anywhere
    assert find_in_file(db, str(mid), secret) == []
    # the generation is still trusted and still right
    assert idx.health(1).usable
    assert idx.candidates(1)[0] == sql_oracle(db, 1)
    assert idx.validate(idx.current_generation(1)).ok


def test_purging_the_newest_memory_takes_its_id_out_of_the_generation_watermark(file_db, idx):
    """P0-4, the last place the id hid.  A generation's ``watermark_id`` is the largest source
    id in its snapshot, so purging the newest memory left that id in
    ``anatid_index_generations`` -- the same leak ``anatid_meta.fts_indexed_max_id`` had, in the
    table this framework added.  Only found by scanning the whole file, which is why the test
    for it does.
    """
    db = file_db
    older = db.remember("older", now=T0).memory_id
    newest = db.remember("newest", now=T0 + MINUTE).memory_id
    other = db.remember("another tenant", tenant=2, now=T0 + MINUTE).memory_id
    g = idx.publish(idx.validate(idx.build_next(1)).generation)
    assert g.watermark_id == newest > older
    assert (f"main.{S.INDEX_GENERATIONS_TABLE}", "watermark_id", str(newest)) in find_in_file(
        db, str(newest)
    )

    db.forget(newest, hard=True, now=T0 + 2 * MINUTE)
    assert find_in_file(db, str(newest)) == []
    # clamped to the largest surviving id of this tenant, so nothing stops being pending
    after = idx.generation(1, g.generation)
    assert after.watermark_id == older
    assert after.watermark_ts is not None
    assert idx.candidates(1)[0] == sql_oracle(db, 1) == {older}
    assert idx.validate(after).ok

    # and with the whole tenant gone the watermark goes to NULL rather than to another
    # tenant's id
    db.forget(older, hard=True, now=T0 + 3 * MINUTE)
    empty = idx.generation(1, g.generation)
    assert empty.watermark_id is None and empty.watermark_ts is None
    assert find_in_file(db, str(older)) == []
    assert db.get(other, tenant=2) is not None  # the other tenant is untouched


def test_an_index_whose_storage_cannot_delete_has_its_generations_invalidated(file_db):
    """P0-4, the other branch: an index that only rebuilds is taken out of service instead of
    keeping the erased document."""

    class NoErase(TableIndex):
        name = "noerase"

        def _erase(self, gen, tenant_id, doc_ids):
            return None  # this storage rebuilds, it does not delete

    db = file_db
    idx = db.indexes.register(NoErase(db))
    m = db.remember("secret", now=T0).memory_id
    db.remember("kept", now=T0 + MINUTE)
    idx.publish(idx.validate(idx.build_next(1)).generation)
    later = db.remember("also secret", now=T0 + 2 * MINUTE).memory_id
    assert journal(db, "noerase") == [(1, later, "insert", None)]

    receipt = db.forget(m, hard=True, now=T0 + 3 * MINUTE)
    assert receipt.invalidated_generations == 1
    h = idx.health(1)
    assert h.reason is HealthReason.STALE_GENERATION and not h.usable
    assert "hard erasure" in h.detail
    assert idx.candidates(1) == (None, "stale_generation")
    # the journal rows go even where the storage cannot be cleaned
    db.forget(later, hard=True, now=T0 + 4 * MINUTE)
    assert journal(db, "noerase") == []
    # and it is force-publish-proof: the base is missing rows the journal cannot supply
    with pytest.raises(IndexGenerationError):
        idx.publish(idx.current_generation(1), force=True)
    # a rebuild puts it back in service
    assert maintain(idx, 1).action == "published"
    assert idx.health(1).usable and idx.candidates(1)[0] == sql_oracle(db, 1)


def test_a_hard_forget_from_a_handle_without_the_code_invalidates_rather_than_retains(file_db):
    """A handle that cannot reach an index's storage must not pretend the erasure was complete."""
    db = file_db
    idx = db.indexes.register(TableIndex(db))
    db.remember("kept", now=T0)
    idx.publish(idx.validate(idx.build_next(1)).generation)
    m = db.remember("secret", now=T0 + MINUTE).memory_id
    assert journal(db) == [(1, m, "insert", None)]

    other = Anatid.open(db.path, tenant=1, embedding_dim=DIM, ensure=False, accelerators=False)
    try:
        receipt = other.forget(m, hard=True, now=T0 + 2 * MINUTE)
    finally:
        other.close()
    assert receipt.invalidated_generations == 1
    assert receipt.derived_rows_deleted == 1  # the journal row, which it can reach
    assert journal(db) == []
    h = idx.health(1)
    assert h.reason is HealthReason.STALE_GENERATION and not h.usable
    assert "no implementation" in h.detail
    assert idx.candidates(1) == (None, "stale_generation")


def test_a_build_that_finishes_before_a_purge_commits_does_not_go_live(file_db):
    """The erasure hand-off must not depend on the purge's SNAPSHOT.

    A purge that starts while a generation is being built calls that generation ``building`` for
    as long as its transaction lasts, so it skips its storage; the erasure counter is the only
    thing that reaches such a build.  Bumped when the purge got as far as the index, the build
    could finish first, check an unchanged counter, compare itself with canonical rows the purge
    had not deleted yet, pass, and publish a base holding the erased document.  So the counter
    is bumped before the purge opens its transaction, and :meth:`DerivedIndex.validate` refuses
    a generation carrying an invalidation note however well it compares.
    """
    db = file_db

    class Gated(TableIndex):
        name = "gated"

        def __init__(self, db):
            super().__init__(db)
            self.started = threading.Event()
            self.gate = threading.Event()
            self.arm = False

        def _build(self, gen):
            stats = super()._build(gen)
            if self.arm:
                self.arm = False
                self.started.set()
                assert self.gate.wait(10)
            return stats

    idx = db.indexes.register(Gated(db))
    secret = db.remember("secret", now=T0).memory_id
    idx.publish(idx.validate(idx.build_next(1)).generation)
    assert journal(db, "gated") == []  # pruned, so the purge and the build touch no same row
    db.remember("kept", now=T0 + MINUTE)

    purge_open, build_done = threading.Event(), threading.Event()
    original = IndexRegistry.erase

    def slow_erase(self, *args, **kwargs):
        purge_open.set()
        assert build_done.wait(10)
        return original(self, *args, **kwargs)

    outcome: list[bool] = []
    errors: list[BaseException] = []

    def build():
        try:
            gen = idx.build_next(1)
            report = idx.validate(gen)
            outcome.append(report.ok)
            if report.ok:
                idx.publish(gen)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def purge():
        try:
            db.forget(secret, hard=True, now=T0 + 2 * MINUTE)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    IndexRegistry.erase = slow_erase
    b = threading.Thread(target=build)
    p = threading.Thread(target=purge)
    try:
        idx.arm = True
        b.start()
        assert idx.started.wait(10)  # generation 2 announced; its snapshot holds the secret
        p.start()
        assert purge_open.wait(10)  # the purge has announced and opened its transaction
        idx.gate.set()  # the build finishes while the purge is still uncommitted
        b.join(10)
        build_done.set()
        p.join(10)
    finally:
        IndexRegistry.erase = original
        idx.gate.set()
        build_done.set()
        b.join(10)
        p.join(10)

    assert errors == [], errors[:2]
    assert outcome == [False]  # the build learned a purge had started and was refused
    stale = idx.generation(1, 2)
    assert stale is not None and not stale.validated
    assert "hard erasure" in stale.notes
    assert idx.current_generation(1).generation == 1  # generation 1 is still the live one

    # the purge committed, and the only copy of the secret left is in the generation that must
    # never go live; the next rebuild retires it and takes it with it
    assert (
        db.execute("SELECT count(*) FROM memories WHERE memory_id = ?", [secret]).fetchone()[0] == 0
    )
    assert find_in_file(db, str(secret)) == [(f"main.{idx.table(stale)}", "memory_id", str(secret))]
    assert maintain(idx, 1).action == "published"
    assert find_in_file(db, str(secret)) == []
    assert idx.candidates(1)[0] == sql_oracle(db, 1)


def test_a_generation_built_across_a_hard_erasure_does_not_go_live_validated(file_db):
    """A build's snapshot cannot see a purge that commits after it opened, so the generation it
    produces may still hold the erased document.  It is marked invalidated instead."""

    class Slow(TableIndex):
        name = "slow"

        def __init__(self, db, gate):
            super().__init__(db)
            self.gate = gate

        def _build(self, gen):
            stats = super()._build(gen)
            self.gate.wait()  # the purge runs now, in another thread and another transaction
            time.sleep(0.05)
            return stats

    db = file_db
    gate = threading.Barrier(2)
    idx = db.indexes.register(Slow(db, gate))
    m = db.remember("secret", now=T0).memory_id
    db.remember("kept", now=T0 + MINUTE)
    built: list[Generation] = []
    errors: list[BaseException] = []

    def build():
        try:
            built.append(idx.build_next(1))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=build)
    t.start()
    gate.wait()
    db.forget(m, hard=True, now=T0 + 2 * MINUTE)
    t.join()
    assert errors == [] and built
    gen = built[0]
    assert not gen.validated and gen.notes is not None
    assert "hard erasure" in gen.notes
    with pytest.raises(IndexGenerationError):
        idx.publish(gen, force=True)


# --------------------------------------------------------------------------- lifecycle locking


def test_a_generation_cannot_be_retired_between_choosing_it_and_pinning_it(file_db):
    """P1-6.  Without one lock per file there is a window between a reader selecting the
    published generation and registering its pin, and a retire in that window drops the storage
    the reader is about to read."""
    db = file_db

    class Racy(TableIndex):
        name = "racy"

        def __init__(self, db, gate):
            super().__init__(db)
            self.gate = gate
            self.arm = False

        def _cached_current(self, key):
            gen = super()._cached_current(key)
            if self.arm:
                self.arm = False
                self.gate.wait()  # release the other thread, still holding the lifecycle lock
                time.sleep(0.05)  # it would drop this generation's storage inside this window
            return gen

    gate = threading.Barrier(2)
    idx = db.indexes.register(Racy(db, gate))
    db.remember("m", now=T0)
    g1 = idx.publish(idx.validate(idx.build_next(1)).generation)
    db.remember("n", now=T0 + MINUTE)
    g2 = idx.validate(idx.build_next(1)).generation
    outcome: list[object] = []
    errors: list[BaseException] = []

    def publish_and_retire():
        try:
            gate.wait()
            idx.publish(g2)
            outcome.append(idx.retire(g1))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=publish_and_retire)
    t.start()
    idx.arm = True
    with idx.pin(1) as pin:
        assert pin and pin.generation.generation == g1.generation
        t.join()
        assert errors == []
        assert idx.pinned(pin.generation) == 1
        assert outcome == [False]  # the retire lost the race and said so
        assert idx.table(g1) in tables(db)
        assert idx.base(g1, 1)  # the storage is still there and still readable
    assert idx.pinned(g1) == 0
    assert idx.retire(g1) is True  # and now it goes


def test_abandon_builds_leaves_a_build_this_process_owns_alone(file_db):
    """P1-6.  Build ownership is process-wide, so a second handle cannot mistake a live build
    for the leftovers of a dead process and delete its catalog row."""
    db = file_db

    class Slow(TableIndex):
        name = "slow"

        def __init__(self, db, started, release):
            super().__init__(db)
            self.started = started
            self.release = release

        def _build(self, gen):
            stats = super()._build(gen)
            self.started.set()
            self.release.wait(5)
            return stats

    started, release = threading.Event(), threading.Event()
    idx = db.indexes.register(Slow(db, started, release))
    db.remember("m", now=T0)
    built: list[Generation] = []
    errors: list[BaseException] = []

    def build():
        try:
            built.append(idx.build_next(1))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=build)
    t.start()
    assert started.wait(5)
    other = Anatid.open(db.path, tenant=1, embedding_dim=DIM, ensure=False, accelerators=False)
    try:
        o_idx = other.indexes.register(TableIndex(other, name="slow"))
        assert o_idx.is_building(1)
        assert o_idx.abandon_builds(1) == 0  # this process owns it
        with pytest.raises(IndexGenerationError):
            o_idx.build_next(1)
    finally:
        release.set()
        other.close()
    t.join()
    assert errors == [] and built and built[0].generation == 1
    assert not idx.is_building(1)
    assert derived_mod.lifecycle_lock(derived_mod.database_key(db)) is derived_mod.lifecycle_lock(
        derived_mod.database_key(db)
    )


# --------------------------------------------------------------------------- bulk load


def test_load_parquet_rolls_the_insert_back_when_the_invalidation_fails(file_db, tmp_path, idx):
    """P1-7.  The insert and the invalidation are one transaction: there is no window in which
    the rows are committed and a generation that cannot contain them is still trusted."""
    db = file_db
    db.remember("m0", now=T0)
    idx.publish(idx.validate(idx.build_next(1)).generation)
    con = duckdb.connect()
    con.execute(
        f"CREATE TABLE memories (memory_id BIGINT, tenant_id INTEGER, content VARCHAR, "
        f"kind VARCHAR, embedding FLOAT[{DIM}], created_at TIMESTAMP, valid_from TIMESTAMP, "
        f"tx_from TIMESTAMP)"
    )
    con.execute(
        "INSERT INTO memories VALUES (777, 1, 'loaded', 'fact', NULL, ?, ?, ?)", [T0, T0, T0]
    )
    con.execute(f"COPY memories TO '{tmp_path / 'memories.parquet'}' (FORMAT PARQUET)")
    con.close()

    def boom(self, table, *, reason):
        raise RuntimeError("catalog write refused")

    original = IndexRegistry.invalidate
    IndexRegistry.invalidate = boom
    try:
        with pytest.raises(RuntimeError):
            db.load_parquet(tmp_path, rebuild_fts=False)
    finally:
        IndexRegistry.invalidate = original

    assert db.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
    assert 777 not in sql_oracle(db, 1)
    gen = idx.current_generation(1)
    assert gen is not None and gen.validated and idx.health(1).usable
    # the same load, with the invalidation working, commits both halves
    assert db.load_parquet(tmp_path, rebuild_fts=False)["memories"] == 1
    assert 777 in sql_oracle(db, 1)
    assert not idx.current_generation(1).validated


# --------------------------------------------------------------------------- force publication


def test_a_force_published_generation_is_usable_and_says_it_was_never_validated(file_db, idx):
    """P2-9.  publish(force=True) and MaintenancePolicy(validate=False) used to produce
    generations pin() always rejected, so nothing could read them and maintain rebuilt forever.

    Kept rather than removed, with an explicit state: candidate generation only narrows and the
    canonical rows decide the answer, so skipping an O(corpus) oracle check is the operator's
    call to make; what was wrong was publishing a generation and then refusing to read it.
    """
    db = file_db
    db.remember("m", now=T0)
    gen = idx.publish(idx.build_next(1), force=True)
    assert gen.published and not gen.validated and gen.published_unvalidated
    assert gen.usable_without_validation and gen.as_dict()["published_unvalidated"]

    h = idx.health(1)
    assert h.reason is HealthReason.UNVALIDATED and h.usable
    assert "never compared with the oracle" in h.detail
    with idx.pin(1) as pin:
        assert pin and pin.reason is HealthReason.UNVALIDATED
    assert idx.candidates(1) == (sql_oracle(db, 1), "unvalidated")

    # and it does not loop: the same policy that published it does not find it due
    lenient = MaintenancePolicy(
        rebuild_after_rows=None,
        rebuild_after_ratio=None,
        rebuild_after_seconds=None,
        validate=False,
    )
    assert maintain(idx, 1, lenient).action == "none"
    assert idx.current_generation(1).generation == gen.generation

    # validating it later clears the flag
    assert idx.validate(idx.current_generation(1)).ok
    after = idx.current_generation(1)
    assert after.validated and not after.published_unvalidated
    assert idx.health(1).reason is HealthReason.FRESH

    # an INVALIDATED generation is a different state and force cannot publish it
    db.indexes.invalidate("memories", reason="bulk load")
    invalid = idx.current_generation(1)
    assert not invalid.validated and not invalid.published_unvalidated
    assert idx.health(1).reason is HealthReason.STALE_GENERATION and not idx.health(1).usable
    with pytest.raises(IndexGenerationError) as exc:
        idx.publish(invalid, force=True)
    assert "not in the journal either" in str(exc.value)


# --------------------------------------------------------------------------- enabling


def test_unregistering_an_index_takes_its_generations_out_of_service(file_db):
    """``enabled`` is a journal switch, and a generation that keeps ``validated`` across the gap
    is read as complete when it is not.

    The P0-1 failure (a write that reaches no delta) reached through the registry rather than
    through a second handle: unregister, write, register again, and the published base was
    still marked validated and still missing the row the SQL path returns.
    """
    db = file_db
    idx = db.indexes.register(TableIndex(db))
    a = db.remember("a", now=T0).memory_id
    idx.publish(idx.validate(idx.build_next(1)).generation)
    assert idx.health(1).usable

    assert db.indexes.unregister("tidx") is idx
    assert db.indexes.definitions()["tidx"].enabled is False
    assert not db.indexes.wants("memories")
    b = db.remember("b", now=T0 + MINUTE).memory_id  # journalled nowhere
    assert journal(db) == []
    gen = idx.current_generation(1)
    assert gen.published and not gen.validated and "unregistered" in gen.notes

    again = db.indexes.register(TableIndex(db))
    assert db.indexes.definitions()["tidx"].enabled is True
    h = again.health(1)
    assert h.reason is HealthReason.STALE_GENERATION and not h.usable
    assert "journalling for this index was switched" in h.detail
    assert again.candidates(1) == (None, "stale_generation")  # the SQL path, not a wrong answer

    c = db.remember("c", now=T0 + 2 * MINUTE).memory_id  # journalled again
    assert journal(db) == [(1, c, "insert", None)]
    assert maintain(again, 1).action == "published"
    assert again.candidates(1)[0] == sql_oracle(db, 1) == {a, b, c}


def test_a_hard_forget_reaches_an_index_whose_definition_is_disabled(file_db):
    """Erasure follows the storage, not the journal: a disabled definition still has
    generations, and their base still holds whatever the index put in it."""
    db = file_db
    idx = db.indexes.register(ContentIndex(db))
    secret = "erase-me-95c1a730"
    mid = db.remember(secret, now=T0).memory_id
    db.remember("kept", now=T0 + MINUTE)
    g = idx.publish(idx.validate(idx.build_next(1)).generation)
    assert (f"main.{idx.table(g)}", "content", secret) in find_in_file(db, secret)

    # disabled in the file, while this handle still holds the implementation
    db.indexes.define(replace(idx.definition(), enabled=False))
    assert db.indexes.definitions()["tidx_content"].enabled is False
    assert not db.indexes.wants("memories")

    receipt = db.forget(mid, hard=True, now=T0 + 2 * MINUTE)
    assert receipt.derived_rows_deleted == 1  # the row in the disabled index's base
    assert receipt.invalidated_generations == 0
    assert find_in_file(db, secret, str(mid)) == []


def test_a_purge_that_cannot_clean_a_generation_says_so_every_time(file_db):
    """The count on the receipt is what the purge failed to clean, not what it changed: a
    generation that was already out of service still holds the document."""

    class NoErase(TableIndex):
        name = "noerase2"

        def _erase(self, gen, tenant_id, doc_ids):
            return None

    db = file_db
    idx = db.indexes.register(NoErase(db))
    first = db.remember("one", now=T0).memory_id
    second = db.remember("two", now=T0 + MINUTE).memory_id
    idx.publish(idx.validate(idx.build_next(1)).generation)
    assert db.forget(first, hard=True, now=T0 + 2 * MINUTE).invalidated_generations == 1
    # the generation is already out of service, and it still holds the second document
    assert db.forget(second, hard=True, now=T0 + 3 * MINUTE).invalidated_generations == 1


def test_unregister_with_retire_leaves_no_generation_storage_behind(file_db):
    """Without it there is no supported way to reclaim a PUBLISHED generation's storage, so an
    unregistered content index left a copy of everything it had indexed in the file, out of
    reach of forget(hard=True)."""
    db = file_db
    idx = db.indexes.register(ContentIndex(db))
    secret = "erase-me-6b0f42ae"
    db.remember(secret, now=T0)
    g = idx.publish(idx.validate(idx.build_next(1)).generation)
    assert idx.table(g) in tables(db)
    with pytest.raises(IndexGenerationError):
        idx.retire(g)  # published: retire on its own cannot reclaim it

    assert db.indexes.unregister("tidx_content", retire=True) is idx
    assert idx.table(g) not in tables(db)
    assert idx.all_generations() == [] and journal(db, "tidx_content") == []
    assert find_in_file(db, secret) == [("main.memories", "content", secret)]

    other = Anatid.open(db.path, tenant=1, embedding_dim=DIM, ensure=False, accelerators=False)
    try:
        # a handle holding no implementation is told so rather than reporting a removal it
        # could not make, and keeps the placeholder it was asked about
        with pytest.raises(IndexGenerationError, match="no implementation"):
            other.indexes.unregister("fts", retire=True)
        assert "fts" in other.indexes.names
        # and it refuses while a read is pinned
        o_idx = other.indexes.register(TableIndex(other))
        other.remember("m", now=T0 + MINUTE)
        o_idx.publish(o_idx.validate(o_idx.build_next(1)).generation)
        with o_idx.pin(1) as pin:
            assert pin
            with pytest.raises(IndexGenerationError, match="pinned"):
                other.indexes.unregister("tidx", retire=True)
        assert "tidx" in other.indexes.names
    finally:
        other.close()


# --------------------------------------------------------------------------- the oracle test


def test_random_mutations_merge_exactly_like_the_sql_oracle(file_db, idx):
    """The design's primary acceptance test: base + delta - tombstones == SQL, always."""
    db = file_db
    rng = random.Random(20260902)
    live: dict[int, list[int]] = {1: [], 2: []}
    policy = MaintenancePolicy(
        rebuild_after_rows=7, rebuild_after_ratio=None, rebuild_after_seconds=None
    )
    clock = T0
    compared = 0
    for step in range(220):
        clock += MINUTE
        tenant = rng.choice((1, 2))
        op = rng.random()
        if op < 0.45 or not live[tenant]:
            explicit = rng.random() < 0.2
            kw = {"memory_id": rng.randint(1, 50)} if explicit else {}
            try:
                m = db.remember(f"m{step}", tenant=tenant, now=clock, **kw)
            except Exception as exc:  # noqa: BLE001 - duplicate explicit id
                assert "already exists" in str(exc)
                continue
            live[tenant].append(m.memory_id)
        elif op < 0.65:
            old = rng.choice(live[tenant])
            new = db.supersede(old, f"s{step}", tenant=tenant, now=clock)
            live[tenant].remove(old)
            live[tenant].append(new.memory_id)
        elif op < 0.8:
            mid = live[tenant].pop(rng.randrange(len(live[tenant])))
            db.forget(mid, tenant=tenant, now=clock)
        elif op < 0.9:
            mid = live[tenant].pop(rng.randrange(len(live[tenant])))
            db.forget(mid, hard=True, tenant=tenant, now=clock)
        else:
            maintain(idx, tenant, policy)
        for t in (1, 2):
            got, reason = idx.candidates(t)
            if got is None:
                assert reason == "absent"
                continue
            assert got == sql_oracle(db, t) == set(live[t]), (step, t, reason)
            compared += 1
    assert compared > 150
    for t in (1, 2):
        assert idx.current_generation(t) is not None
        assert idx.health(t).reason in (HealthReason.FRESH, HealthReason.STALE_GENERATION)


@pytest.mark.slow
def test_readers_writers_and_maintenance_run_together_without_a_wrong_answer(file_db):
    """The lifecycle under contention: one writer, two readers and a maintenance loop.

    Each reader takes its snapshot inside a transaction, so the merged candidate set and the SQL
    oracle it is compared with are one instant, and any difference is a real one.  A generation
    the maintenance loop retires while a reader is on it would take the reader's storage with
    it, which is what the per-file lifecycle lock exists to stop.
    """
    db = file_db
    idx = db.indexes.register(TableIndex(db))
    for i in range(20):
        db.remember(f"seed {i}", tenant=1 + (i % 2), now=T0 + i * MINUTE)
    idx.publish(idx.validate(idx.build_next(1)).generation)
    idx.publish(idx.validate(idx.build_next(2)).generation)

    stop = threading.Event()
    mismatches: list[object] = []
    errors: list[BaseException] = []
    reads = [0]
    policy = MaintenancePolicy(
        rebuild_after_rows=3, rebuild_after_ratio=None, rebuild_after_seconds=None
    )

    def reader():
        try:
            while not stop.is_set():
                for tenant in (1, 2):
                    with db.transaction():
                        got, reason = idx.candidates(tenant)
                        expected = sql_oracle(db, tenant)
                    if got is not None and got != expected:
                        mismatches.append((tenant, reason, sorted(got ^ expected)))
                    reads[0] += 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def writer():
        clock = T0 + 100 * MINUTE
        live = {1: [], 2: []}
        rng = random.Random(4242)
        try:
            while not stop.is_set():
                clock += MINUTE
                tenant = rng.choice((1, 2))
                try:
                    if not live[tenant] or rng.random() < 0.5:
                        live[tenant].append(
                            db.remember(f"w{clock}", tenant=tenant, now=clock).memory_id
                        )
                    elif rng.random() < 0.5:
                        mid = live[tenant].pop(rng.randrange(len(live[tenant])))
                        db.forget(mid, hard=rng.random() < 0.5, tenant=tenant, now=clock)
                    else:
                        old = live[tenant].pop(rng.randrange(len(live[tenant])))
                        live[tenant].append(
                            db.supersede(old, f"s{clock}", tenant=tenant, now=clock).memory_id
                        )
                except ConflictError:
                    pass  # a maintenance publish touched the same catalog row; retryable
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def maintainer():
        try:
            while not stop.is_set():
                for tenant in (1, 2):
                    try:
                        maintain(idx, tenant, policy)
                    except (ConflictError, IndexGenerationError):
                        pass
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=reader),
        threading.Thread(target=reader),
        threading.Thread(target=writer),
        threading.Thread(target=maintainer),
    ]
    for t in threads:
        t.start()
    time.sleep(3.0)
    stop.set()
    for t in threads:
        t.join(30)
    assert not any(t.is_alive() for t in threads)
    assert errors == [], errors[:3]
    assert mismatches == [], mismatches[:3]
    assert reads[0] > 20
    for tenant in (1, 2):
        assert idx.candidates(tenant)[0] == sql_oracle(db, tenant)
        assert idx.validate(idx.current_generation(tenant)).ok


# --------------------------------------------------------------------------- migration


def test_migration_3_to_4_on_a_v3_file_with_data(tmp_path):
    """A v3 file keeps its rows and its BM25 index and gains the empty catalog."""
    path = tmp_path / "v3.anatid"
    with Anatid.open(path, tenant=1, embedding_dim=DIM, accelerators=False) as db:
        db.relate("Ada", "coffee", now=T0)
        m = db.remember("Ada likes coffee", entities=["Ada"], now=T0)
        n = db.supersede(m.memory_id, "Ada likes decaf", now=T0 + MINUTE)
        db.remember("tenant two", tenant=2, now=T0)
        db.rebuild_fts_index(now=T0 + 2 * MINUTE)
        # rewind to v3: drop the v4 catalog and the version stamp
        for table in S.INDEX_TABLES:
            db.execute(f"DROP TABLE {table}")
        db.execute("UPDATE anatid_meta SET schema_version = 3")
        assert set(S.missing_tables(db.connection)) == set(S.INDEX_TABLES)
    assert S.current_version(duckdb.connect(str(path), read_only=True)) == 3

    with Anatid.open(path, tenant=1, embedding_dim=DIM, accelerators=False) as db:
        assert db.info().schema_version == 4 and S.missing_tables(db.connection) == []
        assert "derived indexes (schema v4)" in db.info().contract
        for table in S.INDEX_TABLES:
            assert db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
        # data survived, history survived, the fts index survived
        assert [x.memory_id for x in db.recall_2hop("Ada")] == [n.memory_id]
        assert [x.memory_id for x in db.as_of(T0).recall_2hop("Ada")] == [m.memory_id]
        assert db.fts_status().available and db.recall("decaf", k=5)[0].memory_id == n.memory_id
        assert db.stats(tenant=2)["memories"] == 1
        assert db.doctor(all_tenants=True).ok
        # and the framework works on the migrated file
        idx = db.indexes.register(TableIndex(db))
        assert idx.health(1).reason is HealthReason.ABSENT
        assert maintain(idx, 1).action == "published"
        assert idx.candidates(1)[0] == sql_oracle(db, 1) == {n.memory_id}
        late = db.remember("after migration", now=T0 + 3 * MINUTE).memory_id
        assert idx.pending(1) == [late]
        assert idx.candidates(1)[0] == sql_oracle(db, 1)


def test_a_v4_file_written_before_the_journal_existed_is_repaired_on_open(tmp_path):
    """The v4 catalog grew a table, a sequence and a column while v4 was in development, so an
    open repairs those in place rather than asking for a schema bump nobody could migrate to."""
    path = tmp_path / "devv4.anatid"
    with Anatid.open(path, tenant=1, embedding_dim=DIM, accelerators=False) as db:
        db.remember("m", now=T0)
        db.execute(f"DROP TABLE {S.INDEX_JOURNAL_TABLE}")
        db.execute(f"DROP TABLE {S.INDEX_REGISTRY_TABLE}")
        db.execute(f"DROP SEQUENCE {S.INDEX_CHANGE_SEQUENCE}")
        db.execute(f"ALTER TABLE {S.INDEX_GENERATIONS_TABLE} DROP COLUMN published_unvalidated")
        assert db.doctor(all_tenants=True).findings  # drift is reported, not silently repaired
        drift = [f for f in db.doctor(all_tenants=True).findings if f.check == "schema_drift"]
        assert (
            drift
            and ("missing_column", S.INDEX_GENERATIONS_TABLE, "published_unvalidated")
            in drift[0].samples
        )

    with Anatid.open(path, tenant=1, embedding_dim=DIM, accelerators=False) as db:
        assert S.missing_tables(db.connection) == []
        assert S.ensure_index_columns(db.connection) == []
        assert db.doctor(all_tenants=True).ok
        idx = db.indexes.register(TableIndex(db))
        assert maintain(idx, 1).action == "published"
        later = db.remember("later", now=T0 + MINUTE).memory_id
        assert idx.pending(1) == [later]
        assert idx.candidates(1)[0] == sql_oracle(db, 1)


def test_generation_storage_names_are_identifiers_and_index_names_are_validated(db):
    g = Generation(index_name="fts", generation=3, tenant_id=7, built_at=T0)
    assert g.storage_suffix == "fts_t7_g3" and g.storage_name() == "anatid_idx_fts_t7_g3"
    assert (
        Generation(index_name="fts", generation=1, tenant_id=None, built_at=T0).storage_suffix
        == "fts_g1"
    )
    # A negative tenant id renders its sign as `n`: "-" is not an identifier character, so
    # `_t-7` would fail quote_ident and take the build down with it.  Namespace accepts a
    # negative id even though nothing in anatid mints one.
    negative = Generation(index_name="fts", generation=3, tenant_id=-7, built_at=T0)
    assert negative.storage_suffix == "fts_tn7_g3"
    assert negative.storage_name() == "anatid_idx_fts_tn7_g3"
    assert negative.storage_suffix != g.storage_suffix
    assert g.key == ("fts", 7, 3) and g.watermark.id is None
    with pytest.raises(ValueError):
        TableIndex(db, name="bad name; DROP TABLE memories")
    assert repr(db.indexes).startswith("<IndexRegistry")


def test_file_wide_index_uses_null_tenant_and_one_generation_for_the_file(file_db):
    db = file_db
    wide = db.indexes.register(FileWide(db))
    a = db.remember("a", tenant=1, now=T0).memory_id
    b = db.remember("b", tenant=2, now=T0).memory_id
    g = wide.publish(wide.validate(wide.build_next()).generation)
    assert g.tenant_id is None and g.storage_suffix == "wide_g1" and g.stats["rows"] == 2
    assert wide.current_generation() == g and wide.current_generation(5) == g
    c = db.remember("c", tenant=2, now=T0 + MINUTE).memory_id
    assert wide.pending() == [c] and wide.pending_keys() == [(2, c)]
    assert wide.pending(1) == [] and wide.pending(2) == [c]
    db.forget(a, tenant=1, now=T0 + 2 * MINUTE)
    assert wide.tombstone_keys() == [(1, a)]
    assert wide.tombstones(1) == [a] and wide.tombstones(2) == []
    lenient = MaintenancePolicy(
        rebuild_after_rows=None, rebuild_after_ratio=None, rebuild_after_seconds=None
    )
    h = wide.health(policy=lenient)
    assert h.reason is HealthReason.FRESH and h.tenant_id is None
    assert h.pending_rows == 1 and h.tombstone_rows == 1 and h.pending_ratio == 1.0
    assert wide.health().reason is HealthReason.STALE_GENERATION  # 2 of 2 rows changed
    merged = (wide.base_keys(g) | set(wide.pending_keys())) - set(wide.tombstone_keys())
    assert merged == {(2, b), (2, c)}
    for tenant in (1, 2):
        assert wide.candidates(tenant)[0] == sql_oracle(db, tenant)


def test_events_carry_ids_for_every_edge_table_the_verbs_touch(db):
    seen: list[derived_mod.IndexEvent] = []

    class Spy(DerivedIndex):
        name = "spy"
        source_table = "edges_about"
        source_id_column = "edge_id"

        def on_write(self, event):
            seen.append(event)

        def _build(self, gen):
            return {}

        def _validate(self, gen):
            return ValidationReport(ok=True, generation=gen)

        def _drop(self, gen):
            return None

    db.indexes.register(Spy(db))
    m = db.remember("m", entities=["Ada", "Bob"], now=T0)
    assert [(e.kind, len(e.doc_ids)) for e in seen] == [("insert", 2)]
    n = db.supersede(m.memory_id, "n", now=T0 + MINUTE, close_about_edges=True)
    assert [(e.kind, len(e.doc_ids), e.reason) for e in seen[1:]] == [
        ("insert", 2, None),
        ("close", 2, "supersede"),
    ]
    db.forget(n.memory_id, now=T0 + 2 * MINUTE)
    assert seen[-1].kind == "close" and len(seen[-1].doc_ids) == 2 and seen[-1].reason == "forget"
    db.forget(n.memory_id, hard=True, now=T0 + 3 * MINUTE)
    assert seen[-1].kind == "purge" and len(seen[-1].doc_ids) == 2
    assert all(e.table == "edges_about" and e.tenant_id == 1 for e in seen)
