"""Core-engine tests for anatid.

The last test in this file is the one that matters most: it loads the Phase 0 spike's 100k-memory
dataset into a real anatid database and checks ``recall_2hop`` against
``spike/bench/common.py``'s authoritative pure-numpy ``reference_r1`` for 200 query ids, ids and
timestamps, demanding ZERO mismatches.  Everything else checks that the verbs mean what their
docstrings say.
"""

from __future__ import annotations

import datetime as _dt
import threading
from pathlib import Path

import duckdb
import pytest

import anatid
from anatid import (
    Anatid,
    AsOf,
    BruteForceCeilingError,
    ConflictError,
    DatabasePool,
    EmbeddingDimensionError,
    Isolation,
    Namespace,
    NotFoundError,
    SchemaVersionError,
    TenantIsolationError,
)
from anatid import database as database_mod
from anatid import recall as recall_mod
from anatid import schema as schema_mod
from anatid import verbs as verbs_mod
from anatid.csr import (
    CsrBackend,
    discover_extension_path,
    extension_unsupported,
    frontier_sql,
)

from conftest import BUILT_EXTENSION, DIM, SPIKE_EXTENSION, SPIKE_SMALL, T0, vec

MINUTE = _dt.timedelta(minutes=1)

#: The C++ extension to test against: ``$ANATID_EXTENSION_PATH``, then ``ext/build``, then the
#: Phase 0 spike build (the order :func:`anatid.csr.discover_extension_path` uses).  The spike
#: tree alone is not enough: the productionised binary lives under ``ext/`` and the extension
#: oracle test below has to run wherever either has been built.
def _usable(path):
    """``path`` if the library will accept that binary, else None.

    The Phase 0 spike build (``anatid 0.0.0-spike``) filters edges on ``valid_to IS NULL``
    alone, so on a schema-v4 file it traverses the old version row a correction closed and
    disagrees with the SQL oracle.  ``anatid.csr`` refuses it; these tests skip on it rather
    than reporting a failure that is the binary's age.
    """
    if path is None or not Path(path).is_file():
        return None
    return None if extension_unsupported(path) else Path(path)


EXTENSION = (_usable(discover_extension_path()) or _usable(BUILT_EXTENSION)
             or _usable(SPIKE_EXTENSION))


# ============================================================================ schema

def test_schema_creates_every_table_and_the_catalog(db):
    con = db.connection
    missing = schema_mod.missing_tables(con)
    assert missing == []

    info = db.info()
    assert info.schema_version == schema_mod.SCHEMA_VERSION == 4
    assert info.embedding_dim == DIM
    assert info.duckdb_version == duckdb.__version__
    assert info.anatid_version == anatid.__version__
    # The contract is stored in the file, not only in the docs.
    for phrase in ("AS OF SYSTEM TIME", "file-per-tenant", "NOT incremental",
                   "NOT serializable", "brute-force"):
        assert phrase in info.contract, phrase

    # Schema v3: the four BM25 sidecar tables are part of the catalog set, the entity key is a
    # generated column, and the UNIQUE index that makes entity creation race-safe is present
    # on every file (REQUIRED_INDEXES, not selectable away through SchemaConfig.indexes).
    present = schema_mod.table_names(con)
    assert set(schema_mod.FTS_TABLES) <= set(schema_mod.CATALOG_TABLES) <= set(schema_mod.ALL_TABLES)
    assert set(schema_mod.ALL_TABLES) <= present
    assert set(schema_mod.FTS_TABLES) == {
        "anatid_fts_documents", "anatid_fts_docmap", "anatid_fts_dict", "anatid_fts_stats"}
    entity_cols = {r[1] for r in con.execute("PRAGMA table_info(entities)").fetchall()}
    assert "entity_key" in entity_cols
    assert "entity_key" not in schema_mod.insertable_columns(con, "entities")
    indexes = {r[0] for r in con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
    assert set(schema_mod.REQUIRED_INDEXES) <= indexes
    assert "ux_entities_tenant_key" in indexes

    # Schema v4: the derived-index catalog is part of the catalog set, and the journal's
    # ordering sequence exists (it is not a table, so missing_tables cannot see it).
    assert set(schema_mod.INDEX_TABLES) <= set(schema_mod.CATALOG_TABLES)
    assert set(schema_mod.INDEX_TABLES) == {
        "anatid_index_generations", "anatid_index_registry", "anatid_index_journal"}
    # A default handle registers the accelerators that cost nothing to have, so the registry
    # carries their DEFINITIONS from the first open -- that is what makes every other handle on
    # the file journal writes for them.  Nothing is built and nothing has been written yet.
    assert {r[0] for r in con.execute(
        f"SELECT index_name FROM {schema_mod.INDEX_REGISTRY_TABLE} WHERE enabled"
    ).fetchall()} == {"fts", "csr"}
    for table in (schema_mod.INDEX_GENERATIONS_TABLE, schema_mod.INDEX_JOURNAL_TABLE):
        assert con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    first = con.execute(f"SELECT nextval('{schema_mod.INDEX_CHANGE_SEQUENCE}')").fetchone()[0]
    second = con.execute(f"SELECT nextval('{schema_mod.INDEX_CHANGE_SEQUENCE}')").fetchone()[0]
    assert second > first
    assert "derived indexes (schema v4)" in info.contract


def test_system_columns_are_on_by_default_on_every_node_and_edge_table(db):
    con = db.connection
    system = {name for name, _t in schema_mod.SYSTEM_COLUMNS}
    for table in ("memories", "entities", "edges_about", "edges_relates"):
        cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
        assert system <= cols, f"{table} is missing {system - cols}"


def test_user_defined_labels_get_system_columns_unless_opted_out(db):
    db.create_node_label("document", [("title", "VARCHAR")])
    db.create_edge_type("cites", [("note", "VARCHAR")])
    con = db.connection
    for table in ("document", "edges_cites"):
        cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
        assert "valid_from" in cols and "tx_to" in cols and "episode_id" in cols

    db.create_node_label("scratch", [("x", "INTEGER")], system_columns=False)
    cols = {r[1] for r in con.execute("PRAGMA table_info(scratch)").fetchall()}
    assert "valid_from" not in cols
    assert cols == {"scratch_id", "tenant_id", "x"}


def test_identifiers_are_validated_not_interpolated():
    with pytest.raises(ValueError):
        schema_mod.quote_ident("memories; DROP TABLE memories")
    with pytest.raises(ValueError):
        schema_mod.quote_ident('a"b')
    assert schema_mod.quote_ident("edges_about") == '"edges_about"'


def test_schema_version_from_the_future_is_refused(tmp_path):
    path = tmp_path / "future.anatid"
    with Anatid.open(path, tenant=0, embedding_dim=DIM) as db:
        db.execute("UPDATE anatid_meta SET schema_version = 999")
    with pytest.raises(SchemaVersionError) as exc:
        Anatid.open(path, tenant=0, embedding_dim=DIM)
    assert exc.value.found == 999


def test_migration_hook_is_registered_and_used(tmp_path, monkeypatch):
    """Registered migrations run in order on open and bump the stored version."""
    path = tmp_path / "mig.anatid"
    with Anatid.open(path, tenant=0, embedding_dim=DIM) as db:
        db.execute("UPDATE anatid_meta SET schema_version = 0")

    ran = []
    monkeypatch.setattr(schema_mod, "MIGRATIONS",
                        {1: lambda con: ran.append(1), 2: lambda con: ran.append(2),
                         3: lambda con: ran.append(3), 4: lambda con: ran.append(4)})
    with Anatid.open(path, tenant=0, embedding_dim=DIM) as db:
        assert db.info().schema_version == schema_mod.SCHEMA_VERSION == 4
    assert ran == [1, 2, 3, 4]


def test_the_real_v1_to_v2_migration_adds_the_columns_and_backfills(tmp_path):
    """A genuine schema-1 file gains the fts id watermark and the audit counterpart column.

    Both columns exist because a claim in the docs was false without them (BM25 staleness that
    count(*) alone cannot see; a purged id surviving in another row's audit `reason` text), so
    the migration has to move the old free text into the new column, not just add it.

    Since schema v3 the ladder continues 2->3 on the same open, and since v4 on to 3->4, so
    the file is rewound all the way to v1 -- no derived-index catalog, no fts sidecar tables,
    no entity_key, no unique entity index -- and the whole ladder is asserted, not only its
    first rung.
    """
    path = tmp_path / "v1.anatid"
    with Anatid.open(path, tenant=0, embedding_dim=DIM) as db:
        m = db.remember("v1", entities=["Ada"])
        n = db.supersede(m.memory_id, "v2")
        con = db.connection
        # rewind the file to what schema v1 actually looked like: first the v4 objects, then
        # the v3 ones ...
        for table in schema_mod.INDEX_TABLES:
            db.execute(f"DROP TABLE IF EXISTS {table}")
        db.execute("DROP INDEX IF EXISTS ux_entities_tenant_key")
        for table in schema_mod.FTS_TABLES:
            db.execute(f"DROP TABLE IF EXISTS {table}")
        v1_entity_cols = [
            (r[1], r[2], bool(r[3]))
            for r in con.execute("PRAGMA table_info(entities)").fetchall()
            if r[1] != "entity_key"]
        db.execute("CREATE TABLE entities__v1 (" + ", ".join(
            f"{name} {typ}{' NOT NULL' if notnull else ''}"
            for name, typ, notnull in v1_entity_cols) + ")")
        db.execute("INSERT INTO entities__v1 SELECT "
                   + ", ".join(c for c, _t, _n in v1_entity_cols) + " FROM entities")
        db.execute("DROP TABLE entities")
        db.execute("ALTER TABLE entities__v1 RENAME TO entities")
        # ... then the v2 ones
        db.execute("ALTER TABLE anatid_meta DROP COLUMN fts_indexed_max_id")
        db.execute("UPDATE anatid_audit SET reason = 'superseded by ' || related_memory_id")
        db.execute("ALTER TABLE anatid_audit DROP COLUMN related_memory_id")
        db.execute("UPDATE anatid_meta SET schema_version = 1")
        old_id, new_id_ = m.memory_id, n.memory_id
        assert set(schema_mod.missing_tables(con)) == set(schema_mod.FTS_TABLES) | set(
            schema_mod.INDEX_TABLES)
        assert "entity_key" not in {
            r[1] for r in con.execute("PRAGMA table_info(entities)").fetchall()}

    with Anatid.open(path, tenant=0, embedding_dim=DIM) as db:
        assert db.info().schema_version == schema_mod.SCHEMA_VERSION == 4
        cols = {r[1] for r in db.execute("PRAGMA table_info(anatid_audit)").fetchall()}
        assert "related_memory_id" in cols
        assert {r[1] for r in db.execute("PRAGMA table_info(anatid_meta)").fetchall()} \
            >= {"fts_indexed_max_id"}
        # the 2->3 rung ran on the same open: sidecar tables, generated key, required index
        assert schema_mod.missing_tables(db.connection) == []
        assert "entity_key" in {
            r[1] for r in db.execute("PRAGMA table_info(entities)").fetchall()}
        assert "ux_entities_tenant_key" in {
            r[0] for r in db.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
        assert [e.name for e in db.entities_of(new_id_)] == ["Ada"]
        row = db.execute("SELECT memory_id, related_memory_id, reason FROM anatid_audit "
                         "WHERE action = 'supersede'").fetchone()
        assert row == (old_id, new_id_, "superseded")
        # and the purge that the column exists for now actually reaches that row
        receipt = db.forget(new_id_, hard=True)
        assert receipt.audit_rows_deleted == 1
        assert db.execute("SELECT count(*) FROM anatid_audit").fetchone()[0] == 0


def test_embedding_dim_is_taken_from_the_file_not_the_argument(tmp_path):
    path = tmp_path / "dim.anatid"
    Anatid.open(path, tenant=0, embedding_dim=16).close()
    with Anatid.open(path, tenant=0, embedding_dim=1536) as db:
        assert db.config.embedding_dim == 16


def test_the_version_string_agrees_everywhere(db):
    """pyproject, ``anatid.__version__``, ``anatid.database.__version__`` and the version a
    new file records in ``anatid_meta`` are one string; CI checks the first two, this checks
    the rest, because a wheel that reports 0.1.0 while pip says 0.1.1 is a bug report waiting
    to happen."""
    import pathlib
    import re

    pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.MULTILINE).group(1)
    assert anatid.__version__ == database_mod.__version__ == declared
    assert db.info().anatid_version == anatid.__version__
    assert db.execute("SELECT anatid_version FROM anatid_meta").fetchone()[0] == anatid.__version__

    def parts(text: str) -> tuple[int, ...]:
        return tuple(int(n) for n in re.findall(r"\d+", text)[:3])

    # The version string has to be at least the release that introduced the schema this build
    # writes.  0.1.1 is published on PyPI carrying schema v3; a build that writes v4 cannot also
    # call itself 0.1.1, because a file's recorded version is what says which release can open
    # it, and pip would refuse the upload anyway.
    introduced = schema_mod.SCHEMA_VERSION_RELEASES[schema_mod.SCHEMA_VERSION]
    assert parts(introduced) <= parts(anatid.__version__), (
        f"schema v{schema_mod.SCHEMA_VERSION} was introduced in {introduced}, but this build "
        f"calls itself {anatid.__version__}"
    )

    # The C++ extension is versioned with the package.  Source-checkout only (the wheel does not
    # ship ext/), so this checks the file when it is there.
    banner = pathlib.Path(__file__).resolve().parent.parent / "ext" / "src" / "anatid_extension.cpp"
    if banner.is_file():
        found = re.search(r'ANATID_EXT_VERSION\s*=\s*"([^"]+)"', banner.read_text())
        assert found is not None, "ANATID_EXT_VERSION is not in ext/src/anatid_extension.cpp"
        assert found.group(1) == anatid.__version__, (
            f"the extension reports {found.group(1)} and the package {anatid.__version__}"
        )


def test_recall_refuses_the_vector_arm_past_the_brute_force_ceiling(db, monkeypatch):
    """``BRUTE_FORCE_CEILING`` is behaviour, not a comment.

    The ceiling is lowered with a monkeypatch rather than by writing 100k rows; the check
    counts the rows the vector arm would scan (this tenant, this as_of, this kind filter, rows
    with an embedding), so every one of those predicates is exercised below.
    """
    monkeypatch.setattr(recall_mod, "BRUTE_FORCE_CEILING", 5)
    ids = [db.remember(f"m{i}", kind="note" if i % 2 else "fact", embedding=vec(1, i / 10),
                       now=T0 + i * MINUTE).memory_id for i in range(6)]
    db.remember("elsewhere", embedding=vec(0, 1), tenant=2, now=T0)     # another tenant
    q = vec(1, 0)

    with pytest.raises(BruteForceCeilingError) as exc:
        db.recall(embedding=q)
    assert exc.value.rows == 6 and exc.value.ceiling == 5 and exc.value.tenant_id == 1
    assert exc.value.retryable is False
    assert isinstance(exc.value, anatid.AnatidError)
    assert "allow_slow=True" in str(exc.value)

    # the escape hatch, on the method, the as-of view and the function form
    hits = db.recall(embedding=q, allow_slow=True)
    assert "vector" in hits.arms and len(hits) == 6
    with pytest.raises(BruteForceCeilingError):
        db.as_of(T0 + 10 * MINUTE).recall(embedding=q)
    assert "vector" in db.as_of(T0 + 10 * MINUTE).recall(embedding=q, allow_slow=True).arms
    with pytest.raises(BruteForceCeilingError):
        verbs_mod.recall(db, embedding=q)
    assert "vector" in verbs_mod.recall(db, embedding=q, allow_slow=True).arms

    # only the vector arm is guarded: text and graph answer at any size
    assert db.recall("m1").arms == ("text",)                # no ceiling on the text arm
    db.rebuild_fts_index()
    assert db.recall("m1").arms == ("text",)

    # the count is of the rows the arm would scan, nothing else
    assert db.recall(embedding=q, tenant=2).arms == ("vector",)          # 1 row there
    assert db.recall(embedding=q, kinds=["note"]).arms == ("vector",)   # 3 rows pass the filter
    assert db.as_of(T0 + 2 * MINUTE).recall(embedding=q).arms == ("vector",)   # 3 visible then
    db.forget(ids[0], now=T0 + 10 * MINUTE)                             # soft: 5 current rows
    assert db.recall(embedding=q).arms == ("vector",)
    assert recall_mod.vector_scan_rows(db.connection, tenant_id=1) == 5
    assert recall_mod.vector_scan_rows(db.connection, tenant_id=1, as_of=AsOf.coerce(T0)) == 1


# ============================================================================ verbs

def test_remember_round_trips_with_entities_and_edges(db):
    m = db.remember("Ada prefers dark roast", entities=["Ada", "coffee"], kind="preference",
                    embedding=vec(1, 0), writer="agent-1", confidence=0.9, now=T0)
    assert m.memory_id > 0 and m.tenant_id == 1 and m.is_current

    got = db.get(m.memory_id)
    assert got.content == "Ada prefers dark roast"
    assert got.kind == "preference"
    assert got.writer == "agent-1"
    assert got.valid_from == T0 and got.tx_from == T0 and got.created_at == T0
    assert got.valid_to is None and got.tx_to is None
    assert pytest.approx(got.confidence, abs=1e-6) == 0.9
    assert got.embedding == tuple(vec(1, 0))

    names = sorted(e.name for e in db.entities_of(m.memory_id))
    assert names == ["Ada", "coffee"]
    edges = db.execute("SELECT count(*) FROM edges_about WHERE src = ?", [m.memory_id]).fetchone()
    assert edges[0] == 2


def test_entities_are_deduplicated_by_name(db):
    a = db.remember("one", entities=["Ada"], now=T0)
    b = db.remember("two", entities=["Ada"], now=T0)
    ids = {e.entity_id for e in db.entities_of(a.memory_id)} | \
          {e.entity_id for e in db.entities_of(b.memory_id)}
    assert len(ids) == 1


def test_remember_rejects_a_wrong_length_embedding(db):
    with pytest.raises(EmbeddingDimensionError) as exc:
        db.remember("x", embedding=[1.0, 2.0])
    assert exc.value.expected == DIM and exc.value.got == 2


def test_remember_is_one_transaction(db, monkeypatch):
    """If the ABOUT-edge insert fails, the memory row must not survive either."""
    before = db.stats()["memories"]
    real = db.execute

    def boom(sql, params=None, *, con=None):
        if sql.startswith("INSERT INTO edges_about"):
            raise duckdb.Error("synthetic failure")
        return real(sql, params, con=con)

    monkeypatch.setattr(db, "execute", boom)
    with pytest.raises(duckdb.Error):
        db.remember("should roll back", entities=["Ada"], now=T0)
    monkeypatch.undo()
    assert db.stats()["memories"] == before


def test_episode_is_written_before_the_belief(db):
    m = db.remember("Ada drinks decaf", entities=["Ada"],
                    episode="transcript: 'these days I only drink decaf'",
                    episode_source="chat/2026-01-01", writer="extractor", now=T0)
    assert m.episode_id is not None
    ep = db.get_episode(m.episode_id)
    assert ep.content.startswith("transcript:")
    assert ep.source == "chat/2026-01-01"
    # the episode's tx_from is not after the memory's: evidence first
    assert ep.tx_from <= m.tx_from


def test_recall_2hop_walks_two_hops_and_respects_the_tenant(db):
    db.relate("Ada", "coffee", now=T0)
    db.relate("coffee", "Colombia", now=T0)
    db.relate("Colombia", "altitude", now=T0)          # 3 hops from Ada
    near = db.remember("about Colombia", entities=["Colombia"], now=T0)
    far = db.remember("about altitude", entities=["altitude"], now=T0)

    ids2 = [m.memory_id for m in db.recall_2hop("Ada")]
    assert near.memory_id in ids2
    assert far.memory_id not in ids2

    ids3 = [m.memory_id for m in db.recall_2hop("Ada", hops=3)]
    assert far.memory_id in ids3

    # A different tenant sees nothing (scoping inside one file).
    assert db.recall_2hop(db.entity_id("Ada"), tenant=2) == []


def test_recall_2hop_orders_newest_first(db):
    db.relate("Ada", "coffee", now=T0)
    ids = []
    for i in range(5):
        ids.append(db.remember(f"m{i}", entities=["coffee"], now=T0 + i * MINUTE).memory_id)
    got = [m.memory_id for m in db.recall_2hop("Ada", limit=3)]
    assert got == list(reversed(ids))[:3]


def test_context_returns_memories_about_one_entity(db):
    db.relate("Ada", "coffee", now=T0)
    direct = db.remember("about Ada", entities=["Ada"], now=T0)
    neighbour = db.remember("about coffee", entities=["coffee"], now=T0)
    assert [m.memory_id for m in db.context("Ada")] == [direct.memory_id]
    assert neighbour.memory_id in [m.memory_id for m in db.context("Ada", hops=1)]


def test_supersede_closes_the_old_and_recall_returns_the_new(db):
    db.relate("Ada", "coffee", now=T0)
    old = db.remember("Ada prefers dark roast", entities=["Ada"], writer="a1", now=T0)
    t1 = T0 + MINUTE
    new = db.supersede(old.memory_id, "Ada switched to decaf", writer="a2", now=t1)

    stored_old = db.get(old.memory_id)
    assert stored_old.valid_to == t1 and not stored_old.is_current
    assert db.get(new.memory_id).is_current

    assert [m.memory_id for m in db.recall_2hop("Ada")] == [new.memory_id]
    # ABOUT entities are inherited so the new memory stays reachable from the same graph.
    assert [e.name for e in db.entities_of(new.memory_id)] == ["Ada"]
    edge = db.execute("SELECT src, dst FROM edges_supersedes WHERE src = ?",
                      [new.memory_id]).fetchone()
    assert edge == (new.memory_id, old.memory_id)


def test_as_of_returns_the_pre_supersede_answer(db):
    db.relate("Ada", "coffee", now=T0)
    old = db.remember("dark roast", entities=["Ada"], now=T0)
    t1 = T0 + MINUTE
    new = db.supersede(old.memory_id, "decaf", now=t1)

    before = db.as_of(t1 - _dt.timedelta(seconds=1))
    assert [m.memory_id for m in before.recall_2hop("Ada")] == [old.memory_id]
    assert before.get(old.memory_id).content == "dark roast"
    assert before.get(new.memory_id) is None

    # Half-open intervals: exactly at t1 the new one is already the answer.
    assert [m.memory_id for m in db.as_of(t1).recall_2hop("Ada")] == [new.memory_id]
    # ... and now() sees only the new one.
    assert [m.memory_id for m in db.recall_2hop("Ada")] == [new.memory_id]


def test_as_of_separates_valid_time_from_transaction_time(db):
    """A fact true since 2020 but recorded today is invisible to a tx-time query about 2021."""
    db.relate("Ada", "coffee", now=T0)
    m = db.remember("Ada has always liked coffee", entities=["Ada"],
                    valid_from=_dt.datetime(2020, 1, 1), now=T0)
    old_valid = AsOf(valid_time=_dt.datetime(2021, 1, 1), tx_time=None)
    assert [x.memory_id for x in db.recall_2hop("Ada", as_of=old_valid)] == [m.memory_id]
    old_tx = AsOf(valid_time=None, tx_time=_dt.datetime(2021, 1, 1))
    assert db.recall_2hop("Ada", as_of=old_tx) == []


def test_reinforce_bumps_access_count(db):
    m = db.remember("x", now=T0)
    t1 = T0 + MINUTE
    got = db.reinforce(m.memory_id, now=t1)
    assert got.access_count == 1 and got.last_access_at == t1
    got = db.reinforce(m.memory_id, amount=3, confidence=0.5, now=t1)
    assert got.access_count == 4
    assert pytest.approx(got.confidence, abs=1e-6) == 0.5
    with pytest.raises(NotFoundError):
        db.reinforce(-1)


def test_soft_forget_keeps_the_audit_trail_and_history(db):
    db.relate("Ada", "coffee", now=T0)
    m = db.remember("secret-ish", entities=["Ada"], now=T0)
    t1 = T0 + MINUTE
    receipt = db.forget(m.memory_id, reason="user asked", writer="a1", now=t1)
    assert receipt.hard is False and receipt.audit_rows_written == 1

    assert db.recall_2hop("Ada") == []
    assert db.get(m.memory_id).valid_to == t1
    assert [x.memory_id for x in db.as_of(T0).recall_2hop("Ada")] == [m.memory_id]
    audit = db.execute(
        "SELECT action, reason, writer FROM anatid_audit WHERE memory_id = ?",
        [m.memory_id]).fetchall()
    assert audit == [("forget_soft", "user asked", "a1")]
    # The ABOUT edge is closed the way the memory is: a new version carries the valid_to, the
    # original version keeps its open interval and is retired on the transaction axis.
    edge_versions = db.execute(
        "SELECT version, valid_to, tx_to FROM edges_about WHERE src = ? ORDER BY version",
        [m.memory_id]).fetchall()
    assert edge_versions == [(1, None, t1), (2, t1, None)]
    assert db.entities_of(m.memory_id) == []
    assert [e.name for e in db.entities_of(m.memory_id, as_of=T0)] == ["Ada"]


def test_hard_forget_purges_the_row_its_edges_its_embedding_and_its_provenance(db):
    db.relate("Ada", "coffee", now=T0)
    old = db.remember("v1", entities=["Ada"], embedding=vec(1, 0),
                      episode="the only evidence", now=T0)
    new = db.supersede(old.memory_id, "v2", now=T0 + MINUTE)
    episode_id = old.episode_id
    db.forget(old.memory_id, reason="right to erasure", now=T0 + 2 * MINUTE)   # soft first

    receipt = db.forget(new.memory_id, hard=True, now=T0 + 3 * MINUTE)
    assert receipt.hard and receipt.memories_deleted == 1

    purged = new.memory_id
    con = db.connection
    # Zero rows in EVERY table that could reference the purged memory.
    assert con.execute("SELECT count(*) FROM memories WHERE memory_id = ?", [purged]).fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM edges_about WHERE src = ?", [purged]).fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM edges_supersedes WHERE src = ? OR dst = ?",
                       [purged, purged]).fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM anatid_audit WHERE memory_id = ?", [purged]).fetchone()[0] == 0
    # The embedding lived in the purged row, so it is gone with it.
    assert con.execute("SELECT count(*) FROM memories WHERE embedding IS NOT NULL "
                       "AND memory_id = ?", [purged]).fetchone()[0] == 0
    # No as-of view can resurrect it.
    assert db.as_of(T0).get(purged) is None
    assert db.get(purged) is None

    # Now purge the original too; its orphaned episode goes with it.
    r2 = db.forget(old.memory_id, hard=True, now=T0 + 4 * MINUTE)
    assert r2.episodes_deleted == 1
    assert con.execute("SELECT count(*) FROM episodes WHERE episode_id = ?",
                       [episode_id]).fetchone()[0] == 0
    for table in ("memories", "edges_about", "edges_supersedes", "anatid_audit"):
        col = "memory_id" if table in ("memories", "anatid_audit") else "src"
        assert con.execute(f"SELECT count(*) FROM {table} WHERE {col} = ?",
                           [old.memory_id]).fetchone()[0] == 0


def test_hard_forget_keeps_an_episode_that_other_memories_cite(db):
    ep = db.episode("shared evidence", now=T0)
    a = db.remember("a", episode_id=ep.episode_id, now=T0)
    db.remember("b", episode_id=ep.episode_id, now=T0)
    r = db.forget(a.memory_id, hard=True, now=T0 + MINUTE)
    assert r.episodes_deleted == 0
    assert db.get_episode(ep.episode_id) is not None


def test_prune_is_dry_by_default_and_needs_a_policy(db):
    old = db.remember("stale", now=T0)
    fresh = db.remember("fresh", now=T0 + 10 * MINUTE)
    db.reinforce(fresh.memory_id, amount=5, now=T0 + 11 * MINUTE)

    with pytest.raises(ValueError):
        db.prune()

    report = db.prune(older_than=T0 + 5 * MINUTE)
    assert report.dry_run and report.memory_ids == (old.memory_id,)
    assert db.get(old.memory_id).is_current           # nothing happened

    report = db.prune(older_than=T0 + 5 * MINUTE, max_access_count=0, dry_run=False,
                      now=T0 + 20 * MINUTE)
    assert report.memory_ids == (old.memory_id,)
    assert not db.get(old.memory_id).is_current
    assert db.get(fresh.memory_id).is_current

    hard = db.prune(max_access_count=99, dry_run=False, hard=True, now=T0 + 30 * MINUTE)
    assert hard.memory_ids == (fresh.memory_id,)
    assert db.get(fresh.memory_id) is None


def test_provenance_walks_the_chain_back_to_the_source(db):
    v1 = db.remember("v1", writer="a1", episode="first observation", now=T0)
    v2 = db.supersede(v1.memory_id, "v2", writer="a2", now=T0 + MINUTE)
    v3 = db.supersede(v2.memory_id, "v3", writer="a3", now=T0 + 2 * MINUTE)

    p = db.provenance(v3.memory_id)
    assert [m.memory_id for m in p.chain] == [v3.memory_id, v2.memory_id, v1.memory_id]
    assert p.root.memory_id == v1.memory_id
    assert p.depth == 2
    assert p.writers == ("a3", "a2", "a1")
    assert p.source_text == "first observation"
    assert [e.src for e in p.edges] == [v3.memory_id, v2.memory_id]
    with pytest.raises(NotFoundError):
        db.provenance(-1)


def test_verbs_are_deterministic_when_now_is_supplied(db):
    m = db.remember("x", now=T0, created_at=T0, valid_from=T0)
    assert m.created_at == m.valid_from == m.tx_from == T0
    e = db.relate("a", "b", now=T0)
    assert e.tx_from == T0 and e.valid_from == T0
    ep = db.episode("src", now=T0)
    assert ep.created_at == T0


def test_function_forms_delegate_to_the_methods(db):
    from anatid import verbs

    m = verbs.remember(db, "via function", now=T0)
    assert verbs.provenance(db, m.memory_id).memory_id == m.memory_id
    assert isinstance(verbs.as_of(T0), AsOf)


# ============================================================================ recall

def test_recall_fuses_the_three_arms_with_rrf(db):
    db.relate("Ada", "coffee", now=T0)
    graphy = db.remember("colombian beans", entities=["coffee"], embedding=vec(0, 1), now=T0)
    texty = db.remember("dark roast beans", embedding=vec(0, 0, 1), now=T0)
    vecy = db.remember("unrelated words", embedding=vec(1, 0), now=T0)
    db.rebuild_fts_index()

    hits = db.recall("dark roast", embedding=vec(1, 0), seed_entity="Ada", k=5)
    by_id = {h.memory_id: h for h in hits}
    assert set(hits.arms) == {"vector", "text", "graph"}
    assert by_id[texty.memory_id].text_rank == 1
    assert by_id[vecy.memory_id].vector_rank == 1
    assert by_id[graphy.memory_id].graph_rank is not None
    assert "graph" in by_id[graphy.memory_id].sources
    assert hits == sorted(hits, key=lambda h: (-h.score, h.memory_id))
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    # RRF: a memory only in one arm at rank 1 scores 1/(60+1)
    assert pytest.approx(by_id[texty.memory_id].score, rel=1e-9) == \
        sum(1.0 / (60 + r) for r in (by_id[texty.memory_id].text_rank,
                                     by_id[texty.memory_id].vector_rank)
            if r is not None)


def test_recall_reports_that_bm25_is_stale_rather_than_hiding_it(legacy_db, caplog):
    """0.1.1's index cannot see a row written after its rebuild, and recall says so.

    ``accelerators=False``: the claim is about the non-incremental index.  The derived one makes
    the row searchable inside the writing transaction, so there is nothing to hide and nothing
    to report -- see the test below.
    """
    db = legacy_db
    db.remember("indexed content", embedding=vec(1, 0), now=T0)
    db.rebuild_fts_index()
    status = db.fts_status()
    assert status.available and not status.stale and status.indexed_rows == 1

    db.remember("content written after the index", embedding=vec(0, 1), now=T0)
    status = db.fts_status()
    assert status.stale and status.pending_rows == 1

    with caplog.at_level("WARNING", logger="anatid.recall"):
        hits = db.recall("content", embedding=vec(1, 0), k=5)
    assert hits.bm25_stale and hits.pending_fts_rows == 1
    assert any("stale" in r.message.lower() or "stale" in r.getMessage().lower()
               for r in caplog.records)
    assert any("stale" in n for n in hits.notes)

    with pytest.raises(anatid.StaleIndexError):
        db.recall("content", on_stale_fts="error")

    db.rebuild_fts_index()
    hits = db.recall("content", embedding=vec(1, 0), k=5)
    assert not hits.bm25_stale and hits.pending_fts_rows == 0


def test_recall_without_an_fts_index_says_the_text_arm_was_skipped(legacy_db):
    """No index of either kind: the arm is skipped and the result says which arm and why."""
    db = legacy_db
    db.remember("no index yet", embedding=vec(1, 0), now=T0)
    hits = db.recall("index", embedding=vec(1, 0), k=5)
    assert not hits.bm25_available
    assert "text" not in hits.arms
    assert any("no fts index" in n for n in hits.notes)


def test_on_a_default_handle_the_text_arm_answers_before_anything_is_built(db):
    """The default handle attaches the derived full-text index, so there is no such state.

    Nothing has been built and no generation is published; the text arm answers by scanning the
    tenant's visible documents exactly, and reports that it is available and not stale.  This is
    what ``Anatid.open(accelerators=True)`` buys over the test above.
    """
    m = db.remember("no index yet", embedding=vec(1, 0), now=T0)
    hits = db.recall("index", embedding=vec(1, 0), k=5)
    assert hits.bm25_available
    assert "text" in hits.arms
    assert [h.memory.memory_id for h in hits] == [m.memory_id]
    assert db.fts_status().stale is False


def test_recall_with_no_usable_arm_returns_nothing(db):
    db.remember("something", embedding=vec(1, 0), now=T0)
    assert list(db.recall()) == []


def test_recall_filters_by_kind_and_tenant(db):
    db.remember("a fact", kind="fact", embedding=vec(1, 0), now=T0)
    note = db.remember("a note", kind="note", embedding=vec(1, 0), now=T0)
    hits = db.recall(embedding=vec(1, 0), kinds=["note"], k=5)
    assert [h.memory_id for h in hits] == [note.memory_id]
    assert list(db.recall(embedding=vec(1, 0), tenant=7, k=5)) == []


def test_recall_hits_carry_about_names(db):
    db.relate("Ada", "coffee", now=T0)
    m = db.remember("beans", entities=["Ada", "coffee"], embedding=vec(1, 0), now=T0)
    hits = db.recall(embedding=vec(1, 0), k=5)
    assert hits[0].memory_id == m.memory_id
    assert set(hits[0].about) == {"Ada", "coffee"}


def test_literal_and_parameterized_recall_paths_agree(db):
    """The int-literal fast path and the bound-parameter path must return identical rows.

    ``recall_2hop_ids`` inlines integers when the read is current-state with no kind filter
    (measured 1.32 ms vs 1.91 ms at 100k memories); any other read binds parameters.  Both are
    exercised here on the same data.
    """
    db.relate("Ada", "coffee", now=T0)
    db.relate("coffee", "Colombia", now=T0)
    ids = [db.remember(f"m{i}", entities=["Colombia"], kind="fact",
                       now=T0 + i * MINUTE).memory_id for i in range(5)]

    fast = db.recall_2hop_ids("Ada", limit=10)                       # literals, no params
    slow = db.recall_2hop_ids("Ada", limit=10, kinds=["fact"])       # bound parameters
    later = db.recall_2hop_ids("Ada", limit=10, as_of=T0 + 100 * MINUTE)
    assert [m for m, _ in fast] == [m for m, _ in slow] == [m for m, _ in later]
    assert [m for m, _ in fast] == list(reversed(ids))


def test_rebuild_fts_index_also_builds_the_terms_index(legacy_db):
    """The postings index the spike measured at BM25 8.57 ms vs 12.51 ms must survive a rebuild.

    ``accelerators=False``: this is 0.1.1's single file-wide index and its fixed table names.  A
    generation carries the same index under its own name (``anatid_idx_fts_g<n>_termid``), which
    ``tests/test_fts_framework.py`` checks.
    """
    db = legacy_db
    db.remember("indexable words here", now=T0)
    db.rebuild_fts_index()
    idx = {r[0] for r in db.connection.execute(
        "SELECT index_name FROM duckdb_indexes()").fetchall()}
    assert "idx_fts_terms_termid" in idx
    db.remember("more words", now=T0)
    db.rebuild_fts_index()          # create_fts_index drops the schema; it must be recreated
    idx = {r[0] for r in db.connection.execute(
        "SELECT index_name FROM duckdb_indexes()").fetchall()}
    assert "idx_fts_terms_termid" in idx
    assert db.rebuild_fts_index(terms_index=False).indexed_rows == 2


def test_rrf_fuse_is_the_documented_formula():
    from anatid.recall import rrf_fuse

    fused = rrf_fuse({"a": [(1, 0.9), (2, 0.8)], "b": [(2, 5.0), (3, 4.0)]}, k=60)
    scores = {mid: sc for mid, sc, _r, _s in fused}
    assert pytest.approx(scores[1]) == 1 / 61
    assert pytest.approx(scores[2]) == 1 / 62 + 1 / 61
    assert [mid for mid, *_ in fused] == [2, 1, 3]


# ============================================================================ transactions

def test_write_write_conflict_raises_ConflictError(file_db, tmp_path):
    """DuckDB's optimistic MVCC aborts the second updater of the same row."""
    m = file_db.remember("contended", now=T0)
    a = file_db.connection.cursor()
    b = file_db.connection.cursor()
    a.execute("BEGIN")
    b.execute("BEGIN")
    a.execute("UPDATE memories SET content = 'a' WHERE memory_id = ?", [m.memory_id])
    with pytest.raises(ConflictError) as exc:
        file_db.execute("UPDATE memories SET content = 'b' WHERE memory_id = ?",
                        [m.memory_id], con=b)
    assert exc.value.retryable
    assert "conflict" in str(exc.value).lower()
    b.execute("ROLLBACK")
    a.execute("COMMIT")
    a.close()
    b.close()
    assert file_db.get(m.memory_id).content == "a"


def test_concurrent_reinforce_of_the_same_row_conflicts_through_the_verb(file_db):
    m = file_db.remember("hot row", now=T0)
    other = Anatid.open(file_db.path, tenant=1, embedding_dim=DIM, ensure=False)
    try:
        with file_db.transaction():
            file_db.execute("UPDATE memories SET access_count = 1 WHERE memory_id = ?",
                            [m.memory_id])
            with pytest.raises(ConflictError):
                other.reinforce(m.memory_id, now=T0)
    finally:
        other.close()


def test_appends_never_conflict(file_db):
    """The spike measured 4 writers + 2 readers for 30 s with 0 errors; this is the shape of it."""
    errors: list[BaseException] = []
    written: list[int] = []
    lock = threading.Lock()

    def worker(i: int):
        try:
            for j in range(20):
                m = file_db.remember(f"w{i}-{j}", entities=[f"e{i}"], now=T0)
                with lock:
                    written.append(m.memory_id)
        except BaseException as exc:      # noqa: BLE001 - the test is about there being none
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(set(written)) == 80
    assert file_db.stats()["memories"] == 80


def test_transaction_rolls_back_on_error_and_nests(db):
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.remember("inner", now=T0)
            raise RuntimeError("boom")
    assert db.stats()["memories"] == 0

    with db.transaction():
        with db.transaction():                 # joins the outer transaction
            db.remember("kept", now=T0)
    assert db.stats()["memories"] == 1


def test_each_thread_gets_its_own_connection(db):
    seen = []

    def grab():
        seen.append(id(db.connection))

    t = threading.Thread(target=grab)
    t.start()
    t.join()
    assert seen and seen[0] != id(db.connection)


# ============================================================================ tenancy

def test_scoped_tenants_share_a_file_and_are_only_filtered(file_db):
    a = file_db.remember("tenant 1", entities=["shared"], now=T0)
    b = file_db.remember("tenant 2", entities=["shared"], tenant=2, now=T0)
    assert file_db.get(a.memory_id) is not None
    assert file_db.get(a.memory_id, tenant=2) is None
    assert file_db.get(b.memory_id, tenant=2) is not None
    # ... but raw SQL sees both: scoping is not isolation, and the docstring says so.
    total = file_db.connection.execute("SELECT count(*) FROM memories").fetchone()[0]
    assert total == 2


def test_file_per_tenant_refuses_cross_tenant_access(tmp_path):
    with Anatid.open(tmp_path / "t7.anatid", tenant=7, embedding_dim=DIM,
                     isolation=Isolation.FILE_PER_TENANT) as db:
        db.remember("mine", now=T0)
        with pytest.raises(TenantIsolationError):
            db.remember("theirs", tenant=8, now=T0)
        with pytest.raises(TenantIsolationError):
            db.recall_2hop(1, tenant=8)
        assert db.resolve_tenant(7).tenant_id == 7


def test_database_pool_gives_one_file_per_tenant_with_lru_close(tmp_path):
    with DatabasePool(str(tmp_path / "t_{tenant}.anatid"), max_open=2,
                      embedding_dim=DIM) as pool:
        one, two = pool.get(1), pool.get(2)
        one.remember("one", entities=["x"], now=T0)
        two.remember("two", entities=["x"], now=T0)
        assert one.namespace.isolation is Isolation.FILE_PER_TENANT
        assert (tmp_path / "t_1.anatid").is_file() and (tmp_path / "t_2.anatid").is_file()
        assert one.stats()["memories"] == 1 and two.stats()["memories"] == 1

        three = pool.get(3)                     # evicts tenant 1 (LRU)
        assert len(pool) == 2
        assert one.closed
        assert pool.known_tenants() == [2, 3]
        three.remember("three", now=T0)

        again = pool.get(1)                     # reopened from its file
        assert not again.closed and again.stats()["memories"] == 1
    assert again.closed


def test_pool_requires_a_tenant_in_the_path_template(tmp_path):
    with pytest.raises(ValueError):
        DatabasePool(str(tmp_path / "one-file.anatid"))


def test_cross_tenant_read_needs_an_explicit_read_only_attach(tmp_path):
    with DatabasePool(str(tmp_path / "t_{tenant}.anatid"), embedding_dim=DIM) as pool:
        pool.get(1).remember("tenant one secret", now=T0)
        pool.get(2).remember("tenant two secret", now=T0)
        alias = pool.attach_read_only(1, 2)
        host = pool.get(1)
        assert host.attached == {alias: str(tmp_path / "t_2.anatid")}
        rows = host.connection.execute(f"SELECT content FROM {alias}.memories").fetchall()
        assert rows == [("tenant two secret",)]
        with pytest.raises(duckdb.Error):
            host.connection.execute(f"DELETE FROM {alias}.memories")
        host.detach(alias)
        assert host.attached == {}


def test_namespace_value_type():
    ns = Namespace(3, "acme", Isolation.FILE_PER_TENANT)
    assert ns.is_isolated and ns.name == "acme"
    assert Namespace.coerce(5).tenant_id == 5
    assert Namespace.coerce(None, ns) is ns
    with pytest.raises(TypeError):
        Namespace.coerce(True)


# ============================================================================ csr backend

def test_frontier_sql_reports_which_path_it_used():
    _sql, _p, path = frontier_sql(1, 2, 2, backend=None)
    assert path == "sql"
    backend = CsrBackend(enabled=False)
    assert backend.active == "sql" and not backend.available
    assert backend.describe()["active"] == "sql"


def test_extension_is_optional_and_the_default_is_sql(db):
    assert db.expand_path == "sql"
    assert db.build_csr() is None                  # not enabled -> no-op
    with pytest.raises(anatid.ExtensionUnavailable):
        db.require_csr_extension()


@pytest.mark.skipif(EXTENSION is None, reason="C++ anatid extension not built")
def test_csr_extension_and_sql_expansion_agree(tmp_path):
    """Same results either way -- the promise the fallback rests on."""
    path = tmp_path / "csr.anatid"
    with Anatid.open(path, tenant=1, embedding_dim=DIM, use_csr_extension=True,
                     extension_path=EXTENSION, require_extension=True) as db:
        # Dense entity ids: the extension builds a dense per-tenant CSR.
        edges = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (3, 7), (7, 8)]
        for i, (a, b) in enumerate(edges):
            db.relate(a, b, now=T0 + i * MINUTE)
        expected_by_entity = {}
        for e in range(9):
            m = db.remember(f"memory for {e}", entities=[e], memory_id=1000 + e,
                            now=T0 + e * MINUTE)
            expected_by_entity[e] = m.memory_id

        sql_ids = {h: [m.memory_id for m in db.recall_2hop(h, limit=50)] for h in range(9)}
        assert db.expand_path == "sql"             # writes marked the snapshot stale

        info = db.build_csr()
        assert info is not None and info.edges > 0
        assert db.expand_path == "extension"
        db.require_csr_extension()
        ext_ids = {h: [m.memory_id for m in db.recall_2hop(h, limit=50)] for h in range(9)}

        assert ext_ids == sql_ids
        assert sorted(ext_ids[0]) == sorted(expected_by_entity[e] for e in (0, 1, 2, 4, 5))

        # A new edge is invisible to the stale snapshot until it is rebuilt -- documented, and
        # anatid protects against it by falling back to SQL after every RELATES_TO write.
        db.relate(6, 100, now=T0 + 99 * MINUTE)
        assert db.expand_path == "sql"


def test_require_extension_raises_when_the_binary_is_missing(tmp_path):
    with pytest.raises(anatid.ExtensionUnavailable):
        Anatid.open(tmp_path / "x.anatid", tenant=1, embedding_dim=DIM,
                    extension_path=tmp_path / "nope.duckdb_extension",
                    require_extension=True)


# ============================================================================ maintenance

def test_recluster_preserves_the_rows_and_the_indexes(file_db):
    for i in range(50):
        file_db.remember(f"m{i}", entities=[f"e{i % 5}"], now=T0 + i * MINUTE)
    before = [m.memory_id for m in file_db.recall_2hop(file_db.entity_id("e0"), limit=50)]
    counts = file_db.recluster(["memories", "edges_about"])
    assert counts["memories"] == 50
    after = [m.memory_id for m in file_db.recall_2hop(file_db.entity_id("e0"), limit=50)]
    assert before == after
    idx = {r[0] for r in file_db.connection.execute(
        "SELECT index_name FROM duckdb_indexes()").fetchall()}
    assert "idx_memories_id" in idx


def test_close_is_idempotent_and_the_context_manager_closes(tmp_path):
    db = Anatid.open(tmp_path / "c.anatid", tenant=1, embedding_dim=DIM)
    db.close()
    db.close()
    assert db.closed
    with pytest.raises(anatid.AnatidError):
        db.connection


# ============================================================================ THE ORACLE

@pytest.mark.slow
@pytest.mark.oracle
def test_recall_2hop_matches_the_spike_reference_for_200_queries(spike_db, spike_common,
                                                                 spike_queries):
    """anatid's 2-hop recall == ``spike/bench/common.py`` ``reference_r1`` -- zero mismatches.

    ``reference_r1`` is the Phase 0 spike's authoritative pure-numpy implementation, computed
    straight from the Parquet files: memories ABOUT any entity in the seed's 2-hop frontier,
    ``valid_to IS NULL``, same tenant, ordered ``created_at DESC, memory_id DESC``, first 20.
    Ids AND timestamps are compared.
    """
    qids = list(spike_common.VERIFY_QUERY_IDS)          # 0..199
    assert len(qids) >= 200

    mismatches = []
    for qid in qids:
        q = spike_queries[qid]
        got = spike_db.recall_2hop_ids(
            int(q["seed_entity_id"]), tenant=int(q["tenant_id"]),
            limit=spike_common.R1_LIMIT)
        expected = spike_common.reference_r1(
            "small", int(q["tenant_id"]), int(q["seed_entity_id"]),
            limit=spike_common.R1_LIMIT)
        ok, message = spike_common.compare_r1(got, expected, check_created_at=True)
        if not ok:
            mismatches.append((qid, message))

    assert mismatches == [], f"{len(mismatches)}/{len(qids)} queries differ: {mismatches[:3]}"

    # Non-trivial: the frontiers really do reach a lot of memories.
    total = sum(spike_common.reference_r1_count("small", int(spike_queries[q]["tenant_id"]),
                                                int(spike_queries[q]["seed_entity_id"]))
                for q in qids[:20])
    assert total > 20 * 30


@pytest.mark.slow
@pytest.mark.oracle
def test_recall_2hop_hydrated_matches_the_id_query(spike_db, spike_queries):
    """``recall_2hop`` (Memory objects) returns exactly what the benchmarked id query returns."""
    for qid in range(0, 25):
        q = spike_queries[qid]
        ids = [m for m, _ in spike_db.recall_2hop_ids(int(q["seed_entity_id"]),
                                                      tenant=int(q["tenant_id"]), limit=20)]
        mems = spike_db.recall_2hop(int(q["seed_entity_id"]), tenant=int(q["tenant_id"]),
                                    limit=20)
        assert [m.memory_id for m in mems] == ids
        assert all(m.tenant_id == int(q["tenant_id"]) and m.is_current for m in mems)


@pytest.mark.slow
@pytest.mark.oracle
@pytest.mark.skipif(EXTENSION is None, reason="C++ anatid extension not built")
def test_csr_extension_matches_the_reference_on_the_spike_dataset(tmp_path_factory,
                                                                  spike_common, spike_queries):
    """The extension path gives the same 2-hop answers as the reference, on real data.

    Zero mismatches over the same 200 queries as the SQL-path oracle test, and the frontier
    really comes from the extension: ``frontier_sql`` is asked which path it took.
    """
    path = tmp_path_factory.mktemp("csr-oracle") / "ext.anatid"
    with Anatid.open(path, tenant=0, embedding_dim=64, use_csr_extension=True,
                     extension_path=EXTENSION, require_extension=True) as db:
        db.load_parquet(SPIKE_SMALL, rebuild_fts=False, build_csr=True)
        assert db.expand_path == "extension"
        mismatches = []
        for qid in range(0, 200):
            q = spike_queries[qid]
            _sql, _params, used = frontier_sql(int(q["tenant_id"]), int(q["seed_entity_id"]), 2,
                                               backend=db.csr)
            assert used == "extension"
            got = db.recall_2hop_ids(int(q["seed_entity_id"]), tenant=int(q["tenant_id"]),
                                     limit=spike_common.R1_LIMIT)
            expected = spike_common.reference_r1("small", int(q["tenant_id"]),
                                                 int(q["seed_entity_id"]),
                                                 limit=spike_common.R1_LIMIT)
            ok, message = spike_common.compare_r1(got, expected, check_created_at=True)
            if not ok:
                mismatches.append((qid, message))
        assert mismatches == [], f"{len(mismatches)} differ: {mismatches[:3]}"


@pytest.mark.slow
def test_hybrid_recall_on_the_spike_dataset_returns_relevant_results(spike_db, spike_queries):
    """The hybrid path runs end to end on 100k memories and both arms contribute."""
    spike_db.rebuild_fts_index()
    q = spike_queries[1000]
    hits = spike_db.recall(q["query_text"], embedding=list(q["query_embedding"]),
                           tenant=int(q["tenant_id"]), k=20)
    assert len(hits) == 20
    assert set(hits.arms) == {"vector", "text"}
    assert not hits.bm25_stale
    assert all(h.memory.tenant_id == int(q["tenant_id"]) for h in hits)
    assert any(h.vector_rank is not None for h in hits)
    assert any(h.text_rank is not None for h in hits)


# ============================================================================ review regressions
# Each test below pins one defect a three-lens review found and this pass fixed.  They are named
# for the guarantee, not for the bug, so a future rewrite that reintroduces the bug fails loudly.


def test_column_types_are_validated_not_interpolated(db):
    """A property *type* is structure, like a name, and must not reach DDL unchecked.

    ``create_node_label("victim", [("x", "INTEGER); DROP TABLE memories; CREATE TABLE zz (a INT")])``
    used to drop ``memories``: names went through ``quote_ident`` but the type was interpolated
    raw into the f-string.
    """
    db.remember("still here", entities=["Ada"])
    payload = "INTEGER); DROP TABLE memories; CREATE TABLE zz (a INTEGER"
    with pytest.raises(ValueError):
        db.create_node_label("victim", [("x", payload)])
    with pytest.raises(ValueError):
        db.create_edge_type("victim_edge", [("x", payload)])
    # nothing ran: the real tables are intact and the injected one does not exist
    assert db.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
    assert "zz" not in schema_mod.table_names(db.connection)
    assert "victim" not in schema_mod.table_names(db.connection)

    for bad in ("VARCHAR'", "INTEGER -- x", 'VARCHAR"', "INTEGER, y VARCHAR", "1BAD", "INT)"):
        with pytest.raises(ValueError):
            schema_mod.check_type(bad)
    for good in ("VARCHAR", "DECIMAL(18, 3)", "FLOAT[64]", "BIGINT[]",
                 "TIMESTAMP WITH TIME ZONE", "STRUCT(a INTEGER, b VARCHAR)", "VARCHAR NOT NULL"):
        assert schema_mod.check_type(good) == good
    # and a legitimate type still builds a working table
    db.create_node_label("document", [("title", "VARCHAR"), ("scores", "FLOAT[4]")])
    assert "document" in schema_mod.table_names(db.connection)


def test_hydration_never_crosses_a_tenant_in_a_shared_file(file_db):
    """``recall``/``recall_2hop`` hydrate by (memory_id, tenant_id), not by id alone.

    ``memory_id`` is not unique across tenants in a SCOPED file -- ``remember(memory_id=...)`` and
    per-tenant Parquet imports both produce collisions -- and hydrating by id alone handed tenant
    2's content back inside a tenant-1 result.
    """
    file_db.remember("TENANT-1 PUBLIC", entities=["Ada"], memory_id=4242, tenant=1, now=T0)
    file_db.remember("TENANT-2 SECRET", entities=["Zed"], memory_id=4242, tenant=2, now=T0)

    ids = file_db.recall_2hop_ids("Ada", tenant=1)
    assert [m for m, _ in ids] == [4242]

    rows = file_db.recall_2hop("Ada", tenant=1)
    assert [(m.tenant_id, m.content) for m in rows] == [(1, "TENANT-1 PUBLIC")]

    hits = file_db.recall(seed_entity="Ada", tenant=1, k=5)
    assert [(h.memory.tenant_id, h.memory.content) for h in hits] == [(1, "TENANT-1 PUBLIC")]
    assert hits[0].about == ("Ada",)          # not ("Ada", "Zed")

    other = file_db.recall_2hop("Zed", tenant=2)
    assert [(m.tenant_id, m.content) for m in other] == [(2, "TENANT-2 SECRET")]


def test_bm25_staleness_survives_an_insert_cancelled_out_by_a_purge(legacy_file_db):
    """Row count alone is not a watermark; the id watermark is what makes the claim true.

    ``accelerators=False``: a watermark is how 0.1.1's index guesses at staleness.  The derived
    index does not guess -- the journal has a row per change, so an insert and a purge do not
    cancel out -- and ``tests/test_fts_framework.py`` proves that directly.
    """
    file_db = legacy_file_db
    a = file_db.remember("alpha alpha", now=T0)
    file_db.remember("bravo bravo", now=T0)
    file_db.rebuild_fts_index(now=T0)
    assert file_db.fts_status().stale is False
    assert file_db.recall("alpha", k=5)[0].memory.content == "alpha alpha"

    # one insert + one hard purge: count(*) is back where it started
    file_db.remember("charlie charlie charlie", now=T0)
    file_db.forget(a.memory_id, hard=True, now=T0)

    status = file_db.fts_status()
    assert status.current_rows == status.indexed_rows == 2      # the count really did cancel out
    assert status.indexed_max_id != status.current_max_id
    assert status.stale is True

    hits = file_db.recall("charlie", k=5)
    assert hits.bm25_stale is True
    assert any("stale" in n for n in hits.notes)

    file_db.rebuild_fts_index(now=T0)
    assert file_db.fts_status().stale is False
    assert file_db.recall("charlie", k=5)[0].memory.content == "charlie charlie charlie"


def test_hard_forget_leaves_no_audit_row_naming_the_purged_id(db):
    """The supersede counterpart is a column, so the purge can actually find it."""
    old = db.remember("v1", entities=["Ada"], now=T0)
    new = db.supersede(old.memory_id, "v2", now=T0 + MINUTE)
    row = db.execute("SELECT memory_id, related_memory_id, reason FROM anatid_audit "
                     "WHERE action = 'supersede'").fetchone()
    assert row == (old.memory_id, new.memory_id, "superseded")

    receipt = db.forget(new.memory_id, hard=True, now=T0 + 2 * MINUTE)
    assert receipt.audit_rows_deleted == 1
    purged = str(new.memory_id)
    leftovers = db.execute(
        "SELECT count(*) FROM anatid_audit WHERE memory_id = ? OR related_memory_id = ? "
        "OR contains(coalesce(reason, ''), ?)",
        [new.memory_id, new.memory_id, purged]).fetchone()[0]
    assert leftovers == 0


def test_supersede_of_a_stale_memory_refuses_to_fork_the_chain(db):
    """The serialized version of the supersede race must fail, not silently branch."""
    a = db.remember("v1", entities=["Ada"], now=T0)
    b = db.supersede(a.memory_id, "v2-branch-1", now=T0 + MINUTE)
    with pytest.raises(ConflictError) as exc:
        db.supersede(a.memory_id, "v2-branch-2", now=T0 + 2 * MINUTE)
    assert str(b.memory_id) in str(exc.value)

    assert [m.content for m in db.recall_2hop("Ada")] == ["v2-branch-1"]
    assert db.execute("SELECT count(*) FROM edges_supersedes WHERE dst = ?",
                      [a.memory_id]).fetchone()[0] == 1

    # opt in explicitly and you get the fork you asked for
    c = db.supersede(a.memory_id, "v2-branch-2", now=T0 + 3 * MINUTE, allow_fork=True)
    assert {m.content for m in db.recall_2hop("Ada")} == {"v2-branch-1", "v2-branch-2"}
    assert c.is_current


def test_deterministic_transaction_errors_are_not_retryable_conflicts(file_db):
    """``ConflictError`` promises a retry helps.  These three fail identically forever."""
    file_db.execute("BEGIN")
    try:
        with pytest.raises(duckdb.Error) as exc:
            file_db.remember("inside a manual transaction")
        assert not isinstance(exc.value, ConflictError)
        assert "transaction within a transaction" in str(exc.value)
    finally:
        file_db.execute("ROLLBACK")

    for stmt in ("COMMIT", "ROLLBACK"):
        with pytest.raises(duckdb.Error) as exc:
            file_db.execute(stmt)
        assert not isinstance(exc.value, ConflictError), stmt
        assert "no transaction is active" in str(exc.value)

    # the genuine MVCC abort is still translated (covered in full by the conflict tests above)
    assert issubclass(ConflictError, anatid.AnatidError)


def test_recluster_preserves_constraints_and_defaults(db):
    """A CTAS rebuild silently dropped every NOT NULL and every column DEFAULT."""
    def snapshot():
        return {r[0]: (r[1], r[2]) for r in db.execute(
            "SELECT column_name, is_nullable, column_default FROM duckdb_columns() "
            "WHERE table_name = 'memories' ORDER BY column_index").fetchall()}

    db.remember("one", entities=["Ada"], now=T0)
    before = snapshot()
    assert before["memory_id"][0] is False and before["access_count"][1] == "0"

    db.recluster(["memories"])
    assert snapshot() == before

    with pytest.raises(duckdb.Error):
        db.execute("INSERT INTO memories (memory_id, tenant_id, content, created_at) "
                   "VALUES (NULL, NULL, 'unreachable', NULL)")
    db.execute("INSERT INTO memories (memory_id, tenant_id, content, created_at) "
               "VALUES (7, 1, 'defaulted', now())")
    assert db.execute("SELECT access_count FROM memories WHERE memory_id = 7").fetchone()[0] == 0


def test_extension_discovery_walks_past_src_and_prefers_the_ext_build_tree(tmp_path, monkeypatch):
    """Discovery used to stop at ``src``, so neither build tree in this repo was reachable."""
    from anatid import csr as csr_mod

    repo = tmp_path / "repo"
    pkg = repo / "src" / "anatid"
    pkg.mkdir(parents=True)
    monkeypatch.delenv(csr_mod.EXTENSION_PATH_ENV, raising=False)
    monkeypatch.setattr(csr_mod, "__file__", str(pkg / "csr.py"))
    assert csr_mod.discover_extension_path() is None

    spike_ext = repo / "spike" / "extension" / "build" / "release" / "extension" / "anatid"
    spike_ext.mkdir(parents=True)
    (spike_ext / "anatid.duckdb_extension").write_bytes(b"spike")
    assert csr_mod.discover_extension_path() == spike_ext / "anatid.duckdb_extension"

    prod_ext = repo / "ext" / "build" / "release" / "extension" / "anatid"
    prod_ext.mkdir(parents=True)
    (prod_ext / "anatid.duckdb_extension").write_bytes(b"prod")
    assert csr_mod.discover_extension_path() == prod_ext / "anatid.duckdb_extension"

    monkeypatch.setenv(csr_mod.EXTENSION_PATH_ENV, str(tmp_path / "nope"))
    assert csr_mod.discover_extension_path() is None      # an explicit path never falls back


def test_erasure_hooks_run_inside_the_purge_transaction(file_db):
    """``forget(hard=True)`` reaches tables anatid does not own, and only through a hook."""
    file_db.create_node_label("notes", [("body", "VARCHAR"), ("memory_id", "BIGINT")])
    m = file_db.remember("Ada's home address is 12 Elm Street", entities=["Ada"], now=T0)
    file_db.execute("INSERT INTO notes (notes_id, tenant_id, body, memory_id) VALUES (1, 1, ?, ?)",
                    ["Ada's home address is 12 Elm Street", m.memory_id])

    seen: list[tuple] = []

    def hook(db, memory_id, tenant_id, content):
        seen.append((memory_id, tenant_id, content))
        return _count_rows(db.execute("DELETE FROM notes WHERE memory_id = ? AND tenant_id = ?",
                                      [memory_id, tenant_id]))

    file_db.register_erasure_hook(hook)
    file_db.register_erasure_hook(hook)                 # idempotent registration
    receipt = file_db.forget(m.memory_id, hard=True, now=T0 + MINUTE)
    assert seen == [(m.memory_id, 1, "Ada's home address is 12 Elm Street")]
    assert receipt.extra_rows_deleted == 1
    assert file_db.execute("SELECT count(*) FROM notes").fetchone()[0] == 0

    # a raising hook aborts the whole purge rather than committing a half-erased file
    other = file_db.remember("keep me", entities=["Ada"], now=T0)
    file_db.erasure_hooks.clear()
    file_db.register_erasure_hook(lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        file_db.forget(other.memory_id, hard=True, now=T0 + MINUTE)
    assert file_db.get(other.memory_id) is not None


def _count_rows(result) -> int:
    row = result.fetchone()
    return int(row[0]) if row and row[0] is not None else 0
