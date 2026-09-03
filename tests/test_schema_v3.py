"""Schema v3: the cross-tenant BM25 leak and the entity-creation race, at the storage layer.

Two defects are fixed *below* the verbs, because that is the only level at which they stay
fixed:

* the fts document key was ``memory_id``, which collides across tenants, so one tenant's BM25
  hit could return another tenant's row -- and the corpus statistics behind every score were
  computed over every tenant's text;
* entity creation was ``SELECT`` then ``INSERT`` with nothing in between, so concurrent writers
  minted several entities for one name.

Every test here either reproduces the reported defect against the v2 shape or proves the v3
shape cannot express it.  The migration tests build a real v2-shaped file with the v2 DDL and
migrate it, rather than asserting about a v3 file.
"""

from __future__ import annotations

import datetime as _dt
import threading

import duckdb
import pytest

from anatid import schema as S

T0 = _dt.datetime(2026, 1, 1, 0, 0, 0)


# --------------------------------------------------------------------------- the v2 file shape

#: The schema-v2 DDL, verbatim, so the migration tests run against a file this build did not
#: write.  Do not "modernise" this: it is the old shape, on purpose.
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
    audit_id   BIGINT NOT NULL, tenant_id INTEGER NOT NULL, memory_id BIGINT,
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
    rel_kind VARCHAR, valid_from TIMESTAMP, valid_to TIMESTAMP, tx_from TIMESTAMP, tx_to TIMESTAMP,
    writer VARCHAR, episode_id BIGINT, confidence FLOAT)""",
    """CREATE TABLE edges_supersedes (
    edge_id BIGINT NOT NULL, src BIGINT NOT NULL, dst BIGINT NOT NULL, tenant_id INTEGER NOT NULL,
    tx_from TIMESTAMP, writer VARCHAR)""",
]

#: The v2 fts index: keyed on memory_id, which is exactly the bug.
V2_FTS_SQL = (r"PRAGMA create_fts_index('memories', 'memory_id', 'content', stemmer='none', "
              r"stopwords='none', ignore='(\.|[^a-z])+', strip_accents=0, lower=1, overwrite=1)")


def v2_file(path, *, version: int = 2, fts: bool = False) -> duckdb.DuckDBPyConnection:
    """A database written the way schema v2 (or v1) wrote it."""
    con = duckdb.connect(str(path))
    con.execute("INSTALL fts")
    con.execute("LOAD fts")
    for stmt in V2_DDL:
        if version == 1 and stmt.startswith("CREATE TABLE anatid_meta"):
            stmt = stmt.replace("    fts_indexed_max_id BIGINT,\n", "")
        if version == 1 and stmt.startswith("CREATE TABLE anatid_audit"):
            stmt = stmt.replace("    related_memory_id BIGINT, ", "    ")
        con.execute(stmt)
    con.execute("CREATE INDEX idx_entities_name ON entities (tenant_id, name)")
    con.execute(
        "INSERT INTO anatid_meta VALUES (?, ?, 8, '0.1.0', ?, TRUE, NULL, NULL, "
        + ("" if version == 1 else "NULL, ") + "'v2 contract')",
        [version, T0, duckdb.__version__])
    if fts:
        con.execute(V2_FTS_SQL)
    return con


def add_memory(con, memory_id, tenant_id, content, *, at=T0):
    con.execute(
        "INSERT INTO memories (memory_id, tenant_id, content, kind, created_at, valid_from, "
        "tx_from, confidence, access_count) VALUES (?, ?, ?, 'fact', ?, ?, ?, 1.0, 0)",
        [memory_id, tenant_id, content, at, at, at])


def add_entity(con, entity_id, tenant_id, name, *, at=T0):
    con.execute(
        "INSERT INTO entities (entity_id, tenant_id, kind, name, valid_from, tx_from, confidence) "
        "VALUES (?, ?, 'person', ?, ?, ?, 1.0)", [entity_id, tenant_id, name, at, at])


# --------------------------------------------------------------------------- the v3 BM25 query

#: The BM25 SQL schema v3 is designed for -- the query anatid.recall compiles.  It reads df from
#: the PER-TENANT dictionary and (num_docs, avgdl) from the per-tenant stats, and it prunes the
#: candidate documents to the tenant BEFORE scoring, so no other tenant's text can influence
#: either the ranking or the score.  Kept here as well as in recall so this file can prove the
#: property without depending on that module.
BM25_SQL = f"""
    WITH q AS (
        SELECT DISTINCT term FROM (
            SELECT unnest(string_split_regex(
                regexp_replace(lower(?), '(\\.|[^a-z])+', ' ', 'g'), '\\s+')) AS term)
        WHERE term <> ''
    ), dt AS (
        SELECT docid, memory_id, len FROM {S.FTS_DOCS_TABLE} WHERE tenant_id = ?
    ), qt AS (
        SELECT d.termid, td.df FROM {S.FTS_INDEX_SCHEMA}.dict d
        JOIN q ON d.term = q.term
        JOIN {S.FTS_DICT_TABLE} td ON td.termid = d.termid AND td.tenant_id = ?
    ), st AS (
        SELECT num_docs, avgdl FROM {S.FTS_STATS_TABLE} WHERE tenant_id = ?
    ), tf AS (
        SELECT t.docid, t.termid, count(*)::DOUBLE AS tf
        FROM {S.FTS_INDEX_SCHEMA}.terms t
        JOIN qt ON t.termid = qt.termid
        JOIN dt ON dt.docid = t.docid
        GROUP BY t.docid, t.termid
    ), sc AS (
        SELECT tf.docid, sum(ln((s.num_docs - qt.df + 0.5) / (qt.df + 0.5) + 1)
               * tf.tf * (1.2 + 1)
               / (tf.tf + 1.2 * (1 - 0.75 + 0.75 * dt.len / s.avgdl))) AS score
        FROM tf JOIN qt ON tf.termid = qt.termid
        JOIN dt ON dt.docid = tf.docid
        CROSS JOIN st s
        GROUP BY tf.docid
    )
    SELECT m.memory_id, sc.score
    FROM sc JOIN dt ON dt.docid = sc.docid
    JOIN memories m ON m.memory_id = dt.memory_id AND m.tenant_id = ?
    WHERE m.valid_to IS NULL AND m.tx_to IS NULL
    ORDER BY sc.score DESC, m.memory_id ASC
    LIMIT ?
"""


def bm25(con, tenant_id: int, query: str, limit: int = 10):
    return [(int(r[0]), float(r[1])) for r in con.execute(
        BM25_SQL, [query, tenant_id, tenant_id, tenant_id, tenant_id, limit]).fetchall()]


#: The v2 BM25 query, for the "this used to leak" half of the reproduction.
V2_BM25_SQL = """
    WITH q AS (
        SELECT DISTINCT term FROM (
            SELECT unnest(string_split_regex(
                regexp_replace(lower(?), '(\\.|[^a-z])+', ' ', 'g'), '\\s+')) AS term)
        WHERE term <> ''
    ), qt AS (SELECT d.termid, d.df FROM fts_main_memories.dict d JOIN q ON d.term = q.term
    ), tf AS (
        SELECT t.docid, t.termid, count(*)::DOUBLE AS tf
        FROM fts_main_memories.terms t JOIN qt ON t.termid = qt.termid GROUP BY 1, 2
    ), sc AS (
        SELECT tf.docid, sum(ln((s.num_docs - qt.df + 0.5) / (qt.df + 0.5) + 1)
               * tf.tf * 2.2 / (tf.tf + 1.2 * (0.25 + 0.75 * d.len / s.avgdl))) AS score
        FROM tf JOIN qt ON tf.termid = qt.termid
        JOIN fts_main_memories.docs d ON d.docid = tf.docid
        CROSS JOIN fts_main_memories.stats s GROUP BY tf.docid
    )
    SELECT m.memory_id, sc.score FROM sc
    JOIN fts_main_memories.docs d ON d.docid = sc.docid
    JOIN memories m ON m.memory_id = d.name
    WHERE m.tenant_id = ? AND m.valid_to IS NULL AND m.tx_to IS NULL
    ORDER BY sc.score DESC, m.memory_id ASC
"""


def rebuild_fts(con):
    for stmt in S.fts_rebuild_statements():
        con.execute(stmt)


# ======================================================================= defect 1: BM25 leak

def test_v2_leaks_across_tenants_and_v3_does_not(tmp_path):
    """The reviewer's reproduction, both halves, in one test.

    Same ``memory_id`` in two tenants, different text.  Under v2 a search for the OTHER
    tenant's word came back with this tenant's row -- which both corrupts the result and tells
    tenant 1 that the word appears somewhere in the file.
    """
    con = v2_file(tmp_path / "leak.anatid")
    add_memory(con, 4242, 1, "apples are crisp")
    add_memory(con, 4242, 2, "swordfish are large")
    con.execute(V2_FTS_SQL)

    # --- v2: the leak, reproduced.
    leaked = con.execute(V2_BM25_SQL, ["swordfish", 1]).fetchall()
    assert leaked, "expected the v2 index to leak; if this is empty the repro is wrong"
    assert int(leaked[0][0]) == 4242
    assert con.execute("SELECT content FROM memories WHERE memory_id = 4242 AND tenant_id = 1"
                       ).fetchone()[0] == "apples are crisp"

    # --- v3: migrate the same file, rebuild, ask again.
    S.ensure_schema(con)
    rebuild_fts(con)

    assert bm25(con, 1, "swordfish") == []          # tenant 1 has no such word
    assert bm25(con, 2, "apples") == []             # nor tenant 2
    assert [m for m, _ in bm25(con, 1, "apples")] == [4242]
    assert [m for m, _ in bm25(con, 2, "swordfish")] == [4242]
    # ... and the two ids are two documents now, not one.
    assert sorted(r[0] for r in con.execute(
        f"SELECT fts_doc_id FROM {S.FTS_SOURCE_TABLE} ORDER BY 1").fetchall()) == ["1:4242", "2:4242"]
    con.close()


def test_bm25_corpus_statistics_are_per_tenant(tmp_path):
    """df/idf must not be able to see another tenant's corpus.

    Tenant 1 has one document containing "swordfish" out of two; tenant 2 has fifty documents,
    all containing it.  A shared dictionary would give tenant 1 df=51 and flatten the term's
    idf to nothing.  Per-tenant statistics give it df=1.
    """
    con = v2_file(tmp_path / "df.anatid")
    S.ensure_schema(con)
    add_memory(con, 1, 1, "swordfish are large")
    add_memory(con, 2, 1, "apples are crisp")
    for i in range(50):
        add_memory(con, 100 + i, 2, "swordfish swim here")
    rebuild_fts(con)

    termid = con.execute(
        f"SELECT termid FROM {S.FTS_INDEX_SCHEMA}.dict WHERE term = 'swordfish'").fetchone()[0]
    df = dict(con.execute(f"SELECT tenant_id, df FROM {S.FTS_DICT_TABLE} WHERE termid = ?",
                          [termid]).fetchall())
    assert df == {1: 1, 2: 50}
    global_df = con.execute(f"SELECT df FROM {S.FTS_INDEX_SCHEMA}.dict WHERE termid = ?",
                            [termid]).fetchone()[0]
    assert int(global_df) == 51, "the shared dictionary still counts every tenant -- unused on purpose"

    stats = dict((int(t), (int(n), float(a))) for t, n, a in con.execute(
        f"SELECT tenant_id, num_docs, avgdl FROM {S.FTS_STATS_TABLE}").fetchall())
    assert stats[1][0] == 2 and stats[2][0] == 50

    # The score tenant 1 gets is computed from df=1 of 2 documents, so it is the score it would
    # get if tenant 2 did not exist in the file at all.
    alone = duckdb.connect(str(tmp_path / "alone.anatid"))
    alone.execute("INSTALL fts")
    alone.execute("LOAD fts")
    S.ensure_schema(alone)
    add_memory(alone, 1, 1, "swordfish are large")
    add_memory(alone, 2, 1, "apples are crisp")
    rebuild_fts(alone)
    assert bm25(con, 1, "swordfish")[0][1] == pytest.approx(bm25(alone, 1, "swordfish")[0][1])
    con.close()
    alone.close()


def test_migration_drops_the_v2_index_so_nothing_serves_from_it(tmp_path):
    con = v2_file(tmp_path / "drop.anatid", fts=False)
    add_memory(con, 4242, 1, "apples are crisp")
    con.execute(V2_FTS_SQL)
    assert con.execute("SELECT count(*) FROM information_schema.schemata "
                       "WHERE schema_name = 'fts_main_memories'").fetchone()[0] == 1

    S.ensure_schema(con)
    assert con.execute("SELECT count(*) FROM information_schema.schemata "
                       "WHERE schema_name = 'fts_main_memories'").fetchone()[0] == 0
    # ... and the staleness watermarks are cleared, so nothing claims the index is fresh.
    assert con.execute("SELECT fts_indexed_at, fts_indexed_rows, fts_indexed_max_id "
                       "FROM anatid_meta").fetchone() == (None, None, None)
    con.close()


def test_fts_purge_removes_every_verbatim_copy(tmp_path):
    """The source table holds content verbatim, so a hard forget has to reach it."""
    con = v2_file(tmp_path / "purge.anatid")
    S.ensure_schema(con)
    add_memory(con, 7, 1, "the pangolin ate my homework")
    add_memory(con, 8, 1, "unrelated")
    add_memory(con, 7, 2, "another tenant, same id")
    rebuild_fts(con)
    assert S.fts_purge(con, 1, 7) > 0
    con.execute("DELETE FROM memories WHERE memory_id = 7 AND tenant_id = 1")

    for table in S.FTS_TABLES:
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]
        for col in cols:
            hits = con.execute(
                f"SELECT count(*) FROM {table} WHERE CAST({col} AS VARCHAR) LIKE '%pangolin%'"
            ).fetchone()[0]
            assert hits == 0, f"{table}.{col} still holds the erased text"
    assert bm25(con, 1, "pangolin") == []
    # The other tenant's identically-numbered memory is untouched.
    assert con.execute(f"SELECT count(*) FROM {S.FTS_SOURCE_TABLE} WHERE memory_id = 7"
                       ).fetchone()[0] == 1
    assert S.fts_purge(con, 1, 7) == 0            # idempotent
    con.close()


def test_fts_purge_is_safe_before_any_index_exists(tmp_path):
    con = v2_file(tmp_path / "nopurge.anatid")
    S.ensure_schema(con)
    add_memory(con, 1, 1, "never indexed")
    assert S.fts_purge(con, 1, 1) == 0
    con.close()


def test_duplicate_memory_ids_do_not_duplicate_a_document(tmp_path):
    """A duplicate id inside one tenant is an integrity fault, not a licence to break the key."""
    con = v2_file(tmp_path / "dupid.anatid")
    S.ensure_schema(con)
    add_memory(con, 5, 1, "first")
    add_memory(con, 5, 1, "second", at=T0 + _dt.timedelta(days=1))
    rebuild_fts(con)
    assert con.execute(f"SELECT count(*) FROM {S.FTS_SOURCE_TABLE}").fetchone()[0] == 1
    assert con.execute(f"SELECT count(*) FROM {S.FTS_INDEX_SCHEMA}.docs").fetchone()[0] == 1
    con.close()


def test_fts_indexes_superseded_rows_so_as_of_bm25_still_works(tmp_path):
    con = v2_file(tmp_path / "asof.anatid")
    S.ensure_schema(con)
    add_memory(con, 1, 1, "the old belief")
    con.execute("UPDATE memories SET valid_to = ? WHERE memory_id = 1", [T0])
    rebuild_fts(con)
    assert con.execute(f"SELECT count(*) FROM {S.FTS_SOURCE_TABLE}").fetchone()[0] == 1
    con.close()


# =================================================================== defect 2: entity race

def test_entity_key_is_generated_and_canonical(tmp_path):
    con = duckdb.connect(str(tmp_path / "keys.anatid"))
    S.ensure_schema(con)
    add_entity(con, 1, 1, "  Ada   LOVELACE\t")
    key = con.execute("SELECT entity_key FROM entities").fetchone()[0]
    assert key == "ada lovelace"
    assert S.entity_key("  Ada   LOVELACE\t") == key
    # It is derived, not stored: renaming the entity moves the key with it.
    con.execute("UPDATE entities SET name = 'Grace  Hopper' WHERE entity_id = 1")
    assert con.execute("SELECT entity_key FROM entities").fetchone()[0] == "grace hopper"
    con.close()


@pytest.mark.parametrize("second", ["Ada Lovelace", "ada lovelace", "  ADA   lovelace  ",
                                    "Ada\tLovelace", "Ada\nLovelace"])
def test_duplicate_entity_names_are_rejected(tmp_path, second):
    con = duckdb.connect(str(tmp_path / f"dup{abs(hash(second))}.anatid"))
    S.ensure_schema(con)
    add_entity(con, 1, 1, "Ada Lovelace")
    with pytest.raises(duckdb.ConstraintException):
        add_entity(con, 2, 1, second)
    assert con.execute("SELECT count(*) FROM entities").fetchone()[0] == 1
    con.close()


def test_the_constraint_is_per_tenant(tmp_path):
    con = duckdb.connect(str(tmp_path / "pertenant.anatid"))
    S.ensure_schema(con)
    add_entity(con, 1, 1, "Ada Lovelace")
    add_entity(con, 2, 2, "Ada Lovelace")
    assert con.execute("SELECT count(*) FROM entities").fetchone()[0] == 2
    con.close()


def test_unnamed_entities_are_not_deduplicated(tmp_path):
    """A NULL name has no canonical key, so two of them are two entities, not a violation."""
    con = duckdb.connect(str(tmp_path / "null.anatid"))
    S.ensure_schema(con)
    con.execute("INSERT INTO entities (entity_id, tenant_id, name) VALUES (1, 1, NULL), (2, 1, NULL)")
    assert con.execute("SELECT count(*) FROM entities").fetchone()[0] == 2
    con.close()


def test_concurrent_creation_of_one_entity_yields_one_row(tmp_path):
    """Defect 2's reproduction: 100 racing writers, one new name.

    The reviewer got four entity rows out of the v2 SELECT-then-INSERT.  The constraint makes
    that unrepresentable: the losers get a retryable error and re-read.
    """
    db = duckdb.connect(str(tmp_path / "race.anatid"))
    S.ensure_schema(db)
    started = threading.Barrier(100)
    outcomes: list[str] = []
    lock = threading.Lock()

    def writer(i: int) -> None:
        con = db.cursor()
        started.wait()
        for _attempt in range(20):
            row = con.execute(
                "SELECT entity_id FROM entities WHERE tenant_id = 1 AND entity_key = "
                + S.entity_key_sql("?"), ["Ada Lovelace"]).fetchone()
            if row is not None:
                with lock:
                    outcomes.append("found")
                return
            try:
                add_entity(con, i, 1, "Ada Lovelace")
            except (duckdb.ConstraintException, duckdb.TransactionException):
                continue                              # somebody else won; loop re-reads
            with lock:
                outcomes.append("created")
            return
        with lock:
            outcomes.append("gave-up")

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert db.execute("SELECT count(*) FROM entities").fetchone()[0] == 1
    assert outcomes.count("created") == 1
    assert "gave-up" not in outcomes
    db.close()


# ==================================================================== the 2 -> 3 migration

def test_migration_merges_duplicate_entities_and_repoints_edges(tmp_path):
    con = v2_file(tmp_path / "merge.anatid")
    # Four rows for one name in tenant 1 -- exactly what the reported race produced.
    add_entity(con, 10, 1, "Ada Lovelace")
    add_entity(con, 11, 1, "ada lovelace")
    add_entity(con, 12, 1, "  Ada  Lovelace ")
    add_entity(con, 13, 1, "ADA LOVELACE")
    add_entity(con, 20, 1, "Grace Hopper")           # innocent bystander
    add_entity(con, 30, 2, "Ada Lovelace")           # another tenant, must survive intact
    add_memory(con, 100, 1, "Ada likes coffee")
    add_memory(con, 101, 1, "Ada likes tea")
    # ABOUT edges spread over the duplicates, one of which already points at the winner.
    for edge_id, src, dst in [(1, 100, 10), (2, 100, 11), (3, 101, 12), (4, 101, 13), (5, 101, 20)]:
        con.execute("INSERT INTO edges_about (edge_id, src, dst, tenant_id, weight, valid_from, "
                    "tx_from) VALUES (?, ?, ?, 1, 1.0, ?, ?)", [edge_id, src, dst, T0, T0])
    con.execute("INSERT INTO edges_about (edge_id, src, dst, tenant_id, weight, valid_from, "
                "tx_from) VALUES (9, 100, 30, 2, 1.0, ?, ?)", [T0, T0])
    # RELATES edges: one between two of the losers (a self-loop after the merge, and the user's
    # edge, so it is KEPT), two from different losers to the bystander that become verbatim
    # copies of each other (collapsed to the older one), and one with a different rel_kind that
    # must survive alongside them.
    con.execute("INSERT INTO edges_relates (edge_id, src, dst, tenant_id, rel_kind, valid_from, "
                "tx_from) VALUES (6, 11, 12, 1, 'same_as', ?, ?)", [T0, T0])
    con.execute("INSERT INTO edges_relates (edge_id, src, dst, tenant_id, rel_kind, valid_from, "
                "tx_from) VALUES (7, 13, 20, 1, 'knows', ?, ?)", [T0, T0])
    con.execute("INSERT INTO edges_relates (edge_id, src, dst, tenant_id, rel_kind, valid_from, "
                "tx_from) VALUES (8, 12, 20, 1, 'knows', ?, ?)", [T0, T0])
    con.execute("INSERT INTO edges_relates (edge_id, src, dst, tenant_id, rel_kind, valid_from, "
                "tx_from) VALUES (9, 10, 20, 1, 'mentors', ?, ?)", [T0, T0])

    assert S.ensure_schema(con) == 3

    ids = [r[0] for r in con.execute(
        "SELECT entity_id FROM entities WHERE tenant_id = 1 ORDER BY entity_id").fetchall()]
    assert ids == [10, 20], "lowest entity_id wins, the other three are gone"
    assert con.execute("SELECT entity_id, name FROM entities WHERE tenant_id = 2"
                       ).fetchall() == [(30, "Ada Lovelace")]

    # Every ABOUT edge that pointed at a loser now points at the winner ...
    about = sorted(con.execute("SELECT src, dst FROM edges_about WHERE tenant_id = 1").fetchall())
    assert about == [(100, 10), (101, 10), (101, 20)], about
    # ... exactly once: (100, 10) and (100, 11) collapsed into one edge, not two identical ones.
    assert con.execute("SELECT count(*) FROM edges_about WHERE tenant_id = 1 AND src = 100"
                       ).fetchone()[0] == 1
    assert con.execute("SELECT dst FROM edges_about WHERE tenant_id = 2").fetchall() == [(30,)]

    relates = sorted(con.execute("SELECT edge_id, src, dst, rel_kind FROM edges_relates").fetchall())
    assert relates == [(6, 10, 10, "same_as"),     # 11->12: repointed, kept as a self-loop
                       (7, 10, 20, "knows"),       # 13->20 and 12->20 collapsed onto the older
                       (9, 10, 20, "mentors")], relates   # different rel_kind: not a duplicate

    # The constraint is live afterwards.
    with pytest.raises(duckdb.ConstraintException):
        add_entity(con, 99, 1, "ADA  lovelace")
    con.close()


def test_migration_keeps_the_edge_that_already_pointed_at_the_winner(tmp_path):
    """The verifier's case: a loser's edge carries a LOWER edge_id than the original.

    After repointing, the two edges are verbatim copies.  "Older" cannot mean "lower id" --
    ids can be hand-assigned -- and it cannot mean tx_from alone either, because both edges
    here were recorded at the same instant.  The rule is: an edge the merge did not touch (it
    already pointed at the winner) survives a repointed copy of itself, whatever their ids;
    between two repointed copies the earlier tx_from wins, then the lower id.  A pair the merge
    never touched is left alone and reported by doctor() as duplicate_live_edges.
    """
    con = v2_file(tmp_path / "edge-order.anatid")
    add_entity(con, 10, 1, "Ada")
    add_entity(con, 11, 1, "ada")                    # the loser
    add_entity(con, 20, 1, "Kestrel")
    add_entity(con, 30, 1, "Bo")
    add_memory(con, 100, 1, "Ada leads Kestrel")
    later = T0 + _dt.timedelta(hours=1)
    # columns: edge_id, src, dst, rel_kind, tx_from
    rows = [
        (883, 10, 20, "leads", T0),                   # original: winner -> Kestrel, HIGH id
        (800, 11, 20, "leads", T0),                   # loser's copy with a LOWER id, same instant
        (700, 11, 30, "knows", later),                # two repointed copies: the one recorded
        (600, 11, 30, "knows", T0),                   #   first (600, at T0) survives, 700 goes
        (500, 10, 30, "mentors", later),              # untouched pair: both survive
        (400, 10, 30, "mentors", T0),
    ]
    for edge_id, src, dst, kind, tx in rows:
        con.execute("INSERT INTO edges_relates (edge_id, src, dst, tenant_id, rel_kind, "
                    "valid_from, tx_from) VALUES (?, ?, ?, 1, ?, ?, ?)",
                    [edge_id, src, dst, kind, T0, tx])
    con.execute("INSERT INTO edges_about (edge_id, src, dst, tenant_id, weight, valid_from, "
                "tx_from) VALUES (9, 100, 10, 1, 1.0, ?, ?)", [T0, T0])
    con.execute("INSERT INTO edges_about (edge_id, src, dst, tenant_id, weight, valid_from, "
                "tx_from) VALUES (1, 100, 11, 1, 1.0, ?, ?)", [T0, T0])     # lower id, loser

    assert S.ensure_schema(con) == 3

    relates = sorted(con.execute(
        "SELECT edge_id, src, dst, rel_kind FROM edges_relates ORDER BY edge_id").fetchall())
    assert relates == [
        (400, 10, 30, "mentors"), (500, 10, 30, "mentors"),   # untouched pair, both kept
        (600, 10, 30, "knows"),                               # earlier of the two repointed
        (883, 10, 20, "leads"),                               # the original, despite its id
    ], relates
    about = con.execute("SELECT edge_id, src, dst FROM edges_about").fetchall()
    assert about == [(9, 100, 10)], about
    con.close()


def test_migration_preserves_entity_columns_and_values(tmp_path):
    con = v2_file(tmp_path / "cols.anatid")
    con.execute(
        "INSERT INTO entities (entity_id, tenant_id, kind, name, valid_from, valid_to, tx_from, "
        "tx_to, writer, episode_id, confidence) VALUES (1, 7, 'org', 'ACME', ?, NULL, ?, NULL, "
        "'alice', 42, 0.5)", [T0, T0])
    before = con.execute(f"SELECT {', '.join(S.ENTITY_COLUMNS)} FROM entities").fetchall()

    S.ensure_schema(con)

    after = con.execute(f"SELECT {', '.join(S.ENTITY_COLUMNS)} FROM entities").fetchall()
    assert after == before
    info = {r[1]: (r[2], r[3]) for r in con.execute("PRAGMA table_info(entities)").fetchall()}
    assert info["entity_id"] == ("BIGINT", True), "NOT NULL survived the rebuild"
    assert info["tenant_id"] == ("INTEGER", True)
    assert S.ENTITY_KEY_COLUMN in info
    assert S.insertable_columns(con, "entities") == list(S.ENTITY_COLUMNS)
    con.close()


def test_migration_is_idempotent_and_reopening_is_a_no_op(tmp_path):
    path = tmp_path / "twice.anatid"
    con = v2_file(path)
    add_entity(con, 1, 1, "Ada")
    add_entity(con, 2, 1, "ada")
    assert S.ensure_schema(con) == 3
    assert S.ensure_schema(con) == 3
    con.close()
    again = duckdb.connect(str(path))
    assert S.ensure_schema(again) == 3
    assert again.execute("SELECT count(*) FROM entities").fetchone()[0] == 1
    again.close()


def test_migration_from_v1_runs_both_steps(tmp_path):
    con = v2_file(tmp_path / "v1.anatid", version=1)
    con.execute("INSERT INTO anatid_audit (audit_id, tenant_id, memory_id, action, reason, "
                "happened_at) VALUES (1, 1, 5, 'supersede', 'superseded by 6', ?)", [T0])
    add_entity(con, 1, 1, "Ada")
    add_entity(con, 2, 1, "ADA")
    assert S.ensure_schema(con) == 3
    assert con.execute("SELECT related_memory_id, reason FROM anatid_audit").fetchone() == (
        6, "superseded")
    assert con.execute("SELECT count(*) FROM entities").fetchone()[0] == 1
    assert S.missing_tables(con) == []
    con.close()


def test_migration_rolls_back_as_one_transaction(tmp_path, monkeypatch):
    """A migration that fails half way must leave the v2 file exactly as it was."""
    con = v2_file(tmp_path / "rollback.anatid")
    add_entity(con, 1, 1, "Ada")
    add_entity(con, 2, 1, "ada")

    def boom(_con):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(S, "_rebuild_entities_with_key", boom)
    with pytest.raises(RuntimeError):
        S.ensure_schema(con)

    assert con.execute("SELECT schema_version FROM anatid_meta").fetchone()[0] == 2
    assert con.execute("SELECT count(*) FROM entities").fetchone()[0] == 2
    assert S.ENTITY_KEY_COLUMN not in {
        r[1] for r in con.execute("PRAGMA table_info(entities)").fetchall()}
    con.close()


def test_a_v3_file_is_not_downgraded(tmp_path):
    con = duckdb.connect(str(tmp_path / "future.anatid"))
    S.ensure_schema(con)
    con.execute("UPDATE anatid_meta SET schema_version = 99")
    with pytest.raises(Exception) as excinfo:
        S.ensure_schema(con)
    assert "never downgrades" in str(excinfo.value)
    con.close()


# ============================================================ the surface other code depends on

def test_fresh_database_has_every_table_and_the_required_index(tmp_path):
    con = duckdb.connect(str(tmp_path / "fresh.anatid"))
    assert S.ensure_schema(con) == S.SCHEMA_VERSION == 3
    assert S.missing_tables(con) == []
    indexes = {r[0] for r in con.execute(
        "SELECT index_name FROM duckdb_indexes()").fetchall()}
    assert "ux_entities_tenant_key" in indexes
    for table in S.FTS_TABLES:
        assert con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    con.close()


def test_required_indexes_are_not_selectable_away(tmp_path):
    """A caller may switch off tuning indexes; it may not switch off a constraint."""
    con = duckdb.connect(str(tmp_path / "noindex.anatid"))
    S.ensure_schema(con, S.SchemaConfig(embedding_dim=8, indexes=()))
    add_entity(con, 1, 1, "Ada")
    with pytest.raises(duckdb.ConstraintException):
        add_entity(con, 2, 1, "ada")
    con.close()


def test_user_defined_labels_and_types_still_work(tmp_path):
    con = duckdb.connect(str(tmp_path / "labels.anatid"))
    S.ensure_schema(con)
    con.execute(S.node_table_ddl("project", [("title", "VARCHAR"), ("budget", "DECIMAL(18, 3)")]))
    con.execute(S.edge_table_ddl("owns", [("share", "FLOAT")]))
    cols = [r[1] for r in con.execute("PRAGMA table_info(project)").fetchall()]
    assert cols[:4] == ["project_id", "tenant_id", "title", "budget"]
    assert "confidence" in cols
    with pytest.raises(ValueError):
        S.check_type("INTEGER); DROP TABLE memories; CREATE TABLE zz (a INTEGER")
    with pytest.raises(ValueError):
        S.node_table_ddl("t", [("x", "INTEGER, y INTEGER")])
    con.close()


def test_table_ddl_and_insertable_columns_agree_for_a_rebuild(tmp_path):
    """The shape ``Anatid.recluster()`` relies on: build from table_ddl, fill from
    insertable_columns.  Naming the generated column in the INSERT is an error, which is why
    recluster must not take its column list straight from PRAGMA table_info."""
    con = duckdb.connect(str(tmp_path / "rebuild.anatid"))
    S.ensure_schema(con, S.SchemaConfig(embedding_dim=8))
    add_entity(con, 1, 1, "Ada")
    cols = ", ".join(S.quote_ident(c) for c in S.insertable_columns(con, "entities"))
    con.execute(S.table_ddl("entities", S.SchemaConfig(embedding_dim=8), as_table="entities__t"))
    con.execute(f"INSERT INTO entities__t ({cols}) SELECT {cols} FROM entities "
                f"ORDER BY {S.CLUSTER_ORDER['entities']}")
    assert con.execute("SELECT entity_key FROM entities__t").fetchone()[0] == "ada"

    all_cols = ", ".join(S.quote_ident(r[1]) for r in
                         con.execute("PRAGMA table_info(entities)").fetchall())
    with pytest.raises(duckdb.BinderException):
        con.execute(f"INSERT INTO entities__t ({all_cols}) SELECT {all_cols} FROM entities")
    con.close()


def test_contract_notes_are_stamped_into_the_file(tmp_path):
    con = duckdb.connect(str(tmp_path / "contract.anatid"))
    S.ensure_schema(con)
    contract = con.execute("SELECT contract FROM anatid_meta").fetchone()[0]
    assert "'<tenant_id>:<memory_id>'" in contract
    assert "PER TENANT" in contract
    assert "entity_key" in contract
    con.close()


def test_migration_refreshes_the_stored_contract(tmp_path):
    con = v2_file(tmp_path / "oldcontract.anatid")
    assert con.execute("SELECT contract FROM anatid_meta").fetchone()[0] == "v2 contract"
    S.ensure_schema(con)
    assert "PER TENANT" in con.execute("SELECT contract FROM anatid_meta").fetchone()[0]
    con.close()
