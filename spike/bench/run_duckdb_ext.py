#!/usr/bin/env python3
"""anatid Phase 0 spike -- engine "duckdb_ext" (Layer B).

Same DuckDB tables as Layer A (duckdb_sql/schema.sql, mirrored below, not imported), but R1's 2-hop
frontier comes from the C++ `anatid` extension: `anatid_build_csr('edges_relates')` builds an
in-memory undirected CSR (per tenant, current edges only) that lives in the DatabaseInstance's
ObjectCache, and `graph_expand(tenant_id, seed, 2)` is a table function doing a BFS over it.
R1 is then plain SQL: memories ABOUT any frontier entity, tenant + validity filters, ORDER BY
created_at DESC, memory_id DESC LIMIT n.

Usage (always with the spike's interpreter):
    .venv/bin/python spike/bench/run_duckdb_ext.py --scale small
    .venv/bin/python spike/bench/run_duckdb_ext.py --scale full [--phases warmup,r1_only,...] [--threads N]

`load` always runs (the DB file results/anatid_ext.duckdb is deleted and rebuilt); --phases selects
which of warmup,r1_only,r2_only,mixed,concurrent,verify run afterwards (default: all, in SPEC order).
Writes results/duckdb_ext.<scale>.json via common.write_result.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import duckdb  # noqa: E402

ENGINE = "duckdb_ext"
EXT_PATH = common.SPIKE_DIR / "extension" / "build" / "release" / "extension" / "anatid" / "anatid.duckdb_extension"
DB_PATH = common.RESULTS_DIR / "anatid_ext.duckdb"
ALL_PHASES = ["warmup", "r1_only", "r2_only", "mixed", "concurrent", "verify"]

# ----------------------------------------------------------------------------- schema (mirrors duckdb_sql/schema.sql)

SCHEMA = [
    """CREATE TABLE entities (
        entity_id BIGINT NOT NULL, tenant_id INTEGER NOT NULL, kind VARCHAR, name VARCHAR)""",
    """CREATE TABLE memories (
        memory_id BIGINT NOT NULL, tenant_id INTEGER NOT NULL, content VARCHAR, kind VARCHAR,
        embedding FLOAT[64], created_at TIMESTAMP NOT NULL, valid_from TIMESTAMP, valid_to TIMESTAMP,
        tx_from TIMESTAMP, tx_to TIMESTAMP, writer VARCHAR, episode_id BIGINT, confidence FLOAT)""",
    """CREATE TABLE edges_about (
        edge_id BIGINT NOT NULL, src BIGINT NOT NULL, dst BIGINT NOT NULL, tenant_id INTEGER NOT NULL,
        weight FLOAT, valid_from TIMESTAMP, valid_to TIMESTAMP, tx_from TIMESTAMP, tx_to TIMESTAMP)""",
    """CREATE TABLE edges_relates (
        edge_id BIGINT NOT NULL, src BIGINT NOT NULL, dst BIGINT NOT NULL, tenant_id INTEGER NOT NULL,
        rel_kind VARCHAR, valid_from TIMESTAMP, valid_to TIMESTAMP, tx_from TIMESTAMP, tx_to TIMESTAMP)""",
    """CREATE TABLE edges_supersedes (
        edge_id BIGINT NOT NULL, src BIGINT NOT NULL, dst BIGINT NOT NULL, tenant_id INTEGER NOT NULL,
        tx_from TIMESTAMP)""",
]

# Bulk load. memories and edges_about are clustered by tenant on the way in (ORDER BY): R1's plan is
# two SEMI hash joins whose probe sides are sequential scans of memories / edges_about with the
# tenant filter, and DuckDB's zone maps then skip the other tenants' row groups. Semantics are
# unchanged (same rows, same columns); only the physical order differs from a plain INSERT..SELECT.
LOAD = [
    ("entities", "INSERT INTO entities SELECT entity_id, tenant_id, kind, name FROM read_parquet('{d}/entities.parquet')"),
    ("memories", "INSERT INTO memories SELECT memory_id, tenant_id, content, kind, embedding::FLOAT[64], created_at, "
                 "valid_from, valid_to, tx_from, tx_to, writer, episode_id, confidence "
                 "FROM read_parquet('{d}/memories.parquet') ORDER BY tenant_id, memory_id"),
    ("edges_about", "INSERT INTO edges_about SELECT edge_id, src, dst, tenant_id, weight, valid_from, valid_to, tx_from, tx_to "
                    "FROM read_parquet('{d}/edges_about.parquet') ORDER BY tenant_id, dst"),
    ("edges_relates", "INSERT INTO edges_relates SELECT edge_id, src, dst, tenant_id, rel_kind, valid_from, valid_to, tx_from, tx_to "
                      "FROM read_parquet('{d}/edges_relates.parquet')"),
    ("edges_supersedes", "INSERT INTO edges_supersedes SELECT edge_id, src, dst, tenant_id, tx_from "
                         "FROM read_parquet('{d}/edges_supersedes.parquet')"),
]

# ----------------------------------------------------------------------------- operations

def r1_sql(tenant_id: int, seed_entity_id: int, limit: int) -> str:
    """R1 recall_2hop. graph_expand's arguments are constants (the BFS runs at bind time and the
    optimizer sees the exact frontier size), hence literals instead of prepared-statement parameters."""
    t, s, n = int(tenant_id), int(seed_entity_id), int(limit)
    return (
        "SELECT m.memory_id, m.created_at FROM memories m "
        f"WHERE m.tenant_id = {t} AND m.valid_to IS NULL "
        f"AND m.memory_id IN (SELECT a.src FROM edges_about a WHERE a.tenant_id = {t} "
        f"AND a.dst IN (SELECT entity_id FROM graph_expand({t}, {s}, 2))) "
        f"ORDER BY m.created_at DESC, m.memory_id DESC LIMIT {n}"
    )


def run_r1(con, tenant_id: int, seed_entity_id: int, limit: int = common.R1_LIMIT) -> list:
    return con.execute(r1_sql(tenant_id, seed_entity_id, limit)).fetchall()


def emb_literal(embedding) -> str:
    """Embeddings travel as a string literal cast to FLOAT[64], not as bound parameters: duckdb-python
    1.5.5 needs ~5 ms (GIL held) to convert a 64-element list/ndarray parameter, and a bare list
    literal [0.1, ...] is parsed as DECIMAL(18,17) and loses precision; '[...]'::FLOAT[64] binds in
    ~0.15 ms and round-trips the float32 values exactly (checked at small scale, max |diff| = 0)."""
    return "'[" + ",".join(repr(float(x)) for x in embedding) + "]'::FLOAT[64]"


# R2 candidate lists. $1 is always tenant_id.
R2_COSINE_TPL = (
    "SELECT memory_id, array_cosine_similarity(embedding, {emb}) AS s "
    "FROM memories WHERE tenant_id = $1 AND valid_to IS NULL "
    f"ORDER BY s DESC, memory_id ASC LIMIT {common.R2_TOPN}"
)
# BM25 (Okapi, k1/b from the SPEC, idf = ln((N - df + 0.5)/(df + 0.5) + 1), corpus stats over all
# indexed memories) over bm25_postings, a (termid, memory_id, tf, len) table derived at load from the
# fts extension's index tables (fts_main_memories.terms/docs) and clustered by termid so the IN
# filter prunes row groups. Query terms come from the fts tokenizer + dict (see bm25_terms()).
BM25_TERMS_SQL = ("SELECT termid, df FROM fts_main_memories.dict "
                  "WHERE term IN (SELECT unnest(fts_main_memories.tokenize($1)))")


def bm25_terms(con, query_text: str) -> list[tuple[int, int]]:
    return [(int(t), int(df)) for t, df in con.execute(BM25_TERMS_SQL, [str(query_text)]).fetchall()]


def bm25_ctes(terms: list[tuple[int, int]]) -> str:
    """CTEs q(termid, df), st(num_docs, avgdl) and bm(memory_id, s) = BM25 top-50 of the tenant."""
    if terms:
        q = "VALUES " + ", ".join(f"({t}, {df})" for t, df in terms)
        inlist = ", ".join(str(t) for t, _ in terms)
    else:  # no query term is in the corpus -> empty BM25 list
        q, inlist = "SELECT 0::BIGINT, 0::BIGINT WHERE false", "-1"
    return (
        f"q(termid, df) AS ({q}), st AS (SELECT num_docs, avgdl FROM fts_main_memories.stats), "
        "bm AS (SELECT m.memory_id, sc.s FROM ("
        f"  SELECT p.memory_id, sum(ln((st.num_docs - q.df + 0.5) / (q.df + 0.5) + 1) * p.tf * ({common.BM25_K1} + 1) "
        f"      / (p.tf + {common.BM25_K1} * (1 - {common.BM25_B} + {common.BM25_B} * p.len / st.avgdl))) AS s "
        f"  FROM bm25_postings p JOIN q USING (termid), st WHERE p.termid IN ({inlist}) GROUP BY p.memory_id) sc "
        "  JOIN memories m ON m.memory_id = sc.memory_id WHERE m.tenant_id = $1 AND m.valid_to IS NULL "
        f"  ORDER BY sc.s DESC, m.memory_id ASC LIMIT {common.R2_TOPN})"
    )


def r2_sql(k: int, query_embedding, terms: list[tuple[int, int]]) -> str:
    """R2 recall_hybrid: brute-force cosine top-50 + BM25 top-50, RRF (k=60, 1-based ranks), top k with
    the ABOUT entity names (1-hop context). $1 tenant_id; the embedding and term ids are inlined."""
    return (
        f"WITH cos AS ({R2_COSINE_TPL.format(emb=emb_literal(query_embedding))}), {bm25_ctes(terms)}, "
        "ranked AS ("
        "  SELECT memory_id, row_number() OVER (ORDER BY s DESC, memory_id ASC) AS rnk FROM cos "
        "  UNION ALL "
        "  SELECT memory_id, row_number() OVER (ORDER BY s DESC, memory_id ASC) AS rnk FROM bm), "
        "fused AS ("
        f"  SELECT memory_id, sum(1.0 / ({common.RRF_K} + rnk)) AS rrf_score FROM ranked GROUP BY memory_id "
        f"  ORDER BY rrf_score DESC, memory_id ASC LIMIT {int(k)}) "
        "SELECT f.memory_id, f.rrf_score, "
        "  list(e.name ORDER BY e.entity_id) FILTER (WHERE e.entity_id IS NOT NULL) AS about_names "
        "FROM fused f LEFT JOIN edges_about a ON a.src = f.memory_id LEFT JOIN entities e ON e.entity_id = a.dst "
        "GROUP BY f.memory_id, f.rrf_score ORDER BY f.rrf_score DESC, f.memory_id ASC"
    )


def run_r2(con, tenant_id: int, query_text: str, query_embedding: list, k: int = common.R2_K) -> list:
    """Two statements per op: term lookup in the fts dictionary, then the fused query."""
    terms = bm25_terms(con, query_text)
    return con.execute(r2_sql(k, query_embedding, terms), [int(tenant_id)]).fetchall()


def run_cosine_top(con, tenant_id: int, query_embedding) -> list:
    return con.execute(R2_COSINE_TPL.format(emb=emb_literal(query_embedding)), [int(tenant_id)]).fetchall()


def run_bm25_top(con, tenant_id: int, query_text: str) -> list:
    terms = bm25_terms(con, query_text)
    return con.execute(f"WITH {bm25_ctes(terms)} SELECT memory_id, s FROM bm", [int(tenant_id)]).fetchall()


INSERT_MEMORY = "INSERT INTO memories VALUES (?, ?, ?, ?, {emb}, ?, ?, ?, ?, ?, ?, ?, ?)"
ABOUT_ROW = "(?, ?, ?, ?, ?, ?, ?, ?, ?)"
INSERT_SUPERSEDES = "INSERT INTO edges_supersedes VALUES (?, ?, ?, ?, ?)"
UPDATE_OLD = "UPDATE memories SET valid_to = ? WHERE memory_id = ?"


def _insert_memory(con, m: dict) -> None:
    con.execute(INSERT_MEMORY.format(emb=emb_literal(m["embedding"])),
                [m["memory_id"], m["tenant_id"], m["content"], m["kind"], m["created_at"], m["valid_from"],
                 m["valid_to"], m["tx_from"], m["tx_to"], m["writer"], m["episode_id"], m["confidence"]])


def _in_transaction(con, body) -> None:
    con.begin()
    try:
        body()
        con.commit()
    except Exception:
        try:
            con.rollback()
        except Exception:  # noqa: BLE001 -- a failed COMMIT already rolled back
            pass
        raise


def run_w1(con, op: dict) -> None:
    """remember: one transaction, insert the memory + one ABOUT edge per dst."""
    m, now = op["memory"], op["now"]

    def body():
        _insert_memory(con, m)
        rows = [[eid, m["memory_id"], int(dst), op["tenant_id"], 1.0, now, None, now, None]
                for dst, eid in zip(op["about_dsts"], op["about_edge_ids"])]
        con.execute("INSERT INTO edges_about VALUES " + ", ".join([ABOUT_ROW] * len(rows)),
                    [v for r in rows for v in r])

    _in_transaction(con, body)


def run_w2(con, op: dict) -> None:
    """supersede: one transaction, insert the new memory, close the old one, add the SUPERSEDES edge."""
    m, now = op["memory"], op["now"]

    def body():
        _insert_memory(con, m)
        con.execute(UPDATE_OLD, [now, op["old_memory_id"]])
        con.execute(INSERT_SUPERSEDES, [op["supersedes_edge_id"], m["memory_id"], op["old_memory_id"], op["tenant_id"], now])

    _in_transaction(con, body)


def rebuild_csr(con, phase: str) -> dict:
    t0 = time.perf_counter()
    tenants, vertices, edges, build_ms = con.execute("SELECT * FROM anatid_build_csr('edges_relates')").fetchone()
    return {"phase": phase, "tenants": int(tenants), "vertices": int(vertices), "edges": int(edges),
            "build_ms": float(build_ms), "wall_ms": (time.perf_counter() - t0) * 1000.0}


# ----------------------------------------------------------------------------- phases

def connect(threads: int | None):
    con = duckdb.connect(str(DB_PATH), config={"allow_unsigned_extensions": "true"})
    if threads is not None:
        con.execute(f"SET threads = {int(threads)}")
    con.execute(f"LOAD '{EXT_PATH}'")
    con.execute("INSTALL fts")
    con.execute("LOAD fts")
    return con


def phase_load(con, scale: str, result: dict) -> dict:
    d = str(common.data_dir(scale))
    steps: dict[str, float] = {}
    with common.Stopwatch() as sw:
        for ddl in SCHEMA:
            con.execute(ddl)
        for name, sql in LOAD:
            t0 = time.perf_counter()
            con.execute(sql.format(d=d))
            steps[f"insert_{name}"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        con.execute("CREATE INDEX idx_memories_id ON memories (memory_id)")
        steps["art_index_memories_id"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        con.execute("PRAGMA create_fts_index('memories', 'memory_id', 'content', "
                    "stemmer = 'none', stopwords = 'none', lower = 1, strip_accents = 0, overwrite = 1)")
        steps["fts_index"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        con.execute("CREATE TABLE bm25_postings AS "
                    "SELECT t.termid, d.name AS memory_id, count(*)::DOUBLE AS tf, d.len::DOUBLE AS len "
                    "FROM fts_main_memories.terms t JOIN fts_main_memories.docs d ON d.docid = t.docid "
                    "GROUP BY t.termid, d.name, d.len ORDER BY t.termid, d.name")
        steps["bm25_postings"] = time.perf_counter() - t0
        csr = rebuild_csr(con, "load")
        steps["csr_build"] = csr["wall_ms"] / 1000.0
        t0 = time.perf_counter()
        con.execute("CHECKPOINT")
        steps["checkpoint"] = time.perf_counter() - t0
    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
              for t in ("entities", "memories", "edges_about", "edges_relates", "edges_supersedes")}
    result["load"] = {
        "seconds": sw.seconds,
        "db_bytes": common.db_bytes(DB_PATH),
        "index_notes": (
            f"ART index memories(memory_id) {steps['art_index_memories_id']:.1f}s (for W2's point UPDATE); "
            f"fts index on memories.content (stemmer none, stopwords none, lower, default ignore regex) {steps['fts_index']:.1f}s "
            f"+ bm25_postings(termid, memory_id, tf, len) derived from it, clustered by termid, {steps['bm25_postings']:.1f}s, "
            f"both NOT incremental; in-memory CSR over current edges_relates via anatid_build_csr: "
            f"{csr['tenants']} tenants, {csr['vertices']} vertices, {csr['edges']} edges, {csr['build_ms']:.1f} ms "
            f"(rebuilt at the start of every phase); memories clustered ORDER BY (tenant_id, memory_id), "
            f"edges_about ORDER BY (tenant_id, dst) on load; no other indexes"
        ),
        "steps_s": {k: round(v, 3) for k, v in steps.items()},
        "row_counts": counts,
    }
    return csr


def phase_warmup(con, scale: str) -> None:
    qs = common.load_queries(scale)
    for qid in common.WARMUP_QUERY_IDS:
        q = qs[qid]
        run_r1(con, q["tenant_id"], q["seed_entity_id"])


def phase_r1_only(con, scale: str, result: dict) -> dict:
    qs = common.load_queries(scale)
    timer = common.Timer()
    got: dict[int, list] = {}
    with common.Stopwatch() as sw:
        for qid in common.R1_QUERY_IDS:
            q = qs[qid]
            with timer:
                rows = run_r1(con, q["tenant_id"], q["seed_entity_id"], common.R1_LIMIT)
            got[qid] = rows
    phase = timer.summary(wall_s=sw.seconds)
    # authoritative check, after the timed loop so the reference build does not perturb the numbers
    mismatches = []
    tenants = set()
    for qid, rows in got.items():
        q = qs[qid]
        tenants.add(q["tenant_id"])
        ok, msg = common.compare_r1(rows, common.reference_r1(scale, q["tenant_id"], q["seed_entity_id"], common.R1_LIMIT))
        if not ok:
            mismatches.append((qid, msg))
    phase["reference_checked"] = len(got)
    phase["reference_mismatches"] = len(mismatches)
    phase["reference_tenants_covered"] = len(tenants)
    result["notes"].append(
        f"r1_only: {len(got)} R1 results compared with common.reference_r1 across {len(tenants)} tenants: "
        f"{len(mismatches)} mismatches" + (f" (first: {mismatches[0]})" if mismatches else ""))
    return phase


def phase_r2_only(con, scale: str, result: dict) -> dict:
    qs = common.load_queries(scale)
    timer = common.Timer()
    got: dict[int, list] = {}
    with common.Stopwatch() as sw:
        for qid in common.R2_QUERY_IDS:
            q = qs[qid]
            with timer:
                rows = run_r2(con, q["tenant_id"], q["query_text"], q["query_embedding"], common.R2_K)
            got[qid] = rows
    phase = timer.summary(wall_s=sw.seconds)
    # recall vs brute-force truth (untimed); also the two candidate lists separately, for diagnosis
    rec, rec_cos, rec_bm, exact = [], [], [], 0
    for qid, rows in got.items():
        q = qs[qid]
        truth = common.reference_r2_truth(scale, q)
        ids = [int(r[0]) for r in rows]
        rec.append(common.recall_at_k(truth["rrf_ids"], ids, common.R2_K))
        exact += int(ids == truth["rrf_ids"])
        cos_ids = [r[0] for r in run_cosine_top(con, q["tenant_id"], q["query_embedding"])]
        bm_ids = [r[0] for r in run_bm25_top(con, q["tenant_id"], q["query_text"])]
        rec_cos.append(common.recall_at_k(truth["cosine_ids"], cos_ids, common.R2_TOPN))
        rec_bm.append(common.recall_at_k(truth["bm25_ids"], bm_ids, common.R2_TOPN))
    n = max(len(rec), 1)
    phase["recall_at_20"] = sum(rec) / n
    phase["exact_top20_lists"] = exact
    phase["cosine_recall_at_50"] = sum(rec_cos) / n
    phase["bm25_recall_at_50"] = sum(rec_bm) / n
    phase["index"] = ("cosine: brute force array_cosine_similarity over the tenant's current rows (no ANN); "
                      "BM25: bm25_postings table derived from the fts extension index at load, clustered by termid (not incremental)")
    result["notes"].append(
        f"r2_only: recall@20 vs brute-force RRF truth {phase['recall_at_20']:.4f} ({exact}/{len(got)} lists identical); "
        f"cosine top-50 recall {phase['cosine_recall_at_50']:.4f}, BM25 (postings) top-50 recall {phase['bm25_recall_at_50']:.4f}")
    return phase


def phase_mixed(con, scale: str, result: dict) -> dict:
    ops = common.schedule(scale)
    timers = {k: common.Timer() for k in ("W1", "W2", "R1", "R2")}
    with common.Stopwatch() as sw:
        for op in ops:
            kind = op["op"]
            if kind == "W1":
                with timers[kind]:
                    run_w1(con, op)
            elif kind == "W2":
                with timers[kind]:
                    run_w2(con, op)
            elif kind == "R1":
                with timers[kind]:
                    run_r1(con, op["tenant_id"], op["seed_entity_id"], op["limit"])
            else:
                with timers[kind]:
                    run_r2(con, op["tenant_id"], op["query_text"], op["query_embedding"], op["k"])
    phase = {k: t.summary() for k, t in timers.items()}
    phase["wall_s"] = sw.seconds
    phase["ops"] = len(ops)
    # write-effect spot check (untimed): the last W1 memories must be visible to R1 through their
    # ABOUT edges (each is newer than every base row and at most 9 later W1s exist), the old memory
    # of every one of the last W2 ops must be closed.
    w1_ops = [op for op in ops if op["op"] == "W1"][-10:]
    w2_ops = [op for op in ops if op["op"] == "W2"][-10:]
    w1_seen = 0
    for op in w1_ops:
        rows = run_r1(con, op["tenant_id"], op["about_dsts"][0], common.R1_LIMIT)
        w1_seen += int(any(int(r[0]) == op["memory"]["memory_id"] for r in rows))
    w2_closed = 0
    for op in w2_ops:
        closed = con.execute("SELECT valid_to FROM memories WHERE memory_id = ?", [op["old_memory_id"]]).fetchone()
        w2_closed += int(closed is not None and closed[0] == op["now"])
    phase["write_spot_check"] = {"w1_checked": len(w1_ops), "w1_visible_in_r1": w1_seen,
                                 "w2_checked": len(w2_ops), "w2_old_closed": w2_closed}
    result["notes"].append(
        f"mixed: write spot check -- {w1_seen}/{len(w1_ops)} of the last W1 memories visible in R1 via their first "
        f"ABOUT dst, {w2_closed}/{len(w2_ops)} of the last W2 old memories have valid_to = now")
    return phase


def phase_concurrent(db_con, scale: str, threads: int | None) -> dict:
    # One W1 op stream per writer index, shared by the baseline sub-runs and the main run so every
    # inserted memory_id / edge_id stays unique (fresh generators would restart the id sequence).
    w1_gens = {idx: common.concurrent_w1_ops(scale, idx) for idx in range(common.N_WRITERS)}

    def writer_factory(idx: int):
        con = db_con.cursor()  # own connection to the same database
        gen = w1_gens[idx]

        def step():
            run_w1(con, next(gen))

        return step

    def reader_factory(idx: int):
        con = db_con.cursor()
        gen = common.concurrent_r1_queries(scale, idx)

        def step():
            op = next(gen)
            run_r1(con, op["tenant_id"], op["seed_entity_id"], op["limit"])

        return step

    # short baselines so the main number can be interpreted: writers alone, 1 then 4 threads
    base_1w = common.run_concurrent(3.0, writer_factory, reader_factory, 1, 0)
    base_4w = common.run_concurrent(3.0, writer_factory, reader_factory, common.N_WRITERS, 0)
    res = common.run_concurrent(common.CONCURRENT_SECONDS[scale], writer_factory, reader_factory,
                                common.N_WRITERS, common.N_READERS)
    res["baseline_1_writer_only"] = {k: base_1w[k] for k in ("W1_ops_per_s", "W1_p50_ms", "W1_p95_ms", "errors", "seconds")}
    res["baseline_4_writers_only"] = {k: base_4w[k] for k in ("W1_ops_per_s", "W1_p50_ms", "W1_p95_ms", "errors", "seconds")}
    res["notes"] = (
        f"DuckDB optimistic MVCC, one cursor() per thread on the same open database, every W1 is one "
        f"BEGIN..COMMIT, no retries (a failed transaction is rolled back and counted as an error); "
        f"engine threads={threads if threads is not None else 'default (all cores)'}; "
        f"R1 uses the CSR snapshot (concurrent W1 adds no RELATES_TO edges, so it stays exact); "
        f"baselines (3 s each, writers only): 1 writer {base_1w['W1_ops_per_s']:.0f} W1/s "
        f"(p50 {base_1w['W1_p50_ms']:.1f} ms), 4 writers {base_4w['W1_ops_per_s']:.0f} W1/s (p50 {base_4w['W1_p50_ms']:.1f} ms); "
        f"main run 4 writers + 2 readers: W1 p50 {res['W1_p50_ms']:.1f} ms, R1 p50 {res['R1_p50_ms']:.1f} ms; "
        f"W1 errors {res['W1_errors']}, R1 errors {res['R1_errors']}, "
        f"threads with zero ops: W1 {res['W1_threads_with_zero_ops']}, R1 {res['R1_threads_with_zero_ops']}"
    )
    return res


def phase_verify(con, scale: str, result: dict, after_mixed: bool) -> dict:
    qs = common.load_queries(scale)
    got: dict[int, list] = {}
    for qid in common.VERIFY_QUERY_IDS:
        q = qs[qid]
        got[qid] = run_r1(con, q["tenant_id"], q["seed_entity_id"], common.R1_LIMIT)
    mismatches = []
    for qid, rows in got.items():
        q = qs[qid]
        ok, msg = common.compare_r1(rows, common.reference_r1(scale, q["tenant_id"], q["seed_entity_id"],
                                                              common.R1_LIMIT, after_mixed=after_mixed))
        if not ok:
            mismatches.append((qid, msg))
    result["notes"].append(
        f"verify: {len(got)} R1 results compared with common.reference_r1(after_mixed={after_mixed}): "
        f"{len(mismatches)} mismatches" + (f" (first: {mismatches[0]})" if mismatches else ""))
    result["checks"]["verify_reference_mismatches"] = len(mismatches)
    return common.verify_payload(got)


# ----------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=common.SCALES, required=True)
    ap.add_argument("--phases", default=",".join(ALL_PHASES),
                    help="comma-separated subset of " + ",".join(ALL_PHASES) + " (load always runs)")
    ap.add_argument("--threads", type=int, default=None, help="SET threads for the engine (default: DuckDB default)")
    args = ap.parse_args()
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    bad = [p for p in phases if p not in ALL_PHASES]
    if bad:
        ap.error(f"unknown phases {bad}; choose from {ALL_PHASES}")
    phases = [p for p in ALL_PHASES if p in phases]  # SPEC order
    scale = args.scale

    if not EXT_PATH.exists():
        print(f"extension not built: {EXT_PATH}\n  cd spike/extension && PATH=.venv/bin:$PATH GEN=ninja make release",
              file=sys.stderr)
        return 2

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for f in common.RESULTS_DIR.glob(DB_PATH.name + "*"):  # previous DB file + .wal
        f.unlink()

    con = connect(args.threads)
    ext_version = con.execute("SELECT anatid_version('')").fetchone()[0].split(":")[0]
    threads = con.execute("SELECT current_setting('threads')").fetchone()[0]
    result = common.result_skeleton(ENGINE, f"duckdb {duckdb.__version__} + {ext_version}", scale)
    result["checks"] = {}
    result["csr_rebuilds"] = []
    result["notes"] += [
        f"engine threads = {threads} (DuckDB default = all cores unless --threads given); Python client single-threaded except the concurrent phase",
        "R1 = SELECT memories m WHERE tenant AND valid_to IS NULL AND m.memory_id IN (SELECT src FROM edges_about WHERE tenant AND dst IN "
        "(SELECT entity_id FROM graph_expand(t, seed, 2))) ORDER BY created_at DESC, memory_id DESC LIMIT n; plan = two SEMI hash joins "
        "(graph_expand -> edges_about -> memories) with dynamic min/max + bloom filters pushed into the two sequential scans",
        "graph_expand is a C++ table function: BFS (visited bitmap) over an undirected per-tenant CSR held in the DatabaseInstance ObjectCache; "
        "the BFS runs at bind time (arguments are literals) so the optimizer sees the exact frontier cardinality; results stream in 2048-row chunks",
        "DESIGN LIMITATION: the CSR is a snapshot of edges_relates (valid_to IS NULL) taken by anatid_build_csr(); RELATES_TO writes are not "
        "reflected until the next rebuild. The SPEC workload never writes RELATES_TO (W1/W2 add ABOUT / SUPERSEDES edges only), so R1 stays exact; "
        "the runner still rebuilds the CSR at the start of every phase and records rebuild_ms (see csr_rebuilds)",
        "R2 = 2 statements per op: (1) query terms -> (termid, df) via the fts tokenizer + fts_main_memories.dict, (2) one fused statement: "
        "cosine top-50 (brute force, FLOAT) UNION BM25 top-50 computed in SQL over bm25_postings (Okapi k1=1.2 b=0.75, idf=ln((N-df+0.5)/(df+0.5)+1), "
        "N/avgdl from fts_main_memories.stats), RRF k=60, top-k, ABOUT entity names via LEFT JOIN edges_about/entities + list(). The fts "
        "match_bm25 macro was measured too (small: 7.6 ms vs 3.2 ms for the postings table, identical top-50 lists; it uses log10 so scores are "
        "scaled by 1/ln(10) but ranking is identical)",
        "fts index + bm25_postings are built once at load and are NOT incremental: memories inserted by W1/W2 are not BM25 candidates in "
        "mixed-phase R2 (the cosine side sees them); superseded memories are excluded by the valid_to filter on the memories join",
        "embeddings (W1/W2 rows and R2 query vectors) are sent as '[...]'::FLOAT[64] string literals: a bound list parameter costs ~5 ms in "
        "duckdb-python 1.5.5 and a bare list literal is parsed as DECIMAL(18,17) (lossy); the string form is exact and binds in ~0.15 ms",
        "load clusters memories by (tenant_id, memory_id) and edges_about by (tenant_id, dst) so zone maps skip other tenants' row groups; "
        "small-scale alternatives measured (200 R1 queries, all 0 mismatches): natural order p50 1.25 ms vs clustered 1.03 ms (10 threads); "
        "join+DISTINCT formulation 1.27 ms vs semi-join 1.03 ms; ART index on edges_about(dst) not used by the plan (1.13 ms)",
        f"extension binary: {EXT_PATH.relative_to(common.SPIKE_DIR)} (built from spike/extension against duckdb v1.5.5, loaded with allow_unsigned_extensions)",
    ] + list(common.DATASET_NOTES)

    print(f"[{ENGINE}/{scale}] load ...", flush=True)
    try:
        csr = phase_load(con, scale, result)
        result["csr_rebuilds"].append(csr)
        print(f"  load {result['load']['seconds']:.1f}s, db {result['load']['db_bytes']/1e6:.1f} MB, "
              f"steps {result['load']['steps_s']}, csr {csr}", flush=True)
    except Exception as e:  # noqa: BLE001
        result["load"] = {"seconds": None, "db_bytes": None, "index_notes": "", **common.phase_error(e)}
        result["notes"].append(f"load FAILED: {type(e).__name__}: {e}")
        print(f"  load FAILED: {e}", flush=True)
        common.write_result(ENGINE, scale, result)
        return 1

    for phase in phases:
        print(f"[{ENGINE}/{scale}] {phase} ...", flush=True)
        try:
            result["csr_rebuilds"].append(rebuild_csr(con, phase))  # SPEC: refresh the snapshot, record the cost
            if phase == "warmup":
                with common.Stopwatch() as sw:
                    phase_warmup(con, scale)
                result["checks"]["warmup_s"] = sw.seconds
                print(f"  warmup {sw.seconds:.2f}s", flush=True)
            elif phase == "r1_only":
                ph = phase_r1_only(con, scale, result)
                result["phases"]["r1_only"] = ph
                print(f"  r1_only p50 {ph['p50_ms']:.3f} ms p95 {ph['p95_ms']:.3f} ms, "
                      f"reference mismatches {ph['reference_mismatches']}/{ph['reference_checked']}", flush=True)
            elif phase == "r2_only":
                ph = phase_r2_only(con, scale, result)
                result["phases"]["r2_only"] = ph
                print(f"  r2_only p50 {ph['p50_ms']:.3f} ms p95 {ph['p95_ms']:.3f} ms, recall@20 {ph['recall_at_20']:.4f}", flush=True)
            elif phase == "mixed":
                ph = phase_mixed(con, scale, result)
                result["phases"]["mixed"] = ph
                print("  mixed " + ", ".join(f"{k} n={ph[k]['count']} p50={ph[k]['p50_ms']:.3f}" for k in ("W1", "W2", "R1", "R2"))
                      + f", wall {ph['wall_s']:.1f}s, spot check {ph['write_spot_check']}", flush=True)
            elif phase == "concurrent":
                ph = phase_concurrent(con, scale, args.threads)
                result["phases"]["concurrent"] = ph
                print(f"  concurrent W1 {ph['W1_ops_per_s']:.1f} ops/s, R1 {ph['R1_ops_per_s']:.1f} ops/s, errors {ph['errors']}", flush=True)
            elif phase == "verify":
                result["verify"] = phase_verify(con, scale, result, after_mixed="mixed" in phases)
                print(f"  verify mismatches vs reference: {result['checks']['verify_reference_mismatches']}", flush=True)
        except Exception as e:  # noqa: BLE001
            err = common.phase_error(e)
            if phase in ("r1_only", "r2_only", "mixed", "concurrent"):
                result["phases"][phase] = err
            else:
                result["checks"][f"{phase}_error"] = err
            result["notes"].append(f"{phase} FAILED: {err['error']}")
            print(f"  {phase} FAILED: {err['error']}", flush=True)

    result["notes"].append("csr rebuilds (phase: build_ms): " + ", ".join(
        f"{c['phase']}: {c['build_ms']:.1f}" for c in result["csr_rebuilds"]))
    result["peak_rss_mb"] = common.peak_rss_mb()
    con.execute("CHECKPOINT")
    con.close()
    path = common.write_result(ENGINE, scale, result)
    print(f"[{ENGINE}/{scale}] wrote {path}; peak RSS {result['peak_rss_mb']:.0f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
