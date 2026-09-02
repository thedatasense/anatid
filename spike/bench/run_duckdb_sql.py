#!/usr/bin/env python
"""anatid Phase 0 spike -- engine "duckdb_sql" (Layer A).

DuckDB 1.5.5 with plain tables (duckdb_sql/schema.sql) and SQL macros (duckdb_sql/macros.sql),
no C++. Implements every phase of SPEC.md's workload schedule (load, warmup, r1_only, r2_only,
mixed, concurrent, verify) and writes results/duckdb_sql.<scale>.json via common.write_result.

    .venv/bin/python bench/run_duckdb_sql.py --scale small
    .venv/bin/python bench/run_duckdb_sql.py --scale full

Options (defaults are the measured-best configuration, see the result JSON "notes"):
    --phases a,b,c      subset of load,warmup,r1_only,r2_only,mixed,concurrent,verify (default all;
                        load always runs first; later phases see whatever state earlier ones left)
    --threads auto|N    DuckDB thread count for the query phases (auto = pick the faster of
                        1 and the DuckDB default by measuring R1, both recorded in notes)
    --expand auto|join|rec   graph_expand formulation for R1 (auto = pick the faster, record both)
    --bm25 macro|match  BM25 path for R2: bm25_top (fts tables read directly) or the fts
                        extension's match_bm25 macro (both give identical rankings)
    --indexes a,b       ART indexes to create (default idx_memories_id; '' for none)
    --no-cluster        load tables in Parquet order instead of clustered by tenant
    --no-ab             skip the A/B pre-measurements (variants, threads, indexes, bm25 path)
    --fts-terms-index auto|on|off   ART index on fts_main_memories.terms(termid) for BM25
                        (auto = build it, keep it only if the A/B says it is faster; recorded)
    --no-fts-rebuild    skip timing the fts index rebuild after the phases
    --db PATH           database file (default results/anatid.duckdb, deleted before load)
    --results-dir DIR   where <engine>.<scale>.json goes (default spike/results)

Correctness: every r1_only result is compared with common.reference_r1 (ids and created_at),
every r2_only result with common.reference_r2_truth (recall@20), the verify lists with
common.reference_r1(after_mixed=True); the counts land in the result JSON notes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import duckdb  # noqa: E402

ENGINE = "duckdb_sql"
SQL_DIR = common.SPIKE_DIR / "duckdb_sql"
DEFAULT_DB = common.RESULTS_DIR / "anatid.duckdb"
ALL_PHASES = ["load", "warmup", "r1_only", "r2_only", "mixed", "concurrent", "verify"]
ALL_INDEXES = ["idx_memories_id", "idx_relates_tenant_src", "idx_relates_tenant_dst", "idx_about_dst"]
DEFAULT_INDEXES = ["idx_memories_id"]
MEM_COLS = ["memory_id", "tenant_id", "content", "kind", "embedding", "created_at", "valid_from",
            "valid_to", "tx_from", "tx_to", "writer", "episode_id", "confidence"]
AB_R1_QIDS = list(range(1300, 1400))          # A/B pre-measurements use mixed-phase query ids,
AB_R2_QIDS = list(range(1300, 1350))          # never the r1_only / r2_only ids
FTS_INDEX_SQL = (r"PRAGMA create_fts_index('memories', 'memory_id', 'content', stemmer='none', "
                 r"stopwords='none', ignore='(\.|[^a-z])+', strip_accents=0, lower=1, overwrite=1)")
FTS_TERMS_INDEX_SQL = "CREATE INDEX idx_fts_terms_termid ON fts_main_memories.terms(termid)"

# W1 / W2 statements (documented in duckdb_sql/macros.sql). The embedding is bound as the text
# '[f1,...,f64]' and cast in SQL: binding a 64-element Python list costs ~4.5 ms in duckdb-python.
SQL_INSERT_MEMORY = ("INSERT INTO memories VALUES (?, ?, ?, ?, ?::FLOAT[64], ?, ?, ?, ?, ?, ?, ?, ?)")
SQL_INSERT_ABOUT = ("INSERT INTO edges_about "
                    "SELECT eid, ?, dst, ?, 1.0, ?, NULL, ?, NULL "
                    "FROM (SELECT unnest(?::BIGINT[]) AS eid, unnest(?::BIGINT[]) AS dst)")
SQL_UPDATE_OLD = "UPDATE memories SET valid_to = ? WHERE memory_id = ?"
SQL_INSERT_SUPERSEDES = "INSERT INTO edges_supersedes VALUES (?, ?, ?, ?, ?)"


# ----------------------------------------------------------------------------- helpers

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def strip_comment(line: str) -> str:
    """Remove a trailing '-- comment' unless the '--' sits inside a quoted literal."""
    i = line.find("--")
    while i >= 0:
        if line[:i].count("'") % 2 == 0:
            return line[:i]
        i = line.find("--", i + 2)
    return line


def split_sql(text: str) -> list[str]:
    lines = [strip_comment(l) for l in text.splitlines()]
    body = "\n".join(lines)
    return [s.strip() for s in body.split(";") if s.strip()]


def emb_str(vec) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def pct(xs_ms: list[float], p: float) -> float:
    xs = sorted(xs_ms)
    return xs[min(len(xs) - 1, max(0, int(round(p / 100.0 * len(xs) + 0.5)) - 1))]


def summ(xs_ms: list[float]) -> str:
    return f"p50={statistics.median(xs_ms):.3f} p95={pct(xs_ms, 95):.3f} mean={statistics.fmean(xs_ms):.3f} ms (n={len(xs_ms)})"


# ----------------------------------------------------------------------------- engine

class DuckSQL:
    def __init__(self, scale: str, db_path: Path, cluster: bool, indexes: list[str], bm25: str):
        self.scale = scale
        self.db_path = Path(db_path)
        self.cluster = cluster
        self.indexes = list(indexes)
        self.bm25 = bm25
        self.expand = "join"
        self.fts_terms_index = False
        self.default_threads: int | None = None
        self.threads: int | None = None
        self.con: duckdb.DuckDBPyConnection | None = None

    # --- lifecycle
    def open(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        for f in self.db_path.parent.glob(self.db_path.name + "*"):
            if f.is_file():
                f.unlink()
        self.con = duckdb.connect(str(self.db_path))
        self.con.execute("INSTALL fts; LOAD fts;")
        self.default_threads = int(self.con.execute("SELECT current_setting('threads')").fetchone()[0])
        self.threads = self.default_threads

    def set_threads(self, n: int) -> None:
        self.con.execute(f"SET threads={int(n)}")
        self.threads = int(n)

    def close(self) -> None:
        if self.con is not None:
            self.con.close()
            self.con = None

    # --- load
    def schema_statements(self) -> list[str]:
        text = (SQL_DIR / "schema.sql").read_text().replace("@DATA_DIR@", str(common.data_dir(self.scale)))
        if not self.cluster:
            text = "\n".join((";" if "@cluster" in l else l) for l in text.splitlines())
        return split_sql(text)

    def load(self) -> dict:
        con = self.con
        t0 = time.perf_counter()
        index_stmts = {}
        for st in self.schema_statements():
            if st.upper().startswith("CREATE INDEX"):
                index_stmts[st.split()[2]] = st
                continue
            con.execute(st)
        t_tables = time.perf_counter() - t0
        t1 = time.perf_counter()
        for name in self.indexes:
            con.execute(index_stmts[name])
        t_idx = time.perf_counter() - t1
        t2 = time.perf_counter()
        con.execute(FTS_INDEX_SQL)
        t_fts = time.perf_counter() - t2
        t3 = time.perf_counter()
        con.execute("CHECKPOINT")
        t_ckpt = time.perf_counter() - t3
        total = time.perf_counter() - t0
        self.index_stmts = index_stmts
        counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                  for t in ("entities", "memories", "edges_about", "edges_relates", "edges_supersedes")}
        return {"seconds": total, "tables_s": t_tables, "art_index_s": t_idx, "fts_index_s": t_fts,
                "checkpoint_s": t_ckpt, "counts": counts}

    def create_macros(self) -> None:
        for st in split_sql((SQL_DIR / "macros.sql").read_text()):
            self.con.execute(st)

    # --- R1
    def prepare_r1(self, con, expand: str | None = None) -> None:
        macro = {"join": "recall_2hop", "rec": "recall_2hop_rec"}[expand or self.expand]
        try:
            con.execute("DEALLOCATE r1")
        except duckdb.Error:
            pass
        con.execute(f"PREPARE r1 AS SELECT memory_id, created_at FROM {macro}($1, $2, $3)")

    @staticmethod
    def r1(con, tenant_id: int, seed_entity_id: int, limit: int = common.R1_LIMIT) -> list:
        # EXECUTE does not accept '?' parameters; the three arguments are integers from the dataset.
        return con.execute(f"EXECUTE r1({int(tenant_id)}, {int(seed_entity_id)}, {int(limit)})").fetchall()

    # --- R2
    def r2(self, con, tenant_id: int, query_text: str, query_embedding, k: int = common.R2_K, bm25: str | None = None) -> list:
        macro = {"macro": "recall_hybrid", "match": "recall_hybrid_match"}[bm25 or self.bm25]
        # k inside LIMIT cannot be a bound parameter of a table macro, it is an int literal.
        return con.execute(
            f"SELECT memory_id, rrf_score, about_names FROM {macro}(?, ?, ?::FLOAT[64], {int(k)})",
            [int(tenant_id), query_text, emb_str(query_embedding)]).fetchall()

    # --- W1 / W2
    @staticmethod
    def _memory_params(m: dict) -> list:
        vals = [m[c] for c in MEM_COLS]
        vals[4] = emb_str(m["embedding"])
        return vals

    def w1(self, con, op: dict) -> None:
        m = op["memory"]
        now = op["now"]
        con.execute("BEGIN")
        try:
            con.execute(SQL_INSERT_MEMORY, self._memory_params(m))
            con.execute(SQL_INSERT_ABOUT, [m["memory_id"], op["tenant_id"], now, now,
                                           list(op["about_edge_ids"]), list(op["about_dsts"])])
            con.execute("COMMIT")
        except Exception:
            try:
                con.execute("ROLLBACK")
            except duckdb.Error:
                pass
            raise

    def w2(self, con, op: dict) -> None:
        m = op["memory"]
        now = op["now"]
        con.execute("BEGIN")
        try:
            con.execute(SQL_INSERT_MEMORY, self._memory_params(m))
            con.execute(SQL_UPDATE_OLD, [now, op["old_memory_id"]])
            con.execute(SQL_INSERT_SUPERSEDES, [op["supersedes_edge_id"], m["memory_id"], op["old_memory_id"],
                                                op["tenant_id"], now])
            con.execute("COMMIT")
        except Exception:
            try:
                con.execute("ROLLBACK")
            except duckdb.Error:
                pass
            raise


# ----------------------------------------------------------------------------- A/B pre-measurements

def time_r1(eng: DuckSQL, queries: list[dict], qids, warm: int = 20) -> list[float]:
    for qid in list(qids)[:warm]:
        eng.r1(eng.con, queries[qid]["tenant_id"], queries[qid]["seed_entity_id"])
    out = []
    for qid in qids:
        q = queries[qid]
        t0 = time.perf_counter_ns()
        eng.r1(eng.con, q["tenant_id"], q["seed_entity_id"])
        out.append((time.perf_counter_ns() - t0) / 1e6)
    return out


def time_r2(eng: DuckSQL, queries: list[dict], qids, bm25: str, warm: int = 5) -> list[float]:
    for qid in list(qids)[:warm]:
        q = queries[qid]
        eng.r2(eng.con, q["tenant_id"], q["query_text"], q["query_embedding"], bm25=bm25)
    out = []
    for qid in qids:
        q = queries[qid]
        t0 = time.perf_counter_ns()
        eng.r2(eng.con, q["tenant_id"], q["query_text"], q["query_embedding"], bm25=bm25)
        out.append((time.perf_counter_ns() - t0) / 1e6)
    return out


def ab_r1(eng: DuckSQL, queries: list[dict], threads_opt: str, expand_opt: str, notes: list[str]) -> None:
    """Measure R1 for expand variant x thread count, pick the fastest p50 (unless fixed by options)."""
    thread_choices = [eng.default_threads, 1] if threads_opt == "auto" else [int(threads_opt)]
    if len(thread_choices) > 1 and thread_choices[0] == 1:
        thread_choices = [1]
    results = {}
    for th in thread_choices:
        eng.set_threads(th)
        for variant in ("join", "rec"):
            eng.prepare_r1(eng.con, variant)
            xs = time_r1(eng, queries, AB_R1_QIDS)
            results[(th, variant)] = xs
            notes.append(f"A/B R1 graph_expand={variant} threads={th}: {summ(xs)}")
            log(f"  A/B R1 expand={variant} threads={th}: {summ(xs)}")
    best_th, best_variant = min(results, key=lambda k: statistics.median(results[k]))
    if expand_opt != "auto":
        best_variant = expand_opt
    eng.set_threads(best_th)
    eng.expand = best_variant
    eng.prepare_r1(eng.con)
    notes.append(f"chosen for the timed phases: graph_expand={'two self-join levels, UNION ALL' if best_variant == 'join' else 'recursive CTE USING KEY'} "
                 f"({best_variant}), SET threads={best_th} (DuckDB default here is {eng.default_threads})")


def ab_indexes(eng: DuckSQL, queries: list[dict], notes: list[str]) -> None:
    """R1 and point-UPDATE latency with the configured ART indexes vs none (indexes are re-created after)."""
    con = eng.con
    n_mem = con.execute("SELECT count(*) FROM memories").fetchone()[0]

    def update_micro(n: int = 40) -> list[float]:
        xs = []
        con.execute("BEGIN")
        for i in range(n):
            mid = (i * 7919 + 13) % n_mem
            t0 = time.perf_counter_ns()
            con.execute(SQL_UPDATE_OLD, [dt.datetime(2030, 1, 1), mid])
            xs.append((time.perf_counter_ns() - t0) / 1e6)
        con.execute("ROLLBACK")
        return xs

    with_idx_r1 = time_r1(eng, queries, AB_R1_QIDS)
    with_idx_upd = update_micro()
    for name in eng.indexes:
        con.execute(f"DROP INDEX {name}")
    no_idx_r1 = time_r1(eng, queries, AB_R1_QIDS)
    no_idx_upd = update_micro()
    all_r1 = None
    if set(eng.indexes) != set(ALL_INDEXES):
        for name in ALL_INDEXES:
            con.execute(eng.index_stmts[name])
        all_r1 = time_r1(eng, queries, AB_R1_QIDS)
        for name in ALL_INDEXES:
            con.execute(f"DROP INDEX {name}")
    t0 = time.perf_counter()
    for name in eng.indexes:
        con.execute(eng.index_stmts[name])
    rebuild_s = time.perf_counter() - t0
    idx_desc = ", ".join(eng.indexes) if eng.indexes else "none"
    notes.append(f"A/B ART indexes: R1 with [{idx_desc}] {summ(with_idx_r1)}; R1 with no index {summ(no_idx_r1)}"
                 + (f"; R1 with all four ({', '.join(ALL_INDEXES)}) {summ(all_r1)}" if all_r1 else ""))
    notes.append(f"A/B ART indexes: point UPDATE memories SET valid_to WHERE memory_id=? (W2 step) with [{idx_desc}] "
                 f"{summ(with_idx_upd)}; with no index {summ(no_idx_upd)}; re-creating [{idx_desc}] took {rebuild_s:.2f} s")
    log(f"  A/B indexes: R1 with={statistics.median(with_idx_r1):.3f} none={statistics.median(no_idx_r1):.3f}"
        + (f" all4={statistics.median(all_r1):.3f}" if all_r1 else "")
        + f" | UPDATE with={statistics.median(with_idx_upd):.3f} none={statistics.median(no_idx_upd):.3f} ms")


def create_fts_terms_index(eng: DuckSQL) -> float:
    t0 = time.perf_counter()
    eng.con.execute(FTS_TERMS_INDEX_SQL)
    eng.fts_terms_index = True
    return time.perf_counter() - t0


def ab_r2(eng: DuckSQL, queries: list[dict], threads_opt: str, fts_terms_opt: str, notes: list[str]) -> None:
    chosen = eng.threads
    base = {}
    for bm25 in ("macro", "match"):
        xs = time_r2(eng, queries, AB_R2_QIDS, bm25)
        base[bm25] = xs
        notes.append(f"A/B R2 bm25={'bm25_top (fts tables read directly)' if bm25 == 'macro' else 'fts match_bm25 macro'} "
                     f"threads={chosen}, no ART index on fts terms: {summ(xs)}")
        log(f"  A/B R2 bm25={bm25} threads={chosen}: {summ(xs)}")
    if fts_terms_opt != "off":
        # ART index on the fts posting table: DuckDB 1.5 turns the dict->terms hash join into index
        # lookups at execution time (visible as 'Index Scan' in the profile, not in EXPLAIN).
        build_s = create_fts_terms_index(eng)
        with_idx = {bm25: time_r2(eng, queries, AB_R2_QIDS, bm25) for bm25 in ("macro", "match")}
        keep = fts_terms_opt == "on" or statistics.median(with_idx[eng.bm25]) < statistics.median(base[eng.bm25])
        if not keep:
            eng.con.execute("DROP INDEX fts_main_memories.idx_fts_terms_termid")
            eng.fts_terms_index = False
        notes.append(f"A/B ART index on fts_main_memories.terms(termid) (build {build_s:.1f} s): R2 with it bm25_top {summ(with_idx['macro'])}, "
                     f"match_bm25 {summ(with_idx['match'])}; without it bm25_top {summ(base['macro'])}, match_bm25 {summ(base['match'])} "
                     f"-> {'kept' if keep else 'dropped'} for the timed phases")
        log(f"  A/B fts terms index: macro with={statistics.median(with_idx['macro']):.2f} without={statistics.median(base['macro']):.2f} ms -> {'kept' if keep else 'dropped'}")
    if threads_opt == "auto" and eng.default_threads != 1:
        other = 1 if chosen != 1 else eng.default_threads
        eng.set_threads(other)
        xs = time_r2(eng, queries, AB_R2_QIDS, eng.bm25)
        notes.append(f"A/B R2 bm25={eng.bm25} threads={other}: {summ(xs)} (timed phases use threads={chosen}, chosen by R1)")
        log(f"  A/B R2 bm25={eng.bm25} threads={other}: {summ(xs)}")
        eng.set_threads(chosen)


# ----------------------------------------------------------------------------- phases

def phase_r1_only(eng: DuckSQL, queries: list[dict], scale: str, notes: list[str]) -> dict:
    timer = common.Timer()
    got = {}
    with common.Stopwatch() as sw:
        for qid in common.R1_QUERY_IDS:
            q = queries[qid]
            with timer:
                got[qid] = eng.r1(eng.con, q["tenant_id"], q["seed_entity_id"], common.R1_LIMIT)
    out = timer.summary(wall_s=sw.seconds)
    out["peak_rss_mb_before_reference_check"] = common.peak_rss_mb()
    bad = []
    for qid, rows in got.items():
        q = queries[qid]
        ok, msg = common.compare_r1(rows, common.reference_r1(scale, q["tenant_id"], q["seed_entity_id"], common.R1_LIMIT))
        if not ok:
            bad.append((qid, msg))
    out["reference_mismatches"] = len(bad)
    out["reference_checked"] = len(got)
    notes.append(f"r1_only: {len(got)} R1 results compared with common.reference_r1 (ids + created_at): {len(bad)} mismatches"
                 + (f"; first: {bad[:3]}" if bad else ""))
    return out


def phase_r2_only(eng: DuckSQL, queries: list[dict], scale: str, notes: list[str]) -> dict:
    timer = common.Timer()
    got = {}
    with common.Stopwatch() as sw:
        for qid in common.R2_QUERY_IDS:
            q = queries[qid]
            with timer:
                got[qid] = eng.r2(eng.con, q["tenant_id"], q["query_text"], q["query_embedding"], common.R2_K)
    out = timer.summary(wall_s=sw.seconds)
    recalls, exact = [], 0
    for qid, rows in got.items():
        truth = common.reference_r2_truth(scale, queries[qid])
        ids = [r[0] for r in rows]
        recalls.append(common.recall_at_k(truth["rrf_ids"], ids))
        exact += ids == truth["rrf_ids"]
    out["recall_at_20"] = statistics.fmean(recalls) if recalls else None
    out["recall_at_20_min"] = min(recalls) if recalls else None
    out["exact_top20_lists"] = f"{exact}/{len(got)}"
    out["vector_index"] = "none (brute-force array_cosine_similarity over the tenant's current memories)"
    out["bm25"] = (("fts extension inverted index, BM25 k1=1.2 b=0.75 read from the fts_main_memories tables"
                    if eng.bm25 == "macro" else "fts extension match_bm25 macro, k1=1.2 b=0.75")
                   + (" + ART index on fts_main_memories.terms(termid)" if eng.fts_terms_index else "")
                   + "; NOT incremental (stale after writes)")
    notes.append(f"r2_only: recall@20 vs common.reference_r2_truth over all {len(got)} queries: mean {out['recall_at_20']:.4f}, "
                 f"min {out['recall_at_20_min']:.3f}, exact top-20 lists {exact}/{len(got)}")
    return out


def phase_mixed(eng: DuckSQL, scale: str, notes: list[str]) -> dict:
    ops = common.schedule(scale)
    timers = {k: common.Timer() for k in ("W1", "W2", "R1", "R2")}
    con = eng.con
    with common.Stopwatch() as sw:
        for op in ops:
            k = op["op"]
            if k == "W1":
                with timers["W1"]:
                    eng.w1(con, op)
            elif k == "W2":
                with timers["W2"]:
                    eng.w2(con, op)
            elif k == "R1":
                with timers["R1"]:
                    eng.r1(con, op["tenant_id"], op["seed_entity_id"], op["limit"])
            else:
                with timers["R2"]:
                    eng.r2(con, op["tenant_id"], op["query_text"], op["query_embedding"], op["k"])
    out = {k: t.summary(wall_s=None) for k, t in timers.items()}
    out["wall_s"] = sw.seconds
    out["ops"] = len(ops)
    out["ops_per_s"] = len(ops) / sw.seconds if sw.seconds > 0 else None
    # spot check: the first W1 memory must be visible to R1 from one of its ABOUT entities
    first = next(op for op in ops if op["op"] == "W1")
    rows = con.execute("SELECT memory_id FROM recall_2hop(?, ?, ?)",
                       [first["tenant_id"], first["about_dsts"][0], 10_000_000]).fetchall()
    visible = first["memory"]["memory_id"] in {r[0] for r in rows}
    first_w2 = next((op for op in ops if op["op"] == "W2"), None)
    superseded = None
    if first_w2 is not None:
        superseded = con.execute("SELECT valid_to FROM memories WHERE memory_id = ?", [first_w2["old_memory_id"]]).fetchone()[0] == first_w2["now"]
    stale = con.execute("SELECT count(*) FROM memories m WHERE NOT EXISTS (SELECT 1 FROM fts_main_memories.docs d WHERE d.name = m.memory_id)").fetchone()[0]
    notes.append(f"mixed: {len(ops)} ops in {sw.seconds:.1f} s; first W1 memory visible to R1 via its ABOUT entity: {visible}; "
                 f"first W2 old memory has valid_to = now: {superseded}; fts index stale: {stale} memories inserted by the mixed phase "
                 f"are invisible to BM25 (mixed-phase R2 used the stale index, cosine side is live)")
    out["fts_stale_memories"] = stale
    if not visible or superseded is False:
        notes.append("WARNING: mixed-phase write visibility check FAILED")
    return out


def phase_concurrent(eng: DuckSQL, scale: str, notes: list[str]) -> dict:
    base_con = eng.con

    def writer_factory(idx: int):
        c = base_con.cursor()
        it = common.concurrent_w1_ops(scale, idx)

        def step():
            eng.w1(c, next(it))
        return step

    def reader_factory(idx: int):
        c = base_con.cursor()
        eng.prepare_r1(c)
        it = common.concurrent_r1_queries(scale, idx)

        def step():
            op = next(it)
            return eng.r1(c, op["tenant_id"], op["seed_entity_id"], op["limit"])
        return step

    n_before = base_con.execute("SELECT count(*) FROM memories").fetchone()[0]
    res = common.run_concurrent(common.CONCURRENT_SECONDS[scale], writer_factory, reader_factory)
    n_after = base_con.execute("SELECT count(*) FROM memories").fetchone()[0]
    res["memories_inserted"] = n_after - n_before
    res["notes"] = (f"{common.N_WRITERS} writer + {common.N_READERS} reader threads, each on its own connection (duckdb cursor()) to the same file; "
                    f"DuckDB optimistic MVCC, one W1 = one BEGIN/COMMIT transaction, no retries implemented, "
                    f"{res['W1_errors']} write errors / {res['R1_errors']} read errors; {n_after - n_before} memories committed "
                    f"({res['W1_ops']} W1 ops counted); SET threads={eng.threads}")
    notes.append(f"concurrent: W1 {res['W1_ops_per_s']:.1f} ops/s (p50 {res['W1_p50_ms']:.2f} ms), R1 {res['R1_ops_per_s']:.1f} ops/s "
                 f"(p50 {res['R1_p50_ms']:.2f} ms), errors {res['errors']}; {res['notes']}")
    return res


def phase_verify(eng: DuckSQL, queries: list[dict], scale: str, mixed_ran: bool, notes: list[str]) -> dict:
    got = {}
    for qid in common.VERIFY_QUERY_IDS:
        q = queries[qid]
        got[qid] = eng.r1(eng.con, q["tenant_id"], q["seed_entity_id"], common.R1_LIMIT)
    bad = 0
    for qid, rows in got.items():
        q = queries[qid]
        ok, _ = common.compare_r1(rows, common.reference_r1(scale, q["tenant_id"], q["seed_entity_id"], common.R1_LIMIT, after_mixed=mixed_ran))
        bad += (not ok)
    notes.append(f"verify: {len(got)} R1 id lists dumped; compared with common.reference_r1(after_mixed={mixed_ran}): {bad} mismatches")
    return common.verify_payload(got)


# ----------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=common.SCALES, required=True)
    ap.add_argument("--phases", default=",".join(ALL_PHASES))
    ap.add_argument("--threads", default="auto")
    ap.add_argument("--expand", default="auto", choices=["auto", "join", "rec"])
    ap.add_argument("--bm25", default="macro", choices=["macro", "match"])
    ap.add_argument("--fts-terms-index", default="auto", choices=["auto", "on", "off"],
                    help="ART index on fts_main_memories.terms(termid) for BM25 (auto = keep if the A/B says it is faster)")
    ap.add_argument("--indexes", default=",".join(DEFAULT_INDEXES))
    ap.add_argument("--no-cluster", action="store_true")
    ap.add_argument("--no-ab", action="store_true")
    ap.add_argument("--no-fts-rebuild", action="store_true")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--results-dir", default=str(common.RESULTS_DIR), help="where <engine>.<scale>.json is written (default spike/results)")
    args = ap.parse_args()
    results_dir = Path(args.results_dir)

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    unknown = [p for p in phases if p not in ALL_PHASES]
    if unknown:
        ap.error(f"unknown phases {unknown}; choose from {ALL_PHASES}")
    indexes = [x.strip() for x in args.indexes.split(",") if x.strip()]
    bad_idx = [x for x in indexes if x not in ALL_INDEXES]
    if bad_idx:
        ap.error(f"unknown indexes {bad_idx}; choose from {ALL_INDEXES}")

    scale = args.scale
    result = common.result_skeleton(ENGINE, duckdb.__version__, scale)
    notes: list[str] = result["notes"]
    notes.append(f"config: duckdb {duckdb.__version__} python {sys.version.split()[0]}; db={args.db}; "
                 f"clustered load={'yes (memories ORDER BY tenant_id, created_at DESC, memory_id DESC; edges_about ORDER BY tenant_id, dst, src; edges_relates ORDER BY tenant_id, src)' if not args.no_cluster else 'no (Parquet order)'}; "
                 f"ART indexes={indexes or 'none'}; bm25 path={args.bm25}; phases={phases}")
    notes.append("implementation: R1 = PREPARE'd `SELECT memory_id, created_at FROM recall_2hop($1,$2,$3)` executed with EXECUTE r1(t, seed, 20) "
                 "(table macro, frontier via IN (...) semi joins, no ART index used by the planner for joins); "
                 "R2 = recall_hybrid table macro (brute-force cosine top-50 UNION ALL fts BM25 top-50, RRF k=60 in SQL, ABOUT entity names via list()); "
                 "W1/W2 = parameterized INSERT/UPDATE statements inside explicit BEGIN/COMMIT (DuckDB macros cannot contain DML, see macros.sql); "
                 "embeddings are bound as '[...]' text and cast to FLOAT[64] in SQL (a 64-float Python list parameter costs ~4.5 ms to bind in duckdb-python)")

    def save() -> None:
        result["peak_rss_mb"] = common.peak_rss_mb()
        common.write_result(ENGINE, scale, result, results_dir=results_dir)

    def loadavg() -> str:
        try:
            return "/".join(f"{x:.1f}" for x in os.getloadavg())
        except OSError:
            return "n/a"

    notes.append(f"host load average (1/5/15 min) at start: {loadavg()} on {os.cpu_count()} cpus "
                 "(SPEC: numbers are only comparable if nothing else was running)")

    eng = DuckSQL(scale, Path(args.db), cluster=not args.no_cluster, indexes=indexes, bm25=args.bm25)
    queries = common.load_queries(scale)
    mixed_ran = False

    # ---- load (always)
    log(f"load {scale} -> {args.db}")
    try:
        eng.open()
        ld = eng.load()
        result["load"] = {
            "seconds": ld["seconds"],
            "db_bytes": common.db_bytes(eng.db_path),
            "index_notes": (f"tables {ld['tables_s']:.1f} s (read_parquet, {'clustered by tenant' if eng.cluster else 'Parquet order'}), "
                            f"ART {', '.join(indexes) if indexes else 'none'} {ld['art_index_s']:.1f} s, "
                            f"fts index on memories(content) {ld['fts_index_s']:.1f} s (PRAGMA create_fts_index, not incremental), "
                            f"CHECKPOINT {ld['checkpoint_s']:.1f} s; rows {ld['counts']}; load used the DuckDB default of {eng.default_threads} threads"),
        }
        eng.create_macros()
        eng.prepare_r1(eng.con)
        notes.append(f"peak RSS after load (engine + harness query table, before any numpy reference state): {common.peak_rss_mb():.0f} MB; "
                     "the final peak_rss_mb also includes the harness's pure-numpy reference R1/R2 states used for the correctness checks")
        log(f"  loaded in {ld['seconds']:.1f} s ({result['load']['db_bytes'] / 1e6:.0f} MB): {result['load']['index_notes']}")
    except Exception as e:  # noqa: BLE001
        result["load"] = common.phase_error(e)
        notes.append(f"load FAILED: {type(e).__name__}: {e}")
        save()
        log(f"load failed: {e}")
        return 1
    save()

    # ---- A/B pre-measurements (warm, not part of the timed phases)
    try:
        if args.no_ab:
            th = eng.default_threads if args.threads == "auto" else int(args.threads)
            eng.set_threads(th)
            eng.expand = "join" if args.expand == "auto" else args.expand
            eng.prepare_r1(eng.con)
            if args.fts_terms_index == "on":
                s = create_fts_terms_index(eng)
                notes.append(f"ART index on fts_main_memories.terms(termid) created in {s:.1f} s (--fts-terms-index on)")
            notes.append(f"A/B skipped (--no-ab): graph_expand={eng.expand}, SET threads={th}, fts terms index={eng.fts_terms_index}")
        else:
            log("A/B pre-measurements (100 R1 / 50 R2 queries from the mixed-phase id range, warm)")
            ab_r1(eng, queries, args.threads, args.expand, notes)
            ab_indexes(eng, queries, notes)
            ab_r2(eng, queries, args.threads, args.fts_terms_index, notes)
    except Exception as e:  # noqa: BLE001
        notes.append(f"A/B pre-measurement FAILED: {type(e).__name__}: {e}")
        log(f"A/B failed: {e}")
        eng.prepare_r1(eng.con)
    save()

    # ---- warmup
    if "warmup" in phases:
        log("warmup: 50 R1 (not timed)")
        try:
            for qid in common.WARMUP_QUERY_IDS:
                q = queries[qid]
                eng.r1(eng.con, q["tenant_id"], q["seed_entity_id"], common.R1_LIMIT)
        except Exception as e:  # noqa: BLE001
            notes.append(f"warmup FAILED: {type(e).__name__}: {e}")

    # ---- r1_only
    if "r1_only" in phases:
        log("r1_only: 1,000 R1 queries")
        try:
            result["phases"]["r1_only"] = phase_r1_only(eng, queries, scale, notes)
            p = result["phases"]["r1_only"]
            log(f"  R1 p50={p['p50_ms']:.3f} p95={p['p95_ms']:.3f} p99={p['p99_ms']:.3f} ms, {p['ops_per_s']:.0f} ops/s, "
                f"reference mismatches {p['reference_mismatches']}")
        except Exception as e:  # noqa: BLE001
            result["phases"]["r1_only"] = common.phase_error(e)
            log(f"  r1_only failed: {e}")
        save()

    # ---- r2_only
    if "r2_only" in phases:
        log("r2_only: 300 R2 queries")
        try:
            result["phases"]["r2_only"] = phase_r2_only(eng, queries, scale, notes)
            p = result["phases"]["r2_only"]
            log(f"  R2 p50={p['p50_ms']:.3f} p95={p['p95_ms']:.3f} ms, {p['ops_per_s']:.0f} ops/s, recall@20={p['recall_at_20']:.4f}")
        except Exception as e:  # noqa: BLE001
            result["phases"]["r2_only"] = common.phase_error(e)
            log(f"  r2_only failed: {e}")
        save()

    # ---- mixed
    if "mixed" in phases:
        log(f"mixed: {common.MIXED_OPS[scale]} ops")
        try:
            result["phases"]["mixed"] = phase_mixed(eng, scale, notes)
            mixed_ran = True
            m = result["phases"]["mixed"]
            log("  " + ", ".join(f"{k} n={m[k]['count']} p50={m[k]['p50_ms']:.3f} p95={m[k]['p95_ms']:.3f}" for k in ("W1", "W2", "R1", "R2"))
                + f", wall {m['wall_s']:.1f} s")
        except Exception as e:  # noqa: BLE001
            result["phases"]["mixed"] = common.phase_error(e)
            log(f"  mixed failed: {e}")
        save()

    # ---- concurrent
    if "concurrent" in phases:
        log(f"concurrent: {common.N_WRITERS} writers (W1) + {common.N_READERS} readers (R1) for {common.CONCURRENT_SECONDS[scale]} s")
        try:
            result["phases"]["concurrent"] = phase_concurrent(eng, scale, notes)
            c = result["phases"]["concurrent"]
            log(f"  W1 {c['W1_ops_per_s']:.1f} ops/s, R1 {c['R1_ops_per_s']:.1f} ops/s, errors {c['errors']}")
        except Exception as e:  # noqa: BLE001
            result["phases"]["concurrent"] = common.phase_error(e)
            log(f"  concurrent failed: {e}")
        save()

    # ---- verify
    if "verify" in phases:
        log("verify: R1 for query_ids 0..199")
        try:
            result["verify"] = phase_verify(eng, queries, scale, mixed_ran, notes)
            log("  " + notes[-1])
        except Exception as e:  # noqa: BLE001
            result["verify"] = {"r1": {}, "error": f"{type(e).__name__}: {e}"}
            notes.append(f"verify FAILED: {type(e).__name__}: {e}")
            log(f"  verify failed: {e}")
        save()

    # ---- post: fts rebuild cost (the non-incremental part of R2), final size
    if not args.no_fts_rebuild:
        try:
            t0 = time.perf_counter()
            eng.con.execute(FTS_INDEX_SQL)
            s = time.perf_counter() - t0
            n = eng.con.execute("SELECT count(*) FROM memories").fetchone()[0]
            notes.append(f"post: rebuilding the fts index over {n} memories after the phases took {s:.1f} s (this is the cost of making BM25 see the writes"
                         + ("; the rebuild drops the fts schema and with it the ART index on terms(termid), which would have to be re-created" if eng.fts_terms_index else "") + ")")
            log(f"fts rebuild {s:.1f} s")
        except Exception as e:  # noqa: BLE001
            notes.append(f"post: fts rebuild FAILED: {type(e).__name__}: {e}")
    try:
        eng.con.execute("CHECKPOINT")
        notes.append(f"post: final db size after CHECKPOINT {common.db_bytes(eng.db_path) / 1e6:.0f} MB")
    except Exception as e:  # noqa: BLE001
        notes.append(f"post: final CHECKPOINT failed: {type(e).__name__}: {e}")
    notes.append(f"host load average (1/5/15 min) at end: {loadavg()}")
    notes.extend(common.DATASET_NOTES)
    eng.close()
    save()
    path = results_dir / f"{ENGINE}.{scale}.json"
    log(f"wrote {path} (peak RSS {result['peak_rss_mb']:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
