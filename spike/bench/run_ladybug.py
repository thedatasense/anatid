#!/usr/bin/env python
"""anatid Phase 0 spike -- engine "ladybug" (LadybugDB 0.20.2, the maintained Kuzu fork, Cypher).

File-backed database results/anatid.lbdb (deleted before load), schema in ladybug/schema.cypher,
bulk load with COPY ... FROM parquet, every SPEC.md operation in Cypher with $parameters, all
phases of the workload schedule (load, warmup, r1_only, r2_only, mixed, concurrent, verify) and the
result JSON via common.write_result -> results/ladybug.<scale>.json.

    .venv/bin/python bench/run_ladybug.py --scale small
    .venv/bin/python bench/run_ladybug.py --scale full

Options (defaults = measured-best configuration; every choice is recorded in the result "notes"):
    --phases a,b,c      subset of load,warmup,r1_only,r2_only,mixed,concurrent,verify (default all;
                        load always runs first, later phases see the state earlier ones left)
    --r1 auto|<variant> R1 formulation (auto = A/B on query_ids 1300..1399, keep the fastest p50 with
                        0 reference mismatches). Variants:
                        two_stmt_in  statement 1: recursive -[:RELATES_TO*0..2 (e,_ | WHERE ...)]- frontier
                                     -> DISTINCT entity ids; statement 2: MATCH (x:Entity)<-[a:ABOUT]-(m)
                                     WHERE x.entity_id IN $ids AND m filters, HINT (x JOIN a) JOIN m,
                                     DISTINCT / ORDER BY / LIMIT
                        rec_hint     one statement: recursive frontier, WITH DISTINCT x, ABOUT expansion
                                     with HINT (x JOIN a) JOIN m
                        rec_hint2    the same with HINT (m JOIN a) JOIN x
                        rec          the same without a HINT (planner's own join order)
                        opt_hint     explicit expansion in one statement: two OPTIONAL MATCH hops,
                                     collect/UNWIND the 0/1/2-hop entity ids, PK lookup, ABOUT expansion
                                     with the HINT
                        two_stmt     statement 1 as two_stmt_in, statement 2 with UNWIND $ids + PK lookup
    --r1-threads auto|N max query threads per connection for R1 (auto = A/B 1 vs engine default)
    --r2-threads auto|N the same for R2 (its cosine / postings scans parallelize; R1 mostly does not)
    --no-ab             skip all A/B pre-measurements
    --no-multi-writes   skip the extra concurrent run with Database(enable_multi_writes=True)
    --db PATH           database file (default results/anatid.lbdb)
    --results-dir DIR   where <engine>.<scale>.json goes (default spike/results)

Correctness: every r1_only result is compared with common.reference_r1 (ids and created_at), every
r2_only result with common.reference_r2_truth (recall@20 of the RRF list, recall@50 of the cosine
and BM25 candidate lists), the verify lists with common.reference_r1(after_mixed=True), and the
mixed phase spot-checks that W1 memories are visible to R1 and W2 old memories got valid_to = now.
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import ladybug as lb  # noqa: E402

warnings.simplefilter("ignore", DeprecationWarning)   # ladybug deprecates prepare()+execute(); we use execute(str, params)

ENGINE = "ladybug"
SCHEMA_PATH = common.SPIKE_DIR / "ladybug" / "schema.cypher"
DEFAULT_DB = common.RESULTS_DIR / "anatid.lbdb"
ALL_PHASES = ["load", "warmup", "r1_only", "r2_only", "mixed", "concurrent", "verify"]
R1_VARIANTS = ["two_stmt_in", "rec_hint", "rec_hint2", "rec", "opt_hint", "two_stmt"]
AB_R1_QIDS = list(range(1300, 1400))          # A/B pre-measurements use mixed-phase query ids,
AB_R2_QIDS = list(range(1300, 1330))          # never the r1_only / r2_only ids
WRITE_CONFLICT_MARKER = "one write transaction"   # LadybugDB: "Cannot start a new write transaction ... Only one write transaction at a time"
MEM_PARAM_KEYS = ("memory_id", "tenant_id", "content", "kind", "embedding", "created_at", "valid_from",
                  "tx_from", "writer", "episode_id", "confidence")
CPU = int(os.cpu_count() or 1)

# ----------------------------------------------------------------------------- Cypher

# R1 (SPEC): frontier = seed + 1..2 hops over current same-tenant RELATES_TO (either direction);
# memories ABOUT the frontier, current, same tenant; ORDER BY created_at DESC, memory_id DESC LIMIT n.
# The recursive predicate (e, _ | WHERE ...) filters every edge on the path. LadybugDB fetches node
# properties of an expanded node set by scanning the whole node table with a semi-mask (cost O(|Memory|),
# ~36 ms at 1M rows) unless the planner sees a small cardinality; the two_stmt_in variant hands it the
# frontier as a constant list so it takes the per-tuple lookup path. The join HINT forces "expand ABOUT
# backwards from the frontier, then fetch the memory rows" (without it the planner scans all ABOUT edges).
R1_FRONTIER_REC = (
    "MATCH (s:Entity {entity_id: $seed})-[:RELATES_TO*0..2 (e, _ | WHERE e.valid_to IS NULL AND e.tenant_id = $t)]-(x:Entity)\n"
    "WITH DISTINCT x\n")
R1_FRONTIER_OPT = (
    "MATCH (s:Entity {entity_id: $seed})\n"
    "OPTIONAL MATCH (s)-[e1:RELATES_TO]-(y:Entity) WHERE e1.valid_to IS NULL AND e1.tenant_id = $t\n"
    "OPTIONAL MATCH (y)-[e2:RELATES_TO]-(z:Entity) WHERE e2.valid_to IS NULL AND e2.tenant_id = $t\n"
    "WITH collect(DISTINCT s.entity_id) + collect(DISTINCT y.entity_id) + collect(DISTINCT z.entity_id) AS ids\n"
    "UNWIND ids AS id\n"
    "WITH DISTINCT id\n"
    "MATCH (x:Entity {entity_id: id})\n")
R1_TAIL = (
    "RETURN DISTINCT m.memory_id, m.created_at\n"
    "ORDER BY m.created_at DESC, m.memory_id DESC\n"
    "LIMIT $lim")
R1_MEMORIES_HINT = "MATCH (x)<-[a:ABOUT]-(m:Memory)\nWHERE m.valid_to IS NULL AND m.tenant_id = $t\nHINT (x JOIN a) JOIN m\n" + R1_TAIL
R1_MEMORIES_HINT2 = "MATCH (x)<-[a:ABOUT]-(m:Memory)\nWHERE m.valid_to IS NULL AND m.tenant_id = $t\nHINT (m JOIN a) JOIN x\n" + R1_TAIL
R1_MEMORIES_PLAIN = "MATCH (x)<-[a:ABOUT]-(m:Memory)\nWHERE m.valid_to IS NULL AND m.tenant_id = $t\n" + R1_TAIL
R1_SINGLE = {
    "rec_hint": R1_FRONTIER_REC + R1_MEMORIES_HINT,
    "rec_hint2": R1_FRONTIER_REC + R1_MEMORIES_HINT2,
    "rec": R1_FRONTIER_REC + R1_MEMORIES_PLAIN,
    "opt_hint": R1_FRONTIER_OPT + R1_MEMORIES_HINT,
}
R1_STMT1_FRONTIER = (
    "MATCH (s:Entity {entity_id: $seed})-[:RELATES_TO*0..2 (e, _ | WHERE e.valid_to IS NULL AND e.tenant_id = $t)]-(x:Entity)\n"
    "RETURN DISTINCT x.entity_id")
R1_STMT2 = {
    "two_stmt_in": ("MATCH (x:Entity)<-[a:ABOUT]-(m:Memory)\n"
                    "WHERE x.entity_id IN $ids AND m.valid_to IS NULL AND m.tenant_id = $t\n"
                    "HINT (x JOIN a) JOIN m\n" + R1_TAIL),
    "two_stmt": "UNWIND $ids AS id\nMATCH (x:Entity {entity_id: id})\n" + R1_MEMORIES_HINT,
}

# R2: cosine top-50 (brute force over the tenant's current memories; no HNSW index available, see
# notes), BM25 top-50 over the Term/Posting inverted index (Okapi BM25 k1=1.2 b=0.75,
# idf = ln((N-df+0.5)/(df+0.5)+1), N/avgdl over all base memories; candidates ranked over the static
# postings, then the current-only filter via PK lookups, re-fetching a longer candidate list if fewer
# than 50 survive), RRF (k=60) in Python via common.rrf_fuse, then the ABOUT entity names of the fused top-k.
R2_COSINE = (
    "MATCH (m:Memory)\n"
    "WHERE m.tenant_id = $t AND m.valid_to IS NULL\n"
    "RETURN m.memory_id AS mid, array_cosine_similarity(m.embedding, $q) AS s\n"
    "ORDER BY s DESC, mid ASC\n"
    "LIMIT $topn")
R2_TERM_IDS = "MATCH (t:Term) WHERE t.term IN $terms RETURN t.term_id"
R2_BM25_CANDIDATES = (
    "MATCH (p:Posting)\n"
    "WHERE p.term_id IN $tids AND p.tenant_id = $t\n"
    "WITH p.memory_id AS mid, sum(ln(($N - p.df + 0.5) / (p.df + 0.5) + 1.0) * to_double(p.tf) * ($k1 + 1.0)"
    " / (to_double(p.tf) + $k1 * (1.0 - $b + $b * to_double(p.dl) / $avgdl))) AS score\n"
    "RETURN mid, score\n"
    "ORDER BY score DESC, mid ASC\n"
    "LIMIT $k")
R2_VALID = "UNWIND $ids AS id\nMATCH (m:Memory {memory_id: id})\nWHERE m.valid_to IS NULL\nRETURN id"
R2_NAMES = (
    "MATCH (m:Memory)-[a:ABOUT]->(e:Entity)\n"
    "WHERE m.memory_id IN $ids\n"
    "HINT (m JOIN a) JOIN e\n"
    "RETURN m.memory_id, collect(e.name)")

# W1 / W2 (one explicit transaction each: BEGIN TRANSACTION ... COMMIT). Properties not listed are NULL
# (valid_to, tx_to). $dsts / $eids are parallel lists (Kuzu lists are 1-based).
W1_CYPHER = (
    "CREATE (m:Memory {memory_id: $memory_id, tenant_id: $tenant_id, content: $content, kind: $kind, embedding: $embedding,"
    " created_at: $created_at, valid_from: $valid_from, tx_from: $tx_from, writer: $writer, episode_id: $episode_id, confidence: $confidence})\n"
    "WITH m\n"
    "UNWIND range(1, size($dsts)) AS i\n"
    "MATCH (e:Entity {entity_id: $dsts[i]})\n"
    "CREATE (m)-[:ABOUT {edge_id: $eids[i], tenant_id: $tenant_id, weight: 1.0, valid_from: $now, tx_from: $now}]->(e)")
W2_CYPHER = (
    "CREATE (n:Memory {memory_id: $memory_id, tenant_id: $tenant_id, content: $content, kind: $kind, embedding: $embedding,"
    " created_at: $created_at, valid_from: $valid_from, tx_from: $tx_from, writer: $writer, episode_id: $episode_id, confidence: $confidence})\n"
    "WITH n\n"
    "MATCH (o:Memory {memory_id: $old})\n"
    "SET o.valid_to = $now\n"
    "CREATE (n)-[:SUPERSEDES {edge_id: $eid, tenant_id: $tenant_id, tx_from: $now}]->(o)")

EXT_PROBE_CODE = r"""
import ladybug as lb, sys
def say(m): print(m, flush=True)
db = lb.Database(":memory:")
con = lb.Connection(db)
con.execute("CREATE NODE TABLE M(id INT64, content STRING, embedding FLOAT[64], PRIMARY KEY(id))")
con.execute("CREATE (:M {id: 1, content: 'alpha beta', embedding: $e})", {"e": [0.1] * 64})
for ext in ("FTS", "VECTOR"):
    try:
        con.execute(f"INSTALL {ext}"); con.execute(f"LOAD {ext}")
        loaded = [r for r in con.execute("CALL show_loaded_extensions() RETURN *").get_all() if r[0].upper() == ext]
        say(f"{ext} extension installs/loads ({loaded[0][2] if loaded else 'path unknown'})")
    except Exception as e:
        say(f"INSTALL/LOAD {ext} failed: {str(e)[:160]}")
for label, stmt in (("CREATE_FTS_INDEX", "CALL CREATE_FTS_INDEX('M', 'm_fts', ['content'], stemmer := 'none')"),
                    ("CREATE_VECTOR_INDEX", "CALL CREATE_VECTOR_INDEX('M', 'm_vec', 'embedding', metric := 'cosine')")):
    try:
        con.execute(stmt); say(f"{label} works")
    except Exception as e:
        say(f"{label} fails: {str(e)[:160]}")
say("probe statements done")
con.close(); db.close()
say("Database.close() ok")
"""


# ----------------------------------------------------------------------------- helpers

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def split_cypher(text: str) -> list[str]:
    body = "\n".join(l for l in text.splitlines() if not l.strip().startswith("//"))
    return [s.strip() for s in body.split(";") if s.strip()]


def pct(xs_ms: list[float], p: float) -> float:
    xs = sorted(xs_ms)
    return xs[min(len(xs) - 1, max(0, int(round(p / 100.0 * len(xs) + 0.5)) - 1))]


def summ(xs_ms: list[float]) -> str:
    return f"p50={statistics.median(xs_ms):.3f} p95={pct(xs_ms, 95):.3f} mean={statistics.fmean(xs_ms):.3f} ms (n={len(xs_ms)})"


def loadavg() -> str:
    try:
        return "/".join(f"{x:.1f}" for x in os.getloadavg())
    except OSError:
        return "n/a"


def rm_db_files(db_path: Path) -> None:
    for f in db_path.parent.glob(db_path.name + "*"):
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)


def query_terms(text: str) -> list[str]:
    out: list[str] = []
    for t in str(text).lower().split():
        if t not in out:
            out.append(t)
    return out


def memory_params(m: dict) -> dict:
    p = {k: m[k] for k in MEM_PARAM_KEYS}
    p["embedding"] = [float(x) for x in m["embedding"]]
    p["confidence"] = float(m["confidence"])
    return p


def thr_label(n: int | None) -> str:
    return "engine default (all cores)" if n is None else str(n)


def probe_extensions() -> str:
    """Try the FTS / VECTOR extensions on a throwaway in-memory database IN A SUBPROCESS and report what
    happens. Kept out of the benchmark database (a failing CREATE_FTS_INDEX leaves internal tables behind)
    and out of this process (closing that database after the failed calls aborts the interpreter, SIGABRT)."""
    try:
        r = subprocess.run([sys.executable, "-c", EXT_PROBE_CODE], capture_output=True, text=True, timeout=300)
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        tail = "" if r.returncode == 0 else (f"; probe subprocess exited with code {r.returncode} (native crash -- SIGSEGV/SIGABRT -- at Database.close() after the "
                                             f"failed CREATE_* calls; stderr: {r.stderr.strip()[-160:]!r})")
        return "Extensions (probed in a subprocess): " + "; ".join(lines) + tail + "."
    except Exception as e:  # noqa: BLE001
        return f"Extensions: probe failed to run: {type(e).__name__}: {str(e)[:160]}."


# ----------------------------------------------------------------------------- engine

class Ladybug:
    def __init__(self, scale: str, db_path: Path, r1_variant: str, r1_threads: int | None, r2_threads: int | None):
        self.scale = scale
        self.db_path = Path(db_path)
        self.r1_variant = r1_variant
        self.r1_threads = r1_threads           # None = engine default (all cores)
        self.r2_threads = r2_threads
        self.db: lb.Database | None = None
        self.con: lb.Connection | None = None
        self._con_threads: int | None = -1     # what self.con is currently set to (-1 = unknown)
        self.n_docs = 0                        # BM25 corpus size (base memories at load)
        self.avgdl = 0.0
        self.multi_writes = False
        self.bm25_refetches = 0                # times the candidate list had to be enlarged

    # --- lifecycle
    def open(self, delete: bool = False, multi_writes: bool = False) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if delete:
            rm_db_files(self.db_path)
        self.db = lb.Database(str(self.db_path), enable_multi_writes=multi_writes)
        self.multi_writes = multi_writes
        self.con = lb.Connection(self.db)
        self._con_threads = -1

    def connection(self, threads: int | None) -> lb.Connection:
        con = lb.Connection(self.db)
        con.set_max_threads_for_exec(int(threads) if threads is not None else CPU)
        return con

    def use_threads(self, n: int | None) -> None:
        if n != self._con_threads:
            self.con.set_max_threads_for_exec(int(n) if n is not None else CPU)
            self._con_threads = n

    def close(self) -> None:
        if self.con is not None:
            try:
                self.con.close()
            except Exception:  # noqa: BLE001
                pass
            self.con = None
        if self.db is not None:
            try:
                self.db.close()
            except Exception:  # noqa: BLE001
                pass
            self.db = None
        gc.collect()

    def ex(self, q: str, params: dict | None = None, con: lb.Connection | None = None) -> list[list]:
        return (con or self.con).execute(q, params).get_all()

    def count(self, what: str) -> int:
        q = f"MATCH ()-[r:{what}]->() RETURN count(*)" if what.isupper() else f"MATCH (n:{what}) RETURN count(*)"
        return int(self.ex(q)[0][0])

    # --- load
    def load(self, result: dict) -> None:
        d = str(common.data_dir(self.scale))
        steps: dict[str, float] = {}
        t_all = time.perf_counter()

        t = time.perf_counter()
        for stmt in split_cypher(SCHEMA_PATH.read_text()):
            self.ex(stmt)
        steps["schema"] = time.perf_counter() - t

        for name, stmt in (
            ("entities", f"COPY Entity FROM '{d}/entities.parquet'"),
            ("memories", f"COPY Memory FROM '{d}/memories.parquet'"),
            ("edges_about", f"COPY ABOUT FROM (LOAD FROM '{d}/edges_about.parquet' RETURN src, dst, edge_id, tenant_id, weight, valid_from, valid_to, tx_from, tx_to)"),
            ("edges_relates", f"COPY RELATES_TO FROM (LOAD FROM '{d}/edges_relates.parquet' RETURN src, dst, edge_id, tenant_id, rel_kind, valid_from, valid_to, tx_from, tx_to)"),
            ("edges_supersedes", f"COPY SUPERSEDES FROM (LOAD FROM '{d}/edges_supersedes.parquet' RETURN src, dst, edge_id, tenant_id, tx_from)"),
        ):
            t = time.perf_counter()
            self.ex(stmt)
            steps["copy_" + name] = time.perf_counter() - t

        # BM25 inverted index materialized as Term + Posting node tables (see schema.cypher): the FTS
        # extension's CREATE_FTS_INDEX is not usable in this build (probe below records the error).
        t = time.perf_counter()
        tmp_dir = self.db_path.parent / "_ladybug_load_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        n_terms, n_postings = self.build_postings(tmp_dir)
        steps["bm25_postings_compute"] = time.perf_counter() - t
        t = time.perf_counter()
        self.ex(f"COPY Term FROM '{tmp_dir}/terms.parquet'")
        self.ex(f"COPY Posting FROM '{tmp_dir}/postings.parquet'")
        steps["bm25_postings_copy"] = time.perf_counter() - t
        shutil.rmtree(tmp_dir, ignore_errors=True)

        t = time.perf_counter()
        self.ex("CHECKPOINT")
        steps["checkpoint"] = time.perf_counter() - t
        seconds = time.perf_counter() - t_all

        rows = {k: self.count(v) for k, v in (("entities", "Entity"), ("memories", "Memory"), ("edges_about", "ABOUT"),
                                              ("edges_relates", "RELATES_TO"), ("edges_supersedes", "SUPERSEDES"),
                                              ("terms", "Term"), ("postings", "Posting"))}
        ext_notes = probe_extensions()
        result["load"] = {
            "seconds": seconds,
            "db_bytes": common.db_bytes(self.db_path),
            "index_notes": (
                "no secondary indexes: LadybugDB keeps a hash index on each node PRIMARY KEY (Entity.entity_id, "
                "Memory.memory_id, Term.term_id; Posting.pid is SERIAL) and forward+backward adjacency lists per REL TABLE "
                "(ABOUT, RELATES_TO, SUPERSEDES); R1 uses the RELATES_TO/ABOUT adjacency lists + PK lookup of the seed. "
                f"BM25 index = Term + Posting node tables materialized at load from memories.content ({n_terms} terms, "
                f"{n_postings} (memory, term) postings sorted by term_id/tenant_id, N={self.n_docs}, avgdl={self.avgdl:.3f}; "
                f"pyarrow tokenization {steps['bm25_postings_compute']:.1f} s + COPY {steps['bm25_postings_copy']:.1f} s), "
                "NOT incremental (W1/W2 memories are invisible to BM25). Vector: no ANN index, brute-force "
                "array_cosine_similarity over the tenant's current memories. " + ext_notes +
                " Load parallelism: engine default (all cores). Memory rows loaded in Parquet order (tenant-sorted order "
                "was measured at full: no R1 gain, 4x slower COPY)."),
            "steps_s": {k: round(v, 3) for k, v in steps.items()},
            "row_counts": rows,
        }
        log(f"load {seconds:.1f} s, db {result['load']['db_bytes'] / 1e6:.0f} MB, rows {rows}")

    def build_postings(self, tmp_dir: Path) -> tuple[int, int]:
        """Whitespace tokens of lower-cased content (same tokenizer as common.reference_r2_truth):
        Term(term_id, term, df) and Posting(term_id, tenant_id, memory_id, tf, dl, df) as Parquet for COPY."""
        mem = common.load_table(self.scale, "memories", ["memory_id", "tenant_id", "content"])
        n = mem.num_rows
        toks = pc.utf8_split_whitespace(pc.utf8_lower(mem["content"])).combine_chunks()
        doclen = pc.list_value_length(toks).to_numpy().astype(np.int64)
        flat = pc.dictionary_encode(toks.flatten())
        term_of_tok = flat.indices.to_numpy().astype(np.int64)
        vocab = flat.dictionary.to_pylist()
        doc_of_tok = np.repeat(mem["memory_id"].to_numpy(), doclen)
        key, tf = np.unique(term_of_tok * n + doc_of_tok, return_counts=True)
        term_sorted, doc_sorted = key // n, key % n
        df = np.bincount(term_sorted, minlength=len(vocab))
        tenant = mem["tenant_id"].to_numpy()
        self.n_docs = int(n)
        self.avgdl = float(doclen.mean())
        order = np.lexsort((doc_sorted, tenant[doc_sorted], term_sorted))
        pq.write_table(pa.table({"term_id": pa.array(np.arange(len(vocab)), pa.int64()), "term": pa.array(vocab, pa.string()),
                                 "df": pa.array(df, pa.int64())}), tmp_dir / "terms.parquet")
        pq.write_table(pa.table({"term_id": pa.array(term_sorted[order], pa.int64()),
                                 "tenant_id": pa.array(tenant[doc_sorted][order], pa.int32()),
                                 "memory_id": pa.array(doc_sorted[order], pa.int64()),
                                 "tf": pa.array(tf[order].astype(np.int32), pa.int32()),
                                 "dl": pa.array(doclen[doc_sorted][order].astype(np.int32), pa.int32()),
                                 "df": pa.array(df[term_sorted][order], pa.int64())}), tmp_dir / "postings.parquet")
        return len(vocab), int(len(key))

    # --- operations
    def r1(self, tenant_id: int, seed: int, limit: int = common.R1_LIMIT, con: lb.Connection | None = None) -> list[list]:
        if con is None:
            self.use_threads(self.r1_threads)
            con = self.con
        v = self.r1_variant
        if v in R1_STMT2:
            ids = [r[0] for r in con.execute(R1_STMT1_FRONTIER, {"seed": int(seed), "t": int(tenant_id)}).get_all()]
            return con.execute(R1_STMT2[v], {"ids": ids, "t": int(tenant_id), "lim": int(limit)}).get_all()
        return con.execute(R1_SINGLE[v], {"seed": int(seed), "t": int(tenant_id), "lim": int(limit)}).get_all()

    def r2_parts(self, tenant_id: int, query_text: str, query_embedding, topn: int = common.R2_TOPN) -> tuple[list[int], list[int]]:
        self.use_threads(self.r2_threads)
        cos = self.con.execute(R2_COSINE, {"t": int(tenant_id), "q": [float(x) for x in query_embedding], "topn": int(topn)}).get_all()
        tids = [r[0] for r in self.con.execute(R2_TERM_IDS, {"terms": query_terms(query_text)}).get_all()]
        bm: list[int] = []
        if tids:
            k = 2 * topn
            while True:
                cand = self.con.execute(R2_BM25_CANDIDATES, {"tids": tids, "t": int(tenant_id), "N": float(self.n_docs), "k1": common.BM25_K1,
                                                             "b": common.BM25_B, "avgdl": self.avgdl, "k": int(k)}).get_all()
                valid = {r[0] for r in self.con.execute(R2_VALID, {"ids": [r[0] for r in cand]}).get_all()}
                bm = [r[0] for r in cand if r[0] in valid][:topn]
                if len(bm) >= topn or len(cand) < k:
                    break
                k *= 4
                self.bm25_refetches += 1
        return [r[0] for r in cos], bm

    def r2(self, tenant_id: int, query_text: str, query_embedding, k: int = common.R2_K) -> list[tuple]:
        cos_ids, bm_ids = self.r2_parts(tenant_id, query_text, query_embedding)
        fused = common.rrf_fuse(cos_ids, bm_ids, k=common.RRF_K, top=k)
        names = {r[0]: r[1] for r in self.con.execute(R2_NAMES, {"ids": [m for m, _ in fused]}).get_all()}
        return [(m, s, names.get(m, [])) for m, s in fused]

    def w1(self, op: dict, con: lb.Connection | None = None) -> None:
        con = con or self.con
        p = memory_params(op["memory"])
        p.update({"dsts": [int(x) for x in op["about_dsts"]], "eids": [int(x) for x in op["about_edge_ids"]], "now": op["now"]})
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(W1_CYPHER, p)
            con.execute("COMMIT")
        except Exception:
            try:
                con.execute("ROLLBACK")
            except Exception:  # noqa: BLE001
                pass
            raise

    def w2(self, op: dict) -> None:
        p = memory_params(op["memory"])
        p.update({"old": int(op["old_memory_id"]), "eid": int(op["supersedes_edge_id"]), "now": op["now"]})
        self.con.execute("BEGIN TRANSACTION")
        try:
            self.con.execute(W2_CYPHER, p)
            self.con.execute("COMMIT")
        except Exception:
            try:
                self.con.execute("ROLLBACK")
            except Exception:  # noqa: BLE001
                pass
            raise


# ----------------------------------------------------------------------------- phases

def ab_r1(eng: Ladybug, queries: list[dict], variants: list[str], thread_opts: list[int | None], notes: list[str]) -> tuple[str, int | None]:
    """Time each (variant, threads) on AB_R1_QIDS (results checked against the reference); fastest p50 wins."""
    best = None
    for thr in thread_opts:
        eng.r1_threads = thr
        for v in variants:
            eng.r1_variant = v
            try:
                for qid in list(common.WARMUP_QUERY_IDS)[:10]:
                    q = queries[qid]
                    eng.r1(q["tenant_id"], q["seed_entity_id"])
                lat, mism = [], 0
                for qid in AB_R1_QIDS:
                    q = queries[qid]
                    t0 = time.perf_counter_ns()
                    rows = eng.r1(q["tenant_id"], q["seed_entity_id"])
                    lat.append((time.perf_counter_ns() - t0) / 1e6)
                    ok, _ = common.compare_r1(rows, common.reference_r1(eng.scale, q["tenant_id"], q["seed_entity_id"]))
                    mism += (not ok)
                p50 = statistics.median(lat)
                notes.append(f"A/B R1 {v} threads={thr_label(thr)}: {summ(lat)}; reference mismatches {mism}/{len(lat)}")
                log(f"  A/B R1 {v} threads={thr_label(thr)}: {summ(lat)} mismatches={mism}")
                if mism == 0 and (best is None or p50 < best[0]):
                    best = (p50, v, thr)
            except Exception as e:  # noqa: BLE001
                notes.append(f"A/B R1 {v} threads={thr_label(thr)}: FAILED {type(e).__name__}: {str(e)[:200]}")
                log(f"  A/B R1 {v} threads={thr_label(thr)}: FAILED {str(e)[:120]}")
    if best is None:
        raise RuntimeError("every R1 variant failed or mismatched in the A/B")
    return best[1], best[2]


def ab_r2(eng: Ladybug, queries: list[dict], thread_opts: list[int | None], notes: list[str]) -> int | None:
    best = None
    for thr in thread_opts:
        eng.r2_threads = thr
        lat, rec = [], []
        for qid in AB_R2_QIDS:
            q = queries[qid]
            t0 = time.perf_counter_ns()
            rows = eng.r2(q["tenant_id"], q["query_text"], q["query_embedding"])
            lat.append((time.perf_counter_ns() - t0) / 1e6)
            rec.append(common.recall_at_k(common.reference_r2_truth(eng.scale, q)["rrf_ids"], [r[0] for r in rows]))
        p50 = statistics.median(lat)
        notes.append(f"A/B R2 threads={thr_label(thr)}: {summ(lat)}; recall@20 {statistics.fmean(rec):.4f}")
        log(f"  A/B R2 threads={thr_label(thr)}: {summ(lat)} recall@20={statistics.fmean(rec):.4f}")
        if best is None or p50 < best[0]:
            best = (p50, thr)
    return best[1]


def phase_r1_only(eng: Ladybug, queries: list[dict], result: dict) -> None:
    timer = common.Timer()
    got: dict[int, list] = {}
    with common.Stopwatch() as sw:
        for qid in common.R1_QUERY_IDS:
            q = queries[qid]
            with timer:
                got[qid] = eng.r1(q["tenant_id"], q["seed_entity_id"], common.R1_LIMIT)
    phase = timer.summary(sw.seconds)
    rss_before = common.peak_rss_mb()
    mism = 0
    tenants = set()
    for qid, rows in got.items():
        q = queries[qid]
        tenants.add(q["tenant_id"])
        ok, msg = common.compare_r1(rows, common.reference_r1(eng.scale, q["tenant_id"], q["seed_entity_id"]))
        if not ok:
            mism += 1
            if mism <= 3:
                result["notes"].append(f"r1_only mismatch qid {qid}: {msg}")
    phase["reference_checked"] = len(got)
    phase["reference_mismatches"] = mism
    phase["reference_tenants_covered"] = len(tenants)
    phase["peak_rss_mb_before_reference_check"] = rss_before
    phase["formulation"] = eng.r1_variant
    phase["threads"] = thr_label(eng.r1_threads)
    result["phases"]["r1_only"] = phase
    result["notes"].append(f"r1_only: {len(got)} R1 results compared with common.reference_r1 (ids + created_at) across {len(tenants)} tenants: {mism} mismatches")
    log(f"r1_only: p50={phase['p50_ms']:.3f} p95={phase['p95_ms']:.3f} ms, mismatches={mism}")


def phase_r2_only(eng: Ladybug, queries: list[dict], result: dict) -> None:
    timer = common.Timer()
    got: dict[int, list] = {}
    eng.bm25_refetches = 0
    with common.Stopwatch() as sw:
        for qid in common.R2_QUERY_IDS:
            q = queries[qid]
            with timer:
                got[qid] = eng.r2(q["tenant_id"], q["query_text"], q["query_embedding"], common.R2_K)
    phase = timer.summary(sw.seconds)
    refetches = eng.bm25_refetches
    # recall vs brute-force truth (candidate lists recomputed outside the timed loop)
    rec, rec_cos, rec_bm, exact = [], [], [], 0
    for qid, rows in got.items():
        q = queries[qid]
        truth = common.reference_r2_truth(eng.scale, q)
        ids = [r[0] for r in rows]
        rec.append(common.recall_at_k(truth["rrf_ids"], ids))
        exact += (ids == truth["rrf_ids"])
        cos_ids, bm_ids = eng.r2_parts(q["tenant_id"], q["query_text"], q["query_embedding"])
        rec_cos.append(common.recall_at_k(truth["cosine_ids"], cos_ids, common.R2_TOPN))
        rec_bm.append(common.recall_at_k(truth["bm25_ids"], bm_ids, common.R2_TOPN))
    phase["recall_at_20"] = statistics.fmean(rec)
    phase["recall_at_20_min"] = min(rec)
    phase["exact_top20_lists"] = f"{exact}/{len(got)}"
    phase["cosine_recall_at_50"] = statistics.fmean(rec_cos)
    phase["bm25_recall_at_50"] = statistics.fmean(rec_bm)
    phase["bm25_candidate_refetches"] = refetches
    phase["threads"] = thr_label(eng.r2_threads)
    phase["vector_index"] = "none (CREATE_VECTOR_INDEX unavailable in this build); brute-force array_cosine_similarity over the tenant's current memories"
    phase["bm25"] = ("Term/Posting inverted index built at load (Okapi BM25 k1=1.2 b=0.75 in Cypher over Posting, top-100 candidates, then "
                     "current-only filter via PK lookups, re-fetch x4 if fewer than 50 survive); NOT incremental (stale after writes); "
                     "RRF k=60 in Python (common.rrf_fuse); ABOUT entity names via one more statement; 5 statements per op, all inside the timed op")
    result["phases"]["r2_only"] = phase
    result["notes"].append(f"r2_only: recall@20 vs common.reference_r2_truth over {len(got)} queries: mean {phase['recall_at_20']:.4f}, "
                           f"min {phase['recall_at_20_min']:.3f}, exact top-20 lists {exact}/{len(got)}; cosine top-50 recall {phase['cosine_recall_at_50']:.4f}, "
                           f"BM25 top-50 recall {phase['bm25_recall_at_50']:.4f}; BM25 candidate list enlarged {refetches} times")
    log(f"r2_only: p50={phase['p50_ms']:.3f} p95={phase['p95_ms']:.3f} ms, recall@20={phase['recall_at_20']:.4f}")


def phase_mixed(eng: Ladybug, ops: list[dict], result: dict) -> None:
    timers = {k: common.Timer() for k in ("W1", "W2", "R1", "R2")}
    errors: list[str] = []
    with common.Stopwatch() as sw:
        for op in ops:
            kind = op["op"]
            try:
                with timers[kind]:
                    if kind == "W1":
                        eng.w1(op)
                    elif kind == "W2":
                        eng.w2(op)
                    elif kind == "R1":
                        eng.r1(op["tenant_id"], op["seed_entity_id"], op["limit"])
                    else:
                        eng.r2(op["tenant_id"], op["query_text"], op["query_embedding"], op["k"])
            except Exception as e:  # noqa: BLE001
                errors.append(f"{kind} seq {op['seq']}: {type(e).__name__}: {str(e)[:200]}")
                if len(errors) > 20:
                    raise
    phase = {k: t.summary() for k, t in timers.items()}
    phase["wall_s"] = sw.seconds
    phase["ops"] = len(ops)
    phase["errors"] = len(errors)
    if errors:
        phase["error_samples"] = errors[:10]
    # spot checks: W1 memories visible to R1 via their first ABOUT entity; W2 old memories closed
    w1_ops = [o for o in ops if o["op"] == "W1"][-10:]
    w2_ops = [o for o in ops if o["op"] == "W2"][-10:]
    seen = 0
    for o in w1_ops:
        ids = [r[0] for r in eng.r1(o["tenant_id"], o["about_dsts"][0], common.R1_LIMIT)]
        seen += (o["memory"]["memory_id"] in ids)
    closed = 0
    for o in w2_ops:
        vt = eng.ex("MATCH (m:Memory {memory_id: $id}) RETURN m.valid_to", {"id": int(o["old_memory_id"])})[0][0]
        closed += (vt == o["now"])
    n_written = sum(1 for o in ops if o["op"] in ("W1", "W2"))
    phase["write_spot_check"] = {"w1_visible_in_r1": f"{seen}/{len(w1_ops)}", "w2_old_valid_to_set": f"{closed}/{len(w2_ops)}"}
    phase["bm25_stale_memories"] = n_written
    result["phases"]["mixed"] = phase
    counts = common.schedule_counts(ops)
    result["notes"].append(f"mixed: {len(ops)} ops ({counts}) in {sw.seconds:.1f} s, {len(errors)} errors; spot check: {seen}/{len(w1_ops)} of the last W1 memories visible "
                           f"to R1 via their first ABOUT entity, {closed}/{len(w2_ops)} of the last W2 old memories have valid_to = now; the {n_written} memories "
                           "inserted by W1/W2 are invisible to BM25 (Term/Posting index is not incremental; the cosine side sees them)")
    log(f"mixed: {len(ops)} ops in {sw.seconds:.1f} s; W1 p50={phase['W1']['p50_ms']:.3f} W2 p50={phase['W2']['p50_ms']:.3f} "
        f"R1 p50={phase['R1']['p50_ms']:.3f} R2 p50={phase['R2']['p50_ms']:.3f} ms; errors={len(errors)}")


VARIANT_ID_OFFSET = 15_000_000   # 2nd concurrent run: shift memory/edge ids inside each writer's 30M stride (still >= 1e9)


def run_concurrent_once(eng: Ladybug, seconds: float, retry: bool, id_offset: int = 0) -> dict:
    """SPEC step 6 with one Connection per thread. LadybugDB (default) allows one write transaction at a
    time and rejects a second BEGIN immediately; with retry=True a writer retries its BEGIN (0.5 ms sleep)
    until it gets the lock (retries counted separately, not as errors). id_offset shifts the harness's
    concurrent memory/edge ids so a second run on the same database does not collide with the first."""
    lock = threading.Lock()
    stats = {"retries": 0, "max_retries_one_op": 0, "committed": 0}

    def writer_factory(idx: int):
        con = eng.connection(1)
        it = common.concurrent_w1_ops(eng.scale, idx)

        def step():
            op = next(it)
            if id_offset:
                op["memory"]["memory_id"] += id_offset
                op["about_edge_ids"] = [e + id_offset for e in op["about_edge_ids"]]
            p = memory_params(op["memory"])
            p.update({"dsts": [int(x) for x in op["about_dsts"]], "eids": [int(x) for x in op["about_edge_ids"]], "now": op["now"]})
            attempts = 0
            while True:
                try:
                    con.execute("BEGIN TRANSACTION")
                    break
                except Exception as e:  # noqa: BLE001
                    if retry and WRITE_CONFLICT_MARKER in str(e) and attempts < 20000:
                        attempts += 1
                        time.sleep(0.0005)
                        continue
                    raise
            try:
                con.execute(W1_CYPHER, p)
                con.execute("COMMIT")
            except Exception:
                try:
                    con.execute("ROLLBACK")
                except Exception:  # noqa: BLE001
                    pass
                raise
            with lock:
                stats["committed"] += 1
                if attempts:
                    stats["retries"] += attempts
                    stats["max_retries_one_op"] = max(stats["max_retries_one_op"], attempts)
        return step

    def reader_factory(idx: int):
        con = eng.connection(eng.r1_threads)
        it = common.concurrent_r1_queries(eng.scale, idx)

        def step():
            op = next(it)
            eng.r1(op["tenant_id"], op["seed_entity_id"], op["limit"], con=con)
        return step

    n_before = eng.count("Memory")
    conc = common.run_concurrent(seconds, writer_factory, reader_factory)
    conc["memories_inserted"] = eng.count("Memory") - n_before
    conc["begin_retries"] = stats["retries"]
    conc["max_begin_retries_one_op"] = stats["max_retries_one_op"]
    return conc


def phase_concurrent(eng: Ladybug, scale: str, result: dict, multi_writes: bool) -> None:
    seconds = common.CONCURRENT_SECONDS[scale]
    conc = run_concurrent_once(eng, seconds, retry=True)
    conc["notes"] = (
        f"{common.N_WRITERS} writer + {common.N_READERS} reader threads, one lb.Connection per thread on the same open Database "
        f"(default settings: single writer). Every W1 = BEGIN TRANSACTION / CREATE memory + ABOUT edges / COMMIT. A second concurrent "
        f"writer's BEGIN fails immediately with 'Cannot start a new write transaction ... Only one write transaction at a time'; the "
        f"writer retries BEGIN after a 0.5 ms sleep (retries are counted, not errors): {conc['begin_retries']} retries in total, max "
        f"{conc['max_begin_retries_one_op']} for one op; {conc['memories_inserted']} memories committed ({conc['W1_ops']} W1 ops counted); "
        f"W1 errors {conc['W1_errors']}, R1 errors {conc['R1_errors']}; threads with zero ops: W1 {conc['W1_threads_with_zero_ops']}, "
        f"R1 {conc['R1_threads_with_zero_ops']}; query threads per connection: writers 1, readers {thr_label(eng.r1_threads)}; "
        f"R1 formulation {eng.r1_variant}")
    log(f"concurrent: W1 {conc['W1_ops_per_s']:.1f} ops/s (p50 {conc['W1_p50_ms']}), R1 {conc['R1_ops_per_s']:.1f} ops/s (p50 {conc['R1_p50_ms']}), "
        f"errors {conc['errors']}, retries {conc['begin_retries']}")
    if multi_writes:
        # extra run: re-open the same database with enable_multi_writes=True (LadybugDB-specific option)
        try:
            eng.close()
            eng.open(delete=False, multi_writes=True)
            mw = run_concurrent_once(eng, seconds, retry=True, id_offset=VARIANT_ID_OFFSET)
            mw["notes"] = (f"same workload with lb.Database(enable_multi_writes=True) (memory/edge ids shifted by {VARIANT_ID_OFFSET} inside each "
                           f"writer's id stride so they do not collide with the first run): {mw['memories_inserted']} memories committed, "
                           f"{mw['begin_retries']} BEGIN retries, W1 errors {mw['W1_errors']}, R1 errors {mw['R1_errors']}")
            if not mw["errors"]:
                mw.pop("error_samples", None)
            conc["multi_writes_variant"] = mw
            log(f"concurrent (enable_multi_writes=True): W1 {mw['W1_ops_per_s']:.1f} ops/s, R1 {mw['R1_ops_per_s']:.1f} ops/s, errors {mw['errors']}, retries {mw['begin_retries']}")
        except Exception as e:  # noqa: BLE001
            conc["multi_writes_variant"] = common.phase_error(e)
            log(f"concurrent multi-writes variant FAILED: {e}")
        finally:
            eng.close()
            eng.open(delete=False, multi_writes=False)
    result["phases"]["concurrent"] = conc
    result["notes"].append(f"concurrent: W1 {conc['W1_ops_per_s']:.1f} ops/s (p50 {conc['W1_p50_ms']:.2f} ms), R1 {conc['R1_ops_per_s']:.1f} ops/s "
                           f"(p50 {conc['R1_p50_ms']:.2f} ms), errors {conc['errors']}; {conc['notes']}")
    mw = conc.get("multi_writes_variant")
    if isinstance(mw, dict) and "W1_ops_per_s" in mw:
        result["notes"].append(f"concurrent variant enable_multi_writes=True (LadybugDB option, same {seconds} s workload after re-opening the database): "
                               f"W1 {mw['W1_ops_per_s']:.1f} ops/s (p50 {mw['W1_p50_ms']:.2f} ms), R1 {mw['R1_ops_per_s']:.1f} ops/s (p50 {mw['R1_p50_ms']:.2f} ms), "
                               f"errors {mw['errors']}, BEGIN retries {mw['begin_retries']}, {mw['memories_inserted']} memories committed")
    elif isinstance(mw, dict):
        result["notes"].append(f"concurrent variant enable_multi_writes=True FAILED: {mw.get('error')}")


def phase_verify(eng: Ladybug, queries: list[dict], result: dict) -> None:
    got: dict[int, list] = {}
    for qid in common.VERIFY_QUERY_IDS:
        q = queries[qid]
        got[qid] = eng.r1(q["tenant_id"], q["seed_entity_id"], common.R1_LIMIT)
    result["verify"] = common.verify_payload(got)
    mism = 0
    for qid, rows in got.items():
        q = queries[qid]
        ok, msg = common.compare_r1(rows, common.reference_r1(eng.scale, q["tenant_id"], q["seed_entity_id"], after_mixed=True))
        if not ok:
            mism += 1
            if mism <= 3:
                result["notes"].append(f"verify mismatch qid {qid}: {msg}")
    result["notes"].append(f"verify: {len(got)} R1 id lists dumped; compared with common.reference_r1(after_mixed=True): {mism} mismatches")
    log(f"verify: {len(got)} lists, {mism} mismatches vs reference(after_mixed)")


# ----------------------------------------------------------------------------- main

def parse_threads(s: str) -> int | None:
    return None if s in ("auto", "default", "0") else int(s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=common.SCALES, required=True)
    ap.add_argument("--phases", default=",".join(ALL_PHASES))
    ap.add_argument("--r1", default="auto", choices=["auto"] + R1_VARIANTS)
    ap.add_argument("--r1-threads", default="auto")
    ap.add_argument("--r2-threads", default="auto")
    ap.add_argument("--no-ab", action="store_true")
    ap.add_argument("--no-multi-writes", action="store_true")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--results-dir", default=str(common.RESULTS_DIR))
    args = ap.parse_args()

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    for p in phases:
        if p not in ALL_PHASES:
            ap.error(f"unknown phase {p!r}; choose from {ALL_PHASES}")
    scale = args.scale

    version = lb.__version__
    result = common.result_skeleton(ENGINE, version, scale)
    result["notes"].append(f"config: ladybug {version} (python package 'ladybug', import ladybug as lb; Kuzu 0.11-compatible API) python {sys.version.split()[0]}; "
                           f"db={args.db}; phases={phases}; r1={args.r1}; r1-threads={args.r1_threads}; r2-threads={args.r2_threads}")
    result["notes"].append(f"host load average (1/5/15 min) at start: {loadavg()} on {CPU} cpus (SPEC: numbers are only comparable if nothing else was running)")
    result["notes"].append("statements are executed with Connection.execute(query, params) ($name parameters, no string building for the timed ops); "
                           "the client keeps an implicit prepared-statement cache keyed by (query text, parameter signature), so repeated executions "
                           "reuse the compiled plan. Connection.prepare()+execute(prepared, params) is deprecated in 0.20.2 and is wrong for parameters "
                           "inside a recursive-rel predicate (prepare without values raises 'Cannot evaluate $_ because it does not depend on _ or e'; "
                           "prepare with sample values bakes them into the plan), so it is not used.")
    result["notes"].append("engine behaviour found while tuning (LadybugDB 0.20.2, measured on the full dataset in scratch databases): reading any Memory "
                           "property for an expanded node set is a semi-masked scan of the whole node table (~36 ms per query at 1M memories, "
                           "independent of the frontier size) unless the planner sees a small constant cardinality, which the two_stmt_in variant "
                           "provides (x.entity_id IN $ids -> per-tuple lookups; ~8 ms for frontiers <= 30 entities, ~36 ms above 500); without a join "
                           "HINT the planner scans the whole ABOUT rel table and hash-joins it with the frontier (118 ms); a REL-TABLE inverted index "
                           "(Memory-[:HAS_TERM]->Term) took 800 ms per BM25 query at full (20 ms at small), the Posting node table 40 ms single-threaded / "
                           "8 ms with all threads; node-table zone maps did not prune scans (count with term_id IN [...] slower than a full count); "
                           "loading Memory sorted by tenant gave no R1 gain.")

    queries = common.load_queries(scale)
    eng = Ladybug(scale, Path(args.db), "two_stmt_in" if args.r1 == "auto" else args.r1,
                  parse_threads(args.r1_threads), parse_threads(args.r2_threads))
    log(f"opening {args.db} (deleting previous files)")
    eng.open(delete=True)
    try:
        # ---- load (always)
        eng.load(result)
        result["notes"].append(f"peak RSS after load (engine + harness, before any numpy reference state): {common.peak_rss_mb():.0f} MB")

        # ---- A/B pre-measurements
        if not args.no_ab:
            variants = R1_VARIANTS if args.r1 == "auto" else [args.r1]
            r1_thread_opts: list[int | None] = [1, None] if args.r1_threads == "auto" else [eng.r1_threads]
            log("A/B R1 formulations / threads")
            best_v, best_t = ab_r1(eng, queries, variants, r1_thread_opts, result["notes"])
            eng.r1_variant = best_v
            eng.r1_threads = best_t
            result["notes"].append(f"chosen for the timed phases: R1 formulation={best_v}, R1 query threads per connection={thr_label(best_t)} (fastest A/B p50 with 0 mismatches)")
            log(f"chosen R1={best_v} threads={thr_label(best_t)}")
            r2_thread_opts: list[int | None] = [1, None] if args.r2_threads == "auto" else [eng.r2_threads]
            log("A/B R2 threads")
            eng.r2_threads = ab_r2(eng, queries, r2_thread_opts, result["notes"])
            result["notes"].append(f"chosen R2 query threads per connection={thr_label(eng.r2_threads)}")
            log(f"chosen R2 threads={thr_label(eng.r2_threads)}")
        else:
            result["notes"].append(f"no A/B (--no-ab): R1 formulation={eng.r1_variant}, R1 threads={thr_label(eng.r1_threads)}, R2 threads={thr_label(eng.r2_threads)}")
        r1_text = R1_SINGLE.get(eng.r1_variant) or (R1_STMT1_FRONTIER + "\n-- then --\n" + R1_STMT2[eng.r1_variant])
        result["notes"].append(f"R1 statement(s) in use ({eng.r1_variant}):\n" + r1_text)
        result["notes"].append("R2 statements: cosine top-50 = " + R2_COSINE.replace("\n", " ") + " | term ids = " + R2_TERM_IDS +
                               " | BM25 candidates (k=100, x4 on shortfall) = " + R2_BM25_CANDIDATES.replace("\n", " ") +
                               " | current-only filter = " + R2_VALID.replace("\n", " ") +
                               " | RRF k=60 in Python (common.rrf_fuse) | names = " + R2_NAMES.replace("\n", " "))
        result["notes"].append("W1 = BEGIN TRANSACTION; " + W1_CYPHER.replace("\n", " ") + "; COMMIT  |  W2 = BEGIN TRANSACTION; " + W2_CYPHER.replace("\n", " ") + "; COMMIT")

        # ---- warmup
        if "warmup" in phases:
            for qid in common.WARMUP_QUERY_IDS:
                q = queries[qid]
                eng.r1(q["tenant_id"], q["seed_entity_id"])
            for qid in list(common.WARMUP_QUERY_IDS)[:10]:
                q = queries[qid]
                eng.r2(q["tenant_id"], q["query_text"], q["query_embedding"])
            log("warmup done (50 R1 + 10 R2, not timed)")

        # ---- timed phases
        if "r1_only" in phases:
            try:
                phase_r1_only(eng, queries, result)
            except Exception as e:  # noqa: BLE001
                result["phases"]["r1_only"] = common.phase_error(e)
                log(f"r1_only FAILED: {e}")
        if "r2_only" in phases:
            try:
                phase_r2_only(eng, queries, result)
            except Exception as e:  # noqa: BLE001
                result["phases"]["r2_only"] = common.phase_error(e)
                log(f"r2_only FAILED: {e}")
        if "mixed" in phases:
            try:
                phase_mixed(eng, common.schedule(scale), result)
            except Exception as e:  # noqa: BLE001
                result["phases"]["mixed"] = common.phase_error(e)
                log(f"mixed FAILED: {e}")
        if "concurrent" in phases:
            try:
                phase_concurrent(eng, scale, result, multi_writes=not args.no_multi_writes)
            except Exception as e:  # noqa: BLE001
                result["phases"]["concurrent"] = common.phase_error(e)
                log(f"concurrent FAILED: {e}")
        if "verify" in phases:
            try:
                phase_verify(eng, queries, result)
            except Exception as e:  # noqa: BLE001
                result["verify"] = {"r1": {}, "error": common.phase_error(e)}
                log(f"verify FAILED: {e}")

        # ---- post
        try:
            eng.ex("CHECKPOINT")
            result["notes"].append(f"post: final db size after CHECKPOINT {common.db_bytes(eng.db_path) / 1e6:.0f} MB "
                                   f"(load-time size {result['load']['db_bytes'] / 1e6:.0f} MB); memories now {eng.count('Memory')}")
        except Exception as e:  # noqa: BLE001
            result["notes"].append(f"post: CHECKPOINT failed: {e}")
    finally:
        eng.close()
    result["notes"].append(f"host load average (1/5/15 min) at end: {loadavg()}")
    result["notes"] += common.DATASET_NOTES
    result["peak_rss_mb"] = common.peak_rss_mb()
    path = common.write_result(ENGINE, scale, result, Path(args.results_dir))
    log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
