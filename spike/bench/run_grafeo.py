#!/usr/bin/env python
"""anatid Phase 0 spike -- engine "grafeo" (best-effort third baseline, SPEC.md "Engine notes").

Grafeo 0.5.42 (Python package `grafeo`, Rust embedded LPG database with GQL/Cypher, HNSW, BM25, MVCC).
Implements every phase of SPEC.md's workload schedule (load, warmup, r1_only, r2_only, mixed,
concurrent, verify) and writes results/grafeo.<scale>.json via common.write_result.

    .venv/bin/python bench/run_grafeo.py --scale small
    .venv/bin/python bench/run_grafeo.py --scale full

Options (defaults are the measured-best configuration; every choice is recorded in the result "notes"):
    --phases a,b,c        subset of load,warmup,r1_only,r2_only,mixed,concurrent,verify (default all;
                          load always runs first; later phases see whatever state earlier ones left)
    --load-path save|direct
                          save   (default): bulk import into an in-memory GrafeoDB, save() it to --db,
                                 close, reopen the file (file-backed, WAL on) and build the indexes there
                          direct: import straight into the file-backed GrafeoDB (every row goes through the
                                 WAL; ~16x slower import at small scale, same query speed afterwards)
    --writes cypher|api   W1/W2 implementation (default cypher):
                          cypher: ONE GQL statement per op inside begin_transaction()/commit() (SPEC's
                                  "one transaction"); Grafeo's planner does not use property indexes in
                                  mutation statements, so each op scans the Entity (W1) or Memory (W2) label
                          api:    direct Python API calls (batch_create_nodes_with_props, find_nodes_by_property,
                                  create_edge, set_node_property); each call is its own implicit transaction,
                                  so a W1/W2 is NOT atomic -- recorded as a deviation when chosen
    --no-vector-sync      do not call set_node_property(embedding) after a cypher write (Grafeo's HNSW index
                          ignores nodes created by INSERT/CREATE statements; the extra call, timed as part of
                          the op, keeps R2's vector side live during the mixed phase)
    --vector-k N          HNSW candidates fetched per R2 (default 2000), then filtered to the tenant's
                          current memories and cut to 50 (vector_search cannot filter on valid_to IS NULL;
                          its tenant filter is ~2x slower than over-fetching, see the A/B notes)
    --text-k N            BM25 candidates fetched per R2 (default 2000; text_search has no filters at all)
    --hnsw-m N / --hnsw-ef-construction N   HNSW parameters (defaults 16 / 128 = Grafeo defaults)
    --no-ab               skip the A/B pre-measurements (alternative R1 formulation, vector filter mode,
                          the other write path)
    --db PATH             database file (default results/anatid.grafeo; `PATH.wal/` and `PATH.spill/` are
                          Grafeo's sidecars; all three are deleted before load)
    --results-dir DIR     where <engine>.<scale>.json goes (default spike/results)

Data model (one label per node table, one edge type per rel table, same columns as the Parquet files):
  (:Entity {entity_id, tenant_id, kind, name})
  (:Memory {memory_id, tenant_id, content, kind, embedding (vector 64), ctime, valid_from, valid_to,
            tx_from, tx_to, writer, episode_id, confidence})
  (:Entity)-[:RELATES_TO {edge_id, tenant_id, rel_kind, valid_from, valid_to, tx_from, tx_to}]->(:Entity)
  (:Memory)-[:ABOUT {edge_id, tenant_id, weight, valid_from, valid_to, tx_from, tx_to}]->(:Entity)
  (:Memory)-[:SUPERSEDES {edge_id, tenant_id, tx_from}]->(:Memory)
  * all timestamps are INT64 microseconds since 1970-01-01 UTC (naive-datetime parameters are shifted by
    the local UTC offset by grafeo 0.5.42; integers are exact and compare/sort correctly)
  * created_at is stored under the property name `ctime`: any identifier containing "created" in a query
    (property or alias) makes grafeo 0.5.42 take a slow path (+1.4 ms per query at 100k memories, +7 ms
    when it is a RETURN alias) -- measured on identical values under different names
  * node ids: entities are imported first so node id == entity_id; base memories get node id N_ENT + memory_id
    (both asserted after load); memories inserted later get arbitrary ids and are always addressed by memory_id

Operations:
  R1  two statements per op (Grafeo has no correct single-statement formulation, see notes):
        1. GQL UNION of the seed / 1-hop / 2-hop patterns over RELATES_TO (undirected, valid_to IS NULL,
           same tenant) -> frontier entity ids
        2. Cypher  MATCH (e:Entity) WHERE e.entity_id IN $ids WITH e MATCH (e)<-[:ABOUT]-(m:Memory)
           WHERE m.valid_to IS NULL AND m.tenant_id = $t RETURN DISTINCT m.memory_id, m.ctime
           ORDER BY ctime DESC, memory_id DESC LIMIT $limit
  R2  vector_search (HNSW, cosine) + text_search (BM25) over-fetched, filtered to current same-tenant
      memories via get_property_batch, cut to 50 each, RRF (k=60) in Python, ABOUT entity names with one
      GQL query (m.memory_id IN [...]).
  W1/W2 see --writes.

Correctness: every r1_only result is compared with common.reference_r1 (ids and created_at), every r2_only
result with common.reference_r2_truth (recall@20), the verify lists with common.reference_r1(after_mixed=True);
the counts land in the result JSON notes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import grafeo  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from grafeo import GrafeoDB  # noqa: E402

ENGINE = "grafeo"
DEFAULT_DB = common.RESULTS_DIR / "anatid.grafeo"
ALL_PHASES = ["load", "warmup", "r1_only", "r2_only", "mixed", "concurrent", "verify"]
EPOCH = dt.datetime(1970, 1, 1)
TS_COLS = ["created_at", "valid_from", "valid_to", "tx_from", "tx_to"]
# memories.parquet columns -> Grafeo property names (created_at -> ctime, see module doc)
MEM_PROPS = [("memory_id", "memory_id"), ("tenant_id", "tenant_id"), ("content", "content"), ("kind", "kind"),
             ("embedding", "embedding"), ("created_at", "ctime"), ("valid_from", "valid_from"),
             ("valid_to", "valid_to"), ("tx_from", "tx_from"), ("tx_to", "tx_to"), ("writer", "writer"),
             ("episode_id", "episode_id"), ("confidence", "confidence")]
AB_R1_QIDS = list(range(1300, 1320))          # A/B pre-measurements use mixed-phase query ids,
AB_R2_QIDS = list(range(1300, 1330))          # never the r1_only / r2_only ids
AB_WRITE_SKIP = 20_000                        # A/B writes use concurrent_w1_ops(writer 0) rows after this many

# ----------------------------------------------------------------------------- statements

# R1 step 1 (GQL): frontier = seed + 1-hop + 2-hop over current same-tenant RELATES_TO, either direction.
Q_FRONTIER = """MATCH (s:Entity {entity_id: $seed}) RETURN s.entity_id AS e
UNION
MATCH (s:Entity {entity_id: $seed})-[r1:RELATES_TO]-(h1:Entity)
WHERE r1.valid_to IS NULL AND r1.tenant_id = $t RETURN h1.entity_id AS e
UNION
MATCH (s:Entity {entity_id: $seed})-[r1:RELATES_TO]-(h1:Entity)-[r2:RELATES_TO]-(h2:Entity)
WHERE r1.valid_to IS NULL AND r1.tenant_id = $t AND r2.valid_to IS NULL AND r2.tenant_id = $t RETURN h2.entity_id AS e"""

# R1 step 2 (Cypher mode: GQL mode has no WITH). The WITH boundary is what makes the planner apply the
# IN filter on the Entity scan first ("label-first"); in a single pattern it filters after expanding
# every entity's ABOUT edges (~750 ms at small scale).
Q_MEMORIES = """MATCH (e:Entity) WHERE e.entity_id IN $ids
WITH e
MATCH (e)<-[:ABOUT]-(m:Memory) WHERE m.valid_to IS NULL AND m.tenant_id = $t
RETURN DISTINCT m.memory_id AS memory_id, m.ctime AS ctime ORDER BY ctime DESC, memory_id DESC LIMIT $limit"""

# Alternative single-statement R1 (Cypher mode). Correct, but OPTIONAL MATCH is planned as LeftJoin over
# full node scans (~360 ms at small scale) -- only used by the A/B pre-measurement.
Q_R1_SINGLE = """MATCH (s:Entity {entity_id: $seed})
OPTIONAL MATCH (s)-[r1:RELATES_TO]-(h1:Entity) WHERE r1.valid_to IS NULL AND r1.tenant_id = $t
OPTIONAL MATCH (h1)-[r2:RELATES_TO]-(h2:Entity) WHERE r2.valid_to IS NULL AND r2.tenant_id = $t
WITH collect(DISTINCT s) + collect(DISTINCT h1) + collect(DISTINCT h2) AS fs
UNWIND fs AS e
WITH DISTINCT e
MATCH (e)<-[:ABOUT]-(m:Memory) WHERE m.valid_to IS NULL AND m.tenant_id = $t
RETURN DISTINCT m.memory_id AS memory_id, m.ctime AS ctime ORDER BY ctime DESC, memory_id DESC LIMIT $limit"""

# R2: ABOUT entity names for the fused top-k (IN on the indexed memory_id is planned "label-first").
Q_NAMES = """MATCH (m:Memory)-[:ABOUT]->(e:Entity) WHERE m.memory_id IN $mids
RETURN m.memory_id AS mid, collect(e.name) AS names"""

MEM_NODE = "(m:Memory {" + ", ".join(f"{prop}: ${prop}" for _, prop in MEM_PROPS) + "})"
ABOUT_PROPS = "tenant_id: $t, weight: 1.0, valid_from: $now, valid_to: null, tx_from: $now, tx_to: null"


def w1_statement(n_dsts: int) -> str:
    """One GQL statement: look up the 1..3 ABOUT entities, insert the memory node and its ABOUT edges."""
    match = ", ".join(f"(e{j}:Entity {{entity_id: $d{j}}})" for j in range(n_dsts))
    edges = ", ".join(f"(m)-[:ABOUT {{edge_id: $eid{j}, {ABOUT_PROPS}}}]->(e{j})" for j in range(n_dsts))
    return f"MATCH {match} INSERT {MEM_NODE}, {edges}"


# One GQL statement: find the old memory, insert the new memory + SUPERSEDES edge, expire the old one.
Q_W2 = (f"MATCH (o:Memory {{memory_id: $old}}) INSERT {MEM_NODE.replace('(m:', '(n:')}"
        "-[:SUPERSEDES {edge_id: $eid, tenant_id: $t, tx_from: $now}]->(o) SET o.valid_to = $now")


# ----------------------------------------------------------------------------- helpers

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def us(d: dt.datetime | None) -> int | None:
    """naive UTC datetime -> int64 microseconds since epoch (None stays None)."""
    return None if d is None else (d - EPOCH) // dt.timedelta(microseconds=1)


def from_us(v: int) -> dt.datetime:
    return EPOCH + dt.timedelta(microseconds=int(v))


def pct(xs_ms: list[float], p: float) -> float:
    xs = sorted(xs_ms)
    return xs[min(len(xs) - 1, max(0, int(round(p / 100.0 * len(xs) + 0.5)) - 1))]


def summ(xs_ms: list[float]) -> str:
    if not xs_ms:
        return "n=0"
    return f"p50={statistics.median(xs_ms):.3f} p95={pct(xs_ms, 95):.3f} mean={statistics.fmean(xs_ms):.3f} ms (n={len(xs_ms)})"


def mem_params(m: dict) -> dict:
    """W1/W2 memory dict (memories.parquet columns) -> query parameters (ctime, int64 microseconds)."""
    p = {}
    for col, prop in MEM_PROPS:
        v = m[col]
        if col in TS_COLS:
            v = us(v)
        elif col == "embedding":
            v = [float(x) for x in v]
        elif col == "confidence" and v is not None:
            v = float(v)
        p[prop] = v
    return p


def remove_db(path: Path) -> None:
    for f in path.parent.glob(path.name + "*"):
        if f.is_dir():
            shutil.rmtree(f)
        elif f.is_file():
            f.unlink()


# ----------------------------------------------------------------------------- engine

class Grafeo:
    def __init__(self, scale: str, db_path: Path, load_path: str, writes: str, vector_sync: bool,
                 vector_k: int, text_k: int, hnsw_m: int, hnsw_ef: int):
        self.scale = scale
        self.db_path = Path(db_path)
        self.load_path = load_path
        self.writes = writes
        self.vector_sync = vector_sync
        self.vector_k = vector_k
        self.text_k = text_k
        self.hnsw_m = hnsw_m
        self.hnsw_ef = hnsw_ef
        self.db: GrafeoDB | None = None
        self.n_ent = 0
        self.counts: dict[str, int] = {}

    # --- load
    def _import(self, db: GrafeoDB) -> dict:
        t = {}
        scale = self.scale
        t0 = time.perf_counter()
        ent = common.load_table(scale, "entities")
        self.n_ent = ent.num_rows
        db.import_df(pl.from_arrow(ent), mode="nodes", label="Entity")
        t["entities"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        mem = pl.from_arrow(common.load_table(scale, "memories"))
        mem = mem.with_columns([pl.col(c).cast(pl.Int64) for c in TS_COLS]).rename({"created_at": "ctime"})
        db.import_df(mem, mode="nodes", label="Memory")
        t["memories"] = time.perf_counter() - t0
        del mem

        t0 = time.perf_counter()
        rel = pl.from_arrow(common.load_table(scale, "edges_relates"))
        rel = rel.with_columns([pl.col(c).cast(pl.Int64) for c in TS_COLS[1:]]).rename({"src": "source", "dst": "target"})
        db.import_df(rel, mode="edges", edge_type="RELATES_TO")
        t["edges_relates"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        ab = pl.from_arrow(common.load_table(scale, "edges_about"))
        ab = (ab.with_columns([pl.col(c).cast(pl.Int64) for c in TS_COLS[1:]])
                .with_columns((pl.col("src") + self.n_ent).alias("source")).rename({"dst": "target"}).drop("src"))
        db.import_df(ab, mode="edges", edge_type="ABOUT")
        t["edges_about"] = time.perf_counter() - t0
        del ab

        t0 = time.perf_counter()
        sup = pl.from_arrow(common.load_table(scale, "edges_supersedes"))
        sup = (sup.with_columns([pl.col("tx_from").cast(pl.Int64), (pl.col("src") + self.n_ent).alias("source"),
                                 (pl.col("dst") + self.n_ent).alias("target")]).drop(["src", "dst"]))
        db.import_df(sup, mode="edges", edge_type="SUPERSEDES")
        t["edges_supersedes"] = time.perf_counter() - t0
        return t

    def load(self) -> dict:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        remove_db(self.db_path)
        t_all = time.perf_counter()
        t = {}
        if self.load_path == "save":
            db = GrafeoDB()
            t.update(self._import(db))
            t0 = time.perf_counter()
            db.save(str(self.db_path))
            db.close()
            t["save"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            self.db = GrafeoDB(str(self.db_path))
            t["reopen"] = time.perf_counter() - t0
        else:
            self.db = GrafeoDB(str(self.db_path))
            t.update(self._import(self.db))
            t0 = time.perf_counter()
            self.db.wal_checkpoint()
            t["wal_checkpoint"] = time.perf_counter() - t0
        db = self.db
        t0 = time.perf_counter()
        db.create_property_index("entity_id")
        db.create_property_index("memory_id")
        t["property_indexes"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        db.create_text_index("Memory", "content")
        t["text_index"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        c0 = time.process_time()
        db.create_vector_index("Memory", "embedding", dimensions=common.DIM, metric="cosine",
                               m=self.hnsw_m, ef_construction=self.hnsw_ef)
        t["vector_index"] = time.perf_counter() - t0
        t["vector_index_cpu_s"] = time.process_time() - c0
        t["total"] = time.perf_counter() - t_all
        # sanity: node id mapping + counts
        sch = db.schema()
        self.counts = {x["name"]: x["count"] for x in sch["labels"]}
        self.counts.update({x["name"]: x["count"] for x in sch["edge_types"]})
        for eid in (0, 1, self.n_ent // 2, self.n_ent - 1):
            r = db.execute("MATCH (e:Entity {entity_id: $i}) RETURN id(e) AS i", {"i": eid}).to_list()
            assert r and r[0]["i"] == eid, f"entity node id mapping broken: {eid} -> {r}"
        n_mem = self.counts["Memory"]
        for mid in (0, 12345 % n_mem, n_mem - 1):
            r = db.execute("MATCH (m:Memory {memory_id: $i}) RETURN id(m) AS i, m.ctime AS c", {"i": mid}).to_list()
            assert r and r[0]["i"] == self.n_ent + mid, f"memory node id mapping broken: {mid} -> {r}"
        return t

    def close(self) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None

    # --- R1
    def r1(self, tenant_id: int, seed_entity_id: int, limit: int = common.R1_LIMIT) -> list[tuple[int, dt.datetime]]:
        db = self.db
        ids = [r["e"] for r in db.execute(Q_FRONTIER, {"seed": int(seed_entity_id), "t": int(tenant_id)}).to_list()]
        rows = db.execute_cypher(Q_MEMORIES, {"ids": ids, "t": int(tenant_id), "limit": int(limit)}).to_list()
        return [(r["memory_id"], from_us(r["ctime"])) for r in rows]

    def r1_single(self, tenant_id: int, seed_entity_id: int, limit: int = common.R1_LIMIT) -> list[tuple[int, dt.datetime]]:
        rows = self.db.execute_cypher(Q_R1_SINGLE, {"seed": int(seed_entity_id), "t": int(tenant_id), "limit": int(limit)}).to_list()
        return [(r["memory_id"], from_us(r["ctime"])) for r in rows]

    # --- R2
    def cosine_candidates(self, tenant_id: int, query_embedding, mode: str | None = None, k: int | None = None) -> list[int]:
        """Top-50 current same-tenant memories (node ids) by cosine via the HNSW index."""
        db = self.db
        q = [float(x) for x in query_embedding]
        if (mode or "overfetch") == "overfetch":
            vs = db.vector_search("Memory", "embedding", q, int(k or self.vector_k))
        else:  # Grafeo's own equality filter on tenant_id (cannot express valid_to IS NULL)
            vs = db.vector_search("Memory", "embedding", q, int(k or 60), filters={"tenant_id": int(tenant_id)})
        nids = [n for n, _ in vs]
        ten = db.get_property_batch(nids, "tenant_id")
        vt = db.get_property_batch(nids, "valid_to")
        return [n for n, t, v in zip(nids, ten, vt) if t == tenant_id and v is None][:common.R2_TOPN]

    def bm25_candidates(self, tenant_id: int, query_text: str) -> list[int]:
        db = self.db
        ts = db.text_search("Memory", "content", query_text, int(self.text_k))
        nids = [n for n, _ in ts]
        ten = db.get_property_batch(nids, "tenant_id")
        vt = db.get_property_batch(nids, "valid_to")
        return [n for n, t, v in zip(nids, ten, vt) if t == tenant_id and v is None][:common.R2_TOPN]

    def r2(self, tenant_id: int, query_text: str, query_embedding, k: int = common.R2_K) -> list[tuple[int, float, list[str]]]:
        db = self.db
        tenant_id = int(tenant_id)
        a = self.cosine_candidates(tenant_id, query_embedding)
        b = self.bm25_candidates(tenant_id, query_text)
        fused = common.rrf_fuse(a, b, k=common.RRF_K, top=k)
        nids = [n for n, _ in fused]
        mids = db.get_property_batch(nids, "memory_id")
        names = {r["mid"]: r["names"] for r in db.execute(Q_NAMES, {"mids": mids}).to_list()}
        return [(int(mid), float(score), list(names.get(mid, []))) for (n, score), mid in zip(fused, mids)]

    # --- W1 / W2
    def _sync_vector(self, memory_id: int, embedding: list[float]) -> None:
        # HNSW index ignores nodes created by INSERT statements; a property write re-indexes the node.
        nid = self.db.find_nodes_by_property("memory_id", int(memory_id))[0]
        self.db.set_node_property(nid, "embedding", embedding)

    def w1(self, op: dict) -> None:
        if self.writes == "api":
            return self.w1_api(op)
        m = op["memory"]
        p = mem_params(m)
        p["now"] = us(op["now"])
        p["t"] = int(op["tenant_id"])
        for j, (d, e) in enumerate(zip(op["about_dsts"], op["about_edge_ids"])):
            p[f"d{j}"] = int(d)
            p[f"eid{j}"] = int(e)
        tx = self.db.begin_transaction()
        try:
            tx.execute(w1_statement(len(op["about_dsts"])), p)
            tx.commit()
        except Exception:
            try:
                tx.rollback()
            except Exception:  # noqa: BLE001
                pass
            raise
        if self.vector_sync:
            self._sync_vector(m["memory_id"], p["embedding"])

    def w2(self, op: dict) -> None:
        if self.writes == "api":
            return self.w2_api(op)
        m = op["memory"]
        p = mem_params(m)
        p["now"] = us(op["now"])
        p["t"] = int(op["tenant_id"])
        p["old"] = int(op["old_memory_id"])
        p["eid"] = int(op["supersedes_edge_id"])
        tx = self.db.begin_transaction()
        try:
            tx.execute(Q_W2, p)
            tx.commit()
        except Exception:
            try:
                tx.rollback()
            except Exception:  # noqa: BLE001
                pass
            raise
        if self.vector_sync:
            self._sync_vector(m["memory_id"], p["embedding"])

    def w1_api(self, op: dict) -> None:
        db = self.db
        m = op["memory"]
        p = mem_params(m)
        now = us(op["now"])
        nid = db.batch_create_nodes_with_props("Memory", [p])[0]       # auto-syncs text + vector index
        for d, e in zip(op["about_dsts"], op["about_edge_ids"]):
            eid = db.find_nodes_by_property("entity_id", int(d))[0]
            db.create_edge(nid, eid, "ABOUT", {"edge_id": int(e), "tenant_id": int(op["tenant_id"]), "weight": 1.0,
                                               "valid_from": now, "valid_to": None, "tx_from": now, "tx_to": None})

    def w2_api(self, op: dict) -> None:
        db = self.db
        m = op["memory"]
        p = mem_params(m)
        now = us(op["now"])
        nid = db.batch_create_nodes_with_props("Memory", [p])[0]
        old = db.find_nodes_by_property("memory_id", int(op["old_memory_id"]))[0]
        db.set_node_property(old, "valid_to", now)
        db.create_edge(nid, old, "SUPERSEDES", {"edge_id": int(op["supersedes_edge_id"]),
                                                 "tenant_id": int(op["tenant_id"]), "tx_from": now})

    # --- checks
    def memory_row(self, memory_id: int) -> dict | None:
        r = self.db.execute("MATCH (m:Memory {memory_id: $i}) RETURN m.valid_to AS valid_to, m.ctime AS ctime, id(m) AS nid",
                            {"i": int(memory_id)}).to_list()
        return r[0] if r else None

    def about_dsts(self, memory_id: int) -> list[tuple[int, int]]:
        r = self.db.execute("MATCH (m:Memory {memory_id: $i})-[a:ABOUT]->(e:Entity) RETURN e.entity_id AS e, a.edge_id AS eid",
                            {"i": int(memory_id)}).to_list()
        return sorted((x["e"], x["eid"]) for x in r)

    def supersedes(self, new_memory_id: int) -> list[tuple[int, int, int]]:
        r = self.db.execute("MATCH (n:Memory {memory_id: $i})-[s:SUPERSEDES]->(o:Memory) RETURN o.memory_id AS o, s.edge_id AS eid, s.tx_from AS tf",
                            {"i": int(new_memory_id)}).to_list()
        return [(x["o"], x["eid"], x["tf"]) for x in r]

    def in_vector_index(self, memory_id: int, embedding) -> bool:
        nid = self.db.find_nodes_by_property("memory_id", int(memory_id))
        vs = self.db.vector_search("Memory", "embedding", [float(x) for x in embedding], 3)
        return any(n in nid for n, _ in vs)

    def in_text_index(self, memory_id: int, content: str) -> bool:
        nid = set(self.db.find_nodes_by_property("memory_id", int(memory_id)))
        ts = self.db.text_search("Memory", "content", " ".join(content.split()[:3]), 5000)
        return any(n in nid for n, _ in ts)


# ----------------------------------------------------------------------------- A/B pre-measurements

def time_r1(fn, queries: list[dict], qids, scale: str, warm: int = 5) -> tuple[list[float], int]:
    for qid in list(qids)[:warm]:
        fn(queries[qid]["tenant_id"], queries[qid]["seed_entity_id"])
    xs, bad = [], 0
    for qid in qids:
        q = queries[qid]
        t0 = time.perf_counter_ns()
        got = fn(q["tenant_id"], q["seed_entity_id"])
        xs.append((time.perf_counter_ns() - t0) / 1e6)
        ok, _ = common.compare_r1(got, common.reference_r1(scale, q["tenant_id"], q["seed_entity_id"]))
        bad += (not ok)
    return xs, bad


def ab_r1(eng: Grafeo, queries: list[dict], scale: str, notes: list[str]) -> None:
    xs2, bad2 = time_r1(eng.r1, queries, AB_R1_QIDS, scale)
    xs1, bad1 = time_r1(eng.r1_single, queries, AB_R1_QIDS, scale)
    notes.append(f"A/B R1 formulation ({len(AB_R1_QIDS)} mixed-range queries): two statements (GQL UNION frontier + Cypher IN/WITH memory step) "
                 f"{summ(xs2)}, {bad2} reference mismatches; single Cypher statement (OPTIONAL MATCH x2 + collect/UNWIND, planned as "
                 f"LeftJoin over full node scans) {summ(xs1)}, {bad1} mismatches -> two statements used for the timed phases")
    log(f"  A/B R1: two-statement {summ(xs2)} | single-statement {summ(xs1)}")


def ab_r2_vector(eng: Grafeo, queries: list[dict], scale: str, notes: list[str]) -> None:
    out = {}
    for mode, k in (("overfetch", eng.vector_k), ("overfetch", 1000), ("filter", 60)):
        xs, rec = [], []
        for qid in AB_R2_QIDS:
            q = queries[qid]
            truth = common.reference_r2_truth(scale, q)
            t0 = time.perf_counter_ns()
            nids = eng.cosine_candidates(q["tenant_id"], q["query_embedding"], mode=mode, k=k)
            xs.append((time.perf_counter_ns() - t0) / 1e6)
            mids = eng.db.get_property_batch(nids, "memory_id")
            rec.append(common.recall_at_k(truth["cosine_ids"], mids, k=common.R2_TOPN))
        out[(mode, k)] = (xs, statistics.fmean(rec) if rec else None)
    desc = "; ".join(f"{'vector_search k=' + str(k) + ' unfiltered + tenant/valid_to post-filter' if mode == 'overfetch' else 'vector_search k=60 with filters={tenant_id} + valid_to post-filter'}: "
                     f"{summ(xs)}, cosine recall@50 vs brute force {rec:.3f}" for (mode, k), (xs, rec) in out.items())
    notes.append(f"A/B R2 vector side ({len(AB_R2_QIDS)} mixed-range queries; HNSW m={eng.hnsw_m} ef_construction={eng.hnsw_ef}, "
                 f"default search ef): {desc} -> timed phases use vector_search k={eng.vector_k} unfiltered "
                 f"(filters={{'valid_to': None}} matches nothing, so the current-only filter is always applied via get_property_batch)")
    log("  A/B R2 vector: " + desc)


def ab_writes(eng: Grafeo, scale: str, notes: list[str]) -> None:
    """Time both write paths on concurrent-pool rows (created_at=1990, ids >= 1e9: never in an R1 top-20)."""
    it = common.concurrent_w1_ops(scale, 0)
    for _ in range(AB_WRITE_SKIP):
        next(it)
    ops = [next(it) for _ in range(30)]
    res = {}
    chosen = eng.writes
    for path in ("cypher", "api"):
        eng.writes = path
        xs = []
        for op in ops[:15] if path == "cypher" else ops[15:]:
            t0 = time.perf_counter_ns()
            eng.w1(op)
            xs.append((time.perf_counter_ns() - t0) / 1e6)
        res[path] = xs
    eng.writes = chosen
    # index visibility of the rows written by each path
    vis = {}
    for path, op in (("cypher", ops[0]), ("api", ops[15])):
        m = op["memory"]
        vis[path] = (eng.in_vector_index(m["memory_id"], m["embedding"]), eng.in_text_index(m["memory_id"], m["content"]))
    notes.append(f"A/B W1 write path (15 ops each on concurrent-pool rows with created_at=1990): "
                 f"cypher = one GQL MATCH..INSERT statement in begin_transaction()/commit()"
                 f"{' + set_node_property(embedding) for the HNSW index' if eng.vector_sync else ''}: {summ(res['cypher'])}, "
                 f"row visible in (vector index, text index) = {vis['cypher']}; "
                 f"api = batch_create_nodes_with_props + find_nodes_by_property + create_edge (not one transaction): {summ(res['api'])}, "
                 f"visible in (vector index, text index) = {vis['api']} -> timed phases use --writes {chosen}")
    log(f"  A/B W1: cypher {summ(res['cypher'])} | api {summ(res['api'])}")


# ----------------------------------------------------------------------------- phases

def phase_r1_only(eng: Grafeo, queries: list[dict], scale: str, notes: list[str]) -> dict:
    timer = common.Timer()
    got = {}
    c0 = time.process_time()
    with common.Stopwatch() as sw:
        for qid in common.R1_QUERY_IDS:
            q = queries[qid]
            with timer:
                got[qid] = eng.r1(q["tenant_id"], q["seed_entity_id"], common.R1_LIMIT)
    cpu = time.process_time() - c0
    out = timer.summary(wall_s=sw.seconds)
    out["cpu_wall_ratio"] = cpu / sw.seconds if sw.seconds > 0 else None
    bad = []
    for qid, rows in got.items():
        q = queries[qid]
        ok, msg = common.compare_r1(rows, common.reference_r1(scale, q["tenant_id"], q["seed_entity_id"], common.R1_LIMIT))
        if not ok:
            bad.append((qid, msg))
    out["reference_mismatches"] = len(bad)
    out["reference_checked"] = len(got)
    notes.append(f"r1_only: {len(got)} R1 results compared with common.reference_r1 (ids + created_at): {len(bad)} mismatches"
                 + (f"; first: {bad[:3]}" if bad else "")
                 + f"; process CPU / wall during the phase = {out['cpu_wall_ratio']:.2f} (Grafeo executes a query on the calling thread, single-threaded)")
    return out


def phase_r2_only(eng: Grafeo, queries: list[dict], scale: str, notes: list[str]) -> dict:
    timer = common.Timer()
    got = {}
    with common.Stopwatch() as sw:
        for qid in common.R2_QUERY_IDS:
            q = queries[qid]
            with timer:
                got[qid] = eng.r2(q["tenant_id"], q["query_text"], q["query_embedding"], common.R2_K)
    out = timer.summary(wall_s=sw.seconds)
    recalls, exact, cos_rec, bm_rec = [], 0, [], []
    for qid, rows in got.items():
        q = queries[qid]
        truth = common.reference_r2_truth(scale, q)
        ids = [r[0] for r in rows]
        recalls.append(common.recall_at_k(truth["rrf_ids"], ids))
        exact += ids == truth["rrf_ids"]
    # side-by-side recall of the two candidate lists on a subset (not timed)
    for qid in list(common.R2_QUERY_IDS)[:60]:
        q = queries[qid]
        truth = common.reference_r2_truth(scale, q)
        a = eng.db.get_property_batch(eng.cosine_candidates(q["tenant_id"], q["query_embedding"]), "memory_id")
        b = eng.db.get_property_batch(eng.bm25_candidates(q["tenant_id"], q["query_text"]), "memory_id")
        cos_rec.append(common.recall_at_k(truth["cosine_ids"], a, k=common.R2_TOPN))
        bm_rec.append(common.recall_at_k(truth["bm25_ids"], b, k=common.R2_TOPN))
    out["recall_at_20"] = statistics.fmean(recalls) if recalls else None
    out["recall_at_20_min"] = min(recalls) if recalls else None
    out["exact_top20_lists"] = f"{exact}/{len(got)}"
    out["cosine_recall_at_50"] = statistics.fmean(cos_rec) if cos_rec else None
    out["bm25_recall_at_50"] = statistics.fmean(bm_rec) if bm_rec else None
    out["vector_index"] = (f"Grafeo HNSW (create_vector_index cosine, m={eng.hnsw_m}, ef_construction={eng.hnsw_ef}); "
                           f"vector_search k={eng.vector_k} unfiltered, then tenant_id/valid_to post-filter via get_property_batch, top 50; "
                           f"incremental via the API write path or set_node_property (INSERT statements do not update it)")
    out["bm25"] = (f"Grafeo text index (create_text_index, BM25; scores identical to the reference k1=1.2 b=0.75 on the checked queries); "
                   f"text_search k={eng.text_k} (no filters available), then tenant_id/valid_to post-filter, top 50; incremental (auto-sync on writes)")
    notes.append(f"r2_only: recall@20 vs common.reference_r2_truth over all {len(got)} queries: mean {out['recall_at_20']:.4f}, "
                 f"min {out['recall_at_20_min']:.3f}, exact top-20 lists {exact}/{len(got)}; on 60 queries cosine top-50 recall "
                 f"{out['cosine_recall_at_50']:.3f} (HNSW approximation), BM25 top-50 recall {out['bm25_recall_at_50']:.3f}; "
                 f"RRF fusion (k=60) done in Python on the two 50-id lists (Grafeo's hybrid_search cannot filter by tenant/validity)")
    return out


def phase_mixed(eng: Grafeo, scale: str, notes: list[str]) -> dict:
    ops = common.schedule(scale)
    timers = {k: common.Timer() for k in ("W1", "W2", "R1", "R2")}
    checks = {}
    first_w1 = next(op for op in ops if op["op"] == "W1")
    first_w2 = next((op for op in ops if op["op"] == "W2"), None)
    with common.Stopwatch() as sw:
        for op in ops:
            k = op["op"]
            if k == "W1":
                with timers["W1"]:
                    eng.w1(op)
                if op is first_w1:   # untimed spot check right after the write
                    m = op["memory"]
                    top = eng.r1(op["tenant_id"], op["about_dsts"][0], common.R1_LIMIT)
                    checks["w1_edges_ok"] = eng.about_dsts(m["memory_id"]) == sorted(zip(op["about_dsts"], op["about_edge_ids"]))
                    checks["w1_visible_to_r1"] = bool(top) and top[0][0] == m["memory_id"] and top[0][1] == m["created_at"]
                    checks["w1_in_vector_index"] = eng.in_vector_index(m["memory_id"], m["embedding"])
                    checks["w1_in_text_index"] = eng.in_text_index(m["memory_id"], m["content"])
            elif k == "W2":
                with timers["W2"]:
                    eng.w2(op)
                if op is first_w2:
                    m = op["memory"]
                    old = eng.memory_row(op["old_memory_id"])
                    checks["w2_old_valid_to_ok"] = old is not None and old["valid_to"] == us(op["now"])
                    checks["w2_edge_ok"] = eng.supersedes(m["memory_id"]) == [(op["old_memory_id"], op["supersedes_edge_id"], us(op["now"]))]
                    checks["w2_new_current"] = (eng.memory_row(m["memory_id"]) or {}).get("valid_to", 1) is None
            elif k == "R1":
                with timers["R1"]:
                    eng.r1(op["tenant_id"], op["seed_entity_id"], op["limit"])
            else:
                with timers["R2"]:
                    eng.r2(op["tenant_id"], op["query_text"], op["query_embedding"], op["k"])
    out = {k: t.summary(wall_s=None) for k, t in timers.items()}
    out["wall_s"] = sw.seconds
    out["ops"] = len(ops)
    out["ops_per_s"] = len(ops) / sw.seconds if sw.seconds > 0 else None
    out["checks"] = checks
    notes.append(f"mixed: {len(ops)} ops in {sw.seconds:.1f} s (--writes {eng.writes}, vector sync {'on' if eng.vector_sync else 'off'}); "
                 f"spot checks after the first W1/W2: {checks}")
    if not all(checks.values()):
        notes.append("WARNING: a mixed-phase write visibility check FAILED: " + str({k: v for k, v in checks.items() if not v}))
    return out


def phase_concurrent(eng: Grafeo, scale: str, notes: list[str]) -> dict:
    def writer_factory(idx: int):
        it = common.concurrent_w1_ops(scale, idx)

        def step():
            eng.w1(next(it))
        return step

    def reader_factory(idx: int):
        it = common.concurrent_r1_queries(scale, idx)

        def step():
            op = next(it)
            return eng.r1(op["tenant_id"], op["seed_entity_id"], op["limit"])
        return step

    n_before = eng.db.node_count
    res = common.run_concurrent(common.CONCURRENT_SECONDS[scale], writer_factory, reader_factory)
    n_after = eng.db.node_count
    res["memories_inserted"] = n_after - n_before
    res["notes"] = (f"{common.N_WRITERS} writer + {common.N_READERS} reader threads sharing ONE GrafeoDB object (a second open of the same "
                    f"file fails with 'database file is locked by another process', so per-thread connections are impossible in-process); "
                    f"GrafeoDB.execute holds the GIL for the whole query, so the threads are serialized; W1 via --writes {eng.writes}"
                    f"{' (one GQL statement in begin_transaction/commit + set_node_property for the HNSW index)' if eng.writes == 'cypher' else ' (API calls, each its own implicit transaction)'}, "
                    f"no retries; {res['W1_errors']} write errors / {res['R1_errors']} read errors; {n_after - n_before} memory nodes added "
                    f"({res['W1_ops']} W1 ops counted)")
    notes.append(f"concurrent: W1 {res['W1_ops_per_s']:.1f} ops/s (p50 {res['W1_p50_ms']} ms), R1 {res['R1_ops_per_s']:.1f} ops/s "
                 f"(p50 {res['R1_p50_ms']} ms), errors {res['errors']}; {res['notes']}"
                 + (f"; error samples: {res['error_samples'][:3]}" if res["error_samples"] else ""))
    return res


def phase_verify(eng: Grafeo, queries: list[dict], scale: str, mixed_ran: bool, notes: list[str]) -> dict:
    got = {}
    for qid in common.VERIFY_QUERY_IDS:
        q = queries[qid]
        got[qid] = eng.r1(q["tenant_id"], q["seed_entity_id"], common.R1_LIMIT)
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
    ap.add_argument("--load-path", default="save", choices=["save", "direct"])
    ap.add_argument("--writes", default="cypher", choices=["cypher", "api"])
    ap.add_argument("--no-vector-sync", action="store_true")
    ap.add_argument("--vector-k", type=int, default=2000)
    ap.add_argument("--text-k", type=int, default=2000)
    ap.add_argument("--hnsw-m", type=int, default=16)
    ap.add_argument("--hnsw-ef-construction", type=int, default=128)
    ap.add_argument("--no-ab", action="store_true")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--results-dir", default=str(common.RESULTS_DIR))
    args = ap.parse_args()
    results_dir = Path(args.results_dir)

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    unknown = [p for p in phases if p not in ALL_PHASES]
    if unknown:
        ap.error(f"unknown phases {unknown}; choose from {ALL_PHASES}")

    scale = args.scale
    result = common.result_skeleton(ENGINE, grafeo.__version__, scale)
    notes: list[str] = result["notes"]
    notes.append(f"config: grafeo {grafeo.__version__} (simd {grafeo.simd_support()}) python {sys.version.split()[0]}; db={args.db}; "
                 f"load-path={args.load_path}; writes={args.writes}; vector sync after cypher writes={not args.no_vector_sync}; "
                 f"vector_k={args.vector_k}; text_k={args.text_k}; hnsw m={args.hnsw_m} ef_construction={args.hnsw_ef_construction}; phases={phases}")
    notes.append("implementation: R1 = 2 statements per op: (1) GQL `MATCH seed RETURN e UNION MATCH seed-[r1:RELATES_TO]-h1 WHERE r1.valid_to IS NULL "
                 "AND r1.tenant_id=$t RETURN h1 UNION MATCH seed-[r1]-h1-[r2]-h2 WHERE ... RETURN h2` -> frontier entity ids, "
                 "(2) Cypher `MATCH (e:Entity) WHERE e.entity_id IN $ids WITH e MATCH (e)<-[:ABOUT]-(m:Memory) WHERE m.valid_to IS NULL AND "
                 "m.tenant_id=$t RETURN DISTINCT m.memory_id, m.ctime ORDER BY ctime DESC, memory_id DESC LIMIT $limit`; "
                 "R2 = vector_search (HNSW) + text_search (BM25) over-fetched and post-filtered to the tenant's current memories via "
                 "get_property_batch, top 50 each, RRF k=60 in Python, ABOUT names via one GQL `WHERE m.memory_id IN $mids ... collect(e.name)`; "
                 "W1 = one GQL `MATCH (e0:Entity {entity_id:$d0}), ... INSERT (m:Memory {...}), (m)-[:ABOUT {...}]->(e0), ...` in "
                 "begin_transaction()/commit(); W2 = one GQL `MATCH (o:Memory {memory_id:$old}) INSERT (n:Memory {...})-[:SUPERSEDES {...}]->(o) "
                 "SET o.valid_to=$now` in a transaction; after each cypher write set_node_property(node, 'embedding', ...) re-indexes the "
                 "node in the HNSW index (INSERT statements do not); timestamps stored as INT64 microseconds; created_at stored as `ctime`")
    notes.append("grafeo 0.5.42 limitations found while building this runner (each verified against common.reference_r1): "
                 "(a) variable-length patterns `-[:RELATES_TO*0..2 {valid_to: null}]-` apply the property filter only to the first hop, and "
                 "`ALL(r IN rs WHERE r.valid_to IS NULL)` returns no rows, so the SPEC's `*0..2` formulation cannot express the validity "
                 "filter (122/150 frontier mismatches) -> explicit 1-hop/2-hop patterns; (b) ORDER BY/LIMIT after UNION binds to the last branch "
                 "only, so the frontier UNION cannot be composed with the memory lookup in one statement (no CALL {} subqueries, GQL has no "
                 "WITH/NEXT composition that avoids a full node scan); (c) property indexes only annotate equality lookups in read queries "
                 "(a lookup by indexed memory_id still costs a label scan: ~3.6 ms per 100k nodes) and are never used in INSERT/SET statements "
                 "(nested lookups are cartesian scans: a 3-entity W1 is one statement but ~3 Entity scans); (d) identifiers containing 'created' "
                 "in a query take a slow path (+1.4 ms / +7 ms per query at 100k memories) -> created_at stored and returned as `ctime`; "
                 "(e) naive datetime parameters are shifted by the local UTC offset -> INT64 microseconds everywhere; (f) the HNSW index ignores "
                 "nodes created by INSERT/CREATE statements (the text index does not) and vector_search filters cannot express IS NULL "
                 "(filters={'valid_to': None} matches nothing); (g) indexes (property/text/vector) are not persisted: they are rebuilt after "
                 "every open, and the HNSW build is single-threaded (~55 s per 100k 64-d vectors)")

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
    notes.extend(common.DATASET_NOTES)

    eng = Grafeo(scale, Path(args.db), args.load_path, args.writes, not args.no_vector_sync,
                 args.vector_k, args.text_k, args.hnsw_m, args.hnsw_ef_construction)
    queries = common.load_queries(scale)
    mixed_ran = False

    # ---- load (always)
    log(f"load {scale} -> {args.db} (load-path {args.load_path})")
    try:
        ld = eng.load()
        idx_notes = (f"import via polars import_df (nodes: Entity {ld['entities']:.1f} s, Memory {ld['memories']:.1f} s; edges: RELATES_TO "
                     f"{ld['edges_relates']:.1f} s, ABOUT {ld['edges_about']:.1f} s, SUPERSEDES {ld['edges_supersedes']:.1f} s)")
        if args.load_path == "save":
            idx_notes += f" into an in-memory GrafeoDB, save() {ld['save']:.1f} s, reopen file-backed {ld['reopen']:.1f} s"
        else:
            idx_notes += f" straight into the file-backed GrafeoDB (WAL), wal_checkpoint {ld['wal_checkpoint']:.1f} s"
        idx_notes += (f"; property indexes entity_id+memory_id {ld['property_indexes']:.2f} s; BM25 text index Memory(content) "
                      f"{ld['text_index']:.1f} s; HNSW vector index Memory(embedding) cosine m={eng.hnsw_m} ef_construction={eng.hnsw_ef} "
                      f"{ld['vector_index']:.1f} s wall / {ld['vector_index_cpu_s']:.1f} s CPU (single-threaded); indexes are not persisted "
                      f"(rebuilt on every open); node/edge counts {eng.counts}; storage tiers {eng.db.storage_tiers()}")
        result["load"] = {"seconds": ld["total"], "db_bytes": common.db_bytes(eng.db_path), "index_notes": idx_notes, "breakdown_s": ld}
        notes.append(f"peak RSS after load (engine + polars frames, before any numpy reference state): {common.peak_rss_mb():.0f} MB; "
                     "the final peak_rss_mb also includes the harness's pure-numpy reference R1/R2 states used for the correctness checks")
        log(f"  loaded in {ld['total']:.1f} s ({result['load']['db_bytes'] / 1e6:.0f} MB on disk): {idx_notes}")
    except Exception as e:  # noqa: BLE001
        result["load"] = common.phase_error(e)
        notes.append(f"load FAILED: {type(e).__name__}: {e}")
        save()
        log(f"load failed: {e}")
        return 1
    save()

    # ---- A/B pre-measurements (warm, not part of the timed phases)
    if not args.no_ab:
        log("A/B pre-measurements (mixed-range query ids, concurrent-pool write rows)")
        for fn, argv in ((ab_r1, (eng, queries, scale, notes)), (ab_r2_vector, (eng, queries, scale, notes)), (ab_writes, (eng, scale, notes))):
            try:
                fn(*argv)
            except Exception as e:  # noqa: BLE001
                notes.append(f"A/B {fn.__name__} FAILED: {type(e).__name__}: {e}")
                log(f"  A/B {fn.__name__} failed: {e}")
        save()

    # ---- warmup
    if "warmup" in phases:
        log("warmup: 50 R1 (not timed)")
        try:
            for qid in common.WARMUP_QUERY_IDS:
                q = queries[qid]
                eng.r1(q["tenant_id"], q["seed_entity_id"], common.R1_LIMIT)
        except Exception as e:  # noqa: BLE001
            notes.append(f"warmup FAILED: {type(e).__name__}: {e}")

    # ---- r1_only
    if "r1_only" in phases:
        log("r1_only: 1,000 R1 queries")
        try:
            result["phases"]["r1_only"] = phase_r1_only(eng, queries, scale, notes)
            p = result["phases"]["r1_only"]
            log(f"  R1 p50={p['p50_ms']:.3f} p95={p['p95_ms']:.3f} p99={p['p99_ms']:.3f} ms, {p['ops_per_s']:.1f} ops/s, "
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
            log(f"  R2 p50={p['p50_ms']:.3f} p95={p['p95_ms']:.3f} ms, {p['ops_per_s']:.1f} ops/s, recall@20={p['recall_at_20']:.4f}")
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
                + f", wall {m['wall_s']:.1f} s, checks {m['checks']}")
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

    # ---- post: final size, WAL status
    try:
        st = eng.db.wal_status()
        result["load"]["db_bytes_final"] = common.db_bytes(eng.db_path)
        notes.append(f"final on-disk size {result['load']['db_bytes_final'] / 1e6:.0f} MB (file + .wal/ + .spill/); WAL status after all phases: "
                     f"{ {k: st.get(k) for k in ('size_bytes', 'record_count', 'current_epoch')} }; engine memory_usage total "
                     f"{eng.db.memory_usage()['total_bytes'] / 1e6:.0f} MB")
    except Exception as e:  # noqa: BLE001
        notes.append(f"post-run status failed: {type(e).__name__}: {e}")
    eng.close()
    try:
        # GrafeoDB.close() folds the WAL into the main file and removes the .wal/ and .spill/ sidecars
        result["load"]["db_bytes_after_close"] = common.db_bytes(eng.db_path)
        notes.append(f"on-disk size after close() (WAL folded into the file, sidecars removed): {result['load']['db_bytes_after_close'] / 1e6:.0f} MB; "
                     f"load.db_bytes ({result['load']['db_bytes'] / 1e6:.0f} MB) was measured right after load with the WAL sidecar open")
    except Exception as e:  # noqa: BLE001
        notes.append(f"post-close size failed: {type(e).__name__}: {e}")
    save()
    log(f"done -> {results_dir / f'{ENGINE}.{scale}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
