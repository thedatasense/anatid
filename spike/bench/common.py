"""Shared harness for the anatid Phase 0 spike (contract: spike/SPEC.md).

Every engine runner imports this module and ONLY this module for: dataset paths and rows, the
deterministic workload schedule, timing/metrics, the result JSON writer, the concurrency driver,
and the pure-numpy reference implementations of R1/R2 that runners verify themselves against.

Import it either as `import common` (runner executed as a script from bench/) or
`from bench import common` (spike/ on sys.path). It has no dependency on any engine.

Quick tour
----------
    ops = schedule("small")                       # mixed-phase op list (1,000 ops; 10,000 at full)
    queries = load_queries("small")               # queries[qid] -> dict(query_id, tenant_id, seed_entity_id, query_text, query_embedding)
    for qid in R1_QUERY_IDS: ...                  # r1_only: 1,000 R1 queries -> compare with reference_r1(...)
    for qid in R2_QUERY_IDS: ...                  # r2_only: 300 R2 queries  -> recall_at_k(reference_r2_truth(...)["rrf_ids"], got_ids)
    t = Timer(); with t: rows = run_r1(...)      # perf_counter_ns around the whole op incl. materialization
    phase = t.summary(wall_s=...)                 # count/p50/p95/p99/mean/ops_per_s/wall_s + raw_ms
    conc = run_concurrent(CONCURRENT_SECONDS[scale], writer_factory, reader_factory)
    write_result("duckdb_sql", "small", result)  # -> results/duckdb_sql.small.json (SPEC shape)

Write-op conventions (so every engine applies identical writes and verify can compare them):
  * W1 op: insert op["memory"] (a dict with exactly the memories.parquet columns; valid_from and
    tx_from are already set to op["now"], valid_to/tx_to None, created_at from the dataset), then one
    ABOUT edge per (dst, edge_id) in zip(op["about_dsts"], op["about_edge_ids"]) with
    tenant_id = memory tenant, weight = 1.0, valid_from = tx_from = op["now"], valid_to = tx_to = NULL.
  * W2 op: insert op["memory"] the same way, UPDATE the old memory (op["old_memory_id"]) setting
    valid_to = op["now"] (nothing else), insert one SUPERSEDES edge (edge_id = op["supersedes_edge_id"],
    src = new memory_id, dst = old_memory_id, tenant_id, tx_from = op["now"]). NO ABOUT edges in W2.
  * `now` is deterministic: op["now"] = T0 + (seq + 1) seconds, and the new memory's created_at is
    set to `now` as well (created_at = valid_from = tx_from = now), so all engines hold identical data.
  * Concurrent-phase W1 rows (concurrent_w1_ops) carry created_at = 1990-01-01 and memory ids >= 1e9
    (edge ids >= 2e9), so however many of them an engine manages to insert they never enter an R1
    top-20 (every query frontier holds >= 30 newer current memories) and verify stays comparable.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import platform
import resource
import socket
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# ----------------------------------------------------------------------------- constants

SPIKE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SPIKE_DIR / "data"
RESULTS_DIR = SPIKE_DIR / "results"

SCALES = ("small", "full")
FILES = ("entities", "memories", "edges_about", "edges_relates", "edges_supersedes",
         "queries", "writes_memories", "writes_supersedes")
N_TENANTS = 10
DIM = 64
T0 = _dt.datetime(2025, 1, 1)                     # dataset epoch: all base rows are created before T0

# workload schedule (SPEC "Workload schedule")
WARMUP_QUERY_IDS = range(1950, 2000)              # 50 R1 queries, not timed, disjoint from r1_only
R1_QUERY_IDS = range(0, 1000)                     # r1_only, the kill-criterion measurement
R2_QUERY_IDS = range(1000, 1300)                  # r2_only
MIXED_QUERY_IDS = range(1300, 2000)               # reads in the mixed phase (wrapping)
VERIFY_QUERY_IDS = range(0, 200)                  # re-run after mixed+concurrent, dumped into the JSON
MIXED_OPS = {"small": 1_000, "full": 10_000}
CONCURRENT_SECONDS = {"small": 10, "full": 30}
N_WRITERS, N_READERS = 4, 2
R1_LIMIT = 20
R2_K = 20
R2_TOPN = 50                                      # size of the cosine and BM25 candidate lists
RRF_K = 60                                        # rrf score = sum 1/(RRF_K + rank), rank is 1-based
BM25_K1, BM25_B = 1.2, 0.75

# id allocation for rows inserted by the harness (base ABOUT edge_ids are 0..~2M, memories 0..1.1M)
WRITE_EDGE_ID_BASE = 100_000_000                  # mixed phase: about edge_id = BASE + 3*i + j, supersedes edge_id = BASE + 3*i
CONCURRENT_ID_BASE = 1_000_000_000                # concurrent phase memory_id = BASE + writer*STRIDE + seq
CONCURRENT_EDGE_ID_BASE = 2_000_000_000           # concurrent phase about edge_id = BASE + writer*STRIDE + 3*seq + j
CONCURRENT_STRIDE = 30_000_000
CONCURRENT_CREATED_AT = _dt.datetime(1990, 1, 1)  # older than every dataset memory (see module doc)

DATASET_NOTES = [
    "dataset: queries.parquet has 2,000 rows at both scales (schedule addresses query_ids 0..1999); everything else is /10 at small",
    "dataset: query_text is 2..3 distinct topic words of the seed (an entity owns 3 topic words; SPEC says 2..4)",
    "dataset: writes_supersedes columns are new_memory_id, old_memory_id, tenant_id; new ids are every 10th writes_memories row, the other 90% are the W1 pool",
    "dataset: RELATES_TO out-degree ~ Zipf(2.2) capped at 500 -> ~2.8 avg out-degree (~280k edges at full), 5% of RELATES_TO edges expired; ABOUT edges all current",
    "dataset: query seeds are re-drawn until their 2-hop frontier holds >= 30 current memories",
    "harness: `now` for mixed-phase op seq is T0 + (seq+1) s (deterministic, not wall clock); the written memory's created_at = valid_from = tx_from = now",
    "harness: concurrent-phase W1 rows use created_at=1990-01-01 and memory ids >= 1e9 (edge ids >= 2e9) so they cannot enter any R1 top-20 and verify stays comparable across engines",
]


# ----------------------------------------------------------------------------- dataset access

def data_dir(scale: str) -> Path:
    """Directory holding the Parquet files for `scale` ('small' | 'full')."""
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
    return DATA_DIR / scale


def dataset_path(scale: str, name: str) -> Path:
    """Path of one Parquet file, e.g. dataset_path('full', 'memories') -> spike/data/full/memories.parquet."""
    if name not in FILES:
        raise ValueError(f"unknown dataset file {name!r}; expected one of {FILES}")
    p = data_dir(scale) / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run: .venv/bin/python bench/gen_dataset.py --scale {scale}")
    return p


def load_table(scale: str, name: str, columns: list[str] | None = None) -> pa.Table:
    """Read one dataset file as a pyarrow Table (optionally a column subset)."""
    return pq.read_table(dataset_path(scale, name), columns=columns)


def _to_py(v: Any) -> Any:
    """pyarrow/numpy scalar -> plain Python (datetimes naive UTC, lists of floats for embeddings)."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, _dt.datetime) and v.tzinfo is not None:
        return v.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    return v


def _rows(table: pa.Table, indices: np.ndarray | None = None) -> list[dict]:
    if indices is not None:
        table = table.take(pa.array(np.asarray(indices, dtype=np.int64)))
    return [{k: _to_py(v) for k, v in r.items()} for r in table.to_pylist()]


_QUERIES: dict[str, list[dict]] = {}


def load_queries(scale: str) -> list[dict]:
    """All 2,000 queries as dicts, list index == query_id.
    Keys: query_id, tenant_id, seed_entity_id, query_text, query_embedding (list[float], 64, unit norm)."""
    if scale not in _QUERIES:
        rows = _rows(load_table(scale, "queries"))
        assert [r["query_id"] for r in rows] == list(range(len(rows)))
        _QUERIES[scale] = rows
    return _QUERIES[scale]


def load_writes(scale: str) -> dict[str, list[dict]]:
    """Every write row as Python dicts: {"memories": [...], "supersedes": [...]}.
    memories rows have the memories.parquet columns plus about_dsts (list[int]).
    supersedes rows: new_memory_id, old_memory_id, tenant_id, new_row (the matching memories row).
    schedule() is what runners normally use; this is the raw pool (100k / 10k rows, ~2 s at full)."""
    mems = _rows(load_table(scale, "writes_memories"))
    by_id = {m["memory_id"]: m for m in mems}
    sups = _rows(load_table(scale, "writes_supersedes"))
    for s in sups:
        s["new_row"] = by_id[s["new_memory_id"]]
    return {"memories": mems, "supersedes": sups}


# ----------------------------------------------------------------------------- schedule

def _memory_for_insert(row: dict, now: _dt.datetime, created_at: _dt.datetime | None = None) -> dict:
    m = {k: row[k] for k in ("memory_id", "tenant_id", "content", "kind", "embedding", "created_at",
                             "valid_from", "valid_to", "tx_from", "tx_to", "writer", "episode_id", "confidence")}
    m["created_at"] = now if created_at is None else created_at
    m["valid_from"] = now
    m["tx_from"] = now
    m["valid_to"] = None
    m["tx_to"] = None
    return m


def _w1_op(row: dict, seq: int, now: _dt.datetime, edge_ids: list[int], created_at: _dt.datetime | None = None) -> dict:
    return {"op": "W1", "seq": seq, "now": now, "memory": _memory_for_insert(row, now, created_at),
            "tenant_id": row["tenant_id"], "about_dsts": list(row["about_dsts"]),
            "about_edge_ids": edge_ids[: len(row["about_dsts"])]}


def schedule(scale: str, seed: int = 7) -> list[dict]:
    """Deterministic mixed-phase op list (SPEC step 5): MIXED_OPS[scale] ops, 70% writes / 30% reads,
    writes 90% W1 / 10% W2, reads 80% R1 / 20% R2. W1 rows come in order from the W1 pool
    (writes_memories rows with index % 10 != 0), W2 rows in order from writes_supersedes, read
    query_ids cycle through MIXED_QUERY_IDS. Every op dict has "op", "seq", "now" (= T0 + (seq+1) s);
    W1/W2 memory rows have created_at = valid_from = tx_from = now. See module doc
    for the per-op payload ("memory", "about_dsts", "about_edge_ids" | "old_memory_id",
    "supersedes_edge_id" | "query_id", "tenant_id", "seed_entity_id", "limit" | "query_text",
    "query_embedding", "k")."""
    n_ops = MIXED_OPS[scale]
    rng = np.random.default_rng(seed)
    u = rng.random((n_ops, 2))
    is_write = u[:, 0] < 0.7
    kinds = np.where(is_write, np.where(u[:, 1] < 0.9, "W1", "W2"), np.where(u[:, 1] < 0.8, "R1", "R2"))
    n_w1, n_w2 = int((kinds == "W1").sum()), int((kinds == "W2").sum())

    wm = load_table(scale, "writes_memories")
    n_mem_base = int(wm["memory_id"][0].as_py())
    w1_pool = np.flatnonzero(np.arange(wm.num_rows) % 10 != 0)
    if n_w1 > len(w1_pool):
        raise RuntimeError(f"schedule needs {n_w1} W1 rows, pool has {len(w1_pool)}")
    w1_rows = _rows(wm, w1_pool[:n_w1])
    ws = load_table(scale, "writes_supersedes")
    if n_w2 > ws.num_rows:
        raise RuntimeError(f"schedule needs {n_w2} W2 rows, pool has {ws.num_rows}")
    ws_rows = _rows(ws.slice(0, n_w2))
    w2_new_rows = _rows(wm, np.array([r["new_memory_id"] - n_mem_base for r in ws_rows], dtype=np.int64))
    queries = load_queries(scale)

    ops: list[dict] = []
    i1 = i2 = ir = 0
    for seq, kind in enumerate(kinds.tolist()):
        now = T0 + _dt.timedelta(seconds=seq + 1)
        if kind == "W1":
            row = w1_rows[i1]
            i1 += 1
            i = row["memory_id"] - n_mem_base
            ops.append(_w1_op(row, seq, now, [WRITE_EDGE_ID_BASE + 3 * i + j for j in range(3)]))
        elif kind == "W2":
            s, row = ws_rows[i2], w2_new_rows[i2]
            i2 += 1
            assert row["memory_id"] == s["new_memory_id"]
            i = row["memory_id"] - n_mem_base
            ops.append({"op": "W2", "seq": seq, "now": now, "memory": _memory_for_insert(row, now),
                        "tenant_id": row["tenant_id"], "old_memory_id": s["old_memory_id"],
                        "supersedes_edge_id": WRITE_EDGE_ID_BASE + 3 * i})
        else:
            q = queries[MIXED_QUERY_IDS[ir % len(MIXED_QUERY_IDS)]]
            ir += 1
            if kind == "R1":
                ops.append({"op": "R1", "seq": seq, "now": now, "query_id": q["query_id"],
                            "tenant_id": q["tenant_id"], "seed_entity_id": q["seed_entity_id"], "limit": R1_LIMIT})
            else:
                ops.append({"op": "R2", "seq": seq, "now": now, "query_id": q["query_id"],
                            "tenant_id": q["tenant_id"], "query_text": q["query_text"],
                            "query_embedding": q["query_embedding"], "k": R2_K})
    return ops


def schedule_counts(ops: list[dict]) -> dict[str, int]:
    """{'W1': n, 'W2': n, 'R1': n, 'R2': n} for an op list."""
    out = {"W1": 0, "W2": 0, "R1": 0, "R2": 0}
    for op in ops:
        out[op["op"]] += 1
    return out


def concurrent_w1_ops(scale: str, writer_idx: int) -> Iterator[dict]:
    """Infinite iterator of W1 ops for concurrent writer `writer_idx` (0..3). Content/embedding/about_dsts
    cycle through the writes_memories pool; memory_id (>= CONCURRENT_ID_BASE) and about edge_ids
    (>= CONCURRENT_EDGE_ID_BASE) are unique per writer, created_at = CONCURRENT_CREATED_AT,
    now = valid_from = tx_from = T0 + 365 d + seq s."""
    wm = load_table(scale, "writes_memories")
    n = wm.num_rows
    start = (writer_idx * 7919) % n
    seq = 0
    while True:
        idx = np.arange(start, start + 512) % n
        start = int((start + 512) % n)
        for row in _rows(wm, idx):
            row["memory_id"] = CONCURRENT_ID_BASE + writer_idx * CONCURRENT_STRIDE + seq
            now = T0 + _dt.timedelta(days=365, seconds=seq)
            ebase = CONCURRENT_EDGE_ID_BASE + writer_idx * CONCURRENT_STRIDE
            edge_ids = [ebase + 3 * seq + j for j in range(3)]
            yield _w1_op(row, seq, now, edge_ids, created_at=CONCURRENT_CREATED_AT)
            seq += 1


def concurrent_r1_queries(scale: str, reader_idx: int) -> Iterator[dict]:
    """Infinite iterator of R1 op dicts for concurrent reader `reader_idx`, cycling all 2,000 queries."""
    queries = load_queries(scale)
    i = (reader_idx * 997) % len(queries)
    seq = 0
    while True:
        q = queries[i % len(queries)]
        i += 1
        yield {"op": "R1", "seq": seq, "now": None, "query_id": q["query_id"], "tenant_id": q["tenant_id"],
               "seed_entity_id": q["seed_entity_id"], "limit": R1_LIMIT}
        seq += 1


# ----------------------------------------------------------------------------- timing & metrics

class Timer:
    """Collects per-op latencies in ns. Use `with timer: ...` around one op (including materializing the
    result to Python lists) or `timer.time(fn, *args)`. `.ns` is the raw list; `.summary()` -> phase dict."""

    def __init__(self) -> None:
        self.ns: list[int] = []
        self._t0 = 0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter_ns()
        return self

    def __exit__(self, *exc) -> None:
        self.ns.append(time.perf_counter_ns() - self._t0)

    def time(self, fn: Callable, *args, **kwargs):
        t0 = time.perf_counter_ns()
        out = fn(*args, **kwargs)
        self.ns.append(time.perf_counter_ns() - t0)
        return out

    def summary(self, wall_s: float | None = None) -> dict:
        return summarize(self.ns, wall_s)


class Stopwatch:
    """`with Stopwatch() as sw: ...; sw.seconds` -- wall time of a whole phase."""

    def __enter__(self) -> "Stopwatch":
        self._t0 = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0


def summarize(latencies_ns: list[int], wall_s: float | None = None) -> dict:
    """Phase metrics dict in the SPEC shape: count, p50_ms, p95_ms, p99_ms, mean_ms, ops_per_s, wall_s,
    plus raw_ms (every latency, ms, 4 decimals) so auditors can recompute. ops_per_s = count / wall_s
    when wall_s is given, else count / sum(latencies)."""
    ms = np.asarray(latencies_ns, dtype=np.float64) / 1e6
    n = int(ms.size)
    if n == 0:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "mean_ms": None,
                "ops_per_s": None, "wall_s": wall_s, "raw_ms": []}
    total_s = float(ms.sum() / 1000.0)
    wall = float(wall_s) if wall_s is not None else total_s
    return {
        "count": n,
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "mean_ms": float(ms.mean()),
        "ops_per_s": (n / wall) if wall > 0 else None,
        "wall_s": wall,
        "raw_ms": [round(float(x), 4) for x in ms],
    }


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MiB (resource.getrusage; bytes on macOS, KiB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def db_bytes(path: str | os.PathLike) -> int:
    """Size on disk of a database: a directory (recursive) or a file plus its sidecars
    (anything in the same directory whose name starts with the file's name, e.g. .wal)."""
    p = Path(path)
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    total = 0
    for f in p.parent.glob(p.name + "*"):
        if f.is_file():
            total += f.stat().st_size
        elif f.is_dir():
            total += sum(x.stat().st_size for x in f.rglob("*") if x.is_file())
    return total


def host_info() -> str:
    """One-line host description for the result JSON."""
    return f"{socket.gethostname()} {platform.platform()} {platform.machine()} cpus={os.cpu_count()} python={platform.python_version()}"


def recall_at_k(truth_ids: list[int], got_ids: list[int], k: int = 20) -> float:
    """|truth[:k] ∩ got[:k]| / min(k, len(truth[:k])) (1.0 when truth is empty)."""
    t = list(truth_ids)[:k]
    if not t:
        return 1.0
    return len(set(t) & set(list(got_ids)[:k])) / len(t)


def rrf_fuse(*ranked_lists: list[int], k: int = RRF_K, top: int = R2_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(id) = sum over lists of 1/(k + rank), rank 1-based.
    Returns the top `top` (id, score) ordered by score desc, id asc."""
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, mid in enumerate(lst, start=1):
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top]


# ----------------------------------------------------------------------------- result JSON

def result_skeleton(engine: str, engine_version: str, scale: str) -> dict:
    """Empty result dict in the SPEC shape; fill phases as you go and hand it to write_result()."""
    return {
        "engine": engine, "engine_version": engine_version, "scale": scale, "host": host_info(),
        "load": {"seconds": None, "db_bytes": None, "index_notes": ""},
        "phases": {"r1_only": None, "r2_only": None, "mixed": None, "concurrent": None},
        "verify": {"r1": {}},
        "peak_rss_mb": None,
        "notes": [],
    }


def phase_error(exc: BaseException) -> dict:
    """Record a failed phase (SPEC rule: errors are recorded, never skipped)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def verify_payload(r1_by_qid: dict[int, list]) -> dict:
    """{"r1": {"<qid>": [memory_id, ...]}} from {qid: rows} where rows are (memory_id, created_at) tuples
    or bare ids."""
    out = {}
    for qid, rows in r1_by_qid.items():
        out[str(int(qid))] = [int(r[0] if isinstance(r, (tuple, list)) else r) for r in rows]
    return {"r1": out}


def _json_default(o: Any) -> Any:
    if isinstance(o, (_dt.datetime, _dt.date)):
        return o.isoformat()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, BaseException):
        return f"{type(o).__name__}: {o}"
    return str(o)


def write_result(engine: str, scale: str, result: dict, results_dir: Path | None = None) -> Path:
    """Write results/<engine>.<scale>.json in the SPEC shape (fills engine/scale/host/peak_rss_mb/
    written_at if absent, appends nothing else). Returns the path."""
    results_dir = Path(results_dir or RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = dict(result)
    out["engine"] = engine
    out["scale"] = scale
    out.setdefault("engine_version", "unknown")
    out.setdefault("host", host_info())
    out.setdefault("load", {"seconds": None, "db_bytes": None, "index_notes": ""})
    out.setdefault("phases", {})
    out.setdefault("verify", {"r1": {}})
    if out.get("peak_rss_mb") is None:
        out["peak_rss_mb"] = peak_rss_mb()
    out.setdefault("notes", [])
    out["written_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    path = results_dir / f"{engine}.{scale}.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, default=_json_default)
    os.replace(tmp, path)
    return path


# ----------------------------------------------------------------------------- concurrency

def run_concurrent(seconds: float, writer_fn_factory: Callable[[int], Callable[[], Any]],
                   reader_fn_factory: Callable[[int], Callable[[], Any]],
                   n_writers: int = N_WRITERS, n_readers: int = N_READERS) -> dict:
    """SPEC step 6. Spawns n_writers + n_readers threads. Inside each thread the factory is called with
    the thread index and must return a zero-arg callable that performs ONE op (W1 for writers, R1 for
    readers) -- open the thread's own connection inside the factory. All threads start together and
    loop until `seconds` elapse; exceptions are counted as errors (first 5 messages kept) and the loop
    continues. Returns {"W1_ops_per_s", "R1_ops_per_s", "errors", "notes", ...} plus per-type
    counts, error counts, p50/p95 latencies and error samples."""
    recs: dict[tuple[str, int], dict] = {}
    barrier = threading.Barrier(n_writers + n_readers + 1)

    def worker(kind: str, idx: int, factory):
        rec = {"kind": kind, "idx": idx, "ops": 0, "errors": 0, "samples": [], "lat_ns": [], "elapsed": 0.0}
        recs[(kind, idx)] = rec
        step = None
        try:
            step = factory(idx)
        except Exception as e:  # noqa: BLE001
            rec["errors"] += 1
            rec["samples"].append(f"factory({idx}) failed: {type(e).__name__}: {e}")
        try:
            barrier.wait(timeout=120)
        except threading.BrokenBarrierError:
            pass
        if step is None:
            return
        t_start = time.perf_counter()
        deadline = t_start + seconds
        while time.perf_counter() < deadline:
            t = time.perf_counter_ns()
            try:
                step()
                rec["ops"] += 1
                rec["lat_ns"].append(time.perf_counter_ns() - t)
            except Exception as e:  # noqa: BLE001
                rec["errors"] += 1
                if len(rec["samples"]) < 5:
                    rec["samples"].append(f"{type(e).__name__}: {e}"[:500])
                if rec["ops"] == 0 and rec["errors"] >= 1000:
                    rec["samples"].append("giving up: 1000 consecutive errors before the first success")
                    break
        rec["elapsed"] = time.perf_counter() - t_start

    threads = [threading.Thread(target=worker, args=("W1", i, writer_fn_factory), daemon=True) for i in range(n_writers)]
    threads += [threading.Thread(target=worker, args=("R1", i, reader_fn_factory), daemon=True) for i in range(n_readers)]
    for t in threads:
        t.start()
    try:
        barrier.wait(timeout=120)
    except threading.BrokenBarrierError:
        pass
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    elapsed = max(time.perf_counter() - t0, 1e-9)

    def agg(kind: str) -> dict:
        rs = [r for r in recs.values() if r["kind"] == kind]
        lat = [x for r in rs for x in r["lat_ns"]]
        s = summarize(lat, wall_s=elapsed)
        s.pop("raw_ms", None)
        return {"ops": sum(r["ops"] for r in rs), "errors": sum(r["errors"] for r in rs),
                "p50_ms": s["p50_ms"], "p95_ms": s["p95_ms"], "p99_ms": s["p99_ms"], "mean_ms": s["mean_ms"],
                "threads_with_zero_ops": sum(1 for r in rs if r["ops"] == 0)}

    w, r = agg("W1"), agg("R1")
    samples = [f"{k[0]}[{k[1]}] {m}" for k, rec in sorted(recs.items()) for m in rec["samples"]]
    return {
        "W1_ops_per_s": w["ops"] / elapsed, "R1_ops_per_s": r["ops"] / elapsed,
        "errors": w["errors"] + r["errors"], "notes": "",
        "seconds": seconds, "elapsed_s": elapsed, "n_writers": n_writers, "n_readers": n_readers,
        "W1_ops": w["ops"], "R1_ops": r["ops"], "W1_errors": w["errors"], "R1_errors": r["errors"],
        "W1_p50_ms": w["p50_ms"], "W1_p95_ms": w["p95_ms"], "R1_p50_ms": r["p50_ms"], "R1_p95_ms": r["p95_ms"],
        "W1_threads_with_zero_ops": w["threads_with_zero_ops"], "R1_threads_with_zero_ops": r["threads_with_zero_ops"],
        "error_samples": samples[:20],
    }


# ----------------------------------------------------------------------------- reference R1

_REF: dict[tuple[str, bool], SimpleNamespace] = {}
_REF_LOCK = threading.Lock()


def _ts_us(col: pa.ChunkedArray) -> np.ndarray:
    return col.to_numpy(zero_copy_only=False).astype("datetime64[us]").astype(np.int64)


def _is_null(col: pa.ChunkedArray) -> np.ndarray:
    return pc.is_null(col).to_numpy(zero_copy_only=False)


def _csr(u: np.ndarray, v: np.ndarray, w: np.ndarray, n_nodes: int):
    order = np.argsort(u, kind="stable")
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(u, minlength=n_nodes), out=indptr[1:])
    return indptr, v[order], w[order]


def _expand(indptr: np.ndarray, adj: np.ndarray, adj_tenant: np.ndarray, nodes: np.ndarray, tenant_id: int) -> np.ndarray:
    """All adjacency entries of `nodes` whose edge tenant == tenant_id (with repeats)."""
    if len(nodes) == 0:
        return np.empty(0, dtype=np.int64)
    starts, lens = indptr[nodes], indptr[nodes + 1] - indptr[nodes]
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    idx = np.repeat(starts, lens) + (np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens))
    idx = idx[adj_tenant[idx] == tenant_id]
    return adj[idx]


def _ref_state(scale: str, after_mixed: bool) -> SimpleNamespace:
    key = (scale, after_mixed)
    st = _REF.get(key)
    if st is not None:
        return st
    with _REF_LOCK:
        if key in _REF:
            return _REF[key]
        t0 = time.perf_counter()
        mem = load_table(scale, "memories", ["memory_id", "tenant_id", "created_at", "valid_to"])
        wm = load_table(scale, "writes_memories", ["memory_id", "tenant_id", "created_at"])
        n_mem, n_w = mem.num_rows, wm.num_rows
        n_all = n_mem + n_w
        assert np.array_equal(mem["memory_id"].to_numpy(), np.arange(n_mem))
        assert np.array_equal(wm["memory_id"].to_numpy(), n_mem + np.arange(n_w))
        tenant = np.concatenate([mem["tenant_id"].to_numpy(), wm["tenant_id"].to_numpy()]).astype(np.int32)
        created = np.concatenate([_ts_us(mem["created_at"]), _ts_us(wm["created_at"])])
        current = np.zeros(n_all, dtype=bool)
        current[:n_mem] = _is_null(mem["valid_to"])
        exists = np.zeros(n_all, dtype=bool)
        exists[:n_mem] = True

        ent = load_table(scale, "entities", ["entity_id", "tenant_id"])
        n_ent = ent.num_rows
        rel = load_table(scale, "edges_relates", ["src", "dst", "tenant_id", "valid_to"])
        cur = _is_null(rel["valid_to"])
        rs, rd, rt = rel["src"].to_numpy()[cur], rel["dst"].to_numpy()[cur], rel["tenant_id"].to_numpy()[cur]
        ab = load_table(scale, "edges_about", ["src", "dst", "tenant_id"])
        a_mem, a_ent, a_ten = ab["src"].to_numpy(), ab["dst"].to_numpy(), ab["tenant_id"].to_numpy()

        if after_mixed:
            add_mem, add_ent, add_ten = [], [], []
            epoch = _dt.datetime(1970, 1, 1)
            for op in schedule(scale):
                if op["op"] == "W1":
                    m = op["memory"]["memory_id"]
                    exists[m] = current[m] = True
                    created[m] = (op["memory"]["created_at"] - epoch) // _dt.timedelta(microseconds=1)
                    for d in op["about_dsts"]:
                        add_mem.append(m)
                        add_ent.append(d)
                        add_ten.append(op["tenant_id"])
                elif op["op"] == "W2":
                    m = op["memory"]["memory_id"]
                    exists[m] = current[m] = True
                    created[m] = (op["memory"]["created_at"] - epoch) // _dt.timedelta(microseconds=1)
                    current[op["old_memory_id"]] = False
            a_mem = np.concatenate([a_mem, np.asarray(add_mem, dtype=np.int64)])
            a_ent = np.concatenate([a_ent, np.asarray(add_ent, dtype=np.int64)])
            a_ten = np.concatenate([a_ten, np.asarray(add_ten, dtype=np.int32)])

        r_indptr, r_adj, r_ten = _csr(np.concatenate([rs, rd]), np.concatenate([rd, rs]), np.concatenate([rt, rt]), n_ent)
        a_indptr, a_adj, a_eten = _csr(a_ent, a_mem, a_ten, n_ent)
        st = SimpleNamespace(scale=scale, after_mixed=after_mixed, n_ent=n_ent, n_mem=n_mem,
                             ent_tenant=ent["tenant_id"].to_numpy(), tenant=tenant, created=created,
                             current=current, exists=exists, r_indptr=r_indptr, r_adj=r_adj, r_ten=r_ten,
                             a_indptr=a_indptr, a_adj=a_adj, a_ten=a_eten, build_s=time.perf_counter() - t0)
        _REF[key] = st
        return st


def reference_frontier(scale: str, tenant_id: int, seed_entity_id: int, after_mixed: bool = False) -> list[int]:
    """Sorted entity ids of the R1 frontier: seed + 1-hop + 2-hop over current (valid_to IS NULL),
    same-tenant RELATES_TO edges in either direction. Handy for checking graph_expand()."""
    st = _ref_state(scale, after_mixed)
    return sorted(int(x) for x in _frontier(st, int(tenant_id), int(seed_entity_id)))


def _frontier(st: SimpleNamespace, tenant_id: int, seed: int) -> np.ndarray:
    if seed < 0 or seed >= st.n_ent:
        return np.array([seed], dtype=np.int64)
    s = np.array([seed], dtype=np.int64)
    h1 = np.unique(_expand(st.r_indptr, st.r_adj, st.r_ten, s, tenant_id))
    h2 = _expand(st.r_indptr, st.r_adj, st.r_ten, h1, tenant_id)
    return np.unique(np.concatenate([s, h1, h2]))


def reference_r1(scale: str, tenant_id: int, seed_entity_id: int, limit: int = R1_LIMIT,
                 after_mixed: bool = False) -> list[tuple[int, _dt.datetime]]:
    """Authoritative R1 straight from the Parquet files (SPEC semantics, pure numpy, cached per scale):
    memories ABOUT any frontier entity with valid_to IS NULL and tenant_id = tenant_id, ordered by
    created_at DESC, memory_id DESC, first `limit`. Returns [(memory_id, created_at naive UTC)].
    after_mixed=True applies the schedule()'s W1/W2 writes first (state after the mixed phase, i.e.
    what verify should see if the concurrent-phase rows stay invisible as designed)."""
    st = _ref_state(scale, after_mixed)
    tenant_id, seed_entity_id = int(tenant_id), int(seed_entity_id)
    f = _frontier(st, tenant_id, seed_entity_id)
    f = f[(f >= 0) & (f < st.n_ent)]
    mems = np.unique(_expand(st.a_indptr, st.a_adj, st.a_ten, f, tenant_id))
    mems = mems[st.exists[mems] & st.current[mems] & (st.tenant[mems] == tenant_id)]
    if len(mems) == 0:
        return []
    created = st.created[mems]
    order = np.lexsort((mems, created))[::-1][:limit]     # created_at DESC, memory_id DESC
    ids = mems[order]
    dts = created[order].astype("datetime64[us]").tolist()
    return [(int(m), d) for m, d in zip(ids, dts)]


def reference_r1_count(scale: str, tenant_id: int, seed_entity_id: int, after_mixed: bool = False) -> int:
    """Number of memories R1 would return without the limit (for checking that a frontier is non-trivial)."""
    st = _ref_state(scale, after_mixed)
    f = _frontier(st, int(tenant_id), int(seed_entity_id))
    f = f[(f >= 0) & (f < st.n_ent)]
    mems = np.unique(_expand(st.a_indptr, st.a_adj, st.a_ten, f, int(tenant_id)))
    return int((st.exists[mems] & st.current[mems] & (st.tenant[mems] == int(tenant_id))).sum())


def _norm_dt(d: Any) -> Any:
    if isinstance(d, _dt.datetime):
        if d.tzinfo is not None:
            d = d.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        return d
    if isinstance(d, np.datetime64):
        return d.astype("datetime64[us]").item()
    if isinstance(d, str):
        return _dt.datetime.fromisoformat(d.replace("Z", "+00:00")).replace(tzinfo=None)
    return d


def compare_r1(got: list, expected: list, check_created_at: bool = True) -> tuple[bool, str]:
    """Compare an engine's R1 rows [(memory_id, created_at), ...] (or bare ids) to reference_r1 output.
    Returns (ok, message). created_at is compared as naive UTC datetimes at microsecond precision."""
    g_ids = [int(r[0] if isinstance(r, (tuple, list)) else r) for r in got]
    e_ids = [int(r[0]) for r in expected]
    if g_ids != e_ids:
        return False, f"ids differ: got {g_ids[:5]}... expected {e_ids[:5]}... (len {len(g_ids)} vs {len(e_ids)})"
    if check_created_at and got and isinstance(got[0], (tuple, list)) and len(got[0]) > 1:
        for (gm, gc), (em, ec) in zip(got, expected):
            if _norm_dt(gc) != _norm_dt(ec):
                return False, f"created_at differs for memory {gm}: got {gc!r} expected {ec!r}"
    return True, "ok"


# ----------------------------------------------------------------------------- reference R2

_R2: dict[str, SimpleNamespace] = {}


def _r2_state(scale: str) -> SimpleNamespace:
    st = _R2.get(scale)
    if st is not None:
        return st
    with _REF_LOCK:
        if scale in _R2:
            return _R2[scale]
        t0 = time.perf_counter()
        mem = load_table(scale, "memories", ["memory_id", "tenant_id", "content", "embedding", "valid_to"])
        n = mem.num_rows
        assert np.array_equal(mem["memory_id"].to_numpy(), np.arange(n))
        emb = mem["embedding"].combine_chunks().flatten().to_numpy().reshape(n, DIM).astype(np.float32)
        tenant = mem["tenant_id"].to_numpy()
        current = _is_null(mem["valid_to"])
        toks = pc.utf8_split_whitespace(pc.utf8_lower(mem["content"])).combine_chunks()
        doclen = pc.list_value_length(toks).to_numpy().astype(np.int64)
        flat = pc.dictionary_encode(toks.flatten())
        term_of_tok = flat.indices.to_numpy().astype(np.int64)
        vocab = flat.dictionary.to_pylist()
        doc_of_tok = np.repeat(np.arange(n, dtype=np.int64), doclen)
        key, tf = np.unique(term_of_tok * n + doc_of_tok, return_counts=True)
        term_sorted, doc_sorted = key // n, key % n
        df = np.bincount(term_sorted, minlength=len(vocab))
        st = SimpleNamespace(n=n, emb=emb, tenant=tenant, current=current, doclen=doclen,
                             avgdl=float(doclen.mean()), term_index={w: i for i, w in enumerate(vocab)},
                             term_sorted=term_sorted, doc_sorted=doc_sorted, tf=tf.astype(np.float64), df=df,
                             build_s=time.perf_counter() - t0)
        _R2[scale] = st
        return st


def reference_r2_truth(scale: str, query: dict, topn: int = R2_TOPN, k: int = R2_K) -> dict:
    """Brute-force R2 truth over the base dataset (pre-writes, i.e. the r2_only phase state):
      cosine_ids  top-`topn` current same-tenant memories by cosine(query_embedding, embedding)
      bm25_ids    top-`topn` by Okapi BM25 (k1=1.2, b=0.75, idf = ln((N-df+0.5)/(df+0.5)+1), corpus
                  statistics over ALL memories rows, whitespace tokens; only docs matching >=1 term)
      rrf         top-`k` (memory_id, score) fusing the two with rrf_fuse (k=60, 1-based ranks)
      rrf_ids     just the ids of `rrf`
    Ties break by memory_id ascending. `query` needs tenant_id, query_text, query_embedding."""
    st = _r2_state(scale)
    tenant_id = int(query["tenant_id"])
    cand = np.flatnonzero((st.tenant == tenant_id) & st.current)
    q = np.asarray(query["query_embedding"], dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    sims = (st.emb[cand].astype(np.float64) @ q.astype(np.float64))
    order = np.lexsort((cand, -sims))[:topn]
    cosine_ids, cosine_scores = cand[order], sims[order]

    terms = []
    for t in str(query["query_text"]).lower().split():
        if t not in terms:
            terms.append(t)
    scores = np.zeros(st.n, dtype=np.float64)
    for t in terms:
        tid = st.term_index.get(t)
        if tid is None:
            continue
        lo = int(np.searchsorted(st.term_sorted, tid, "left"))
        hi = int(np.searchsorted(st.term_sorted, tid, "right"))
        docs, tf = st.doc_sorted[lo:hi], st.tf[lo:hi]
        df = float(st.df[tid])
        idf = math.log((st.n - df + 0.5) / (df + 0.5) + 1.0)
        denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * st.doclen[docs] / st.avgdl)
        scores[docs] += idf * tf * (BM25_K1 + 1.0) / denom
    s = scores[cand]
    nz = s > 0
    c2, s2 = cand[nz], s[nz]
    order = np.lexsort((c2, -s2))[:topn]
    bm25_ids, bm25_scores = c2[order], s2[order]

    rrf = rrf_fuse(cosine_ids.tolist(), bm25_ids.tolist(), k=RRF_K, top=k)
    return {"cosine_ids": [int(x) for x in cosine_ids], "cosine_scores": [float(x) for x in cosine_scores],
            "bm25_ids": [int(x) for x in bm25_ids], "bm25_scores": [float(x) for x in bm25_scores],
            "rrf": [(int(m), float(s)) for m, s in rrf], "rrf_ids": [int(m) for m, _ in rrf],
            "n_candidates": int(len(cand))}


__all__ = [
    "SPIKE_DIR", "DATA_DIR", "RESULTS_DIR", "SCALES", "FILES", "N_TENANTS", "DIM", "T0",
    "WARMUP_QUERY_IDS", "R1_QUERY_IDS", "R2_QUERY_IDS", "MIXED_QUERY_IDS", "VERIFY_QUERY_IDS",
    "MIXED_OPS", "CONCURRENT_SECONDS", "N_WRITERS", "N_READERS", "R1_LIMIT", "R2_K", "R2_TOPN", "RRF_K",
    "BM25_K1", "BM25_B", "WRITE_EDGE_ID_BASE", "CONCURRENT_ID_BASE", "CONCURRENT_EDGE_ID_BASE", "CONCURRENT_STRIDE",
    "CONCURRENT_CREATED_AT", "DATASET_NOTES",
    "data_dir", "dataset_path", "load_table", "load_queries", "load_writes",
    "schedule", "schedule_counts", "concurrent_w1_ops", "concurrent_r1_queries",
    "Timer", "Stopwatch", "summarize", "peak_rss_mb", "db_bytes", "host_info", "recall_at_k", "rrf_fuse",
    "result_skeleton", "phase_error", "verify_payload", "write_result", "run_concurrent",
    "reference_frontier", "reference_r1", "reference_r1_count", "compare_r1", "reference_r2_truth",
]
