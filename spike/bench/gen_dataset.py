#!/usr/bin/env python
"""anatid Phase 0 spike: deterministic synthetic dataset generator (SPEC.md, section "Dataset").

    .venv/bin/python bench/gen_dataset.py --scale small|full [--out DIR] [--no-check]

Writes eight Parquet files into spike/data/<scale>/ (numpy seed 42), then re-reads them and
prints row counts plus sanity checks. Everything is vectorized with numpy; strings are built
with pyarrow (dictionary take + binary_join), never a per-row Python loop over memories.

Design decisions that go beyond the SPEC text (also recorded in common.DATASET_NOTES):
  * queries.parquet has 2,000 rows at BOTH scales (the workload schedule addresses
    query_ids 0..1999 at both scales), everything else is /10 at `small`.
  * query_text uses 2..3 DISTINCT topic words of the seed entity (an entity owns only 3 topic
    words; a 4th word would have to repeat one and make BM25 scoring tokenizer-dependent).
  * writes_supersedes.parquet columns: new_memory_id, old_memory_id, tenant_id (tenant added
    for convenience). Its new_memory_ids are exactly every 10th writes_memories row
    (row index % 10 == 0); the other 90% are the W1 pool. old_memory_ids are distinct,
    currently-valid (valid_to IS NULL) base memories of the same tenant.
  * the 5% of base memories with valid_to set are exactly the targets of edges_supersedes
    (valid_to = created_at of the superseding memory), so the two are consistent.
  * RELATES_TO out-degree ~ Zipf(2.2) clipped to [1, 500] (mean ~3, ~0.1% hubs with 200+);
    5% of RELATES_TO edges are expired (valid_to set) so the validity filter matters.
    ABOUT edges are all current (valid_to NULL) so filtering or not filtering on edge validity
    gives identical R1 answers.
  * query seeds are re-drawn until the seed's 2-hop frontier holds >= 30 current memories, so R1
    always fills its top-20 (also after the mixed phase's W2s) and the verify phase cannot be polluted
    by the non-deterministic number of concurrent-phase rows (which carry created_at = 1990).
  * vocabulary words are pseudo-words built from CVC syllables whose final consonant is one of
    k,m,n,p,t,b,g,x,z,v: they are invariant under the Porter stemmer and never English stopwords,
    so FTS engines that stem/stop tokenize them identically to a plain whitespace split.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SEED = 42
N_TENANTS = 10
DIM = 64
VOCAB_SIZE = 5000
TOPIC_WORDS = 3
US = 1_000_000
DAY_US = 86_400 * US
T0_US = np.datetime64("2025-01-01T00:00:00", "us").astype(np.int64)  # dataset epoch (all base rows < T0)
HISTORY_US = 730 * DAY_US          # base memories/edges span the two years before T0
WRITE_STEP_US = 10 * US            # writes_memories[i].created_at = T0 + i * 10 s
MIN_WORDS, MAX_WORDS = 12, 30
ENTITY_KINDS = np.array(["person", "project", "topic", "tool", "doc"])
ENTITY_KIND_P = [0.35, 0.20, 0.20, 0.15, 0.10]
MEMORY_KINDS = np.array(["episodic", "semantic", "preference", "procedural"])
MEMORY_KIND_P = [0.5, 0.3, 0.1, 0.1]
REL_KINDS = np.array(["works_on", "knows", "part_of", "uses", "mentions", "depends_on"])
N_AGENTS = 20
ZIPF_A = 2.2
DEG_CAP = 500
EMB_NOISE = 0.35
QUERY_NOISE = 0.10
TOPIC_FRACTION = 0.6
NEIGHBOUR_BIAS = 0.7
ABOUT_K_P = [0.3, 0.4, 0.3]        # P(#ABOUT entities = 1, 2, 3) -> mean 2.0
EXPIRED_FRACTION = 0.05            # memories with valid_to set (== rows of edges_supersedes)
EDGE_EXPIRED_FRACTION = 0.05       # RELATES_TO edges with valid_to set
MIN_QUERY_RESULTS = 30             # every query seed's 2-hop frontier holds >= 30 current memories

SCALES = {
    "full": dict(entities=100_000, memories=1_000_000, queries=2_000, writes=100_000),
    "small": dict(entities=10_000, memories=100_000, queries=2_000, writes=10_000),
}

TS = pa.timestamp("us")
EMB_T = pa.list_(pa.float32(), DIM)

SCHEMAS = {
    "entities": pa.schema([("entity_id", pa.int64()), ("tenant_id", pa.int32()),
                           ("kind", pa.string()), ("name", pa.string())]),
    "memories": pa.schema([("memory_id", pa.int64()), ("tenant_id", pa.int32()), ("content", pa.string()),
                           ("kind", pa.string()), ("embedding", EMB_T), ("created_at", TS),
                           ("valid_from", TS), ("valid_to", TS), ("tx_from", TS), ("tx_to", TS),
                           ("writer", pa.string()), ("episode_id", pa.int64()), ("confidence", pa.float32())]),
    "edges_about": pa.schema([("edge_id", pa.int64()), ("src", pa.int64()), ("dst", pa.int64()),
                              ("tenant_id", pa.int32()), ("weight", pa.float32()), ("valid_from", TS),
                              ("valid_to", TS), ("tx_from", TS), ("tx_to", TS)]),
    "edges_relates": pa.schema([("edge_id", pa.int64()), ("src", pa.int64()), ("dst", pa.int64()),
                                ("tenant_id", pa.int32()), ("rel_kind", pa.string()), ("valid_from", TS),
                                ("valid_to", TS), ("tx_from", TS), ("tx_to", TS)]),
    "edges_supersedes": pa.schema([("edge_id", pa.int64()), ("src", pa.int64()), ("dst", pa.int64()),
                                   ("tenant_id", pa.int32()), ("tx_from", TS)]),
    "queries": pa.schema([("query_id", pa.int32()), ("tenant_id", pa.int32()), ("seed_entity_id", pa.int64()),
                          ("query_text", pa.string()), ("query_embedding", EMB_T)]),
    "writes_supersedes": pa.schema([("new_memory_id", pa.int64()), ("old_memory_id", pa.int64()),
                                    ("tenant_id", pa.int32())]),
}
SCHEMAS["writes_memories"] = SCHEMAS["memories"].append(pa.field("about_dsts", pa.list_(pa.int64())))


# ----------------------------------------------------------------------------- helpers

def log(msg: str) -> None:
    print(msg, flush=True)


def unit(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def ts(us: np.ndarray, null_mask: np.ndarray | None = None) -> pa.Array:
    """int64 microseconds since epoch -> pyarrow timestamp[us] (UTC-naive); null where mask is True."""
    arr = np.asarray(us, dtype=np.int64).astype("datetime64[us]")
    return pa.array(arr, mask=None if null_mask is None else np.asarray(null_mask, dtype=bool))


def null_ts(n: int) -> pa.Array:
    return pa.nulls(n, TS)


def emb_array(m: np.ndarray) -> pa.Array:
    m = np.ascontiguousarray(m, dtype=np.float32)
    assert m.shape[1] == DIM
    return pa.FixedSizeListArray.from_arrays(pa.array(m.ravel(), type=pa.float32()), DIM)


def list_array(offsets: np.ndarray, values: pa.Array) -> pa.Array:
    return pa.ListArray.from_arrays(pa.array(np.asarray(offsets, dtype=np.int32)), values)


def csr_from_edges(u: np.ndarray, v: np.ndarray, n_nodes: int):
    """Sort (u, v) by u and return (indptr, adj) with adj[indptr[i]:indptr[i+1]] = neighbours of i."""
    order = np.argsort(u, kind="stable")
    counts = np.bincount(u, minlength=n_nodes)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, v[order]


def make_vocab(rng: np.random.Generator) -> np.ndarray:
    onset, vowel, coda = "bdfghjklmnprstvwz", "aeiou", "kmnptbgxzv"
    syll = np.array([o + v + c for o in onset for v in vowel for c in coda])
    cand = VOCAB_SIZE * 8
    n_syl = rng.integers(2, 4, cand)
    s = rng.integers(0, len(syll), (cand, 3))
    words = [syll[a] + syll[b] + (syll[c] if k == 3 else "") for (a, b, c), k in zip(s, n_syl)]
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
            if len(out) == VOCAB_SIZE:
                break
    assert len(out) == VOCAB_SIZE
    return np.array(out)


# ----------------------------------------------------------------------------- entities / relates

def gen_entities(rng, n_ent: int, vocab: np.ndarray):
    per_tenant = n_ent // N_TENANTS
    entity_id = np.arange(n_ent, dtype=np.int64)
    tenant = (entity_id // per_tenant).astype(np.int32)
    kind_idx = rng.choice(len(ENTITY_KINDS), n_ent, p=ENTITY_KIND_P)
    topic = rng.integers(0, VOCAB_SIZE, (n_ent, TOPIC_WORDS), dtype=np.int32)
    # make the three topic words of an entity distinct
    for _ in range(8):
        d1 = topic[:, 1] == topic[:, 0]
        topic[d1, 1] = (topic[d1, 1] + 1) % VOCAB_SIZE
        d2 = (topic[:, 2] == topic[:, 0]) | (topic[:, 2] == topic[:, 1])
        topic[d2, 2] = (topic[d2, 2] + 2) % VOCAB_SIZE
        if not (d1.any() or d2.any()):
            break
    centroid = unit(rng.standard_normal((n_ent, DIM), dtype=np.float32))
    kinds = ENTITY_KINDS[kind_idx]
    names = [f"{k}-{w}-{i}" for k, w, i in zip(kinds, vocab[topic[:, 0]], entity_id)]
    table = pa.table({"entity_id": pa.array(entity_id), "tenant_id": pa.array(tenant),
                      "kind": pa.array(kinds), "name": pa.array(names)}, schema=SCHEMAS["entities"])
    return table, tenant, topic, centroid, per_tenant


def gen_relates(rng, n_ent: int, ent_tenant: np.ndarray, per_tenant: int):
    deg = np.minimum(rng.zipf(ZIPF_A, n_ent), DEG_CAP).astype(np.int64)
    src = np.repeat(np.arange(n_ent, dtype=np.int64), deg)
    base = (src // per_tenant) * per_tenant
    local = src - base
    d = rng.integers(0, per_tenant - 1, len(src), dtype=np.int64)
    d += (d >= local)                      # never a self loop, always same tenant
    dst = base + d
    key = np.unique(src * n_ent + dst)     # dedupe parallel edges; sorted by (src, dst)
    src, dst = key // n_ent, key % n_ent
    e = len(src)
    valid_from = T0_US - rng.integers(1, HISTORY_US, e, dtype=np.int64)
    expired = rng.random(e) < EDGE_EXPIRED_FRACTION
    valid_to = valid_from + np.maximum(1, (rng.random(e) * (T0_US - valid_from)).astype(np.int64))
    table = pa.table({
        "edge_id": pa.array(np.arange(e, dtype=np.int64)),
        "src": pa.array(src), "dst": pa.array(dst),
        "tenant_id": pa.array(ent_tenant[src]),
        "rel_kind": pa.array(REL_KINDS[rng.integers(0, len(REL_KINDS), e)]),
        "valid_from": ts(valid_from), "valid_to": ts(valid_to, ~expired),
        "tx_from": ts(valid_from), "tx_to": null_ts(e),
    }, schema=SCHEMAS["edges_relates"])
    cur = ~expired
    return table, src[cur], dst[cur], deg


# ----------------------------------------------------------------------------- memories

class Graph:
    """Everything a memory block needs about the entity graph."""

    def __init__(self, n_ent, per_tenant, cur_src, cur_dst, topic, centroid):
        self.n_ent, self.per_tenant = n_ent, per_tenant
        self.indptr, self.adj = csr_from_edges(np.concatenate([cur_src, cur_dst]),
                                               np.concatenate([cur_dst, cur_src]), n_ent)
        self.deg = np.diff(self.indptr)
        self.topic, self.centroid = topic, centroid


def biased_pick(rng, home: np.ndarray, g: Graph) -> np.ndarray:
    """For each home entity: with prob NEIGHBOUR_BIAS a random current RELATES_TO neighbour, else a random
    same-tenant entity."""
    n = len(home)
    deg = g.deg[home]
    off = (rng.random(n) * deg).astype(np.int64)
    nb = g.adj[g.indptr[home] + np.minimum(off, np.maximum(deg - 1, 0))]
    rnd = (home // g.per_tenant) * g.per_tenant + rng.integers(0, g.per_tenant, n, dtype=np.int64)
    use_nb = (rng.random(n) < NEIGHBOUR_BIAS) & (deg > 0)
    return np.where(use_nb, nb, rnd)


def gen_content(rng, n: int, indptr: np.ndarray, about_dst: np.ndarray, topic: np.ndarray, lengths: np.ndarray):
    """Word-index matrix -> flat int32 token ids + int64 offsets (12..30 words/memory, 60% topic words)."""
    chunks = []
    step = 200_000
    pos = np.arange(MAX_WORDS)[None, :]
    for s in range(0, n, step):
        e = min(n, s + step)
        m = e - s
        cnt = indptr[s + 1:e + 1] - indptr[s:e]
        j = (rng.random((m, MAX_WORDS)) * cnt[:, None]).astype(np.int64)
        ent = about_dst[indptr[s:e][:, None] + j]
        tw = topic[ent, rng.integers(0, TOPIC_WORDS, (m, MAX_WORDS))]
        gw = rng.integers(0, VOCAB_SIZE, (m, MAX_WORDS), dtype=np.int32)
        use_topic = rng.random((m, MAX_WORDS)) < TOPIC_FRACTION
        w = np.where(use_topic, tw, gw).astype(np.int32)
        chunks.append(w[pos < lengths[s:e, None]])
    flat = np.concatenate(chunks)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    assert offsets[-1] == len(flat)
    return flat, offsets


def content_strings(flat: np.ndarray, offsets: np.ndarray, vocab_arr: pa.Array) -> pa.Array:
    words = pc.take(vocab_arr, pa.array(flat, type=pa.int32()))
    return pc.binary_join(list_array(offsets, words), " ")


def gen_memory_block(rng, mem_ids: np.ndarray, tenants: np.ndarray, created_us: np.ndarray, g: Graph,
                     vocab_arr: pa.Array):
    """Home entity + ABOUT edges + embedding + content + misc columns for a dense block of memory ids.
    Returns (columns dict, about_src, about_dst, about_indptr) with ABOUT edges sorted by (src, dst)."""
    n = len(mem_ids)
    first = int(mem_ids[0])
    assert np.array_equal(mem_ids, np.arange(first, first + n))
    home = np.empty(n, dtype=np.int64)
    for t in range(N_TENANTS):
        idx = np.flatnonzero(tenants == t)
        base = t * g.per_tenant
        w = np.sqrt(g.deg[base:base + g.per_tenant] + 1.0)
        w /= w.sum()
        home[idx] = base + rng.choice(g.per_tenant, size=len(idx), p=w)
    k = rng.choice([1, 2, 3], n, p=ABOUT_K_P)
    extra1, extra2 = biased_pick(rng, home, g), biased_pick(rng, home, g)
    src = np.concatenate([mem_ids, mem_ids[k >= 2], mem_ids[k == 3]])
    dst = np.concatenate([home, extra1[k >= 2], extra2[k == 3]])
    key = np.unique(src * g.n_ent + dst)
    src, dst = key // g.n_ent, key % g.n_ent
    counts = np.bincount(src - first, minlength=n)
    assert counts.min() >= 1
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])

    sums = np.add.reduceat(g.centroid[dst], indptr[:-1], axis=0)
    emb = unit(sums / counts[:, None] + rng.standard_normal((n, DIM), dtype=np.float32) * EMB_NOISE)

    lengths = rng.integers(MIN_WORDS, MAX_WORDS + 1, n)
    flat, offsets = gen_content(rng, n, indptr, dst, g.topic, lengths)
    content = content_strings(flat, offsets, vocab_arr)

    cols = {
        "memory_id": pa.array(mem_ids), "tenant_id": pa.array(tenants.astype(np.int32)),
        "content": content,
        "kind": pa.array(MEMORY_KINDS[rng.choice(len(MEMORY_KINDS), n, p=MEMORY_KIND_P)]),
        "embedding": emb_array(emb),
        "created_at": ts(created_us), "valid_from": ts(created_us),
        "valid_to": null_ts(n), "tx_from": ts(created_us), "tx_to": null_ts(n),
        "writer": pa.array(np.array([f"agent-{i:02d}" for i in range(N_AGENTS)])[rng.integers(0, N_AGENTS, n)]),
        "episode_id": pa.array((mem_ids // 8).astype(np.int64)),
        "confidence": pa.array(rng.uniform(0.5, 1.0, n).astype(np.float32)),
    }
    return cols, src, dst, indptr


def about_edges_table(src, dst, tenants_of_src, created_us_of_src, rng) -> pa.Table:
    e = len(src)
    return pa.table({
        "edge_id": pa.array(np.arange(e, dtype=np.int64)),
        "src": pa.array(src), "dst": pa.array(dst),
        "tenant_id": pa.array(tenants_of_src.astype(np.int32)),
        "weight": pa.array(rng.uniform(0.3, 1.0, e).astype(np.float32)),
        "valid_from": ts(created_us_of_src), "valid_to": null_ts(e),
        "tx_from": ts(created_us_of_src), "tx_to": null_ts(e),
    }, schema=SCHEMAS["edges_about"])


def gen_supersedes(rng, n_mem: int, tenants: np.ndarray):
    """5% of base memories get superseded by a newer same-tenant memory. Memory ids are in created_at
    order, so 'newer' == larger id. Returns (new_ids, old_ids) sorted by new id."""
    per_t = int(n_mem * EXPIRED_FRACTION) // N_TENANTS
    olds, news = [], []
    for t in range(N_TENANTS):
        ids_t = np.flatnonzero(tenants == t)
        n_t = len(ids_t)
        p = rng.choice(int(n_t * 0.9), size=per_t, replace=False)
        q = p + 1 + (rng.random(per_t) * (n_t - p - 1)).astype(np.int64)
        olds.append(ids_t[p])
        news.append(ids_t[q])
    old, new = np.concatenate(olds), np.concatenate(news)
    order = np.argsort(new * n_mem + old, kind="stable")
    return new[order].astype(np.int64), old[order].astype(np.int64)


# ----------------------------------------------------------------------------- queries

def frontier_counts(seeds, r_indptr, r_adj, a_indptr, a_adj, mem_current) -> np.ndarray:
    """Per seed: number of current memories ABOUT its undirected 2-hop RELATES_TO frontier
    (edges never cross tenants, so no tenant filter is needed here)."""
    out = np.empty(len(seeds), dtype=np.int64)
    for i, s in enumerate(np.asarray(seeds, dtype=np.int64)):
        h1 = r_adj[r_indptr[s]:r_indptr[s + 1]]
        parts = [np.array([s], dtype=np.int64), h1] + [r_adj[r_indptr[x]:r_indptr[x + 1]] for x in h1]
        f = np.unique(np.concatenate(parts))
        mems = np.unique(np.concatenate([a_adj[a_indptr[x]:a_indptr[x + 1]] for x in f]))
        out[i] = int(mem_current[mems].sum())
    return out


def gen_queries(rng, n_q: int, per_tenant: int, topic, centroid, vocab, count_fn):
    tenant = rng.integers(0, N_TENANTS, n_q).astype(np.int32)
    seed = tenant.astype(np.int64) * per_tenant + rng.integers(0, per_tenant, n_q, dtype=np.int64)
    for _ in range(100):
        bad = np.flatnonzero(count_fn(seed) < MIN_QUERY_RESULTS)
        if len(bad) == 0:
            break
        tenant[bad] = rng.integers(0, N_TENANTS, len(bad)).astype(np.int32)
        seed[bad] = tenant[bad].astype(np.int64) * per_tenant + rng.integers(0, per_tenant, len(bad), dtype=np.int64)
    else:
        raise RuntimeError("could not find query seeds with enough memories in their frontier")
    n_words = rng.integers(2, 4, n_q)   # 2 or 3 distinct topic words (entity owns exactly 3)
    texts = []
    for i in range(n_q):
        perm = rng.permutation(TOPIC_WORDS)[: n_words[i]]
        texts.append(" ".join(vocab[topic[seed[i], perm]]))
    qemb = unit(centroid[seed] + rng.standard_normal((n_q, DIM), dtype=np.float32) * QUERY_NOISE)
    return pa.table({
        "query_id": pa.array(np.arange(n_q, dtype=np.int32)), "tenant_id": pa.array(tenant),
        "seed_entity_id": pa.array(seed), "query_text": pa.array(texts), "query_embedding": emb_array(qemb),
    }, schema=SCHEMAS["queries"])


# ----------------------------------------------------------------------------- driver

def equal_split_tenants(rng, n: int) -> np.ndarray:
    assert n % N_TENANTS == 0
    return rng.permutation(np.repeat(np.arange(N_TENANTS, dtype=np.int32), n // N_TENANTS))


def write(table: pa.Table, out: Path, name: str, row_group_size: int = 131_072) -> None:
    assert table.schema.equals(SCHEMAS[name]), f"{name}: schema mismatch\n{table.schema}\n!=\n{SCHEMAS[name]}"
    pq.write_table(table, out / f"{name}.parquet", compression="snappy", row_group_size=row_group_size)


def generate(scale: str, out: Path) -> None:
    cfg = SCALES[scale]
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t_all = time.perf_counter()

    def stage(name, t):
        log(f"  [{scale}] {name:<22s} {time.perf_counter() - t:7.2f} s")

    t = time.perf_counter()
    vocab = make_vocab(rng)
    vocab_arr = pa.array(vocab)
    ent_table, ent_tenant, topic, centroid, per_tenant = gen_entities(rng, cfg["entities"], vocab)
    write(ent_table, out, "entities")
    stage("entities", t)

    t = time.perf_counter()
    rel_table, cur_src, cur_dst, out_deg = gen_relates(rng, cfg["entities"], ent_tenant, per_tenant)
    write(rel_table, out, "edges_relates")
    g = Graph(cfg["entities"], per_tenant, cur_src, cur_dst, topic, centroid)
    stage("edges_relates", t)

    t = time.perf_counter()
    n_mem = cfg["memories"]
    mem_ids = np.arange(n_mem, dtype=np.int64)
    mem_tenant = equal_split_tenants(rng, n_mem)
    created = T0_US - HISTORY_US + np.sort(rng.integers(0, HISTORY_US, n_mem, dtype=np.int64))
    cols, a_src, a_dst, _ = gen_memory_block(rng, mem_ids, mem_tenant, created, g, vocab_arr)
    stage("memories (gen)", t)

    t = time.perf_counter()
    sup_new, sup_old = gen_supersedes(rng, n_mem, mem_tenant)
    valid_to = np.zeros(n_mem, dtype=np.int64)
    valid_to[sup_old] = created[sup_new]
    is_null = np.ones(n_mem, dtype=bool)
    is_null[sup_old] = False
    cols["valid_to"] = ts(valid_to, is_null)
    write(pa.table(cols, schema=SCHEMAS["memories"]), out, "memories")
    del cols
    write(pa.table({
        "edge_id": pa.array(np.arange(len(sup_new), dtype=np.int64)),
        "src": pa.array(sup_new), "dst": pa.array(sup_old),
        "tenant_id": pa.array(mem_tenant[sup_new].astype(np.int32)),
        "tx_from": ts(created[sup_new]),
    }, schema=SCHEMAS["edges_supersedes"]), out, "edges_supersedes")
    stage("memories + supersedes", t)

    t = time.perf_counter()
    write(about_edges_table(a_src, a_dst, mem_tenant[a_src], created[a_src], rng), out, "edges_about")
    stage("edges_about", t)

    t = time.perf_counter()
    a_indptr, a_adj = csr_from_edges(a_dst, a_src, cfg["entities"])       # entity -> memories
    count_fn = lambda seeds: frontier_counts(seeds, g.indptr, g.adj, a_indptr, a_adj, is_null)  # noqa: E731
    write(gen_queries(rng, cfg["queries"], per_tenant, topic, centroid, vocab, count_fn), out, "queries")
    del a_indptr, a_adj
    stage("queries", t)

    t = time.perf_counter()
    n_w = cfg["writes"]
    w_ids = n_mem + np.arange(n_w, dtype=np.int64)
    w_tenant = equal_split_tenants(rng, n_w)
    w_created = T0_US + np.arange(n_w, dtype=np.int64) * WRITE_STEP_US
    wcols, w_src, w_dst, w_indptr = gen_memory_block(rng, w_ids, w_tenant, w_created, g, vocab_arr)
    wcols["about_dsts"] = list_array(w_indptr, pa.array(w_dst, type=pa.int64()))
    write(pa.table(wcols, schema=SCHEMAS["writes_memories"]), out, "writes_memories")
    del wcols

    ws_new = w_ids[::10]
    ws_tenant = w_tenant[::10]
    ws_old = np.empty(len(ws_new), dtype=np.int64)
    for tn in range(N_TENANTS):
        sel = np.flatnonzero(ws_tenant == tn)
        pool = np.flatnonzero((mem_tenant == tn) & is_null)          # current base memories of this tenant
        ws_old[sel] = rng.choice(pool, size=len(sel), replace=False)
    write(pa.table({"new_memory_id": pa.array(ws_new), "old_memory_id": pa.array(ws_old),
                    "tenant_id": pa.array(ws_tenant.astype(np.int32))},
                   schema=SCHEMAS["writes_supersedes"]), out, "writes_supersedes")
    stage("writes", t)
    log(f"  [{scale}] generated in {time.perf_counter() - t_all:.1f} s -> {out}")


# ----------------------------------------------------------------------------- checks

def _csr(u, v, n_nodes):
    return csr_from_edges(np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64), n_nodes)


def check(out: Path) -> bool:
    ok = True

    def expect(cond, msg):
        nonlocal ok
        status = "ok  " if cond else "FAIL"
        if not cond:
            ok = False
        log(f"    {status} {msg}")

    log("  row counts:")
    tables = {}
    for name in SCHEMAS:
        pf = pq.ParquetFile(out / f"{name}.parquet")
        tables[name] = pf.read()
        size_mb = (out / f"{name}.parquet").stat().st_size / 1e6
        log(f"    {name:<20s} {pf.metadata.num_rows:>10,d} rows  {size_mb:8.1f} MB")
        expect(tables[name].schema.equals(SCHEMAS[name]), f"{name}: schema matches contract")

    ent, mem, ab, rel, sup = (tables[k] for k in ("entities", "memories", "edges_about", "edges_relates", "edges_supersedes"))
    q, wm, ws = tables["queries"], tables["writes_memories"], tables["writes_supersedes"]
    ent_tenant = ent["tenant_id"].to_numpy()
    mem_tenant = mem["tenant_id"].to_numpy()
    n_ent, n_mem = len(ent), len(mem)

    log("  sanity:")
    expect(np.array_equal(ent["entity_id"].to_numpy(), np.arange(n_ent)), "entity_id dense 0..N-1")
    expect(np.array_equal(mem["memory_id"].to_numpy(), np.arange(n_mem)), "memory_id dense 0..N-1")
    expect(np.array_equal(wm["memory_id"].to_numpy(), n_mem + np.arange(len(wm))), "writes memory_id dense after memories")

    for name, tbl, src_t, dst_t in (("edges_about", ab, mem_tenant, ent_tenant),
                                    ("edges_relates", rel, ent_tenant, ent_tenant),
                                    ("edges_supersedes", sup, mem_tenant, mem_tenant)):
        s, d, tt = tbl["src"].to_numpy(), tbl["dst"].to_numpy(), tbl["tenant_id"].to_numpy()
        bad = int(((src_t[s] != tt) | (dst_t[d] != tt)).sum())
        expect(bad == 0, f"{name}: cross-tenant edges = {bad}")
    rs, rd = rel["src"].to_numpy(), rel["dst"].to_numpy()
    expect(int((rs == rd).sum()) == 0, "edges_relates: no self loops")
    expect(len(np.unique(rs * n_ent + rd)) == len(rs), "edges_relates: no duplicate (src,dst)")

    emb = ab_emb = mem["embedding"].combine_chunks().flatten().to_numpy().reshape(n_mem, DIM)
    norms = np.linalg.norm(emb.astype(np.float64), axis=1)
    expect(np.abs(norms - 1).max() < 1e-3, f"memory embeddings unit-norm (max |norm-1| = {np.abs(norms - 1).max():.2e})")
    qe = q["query_embedding"].combine_chunks().flatten().to_numpy().reshape(len(q), DIM)
    expect(np.abs(np.linalg.norm(qe, axis=1) - 1).max() < 1e-3, "query embeddings unit-norm")

    vt_null = pc.is_null(mem["valid_to"]).to_numpy(zero_copy_only=False)
    frac = 1 - vt_null.mean()
    expect(0.045 <= frac <= 0.055, f"memories with valid_to set = {frac * 100:.2f}% (target ~5%)")
    expect(int((~vt_null).sum()) == len(sup), "expired memories == edges_supersedes rows")
    sup_src, sup_dst = sup["src"].to_numpy(), sup["dst"].to_numpy()
    expect(bool((sup_src > sup_dst).all()), "edges_supersedes: src (newer) id > dst (older) id")
    created = mem["created_at"].to_numpy().astype("datetime64[us]").astype(np.int64)
    expect(bool((np.diff(created) >= 0).all()), "memories created_at non-decreasing with memory_id")
    vt = mem["valid_to"].to_numpy(zero_copy_only=False)
    vt_i = vt[~vt_null].astype("datetime64[us]").astype(np.int64)
    expect(bool((vt_i > created[~vt_null]).all()), "valid_to > created_at where set")
    exp_vt = np.zeros(n_mem, dtype=np.int64)
    exp_vt[sup_dst] = created[sup_src]
    expect(np.array_equal(np.sort(sup_dst), np.flatnonzero(~vt_null)) and bool((vt_i == exp_vt[~vt_null]).all()),
           "valid_to of superseded == created_at of superseding memory")
    rel_vt_null = pc.is_null(rel["valid_to"]).to_numpy(zero_copy_only=False)
    log(f"    info edges_relates expired (valid_to set) = {(1 - rel_vt_null.mean()) * 100:.2f}%")
    expect(pc.all(pc.is_null(ab["valid_to"])).as_py(), "edges_about all current (valid_to NULL)")
    expect(pc.all(pc.is_null(mem["tx_to"])).as_py(), "memories tx_to all NULL")

    qs, qt = q["seed_entity_id"].to_numpy(), q["tenant_id"].to_numpy()
    expect(bool(((qs >= 0) & (qs < n_ent)).all()) and bool((ent_tenant[qs] == qt).all()), "query seeds exist and match tenant")
    qtext = q["query_text"].to_pylist()
    nw = np.array([len(s.split()) for s in qtext])
    expect(bool(((nw >= 2) & (nw <= 3)).all()), f"query_text 2..3 words (mean {nw.mean():.2f})")
    expect(all(len(set(s.split())) == len(s.split()) for s in qtext), "query_text words distinct")
    contents_sample = mem["content"].slice(0, 20000).to_pylist()
    cw = np.array([len(s.split()) for s in contents_sample])
    expect(cw.min() >= MIN_WORDS and cw.max() <= MAX_WORDS, f"content 12..30 words (sample min {cw.min()} max {cw.max()} mean {cw.mean():.1f})")

    out_deg = np.bincount(rs, minlength=n_ent)
    udeg = np.bincount(np.concatenate([rs, rd]), minlength=n_ent)
    log(f"    info RELATES_TO out-degree: mean {out_deg.mean():.2f}, max {out_deg.max()}, "
        f">=200: {(out_deg >= 200).sum()}, ==0: {(out_deg == 0).sum()}")
    log(f"    info RELATES_TO undirected degree: mean {udeg.mean():.2f}, p50 {np.percentile(udeg, 50):.0f}, "
        f"p99 {np.percentile(udeg, 99):.0f}, max {udeg.max()}")
    expect((out_deg >= 200).sum() >= 1, "degree distribution has hubs (>=200)")
    ab_per_mem = np.bincount(ab["src"].to_numpy(), minlength=n_mem)
    log(f"    info ABOUT edges/memory: mean {ab_per_mem.mean():.3f}, min {ab_per_mem.min()}, max {ab_per_mem.max()}")
    expect(ab_per_mem.min() >= 1 and ab_per_mem.max() <= 3, "every memory has 1..3 ABOUT edges")
    mem_per_ent = np.bincount(ab["dst"].to_numpy(), minlength=n_ent)
    log(f"    info ABOUT memories/entity: mean {mem_per_ent.mean():.1f}, p50 {np.percentile(mem_per_ent, 50):.0f}, "
        f"p99 {np.percentile(mem_per_ent, 99):.0f}, max {mem_per_ent.max()}")

    cur_rel = rel_vt_null
    r_indptr, r_adj = _csr(np.concatenate([rs[cur_rel], rd[cur_rel]]), np.concatenate([rd[cur_rel], rs[cur_rel]]), n_ent)
    a_indptr, a_adj = _csr(ab["dst"].to_numpy(), ab["src"].to_numpy(), n_ent)
    fsz = np.empty(len(qs), dtype=np.int64)
    for i, sd in enumerate(qs):
        h1 = r_adj[r_indptr[sd]:r_indptr[sd + 1]]
        fsz[i] = len(np.unique(np.concatenate([np.array([sd]), h1] + [r_adj[r_indptr[x]:r_indptr[x + 1]] for x in h1])))
    qcnt = frontier_counts(qs, r_indptr, r_adj, a_indptr, a_adj, vt_null)
    log(f"    info query 2-hop frontier size: min {fsz.min()}, p10 {np.percentile(fsz, 10):.0f}, p50 {np.percentile(fsz, 50):.0f}, "
        f"p90 {np.percentile(fsz, 90):.0f}, max {fsz.max()}")
    log(f"    info current memories in query frontier: min {qcnt.min()}, p50 {np.percentile(qcnt, 50):.0f}, max {qcnt.max()}")
    expect(qcnt.min() >= MIN_QUERY_RESULTS, f"every query frontier has >= {MIN_QUERY_RESULTS} current memories")

    ad = pc.list_value_length(wm["about_dsts"]).to_numpy()
    expect(ad.min() >= 1 and ad.max() <= 3, f"writes about_dsts 1..3 (mean {ad.mean():.2f})")
    wflat = wm["about_dsts"].combine_chunks().flatten().to_numpy()
    wten = np.repeat(wm["tenant_id"].to_numpy(), ad)
    expect(bool((ent_tenant[wflat] == wten).all()), "writes about_dsts same tenant as memory")
    wcreated = wm["created_at"].to_numpy().astype("datetime64[us]").astype(np.int64)
    expect(bool((wcreated > created.max()).all()), "writes created_at after every base memory")
    expect(pc.all(pc.is_null(wm["valid_to"])).as_py(), "writes valid_to all NULL")

    ws_new, ws_old, ws_t = ws["new_memory_id"].to_numpy(), ws["old_memory_id"].to_numpy(), ws["tenant_id"].to_numpy()
    wm_tenant = wm["tenant_id"].to_numpy()
    expect(np.array_equal(ws_new, wm["memory_id"].to_numpy()[::10]), "writes_supersedes new ids == every 10th writes_memories row")
    expect(len(np.unique(ws_old)) == len(ws_old), "writes_supersedes old ids distinct")
    expect(bool(vt_null[ws_old].all()), "writes_supersedes old memories are current (valid_to NULL)")
    expect(bool((mem_tenant[ws_old] == ws_t).all()) and bool((wm_tenant[ws_new - n_mem] == ws_t).all()),
           "writes_supersedes tenant matches old and new memory")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=sorted(SCALES), required=True)
    ap.add_argument("--out", type=Path, default=None, help="output dir (default spike/data/<scale>)")
    ap.add_argument("--no-check", action="store_true", help="skip the read-back sanity checks")
    args = ap.parse_args(argv)
    out = args.out or (Path(__file__).resolve().parent.parent / "data" / args.scale)
    log(f"generating scale={args.scale} seed={SEED} -> {out}")
    generate(args.scale, out)
    if args.no_check:
        return 0
    t = time.perf_counter()
    ok = check(out)
    log(f"  checks {'passed' if ok else 'FAILED'} in {time.perf_counter() - t:.1f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
