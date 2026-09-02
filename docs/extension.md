# The `anatid` DuckDB extension

An optional C++ DuckDB extension that answers anatid's graph-expansion question — *which entities
are within k hops of this seed, in this tenant, right now?* — from an in-memory CSR instead of SQL
joins. Source lives in [`ext/`](../ext). It is **optional**: anatid works, and returns the same
answers, without it.

---

## Why it exists

Phase 0 measured three engines on the same dataset — 1,000,000 memories, 2.3M edges, 10 tenants —
running the same 1,000 2-hop recall queries (`spike/SPEC.md`, results in `spike/results/*.full.json`):

| engine | 2-hop recall p50 | p95 | vs LadybugDB |
|---|---|---|---|
| LadybugDB 0.20.2 (best of 5 hand-tuned Cypher formulations) | **7.35 ms** | 28.73 ms | 1.00x |
| DuckDB 1.5.5, pure SQL | **2.88 ms** | 3.50 ms | 0.39x |
| DuckDB 1.5.5 + this extension | **2.04 ms** | 3.07 ms | 0.28x |

All three returned **identical result id-lists** on 200 verification queries — zero mismatches. The
extension is worth about 0.84 ms of the 2.88 ms, roughly a 29% cut, and it flattens the tail
(p95 3.07 ms vs 3.50 ms) because a BFS over a contiguous adjacency array has none of the hash-join
variance the SQL formulation has.

That is the whole pitch. **The extension is a constant-factor win on one hot path, not a different
answer.** `tests/test_extension.py` re-proves the equality on every run: 200 benchmark seeds at 1 and
2 hops, the C++ path and `anatid.csr.frontier_sql`'s SQL path, byte-for-byte the same entity ids,
both also checked against `spike/bench/common.py`'s pure-numpy oracle.

---

## Building it

### Prerequisites

* a C++11 compiler (Apple clang, GCC, MSVC), CMake ≥ 3.5, and preferably Ninja
* the two git submodules under `ext/`: `duckdb` (pinned to **v1.5.5**, commit `d8cdaa33`) and
  `extension-ci-tools`

```sh
git submodule update --init --recursive        # from the repo root; ~300 MB, takes a few minutes
```

> **In this working tree** `ext/duckdb` and `ext/extension-ci-tools` are **symlinks** into
> `spike/extension/`, because the repo predates its own git history and the Phase 0 spike already
> holds the exact pinned checkouts. Nothing is duplicated and nothing under `spike/` is ever
> written to — the DuckDB source tree is read-only input to an out-of-source build in `ext/build/`.
> When `ext/` becomes a real git checkout, replace the two symlinks with the submodules declared in
> [`ext/.gitmodules`](../ext/.gitmodules); the build is otherwise identical.

### Build

```sh
cd ext
GEN=ninja make release
```

Artifacts:

| path | what |
|---|---|
| `ext/build/release/extension/anatid/anatid.duckdb_extension` | the loadable binary anatid loads |
| `ext/build/release/duckdb` | a DuckDB shell with the extension linked in |
| `ext/build/release/test/unittest` | the sqllogictest runner |

The first build compiles all of DuckDB and takes 10–20 minutes on a 10-core machine. After that,
editing `ext/src/anatid_extension.cpp` and re-running `GEN=ninja make release` is a ~5 second
incremental rebuild. `make debug` produces `ext/build/debug/...` the same way.

Everything the build needs is in the tree; there are no vcpkg dependencies (`ext/vcpkg.json`
declares none).

### Point anatid at it

```python
from anatid import Anatid

db = Anatid.open("memories.anatid", use_csr_extension=True)   # discovery, best effort
db = Anatid.open("memories.anatid", use_csr_extension=True,
                 extension_path="ext/build/release/extension/anatid/anatid.duckdb_extension",
                 require_extension=True)                       # or say exactly where
```

Discovery (`anatid.csr.discover_extension_path`) looks, in order, at `$ANATID_EXTENSION_PATH`,
`<package>/_ext/`, `<repo>/ext/build/release/extension/anatid/` — what the build above produces —
and finally the Phase 0 tree `<repo>/spike/extension/build/release/extension/anatid/`. Or set
`ANATID_EXTENSION_PATH` in the environment. Three things that bite:

1. Loading an unsigned extension needs `allow_unsigned_extensions` in the **connect** config, which
   DuckDB only reads when the database is opened. `use_csr_extension=True` sets it. You cannot turn
   the extension on after `Anatid.open()`.
2. `require_extension=True` raises `anatid.errors.ExtensionUnavailable` if the binary is missing or
   will not load. Without it, a failed load is silent and anatid stays on the SQL path — check
   `db.expand_path` (`"extension"` or `"sql"`) and `db.csr.describe()["load_error"]`.
3. If `$ANATID_EXTENSION_PATH` is set it is the whole answer — a typo there returns nothing rather
   than falling back to a build tree, so the miss is visible.

---

## What it registers

```sql
anatid_version()                        -- 'anatid 0.1.0 (DuckDB v1.5.5)'
anatid_version(VARCHAR)                 -- banner + echo (kept for the Phase 0 benchmark harness)

anatid_build_csr(edge_table VARCHAR [, max_bytes := BIGINT] [, max_span_factor := BIGINT])
    -> (tenants BIGINT, vertices BIGINT, edges BIGINT, build_ms DOUBLE)

graph_expand(tenant_id BIGINT, seed BIGINT, hops INTEGER
             [, strict := BOOLEAN] [, max_hops := INTEGER])
    -> (entity_id BIGINT, depth INTEGER)

anatid_csr_stats()
    -> (edge_table VARCHAR, current_filter VARCHAR, tenants BIGINT, vertices BIGINT,
        edges BIGINT, bytes BIGINT, build_ms DOUBLE, built_at TIMESTAMP)

anatid_csr_tenants()
    -> (tenant_id BIGINT, vertices BIGINT, edges BIGINT,
        min_entity_id BIGINT, max_entity_id BIGINT, bytes BIGINT)
```

```sql
SELECT * FROM anatid_build_csr('edges_relates');
-- ┌─────────┬──────────┬───────┬──────────┐
-- │ tenants │ vertices │ edges │ build_ms │
-- │      10 │     9963 │ 26479 │     1.39 │
-- └─────────┴──────────┴───────┴──────────┘

SELECT * FROM anatid_csr_stats();
-- edge_table     | edges_relates
-- current_filter | valid_to IS NULL AND tx_to IS NULL
-- tenants        | 10
-- vertices       | 9963
-- edges          | 26479
-- bytes          | 504752
-- build_ms       | 1.385625
-- built_at       | 2026-09-02 19:18:16.528672

SELECT m.memory_id, m.created_at FROM memories m
WHERE m.tenant_id = 3 AND m.valid_to IS NULL
  AND m.memory_id IN (SELECT a.src FROM edges_about a
                      WHERE a.tenant_id = 3
                        AND a.dst IN (SELECT entity_id FROM graph_expand(3, 4211, 2)))
ORDER BY m.created_at DESC, m.memory_id DESC LIMIT 20;
```

### How it works

`anatid_build_csr(t)` reads `SELECT tenant_id, src, dst FROM t` restricted to the **current** rows
and builds one undirected CSR per tenant: an `offsets` array indexed by `entity_id - min_id` and a
`neighbours` array holding both directions of every edge. `graph_expand` is then a breadth-first
search over that, with a byte-per-vertex visited map, emitting `(entity_id, depth)`.

Two deliberate choices carried over from the benchmarked spike, because they are what made it fast:

* **The BFS runs at bind time.** The three positional arguments must be constant, and in exchange
  the optimizer is told the frontier's exact cardinality, which is what makes DuckDB pick the right
  build side for the two semi-joins in 2-hop recall. anatid renders those arguments as SQL integer
  literals (already through `int()`, so they cannot carry SQL) rather than bound parameters.
* **The CSR lives in the `DatabaseInstance`'s `ObjectCache`**, non-evictable, shared by every
  connection, surviving across queries. A rebuild swaps a `shared_ptr`, so readers holding the old
  snapshot keep using it and a rebuild concurrent with reads is safe. A rebuild that *throws* leaves
  the previous snapshot in place — the new one is published only once it is fully built.

### "Current" means anatid's predicate, not the spike's

The CSR includes an edge row when `valid_to IS NULL AND tx_to IS NULL` — anatid's documented
current-state predicate, and exactly what the `relates_undirected` view applies. (The Phase 0 spike
macro filtered on `valid_to` alone. Adding the `tx_to` conjunct provably cannot change the benchmark
answers: `tx_to` is NULL on 100% of the spike dataset's rows, and both binaries were re-run to
confirm — 185,102 `(entity_id, depth)` rows over 1,188 frontiers, zero differences.) Each conjunct
is applied only if the table has that column, so a plain three-column edge list still works and is
treated as all-current. `anatid_csr_stats().current_filter` reports which conjuncts were applied.

---

## The snapshot limitation — read this one

**The CSR does not see `RELATES_TO` writes until it is rebuilt.** It is a snapshot taken by
`anatid_build_csr()`, not a live index. There is no incremental maintenance, no trigger, no
invalidation hook inside DuckDB. An edge inserted after the build is invisible to `graph_expand`,
and an edge deleted or expired after the build is still traversed.

anatid handles this so you mostly do not have to:

* `CsrBackend.note_edge_write()` fires on **every** `RELATES_TO` write (`relate()`), marking the
  snapshot stale.
* While the snapshot is stale, `db.expand_path` reports `"sql"` and every read silently takes the
  pure-SQL path — the identical-result path. **You get a slower correct answer, never a fast wrong
  one.**
* `db.build_csr()` rebuilds and flips it back to `"extension"`.

So the extension speeds up read-mostly graphs, and quietly gets out of the way of write-heavy ones.

### When to rebuild

| situation | do |
|---|---|
| after `load_parquet()` / any bulk load | `db.build_csr()` — `load_parquet(build_csr=True)` does it for you |
| after a batch of `relate()` calls | one `db.build_csr()` at the end of the batch, not per edge |
| steady state, `RELATES_TO` rarely changes | rebuild on a timer or at process start; check `db.expand_path` |
| steady state, `RELATES_TO` changes constantly | don't bother — stay on the SQL path |

Cost of a rebuild, measured: **1.39 ms** at 26,479 edges (the spike `small` dataset) and **30.4 ms**
at 270,637 edges / 99,780 vertices (the `full` dataset). It is a full read of the edge table plus
two passes, so it scales linearly with current edge count. Rebuilding once per bulk load is free;
rebuilding per write is not.

`SELECT * FROM anatid_csr_stats()` tells you what is actually in memory — which table, which
filter, how many tenants/vertices/edges, how many bytes, and `built_at`. Zero rows means no CSR has
been built on this database; it is a health probe, not an error.

---

## The other limitation: entity ids must be dense per tenant

The CSR indexes by `entity_id - min_id`, so the `offsets` array costs **8 bytes for every id between
a tenant's smallest and largest**, occupied or not. **anatid's default ids are 63-bit time-ordered
values** (`anatid.ids.new_id()`: 41 bits of milliseconds, then 22 low bits), so two entities created
10 ms apart are about 42,000,000 ids apart — a third of a gigabyte of offsets for a single edge.

`anatid_build_csr` refuses rather than allocating that, with three bounds, checked in this order:

| bound | default | override |
|---|---|---|
| a tenant's id range must be addressable by the dense array | 2^31 ids | — (hard) |
| **density**: a tenant's id span ≤ `max_span_factor` × its current edge count, floor 4096 | `max_span_factor := 128` | `max_span_factor := 0` disables |
| total snapshot allocation | `max_bytes := 1073741824` (1 GiB) | any BIGINT; `0` disables |

The density bound is the one that catches real anatid ids:

```
SELECT * FROM anatid_build_csr('edges_relates');
-- Invalid Input Error: anatid_build_csr: tenant 0 has 1 current edge(s) but its entity ids span
-- 883069443130717184..883069443193631744 (62914561 ids), which would cost 503316496 bytes of CSR
-- offsets. The CSR is a dense array indexed by (entity_id - min_id); ids this sparse are the shape
-- anatid's default 63-bit time-ordered ids have. Supply dense per-tenant entity ids
-- (anatid.ids.set_allocator), or pass max_span_factor := 0 to build it anyway.
```

The previous snapshot survives any of these refusals, and anatid stays on the SQL path — which is
why a refused build costs you latency and nothing else.

To use the extension in production you must supply dense per-tenant entity ids — via
`anatid.ids.set_allocator()`, or by passing explicit ids to `relate()` (an `int` passes through
`entity_id()` verbatim). The Phase 0 dataset has them by construction (tenant *n* owns ids
`1000n .. 1000n+999`), which is why the benchmark numbers exist at all.

---

## Errors, and the shape of them

Nothing here returns a wrong answer quietly, and nothing crashes.

| you did | you get |
|---|---|
| `graph_expand(...)` with no CSR built | *"no CSR has been built for this database yet. Run `SELECT * FROM anatid_build_csr('edges_relates')` first…"* |
| `anatid_build_csr('typo')` | *"cannot read edge table 'typo': …"* |
| `anatid_build_csr` on a table without `tenant_id`/`src`/`dst` | *"has no column 'tenant_id' (needs tenant_id, src, dst; found: a, b)"* |
| `hops` above the limit | *"hops=31 exceeds the limit of 30 … Pass `max_hops := 31` to allow it for this query."* |
| `hops < 0`, NULL arguments, `max_bytes < 0`, `max_span_factor` out of range | a `BinderException` naming the argument |
| entity ids too sparse for a dense CSR | the tenant id, its edge count, its id span, the byte cost, and the knob that overrides it |

**The hop limit** defaults to **30** — the same bound Kuzu puts on a recursive join — because a
deeper traversal of a connected graph is a full scan wearing a hop count. It is a guard rail, not a
wall: `max_hops := N` raises it for one query (up to 1,000,000). anatid itself only ever asks for 1
or 2.

**An unknown tenant, or a seed with no current edge, is not an error.** `graph_expand` returns just
the seed at depth 0 — precisely what the pure-SQL expansion returns, which is what keeps the two
paths interchangeable. When you are debugging *"why did I get only the seed back?"*, pass
`strict := true` and the same call raises instead, naming the tenants the CSR does hold and the id
range each covers:

```sql
SELECT * FROM graph_expand(42, 1, 2, strict := true);
-- Invalid Input Error: graph_expand: tenant 42 has no edges in the CSR (CSR built from
-- 'edges_relates', 10 tenant(s) [0, 1, 2, 3, 4, 5, 6, 7, ...], 26479 edge(s)). strict := false
-- (the default) returns just the seed. If RELATES_TO rows were written since the build, rebuild
-- with SELECT * FROM anatid_build_csr('edges_relates').
```

`anatid_csr_tenants()` answers the same question without an exception.

---

## Testing

**sqllogictests** (`ext/test/sql/*.test`) — build and expand on a tiny inline graph, hops 0/1/2,
the temporal filter, tenant scoping, unknown tenants, the snapshot contract, every error path, the
hop limit, strict mode, the sparse-id bounds, and both stats functions:

```sh
cd ext && GEN=ninja make test
# [0/3] test/sql/anatid_csr_errors.test
# [1/3] test/sql/anatid_csr_stats.test
# [2/3] test/sql/anatid_csr.test
# All tests passed (179 assertions in 3 test cases)
```

**Python** (`tests/test_extension.py`) — the C++ and SQL paths on the spike's 100k-memory dataset,
plus the hardening, from anatid's own API:

```sh
.venv/bin/python -m pytest tests/test_extension.py -q
```

It skips cleanly, with the build command in the skip reason, when the extension has not been built.
Pointing `ANATID_EXTENSION_PATH` at an older binary skips only the hardening tests; the parity tests
still run.

---

## What it deliberately does not do

* **No incremental maintenance.** See above. That is a design decision, not a TODO: making it
  incremental means hooking DuckDB's storage layer, and the fallback path already returns the right
  answer.
* **No time travel.** The CSR has no notion of valid or transaction time, so anatid uses it only for
  `as_of`-free reads. Any `as_of` query goes to SQL. There is nothing to configure.
* **No shortest paths, no path enumeration, no weights.** It answers "which entities are within k
  hops", which is the only graph question anatid's recall asks.
* **No persistence.** The CSR is rebuilt from the edge table on demand; nothing is written to disk.
