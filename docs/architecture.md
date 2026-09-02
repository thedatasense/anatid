# anatid architecture

anatid is a Python library over one DuckDB database file. There is no server, no daemon, and no
background thread. Every write verb (`remember`, `supersede`, `forget`, `relate`, `reinforce`,
`episode`, `entity_id`) is one DuckDB transaction issued from your process. `prune` is a query plus
one transaction per memory it forgets, so a failure part-way leaves the earlier deletions
committed; read `PruneReport.memory_ids` from a `dry_run` first. Read verbs (`recall`,
`recall_2hop`, `context`, `get`, `provenance`, `stats`) run their statements without an enclosing
transaction, so a concurrent commit can land between a `recall`'s arms and its hydration step. Wrap
the call in `db.transaction()` when you need all of it on one snapshot.

This document describes what is in the file, how the graph is traversed, what isolation the library
provides, how time travel is implemented, and how a `recall()` call is answered. Where DuckDB does
not provide something anatid needs, the text says so.

## 1. Storage layout

One node label becomes one table and one edge type becomes one table.

```
anatid_meta          one row: schema_version, embedding_dim, versions, contract,
                     fts_indexed_at / fts_indexed_rows / fts_indexed_max_id (the BM25 watermarks)
anatid_audit         append-only trail of forget/supersede/prune actions (happened_at; "at" is a
                     reserved word). related_memory_id is a COLUMN, not text inside `reason`, so
                     forget(hard=True) can find and delete every row that names an erased id.

entities             entity_id, tenant_id, kind, name                       + system columns
memories             memory_id, tenant_id, content, kind, embedding FLOAT[N],
                     created_at, access_count, last_access_at               + system columns
episodes             episode_id, tenant_id, source, content, kind, created_at + system columns

edges_about          memory -> entity   (edge_id, src, dst, tenant_id, weight)     + system columns
edges_relates        entity -> entity   (edge_id, src, dst, tenant_id, rel_kind)   + system columns
edges_supersedes     newer memory -> older memory (edge_id, src, dst, tenant_id, tx_from, writer)

relates_undirected   VIEW: current edges_relates unioned with itself, src/dst swapped
```

`db.create_node_label(...)` and `db.create_edge_type(...)` add your own labels and edge types with
the same shape.

### System columns, and why they are on by default

Every node and edge table carries seven columns that are not user data:

| column | meaning |
|---|---|
| `valid_from` / `valid_to` | when the fact was true in the world; `valid_to IS NULL` = still true |
| `tx_from` / `tx_to` | when the database believed it; `tx_to IS NULL` = still believed |
| `writer` | which agent or process asserted it |
| `episode_id` | the raw source this was derived from |
| `confidence` | the asserter's own confidence, 0..1 |

They default to on because the temporal verbs are defined in terms of them. `as_of()` is a filter
over the four timestamps. `supersede()` closes `valid_to` rather than deleting. Soft `forget()`
closes `tx_to` and leaves an audit row. `provenance()` walks `episode_id` and `writer` back through
`edges_supersedes`. Without the columns none of that works, so the built-in tables do not offer a
switch to turn them off. User-defined labels do offer one
(`create_node_label(..., system_columns=False)`), where a plain lookup table with no history is a
legitimate thing to want.

`edges_supersedes` is the exception. It records when a write happened rather than a fact about the
world, so it carries transaction time only (`tx_from`, `writer`) and is never re-dated.

### No primary keys, and where ids come from

There is no `PRIMARY KEY` or `UNIQUE` constraint anywhere. In DuckDB both create an implicit ART
index, and ART index maintenance is the dominant cost on the write path. The only ART indexes
anatid creates by default are `memories(memory_id)` (needed for `supersede`'s point update),
`entities(tenant_id, name)` (entity resolution) and `episodes(episode_id)`. Five more are
available via `Anatid.open(indexes=[...])` and are off because the spike measured them as neutral
for 2-hop recall.

Ids come from `anatid.ids.new_id()`: 41 bits of milliseconds since 2020, 12 bits of per-process
random worker id, and 10 bits of sequence, for 63 bits that are time-ordered and coordination-free.
A shared counter row was rejected: under DuckDB's optimistic MVCC every `remember()` would be a
write-write conflict on the same row, which would destroy the concurrent-append behavior the
benchmark measured. The cost of that choice is that a cross-process collision needs the same worker
id, the same millisecond, and the same sequence number. Every verb accepts an explicit id, and
`ids.set_allocator()` replaces the scheme entirely, which is how to supply the dense ids the CSR
extension needs (see §2).

### Physical clustering

DuckDB has no clustered index, but it does prune row groups by min/max zone maps, so the order rows
are written in matters. `load_parquet()` and `recluster()` write each table in the order the spike
measured as fastest:

```
memories       ORDER BY tenant_id, created_at DESC, memory_id DESC
edges_about    ORDER BY tenant_id, dst, src
edges_relates  ORDER BY tenant_id, src
entities       ORDER BY tenant_id, entity_id
episodes       ORDER BY tenant_id, episode_id
```

`recluster()` is compaction. Appends stay correct without it. It rewrites a table with
`CREATE OR REPLACE TABLE ... AS SELECT ... ORDER BY`, so it drops indexes anatid does not know
about and is not safe to run alongside writers.

## 2. Graph traversal: SQL first, CSR as an accelerator

2-hop recall is the query anatid is built around: from a seed entity, expand 1 and 2 hops over
`RELATES_TO` in both directions, then return the current memories `ABOUT` anything in that
frontier, newest first.

There are two implementations and they must return identical rows.

The SQL path is always available. Two semi-joins over `relates_undirected` produce the frontier,
which is pushed into a semi-join against `edges_about` and then `memories`. DuckDB's optimizer
turns this into hash semi-joins with dynamic min/max and Bloom filters pushed into the scans.
Measured 2.88 ms p50 at 1M memories.

The CSR path needs the optional C++ extension. `anatid_build_csr()` reads the current
`edges_relates` rows once and materializes a per-tenant compressed sparse row adjacency structure
in the `DatabaseInstance` object cache. `graph_expand(tenant, seed, hops)` is a table function that
runs the BFS at bind time, so the optimizer sees the exact frontier cardinality before planning the
rest of the query. Measured 2.04 ms p50.

### How the CSR stays MVCC-correct

The CSR is a cache of `edges_relates`, and three rules keep it from answering a query the SQL path
would answer differently:

1. The structure is a snapshot, and anatid tracks that. Every `relate()` calls
   `CsrBackend.note_edge_write()`, which marks the snapshot stale.
2. A stale snapshot is not used. `frontier_sql()` emits the SQL form instead, which reads the same
   MVCC snapshot as the rest of the statement, and `db.expand_path` reports `"sql"`. Nothing is
   served from a stale structure.
3. Both paths are checked to agree. The spike ran 1,000 R1 queries per engine against a
   pure-Python oracle (`spike/bench/common.py::reference_r1`) and 200 verify queries across
   engines: 0 mismatches, SQL vs CSR vs LadybugDB.

`build_csr()` buys latency and does not change the answer; a wrong answer from it is a bug, and the
oracle tests in rule 3 above are the check for it. Two further limits apply. The CSR is rebuilt in
full, with no incremental update, and it requires dense per-tenant vertex ids. anatid's 63-bit
time-ordered ids are not dense, so the extension is usable only when you supply your own dense
entity ids through `ids.set_allocator()` or explicit ids, as the spike dataset does. When the
extension path is not live, `require_csr_extension()` raises `ExtensionUnavailable` and the SQL path
answers the query.

### Integer literals on the fast path

On the current-state, no-kind-filter fast path, integer arguments are rendered as SQL literals
rather than bound parameters. duckdb-python 1.5.5 attempts `import pandas` twice per bound
parameter, so about 14 times per `execute()` of the seven-parameter 2-hop statement, and with
pandas absent each attempt re-walks `sys.path`, costing about 0.6 ms per call. Literal rendering
took `recall_2hop_ids` from 1.911 ms to 1.345 ms p50 at 100k memories, single-threaded. Only values
already coerced with `int()` take this path; strings, timestamps and embeddings are always bound.
`tests/test_core.py::test_literal_and_parameterized_recall_paths_agree` asserts both paths return
identical rows.

## 3. Isolation: the file is the boundary

DuckDB has no row-level security, no schema-level access control, and no notion of a user. Anything
a connection can reach, it can read. Two consequences follow.

Inside one file, `tenant_id` is a scoping predicate rather than a boundary. Every anatid query
carries it, and `resolve_tenant()` raises `TenantIsolationError` when a file-per-tenant handle is
asked for another tenant, but `db.connection.execute("SELECT * FROM memories")` reads every tenant
in that file. Raw SQL bypasses the wrapper, and the wrapper is the only thing enforcing the filter.
`Namespace(isolation=Isolation.SCOPED)` names that arrangement in the type system.

One file per tenant is the arrangement that enforces a boundary, and `DatabasePool` manages a
directory of such files:

```python
pool = DatabasePool("/var/lib/anatid/t_{tenant}.anatid", embedding_dim=1536)
db = pool[42]                     # opens /var/lib/anatid/t_42.anatid, Isolation.FILE_PER_TENANT
```

The boundary is then the filesystem's, enforced by file permissions, and a bug in a tenant
predicate cannot leak across it. Cross-tenant reads are explicit and one-directional:
`pool.attach_read_only(host, other)` closes the other tenant's handle, because DuckDB refuses to
attach a file the same process already holds open ("Unique file handle conflict"), and attaches it
`READ_ONLY`. If a different process holds it open read-write, the attach fails outright; that is
DuckDB's file lock, not anatid's policy.

### The transaction contract

DuckDB's MVCC is optimistic and gives snapshot isolation rather than serializability.

- Appends never conflict. Concurrent `remember()` from many threads all commit. The spike measured
  4 writers + 2 readers for 30 s at 152-189 W1/s and 583-825 R1/s with 0 errors and no retry logic
  at all.
- Two concurrent updates to the same row abort the second. anatid translates that into
  `ConflictError`, which carries `retryable=True` and the underlying cause. `supersede()` and
  `reinforce()` are the verbs that update rows. anatid does not retry, because whether the write
  should be re-derived from a fresh read depends on the caller.
- One process holds the write lock on a file. Other processes can open it read-only. anatid is a
  single-writer-process library; multi-process write coordination is not something it provides and
  not something DuckDB provides for it.

## 4. Time: two axes, filtered by generated SQL

anatid is bitemporal. Every fact carries valid time, when it was true in the world, and transaction
time, when the database believed it.

```
valid_from ──────────────── valid_to      "Ada preferred dark roast, until March 31"
tx_from    ──────────────── tx_to         "we believed that from Mar 1 until we were told otherwise"
```

The current-state predicate is `valid_to IS NULL AND tx_to IS NULL`. That is one conjunct more than
the spike's macro used, and it costs about 0.01 ms.

`db.as_of(t)` returns an `AsOfView` whose reads are scoped to `t`. The scoping is anatid's own
`WHERE` clause. DuckDB has no `AS OF SYSTEM TIME` and nothing rewinds.
`schema.temporal_predicate()` generates the filter; `AsOf(valid_time=..., tx_time=...)` separates
the two axes, so "what was true then" can be asked apart from "what did we know then". Two
consequences follow:

- Time travel only reaches back as far as the rows still in the table. A hard purge removes the row
  from every as-of view too, because the row is gone.
- `forget(hard=False)` closes `tx_to` and writes an audit row: the memory stops being current,
  history stays intact, and `as_of()` before the forget still finds it. `forget(hard=True)` deletes
  the memory, its ABOUT and SUPERSEDES edges, its embedding, and its provenance rows, and returns a
  `ForgetReceipt` counting exactly what was removed. A right-to-erasure request needs the second
  one.

## 5. The `recall()` pipeline

`recall()` runs up to three independent retrieval arms and fuses them with Reciprocal Rank Fusion
(k=60). An arm runs only when its input is present, so `recall(query="x")` is pure BM25 and
`recall(seed_entity="Ada")` is pure graph.

```
                     db.recall(query="coffee roast",
                               embedding=[...], seed_entity="Ada", k=3)
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        │ needs embedding=                │ needs query= + fts index        │ needs seed_entity=
        ▼                                 ▼                                 ▼
  ┌───────────────┐              ┌──────────────────┐            ┌─────────────────────┐
  │  VECTOR arm   │              │    BM25 arm      │            │     GRAPH arm       │
  │ array_cosine_ │              │ fts_main_        │            │ frontier = seed +   │
  │ similarity    │              │ memories.dict/   │            │ 1..2 hops over      │
  │ brute force,  │              │ terms/docs/stats │            │ relates_undirected  │
  │ no ANN index  │              │ Okapi k1=1.2     │            │ (CSR ext or SQL)    │
  │               │              │ b=0.75 in SQL    │            │                     │
  │               │              │ (NOT match_bm25) │            │                     │
  │               │              │ NOT incremental  │            │         │           │
  │ memories.     │              │                  │            │         ▼           │
  │ embedding     │              │ fts index built  │            │ edges_about ⋈       │
  │ FLOAT[N]      │              │ by rebuild_fts_  │            │ memories            │
  │               │              │ index()          │            │ ORDER BY created_at │
  │ ~1e5/tenant   │              │                  │            │ DESC, memory_id DESC│
  └───────┬───────┘              └────────┬─────────┘            └──────────┬──────────┘
          │ top `candidates` (50)         │ top 50                          │ top 50
          └───────────────┬───────────────┴─────────────────┬───────────────┘
                          ▼                                 │
              ┌──────────────────────┐                      │
              │  RRF fuse, k=60      │◄─────────────────────┘
              │  score = Σ 1/(60+r)  │   every arm's rank kept, per hit
              └──────────┬───────────┘
                         ▼
        hydrate(memories)  +  about_names(edges_about ⋈ entities)
                         ▼
        RecallHits  ── .arms  .bm25_stale  .pending_fts_rows  .notes
          └─ RecallHit(memory, score, rank, vector_rank, text_rank, graph_rank,
                       vector_score, text_score, about)

   every table read above also carries:  tenant_id = ?  AND  <temporal predicate for as_of>
```

### The three arms, and what each one costs

The vector arm is a brute-force `array_cosine_similarity` scan over the tenant's current
embeddings. DuckDB ships no ANN index, so anatid has none either, and the cost is linear in one
tenant's row count rather than the file's. Measured at 64 dims on the spike hardware with DuckDB's
default thread count, all rows in one tenant: 2.0 ms p50 at 10k, 8.6 ms at 100k (an independent run
of the same measurement got 11.4 ms) and 23.3 ms at 1M. `BRUTE_FORCE_CEILING = 100_000` is
documented and not enforced. At that ceiling a recall already costs 9-11 ms, so past roughly 1e5
memories per tenant this arm is the wrong tool, and an owned ANN index is on the roadmap for v1.0.

The BM25 arm uses DuckDB's `fts` extension index, which is not incremental. Rows inserted after
`PRAGMA create_fts_index` are invisible to BM25 until the index is rebuilt, and rebuilding drops
and recreates the whole `fts_main_memories` schema. The API accounts for that in four places:

- anatid computes BM25 in SQL off the fts extension's index tables (`fts_main_memories.dict`,
  `terms`, `docs`, `stats`; Okapi k1=1.2, b=0.75) rather than calling `match_bm25`. Identical
  ranking, and the spike measured 19.9 ms vs 29.2 ms p50 at 1M rows, because it skips the
  extension's per-document correlated lookup. Do not "simplify" `recall.py` back to the macro.
- `rebuild_fts_index()` is explicit, records both the row count and the largest `memory_id`
  present (`anatid_meta.fts_indexed_rows` / `fts_indexed_max_id`), and also builds an ART index on
  `fts_main_memories.terms(termid)`. The spike A/B'd that index at 8.57 ms vs 12.51 ms BM25 p50 at
  100k memories, for a 0.7 s build.
- `fts_status()` compares both watermarks against the table. The id watermark exists because a row
  count alone can cancel out: one insert plus one hard purge leaves `count(*)` where it was while
  the new document stays invisible to BM25. Ids are time-ordered, so any insert raises
  `max(memory_id)`. Neither watermark can see a raw `UPDATE memories SET content = ...`; no anatid
  verb issues one (`supersede` inserts a new row), so that is reachable only through
  `db.connection`, and if you do it you must rebuild yourself. `fts_status(deep=True)` adds a
  `max(tx_from)` scan that catches back-dated rows.
- Every `recall()` result reports `.bm25_stale` and `.pending_fts_rows`, and logs a warning on the
  `anatid.recall` logger. `on_stale_fts="error"` raises `StaleIndexError` instead.

The staleness window is the interval between rebuilds. A rebuild is O(corpus), so anatid never
triggers one inside a read; choosing the interval is the caller's job.

The graph arm is the 2-hop frontier of §2, semi-joined into `edges_about` and `memories`.

### Where fusion happens, and what that costs

RRF runs in Python rather than in one SQL statement, because each hit has to report which arms
found it and at what rank, and the result object has to be able to answer `bm25_stale`. The price
is measurable: hybrid recall is 11.6 ms p50 at 100k memories against the spike macro's 8.58 ms,
split between work the macro did not do (hydrating full `Memory` rows, resolving ABOUT names, the
staleness check) and per-arm bound-parameter overhead. If a future version needs those milliseconds
back, the fix is a single-statement CTE fusion that projects each arm's rank out; the per-arm
reporting described above is the reason it has not been written.

## 6. What runs where

```
your process
├── anatid (pure Python)          verbs, tenant predicates, RRF fusion, staleness accounting
├── duckdb (C++, embedded)        storage, MVCC, joins, BM25, array_cosine_similarity
└── anatid extension (C++, opt.)  anatid_build_csr / graph_expand, loaded unsigned
```

`Anatid.connection` returns the raw DuckDB cursor for this thread. Data in the file is queryable by
anything that speaks SQL, joinable against Parquet and CSV in place, and readable by any DuckDB
client. Raw SQL runs outside the wrapper, so a caller writing it supplies the tenant predicate and
the temporal filter.
