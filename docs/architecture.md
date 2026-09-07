# anatid architecture

anatid is a Python library over one DuckDB database file. In the default embedded profile there is
no server, no daemon, and no background thread; the optional server profile, which exists only for
callers who need several processes writing one file, is described in
[`server.md`](server.md) and changes none of what follows. Every write verb (`remember`, `supersede`, `forget`, `relate`, `reinforce`,
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

entities             entity_id, tenant_id, kind, name, entity_key (generated)  + system columns
memories             memory_id, tenant_id, content, kind, embedding FLOAT[N],
                     created_at, access_count, last_access_at, version       + system columns
episodes             episode_id, tenant_id, source, content, kind, created_at + system columns

edges_about          memory -> entity   (edge_id, src, dst, tenant_id, weight, version) + system cols
edges_relates        entity -> entity   (edge_id, src, dst, tenant_id, rel_kind, version) + system
edges_supersedes     newer memory -> older memory (edge_id, src, dst, tenant_id, tx_from, writer)

relates_undirected   VIEW: current edges_relates unioned with itself, src/dst swapped

anatid_fts_documents / _docmap / _dict / _stats    the BM25 sidecar tables (§6)

anatid_index_registry     one row per derived index DEFINITION: name, kind, source table and id
                          column, delta mode, params, enabled. In the FILE, not on the handle.
anatid_index_generations  one row per built generation: index, tenant (NULL = file-wide), number,
                          watermark, built_at, validated, published, stats
anatid_index_journal      one row per change a generation has not absorbed: index, tenant, doc id,
                          change_seq (from a sequence), op, absorbed_by
anatid_idx_*              a generation's own storage, named for the generation it belongs to
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

### Version rows

`memories`, `edges_about` and `edges_relates` carry a `version` column, and their rows are
immutable. `memory_id` is the logical id; `version` numbers the physical rows of that id from 1. A
correction (`supersede`, soft `forget`, `unrelate`, a `reinforce` that changes confidence) closes
the current version's `tx_to` and inserts version n+1 with the corrected valid interval, in one
transaction. Nothing rewrites `valid_to` in place.

That is what makes the transaction axis real. "What did the database believe on January 2 would be
true on January 4" has to return the open-ended belief version 1 carried on January 2, even though
a January 3 forget later closed it, and it does: every read predicate over `(valid_at, tx_at)`
selects at most one version of a logical id. `db.versions(id)` lists them oldest first and
`Provenance.versions` carries the same list beside the SUPERSEDES chain between logical ids.
`access_count` and `last_access_at` are usage counters, updated in place on the live version, and
are deliberately not bitemporal.

Rows written before the 3 -> 4 migration carry over as version 1. Corrections made to them before
the migration were rewritten in place and are not recoverable.

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
extension needs (see §3).

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

## 2. Visibility, and the derived-index framework

None of anatid's three retrieval accelerators is the source of truth. The full-text index, the CSR
adjacency structure and the vector index are all derived from the canonical tables, which are.
Everything in this section follows from that one relationship.

### visible_at: one predicate, generated once

Every read path renders its tenant and time predicate through `anatid.visibility`:

```python
from anatid import Visibility, visible_at

sql, params = Visibility.at(tenant_id, as_of).predicate("m")
#   current:  "m.tenant_id = ? AND m.valid_to IS NULL AND m.tx_to IS NULL"
#   as_of:    "m.tenant_id = ? AND m.valid_from <= ? AND (m.valid_to IS NULL OR m.valid_to > ?)
#              AND m.tx_from <= ? AND (m.tx_to IS NULL OR m.tx_to > ?)"

visible_at(tenant_id, valid_time=t1, transaction_time=t2)   # the same object, named for the axes
```

`Visibility.admits(row)` is the Python mirror, half-open intervals and all, for an accelerator
that has to post-filter in Python rather than in SQL. `tests/test_visibility.py` checks two things:
that no module under `src/anatid` writes the predicate by hand (a static scan of each module's SQL
string literals, with the abstraction itself exempt), and that every public read verb agrees with a
pure-Python oracle for two tenants across five time scopes, on both expansion paths.

The hot path is unchanged by this. `recall_2hop_ids` renders
`Visibility.predicate("m", inline_tenant=True)`, which produces the same SQL text with the same
zero bound parameters that 0.1.1 emitted by hand.

### Base generation, journal, tombstone

A derived index is a versioned base generation plus a journal written in the same
transaction as the canonical row.

```
write transaction
   ├── canonical row                    the source of truth, always correct
   └── anatid_index_journal row         (index, tenant, doc_id, change_seq, op)

read
   ├── base generation, pinned for the duration of the read
   ├── + the journal rows the generation has not absorbed   (insert / close / purge)
   └── then, and only then, the tenant and time predicate from anatid.visibility

maintenance (maintain_indexes(), explicit, no background thread)
   ├── build the next generation beside the live one
   ├── validate it against the oracle
   └── publish: one UPDATE of one metadata row
```

Five properties follow, and each one is a test:

A write is in the next read. The journal row and the canonical row are one transaction, so a
reader either sees both or neither, including a reader inside the writing transaction.

Id reuse is representable. The journal is ordered by `change_seq`, drawn from a DuckDB
sequence, and the newest operation for a `(tenant, doc)` key wins. `insert 5 -> build -> purge 5
-> insert a new 5` therefore ends as an insert, which a delta set and a tombstone set held
separately could not express.

One tenant's change cannot touch another's document. Every journal row is keyed
`(tenant_id, doc_id)`. `memory_id` is unique within a tenant, not within a file, so a bare id would
let a purge in tenant 1 suppress tenant 2's document 42 out of a file-wide generation.

A handle that holds no accelerator code still journals. Definitions live in
`anatid_index_registry`, in the file. `IndexRegistry.emit` is driven by those rows, not by which
Python objects this handle happens to hold, so a maintenance script that opened the file with
`Anatid.open(...)` and never imported `anatid.fts` still records every write the full-text index
will need.

Publication does not interrupt a read. A build writes new tables beside the live ones;
publishing flips `published` on one row inside a transaction; a read pins a generation under a
process-wide per-file lifecycle lock, so a generation a reader chose cannot be retired and dropped
underneath it.

### An index may be stale, corrupt or absent

The SQL path over the canonical tables is always the oracle, so an index that cannot be used costs
latency and nothing else. What a fallback must not do is happen silently, so every read path
reports a machine-readable reason:

| `HealthReason` | what it means |
|---|---|
| `fresh` | a validated generation is published and its journal is within policy |
| `stale_generation` | published but due for a rebuild, or invalidated by a bulk load or an erasure |
| `unvalidated` | published with `force=True`; usable and flagged |
| `historical_query` | the read carries `as_of` and this index only knows current state |
| `rebuild_in_progress` | nothing published and a build is running |
| `load_failure` | the storage or the extension would not load, or a statement against it raised |
| `damaged_base` | the storage is present and queryable but no longer holds what its build recorded |
| `absent` | no generation has ever been published |

`db.index_health()` returns one report per index (`as_of=` asks what a historical read would do),
`FtsSearch.reason`, `VectorSearch.reason` and `ExpandPath.reason` carry it per read, and
`db.doctor()` raises `unusable_derived_index` when a published generation cannot serve reads.

`damaged_base` is the one that needed looking for. Damage that raises is caught for nothing;
damage that leaves a structure queryable and quietly incomplete is not. Each index checks one
cheap invariant on every read: full text compares its source table's cardinality with its document
map's, and the vector arm compares the base's row count with what the build recorded and the
number of candidates the approximate scan returned with the number that were available. Neither is
a full validation, which costs a scan of the corpus and is what a rebuild runs.

### Erasure reaches the accelerators

`forget(hard=True)` deletes the document from every generation's storage inside the purge
transaction, deletes its journal rows rather than tombstoning them (a tombstone would keep the
erased id in the file), and lowers any generation watermark that was the erased id. A generation
whose storage cannot delete one document is invalidated instead, which takes it out of service
until the next rebuild. `ForgetReceipt.derived_rows_deleted` and `.invalidated_generations` report
both halves, and the erasure tests scan every table in the file, including generation storage
nobody named, for the erased id and its text.

### What it costs

Measured on this machine, DuckDB 1.5.5, macOS arm64.

```
remember() with entities, accelerators=False        2.78 ms p50
remember() with entities, default (fts + csr)       3.49 ms p50     one journal INSERT per index
```

The default is on because a searchable write is worth 0.7 ms; `Anatid.open(accelerators=False)`
turns it off for a write-heavy database that never searches text.

## 3. Graph traversal: SQL first, CSR as an accelerator

2-hop recall is the query anatid is built around: from a seed entity, expand 1 and 2 hops over
`RELATES_TO` in both directions, then return the current memories `ABOUT` anything in that
frontier, newest first.

There are three implementations and they must return identical rows.

The SQL path is always available and is the oracle. Two semi-joins over `relates_undirected`
produce the frontier, which is pushed into a semi-join against `edges_about` and then `memories`.
DuckDB's optimizer turns this into hash semi-joins with dynamic min/max and Bloom filters pushed
into the scans. Measured 2.88 ms p50 at 1M memories.

The derived CSR index is a generation of the framework in §2, attached by default. Each generation
owns two tables, a dense vertex map `(tenant_id, vertex_id, entity_id)` and an edge list in that
dense space, and the C++ extension holds an in-memory structure named for the generation, so
generations coexist and a pinned read keeps the numbering it started with. A read resolves the
journal into added and removed entity pairs and expands level by level: hop k+1 is the base
neighbours of hop k, minus the pairs every one of whose base edges has been retired, plus the pairs
a pending live edge connects. Retiring one of two parallel edges therefore does not disconnect a
pair. The merge runs either in one SQL statement over the generation's edge list or inside
`graph_expand` itself; the C++ merge is faster at every journal size measured, so `strategy="auto"`
picks it when the extension is loaded.

Dense ids never leave the module. Seeds and results are entity ids on every path, which is what
makes the extension usable on an ordinary anatid database at all: its raw build over
`edges_relates` still refuses anatid's sparse 63-bit ids, and correctly so.

The 0.1 snapshot path is still there and still what `build_csr()` drives. It is a single unnamed
CSR over the current `edges_relates` rows, used only while `CsrBackend.fresh` (no `RELATES_TO`
write since the build). The precedence is: a published generation, then the fresh 0.1 snapshot,
then SQL.

### How the CSR stays MVCC-correct

The CSR is derived from `edges_relates`, and four rules keep it from answering a query the SQL path
would answer differently:

1. A generation's changes are journalled in the writing transaction, and the expansion applies
   them. That is what replaced 0.1's "any write makes the whole structure unusable".
2. The 0.1 snapshot has no journal, so it keeps 0.1's rule: every `relate()` calls
   `CsrBackend.note_edge_write()`, which marks it stale, and a stale snapshot is never used.
3. Everything that declines says why. `frontier_sql()` returns an `ExpandPath`, which IS the
   string `"sql"` / `"csr"` / `"extension"` and also carries `.reason` and `.explain()`. A
   historical `as_of` read always declines: a current-state structure cannot answer it.
   `db.expand_path` is the forecast for the next current-state read on this handle's own tenant,
   `db.last_expansion` is what the last read actually did.
4. The paths are checked to agree. The spike ran 1,000 R1 queries per engine against a
   pure-Python oracle (`spike/bench/common.py::reference_r1`) and 200 verify queries across
   engines: 0 mismatches, SQL vs CSR vs LadybugDB. `tests/test_csr_framework.py` adds a
   1,000-mutation random walk comparing the merged expansion with the SQL oracle after every step.

Two limits still apply. A generation is built in full rather than updated in place, so a large
journal eventually costs more than the expansion saves (measured: 1.50 ms against 0.88 ms of pure
SQL at about 550 journal rows on the spike graph), which is what `MaintenancePolicy`'s ratio
trigger exists to prevent. And the in-memory structure is never evicted by DuckDB's object cache,
so memory grows with the number of resident generations; retiring one frees it.

When no structure can answer, `require_csr_extension()` raises `ExtensionUnavailable` and the SQL
path answers the query.

### Integer literals on the fast path

On the current-state, no-kind-filter fast path, integer arguments are rendered as SQL literals
rather than bound parameters. duckdb-python 1.5.5 attempts `import pandas` twice per bound
parameter, so about 14 times per `execute()` of the seven-parameter 2-hop statement, and with
pandas absent each attempt re-walks `sys.path`, costing about 0.6 ms per call. Literal rendering
took `recall_2hop_ids` from 1.911 ms to 1.345 ms p50 at 100k memories, single-threaded, on the
machine that measurement was taken on. Treat the ratio rather than the absolute: the same
comparison re-run on a loaded 10-core laptop reproduces the improvement but lands at 1.60 ms, and
the 0.2.0 release was gated on a paired A/B against the published 0.1.1 wheel on one machine
(1.598 ms against 1.608 ms p50, 600 queries per round, four interleaved rounds) rather than on the
number above. Only values already coerced with `int()` take this path; strings, timestamps and
embeddings are always bound.
`tests/test_core.py::test_literal_and_parameterized_recall_paths_agree` asserts both paths return
identical rows.

## 4. Isolation: the file is the boundary

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
- One process holds a file, and it holds it against everybody. Measured on duckdb 1.5.5, a second
  process is refused whether it asks read-write or read-only: `IO Error: Could not set lock on
  file`. So there is no "one writer, many read-only readers" arrangement to build. Since 0.3.0
  anatid ships the arrangement that does follow from the lock, as an opt-in second profile: one
  process owns the files and the others reach it over a socket, for reads as well as writes
  ([`server.md`](server.md)). It relocates the single writer process, it does not remove it, and
  it changes nothing above: the isolation level is still DuckDB's snapshot isolation, and
  `ConflictError` still means what it means here.

## 5. Time: two axes, filtered by generated SQL

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
`anatid.visibility.temporal_predicate()` generates the filter (§2);
`AsOf(valid_time=..., tx_time=...)` separates the two axes, so "what was true then" can be asked
apart from "what did we know then". Three consequences follow:

- Time travel only reaches back as far as the rows still in the table. A hard purge removes the row
  from every as-of view too, because the row is gone.
- `forget(hard=False)` closes the current version's `tx_to` and inserts the next version with
  `valid_to` set, and writes an audit row: the memory stops being current, history stays intact,
  and `as_of()` before the forget still finds the belief the database held then, open-ended, as it
  was. `forget(hard=True)` deletes every version, its ABOUT and SUPERSEDES edges, its embedding,
  its provenance rows and its rows in every derived index's storage and journal, and returns a
  `ForgetReceipt` counting exactly what was removed. A right-to-erasure request needs the second
  one.
- A current-state accelerator cannot answer an `as_of` read at all, so it does not try: the graph
  and vector arms take the SQL path and report `historical_query`. The full-text index is the
  exception and answers from a generation, because its base holds document identity and content,
  both of which are the same for every version of a memory, and the time predicate is applied to
  the canonical rows it joins.

## 6. The `recall()` pipeline

`recall()` runs up to three independent retrieval arms and fuses them with weighted Reciprocal
Rank Fusion (k=60). An arm runs only when its input is present, so `recall(seed_entity="Ada")` with
no query is pure graph and `recall(query="x", seed_entity=None)` is pure BM25. The weights are not
equal: when the vector arm runs it leads (vector 1.0, graph 0.5, text 0.25), otherwise the text arm
does (text 1.0, graph 0.5), and `arm_weights=` overrides any of them by name. Before the fusion the
graph arm's candidates are reordered by the query's own signal, cosine with an embedding and BM25
without one, instead of newest first, so the arm votes for what the question is about rather than
for what was written last. Both settings come from the answer-quality benchmark
([`quality.md`](quality.md)), where equal votes and a newest-first graph arm cost the fusion
questions the vector arm alone had right; `hits.weights` reports what was used.

```
                     db.recall(query="coffee roast",
                               embedding=[...], seed_entity="Ada", k=3)
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        │ needs embedding=                │ needs query= + fts index        │ needs seed_entity=
        ▼                                 ▼                                 ▼
  ┌───────────────┐              ┌──────────────────┐            ┌─────────────────────┐
  │  VECTOR arm   │              │    BM25 arm      │            │     GRAPH arm       │
  │ exact cosine  │              │ Okapi k1=1.2     │            │ frontier = seed +   │
  │ scan, or an   │              │ b=0.75 in SQL    │            │ 1..2 hops over      │
  │ HNSW          │              │ (NOT match_bm25) │            │ RELATES_TO          │
  │ generation    │              │ over a generation│            │ over a CSR          │
  │ + the journal │              │ + the journal,   │            │ generation + the    │
  │ (opt in)      │              │ or an exact scan │            │ journal, the 0.1    │
  │               │              │                  │            │ snapshot, or SQL    │
  │ memories.     │              │ anatid_idx_fts_* │            │         │           │
  │ embedding     │              │ or anatid_fts_*  │            │         ▼           │
  │ FLOAT[N]      │              │                  │            │ edges_about ⋈       │
  │               │              │                  │            │ memories            │
  │ score is      │              │ one corpus over  │            │ ORDER BY created_at │
  │ always exact  │              │ base + journal   │            │ DESC, memory_id DESC│
  └───────┬───────┘              └────────┬─────────┘            └──────────┬──────────┘
          │ top `candidates` (50)         │ top 50                          │ top 50, then
          │                               │                                 │ re-ranked by cosine
          │                               │                                 │ (or BM25) to the query
          └───────────────┬───────────────┴─────────────────┬───────────────┘
                          ▼                                 │
              ┌──────────────────────┐                      │
              │  RRF fuse, k=60      │◄─────────────────────┘
              │  score = Σ w/(60+r)  │   every arm's rank kept, per hit; w from
              │                      │   default_arm_weights() or arm_weights=
              └──────────┬───────────┘
                         ▼
        hydrate(memories)  +  about_names(edges_about ⋈ entities)
                         ▼
        RecallHits  ── .arms  .weights  .seeds  .bm25_stale  .pending_fts_rows  .notes
          └─ RecallHit(memory, score, rank, vector_rank, text_rank, graph_rank,
                       vector_score, text_score, about)

   candidate generation happens first; the tenant and time predicate from anatid.visibility is
   then applied to the CANONICAL rows, before the top-N cut.
```

### The three arms, and what each one costs

The vector arm has two backends and `"exact"` is the default. Exact is a brute-force
`array_cosine_similarity` scan over the tenant's visible embeddings; it is also the oracle every
other backend is measured against and the fallback from every state in which another one cannot be
used. Its cost is linear in one tenant's row count rather than the file's. Measured at 64 dims on
the spike hardware with DuckDB's default thread count, all rows in one tenant: 2.0 ms p50 at 10k,
8.6 ms at 100k (an independent run of the same measurement got 11.4 ms) and 23.3 ms at 1M.
`BRUTE_FORCE_CEILING = 100_000` is enforced: `recall(embedding=...)` raises
`BruteForceCeilingError` when the scan would cover more rows than that, unless the caller passes
`allow_slow=True`.

`Anatid.open(vector_backend="duckdb_vss")` opts in to an HNSW generation instead. The generation
holds a frozen copy of the tenant's visible embeddings with an HNSW index over it; a read takes
`topn * overfetch` candidates from it, unions an exact scan of the journal's pending documents and
anything past the watermark, subtracts the journal's tombstones, applies the visibility predicate
to the canonical `memories` rows, and takes the exact cosine top-k. The approximate structure only
ever changes which rows were considered; the score that ranks them is bit-identical to the oracle's.

It is opt in for three reasons, all measured. DuckDB documents HNSW persistence as experimental
with write-ahead-log and crash-recovery caveats. A persisted HNSW index loses the `ef_search` it
was created with, so recall at 95,000 rows falls from 0.999 to 0.641 after a reopen unless the
setting is reissued per connection, which `anatid.vector` does. And it does not pay below roughly
15,000 rows per tenant: 0.85x the exact scan at 9,500 rows, 2.2-2.8x at 95,000. Recall at k
against the exact oracle is 1.0000 at k=10 and 0.9982-0.9984 at k=50, at both sizes.

The BM25 arm computes Okapi BM25 (k1=1.2, b=0.75) in SQL rather than calling `match_bm25`.
Identical ranking, and the spike measured 19.9 ms against 29.2 ms p50 at 1M rows, because it skips
the extension's per-document correlated lookup. Do not "simplify" `recall.py` back to the macro.

DuckDB's own full-text index is not incremental: `PRAGMA create_fts_index` rebuilds it wholesale
and rows written afterwards are invisible to it. `anatid.fts` builds incremental behaviour above
that, on the framework in §2, and `Anatid.open(accelerators=True)` (the default) attaches it. One
generation covers the whole file, because `create_fts_index` has a fixed per-call cost and
per-tenant scoping is a predicate rather than a structure; the document key is
`'<tenant_id>:<memory_id>'`, unique across tenants where `memory_id` alone is not.

A search is one statement: the base generation's documents the journal has NOT touched, unioned
with the ones it HAS, re-read from `memories` and tokenised with the index's own tokenizer, then
the visibility predicate applied to the canonical rows before the top-N cut. Tombstones are
subtracted from the base rather than from the answer, because a journalled close means "the base's
view of this document is out of date", not "this document is gone", and for an `as_of` read a
closed document is a legitimate candidate.

Scores are the part that had to be got right. Two independently built BM25 corpora do not produce
comparable scores, so anatid does not build a second corpus and does not rank-fuse. It
reconstructs the one set of corpus statistics that describes base plus journal together:
`num_docs` and `avgdl` from the untouched base documents (lengths the build recorded) plus the
rescanned ones (measured now), and `df` the same way. The two sets are disjoint and their union is
exactly what a rebuild would index, which gives the property the tests assert after random
mutations: a rebuild never changes an answer. The first implementation rescored the journal
against the base's stale statistics and reordered 53 of 120 random queries; it was replaced.

Consequences worth knowing:

- `FtsStatus.stale` means something different on each half. On 0.1.1's index it is "N rows the
  arm cannot see". On a generation a write is searchable at once, so it is "the answer would be
  incomplete", which happens only when there is no usable generation AND the corpus is above
  `SCAN_CEILING`. `pending_rows` is how many documents a search re-reads, not how many are hidden.
- With no generation published the answer is still exact, by scanning. Below a few tens of
  thousands of documents that scan is also the faster path, which is why `SCAN_CEILING` is 100,000
  rather than lower. Above the ceiling with no generation, 0.1.1's index answers if the file still
  has one, and the result says so.
- `rebuild_fts_index()` still exists and still means "build now". On the framework it builds,
  validates and publishes a generation, with reads answering from the previous one throughout;
  `maintain_indexes()` does the same on `MaintenancePolicy`'s triggers (10,000 rows, 5%, 900 s).
- This is what a rebuild is for. Measured on this machine, top-10 over one tenant:

  ```
  corpus     0.1.1 index    derived, nothing built    derived, generation    + 500 journalled
   10,000       8.7 ms            10.1 ms                  11.5 ms               14.3 ms
  100,000      17.9 ms            27.1 ms                  19.1 ms               21.4 ms
  ```

  The 0.1.1 column is the only one that cannot see a write made since its last rebuild.

The graph arm is the 2-hop frontier of §3, semi-joined into `edges_about` and `memories`.

### Where fusion happens, and what that costs

RRF runs in Python rather than in one SQL statement, because each hit has to report which arms
found it and at what rank, and the result object has to be able to answer `bm25_stale`. The price
is measurable: hybrid recall is 11.6 ms p50 at 100k memories against the spike macro's 8.58 ms,
split between work the macro did not do (hydrating full `Memory` rows, resolving ABOUT names, the
staleness check) and per-arm bound-parameter overhead. If a future version needs those milliseconds
back, the fix is a single-statement CTE fusion that projects each arm's rank out; the per-arm
reporting described above is the reason it has not been written.

## 7. What runs where

```
your process
├── anatid (pure Python)          verbs, visibility predicates, the derived-index framework,
│                                 RRF fusion, health and staleness accounting
├── duckdb (C++, embedded)        storage, MVCC, joins, BM25, array_cosine_similarity
├── duckdb fts extension          the postings a full-text generation is built on
├── duckdb vss extension (opt.)   the HNSW index behind vector_backend="duckdb_vss"
└── anatid extension (C++, opt.)  anatid_build_csr / graph_expand / anatid_drop_csr, one named
                                  in-memory CSR per generation, loaded unsigned
```

Nothing runs on a schedule. There is no background thread, no maintenance daemon and no worker
process: `maintain_indexes()` is a call you make after a batch of writes, on a timer of your own,
or when `index_health()` reports a stale generation. The server profile adds a process, not a
schedule: its write queue drains on worker threads inside that one process and it still rebuilds an
index only when a caller asks.

`Anatid.connection` returns the raw DuckDB cursor for this thread. Data in the file is queryable by
anything that speaks SQL, joinable against Parquet and CSV in place, and readable by any DuckDB
client. Raw SQL runs outside the wrapper, so a caller writing it supplies the tenant predicate and
the temporal filter.
