# Roadmap

anatid's goal is narrow: be the memory layer an AI agent can run in production, embedded, MIT
licensed, with its limits written down. Everything below is ordered by that goal.

This document carries no dates. A milestone ships when its exit criteria are met, and the items
under each milestone state intent rather than commitment.

## v0.1: the core (current)

Exit criteria: a single-file agent memory you can `pip install`, wire into the OpenAI Agents SDK in
ten lines, and reason about when it goes wrong.

### Shipped

- Storage is one DuckDB file. A node label maps to a table and an edge type maps to a table. Seven
  system columns (`valid_from`/`valid_to`, `tx_from`/`tx_to`, `writer`, `episode_id`, `confidence`)
  are on by default everywhere, so every temporal query and every provenance walk reads columns
  that are already on the row. There is no `PRIMARY KEY` or `UNIQUE` constraint anywhere; ids are
  63-bit, time-ordered and coordination-free, in place of a counter row that would make every write
  a conflict.
- The verbs are `remember`, `recall`, `recall_2hop`, `context`, `supersede`, `reinforce`, `forget`,
  `prune`, `provenance`, `as_of`, plus the graph primitives (`relate`, `upsert_entity`, `entity_id`,
  `episode`, `entities_of`). Each write verb is one transaction and takes `now=`; the temporal read
  verbs take `as_of=`, so runs are deterministic and testable.
- Hybrid recall combines brute-force cosine, BM25, and 2-hop graph expansion, fused with RRF
  (k=60), with every arm's rank preserved on the result and the BM25 staleness window reported on
  every call.
- Bitemporal time travel, implemented as anatid's own filter because DuckDB has no
  `AS OF SYSTEM TIME`.
- `forget(hard=True)` removes the row, its edges, its embedding and its provenance rows, and
  returns a receipt counting exactly what went.
- `DatabasePool` gives one file per tenant. DuckDB has no row- or schema-level access control, so a
  `tenant_id` column scopes a query rather than isolating it.
- The optional C++ CSR extension (`anatid_build_csr` / `graph_expand`), off by default, is 29%
  faster on 2-hop recall when its constraints are met.
- OpenAI Agents SDK integration: a DuckDB-backed `Session` and memory tools that declare
  `needs_approval`, so writes to memory go through the SDK's human-in-the-loop interruption flow.
  As of this release, no other open-source project combines the Agents SDK, DuckDB, a graph store,
  and approval-gated memory writes.
- MCP server (`anatid-mcp`), so any MCP client gets the same memory.
- Evidence in `docs/benchmarks.md`: 2-hop recall at 1M memories is 2.04-2.88 ms p50 against a tuned
  LadybugDB's 7.35 ms, with 0 result mismatches against a pure-Python oracle.

### Limitations

No ANN index; the full-text index is not incremental; one writer process per file; brute-force
cosine is comfortable to roughly 1e5 memories per tenant; DuckDB gives snapshot isolation rather
than serializability.

## v0.2: framework drivers and a Cypher subset

Graphiti deprecated its Kuzu driver, Mem0 removed open-source graph memory in v2.0.0, and Cognee is
migrating away. Those projects' users need a graph backend that is maintained and MIT licensed.
v0.2 is about being droppable into what they already run, and readable by people who already know
Cypher.

Exit criteria: an existing Graphiti or Cognee deployment can switch its graph store to anatid by
changing configuration, and its test suite passes.

- A Cypher subset lowered to DuckDB plans: a defined, documented subset with a compatibility table
  saying exactly which clauses are in, which are out, and which are planned. `MATCH` with fixed and
  variable-length patterns, `WHERE`, `RETURN`/`ORDER BY`/`LIMIT`, `CREATE`/`MERGE`/`SET`/`DELETE`,
  `WITH` composition, parameters. The temporal predicate is injected automatically so a Cypher
  query respects `as_of` like every other read.
- Framework drivers: a Graphiti graph driver and a Cognee adapter, each maintained against that
  project's own test suite in anatid's CI, plus a Mem0-shaped `GraphStore` for the users its
  v2.0.0 removal stranded.
- Migration tooling. `anatid import` reads Kuzu and LadybugDB database files and Graphiti, Cognee
  and Mem0 exports, and reports what did and did not translate.
- Node.js bindings over `@duckdb/node-api`, exposing the same verbs. The agent ecosystem is not
  Python-only, and DuckDB already has the binding anatid would otherwise have to write.
- Benchmarks against the frameworks' own workloads as well as anatid's, published the same way,
  with the losses included.

## v0.5: run it in production without surprises

Exit criteria: a team can operate anatid for a year without reading the source, and the failure
modes are ones the docs already named.

- Incremental text search, closing the staleness window instead of reporting it: a small hot index
  over recent rows, unioned with the cold `fts` index and merged in the background, so `recall()`
  sees writes immediately. `bm25_stale` and `pending_fts_rows` remain on the result and report zero.
- A persistent, incrementally maintained CSR. Today the adjacency snapshot is in memory, rebuilt in
  full, invalidated by any `relate()`, and dependent on dense vertex ids. All four are fixable:
  store it in the file, apply edge deltas, and keep an internal dense-id mapping so callers keep
  their 63-bit ids. This is the item that would make the extension worth turning on by default.
- A multi-process story that is documented and tested. DuckDB allows one writing process per file.
  v0.5 ships the pattern that follows from that: a writer process plus read-only readers, a
  documented handoff, and a test that proves what happens on lock contention.
- Graph algorithms over the CSR: shortest path, k-hop with edge predicates, PageRank, and community
  detection for memory consolidation, which are the operations "which memories matter" needs.
- Consolidation and forgetting policies. `prune` today takes age and access-count policies. v0.5
  adds decay curves, duplicate detection via the vector arm, and summarization hooks, all producing
  receipts and audit rows so a deletion is always explainable.
- Schema migrations with a real `MIGRATIONS` chain, forward-tested against files written by every
  prior release.
- Observability: structured events per verb, query plans on demand, and a `db.explain()` that shows
  which recall arms ran and what they cost.

## v1.0: the parts that need a real engine underneath

Exit criteria: stable on-disk format, semver guarantees on the public API, and the two weak arms
(vector search, adjacency) are no longer weak.

- An owned ANN index, the single biggest gap in v0.1. DuckDB's team-maintained `vss` extension
  provides an HNSW index, but its on-disk persistence is experimental and not recommended for
  production, so anatid's vector arm is a brute-force scan with an enforced ~1e5-per-tenant
  ceiling. v1.0 ships an HNSW index (or
  IVF-PQ, decided by measurement) as an anatid DuckDB extension, with MVCC-correct incremental
  maintenance. An index that goes stale on write, or that ignores rows created inside a
  transaction, is not acceptable: the spike hit exactly that in Grafeo 0.5.42, whose HNSW index
  ignores nodes created by `INSERT`/`CREATE` statements while its text index does not
  (`spike/results/grafeo.full.json`, `notes[2]` item f). Recall@k against brute-force truth gets
  published alongside latency.
- CSR persistence and default-on graph acceleration, building on v0.5.
- duckdb-wasm, putting anatid in the browser and at the edge. DuckDB already compiles to
  WebAssembly; the work is the anatid extension and a Python-free surface. The target is a
  local-first agent whose memory stays on the user's device.
- Format and API stability: schema v1 files readable by every 1.x release, deprecations with a
  minor-version runway, and a compatibility test suite that opens files written by old versions.
- Security posture: a documented threat model for a memory store that holds personal data, an
  encryption-at-rest story, and erasure guarantees that survive an audit.

## The DuckDB 2.0 dependency

Two v1.0 items are outside anatid's control.

The first is the extension ABI. DuckDB extensions are compiled against a specific DuckDB version,
and the C++ extension API is not stable across releases; the spike's extension is pinned to v1.5.5.
Today that means an anatid extension build per DuckDB version, which is a distribution problem (a
matrix of binaries) rather than an engineering one, but it is a real tax on shipping the ANN index
and the persistent CSR. DuckDB's C extension API exists to fix this. How much of it anatid can
build against, and when, decides whether v1.0's extensions ship as one artifact or as a matrix.

The second is storage format stability. anatid's promise that a v1.0 file opens in every 1.x
release can only be as strong as DuckDB's own format guarantee underneath it. anatid inherits that
promise and cannot issue it.

anatid v1.0 therefore targets DuckDB 2.0. If DuckDB 2.0 slips, the ANN index and the persistent CSR
slip with it, or ship first on 1.x with a rebuild-per-version distribution matrix, which is the
fallback anatid would rather not choose. What anatid needs from 2.0 is a stable extension ABI and a
format commitment. That is an expectation of another project's roadmap, not a promise from it.
Everything in v0.2 and v0.5 is pure Python plus SQL and depends on none of it.

## Deliberately out of scope

- A server. anatid is embedded. Callers who need a network service have DuckDB's own story for
  that, and it is not anatid's to invent.
- Rebuilding the OpenAI Agents SDK's human-in-the-loop machinery. The SDK already has
  `needs_approval`, `RunResult.interruptions`, and serializable `RunState`. anatid supplies the
  approval-gated tools and the memory behind them; the approval flow itself belongs to the SDK.
- A distributed graph database. Multi-node is a different product with different tradeoffs.
- Claiming ACID guarantees DuckDB does not give. anatid documents DuckDB's isolation level as
  snapshot isolation and makes no serializability claim.
