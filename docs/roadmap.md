# Roadmap

anatid's goal is narrow: be the memory layer an AI agent can actually run in production — embedded,
MIT, no server, honest about its limits. Everything below is ordered by that goal, not by what is
interesting to build.

Dates are deliberately absent. Milestones ship when their exit criteria are met.

---

## v0.1 — the core, and the hole nothing else fills *(current)*

Exit criteria: a single-file agent memory you can `pip install`, wire into the OpenAI Agents SDK in
ten lines, and reason about when it goes wrong.

**Shipped**

- **Storage.** One DuckDB file. Node label → table, edge type → table. Seven system columns
  (`valid_from`/`valid_to`, `tx_from`/`tx_to`, `writer`, `episode_id`, `confidence`) default-on
  everywhere, so temporal queries and provenance are storage-level facts, not bolted-on metadata.
  No `PRIMARY KEY`/`UNIQUE` anywhere; 63-bit time-ordered coordination-free ids instead of a counter
  row that would make every write a conflict.
- **Verbs.** `remember`, `recall`, `recall_2hop`, `context`, `supersede`, `reinforce`, `forget`,
  `prune`, `provenance`, `as_of`, plus the graph primitives the graph is unusable without
  (`relate`, `upsert_entity`, `entity_id`, `episode`, `entities_of`). Each is one transaction and
  each takes `now=`/`as_of=` so runs are deterministic and testable.
- **Hybrid recall.** Brute-force cosine + BM25 + 2-hop graph expansion, fused with RRF (k=60), with
  every arm's rank preserved on the result and the BM25 staleness window reported on every call.
- **Bitemporal time travel**, implemented as our own filter (DuckDB has no `AS OF SYSTEM TIME`).
- **Erasure that erases.** `forget(hard=True)` removes the row, its edges, its embedding and its
  provenance, and returns a receipt counting exactly what went.
- **Isolation that isolates.** `DatabasePool` gives one file per tenant, because DuckDB has no
  row- or schema-level access control and a `tenant_id` column is scoping, not a boundary.
- **The optional C++ CSR extension** (`anatid_build_csr` / `graph_expand`), off by default, 29%
  faster 2-hop recall when its constraints are met.
- **OpenAI Agents SDK integration.** A DuckDB-backed `Session` and memory tools with
  `needs_approval=True`, so writes to memory go through the SDK's human-in-the-loop interruption
  flow. As of this release, no other open-source project combines the Agents SDK, DuckDB, a graph
  store, and approval-gated memory writes.
- **MCP server** (`anatid-mcp`), so any MCP client gets the same memory.
- **Evidence.** `docs/benchmarks.md`: 2-hop recall at 1M memories is 2.04-2.88 ms p50 against a
  tuned LadybugDB's 7.35 ms, with 0 result mismatches against a pure-Python oracle.

**Known limits, all documented, none hidden:** no ANN index; the full-text index is not incremental;
single-writer process; brute-force cosine is comfortable to ~1e5 memories per tenant; DuckDB gives
snapshot isolation, not serializability.

---

## v0.2 — meet people where they already are

The wedge is stranded users. Graphiti deprecated its Kuzu driver, Mem0 removed open-source graph
memory in v2.0.0, and Cognee is migrating away. Those projects' users need a graph backend that is
maintained and MIT. v0.2 is about being droppable into what they already run, and readable by
people who already know Cypher.

Exit criteria: an existing Graphiti or Cognee deployment can switch its graph store to anatid by
changing configuration, and its test suite passes.

- **Cypher subset**, lowered to DuckDB plans. Not "Cypher support" — a defined, documented subset
  with a compatibility table saying exactly which clauses are in, which are out, and which are
  planned. `MATCH` with fixed and variable-length patterns, `WHERE`, `RETURN`/`ORDER BY`/`LIMIT`,
  `CREATE`/`MERGE`/`SET`/`DELETE`, `WITH` composition, parameters. The temporal predicate is
  injected automatically so a Cypher query respects `as_of` like every other read.
- **Framework drivers.** A Graphiti graph driver and a Cognee adapter, each maintained against that
  project's own test suite in our CI, plus a Mem0-shaped `GraphStore` for the users its v2.0.0
  removal stranded.
- **Migration tooling.** `anatid import` from Kuzu and LadybugDB database files and from Graphiti /
  Cognee / Mem0 exports, with a report of what did and did not translate. Nobody adopts a memory
  store they cannot get their existing memories into.
- **Node.js bindings** over `@duckdb/node-api`, exposing the same verbs. The agent ecosystem is not
  Python-only, and DuckDB already has the binding we would otherwise have to write.
- **Benchmarks against the frameworks' own workloads**, not just ours, published the same way — with
  the losses included.

---

## v0.5 — run it in production without surprises

Exit criteria: a team can operate anatid for a year without reading the source, and the failure
modes are ones the docs already named.

- **Incremental text search.** Kill the staleness window rather than merely reporting it: a small
  hot index over recent rows, unioned with the cold `fts` index and merged in the background, so
  `recall()` sees writes immediately and a rebuild stops being an event. The reporting fields stay —
  they just stop mattering.
- **Persistent, incrementally-maintained CSR.** Today the adjacency snapshot is in-memory, rebuilt
  in full, invalidated by any `relate()`, and needs dense vertex ids. All four of those are fixable:
  store it in the file, apply edge deltas, keep an internal dense-id mapping so callers keep their
  63-bit ids. This is the item that makes the extension worth turning on by default.
- **Multi-process story, stated and tested.** DuckDB allows one writing process per file. v0.5
  ships the honest pattern rather than pretending otherwise: a writer process plus read-only
  readers, a documented handoff, and a test that proves what happens on lock contention.
- **Graph algorithms over the CSR**: shortest path, k-hop with edge predicates, PageRank and
  community detection for memory consolidation — the operations "which memories matter" needs.
- **Consolidation and forgetting policies.** `prune` today takes age and access-count policies.
  v0.5 adds decay curves, duplicate detection via the vector arm, and summarization hooks, all
  producing receipts and audit rows so a deletion is always explainable.
- **Schema migrations** with a real `MIGRATIONS` chain, forward-tested against files written by
  every prior release.
- **Observability**: structured events per verb, query plans on demand, a `db.explain()` that shows
  which recall arms ran and what they cost.

---

## v1.0 — the parts that need a real engine underneath

Exit criteria: stable on-disk format, semver guarantees on the public API, and the two weak arms
(vector search, adjacency) are no longer weak.

- **An owned ANN index.** The single biggest gap in v0.1. DuckDB ships no ANN index, so anatid's
  vector arm is a brute-force scan with a documented ~1e5-per-tenant ceiling. v1.0 ships an HNSW (or
  IVF-PQ, decided by measurement) index as an anatid DuckDB extension, with **MVCC-correct
  incremental maintenance** — an index that goes stale on write, or that ignores rows created inside
  a transaction, is not acceptable; the spike watched exactly that failure mode in another engine.
  Recall@k against brute-force truth gets published, not just latency.
- **CSR persistence and default-on graph acceleration**, building on v0.5.
- **duckdb-wasm**: anatid in the browser and at the edge. DuckDB already compiles to WebAssembly;
  the work is our extension and the Python-free surface. Local-first agents that keep their memory
  on the user's device are the point.
- **Format and API stability.** Schema v1 files readable by every 1.x release, deprecations with a
  minor-version runway, and a compatibility test suite that opens files written by old versions.
- **Security posture**: a documented threat model for a memory store that holds personal data, an
  encryption-at-rest story, and erasure guarantees that survive an audit.

---

## The DuckDB 2.0 dependency — call it out

Two v1.0 items are **not fully in our control**, and pretending otherwise would be exactly the kind
of overclaiming this project is trying to avoid.

**The extension ABI.** DuckDB extensions are compiled against a specific DuckDB version — the
spike's extension is pinned to v1.5.5 — and the C++ extension API is not stable across releases.
Today that means an anatid extension build per DuckDB version, which is a distribution problem
(a matrix of binaries) rather than an engineering one, but it is a real tax on shipping the ANN
index and the persistent CSR. DuckDB's C extension API exists to fix this; how much of it we can
build against, and when, decides whether v1.0's extensions ship as one artifact or as a matrix.

**Storage format stability.** anatid's promise that a v1.0 file opens in every 1.x release can only
be as strong as DuckDB's own format guarantee underneath it. We inherit that promise; we cannot
issue it.

So: **anatid v1.0 targets DuckDB 2.0**, and if DuckDB 2.0 slips, the ANN index and the persistent
CSR slip with it — or ship first on 1.x with a rebuild-per-version distribution matrix, which is the
fallback we would rather not choose. What we need from 2.0 is a stable extension ABI and a format
commitment. That is our expectation of another project's roadmap, not a promise from it, and we will
say so every time the question comes up. Everything in v0.2 and v0.5 is pure Python plus SQL and
depends on none of it.

---

## Things we are deliberately not doing

- **A server.** anatid is embedded. If you need a network service, DuckDB has one story and it is
  not ours to invent.
- **Rebuilding the OpenAI Agents SDK's human-in-the-loop machinery.** The SDK already has
  `needs_approval`, `RunResult.interruptions`, and serializable `RunState`. anatid supplies the
  approval-gated *tools* and the memory behind them; the approval flow itself belongs to the SDK.
- **A distributed graph database.** Multi-node is a different product with different tradeoffs.
- **Claiming ACID guarantees DuckDB does not give.** Snapshot isolation is snapshot isolation.
