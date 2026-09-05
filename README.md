# anatid

anatid is an embedded graph memory for AI agents, built on DuckDB and released under the MIT
license. The database is a single file, and the default way to use it has no server to run and no
daemon to supervise. There is an optional server profile for the one case that needs it, described
below.

Install it and an agent gains a memory that stores entities, the edges between them, and facts
attached to both. That memory records two kinds of time: what was true, and what the agent believed
at any past instant. Queries run three ways at once, through vector similarity, Best Match 25
(BM25) text scoring, and graph traversal, fused into a single ranked list. Writes can be routed
through the human-in-the-loop approval flow in the OpenAI Agents Software Development Kit (SDK), so
an agent proposes a change to its own memory and a person decides whether it lands.

Every retrieval structure in anatid is derived rather than canonical. The full-text index, the
graph adjacency structure, and the optional vector index are built the same way: a versioned base
generation, plus a journal written inside the same transaction as the row it describes, merged on
every read before any tenant or time filter runs. So a write is findable by the next read with
nothing rebuilt. An index that has gone stale, or been damaged, or was never built at all, costs
latency rather than correctness, and every fallback reports which of eight reasons applied. The SQL
path over the canonical tables remains the oracle.

Why this project exists: [Kuzu was archived on 2025-10-10](https://github.com/kuzudb/kuzu).
Graphiti deprecated its Kuzu driver, Mem0 removed open-source graph memory in v2.0.0, and Cognee
began migrating away. A number of people were left with an embedded graph memory and nowhere
obvious to go.

Two-hop recall over 1,000,000 memories has a median latency of 2.88 ms on DuckDB against 7.35 ms on
a tuned LadybugDB, the maintained MIT fork of Kuzu. That is a factor of 2.5, and the two engines
return identical result identifier lists. The measurement came before the library, and it is why
anatid sits on DuckDB. Method and caveats are in [`docs/benchmarks.md`](docs/benchmarks.md).

## Install

```bash
pip install anatid                    # just duckdb
pip install "anatid[agents]"          # + the OpenAI Agents SDK integration
pip install "anatid[mcp]"             # + the Model Context Protocol server
```

To track `main` instead:

```bash
pip install "git+https://github.com/thedatasense/anatid"
```

anatid runs on Python 3.10 through 3.13 and requires one dependency, `duckdb>=1.5`. Continuous
integration runs the test suite on Linux, macOS and Windows across all four Python versions,
including the two integration suites, which is why the `[dev]` extra installs `openai-agents` and
`mcp`. Tests that load the 100k-row benchmark dataset, and those needing the compiled C++
extension, skip in continuous integration because neither artifact lives in the repository. Both
run locally before a release.

## Quickstart

```python
from anatid import Anatid, utcnow

vec = [0.0] * 63 + [1.0]                                        # your embedding model's output

with Anatid.open("agent.anatid", tenant=1, embedding_dim=64) as db:
    db.relate("Ada", "Kestrel", rel_kind="leads")               # an entity to entity edge
    m = db.remember("Ada prefers dark roast coffee",            # a fact, filed under 2 entities
                    entities=["Ada", "coffee"], kind="preference",
                    embedding=vec, writer="agent-1",
                    episode="Standup 2026-03-01: Ada takes it dark roast.")  # raw evidence first
    t0 = utcnow()
    hits = db.recall("coffee", embedding=vec, seed_entity="Ada", k=3)   # findable already
    print(hits[0].content, hits[0].sources, "| bm25_stale:", hits.bm25_stale)

    new = db.supersede(m.memory_id, "Ada switched to decaf")    # closes the old row, keeps it
    print("old still current?", db.get(m.memory_id).is_current) # False, history intact
    print("at t0:", [x.content for x in db.as_of(t0).recall_2hop("Ada")])
    print("evidence:", db.provenance(new.memory_id).source_text)
```

```
Ada prefers dark roast coffee ('vector', 'text', 'graph') | bm25_stale: False
old still current? False
at t0: ['Ada prefers dark roast coffee']
evidence: Standup 2026-03-01: Ada takes it dark roast.
```

Running the block above produces exactly that. Nothing was rebuilt before the `recall` call. The
write was journalled inside its own transaction and the text arm merged it. Calling
`db.maintain_indexes()` folds the journal into fresh generations when you want the speed of a built
index, and `db.index_health()` reports whether that is due, and why.

A longer commented walkthrough covering `recall_2hop`, `forget(hard=True)` and `stats()` lives in
[`examples/quickstart.py`](examples/quickstart.py). It needs no API key and finishes in under a
second.

For something closer to how memory tends to fail in practice, run
[`examples/dinner_party.py`](examples/dinner_party.py). Six months of ordinary household facts, a
cook who asks whether Friday's menu is safe, and an allergy that neither the question nor any
single stored sentence mentions. The graph walks from the dinner to a guest to an ingredient to the
dish. Word search alone returns the recipe cards and stops.

## The verbs

| verb | what it does |
|---|---|
| `remember(content, entities=[...])` | write a fact and the ABOUT edges that make it reachable |
| `recall(query, embedding=, seed_entity=)` | hybrid retrieval: cosine, BM25 and 2-hop graph, fused with reciprocal rank fusion |
| `recall_2hop(seed)` / `context(entity)` | pure graph recall; `context` defaults to 0 hops |
| `supersede(old_id, content)` | replace a belief, keeping the old one closed and linked |
| `unrelate(a, b)` | close an edge that stopped being true |
| `reinforce(id)` / `prune(...)` | strengthen what gets used, drop what does not |
| `forget(id, hard=False)` | stop believing, with the audit trail kept, or erase completely |
| `as_of(t)` | every read, as the database saw the world at `t` |
| `provenance(id)` | the supersession chain, the raw episodes, every writer involved |
| `relate(a, b)` / `upsert_entity` / `episode` | the graph and evidence primitives underneath |
| `update(id, content, expected_version=n)` | compare and swap: read the version, write, one transaction |
| `atomic(callback)` | re-run the whole callback on a retryable conflict, with jittered backoff |
| `maintain_indexes()` / `index_health()` | build the derived indexes that are due; report each one's state |
| `doctor()` | integrity and upkeep checks, with severities and samples |

Each write verb is exactly one DuckDB transaction. Reads run their statements outside an explicit
transaction, so a concurrent commit can land between a recall's arms and its hydration step. Wrap
the call in `db.transaction()` when you need a single snapshot.

`prune` behaves differently: a query, then one transaction per memory it forgets. A failure part-way
leaves earlier deletions committed. Taking its `dry_run` list first shows what it will touch.

Write verbs accept `now=` and the temporal read verbs accept `as_of=`, which keeps tests
deterministic. Function forms exist as well, through `from anatid.verbs import remember`. And
`db.connection` hands you the raw DuckDB cursor whenever you want SQL. The memory is ordinary
tables, joinable against your Parquet and CSV files in place.

## Two deployment profiles

Embedded is the default and nothing changes about it. One process opens the file and writes from as
many threads as it likes. If your agent is one process, this is the whole answer, and it is faster
than the alternative.

The server profile exists for one case: two or more processes that must write the same memory.
DuckDB gives one process exclusive use of a database file, and a second process is refused even
when it asks for read-only access, measured on duckdb 1.5.5 as `IO Error: Could not set lock on
file`. So one process owns the files and the others reach it over a Unix domain socket or over the
Hypertext Transfer Protocol (HTTP). Reads cross the wire along with writes, because while the
server holds a file nothing else can open it.

| | embedded | server |
|---|---|---|
| When to use it | one process, any number of threads | several processes writing one memory |
| How you open it | `Anatid.open(...)` | `AnatidClient.connect(...)` |
| What runs | nothing extra | one server process you supervise |
| Where the tenant boundary is | one file per tenant | the same, plus a principal checked before any file is opened |
| Backpressure | none; a thread waits its turn | a typed `BusyError` with a wait hint, over HTTP a 429 |
| Backups | copy the file while nothing holds it | the server takes them, because nothing else can open the file |

Switching is one line. Every verb keeps its name, its parameters and its return type:

```python
from anatid import Anatid                                   # embedded
from anatid.server.client import AnatidClient               # server

with Anatid.open("agent.anatid", tenant=1) as memory:       # one process
    memory.remember("the deploy at 14:05 rolled back cleanly")

with AnatidClient.connect("/run/anatid/anatid.sock", tenant=1) as memory:   # many
    memory.remember("the deploy at 14:05 rolled back cleanly")
```

The wire is not free, and the cost depends entirely on which call you make. Measured on one tenant
of 3,000 memories with 384-dimension embeddings, 150 timed repetitions after warmup, median:

| call | embedded | over the socket | ratio |
|---|---:|---:|---:|
| `get()` | 0.395 ms | 0.863 ms | 2.18x |
| `recall()`, vector arm on | 18.011 ms | 18.966 ms | 1.05x |
| `recall_2hop()` | 3.991 ms | 4.821 ms | 1.21x |
| `stats()` | 1.725 ms | 1.996 ms | 1.16x |

Read that plainly. For `recall`, the call this profile exists to serve, the wire is close to free.
For `get`, the cheapest read anatid has, it is not: half a millisecond of fixed cost doubles the
call, and the ratio flatters the server only because `recall` is slow. Writes cost more too, at
roughly 0.71 times embedded throughput with four concurrent writers, 213 against 300 writes per
second on this machine. Most of the fixed cost is the JavaScript Object Notation (JSON) codec
rather than the socket, and `ServerConfig(embeddings="f32")` removes about a quarter of it on
embedding-carrying replies.

Running one is a command:

```bash
anatid-server start \
  --socket /run/anatid/anatid.sock \
  --pool '/var/lib/anatid/tenant-{tenant}.anatid' \
  --tenant 1 --tenant 2
```

[`docs/server.md`](docs/server.md) covers the security model, the operator surface, systemd and
launchd units, health and readiness, backup and restore, and what the shutdown guarantees.
[`examples/server_demo.py`](examples/server_demo.py) demonstrates the lock, the server, and several
processes writing one memory, in three acts and under a minute.

## Why DuckDB, with numbers

Phase 0 was a benchmark, run before any of the library existed: 1,000,000 memories, 2.3M edges, ten
tenants, four engines, the same operations under identical semantics, all checked against a
pure-Python oracle.

Two-hop recall is the query shape agent memory hits hardest. Over 1,000 queries on a single thread:

| engine | p50 | p95 | load | on disk | concurrent reads |
|---|---:|---:|---:|---:|---:|
| DuckDB with the C++ CSR extension | 2.04 ms | 3.07 ms | 4.8 s | 481 MiB | 825/s |
| DuckDB, plain SQL | 2.88 ms | 3.50 ms | 4.6 s | 434 MiB | 583/s |
| LadybugDB 0.20.2, tuned | 7.35 ms | 28.73 ms | 16.2 s | 1,158 MiB | 147/s |

The kill criterion set beforehand was to abandon DuckDB if it ran more than five times slower. It
came in at 0.39x on plain SQL and 0.28x with the extension. At the 95th percentile those figures
are 0.12x and 0.11x.

All three engines returned identical result identifier lists across 1,000 oracle-checked queries
and 200 post-write verification queries. LadybugDB's figure is the fastest of six Cypher
formulations across two thread settings; the naive formulation ran 16 times slower, and reporting
that one would have flattered DuckDB.

Where DuckDB loses is worth stating plainly. Hybrid recall runs about 22% slower, 16.4 ms against
20.0 ms median, though no engine in the run had an approximate nearest neighbour index, so that
comparison measures scan speed. Concurrent readers cost DuckDB writers real throughput, dropping
from 397 writes per second with writers alone to between 152 and 189 once two readers join.
LadybugDB with `enable_multi_writes=True` commits more writes per second than DuckDB does.

Full tables covering every phase, the mixed workload, concurrency, correctness, and nine
limitations of the benchmark itself are in [`docs/benchmarks.md`](docs/benchmarks.md). Raw JSON
with per-operation latency arrays sits in `spike/results/`.

## OpenAI Agents SDK integration

The [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) already carries what
human-in-the-loop review needs: `needs_approval=True` on a `function_tool`,
`RunResult.interruptions`, a serializable `RunState`, and `state.approve()` alongside
`state.reject()`. It also defines a `Session` protocol for conversation history, with backends for
SQLite, SQLAlchemy and Redis.

Missing from it are a DuckDB session, graph memory, and approval-gated memory writes. As far as we
can establish, no open-source project combines all four of the Agents SDK, DuckDB, a graph store,
and human approval on memory writes. anatid supplies the missing three while rebuilding none of the
SDK's machinery.

```python
from agents import Agent, Runner
from anatid import Anatid
from anatid.integrations.openai_agents import AnatidSession, create_memory_tools

db = Anatid.open("agent.anatid", tenant=1)
session = AnatidSession("conv-1", db)                # conversation history, same file as the graph
tools = create_memory_tools(db, session=session)     # 3 read tools, 3 write tools

agent = Agent(name="assistant", tools=tools)
result = await Runner.run(agent, "Ada switched to decaf, remember that", session=session)

while result.interruptions:                          # writes stop here; reads never do
    state = result.to_state()
    for item in result.interruptions:
        print(item.tool_name, item.raw_item.arguments)   # "anatid_remember" {"content": ...}
        state.approve(item)                              # or state.reject(item)
    result = await Runner.run(agent, state, session=session)
```

Writes are gated and reads run straight through. The tools `anatid_remember`, `anatid_supersede`
and `anatid_forget` carry `needs_approval`, while `anatid_recall`, `anatid_context` and
`anatid_provenance` do not. Nothing reaches the database until somebody approves.

The approval policy is a callable, so you can shape it. `approve_low_risk()` waves through small
ordinary writes and still stops for hard deletes. Unless you opt out explicitly,
`anatid_forget(hard=True)` always requires approval, since a hard forget removes the row, its edges,
its embedding and its provenance together.

Approval can also happen later, and somewhere else entirely. `RunStateStore(db)` parks the SDK's
serialized `RunState` in the same anatid file, so an interrupted run can be reviewed and resumed
minutes or days afterwards by a different process. That turns approval into a review queue rather
than a blocking prompt.

History and knowledge stay joinable, because `AnatidSession` writes conversation turns into a table
inside the same DuckDB file as the memory graph. Calling `await session.entities_mentioned()`
becomes one SQL join against `entities`, rather than two round-trips to two different stores, and
`memories_written_here()` reports what a given conversation committed to memory.

## Model Context Protocol server

```bash
pip install "anatid[mcp]"
anatid-mcp --db memory.anatid        # stdio; point Claude Desktop, Claude Code or Cursor at it
```

That exposes the memory verbs over the Model Context Protocol (MCP), so any MCP client gains
persistent, bitemporal, graph-shaped memory. The write side offers `remember`, `relate`,
`supersede`, `reinforce`, `forget`, `prune` and `rebuild_fts_index`. The read side offers `recall`,
`context`, `get`, `provenance` and `stats`. Those are MCP tool names; the `anatid_`-prefixed names
belong to the Agents SDK integration above. Passing `--read-only` registers the read tools alone.

Identifiers cross that boundary as decimal strings, never as JSON numbers. anatid identifiers
exceed what JavaScript integers carry safely, and a client that parsed them as numbers would
silently address the wrong row. Tools accept either spelling on the way in.

One deliberate escape hatch exists: a `sql` tool, off by default, for questions the verbs do not
answer. How many memories per kind, say, or show me the audit trail. It is read-only, and DuckDB
enforces that in three layers rather than a regular expression over the query text. DuckDB's own
statement classifier admits only SELECT and EXPLAIN, and every statement in the text must pass. A
scan of DuckDB's parse tree rejects file-reading functions and base-table names that are not plain
identifiers, since DuckDB's replacement scan would otherwise turn `SELECT * FROM '/etc/passwd.csv'`
into an ordinary SELECT. Execution then happens inside `BEGIN TRANSACTION READ ONLY` on a private
cursor that is always rolled back.

`from anatid.integrations.mcp import build_server` embeds the server in your own process.

## Limitations

Everything here is measured, or documented in the source. Behaviour that contradicts the
documentation and is absent from this list is a bug, and we would like the report.

| area | where it stands |
|---|---|
| Vector search | Exact scan by default. An HNSW generation is opt-in |
| Full-text | Journalled writes are searchable at once; rebuilds buy latency |
| Concurrency | One writing process per file, many threads inside it. Several processes need the server profile |
| Isolation | Snapshot, with retryable conflicts. Not serializable |
| Tenancy | One file per tenant is the real boundary |
| Query language | The verbs above, plus SQL. No Cypher yet |
| Maintenance | A call you make, not a background thread |

Several of those deserve more than a row.

The default vector backend performs an exact scan. Opting into
`Anatid.open(vector_backend="duckdb_vss")` builds a Hierarchical Navigable Small World (HNSW)
generation, which measured recall at k of 1.0000 for k=10, and between 0.9982 and 0.9984 for k=50,
against the exact oracle at 9,500 and 95,000 rows per tenant, running 2.2 to 2.8 times faster at
the larger size. It stays opt-in for three reasons. DuckDB documents HNSW persistence as
experimental, with write-ahead-log and crash-recovery caveats. A persisted HNSW index silently
loses its `ef_search` setting across a reopen, which anatid works around by reissuing the setting
per connection. And below roughly 15,000 rows per tenant, the exact scan tends to be faster anyway.
The 1M and 10M measurements named in the promotion criterion have not been taken. Since 0.1.1,
`recall(embedding=...)` raises `BruteForceCeilingError` when an exact scan would cover more than
`BRUTE_FORCE_CEILING = 100_000` rows, unless you pass `allow_slow=True`.

DuckDB's own full-text index does not update incrementally, and anatid builds incremental behaviour
above it rather than exposing that limitation. A write is journalled in its own transaction and
merged into the next search, so `.bm25_stale` reads False and the row is findable. What you still
choose is when to pay for a rebuild, either through `maintain_indexes()` on a policy or
`rebuild_fts_index()` by hand. Merging costs read latency in proportion to the journal rather than
the corpus, measured at an extra 2.3 ms for 500 journalled writes over a 100,000-document corpus.
With no generation published at all, a search scans the corpus exactly, which is refused above
`SCAN_CEILING = 100_000` documents per tenant.

An index can be damaged in ways a read cannot afford to detect. Every read checks one cheap
invariant per index and falls back to the oracle with `HealthReason.damaged_base` when it fails. A
base that is structurally consistent yet wrong, postings lost from under a document map that still
points at them, gets caught by `validate()` during a rebuild rather than by a read.

One writing process per file is DuckDB's model, and the engine enforces it. A second read-write
process cannot even open the file, failing with `IO Error: Could not set lock on file`. Many
threads inside that one process write concurrently, and appends never conflict, measured at zero
errors across a 30-second six-thread benchmark with no retry logic. When you need several
processes, the server profile above puts one of them in charge of the files. That does not change
the model, it relocates it: one process still owns each file, and it is a single point of failure
rather than a cluster.

Isolation is snapshot rather than serializable. Two concurrent updates to the same row abort the
second with a retryable `ConflictError`. anatid does not retry on your behalf, because whether the
write should be re-derived from a fresh read depends on what you were trying to do.

Tenant isolation is file-per-tenant. DuckDB offers no row-level or schema-level access control, so
a `tenant_id` column scopes queries while the real boundary is one file per tenant through
`DatabasePool`, enforced by the filesystem. Raw SQL through `db.connection` sees every tenant in
the file, and the docstrings say so.

DuckDB has no `AS OF SYSTEM TIME` clause. `as_of()` generates a `WHERE` clause over `valid_from`,
`valid_to`, `tx_from` and `tx_to`. It reaches back exactly as far as the rows still present, so a
hard purge disappears from every historical view as well.

The Compressed Sparse Row (CSR) graph structure still has sharp edges, though fewer than in 0.1. A
generation numbers its own vertices, so dense entity identifiers are no longer required of you. A
generation is built in full rather than updated in place, so a large journal eventually costs more
than the expansion saves, measured at 1.50 ms against 0.88 ms of pure SQL at roughly 550 journal
rows. The ratio trigger in `MaintenancePolicy` exists to prevent that. The in-memory structure is
not evicted by DuckDB's object cache, so memory grows with the number of resident generations. The
C++ extension remains optional; without it the merge runs in SQL and returns the same rows.

This is v0.3. The API may still move, so pin the version.

## Documentation

| document | what it covers |
|---|---|
| [`docs/server.md`](docs/server.md) | the optional server profile: why it exists, what it costs, the security model, and how to operate it |
| [`docs/architecture.md`](docs/architecture.md) | storage layout, the visibility predicate, the derived-index framework, graph paths, the isolation contract, the temporal model, the recall pipeline |
| [`docs/design/derived-index-framework.md`](docs/design/derived-index-framework.md) | the design the accelerators are built to, and what shipped against what was deferred |
| [`docs/benchmarks.md`](docs/benchmarks.md) | Phase 0 method, every result, and what the benchmark does not tell you |
| [`docs/roadmap.md`](docs/roadmap.md) | what comes next, and what is deliberately out of scope |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | how to build it, what we care about in a change, third-party notices |
| `spike/` | the Phase 0 evidence, kept read-only |

## License

MIT. Copyright (c) 2026 anatid contributors. Code adapted from DuckDB (MIT), or from Kuzu and
LadybugDB (MIT, Copyright 2022-2025 Kùzu Inc.), carries its original notice alongside ours. See
[`CONTRIBUTING.md`](CONTRIBUTING.md#third-party-notices).
