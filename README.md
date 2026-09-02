# anatid

anatid is an embedded graph memory for AI agents, built on DuckDB and MIT licensed. The database is
a single file with no server or daemon to run. `pip install anatid` gives an agent a memory that
stores entities and the edges between them, records both what was true and what the agent believed
at any past instant, and answers a query three ways at once (vector similarity, BM25 text, and
graph traversal, fused into one ranked list). Writes can be routed through the OpenAI Agents SDK's
human-in-the-loop approval flow, so an agent proposes a change to its memory and a person decides
whether it lands.

anatid exists because [Kuzu was archived on 2025-10-10](https://github.com/kuzudb/kuzu). Graphiti
deprecated its Kuzu driver, Mem0 removed open-source graph memory in v2.0.0, and Cognee is
migrating away.

2-hop recall at 1,000,000 memories has a p50 of 2.88 ms on DuckDB against 7.35 ms on a tuned
LadybugDB (the maintained MIT fork of Kuzu), a factor of 2.5, and the two engines return identical
result id-lists. That measurement is why anatid is built on DuckDB. The numbers, the method, and
the caveats are in [`docs/benchmarks.md`](docs/benchmarks.md).

## Install

```bash
pip install anatid                    # just duckdb
pip install "anatid[agents]"          # + the OpenAI Agents SDK integration
pip install "anatid[mcp]"             # + the MCP server
```

Released on PyPI as [`anatid` 0.1.0](https://pypi.org/project/anatid/0.1.0/). To track `main` instead:

```bash
pip install "git+https://github.com/thedatasense/anatid"
```

anatid runs on Python 3.10 through 3.13 and has one required dependency, `duckdb>=1.5`. CI runs the
test suite on Linux and macOS across all four Python versions, including both integration suites;
that is why `[dev]` installs `openai-agents` and `mcp`. The tests that load the 100k-row spike
dataset and the ones that need the compiled C++ extension skip in CI, because neither is in the
repository; they are run locally before a release. Windows should work, since DuckDB supports it,
but is not tested.

## Quickstart

```python
from anatid import Anatid, utcnow

vec = [0.0] * 63 + [1.0]                                        # your embedding model's output

with Anatid.open("agent.anatid", tenant=1, embedding_dim=64) as db:
    db.relate("Ada", "Kestrel", rel_kind="leads")               # an entity -> entity edge
    m = db.remember("Ada prefers dark roast coffee",            # a fact, filed under 2 entities
                    entities=["Ada", "coffee"], kind="preference",
                    embedding=vec, writer="agent-1",
                    episode="Standup 2026-03-01: Ada takes it dark roast.")  # raw evidence first
    t0 = utcnow()
    db.rebuild_fts_index()                                      # BM25 is not incremental: you say when

    hits = db.recall("coffee", embedding=vec, seed_entity="Ada", k=3)
    print(hits[0].content, hits[0].sources, "| bm25_stale:", hits.bm25_stale)

    new = db.supersede(m.memory_id, "Ada switched to decaf")    # closes the old row, keeps it
    print("old still current?", db.get(m.memory_id).is_current) # False -- history is intact
    print("at t0:", [x.content for x in db.as_of(t0).recall_2hop("Ada")])
    print("evidence:", db.provenance(new.memory_id).source_text)
```

```
Ada prefers dark roast coffee ('vector', 'text', 'graph') | bm25_stale: False
old still current? False
at t0: ['Ada prefers dark roast coffee']
evidence: Standup 2026-03-01: Ada takes it dark roast.
```

That is the output of running the block above. A longer, commented version covering `recall_2hop`,
`forget(hard=True)` and `stats()` is in [`examples/quickstart.py`](examples/quickstart.py). It needs
no API key and finishes in under a second.

```
$ python examples/quickstart.py
anatid schema v2 on duckdb 1.5.5, tenant 1, expand path: sql
before rebuild: bm25 stale=True, rows waiting=3

recall(query + embedding + seed): arms=('vector', 'text', 'graph') stale=False
  [1] 0.0487 'Ada prefers dark roast coffee' via vector+text+graph about=['Ada', 'coffee']
  [2] 0.0325 'The ingest service is maintained by Bo' via vector+graph about=['ingest service', 'Bo']
  [3] 0.0320 'Ada leads Project Kestrel' via vector+graph about=['Ada', 'Kestrel']

recall_2hop('Ada'):
  'The ingest service is maintained by Bo'
  'Ada leads Project Kestrel'
  'Ada prefers dark roast coffee'

supersede: old is_current=False valid_to=2026-03-31 09:00:00 -> new 'Ada switched to decaf'

as_of(day 1)  : ['The ingest service is maintained by Bo', 'Ada leads Project Kestrel', 'Ada prefers dark roast coffee']
current       : ['Ada switched to decaf', 'The ingest service is maintained by Bo', 'Ada leads Project Kestrel']

provenance(depth=1, writers=['agent-2', 'agent-1']):
  current  'Ada switched to decaf' (by agent-2)
  closed   'Ada prefers dark roast coffee' (by agent-1)
  source: 'Standup 2026-03-01: Ada is leading Project Kestrel; she take'...

forget(hard=True): rows_removed=5 about_edges=2 supersedes_edges=1 audit_rows_deleted=1

stats: memories=3 current=2 entities=5 about=6 relates=2
```

`Bo` never appears in the query and has no edge to `Ada`. The graph arm reached that memory in two
hops, `Ada → Kestrel → ingest service`.

## The verbs

| verb | what it does |
|---|---|
| `remember(content, entities=[...])` | write a fact and the ABOUT edges that make it reachable |
| `recall(query, embedding=, seed_entity=)` | hybrid retrieval: cosine + BM25 + 2-hop graph, fused with RRF |
| `recall_2hop(seed)` / `context(entity)` | pure graph recall; `context` defaults to 0 hops |
| `supersede(old_id, content)` | replace a belief, keeping the old one closed and linked |
| `reinforce(id)` / `prune(...)` | strengthen what gets used, drop what does not |
| `forget(id, hard=False)` | stop believing (audit trail kept) or erase completely |
| `as_of(t)` | every read, as the database saw the world at `t` |
| `provenance(id)` | the supersession chain, the raw episodes, and every writer involved |
| `relate(a, b)` / `upsert_entity` / `episode` | the graph and evidence primitives underneath |

Each write verb is exactly one DuckDB transaction. Reads (`recall`, `recall_2hop`, `context`,
`get`, `provenance`, `stats`) run their statements outside an explicit transaction, so a concurrent
commit can land between a recall's arms and its hydration step; wrap the call in `db.transaction()`
if you need one snapshot. `prune` is a query plus one transaction per memory it forgets, so a
failure part-way leaves the earlier deletions committed, and taking its `dry_run` list first is the
way to see what it will touch. Write verbs take `now=` and the temporal read verbs take `as_of=`,
which keeps tests deterministic. There are function forms too (`from anatid.verbs import
remember`), and `db.connection` hands you the raw DuckDB cursor whenever you want to write SQL. The
memory is ordinary tables, joinable against your Parquet and CSV in place.

## Why DuckDB, with numbers

Phase 0 was a benchmark run before any of the library was written: 1,000,000 memories, 2.3M edges,
10 tenants, four engines, the same operations with identical semantics, checked against a
pure-Python oracle.

2-hop recall is the query shape agent memory hits hardest. Over 1,000 queries on a single thread:

| engine | p50 | p95 | load | on disk | concurrent reads |
|---|---:|---:|---:|---:|---:|
| DuckDB + C++ CSR extension | 2.04 ms | 3.07 ms | 4.8 s | 481 MiB | 825 R1/s |
| DuckDB, plain SQL | 2.88 ms | 3.50 ms | 4.6 s | 434 MiB | 583 R1/s |
| LadybugDB 0.20.2 (tuned) | 7.35 ms | 28.73 ms | 16.2 s | 1,158 MiB | 147 R1/s |

- The kill criterion was "abandon DuckDB if it is more than 5x slower". It came in at 0.39x (SQL)
  and 0.28x (extension). At p95 it is 0.12x and 0.11x.
- All three engines returned identical result id-lists on 1,000 oracle-checked queries and on 200
  post-write verify queries.
- LadybugDB's number is the fastest of six Cypher formulations across two thread settings. The
  naive formulation was 16x slower than the tuned one; reporting it would have flattered DuckDB.
- Where DuckDB loses: hybrid recall is 22% slower (16.4 ms against 20.0 ms p50; no engine in the
  run had an ANN index, so this is a scan-speed comparison), and concurrent readers cost DuckDB
  writers real throughput (397 W1/s with writers alone, 152-189 W1/s with 2 readers added).
  LadybugDB with `enable_multi_writes=True` commits more writes per second than DuckDB does.

Full tables covering every phase, p50/p95/p99, the mixed workload, concurrency, correctness, and
nine limitations of the benchmark itself are in [`docs/benchmarks.md`](docs/benchmarks.md). The raw
JSON with per-operation latency arrays is in `spike/results/`.

## OpenAI Agents SDK integration

The [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) already has everything
needed for human-in-the-loop: `needs_approval=True` on a `function_tool`,
`RunResult.interruptions`, a serializable `RunState`, `state.approve()` / `state.reject()`. It also
has a `Session` protocol for conversation history, with SQLite, SQLAlchemy and Redis backends.

What it does not have is a DuckDB session, graph memory, or approval-gated memory writes. As far as
we can establish, no open-source project combines all four of the Agents SDK, DuckDB, a graph
store, and human approval on memory writes. anatid supplies the missing three and rebuilds none of
the SDK's machinery:

```python
from agents import Agent, Runner
from anatid import Anatid
from anatid.integrations.openai_agents import AnatidSession, create_memory_tools

db = Anatid.open("agent.anatid", tenant=1)
session = AnatidSession("conv-1", db)                # conversation history, same file as the graph
tools = create_memory_tools(db, session=session)     # 3 read tools + 3 write tools

agent = Agent(name="assistant", tools=tools)
result = await Runner.run(agent, "Ada switched to decaf, remember that", session=session)

while result.interruptions:                          # writes stop here; reads never do
    state = result.to_state()
    for item in result.interruptions:
        print(item.tool_name, item.raw_item.arguments)   # "anatid_remember" {"content": ...}
        state.approve(item)                              # or state.reject(item)
    result = await Runner.run(agent, state, session=session)
```

- Writes are gated and reads are not. `anatid_remember`, `anatid_supersede` and `anatid_forget`
  carry `needs_approval`; `anatid_recall`, `anatid_context` and `anatid_provenance` do not. Nothing
  touches the database until someone approves.
- The approval policy is a callable. `approve_low_risk()` auto-approves small, ordinary writes and
  still stops for hard deletes. Unless you opt out explicitly, `anatid_forget(hard=True)` requires
  approval regardless, because a hard forget removes the row, its edges, its embedding and its
  provenance.
- Approval can happen later and elsewhere. `RunStateStore(db)` parks the SDK's serialized
  `RunState` in the same anatid file, so an interrupted run can be reviewed and resumed minutes or
  days later by another process, as a review queue rather than a blocking prompt.
- History and knowledge are joinable, because `AnatidSession` writes turns into a table in the same
  DuckDB file as the memory graph. `await session.entities_mentioned()` is one SQL join against
  `entities` rather than two round-trips to two different stores, and `memories_written_here()`
  reports what this conversation committed to memory.

## MCP server

```bash
pip install "anatid[mcp]"
anatid-mcp --db memory.anatid        # stdio; point Claude Desktop, Claude Code or Cursor at it
```

This exposes the memory verbs over the Model Context Protocol, so any MCP client gets persistent,
bitemporal, graph-shaped memory: `remember`, `relate`, `supersede`, `reinforce`, `forget`, `prune`
and `rebuild_fts_index` on the write side, `recall`, `context`, `get`, `provenance` and `stats` on
the read side. (Those are the MCP tool names; the `anatid_`-prefixed names belong to the Agents SDK
integration above.) `--read-only` registers the read tools only.

There is also one deliberate escape hatch: a `sql` tool for the questions the verbs do not answer
("how many memories per kind?", "show me the audit trail"). It is read-only, and enforced in three
layers by DuckDB rather than by a regex over the query text: DuckDB's own statement classifier
(only SELECT/EXPLAIN, and every statement in the text must pass), a scan of DuckDB's parse tree for
file-reading functions and for base-table names that are not plain identifiers (DuckDB's
replacement scan makes `SELECT * FROM '/etc/passwd.csv'` an ordinary SELECT), and execution inside
`BEGIN TRANSACTION READ ONLY` on a private cursor that is always rolled back. DuckDB will not give
a second read-only connection to a file the process already holds, so the read-only transaction is
the mechanism. `PRAGMA create_fts_index(...)`, which expands into DDL at bind time, is rejected on
what it really is. Turn the tool off with `--no-sql-tool`.

`from anatid.integrations.mcp import build_server` if you want to embed the server in your own
process.

## Limitations

Every item here is measured or documented in the source. Behavior that contradicts the docs and is
not listed below is a bug; please report it.

- There is no ANN index. The vector arm is a brute-force `array_cosine_similarity` scan, because
  DuckDB ships no ANN index. The cost is linear in one tenant's row count: measured at 64 dims on
  the spike hardware, 2.0 ms p50 with 10k memories in the tenant, 8.6 ms at 100k (an independent
  run of the same measurement got 11.4 ms) and 23.3 ms at 1M. `BRUTE_FORCE_CEILING = 100_000` is
  documented and not enforced. At that ceiling a recall already costs roughly 9-11 ms, and past it
  this is the wrong tool. An owned ANN index is the headline item of v1.0.
- The full-text index is not incremental. DuckDB's `fts` index does not see rows inserted after it
  was built. The API reports it in three places: `rebuild_fts_index()` is explicit, `fts_status()`
  reports how stale the index is, and every `recall()` result carries `.bm25_stale` and
  `.pending_fts_rows` (with `on_stale_fts="error"` if you would rather raise). The staleness window
  is the interval between rebuilds.
- One writing process per file. That is DuckDB's model, and the engine enforces it: a second
  read-write process cannot even open the file (`IO Error: Could not set lock on file ...:
  Conflicting lock is held`). Many threads inside that one process write concurrently and appends
  never conflict (0 errors in a 30 s, 6-thread benchmark with no retry logic), but anatid provides
  nothing for multi-process writes.
- Isolation is snapshot, not serializable. Two concurrent updates to the same row abort the second
  with a retryable `ConflictError`. anatid does not retry, because whether the write should be
  re-derived from a fresh read depends on the caller.
- Tenant isolation is file-per-tenant. DuckDB has no row-level or schema-level access control. A
  `tenant_id` column scopes queries; the real boundary is one file per tenant via `DatabasePool`,
  enforced by the filesystem. Raw SQL through `db.connection` sees every tenant in the file, and
  the docstrings say so.
- DuckDB has no `AS OF SYSTEM TIME`. `as_of()` is a `WHERE` clause over
  `valid_from`/`valid_to`/`tx_from`/`tx_to` that anatid generates. It reaches back exactly as far as
  the rows still in the table, so a hard purge is gone from every as-of view too.
- The CSR extension has sharp edges. It needs dense per-tenant entity ids (anatid's default 63-bit
  time-ordered ids are not dense), it is rebuilt in full rather than incrementally, and any
  `relate()` marks it stale, at which point recall silently falls back to the SQL path and returns
  identical rows. It is an accelerator, off by default.
- No Cypher yet. That is v0.2. Today the API is the verbs above plus SQL.
- This is v0.1. The API may still move, so pin the version.

## Documentation

- [`docs/architecture.md`](docs/architecture.md): storage layout, the derived CSR and how it stays
  MVCC-correct, the isolation contract, the temporal model, the recall pipeline.
- [`docs/benchmarks.md`](docs/benchmarks.md): Phase 0 method, every result, and what the benchmark
  does not tell you.
- [`docs/roadmap.md`](docs/roadmap.md): v0.2 (Cypher subset, Graphiti/Cognee drivers, Node
  bindings), v0.5 (production operation), v1.0 (ANN index, persistent CSR, duckdb-wasm).
- [`CONTRIBUTING.md`](CONTRIBUTING.md): how to build it, what we care about in a change, and the
  third-party notices.
- `spike/`: the Phase 0 evidence, kept read-only.

## License

MIT. Copyright (c) 2026 anatid contributors. Code adapted from DuckDB (MIT) or from Kuzu /
LadybugDB (MIT, Copyright 2022-2025 Kùzu Inc.) carries its original notice alongside ours; see
[`CONTRIBUTING.md`](CONTRIBUTING.md#third-party-notices).
