<h1><img src="https://raw.githubusercontent.com/thedatasense/anatid/main/assets/brand/anatid-logo.png" alt="anatid" width="280" height="96"></h1>

Local memory for AI agents, with evidence, corrections and history. One DuckDB file.

anatid is for developers building assistants that have to remember decisions, ownership and
changing constraints: who owns a service, what was decided and when, which rule still holds. Every
fact is filed under the entities it names, linked to the raw text it came from, and kept when it
is corrected rather than overwritten. Retrieval runs three ways at once, through text scoring,
graph traversal and, with an embedder configured, vector similarity. Nothing runs but your
process, and the memory is a single file you can copy, back up and query with SQL. MIT licensed.

[Documentation](https://github.com/thedatasense/anatid/blob/main/docs/README.md) ·
[Examples](https://github.com/thedatasense/anatid/blob/main/examples/README.md) ·
[PyPI](https://pypi.org/project/anatid/) ·
[Releases](https://github.com/thedatasense/anatid/releases)

## Watch a procedure improve

A passing test says a fictional infusion-pump lot is ready for review. A linked correction
withdraws that test. The visual **Procedural Studio** shows how adding a reconciliation step
changes the outcome, how validation rejects a harmful shortcut, and how anatid preserves the
original procedure and the evidence behind each revision.

<img src="https://raw.githubusercontent.com/thedatasense/anatid/main/assets/procedural-studio.png" alt="Anatid Procedural Studio: a repaired graph reconciles a passing test with its withdrawal, then holds the packet for human review." width="960">

From a source checkout:

```bash
git clone https://github.com/thedatasense/anatid.git
cd anatid
python -m pip install -e .
python -m examples.procedural_studio
```

Open <http://127.0.0.1:8766>. Walk through **The shortcut → The repair → The guardrail**.
Add `--live` with `OPENROUTER_API_KEY` configured to ask a model for next-step guidance.
The graph, evidence, and history are read from a real in-memory anatid database. The five
synthetic test cases improve from **1/5 to 5/5** under a scripted evaluator; live model responses
are separate from those scores. These are fictional records and review rules, with no actual
device release decisions.

[Run the visual demo](https://github.com/thedatasense/anatid/blob/main/examples/procedural_studio/README.md) ·
[How procedural graphs map to anatid](https://github.com/thedatasense/anatid/blob/main/docs/procedural-graphs.md)

## The demo

Three project notes, six months apart, go in as prose. What follows is the real output of
[`examples/ingest_notes.py`](https://github.com/thedatasense/anatid/blob/main/examples/ingest_notes.py), trimmed only for width. It runs offline,
without a key, in about a second.

The first note says who owns what. A model reads it and proposes a patch; the pipeline shows the
patch as a diff before anything is written, then applies it in one transaction with the note
stored as evidence.

```
2026-03-02  notes/2026-03-02.md
  > Ada leads the Kestrel team. Kestrel owns the ingest service, and Bo maintains it day to day.

  memory patch: 3 facts, 3 relation(s) added
    + fact        "Ada leads Kestrel"  about: Ada, Kestrel
    + fact        "Kestrel owns the ingest service"  about: Kestrel, ingest service
    + fact        "Bo maintains the ingest service"  about: Bo, ingest service
    + relation    Ada -leads-> Kestrel
    + relation    Kestrel -owns-> ingest service
    + relation    Bo -maintains-> ingest service
  applied: episode 883936403279032320 stored; 3 memories created; 3 relations opened.
```

Then the owner changes. The second note does not repeat the old fact; the patch corrects it, and
the edge from Bo to the service closes in the same transaction that opens the edge from Cy.

```
2026-06-15  notes/2026-06-15.md
  > Bo moved to the platform group. Cy took over the ingest service from Bo this week.

  memory patch: 1 fact, 1 correction, 2 relation(s) added, 1 relation(s) removed
    + fact        "Bo works in the platform group"  about: Bo, platform group
    ~ correction  memory 883936403329363968 "Bo maintains the ingest service"
                  -> "Cy maintains the ingest service"  about: Cy, ingest service
    - relation    Bo -maintains-> ingest service
    + relation    Cy -maintains-> ingest service
    + relation    Bo -member_of-> platform group
  applied: episode 883936403509719040 stored; 2 memories created; 1 superseded; 2 relations opened; 1 relations closed.
```

Ask who maintains the ingest service now, and the assistant answers with the new owner. Ask why,
and the chain of evidence runs back through both notes.

```
3. Who maintains the ingest service now, and why
  now:     Cy maintains the ingest service
  before:  Bo maintains the ingest service  (valid until 2026-06-15)
  chain of evidence, newest first:
    2026-06-15  notes/2026-06-15.md
      > Bo moved to the platform group. Cy took over the ingest service from Bo this week.
    2026-03-02  notes/2026-03-02.md
      > Ada leads the Kestrel team. Kestrel owns the ingest service, and Bo maintains it day to day.
  writers: notes-bot
```

Ask what was true in April, and the previous owner comes back, because the correction closed the
old fact instead of deleting it.

```
4. as_of(2026-04-01): what the database believed about the ingest service in April
    2026-03-02  Bo maintains the ingest service
    2026-03-02  Kestrel owns the ingest service

5. The same read today
    2026-08-20  The ingest service must stay on Python 3.10 until the Kestrel migration finishes [constraint]
    2026-06-15  Cy maintains the ingest service
    2026-03-02  Kestrel owns the ingest service
```

A question about Ada reaches the ingest service too, though no fact about the service mentions
her: Ada leads Kestrel, Kestrel owns the service, and the graph walk covers the two hops. Run the
script with `--live` and GLM 5.3 Flash through OpenRouter proposes the patches instead of the
scripted extractor; the output above is the offline run.

## Five minutes to a working memory

Install it.

```bash
pip install anatid                    # the library and the anatid-server command
pip install "anatid[mcp]"             # + the Model Context Protocol server
pip install "anatid[agents]"          # + the OpenAI Agents Software Development Kit (SDK) tools
```

Give it to an assistant over the Model Context Protocol (MCP). Claude Desktop, Claude Code and
Cursor all take this block, with the path `which anatid-mcp` prints; [`docs/mcp.md`](https://github.com/thedatasense/anatid/blob/main/docs/mcp.md)
says where each client keeps it.

```json
{
  "mcpServers": {
    "anatid": {
      "command": "/ABSOLUTE/PATH/TO/anatid-mcp",
      "env": { "ANATID_DB": "/Users/you/.anatid/memory.anatid" }
    }
  }
}
```

Or use it from Python. Three statements: open a file, remember a fact with its evidence, ask.

```python
from anatid import Anatid
db = Anatid.open("team.anatid", tenant=1)
db.remember("Cy maintains the ingest service", entities=["Cy", "ingest service"],
            episode="Handover note, 2026-06-15: Cy took the ingest service over from Bo.")
print(db.recall("who maintains the ingest service")[0].content)
```

```
Cy maintains the ingest service
```

That `recall` ran the text arm and the graph arm: the query names the ingest service, so the walk
started there without anyone naming a seed. `db.provenance(memory_id).source_text` returns the
handover note. To have text go in as in the demo rather than one fact at a time, see
[`docs/ingest.md`](https://github.com/thedatasense/anatid/blob/main/docs/ingest.md).

## What anatid is

anatid is an embedded graph memory for AI agents, built on DuckDB and released under the MIT
license. The database is a single file, and the default way to use it has no server to run and no
daemon to supervise. There is an optional server profile for the one case that needs it, described
below.

Install it and an agent gains a memory that stores entities, the edges between them, and facts
attached to both. That memory records two kinds of time: what was true, and what the agent believed
at any past instant. Queries run three ways at once, through vector similarity, Best Match 25
(BM25) text scoring, and graph traversal, fused into a single ranked list. Writes can be routed
through the human-in-the-loop approval flow in the OpenAI Agents SDK, so an agent proposes a
change to its own memory and a person decides whether it lands.

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
anatid sits on DuckDB. Method and caveats are in [`docs/benchmarks.md`](https://github.com/thedatasense/anatid/blob/main/docs/benchmarks.md).

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
[`examples/quickstart.py`](https://github.com/thedatasense/anatid/blob/main/examples/quickstart.py). It needs no API key and finishes in under a
second. [`examples/README.md`](https://github.com/thedatasense/anatid/blob/main/examples/README.md) lists every example and which ones need a key,
and [`examples/notebooks/`](https://github.com/thedatasense/anatid/blob/main/examples/notebooks/) walks through the same material as four executed
Jupyter notebooks.

For something closer to how memory tends to fail in practice, run
[`examples/dinner_party.py`](https://github.com/thedatasense/anatid/blob/main/examples/dinner_party.py). Six months of ordinary household facts, a
cook who asks whether Friday's menu is safe, and an allergy that neither the question nor any
single stored sentence mentions. The graph walks from the dinner to a guest to an ingredient to the
dish. Word search alone returns the recipe cards and stops.

## The verbs

| verb | what it does |
|---|---|
| `remember(content, entities=[...])` | write a fact and the ABOUT edges that make it reachable |
| `recall(query, embedding=, seed_entity="auto", arm_weights=)` | hybrid retrieval: cosine, BM25 and 2-hop graph, fused with weighted reciprocal rank fusion; the graph arm seeds itself from the entity names in the query and ranks its candidates by the query's own signal |
| `recall_2hop(seed)` / `context(entity)` | pure graph recall; `context` defaults to 0 hops |
| `supersede(old_id, content)` | replace a belief, keeping the old one closed and linked |
| `correct(old_id, content, add_relations=, remove_relations=)` | supersede a belief and move the edges that change with it, in one transaction |
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

`recall(query)` runs the text arm and the graph arm by default. The graph arm's seeds are the
entity names that occur in the query, matched case-insensitively, longest name first, at most
three; `hits.seeds` lists them and `hits.arms` says which arms ran. Pass `seed_entity="Ada"` to
expand from exactly that entity, or `seed_entity=None` to run without the graph arm. The vector arm
runs when you pass an `embedding`, or when the handle was opened with an embedder:
`Anatid.open(path, embedder=OpenAICompatibleEmbedder(base_url, api_key, model, dim))` embeds
every `remember` and every query it is not given a vector for, so all three arms run with no
application code. `HashEmbedder(dim)` is an offline stand-in for demos and tests. The arms do not
vote equally: with a vector arm it leads (vector 1.0, graph 0.5, text 0.25), without one the text
arm does (text 1.0, graph 0.5), and the graph arm ranks its neighbourhood by cosine or BM25 to the
query before it votes, not newest first. `arm_weights={"text": 0}` overrides a weight by name and
`hits.weights` reports what was used. The weights come from the answer-quality benchmark in
[`docs/quality.md`](https://github.com/thedatasense/anatid/blob/main/docs/quality.md).

Write verbs accept `now=` and the temporal read verbs accept `as_of=`, which keeps tests
deterministic. Function forms exist as well, through `from anatid.verbs import remember`. And
`db.connection` hands you the raw DuckDB cursor whenever you want SQL. The memory is ordinary
tables, joinable against your Parquet and CSV files in place.

## Text in, a reviewed patch out

The verbs take facts one at a time. `anatid.ingest` takes text. An extractor, any object with
`extract(text, existing) -> MemoryPatch`, proposes what a note changes: facts to add, facts to
correct by id, edges to open and close, and names that mean an existing entity. The pipeline
resolves the proposal against what the graph already holds and writes a note for everything it
changed: a duplicate dropped, an alias rewritten, a correction whose target is gone downgraded to
a fact. A review hook may edit or decline the patch. `MemoryPatch.apply` then commits the whole
patch in one transaction, with the raw text stored first as the episode every new row cites.

```python
from anatid.ingest import OpenAICompatibleExtractor, ingest

extractor = OpenAICompatibleExtractor(model="z-ai/glm-5.3-flash",
                                      base_url="https://openrouter.ai/api/v1", api_key=key)
receipt = ingest(db, note, extractor=extractor, writer="notes-bot", source="notes/2026-06-15.md",
                 review=lambda patch: patch if input(patch.describe() + "\napply? ") == "y" else None)
```

`OpenAICompatibleExtractor` talks to any OpenAI-compatible chat endpoint; `ScriptedExtractor`
returns prepared patches for tests and the offline example. The same pipeline is behind the
`anatid_ingest` tool in the Agents SDK integration and the `ingest` and `apply_patch` tools in the
MCP server, described below. [`docs/ingest.md`](https://github.com/thedatasense/anatid/blob/main/docs/ingest.md) has the patch schema and the apply
order.

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

[`docs/server.md`](https://github.com/thedatasense/anatid/blob/main/docs/server.md) covers the security model, the operator surface, systemd and
launchd units, health and readiness, backup and restore, and what the shutdown guarantees.
[`examples/server_demo.py`](https://github.com/thedatasense/anatid/blob/main/examples/server_demo.py) demonstrates the lock, the server, and several
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
limitations of the benchmark itself are in [`docs/benchmarks.md`](https://github.com/thedatasense/anatid/blob/main/docs/benchmarks.md). Raw JSON
with per-operation latency arrays sits in `spike/results/`.

## Does an agent answer better?

Phase 0 measures storage. It says nothing about whether an agent answers better with anatid in
front of it than with the obvious alternatives, so a second benchmark asks that. Eleven memory
systems answer the same 150 questions about a synthetic engineering organisation (about 178 notes
over eighteen months: handovers, on-call rotations, incidents, decisions, and three wrong records
corrected weeks later) with the same model, `z-ai/glm-5.3-flash` at temperature 0, the same prompt
and the same 1,200-token memory budget. A judge that never learns which system answered grades each
answer against an exact gold, a lexical scorer is reported next to it, and every model call is
cached so one command reproduces every number. Two seeds of the generator give the two worlds the
retrieval settings were chosen on; a third, held out until then, is reported below the table.
Accuracy under the judge, then multi-hop accuracy, false refusals (`I don't know` on a question the
notes do answer) and mean context size, each as seed 20260905 / seed 7:

| memory system | accuracy, seed 20260905 | accuracy, seed 7 | multi-hop | false refusals | context tokens |
|---|---:|---:|---:|---:|---:|
| Markdown file, most recent notes that fit | 39% | 41% | 36% / 28% | 68% / 61% | 1200 / 1187 |
| Markdown file, whole, no budget | 92% | 99% | 88% / 96% | 2% / 0% | 6338 / 6310 |
| BM25 over the notes | 89% | 88% | 40% / 32% | 2% / 3% | 1150 / 1145 |
| vectors over the notes | 92% | 93% | 52% / 56% | 4% / 3% | 1178 / 1179 |
| BM25 and vectors fused | 91% | 91% | 48% / 48% | 3% / 2% | 1179 / 1179 |
| vectors with one feedback round | 93% | 92% | 60% / 56% | 3% / 2% | 1178 / 1179 |
| anatid, every arm | 89% | 89% | 52% / 56% | 9% / 8% | 1098 / 1078 |
| anatid, text arm only | 84% | 77% | 52% / 12% | 15% / 20% | 1122 / 1041 |
| anatid, vector arm only | 91% | 87% | 64% / 60% | 8% / 8% | 1090 / 1079 |
| anatid, graph arm only | 30% | 22% | 12% / 8% | 82% / 92% | 568 / 625 |
| anatid built from gold patches (oracle) | 96% | 96% | 84% / 84% | 3% / 4% | 1004 / 1018 |

Where anatid wins. It answers as many of the 25 multi-hop questions as vectors over the raw notes
in both worlds (13 and 14 against 13 and 14), its memory block is smaller than any raw-note
system's, at about 1,080 to 1,100 tokens against 1,180, for answers that cost the same two to three
cents per 150 questions, and it refused every unanswerable question in both worlds, as did nearly
every other system. The same store built from gold patches instead of the model's, an oracle for
extraction rather than a product, is the best budgeted system in both worlds at 96%, so the
retrieval is not the limit. Its fused recall now stands within two points of its own vector arm on
the committed corpus and two points above it on seed 7; in 0.4.1 the vector arm alone beat the
fusion by seven and two, which is what changing the fusion weights and the graph arm's ordering
bought. On the held-out world (seed 11), built and answered once after that choice, anatid answers
88% with every arm and 87% with its vector arm alone, vectors over the raw notes 96%, and the gold
store 99%.

Where anatid loses. It trails vectors over the raw notes by three points on the committed corpus
and four on seed 7, wins no question outright in either world, and loses 11 and nine: six and five
of them refusals on facts the extractor never wrote down in a findable form, the rest stale
values, team-service lists with one entry wrong, and provenance answered with a later note. In
both worlds it refuses answerable questions two to four times as often as the raw-note systems.
Building the store costs a model pass over every note, about $0.10 and 40 to 70 minutes of model
time for 178 notes, where the vector index costs a cent, and a rebuild re-rolls the extraction by a
few points either way. The whole file in the prompt beats everything at this corpus size, which is
the honest answer at 6,300 tokens of notes and says nothing about 60,000.

The corpus is ours, the judge is the answering model, and 25 questions per category means one
question is four points, so the differences among the raw-note systems are noise and anatid's
three-to-four-point gap is at the edge of it; 0.4.1's ten-point gap on the committed corpus, part
of it a note lost to a cached provider error, was not. The method, the per-category tables, the arm
ablations, every loss question by question, the offline coverage proxy the fusion was chosen with,
and the one command that reproduces it all are in [`docs/quality.md`](https://github.com/thedatasense/anatid/blob/main/docs/quality.md).

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
tools = create_memory_tools(db, session=session)     # 3 read tools, 6 write tools

agent = Agent(name="assistant", tools=tools)
result = await Runner.run(agent, "Ada switched to decaf, remember that", session=session)

while result.interruptions:                          # writes stop here; reads never do
    state = result.to_state()
    for item in result.interruptions:
        print(item.tool_name, item.raw_item.arguments)   # "anatid_remember" {"content": ...}
        state.approve(item)                              # or state.reject(item)
    result = await Runner.run(agent, state, session=session)
```

Writes are gated and reads run straight through. Six tools carry `needs_approval`:
`anatid_remember`, `anatid_relate`, `anatid_supersede`, `anatid_correct`, `anatid_unrelate` and
`anatid_forget`. Three do not: `anatid_recall`, `anatid_context` and `anatid_provenance`. Nothing
reaches the database until somebody approves. The three graph tools exist because memories are
filed under the entities they name and nothing links those entities until an edge does. Three facts
stored as "Ada leads Kestrel", "Kestrel owns the ingest service" and "Bo maintains the ingest
service" are three islands until `anatid_relate` connects them, and when the maintainer changes,
`anatid_correct` supersedes the fact and moves the edge in one transaction, so the graph never says
two things at once.

A tenth tool, `anatid_ingest`, appears when `create_memory_tools` is given an
`extractor`. It takes a note, proposes a patch of facts and edges through the ingestion pipeline,
and applies it as one gated write. `dry_run=True` returns the diff without writing and never waits
for approval, and the `patch_json` a dry run returns can be handed back, edited or not, to apply
exactly that patch.

The approval policy is a callable, so you can shape it. `approve_low_risk()` waves through small
ordinary writes and still stops for hard deletes, edge changes and ingestion. Unless you opt out
explicitly, `anatid_forget(hard=True)` always requires approval, since a hard forget removes the
row, its edges, its embedding and its provenance together.

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

That exposes the memory verbs over MCP, so any MCP client gains persistent, bitemporal,
graph-shaped memory. The write side offers `remember`, `relate`, `unrelate`, `supersede`,
`correct`, `reinforce`, `forget`, `prune` and `rebuild_fts_index`. The read side offers `recall`,
`context`, `get`, `provenance` and `stats`. With `ANATID_EXTRACT_BASE_URL` and
`ANATID_EXTRACT_MODEL` set, `ingest` proposes a patch from text and returns its diff with a
`patch_id`, and `apply_patch` commits it. With `ANATID_EMBED_BASE_URL` and `ANATID_EMBED_MODEL`
set, the server embeds every write and every query and `recall` runs the vector arm too. Those are
MCP tool names; the `anatid_`-prefixed names belong to the Agents SDK integration above. Passing
`--read-only` registers the read tools alone.

Two MCP clients cannot both open one file, because DuckDB gives one process exclusive use of it.
`anatid-mcp --socket /tmp/anatid/anatid.sock` talks to a running `anatid-server` instead, so
Claude Desktop and Claude Code share one memory with the same tools; a second `anatid-mcp --db` on a
held file exits with the two commands to run instead of a traceback.

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
[`docs/mcp.md`](https://github.com/thedatasense/anatid/blob/main/docs/mcp.md) has the config blocks, every environment variable, and the sharing
recipe.

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
| Ingestion | The extractor proposes and a person or a hook decides; the pipeline never decides what is true |

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
`BRUTE_FORCE_CEILING = 100_000` rows, unless you pass `allow_slow=True`; an embedding the handle's
own embedder produced skips the arm and says so in `hits.notes` instead.

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

Automatic seeding matches lowercased entity names against the words of the query, so a stored name
with irregular internal whitespace is matched only when the query repeats it, and a query that
names no entity runs the text arm alone as before. The match costs about 1.9 ms with 100,000
entities in a tenant on a laptop, because `entity_key` is a generated column DuckDB's index does not
serve; below 10,000 entities it is under a millisecond.

This is v0.4. The API may still move, so pin the version.

## Documentation

| document | what it covers |
|---|---|
| [`docs/mcp.md`](https://github.com/thedatasense/anatid/blob/main/docs/mcp.md) | the MCP server: config blocks for each client, every variable, sharing one memory between clients, embeddings, ingestion, the SQL escape hatch |
| [`docs/ingest.md`](https://github.com/thedatasense/anatid/blob/main/docs/ingest.md) | text in, a reviewed patch out: the pipeline, the patch schema, apply order, the review hook, the extractors |
| [`docs/server.md`](https://github.com/thedatasense/anatid/blob/main/docs/server.md) | the optional server profile: why it exists, what it costs, the security model, and how to operate it |
| [`docs/architecture.md`](https://github.com/thedatasense/anatid/blob/main/docs/architecture.md) | storage layout, the visibility predicate, the derived-index framework, graph paths, the isolation contract, the temporal model, the recall pipeline |
| [`docs/design/derived-index-framework.md`](https://github.com/thedatasense/anatid/blob/main/docs/design/derived-index-framework.md) | the design the accelerators are built to, and what shipped against what was deferred |
| [`docs/extension.md`](https://github.com/thedatasense/anatid/blob/main/docs/extension.md) | the optional C++ extension: what it accelerates and how to build it |
| [`docs/benchmarks.md`](https://github.com/thedatasense/anatid/blob/main/docs/benchmarks.md) | Phase 0 method, every result, and what the benchmark does not tell you |
| [`docs/quality.md`](https://github.com/thedatasense/anatid/blob/main/docs/quality.md) | the answer-quality benchmark: eleven memory systems, one model, one budget, the losses next to the wins, and the command that reproduces it |
| [`docs/roadmap.md`](https://github.com/thedatasense/anatid/blob/main/docs/roadmap.md) | what comes next, and what is deliberately out of scope |
| [`CONTRIBUTING.md`](https://github.com/thedatasense/anatid/blob/main/CONTRIBUTING.md) | how to build it, what we care about in a change, third-party notices |
| `spike/` | the Phase 0 evidence, kept read-only |

## License

MIT. Copyright (c) 2026 anatid contributors. Code adapted from DuckDB (MIT), or from Kuzu and
LadybugDB (MIT, Copyright 2022-2025 Kùzu Inc.), carries its original notice alongside ours. See
[`CONTRIBUTING.md`](https://github.com/thedatasense/anatid/blob/main/CONTRIBUTING.md#third-party-notices).

<!-- mcp-name: io.github.thedatasense/anatid -->
