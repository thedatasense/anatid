# anatid over MCP

`anatid-mcp` serves an anatid database over the [Model Context Protocol](https://modelcontextprotocol.io).
Claude Code, Claude Desktop, Cursor, and anything else that speaks MCP get a persistent,
bitemporal, graph-shaped memory backed by one embedded DuckDB file. By default there is no server
process to run, no container, and no API key: `anatid-mcp` opens the file itself. That default
serves one client at a time, because DuckDB gives one process exclusive use of a file. When two
clients need the same memory at once, `anatid-mcp --socket` talks to a running `anatid-server`
instead of opening the file; see [Sharing one memory between clients](#sharing-one-memory-between-clients).

Built against mcp 2.x (`mcp.server.mcpserver.MCPServer`, the class mcp 1.x called `FastMCP`).

## Install

```sh
pip install 'anatid[mcp]'          # or: pip install anatid 'mcp>=2.1'
```

That puts an `anatid-mcp` console script on your PATH. GUI apps do not inherit your shell's PATH,
so find its absolute path first; the config block below needs it.

```sh
which anatid-mcp
```

With [uv](https://docs.astral.sh/uv/) there is nothing to install or to locate: `uvx` fetches the
release, the `mcp` extra and the `anatid` executable, which is the same server under the package's
own name, and this is also the command a client that installs anatid from the MCP Registry runs
([`mcp-registry.md`](mcp-registry.md)).

```sh
uvx --with "anatid[mcp]" anatid --db ~/.anatid/memory.anatid
```

## The config block

All three clients use the same `mcpServers` shape. Paste this, replacing
`/ABSOLUTE/PATH/TO/anatid-mcp` with what `which anatid-mcp` printed.

```json
{
  "mcpServers": {
    "anatid": {
      "command": "/ABSOLUTE/PATH/TO/anatid-mcp",
      "args": [],
      "env": {
        "ANATID_DB": "/Users/you/.anatid/memory.anatid",
        "ANATID_TENANT": "0"
      }
    }
  }
}
```

Where it goes:

| Client | File |
| --- | --- |
| Claude Desktop (macOS) | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Claude Desktop (Windows) | `%APPDATA%\Claude\claude_desktop_config.json` |
| Claude Code (per project) | `.mcp.json` in the project root |
| Claude Code (per user) | `~/.claude.json`, under `"mcpServers"` |
| Cursor (per project) | `.cursor/mcp.json` |
| Cursor (per user) | `~/.cursor/mcp.json` |

Restart the client afterwards. Claude Desktop and Cursor only read the file at launch.

### Claude Code, without editing a file

```sh
claude mcp add anatid --env ANATID_DB=$HOME/.anatid/memory.anatid -- "$(which anatid-mcp)"
```

### If it is not on your PATH

Point at the interpreter instead. This form does not depend on PATH at all, and is the one to use
inside a virtualenv or a checkout.

```json
{
  "mcpServers": {
    "anatid": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "anatid.integrations.mcp.server"],
      "env": { "ANATID_DB": "/Users/you/.anatid/memory.anatid" }
    }
  }
}
```

## Configuration

Every setting reads an environment variable, which is all a client config block can set, and has a
matching command-line flag for running the server by hand.

| Env | Flag | Default | Meaning |
| --- | --- | --- | --- |
| `ANATID_DB` | `--db` | `~/.anatid/memory.anatid` | Database file, or `:memory:`. Parent directories are created. Opened in this process. |
| `ANATID_SOCKET` | `--socket` | unset | A running `anatid-server`'s Unix socket. The tools talk to that server instead of opening a file. Refused together with `--db` or `--enable-sql`. |
| `ANATID_HTTP_URL` | `--http-url` | unset | A running `anatid-server`'s HTTP listener, for a server on another host. Needs a token. |
| `ANATID_TOKEN` / `ANATID_TOKEN_FILE` | `--token` / `--token-file` | unset | Bearer token for `ANATID_HTTP_URL`, or a file whose first token is used. An `anatid-server` token file works as is. |
| `ANATID_TENANT` | `--tenant` | `0` | Tenant id. Every tool is pinned to it. |
| `ANATID_EMBEDDING_DIM` | `--embedding-dim` | `1536` | `N` in `FLOAT[N]`. Only used when creating a new file; an existing file keeps its own. Over a socket the server's file decides. |
| `ANATID_EMBED_BASE_URL` / `ANATID_EMBED_MODEL` / `ANATID_EMBED_API_KEY` | none | unset | An OpenAI-compatible `/embeddings` endpoint. With both URL and model set, every `remember` stores a vector and every `recall` runs the vector arm; the key is optional for a local endpoint. Environment only, because a key belongs out of `ps`. Refused with `--socket`: the embedder belongs to the process that holds the file. |
| `ANATID_EMBED_HASH` | `--embed-hash` | off | Embed with the deterministic offline stand-in instead of an endpoint (similarity means shared words). For demos and tests. Refused together with an endpoint, and with `--socket`. |
| `ANATID_EXTRACT_BASE_URL` / `ANATID_EXTRACT_MODEL` / `ANATID_EXTRACT_API_KEY` | none | unset | An OpenAI-compatible chat endpoint. With both URL and model set, the `ingest` and `apply_patch` tools are registered. Environment only. Refused with `--socket`: the pipeline runs on the file's own connection. |
| `ANATID_EXTRACT_REASONING` | none | off | `1` sends `{"reasoning": {"enabled": true}}` with every extraction request, which OpenRouter's GLM models take. |
| `ANATID_READ_ONLY` | `--read-only` | off | Open the database read-only. No write tools are registered at all. Over a socket this restricts this MCP server alone; the server keeps its own policy. |
| `ANATID_ENABLE_SQL` | `--enable-sql` | off | The read-only SQL escape hatch is off unless this is set. |
| `ANATID_SQL_TOOL` | `--no-sql-tool` | unset | Legacy switch, kept for older configs. `on` enables the tool the way `ANATID_ENABLE_SQL` does; `off` forces it off even when `ANATID_ENABLE_SQL` is set. |
| `ANATID_MAX_ROWS` | `--max-rows` | `200` | Row cap for the SQL tool. |
| `ANATID_MCP_TRANSPORT` | `--transport` | `stdio` | `stdio`, `streamable-http`, or `sse`. |
| `ANATID_MCP_HOST` / `ANATID_MCP_PORT` | `--host` / `--port` | `127.0.0.1:8765` | Bind address for the HTTP transports. |

### Over HTTP instead of stdio

```sh
anatid-mcp --transport streamable-http --port 8765
```

then point the client at `http://127.0.0.1:8765/mcp`.

## Sharing one memory between clients

`anatid-mcp --db` opens the file in its own process, and DuckDB gives one process exclusive use
of a database file. Measured on duckdb 1.5.5 ([`server.md`](server.md), section 1):

| first process holds | second process wants | result |
| --- | --- | --- |
| read-write | read-write | `IO Error: Could not set lock on file` |
| read-write | read-only | `IO Error: Could not set lock on file` |

So Claude Desktop and Claude Code pointed at the same `ANATID_DB` cannot both be running. The
second one to start is refused, and `anatid-mcp` now says so on stderr, in one paragraph, followed
by the two commands below. A read-only open does not get around it; the second row is
the reason.

The way to share one memory is anatid's server profile: one `anatid-server` process holds the
file, and every `anatid-mcp` talks to it over a Unix socket, reads and writes alike. Two commands.

```sh
# 1. once, the process that holds the file
anatid-server start --socket /tmp/anatid/anatid.sock --db ~/.anatid/memory.anatid --tenant 0

# 2. in every client, the socket in place of the file
anatid-mcp --socket /tmp/anatid/anatid.sock --tenant 0
```

`anatid-server` is in the base install, so nothing extra is installed. Use a short socket path:
`sockaddr_un` holds 104 bytes on macOS, and `/tmp/anatid/anatid.sock` is writable without root,
which `/run/anatid` is not. To keep the server running across logins, [`server.md`](server.md)
section 6 has the launchd and systemd units.

Every tool is the same over the socket as over the file: the same names, arguments, results and
id spelling, because `AnatidClient` answers every verb `Anatid` does with the same types.
`tests/test_mcp_over_server.py` runs every tool against both and compares the answers. Five
things differ, and each is stated rather than hidden:

- `stats.path` reports `unix:/tmp/anatid/anatid.sock` instead of a file path. The file is the
  server's.
- `--enable-sql` is refused with `--socket`. The escape hatch runs statements on the file's own
  DuckDB connection, and over a socket the only such connection is in the server process.
- `--db` is refused with `--socket`. A socket client opens no file, so a config block naming both
  is a mistake, and the message says which one to drop.
- `ANATID_EMBED_*` and `--embed-hash` are refused with `--socket`. Embedding happens in the
  process that holds the file, and a socket client forwards verbs without embedding.
- `ANATID_EXTRACT_*` is refused with `--socket`. The ingest pipeline applies a whole patch in one
  transaction on the file's own connection, which only the server process has.

### Claude Desktop

```json
{
  "mcpServers": {
    "anatid": {
      "command": "/ABSOLUTE/PATH/TO/anatid-mcp",
      "args": ["--socket", "/tmp/anatid/anatid.sock", "--tenant", "0"]
    }
  }
}
```

### Claude Code

`.mcp.json` in the project root, or `~/.claude.json` under `"mcpServers"`:

```json
{
  "mcpServers": {
    "anatid": {
      "command": "/ABSOLUTE/PATH/TO/anatid-mcp",
      "args": ["--socket", "/tmp/anatid/anatid.sock", "--tenant", "0"]
    }
  }
}
```

or, without editing a file:

```sh
claude mcp add anatid -- "$(which anatid-mcp)" --socket /tmp/anatid/anatid.sock --tenant 0
```

`ANATID_SOCKET` in the block's `env` does the same as `--socket`, for a client that can only set
environment variables. With both clients pointed at the socket, a fact remembered in a Claude
Desktop conversation is in the next Claude Code `recall`, and the two can run at the same time.

### The security model, in two sentences

The Unix socket is created mode 0600 in a directory the server sets to 0700, so the user the
server runs as is the only user who can connect, and the kernel enforces it with no token to leak
or rotate; `--allow-uid` on the server narrows that further by peer credentials. A server on
another host is reached with `--http-url` plus `--token`, and `anatid-server` refuses to serve
HTTP without a token, so `anatid-mcp --http-url` without one is refused too.

### From Python

```python
from anatid.integrations.mcp import ServerConfig, build_server, open_backend

db = open_backend(ServerConfig(socket="/tmp/anatid/anatid.sock", tenant=0))
build_server(db).run("stdio")
```

`open_backend` returns an `Anatid` for a file and an `AnatidClient` for a socket or URL; the
tools do not branch on which. Close it yourself.

## Tools

| Tool | Annotation | What it does |
| --- | --- | --- |
| `remember` | write | Write one memory plus the ABOUT edges to the entities it concerns. Entities are given by name and created on demand. `episode` records the raw source text first, as provenance. |
| `relate` | write | A RELATES_TO edge between two entities. Traversed undirected. This is what lets recall reach past one hop. |
| `unrelate` | destructive | Close the RELATES_TO edges between two entities, or only those with a given `rel_kind`. Closing writes a new edge version; `as_of` before the close still walks it. |
| `supersede` | write | Replace a memory with a corrected version. The old row survives, closed, with a SUPERSEDES edge. |
| `correct` | destructive | `supersede` plus the edges named in `remove_relations` closed and those in `add_relations` opened, in one transaction. For a correction that changes who is connected to what. |
| `reinforce` | write | Bump `access_count` / `last_access_at`, optionally set confidence. |
| `forget` | destructive | `hard=false` (default) closes validity and keeps the history; `hard=true` is a right-to-erasure purge. |
| `prune` | destructive | Forget by age and/or usage. `dry_run=true` by default. Needs at least one policy argument. |
| `rebuild_fts_index` | write | Fold the journal into a new BM25 generation. See "Text search" below. |
| `ingest` | read-only | With an extractor configured: propose a memory patch from text, returning `patch_id`, `diff` and `patch`. Writes nothing. Sends the text to the extraction model. |
| `apply_patch` | destructive | Commit a proposed patch, unchanged or edited, in one transaction with the note stored as the episode. Destructive when the patch corrects a memory or removes a relation. |
| `recall` | read-only | Hybrid retrieval: BM25 + graph expansion + cosine when an embedding or an embedder is at hand, fused with weighted RRF (vector 1.0, graph 0.5, text 0.25 when the vector arm runs; text 1.0, graph 0.5 otherwise). `arm_weights` overrides a weight by name and `weights` in the result says what was used. With a query and no `seed_entity` the graph arm seeds itself from the entity names in the query, and `seeds` in the result says which. |
| `context` | read-only | Everything about one entity. `hops=0` direct, `1` neighbors, `2` two hops. |
| `get` | read-only | One memory by id, with its entities. |
| `provenance` | read-only | Walk the SUPERSEDES chain back to the original assertion and its source text. |
| `stats` | read-only | Row counts, schema metadata, BM25 staleness, the graph-expansion path in use. |
| `sql` | read-only | Read-only SQL escape hatch, described below. |

`forget`, `prune`, `unrelate`, `correct` and `apply_patch` carry `destructiveHint: true` in their
MCP tool annotations, so a client can prompt before running them. MCP annotations apply to a
whole tool rather than to individual arguments, so all five are marked even though only
`forget(hard=true)` and `prune(dry_run=false)` remove anything: `unrelate`, `correct` and
`apply_patch` close edge or memory versions that a read with `as_of` before the close still sees.
The graph stops saying something it said, and that is what the hint is for.

## Recall without a seed

`recall(query)` used to run the text arm alone unless the client named a `seed_entity`, so the
graph helped only a caller who already knew which entity to ask about. Since 0.4.0 the graph arm
seeds itself: the query's words are matched against this tenant's entity names, case-insensitive,
longest name first, at most three, and the graph arm expands two hops from each match. The
result's `seeds` lists the names it used and `arms` the arms that ran. Three facts stored by name
alone, "Ada leads Kestrel", "Kestrel owns the ingest service" and "Bo maintains the ingest
service", answer `recall("who maintains the ingest service")` with `arms: ["text", "graph"]` and
`seeds: ["ingest service"]`. Naming a `seed_entity` expands from exactly that one, as before.
Through the library, `seed_entity=None` switches the graph arm off.

The match costs about 1.9 ms with 100,000 entities in the tenant on a laptop, and less than a
millisecond below 10,000. It compares lowercased names, so a stored name with irregular internal
whitespace is matched only when the query repeats it.

## Embeddings

The server embeds nothing by default and the vector arm runs only when a client passes an
`embedding`. Set `ANATID_EMBED_BASE_URL` and `ANATID_EMBED_MODEL` (plus `ANATID_EMBED_API_KEY`
when the endpoint needs one) and the handle embeds every `remember` and every `recall` query
itself, with `stats.embedder` reporting the model and dimension and never the key. The endpoint
has to return vectors of the file's `embedding_dim`, or the server refuses to start with
`EmbeddingDimensionError`. `ANATID_EMBED_HASH=1` (or `--embed-hash`) substitutes a deterministic
offline embedder whose similarity means shared words; it exists for demos and tests. Nothing about
the tools changes either way: `recall` reports `"vector"` in `arms` when the arm ran.

## Ingesting text

With `ANATID_EXTRACT_BASE_URL` and `ANATID_EXTRACT_MODEL` set, two tools take a note instead of
one fact at a time. `ingest(text, source?, writer?)` hands the text and the current facts about
the entities it names to the model, which proposes a patch: facts to add, facts to correct by
memory id, edges to open and close, and names that mean an existing entity. The proposal is
resolved against the database (aliases rewritten, a correction whose target is gone downgraded to
a fact, facts and edges already held dropped, each with a note) and returned with a `patch_id`,
a `diff` for a person to read and the `patch` as JSON. Nothing is written.

```
memory patch: 1 fact, 1 correction, 2 relation(s) added, 1 relation(s) removed
  + fact        "Bo works in the platform group"  about: Bo, platform group
  ~ correction  memory 883936403329363968 "Bo maintains the ingest service"
                -> "Cy maintains the ingest service"  about: Cy, ingest service
  - relation    Bo -maintains-> ingest service
  + relation    Cy -maintains-> ingest service
  + relation    Bo -member_of-> platform group
```

`apply_patch(patch_id, patch?, writer?)` commits it in one transaction: the note is stored
first as an episode, then aliases, new facts, corrections, edges closed and edges opened, every
row citing that episode. Pass an edited `patch` to change the proposal first; memory ids in it
stay decimal strings. If any step fails nothing lands and the proposal stays pending. Declining
is not calling `apply_patch`. A proposal is applied at most once: the server takes it out of
the pending table before writing, so two calls that name the same `patch_id` at the same moment
cannot both commit it, and a `patch_id` that was applied already is answered with the receipt
of that apply, `already_applied` set to true and a `note` saying nothing was written. That
makes a call repeated after a lost reply safe; to store a changed version of the note, call
`ingest` again. Proposals live in the server's memory, at most 64 at a time, as do the receipts
of the last 64 applied, and both are gone when the process exits. [`ingest.md`](ingest.md)
describes the pipeline and the patch schema.

Every read takes `as_of` (an ISO-8601 timestamp) to ask what the database believed at that time.
Time travel is anatid's own filter over the valid-time and transaction-time columns. DuckDB has no
`AS OF SYSTEM TIME`, nothing rewinds, and rows removed by a hard purge are gone from every as-of
view too.

## Ids are decimal strings

Every id in a tool result is a decimal string: `"memory_id": "883768514279557120"`, never
`883768514279557120`. That holds for `memory_id`, `entity_id`, `edge_id`, `episode_id` and
`tenant_id` wherever they appear, including inside a provenance chain, a forget receipt and a
prune report. A tool that takes an id accepts either spelling, a string or an integer, so a client
written against 0.2.0 keeps working. The tool's JSON schema declares the parameter as a string, so
a model reading the schema sends one.

anatid mints 63-bit ids, and a JSON number is an IEEE-754 double in every JavaScript client, so an
id above 2^53 sent as a number comes back changed with nothing raised:

```sh
node -e 'console.log(JSON.parse(String.raw`{"memory_id": 883768514279557120}`).memory_id)'
# 883768514279557100
```

The client then holds an id that addresses no row, and every `get`, `supersede` and `forget` it
makes with that id is wrong. Sending the id as a string is what stops that.

The `sql` tool cannot name the id columns of an arbitrary `SELECT`, so it applies the rule by
range: an integer a JSON number cannot carry exactly leaves as a decimal string and everything
else is untouched. A `memory_id` you select is the same string the other tools take, and a
`count(*)` is still a number.

`version` is a small per-row counter rather than an id, and counts, ranks and scores are not ids
either, so all of them stay numbers.

## The `sql` tool is read-only, and DuckDB is what enforces it

Enforcement is three layers, and none of them is a regex over the SQL text.

1. DuckDB's parser classifies every statement. `extract_statements()` returns one
   `duckdb.StatementType` per statement in the text; only `SELECT` and `EXPLAIN` run, and every
   one of them must pass, so `SELECT 1; DELETE FROM memories` is refused on the second rather than
   half-executed. Because the classification is DuckDB's, it sees through things a regex cannot.
   `PRAGMA create_fts_index(...)` expands at bind time into the `CREATE`/`INSERT`/`UPDATE` it
   really is, and is refused on those. `EXPLAIN ANALYZE <stmt>` executes what it explains and
   DuckDB still types the whole thing `EXPLAIN`, so an `EXPLAIN` is accepted only when the
   statement it wraps is itself a `SELECT`.
2. The parse tree is scanned for anything that reads outside the database.
   `json_serialize_sql()` gives DuckDB's own AST, and two things in it are checked. Function
   names: any `read_csv`, `read_parquet`, `glob`, `read_text`, `postgres_scan`, `duckdb_secrets`,
   and so on is refused, and so are `query()` and `query_table()`, which take SQL as a string; a
   denied function nested inside one of those is a plain string constant in the AST and would
   otherwise be invisible to this scan. Base-table names: each must be a plain identifier. DuckDB's
   replacement scan makes `SELECT * FROM '/etc/passwd.csv'` an ordinary `SELECT` whose AST contains
   no function at all, because the path itself is the table name, and the same trick reaches globs
   (`FROM '/data/*.parquet'`) and, by autoloading `httpfs`, arbitrary URLs
   (`FROM 'https://attacker.example/x.csv'`), which would make the MCP host issue outbound requests
   a prompt-injected model chose. A function deny-list cannot see any of that, so table names are
   allow-listed by shape instead.
3. The statement runs in `BEGIN TRANSACTION READ ONLY` on a private cursor, and is always
   `ROLLBACK`ed. DuckDB's transaction manager refuses any write to the database regardless of what
   got past layers 1 and 2.

Layer 1 is not redundant: a read-only transaction alone does not stop `ATTACH`,
`COPY … TO 'file'`, `INSTALL`, or `CHECKPOINT`, because those do not write the current database.
Layer 1 rejects all four by statement type before layer 3 is reached. Layer 2 is not redundant
either: every construct in its paragraph above is classified `SELECT` by layer 1 and blocked by
layer 3 only from writing, so without it the tool reads any file the server process can read.

The tool does two things a caller might not expect. It is not tenant-filtered: `Isolation.SCOPED`
is a column predicate that anatid's verbs add, and raw SQL does not get it, so add
`WHERE tenant_id = <n>` yourself, or use one file per tenant. It also does not make a `SELECT`
cheap; output is capped by `ANATID_MAX_ROWS`, but a full scan of a large table costs what it costs.

Set `ANATID_SQL_TOOL=off` to remove the tool entirely.

Tables you can query: `memories`, `entities`, `episodes`, `edges_about` (memory→entity),
`edges_relates` (entity→entity), `edges_supersedes` (new→old), `anatid_audit`, `anatid_meta`,
and the view `relates_undirected`. Current rows are `valid_to IS NULL AND tx_to IS NULL`.

## Multi-tenant

The server is pinned to one tenant, and no tool takes a `tenant` argument. DuckDB has no row-level
security, so a `tenant_id` column scopes queries without isolating them: anything that reaches raw
SQL sees the whole file. Real isolation is one file per tenant, which over MCP means one server
entry per tenant:

```json
{
  "mcpServers": {
    "anatid-work": {
      "command": "/ABSOLUTE/PATH/TO/anatid-mcp",
      "env": { "ANATID_DB": "/Users/you/.anatid/work.anatid", "ANATID_TENANT": "1" }
    },
    "anatid-personal": {
      "command": "/ABSOLUTE/PATH/TO/anatid-mcp",
      "env": { "ANATID_DB": "/Users/you/.anatid/personal.anatid", "ANATID_TENANT": "2" }
    }
  }
}
```

One DuckDB file cannot be held open by two processes at once, so give each server its own
`ANATID_DB`, or let one `anatid-server` hold the file and point every `anatid-mcp` at its socket
(see [Sharing one memory between clients](#sharing-one-memory-between-clients)). A second
`anatid-mcp --db` on a file another process holds is refused with that recipe.

## Text search, and what `rebuild_fts_index` is for

DuckDB's own `fts` extension builds a static index: rows written after the last build are
invisible to it. Since 0.2.0 the server's database journals every write in the writing
transaction and merges the journal into each search, so a memory the client just wrote is matched
by the very next `recall` and `bm25_stale` is False. `pending_fts_rows` is how many documents a
search re-reads from the canonical rows, not how many are hidden.

`rebuild_fts_index` is therefore about speed rather than correctness: it folds the journal into a
new generation, which is published in one metadata switch with reads answering from the previous
one throughout. Call it after a large batch of writes. `stats` reports the same numbers, and
`index_health` (through the library) names the state of each derived index.

What it buys is read latency in proportion to the journal, not to the corpus. Measured on this
machine, one tenant of 8-word memories, `recall(query, k=10)` p50 over 15 calls:

| corpus | pending | recall p50 | the rebuild itself |
| --- | --- | --- | --- |
| 20,000 | 20,000 (nothing ever built) | 19.01 ms | |
| 20,000 | 0 (just rebuilt) | 15.56 ms | 489 ms |
| 21,000 | 1,000 | 17.32 ms | |
| 26,000 | 6,000 | 21.33 ms | |
| 26,000 | 0 (rebuilt again) | 16.61 ms | 724 ms |

Repeated runs move those by up to 10%. The top-10 ids across each rebuild are identical, which is
the point of the numbers: a rebuild moves milliseconds and never membership. Skipping it costs
about 1 ms per 1,000 documents in the journal, and skipping it forever is a slow read rather than
a wrong one.

One case still returns nothing from the text arm: no generation has ever been published AND the
tenant holds more than `SCAN_CEILING` (100,000) documents, which is more than an exact scan is
worth. The result says so, and `rebuild_fts_index` fixes it.

The vector arm is a brute-force cosine scan by default. DuckDB's `vss` extension provides an HNSW
index, and 0.2.0 puts it behind the derived-index framework as an opt-in backend, but its on-disk
persistence is experimental and not recommended for production, so the server does not turn it on.
The scan's cost is linear in the tenant's current row count at every size. Past
`BRUTE_FORCE_CEILING` (100,000 memories per tenant) `recall` refuses to run the vector arm and
reports the error to the client; the text and graph arms still answer.

## Troubleshooting

### "Server disconnected", or the server never appears

The client could not run `command`. Use an absolute path, since GUI apps do not inherit your shell
PATH. Check the client's MCP log; on macOS Claude Desktop writes to
`~/Library/Logs/Claude/mcp-server-anatid.log`.

### `IO Error: Could not set lock on file`

Another process holds that database, and DuckDB refuses a second opener even for reading.
`anatid-mcp --db` on such a file exits 2 and prints the explanation with the `anatid-server start`
and `anatid-mcp --socket` commands to run instead. Three ways out: close the other client, give
this one its own `ANATID_DB`, or share the file through a server as
[Sharing one memory between clients](#sharing-one-memory-between-clients) describes. If the
process holding the file is already an `anatid-server`, point this client at its socket.

### Tools appear but every call errors

Run the server by hand to see stderr: `ANATID_DB=/tmp/t.anatid anatid-mcp`. It should sit silently
waiting for JSON-RPC on stdin.

### Everything is read-only and the write tools are missing

`ANATID_READ_ONLY` is set, or the database file is not writable by the client's user.

## Embedding it in your own server

`build_server` takes an already-open `Anatid` handle and returns an `MCPServer` you can extend
with your own tools before running it:

```python
from anatid import Anatid
from anatid.integrations.mcp import build_server

db = Anatid.open("memory.anatid", tenant=0)
server = build_server(db)

@server.tool(title="Today's standup")
def standup() -> str:
    return "\n".join(m.content for m in db.context("standup", limit=10))

server.run("stdio")
```

The server does not take ownership of `db`; close it yourself. `db` can also be an
`AnatidClient` connected to a running `anatid-server`, which is what `open_backend` returns for a
`ServerConfig` with `socket=` set; the tools are the same either way. `build_server(db,
extractor=...)` registers the `ingest` and `apply_patch` tools with an `anatid.ingest.Extractor`
of your own, which needs the embedded handle.
