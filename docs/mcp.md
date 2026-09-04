# anatid over MCP

`anatid-mcp` serves an anatid database over the [Model Context Protocol](https://modelcontextprotocol.io).
Claude Code, Claude Desktop, Cursor, and anything else that speaks MCP get a persistent,
bitemporal, graph-shaped memory backed by one embedded DuckDB file. There is no server process to
run, no container, and no API key.

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
| `ANATID_DB` | `--db` | `~/.anatid/memory.anatid` | Database file, or `:memory:`. Parent directories are created. |
| `ANATID_TENANT` | `--tenant` | `0` | Tenant id. Every tool is pinned to it. |
| `ANATID_EMBEDDING_DIM` | `--embedding-dim` | `1536` | `N` in `FLOAT[N]`. Only used when creating a new file; an existing file keeps its own. |
| `ANATID_READ_ONLY` | `--read-only` | off | Open the database read-only. No write tools are registered at all. |
| `ANATID_ENABLE_SQL` | `--enable-sql` | off | The read-only SQL escape hatch is off unless this is set. |
| `ANATID_SQL_TOOL` | `--no-sql-tool` | on | Legacy switch, kept for older configs. `off` removes the tool entirely. |
| `ANATID_MAX_ROWS` | `--max-rows` | `200` | Row cap for the SQL tool. |
| `ANATID_MCP_TRANSPORT` | `--transport` | `stdio` | `stdio`, `streamable-http`, or `sse`. |
| `ANATID_MCP_HOST` / `ANATID_MCP_PORT` | `--host` / `--port` | `127.0.0.1:8765` | Bind address for the HTTP transports. |

### Over HTTP instead of stdio

```sh
anatid-mcp --transport streamable-http --port 8765
```

then point the client at `http://127.0.0.1:8765/mcp`.

## Tools

| Tool | Annotation | What it does |
| --- | --- | --- |
| `remember` | write | Write one memory plus the ABOUT edges to the entities it concerns. Entities are given by name and created on demand. `episode` records the raw source text first, as provenance. |
| `relate` | write | A RELATES_TO edge between two entities. Traversed undirected. This is what lets recall reach past one hop. |
| `supersede` | write | Replace a memory with a corrected version. The old row survives, closed, with a SUPERSEDES edge. |
| `reinforce` | write | Bump `access_count` / `last_access_at`, optionally set confidence. |
| `forget` | destructive | `hard=false` (default) closes validity and keeps the history; `hard=true` is a right-to-erasure purge. |
| `prune` | destructive | Forget by age and/or usage. `dry_run=true` by default. Needs at least one policy argument. |
| `rebuild_fts_index` | write | Fold the journal into a new BM25 generation. See "Text search" below. |
| `recall` | read-only | Hybrid retrieval: BM25 + graph expansion + optional cosine, fused with RRF. |
| `context` | read-only | Everything about one entity. `hops=0` direct, `1` neighbors, `2` two hops. |
| `get` | read-only | One memory by id, with its entities. |
| `provenance` | read-only | Walk the SUPERSEDES chain back to the original assertion and its source text. |
| `stats` | read-only | Row counts, schema metadata, BM25 staleness, the graph-expansion path in use. |
| `sql` | read-only | Read-only SQL escape hatch, described below. |

`forget` and `prune` carry `destructiveHint: true` in their MCP tool annotations, so a client can
prompt before running them. MCP annotations apply to a whole tool rather than to individual
arguments, so both are marked even though only `forget(hard=true)` and `prune(dry_run=false)`
actually remove anything.

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

One DuckDB file cannot be held open read-write by two processes at once, so give each server its
own `ANATID_DB`. A second client trying to open the same file gets a DuckDB file-lock error.

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

Another process holds that database read-write. Close the other client, or give this one its own
`ANATID_DB`.

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

The server does not take ownership of `db`; close it yourself.
