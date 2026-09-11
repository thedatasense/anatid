<p><a href="README.md"><img src="assets/brand/anatid-logo.png" alt="anatid" width="140" height="48"></a></p>

# Changelog

All notable changes to anatid are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); anatid uses semantic versioning.

## [Unreleased]

## [0.4.2] - 2026-09-10

The duck-and-wordmark identity now appears in the README, PyPI description, documentation, and
Procedural Studio. Shared SVG/PNG logos, a favicon, a registry icon, and a GitHub sharing card
live in `assets/brand`. Documentation now has an index and a brand guide, and ships in the source
distribution with the example assets.

Procedural Studio rebuilds the procedural-graph example around fictional medical-device
manufacturing records: reconcile a withdrawn test, validate a repair, reject a shortcut, and
inspect historical procedures and their evidence. Five scripted test cases improve from 1/5
to 5/5. Optional server-side OpenRouter calls provide bounded next-step advice separately from
the evaluator; standalone exports remain offline and contain no API key or session token.

### Changed

- `recall()` fuses its arms with weights, and the graph arm votes for what the question is about.
  With a vector arm running the weights are vector 1.0, graph 0.5, text 0.25; without one, text
  1.0, graph 0.5 (`anatid.recall.default_arm_weights`). Before the fusion the graph arm's
  candidates are ordered by cosine similarity to the query, or by BM25 without an embedding,
  instead of newest first (`anatid.recall.rank_graph_candidates`). `recall(arm_weights={"text":
  0})` overrides any weight by name, on the handle, on `AnatidClient`, through the server's
  verb table and as the MCP `recall` tool's `arm_weights`; `RecallHits.weights` and the tool
  result's `weights` report what was used. Both settings come from the
  answer-quality benchmark in `docs/quality.md`, where equal votes let a BM25 arm over short
  extracted sentences outvote the vector arm and a newest-first graph arm crowded the memory
  block with whatever was written last; the product review had found that anatid's own vector
  arm beat the fusion on the same memories. Chosen with an offline support-coverage proxy
  (`python -m bench.quality.coverage`) on the two published worlds and confirmed on a third the
  choice never saw, the fusion is worth three to five points of coverage on every store. Under the
  LLM judge anatid went from 82% to 89% on the committed corpus and from 91% to 89% on seed 7,
  where rebuilding the store re-rolled the extraction, and answers 88% on the held-out world, one
  point above its own vector arm; the gold-extracted oracle answers 96%, 96% and 99%. `rrf_fuse`
  takes `weights=`; `hybrid_recall` takes `arm_weights=`; the RRF score of a hit is now
  `sum(weight / (60 + rank))`.
- `anatid.ingest.prepare` gained a step, `move_relations`, between `resolve_corrections` and
  `dedupe`: a correction that swaps exactly one entity of the old fact for one new entity is a
  handover, and every edge between the replaced entity and a kept one that the old fact's own
  note opened is closed and reopened with the new entity in its place, unless the patch already
  says so. Edges other notes stated do not move. The benchmark's extraction model corrected the
  fact and left the old edge open in most handovers, so the graph arm kept walking through the
  previous owner.

### Added

- Four executed example notebooks under `examples/notebooks/`: the core verbs in ten minutes, the
  ingestion pipeline step by step, how recall's three arms and their fusion work, and the Agents
  SDK and MCP integrations driven offline. Outputs are saved, so they read without running; the
  cells that talk to a model are guarded. `pip install "anatid[notebooks]"` installs what
  re-executing them needs.
- The benchmark's extraction context uses a versioned `weighted-v1` fusion policy with explicit
  weights, RRF constant and candidate budget. Changing answer-time default weights no longer
  changes the extractor's context. Ingestion reports record the policy; changes to the underlying
  graph ordering or ingestion semantics can still require a rebuild.
- An `anatid` console script, the MCP server under the package's own name beside `anatid-mcp`.
  A client that installs from the MCP Registry runs `uvx <runtimeArguments> anatid@<version>`, and
  `uvx` runs the executable named like the package; `uvx anatid` failed with "an executable named
  `anatid` is not provided by package `anatid`". `mcp-registry/server.json` now says
  `runtimeHint: uvx` and passes `--with anatid[mcp]==<version>`, because the base wheel does not
  require `mcp` and the server cannot import without it. `tests/test_mcp_registry.py` holds the
  manifest, the scripts table, the version and the README's ownership marker together, and
  `docs/mcp-registry.md` describes the launch. Verified against a locally built wheel with the exact
  command VS Code assembles.

### Fixed

- The ingestion pipeline's handover step could move an edge that belonged to another fact. When
  one note stated "Ada reports to Bo" and "Ada mentors Bo", correcting the manager to Cy moved
  the mentorship edge too, while the stored fact still said Ada mentors Bo. When the notes behind
  a fact state more than one fact about a pair, an edge now moves only when the correction's
  wording names its kind, and the edge left alone is noted.
- The same step missed the edge after an earlier wording correction: "Atlas owns the ledger",
  then a clarification that kept the entities, then "Cinder owns the ledger" left the graph
  naming Atlas, because the clarification gave the fact a new episode while the edge kept the
  first note's. The lookup now follows the fact's whole supersede chain.
- A vector arm that ran and found nothing, over memories that have no embeddings, still
  quartered the text arm's weight and ordered the graph arm by a cosine it could not compute, so
  supplying a query embedding could push the right memory behind unrelated newer ones. An arm
  that returns no candidates now casts no vote: the text arm leads, the graph arm is ranked by
  BM25, `hits.weights` omits the empty arm and `hits.notes` says so.
- The answer-quality benchmark's model client (`bench/quality/llm.py`) cached a provider failure
  as an answer: OpenRouter can return HTTP 200 with `finish_reason: "error"` and no content when
  the upstream model fails, and one such reply, for the note that moves Marcus to Atlas and hands
  the Boreal pager to Lena, was stored as that note's extraction in the 0.4.0 build of the
  committed corpus. Six questions rest on that note, and every offline replay reproduced the
  loss. The client now refuses such a reply, retries it with the transport's backoff, counts it
  in `Stats.provider_failures`, never caches it, and treats one an older run cached as a miss.
- `examples/ingest_notes.py` deleted whatever database `--db` pointed at before it started, so
  pointing the demo at a memory you meant to keep emptied it, and the run reported success. The
  script now recreates only its own `ingest_demo.anatid` beside itself; a path given with `--db`
  must not exist yet, and `--reset` is the explicit way to have it deleted first.
- Two MCP `apply_patch` calls for one `patch_id`, concurrent or a retry after a lost reply,
  could both commit the proposal, leaving two copies of every memory and every edge in the graph.
  The server now takes the proposal out of the pending table before it writes anything, puts it
  back if the apply fails, and keeps the receipt of a committed one: a second call for the same
  id is answered with that receipt, `already_applied` true and a `note` that nothing was written.
  `PendingPatches` gained `claim`, `settle` and `restore` for this, and `AppliedPatch` is the
  record it keeps.

## [0.4.1] - 2026-09-05

Nothing yet.

## [0.4.0] - 2026-09-05

The release that closes the four gaps a product review found once the tools were used the way an
agent uses them. The embedded and server profiles are unchanged: the file format, the schema
(v4), the verbs 0.3.0 shipped and their signatures all stand, and every id at an external
boundary is still a decimal string.

The first gap: agents could not maintain the graph through the tools. The Agents SDK had no
`relate` and neither integration had `unrelate`, so three connected facts stored by entity name
alone ("Ada leads Kestrel", "Kestrel owns the ingest service", "Bo maintains the ingest service")
left `recall_2hop("Ada")` with one hit, and nothing let an agent correct a fact together with its
edges. The Agents SDK now has nine tools, with `anatid_relate`, `anatid_unrelate` and
`anatid_correct` gated like the other writes; MCP gains `unrelate` and `correct`. Underneath them
is one new core verb, `Anatid.correct(old_id, content, add_relations=, remove_relations=)`, which
runs `supersede`, the closes and the opens in one transaction and returns a `CorrectionReceipt`.
The verb is also on `AnatidClient` and in the server's verb table, so the client is a drop-in for
the handle again and `anatid-mcp` lists the same tools over a socket as over a file.

The second gap: the default recall ran the text arm alone. `recall(query)` with no seed and no
embedding is now text plus graph: the query's words are matched against the tenant's entity
names, longest name first, at most three, and the graph arm expands from each; `RecallHits.seeds`
names them and `RecallHits.arms` still reports which arms ran. Pass `seed_entity=None` to run
without the graph arm, or name an entity to expand from exactly that one; `seed_entity="auto"` is
the default everywhere, on the handle, on `AnatidClient`, and in both integrations' recall tools.
Embeddings became a protocol: `Anatid.open(embedder=...)` takes any object with `dim` and
`embed(texts)`, `OpenAICompatibleEmbedder` speaks to any `/embeddings` endpoint over the standard
library, `HashEmbedder` is an offline stand-in for demos and tests, and a handle with an embedder
embeds every `remember`, `supersede` and `recall` it is not given a vector for, so the vector arm
runs with no application code. `anatid-mcp` builds the embedder from `ANATID_EMBED_BASE_URL`,
`ANATID_EMBED_MODEL`, `ANATID_EMBED_API_KEY` or `ANATID_EMBED_HASH` (`--embed-hash`), and
`stats.embedder` reports which one, never the key.

The third gap: getting real information in took too much application code. `anatid.ingest` takes
text. An extractor (`OpenAICompatibleExtractor` for any chat endpoint, `ScriptedExtractor` for
tests) proposes a `MemoryPatch` of facts to add, facts to correct, edges to open and close and
names that mean an existing entity; the pipeline resolves it against what the graph holds and
writes a note for everything it changed; a review hook may edit or decline it; `MemoryPatch.apply`
commits the whole patch in one transaction with the raw text stored first as the episode every
row cites. The Agents SDK gets `anatid_ingest` when `create_memory_tools` is given an
`extractor`, approval-gated, with `dry_run=true` returning the diff and `patch=` applying a
reviewed patch as is. MCP gets `ingest`, which proposes and returns a `patch_id` with the diff,
and `apply_patch`, which commits it, from `ANATID_EXTRACT_BASE_URL` and `ANATID_EXTRACT_MODEL`.
`examples/ingest_notes.py` runs three notes through it offline and shows the owner change, the
evidence behind it and what was believed before.

The fourth gap: `anatid-mcp` opened the file directly, so two MCP clients on one writable file
hit DuckDB's exclusive lock with a raw traceback. `anatid-mcp --socket` (or `ANATID_SOCKET`) now
talks to a running `anatid-server` instead, and any number of clients share one memory with the
same tools, arguments, results and id spelling. A held file exits 2 with a one-paragraph
explanation and the two commands to run instead. `--enable-sql`, the embedder and the extractor
are refused with a socket, each with the reason: all three need the file's own connection.

### Added

- `bench/quality`: the answer-quality benchmark. Eleven memory systems (a Markdown file, BM25,
  vectors, their fusion, vectors with one feedback round, anatid with every arm and with each arm
  alone, and anatid built from gold patches as an oracle) answer the same 150 questions with the
  same model, prompt and 1,200-token memory budget, judged blind; every model call is cached so
  `python -m bench.quality.run` reproduces every number. `docs/quality.md` has the method, the
  tables for two seeds, the ablations and the losses next to the wins.
- `Anatid.correct` and `CorrectionReceipt`; the function form `anatid.verbs.correct`; `correct`
  on `AnatidClient` and in the server's verb table, with the receipt registered in the wire codec.
- Agents SDK tools `anatid_relate`, `anatid_unrelate`, `anatid_correct` and, with an extractor,
  `anatid_ingest`. `approve_low_risk` gained `relate=` and `ingest=`, both False by default.
  `Relation` is the element type of the correction tool's relation lists.
- MCP tools `unrelate`, `correct`, `ingest` and `apply_patch`. `unrelate`, `correct` and
  `apply_patch` carry `destructiveHint: true` because they close versions.
- `seed_entity="auto"` and `RecallHits.seeds`; `anatid.recall.auto_seeds`; the constants
  `AUTO_SEED`, `AUTO_SEED_LIMIT`.
- `anatid.embed`: the `Embedder` protocol, `OpenAICompatibleEmbedder`, `HashEmbedder`,
  `EmbedderError`; `Anatid.open(embedder=)` and `db.embedder`.
- `anatid.ingest`: `MemoryPatch`, `PatchReceipt`, `AddFact`, `Correction`, `Relation`, `Alias`,
  `Span`, `Extractor`, `OpenAICompatibleExtractor`, `ScriptedExtractor`, `existing_context`,
  `prepare`, `propose`, `ingest`, `PATCH_JSON_SCHEMA`.
- `anatid.integrations.mcp.backend` (`open_backend`, `ServerHandle`, `DatabaseLocked`),
  `anatid.integrations.mcp.embedding` and `anatid.integrations.mcp.ingest`; `anatid-mcp --socket`,
  `--http-url`, `--token`, `--token-file`, `--embed-hash`, and the `ANATID_SOCKET`,
  `ANATID_HTTP_URL`, `ANATID_TOKEN`, `ANATID_TOKEN_FILE`, `ANATID_EMBED_*` and `ANATID_EXTRACT_*`
  variables. `stats` reports `seeds`-aware recall, the embedder, the extractor and the pending
  patch count.
- `examples/ingest_notes.py` and `docs/ingest.md`; `examples/README.md` lists every example.

### Changed

- `recall(query)` defaults to `seed_entity="auto"` on `Anatid`, `AnatidClient` and in both
  integrations. The old behaviour is `seed_entity=None`. A query that names no entity runs as
  before.
- An empty or whitespace-only entity name raises `ValidationError` from `entity_id`,
  `upsert_entity`, `relate`, `remember` and `correct`, and the write that carried it is rolled
  back. Before, it created an entity named "" that nothing could address by name.
- The Agents SDK recall tool falls back to the query's own seeds when a `seed_entity` it was given
  does not exist, and says so in `notes`, rather than running text only.
- The extension reports version 0.4.0. Its surface is unchanged.

## [0.3.0] - 2026-09-04

The server release. anatid gains a second deployment profile, and the first one is unchanged.

Embedded is still the default and still what most callers should use. `Anatid.open` is one process
with as many writer threads as it likes, no daemon, no socket, no extra hop, and it is faster than
anything that adds one. Nothing in this release changes its behaviour, its file format or its API.

The server profile is for the case embedded cannot serve: two or more processes that must write the
same memory. It exists because of a measurement, not a preference. DuckDB gives one process
exclusive use of a database file, and on duckdb 1.5.5 a second process is refused even when it asks
for read-only access, with `IO Error: Could not set lock on file`. So there is no arrangement where
one process writes through a server and the others read the file directly. One process owns the
files and everybody else asks it, over a Unix domain socket or over HTTP, for reads as well as
writes. Switching is one line: `AnatidClient.connect(path, tenant=1)` in place of
`Anatid.open(path, tenant=1)`, with every verb keeping its name, its parameters and its return type.

The security model is stated rather than implied. A Unix socket is authenticated by the permissions
on the socket and its 0700 directory, plus peer credentials where the platform reports them. An
HTTP listener requires a bearer token and refuses to bind anything but loopback without one. A
principal carries the tenants it may name, and the check happens before any file is opened, so a
client authorised for one tenant cannot reach another and cannot learn from the error whether that
tenant exists. Tenant isolation is still file per tenant; a single shared file gives namespaces,
which is not a security boundary, and the server says so rather than papering over it.

The honest limits, which have not moved:

- One process still owns the files. The server is a single point of failure, not a cluster. There
  is no replication, no sharding and no failover.
- Isolation is DuckDB's optimistic snapshot isolation with write-write aborts. It is not
  serializable, and funnelling writes through one process does not make it so. `ConflictError` is
  still something a caller handles.
- Reads cross the wire too. The exclusive lock rules out reading the file directly while the server
  holds it. Measured on 3,000 memories at 384 dimensions, median: `recall()` costs 1.05x over the
  socket, `get()` costs 2.18x, and writes run at about 0.71x embedded throughput with four
  concurrent writers.
- Backpressure is visible to callers. A tenant's queue is bounded, and a full one answers a
  retryable `BusyError` carrying a wait hint (429 over HTTP) rather than blocking. The write was not
  performed, which is what makes that answer safe to send again.

### Added

- `anatid.server`: `AnatidServer`, `AnatidClient`, the wire protocol, per-tenant write queues with
  batching and idempotency keys, bearer-token and Unix-peer authentication, Prometheus metrics,
  online per-tenant backup, and the `anatid-server` command (`start`, `stop`, `status`, `backup`,
  `restore`, `doctor`, also reachable as `python -m anatid.server`). It needs no dependency beyond
  `duckdb`, so it is in the base install and costs nothing to the callers who never import it.
- Writes for one tenant that are queued at the same moment commit in one transaction. Measured
  with 16 clients writing 100 memories each to one tenant, `--batch-max 32` turns 1,600
  transactions into 200 and is worth between 1.13x and 1.30x depending on how loaded the machine
  is. Verbs that must not share a transaction (`forget`, `prune`, `maintain_indexes`,
  `rebuild_fts_index`, `recluster`) get one of their own.
- Idempotency keys, on by default with a 24 hour lifetime. The key and the write it guards commit
  in the same transaction, in a table in the tenant's own file, so a crash cannot separate them and
  a retry after a restart still writes once.
- Per-tenant fairness. A tenant is served, then goes to the back of the ring whether or not it
  still has work, so a tenant with a thousand queued writes cannot starve one with five. Measured:
  a quiet tenant kept 59% of its solo rate while four processes hammered another.
- Health and readiness as separate questions. A busy server is healthy and not ready, which is what
  keeps a supervisor from restarting it and a load balancer from sending it more.
- `Anatid.checkpoint()`, which folds a file's write-ahead log in. It exists because
  `db.execute("CHECKPOINT")` cannot do it: measured on duckdb 1.5.5, through a thread cursor the
  statement succeeds on a handle for a file this process created and raises
  `TransactionException: Cannot CHECKPOINT: there are other write transactions active` on a handle
  for a file that already existed, from any thread, on a handle that has run nothing else, and
  `FORCE CHECKPOINT`, which DuckDB's message suggests, does not raise and does not return. Issued
  on the handle's root connection it works in both cases (measured: a 2,146,504 byte `.wal` folded
  to 0). It does not force, so a thread with a write transaction genuinely open still raises.
- `ServerConfig(create_tenants=False)` and `anatid-server start --no-create-tenants`. By default a
  server opens a tenant's file the first time a request names it, which is what a service that
  provisions tenants from its own traffic wants; it also lets a client that may name any tenant
  turn a loop over `remember(tenant=i)` into a directory of files. That is not a tenant boundary
  problem, because the caller was entitled to name them, and it is unbounded resource use. With
  the flag the server serves the tenants it was configured with plus the files already on disk,
  and refuses anything else without creating it.
- `docs/server.md`, and `examples/server_demo.py`, which demonstrates the lock, the server and
  several processes writing one memory in three acts.

### Fixed

- SIGTERM is now bounded by `--drain-timeout`. `shutdown()` waited on the listener before it
  stopped the queue accepting or cancelled the connections, and that wait does not return while a
  connection handler is running, so a single connected client held the process open indefinitely:
  measured at over 25 seconds against a 5 second budget with one idle connection, and about 2,100
  further writes accepted and committed after the signal with 64 writing clients. The queue now
  stops accepting first, so a request already decoded gets `ShuttingDown` as documented, the
  listener is waited on last and under a bound, and a second SIGTERM reaches the drain instead of
  queueing behind it.
- Ids no longer leave the server as JSON numbers. anatid ids are 63-bit and a JSON number is an
  IEEE-754 double in every JavaScript client, so an id above 2^53 was rounded silently: 12 of 12
  ids sent through a real Node process came back with different digits and none of them addressed a
  row. The server protocol is a new external boundary and now applies the rule
  `anatid.integrations.wire` already applies at the Model Context Protocol and Agents SDK
  boundaries, from that module's single definition. An integer a JSON number cannot carry exactly
  travels tagged, with its digits in a string, in both directions and in error details as well as
  results; a Python client still sees `int` on both sides. Every reply of every verb is swept for
  the shape, and the frames are round-tripped through a real Node process in the test suite.
- The backup drain is per tenant. It waited on every tenant's queue, so backing up an untouched
  tenant cost 0.16 seconds idle and 3.90 seconds while a different tenant was being written, which
  is not the "only this tenant pauses" its docstring claimed. The drain's outcome is also reported
  now rather than discarded: a copy taken with writes still queued is consistent, but it is not the
  point where everything acknowledged so far had landed, and the command says which one you got.
- The `busy` wait hint scales with the crowd. It was computed from queue depth alone, which is the
  time until one more write fits and not the time until this caller's turn, so sixteen clients
  refused at the same instant were each told the same few milliseconds, came back together, and
  fifteen were refused again. The hint now scales with the number of writes that tenant has refused
  since it was last under its high-water mark, and is jittered so a crowd does not return in step.
- `GET /ready` no longer hands an anonymous caller the tenant list and the per-tenant schema
  versions on a token-protected listener, and the `ready` verb no longer hands them to a principal
  scoped to one tenant. The verdict a load balancer reads stays open, because it describes the
  process and not a tenant; the detail is scoped to the tenants the caller may name, because a
  principal must not be able to learn whether a tenant it may not name exists. `GET /health` is
  unchanged and stays open: it carries a pid, an uptime and a version, and a supervisor has no
  token.
- An online backup holds a barrier rather than draining. `WriteQueue.drain_tenant` waits for a
  tenant's queue to empty and then returns, stopping nothing, so a write submitted between the
  drain returning and the copy starting committed into the copy: the boundary was the weaker
  snapshot one however long the drain waited, and under continuous writes the drain could spend
  its whole budget and still deliver only that. Both `AnatidServer.backup_tenant` and the
  `anatid-server backup` verb now go through `BackupCoordinator`, which takes the tenant's single
  serving slot for the length of the copy, so every write acknowledged before the call is in it
  and nothing that committed after the barrier closed is. Measured on a 42.3 MiB file of 100,000
  memories: 502 ms, 8.8 ms more than the same copy with nothing paused. No other tenant is ever
  paused, and a barrier that cannot be taken inside the budget refuses the copy rather than
  quietly downgrading it.
- `DrainReport.drained` counts what that drain drained. It was the queue's lifetime
  completed-plus-failed counter, so a server that had served two hundred writes and drained three
  on the way out logged "shutdown drained 203 writes cleanly" and the command line printed the
  same number. It is now a delta taken across the call, and it is read after the workers are
  joined rather than a moment before the abandon, so `drained` and `abandoned` partition what was
  outstanding instead of leaving a gap: a run that reported `drained=44` alongside 45 completed
  futures and 355 abandoned ones now adds up.
- A bearer token configured with leading or trailing whitespace is refused at construction. The
  header value is stripped before it is compared, because a client library that appends a newline
  is routine, so such a token could never match anything and the server started happily and
  answered every request with "the bearer token was not accepted". A token containing whitespace
  or a control character is refused for the same reason: a space ends the value in an
  `Authorization` header and a newline ends the header.
- Binding HTTP to a loopback address with an authenticator that requires no token now logs a
  warning. It is correct on a single-user machine and wrong on a shared host, where `127.0.0.1`
  is reachable by every local user and no filesystem permission stands between them and the port.
  Only the operator knows which this is, so the library warns; `anatid-server` already refuses
  `--http` without `--token-file` unless `--http-no-auth` says the choice was deliberate.
- Two classes can no longer claim one wire tag. `anatid.types.PruneReport` (what `prune()`
  removed from a tenant) and `anatid.server.backup.PruneReport` (what backup retention removed
  from a directory) share a class name; the second registration silently replaced the first, and
  the failure surfaced three modules away as `cannot rebuild PruneReport from the wire`. A second
  claim on a tag now raises at import time and names both classes. The escape hatch it points at
  also works now: `register_dataclass(cls, name)` set the decoder's key while the encoder still
  tagged by `type(value).__name__`, so a type registered under another name was written under its
  own and could not be read back at all.
- `queue_stats` works on a server built through the library. The verb is in the public table, but
  its return type was registered with the codec only in `anatid.server.cli`, so a server built with
  `AnatidServer(pool=...)` ran the verb and then failed to encode the answer, with an error that
  blamed a version difference that did not exist. The registration now lives next to the type.

### Changed

- The C++ CSR extension reports version 0.3.0, matching the package. The test suite compares the
  two, and its feature gates now compare parsed versions rather than version-string prefixes, so a
  release cannot silently turn every test behind a gate into a skip. Rebuilding it also exposed a
  timing-dependent assertion: the extension refuses a sparse-id CSR build with one of two messages
  depending on whether the ids landed more than `ANATID_MAX_TENANT_SPAN` apart, which for
  time-ordered ids is a question about how long the test's twelve writes took. The test now asserts
  what both refusals say.
- `AnatidServer.backup_tenant` returns a `BackupReport` rather than a path, so a caller sees the
  guarantee the copy has, its size, how long the tenant was paused and what schema version came
  back. `anatid-server backup` prints the guarantee and its reply carries `guarantee`, `quiesced`,
  `paused_s` and `waited_s`.
- `DatabasePool.backup` refuses an existing destination with `BackupDestinationExists`, which is
  both a `FileExistsError` and an `AnatidError`. The old bare `FileExistsError` is an `OSError`,
  so a command line catching `OSError` reported "the arguments are wrong and nothing was written"
  as a run-time failure, and an HTTP layer that recognises anatid's errors answered 500 where 400
  was the honest answer. It also quotes its source catalog with the new `anatid.schema.quote_name`
  instead of validating it with `quote_ident`, which refused the two pool templates the
  documentation itself shows: `tenant-{tenant}.anatid` gives DuckDB the catalog `tenant-1` and
  `{tenant}.anatid` gives `1`, and neither is a bare SQL identifier. The command line's fallback
  copy for those templates is gone with the reason for it.
- The backup module's types and errors cross the wire. `Guarantee`, `BackupReport`, `BackupInfo`,
  `RestoreReport` and its `PruneReport` are registered with the codec, along with `BackupError`
  and its four subclasses, so a client sees `QuiesceTimeout` rather than a generic `RemoteError`
  that has lost the one thing it says: nothing was copied and nothing was paused, so the same
  call can be made again. `QuiesceTimeout` and `QuiesceUnavailable` are marked retryable and
  answer 503 over HTTP; `DestinationInUse` answers 409. `pathlib.Path` has a codec, because a
  backup report names a file on the server's disk.
- `anatid.server` re-exports the backup and client surfaces. `BackupCoordinator`, `Guarantee`,
  `AnatidClient` and the rest were reachable only through their own modules, so
  `from anatid.server import BackupCoordinator` raised `ImportError`. `inspect` and `prune` are
  exported as `read_backup` and `prune_backups`, because at package level the first collides with
  a standard library module and the second with a verb about memories rather than files.
- `docs/roadmap.md` no longer says a server is not anatid's to invent. The principle it states now
  is that there is no *mandatory* server: embedded by default, the server opt in for callers who
  need several processes writing one file, and not a network database.

## [0.2.1] - 2026-09-04

Identifiers now cross every external boundary as decimal strings. anatid identifiers are 64 bit and
exceed what JavaScript integers carry safely: sent as a JSON number, 883768514279557120 comes back
from Node as 883768514279557100. An agent calling supersede or provenance on a memory it had just
stored could address a different row, and nothing would raise. Tools accept an identifier as a
string or an integer, so clients written against 0.2.0 keep working, and the tool schemas declare
string. Any client that parsed identifiers as numbers should now read them as strings.

The text index documentation described 0.1 behaviour. It said writes stayed invisible until
rebuild_fts_index() ran, which the derived index made false in 0.2, where the journal carries a
write to the very next read. The Model Context Protocol instruction text mattered most, since
agents are given it as guidance. Every stale claim now describes what happens, and says what
rebuilding is still for.

The dinner example corrected its sentence without correcting its graph, leaving Priya recorded as
reacting to both pine nuts and prawns. Scenarios now carry removed_relations and apply supersede,
unrelate and relate inside one transaction.

Loopback detection in the Model Context Protocol server was case sensitive, so LOCALHOST was
treated as a public interface. Hostnames are normalised, including the trailing dot and the
bracketed IPv6 forms.

The README was rewritten.

### Changed

- Every id crossing an integration boundary is now a decimal string. The MCP tools and the OpenAI
  Agents SDK tools return `memory_id`, `entity_id`, `edge_id`, `episode_id` and `tenant_id` as
  strings wherever they appear, including inside a provenance chain, a forget receipt and a prune
  report, and the `sql` escape hatch turns any integer a JSON number cannot carry exactly into a
  string the same way. A tool accepts an id as a string or as an integer, and a tool's JSON schema
  declares an id parameter as a string and says why. This is a breaking change for any client that
  parsed an id as a number: anatid ids are 63-bit, a JSON number is an IEEE-754 double in every
  JavaScript client, and an id above 2^53 was rounded silently, so the id such a client held
  addressed no row and every call it made with that id was wrong. The contract has one definition,
  `anatid.integrations.wire`, and both boundaries use it.

### Fixed

- The documentation said a write was invisible to the text arm until `rebuild_fts_index()` ran.
  That stopped being true in 0.2.0, which attaches the derived text index by default: a new memory
  is matched by the very next `recall` with nothing rebuilt, and a `supersede` or a `forget` leaves
  the text results on that same read. The package docstring, `recall()`'s docstring, the `Anatid`
  class contract, `StaleIndexError`, `FtsStatus`, the contract stamped into every new file,
  `docs/mcp.md` and the MCP server instructions a model is given now describe that, and say what a
  rebuild is for: it compacts the journal into a new base generation and buys read latency.
  `doctor()`'s stale-index detail comes from the same source `recall()` uses, so it describes the
  half the database is actually running. The old claim is kept where it is still true, on
  `Anatid.open(accelerators=False)` and on DuckDB's own `PRAGMA create_fts_index`.
- The examples corrected the sentence and left the graph contradicting it. `examples/scenarios.py`
  only ever added relations, so after the dinner scenario's handover Priya reacted to both pine
  nuts and prawns, and after the on-call handover both Bo and Cy maintained the service. A
  `Supersede` now declares the relations a correction closes as well as the ones it opens, and
  `supersede`, `unrelate` and `relate` run inside one transaction, so a correction cannot half
  land. `examples/dinner_party.py`, `examples/glm_openrouter_agent.py` and the studio print and
  return the edges that were closed beside the ones that were opened, and read current edges with
  the predicate anatid means by current.
- `anatid-mcp` refused an HTTP bind host that is loopback spelled another way. The check now
  normalises a host before deciding: whitespace, a trailing dot, an IPv6 URL literal in brackets,
  an interface scope such as `::1%lo0`, and case. `127.0.0.1.` and `[::1].` were refused on every
  platform, and `LOCALHOST` on any platform whose resolver does not fold case. A host that is not
  loopback is still refused, including a name that also resolves off this machine.

## [0.2.0] - 2026-09-03

The derived-index release. Every retrieval structure anatid keeps beside the canonical tables is
now built by one mechanism, described in `docs/design/derived-index-framework.md`: a versioned base
generation plus a journal written in the same transaction as the row it describes, merged on every
read before any tenant or time filter runs. A write is findable by the next read with nothing
rebuilt, an index that is stale, damaged or absent costs latency rather than correctness, and every
fallback reports which of eight reasons applies.

Files written by 0.1.x (schema v3) migrate to schema v4 the first time they are opened for writing;
the migration runs inside one transaction and adds columns and tables without rewriting a row. It
is not reversible: 0.1.1 refuses to open a v4 file.

### Added

- The derived-index framework (`anatid.derived`, schema v4). `DerivedIndex` with generations,
  an ordered journal, validation against the oracle, publication by one metadata-row `UPDATE`,
  process-wide pins, retirement, health reporting and a `MaintenancePolicy`. Index definitions live
  in the file (`anatid_index_registry`), so every handle journals every write for every enabled
  index whether or not it holds that accelerator's code. `db.index_health()`, `db.maintain_indexes()`.
- One visibility abstraction (`anatid.visibility`). `Visibility.at(tenant, as_of).predicate()`
  renders the tenant predicate and both time axes; `visible_at(...)` is the same object named for
  the axes; `Visibility.admits(row)` is the Python mirror. Every read path in the library goes
  through it, and a test scans each module's SQL literals to prove none writes it by hand.
- Immutable version rows. `memories`, `edges_about` and `edges_relates` carry a `version`
  column. A correction closes the current version's `tx_to` and inserts the next version rather
  than rewriting `valid_to` in place, so an `as_of(valid_time, tx_time)` read returns the belief
  the database actually held then. `db.versions(id)` lists them; `Provenance.versions` carries
  them beside the SUPERSEDES chain.
- Full-text search on the framework (`anatid.fts`), attached by default. A search merges the
  base generation with the journal and reconstructs one set of corpus statistics over their union,
  so a rebuild never changes an answer. With no generation published the search is an exact scan.
  `forget(hard=True)` deletes the document from every generation's storage.
- The CSR on the framework (`anatid.csr`), attached by default. One generation per tenant with
  its own dense vertex map, so the caller's 63-bit ids work and a `relate()` no longer invalidates
  anything. The journal is applied level by level, in SQL or inside the C++ extension. The
  extension is version 0.2.0: named snapshots, an external-id label mapping, delta arrays on
  `graph_expand`, and `anatid_drop_csr`.
- An opt-in HNSW vector backend (`anatid.vector`). `Anatid.open(vector_backend="duckdb_vss")`.
  The approximate structure only chooses candidates; the score is always the exact cosine. Recall
  at k measured against the exact oracle: 1.0000 at k=10 and 0.9982-0.9984 at k=50, at 9,500 and
  95,000 rows per tenant.
- Conflict primitives (`anatid.atomic`). `db.atomic(callback, max_attempts=3)` re-runs the
  whole callback on a retryable conflict with jittered backoff; `db.update(id, content,
  expected_version=n)` is a compare-and-swap; `relate(..., if_current=True)` refuses an endpoint
  with no current entity row. `ConflictError` carries resource, expected version, current version,
  retryability and attempt.
- Pool hardening. `DatabasePool(opaque=True, secret=...)` maps a tenant to a keyed digest
  instead of a name in a path; interpolated path components that could escape the pool root are
  refused rather than sanitised; directories are created 0700 and files 0600; per-tenant
  `delete()` and `backup()`; a bounded audit log with an `audit=` hook; and
  `Anatid.unsafe_connection(reason=...)` as the named administrative cursor, with a per-handle
  `raw_access` policy.
- `HealthReason.damaged_base`, and a cheap invariant per index checked on every read, so a base
  that is present, queryable and quietly incomplete becomes a reported fallback instead of a short
  answer. `doctor()` gains `unusable_derived_index`.
- `Anatid.last_expansion`, `Anatid.index_health(as_of=...)`, and `anatid.fts` / `anatid.vector` /
  `anatid.csr` / `anatid.atomic` exported from the package root.

### Changed

- `Anatid.open()` gains `accelerators: bool = True`, which attaches the full-text and CSR derived
  indexes, and `vector_backend: str = "exact"`. Attaching costs one journal `INSERT` per write per
  index that derives from the table written, measured at 0.68 ms per `remember()` on this machine;
  `accelerators=False` is 0.1.1's behaviour exactly.
- `FtsStatus.stale` means something different on each half. On 0.1.1's index it is "rows the arm
  cannot see". On a generation a write is searchable at once, so it is "the answer would be
  incomplete", which happens only with no usable generation and a corpus above `SCAN_CEILING`.
  `pending_rows` is how many documents a search re-reads, not how many are hidden.
- `rebuild_fts_index()` on a database with the derived index attached builds, validates and
  publishes a generation, with reads answering from the previous one throughout.
- `db.expand_path` is documented as a forecast for the next current-state read on this handle's
  own tenant rather than a record of the last read, which is what `db.last_expansion` is.
- `doctor()`'s `stale_fts_index` check is about 0.1.1's index and is silent on a database that has
  moved off it, correctly: nothing there is invisible.
- The version string, the schema version and the C++ extension banner are checked against each
  other by a test. 0.1.1 is published at schema v3; a build writing schema v4 cannot share its name.

### Fixed

- A schema-v3 file opened `read_only=True` or `ensure=False` raised a raw DuckDB
  `BinderException` on every hydrating read, because the select list named the `version` column
  that only the 3 -> 4 migration adds and neither open mode runs the migration ladder. Those are
  supported open modes (`anatid-mcp --read-only` is one), and 0.1.1 answered the same reads on the
  same file. The select list is now asked for rather than assembled, and renders `1 AS version`
  against a table that predates the column.
- `forget(hard=True)` now reaches derived-index storage, deletes the journal rows rather than
  tombstoning them, lowers a generation watermark that was the erased id, and invalidates a
  generation whose storage cannot delete one document. `ForgetReceipt` counts both halves.
- A generation's storage name renders a negative tenant id as `n7` rather than `-7`, which is not
  an identifier character.

### Removed

Nothing. Every 0.1.1 name still resolves and means what it meant; `Anatid.open(accelerators=False)`
restores 0.1.1's retrieval behaviour on a v4 file.

## [0.1.1] - 2026-09-02

A correctness and security release. An external review of 0.1.0 reported five defects with
reproductions; all five are fixed here, with tests that run the reproductions. Files written by
0.1.0 (schema v2) migrate to schema v3 automatically the first time they are opened; the migration
runs inside one transaction and a failure part-way leaves the v2 file untouched.

### Security

- Cross-tenant leak through full-text search. The BM25 index was keyed on `memory_id`, which
  is unique per tenant rather than per file, so a search in one tenant could return a row from
  another tenant whose memory shared the id, and a tenant could learn that another tenant's corpus
  contained a term. The index is now built over a composite `tenant_id:memory_id` key, every BM25
  candidate is restricted to the querying tenant before scoring, and the document frequencies and
  corpus statistics behind each score are kept per tenant, so another tenant's writes change
  neither this tenant's hits nor its scores. Opening a v2 file drops the old index; call
  `rebuild_fts_index()` once after the upgrade.

### Fixed

- Entity creation race. Concurrent `remember()` calls naming the same new entity could each
  create it, fracturing the graph. Entities now carry a generated canonical key (lower-cased,
  whitespace-collapsed name) with a `UNIQUE (tenant_id, entity_key)` index; a writer that loses the
  race re-runs its transaction and reads the winner. The migration merges duplicates that already
  exist, repointing their edges to the surviving row.
- MCP `sql` tool failed open. The tool ran a statement whose parse tree could not be
  inspected; it now refuses it. The tool is off by default and must be enabled with
  `ANATID_ENABLE_SQL=1` or `--enable-sql`; enabling it sets DuckDB's `enable_external_access` off
  on the server's database, applies a statement timeout and a memory limit, and the server refuses
  to start an HTTP transport bound to a non-loopback address unless authentication is declared
  with `ANATID_MCP_AUTH` or `--auth`.
- Erasure was incomplete. `forget(hard=True)` left the erased id and its verbatim content in
  the Agents SDK run-state table, and reached the transcript table only on the handle that had
  created the session. The purge now clears the bundled integration tables on every handle, the
  full-text index tables (including tokens that occurred only in the erased text), the episode
  when no other memory cites it, and the BM25 watermark when the erased row was the newest
  indexed one. A test scans every table in the file for the erased id and content after a purge.
- No integrity validation. Verbs now reject non-finite embeddings, confidence and weight
  values outside `[0, 1]`, non-positive limits and candidate counts, and explicit ids already in
  use in the tenant, and refuse to close a memory's interval before it opened. The id allocator no
  longer repeats ids after a clock rollback. `db.doctor()` returns a structured report of duplicate
  ids, duplicate entities, dangling edges and episode references, embedding dimension mismatches,
  non-finite embeddings, out-of-range values, inverted intervals, stale or drifted full-text
  statistics and schema drift.
- `BRUTE_FORCE_CEILING` is enforced: `recall(embedding=...)` raises `BruteForceCeilingError` when
  the vector arm would scan more than 100,000 rows of one tenant, unless `allow_slow=True` is
  passed. The text and graph arms are not affected.

### Changed

- The `[dev]` extra now installs everything the whole test suite needs (numpy and pyarrow were
  undeclared); `test`, `bench`, `lint` and `types` extras are available separately.
- The package ships a `py.typed` marker; the verb mixin's bodyless placeholder methods were
  replaced by a typed `Protocol`, which removed the type errors they caused.
- The wheel is pure Python. The optional C++ extension is built from a source checkout only and
  is not part of the wheel or the sdist.
- Documentation no longer claims that DuckDB ships no ANN index: DuckDB has a team-maintained
  `vss` extension with an HNSW index whose persistence is experimental and not recommended for
  production, which is why anatid's own index remains a roadmap item.
- CI runs ruff format and lint checks, pyright, a coverage floor, a wheel-install job, Windows,
  and a lowest-direct-dependency resolution alongside the latest-version one.

## [0.1.0] - 2026-09

Initial release.
