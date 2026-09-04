# Changelog

All notable changes to anatid are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); anatid uses semantic versioning.

## [Unreleased]

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
