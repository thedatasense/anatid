# Changelog

All notable changes to anatid are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); anatid uses semantic versioning.

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
