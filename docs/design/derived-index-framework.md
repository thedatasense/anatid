# Derived index framework

Status: accepted, not yet implemented. Targets 0.2.

## The problem this replaces

anatid has three accelerators that are not the source of truth: the full-text index, the CSR
adjacency structure, and (soon) an approximate nearest neighbour index. In 0.1 each one solved
staleness its own way. The full-text index was rebuilt by hand and a row count guessed whether it
had gone stale. The CSR was a snapshot that silently fell back to SQL after any write. The vector
arm had no index at all and scanned every row.

Three accelerators with three different staleness stories produce three different ways to be
wrong, and none of them could answer a historical query correctly, because an index built over
current state does not know what was visible at an earlier instant.

## The shape

One mechanism serves all three. A derived index is a versioned base generation plus a
transactionally maintained delta, and every read merges the two before any visibility filter runs.

```
write transaction
    |
    +-- canonical temporal tables            source of truth, always correct
    +-- derived_index_delta / tombstones     written in the same transaction, visible immediately

recall
    |
    +-- base generation (as of its watermark)
    +-- delta rows newer than that watermark
    +-- tombstones subtracted
    +-- tenant and temporal visibility applied to the merged candidate set
    +-- fusion and exact rescoring

maintenance
    |
    +-- build next generation beside the current one
    +-- validate it against the oracle
    +-- publish atomically by switching one metadata row
```

The properties that matter:

A write is visible to the next read in the same transaction, because the delta is written in that
transaction. An index may be stale, corrupt or absent without making a query wrong, because the
delta covers everything the base generation does not and the fallback path is always available. A
generation can be rebuilt or replaced while queries run, because publication is a single metadata
switch and each read pins one generation for its duration.

Visibility filtering happens after candidate generation, never inside the index. An index over
current state cannot answer what was visible last Tuesday, so it is used to narrow candidates and
the temporal predicate decides what survives.

## Visibility is one abstraction

Every read path goes through a single relational abstraction rather than each call site
remembering to add a predicate.

```
visible_at(valid_time, transaction_time)
```

Memories, entities, relationships, provenance, full-text candidates, vector candidates and CSR
results all flow through it. The design goal is that ordinary library code cannot forget the
predicate, because there is no path that takes the raw table.

Supporting rules: versions are immutable rows; a logical object has a stable id distinct from its
version id; interval boundaries are validated and malformed overlaps rejected; late-arriving
corrections have defined semantics; hard purge is explicitly destructive to every historical view
and issues a receipt saying what was removed and when.

## Per accelerator

### Full-text

DuckDB's full-text index does not update in place. Its documented workflow is drop and recreate,
so incremental behaviour has to be built above it.

Each document carries a globally unique composite key of tenant and memory id. A single memory id
is unsafe as a key when several tenants share one file, which was the cross-tenant leak fixed in
0.1.1. The index records a generation and a watermark. A search reads the built index plus the
pending rows newer than the watermark, subtracts tombstones for deleted and superseded documents,
then applies visibility.

Scanning a bounded pending set is acceptable for the first version. A genuinely incremental
implementation eventually owns its postings: terms, document lengths, frequencies, and corpus
statistics scoped per tenant. Scores from two independently built BM25 corpora are not comparable,
so they must not simply be merged.

Maintenance runs on a policy rather than on a human remembering:

```
fts_rebuild_after_rows = 10_000
fts_rebuild_after_ratio = 0.05
fts_rebuild_after_seconds = 900
```

Explicitly callable first. An optional in-process background worker later.

### Vector

DuckDB has an experimental VSS extension providing HNSW indexes. Its persistence is documented as
experimental with write-ahead-log and crash-recovery limitations, so it is offered as a backend
rather than assumed.

```
VectorBackend = Literal["exact", "duckdb_vss", "owned_hnsw"]
```

Exact search stays as the correctness oracle and the fallback. A stable cold generation is
indexed. Everything written or changed after the generation watermark is scanned exactly. The two
candidate sets merge, tombstones are removed, tenant and temporal predicates apply, and surviving
candidates are rescored exactly, so the approximate structure only ever affects which rows are
considered, never the score that ranks them.

Arbitrary historical `as_of` queries use exact search until a versioned vector index exists,
because a current-state index cannot answer historical visibility correctly.

The brute-force ceiling stops being a comment and becomes behaviour: above it, exact search warns
or raises unless the caller passes `allow_slow=True`.

Promoting VSS to the default requires recall at k of at least 0.98 against brute force at the
chosen latency, immediate visibility of new inserts and supersessions, filtered recall tests,
crash recovery with fallback on a corrupted index, and benchmarks at 100k, 1M and 10M rows per
tenant.

### CSR

Dense vertex numbering becomes an internal detail, with an explicit mapping between an external
entity id and a tenant-local dense vertex id.

The CSR is a versioned base generation with a transactional edge delta. New relations enter the
delta immediately and expansion reads base plus delta. Tombstones suppress removed and expired
edges. Compaction produces a new dense map and a new generation, published atomically, and a
recall pins one generation for its duration.

Fallback to SQL remains for unsupported historical queries and unhealthy indexes, and the reason
is reported: stale generation, historical query, rebuild in progress, or load failure. Reporting
only that the CSR was not used tells an operator nothing actionable.

The primary acceptance test generates random graph mutations and proves that CSR plus delta
returns exactly the same rows as the SQL implementation.

## Concurrency and isolation

DuckDB provides snapshot isolation and aborts conflicting transactions. It does not promise
serializable application invariants, so anatid does not claim them either.

Blind retry of a semantic operation stays out. Safer primitives go in:

```
db.atomic(callback, max_attempts=3)
memory.update(..., expected_version=7)
memory.relate(..., if_current=True)
```

`atomic` reruns the whole caller-supplied callback, not the single failed statement, because
rerunning one statement inside an aborted transaction is meaningless.

Invariants are enforced by a unique index where a constraint can express the rule, by
compare-and-swap predicates on a version column, by checking affected row counts, by guard rows
for predicate-level invariants, and by serialising the most sensitive operations through the
writer service. `ConflictError` carries the resource, expected version, current version,
retryability and attempt number.

Single-process writing is a deployment profile rather than a defect. `EmbeddedDatabase` is one
writer process with multiple writer threads. `AnatidServer` is one process owning the files and
accepting writes over a Unix socket or HTTP, with batching, backpressure, authentication,
idempotency keys, per-tenant queues, graceful draining, health endpoints and backup coordination.
DuckDB's Quack remote protocol and DuckLake with a Postgres catalog are optional backends, not a
silent change to the embedded model.

## Tenant isolation

File per tenant stays the secure default, and a shared file provides namespaces rather than a
security boundary. That distinction is stated wherever tenancy is documented.

The pool gains an opaque tenant to file mapping instead of raw names in paths, path traversal
protection, restrictive permissions, optional per-tenant encryption keys, connection eviction with
an open-file limit, per-tenant quotas, backup, deletion and audit events, and a scoped API that
never hands out raw SQL. Administrative access is a separately named `unsafe_connection`.

Every public query path gets adversarial isolation tests: full-text, vector, graph traversal,
provenance, exports and maintenance.

## Order of work

1. Correctness. Composite full-text document identity, the brute-force ceiling enforced or
   explicitly gated, tenant and temporal predicates centralised, uniqueness and compare-and-swap
   protection for concurrent entity and relationship creation, and the ANN documentation
   corrected. Shipped as 0.1.1.
2. The framework. Generations, watermarks, transactional deltas, tombstones, validation, atomic
   publication and health reporting. Full-text and CSR move onto it first, vector follows.
3. Operations. Writer service, idempotency, conflict helpers, quotas, backups and observability.

## References

- DuckDB VSS extension: https://duckdb.org/docs/current/core_extensions/vss
- DuckDB full-text search: https://duckdb.org/docs/stable/guides/sql_features/full_text_search
- DuckDB concurrency: https://duckdb.org/docs/current/connect/concurrency
- DuckDB transactions: https://duckdb.org/docs/current/sql/statements/transactions
