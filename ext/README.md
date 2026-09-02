# `anatid` — the DuckDB extension

The C++ half of [anatid](../README.md): an in-memory per-tenant CSR over the `RELATES_TO` edge
table, plus a BFS table function, so that anatid's 2-hop recall frontier comes from C++ instead of
SQL joins.

Measured at 1,000,000 memories / 2.3M edges / 10 tenants — 2-hop recall p50 **2.04 ms** with this
extension, **2.88 ms** with the equivalent pure SQL, **7.35 ms** with LadybugDB 0.20.2, all three
returning identical id-lists.

**The full story — why it exists, the snapshot limitation, when to rebuild, the dense-id
requirement, every error — is in [`docs/extension.md`](../docs/extension.md). Read that one.**
This file is just the build card.

## Build

```sh
git submodule update --init --recursive     # ext/duckdb (pinned v1.5.5) + ext/extension-ci-tools
cd ext
GEN=ninja make release
```

Out:

* `build/release/extension/anatid/anatid.duckdb_extension` — the loadable binary
* `build/release/duckdb` — a shell with the extension linked in
* `build/release/test/unittest` — the sqllogictest runner

First build ≈ 10–20 min (it compiles DuckDB). Incremental rebuilds after editing
`src/anatid_extension.cpp` are ≈ 5 s. `make debug` builds into `build/debug/` instead.

> In this working tree `duckdb/` and `extension-ci-tools/` are **symlinks** into
> `../spike/extension/`, which already holds the exact pinned checkouts; the build reads them and
> never writes there. Replace them with the submodules in [`.gitmodules`](.gitmodules) once `ext/`
> is a real git checkout.

## Try it

```sh
./build/release/duckdb
```

```sql
CREATE TABLE edges_relates(edge_id BIGINT, src BIGINT, dst BIGINT, tenant_id INTEGER,
                           valid_to TIMESTAMP, tx_to TIMESTAMP);
INSERT INTO edges_relates VALUES (0, 1, 2, 0, NULL, NULL), (1, 2, 3, 0, NULL, NULL);

SELECT * FROM anatid_build_csr('edges_relates');   -- tenants, vertices, edges, build_ms
SELECT * FROM graph_expand(0, 1, 2);               -- entity_id, depth
SELECT * FROM anatid_csr_stats();                  -- what is in memory right now
```

## Test

```sh
GEN=ninja make test          # ext/test/sql/*.test through DuckDB's sqllogictest runner
```

and from the repo root, the cross-check against the pure-SQL path on the Phase 0 dataset:

```sh
.venv/bin/python -m pytest tests/test_extension.py -q
```

## Layout

| path | what |
|---|---|
| `src/anatid_extension.cpp` | everything: CSR build, BFS, the five registered functions |
| `src/include/anatid_extension.hpp` | the extension class + the registered signatures |
| `test/sql/anatid_csr.test` | build, hops 0/1/2, temporal filter, tenant scoping, snapshot contract |
| `test/sql/anatid_csr_errors.test` | every error path, the hop limit, strict mode |
| `test/sql/anatid_csr_stats.test` | `anatid_csr_stats()` / `anatid_csr_tenants()` |
| `CMakeLists.txt`, `extension_config.cmake`, `Makefile` | the DuckDB extension-template build |

Built from [duckdb/extension-template](https://github.com/duckdb/extension-template). MIT licensed,
like the rest of anatid.
