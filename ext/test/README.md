# Tests for the `anatid` extension

`sql/` holds [sqllogictests](https://duckdb.org/dev/sqllogictest/intro.html), DuckDB's own test
format. Run them all from `ext/`:

```sh
GEN=ninja make test          # release build
GEN=ninja make test_debug    # debug build
```

or one file directly:

```sh
./build/release/test/unittest test/sql/anatid_csr_errors.test
```

| file | covers |
|---|---|
| `sql/anatid_csr.test` | `anatid_build_csr` + `graph_expand` on a tiny inline graph: hops 0/1/2, undirected traversal, the `valid_to`/`tx_to` current-state filter, tenant scoping, an unknown tenant, composition with ordinary SQL, and the snapshot contract (a new edge is invisible until a rebuild) |
| `sql/anatid_csr_errors.test` | no CSR built, a missing table, a table without `tenant_id`/`src`/`dst`, NULL and negative arguments, the `max_bytes` budget, the sparse-id density bound and `max_span_factor :=`, the 30-hop limit and `max_hops :=`, and `strict := true` |
| `sql/anatid_csr_stats.test` | `anatid_csr_stats()` and `anatid_csr_tenants()` before a build (zero rows), after a build, after a rebuild, and after repointing at a different edge table |

The bigger correctness question — *does the C++ path return the same frontier as the pure-SQL path
anatid falls back to?* — is answered in Python, against the Phase 0 spike's 100k-memory dataset, in
[`../../tests/test_extension.py`](../../tests/test_extension.py).
