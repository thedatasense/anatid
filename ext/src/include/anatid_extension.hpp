#pragma once

// anatid -- a DuckDB extension providing an in-memory CSR + BFS for anatid's graph recall.
//
// Registered objects
// ------------------
//   anatid_version()                                    -> VARCHAR   version banner
//   anatid_version(VARCHAR)                             -> VARCHAR   banner + echo (kept for the
//                                                                    Phase 0 benchmark harness)
//   anatid_build_csr(edge_table VARCHAR
//                    [, max_bytes := BIGINT] [, max_span_factor := BIGINT])
//                                                                    (re)build the snapshot
//        -> (tenants BIGINT, vertices BIGINT, edges BIGINT, build_ms DOUBLE)
//   graph_expand(tenant_id BIGINT, seed BIGINT, hops INTEGER
//                [, strict := BOOLEAN] [, max_hops := INTEGER])      BFS frontier over the snapshot
//        -> (entity_id BIGINT, depth INTEGER)
//   anatid_csr_stats()                                               one row, or zero rows if no CSR
//        -> (edge_table VARCHAR, current_filter VARCHAR, tenants BIGINT, vertices BIGINT,
//            edges BIGINT, bytes BIGINT, build_ms DOUBLE, built_at TIMESTAMP)
//   anatid_csr_tenants()                                             one row per tenant in the CSR
//        -> (tenant_id BIGINT, vertices BIGINT, edges BIGINT, min_entity_id BIGINT,
//            max_entity_id BIGINT, bytes BIGINT)
//
// The CSR is a SNAPSHOT: it changes only when anatid_build_csr() is called again.  See
// docs/extension.md for the rebuild policy.

#include "duckdb.hpp"

namespace duckdb {

class AnatidExtension : public Extension {
public:
	void Load(ExtensionLoader &db) override;
	std::string Name() override;
	std::string Version() const override;
};

} // namespace duckdb
