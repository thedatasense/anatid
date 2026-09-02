#define DUCKDB_EXTENSION_MAIN

// anatid DuckDB extension: an in-memory undirected CSR over an edge table plus a BFS table
// function, so that anatid's 2-hop recall frontier comes from C++ instead of SQL joins.
//
//   SELECT * FROM anatid_build_csr('edges_relates');  -- (tenants, vertices, edges, build_ms)
//   SELECT * FROM graph_expand(tenant_id, seed, 2);   -- (entity_id BIGINT, depth INTEGER)
//   SELECT * FROM anatid_csr_stats();                 -- what is in memory right now
//   SELECT * FROM anatid_csr_tenants();               -- per-tenant detail
//
// Measured in the Phase 0 spike at 1,000,000 memories / 2.3M edges / 10 tenants (see
// docs/extension.md): 2-hop recall p50 2.04 ms through this extension, 2.88 ms through the
// equivalent pure SQL, 7.35 ms through LadybugDB 0.20.2.  All three returned identical id lists.
//
// The CSR is a SNAPSHOT.  It is rebuilt only by calling anatid_build_csr() again.  It lives in the
// DatabaseInstance's ObjectCache (non-evictable), so it is shared by every connection and survives
// across queries.  Replacing it swaps a shared_ptr; readers that already hold the old snapshot keep
// using it, so a rebuild during concurrent reads is safe.  A failed rebuild leaves the previous
// snapshot in place (the new one is published only after it is fully built).

#include "anatid_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/parser/keyword_helper.hpp"
#include "duckdb/storage/object_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <unordered_map>

namespace duckdb {

//===--------------------------------------------------------------------===//
// limits
//===--------------------------------------------------------------------===//

//! Default maximum BFS depth accepted by graph_expand.  30 is Kuzu's default recursive-join upper
//! bound; anything deeper on a connected graph is a full traversal wearing a hop count.  Raise it
//! for one query with `max_hops := N`.
static constexpr int32_t ANATID_DEFAULT_MAX_HOPS = 30;

//! Absolute ceiling on `max_hops` itself, so a typo cannot ask for a multi-hour traversal.
static constexpr int32_t ANATID_MAX_MAX_HOPS = 1000000;

//! Default ceiling on the bytes anatid_build_csr will allocate for the whole snapshot (1 GiB).
//! Raise it for one build with `max_bytes := N`; 0 disables the check.
static constexpr int64_t ANATID_DEFAULT_MAX_BYTES = 1LL << 30;

//! Hard ceiling on one tenant's dense vertex-id span (entries in its offsets array).  The CSR
//! indexes by (entity_id - min_id), so sparse ids -- e.g. anatid's 63-bit time-ordered ids -- do
//! not fit.  This is the overflow guard; the density check below normally fires long before it.
static constexpr uint64_t ANATID_MAX_TENANT_SPAN = 1ULL << 31;

//! Density guard.  A tenant's id span may exceed its edge count by at most this factor before the
//! build is refused: the offsets array costs 8 bytes per id in [min_id, max_id] whether or not an
//! entity has that id, so sparse ids buy a huge array for a tiny graph.  Two entities minted 10 ms
//! apart by anatid's default allocator are ~63,000,000 ids apart -- half a gigabyte of offsets for
//! one edge.  Override with `max_span_factor := N`; 0 disables the check.
static constexpr int64_t ANATID_DEFAULT_MAX_SPAN_FACTOR = 128;
static constexpr int64_t ANATID_MAX_MAX_SPAN_FACTOR = 1000000;

//! ... but always allow a small span, so a 1-edge tenant with ids 0..999 is not refused.
static constexpr uint64_t ANATID_MIN_SPAN_ALLOWANCE = 4096;

static const char *const ANATID_EXT_VERSION = "0.1.0";

//===--------------------------------------------------------------------===//
// anatid_version
//===--------------------------------------------------------------------===//
static string VersionBanner() {
	return string("anatid ") + ANATID_EXT_VERSION + " (DuckDB " + DuckDB::LibraryVersion() + ")";
}

static void AnatidVersionNoArgFun(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::GetData<string_t>(result)[0] = StringVector::AddString(result, VersionBanner());
}

// Kept because the Phase 0 benchmark harness calls anatid_version('') and splits on ':'.
static void AnatidVersionEchoFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vector = args.data[0];
	auto banner = VersionBanner();
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, banner + ": " + name.GetString());
	});
}

//===--------------------------------------------------------------------===//
// CSR
//===--------------------------------------------------------------------===//
// One tenant's undirected adjacency, keyed by entity id.  Vertex ids are mapped to a dense range
// [min_id, max_id]: offsets has (max_id - min_id + 2) entries, so the ids of a tenant must not be
// sparse.  anatid's default 63-bit time-ordered ids ARE sparse -- the extension is usable only
// with caller-supplied dense entity ids.  BuildCsr says so, by name, when the span is too wide.
struct TenantCsr {
	int64_t min_id = 0;
	int64_t max_id = -1;
	vector<uint64_t> offsets;   // offsets[v - min_id] .. offsets[v - min_id + 1] index into neighbours
	vector<int64_t> neighbours; // both directions of every current edge
	idx_t vertex_count = 0;     // ids with degree > 0
	idx_t edge_count = 0;       // edge rows read for this tenant

	bool Contains(int64_t id) const {
		return id >= min_id && id <= max_id;
	}
	//! Entries in the offsets array.  Only valid once min_id/max_id are set.
	uint64_t Span() const {
		return max_id < min_id ? 0 : static_cast<uint64_t>(max_id - min_id) + 1;
	}
	idx_t Bytes() const {
		return offsets.capacity() * sizeof(uint64_t) + neighbours.capacity() * sizeof(int64_t);
	}
};

class AnatidCsr : public ObjectCacheEntry {
public:
	static string ObjectType() {
		return "anatid_csr";
	}
	string GetObjectType() override {
		return ObjectType();
	}
	// Invalid index -> never evicted by the object cache.
	optional_idx GetEstimatedCacheMemory() const override {
		return optional_idx();
	}

	string source_table;
	string filter_note; // which current-state conjuncts were applied
	unordered_map<int64_t, TenantCsr> tenants;
	idx_t n_vertices = 0;
	idx_t n_edges = 0; // current edge rows read (each contributes two neighbour entries)
	idx_t n_bytes = 0;
	double build_ms = 0.0;
	int64_t built_at_micros = 0; // epoch microseconds, UTC

	//! Tenant ids present, ascending.  Used for error messages and anatid_csr_tenants().
	vector<int64_t> SortedTenants() const {
		vector<int64_t> out;
		out.reserve(tenants.size());
		for (auto &kv : tenants) {
			out.push_back(kv.first);
		}
		std::sort(out.begin(), out.end());
		return out;
	}

	//! "0, 1, 2" (at most `limit` of them, then ", ...").
	string TenantList(idx_t limit = 8) const {
		auto ids = SortedTenants();
		string out;
		for (idx_t i = 0; i < ids.size() && i < limit; i++) {
			if (i > 0) {
				out += ", ";
			}
			out += std::to_string(ids[i]);
		}
		if (ids.size() > limit) {
			out += ", ...";
		}
		return out.empty() ? string("none") : out;
	}

	string Describe() const {
		return "CSR built from '" + source_table + "', " + std::to_string(tenants.size()) +
		       " tenant(s) [" + TenantList() + "], " + std::to_string(n_edges) + " edge(s)";
	}
};

static const char *const ANATID_CSR_KEY = "anatid_csr";

static shared_ptr<AnatidCsr> GetCsr(ClientContext &context) {
	auto &cache = ObjectCache::GetObjectCache(context);
	return cache.GetWithTypePrefix<AnatidCsr>(ANATID_CSR_KEY);
}

//! The CSR or a clear error naming the function the caller must run first.
static shared_ptr<AnatidCsr> RequireCsr(ClientContext &context, const char *fn) {
	auto csr = GetCsr(context);
	if (!csr) {
		throw BinderException("%s: no CSR has been built for this database yet. Run "
		                      "SELECT * FROM anatid_build_csr('edges_relates') first "
		                      "(SELECT * FROM anatid_csr_stats() reports what is currently loaded).",
		                      fn);
	}
	return csr;
}

static string QuoteQualifiedName(const string &name) {
	// Accept "table", "schema.table" or "catalog.schema.table"; quote every part so odd names work.
	auto parts = StringUtil::Split(name, '.');
	if (parts.empty() || parts.size() > 3) {
		throw InvalidInputException("anatid_build_csr: invalid table name '%s' (expected "
		                            "\"table\", \"schema.table\" or \"catalog.schema.table\")",
		                            name);
	}
	string out;
	for (idx_t i = 0; i < parts.size(); i++) {
		if (parts[i].empty()) {
			throw InvalidInputException("anatid_build_csr: invalid table name '%s' (empty name part)", name);
		}
		if (i > 0) {
			out += ".";
		}
		out += KeywordHelper::WriteQuoted(parts[i], '"');
	}
	return out;
}

static bool HasColumn(const vector<string> &names, const char *want) {
	for (auto &n : names) {
		if (StringUtil::CIEquals(n, want)) {
			return true;
		}
	}
	return false;
}

// Reads the current same-tenant edges of <table> through a nested connection on the same
// DatabaseInstance and builds one undirected CSR per tenant.
//
// "Current" means every temporal conjunct anatid's relates_undirected view applies that this table
// actually has a column for: valid_to IS NULL AND tx_to IS NULL.  The Phase 0 spike's macro used
// only `valid_to IS NULL`; adding `tx_to IS NULL` is the bitemporal model anatid documents, and it
// provably cannot change the benchmark answers because tx_to is NULL on 100% of the spike dataset.
// A table with neither column is treated as all-current, which is what a plain edge list means.
static shared_ptr<AnatidCsr> BuildCsr(ClientContext &context, const string &table_name, int64_t max_bytes,
                                      int64_t max_span_factor) {
	auto t_start = std::chrono::steady_clock::now();
	auto csr = make_shared_ptr<AnatidCsr>();
	csr->source_table = table_name;

	auto &db = DatabaseInstance::GetDatabase(context);
	Connection con(db);
	auto qualified = QuoteQualifiedName(table_name);

	// 0. probe the columns, so a missing table or a wrong table gets a precise error rather than a
	//    binder error about a column the caller never mentioned.
	auto probe = con.Query("SELECT * FROM " + qualified + " LIMIT 0");
	if (probe->HasError()) {
		throw InvalidInputException("anatid_build_csr: cannot read edge table '%s': %s", table_name,
		                            probe->GetError());
	}
	auto columns = probe->names;
	for (auto required : {"tenant_id", "src", "dst"}) {
		if (!HasColumn(columns, required)) {
			throw InvalidInputException("anatid_build_csr: edge table '%s' has no column '%s' "
			                            "(needs tenant_id, src, dst; found: %s)",
			                            table_name, required, StringUtil::Join(columns, ", "));
		}
	}
	string where = "src IS NOT NULL AND dst IS NOT NULL";
	vector<string> applied;
	if (HasColumn(columns, "valid_to")) {
		where += " AND valid_to IS NULL";
		applied.push_back("valid_to IS NULL");
	}
	if (HasColumn(columns, "tx_to")) {
		where += " AND tx_to IS NULL";
		applied.push_back("tx_to IS NULL");
	}
	csr->filter_note = applied.empty() ? string("all rows (no valid_to/tx_to column)")
	                                   : StringUtil::Join(applied, " AND ");

	string sql = "SELECT CAST(tenant_id AS BIGINT), CAST(src AS BIGINT), CAST(dst AS BIGINT) FROM " + qualified +
	             " WHERE " + where;
	auto result = con.Query(sql);
	if (result->HasError()) {
		throw InvalidInputException("anatid_build_csr: cannot read edge table '%s' (needs columns tenant_id, src, "
		                            "dst castable to BIGINT): %s",
		                            table_name, result->GetError());
	}

	// 1. materialize the edge list
	vector<int64_t> ten, src, dst;
	auto row_count = result->RowCount();
	ten.reserve(row_count);
	src.reserve(row_count);
	dst.reserve(row_count);
	auto &collection = result->Collection();
	for (auto &chunk : collection.Chunks()) {
		chunk.Flatten();
		auto t_data = FlatVector::GetData<int64_t>(chunk.data[0]);
		auto s_data = FlatVector::GetData<int64_t>(chunk.data[1]);
		auto d_data = FlatVector::GetData<int64_t>(chunk.data[2]);
		auto &t_valid = FlatVector::Validity(chunk.data[0]);
		for (idx_t i = 0; i < chunk.size(); i++) {
			if (!t_valid.RowIsValid(i)) {
				continue; // edges without a tenant cannot be queried by tenant
			}
			ten.push_back(t_data[i]);
			src.push_back(s_data[i]);
			dst.push_back(d_data[i]);
		}
	}
	csr->n_edges = ten.size();

	// 2. per-tenant id ranges and edge counts
	for (idx_t i = 0; i < ten.size(); i++) {
		auto &tc = csr->tenants[ten[i]];
		auto lo = MinValue(src[i], dst[i]);
		auto hi = MaxValue(src[i], dst[i]);
		if (tc.max_id < tc.min_id) { // first edge of this tenant
			tc.min_id = lo;
			tc.max_id = hi;
		} else {
			tc.min_id = MinValue(tc.min_id, lo);
			tc.max_id = MaxValue(tc.max_id, hi);
		}
		tc.edge_count++;
	}

	// 3. allocate the offsets arrays, refusing spans that would blow up memory.  The span is
	//    computed in unsigned arithmetic: max_id - min_id can overflow int64 for ids near the
	//    extremes of the type, and an overflowed span would size the array wrongly.
	idx_t planned_bytes = 0;
	for (auto &kv : csr->tenants) {
		auto &tc = kv.second;
		// max_id - min_id in UNSIGNED arithmetic: the signed difference overflows for ids near the
		// ends of int64, and an overflowed span would size the offsets array wrongly.
		uint64_t reach = static_cast<uint64_t>(tc.max_id) - static_cast<uint64_t>(tc.min_id);
		if (reach >= ANATID_MAX_TENANT_SPAN) {
			throw InvalidInputException(
			    "anatid_build_csr: tenant %s spans entity ids %s..%s, more than %s apart; the dense "
			    "CSR offsets array cannot address that. The CSR indexes by (entity_id - min_id), so "
			    "entity ids must be dense per tenant -- anatid's default 63-bit time-ordered ids "
			    "are not. Supply dense ids (anatid.ids.set_allocator) or stay on the pure-SQL "
			    "expansion path.",
			    std::to_string(kv.first), std::to_string(tc.min_id), std::to_string(tc.max_id),
			    std::to_string(static_cast<uint64_t>(ANATID_MAX_TENANT_SPAN)));
		}
		uint64_t span = reach + 1; // ids in [min_id, max_id], one offsets entry each

		// Density: the offsets array costs 8 bytes per id in the range whether an entity has that
		// id or not, so ids sparse relative to the graph buy a huge array for a tiny graph.
		if (max_span_factor > 0) {
			uint64_t allowance = ANATID_MIN_SPAN_ALLOWANCE;
			uint64_t budget = static_cast<uint64_t>(max_span_factor) * static_cast<uint64_t>(tc.edge_count);
			if (budget > allowance) {
				allowance = budget;
			}
			if (span > allowance) {
				throw InvalidInputException(
				    "anatid_build_csr: tenant %s has %s current edge(s) but its entity ids span "
				    "%s..%s (%s ids), which would cost %s bytes of CSR offsets. The CSR is a dense "
				    "array indexed by (entity_id - min_id); ids this sparse are the shape anatid's "
				    "default 63-bit time-ordered ids have. Supply dense per-tenant entity ids "
				    "(anatid.ids.set_allocator), or pass max_span_factor := 0 to build it anyway.",
				    std::to_string(kv.first), std::to_string(tc.edge_count), std::to_string(tc.min_id),
				    std::to_string(tc.max_id), std::to_string(span),
				    std::to_string((span + 1) * sizeof(uint64_t)));
			}
		}

		// offsets: span + 1 entries of 8 bytes; neighbours: 2 per edge row of 8 bytes.
		planned_bytes += (span + 1) * sizeof(uint64_t) + tc.edge_count * 2 * sizeof(int64_t);
		if (max_bytes > 0 && static_cast<int64_t>(planned_bytes) > max_bytes) {
			throw InvalidInputException(
			    "anatid_build_csr: the CSR for '%s' would need at least %s bytes, over the limit of "
			    "%s (tenant %s spans entity ids %s..%s). Pass max_bytes := N to raise the limit, or "
			    "use dense per-tenant entity ids.",
			    table_name, std::to_string(planned_bytes), std::to_string(max_bytes),
			    std::to_string(kv.first), std::to_string(tc.min_id), std::to_string(tc.max_id));
		}
		tc.offsets.assign(span + 1, 0);
	}

	// 4. degrees (both directions)
	for (idx_t i = 0; i < ten.size(); i++) {
		auto &tc = csr->tenants[ten[i]];
		tc.offsets[static_cast<uint64_t>(src[i] - tc.min_id) + 1]++;
		tc.offsets[static_cast<uint64_t>(dst[i] - tc.min_id) + 1]++;
	}
	// 5. prefix sums + vertex counts
	for (auto &kv : csr->tenants) {
		auto &tc = kv.second;
		for (idx_t v = 1; v < tc.offsets.size(); v++) {
			if (tc.offsets[v] > 0) {
				tc.vertex_count++;
			}
			tc.offsets[v] += tc.offsets[v - 1];
		}
		tc.neighbours.resize(tc.offsets.back());
		csr->n_vertices += tc.vertex_count;
	}
	// 6. fill
	unordered_map<int64_t, vector<uint64_t>> cursors;
	for (auto &kv : csr->tenants) {
		auto &tc = kv.second;
		cursors[kv.first].assign(tc.offsets.begin(), tc.offsets.end() - 1);
	}
	for (idx_t i = 0; i < ten.size(); i++) {
		auto &tc = csr->tenants[ten[i]];
		auto &cur = cursors[ten[i]];
		tc.neighbours[cur[static_cast<uint64_t>(src[i] - tc.min_id)]++] = dst[i];
		tc.neighbours[cur[static_cast<uint64_t>(dst[i] - tc.min_id)]++] = src[i];
	}

	// 7. accounting
	csr->n_bytes = sizeof(AnatidCsr);
	for (auto &kv : csr->tenants) {
		csr->n_bytes += sizeof(std::pair<const int64_t, TenantCsr>) + kv.second.Bytes();
	}
	auto t_end = std::chrono::steady_clock::now();
	csr->build_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
	csr->built_at_micros = std::chrono::duration_cast<std::chrono::microseconds>(
	                           std::chrono::system_clock::now().time_since_epoch())
	                           .count();
	return csr;
}

// BFS from `seed` over the tenant's CSR, at most `hops` hops, undirected, visited set = bitmap over
// the tenant's id range.  Appends (id, depth) pairs; the seed is always emitted at depth 0.
//
// Semantics on a miss (tenant not in the CSR, or seed with no current edge): the seed alone, which
// is exactly what anatid's pure-SQL fallback returns, so the two paths stay interchangeable.  Pass
// strict=true to turn a miss into an error instead.
static void ExpandBfs(const AnatidCsr &csr, int64_t tenant_id, int64_t seed, int32_t hops, bool strict,
                      vector<int64_t> &ids, vector<int32_t> &depths) {
	ids.push_back(seed);
	depths.push_back(0);
	auto it = csr.tenants.find(tenant_id);
	if (it == csr.tenants.end()) {
		if (strict) {
			throw InvalidInputException("graph_expand: tenant %s has no edges in the CSR (%s). "
			                            "strict := false (the default) returns just the seed. If "
			                            "RELATES_TO rows were written since the build, rebuild with "
			                            "SELECT * FROM anatid_build_csr('%s').",
			                            std::to_string(tenant_id), csr.Describe(), csr.source_table);
		}
		return;
	}
	auto &tc = it->second;
	if (!tc.Contains(seed)) {
		if (strict) {
			throw InvalidInputException("graph_expand: seed %s has no current edge in tenant %s "
			                            "(whose CSR covers entity ids %s..%s). strict := false (the "
			                            "default) returns just the seed.",
			                            std::to_string(seed), std::to_string(tenant_id),
			                            std::to_string(tc.min_id), std::to_string(tc.max_id));
		}
		return; // seed has no current edges in this tenant
	}
	if (hops <= 0) {
		return;
	}
	auto span = tc.Span();
	// Structural invariant of a built CSR.  Checked (not asserted) because a corrupt snapshot must
	// raise, never index out of bounds: this loop reads offsets[v] and offsets[v + 1].
	if (tc.offsets.size() != span + 1) {
		throw InternalException("graph_expand: tenant %s has a malformed CSR (offsets %s, span %s)",
		                        std::to_string(tenant_id), std::to_string(tc.offsets.size()),
		                        std::to_string(span));
	}
	vector<uint8_t> visited(span, 0);
	visited[static_cast<uint64_t>(seed - tc.min_id)] = 1;
	idx_t frontier_begin = 0;
	idx_t frontier_end = 1;
	for (int32_t depth = 1; depth <= hops; depth++) {
		for (idx_t f = frontier_begin; f < frontier_end; f++) {
			auto v = static_cast<uint64_t>(ids[f] - tc.min_id);
			auto begin = tc.offsets[v];
			auto end = tc.offsets[v + 1];
			if (end > tc.neighbours.size() || begin > end) {
				throw InternalException("graph_expand: tenant %s has a malformed CSR (offsets "
				                        "%s..%s into %s neighbours)",
				                        std::to_string(tenant_id), std::to_string(begin), std::to_string(end),
				                        std::to_string(tc.neighbours.size()));
			}
			for (auto e = begin; e < end; e++) {
				auto nb = tc.neighbours[e];
				auto slot = static_cast<uint64_t>(nb - tc.min_id);
				if (slot >= span) {
					throw InternalException("graph_expand: tenant %s has a neighbour %s outside "
					                        "its id range %s..%s",
					                        std::to_string(tenant_id), std::to_string(nb),
					                        std::to_string(tc.min_id), std::to_string(tc.max_id));
				}
				if (!visited[slot]) {
					visited[slot] = 1;
					ids.push_back(nb);
					depths.push_back(depth);
				}
			}
		}
		frontier_begin = frontier_end;
		frontier_end = ids.size();
		if (frontier_begin == frontier_end) {
			break;
		}
	}
}

//===--------------------------------------------------------------------===//
// named-parameter helpers
//===--------------------------------------------------------------------===//
static bool NamedBool(TableFunctionBindInput &input, const char *fn, const char *name, bool dflt) {
	auto entry = input.named_parameters.find(name);
	if (entry == input.named_parameters.end()) {
		return dflt;
	}
	if (entry->second.IsNull()) {
		throw BinderException("%s: %s must not be NULL", fn, name);
	}
	return BooleanValue::Get(entry->second);
}

static int64_t NamedInt(TableFunctionBindInput &input, const char *fn, const char *name, int64_t dflt) {
	auto entry = input.named_parameters.find(name);
	if (entry == input.named_parameters.end()) {
		return dflt;
	}
	if (entry->second.IsNull()) {
		throw BinderException("%s: %s must not be NULL", fn, name);
	}
	return entry->second.GetValue<int64_t>();
}

//===--------------------------------------------------------------------===//
// anatid_build_csr(edge_table VARCHAR [, max_bytes := BIGINT])
//     -> (tenants, vertices, edges, build_ms)
//===--------------------------------------------------------------------===//
struct BuildCsrBindData : public TableFunctionData {
	int64_t tenants = 0;
	int64_t vertices = 0;
	int64_t edges = 0;
	double build_ms = 0.0;
};

struct BuildCsrGlobalState : public GlobalTableFunctionState {
	bool done = false;
};

// The build happens at bind time: bind runs synchronously on the client thread inside the statement's
// own transaction, which is the simplest place to run a nested query and publish the snapshot.
static unique_ptr<FunctionData> BuildCsrBind(ClientContext &context, TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types, vector<string> &names) {
	names = {"tenants", "vertices", "edges", "build_ms"};
	return_types = {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::DOUBLE};
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw BinderException("anatid_build_csr: the edge table name must not be NULL");
	}
	auto table_name = input.inputs[0].GetValue<string>();
	if (table_name.empty()) {
		throw BinderException("anatid_build_csr: the edge table name must not be empty");
	}
	auto max_bytes = NamedInt(input, "anatid_build_csr", "max_bytes", ANATID_DEFAULT_MAX_BYTES);
	if (max_bytes < 0) {
		throw BinderException("anatid_build_csr: max_bytes must be >= 0 (0 disables the limit), got %s",
		                      std::to_string(max_bytes));
	}
	auto max_span_factor =
	    NamedInt(input, "anatid_build_csr", "max_span_factor", ANATID_DEFAULT_MAX_SPAN_FACTOR);
	if (max_span_factor < 0 || max_span_factor > ANATID_MAX_MAX_SPAN_FACTOR) {
		throw BinderException("anatid_build_csr: max_span_factor must be between 0 (disabled) and %s, got %s",
		                      std::to_string(static_cast<int64_t>(ANATID_MAX_MAX_SPAN_FACTOR)),
		                      std::to_string(max_span_factor));
	}
	auto csr = BuildCsr(context, table_name, max_bytes, max_span_factor);
	auto result = make_uniq<BuildCsrBindData>();
	result->tenants = NumericCast<int64_t>(csr->tenants.size());
	result->vertices = NumericCast<int64_t>(csr->n_vertices);
	result->edges = NumericCast<int64_t>(csr->n_edges);
	result->build_ms = csr->build_ms;
	// Published only now: a throw anywhere above leaves the previous snapshot untouched.
	ObjectCache::GetObjectCache(context).PutWithTypePrefix<AnatidCsr>(ANATID_CSR_KEY, std::move(csr));
	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> BuildCsrInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<BuildCsrGlobalState>();
}

static void BuildCsrFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<BuildCsrBindData>();
	auto &state = data.global_state->Cast<BuildCsrGlobalState>();
	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	state.done = true;
	output.SetValue(0, 0, Value::BIGINT(bind_data.tenants));
	output.SetValue(1, 0, Value::BIGINT(bind_data.vertices));
	output.SetValue(2, 0, Value::BIGINT(bind_data.edges));
	output.SetValue(3, 0, Value::DOUBLE(bind_data.build_ms));
	output.SetCardinality(1);
}

//===--------------------------------------------------------------------===//
// graph_expand(tenant_id BIGINT, seed BIGINT, hops INTEGER
//              [, strict := BOOLEAN] [, max_hops := INTEGER])
//     -> (entity_id BIGINT, depth INTEGER)
//===--------------------------------------------------------------------===//
struct GraphExpandBindData : public TableFunctionData {
	int64_t tenant_id = 0;
	int64_t seed = 0;
	int32_t hops = 0;
	// The BFS runs at bind time so the optimizer sees the exact cardinality of the frontier (which
	// decides hash-join build sides in 2-hop recall). Arguments are constants, so this is a
	// per-statement cost.
	vector<int64_t> ids;
	vector<int32_t> depths;
};

struct GraphExpandGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
};

static unique_ptr<FunctionData> GraphExpandBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<string> &names) {
	names = {"entity_id", "depth"};
	return_types = {LogicalType::BIGINT, LogicalType::INTEGER};
	if (input.inputs.size() != 3) {
		throw BinderException("graph_expand(tenant_id BIGINT, seed BIGINT, hops INTEGER) takes exactly 3 "
		                      "positional arguments, got %s",
		                      std::to_string(input.inputs.size()));
	}
	for (auto &v : input.inputs) {
		if (v.IsNull()) {
			throw BinderException("graph_expand: arguments must not be NULL");
		}
	}
	auto result = make_uniq<GraphExpandBindData>();
	result->tenant_id = input.inputs[0].GetValue<int64_t>();
	result->seed = input.inputs[1].GetValue<int64_t>();
	result->hops = input.inputs[2].GetValue<int32_t>();

	auto strict = NamedBool(input, "graph_expand", "strict", false);
	auto max_hops_64 = NamedInt(input, "graph_expand", "max_hops", ANATID_DEFAULT_MAX_HOPS);
	if (max_hops_64 < 0 || max_hops_64 > ANATID_MAX_MAX_HOPS) {
		throw BinderException("graph_expand: max_hops must be between 0 and %d, got %s", ANATID_MAX_MAX_HOPS,
		                      std::to_string(max_hops_64));
	}
	auto max_hops = static_cast<int32_t>(max_hops_64);

	if (result->hops < 0) {
		throw BinderException("graph_expand: hops must be >= 0, got %d", result->hops);
	}
	if (result->hops > max_hops) {
		throw BinderException("graph_expand: hops=%d exceeds the limit of %d. anatid caps BFS depth at "
		                      "%d by default (the same bound Kuzu puts on a recursive join) because a "
		                      "deeper traversal of a connected graph is a full scan wearing a hop "
		                      "count. Pass max_hops := %d to allow it for this query.",
		                      result->hops, max_hops, ANATID_DEFAULT_MAX_HOPS, result->hops);
	}
	auto csr = RequireCsr(context, "graph_expand");
	ExpandBfs(*csr, result->tenant_id, result->seed, result->hops, strict, result->ids, result->depths);
	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> GraphExpandInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<GraphExpandGlobalState>();
}

static void GraphExpandFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<GraphExpandBindData>();
	auto &state = data.global_state->Cast<GraphExpandGlobalState>();
	auto total = bind_data.ids.size();
	if (state.offset >= total) {
		output.SetCardinality(0);
		return;
	}
	auto count = MinValue<idx_t>(total - state.offset, STANDARD_VECTOR_SIZE);
	auto ids = FlatVector::GetData<int64_t>(output.data[0]);
	auto depths = FlatVector::GetData<int32_t>(output.data[1]);
	memcpy(ids, bind_data.ids.data() + state.offset, count * sizeof(int64_t));
	memcpy(depths, bind_data.depths.data() + state.offset, count * sizeof(int32_t));
	state.offset += count;
	output.SetCardinality(count);
}

static unique_ptr<NodeStatistics> GraphExpandCardinality(ClientContext &context, const FunctionData *bind_data) {
	if (!bind_data) {
		return nullptr;
	}
	auto &data = bind_data->Cast<GraphExpandBindData>();
	return make_uniq<NodeStatistics>(data.ids.size(), data.ids.size());
}

//===--------------------------------------------------------------------===//
// anatid_csr_stats() -> (edge_table, tenants, vertices, edges, bytes, build_ms, built_at)
// Zero rows when no CSR has been built, so it doubles as a health probe.
//===--------------------------------------------------------------------===//
struct CsrStatsBindData : public TableFunctionData {
	bool present = false;
	string edge_table;
	string filter_note;
	int64_t tenants = 0;
	int64_t vertices = 0;
	int64_t edges = 0;
	int64_t bytes = 0;
	double build_ms = 0.0;
	int64_t built_at_micros = 0;
};

struct OneRowGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
};

static unique_ptr<FunctionData> CsrStatsBind(ClientContext &context, TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types, vector<string> &names) {
	names = {"edge_table", "current_filter", "tenants", "vertices", "edges", "bytes", "build_ms", "built_at"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT,
	                LogicalType::BIGINT,  LogicalType::BIGINT,  LogicalType::DOUBLE, LogicalType::TIMESTAMP};
	auto result = make_uniq<CsrStatsBindData>();
	auto csr = GetCsr(context);
	if (csr) {
		result->present = true;
		result->edge_table = csr->source_table;
		result->filter_note = csr->filter_note;
		result->tenants = NumericCast<int64_t>(csr->tenants.size());
		result->vertices = NumericCast<int64_t>(csr->n_vertices);
		result->edges = NumericCast<int64_t>(csr->n_edges);
		result->bytes = NumericCast<int64_t>(csr->n_bytes);
		result->build_ms = csr->build_ms;
		result->built_at_micros = csr->built_at_micros;
	}
	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> OneRowInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<OneRowGlobalState>();
}

static void CsrStatsFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<CsrStatsBindData>();
	auto &state = data.global_state->Cast<OneRowGlobalState>();
	if (!bind_data.present || state.offset > 0) {
		output.SetCardinality(0);
		return;
	}
	state.offset = 1;
	output.SetValue(0, 0, Value(bind_data.edge_table));
	output.SetValue(1, 0, Value(bind_data.filter_note));
	output.SetValue(2, 0, Value::BIGINT(bind_data.tenants));
	output.SetValue(3, 0, Value::BIGINT(bind_data.vertices));
	output.SetValue(4, 0, Value::BIGINT(bind_data.edges));
	output.SetValue(5, 0, Value::BIGINT(bind_data.bytes));
	output.SetValue(6, 0, Value::DOUBLE(bind_data.build_ms));
	output.SetValue(7, 0, Value::TIMESTAMP(Timestamp::FromEpochMicroSeconds(bind_data.built_at_micros)));
	output.SetCardinality(1);
}

//===--------------------------------------------------------------------===//
// anatid_csr_tenants() -> one row per tenant in the CSR
//===--------------------------------------------------------------------===//
struct CsrTenantRow {
	int64_t tenant_id;
	int64_t vertices;
	int64_t edges;
	int64_t min_id;
	int64_t max_id;
	int64_t bytes;
};

struct CsrTenantsBindData : public TableFunctionData {
	vector<CsrTenantRow> rows;
};

static unique_ptr<FunctionData> CsrTenantsBind(ClientContext &context, TableFunctionBindInput &input,
                                               vector<LogicalType> &return_types, vector<string> &names) {
	names = {"tenant_id", "vertices", "edges", "min_entity_id", "max_entity_id", "bytes"};
	return_types = {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT,
	                LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT};
	auto result = make_uniq<CsrTenantsBindData>();
	auto csr = GetCsr(context);
	if (csr) {
		for (auto tenant_id : csr->SortedTenants()) {
			auto &tc = csr->tenants.at(tenant_id);
			result->rows.push_back({tenant_id, NumericCast<int64_t>(tc.vertex_count),
			                        NumericCast<int64_t>(tc.edge_count), tc.min_id, tc.max_id,
			                        NumericCast<int64_t>(tc.Bytes())});
		}
	}
	return std::move(result);
}

static void CsrTenantsFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<CsrTenantsBindData>();
	auto &state = data.global_state->Cast<OneRowGlobalState>();
	auto total = bind_data.rows.size();
	if (state.offset >= total) {
		output.SetCardinality(0);
		return;
	}
	auto count = MinValue<idx_t>(total - state.offset, STANDARD_VECTOR_SIZE);
	for (idx_t i = 0; i < count; i++) {
		auto &row = bind_data.rows[state.offset + i];
		output.SetValue(0, i, Value::BIGINT(row.tenant_id));
		output.SetValue(1, i, Value::BIGINT(row.vertices));
		output.SetValue(2, i, Value::BIGINT(row.edges));
		output.SetValue(3, i, Value::BIGINT(row.min_id));
		output.SetValue(4, i, Value::BIGINT(row.max_id));
		output.SetValue(5, i, Value::BIGINT(row.bytes));
	}
	state.offset += count;
	output.SetCardinality(count);
}

//===--------------------------------------------------------------------===//
// registration
//===--------------------------------------------------------------------===//
static void LoadInternal(ExtensionLoader &loader) {
	ScalarFunctionSet version_set("anatid_version");
	version_set.AddFunction(ScalarFunction({}, LogicalType::VARCHAR, AnatidVersionNoArgFun));
	version_set.AddFunction(ScalarFunction({LogicalType::VARCHAR}, LogicalType::VARCHAR, AnatidVersionEchoFun));
	loader.RegisterFunction(version_set);

	TableFunction build_csr("anatid_build_csr", {LogicalType::VARCHAR}, BuildCsrFunction, BuildCsrBind, BuildCsrInit);
	build_csr.named_parameters["max_bytes"] = LogicalType::BIGINT;
	build_csr.named_parameters["max_span_factor"] = LogicalType::BIGINT;
	loader.RegisterFunction(build_csr);

	TableFunction graph_expand("graph_expand", {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::INTEGER},
	                           GraphExpandFunction, GraphExpandBind, GraphExpandInit);
	graph_expand.named_parameters["strict"] = LogicalType::BOOLEAN;
	graph_expand.named_parameters["max_hops"] = LogicalType::INTEGER;
	graph_expand.cardinality = GraphExpandCardinality;
	loader.RegisterFunction(graph_expand);

	TableFunction csr_stats("anatid_csr_stats", {}, CsrStatsFunction, CsrStatsBind, OneRowInit);
	loader.RegisterFunction(csr_stats);

	TableFunction csr_tenants("anatid_csr_tenants", {}, CsrTenantsFunction, CsrTenantsBind, OneRowInit);
	loader.RegisterFunction(csr_tenants);
}

void AnatidExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string AnatidExtension::Name() {
	return "anatid";
}
std::string AnatidExtension::Version() const {
	return ANATID_EXT_VERSION;
}

} // namespace duckdb

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(anatid, loader) {
	duckdb::LoadInternal(loader);
}
}
