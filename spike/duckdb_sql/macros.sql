-- anatid Phase 0 spike, engine "duckdb_sql" (Layer A): the operations of spike/SPEC.md as DuckDB SQL
-- macros over the tables of schema.sql. Executed by bench/run_duckdb_sql.py after load and after
-- the fts index exists (bm25_top reads the fts_main_memories.* tables that
-- `PRAGMA create_fts_index('memories', 'memory_id', 'content', ...)` creates).
--
-- Statements are separated by ';' and comment lines start with '--' (the runner strips comment
-- lines before splitting, so keep ';' out of comments).

-- ---------------------------------------------------------------------------------------------
-- Undirected view of the CURRENT RELATES_TO edges: one row per direction so a 1-hop expansion of
-- entity `a` is a plain equality lookup on `a` (no OR-join). valid_to IS NULL = current edge.
CREATE OR REPLACE VIEW relates_undirected AS
    SELECT tenant_id, src AS a, dst AS b FROM edges_relates WHERE valid_to IS NULL
    UNION ALL
    SELECT tenant_id, dst AS a, src AS b FROM edges_relates WHERE valid_to IS NULL;

-- ---------------------------------------------------------------------------------------------
-- graph_expand(tenant, seed, hops): the R1 frontier = seed + every entity reachable in 1..hops
-- undirected hops over current same-tenant RELATES_TO edges.
--
-- Two formulations (SPEC asks to try both, keep the faster, record both):
--
-- (a) graph_expand_rec: recursive CTE with USING KEY (DuckDB >= 1.3). The key makes the recurring
--     table a set of entities, depth is carried to stop after `hops` iterations.
-- (b) graph_expand: two explicit self-join levels (h1, h2). Rows are UNION ALL'ed, NOT
--     deduplicated: recall_2hop consumes the frontier through an IN (...) semi join, for which
--     duplicates are harmless, and each DISTINCT / UNION costs a HASH_GROUP_BY (~0.1 ms single
--     threaded, ~0.8 ms with 10 threads). `hops` must be 1 or 2 for this variant.
--
-- Measured at small scale (see the result JSON notes for the numbers of the actual run): (b) is
-- faster, so recall_2hop uses it. Both return the same entity set (checked against
-- common.reference_frontier / reference_r1 for 1,000 queries).
CREATE OR REPLACE MACRO graph_expand_rec(tenant, seed, hops) AS TABLE
    WITH RECURSIVE frontier USING KEY (entity_id) AS (
        SELECT seed::BIGINT AS entity_id, 0 AS depth
        UNION
        SELECT r.b AS entity_id, f.depth + 1 AS depth
        FROM frontier f
        JOIN relates_undirected r ON r.a = f.entity_id AND r.tenant_id = tenant
        WHERE f.depth < hops
    )
    SELECT entity_id FROM frontier;

CREATE OR REPLACE MACRO graph_expand(tenant, seed, hops) AS TABLE
    WITH h1 AS (
        SELECT r.b AS entity_id
        FROM relates_undirected r
        WHERE r.tenant_id = tenant AND r.a = seed
    ), h2 AS (
        SELECT r.b AS entity_id
        FROM h1
        JOIN relates_undirected r ON r.a = h1.entity_id
        WHERE r.tenant_id = tenant
    )
    SELECT seed::BIGINT AS entity_id
    UNION ALL SELECT entity_id FROM h1 WHERE hops >= 1
    UNION ALL SELECT entity_id FROM h2 WHERE hops >= 2;

-- Deduplicated frontier (one row per entity), for callers that want the set itself rather than
-- a semi-join input (e.g. comparing with common.reference_frontier).
CREATE OR REPLACE MACRO graph_frontier(tenant, seed, hops) AS TABLE
    SELECT DISTINCT entity_id FROM graph_expand(tenant, seed, hops);

-- ---------------------------------------------------------------------------------------------
-- R1 recall_2hop(tenant, seed, lim): memories ABOUT any frontier entity, current (valid_to IS NULL),
-- same tenant, newest first (created_at DESC, memory_id DESC), first `lim`.
-- The frontier enters through IN (...) so the planner builds SEMI joins: frontier -> edges_about
-- (dst) -> memories (memory_id). Both scans are filtered by tenant_id (zone maps prune the other
-- tenants because schema.sql clusters the tables by tenant on load) and the TOP_N pushes a dynamic
-- created_at threshold into the memories scan.
CREATE OR REPLACE MACRO recall_2hop(tenant, seed, lim) AS TABLE
    SELECT m.memory_id, m.created_at
    FROM memories m
    WHERE m.tenant_id = tenant
      AND m.valid_to IS NULL
      AND m.memory_id IN (
            SELECT a.src
            FROM edges_about a
            WHERE a.tenant_id = tenant
              AND a.dst IN (SELECT entity_id FROM graph_expand(tenant, seed, 2)))
    ORDER BY m.created_at DESC, m.memory_id DESC
    LIMIT lim;

-- Same query over the recursive-CTE frontier (the runner times both and records the numbers).
CREATE OR REPLACE MACRO recall_2hop_rec(tenant, seed, lim) AS TABLE
    SELECT m.memory_id, m.created_at
    FROM memories m
    WHERE m.tenant_id = tenant
      AND m.valid_to IS NULL
      AND m.memory_id IN (
            SELECT a.src
            FROM edges_about a
            WHERE a.tenant_id = tenant
              AND a.dst IN (SELECT entity_id FROM graph_expand_rec(tenant, seed, 2)))
    ORDER BY m.created_at DESC, m.memory_id DESC
    LIMIT lim;

-- ---------------------------------------------------------------------------------------------
-- R2 building blocks.
--
-- cosine_top: brute-force cosine similarity over the tenant's current memories (no ANN index).
-- query_embedding must be a FLOAT[64] (callers pass '[...]'::FLOAT[64]). Ties: memory_id ASC.
CREATE OR REPLACE MACRO cosine_top(tenant, query_embedding, topn) AS TABLE
    SELECT memory_id, array_cosine_similarity(embedding, query_embedding) AS score
    FROM memories
    WHERE tenant_id = tenant AND valid_to IS NULL
    ORDER BY score DESC, memory_id ASC
    LIMIT topn;

-- bm25_top: Okapi BM25 (k1 = 1.2, b = 0.75) over the fts extension's inverted index, read directly
-- from its tables (fts_main_memories.dict / terms / docs / stats). The formula is the one the
-- extension's match_bm25 macro uses (idf = ln((N - df + 0.5) / (df + 0.5) + 1), corpus statistics
-- over every indexed row), and it returns exactly the same ranking as match_bm25 and as
-- common.reference_r2_truth. It is ~2x faster than match_bm25 because it skips the extension's
-- per-document correlated lookup. Query tokens: lower-cased, split on whitespace (same as the
-- tokenizer used at index time: ignore='(\.|[^a-z])+', lower=1, stemmer='none', stopwords='none').
-- Only memories matching at least one query term score. Ties: memory_id ASC.
-- NOTE: the fts index is NOT incremental. Memories inserted after `PRAGMA create_fts_index` are
-- invisible to bm25_top until the index is rebuilt (overwrite=1). Superseded memories are still
-- in the index but are filtered out by memories.valid_to IS NULL here.
CREATE OR REPLACE MACRO bm25_top(tenant, query_text, topn) AS TABLE
    WITH q AS (
        SELECT DISTINCT term
        FROM (SELECT unnest(string_split_regex(lower(query_text), '\s+')) AS term)
        WHERE term <> ''
    ), qt AS (
        SELECT d.termid, d.df
        FROM fts_main_memories.dict d JOIN q ON d.term = q.term
    ), tf AS (
        SELECT t.docid, t.termid, count(*)::DOUBLE AS tf
        FROM fts_main_memories.terms t JOIN qt ON t.termid = qt.termid
        GROUP BY t.docid, t.termid
    ), sc AS (
        SELECT tf.docid,
               sum(ln((s.num_docs - qt.df + 0.5) / (qt.df + 0.5) + 1)
                   * tf.tf * (1.2 + 1) / (tf.tf + 1.2 * (1 - 0.75 + 0.75 * d.len / s.avgdl))) AS score
        FROM tf
        JOIN qt ON tf.termid = qt.termid
        JOIN fts_main_memories.docs d ON d.docid = tf.docid
        CROSS JOIN fts_main_memories.stats s
        GROUP BY tf.docid
    )
    SELECT m.memory_id, sc.score
    FROM sc
    JOIN fts_main_memories.docs d ON d.docid = sc.docid
    JOIN memories m ON m.memory_id = d.name
    WHERE m.tenant_id = tenant AND m.valid_to IS NULL
    ORDER BY sc.score DESC, m.memory_id ASC
    LIMIT topn;

-- bm25_top_match: the same list via the extension's own match_bm25 macro (kept for the A/B
-- measurement recorded by the runner).
CREATE OR REPLACE MACRO bm25_top_match(tenant, query_text, topn) AS TABLE
    SELECT memory_id, score
    FROM (
        SELECT memory_id,
               fts_main_memories.match_bm25(memory_id, query_text, k := 1.2, b := 0.75) AS score
        FROM memories
        WHERE tenant_id = tenant AND valid_to IS NULL
    )
    WHERE score IS NOT NULL
    ORDER BY score DESC, memory_id ASC
    LIMIT topn;

-- ---------------------------------------------------------------------------------------------
-- R2 recall_hybrid(tenant, query_text, query_embedding, k): RRF (k = 60, 1-based ranks) of the
-- cosine top-50 and the BM25 top-50, top `k` by (rrf_score DESC, memory_id ASC), each with the
-- names of its ABOUT entities (1-hop context).
CREATE OR REPLACE MACRO recall_hybrid(tenant, query_text, query_embedding, k) AS TABLE
    WITH cos AS (
        SELECT memory_id, row_number() OVER (ORDER BY score DESC, memory_id ASC) AS rnk
        FROM cosine_top(tenant, query_embedding, 50)
    ), bm AS (
        SELECT memory_id, row_number() OVER (ORDER BY score DESC, memory_id ASC) AS rnk
        FROM bm25_top(tenant, query_text, 50)
    ), fused AS (
        SELECT memory_id, sum(1.0 / (60 + rnk)) AS rrf_score
        FROM (SELECT * FROM cos UNION ALL SELECT * FROM bm)
        GROUP BY memory_id
    ), top AS (
        SELECT memory_id, rrf_score
        FROM fused
        ORDER BY rrf_score DESC, memory_id ASC
        LIMIT k
    )
    SELECT t.memory_id,
           t.rrf_score,
           (SELECT list(e.name ORDER BY e.entity_id)
            FROM edges_about a JOIN entities e ON e.entity_id = a.dst
            WHERE a.src = t.memory_id) AS about_names
    FROM top t
    ORDER BY t.rrf_score DESC, t.memory_id ASC;

-- Same, with the BM25 list coming from match_bm25 (A/B only).
CREATE OR REPLACE MACRO recall_hybrid_match(tenant, query_text, query_embedding, k) AS TABLE
    WITH cos AS (
        SELECT memory_id, row_number() OVER (ORDER BY score DESC, memory_id ASC) AS rnk
        FROM cosine_top(tenant, query_embedding, 50)
    ), bm AS (
        SELECT memory_id, row_number() OVER (ORDER BY score DESC, memory_id ASC) AS rnk
        FROM bm25_top_match(tenant, query_text, 50)
    ), fused AS (
        SELECT memory_id, sum(1.0 / (60 + rnk)) AS rrf_score
        FROM (SELECT * FROM cos UNION ALL SELECT * FROM bm)
        GROUP BY memory_id
    ), top AS (
        SELECT memory_id, rrf_score
        FROM fused
        ORDER BY rrf_score DESC, memory_id ASC
        LIMIT k
    )
    SELECT t.memory_id,
           t.rrf_score,
           (SELECT list(e.name ORDER BY e.entity_id)
            FROM edges_about a JOIN entities e ON e.entity_id = a.dst
            WHERE a.src = t.memory_id) AS about_names
    FROM top t
    ORDER BY t.rrf_score DESC, t.memory_id ASC;

-- ---------------------------------------------------------------------------------------------
-- W1 remember / W2 supersede.
--
-- DEVIATION from the SPEC layout line "remember, supersede as macros": DuckDB macros are
-- SELECT-only (no INSERT/UPDATE inside CREATE MACRO), so the two write operations are
-- parameterized statements that bench/run_duckdb_sql.py executes inside an explicit
-- BEGIN ... COMMIT. They are reproduced here verbatim so this file documents every operation.
-- Parameters ($n) are bound by the runner. The embedding is bound as the text '[f1, ..., f64]'
-- and cast to FLOAT[64] in SQL (binding a 64-element Python list costs ~4.5 ms in duckdb-python).
--
-- W1 remember(memory_row, about_dsts):
--   BEGIN;
--   INSERT INTO memories VALUES ($memory_id, $tenant_id, $content, $kind, $embedding::FLOAT[64],
--                                $created_at, $valid_from, NULL, $tx_from, NULL,
--                                $writer, $episode_id, $confidence);
--   INSERT INTO edges_about
--       SELECT eid, $memory_id, dst, $tenant_id, 1.0, $now, NULL, $now, NULL
--       FROM (SELECT unnest($about_edge_ids::BIGINT[]) AS eid, unnest($about_dsts::BIGINT[]) AS dst);
--   COMMIT;
--
-- W2 supersede(old_memory_id, new_memory_row):
--   BEGIN;
--   INSERT INTO memories VALUES (... the new row, as in W1 ...);
--   UPDATE memories SET valid_to = $now WHERE memory_id = $old_memory_id;
--   INSERT INTO edges_supersedes VALUES ($supersedes_edge_id, $new_memory_id, $old_memory_id, $tenant_id, $now);
--   COMMIT;
