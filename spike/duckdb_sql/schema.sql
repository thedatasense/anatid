-- anatid Phase 0 spike, engine "duckdb_sql" (Layer A): plain DuckDB 1.5.5 tables, no C++.
-- Contract: spike/SPEC.md ("Dataset" and "Engine notes / duckdb_sql").
--
-- Executed by bench/run_duckdb_sql.py, which
--   * replaces the token @DATA_DIR@ with spike/data/<scale> before running,
--   * runs every statement in order, EXCEPT the `CREATE INDEX` statements, which it runs (or skips)
--     individually so R1 and the W2 point update can be measured with and without each index
--     (see the "INDEXES" section at the bottom, the runner picks by index name),
--   * with --no-cluster drops the lines tagged `-- @cluster` (the ORDER BY of the bulk INSERTs) so
--     the tables are loaded in Parquet order instead of clustered by tenant.
--
-- Columns are exactly the SPEC columns including the system (bitemporal) columns:
--   valid_from / valid_to  = when the fact was true (valid_to NULL = current)
--   tx_from / tx_to        = when the row was recorded / retired (tx_to NULL = live)
-- No PRIMARY KEY / UNIQUE constraints: in DuckDB those create ART indexes implicitly, and the
-- spike wants indexes to be an explicit, measured choice.
--
-- Physical layout (measured at small scale, single thread, R1 p50): loading memories, edges_about
-- and edges_relates clustered by tenant lets DuckDB's zone maps skip the other nine tenants on
-- every R1/R2 scan (DuckDB does not use ART indexes for joins, so the scans are what R1 costs):
-- Parquet order 2.03 ms, ORDER BY tenant_id 1.28 ms, ORDER BY tenant_id, created_at DESC 1.17 ms
-- (the created_at order also feeds the TOP_N dynamic filter). The cosine scan of R2 goes from
-- 8.1 ms to 1.1 ms. The ORDER BY costs load time only. Rows written later (W1/W2) append in
-- arrival order, which is fine: they are few and DuckDB filters them per row group as usual.

CREATE TABLE entities (
    entity_id   BIGINT   NOT NULL,
    tenant_id   INTEGER  NOT NULL,
    kind        VARCHAR,
    name        VARCHAR
);

CREATE TABLE memories (
    memory_id   BIGINT    NOT NULL,
    tenant_id   INTEGER   NOT NULL,
    content     VARCHAR,
    kind        VARCHAR,
    embedding   FLOAT[64],
    created_at  TIMESTAMP NOT NULL,
    valid_from  TIMESTAMP,
    valid_to    TIMESTAMP,            -- NULL = current
    tx_from     TIMESTAMP,
    tx_to       TIMESTAMP,
    writer      VARCHAR,
    episode_id  BIGINT,
    confidence  FLOAT
);

-- memory -[ABOUT]-> entity
CREATE TABLE edges_about (
    edge_id     BIGINT   NOT NULL,
    src         BIGINT   NOT NULL,    -- memory_id
    dst         BIGINT   NOT NULL,    -- entity_id
    tenant_id   INTEGER  NOT NULL,
    weight      FLOAT,
    valid_from  TIMESTAMP,
    valid_to    TIMESTAMP,
    tx_from     TIMESTAMP,
    tx_to       TIMESTAMP
);

-- entity -[RELATES_TO]-> entity (traversed in both directions by R1)
CREATE TABLE edges_relates (
    edge_id     BIGINT   NOT NULL,
    src         BIGINT   NOT NULL,    -- entity_id
    dst         BIGINT   NOT NULL,    -- entity_id
    tenant_id   INTEGER  NOT NULL,
    rel_kind    VARCHAR,
    valid_from  TIMESTAMP,
    valid_to    TIMESTAMP,            -- NULL = current
    tx_from     TIMESTAMP,
    tx_to       TIMESTAMP
);

-- newer memory -[SUPERSEDES]-> older memory
CREATE TABLE edges_supersedes (
    edge_id     BIGINT   NOT NULL,
    src         BIGINT   NOT NULL,    -- newer memory_id
    dst         BIGINT   NOT NULL,    -- older memory_id
    tenant_id   INTEGER  NOT NULL,
    tx_from     TIMESTAMP
);

-- LOAD (bulk, from the generated Parquet files). Parquet stores the embedding as a plain
-- list<float>; cast to the fixed-size FLOAT[64] array that array_cosine_similarity needs.
INSERT INTO entities
    SELECT entity_id, tenant_id, kind, name
    FROM read_parquet('@DATA_DIR@/entities.parquet');

INSERT INTO memories
    SELECT memory_id, tenant_id, content, kind, embedding::FLOAT[64],
           created_at, valid_from, valid_to, tx_from, tx_to, writer, episode_id, confidence
    FROM read_parquet('@DATA_DIR@/memories.parquet')
    ORDER BY tenant_id, created_at DESC, memory_id DESC;  -- @cluster

INSERT INTO edges_about
    SELECT edge_id, src, dst, tenant_id, weight, valid_from, valid_to, tx_from, tx_to
    FROM read_parquet('@DATA_DIR@/edges_about.parquet')
    ORDER BY tenant_id, dst, src;  -- @cluster

INSERT INTO edges_relates
    SELECT edge_id, src, dst, tenant_id, rel_kind, valid_from, valid_to, tx_from, tx_to
    FROM read_parquet('@DATA_DIR@/edges_relates.parquet')
    ORDER BY tenant_id, src;  -- @cluster

INSERT INTO edges_supersedes
    SELECT edge_id, src, dst, tenant_id, tx_from
    FROM read_parquet('@DATA_DIR@/edges_supersedes.parquet');

-- INDEXES (ART). The runner creates only the ones named by --indexes (default: idx_memories_id)
-- and measures R1 and the W2 point UPDATE with and without them, recording the numbers in the
-- result JSON notes. Measured at small scale: none of the four changes R1 (DuckDB's planner uses
-- ART only for constant-predicate point lookups / range filters and never for joins, and the R1
-- frontier and ABOUT lookups are joins over tenant-pruned scans), idx_memories_id makes the
-- W2 `UPDATE memories SET valid_to WHERE memory_id = ?` a point lookup instead of a scan of
-- memory_id (its benefit grows with table size), and every index adds insert cost and file size,
-- so only idx_memories_id is kept by default.
CREATE INDEX idx_memories_id           ON memories (memory_id);
CREATE INDEX idx_relates_tenant_src    ON edges_relates (tenant_id, src);
CREATE INDEX idx_relates_tenant_dst    ON edges_relates (tenant_id, dst);
CREATE INDEX idx_about_dst             ON edges_about (dst);
