// anatid Phase 0 spike -- LadybugDB (Kuzu-compatible) schema, contract in spike/SPEC.md.
// Executed statement by statement (split on ';', '//' comment lines dropped) by bench/run_ladybug.py
// before the COPY FROM parquet loads. Column names and types mirror the Parquet files; src/dst of the
// edge files become the FROM/TO node references of the REL tables
// (COPY <rel> FROM (LOAD FROM 'edges_*.parquet' RETURN src, dst, ...)); node tables whose Parquet column
// order already matches are loaded with plain COPY <table> FROM 'file.parquet'.

CREATE NODE TABLE Entity(
    entity_id INT64,
    tenant_id INT32,
    kind STRING,
    name STRING,
    PRIMARY KEY(entity_id)
);

CREATE NODE TABLE Memory(
    memory_id INT64,
    tenant_id INT32,
    content STRING,
    kind STRING,
    embedding FLOAT[64],
    created_at TIMESTAMP,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    tx_from TIMESTAMP,
    tx_to TIMESTAMP,
    writer STRING,
    episode_id INT64,
    confidence FLOAT,
    PRIMARY KEY(memory_id)
);

CREATE REL TABLE ABOUT(
    FROM Memory TO Entity,
    edge_id INT64,
    tenant_id INT32,
    weight FLOAT,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    tx_from TIMESTAMP,
    tx_to TIMESTAMP
);

CREATE REL TABLE RELATES_TO(
    FROM Entity TO Entity,
    edge_id INT64,
    tenant_id INT32,
    rel_kind STRING,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    tx_from TIMESTAMP,
    tx_to TIMESTAMP
);

CREATE REL TABLE SUPERSEDES(
    FROM Memory TO Memory,
    edge_id INT64,
    tenant_id INT32,
    tx_from TIMESTAMP
);

// --- BM25 substitute index (runner-built, see run_ladybug.py) --------------------------------------
// LadybugDB 0.20.2 cannot create its FTS/VECTOR indexes here: INSTALL/LOAD FTS|VECTOR succeed, but the
// extension server (https://extension.ladybugdb.com/) only serves v0.20.0 builds (the engine looks in
// ~/.lbdb/extension/0.20.0/, /0.20.2/ and /v0.20.2/ return 404) and after LOAD the internal functions
// _CREATE_FTS_INDEX / _CREATE_HNSW_INDEX are not registered ("Catalog exception: function
// _CREATE_FTS_INDEX does not exist"; CREATE_FTS_INDEX also leaves half-created internal tables behind and
// closing that database afterwards aborts the process). The runner therefore materializes an inverted
// index at load time, the same shape as duckdb_ext's bm25_postings table: one Term node per distinct
// lower-cased whitespace token of memories.content (df = document frequency over all base memories) and
// one Posting node per (memory, term) pair with the term frequency, the document length and the term's df,
// loaded sorted by (term_id, tenant_id, memory_id). BM25 (k1 = 1.2, b = 0.75,
// idf = ln((N - df + 0.5) / (df + 0.5) + 1)) is evaluated in Cypher over Posting; the current-only filter is
// applied afterwards with a primary-key lookup of the candidate memories (valid_to IS NULL).
// A first version used a REL TABLE HAS_TERM(Memory -> Term) instead: 20 ms per BM25 query at small but
// 800 ms at full (the backward adjacency scan of a 13M-edge rel table into one 5,000-node Term node group
// degrades badly), so it was replaced by the node table. Like DuckDB's fts index it is NOT updated by
// W1/W2 writes.

CREATE NODE TABLE Term(
    term_id INT64,
    term STRING,
    df INT64,
    PRIMARY KEY(term_id)
);

CREATE NODE TABLE Posting(
    pid SERIAL,
    term_id INT64,
    tenant_id INT32,
    memory_id INT64,
    tf INT32,
    dl INT32,
    df INT64,
    PRIMARY KEY(pid)
);
