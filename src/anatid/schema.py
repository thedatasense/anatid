"""The anatid physical schema: node label -> table, edge type -> table.

This is the Phase 0 spike's winning DuckDB layout (``spike/duckdb_sql/schema.sql``) with the
product columns added.  What made it fast, kept verbatim:

* ``memories`` is loaded clustered by ``(tenant_id, created_at DESC, memory_id DESC)`` and
  ``edges_about`` by ``(tenant_id, dst, src)``, so DuckDB's zone maps skip every other tenant's
  row groups on the 2-hop recall scan and the TOP_N pushes a dynamic ``created_at`` threshold
  into the ``memories`` scan.  At the spike's *small* scale (100k memories / 10 tenants, single
  thread) this took the 2-hop recall p50 from 2.03 ms in Parquet order to 1.17 ms clustered, and
  the R2 cosine scan from 8.1 ms to 1.1 ms (``spike/duckdb_sql/schema.sql``, which labels those
  numbers small-scale).  The same clustering is what the 1M-row run measured at 2.88 ms p50
  (``spike/results/duckdb_sql.full.json``) -- do not read the 1.17 ms as a 1M-row figure.  The
  clustering costs load time only -- see :data:`CLUSTER_ORDER` and ``Anatid.recluster()``.
* Three ART indexes by default: ``idx_memories_id`` on ``memories(memory_id)`` (turns the
  supersede/forget point ``UPDATE`` into a lookup instead of a scan), ``idx_entities_name`` on
  ``entities(tenant_id, name)`` (entity resolution, maintained by every ``remember()`` that
  mints a new entity) and ``idx_episodes_id`` on ``episodes(episode_id)``.  DuckDB never uses
  ART for joins, so none of them changes the recall plan.  Five further candidate indexes are
  available through :data:`OPTIONAL_INDEXES` but are off by default (they cost insert time and
  file size and bought nothing on 2-hop recall).
* Almost no PRIMARY KEY / UNIQUE constraints: in DuckDB those create implicit ART indexes, and
  anatid wants every index to be an explicit, measured choice.  ``anatid`` mints ids itself
  (:mod:`anatid.ids`), so id uniqueness comes from the allocator, not from a constraint.  The
  ONE exception is entity identity -- see below -- where a constraint is the only thing that
  holds under concurrency, so it is paid for deliberately and listed in
  :data:`REQUIRED_INDEXES` rather than in the optional set.

System columns
--------------
``valid_from, valid_to, tx_from, tx_to, writer, episode_id, confidence`` are present on every
node and edge table by DEFAULT.  User-defined labels created with :func:`node_table_ddl` /
:func:`edge_table_ddl` can opt out with ``system_columns=False``, but then the temporal verbs
(``as_of``, ``supersede``, soft ``forget``) do not apply to that table -- which is why the
built-in tables never opt out.

Entity identity (schema v3)
---------------------------
``entities.entity_key`` is a **generated** column: ``trim(regexp_replace(lower(name), '\\s+',
' ', 'g'))`` -- lower-cased, internal whitespace collapsed, trimmed.  ``UNIQUE (tenant_id,
entity_key)`` is enforced by :data:`REQUIRED_INDEXES`.  Generated is the point: a plain column
can be left NULL by any writer that does not know about it, so the dedupe could be bypassed;
a generated column is computed by DuckDB on every insert and update and cannot be supplied,
which makes ``SELECT``-then-``INSERT`` entity creation safe under concurrency.  Measured on
this build (duckdb 1.5.5): 16 threads inserting the same name concurrently produce exactly one
row, the other 15 raising ``ConstraintException``/``TransactionException``.

Full text (schema v3)
---------------------
DuckDB's ``fts`` index needs a document key that is **unique over the indexed table**.  v2
indexed ``memories(memory_id)``, and ``memory_id`` is *not* unique across tenants in a scoped
file -- so tenant 2's document overwrote tenant 1's in the index, and a BM25 hit joined back to
whichever tenant's row the caller happened to be scoped to.  v3 indexes a purpose-built source
table, :data:`FTS_SOURCE_TABLE`, keyed on ``'<tenant_id>:<memory_id>'``, and carries
**per-tenant** corpus statistics in :data:`FTS_DICT_TABLE` / :data:`FTS_STATS_TABLE` so a
tenant's df/idf depends only on that tenant's rows.  See :func:`fts_rebuild_statements`.

One index over a composite key was chosen over one index per tenant.  Measured on this build
(duckdb 1.5.5, 4 threads, ``spike/data/small``: 100k memories / 10 tenants, 200 spike queries,
BM25 top-50):

====================================  =======  ==========  ==========  ======  =======
design                                build s  query p50   query p95   tables  file MB
====================================  =======  ==========  ==========  ======  =======
A: composite key + per-tenant stats     1.65     4.9 ms      5.9 ms      19      98
B: one index per tenant (10 tenants)    1.84     2.7 ms      3.2 ms      83     126
B: one index per tenant (100 tenants)   3.97     2.3 ms      2.6 ms     713     318
====================================  =======  ==========  ==========  ======  =======

Over 50 queries A and B returned **identical rankings and identical scores** (to 1e-9): the
sidecar tables reproduce a per-tenant index's statistics exactly, so the leak fix does not
depend on the index layout.  B is faster per query (each tenant's postings table is a tenth
the size) but every tenant costs a schema, five tables and an ART index, the rebuild grows with
the tenant count because ``create_fts_index`` has a fixed per-call cost, the file is 30-225%
larger, and every purge / doctor / erasure path would have to enumerate dynamic table names.
A's 4.9 ms is under the v2 baseline the spike recorded for the same data (8.6 ms p50, single
thread, with the termid index), so the constant object count won.

What is still shared in A, precisely: the extension's own ``fts_main_anatid_fts_documents.dict``
(``term -> termid, df``) and ``.stats`` are computed over every tenant's rows.  anatid reads
**only** the ``term -> termid`` mapping from ``dict`` and never its ``df`` nor ``stats``; a term
another tenant used maps to a termid that has no row in that tenant's :data:`FTS_DICT_TABLE`,
so the query returns nothing -- the same as for a term nobody used.  Nothing observable through
``recall()`` depends on another tenant's corpus.  Someone with raw SQL on the file can read the
global ``df``, exactly as they can read ``memories`` itself: ``tenant_id`` inside one file is
scoping, not isolation (see :data:`CONTRACT_NOTES`), and isolation is file-per-tenant.

Time travel
-----------
DuckDB has no ``AS OF SYSTEM TIME``.  These columns plus the WHERE clauses anatid compiles in
:mod:`anatid.recall` *are* the time-travel mechanism.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .errors import SchemaVersionError

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_EMBEDDING_DIM",
    "SchemaConfig",
    "SYSTEM_COLUMNS",
    "MEMORY_COLUMNS",
    "ENTITY_COLUMNS",
    "EPISODE_COLUMNS",
    "NODE_TABLES",
    "EDGE_TABLES",
    "ALL_TABLES",
    "CLUSTER_ORDER",
    "DEFAULT_INDEXES",
    "OPTIONAL_INDEXES",
    "REQUIRED_INDEXES",
    "CONTRACT_NOTES",
    "FTS_INDEX_SQL",
    "FTS_TERMS_INDEX_SQL",
    "FTS_SOURCE_TABLE",
    "FTS_DOCS_TABLE",
    "FTS_DICT_TABLE",
    "FTS_STATS_TABLE",
    "FTS_TABLES",
    "FTS_INDEX_SCHEMA",
    "FTS_TOKENIZER",
    "FTS_DOC_ID_SQL",
    "fts_doc_id",
    "fts_rebuild_statements",
    "fts_purge_statements",
    "fts_purge",
    "fts_objects_present",
    "ENTITY_KEY_COLUMN",
    "ENTITY_KEY_EXPR",
    "entity_key_sql",
    "entity_key",
    "GENERATED_COLUMNS",
    "insertable_columns",
    "ddl_statements",
    "node_table_ddl",
    "edge_table_ddl",
    "index_statements",
    "required_index_statements",
    "ensure_schema",
    "current_version",
    "MIGRATIONS",
    "register_migration",
    "temporal_predicate",
    "quote_ident",
    "check_type",
    "MEMORY_COLUMNS_NO_EMBEDDING",
    "CATALOG_TABLES",
    "table_names",
    "embedding_dim",
    "missing_tables",
    "table_ddl",
]

SCHEMA_VERSION = 3
DEFAULT_EMBEDDING_DIM = 1536

#: The bitemporal / provenance columns present on every node and edge table by default.
SYSTEM_COLUMNS: tuple[tuple[str, str], ...] = (
    ("valid_from", "TIMESTAMP"),
    ("valid_to", "TIMESTAMP"),   # NULL = still true
    ("tx_from", "TIMESTAMP"),
    ("tx_to", "TIMESTAMP"),      # NULL = live row
    ("writer", "VARCHAR"),
    ("episode_id", "BIGINT"),
    ("confidence", "FLOAT"),
)

#: Column order used by ``SELECT`` for a full memory row (matches :meth:`anatid.Memory.from_row`).
MEMORY_COLUMNS: tuple[str, ...] = (
    "memory_id", "tenant_id", "content", "kind", "embedding", "created_at",
    "valid_from", "valid_to", "tx_from", "tx_to", "writer", "episode_id", "confidence",
    "access_count", "last_access_at",
)

#: Same, minus the embedding, for callers that do not need the vector (cheaper scan).
MEMORY_COLUMNS_NO_EMBEDDING: tuple[str, ...] = tuple(
    "NULL" if c == "embedding" else c for c in MEMORY_COLUMNS
)

#: Column order for a full entity row (matches :meth:`anatid.Entity.from_row`).  ``entity_key``
#: is deliberately NOT here: it is derived from ``name``, adding it would change the arity of
#: every ``Entity.from_row`` call site, and nothing outside the uniqueness constraint reads it.
ENTITY_COLUMNS: tuple[str, ...] = (
    "entity_id", "tenant_id", "kind", "name",
    "valid_from", "valid_to", "tx_from", "tx_to", "writer", "episode_id", "confidence",
)

EPISODE_COLUMNS: tuple[str, ...] = (
    "episode_id", "tenant_id", "source", "content", "kind", "created_at",
    "valid_from", "valid_to", "tx_from", "tx_to", "writer",
)

# --------------------------------------------------------------------------- entity identity

#: Name of the generated canonical-name column on ``entities`` (schema v3).
ENTITY_KEY_COLUMN = "entity_key"

#: The canonicalisation, as a SQL expression template.  ``{expr}`` is the name expression.
#:
#: ``lower`` then collapse-whitespace then ``trim`` -- in that order, because DuckDB's ``trim``
#: strips ASCII spaces only, so a name that begins with a tab is only trimmed correctly *after*
#: the whitespace run has been collapsed to a space.
ENTITY_KEY_EXPR = r"trim(regexp_replace(lower({expr}), '\s+', ' ', 'g'))"

#: Whitespace class RE2 (DuckDB's regex engine) matches for ``\s``.  The Python mirror below
#: uses exactly this class so it cannot disagree with the database over, say, ``\v`` or U+00A0.
_RE2_SPACE = re.compile(r"[\t\n\f\r ]+")


def entity_key_sql(expr: str = "name") -> str:
    """The canonical-entity-key SQL for ``expr`` (a column reference or a ``?`` placeholder).

    Use this -- not the Python :func:`entity_key` -- whenever the value is going into a query:
    binding the raw name and letting DuckDB canonicalise it is the only way a lookup is
    guaranteed to agree with the generated column the UNIQUE index is built on.

    ``entity_key_sql("?")`` gives the fragment a lookup should compare against.
    """
    return ENTITY_KEY_EXPR.format(expr=expr)


def entity_key(name: str | None) -> str | None:
    """Python mirror of :func:`entity_key_sql`, for tests, logs and dict keys.

    The database is authoritative: ``entities.entity_key`` is generated by DuckDB.  This mirrors
    it closely (same whitespace class, same ordering) but ``str.lower()`` and DuckDB's ``lower``
    are two implementations of Unicode case folding and can disagree on exotic input, so never
    write a *lookup* against this value -- use :func:`entity_key_sql` and bind the raw name.
    """
    if name is None:
        return None
    return _RE2_SPACE.sub(" ", str(name).lower()).strip(" ")


#: Columns DuckDB computes itself, per built-in table.  They may not appear in an INSERT column
#: list (``Binder Error: Cannot insert into a generated column``), so anything that rebuilds a
#: table by enumerating ``PRAGMA table_info`` must filter them -- see :func:`insertable_columns`.
GENERATED_COLUMNS: dict[str, tuple[str, ...]] = {
    "entities": (ENTITY_KEY_COLUMN,),
}


def insertable_columns(con, table: str) -> list[str]:
    """The columns of ``table`` an ``INSERT`` may name, in table order.

    ``PRAGMA table_info`` lists generated columns like any other, but DuckDB rejects them in an
    INSERT column list.  ``Anatid.recluster()`` and any other "rebuild this table" path must go
    through here instead of using ``table_info`` directly.
    """
    generated = set(GENERATED_COLUMNS.get(table, ()))
    rows = con.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
    return [r[1] for r in rows if r[1] not in generated]


# --------------------------------------------------------------------------- full-text objects

#: The table the fts index is built over.  NOT ``memories``: the fts extension needs a document
#: key that is unique over the indexed table, and ``memory_id`` is only unique *within* a tenant.
FTS_SOURCE_TABLE = "anatid_fts_documents"

#: ``docid -> (tenant_id, memory_id, doc length)``, so the BM25 query can restrict the candidate
#: set to one tenant *before* it scores, and never joins on the document-key string.  Named so
#: that it is not a substring of :data:`FTS_SOURCE_TABLE`: several call sites (erasure, health
#: checks) match table names in SQL text, and ``anatid_fts_docs`` inside ``anatid_fts_documents``
#: is a trap.
FTS_DOCS_TABLE = "anatid_fts_docmap"

#: Per-tenant document frequency: ``(tenant_id, termid, df)``.  This is what makes idf tenant
#: local -- ``fts_main_*.dict.df`` counts documents in every tenant.
FTS_DICT_TABLE = "anatid_fts_dict"

#: Per-tenant corpus statistics: ``(tenant_id, num_docs, avgdl)``, the tenant-local replacement
#: for ``fts_main_*.stats``.
FTS_STATS_TABLE = "anatid_fts_stats"

#: Every table that holds a copy of, or a statistic derived from, memory content.
#: ``forget(hard=True)`` MUST clear the memory's row from these -- :func:`fts_purge` does it.
FTS_TABLES: tuple[str, ...] = (FTS_SOURCE_TABLE, FTS_DOCS_TABLE, FTS_DICT_TABLE, FTS_STATS_TABLE)

#: The schema ``PRAGMA create_fts_index`` creates for :data:`FTS_SOURCE_TABLE`.
FTS_INDEX_SCHEMA = f"fts_main_{FTS_SOURCE_TABLE}"

#: The exact tokenizer settings the spike benchmarked.  They define what BM25 can match, and the
#: BM25 SQL in :mod:`anatid.recall` tokenizes queries the same way.
FTS_TOKENIZER = (r"stemmer='none', stopwords='none', ignore='(\.|[^a-z])+', "
                 r"strip_accents=0, lower=1")

#: How the composite document key is built in SQL.  Mirror of :func:`fts_doc_id`.
FTS_DOC_ID_SQL = "CAST(tenant_id AS VARCHAR) || ':' || CAST(memory_id AS VARCHAR)"


def fts_doc_id(tenant_id: int, memory_id: int) -> str:
    """The fts document key for one memory: ``'<tenant_id>:<memory_id>'``.

    Globally unique inside the file, which ``memory_id`` alone is not -- that collision is what
    let a BM25 hit on tenant 2's text return tenant 1's row in schema v2.
    """
    return f"{int(tenant_id)}:{int(memory_id)}"


#: The fts index PRAGMA.  Kept under its v2 name so ``from .schema import FTS_INDEX_SQL`` still
#: resolves, but it now targets :data:`FTS_SOURCE_TABLE` and its composite key.  On its own it
#: is no longer enough to build the index: run :func:`fts_rebuild_statements` in order.
FTS_INDEX_SQL = (
    f"PRAGMA create_fts_index('{FTS_SOURCE_TABLE}', 'fts_doc_id', 'content', "
    f"{FTS_TOKENIZER}, overwrite=1)"
)

#: ART index on the fts extension's postings table.  The spike A/B'd it: BM25 p50 goes from
#: 12.51 ms to 8.57 ms at 100k memories, for a 0.7 s build.  ``create_fts_index`` drops and
#: recreates the whole ``fts_main_*`` schema, taking this index with it, so it is recreated by
#: every :func:`fts_rebuild_statements`.
FTS_TERMS_INDEX_SQL = f"CREATE INDEX idx_fts_terms_termid ON {FTS_INDEX_SCHEMA}.terms(termid)"


def fts_rebuild_statements(*, terms_index: bool = True) -> list[str]:
    """Every statement that rebuilds the BM25 index, in order.  Run them in one transaction.

    The DuckDB fts index is not incremental -- ``PRAGMA create_fts_index`` rebuilds it wholesale
    -- so the source table is refreshed wholesale in the same breath, and the per-tenant
    statistics are derived from the postings that build produced.

    Order matters:

    1. refill :data:`FTS_SOURCE_TABLE` from ``memories``, keyed ``'<tenant>:<memory>'``.  One
       row per ``(tenant_id, memory_id)``: duplicate ids inside one tenant are an integrity
       fault (``db.doctor()`` reports them) and must not become two documents here, because a
       repeated document key is exactly the bug this key was introduced to fix.
    2. build the index, and the ART index on its postings.
    3. materialise ``docid -> (tenant_id, memory_id, len)``.
    4. materialise per-tenant ``df`` and per-tenant ``(num_docs, avgdl)``.
    5. drop the schema-v2 ``fts_main_memories`` index if a v2 build left one behind: it is keyed
       on ``memory_id`` and leaks across tenants, so it must never coexist with this one.

    Every ``memories`` row is indexed, live or superseded, exactly as in v2 -- the temporal
    predicate is applied by the query, so ``as_of`` BM25 keeps working.
    """
    stmts = [
        f"DELETE FROM {FTS_SOURCE_TABLE}",
        f"""INSERT INTO {FTS_SOURCE_TABLE} (fts_doc_id, tenant_id, memory_id, content)
    SELECT {FTS_DOC_ID_SQL}, tenant_id, memory_id, content
    FROM memories
    QUALIFY row_number() OVER (PARTITION BY tenant_id, memory_id
                               ORDER BY tx_from DESC NULLS LAST,
                                        created_at DESC NULLS LAST) = 1
    ORDER BY tenant_id, memory_id""",
        FTS_INDEX_SQL,
    ]
    if terms_index:
        stmts.append(FTS_TERMS_INDEX_SQL)
    stmts += [
        f"DELETE FROM {FTS_DOCS_TABLE}",
        f"""INSERT INTO {FTS_DOCS_TABLE} (docid, tenant_id, memory_id, len)
    SELECT d.docid, s.tenant_id, s.memory_id, d.len
    FROM {FTS_INDEX_SCHEMA}.docs d
    JOIN {FTS_SOURCE_TABLE} s ON s.fts_doc_id = d.name
    ORDER BY s.tenant_id, d.docid""",
        f"DELETE FROM {FTS_DICT_TABLE}",
        f"""INSERT INTO {FTS_DICT_TABLE} (tenant_id, termid, df)
    SELECT dt.tenant_id, t.termid, count(DISTINCT t.docid)
    FROM {FTS_INDEX_SCHEMA}.terms t
    JOIN {FTS_DOCS_TABLE} dt ON dt.docid = t.docid
    GROUP BY 1, 2
    ORDER BY 1, 2""",
        f"DELETE FROM {FTS_STATS_TABLE}",
        f"""INSERT INTO {FTS_STATS_TABLE} (tenant_id, num_docs, avgdl)
    SELECT tenant_id, count(*), CAST(avg(len) AS DOUBLE)
    FROM {FTS_DOCS_TABLE} GROUP BY 1""",
        "DROP SCHEMA IF EXISTS fts_main_memories CASCADE",
    ]
    return stmts


def fts_objects_present(con) -> bool:
    """True when a v3 BM25 index has been built on this database."""
    row = con.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = ? AND table_name = 'docs'", [FTS_INDEX_SCHEMA]).fetchone()
    return bool(row and int(row[0]) > 0)


def fts_purge_statements() -> tuple[str, ...]:
    """Statements that erase one memory from the BM25 index.  Each binds ``(tenant_id,
    memory_id)`` in that order; run them in this order, inside the purge transaction.

    Needed because :data:`FTS_SOURCE_TABLE` holds the memory's content **verbatim** and the
    postings hold its tokens.  ``forget(hard=True)`` that skips this leaves the erased text in
    the file.  Prefer :func:`fts_purge`, which skips statements whose objects do not exist yet.

    The extension's own ``dict`` table maps every token in the corpus to a ``termid``, so a
    token that occurred in the erased document and nowhere else is a verbatim fragment of the
    erased text (the whole text, for a one-word memory).  The first statement removes those
    dict rows -- the ones whose every posting belongs to the erased document -- before the
    postings themselves go.  A token another document also uses stays, because it is that
    document's word too.

    The per-tenant ``df`` in :data:`FTS_DICT_TABLE` is left slightly high (it counted a document
    that is now gone) until the next :func:`fts_rebuild_statements`; that is a number keyed by
    ``termid``, not content, and it names nothing.
    """
    return tuple(sql for _needs, sql in _FTS_PURGE_PLAN)


#: ``(objects the statement needs, statement)``.  Kept as data so :func:`fts_purge` decides what
#: to skip from the plan rather than by matching table names inside SQL text.  Every ``?`` pair
#: in a statement binds ``(tenant_id, memory_id)``; a statement may name the pair more than once.
_FTS_PURGE_PLAN: tuple[tuple[tuple[str, ...], str], ...] = (
    # Tokens whose only postings are the erased document's: their dict rows spell out the text.
    # The ART index on terms(termid) serves the inner IN-list; the outer GROUP BY touches only
    # the postings of those termids.
    (("index", FTS_DOCS_TABLE),
     f"DELETE FROM {FTS_INDEX_SCHEMA}.dict WHERE termid IN ("
     f"SELECT t.termid FROM {FTS_INDEX_SCHEMA}.terms t WHERE t.termid IN ("
     f"SELECT p.termid FROM {FTS_INDEX_SCHEMA}.terms p JOIN {FTS_DOCS_TABLE} d ON d.docid = p.docid "
     f"WHERE d.tenant_id = ? AND d.memory_id = ?) "
     f"GROUP BY t.termid HAVING count(*) = count(*) FILTER (WHERE t.docid IN ("
     f"SELECT docid FROM {FTS_DOCS_TABLE} WHERE tenant_id = ? AND memory_id = ?)))"),
    (("index", FTS_DOCS_TABLE),
     f"DELETE FROM {FTS_INDEX_SCHEMA}.terms WHERE docid IN "
     f"(SELECT docid FROM {FTS_DOCS_TABLE} WHERE tenant_id = ? AND memory_id = ?)"),
    (("index",),
     f"DELETE FROM {FTS_INDEX_SCHEMA}.docs WHERE name = "
     f"CAST(? AS VARCHAR) || ':' || CAST(? AS VARCHAR)"),
    ((FTS_DOCS_TABLE,),
     f"DELETE FROM {FTS_DOCS_TABLE} WHERE tenant_id = ? AND memory_id = ?"),
    ((FTS_SOURCE_TABLE,),
     f"DELETE FROM {FTS_SOURCE_TABLE} WHERE tenant_id = ? AND memory_id = ?"),
)


def fts_purge(con, tenant_id: int, memory_id: int) -> int:
    """Erase one memory from the BM25 index; returns the number of rows deleted.

    Call it inside ``forget(hard=True)``'s transaction.  Safe to call when no index has been
    built (the fts schema is simply absent) and safe to call twice.
    """
    pair = [int(tenant_id), int(memory_id)]
    available = set(table_names(con))
    if fts_objects_present(con):
        available.add("index")
    deleted = 0
    for needs, sql in _FTS_PURGE_PLAN:
        if any(n not in available for n in needs):
            continue
        row = con.execute(sql, pair * (sql.count("?") // 2)).fetchone()
        deleted += int(row[0]) if row and row[0] is not None else 0
    return deleted


NODE_TABLES: tuple[str, ...] = ("memories", "entities", "episodes")
EDGE_TABLES: tuple[str, ...] = ("edges_about", "edges_relates", "edges_supersedes")
CATALOG_TABLES: tuple[str, ...] = ("anatid_meta", "anatid_audit") + FTS_TABLES
ALL_TABLES: tuple[str, ...] = NODE_TABLES + EDGE_TABLES + CATALOG_TABLES

#: Physical order each table is written in by a bulk load / recluster.  This is the measured
#: layout from the spike -- see the module docstring.
CLUSTER_ORDER: dict[str, str] = {
    "memories": "tenant_id, created_at DESC, memory_id DESC",
    "edges_about": "tenant_id, dst, src",
    "edges_relates": "tenant_id, src",
    "entities": "tenant_id, entity_id",
    "episodes": "tenant_id, episode_id",
}

#: Created by ``ensure_schema()``.  One index, for the point UPDATE in supersede/forget/reinforce,
#: plus entity resolution by name (``remember(entities=["Alice"])`` is a point lookup).
DEFAULT_INDEXES: dict[str, str] = {
    "idx_memories_id": "CREATE INDEX IF NOT EXISTS idx_memories_id ON memories (memory_id)",
    "idx_entities_name": "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities (tenant_id, name)",
    "idx_episodes_id": "CREATE INDEX IF NOT EXISTS idx_episodes_id ON episodes (episode_id)",
}

#: Measured in the spike and found not to change the recall plan (DuckDB does not use ART for
#: joins).  Available for workloads dominated by single-edge point lookups.
OPTIONAL_INDEXES: dict[str, str] = {
    "idx_relates_tenant_src": "CREATE INDEX IF NOT EXISTS idx_relates_tenant_src ON edges_relates (tenant_id, src)",
    "idx_relates_tenant_dst": "CREATE INDEX IF NOT EXISTS idx_relates_tenant_dst ON edges_relates (tenant_id, dst)",
    "idx_about_dst": "CREATE INDEX IF NOT EXISTS idx_about_dst ON edges_about (dst)",
    "idx_about_src": "CREATE INDEX IF NOT EXISTS idx_about_src ON edges_about (src)",
    "idx_supersedes_dst": "CREATE INDEX IF NOT EXISTS idx_supersedes_dst ON edges_supersedes (dst)",
}

#: Indexes that are part of the *contract*, not of the performance tuning, and are therefore NOT
#: selectable through :class:`SchemaConfig`: this one is the enforcement behind "one entity per
#: canonical name per tenant".  :func:`ensure_schema` always creates it.
REQUIRED_INDEXES: dict[str, str] = {
    "ux_entities_tenant_key": (
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_entities_tenant_key "
        f"ON entities (tenant_id, {ENTITY_KEY_COLUMN})"),
}

#: Written into ``anatid_meta.contract``.  These are the engine's real guarantees; they are stored
#: in the file so anyone who opens it later reads the same sentences the docs make.
CONTRACT_NOTES: tuple[str, ...] = (
    "time travel: DuckDB has NO 'AS OF SYSTEM TIME'. as_of is anatid's own WHERE filter over "
    "valid_from/valid_to and tx_from/tx_to. Intervals are half-open [from, to).",
    "isolation: DuckDB has NO schema-level or row-level access control. Per-tenant ISOLATION is "
    "file-per-tenant, enforced by anatid's wrapper and the filesystem. tenant_id inside one file "
    "is SCOPING, not isolation: any connection to the file can read every tenant in it.",
    "full-text: the DuckDB fts index is NOT incremental. Rows inserted after "
    "PRAGMA create_fts_index are invisible to BM25 until rebuild_fts_index() runs. anatid records "
    "both the indexed row count and the largest indexed memory_id in this table and reports "
    "staleness on every recall() result. A raw SQL UPDATE of memories.content (something no "
    "anatid verb ever issues) changes neither watermark and is NOT detected.",
    "full-text scoping (schema v3): the index is built over anatid_fts_documents, whose document "
    "key is '<tenant_id>:<memory_id>' -- unique across tenants, which memory_id alone is not. "
    "df/idf and (num_docs, avgdl) come from anatid_fts_dict / anatid_fts_stats, which are "
    "computed PER TENANT, so one tenant's BM25 scores depend on that tenant's rows only and no "
    "term statistic crosses the boundary through recall(). The fts extension's own dict/stats "
    "tables under fts_main_anatid_fts_documents still count every tenant; anatid reads only the "
    "term->termid mapping from them, never a df or a corpus size, and a raw SQL reader of the file "
    "can see them exactly as it can see every tenant's memories (scoping, not isolation). Schema "
    "v2 files indexed memories(memory_id) and did leak: the 2->3 migration drops that index.",
    "entity identity (schema v3): entities.entity_key is a GENERATED column, "
    "trim(regexp_replace(lower(name), '\\s+', ' ', 'g')), with UNIQUE (tenant_id, entity_key) "
    "enforced by an index. Concurrent remember() calls naming the same new entity can no longer "
    "produce two entity rows; the losers get a retryable constraint/transaction error.",
    "transactions: DuckDB MVCC is optimistic and snapshot-isolated, NOT serializable. Appends "
    "never conflict; two concurrent updates to the SAME row abort the second with a retryable "
    "error, surfaced as anatid.errors.ConflictError.",
    "vector search: brute-force array_cosine_similarity over FLOAT[N]. Fine to roughly 1e5 "
    "memories per tenant; beyond that latency grows linearly with the tenant's row count.",
    "erasure: forget(hard=True) purges the memory GRAPH row, its edges, its embedding, its "
    "orphaned episode, its provenance and its BM25 documents (anatid_fts_documents holds content "
    "verbatim, so anatid.schema.fts_purge() is part of the purge), and no anatid_audit row "
    "survives that references the purged memory_id. It does NOT reach tables anatid does not own. "
    "Conversation transcripts (anatid.integrations.openai_agents.AnatidSession's agent_messages) "
    "quote memory content and ids verbatim; AnatidSession registers an erasure hook so a purge "
    "removes those rows too, but any other table you write into this file is yours to clean -- "
    "see Anatid.erasure_hooks.",
)


@dataclass(frozen=True)
class SchemaConfig:
    """Knobs that change the DDL.

    ``embedding_dim``
        ``N`` in ``FLOAT[N]``.  Default 1536 (OpenAI ``text-embedding-3-small``).  The spike's
        test data is 64-dimensional, so the test suite opens databases with ``embedding_dim=64``.
        The dimension is stamped into ``anatid_meta`` and enforced on write.
    ``system_columns``
        Default True.  See the module docstring; only user-defined labels should turn it off.
    ``indexes``
        Names from :data:`DEFAULT_INDEXES` / :data:`OPTIONAL_INDEXES` to create.  This does not
        reach :data:`REQUIRED_INDEXES`, which are constraints rather than tuning.
    """

    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    system_columns: bool = True
    indexes: tuple[str, ...] = tuple(DEFAULT_INDEXES)

    def __post_init__(self) -> None:
        if int(self.embedding_dim) <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        object.__setattr__(self, "embedding_dim", int(self.embedding_dim))
        unknown = [i for i in self.indexes if i not in DEFAULT_INDEXES and i not in OPTIONAL_INDEXES]
        if unknown:
            raise ValueError(f"unknown index name(s): {unknown}")


# --------------------------------------------------------------------------- identifier safety

_RESERVED = {"select", "from", "where", "table", "index", "order", "group", "insert", "update",
             "delete", "create", "drop", "join", "union", "all", "and", "or", "not", "null"}


def quote_ident(name: str) -> str:
    """Validate and double-quote an identifier.

    anatid never string-builds SQL from *values*; table and column names are structure, and this
    is the one gate they pass through.  Only ``[A-Za-z_][A-Za-z0-9_]*`` up to 63 chars is allowed,
    so nothing can escape the quotes.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("identifier must be a non-empty string")
    if len(name) > 63:
        raise ValueError(f"identifier too long: {name!r}")
    if not (name[0].isalpha() or name[0] == "_"):
        raise ValueError(f"identifier must start with a letter or underscore: {name!r}")
    if not all(c.isalnum() or c == "_" for c in name):
        raise ValueError(f"identifier may only contain letters, digits and underscore: {name!r}")
    if name.lower() in _RESERVED:
        raise ValueError(f"identifier is a reserved word: {name!r}")
    return f'"{name}"'


#: Characters a DuckDB type expression may contain.  Everything that could terminate the column
#: list or the statement -- ``;``, quotes, comment introducers, operators -- is absent.
_TYPE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ ,()[]")


def check_type(type_sql: str) -> str:
    """Validate a DuckDB type expression before it is interpolated into DDL.

    A column *type* is structure, exactly like a column *name*, and it therefore needs the same
    gate :func:`quote_ident` gives names -- a type cannot be bound as a parameter, so if it is
    not validated it is an injection point.  (It was one: ``create_node_label("t", [("x",
    "INTEGER); DROP TABLE memories; CREATE TABLE zz (a INTEGER")])`` used to drop ``memories``.)

    Accepted: letters, digits, underscore, space, comma, ``()`` and ``[]``, starting with a
    letter or underscore, with balanced brackets, no top-level comma, and at most 200 characters.
    That covers everything DuckDB's type grammar needs -- ``VARCHAR``, ``DECIMAL(18, 3)``,
    ``FLOAT[64]``, ``BIGINT[]``, ``TIMESTAMP WITH TIME ZONE``, ``STRUCT(a INTEGER, b VARCHAR)``,
    ``MAP(VARCHAR, INTEGER)``, and a trailing ``NOT NULL`` -- while making it structurally
    impossible to close the column list or start a second statement.  Returns the type unchanged.
    """
    if not isinstance(type_sql, str) or not type_sql.strip():
        raise ValueError("column type must be a non-empty string")
    t = type_sql.strip()
    if len(t) > 200:
        raise ValueError(f"column type too long: {t[:40]!r}...")
    bad = sorted({c for c in t if c not in _TYPE_CHARS})
    if bad:
        raise ValueError(
            f"column type may only contain letters, digits, underscore, space, comma, "
            f"parentheses and square brackets; got {bad} in {t!r}")
    if not (t[0].isalpha() or t[0] == "_"):
        raise ValueError(f"column type must start with a letter or underscore: {t!r}")
    depth = 0
    for c in t:
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unbalanced brackets in column type: {t!r}")
        elif c == "," and depth == 0:
            raise ValueError(
                f"a column type may not contain a top-level comma (that would declare a second "
                f"column): {t!r}")
    if depth != 0:
        raise ValueError(f"unbalanced brackets in column type: {t!r}")
    return t


# --------------------------------------------------------------------------- DDL builders

def _system_column_sql(system_columns: bool) -> str:
    if not system_columns:
        return ""
    return "".join(f",\n    {n:<12} {t}" for n, t in SYSTEM_COLUMNS)


#: The generated-column clause appended to ``entities``.  Last in the column list so that adding
#: it did not renumber any existing column.
_ENTITY_KEY_DDL = (
    f",\n    {ENTITY_KEY_COLUMN:<12} VARCHAR GENERATED ALWAYS AS "
    f"({entity_key_sql('name')}) VIRTUAL"
)


def node_table_ddl(
    label: str,
    properties: Sequence[tuple[str, str]] = (),
    *,
    id_column: str | None = None,
    system_columns: bool = True,
    if_not_exists: bool = True,
) -> str:
    """DDL for a user-defined node label (one label -> one table).

    ``properties`` are ``(column_name, duckdb_type)`` pairs.  The table always gets
    ``<label>_id BIGINT NOT NULL`` (or ``id_column``) and ``tenant_id INTEGER NOT NULL`` first,
    and the system columns last unless ``system_columns=False``.

    Names go through :func:`quote_ident` and types through :func:`check_type`; both raise
    ``ValueError`` rather than reaching the statement, so neither is an injection point.
    """
    table = quote_ident(label)
    idc = quote_ident(id_column or f"{label}_id")
    cols = [f"{idc:<12} BIGINT NOT NULL", f'{"tenant_id":<12} INTEGER NOT NULL']
    cols += [f"{quote_ident(n):<12} {check_type(t)}" for n, t in properties]
    exists = "IF NOT EXISTS " if if_not_exists else ""
    return (f"CREATE TABLE {exists}{table} (\n    " + ",\n    ".join(cols)
            + _system_column_sql(system_columns) + "\n)")


def edge_table_ddl(
    edge_type: str,
    properties: Sequence[tuple[str, str]] = (),
    *,
    table: str | None = None,
    system_columns: bool = True,
    if_not_exists: bool = True,
) -> str:
    """DDL for a user-defined edge type (one type -> one table, ``src``/``dst`` BIGINT).

    Names go through :func:`quote_ident` and types through :func:`check_type`.
    """
    name = quote_ident(table or f"edges_{edge_type.lower()}")
    cols = [f'{"edge_id":<12} BIGINT NOT NULL', f'{"src":<12} BIGINT NOT NULL',
            f'{"dst":<12} BIGINT NOT NULL', f'{"tenant_id":<12} INTEGER NOT NULL']
    cols += [f"{quote_ident(n):<12} {check_type(t)}" for n, t in properties]
    exists = "IF NOT EXISTS " if if_not_exists else ""
    return (f"CREATE TABLE {exists}{name} (\n    " + ",\n    ".join(cols)
            + _system_column_sql(system_columns) + "\n)")


def _fts_table_ddl() -> list[str]:
    """The four tables that make BM25 tenant-scoped (schema v3).  See :data:`FTS_SOURCE_TABLE`."""
    return [
        # The indexed table. fts_doc_id is '<tenant_id>:<memory_id>' -- the fts extension needs a
        # key that is unique over the table it indexes, and memory_id is unique only per tenant.
        # content is a verbatim copy: forget(hard=True) must clear it, see fts_purge().
        f"""CREATE TABLE IF NOT EXISTS {FTS_SOURCE_TABLE} (
    fts_doc_id   VARCHAR NOT NULL,
    tenant_id    INTEGER NOT NULL,
    memory_id    BIGINT  NOT NULL,
    content      VARCHAR
)""",
        f"""CREATE TABLE IF NOT EXISTS {FTS_DOCS_TABLE} (
    docid        BIGINT  NOT NULL,
    tenant_id    INTEGER NOT NULL,
    memory_id    BIGINT  NOT NULL,
    len          BIGINT
)""",
        f"""CREATE TABLE IF NOT EXISTS {FTS_DICT_TABLE} (
    tenant_id    INTEGER NOT NULL,
    termid       BIGINT  NOT NULL,
    df           BIGINT  NOT NULL
)""",
        f"""CREATE TABLE IF NOT EXISTS {FTS_STATS_TABLE} (
    tenant_id    INTEGER NOT NULL,
    num_docs     BIGINT  NOT NULL,
    avgdl        DOUBLE  NOT NULL
)""",
    ]


def ddl_statements(config: SchemaConfig | None = None) -> list[str]:
    """Every CREATE TABLE / CREATE VIEW statement for a fresh anatid database, in order."""
    cfg = config or SchemaConfig()
    dim = cfg.embedding_dim
    sysc = _system_column_sql(cfg.system_columns)

    stmts: list[str] = [
        # ---- catalog ------------------------------------------------------------------
        """CREATE TABLE IF NOT EXISTS anatid_meta (
    schema_version   INTEGER   NOT NULL,
    created_at       TIMESTAMP NOT NULL,
    embedding_dim    INTEGER   NOT NULL,
    anatid_version   VARCHAR   NOT NULL,
    duckdb_version   VARCHAR   NOT NULL,
    system_columns   BOOLEAN   NOT NULL,
    fts_indexed_at   TIMESTAMP,
    fts_indexed_rows BIGINT,
    fts_indexed_max_id BIGINT,
    contract         VARCHAR   NOT NULL
)""",
        # Audit trail for soft forget / supersede. A HARD purge deletes this table's rows for the
        # purged memory too -- erasure means erasure. `related_memory_id` exists so the
        # counterpart of a supersede is a COLUMN and not free text inside `reason`: a purge has to
        # be able to find and delete every row that names the erased id (schema v2).
        """CREATE TABLE IF NOT EXISTS anatid_audit (
    audit_id   BIGINT    NOT NULL,
    tenant_id  INTEGER   NOT NULL,
    memory_id  BIGINT,
    related_memory_id BIGINT,
    action     VARCHAR   NOT NULL,
    reason     VARCHAR,
    writer     VARCHAR,
    happened_at TIMESTAMP NOT NULL
)""",
        # ---- node labels --------------------------------------------------------------
        # entity_key is GENERATED and carries UNIQUE (tenant_id, entity_key) through
        # REQUIRED_INDEXES: that pair, not the SELECT-then-INSERT in verbs.py, is what stops two
        # concurrent remember() calls minting two entities for one name (schema v3).
        f"""CREATE TABLE IF NOT EXISTS entities (
    entity_id    BIGINT   NOT NULL,
    tenant_id    INTEGER  NOT NULL,
    kind         VARCHAR,
    name         VARCHAR{sysc}{_ENTITY_KEY_DDL}
)""",
        f"""CREATE TABLE IF NOT EXISTS memories (
    memory_id      BIGINT    NOT NULL,
    tenant_id      INTEGER   NOT NULL,
    content        VARCHAR,
    kind           VARCHAR,
    embedding      FLOAT[{dim}],
    created_at     TIMESTAMP NOT NULL{sysc},
    access_count   INTEGER   DEFAULT 0,
    last_access_at TIMESTAMP
)""",
        # "Evidence before belief": raw source text is written first, derived facts carry the
        # episode_id, and provenance() walks back here.
        """CREATE TABLE IF NOT EXISTS episodes (
    episode_id   BIGINT    NOT NULL,
    tenant_id    INTEGER   NOT NULL,
    source       VARCHAR,
    content      VARCHAR,
    kind         VARCHAR,
    created_at   TIMESTAMP NOT NULL,
    valid_from   TIMESTAMP,
    valid_to     TIMESTAMP,
    tx_from      TIMESTAMP,
    tx_to        TIMESTAMP,
    writer       VARCHAR
)""",
        # ---- edge types ---------------------------------------------------------------
        f"""CREATE TABLE IF NOT EXISTS edges_about (
    edge_id   BIGINT  NOT NULL,
    src       BIGINT  NOT NULL,
    dst       BIGINT  NOT NULL,
    tenant_id INTEGER NOT NULL,
    weight    FLOAT{sysc}
)""",
        f"""CREATE TABLE IF NOT EXISTS edges_relates (
    edge_id   BIGINT  NOT NULL,
    src       BIGINT  NOT NULL,
    dst       BIGINT  NOT NULL,
    tenant_id INTEGER NOT NULL,
    rel_kind  VARCHAR{sysc}
)""",
        # SUPERSEDES is a record of a write, so it carries tx time only -- it is never re-dated.
        """CREATE TABLE IF NOT EXISTS edges_supersedes (
    edge_id   BIGINT  NOT NULL,
    src       BIGINT  NOT NULL,
    dst       BIGINT  NOT NULL,
    tenant_id INTEGER NOT NULL,
    tx_from   TIMESTAMP,
    writer    VARCHAR
)""",
    ]
    # ---- full text (schema v3) --------------------------------------------------------
    stmts += _fts_table_ddl()
    stmts += [
        # ---- views --------------------------------------------------------------------
        # One row per direction so a 1-hop expansion is a plain equality lookup (no OR-join).
        # NOTE: this view is the CURRENT-state view used by the benchmarked recall path; the
        # as-of paths build their own predicate instead of using it.
        """CREATE OR REPLACE VIEW relates_undirected AS
    SELECT tenant_id, src AS a, dst AS b FROM edges_relates WHERE valid_to IS NULL AND tx_to IS NULL
    UNION ALL
    SELECT tenant_id, dst AS a, src AS b FROM edges_relates WHERE valid_to IS NULL AND tx_to IS NULL""",
    ]
    return stmts


def table_ddl(name: str, config: SchemaConfig | None = None, *, as_table: str | None = None) -> str:
    """The exact ``CREATE TABLE`` statement :func:`ddl_statements` emits for a built-in table.

    ``as_table`` renames the target (and drops ``IF NOT EXISTS``), which is how
    :meth:`anatid.Anatid.recluster` rebuilds a table in a new physical order **without** losing
    its ``NOT NULL`` constraints and column ``DEFAULT``s.  ``CREATE TABLE ... AS SELECT``, the
    obvious way to write that, silently drops both.

    Note for rebuild paths: the ``entities`` DDL contains a GENERATED column, which may not be
    named in the following ``INSERT`` -- take the column list from :func:`insertable_columns`.
    """
    prefix = f"CREATE TABLE IF NOT EXISTS {name} ("
    for stmt in ddl_statements(config):
        if stmt.startswith(prefix):
            if as_table is None:
                return stmt
            return f"CREATE TABLE {quote_ident(as_table)} (" + stmt[len(prefix):]
    raise KeyError(f"no built-in DDL for table {name!r}")


def index_statements(config: SchemaConfig | None = None) -> list[str]:
    """CREATE INDEX statements for the configured index set (tuning, caller-selectable)."""
    cfg = config or SchemaConfig()
    out = []
    for name in cfg.indexes:
        out.append(DEFAULT_INDEXES.get(name) or OPTIONAL_INDEXES[name])
    return out


def required_index_statements() -> list[str]:
    """CREATE INDEX statements that are contract, not tuning.  Always run by :func:`ensure_schema`."""
    return list(REQUIRED_INDEXES.values())


# --------------------------------------------------------------------------- migrations

MigrationFn = Callable[["object"], None]

#: ``version -> callable(con)`` that upgrades a database FROM ``version - 1`` TO ``version``.
#: Version 1 is creation, handled by :func:`ensure_schema`; later versions register here.
MIGRATIONS: dict[int, MigrationFn] = {}


def register_migration(to_version: int) -> Callable[[MigrationFn], MigrationFn]:
    """Decorator registering a migration step to ``to_version`` (the hook future versions use)."""

    def deco(fn: MigrationFn) -> MigrationFn:
        if to_version in MIGRATIONS:
            raise ValueError(f"migration to version {to_version} already registered")
        MIGRATIONS[to_version] = fn
        return fn

    return deco


def _in_transaction(con) -> bool:
    """Is ``con`` already inside an explicit transaction?

    Asked by *reading*, never by trying ``BEGIN``: DuckDB has no nested transactions, and the
    ``TransactionException`` a nested ``BEGIN`` raises does not merely fail -- it ABORTS the
    caller's transaction, so every following statement dies with "Current transaction is aborted
    (please ROLLBACK)".  ``current_transaction_id()`` is stable inside an explicit transaction
    and changes between two autocommit statements, which tells the two apart without touching
    anything.  If the function is unavailable, say "yes": running a migration non-atomically is
    survivable, poisoning the caller's transaction is not.
    """
    try:
        first = con.execute("SELECT current_transaction_id()").fetchone()[0]
        second = con.execute("SELECT current_transaction_id()").fetchone()[0]
    except Exception:
        return True
    return first == second


@contextlib.contextmanager
def _transaction(con):
    """Run migrations atomically, whether or not the caller already opened a transaction.

    When the caller is already in one (``Anatid.ensure_schema`` wraps this call), that
    transaction supplies the atomicity and this rides along.
    """
    started = not _in_transaction(con)
    if started:
        con.execute("BEGIN TRANSACTION")
    try:
        yield
    except BaseException:
        if started:
            with contextlib.suppress(Exception):
                con.execute("ROLLBACK")
        raise
    else:
        if started:
            con.execute("COMMIT")


@register_migration(2)
def _migrate_1_to_2(con) -> None:
    """v1 -> v2.

    Two columns, both so that a *claim* the docs make can actually hold:

    * ``anatid_meta.fts_indexed_max_id`` -- the largest ``memory_id`` present at the last
      ``rebuild_fts_index()``.  v1 detected BM25 staleness with ``count(*)`` alone, so one insert
      plus one hard purge cancelled out and ``recall()`` reported ``bm25_stale=False`` while the
      new row was invisible to the text arm.
    * ``anatid_audit.related_memory_id`` -- the counterpart of a supersede, as a column.  v1 put
      it in the free-text ``reason`` (``"superseded by <id>"``), where ``forget(hard=True)``
      could not find it, so a purged id survived in another row's audit trail.
    """
    con.execute("ALTER TABLE anatid_meta ADD COLUMN IF NOT EXISTS fts_indexed_max_id BIGINT")
    con.execute("ALTER TABLE anatid_audit ADD COLUMN IF NOT EXISTS related_memory_id BIGINT")
    # Backfill the audit column from the v1 free text, then blank the text: the id belongs in the
    # column now, and leaving the copy behind would keep the erasure hole open on migrated files.
    con.execute(
        "UPDATE anatid_audit SET related_memory_id = "
        "  try_cast(substr(reason, 15) AS BIGINT), reason = 'superseded' "
        "WHERE action = 'supersede' AND reason LIKE 'superseded by %' "
        "  AND try_cast(substr(reason, 15) AS BIGINT) IS NOT NULL")


#: Sentinel used to compare NULLable temporal columns for equality during the entity merge.
_NEVER = "TIMESTAMP '9999-12-31 00:00:00'"


def _merge_duplicate_entities(con) -> dict[str, int]:
    """Collapse entities that share a canonical key inside one tenant.  Returns row counts.

    Lowest ``entity_id`` wins (it is the oldest: ids are time-ordered), every edge that pointed
    at a loser is repointed at the winner, an edge the repointing turned into an exact copy of an
    older edge is dropped (it says nothing the older edge does not), and the losers are deleted.
    Caller supplies the transaction.

    Nothing else is deleted.  In particular a ``RELATES_TO`` edge between two of the duplicates
    becomes a self-loop on the winner and is **kept**: it is the user's edge, the migration's job
    is to repoint, not to judge, and ``db.doctor()`` is where a self-loop gets reported.
    """
    key_outer = entity_key_sql("e.name")
    key_inner = entity_key_sql("name")
    con.execute("DROP TABLE IF EXISTS _anatid_entity_merge")
    con.execute(f"""CREATE TEMP TABLE _anatid_entity_merge AS
        SELECT e.tenant_id, e.entity_id AS loser, w.winner
        FROM entities e
        JOIN (SELECT tenant_id, {key_inner} AS k, min(entity_id) AS winner
              FROM entities WHERE name IS NOT NULL
              GROUP BY 1, 2 HAVING count(*) > 1) w
          ON w.tenant_id = e.tenant_id AND w.k = {key_outer}
        WHERE e.entity_id <> w.winner""")
    losers = int(con.execute("SELECT count(*) FROM _anatid_entity_merge").fetchone()[0])
    counts = {"entities_merged": losers, "about_repointed": 0, "relates_repointed": 0,
              "about_duplicates_removed": 0, "relates_duplicates_removed": 0}
    if losers == 0:
        con.execute("DROP TABLE IF EXISTS _anatid_entity_merge")
        return counts

    # Remember which edges we touch, so the duplicate cleanup below can only ever delete an edge
    # THIS migration created a duplicate of -- never a duplicate the file already had.
    con.execute("DROP TABLE IF EXISTS _anatid_repointed_about")
    con.execute("""CREATE TEMP TABLE _anatid_repointed_about AS
        SELECT a.edge_id FROM edges_about a
        JOIN _anatid_entity_merge m ON a.tenant_id = m.tenant_id AND a.dst = m.loser""")
    con.execute("DROP TABLE IF EXISTS _anatid_repointed_relates")
    con.execute("""CREATE TEMP TABLE _anatid_repointed_relates AS
        SELECT r.edge_id FROM edges_relates r
        JOIN _anatid_entity_merge m ON r.tenant_id = m.tenant_id
                                   AND (r.src = m.loser OR r.dst = m.loser)""")
    counts["about_repointed"] = int(
        con.execute("SELECT count(*) FROM _anatid_repointed_about").fetchone()[0])
    counts["relates_repointed"] = int(
        con.execute("SELECT count(*) FROM _anatid_repointed_relates").fetchone()[0])

    con.execute("""UPDATE edges_about SET dst = m.winner FROM _anatid_entity_merge m
        WHERE edges_about.tenant_id = m.tenant_id AND edges_about.dst = m.loser""")
    con.execute("""UPDATE edges_relates SET src = m.winner FROM _anatid_entity_merge m
        WHERE edges_relates.tenant_id = m.tenant_id AND edges_relates.src = m.loser""")
    con.execute("""UPDATE edges_relates SET dst = m.winner FROM _anatid_entity_merge m
        WHERE edges_relates.tenant_id = m.tenant_id AND edges_relates.dst = m.loser""")

    # A repointed edge that now says exactly what another edge says is noise: it would make
    # about_names() repeat an entity and double a 2-hop weight.  Which twin survives:
    #   * an edge that was NOT repointed -- it already pointed at the winner -- always wins over
    #     a repointed copy of it, whatever their ids or timestamps;
    #   * between two repointed copies, the one recorded first (tx_from) wins, with edge_id as
    #     the tie-break.  anatid's own ids are time-ordered, but an edge_id can be hand-assigned,
    #     and a lower id does not by itself make an edge older.
    # Two identical edges that were both untouched by the merge are not the migration's to
    # judge (doctor() reports them as duplicate_live_edges).
    def survives_over(repointed: str) -> str:
        return (f"(b.edge_id NOT IN (SELECT edge_id FROM {repointed}) "
                f"OR coalesce(b.tx_from, {_NEVER}) < coalesce(a.tx_from, {_NEVER}) "
                f"OR (coalesce(b.tx_from, {_NEVER}) = coalesce(a.tx_from, {_NEVER}) "
                f"AND b.edge_id < a.edge_id))")

    counts["about_duplicates_removed"] = _count_changes(con.execute(f"""
        DELETE FROM edges_about WHERE edge_id IN (
            SELECT a.edge_id FROM edges_about a
            JOIN _anatid_repointed_about r ON r.edge_id = a.edge_id
            WHERE EXISTS (SELECT 1 FROM edges_about b
                          WHERE b.tenant_id = a.tenant_id AND b.src = a.src AND b.dst = a.dst
                            AND b.edge_id <> a.edge_id
                            AND {survives_over("_anatid_repointed_about")}
                            AND coalesce(b.valid_to, {_NEVER}) = coalesce(a.valid_to, {_NEVER})
                            AND coalesce(b.tx_to, {_NEVER}) = coalesce(a.tx_to, {_NEVER})))"""))
    # Same rule for RELATES_TO: only a repointed edge that is now a verbatim copy of another.
    counts["relates_duplicates_removed"] = _count_changes(con.execute(f"""
        DELETE FROM edges_relates WHERE edge_id IN (
            SELECT a.edge_id FROM edges_relates a
            JOIN _anatid_repointed_relates r ON r.edge_id = a.edge_id
            WHERE EXISTS (SELECT 1 FROM edges_relates b
                          WHERE b.tenant_id = a.tenant_id AND b.src = a.src AND b.dst = a.dst
                            AND coalesce(b.rel_kind, '') = coalesce(a.rel_kind, '')
                            AND b.edge_id <> a.edge_id
                            AND {survives_over("_anatid_repointed_relates")}
                            AND coalesce(b.valid_to, {_NEVER}) = coalesce(a.valid_to, {_NEVER})
                            AND coalesce(b.tx_to, {_NEVER}) = coalesce(a.tx_to, {_NEVER})))"""))

    con.execute("""DELETE FROM entities WHERE entity_id IN (
        SELECT loser FROM _anatid_entity_merge m WHERE m.tenant_id = entities.tenant_id)""")
    for t in ("_anatid_entity_merge", "_anatid_repointed_about", "_anatid_repointed_relates"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    return counts


def _count_changes(result) -> int:
    row = result.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _rebuild_entities_with_key(con) -> None:
    """Add the generated ``entity_key`` column to an existing ``entities`` table.

    DuckDB cannot ``ALTER TABLE ... ADD COLUMN ... GENERATED`` (``Parser Error: Adding generated
    columns after table creation is not supported yet``), so the table is rebuilt.  The column
    list is read from the catalog rather than assumed, so a file created with
    ``system_columns=False`` or with a hand-added column migrates intact.
    """
    info = con.execute("PRAGMA table_info(entities)").fetchall()
    if any(r[1] == ENTITY_KEY_COLUMN for r in info):
        return
    defs: list[str] = []
    cols: list[str] = []
    for _cid, name, ctype, notnull, dflt, _pk in info:
        cols.append(quote_ident(name))
        try:
            d = f"{quote_ident(name)} {check_type(ctype)}"
        except ValueError as exc:
            raise ValueError(
                f"cannot migrate entities to schema v3: column {name!r} has a type this build "
                f"will not re-emit ({ctype!r}). Drop or simplify the column, then reopen.") from exc
        if notnull:
            d += " NOT NULL"
        if dflt is not None:
            d += f" DEFAULT {dflt}"
        defs.append(d)
    collist = ", ".join(cols)
    con.execute("DROP TABLE IF EXISTS entities__v3")
    con.execute("CREATE TABLE entities__v3 (\n    " + ",\n    ".join(defs)
                + _ENTITY_KEY_DDL + "\n)")
    con.execute(f"INSERT INTO entities__v3 ({collist}) SELECT {collist} FROM entities "
                f"ORDER BY {CLUSTER_ORDER['entities']}")
    con.execute("DROP TABLE entities")
    con.execute("ALTER TABLE entities__v3 RENAME TO entities")


@register_migration(3)
def _migrate_2_to_3(con) -> None:
    """v2 -> v3.  Two correctness fixes, one transaction.

    **Cross-tenant BM25.**  v2 built the fts index over ``memories(memory_id)``.  The fts
    extension needs a document key that is unique over the indexed table, and ``memory_id`` is
    unique only *within* a tenant, so two tenants writing the same id gave the index one
    document for two texts and ``recall()`` returned the wrong tenant's row for a term the
    caller's tenant never used.  v3 adds :data:`FTS_SOURCE_TABLE` (key
    ``'<tenant>:<memory>'``) with per-tenant df/idf sidecars, and this step DROPS the v2
    ``fts_main_memories`` schema outright: half-migrated files must not keep serving from a
    leaking index.  BM25 stays empty until the next ``rebuild_fts_index()``; the watermarks in
    ``anatid_meta`` are cleared so ``fts_status()`` says so instead of claiming to be fresh.

    **Entity identity.**  v2 created entities with an unguarded ``SELECT`` then ``INSERT``, so
    concurrent ``remember()`` calls naming one new entity minted several rows and the graph
    fractured.  v3 adds the generated ``entity_key`` column and ``UNIQUE (tenant_id,
    entity_key)``.  Existing duplicates are merged first -- lowest ``entity_id`` wins, edges are
    repointed, losers deleted -- because the index cannot be created over duplicate keys
    (``Constraint Error: Data contains duplicates on indexed column(s)``).
    """
    for stmt in _fts_table_ddl():
        con.execute(stmt)
    con.execute("DROP SCHEMA IF EXISTS fts_main_memories CASCADE")
    con.execute("UPDATE anatid_meta SET fts_indexed_at = NULL, fts_indexed_rows = NULL, "
                "fts_indexed_max_id = NULL")
    con.execute("UPDATE anatid_meta SET contract = ?", ["\n".join(CONTRACT_NOTES)])
    _merge_duplicate_entities(con)
    _rebuild_entities_with_key(con)
    for stmt in required_index_statements():
        con.execute(stmt)
    con.execute(DEFAULT_INDEXES["idx_entities_name"])


def current_version(con) -> int | None:
    """Schema version stored in the file, or ``None`` when the file has no anatid schema yet."""
    tables = {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()}
    if "anatid_meta" not in tables:
        return None
    row = con.execute("SELECT schema_version FROM anatid_meta LIMIT 1").fetchone()
    return None if row is None else int(row[0])


def table_names(con) -> set[str]:
    """Names of the tables in the ``main`` schema of ``con``."""
    return {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()}


def _create_required_indexes(con) -> None:
    """Create the contract indexes, turning a duplicate-data failure into a readable error."""
    import duckdb as _duckdb

    for stmt in required_index_statements():
        try:
            con.execute(stmt)
        except _duckdb.ConstraintException as exc:
            from .errors import AnatidError
            raise AnatidError(
                "cannot enforce UNIQUE (tenant_id, entity_key) on entities: the file already "
                "contains two entities with the same canonical name in one tenant. Open the "
                "file with a v2 anatid build to inspect it, or de-duplicate with "
                f"anatid.schema._merge_duplicate_entities(con). DuckDB said: {exc}") from exc


def ensure_schema(con, config: SchemaConfig | None = None, *, anatid_version: str = "0.0.0") -> int:
    """Create the schema if missing, migrate it if behind, verify it if current.

    Returns the schema version now in the file.  Raises :class:`SchemaVersionError` when the file
    was written by a newer anatid, or when a migration step is missing.
    """
    import duckdb as _duckdb

    from .types import utcnow

    cfg = config or SchemaConfig()
    found = current_version(con)

    if found is None:
        for stmt in ddl_statements(cfg):
            con.execute(stmt)
        for stmt in index_statements(cfg):
            con.execute(stmt)
        _create_required_indexes(con)
        if con.execute("SELECT count(*) FROM anatid_meta").fetchone()[0] == 0:
            con.execute(
                "INSERT INTO anatid_meta (schema_version, created_at, embedding_dim, anatid_version,"
                " duckdb_version, system_columns, fts_indexed_at, fts_indexed_rows,"
                " fts_indexed_max_id, contract)"
                " VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?)",
                [SCHEMA_VERSION, utcnow(), cfg.embedding_dim, anatid_version,
                 _duckdb.__version__, cfg.system_columns, "\n".join(CONTRACT_NOTES)],
            )
        return SCHEMA_VERSION

    if found > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"database schema version {found} is newer than this anatid build "
            f"(supports {SCHEMA_VERSION}); anatid never downgrades a file",
            found=found, expected=SCHEMA_VERSION)

    if found < SCHEMA_VERSION:
        # One transaction for the whole ladder: a file is never left on a half-applied step, and
        # the recorded version moves with the data it describes.
        with _transaction(con):
            while found < SCHEMA_VERSION:
                step = MIGRATIONS.get(found + 1)
                if step is None:
                    raise SchemaVersionError(
                        f"no migration registered from schema version {found} to {found + 1}",
                        found=found, expected=SCHEMA_VERSION)
                step(con)
                con.execute("UPDATE anatid_meta SET schema_version = ?", [found + 1])
                found += 1

    # Make sure anything added since creation exists (views, new default indexes).
    for stmt in ddl_statements(cfg):
        if stmt.startswith("CREATE OR REPLACE VIEW") or "IF NOT EXISTS" in stmt:
            con.execute(stmt)
    for stmt in index_statements(cfg):
        con.execute(stmt)
    _create_required_indexes(con)
    return found


def embedding_dim(con) -> int | None:
    """Embedding dimension recorded in ``anatid_meta``, or None if there is no anatid schema."""
    if "anatid_meta" not in table_names(con):
        return None
    row = con.execute("SELECT embedding_dim FROM anatid_meta LIMIT 1").fetchone()
    return None if row is None else int(row[0])


def missing_tables(con, expected: Iterable[str] = ALL_TABLES) -> list[str]:
    """Which of ``expected`` are absent from ``con`` (used by health checks and tests)."""
    have = table_names(con)
    return [t for t in expected if t not in have]


# --------------------------------------------------------------------------- temporal predicate

def temporal_predicate(alias: str, as_of, *, valid_only: bool = False) -> tuple[str, list]:
    """WHERE fragment + bind params selecting the rows visible under ``as_of``.

    This -- not any engine feature -- is anatid's time travel.  DuckDB has no ``AS OF SYSTEM
    TIME``; the visibility rule is compiled here and pasted into every scoped query:

    * current (``as_of`` is :data:`anatid.types.CURRENT`): ``valid_to IS NULL AND tx_to IS NULL``
    * as of T (valid time): ``valid_from <= T AND (valid_to IS NULL OR valid_to > T)``
    * as of T (transaction time): ``tx_from <= T AND (tx_to IS NULL OR tx_to > T)``

    Intervals are half-open ``[from, to)``: a row closed exactly at T is already invisible at T,
    and a row opened exactly at T is already visible.  ``valid_only=True`` skips the tx-time half
    for tables that only have valid-time columns.
    """
    if as_of is None or as_of.is_current:
        return (f"{alias}.valid_to IS NULL" if valid_only
                else f"{alias}.valid_to IS NULL AND {alias}.tx_to IS NULL"), []
    parts: list[str] = []
    params: list = []
    if as_of.valid_time is not None:
        parts.append(f"{alias}.valid_from <= ? AND ({alias}.valid_to IS NULL OR {alias}.valid_to > ?)")
        params += [as_of.valid_time, as_of.valid_time]
    else:
        parts.append(f"{alias}.valid_to IS NULL")
    if not valid_only:
        if as_of.tx_time is not None:
            parts.append(f"{alias}.tx_from <= ? AND ({alias}.tx_to IS NULL OR {alias}.tx_to > ?)")
            params += [as_of.tx_time, as_of.tx_time]
        else:
            parts.append(f"{alias}.tx_to IS NULL")
    return " AND ".join(parts), params
