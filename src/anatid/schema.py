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
DuckDB has no ``AS OF SYSTEM TIME``.  These columns plus the WHERE clauses anatid compiles from
:mod:`anatid.visibility` *are* the time-travel mechanism.

Versions (schema v4)
--------------------
:data:`VERSIONED_TABLES` (``memories``, ``edges_about``, ``edges_relates``) carry a
``version INTEGER NOT NULL DEFAULT 1`` column (:data:`VERSION_COLUMN`).  ``memory_id`` and
``edge_id`` stay the logical ids callers hold; ``(tenant_id, memory_id, version)`` identifies a
physical row.  Rows are immutable: a correction closes the current version's ``tx_to`` and
inserts version ``n + 1`` with the corrected valid interval, so the transaction axis records
what the database believed before the correction.  Before v4 ``supersede`` and soft ``forget``
rewrote ``valid_to`` in place and no ``as_of`` could see the earlier belief.  The 3->4
migration carries every existing row over as version 1 with its ``tx_to`` untouched; the
history rewritten before the migration is not recoverable.  Column order is unchanged and
``version`` is last, so ``Memory.from_row`` reads the same positions as before.

Derived indexes (schema v4)
---------------------------
Three catalog tables and one sequence carry the derived-index framework of
``docs/design/derived-index-framework.md``.  :data:`INDEX_GENERATIONS_TABLE` records every base
generation an accelerator has built, with its watermark and its ``validated`` / ``published``
flags.  :data:`INDEX_REGISTRY_TABLE` records the index DEFINITIONS, in the file rather than on
one handle, so every handle that writes into the file journals for every defined index whether
or not it holds that accelerator's code.  :data:`INDEX_JOURNAL_TABLE` is that journal: one
ordered row per change, written by the verbs in the same transaction as the canonical row and
numbered from :data:`INDEX_CHANGE_SEQUENCE`, so a read that merges base + journal sees every
write immediately.  The journal is ordered rather than split into a delta set and a tombstone
set because two independent id sets cannot represent a document id that is purged and then
reused: it would be in both, and the merge would drop it.  Every row of both is keyed by
``(tenant_id, doc_id)``, including for an index whose generations cover the file, so one
tenant's change never suppresses another tenant's document with the same id.  See
:mod:`anatid.derived`.
"""

from __future__ import annotations

import contextlib
import re
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .errors import SchemaVersionError
from .visibility import current_row_sql, temporal_predicate

__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_RELEASES",
    "DEFAULT_EMBEDDING_DIM",
    "SchemaConfig",
    "SYSTEM_COLUMNS",
    "MEMORY_COLUMNS",
    "ENTITY_COLUMNS",
    "EPISODE_COLUMNS",
    "EDGE_ABOUT_COLUMNS",
    "EDGE_RELATES_COLUMNS",
    "VERSION_COLUMN",
    "VERSIONED_TABLES",
    "ensure_version_columns",
    "LEGACY_VERSION_SELECT",
    "versioned_tables",
    "has_version_column",
    "memory_select",
    "version_expr",
    "forget_column_probes",
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
    "INDEX_GENERATIONS_TABLE",
    "INDEX_REGISTRY_TABLE",
    "INDEX_JOURNAL_TABLE",
    "INDEX_CHANGE_SEQUENCE",
    "INDEX_TABLES",
    "INDEX_GENERATION_ADDED_COLUMNS",
    "ensure_index_columns",
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

SCHEMA_VERSION = 4
DEFAULT_EMBEDDING_DIM = 1536

#: The anatid release that introduced each schema version.  A file records the version of the
#: build that wrote it, and a build refuses a file newer than its own :data:`SCHEMA_VERSION`, so
#: "which release can open this file" is a real question with a real answer.  Two builds sharing
#: a version string while writing different schema versions makes that answer a lie, which is
#: what shipping schema v4 under the name 0.1.1 would have done; ``tests/test_core.py`` checks
#: this table against :data:`anatid.__version__`.
SCHEMA_VERSION_RELEASES: dict[int, str] = {1: "0.1.0", 2: "0.1.0", 3: "0.1.1", 4: "0.2.0"}

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

#: Name of the version column on every table in :data:`VERSIONED_TABLES` (schema v4).
VERSION_COLUMN = "version"

#: The tables whose rows are immutable versions of a logical id.  ``entities`` and
#: ``episodes`` are not corrected by any verb and carry no version column.
VERSIONED_TABLES: tuple[str, ...] = ("memories", "edges_about", "edges_relates")

#: Column order used by ``SELECT`` for a full memory row (matches :meth:`anatid.Memory.from_row`).
#: ``version`` is last so the positions a v3 build read are unchanged.
MEMORY_COLUMNS: tuple[str, ...] = (
    "memory_id", "tenant_id", "content", "kind", "embedding", "created_at",
    "valid_from", "valid_to", "tx_from", "tx_to", "writer", "episode_id", "confidence",
    "access_count", "last_access_at", VERSION_COLUMN,
)

#: Every column of ``edges_about`` / ``edges_relates`` in table order.  The verbs copy a closed
#: edge's version into its successor by name from these, so a column added later is carried
#: over instead of silently defaulting.
EDGE_ABOUT_COLUMNS: tuple[str, ...] = (
    "edge_id", "src", "dst", "tenant_id", "weight",
    "valid_from", "valid_to", "tx_from", "tx_to", "writer", "episode_id", "confidence",
    VERSION_COLUMN,
)
EDGE_RELATES_COLUMNS: tuple[str, ...] = (
    "edge_id", "src", "dst", "tenant_id", "rel_kind",
    "valid_from", "valid_to", "tx_from", "tx_to", "writer", "episode_id", "confidence",
    VERSION_COLUMN,
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

# --------------------------------------------------------------------------- derived indexes

#: One row per base generation of a derived index (schema v4).  ``tenant_id`` is NULL for an
#: index whose generations cover the whole file.  ``published`` is the single flag a read pins
#: on; ``validated`` is cleared by a bulk load and set by a successful validation.
INDEX_GENERATIONS_TABLE = "anatid_index_generations"

#: One row per derived index DEFINED in this file, whatever handle defined it.  The verbs
#: journal a write for every ENABLED row here, so a second handle that holds no accelerator
#: code still keeps the journal complete.  See :mod:`anatid.derived`.
INDEX_REGISTRY_TABLE = "anatid_index_registry"

#: The ordered change journal: one row per (index, tenant, document, change), written in the
#: same transaction as the canonical write.  ``op`` is ``insert`` or ``close`` and the row with
#: the largest ``change_seq`` for a ``(tenant_id, doc_id)`` wins, so a document id that is
#: purged and then reused is represented exactly.  ``absorbed_by`` is the generation whose build
#: already reflects the change; the row is deleted once no older generation is alive.
INDEX_JOURNAL_TABLE = "anatid_index_journal"

#: The sequence behind ``anatid_index_journal.change_seq``.  One sequence per file, so the
#: order is total over every index, tenant and handle writing into that file.
INDEX_CHANGE_SEQUENCE = "anatid_index_change_seq"

#: The derived-index catalog tables, in creation order.
INDEX_TABLES: tuple[str, ...] = (
    INDEX_GENERATIONS_TABLE, INDEX_REGISTRY_TABLE, INDEX_JOURNAL_TABLE)

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


def fts_rebuild_statements(*, terms_index: bool = True, con=None) -> list[str]:
    """Every statement that rebuilds the BM25 index, in order.  Run them in one transaction.

    The DuckDB fts index is not incremental -- ``PRAGMA create_fts_index`` rebuilds it wholesale
    -- so the source table is refreshed wholesale in the same breath, and the per-tenant
    statistics are derived from the postings that build produced.

    Order matters:

    1. refill :data:`FTS_SOURCE_TABLE` from ``memories``, keyed ``'<tenant>:<memory>'``.  One
       row per ``(tenant_id, memory_id)``, taken from the newest version: the versions of one
       memory share its content, and a repeated document key is exactly the bug this key was
       introduced to fix.  Duplicate ids inside one tenant are an integrity fault
       (``db.doctor()`` reports them) and do not become two documents either.
    2. build the index, and the ART index on its postings.
    3. materialise ``docid -> (tenant_id, memory_id, len)``.
    4. materialise per-tenant ``df`` and per-tenant ``(num_docs, avgdl)``.
    5. drop the schema-v2 ``fts_main_memories`` index if a v2 build left one behind: it is keyed
       on ``memory_id`` and leaks across tenants, so it must never coexist with this one.

    Every ``memories`` row is indexed, live or superseded, exactly as in v2 -- the temporal
    predicate is applied by the query, so ``as_of`` BM25 keeps working.

    ``con`` is the connection the statements will run on, and is used only to render the
    version expression: a file that predates schema v4 has no ``version`` column, and the
    de-duplication then falls through to ``tx_from``.  Omit it for a v4 file.
    """
    version = VERSION_COLUMN if con is None else version_expr(con)
    stmts = [
        f"DELETE FROM {FTS_SOURCE_TABLE}",
        f"""INSERT INTO {FTS_SOURCE_TABLE} (fts_doc_id, tenant_id, memory_id, content)
    SELECT {FTS_DOC_ID_SQL}, tenant_id, memory_id, content
    FROM memories
    QUALIFY row_number() OVER (PARTITION BY tenant_id, memory_id
                               ORDER BY {version} DESC NULLS LAST,
                                        tx_from DESC NULLS LAST,
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
     (f"DELETE FROM {FTS_INDEX_SCHEMA}.dict WHERE termid IN ("
      f"SELECT t.termid FROM {FTS_INDEX_SCHEMA}.terms t WHERE t.termid IN ("
      f"SELECT p.termid FROM {FTS_INDEX_SCHEMA}.terms p JOIN {FTS_DOCS_TABLE} d ON d.docid = p.docid "
      f"WHERE d.tenant_id = ? AND d.memory_id = ?) "
      f"GROUP BY t.termid HAVING count(*) = count(*) FILTER (WHERE t.docid IN ("
      f"SELECT docid FROM {FTS_DOCS_TABLE} WHERE tenant_id = ? AND memory_id = ?)))")),
    (("index", FTS_DOCS_TABLE),
     (f"DELETE FROM {FTS_INDEX_SCHEMA}.terms WHERE docid IN "
      f"(SELECT docid FROM {FTS_DOCS_TABLE} WHERE tenant_id = ? AND memory_id = ?)")),
    (("index",),
     (f"DELETE FROM {FTS_INDEX_SCHEMA}.docs WHERE name = "
      f"CAST(? AS VARCHAR) || ':' || CAST(? AS VARCHAR)")),
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
CATALOG_TABLES: tuple[str, ...] = ("anatid_meta", "anatid_audit") + FTS_TABLES + INDEX_TABLES
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
    ("time travel: DuckDB has NO 'AS OF SYSTEM TIME'. as_of is anatid's own WHERE filter over "
     "valid_from/valid_to and tx_from/tx_to. Intervals are half-open [from, to)."),
    ("isolation: DuckDB has NO schema-level or row-level access control. Per-tenant ISOLATION is "
     "file-per-tenant, enforced by anatid's wrapper and the filesystem. tenant_id inside one file "
     "is SCOPING, not isolation: any connection to the file can read every tenant in it."),
    ("full-text: the DuckDB fts index is NOT incremental, so anatid does not depend on it for "
     "freshness. Since schema v4 the text arm is a derived index (anatid.fts), which "
     "Anatid.open() attaches by default: a published base generation plus a journal written "
     "inside the transaction that writes the memory, so a row is searchable by the very next "
     "recall() with no rebuild, on any handle on this file, and a supersede or a forget takes "
     "effect on that same read. rebuild_fts_index() compacts the journal into a new generation, "
     "which buys read latency. On a file whose handles never attached that index "
     "(Anatid.open(accelerators=False)) 0.1.1's file-wide index answers instead, and there rows "
     "inserted after PRAGMA create_fts_index ARE invisible to BM25 until rebuild_fts_index() "
     "runs. anatid records both the indexed row count and the largest indexed memory_id in this "
     "table and reports the state on every recall() result. A raw SQL UPDATE of memories.content "
     "(something no anatid verb ever issues) changes neither watermark and is NOT detected."),
    ("full-text scoping (schema v3): the index is built over anatid_fts_documents, whose document "
     "key is '<tenant_id>:<memory_id>' -- unique across tenants, which memory_id alone is not. "
     "df/idf and (num_docs, avgdl) come from anatid_fts_dict / anatid_fts_stats, which are "
     "computed PER TENANT, so one tenant's BM25 scores depend on that tenant's rows only and no "
     "term statistic crosses the boundary through recall(). The fts extension's own dict/stats "
     "tables under fts_main_anatid_fts_documents still count every tenant; anatid reads only the "
     "term->termid mapping from them, never a df or a corpus size, and a raw SQL reader of the file "
     "can see them exactly as it can see every tenant's memories (scoping, not isolation). Schema "
     "v2 files indexed memories(memory_id) and did leak: the 2->3 migration drops that index."),
    ("entity identity (schema v3): entities.entity_key is a GENERATED column, "
     "trim(regexp_replace(lower(name), '\\s+', ' ', 'g')), with UNIQUE (tenant_id, entity_key) "
     "enforced by an index. Concurrent remember() calls naming the same new entity can no longer "
     "produce two entity rows; the losers get a retryable constraint/transaction error."),
    ("transactions: DuckDB MVCC is optimistic and snapshot-isolated, NOT serializable. Appends "
     "never conflict; two concurrent updates to the SAME row abort the second with a retryable "
     "error, surfaced as anatid.errors.ConflictError."),
    ("vector search: brute-force array_cosine_similarity over FLOAT[N]. Fine to roughly 1e5 "
     "memories per tenant; beyond that latency grows linearly with the tenant's row count."),
    ("erasure: forget(hard=True) purges the memory GRAPH row, its edges, its embedding, its "
     "orphaned episode, its provenance and its BM25 documents (anatid_fts_documents holds content "
     "verbatim, so anatid.schema.fts_purge() is part of the purge), and no anatid_audit row "
     "survives that references the purged memory_id. It does NOT reach tables anatid does not own. "
     "Conversation transcripts (anatid.integrations.openai_agents.AnatidSession's agent_messages) "
     "quote memory content and ids verbatim; AnatidSession registers an erasure hook so a purge "
     "removes those rows too, but any other table you write into this file is yours to clean -- "
     "see Anatid.erasure_hooks."),
    ("versions (schema v4): rows of memories, edges_about and edges_relates are immutable. "
     "memory_id / edge_id is the logical id; version numbers the physical rows of one logical id "
     "from 1. supersede, soft forget, unrelate and a reinforce that changes confidence close the "
     "current version's tx_to and insert the next version in the same transaction; nothing "
     "rewrites valid_to in place. as_of(valid_time, tx_time) therefore returns the version the "
     "database believed at tx_time, and get() by id returns the live version (tx_to IS NULL). "
     "access_count and last_access_at are usage counters, updated in place on the live version, "
     "and are not bitemporal. Rows that existed before the 3->4 migration are version 1 with "
     "tx_to NULL; the corrections made to them before the migration were rewritten in place and "
     "are not recoverable."),
    ("derived indexes (schema v4): every accelerator (full-text, CSR, vector) is a versioned base "
     "generation recorded in anatid_index_generations plus an ORDERED journal "
     "(anatid_index_journal, numbered from the anatid_index_change_seq sequence) written in the "
     "SAME transaction as the canonical row. Every journal row is keyed by (tenant_id, doc_id) "
     "and the newest op for a key wins, so a purged and reused id is represented exactly and one "
     "tenant's change never suppresses another tenant's document with the same id. The index "
     "DEFINITIONS live in anatid_index_registry, in the FILE: every handle that writes journals "
     "for every enabled definition, whether or not it holds that accelerator's code. A read "
     "merges base + journal and THEN applies the tenant and time predicate from "
     "anatid.visibility, so an index that is stale, corrupt or absent narrows candidates badly but "
     "never makes an answer wrong; the SQL path over the canonical tables is always the oracle. "
     "Publication of a generation is one row switch inside a transaction and a read pins one "
     "generation for its duration. A current-state index cannot answer an as_of query; those "
     "always take the SQL path and the fallback reason is reported. forget(hard=True) deletes the "
     "document from every generation's storage and from the journal, and invalidates any "
     "generation whose storage cannot delete, so an erasure reaches the accelerators too."),
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

#: The version-column clause appended to every table in :data:`VERSIONED_TABLES`.  Last in the
#: column list for the same reason.  The DEFAULT is what lets a bulk load, a raw INSERT and the
#: 3->4 migration leave the column alone and get version 1.
_VERSION_DDL = f",\n    {VERSION_COLUMN:<12} INTEGER NOT NULL DEFAULT 1"


def ensure_version_columns(con) -> list[str]:
    """Add :data:`VERSION_COLUMN` to every :data:`VERSIONED_TABLES` member that lacks it.

    Returns the tables that were altered.  ``ALTER TABLE ... ADD COLUMN`` cannot carry
    ``NOT NULL`` in DuckDB (``Adding columns with constraints not yet supported``), so a
    migrated table gets ``version INTEGER DEFAULT 1``: existing rows are backfilled with 1 and
    every later insert that omits the column gets 1.  The verbs read ``coalesce(version, 1)``
    where they compute a successor, so a NULL written by raw SQL into a migrated file is
    treated as version 1 rather than poisoning the chain.  Run by the 3->4 migration and by
    :func:`ensure_schema` on every open, so a v4 file written by a build that predates the
    column is repaired on the next open.

    Inside a transaction this must run before any statement modifies rows of these tables:
    DuckDB cannot commit a transaction that updates a table and then alters it.
    """
    present = table_names(con)
    altered: list[str] = []
    for table in VERSIONED_TABLES:
        if table not in present:
            continue
        cols = {r[1] for r in con.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()}
        if VERSION_COLUMN in cols:
            continue
        con.execute(f"ALTER TABLE {quote_ident(table)} ADD COLUMN IF NOT EXISTS "
                    f"{VERSION_COLUMN} INTEGER DEFAULT 1")
        altered.append(table)
    if altered:
        forget_column_probes()
    return altered


# ------------------------------------------------------------- reading a pre-v4 file

# A handle that cannot run the migration ladder still has to answer reads.  There are two such
# handles: ``read_only=True`` (the file may be on a read-only mount, or another process holds
# the single writer slot) and ``ensure=False`` (the caller said not to touch the file).  Both
# are supported open modes, and ``anatid-mcp --read-only`` is the deployment that uses the
# first, so a schema-v3 file has to keep answering ``get`` / ``recall`` / ``provenance`` there.
#
# Exactly one column separates a v3 table from a v4 one: ``version``.  A select list that names
# it fails to bind against a v3 file with a raw DuckDB ``BinderException``, which is neither an
# anatid error nor recoverable.  So the select list is asked for rather than assembled from
# MEMORY_COLUMNS by hand, and against a pre-v4 table it renders ``1 AS version`` -- which is
# the truth: every row of a table that predates the column is version 1, because nothing had
# ever inserted a second version of a logical id.
#
# The probe is memoised per connection, so it costs one dict lookup on a read.  It is a probe
# and not a read of ``anatid_meta.schema_version`` on purpose: a v4 file written by a build
# that predates the column has the same shape as a v3 one, and the column is what the query
# binds against.

#: What a select list renders in place of :data:`VERSION_COLUMN` on a table that predates it.
LEGACY_VERSION_SELECT = f"1 AS {VERSION_COLUMN}"

_COLUMN_PROBE: "weakref.WeakKeyDictionary[Any, tuple[int, frozenset[str]]]" = (
    weakref.WeakKeyDictionary()
)
_COLUMN_EPOCH = 0

#: Rendered select lists, keyed ``(alias, embedding, versioned)``.  Four entries in practice.
_SELECT_CACHE: dict[tuple[str, bool, bool], str] = {}


def forget_column_probes() -> None:
    """Discard every memoised column probe.  Called when a migration adds a column."""
    global _COLUMN_EPOCH
    _COLUMN_EPOCH += 1


def versioned_tables(con) -> frozenset[str]:
    """Which of :data:`VERSIONED_TABLES` carry :data:`VERSION_COLUMN` on this connection.

    Empty for a file written before schema v4.  Memoised per connection and invalidated by
    :func:`forget_column_probes`, which :func:`ensure_version_columns` calls when it alters a
    table, so a handle that migrates the file mid-session sees the new column at once.
    """
    hit = _COLUMN_PROBE.get(con)
    if hit is not None and hit[0] == _COLUMN_EPOCH:
        return hit[1]
    found: set[str] = set()
    for table in VERSIONED_TABLES:
        # A bind, not a catalog read: this asks the question the query itself will ask, which
        # is the one that matters under an ATTACHed catalog or a non-default search path.
        try:
            con.execute(f"SELECT {VERSION_COLUMN} FROM {quote_ident(table)} LIMIT 0")
        except Exception:  # noqa: BLE001 - a missing table has no version column either
            continue
        found.add(table)
    out = frozenset(found)
    with contextlib.suppress(TypeError):  # a connection object that cannot be weak-referenced
        _COLUMN_PROBE[con] = (_COLUMN_EPOCH, out)
    return out


def has_version_column(con, table: str = "memories") -> bool:
    """Whether ``table`` carries :data:`VERSION_COLUMN` on this connection."""
    return table in versioned_tables(con)


def memory_select(con, *, alias: str | None = None, embedding: bool = True) -> str:
    """The ``SELECT`` list for a full memory row, in :data:`MEMORY_COLUMNS` order.

    Use this rather than joining :data:`MEMORY_COLUMNS` directly: against a file written before
    schema v4 it renders ``1 AS version`` for the column that file does not have, so the row
    still arrives in the arity :meth:`anatid.Memory.from_row` expects.  ``embedding=False``
    replaces the vector with ``NULL AS embedding``, which is what every read that does not need
    it asks for.
    """
    versioned = has_version_column(con, "memories")
    key = (alias or "", bool(embedding), versioned)
    hit = _SELECT_CACHE.get(key)
    if hit is not None:
        return hit
    parts: list[str] = []
    for column in MEMORY_COLUMNS:
        if column == "embedding" and not embedding:
            parts.append("NULL AS embedding")
        elif column == VERSION_COLUMN and not versioned:
            parts.append(LEGACY_VERSION_SELECT)
        else:
            parts.append(f"{alias}.{column}" if alias else column)
    out = ", ".join(parts)
    _SELECT_CACHE[key] = out
    return out


def version_expr(con, table: str = "memories", alias: str | None = None) -> str:
    """``coalesce(version, 1)`` for a versioned table, ``1`` for one that predates the column.

    The expression a read orders or groups by.  ``coalesce`` because a table the 3->4 migration
    altered has a nullable column (DuckDB cannot add a NOT NULL one) and a NULL written there
    by raw SQL means version 1.
    """
    if table not in versioned_tables(con):
        # CAST rather than a bare 1: this expression is rendered into ORDER BY clauses, where
        # DuckDB reads an integer literal as a column ordinal.
        return "CAST(1 AS INTEGER)"
    column = f"{alias}.{VERSION_COLUMN}" if alias else VERSION_COLUMN
    return f"coalesce({column}, 1)"


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


#: Columns added to :data:`INDEX_GENERATIONS_TABLE` after the first v4 development files were
#: written.  ``ALTER TABLE ... ADD COLUMN`` cannot add ``NOT NULL`` in DuckDB, so readers of
#: these columns coalesce (see :mod:`anatid.derived`).
INDEX_GENERATION_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("published_unvalidated", "BOOLEAN DEFAULT FALSE"),
)


def ensure_index_columns(con) -> list[str]:
    """Add any :data:`INDEX_GENERATION_ADDED_COLUMNS` the file is missing.  Returns the names.

    Same shape as :func:`ensure_version_columns`: a v4 file written by an earlier development
    build has the generations table without ``published_unvalidated``, and every open repairs
    it rather than requiring a schema bump for a table no released version ever wrote.
    """
    if INDEX_GENERATIONS_TABLE not in table_names(con):
        return []
    have = {r[1] for r in con.execute(
        f"PRAGMA table_info({quote_ident(INDEX_GENERATIONS_TABLE)})").fetchall()}
    added = []
    for name, decl in INDEX_GENERATION_ADDED_COLUMNS:
        if name in have:
            continue
        con.execute(f"ALTER TABLE {quote_ident(INDEX_GENERATIONS_TABLE)} "
                    f"ADD COLUMN IF NOT EXISTS {quote_ident(name)} {decl}")
        added.append(name)
    return added


def _index_table_ddl() -> list[str]:
    """The derived-index catalog (schema v4).  See :mod:`anatid.derived` for the protocol."""
    return [
        # One row per generation.  No constraint, per the house rule; (index_name, tenant_id,
        # generation) is the identity and anatid.derived allocates generation numbers itself.
        # published_unvalidated records that publish(force=True) skipped the oracle check, which
        # is a different state from "invalidated" (validated FALSE with a note): the first is a
        # deliberate, usable-but-flagged generation, the second is one no read may use.
        f"""CREATE TABLE IF NOT EXISTS {INDEX_GENERATIONS_TABLE} (
    index_name   VARCHAR   NOT NULL,
    generation   INTEGER   NOT NULL,
    tenant_id    INTEGER,
    watermark_id BIGINT,
    watermark_ts TIMESTAMP,
    built_at     TIMESTAMP NOT NULL,
    validated    BOOLEAN   NOT NULL DEFAULT FALSE,
    published    BOOLEAN   NOT NULL DEFAULT FALSE,
    published_unvalidated BOOLEAN NOT NULL DEFAULT FALSE,
    stats        JSON,
    notes        VARCHAR
)""",
        # The index DEFINITIONS, in the file rather than on one handle.  The verbs journal a
        # write for every enabled row, so a handle that holds no code for an index still keeps
        # that index's journal complete; source_table and delta_mode are columns because they
        # are what such a handle needs in order to journal correctly.
        f"""CREATE TABLE IF NOT EXISTS {INDEX_REGISTRY_TABLE} (
    index_name       VARCHAR   NOT NULL,
    kind             VARCHAR   NOT NULL,
    per_tenant       BOOLEAN   NOT NULL DEFAULT TRUE,
    source_table     VARCHAR,
    source_id_column VARCHAR,
    delta_mode       VARCHAR   NOT NULL DEFAULT 'table',
    supports_delta   BOOLEAN   NOT NULL DEFAULT TRUE,
    params           JSON,
    created_at       TIMESTAMP NOT NULL,
    enabled          BOOLEAN   NOT NULL DEFAULT TRUE
)""",
        # One totally ordered journal, not two independent id sets: with a delta table and a
        # tombstone table an id that is purged and then reused appears in both and the merge
        # loses it.  change_seq comes from INDEX_CHANGE_SEQUENCE and the largest one for a
        # (tenant_id, doc_id) is the current state of that document.
        # absorbed_by: the generation whose build snapshot already contained this row.  Set by
        # the build transaction, so it is exact under MVCC: a canonical row and its journal row
        # are written in one transaction and a build snapshot sees both or neither.
        f"""CREATE TABLE IF NOT EXISTS {INDEX_JOURNAL_TABLE} (
    index_name   VARCHAR   NOT NULL,
    tenant_id    INTEGER   NOT NULL,
    doc_id       BIGINT    NOT NULL,
    change_seq   BIGINT    NOT NULL,
    op           VARCHAR   NOT NULL,
    written_at   TIMESTAMP NOT NULL,
    reason       VARCHAR,
    absorbed_by  INTEGER
)""",
    ]


def _index_sequence_ddl() -> list[str]:
    """The journal's ordering sequence.  One per file; gaps from rolled-back transactions are
    expected and harmless, only the order matters."""
    return [f"CREATE SEQUENCE IF NOT EXISTS {quote_ident(INDEX_CHANGE_SEQUENCE)} START 1"]


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
        # version (schema v4): rows are immutable versions of one memory_id, see the module
        # docstring.  Last, so every earlier column keeps its position.
        f"""CREATE TABLE IF NOT EXISTS memories (
    memory_id      BIGINT    NOT NULL,
    tenant_id      INTEGER   NOT NULL,
    content        VARCHAR,
    kind           VARCHAR,
    embedding      FLOAT[{dim}],
    created_at     TIMESTAMP NOT NULL{sysc},
    access_count   INTEGER   DEFAULT 0,
    last_access_at TIMESTAMP{_VERSION_DDL}
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
    weight    FLOAT{sysc}{_VERSION_DDL}
)""",
        f"""CREATE TABLE IF NOT EXISTS edges_relates (
    edge_id   BIGINT  NOT NULL,
    src       BIGINT  NOT NULL,
    dst       BIGINT  NOT NULL,
    tenant_id INTEGER NOT NULL,
    rel_kind  VARCHAR{sysc}{_VERSION_DDL}
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
    # ---- derived indexes (schema v4) --------------------------------------------------
    stmts += _index_sequence_ddl()
    stmts += _index_table_ddl()
    stmts += [
        # ---- views --------------------------------------------------------------------
        # One row per direction so a 1-hop expansion is a plain equality lookup (no OR-join).
        # NOTE: this view is the CURRENT-state view used by the benchmarked recall path; the
        # as-of paths build their own predicate instead of using it.  The predicate text comes
        # from anatid.visibility, the one place it is written.
        f"""CREATE OR REPLACE VIEW relates_undirected AS
    SELECT tenant_id, src AS a, dst AS b FROM edges_relates WHERE {current_row_sql()}
    UNION ALL
    SELECT tenant_id, dst AS a, src AS b FROM edges_relates WHERE {current_row_sql()}""",
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


@register_migration(4)
def _migrate_3_to_4(con) -> None:
    """v3 -> v4.  Immutable versions and the derived-index catalog.

    **Versions.**  Adds :data:`VERSION_COLUMN` to every :data:`VERSIONED_TABLES` member
    (:func:`ensure_version_columns`).  Every existing row becomes version 1 with its ``tx_to``
    untouched, which is ``NULL`` for every row a v3 build wrote: v3 never closed the
    transaction axis, it rewrote ``valid_to`` in place.  A memory that was superseded or
    forgotten before the migration therefore stays one version with ``valid_to`` set, and the
    belief it held before that correction is not recoverable.  Corrections made after the
    migration close the transaction axis and insert a successor, so from here on ``as_of``
    answers on both axes.  No row is rewritten and no data is copied.

    **Derived indexes.**  Adds :data:`INDEX_CHANGE_SEQUENCE`, :data:`INDEX_GENERATIONS_TABLE`,
    :data:`INDEX_REGISTRY_TABLE` and :data:`INDEX_JOURNAL_TABLE`, all empty.  The BM25 index a
    v3 file already has keeps working exactly as before (its ``anatid_meta`` watermark is
    unchanged) and reports ``absent`` through the framework until an accelerator builds its
    first generation.  Nothing is journalled until an index is registered, because the registry
    table is what says an index exists; from then on every handle writing into the file
    journals for it, so a generation built later is complete.

    The contract is restated with both notes.
    """
    ensure_version_columns(con)
    for stmt in _index_sequence_ddl():
        con.execute(stmt)
    for stmt in _index_table_ddl():
        con.execute(stmt)
    ensure_index_columns(con)
    con.execute("UPDATE anatid_meta SET contract = ?", ["\n".join(CONTRACT_NOTES)])


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
            # Structure before data.  DuckDB refuses to commit a transaction that modifies a
            # table's rows and then ALTERs it ("Attempting to modify table ... but another
            # transaction has altered this table"), and the 2->3 step rewrites edge rows, so
            # the v4 version columns go in before any step touches a row.  _migrate_3_to_4
            # calls this too and finds nothing left to do.
            if found < 4:
                ensure_version_columns(con)
            while found < SCHEMA_VERSION:
                step = MIGRATIONS.get(found + 1)
                if step is None:
                    raise SchemaVersionError(
                        f"no migration registered from schema version {found} to {found + 1}",
                        found=found, expected=SCHEMA_VERSION)
                step(con)
                con.execute("UPDATE anatid_meta SET schema_version = ?", [found + 1])
                found += 1

    # Make sure anything added since creation exists (views, new default indexes, and the
    # version column on a v4 file written before it existed).
    for stmt in ddl_statements(cfg):
        if stmt.startswith("CREATE OR REPLACE VIEW") or "IF NOT EXISTS" in stmt:
            con.execute(stmt)
    ensure_version_columns(con)
    ensure_index_columns(con)
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
#
# ``temporal_predicate`` lives in :mod:`anatid.visibility` since 0.2 and is imported above so
# ``from anatid.schema import temporal_predicate`` keeps resolving under its v3 name.  It is
# the time half of :meth:`anatid.visibility.Visibility.predicate`; new code should build a
# ``Visibility`` and take the whole predicate from it.
