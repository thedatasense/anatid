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
* No PRIMARY KEY / UNIQUE constraints: in DuckDB those create implicit ART indexes, and anatid
  wants every index to be an explicit, measured choice.  ``anatid`` mints ids itself
  (:mod:`anatid.ids`), so uniqueness comes from the allocator, not from a constraint.

System columns
--------------
``valid_from, valid_to, tx_from, tx_to, writer, episode_id, confidence`` are present on every
node and edge table by DEFAULT.  User-defined labels created with :func:`node_table_ddl` /
:func:`edge_table_ddl` can opt out with ``system_columns=False``, but then the temporal verbs
(``as_of``, ``supersede``, soft ``forget``) do not apply to that table -- which is why the
built-in tables never opt out.

Time travel
-----------
DuckDB has no ``AS OF SYSTEM TIME``.  These columns plus the WHERE clauses anatid compiles in
:mod:`anatid.recall` *are* the time-travel mechanism.
"""

from __future__ import annotations

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
    "CONTRACT_NOTES",
    "FTS_INDEX_SQL",
    "FTS_TERMS_INDEX_SQL",
    "ddl_statements",
    "node_table_ddl",
    "edge_table_ddl",
    "index_statements",
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

SCHEMA_VERSION = 2
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

ENTITY_COLUMNS: tuple[str, ...] = (
    "entity_id", "tenant_id", "kind", "name",
    "valid_from", "valid_to", "tx_from", "tx_to", "writer", "episode_id", "confidence",
)

EPISODE_COLUMNS: tuple[str, ...] = (
    "episode_id", "tenant_id", "source", "content", "kind", "created_at",
    "valid_from", "valid_to", "tx_from", "tx_to", "writer",
)

NODE_TABLES: tuple[str, ...] = ("memories", "entities", "episodes")
EDGE_TABLES: tuple[str, ...] = ("edges_about", "edges_relates", "edges_supersedes")
CATALOG_TABLES: tuple[str, ...] = ("anatid_meta", "anatid_audit")
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
    "transactions: DuckDB MVCC is optimistic and snapshot-isolated, NOT serializable. Appends "
    "never conflict; two concurrent updates to the SAME row abort the second with a retryable "
    "error, surfaced as anatid.errors.ConflictError.",
    "vector search: brute-force array_cosine_similarity over FLOAT[N]. Fine to roughly 1e5 "
    "memories per tenant; beyond that latency grows linearly with the tenant's row count.",
    "erasure: forget(hard=True) purges the memory GRAPH row, its edges, its embedding, its "
    "orphaned episode and its provenance, and no anatid_audit row survives that references the "
    "purged memory_id. It does NOT reach tables anatid does not own. Conversation transcripts "
    "(anatid.integrations.openai_agents.AnatidSession's agent_messages) quote memory content and "
    "ids verbatim; AnatidSession registers an erasure hook so a purge removes those rows too, but "
    "any other table you write into this file is yours to clean -- see Anatid.erasure_hooks.",
)

#: The exact fts index PRAGMA the spike benchmarked (tokenizer settings matter: they define what
#: BM25 can match, and the BM25 SQL in :mod:`anatid.recall` tokenizes queries the same way).
FTS_INDEX_SQL = (
    r"PRAGMA create_fts_index('memories', 'memory_id', 'content', stemmer='none', "
    r"stopwords='none', ignore='(\.|[^a-z])+', strip_accents=0, lower=1, overwrite=1)"
)

#: ART index on the fts extension's postings table.  The spike A/B'd it: BM25 p50 goes from
#: 12.51 ms to 8.57 ms at 100k memories, for a 0.7 s build.  ``create_fts_index`` drops and
#: recreates the whole ``fts_main_memories`` schema, taking this index with it, so
#: :func:`anatid.recall.rebuild_fts_index` re-creates it every time.
FTS_TERMS_INDEX_SQL = "CREATE INDEX idx_fts_terms_termid ON fts_main_memories.terms(termid)"


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
        Names from :data:`DEFAULT_INDEXES` / :data:`OPTIONAL_INDEXES` to create.
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
        f"""CREATE TABLE IF NOT EXISTS entities (
    entity_id    BIGINT   NOT NULL,
    tenant_id    INTEGER  NOT NULL,
    kind         VARCHAR,
    name         VARCHAR{sysc}
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
    """
    prefix = f"CREATE TABLE IF NOT EXISTS {name} ("
    for stmt in ddl_statements(config):
        if stmt.startswith(prefix):
            if as_table is None:
                return stmt
            return f"CREATE TABLE {quote_ident(as_table)} (" + stmt[len(prefix):]
    raise KeyError(f"no built-in DDL for table {name!r}")


def index_statements(config: SchemaConfig | None = None) -> list[str]:
    """CREATE INDEX statements for the configured index set."""
    cfg = config or SchemaConfig()
    out = []
    for name in cfg.indexes:
        out.append(DEFAULT_INDEXES.get(name) or OPTIONAL_INDEXES[name])
    return out


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
