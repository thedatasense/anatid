"""Retrieval: the benchmarked 2-hop graph recall and the hybrid vector + BM25 + graph search.

Every statement here is a port of the SQL the Phase 0 spike measured
(``spike/duckdb_sql/macros.sql`` and ``spike/bench/run_duckdb_sql.py``), not a reinvention.  The
shapes that mattered are preserved:

* The 2-hop frontier enters through ``IN (...)`` so the planner builds SEMI joins
  (frontier -> ``edges_about``.dst -> ``memories``.memory_id), both scans tenant-filtered so zone
  maps prune the other tenants, with the TOP_N pushing a dynamic ``created_at`` threshold into
  the ``memories`` scan.
* The frontier is *not* deduplicated -- a semi join does not care and each DISTINCT costs a
  HASH_GROUP_BY.
* BM25 is computed straight off the inverted index rather than through ``match_bm25``, with
  Okapi k1=1.2, b=0.75 and ``idf = ln((N - df + 0.5) / (df + 0.5) + 1)``.  That is the same
  ranking ``match_bm25`` produces and roughly 2x faster, because it skips the extension's
  per-document correlated lookup.  Only *where the numbers come from* changed in schema v3 --
  see "Tenant-scoped BM25" below.
* The arms are fused with Reciprocal Rank Fusion, ``score = sum 1 / (k + rank)``, k = 60,
  1-based ranks, ties broken by ``memory_id ASC``.

Three limits this module states rather than hides
-------------------------------------------------
**Tenant-scoped BM25 (schema v3).**  DuckDB's fts extension needs a document key that is unique
over the indexed table, and ``memory_id`` is unique only *within* a tenant: ``remember(memory_id=
...)`` and per-tenant Parquet imports both mint the same id under two tenants.  Schema v2 keyed
the index on ``memory_id`` anyway, so a BM25 hit on tenant 2's text returned tenant 1's row --
one tenant could learn that another's corpus contains a term, and a scoped search returned
documents that do not match the query -- and every ``df`` / ``avgdl`` behind every score was
computed over every tenant's text.

Since v3 the index is built over :data:`anatid.schema.FTS_SOURCE_TABLE`, whose document key is
``'<tenant_id>:<memory_id>'`` (:func:`anatid.schema.fts_doc_id`).  :func:`bm25_arm` prunes the
candidate documents to one tenant through :data:`~anatid.schema.FTS_DOCS_TABLE` *before* it
scores, joins ``memories`` on ``(memory_id, tenant_id)``, and reads ``df`` and
``(num_docs, avgdl)`` from the per-tenant :data:`~anatid.schema.FTS_DICT_TABLE` and
:data:`~anatid.schema.FTS_STATS_TABLE`.  It never reads ``fts_main_*.dict.df`` or
``fts_main_*.stats``: those count every tenant's documents, which skews the ranking and is a
weaker leak of its own.  The property that buys is checkable, and ``tests/test_recall_tenancy.py``
checks it: another tenant's writes change neither the rows nor the *scores* this tenant gets,
and a brute-force per-tenant BM25 in numpy agrees with :func:`bm25_arm` on ids, order and
scores for 500 spike queries over 100k memories / 10 tenants (498 exactly, 2 differing only in
the order of equal scores) while the same reference with file-wide statistics -- v2's scoring
-- ranks 492 of them differently.

**Brute-force vector search.**  ``array_cosine_similarity`` over ``FLOAT[N]`` scans every current
memory of the tenant.  anatid builds no ANN index, so the cost is linear in **the tenant's** row
count, not the file's.  (DuckDB does ship a team-maintained ``vss`` extension with an HNSW index;
its persistence is still experimental and its own documentation advises against relying on it in
production, so anatid does not build on it --
https://duckdb.org/docs/stable/core_extensions/vss.)  Measured on the spike hardware at 64
dimensions with DuckDB's default thread count, all rows in one tenant:

===========  ==================
rows/tenant  ``vector_arm`` p50
===========  ==================
10,000       2.0 ms
100,000      8.6 ms  (an independent run of the same measurement got 11.4 ms)
1,000,000    23.3 ms
===========  ==================

:data:`BRUTE_FORCE_CEILING` (1e5) is where anatid stops calling this cheap -- note that you are
already paying roughly 9-11 ms per recall *at* that ceiling, not the ~1 ms this docstring used to
claim.  (That 1 ms is real, but it is the spike's small-scale figure, where 100k memories were
spread over 10 tenants and each scan saw 10k rows.)  Since 0.1.1 the ceiling is **enforced**:
:func:`hybrid_recall` counts the rows the vector arm would score (:func:`vector_scan_rows`) and
raises :class:`~anatid.errors.BruteForceCeilingError` past the ceiling unless the caller passes
``allow_slow=True``.  Past the ceiling you want file-per-tenant sharding
(:class:`anatid.DatabasePool`), a vector store outside anatid, or the ``vss`` extension with the
caveat above.

**Non-incremental BM25.**  DuckDB's ``fts`` index is rebuilt wholesale by
``PRAGMA create_fts_index``; rows inserted afterwards are invisible to BM25 until it runs again.
anatid does not paper over this: :func:`fts_status` reports the pending row count, every
:class:`~anatid.types.RecallHits` carries ``bm25_stale``, and a stale index is logged at WARNING
on the ``anatid.recall`` logger.  The staleness window is whatever your rebuild policy makes it
-- see :func:`rebuild_fts_index`.  A rebuild covers the whole file, but *staleness is reported
per tenant*: ``fts_status(con, tenant_id=...)`` compares that tenant's ``memories`` rows with
the corpus the last rebuild recorded for that tenant -- its ``num_docs`` in
:data:`~anatid.schema.FTS_STATS_TABLE`, the number its scores are computed with, and the ids in
:data:`~anatid.schema.FTS_DOCS_TABLE` -- so tenant 1 is not told its index is stale because
tenant 2 wrote something tenant 1 can never see.  ``fts_status(con)`` with no tenant keeps the
file-wide watermark from ``anatid_meta``, which is what :meth:`anatid.Anatid.fts_status` and
the MCP ``health`` tool report.

What the scoping costs, measured on the spike's 100k memories / 10 tenants on this machine
(duckdb 1.5.5, 100 spike queries x3, top-50): the v3 query runs at 6.5 ms p50 single-threaded
against 6.4 ms for the v2 query over the same rows, 7.5 ms against 6.8 ms with four threads --
within 10% either way, because the tenant's ``docmap`` slice is pruned before the postings are
aggregated.  The rebuild materialises three sidecar tables on top of the index: 1.3 s for 100k
rows here.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Iterable, Sequence

from . import schema as _schema
from .csr import CsrBackend, _num, frontier_sql
from .errors import BruteForceCeilingError, EmbeddingDimensionError, StaleIndexError
from .schema import (
    FTS_DICT_TABLE,
    FTS_DOCS_TABLE,
    FTS_INDEX_SCHEMA,
    FTS_STATS_TABLE,
    MEMORY_COLUMNS,
    temporal_predicate,
)
from .types import (
    CURRENT,
    AsOf,
    FtsStatus,
    Memory,
    RecallHit,
    RecallHits,
)

log = logging.getLogger("anatid.recall")

__all__ = [
    "RRF_K",
    "DEFAULT_CANDIDATES",
    "BM25_K1",
    "BM25_B",
    "BRUTE_FORCE_CEILING",
    "FTS_STALENESS_POLICY",
    "recall_2hop_ids",
    "graph_arm",
    "vector_arm",
    "vector_scan_rows",
    "bm25_arm",
    "hydrate",
    "about_names",
    "rrf_fuse",
    "hybrid_recall",
    "fts_index_present",
    "fts_status",
    "rebuild_fts_index",
    "embedding_literal",
]

RRF_K = 60
DEFAULT_CANDIDATES = 50          # size of each arm's candidate list (spike R2_TOPN)
BM25_K1, BM25_B = 1.2, 0.75

#: Above this many visible memories in one tenant, the brute-force cosine scan stops being
#: cheap -- measured 8.6-11.4 ms p50 *at* this number, 23.3 ms at 1M (64 dims, this machine).
#: :func:`hybrid_recall` (and so :meth:`anatid.Anatid.recall`) enforces it: a vector-arm request
#: whose scan would cover more rows than this raises
#: :class:`~anatid.errors.BruteForceCeilingError` unless ``allow_slow=True`` is passed.  The
#: bare :func:`vector_arm` primitive does not check; it is the scan itself.
BRUTE_FORCE_CEILING = 100_000

FTS_STALENESS_POLICY = (
    "DuckDB's fts index is not incremental. anatid compares the row count AND the largest "
    "memory_id against the index on every recall(), so an insert cancelled out by a hard purge "
    "is still reported stale. A recall() compares only the querying tenant's rows (memories vs "
    f"that tenant's num_docs in {FTS_STATS_TABLE} and its ids in {FTS_DOCS_TABLE}), so another "
    "tenant's writes never report as this tenant's staleness; "
    "fts_status() with no tenant reports the file-wide watermark recorded in anatid_meta "
    "(fts_indexed_rows / fts_indexed_max_id). Default policy: report, never silently rebuild -- "
    "a rebuild is O(corpus) and must be the caller's decision. "
    "Call rebuild_fts_index() after a batch of writes, on a timer, or when pending_rows crosses "
    "your threshold; the staleness window is the interval between those rebuilds."
)

_MEM_COLS_SQL = ", ".join(f"m.{c}" for c in MEMORY_COLUMNS)


# --------------------------------------------------------------------------- helpers

def embedding_literal(vec: Sequence[float]) -> str:
    """Serialise an embedding as the text DuckDB casts to ``FLOAT[N]``.

    It is passed as a bound VARCHAR parameter and cast in SQL (``?::FLOAT[N]``).  Binding a
    Python list of floats directly costs about 4.5 ms per call in duckdb-python 1.5.5; this costs
    about 0.15 ms and round-trips float32 exactly.  It is still a *parameter* -- no value is ever
    concatenated into a statement.
    """
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _kind_filter(alias: str, kinds: Sequence[str] | None) -> tuple[str, list]:
    if not kinds:
        return "", []
    marks = ", ".join("?" for _ in kinds)
    return f" AND {alias}.kind IN ({marks})", [str(k) for k in kinds]


def _int_list(ids: Iterable[int]) -> str:
    """Render ids as a SQL IN-list.  Every element goes through ``int()`` first, so nothing but a
    number can reach the statement.  Used only where a bound parameter defeats the ART index."""
    return ", ".join(str(int(i)) for i in ids)


# --------------------------------------------------------------------------- graph arm

def recall_2hop_ids(
    con,
    *,
    tenant_id: int,
    seed_entity_id: int,
    limit: int = 20,
    hops: int = 2,
    as_of: AsOf = CURRENT,
    backend: CsrBackend | None = None,
    kinds: Sequence[str] | None = None,
) -> list[tuple[int, _dt.datetime]]:
    """The benchmarked 2-hop recall, returning ``[(memory_id, created_at), ...]`` newest first.

    This is the exact query the Phase 0 kill criterion was measured on (p50 2.88 ms pure SQL /
    2.04 ms with the CSR extension at 1M memories, vs LadybugDB's 7.35 ms).  It projects only two
    columns on purpose -- hydrating full rows would pull the ``FLOAT[N]`` embedding column into
    the TOP_N's input.  Use :func:`hydrate` (or ``Anatid.recall_2hop``) when you want
    :class:`~anatid.types.Memory` objects.

    Ordering is ``created_at DESC, memory_id DESC`` and the visibility filter is
    :func:`anatid.schema.temporal_predicate`.
    """
    # Fast path: current state, no kind filter -> every value in the statement is an integer, so
    # the whole query is rendered with literals and runs with NO bound parameters.  Measured at
    # 100k memories, single thread: 1.32 ms p50 that way versus 1.91 ms with bound parameters
    # (duckdb-python 1.5.5 spends ~0.6 ms per call marshalling them).  See csr._num for why
    # inlining integers is safe; nothing else is ever inlined.
    inline = as_of.is_current and not kinds
    fsql, fparams, _path = frontier_sql(
        tenant_id, seed_entity_id, hops, as_of=as_of, backend=backend, inline_ints=inline)
    mw, mp = temporal_predicate("m", as_of)
    aw, ap = temporal_predicate("a", as_of)
    kf, kp = _kind_filter("m", kinds)
    t_sql, t_p = _num(tenant_id, inline)
    l_sql, l_p = _num(limit, inline)

    # frontier_sql may start with WITH; hoist it so the whole thing is one statement.
    if fsql.lstrip().upper().startswith("WITH"):
        head, body = _split_with(fsql)
        prefix = head + ", "
    else:
        prefix, body = "WITH ", fsql
    sql = (
        f"{prefix}anatid_frontier AS ({body}) "
        f"SELECT m.memory_id, m.created_at FROM memories m "
        f"WHERE m.tenant_id = {t_sql} AND {mw}{kf} "
        f"AND m.memory_id IN ("
        f"  SELECT a.src FROM edges_about a WHERE a.tenant_id = {t_sql} AND {aw}"
        f"  AND a.dst IN (SELECT entity_id FROM anatid_frontier)) "
        f"ORDER BY m.created_at DESC, m.memory_id DESC LIMIT {l_sql}"
    )
    params = list(fparams) + list(t_p) + mp + kp + list(t_p) + ap + list(l_p)
    rows = con.execute(sql).fetchall() if not params else con.execute(sql, params).fetchall()
    return [(int(r[0]), r[1]) for r in rows]


def _split_with(sql: str) -> tuple[str, str]:
    """Split ``WITH a AS (...), b AS (...) SELECT ...`` into its CTE prefix and its ``SELECT`` body.

    The generator in :mod:`anatid.csr` is the only producer of these strings, so the split can rely
    on the final top-level ``SELECT`` starting the body.
    """
    depth = 0
    i = 0
    n = len(sql)
    while i < n:
        c = sql[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0 and sql.startswith("SELECT", i) and (i == 0 or not sql[i - 1].isalnum()):
            return sql[:i].rstrip(), sql[i:]
        i += 1
    raise ValueError(f"could not split CTE prefix from {sql!r}")


def graph_arm(
    con,
    *,
    tenant_id: int,
    seed_entity_id: int,
    hops: int,
    topn: int,
    as_of: AsOf = CURRENT,
    backend: CsrBackend | None = None,
    kinds: Sequence[str] | None = None,
) -> list[tuple[int, float]]:
    """Graph candidates for the hybrid fusion: ``[(memory_id, recency_score)]`` newest first."""
    rows = recall_2hop_ids(con, tenant_id=tenant_id, seed_entity_id=seed_entity_id,
                           limit=topn, hops=hops, as_of=as_of, backend=backend, kinds=kinds)
    return [(mid, float(len(rows) - i)) for i, (mid, _ts) in enumerate(rows)]


# --------------------------------------------------------------------------- vector arm

def vector_scan_rows(
    con,
    *,
    tenant_id: int,
    as_of: AsOf = CURRENT,
    kinds: Sequence[str] | None = None,
) -> int:
    """How many rows :func:`vector_arm` would score for this request.

    The same tenant, temporal and kind predicates as the arm itself, over rows that have an
    embedding -- that is the number the brute-force scan is linear in, and the number
    :data:`BRUTE_FORCE_CEILING` is compared against.  A ``count(*)`` with these predicates is a
    columnar scan of a few narrow columns: well under a millisecond at the ceiling, so the check
    costs a small fraction of the scan it guards.
    """
    w, wp = temporal_predicate("m", as_of)
    kf, kp = _kind_filter("m", kinds)
    sql = (f"SELECT count(*) FROM memories m WHERE m.tenant_id = ? AND {w} "
           f"AND m.embedding IS NOT NULL{kf}")
    row = con.execute(sql, [int(tenant_id)] + wp + kp).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def vector_arm(
    con,
    *,
    tenant_id: int,
    embedding: Sequence[float],
    dim: int,
    topn: int = DEFAULT_CANDIDATES,
    as_of: AsOf = CURRENT,
    kinds: Sequence[str] | None = None,
) -> list[tuple[int, float]]:
    """Brute-force cosine top-N over the tenant's visible memories.

    anatid builds no ANN index, so this is a full scan of the tenant's ``embedding`` column.
    (DuckDB's ``vss`` extension can build an HNSW index; persisting one is still experimental --
    see the module docstring.)  See :data:`BRUTE_FORCE_CEILING`.  Ties break on ``memory_id ASC``.
    """
    if len(embedding) != int(dim):
        raise EmbeddingDimensionError(
            f"embedding has {len(embedding)} dimensions, database is FLOAT[{dim}]",
            expected=int(dim), got=len(embedding))
    w, wp = temporal_predicate("m", as_of)
    kf, kp = _kind_filter("m", kinds)
    sql = (
        f"SELECT m.memory_id, array_cosine_similarity(m.embedding, ?::FLOAT[{int(dim)}]) AS score "
        f"FROM memories m WHERE m.tenant_id = ? AND {w} AND m.embedding IS NOT NULL{kf} "
        f"ORDER BY score DESC, m.memory_id ASC LIMIT ?"
    )
    params = [embedding_literal(embedding), int(tenant_id)] + wp + kp + [int(topn)]
    return [(int(r[0]), float(r[1])) for r in con.execute(sql, params).fetchall()]


# --------------------------------------------------------------------------- BM25 arm

def fts_index_present(con) -> bool:
    """True when a BM25 index has been built on this database.

    Delegates to :func:`anatid.schema.fts_objects_present`, which probes the schema
    ``PRAGMA create_fts_index`` creates for :data:`~anatid.schema.FTS_SOURCE_TABLE`.  A
    schema-v2 file's ``fts_main_memories`` does **not** count: it is keyed on ``memory_id`` and
    leaks across tenants, and the 2->3 migration drops it, so a migrated file reports "no index"
    until :func:`rebuild_fts_index` runs.  Reporting no index is the fail-safe answer; serving
    BM25 from the old one would not be.
    """
    return _schema.fts_objects_present(con)


#: BM25 over the schema-v3, per-tenant index.  ``{where}`` is the temporal predicate on ``m`` and
#: ``{kinds}`` the optional kind filter; the rest is fixed.  Parameters, in order:
#: ``(query_text, tenant, tenant, tenant, tenant, *temporal, *kinds, topn)``.
#:
#: Why each tenant predicate is there -- all four are load-bearing, none is belt-and-braces:
#:
#: * ``dt`` restricts the *candidate documents* to the tenant, so a matching document belonging
#:   to another tenant never reaches ``tf``/``sc`` at all.  This is what makes the row set
#:   correct;
#: * ``qt`` takes ``df`` from the per-tenant dictionary, not ``fts_main_*.dict.df``;
#: * ``st`` takes ``(num_docs, avgdl)`` from the per-tenant stats, not ``fts_main_*.stats``.
#:   Those two make the *score* depend on this tenant's corpus alone;
#: * the join to ``memories`` carries ``tenant_id`` because ``memory_id`` is not unique across
#:   tenants -- the same reason :func:`hydrate` takes a tenant.
#:
#: ``fts_main_*.dict`` is still read, but only to map a query term to its ``termid``; the
#: ``df`` column of that table is never selected.
_BM25_SQL = f"""
        WITH q AS (
            SELECT DISTINCT term FROM (
                SELECT unnest(string_split_regex(
                    regexp_replace(lower(?), '(\\.|[^a-z])+', ' ', 'g'), '\\s+')) AS term)
            WHERE term <> ''
        ), dt AS (
            SELECT docid, memory_id, len FROM {FTS_DOCS_TABLE} WHERE tenant_id = ?
        ), qt AS (
            SELECT d.termid, td.df
            FROM {FTS_INDEX_SCHEMA}.dict d
            JOIN q ON d.term = q.term
            JOIN {FTS_DICT_TABLE} td ON td.termid = d.termid AND td.tenant_id = ?
        ), st AS (
            SELECT num_docs, avgdl FROM {FTS_STATS_TABLE} WHERE tenant_id = ?
        ), tf AS (
            SELECT t.docid, t.termid, count(*)::DOUBLE AS tf
            FROM {FTS_INDEX_SCHEMA}.terms t
            JOIN qt ON t.termid = qt.termid
            JOIN dt ON dt.docid = t.docid
            GROUP BY t.docid, t.termid
        ), sc AS (
            SELECT tf.docid,
                   sum(ln((s.num_docs - qt.df + 0.5) / (qt.df + 0.5) + 1)
                       * tf.tf * ({BM25_K1} + 1)
                       / (tf.tf + {BM25_K1} * (1 - {BM25_B} + {BM25_B} * dt.len / s.avgdl))
                      ) AS score
            FROM tf
            JOIN qt ON tf.termid = qt.termid
            JOIN dt ON dt.docid = tf.docid
            CROSS JOIN st s
            GROUP BY tf.docid
        )
        SELECT m.memory_id, sc.score
        FROM sc
        JOIN dt ON dt.docid = sc.docid
        JOIN memories m ON m.memory_id = dt.memory_id
        WHERE m.tenant_id = ? AND {{where}}{{kinds}}
        ORDER BY sc.score DESC, m.memory_id ASC
        LIMIT ?
"""


def bm25_arm(
    con,
    *,
    tenant_id: int,
    query_text: str,
    topn: int = DEFAULT_CANDIDATES,
    as_of: AsOf = CURRENT,
    kinds: Sequence[str] | None = None,
    available: bool | None = None,
) -> list[tuple[int, float]]:
    """Okapi BM25 top-N over **this tenant's** documents, straight off the inverted index.

    Query text is tokenised the way the index was built (lower-cased, ``(\\.|[^a-z])+`` treated as
    a separator -- :data:`anatid.schema.FTS_TOKENIZER`), so query terms and index terms agree.
    Only memories matching at least one term score.  Ties break on ``memory_id ASC``.

    Nothing outside ``tenant_id`` can influence the result: the candidate documents are pruned to
    the tenant before scoring and the corpus statistics are the tenant's own.  See
    :data:`_BM25_SQL` for which predicate does what, and the module docstring for what schema v2
    got wrong.

    Returns ``[]`` when no fts index exists.  Rows written since the last index build cannot
    appear here at all -- see :func:`fts_status`.

    One artifact worth stating: the rebuild keeps a single document per ``(tenant_id,
    memory_id)``, but if ``memories`` holds two *current* rows under one id -- an integrity fault
    ``Anatid.doctor()`` reports -- the final join matches both and the id is returned twice, so
    RRF scores it twice and it costs two candidate slots.  Deduplicating here would put a hash
    aggregate over the whole matching set on every query to compensate for a broken file, so this
    is left to :func:`~anatid.Anatid.doctor` to find and to the write path to prevent.
    """
    if not (fts_index_present(con) if available is None else available):
        return []
    w, wp = temporal_predicate("m", as_of)
    kf, kp = _kind_filter("m", kinds)
    sql = _BM25_SQL.format(where=w, kinds=kf)
    t = int(tenant_id)
    params = [str(query_text), t, t, t, t] + wp + kp + [int(topn)]
    return [(int(r[0]), float(r[1])) for r in con.execute(sql, params).fetchall()]


def fts_status(con, *, deep: bool = False, tenant_id: int | None = None) -> FtsStatus:
    """Report how far the BM25 index has fallen behind ``memories``.

    With ``tenant_id`` the report covers **only that tenant's** rows, and comes from the
    per-tenant tables the last rebuild wrote rather than from the file-wide watermark:
    ``indexed_rows`` is the tenant's ``num_docs`` in :data:`~anatid.schema.FTS_STATS_TABLE` --
    the corpus size its BM25 scores are computed with -- and ``indexed_max_id`` the largest
    ``memory_id`` among its :data:`~anatid.schema.FTS_DOCS_TABLE` rows.  ``num_docs`` rather
    than a count of those rows on purpose: a hard purge removes the document from the index but
    not from the statistics, so a purge with no rebuild after it reports ``pending_rows = -1``
    here exactly as the file-wide report does, because the scores are now computed over a
    corpus that no longer exists.  That is what :func:`hybrid_recall` asks for, because a rebuild is
    file-wide but *visibility* is not: since schema v3 no other tenant's writes can change this
    tenant's BM25 rows or scores, so telling tenant 1 its index is stale because tenant 2 wrote
    a row would be a false alarm -- and one that leaks the fact that tenant 2 wrote at all.
    ``indexed_at`` stays the file-wide build time; there is only one.

    Everything below describes the file-wide report (``tenant_id=None``), which is what
    :meth:`anatid.Anatid.fts_status` returns.

    ``indexed_rows`` is the row count recorded by the last :func:`rebuild_fts_index`;
    ``current_rows`` is ``count(*)`` now (metadata-only in DuckDB, so this is cheap enough to run
    on every recall).  ``pending_rows`` is the difference: rows the BM25 arm cannot see.  A
    negative difference (rows deleted since the build) also counts as stale, because the index
    still holds documents that no longer exist -- they are filtered out by the tenant/validity
    join, but the corpus statistics behind the scores are off.

    A row *count* on its own is not a watermark: one insert plus one hard purge leaves the count
    where it was while the new document is invisible to BM25.  So the cheap path also compares
    ``max(memory_id)`` with ``anatid_meta.fts_indexed_max_id``, the largest id present at the last
    rebuild.  ids from :func:`anatid.ids.new_id` are time-ordered, so any insert raises it.  The
    two checks are OR'ed; either one alone can miss, together they catch every write anatid's own
    verbs can perform.

    ``deep=True`` additionally reads ``max(tx_from)`` (a second column scan) to catch rows
    back-dated into the table without changing the count.

    What no watermark here can see: a raw SQL ``UPDATE memories SET content = ...``.  No anatid
    verb rewrites content in place (``supersede`` inserts a new row), so this is reachable only
    through :attr:`anatid.Anatid.connection`; if you do it, call :func:`rebuild_fts_index`
    yourself.  Stated rather than hidden, like the rest of the fts contract.
    """
    # One statement rather than several: every extra round trip costs a duckdb-python parameter
    # marshalling pass, and this runs on every recall() that has a text query.  count(*) is
    # metadata-only; adding max(memory_id) to it is a one-column scan measured at +0.04 ms on
    # 100k rows and +0.22 ms on 1M (0.12 -> 0.16 and 0.19 -> 0.41 ms p50 on this machine), inside
    # a fts_status() that costs ~1.5 ms either way because of the information_schema probe.  That
    # is the price of not lying about staleness.  The per-tenant variant is the same shape with a
    # tenant predicate on each scan, so its counts are real scans rather than metadata reads:
    # measured 2.26 ms p50 against 1.79 ms for the file-wide one on the spike's 100k memories
    # over 10 tenants.  That +0.5 ms buys a staleness answer that is about the caller's own rows.
    if tenant_id is None:
        sql = (
            "SELECT (SELECT count(*) FROM information_schema.tables "
            "        WHERE table_schema = ? AND table_name = 'docs'),"
            "       (SELECT count(*) FROM memories),"
            "       (SELECT max(memory_id) FROM memories),"
            "       (SELECT fts_indexed_rows FROM anatid_meta LIMIT 1),"
            "       (SELECT fts_indexed_max_id FROM anatid_meta LIMIT 1),"
            "       (SELECT fts_indexed_at FROM anatid_meta LIMIT 1)")
        params: list = [FTS_INDEX_SCHEMA]
        deep_sql, deep_params = "SELECT max(tx_from) FROM memories", []
    else:
        t = int(tenant_id)
        sql = (
            "SELECT (SELECT count(*) FROM information_schema.tables "
            "        WHERE table_schema = ? AND table_name = 'docs'),"
            "       (SELECT count(*) FROM memories WHERE tenant_id = ?),"
            "       (SELECT max(memory_id) FROM memories WHERE tenant_id = ?),"
            f"       (SELECT max(num_docs) FROM {FTS_STATS_TABLE} WHERE tenant_id = ?),"
            f"       (SELECT max(memory_id) FROM {FTS_DOCS_TABLE} WHERE tenant_id = ?),"
            "       (SELECT fts_indexed_at FROM anatid_meta LIMIT 1)")
        params = [FTS_INDEX_SCHEMA, t, t, t, t]
        deep_sql = "SELECT max(tx_from) FROM memories WHERE tenant_id = ?"
        deep_params = [t]
    present, current, current_max, indexed_rows, indexed_max, indexed_at = con.execute(
        sql, params).fetchone()
    present = bool(present)
    current = int(current)
    current_max = None if current_max is None else int(current_max)
    indexed_rows = None if indexed_rows is None else int(indexed_rows)
    indexed_max = None if indexed_max is None else int(indexed_max)
    newest = con.execute(deep_sql, deep_params).fetchone()[0] if deep else None
    if not present:
        return FtsStatus(available=False, stale=True, indexed_rows=None, current_rows=current,
                         pending_rows=current, indexed_at=None, newest_row_at=newest,
                         policy=FTS_STALENESS_POLICY, indexed_max_id=indexed_max,
                         current_max_id=current_max)
    pending = current - (indexed_rows if indexed_rows is not None else 0)
    moved = current_max != indexed_max
    stale = (pending != 0 or moved
             or (indexed_at is not None and newest is not None and newest > indexed_at))
    return FtsStatus(available=True, stale=bool(stale), indexed_rows=indexed_rows,
                     current_rows=current, pending_rows=pending, indexed_at=indexed_at,
                     newest_row_at=newest, policy=FTS_STALENESS_POLICY,
                     indexed_max_id=indexed_max, current_max_id=current_max)


def _ensure_fts_extension(con) -> None:
    """Make the ``fts`` extension available on this connection, without an unnecessary ``INSTALL``.

    ``INSTALL fts`` reads the extension directory, and a hardened connection refuses it: the MCP
    sql gateway sets ``enable_external_access=false`` (DuckDB's own recommendation for an
    untrusted caller), which turns every filesystem operation into a ``PermissionException`` --
    including one for an extension already sitting in memory.  :meth:`anatid.Anatid.open`
    installs and loads fts before any hardening runs, so probe the catalog first and skip both
    statements when the extension is already there.

    The probe is ``duckdb_functions()`` rather than ``duckdb_extensions()`` on purpose: the
    latter reads the extension directory and so raises under exactly the configuration this
    function exists to survive.  When the extension really is missing the two statements run and
    their error propagates -- a BM25 index cannot be built without it, and pretending otherwise
    would leave a caller searching an index that silently does not exist.
    """
    row = con.execute("SELECT count(*) FROM duckdb_functions() "
                      "WHERE function_name = 'create_fts_index'").fetchone()
    if row and int(row[0]):
        return
    con.execute("INSTALL fts")
    con.execute("LOAD fts")


def rebuild_fts_index(con, *, now: _dt.datetime | None = None,
                      terms_index: bool = True) -> FtsStatus:
    """Rebuild the BM25 index over ``memories`` and record the watermark in ``anatid_meta``.

    Runs :func:`anatid.schema.fts_rebuild_statements` in order: refill
    :data:`~anatid.schema.FTS_SOURCE_TABLE` from ``memories`` keyed ``'<tenant>:<memory>'``,
    ``PRAGMA create_fts_index(..., overwrite=1)`` over it, then re-derive the per-tenant
    ``docid -> (tenant, memory, len)`` map, ``df`` dictionary and ``(num_docs, avgdl)`` stats
    from the postings that build produced.  Cost O(corpus).  It runs as **one transaction**,
    joining the caller's if one is open (:meth:`anatid.Anatid.rebuild_fts_index` opens one), and
    opening its own on a bare connection: the index and the tables that scope it are only
    meaningful together, so a rebuild that fails half-way leaves the previous index, its sidecar
    tables and the ``anatid_meta`` watermark exactly as they were.

    It is never called implicitly by a read: an implicit rebuild would turn an unlucky
    ``recall()`` into a multi-second stall.  Call it after a batch of writes, on a timer, or when
    :func:`fts_status`'s ``pending_rows`` crosses your threshold.

    The index covers every row in the file, all tenants; per-tenant filtering happens in the
    query, off tables this rebuild derives.  For file-per-tenant deployments that is exactly one
    tenant's corpus.

    ``terms_index`` also rebuilds the ART index on the postings table, which the spike measured
    at BM25 p50 8.57 ms with versus 12.51 ms without, for a 0.7 s build at 100k memories.
    ``create_fts_index`` drops the whole fts schema, so it has to be recreated here every time.
    """
    from .types import utcnow

    at = now or utcnow()
    _ensure_fts_extension(con)        # INSTALL/LOAD are not transactional: keep them outside
    # schema's own migration wrapper: joins an open transaction (Anatid.rebuild_fts_index opens
    # one) instead of issuing a nested BEGIN, which in DuckDB would abort the caller's.
    with _schema._transaction(con):   # noqa: SLF001 -- shared with ensure_schema on purpose
        for stmt in _schema.fts_rebuild_statements(terms_index=terms_index):
            con.execute(stmt)
        rows, top = con.execute(
            "SELECT count(*), max(memory_id) FROM memories").fetchone()
        con.execute(
            "UPDATE anatid_meta SET fts_indexed_rows = ?, fts_indexed_max_id = ?, "
            "fts_indexed_at = ?",
            [int(rows), None if top is None else int(top), at])
    return fts_status(con)


# --------------------------------------------------------------------------- fusion

def rrf_fuse(
    arms: dict[str, list[tuple[int, float]]],
    *,
    k: int = RRF_K,
    top: int | None = None,
) -> list[tuple[int, float, dict[str, int], dict[str, float]]]:
    """Reciprocal Rank Fusion of any number of ranked candidate lists.

    ``arms`` maps an arm name to its ``[(memory_id, arm_score)]`` list, already in rank order.
    Returns ``[(memory_id, rrf_score, {arm: rank}, {arm: arm_score})]`` ordered by
    ``rrf_score DESC, memory_id ASC``.  Ranks are 1-based; a memory appearing in several arms
    accumulates ``1 / (k + rank)`` from each.
    """
    fused: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    scores: dict[int, dict[str, float]] = {}
    for arm, rows in arms.items():
        for i, (mid, score) in enumerate(rows, start=1):
            fused[mid] = fused.get(mid, 0.0) + 1.0 / (k + i)
            ranks.setdefault(mid, {})[arm] = i
            scores.setdefault(mid, {})[arm] = score
    out = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    if top is not None:
        out = out[:top]
    return [(mid, sc, ranks[mid], scores[mid]) for mid, sc in out]


# --------------------------------------------------------------------------- hydration

def hydrate(
    con,
    memory_ids: Sequence[int],
    *,
    tenant_id: int,
    with_embedding: bool = False,
) -> dict[int, Memory]:
    """Fetch full :class:`~anatid.types.Memory` rows for a small id list.

    The ids go into a literal ``IN`` list after ``int()`` coercion: with a bound-parameter list
    DuckDB builds a MARK join over a column-data scan (measured 1.58 ms at 100k rows), with
    literals it is a filtered sequential scan of two columns (0.24 ms).  Nothing but integers can
    reach the statement.

    ``tenant_id`` is **required**, and is not redundant with the arm queries that produced the
    ids: an id identifies a memory only together with its tenant, so hydrating by id alone would
    return another tenant's row into a result the arms had correctly filtered.  Every generated
    query in anatid carries the tenant predicate -- this one, both ends of the BM25 join
    (:data:`_BM25_SQL`), and the fts document key itself, which is ``'<tenant>:<memory>'`` for
    exactly this reason.
    """
    ids = [int(i) for i in memory_ids]
    if not ids:
        return {}
    cols = ", ".join(
        (c if (with_embedding or c != "embedding") else "NULL AS embedding")
        for c in MEMORY_COLUMNS)
    sql = (f"SELECT {cols} FROM memories "
           f"WHERE memory_id IN ({_int_list(ids)}) AND tenant_id = ?")
    return {int(r[0]): Memory.from_row(r)
            for r in con.execute(sql, [int(tenant_id)]).fetchall()}


def about_names(con, memory_ids: Sequence[int], *, tenant_id: int,
                as_of: AsOf = CURRENT) -> dict[int, tuple[str, ...]]:
    """Entity names each memory is ABOUT (1-hop context), for a small id list.

    ``tenant_id`` is required for the same reason it is on :func:`hydrate`: ids collide across
    tenants in a ``SCOPED`` file, and an ABOUT edge fetched by ``src`` alone would name another
    tenant's entities.
    """
    ids = [int(i) for i in memory_ids]
    if not ids:
        return {}
    aw, ap = temporal_predicate("a", as_of)
    sql = (
        f"SELECT a.src, list(e.name ORDER BY e.entity_id) "
        f"FROM edges_about a JOIN entities e ON e.entity_id = a.dst AND e.tenant_id = a.tenant_id "
        f"WHERE a.src IN ({_int_list(ids)}) AND a.tenant_id = ? AND {aw} GROUP BY a.src"
    )
    return {int(r[0]): tuple(n for n in r[1] if n is not None)
            for r in con.execute(sql, [int(tenant_id)] + ap).fetchall()}


# --------------------------------------------------------------------------- hybrid

def hybrid_recall(
    con,
    *,
    tenant_id: int,
    query: str | None = None,
    embedding: Sequence[float] | None = None,
    dim: int,
    k: int = 10,
    seed_entity: int | None = None,
    hops: int = 2,
    as_of: AsOf = CURRENT,
    kinds: Sequence[str] | None = None,
    candidates: int = DEFAULT_CANDIDATES,
    rrf_k: int = RRF_K,
    backend: CsrBackend | None = None,
    with_embedding: bool = False,
    include_about: bool = True,
    on_stale_fts: str = "report",
    allow_slow: bool = False,
) -> RecallHits:
    """Hybrid retrieval: cosine top-N + BM25 top-N + optional k-hop graph expansion, fused by RRF.

    Arms run only when their input is present: the vector arm needs ``embedding``, the BM25 arm
    needs ``query`` **and** an fts index, the graph arm needs ``seed_entity``.  With no usable arm
    the result is empty rather than a silent full-table scan.

    ``on_stale_fts``: ``"report"`` (default -- set ``bm25_stale`` on the result and log a warning),
    ``"error"`` (raise :class:`~anatid.errors.StaleIndexError`), or ``"ignore"``.  Staleness is
    judged against ``tenant_id``'s own rows only (:func:`fts_status`), so a busy neighbour in a
    shared file does not make this tenant's fresh index look stale.

    The brute-force cosine ceiling is enforced here: when ``embedding`` is given and the rows the
    vector arm would scan (:func:`vector_scan_rows`) exceed :data:`BRUTE_FORCE_CEILING`, this
    raises :class:`~anatid.errors.BruteForceCeilingError` before running any arm, unless
    ``allow_slow=True``.  The other two arms are unaffected by the tenant's size; leave
    ``embedding`` out to use them alone.  See the module docstring for the measured costs.
    """
    arms: dict[str, list[tuple[int, float]]] = {}
    notes: list[str] = []

    if embedding is not None:
        if not allow_slow:
            ceiling = int(BRUTE_FORCE_CEILING)
            rows_to_scan = vector_scan_rows(con, tenant_id=tenant_id, as_of=as_of, kinds=kinds)
            if rows_to_scan > ceiling:
                raise BruteForceCeilingError(
                    f"the vector arm would brute-force scan {rows_to_scan} embeddings in tenant "
                    f"{int(tenant_id)}, above BRUTE_FORCE_CEILING={ceiling}; anatid has no ANN "
                    f"index and refuses to get quietly slower. Pass allow_slow=True to run the "
                    f"scan anyway, omit embedding= to answer from the text and graph arms, or "
                    f"shard the tenant into its own file (anatid.DatabasePool).",
                    tenant_id=int(tenant_id), rows=rows_to_scan, ceiling=ceiling)
        arms["vector"] = vector_arm(con, tenant_id=tenant_id, embedding=embedding, dim=dim,
                                    topn=candidates, as_of=as_of, kinds=kinds)

    # Staleness is asked for per tenant: a rebuild is file-wide, but since schema v3 another
    # tenant's rows can change neither this tenant's BM25 hits nor its scores, so counting them
    # as "pending" would be a false alarm -- and would leak that the other tenant wrote at all.
    status = (fts_status(con, tenant_id=tenant_id) if query else
              FtsStatus(available=False, stale=False, indexed_rows=None, current_rows=0,
                        pending_rows=0, indexed_at=None, policy=FTS_STALENESS_POLICY))
    if query:
        if status.available:
            arms["text"] = bm25_arm(con, tenant_id=tenant_id, query_text=query, topn=candidates,
                                    as_of=as_of, kinds=kinds, available=True)
        else:
            notes.append("no fts index on memories: BM25 arm skipped "
                         "(call rebuild_fts_index() to create it)")

    if seed_entity is not None:
        arms["graph"] = graph_arm(con, tenant_id=tenant_id, seed_entity_id=int(seed_entity),
                                  hops=hops, topn=candidates, as_of=as_of, backend=backend,
                                  kinds=kinds)

    if query and status.stale:
        if status.pending_rows > 0:
            what = (f"{status.pending_rows} of this tenant's memory row(s) written since the "
                    f"last rebuild are invisible to full-text search")
        elif status.pending_rows < 0:
            what = (f"this tenant's full-text statistics still count {-status.pending_rows} "
                    f"document(s) that have since been removed from memories, so the corpus "
                    f"statistics behind the scores are off")
        else:
            # count(*) matched but the id watermark moved: an insert cancelled out by a purge.
            what = (f"the row count is unchanged but memories have been written and removed since "
                    f"the last rebuild (max memory_id {status.indexed_max_id} -> "
                    f"{status.current_max_id}), so the newest rows are invisible to full-text "
                    f"search")
        msg = (f"BM25 index is stale: {what} (indexed_at={status.indexed_at}). "
               f"{FTS_STALENESS_POLICY}")
        if on_stale_fts == "error":
            raise StaleIndexError(msg)
        if on_stale_fts != "ignore":
            log.warning("%s", msg)
            notes.append(msg)

    fused = rrf_fuse(arms, k=rrf_k, top=k)
    ids = [mid for mid, _s, _r, _sc in fused]
    rows = hydrate(con, ids, tenant_id=tenant_id, with_embedding=with_embedding)
    names = (about_names(con, ids, tenant_id=tenant_id, as_of=as_of)
             if include_about else {})

    hits = []
    for rank, (mid, score, ranks, scores) in enumerate(fused, start=1):
        mem = rows.get(mid)
        if mem is None:            # raced with a hard purge between fusion and hydration
            continue
        hits.append(RecallHit(
            memory=mem, score=score, rank=rank,
            vector_rank=ranks.get("vector"), text_rank=ranks.get("text"),
            graph_rank=ranks.get("graph"),
            vector_score=scores.get("vector"), text_score=scores.get("text"),
            about=names.get(mid, ())))

    return RecallHits(
        hits,
        bm25_available=status.available,
        bm25_stale=bool(query) and status.stale,
        pending_fts_rows=status.pending_rows,
        arms=tuple(arms),
        as_of=as_of,
        notes=tuple(notes),
    )
