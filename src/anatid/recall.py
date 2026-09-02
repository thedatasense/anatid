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
* BM25 is computed straight off the fts extension's index tables
  (``fts_main_memories.dict / terms / docs / stats``) with Okapi k1=1.2, b=0.75 and
  ``idf = ln((N - df + 0.5) / (df + 0.5) + 1)``.  That is the same ranking ``match_bm25``
  produces and roughly 2x faster, because it skips the extension's per-document correlated
  lookup.
* The arms are fused with Reciprocal Rank Fusion, ``score = sum 1 / (k + rank)``, k = 60,
  1-based ranks, ties broken by ``memory_id ASC``.

Two limits this module states rather than hides
-----------------------------------------------
**Brute-force vector search.**  ``array_cosine_similarity`` over ``FLOAT[N]`` scans every current
memory of the tenant.  There is no ANN index, and the cost is linear in **the tenant's** row
count, not the file's.  Measured on the spike hardware at 64 dimensions with DuckDB's default
thread count, all rows in one tenant:

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
spread over 10 tenants and each scan saw 10k rows.)  Past the ceiling you want either
file-per-tenant sharding or an ANN index that DuckDB does not yet ship.

**Non-incremental BM25.**  DuckDB's ``fts`` index is rebuilt wholesale by
``PRAGMA create_fts_index``; rows inserted afterwards are invisible to BM25 until it runs again.
anatid does not paper over this: :func:`fts_status` reports the pending row count, every
:class:`~anatid.types.RecallHits` carries ``bm25_stale``, and a stale index is logged at WARNING
on the ``anatid.recall`` logger.  The staleness window is whatever your rebuild policy makes it
-- see :func:`rebuild_fts_index`.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Iterable, Sequence

from .csr import CsrBackend, _num, frontier_sql
from .errors import EmbeddingDimensionError, StaleIndexError
from .schema import MEMORY_COLUMNS, temporal_predicate
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

#: Above roughly this many current memories in one tenant, the brute-force cosine scan stops
#: being cheap -- measured 8.6-11.4 ms p50 *at* this number, 23.3 ms at 1M (64 dims, this
#: machine).  anatid does not enforce it; :func:`vector_arm` logs once past it.
BRUTE_FORCE_CEILING = 100_000

FTS_STALENESS_POLICY = (
    "DuckDB's fts index is not incremental. anatid records the row count AND the largest "
    "memory_id at index build time in anatid_meta (fts_indexed_rows / fts_indexed_max_id) and "
    "compares both with the table on every recall(), so an insert cancelled out by a hard purge "
    "is still reported stale. Default policy: report, never silently rebuild -- a rebuild is "
    "O(corpus) and must be the caller's decision. "
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
    """Split ``WITH a AS (...), b AS (...) SELECT ...`` into (``WITH a AS (...), b AS (...)``, ``SELECT ...``).

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

    No ANN index exists in DuckDB, so this is a full scan of the tenant's ``embedding`` column.
    See :data:`BRUTE_FORCE_CEILING`.  Ties break on ``memory_id ASC``.
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
    """True when ``PRAGMA create_fts_index('memories', ...)`` has been run on this database."""
    row = con.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = 'fts_main_memories' AND table_name = 'docs'").fetchone()
    return bool(row and row[0])


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
    """Okapi BM25 top-N straight off the fts extension's index tables.

    Query text is tokenised the way the index was built (lower-cased, ``(\\.|[^a-z])+`` treated as
    a separator), so query terms and index terms agree.  Only memories matching at least one term
    score.  Ties break on ``memory_id ASC``.

    Returns ``[]`` when no fts index exists.  Rows written since the last index build cannot
    appear here at all -- see :func:`fts_status`.
    """
    if not (fts_index_present(con) if available is None else available):
        return []
    w, wp = temporal_predicate("m", as_of)
    kf, kp = _kind_filter("m", kinds)
    sql = f"""
        WITH q AS (
            SELECT DISTINCT term FROM (
                SELECT unnest(string_split_regex(
                    regexp_replace(lower(?), '(\\.|[^a-z])+', ' ', 'g'), '\\s+')) AS term)
            WHERE term <> ''
        ), qt AS (
            SELECT d.termid, d.df FROM fts_main_memories.dict d JOIN q ON d.term = q.term
        ), tf AS (
            SELECT t.docid, t.termid, count(*)::DOUBLE AS tf
            FROM fts_main_memories.terms t JOIN qt ON t.termid = qt.termid
            GROUP BY t.docid, t.termid
        ), sc AS (
            SELECT tf.docid,
                   sum(ln((s.num_docs - qt.df + 0.5) / (qt.df + 0.5) + 1)
                       * tf.tf * ({BM25_K1} + 1)
                       / (tf.tf + {BM25_K1} * (1 - {BM25_B} + {BM25_B} * d.len / s.avgdl))) AS score
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
        WHERE m.tenant_id = ? AND {w}{kf}
        ORDER BY sc.score DESC, m.memory_id ASC
        LIMIT ?
    """
    params = [str(query_text), int(tenant_id)] + wp + kp + [int(topn)]
    return [(int(r[0]), float(r[1])) for r in con.execute(sql, params).fetchall()]


def fts_status(con, *, deep: bool = False) -> FtsStatus:
    """Report how far the BM25 index has fallen behind ``memories``.

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
    # is the price of not lying about staleness.
    present, current, current_max, indexed_rows, indexed_max, indexed_at = con.execute(
        "SELECT (SELECT count(*) FROM information_schema.tables "
        "        WHERE table_schema = 'fts_main_memories' AND table_name = 'docs'),"
        "       (SELECT count(*) FROM memories),"
        "       (SELECT max(memory_id) FROM memories),"
        "       (SELECT fts_indexed_rows FROM anatid_meta LIMIT 1),"
        "       (SELECT fts_indexed_max_id FROM anatid_meta LIMIT 1),"
        "       (SELECT fts_indexed_at FROM anatid_meta LIMIT 1)").fetchone()
    present = bool(present)
    current = int(current)
    current_max = None if current_max is None else int(current_max)
    indexed_rows = None if indexed_rows is None else int(indexed_rows)
    indexed_max = None if indexed_max is None else int(indexed_max)
    newest = con.execute("SELECT max(tx_from) FROM memories").fetchone()[0] if deep else None
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


def rebuild_fts_index(con, *, now: _dt.datetime | None = None,
                      terms_index: bool = True) -> FtsStatus:
    """Rebuild the BM25 index over ``memories`` and record the watermark in ``anatid_meta``.

    This is ``PRAGMA create_fts_index(..., overwrite=1)``: it rebuilds the whole inverted index,
    cost O(corpus).  It is never called implicitly by a read -- an implicit rebuild would turn an
    unlucky ``recall()`` into a multi-second stall.  Call it after a batch of writes, on a timer,
    or when :func:`fts_status`'s ``pending_rows`` crosses your threshold.

    The index covers every row in the file, all tenants; per-tenant filtering happens in the
    query.  For file-per-tenant deployments that is exactly one tenant's corpus.

    ``terms_index`` also rebuilds the ART index on ``fts_main_memories.terms(termid)``, which the
    spike measured at BM25 p50 8.57 ms with versus 12.51 ms without, for a 0.7 s build at 100k
    memories.  ``create_fts_index`` drops the whole fts schema, so it has to be recreated here
    every time.
    """
    from .schema import FTS_INDEX_SQL, FTS_TERMS_INDEX_SQL
    from .types import utcnow

    at = now or utcnow()
    con.execute("INSTALL fts")
    con.execute("LOAD fts")
    con.execute(FTS_INDEX_SQL)
    if terms_index:
        con.execute(FTS_TERMS_INDEX_SQL)
    rows, top = con.execute(
        "SELECT count(*), max(memory_id) FROM memories").fetchone()
    con.execute(
        "UPDATE anatid_meta SET fts_indexed_rows = ?, fts_indexed_max_id = ?, fts_indexed_at = ?",
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
    ids.  ``memory_id`` is not unique across tenants in a ``SCOPED`` file -- callers may pass
    ``memory_id=`` to :meth:`~anatid.Anatid.remember`, and a per-tenant Parquet import can carry
    colliding ids -- so hydrating by id alone could return another tenant's row into a result the
    arms had correctly filtered.  Every generated query in anatid carries the tenant predicate;
    this one included.
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
) -> RecallHits:
    """Hybrid retrieval: cosine top-N + BM25 top-N + optional k-hop graph expansion, fused by RRF.

    Arms run only when their input is present: the vector arm needs ``embedding``, the BM25 arm
    needs ``query`` **and** an fts index, the graph arm needs ``seed_entity``.  With no usable arm
    the result is empty rather than a silent full-table scan.

    ``on_stale_fts``: ``"report"`` (default -- set ``bm25_stale`` on the result and log a warning),
    ``"error"`` (raise :class:`~anatid.errors.StaleIndexError`), or ``"ignore"``.

    The brute-force cosine ceiling of roughly 1e5 memories per tenant applies here; see the module
    docstring.
    """
    arms: dict[str, list[tuple[int, float]]] = {}
    notes: list[str] = []

    if embedding is not None:
        arms["vector"] = vector_arm(con, tenant_id=tenant_id, embedding=embedding, dim=dim,
                                    topn=candidates, as_of=as_of, kinds=kinds)

    status = (fts_status(con) if query else
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
        if status.pending_rows:
            what = (f"{status.pending_rows} memory row(s) written since the last rebuild are "
                    f"invisible to full-text search")
        else:
            # count(*) matched but the id watermark moved: an insert cancelled out by a purge.
            what = (f"the row count is unchanged but memories have been written and removed since "
                    f"the last rebuild (max memory_id {status.indexed_max_id} -> "
                    f"{status.current_max_id}), so the newest rows are invisible to full-text "
                    f"search")
        msg = f"BM25 index is stale: {what} (indexed_at={status.indexed_at}). {FTS_STALENESS_POLICY}"
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
