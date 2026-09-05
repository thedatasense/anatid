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
  per-document correlated lookup.  The statement moved to :mod:`anatid.fts` in 0.2, where the
  text arm became a derived index; see "The text arm" below.
* The arms are fused with Reciprocal Rank Fusion, ``score = sum 1 / (k + rank)``, k = 60,
  1-based ranks, ties broken by ``memory_id ASC``.

Where the graph arm's seeds come from
------------------------------------
The graph arm needs an entity to expand from.  :func:`hybrid_recall` finds one when the
caller gives none: ``seed_entity="auto"`` (the default of :meth:`anatid.Anatid.recall`)
matches the query's words against the tenant's entity names, case-insensitively, longest name
first, and expands from up to :data:`AUTO_SEED_LIMIT` of them; ``seed_entity=None`` switches
that off and an explicit entity is used as it always was.  The result reports what happened in
``RecallHits.arms`` and ``RecallHits.seeds``.  The match is one statement (:func:`auto_seeds`):
the query's word n-grams travel as one bound parameter, are split and lower-cased in SQL and
hash-joined against ``lower(name)``, so its cost is the scan of the tenant's names and does
not grow with the number of candidates.  Measured on this machine (duckdb 1.5.5, 10 threads):
0.46 ms p50 at 1,000 entities in the tenant, 0.70 ms at 10,000, 1.90 ms at 100,000 (200,000 in
the file), for 5-word and 72-word queries alike.  It compares ``lower(name)`` rather than the
canonical ``entity_key`` because that column is a virtual generated one: filtering on it
evaluates its ``regexp_replace`` per row (36 ms at 100k entities) and DuckDB 1.5.5 uses the ART
index for neither, so a name stored with irregular internal whitespace is matched only when the
query repeats it.

Two limits this module states rather than hides
----------------------------------------------
**The text arm lives in :mod:`anatid.fts`.**  Since 0.2 it is a derived index: a published base
generation plus an ordered journal written in the same transaction as the memory, so a write is
searchable by the very next :func:`hybrid_recall` with no rebuild and a superseded memory leaves
the text results at once.  A database that has not called :func:`anatid.fts.attach` keeps 0.1.1's
single file-wide index, which is NOT incremental: :func:`fts_status` reports how far behind it
is, every :class:`~anatid.types.RecallHits` carries ``bm25_stale``, and a stale index is logged
at WARNING on the ``anatid.recall`` logger.

Both halves score with per-tenant corpus statistics over a per-tenant candidate set, which is
what made a BM25 hit on tenant 2's text stop returning tenant 1's row in 0.1.1, and both judge
staleness per tenant, so tenant 1 is never told its index is stale because tenant 2 wrote
something it can never see.  ``tests/test_recall_tenancy.py`` checks that against a brute-force
per-tenant BM25 in numpy: it agrees with :func:`bm25_arm` on ids, order and scores for 500 spike
queries over 100k memories / 10 tenants (498 exactly, 2 differing only in the order of equal
scores), while the same reference with file-wide statistics -- schema v2's scoring -- ranks 492
of them differently.  What the arm costs, what its scoring model is, and what it does when no
generation is usable are all in :mod:`anatid.fts`.

**Brute-force vector search is the default and the oracle.**  ``array_cosine_similarity`` over
``FLOAT[N]`` scans every current memory of the tenant, so the cost is linear in **the tenant's**
row count, not the file's.  Since 0.2 there is an opt-in alternative: :mod:`anatid.vector`
builds an HNSW index (DuckDB's ``vss`` extension) over a cold generation of the derived-index
framework, and :func:`vector_arm` merges it with an exact scan of everything written since.
That structure only chooses candidates; the scores and the ranking are the exact ones either
way, and any ``as_of`` read, stale generation or load failure falls back to the scan below and
reports why.  It is opt in because the extension's persistence is documented as experimental
(https://duckdb.org/docs/stable/core_extensions/vss).  Measured on the spike hardware at 64
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
``allow_slow=True``.  The number it compares is what the arm will ACTUALLY scan: with a usable
``duckdb_vss`` generation that is the journal's pending rows rather than the tenant's corpus, so
attaching the backend (``anatid.vector.attach``) is what lets a tenant past the ceiling be
served instead of refused.  Otherwise you want file-per-tenant sharding
(:class:`anatid.DatabasePool`) or a vector store outside anatid.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
import string
from typing import Iterable, Sequence

from . import fts as _fts
from .csr import CsrBackend, _num, frontier_sql
from .errors import BruteForceCeilingError, StaleIndexError
from .fts import BM25_B, BM25_K1, FTS_STALENESS_POLICY
from .schema import memory_select
from .types import (
    CURRENT,
    AsOf,
    FtsStatus,
    Memory,
    RecallHit,
    RecallHits,
)
from .visibility import Visibility, tenant_sql

log = logging.getLogger("anatid.recall")

__all__ = [
    "RRF_K",
    "DEFAULT_CANDIDATES",
    "AUTO_SEED",
    "AUTO_SEED_LIMIT",
    "AUTO_SEED_MAX_WORDS",
    "AUTO_SEED_QUERY_WORDS",
    "BM25_K1",
    "BM25_B",
    "BRUTE_FORCE_CEILING",
    "FTS_STALENESS_POLICY",
    "recall_2hop_ids",
    "graph_arm",
    "seed_candidates",
    "auto_seeds",
    "entity_name",
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

#: ``seed_entity="auto"``: find the graph arm's seeds in the query (:func:`auto_seeds`).  It is
#: the default of :meth:`anatid.Anatid.recall`; ``None`` disables the graph arm when no entity
#: is named.  An entity whose name is literally ``auto`` is still reachable by id or as an
#: :class:`~anatid.types.Entity`.
AUTO_SEED = "auto"
#: How many auto-detected seeds the graph arm expands from, longest name first.
AUTO_SEED_LIMIT = 3
#: The longest entity name, in words, the matcher looks for in the query.
AUTO_SEED_MAX_WORDS = 5
#: Only the first this many words of the query are matched; a question is short and a pasted
#: document is not a question.
AUTO_SEED_QUERY_WORDS = 40

#: Above this many visible memories in one tenant, the brute-force cosine scan stops being
#: cheap -- measured 8.6-11.4 ms p50 *at* this number, 23.3 ms at 1M (64 dims, this machine).
#: :func:`hybrid_recall` (and so :meth:`anatid.Anatid.recall`) enforces it: a vector-arm request
#: whose scan would cover more rows than this raises
#: :class:`~anatid.errors.BruteForceCeilingError` unless ``allow_slow=True`` is passed.  The
#: bare :func:`vector_arm` primitive does not check; it is the scan itself.
BRUTE_FORCE_CEILING = 100_000

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

    Ordering is ``created_at DESC, memory_id DESC``.  The frontier only narrows candidates; the
    tenant and time predicate on ``memories`` and ``edges_about`` comes from
    :class:`anatid.visibility.Visibility` and is applied after it, which is what lets the
    extension's current-state CSR serve the frontier without knowing about time.
    """
    # Fast path: current state, no kind filter -> every value in the statement is an integer, so
    # the whole query is rendered with literals and runs with NO bound parameters.  Measured at
    # 100k memories, single thread: 1.32 ms p50 that way versus 1.91 ms with bound parameters
    # (duckdb-python 1.5.5 spends ~0.6 ms per call marshalling them).  See csr._num for why
    # inlining integers is safe; nothing else is ever inlined.
    vis = Visibility.at(tenant_id, as_of)
    inline = vis.is_current and not kinds
    fsql, fparams, _path = frontier_sql(
        tenant_id, seed_entity_id, hops, as_of=vis.as_of, backend=backend, inline_ints=inline)
    mw, mp = vis.predicate("m", inline_tenant=inline)
    aw, ap = vis.predicate("a", inline_tenant=inline)
    kf, kp = _kind_filter("m", kinds)
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
        f"WHERE {mw}{kf} "
        f"AND m.memory_id IN ("
        f"  SELECT a.src FROM edges_about a WHERE {aw}"
        f"  AND a.dst IN (SELECT entity_id FROM anatid_frontier)) "
        f"ORDER BY m.created_at DESC, m.memory_id DESC LIMIT {l_sql}"
    )
    params = list(fparams) + mp + kp + ap + list(l_p)
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


def _graph_candidates(
    con,
    *,
    tenant_id: int,
    seed_entity_ids: Sequence[int],
    hops: int,
    topn: int,
    as_of: AsOf = CURRENT,
    backend: CsrBackend | None = None,
    kinds: Sequence[str] | None = None,
) -> list[tuple[int, float]]:
    """:func:`graph_arm` from several seeds, merged into one newest-first list of ``topn``.

    One frontier per seed (the benchmarked statement, unchanged), then a merge on
    ``(created_at DESC, memory_id DESC)``, the order the single-seed statement returns.  A
    memory reachable from two seeds appears once.  With one seed this is :func:`graph_arm`.
    """
    if len(seed_entity_ids) == 1:
        return graph_arm(con, tenant_id=tenant_id, seed_entity_id=int(seed_entity_ids[0]),
                         hops=hops, topn=topn, as_of=as_of, backend=backend, kinds=kinds)
    merged: dict[int, _dt.datetime] = {}
    for sid in seed_entity_ids:
        for mid, ts in recall_2hop_ids(con, tenant_id=tenant_id, seed_entity_id=int(sid),
                                       limit=topn, hops=hops, as_of=as_of, backend=backend,
                                       kinds=kinds):
            if mid not in merged or ts > merged[mid]:
                merged[mid] = ts
    rows = sorted(merged.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:topn]
    return [(mid, float(len(rows) - i)) for i, (mid, _ts) in enumerate(rows)]


# --------------------------------------------------------------------------- auto seeds
#
# ``seed_entity="auto"``: the query names the entities to expand from.  The matcher is
# deliberately simple: whole words of the query against whole entity names, case-insensitive,
# longest name first.  Anything cleverer belongs to the caller, who knows the domain, and the
# match has to cost less than the graph query it feeds.  See the module docstring for
# the measurement.

#: Separates the candidate n-grams inside the one bound parameter.  A word containing it is
#: dropped from matching rather than escaped: it does not occur in text.
_CANDIDATE_SEP = "\x1f"
_TRIM_CHARS = string.punctuation + "\u201c\u201d\u2018\u2019\u00ab\u00bb\u2026"
_POSSESSIVE = ("'s", "\u2019s")
_WORD_SPLIT = re.compile(r"[-_]")

#: The one statement behind :func:`auto_seeds`.  The candidates arrive as one VARCHAR, are
#: split and lower-cased inside DuckDB, and are hash-joined against ``lower(name)``: the cost is
#: the scan of the tenant's names and does not grow with the candidates (binding a Python list
#: costs about 65 us per element in duckdb-python 1.5.5, which is why it is a string).  ``raw``
#: comes back so the match can be tied to the span of the query it came from without comparing
#: Python's ``lower()`` with DuckDB's.
_SEED_MATCH_SQL = (
    "SELECT e.entity_id, e.name, c.raw FROM entities e "
    "JOIN (SELECT raw, lower(raw) AS key FROM "
    "      (SELECT unnest(string_split(?, chr(31))) AS raw)) c ON lower(e.name) = c.key "
    f"WHERE {tenant_sql('e')}"
)


def _query_words(query: str) -> list[str]:
    """The query as matchable words: punctuation trimmed, possessives dropped, hyphenated and
    underscored compounds split so ``ingest-service`` and ``ingest service`` are the same words.
    Capped at :data:`AUTO_SEED_QUERY_WORDS`."""
    words: list[str] = []
    for token in query.split():
        token = token.strip(_TRIM_CHARS)
        low = token.lower()
        for suffix in _POSSESSIVE:
            if low.endswith(suffix) and len(token) > len(suffix):
                token = token[: -len(suffix)]
                break
        for part in _WORD_SPLIT.split(token):
            part = part.strip(_TRIM_CHARS)
            if part and _CANDIDATE_SEP not in part:
                words.append(part)
                if len(words) >= AUTO_SEED_QUERY_WORDS:
                    return words
    return words


def seed_candidates(
    query: str, *, max_words: int = AUTO_SEED_MAX_WORDS
) -> dict[str, tuple[int, int]]:
    """Every entity name the query could contain, as ``{candidate: (start, end)}`` word spans.

    Candidates are the word n-grams of :func:`_query_words` for ``n`` from ``max_words`` down to
    1, each multi-word n-gram in three spellings (space-, hyphen- and underscore-joined) so a
    name stored as ``ingest-service`` is found by a question that says ``ingest service``.
    Insertion order is longest first, then left to right; a candidate that occurs twice keeps
    its first span.  Case is preserved: the SQL side lower-cases both sides.
    """
    words = _query_words(query)
    out: dict[str, tuple[int, int]] = {}
    for n in range(min(max_words, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            segment = words[i:i + n]
            forms = [" ".join(segment)]
            if n > 1:
                forms.append("-".join(segment))
                forms.append("_".join(segment))
            for form in forms:
                out.setdefault(form, (i, i + n))
    return out


def auto_seeds(
    con,
    *,
    tenant_id: int,
    query: str,
    limit: int = AUTO_SEED_LIMIT,
) -> list[tuple[int, str]]:
    """Entities named in ``query``, as ``[(entity_id, name)]``, longest name first, at most
    ``limit`` of them.

    Matching is whole words against whole names, case-insensitive (``lower()`` on both sides,
    evaluated by DuckDB), over the tenant's ``entities`` table with no time filter, exactly as
    an explicit ``seed_entity="Ada"`` resolves.  Overlapping matches are settled longest first
    and then left to right: a query that says ``Project Kestrel`` seeds from that entity and
    not additionally from one named ``Kestrel``.  One statement, one bound parameter; the
    module docstring has the measured cost.
    """
    candidates = seed_candidates(query)
    if not candidates or limit <= 0:
        return []
    rows = con.execute(
        _SEED_MATCH_SQL, [_CANDIDATE_SEP.join(candidates), int(tenant_id)]
    ).fetchall()
    matches: list[tuple[int, int, int, str, int, int]] = []
    for eid, name, raw in rows:
        span = candidates.get(raw)
        if span is None:  # pragma: no cover - raw round-trips through DuckDB unchanged
            continue
        start, end = span
        matches.append((-(end - start), start, int(eid), str(name), start, end))
    matches.sort()
    chosen: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []
    seen: set[int] = set()
    for _length, _pos, eid, name, start, end in matches:
        if eid in seen or any(start < t_end and t_start < end for t_start, t_end in taken):
            continue
        chosen.append((eid, name))
        taken.append((start, end))
        seen.add(eid)
        if len(chosen) >= limit:
            break
    return chosen


def entity_name(con, *, tenant_id: int, entity_id: int) -> str | None:
    """The stored name of one entity in one tenant, or None.  Used to report an explicit seed
    in ``RecallHits.seeds`` the same way an auto-detected one is reported."""
    row = con.execute(
        f"SELECT name FROM entities e WHERE e.entity_id = ? AND {tenant_sql('e')}",
        [int(entity_id), int(tenant_id)],
    ).fetchone()
    return None if row is None or row[0] is None else str(row[0])


# --------------------------------------------------------------------------- vector arm

_VECTOR_MODULE = None


def _vector():
    """:mod:`anatid.vector`, imported on first use.

    Deferred rather than imported at the top, because that module reuses this one's embedding
    serialisation and kind filter, and importing both ways at module level is a cycle.  After
    the first call this is a dict lookup in ``sys.modules``.
    """
    global _VECTOR_MODULE
    if _VECTOR_MODULE is None:
        from . import vector as _module

        _VECTOR_MODULE = _module
    return _VECTOR_MODULE


def vector_scan_rows(
    con,
    *,
    tenant_id: int,
    as_of: AsOf = CURRENT,
    kinds: Sequence[str] | None = None,
    index=None,
    plan=None,
) -> int:
    """How many rows :func:`vector_arm` would score for this request.

    The same tenant, temporal and kind predicates as the arm itself, over rows that have an
    embedding -- that is the number the brute-force scan is linear in, and the number
    :data:`BRUTE_FORCE_CEILING` is compared against.  A ``count(*)`` with these predicates is a
    columnar scan of a few narrow columns: well under a millisecond at the ceiling, so the check
    costs a small fraction of the scan it guards.

    With a usable ``duckdb_vss`` generation (:func:`anatid.vector.resolve`) the arm scans only
    the rows written since that generation was built, and this returns that number instead, so
    the ceiling stops refusing a tenant the index can actually serve.  Nothing changes for the
    default exact backend.  ``plan`` is an already-resolved
    :class:`~anatid.vector.VectorPlan`, so one recall resolves once.
    """
    return _vector().scan_rows(
        con, tenant_id=tenant_id, as_of=as_of, kinds=kinds, index=index, plan=plan
    )


def vector_arm(
    con,
    *,
    tenant_id: int,
    embedding: Sequence[float],
    dim: int,
    topn: int = DEFAULT_CANDIDATES,
    as_of: AsOf = CURRENT,
    kinds: Sequence[str] | None = None,
    index=None,
    plan=None,
) -> list[tuple[int, float]]:
    """Cosine top-N over the tenant's visible memories.  ``[(memory_id, similarity)]``.

    The default backend is the brute-force scan of the tenant's ``embedding`` column, which is
    also the correctness oracle: see :data:`BRUTE_FORCE_CEILING`.  When the database has an
    HNSW generation from :mod:`anatid.vector` (``anatid.vector.attach(db)`` then
    ``db.maintain_indexes()``), the candidates come from that structure merged with the
    journal's pending rows, and are then filtered and scored exactly, so the answer this
    returns is scored the same way either way.  Ties break on ``memory_id ASC``.

    Call :func:`anatid.vector.search` instead of this when you want to know WHICH backend
    answered and why; this returns the hits alone, as it did in 0.1.1.
    """
    return _vector().search(
        con,
        tenant_id=tenant_id,
        embedding=embedding,
        dim=dim,
        topn=topn,
        as_of=as_of,
        kinds=kinds,
        index=index,
        plan=plan,
    ).hits


# --------------------------------------------------------------------------- BM25 arm
#
# The text arm lives in :mod:`anatid.fts`, which owns both halves of it: the derived-index
# framework (a published generation plus the journal, so a write is searchable with no rebuild)
# and 0.1.1's single file-wide index, for a database that has not attached one.  What is left
# here is the four names this module has always exported, each delegating.

#: 0.1.1's BM25 statement, re-exported under the name it had here.  It is the subject of a
#: source-level guard (``tests/test_recall_tenancy.py``) that the two file-wide corpus statistics
#: never appear in it, and that guard should keep reading the statement wherever it lives.
_BM25_SQL = _fts._BM25_SQL  # noqa: SLF001 -- the guard's subject, re-exported


def fts_index_present(con) -> bool:
    """True when a BM25 index has been built on this database.

    See :func:`anatid.fts.fts_index_present`.  With a framework index this is "a generation is
    published"; without one it probes the schema ``PRAGMA create_fts_index`` creates for
    :data:`~anatid.schema.FTS_SOURCE_TABLE`.  Either way, reporting no index is the fail-safe
    answer, and a schema-v2 ``fts_main_memories`` never counts: it is keyed on ``memory_id`` and
    leaks across tenants.
    """
    return _fts.fts_index_present(con)


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
    """Okapi BM25 top-N over **this tenant's** documents.  ``[(memory_id, score)]``.

    Nothing outside ``tenant_id`` can influence the result, on either path: the candidate
    documents are pruned to the tenant before scoring and ``df``, ``num_docs`` and ``avgdl`` are
    the tenant's own.  With a derived full-text index (:func:`anatid.fts.attach`) the answer
    merges the published generation with the journal, so a memory written a statement ago is
    already findable; without one it is 0.1.1's index, which cannot see rows written since its
    last rebuild.

    Call :func:`anatid.fts.search` instead of this when you want to know WHICH path answered and
    whether the answer was complete; this returns the ranked list alone, as it did in 0.1.1.
    """
    return _fts.search(
        con,
        tenant_id=tenant_id,
        query_text=query_text,
        topn=topn,
        as_of=as_of,
        kinds=kinds,
        available=available,
    ).hits


def fts_status(con, *, deep: bool = False, tenant_id: int | None = None) -> FtsStatus:
    """How far the text index has fallen behind ``memories``.  See :func:`anatid.fts.status`.

    ``tenant_id`` scopes the report to one tenant's own rows, which is what
    :func:`hybrid_recall` asks for: since schema v3 no other tenant's writes can change this
    tenant's BM25 rows or scores, so counting them would be a false alarm and a leak.
    """
    return _fts.status(con, tenant_id=tenant_id, deep=deep)


def rebuild_fts_index(con, *, now: _dt.datetime | None = None,
                      terms_index: bool = True) -> FtsStatus:
    """Rebuild the BM25 index.  See :func:`anatid.fts.rebuild`.

    With a derived full-text index this is ``build_next`` + ``validate`` + ``publish``: the new
    generation is built beside the live one and published by flipping one metadata row, so reads
    are never interrupted.  Without one it is 0.1.1's drop-and-recreate rebuild.  Never called
    implicitly by a read: an implicit rebuild would turn an unlucky ``recall()`` into a
    multi-second stall.
    """
    return _fts.rebuild(con, now=now, terms_index=terms_index)


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
    as_of: AsOf = CURRENT,
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

    ``as_of`` selects the **version** of each memory, with the same predicate the arms applied:
    a memory has one physical row per correction (schema v4), and the row an as-of read hands
    back is the one that was visible then, with the valid interval it had then, not the live
    row with the ``valid_to`` a later correction gave it.  An id whose version is no longer
    visible under ``as_of`` (a correction committed between the arm and this statement) is
    absent from the result; the callers skip it.
    """
    ids = [int(i) for i in memory_ids]
    if not ids:
        return {}
    # Through schema.memory_select, not MEMORY_COLUMNS directly: a handle that could not run
    # the migration ladder (read_only, or ensure=False) may be reading a schema-v3 file, where
    # naming the v4 `version` column fails to bind.
    cols = memory_select(con, alias="m", embedding=with_embedding)
    w, wp = Visibility.at(tenant_id, as_of).predicate("m")
    sql = (f"SELECT {cols} FROM memories m "
           f"WHERE m.memory_id IN ({_int_list(ids)}) AND {w}")
    return {int(r[0]): Memory.from_row(r) for r in con.execute(sql, wp).fetchall()}


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
    aw, ap = Visibility.at(tenant_id, as_of).predicate("a")
    sql = (
        f"SELECT a.src, list(e.name ORDER BY e.entity_id) "
        f"FROM edges_about a JOIN entities e ON e.entity_id = a.dst AND e.tenant_id = a.tenant_id "
        f"WHERE a.src IN ({_int_list(ids)}) AND {aw} GROUP BY a.src"
    )
    return {int(r[0]): tuple(n for n in r[1] if n is not None)
            for r in con.execute(sql, ap).fetchall()}


# --------------------------------------------------------------------------- hybrid

def hybrid_recall(
    con,
    *,
    tenant_id: int,
    query: str | None = None,
    embedding: Sequence[float] | None = None,
    dim: int,
    k: int = 10,
    seed_entity: int | Sequence[int] | str | None = AUTO_SEED,
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
    vector_index=None,
    on_ceiling: str = "error",
) -> RecallHits:
    """Hybrid retrieval: cosine top-N + BM25 top-N + optional k-hop graph expansion, fused by RRF.

    Arms run only when their input is present: the vector arm needs ``embedding``, the BM25 arm
    needs ``query`` **and** an fts index, the graph arm needs a seed.  With no usable arm the
    result is empty rather than a silent full-table scan.

    ``seed_entity`` is where the graph arm starts:

    * :data:`AUTO_SEED` (``"auto"``, the default): the seeds are the entities the query names
      (:func:`auto_seeds`), up to :data:`AUTO_SEED_LIMIT` of them, longest name first.  With no
      query, or a query that names no entity, the graph arm does not run.
    * an ``int`` entity id, or a sequence of them: expand from exactly those.
    * ``None``: no graph arm.

    The result says what happened: ``RecallHits.arms`` lists the arms that ran and
    ``RecallHits.seeds`` the entity names the graph arm expanded from, in the order they ran.

    ``vector_index`` is this handle's :class:`anatid.vector.VectorIndex` when it holds one.  It
    is optional: the vector backend is recorded in the FILE, so the arm finds the published
    generation through the catalog whether or not the calling handle holds the accelerator.
    Passing it only lets a load failure on this handle be reported as such.

    ``on_stale_fts``: ``"report"`` (default -- set ``bm25_stale`` on the result and log a warning),
    ``"error"`` (raise :class:`~anatid.errors.StaleIndexError`), or ``"ignore"``.  Staleness is
    judged against ``tenant_id``'s own rows only (:func:`fts_status`), so a busy neighbour in a
    shared file does not make this tenant's fresh index look stale.

    The brute-force cosine ceiling is enforced here: when ``embedding`` is given and the rows the
    vector arm would scan (:func:`vector_scan_rows`) exceed :data:`BRUTE_FORCE_CEILING`, this
    raises :class:`~anatid.errors.BruteForceCeilingError` before running any arm, unless
    ``allow_slow=True``.  ``on_ceiling="skip"`` drops the vector arm instead and says so in
    ``notes``; :meth:`anatid.Anatid.recall` passes that when the embedding came from the
    handle's own embedder rather than from the caller, so configuring an embedder never turns
    a recall that used to answer into one that raises.  The other two arms are unaffected by
    the tenant's size; leave ``embedding`` out to use them alone.  See the module docstring
    for the measured costs.
    """
    arms: dict[str, list[tuple[int, float]]] = {}
    notes: list[str] = []
    if on_ceiling not in ("error", "skip"):
        raise ValueError(f"on_ceiling must be 'error' or 'skip', got {on_ceiling!r}")

    if embedding is not None:
        # Resolved once and handed to both calls: the ceiling check and the arm must agree on
        # which backend is answering, and a write between the two would otherwise change it.
        plan = _vector().resolve(con, tenant_id=tenant_id, as_of=as_of, index=vector_index)
        if not allow_slow:
            ceiling = int(BRUTE_FORCE_CEILING)
            rows_to_scan = vector_scan_rows(con, tenant_id=tenant_id, as_of=as_of, kinds=kinds,
                                            plan=plan)
            if rows_to_scan > ceiling and on_ceiling == "skip":
                notes.append(
                    f"vector arm skipped: it would brute-force scan {rows_to_scan} embeddings "
                    f"in tenant {int(tenant_id)}, above BRUTE_FORCE_CEILING={ceiling}, and this "
                    f"database has no usable ANN generation ({plan.reason.value}: "
                    f"{plan.detail}); pass embedding= and allow_slow=True to run it anyway, "
                    f"or attach the duckdb_vss backend (anatid.vector.attach)")
                embedding = None
            elif rows_to_scan > ceiling:
                raise BruteForceCeilingError(
                    f"the vector arm would brute-force scan {rows_to_scan} embeddings in tenant "
                    f"{int(tenant_id)}, above BRUTE_FORCE_CEILING={ceiling}; this database has "
                    f"no usable ANN generation ({plan.reason.value}: {plan.detail}) and anatid "
                    f"refuses to get quietly slower. Pass allow_slow=True to run the scan "
                    f"anyway, omit embedding= to answer from the text and graph arms, attach "
                    f"the duckdb_vss backend (anatid.vector.attach), or shard the tenant into "
                    f"its own file (anatid.DatabasePool).",
                    tenant_id=int(tenant_id), rows=rows_to_scan, ceiling=ceiling)
        if embedding is not None:
            arms["vector"] = vector_arm(con, tenant_id=tenant_id, embedding=embedding, dim=dim,
                                        topn=candidates, as_of=as_of, kinds=kinds, plan=plan)

    # Staleness is asked for per tenant: a rebuild is file-wide, but since schema v3 another
    # tenant's rows can change neither this tenant's BM25 hits nor its scores, so counting them
    # as "pending" would be a false alarm -- and would leak that the other tenant wrote at all.
    if query:
        # Resolved once and handed to both calls, as the vector arm does above: the staleness
        # this result reports and the generation the arm actually reads must be the same one.
        text_plan = _fts.resolve(con, tenant_id=tenant_id, as_of=as_of)
        status = _fts.status(con, tenant_id=tenant_id, plan=text_plan)
        if status.available:
            arms["text"] = _fts.search(con, tenant_id=tenant_id, query_text=query,
                                       topn=candidates, as_of=as_of, kinds=kinds,
                                       plan=text_plan, available=True).hits
        else:
            notes.append("no fts index on memories: BM25 arm skipped "
                         "(call rebuild_fts_index() to create it)")
    else:
        status = FtsStatus(available=False, stale=False, indexed_rows=None, current_rows=0,
                           pending_rows=0, indexed_at=None, policy=FTS_STALENESS_POLICY)

    # The graph arm's seeds: named by the caller, or found in the query.
    seeds: list[tuple[int, str]] = []
    if isinstance(seed_entity, str):
        if seed_entity != AUTO_SEED:
            raise TypeError(
                f"hybrid_recall takes seed_entity as an entity id, a sequence of ids, "
                f"{AUTO_SEED!r} or None, not the name {seed_entity!r}; resolve names with "
                f"Anatid.entity_id() or call Anatid.recall()")
        if query:
            seeds = auto_seeds(con, tenant_id=tenant_id, query=query)
    elif isinstance(seed_entity, bool):
        raise TypeError("seed_entity must be an entity id, a sequence of ids, 'auto' or None, "
                        "not a bool")
    elif isinstance(seed_entity, int):
        sid = int(seed_entity)
        seeds = [(sid, entity_name(con, tenant_id=tenant_id, entity_id=sid) or str(sid))]
    elif seed_entity is not None:
        for sid in seed_entity:
            sid = int(sid)
            if all(sid != s for s, _n in seeds):
                seeds.append((sid, entity_name(con, tenant_id=tenant_id, entity_id=sid)
                              or str(sid)))
    if seeds:
        arms["graph"] = _graph_candidates(con, tenant_id=tenant_id,
                                          seed_entity_ids=[sid for sid, _n in seeds],
                                          hops=hops, topn=candidates, as_of=as_of,
                                          backend=backend, kinds=kinds)

    if query and status.stale:
        # What "stale" MEANS depends on which half of anatid.fts answered, so the sentence comes
        # from there: for 0.1.1's index it is how many rows the arm cannot see, for a derived
        # generation it is that no generation was usable and the corpus was too large to scan.
        msg = _fts.staleness_message(status)
        if on_stale_fts == "error":
            raise StaleIndexError(msg)
        if on_stale_fts != "ignore":
            log.warning("%s", msg)
            notes.append(msg)

    fused = rrf_fuse(arms, k=rrf_k, top=k)
    ids = [mid for mid, _s, _r, _sc in fused]
    rows = hydrate(con, ids, tenant_id=tenant_id, with_embedding=with_embedding, as_of=as_of)
    names = (about_names(con, ids, tenant_id=tenant_id, as_of=as_of)
             if include_about else {})

    hits = []
    for rank, (mid, score, ranks, scores) in enumerate(fused, start=1):
        mem = rows.get(mid)
        if mem is None:            # raced with a purge or a correction between arm and hydration
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
        seeds=tuple(name for _sid, name in seeds),
    )
