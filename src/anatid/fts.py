"""Full-text search: the BM25 arm, on the derived-index framework.

This module owns every line of anatid's text retrieval, and it has two halves.

The **framework half** implements ``docs/design/derived-index-framework.md`` for full text: a
published *generation* is a BM25 index built from a snapshot of ``memories``, an ordered
*journal* records every document written or closed since that snapshot, and a search merges the
two before :class:`~anatid.visibility.Visibility` decides what survives.  A write is searchable
by the very next ``recall()`` with no rebuild, on the handle that wrote it and on any other
handle on the file, because the journal row is written inside the transaction that wrote the
memory and the definition lives in the FILE rather than on a handle.

The **legacy half** (:func:`legacy_bm25_arm`, :func:`legacy_rebuild`) is 0.1.1's single
file-wide index over :data:`~anatid.schema.FTS_SOURCE_TABLE`, kept verbatim.  A database where
nobody has called :func:`attach` behaves exactly as it did in 0.1.1.  :func:`resolve` is what
chooses, and it chooses the framework only when the file records an enabled ``fts`` index.

Document identity
-----------------
The document key is the composite ``'<tenant_id>:<memory_id>'``
(:func:`anatid.schema.fts_doc_id`), introduced in schema v3 because ``memory_id`` is unique only
*within* a tenant and DuckDB's fts extension needs a key unique over the indexed table.  Journal
rows are keyed ``(tenant_id, doc_id)`` for the same reason: one tenant's change must not land on
another tenant's document 42.  One generation covers the whole file (``per_tenant = False``) --
``PRAGMA create_fts_index`` has a fixed per-call cost that a per-tenant index would multiply --
and per-tenant scoping is a predicate in the query, not a separate index.

What a search does
------------------
::

    base generation  (this tenant's documents at the watermark,
                      minus every document the journal has touched)
      UNION ALL
    journal rescan   (every document the journal HAS touched: inserts and closes,
                      re-read from `memories` and scored from their current content)
      -> one BM25 model
      -> tenant and time predicate on the canonical rows, before the top-N cut
      -> ORDER BY score DESC, memory_id ASC LIMIT n

The base is authoritative for documents the journal has not mentioned; **the journal is
authoritative for every document it has mentioned**.  That is why tombstones are subtracted from
the base rather than from the answer.  A ``close`` in anatid's journal does not mean the document
is gone: ``reinforce(confidence=...)`` closes one version and opens the next of the same
``memory_id``, so it journals a ``close`` for a memory that is still perfectly current, and
subtracting that id from the answer would lose a live memory.  Subtracting it from the *base* and
rescoring it from the canonical row is exact in both directions -- a superseded or forgotten
memory fails the visibility predicate and disappears, a reinforced one passes and stays -- and
``tests/test_fts_framework.py`` checks both.

Scoring: one model, reconstructed
---------------------------------
The design document is explicit that scores from two independently built BM25 corpora are not
comparable, so anatid does not build a second corpus and does not rank-fuse two lists.  It
reconstructs the ONE set of corpus statistics that describes base + journal together, and scores
both arms with it.  Okapi BM25, k1=1.2, b=0.75, ``idf = ln((N - df + 0.5) / (df + 0.5) + 1)``,
where for one tenant:

* ``N`` and ``avgdl`` come from the base documents the journal has NOT touched -- whose lengths
  the build recorded -- plus the documents it HAS, measured now.  Those two sets are disjoint
  and their union is exactly the document set a rebuild would index;
* ``df`` the same way: the postings, already restricted to the untouched documents, give the
  untouched half for free (one row per matching document), and the rescanned half is counted;
* ``tf`` and the document length of a rescanned document are tokenised in SQL with exactly the
  tokenizer the index was built with (:data:`~anatid.schema.FTS_TOKENIZER`, as
  :func:`tokens_sql`).  ``tests/test_fts_framework.py`` checks that this produces the same
  length the extension recorded in ``fts_main_*.docs.len``.

The property that buys is worth stating plainly: **a rebuild never changes an answer.**  The
scores and the order a search returns with a journal of a thousand documents are the ones it
returns after those documents are absorbed into a new generation; the journal is a performance
question, not a ranking question, and ``tests/test_fts_framework.py`` asserts the two answers
are equal after random sequences of writes, corrections and purges.  Neither the generation's
``df`` dictionary nor its statistics table is read by a search, so an erasure that leaves those
two counting a document it deleted cannot skew a score either.  The dictionary is still joined,
to prune the query's terms to ones this tenant's base has seen, which keeps a term nobody in
this tenant used from costing a posting scan.

Two corpora, one model
----------------------
The fallback :func:`scan` is a different model, deliberately: with no base it can only score
over the documents it can see, so its corpus is the tenant's VISIBLE documents, while a
generation's is every document in the file.  A base that indexed only what is current could not
answer an ``as_of`` read, which is why it indexes superseded versions too and the visibility
predicate decides afterwards.  The two therefore return the same ROWS -- the tests assert that
after every mutation -- and can order two close scores differently, because one corpus contains
documents the other does not.

Corpus statistics are PER TENANT
--------------------------------
``df``, ``num_docs`` and ``avgdl`` describe one tenant's documents and nothing else, on every
path.  A generation reconstructs them from its document map restricted to the tenant and from
that tenant's journal rows; the legacy index reads them from its per-tenant dictionary and
statistics tables; the fallback scan computes them over the tenant's own corpus.  So another
tenant's writes change neither the rows nor the SCORES this tenant gets, and the journal is read
per tenant, so ten thousand documents added by a neighbour do not even make this tenant's index
look stale.  ``tests/test_fts_framework.py`` checks all three.

What is still shared is the extension's own ``fts_main_*.dict``, and only as a term -> termid
map: a term another tenant used has a termid but no posting in this tenant's slice of the
document map, so it contributes nothing.  The ``df`` column of that table and
``fts_main_*.stats`` count every tenant's documents and are never read.

When there is no usable generation
----------------------------------
An index may be stale, corrupt or absent without making a query wrong.  With no usable
generation the search falls back to :func:`scan`: exact BM25 over the tenant's visible documents,
tokenised in SQL, with the corpus statistics computed from that same scan.  It is the oracle --
:meth:`FtsIndex._validate` compares a new generation against it and the tests compare the merged
answer against it after random mutations -- and it is O(corpus) per query, so it is refused above
:data:`SCAN_CEILING` documents rather than turning an unlucky ``recall()`` into a multi-second
stall.  The reason is reported either way, as a :class:`~anatid.derived.HealthReason`: absent,
rebuild in progress, stale generation, load failure.

Historical reads
----------------
A current-state index cannot answer ``as_of``, which is why the framework's
:meth:`~anatid.derived.DerivedIndex.pin` refuses one.  This index is not a current-state index:
its base holds document *identity* and *content*, and every version of a memory carries the same
content, so nothing in the base depends on which version is current.  The time axes live on the
canonical rows the search joins.  :data:`ANSWERS_HISTORICAL` is therefore True, an ``as_of``
search uses a generation like any other, and the predicate from :mod:`anatid.visibility` picks
the version that was visible then.  The one thing that could make that wrong -- a document erased
after the generation was built -- is handled destructively rather than by falling back:
:meth:`FtsIndex._erase` deletes the document from every generation's postings, dictionary,
document map and source table inside ``forget(hard=True)``'s transaction, and a generation whose
storage cannot be reached is invalidated instead.

Maintenance
-----------
:func:`rebuild` is ``build_next`` + ``validate`` + ``publish``: the new generation is built
beside the live one, compared with the oracle, and published by flipping one metadata row, so a
reader that resolved the old generation keeps reading it and a reader that pinned it can stop
:meth:`~anatid.derived.DerivedIndex.retire` from dropping it.
``MaintenancePolicy(rebuild_after_rows=10_000, rebuild_after_ratio=0.05,
rebuild_after_seconds=900)`` -- the framework defaults -- decide when
:func:`anatid.derived.maintain` does that for you.  Those numbers bound how many documents a
search rescans, not how wrong the answer is.

What it costs
-------------
Measured on this build (duckdb 1.5.5, macOS arm64), one tenant of 8-token documents, top-10,
p50 of 15 to 25 calls, the whole of :func:`search` including :func:`resolve`:

=================  ==================  ==========  ===========  ==============  ===========
corpus             journal             generation  scan         0.1.1's index   build
=================  ==================  ==========  ===========  ==============  ===========
10,000 documents   empty               10.1 ms     8.6 ms       n/a             191 ms
10,000 documents   100 documents       11.9 ms     n/a          n/a             n/a
10,000 documents   1,000 documents     12.2 ms     n/a          n/a             n/a
100,000 documents  empty               19.1 ms     34.4 ms      20.5 ms         2.0 s
100,000 documents  1,000 documents     21.0 ms     34.4 ms      20.5 ms         n/a
=================  ==================  ==========  ===========  ==============  ===========

Repeated runs move those by up to 10%.  End to end, ``db.recall("...")`` over the 11,000-document
corpus with a thousand pending is 15.6 ms: the search plus the staleness report, the hydration
and the ABOUT names.  Three things are worth reading off the table rather than assuming.

**The journal is nearly free.**  A thousand documents written since the build cost 1 to 2 ms,
because the rescan is linear in the JOURNAL and not in the corpus.  At 100,000 documents a
search with a thousand pending costs 21.0 ms against 0.1.1's 20.5 ms for an index that cannot
see any of them: being current is not what the index costs you.

**Below a few tens of thousands of documents, the index-free scan is the faster path.**  DuckDB
tokenises and aggregates 10,000 documents in 8.6 ms, against 10.1 ms to consult a generation
(1.8 ms of which is :func:`resolve`: the catalog read, the storage probe and the journal count,
resolved once per recall and handed to both the staleness report and the arm).  The generation
pays for itself from about 100,000 documents, where it is 19.1 ms against 34.4 ms.  That is why
a database with no generation is not a broken database, and why :data:`SCAN_CEILING` sits where
the vector arm's ceiling does rather than lower.

**A build is the expensive operation, and nothing waits for it.**  Two seconds at 100,000
documents (0.1.1's in-place rebuild takes 0.66 s for the same corpus; a generation is written to
new tables rather than over the live ones).  Reads answer from the previous generation for all
of it, and publication is one metadata row.

On the WRITE side, journalling costs ``remember()`` 0.76 ms (1.13 ms without a full-text index,
1.89 ms with one): one ``INSERT`` per event, most of it DuckDB's fixed per-statement cost.  That
is the price of the write being searchable by the next read, and it is what
``Anatid.open(accelerators=False)`` declines.  Opening with the default attaches this index, so
the searchable-immediately behaviour above is what an ordinary anatid database does.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import logging
import random
import threading
import weakref
from dataclasses import dataclass, replace
from typing import Any, Iterator, Sequence

import duckdb

from . import schema as _schema
from .derived import (
    DerivedIndex,
    Generation,
    HealthReason,
    HealthReport,
    IndexDefinition,
    MaintenancePolicy,
    ValidationReport,
    journal_latest_sql,
)
from .errors import IndexValidationError, StaleIndexError
from .schema import (
    FTS_DICT_TABLE,
    FTS_DOC_ID_SQL,
    FTS_DOCS_TABLE,
    FTS_INDEX_SCHEMA,
    FTS_STATS_TABLE,
    FTS_TOKENIZER,
    INDEX_GENERATIONS_TABLE,
    INDEX_REGISTRY_TABLE,
    VERSION_COLUMN,
    quote_ident,
)
from .types import CURRENT, AsOf, FtsStatus, Namespace, to_utc_naive, utcnow
from .visibility import Visibility, live_row_sql, tenant_sql

log = logging.getLogger("anatid.fts")

__all__ = [
    "BM25_K1",
    "BM25_B",
    "FTS_INDEX_NAME",
    "STORAGE_PREFIX",
    "SCAN_CEILING",
    "MAX_MERGED_JOURNAL_ROWS",
    "ANSWERS_HISTORICAL",
    "VALIDATE_SAMPLE",
    "FTS_GENERATION_POLICY",
    "FTS_STALENESS_POLICY",
    "FtsIndex",
    "FtsPlan",
    "FtsSearch",
    "FtsStorage",
    "attach",
    "detach",
    "index_of",
    "resolve",
    "search",
    "scan",
    "corpus_size",
    "status",
    "rebuild",
    "staleness_message",
    "tokens_sql",
    "newest_version_sql",
    "bm25_arm",
    "fts_status",
    "fts_index_present",
    "rebuild_fts_index",
    "legacy_bm25_arm",
    "legacy_fts_status",
    "legacy_rebuild",
    "ensure_fts_extension",
]

#: Okapi BM25 parameters: the spike's numbers, and the ones ``match_bm25`` uses.
BM25_K1, BM25_B = 1.2, 0.75

#: The index name in :data:`~anatid.schema.INDEX_REGISTRY_TABLE`.  Matches the placeholder
#: :class:`~anatid.derived.NullIndex` every :class:`~anatid.derived.IndexRegistry` starts with,
#: so attaching a real one replaces the placeholder.
FTS_INDEX_NAME = "fts"

#: Prefix for a generation's storage.  ``anatid_idx_fts_g<n>`` and four names derived from it.
STORAGE_PREFIX = "anatid_idx"

#: Default candidate count, matching :data:`anatid.recall.DEFAULT_CANDIDATES`.
DEFAULT_TOPN = 50

#: This index answers ``as_of`` reads from a generation; see the module docstring.  A class
#: attribute on :class:`FtsIndex` and a module constant because :func:`resolve` needs it without
#: an index object.
ANSWERS_HISTORICAL = True

#: Above this many documents touched by the journal for one tenant, the rescan stops being
#: cheaper than the corpus and the read falls back to :func:`scan` with ``stale_generation``.
#: Sits at the default :attr:`~anatid.derived.MaintenancePolicy.rebuild_after_rows`, so a
#: database whose maintenance runs never reaches it.
MAX_MERGED_JOURNAL_ROWS = 10_000

#: How many of a tenant's documents :func:`scan` will tokenise before it refuses.  The scan is
#: exact BM25 with no index at all -- the fallback for every state in which a generation cannot
#: be used -- and it is linear in the tenant's corpus: measured 5.5 ms at 1,000 documents,
#: 8.6 ms at 10,000 and 34.4 ms at 100,000 on this machine (duckdb 1.5.5, macOS arm64, 8-token
#: documents), so a search AT this ceiling costs about 34 ms.  It is deliberately the same number
#: as :data:`anatid.recall.BRUTE_FORCE_CEILING`: both are "past here, anatid stops calling a
#: linear scan cheap".  Above it a text ``recall()`` with no usable generation returns nothing
#: and reports why, on :class:`FtsSearch` and on :attr:`anatid.types.FtsStatus.stale`, exactly as
#: the brute-force ceiling reports the vector arm.  Call ``maintain_indexes()``.
SCAN_CEILING = 100_000

#: Documents sampled from a new generation when it is validated; each contributes one query term.
VALIDATE_SAMPLE = 32

#: What :func:`status` reports about the framework index, stored on
#: :attr:`anatid.types.FtsStatus.policy` so a caller can tell the two halves apart.
FTS_GENERATION_POLICY = (
    "full text runs on the derived-index framework (anatid.fts): a published base generation "
    "plus an ordered journal written in the same transaction as the memory. A write is "
    "searchable immediately, on this handle and on any other handle on the file, with no "
    "rebuild -- pending_rows counts the documents a search rescans from the canonical rows, not "
    "documents it cannot see. stale means the answer would NOT be exact: no generation is usable "
    "AND the tenant's corpus is above anatid.fts.SCAN_CEILING, so the fallback scan was refused. "
    "Rebuild policy: MaintenancePolicy(rebuild_after_rows=10_000, rebuild_after_ratio=0.05, "
    "rebuild_after_seconds=900) through maintain_indexes(); a rebuild builds the next generation "
    "beside the live one, validates it against the scan oracle, and publishes it by flipping one "
    "metadata row, so reads are never interrupted."
)

#: 0.1.1's policy sentence, for a database that has not attached the framework index.
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
    "your threshold; the staleness window is the interval between those rebuilds. "
    "anatid.fts.attach(db) moves this database onto the derived-index framework, where a write "
    "is searchable with no rebuild at all."
)

_REGISTRY_COLUMNS = "kind, per_tenant, source_table, source_id_column, delta_mode, params, enabled"
_GENERATION_COLUMNS = (
    "generation, tenant_id, watermark_id, watermark_ts, built_at, validated, published, "
    "coalesce(published_unvalidated, FALSE), stats, notes"
)

#: Does ``PRAGMA create_fts_index`` have a schema for this generation?  ``duckdb_tables()``
#: rather than ``information_schema.tables``: the same answer for 0.31 ms against 0.56 ms on
#: this machine, and this runs on every text recall.
_OBJECTS_PRESENT_SQL = (
    "SELECT count(*) FROM duckdb_tables() WHERE schema_name = ? AND table_name = 'docs'"
)


# --------------------------------------------------------------------------- helpers


#: The tokenizer, as SQL.  Mirror of :data:`anatid.schema.FTS_TOKENIZER`: lower-case, then treat
#: every run of ``(\.|[^a-z])`` as a separator.  Empty strings survive the split and are dropped
#: by every caller.
_TOKEN_SPLIT = r"string_split_regex(regexp_replace(lower({expr}), '(\.|[^a-z])+', ' ', 'g'), '\s+')"


def tokens_sql(expr: str) -> str:
    """SQL splitting ``expr`` into the terms the fts index stores (empty strings included)."""
    return _TOKEN_SPLIT.format(expr=expr)


def newest_version_sql(alias: str | None = None) -> str:
    """``QUALIFY`` selecting one row per ``(tenant_id, memory_id)``: the newest version.

    Every version of a memory carries the same ``content``, so this is a de-duplication and not
    a choice of which belief to index.  Mirror of the QUALIFY in
    :func:`anatid.schema.fts_rebuild_statements`; ``tests/test_fts_framework.py`` checks that
    the two select the same document set.
    """
    col = (lambda name: f"{alias}.{name}") if alias else (lambda name: name)
    return (
        f"QUALIFY row_number() OVER (PARTITION BY {col('tenant_id')}, {col('memory_id')} "
        f"ORDER BY {col(VERSION_COLUMN)} DESC NULLS LAST, {col('tx_from')} DESC NULLS LAST, "
        f"{col('created_at')} DESC NULLS LAST) = 1"
    )


_NEWEST_VERSION = newest_version_sql()


def _kinds(alias: str, kinds: Sequence[str] | None) -> tuple[str, list]:
    """``AND <alias>.kind IN (...)``, from :mod:`anatid.recall`.

    Deferred, like :mod:`anatid.vector` does it: that module imports this one, and importing
    both ways at module level is a cycle.
    """
    from .recall import _kind_filter

    return _kind_filter(alias, kinds)


def _tenant_of(tenant: Any) -> int:
    if isinstance(tenant, Namespace):
        return int(tenant.tenant_id)
    return int(tenant)


def _json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    import json

    try:
        out = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def _bm25_term(tf: str, length: str, df: str, stats: str = "s") -> str:
    """One term's Okapi BM25 contribution, written once so both arms cannot drift apart."""
    return (
        f"ln(({stats}.num_docs - {df} + 0.5) / ({df} + 0.5) + 1) "
        f"* {tf} * ({BM25_K1} + 1) "
        f"/ ({tf} + {BM25_K1} * (1 - {BM25_B} + {BM25_B} * {length} / {stats}.avgdl))"
    )


# --------------------------------------------------------------------------- value types


@dataclass(frozen=True, slots=True)
class FtsStorage:
    """The five objects one generation of the full-text index owns.

    All five names come from :meth:`anatid.derived.Generation.storage_name`, so two generations
    never share a table and a new one is built beside the live one.  ``index_schema`` is what
    ``PRAGMA create_fts_index`` creates for ``source``.
    """

    source: str
    index_schema: str
    docmap: str
    dictionary: str
    stats: str

    @classmethod
    def of(cls, generation: Generation) -> "FtsStorage":
        base = generation.storage_name(STORAGE_PREFIX)
        return cls(
            source=base,
            index_schema=f"fts_main_{base}",
            docmap=quote_ident(f"{base}_docmap"),
            dictionary=quote_ident(f"{base}_dict"),
            stats=quote_ident(f"{base}_stats"),
        )

    @property
    def tables(self) -> tuple[str, ...]:
        return (self.source, self.docmap, self.dictionary, self.stats)

    def bare(self, field: str) -> str:
        """One name without its quotes, for comparing against
        :func:`anatid.schema.table_names`."""
        return str(getattr(self, field)).strip('"')


@dataclass(frozen=True, slots=True)
class FtsPlan:
    """How a text read for one tenant will be answered, and why.

    ``usable`` False means no generation can serve this read and :func:`scan` will; ``reason``
    and ``detail`` say which of the framework's reasons applies.  ``journal_rows`` is how many
    documents the rescan covers, which is the work a write costs a read until the next rebuild.
    """

    usable: bool
    reason: HealthReason
    detail: str = ""
    generation: Generation | None = None
    storage: FtsStorage | None = None
    journal_rows: int = 0
    overflowed: bool = False

    def __bool__(self) -> bool:
        return self.usable


@dataclass(frozen=True, slots=True)
class FtsSearch:
    """One text search: the hits, and how they were obtained.

    ``hits`` is ``[(memory_id, bm25_score)]`` in rank order, ready for
    :func:`anatid.recall.rrf_fuse`.  ``exact`` says whether the answer is complete: True for a
    merged generation + journal answer and for the fallback scan, False only when no generation
    was usable AND the corpus was above :data:`SCAN_CEILING`, or when the legacy index answered
    (it cannot see rows written since its last rebuild).  ``reason`` always says WHY a generation
    was not used.
    """

    hits: list[tuple[int, float]]
    backend: str = "generation"
    reason: HealthReason = HealthReason.FRESH
    exact: bool = True
    detail: str = ""
    generation: int | None = None
    journal_rows: int = 0
    scanned_rows: int | None = None

    def __iter__(self) -> Iterator[tuple[int, float]]:
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "reason": self.reason.value,
            "exact": self.exact,
            "generation": self.generation,
            "journal_rows": self.journal_rows,
            "scanned_rows": self.scanned_rows,
            "hits": len(self.hits),
            "detail": self.detail,
        }


# --------------------------------------------------------------------------- catalog reads

_STATE_LOCK = threading.Lock()
#: Bumped whenever a full-text index is attached to or detached from any database in this
#: process.  The per-connection definition cache compares against it, so the common case (no
#: framework index in the file) costs no statement per query after the first.
_DEFINITIONS_EPOCH = 0
_DEFINITION_CACHE: "weakref.WeakKeyDictionary[Any, tuple[int, dict[str, Any] | None]]"
_DEFINITION_CACHE = weakref.WeakKeyDictionary()
#: Every :class:`FtsIndex` :func:`attach` has created in this process, weakly.  Used only to let
#: the connection-only entry points (:func:`rebuild_fts_index`, which is what
#: ``Anatid.rebuild_fts_index`` calls) find the object that owns the build.  Reads never need it.
_ATTACHED: list[Any] = []


def _bump_definitions() -> None:
    global _DEFINITIONS_EPOCH
    with _STATE_LOCK:
        _DEFINITIONS_EPOCH += 1
    with contextlib.suppress(TypeError):
        _DEFINITION_CACHE.clear()


def _definition(con) -> dict[str, Any] | None:
    """The persisted full-text definition as a plain dict, or ``None`` when the file has none.

    Read from the FILE rather than from a handle's registry, which is the whole point of the
    definition being persisted: a handle that never called :func:`attach` journals for the index
    and can search its generations.  Cached per connection against :data:`_DEFINITIONS_EPOCH`,
    because a database with no framework index must not pay a catalog read on every recall.

    :func:`attach` and :func:`detach` bump that epoch.  Disabling the definition some other way
    (``db.indexes.unregister("fts")``) leaves this cache saying the index is still there, which
    is safe rather than merely tolerable: every path that disables a definition invalidates its
    generations in the same transaction, so :func:`resolve` reads the catalog, finds an
    invalidated generation and falls back to the scan.
    """
    with _STATE_LOCK:
        epoch = _DEFINITIONS_EPOCH
    with contextlib.suppress(TypeError):
        hit = _DEFINITION_CACHE.get(con)
        if hit is not None and hit[0] == epoch:
            return hit[1]
    out: dict[str, Any] | None = None
    try:
        row = con.execute(
            f"SELECT {_REGISTRY_COLUMNS} FROM {INDEX_REGISTRY_TABLE} WHERE index_name = ?",
            [FTS_INDEX_NAME],
        ).fetchone()
    except duckdb.Error:  # a file older than the registry table has no derived indexes
        row = None
    if row is not None and bool(row[6]):
        out = {
            "kind": row[0],
            "per_tenant": bool(row[1]),
            "source_table": row[2],
            "source_id_column": row[3],
            "delta_mode": row[4],
            "params": _json(row[5]),
        }
    with contextlib.suppress(TypeError):
        _DEFINITION_CACHE[con] = (epoch, out)
    return out


def _generation_row(con) -> tuple[Generation | None, bool, FtsStorage | None]:
    """The published file-wide generation, whether one is building, and its storage if present.

    One statement for the catalog and one for the storage probe.  The probe is folded in because
    a generation whose tables an operator dropped by hand, or one retired between resolving and
    reading, has to become a reported fallback rather than an exception inside a caller's
    transaction.
    """
    rows = con.execute(
        f"SELECT {_GENERATION_COLUMNS} FROM {INDEX_GENERATIONS_TABLE} "
        f"WHERE index_name = ? AND tenant_id IS NULL "
        f"AND (published OR notes = 'building') ORDER BY generation",
        [FTS_INDEX_NAME],
    ).fetchall()
    building = any(r[9] == "building" for r in rows)
    published = [r for r in rows if bool(r[6])]
    if not published:
        return None, building, None
    r = published[-1]
    gen = Generation(
        index_name=FTS_INDEX_NAME,
        generation=int(r[0]),
        tenant_id=None if r[1] is None else int(r[1]),
        watermark_id=None if r[2] is None else int(r[2]),
        watermark_ts=r[3],
        built_at=r[4],
        validated=bool(r[5]),
        published=True,
        published_unvalidated=bool(r[7]),
        stats=_json(r[8]),
        notes=r[9],
    )
    st = FtsStorage.of(gen)
    present = bool(con.execute(_OBJECTS_PRESENT_SQL, [st.index_schema]).fetchone()[0])
    return gen, building, (st if present else None)


def _damage(generation: Generation, storage: FtsStorage, indexed: int, mapped: int) -> str | None:
    """Why a generation's base cannot be trusted, given its two row counts, or ``None``.

    Two invariants a build establishes and nothing anatid does breaks:

    * the source table and the document map hold the same documents.  A search reads the map to
      decide which documents the journal has NOT touched and to reconstruct the corpus
      statistics, so a map that has lost rows silently drops documents out of the ranking and
      changes every score.  :meth:`FtsIndex._validate` checks this too; this is the same check
      at read time, where the two counts ride along with the journal count and cost nothing.
    * the map has not GROWN past what the build recorded.  ``_erase`` shrinks both tables
      together inside a purge, so a smaller base is expected; a larger one is not.

    Neither is a full validation -- postings the map still points at could themselves be gone,
    and finding that out costs a scan of the corpus, which is what a rebuild does.  These are
    the checks a read can afford, and they turn the damage that is cheap to detect into a
    reported fallback rather than a short answer.
    """
    if indexed != mapped:
        return f"{storage.source} holds {indexed} document(s) but {storage.docmap} maps {mapped}"
    value = generation.stats.get("rows")
    recorded = int(value) if isinstance(value, (int, float)) else None
    if recorded is not None and mapped > recorded:
        return f"the base holds {mapped} document(s) against the {recorded} its build recorded"
    return None


def _census(con, tenant_id: int | None, generation: Generation, storage: FtsStorage) -> tuple:
    """``(journal rows, source rows, mapped rows)`` for one generation, in ONE statement.

    The journal count is what the plan reports and what the overflow branch compares; the other
    two are what :func:`_damage` reads.  They are fetched together because the cost of a read
    here is the round trip, not the counting: three scalar subqueries cost what one does.
    """
    sql, params = _journal_sql(tenant_id, generation)
    row = con.execute(
        f"SELECT (SELECT count(*) FROM ({sql})), (SELECT count(*) FROM {storage.source}), "
        f"(SELECT count(*) FROM {storage.docmap})",
        params,
    ).fetchone()
    return int(row[0]), int(row[1]), int(row[2])


def _base_damage(con, generation: Generation, storage: FtsStorage) -> str | None:
    """:func:`_damage` for a caller that holds no counts.  Used by the health report."""
    row = con.execute(
        f"SELECT (SELECT count(*) FROM {storage.source}), (SELECT count(*) FROM {storage.docmap})"
    ).fetchone()
    return _damage(generation, storage, int(row[0]), int(row[1]))


def _journal_sql(tenant_id: int | None, generation: Generation | None) -> tuple[str, list]:
    """``(tenant_id, doc_id, op)``: the current state of every document a generation has not
    absorbed.  Inserts AND closes -- both mean "the base's view of this document is out of
    date", and the search re-reads both from the canonical rows.

    The newest ``change_seq`` for a ``(tenant_id, doc_id)`` wins, which is what makes an id that
    was purged and then reused come back as an insert rather than as a tombstone.
    """
    number = 0 if generation is None else int(generation.generation)
    return journal_latest_sql(FTS_INDEX_NAME, tenant_id, number)


def _journal_count(con, tenant_id: int | None, generation: Generation | None) -> int:
    """How many documents a search would rescan, for one tenant or for the whole file."""
    sql, params = _journal_sql(tenant_id, generation)
    row = con.execute(f"SELECT count(*) FROM ({sql})", params).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


# --------------------------------------------------------------------------- planning


def _scan_plan(reason: HealthReason, detail: str, generation: Generation | None = None) -> FtsPlan:
    return FtsPlan(usable=False, reason=reason, detail=detail, generation=generation)


def resolve(
    con,
    *,
    tenant_id: int | None,
    as_of: AsOf | _dt.datetime | None = CURRENT,
    index: "FtsIndex | None" = None,
) -> FtsPlan:
    """Decide how a text read for ``tenant_id`` will be answered.

    ``tenant_id`` of ``None`` counts the journal over every tenant, which is what a file-wide
    :func:`status` wants; a search always names its tenant.

    Every branch that declines a generation names the reason, and every one of them still
    answers the query: :func:`scan` is always correct.  The order is the order in which a reason
    makes a generation unusable, not the order that is cheapest to check, so the reported reason
    is the FIRST thing wrong rather than the last.

    ``index`` is the accelerator object when the caller holds one; it is used for its
    ``load_error`` only.  The generation is read from the catalog either way, so a handle with no
    full-text code plans the same read as the handle that built the index.

    An ``as_of`` read does NOT decline a generation here (:data:`ANSWERS_HISTORICAL`); see the
    module docstring for why this index can answer one and a current-state index cannot.
    """
    scope = AsOf.coerce(as_of)
    if index is not None and index.load_error:
        return _scan_plan(HealthReason.LOAD_FAILURE, index.load_error)
    if not scope.is_current and not ANSWERS_HISTORICAL:  # pragma: no cover - constant is True
        return _scan_plan(
            HealthReason.HISTORICAL_QUERY,
            "a current-state index cannot answer an as_of read; the scan does",
        )
    if _definition(con) is None:
        return _scan_plan(
            HealthReason.ABSENT,
            "this database defines no derived full-text index (anatid.fts.attach)",
        )
    try:
        gen, building, storage = _generation_row(con)
    except duckdb.Error as exc:
        return _scan_plan(HealthReason.LOAD_FAILURE, f"the index catalog could not be read: {exc}")
    if gen is None:
        if building:
            return _scan_plan(
                HealthReason.REBUILD_IN_PROGRESS,
                "no generation is published and one is being built",
            )
        return _scan_plan(HealthReason.ABSENT, "no generation has been published")
    if not gen.validated and not gen.usable_without_validation:
        return _scan_plan(
            HealthReason.STALE_GENERATION,
            gen.notes or f"generation {gen.generation} is not validated",
            gen,
        )
    if storage is None:
        return _scan_plan(
            HealthReason.LOAD_FAILURE,
            f"generation {gen.generation} is published but its full-text objects "
            f"({FtsStorage.of(gen).source}) are not in this database",
            gen,
        )
    whose = "every tenant" if tenant_id is None else f"tenant {int(tenant_id)}"
    try:
        touched, indexed, mapped = _census(
            con, None if tenant_id is None else int(tenant_id), gen, storage
        )
    except duckdb.Error as exc:
        return _scan_plan(HealthReason.LOAD_FAILURE, f"the index journal could not be read: {exc}")
    damage = _damage(gen, storage, indexed, mapped)
    if damage is not None:
        return _scan_plan(
            HealthReason.DAMAGED_BASE,
            f"generation {gen.generation}: {damage}; the scan answers instead",
            gen,
        )
    if touched > MAX_MERGED_JOURNAL_ROWS:
        return FtsPlan(
            usable=False,
            reason=HealthReason.STALE_GENERATION,
            detail=(
                f"{touched} document(s) of {whose} have changed since generation "
                f"{gen.generation}, above MAX_MERGED_JOURNAL_ROWS={MAX_MERGED_JOURNAL_ROWS}; "
                f"rebuild it (maintain_indexes)"
            ),
            generation=gen,
            journal_rows=touched,
            overflowed=True,
        )
    reason = HealthReason.UNVALIDATED if gen.usable_without_validation else HealthReason.FRESH
    detail = (
        f"generation {gen.generation} was published without validation"
        if gen.usable_without_validation
        else f"generation {gen.generation}, {touched} document(s) rescanned from the journal"
    )
    return FtsPlan(
        usable=True,
        reason=reason,
        detail=detail,
        generation=gen,
        storage=storage,
        journal_rows=touched,
    )


# --------------------------------------------------------------------------- the read


def _query_cte() -> str:
    return (
        f"q AS (SELECT DISTINCT term FROM (SELECT unnest({tokens_sql('?')}) AS term) "
        f"WHERE term <> '')"
    )


def _rescan_ctes(source: str) -> str:
    """Tokenise, measure and count the documents a read scores from the canonical rows.

    ``source`` is a relation of ``(memory_id, content)``.  Everything below it is shared by the
    journal arm and by the fallback scan, which is why the two cannot drift apart: the same
    tokenizer, the same document length, the same ``df`` and the same corpus size.

    ``jlen`` comes back through a LEFT JOIN on the source rather than straight from the token
    counts, so a document with no tokens at all still counts as a document with length 0,
    exactly as it does in the base (the extension gives it a ``docs`` row either way).  The
    content is tokenised once and once only: doing it a second time to measure the length cost
    60% on the fallback scan.
    """
    return (
        f"jt AS (SELECT memory_id, term FROM "
        f"(SELECT memory_id, unnest({tokens_sql('content')}) AS term FROM {source}) "
        f"WHERE term <> ''), "
        f"jcount AS (SELECT memory_id, count(*)::DOUBLE AS len FROM jt GROUP BY 1), "
        f"jlen AS (SELECT s.memory_id, coalesce(c.len, 0.0) AS len FROM {source} s "
        f"         LEFT JOIN jcount c ON c.memory_id = s.memory_id), "
        f"jtf AS (SELECT jt.memory_id, jt.term, count(*)::DOUBLE AS tf "
        f"        FROM jt JOIN q ON q.term = jt.term GROUP BY 1, 2), "
        f"jdf AS (SELECT jt.term, count(DISTINCT jt.memory_id)::DOUBLE AS df "
        f"        FROM jt JOIN q ON q.term = jt.term GROUP BY 1), "
        f"jstats AS (SELECT count(*)::DOUBLE AS num_docs, "
        f"           coalesce(sum(len), 0.0)::DOUBLE AS total FROM jlen)"
    )


def _scan_sql(
    *, tenant_id: int, query_text: str, scope: AsOf, kinds: Sequence[str] | None, topn: int
) -> tuple[str, list]:
    """The fallback scan as one statement.  Parameters follow the CTE order: the query text,
    then the visibility predicate and kind filter, then the limit.

    Its corpus is the tenant's VISIBLE documents, which is the only corpus it has; a generation's
    is every document in the file, because a base that indexed only what is current could not
    answer an ``as_of`` read.  The two therefore agree on which rows match and can order two
    close scores differently.  See "Two corpora, one model" in the module docstring.
    """
    w, wp = Visibility.at(int(tenant_id), scope).predicate("m")
    kf, kp = _kinds("m", kinds)
    sql = (
        f"WITH {_query_cte()}, "
        f"jm AS (SELECT m.memory_id, m.content FROM memories m WHERE {w}{kf}), "
        f"{_rescan_ctes('jm')}, "
        f"s AS (SELECT num_docs, "
        f"      CASE WHEN num_docs > 0 THEN total / num_docs ELSE 1.0 END AS avgdl "
        f"      FROM jstats), "
        f"sc AS (SELECT jtf.memory_id, "
        f"       sum({_bm25_term('jtf.tf', 'jlen.len', 'jdf.df')}) AS score "
        f"       FROM jtf JOIN jdf ON jdf.term = jtf.term "
        f"       JOIN jlen ON jlen.memory_id = jtf.memory_id "
        f"       CROSS JOIN s GROUP BY 1) "
        f"SELECT sc.memory_id, sc.score FROM sc "
        f"ORDER BY sc.score DESC, sc.memory_id ASC LIMIT ?"
    )
    return sql, [str(query_text)] + wp + kp + [int(topn)]


def _merge_sql(
    plan: FtsPlan,
    *,
    tenant_id: int,
    query_text: str,
    scope: AsOf,
    kinds: Sequence[str] | None,
    topn: int,
) -> tuple[str, list]:
    """Base generation MINUS the journal, UNION the journal rescored, then visibility, then
    top-N: one statement.

    The corpus statistics are reconstructed rather than read from the generation's stats table,
    and that is what makes the answer identical to the one a rebuild would give:

    * ``num_docs`` and ``avgdl`` come from the base documents the journal has NOT touched
      (``dt``, whose lengths the build recorded) plus the documents it HAS (``jlen``, measured
      now).  Those two sets are disjoint and their union is exactly the document set a rebuild
      would index, so the numbers are the ones a rebuild would compute;
    * ``df`` likewise: the postings already restricted to ``dt`` give the untouched half for
      free (``tf`` holds one row per matching document), and the rescanned half is counted.

    The generation's dictionary is still joined, but only to prune the query's terms to ones
    this tenant's base has seen -- a term nobody in this tenant used costs no posting scan.
    Its ``df`` column is not read, and neither is the stats table, so an erasure that leaves
    those two counting a document it deleted cannot skew a score.

    Parameter order follows the CTE order exactly, which is why the SQL and the parameter list
    are built together here rather than assembled by the caller.
    """
    st = plan.storage
    assert st is not None
    w, wp = Visibility.at(int(tenant_id), scope).predicate("m")
    kf, kp = _kinds("m", kinds)
    journal, jp = _journal_sql(int(tenant_id), plan.generation)
    t = int(tenant_id)
    sql = (
        f"WITH {_query_cte()}, "
        f"touched AS ({journal}), "
        # the journal arm: every document the journal has mentioned, at the version the build
        # would have indexed.  NOT filtered by visibility here -- an invisible document still
        # counts towards the corpus statistics, exactly as it does in a rebuilt base, and the
        # final join is what decides whether it can be returned.
        f"jm AS (SELECT m.memory_id, m.content FROM memories m "
        f"       JOIN touched j ON j.doc_id = m.memory_id AND j.tenant_id = m.tenant_id "
        f"       {newest_version_sql('m')}), "
        f"{_rescan_ctes('jm')}, "
        # the base arm: this tenant's documents the journal has said nothing about
        f"dt AS (SELECT docid, memory_id, len FROM {st.docmap} WHERE {tenant_sql()} "
        f"       AND memory_id NOT IN (SELECT doc_id FROM touched)), "
        f"bstats AS (SELECT count(*)::DOUBLE AS num_docs, "
        f"           coalesce(sum(len), 0.0)::DOUBLE AS total FROM dt), "
        f"s AS (SELECT (b.num_docs + j.num_docs) AS num_docs, "
        f"      CASE WHEN b.num_docs + j.num_docs > 0 "
        f"           THEN (b.total + j.total) / (b.num_docs + j.num_docs) ELSE 1.0 END AS avgdl "
        f"      FROM bstats b CROSS JOIN jstats j), "
        f"qt AS (SELECT d.termid, d.term FROM {st.index_schema}.dict d "
        f"       JOIN q ON d.term = q.term "
        f"       JOIN {st.dictionary} td ON td.termid = d.termid AND {tenant_sql('td')}), "
        f"tf AS (SELECT t.docid, t.termid, count(*)::DOUBLE AS tf "
        f"       FROM {st.index_schema}.terms t JOIN qt ON t.termid = qt.termid "
        f"       JOIN dt ON dt.docid = t.docid GROUP BY 1, 2), "
        f"bdf AS (SELECT qt.term, count(*)::DOUBLE AS df "
        f"        FROM tf JOIN qt ON qt.termid = tf.termid GROUP BY 1), "
        f"df AS (SELECT q.term, "
        f"       coalesce(b.df, 0.0) + coalesce(j.df, 0.0) AS df FROM q "
        f"       LEFT JOIN bdf b ON b.term = q.term LEFT JOIN jdf j ON j.term = q.term), "
        f"bsc AS (SELECT dt.memory_id, "
        f"        sum({_bm25_term('tf.tf', 'dt.len', 'df.df')}) AS score "
        f"        FROM tf JOIN qt ON qt.termid = tf.termid JOIN df ON df.term = qt.term "
        f"        JOIN dt ON dt.docid = tf.docid CROSS JOIN s GROUP BY 1), "
        f"jsc AS (SELECT jtf.memory_id, "
        f"        sum({_bm25_term('jtf.tf', 'jlen.len', 'df.df')}) AS score "
        f"        FROM jtf JOIN df ON df.term = jtf.term "
        f"        JOIN jlen ON jlen.memory_id = jtf.memory_id "
        f"        CROSS JOIN s GROUP BY 1), "
        f"sc AS (SELECT memory_id, score FROM bsc UNION ALL SELECT memory_id, score FROM jsc) "
        f"SELECT m.memory_id, sc.score FROM sc "
        f"JOIN memories m ON m.memory_id = sc.memory_id "
        f"WHERE {w}{kf} "
        f"ORDER BY sc.score DESC, m.memory_id ASC LIMIT ?"
    )
    params = [str(query_text)] + jp + [t, t] + wp + kp + [int(topn)]
    return sql, params


def corpus_size(
    con,
    *,
    tenant_id: int,
    as_of: AsOf | _dt.datetime | None = CURRENT,
    kinds: Sequence[str] | None = None,
) -> int:
    """How many documents of one tenant :func:`scan` would tokenise for this request."""
    w, wp = Visibility.at(int(tenant_id), as_of).predicate("m")
    kf, kp = _kinds("m", kinds)
    row = con.execute(f"SELECT count(*) FROM memories m WHERE {w}{kf}", wp + kp).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def scan(
    con,
    *,
    tenant_id: int,
    query_text: str,
    topn: int = DEFAULT_TOPN,
    as_of: AsOf | _dt.datetime | None = CURRENT,
    kinds: Sequence[str] | None = None,
) -> list[tuple[int, float]]:
    """Exact BM25 over one tenant's visible documents, with no index at all.  The oracle.

    The same Okapi formula and the same tokenizer as a generation, with ``df``, ``num_docs`` and
    ``avgdl`` computed from the scanned set -- which is the whole of that tenant's corpus at that
    instant, so the statistics are exact rather than approximated.  Used as the fallback when no
    generation is usable, as the reference :meth:`FtsIndex._validate` checks a generation
    against, and by the tests as the answer the merged path has to reproduce.  Cost is linear in
    the tenant's corpus: see :data:`SCAN_CEILING`.
    """
    sql, params = _scan_sql(
        tenant_id=int(tenant_id),
        query_text=str(query_text),
        scope=AsOf.coerce(as_of),
        kinds=kinds,
        topn=int(topn),
    )
    return [(int(r[0]), float(r[1])) for r in con.execute(sql, params).fetchall()]


def search(
    con,
    *,
    tenant_id: int,
    query_text: str,
    topn: int = DEFAULT_TOPN,
    as_of: AsOf | _dt.datetime | None = CURRENT,
    kinds: Sequence[str] | None = None,
    index: "FtsIndex | None" = None,
    plan: FtsPlan | None = None,
    on_unusable: str = "scan",
    available: bool | None = None,
) -> FtsSearch:
    """The text arm: BM25 top-N for one tenant, from whichever path can answer it.

    With a framework generation this is base + journal merged into one ranking, filtered by the
    tenant and time predicate before the top-N cut.  With none it is :func:`scan`, which is exact
    and linear, unless the corpus is above :data:`SCAN_CEILING` (``on_unusable="scan"``, the
    default) -- ``"empty"`` never scans and ``"error"`` raises
    :class:`~anatid.errors.StaleIndexError`.  With no framework definition at all it is 0.1.1's
    legacy index, and ``available`` short-circuits the "is there an index" probe exactly as it
    did there.

    Pass ``plan`` to answer against a generation resolved earlier: that is how a read keeps one
    generation across a publication.  A failure inside the merged path is caught and answered by
    the scan; DuckDB does not abort a transaction on a failed statement, so a caller that wrapped
    this in ``db.transaction()`` keeps its transaction.
    """
    scope = AsOf.coerce(as_of)
    p = plan if plan is not None else resolve(con, tenant_id=tenant_id, as_of=scope, index=index)
    if p.usable:
        sql, params = _merge_sql(
            p, tenant_id=tenant_id, query_text=query_text, scope=scope, kinds=kinds, topn=topn
        )
        try:
            hits = [(int(r[0]), float(r[1])) for r in con.execute(sql, params).fetchall()]
        except duckdb.Error as exc:
            gen = p.generation.generation if p.generation else "?"
            detail = f"generation {gen} could not be searched: {exc}"
            log.warning("full-text index: %s; answering by scanning", detail)
            p = _scan_plan(HealthReason.LOAD_FAILURE, detail, p.generation)
        else:
            return FtsSearch(
                hits=hits,
                backend="generation",
                reason=p.reason,
                exact=True,
                detail=p.detail,
                generation=None if p.generation is None else p.generation.generation,
                journal_rows=p.journal_rows,
            )
    if p.reason is HealthReason.ABSENT and _definition(con) is None:
        return FtsSearch(
            hits=legacy_bm25_arm(
                con,
                tenant_id=tenant_id,
                query_text=query_text,
                topn=topn,
                as_of=scope,
                kinds=kinds,
                available=available,
            ),
            backend="legacy",
            reason=HealthReason.ABSENT,
            exact=False,
            detail=(
                "this database has no derived full-text index; 0.1.1's file-wide BM25 index "
                "answered, and rows written since the last rebuild_fts_index() are invisible to "
                "it. anatid.fts.attach(db) moves it onto the framework."
            ),
        )
    if on_unusable == "error":
        raise StaleIndexError(f"{p.detail}. {FTS_GENERATION_POLICY}")
    if on_unusable == "empty":
        return FtsSearch(
            hits=[],
            backend="none",
            reason=p.reason,
            exact=False,
            detail=f"{p.detail}; the text arm was skipped",
            generation=None if p.generation is None else p.generation.generation,
            journal_rows=p.journal_rows,
        )
    rows = corpus_size(con, tenant_id=tenant_id, as_of=scope, kinds=kinds)
    if rows > SCAN_CEILING and _schema.fts_objects_present(con):
        # A corpus too big to scan, no usable generation, and 0.1.1's file-wide index still in
        # the file: this is the shape of a database upgraded from 0.1.1 before anyone has run
        # maintain_indexes().  Answering from the old index is what 0.1.1 did, so the upgrade
        # loses nothing; it is not exact, and the detail and `exact=False` say which index
        # answered and what is missing from it.
        return FtsSearch(
            hits=legacy_bm25_arm(
                con,
                tenant_id=tenant_id,
                query_text=query_text,
                topn=topn,
                as_of=scope,
                kinds=kinds,
                available=True,
            ),
            backend="legacy",
            reason=p.reason,
            exact=False,
            detail=(
                f"{p.detail}, and the fallback scan would tokenise {rows} document(s) of tenant "
                f"{int(tenant_id)}, above SCAN_CEILING={SCAN_CEILING}; 0.1.1's file-wide BM25 "
                f"index answered instead, and rows written since its last rebuild are invisible "
                f"to it. Call maintain_indexes() to publish a generation."
            ),
            generation=None if p.generation is None else p.generation.generation,
            journal_rows=p.journal_rows,
            scanned_rows=rows,
        )
    if rows > SCAN_CEILING:
        return FtsSearch(
            hits=[],
            backend="none",
            reason=p.reason,
            exact=False,
            detail=(
                f"{p.detail}, and the fallback scan would tokenise {rows} document(s) of tenant "
                f"{int(tenant_id)}, above SCAN_CEILING={SCAN_CEILING}; the text arm returned "
                f"nothing. Call maintain_indexes() to publish a generation."
            ),
            generation=None if p.generation is None else p.generation.generation,
            journal_rows=p.journal_rows,
            scanned_rows=rows,
        )
    return FtsSearch(
        hits=scan(
            con,
            tenant_id=tenant_id,
            query_text=query_text,
            topn=topn,
            as_of=scope,
            kinds=kinds,
        ),
        backend="scan",
        reason=p.reason,
        exact=True,
        detail=f"{p.detail}; answered exactly by scanning {rows} document(s)",
        generation=None if p.generation is None else p.generation.generation,
        journal_rows=p.journal_rows,
        scanned_rows=rows,
    )


# --------------------------------------------------------------------------- status


def status(
    con,
    *,
    tenant_id: int | None = None,
    deep: bool = False,
    index: "FtsIndex | None" = None,
    plan: FtsPlan | None = None,
) -> FtsStatus:
    """How the text arm will answer, as :class:`~anatid.types.FtsStatus`: either half.

    On the framework, ``stale`` no longer means "rows are invisible" -- a write is searchable at
    once, so that is not a state this index can be in.  It means the answer would be INCOMPLETE,
    which needs both no usable generation and a corpus above :data:`SCAN_CEILING`.
    ``pending_rows`` is how many documents a search rescans from the canonical rows,
    ``indexed_rows`` the tenant's document count in the published generation, ``indexed_at`` when
    it was built and ``indexed_max_id`` its watermark.  ``available`` is True whenever the text
    arm can answer at all, which on the framework is as soon as it is attached: before the first
    generation the scan answers.

    Pass ``plan`` when the caller has already resolved one, so that a recall reporting its
    staleness and the search that follows agree about which generation is answering and do not
    resolve twice (:func:`anatid.recall.hybrid_recall` does).
    """
    if _definition(con) is None:
        return legacy_fts_status(con, deep=deep, tenant_id=tenant_id)
    p = plan if plan is not None else resolve(con, tenant_id=tenant_id, index=index)
    gen = p.generation
    where, params = (live_row_sql(), []) if tenant_id is None else Visibility(int(tenant_id)).live()
    rows, top = con.execute(
        f"SELECT count(*), max(memory_id) FROM memories WHERE {where}", params
    ).fetchone()
    rows = int(rows)
    newest = None
    if deep:
        newest = con.execute(f"SELECT max(tx_from) FROM memories WHERE {where}", params).fetchone()[
            0
        ]
    exact = p.usable or rows <= SCAN_CEILING
    return FtsStatus(
        available=True,
        stale=not exact,
        indexed_rows=None if gen is None else _indexed_rows(con, gen, tenant_id),
        current_rows=rows,
        pending_rows=_journal_count(con, tenant_id, gen),
        indexed_at=None if gen is None else gen.built_at,
        newest_row_at=newest,
        policy=FTS_GENERATION_POLICY,
        indexed_max_id=None if gen is None else gen.watermark_id,
        current_max_id=None if top is None else int(top),
    )


def _indexed_rows(con, generation: Generation, tenant_id: int | None) -> int | None:
    """Documents the published generation holds, for one tenant or in total."""
    if tenant_id is None:
        value = generation.stats.get("rows")
        return int(value) if isinstance(value, (int, float)) else None
    st = FtsStorage.of(generation)
    try:
        row = con.execute(
            f"SELECT max(num_docs) FROM {st.stats} WHERE {tenant_sql()}", [int(tenant_id)]
        ).fetchone()
    except duckdb.Error:  # the generation's storage is gone; resolve() reports it
        return None
    return None if not row or row[0] is None else int(row[0])


def staleness_message(status: FtsStatus) -> str:
    """The warning :func:`anatid.recall.hybrid_recall` logs and notes for a stale text index.

    Two halves, two messages: the framework says the answer is incomplete and why, the legacy
    index says how far behind it is.  Both end with the policy that produced the state.
    """
    if status.policy is FTS_GENERATION_POLICY:
        return (
            f"full-text search could not answer exactly: no generation is usable and this "
            f"tenant's corpus of {status.current_rows} document(s) is above "
            f"anatid.fts.SCAN_CEILING={SCAN_CEILING}, so the text arm returned nothing. Call "
            f"maintain_indexes(). {FTS_GENERATION_POLICY}"
        )
    if status.pending_rows > 0:
        what = (
            f"{status.pending_rows} of this tenant's memory row(s) written since the last "
            f"rebuild are invisible to full-text search"
        )
    elif status.pending_rows < 0:
        what = (
            f"this tenant's full-text statistics still count {-status.pending_rows} document(s) "
            f"that have since been removed from memories, so the corpus statistics behind the "
            f"scores are off"
        )
    else:
        what = (
            f"the row count is unchanged but memories have been written and removed since the "
            f"last rebuild (max memory_id {status.indexed_max_id} -> {status.current_max_id}), "
            f"so the newest rows are invisible to full-text search"
        )
    return f"BM25 index is stale: {what} (indexed_at={status.indexed_at}). {status.policy}"


# --------------------------------------------------------------------------- the index


class FtsIndex(DerivedIndex):
    """BM25 over ``memories``, as a versioned derived index.

    Attach one with :func:`attach`.  From that moment the definition is in the FILE, so every
    handle journals every ``remember`` / ``supersede`` / ``forget`` / ``prune`` on ``memories``
    inside the writing transaction, whether or not it holds this class.  The read side does not
    need this object at all (:func:`resolve` works from the catalog); building, validating,
    dropping and erasing do.

    ``per_tenant`` is False: one generation covers every tenant in the file, because
    ``PRAGMA create_fts_index`` has a fixed per-call cost and per-tenant scoping is a predicate
    rather than an index.  The JOURNAL is still per tenant, which is what keeps two tenants'
    document 42 apart.
    """

    name = FTS_INDEX_NAME
    kind = "bm25"
    source_table = "memories"
    source_id_column = "memory_id"
    source_ts_column = "tx_from"
    per_tenant = False
    delta_mode = "table"
    supports_delta = True

    #: See :data:`ANSWERS_HISTORICAL` and the module docstring.
    answers_historical = ANSWERS_HISTORICAL

    def __init__(
        self,
        db: Any,
        *,
        name: str | None = None,
        terms_index: bool = True,
        validate_sample: int = VALIDATE_SAMPLE,
    ) -> None:
        super().__init__(db, name=name)
        self.terms_index = bool(terms_index)
        self.validate_sample = int(validate_sample)

    def params(self) -> dict[str, Any]:
        """What an operator reading the file should know about how this index was built."""
        return {
            "tokenizer": FTS_TOKENIZER,
            "k1": BM25_K1,
            "b": BM25_B,
            "doc_key": "tenant_id:memory_id",
            "statistics": "per tenant",
            "terms_index": self.terms_index,
        }

    def definition(self) -> IndexDefinition:
        return replace(super().definition(), params=self.params())

    def _con(self):
        """The connection this handle's statements run on, for the module-level functions."""
        return getattr(self.db, "connection", self.db)

    # ------------------------------------------------------------------ build

    def storage(self, generation: Generation) -> FtsStorage:
        """The five objects :meth:`_build` materialises for ``generation``."""
        return FtsStorage.of(generation)

    def _build(self, generation: Generation) -> dict[str, Any]:
        """Materialise one generation: source table, BM25 index, document map, per-tenant
        dictionary and per-tenant statistics.  Cost O(corpus).

        The same five objects :func:`anatid.schema.fts_rebuild_statements` builds for the legacy
        index, under names carrying the generation number, so this one is built beside the live
        one and publication is a metadata switch rather than a rebuild window.
        """
        st = self.storage(generation)
        ensure_fts_extension(self._con())
        self._drop(generation)
        self.db.execute(
            f"CREATE TABLE {st.source} AS "
            f"SELECT {FTS_DOC_ID_SQL} AS fts_doc_id, tenant_id, memory_id, content "
            f"FROM memories {_NEWEST_VERSION} ORDER BY tenant_id, memory_id"
        )
        self.db.execute(
            f"PRAGMA create_fts_index('{st.source}', 'fts_doc_id', 'content', "
            f"{FTS_TOKENIZER}, overwrite=1)"
        )
        if self.terms_index:
            # The spike A/B'd this: BM25 p50 8.57 ms with, 12.51 ms without, at 100k memories.
            self.db.execute(
                f"CREATE INDEX {quote_ident(st.source + '_termid')} "
                f"ON {st.index_schema}.terms(termid)"
            )
        self.db.execute(
            f"CREATE TABLE {st.docmap} AS "
            f"SELECT d.docid, s.tenant_id, s.memory_id, d.len "
            f"FROM {st.index_schema}.docs d JOIN {st.source} s ON s.fts_doc_id = d.name "
            f"ORDER BY s.tenant_id, d.docid"
        )
        self.db.execute(
            f"CREATE TABLE {st.dictionary} AS "
            f"SELECT dt.tenant_id, t.termid, count(DISTINCT t.docid) AS df "
            f"FROM {st.index_schema}.terms t JOIN {st.docmap} dt ON dt.docid = t.docid "
            f"GROUP BY 1, 2 ORDER BY 1, 2"
        )
        self.db.execute(
            f"CREATE TABLE {st.stats} AS "
            f"SELECT tenant_id, count(*) AS num_docs, CAST(avg(len) AS DOUBLE) AS avgdl "
            f"FROM {st.docmap} GROUP BY 1"
        )
        rows, tenants = self.db.execute(
            f"SELECT count(*), count(DISTINCT tenant_id) FROM {st.docmap}"
        ).fetchone()
        return {
            "rows": int(rows),
            "tenants": int(tenants),
            "terms_index": self.terms_index,
            "storage": st.source,
        }

    def _drop(self, generation: Generation) -> None:
        """Remove a generation's storage.  Tolerates storage that is already gone."""
        st = self.storage(generation)
        self.db.execute(f"DROP SCHEMA IF EXISTS {st.index_schema} CASCADE")
        for table in st.tables:
            self.db.execute(f"DROP TABLE IF EXISTS {table}")

    # ------------------------------------------------------------------ erasure

    def _erase(self, generation: Generation, tenant_id: int, doc_ids: Sequence[int]) -> int | None:
        """Delete documents from one generation's storage, inside the purge's transaction.

        Full text is the accelerator that most needs this: its source table holds the memory's
        content VERBATIM and its postings hold the memory's tokens, so a ``forget(hard=True)``
        that only tombstoned would leave the erased text in the file and findable.  The
        statements mirror :data:`anatid.schema._FTS_PURGE_PLAN` against this generation's names,
        in the same order and for the same reasons: dictionary rows whose every posting belongs
        to the erased documents (a token used nowhere else is a verbatim fragment of the erased
        text, the whole of it for a one-word memory), then the postings, then the extension's
        document rows, then the document map, then the source rows.

        The generation's per-tenant ``num_docs`` and ``df`` tables are left counting the
        document that has gone, until the next generation is built.  That costs a score nothing:
        a search reconstructs both from the document map and the journal (see :func:`_merge_sql`)
        and never reads either table.  ``status().indexed_rows`` is one higher than the
        generation now holds, and says when it was built.
        """
        st = self.storage(generation)
        ids = [int(d) for d in doc_ids]
        if not ids:
            return 0
        available = set(_schema.table_names(self._con()))
        if st.bare("source") not in available or st.bare("docmap") not in available:
            return 0
        tenant = int(tenant_id)
        marks = ", ".join(str(i) for i in ids)
        deleted = 0
        if _objects_present(self._con(), st):
            deleted += _count(
                self.db.execute(
                    f"DELETE FROM {st.index_schema}.dict WHERE termid IN ("
                    f"SELECT t.termid FROM {st.index_schema}.terms t WHERE t.termid IN ("
                    f"SELECT p.termid FROM {st.index_schema}.terms p "
                    f"JOIN {st.docmap} d ON d.docid = p.docid "
                    f"WHERE {tenant_sql('d')} AND d.memory_id IN ({marks})) "
                    f"GROUP BY t.termid HAVING count(*) = count(*) FILTER (WHERE t.docid IN ("
                    f"SELECT docid FROM {st.docmap} "
                    f"WHERE {tenant_sql()} AND memory_id IN ({marks}))))",
                    [tenant, tenant],
                )
            )
            deleted += _count(
                self.db.execute(
                    f"DELETE FROM {st.index_schema}.terms WHERE docid IN "
                    f"(SELECT docid FROM {st.docmap} "
                    f"WHERE {tenant_sql()} AND memory_id IN ({marks}))",
                    [tenant],
                )
            )
            deleted += _count(
                self.db.execute(
                    f"DELETE FROM {st.index_schema}.docs WHERE name IN "
                    f"(SELECT fts_doc_id FROM {st.source} "
                    f"WHERE {tenant_sql()} AND memory_id IN ({marks}))",
                    [tenant],
                )
            )
        deleted += _count(
            self.db.execute(
                f"DELETE FROM {st.docmap} WHERE {tenant_sql()} AND memory_id IN ({marks})",
                [tenant],
            )
        )
        deleted += _count(
            self.db.execute(
                f"DELETE FROM {st.source} WHERE {tenant_sql()} AND memory_id IN ({marks})",
                [tenant],
            )
        )
        return deleted

    # ------------------------------------------------------------------ validation

    def _validate(self, generation: Generation) -> ValidationReport:
        """Compare a new generation with the oracle: it must return a SUPERSET of it.

        A sample of the generation's own documents each contributes one query term.  For those
        terms the oracle is computed with a single scan of ``memories`` -- the same tokenizer and
        the same one-document-per-logical-memory rule :meth:`_build` used -- and every
        ``(term, tenant, memory)`` it finds must be in the generation's postings, unless the
        journal has already touched that document, in which case a search reads it from the
        canonical row and never asks the base about it.

        A superset rather than an exact match, because that is the property the merge needs:
        extra postings are candidates the canonical join and the visibility predicate throw away,
        missing ones are lost rows.  The structural half is exact: the document map has to agree
        with the source table on how many documents were indexed.
        """
        con = self._con()
        st = self.storage(generation)
        if not _objects_present(con, st):
            return ValidationReport(
                ok=False,
                generation=generation,
                detail=f"generation {generation.generation} has no fts objects at {st.source}",
            )
        indexed, mapped = self.db.execute(
            f"SELECT (SELECT count(*) FROM {st.source}), (SELECT count(*) FROM {st.docmap})"
        ).fetchone()
        if int(indexed) != int(mapped):
            return ValidationReport(
                ok=False,
                generation=generation,
                checked=int(indexed),
                detail=(
                    f"{st.source} holds {int(indexed)} document(s) but {st.docmap} maps "
                    f"{int(mapped)}: the build did not finish"
                ),
            )
        terms = self._sample_terms(st)
        if not terms:
            return ValidationReport(
                ok=True,
                generation=generation,
                checked=0,
                detail=(
                    f"generation {generation.generation} indexes {int(indexed)} document(s); no "
                    f"term could be sampled (an empty or content-free corpus)"
                ),
            )
        oracle = self._oracle_postings(terms)
        base = self._base_postings(st, terms)
        sql, params = _journal_sql(None, generation)
        touched = {(int(r[0]), int(r[1])) for r in self.db.execute(sql, params).fetchall()}
        missing = sorted({p for p in oracle if (p[1], p[2]) not in touched} - base)
        ok = not missing
        return ValidationReport(
            ok=ok,
            generation=generation,
            checked=len(terms),
            mismatches=tuple(missing[:20]),
            detail=(
                f"generation {generation.generation} indexes {int(indexed)} document(s) and "
                f"returns a superset of the oracle for {len(terms)} sampled term(s)"
                if ok
                else (
                    f"{len(missing)} (term, tenant, memory) posting(s) the oracle returns are "
                    f"missing from generation {generation.generation}: {missing[:5]}"
                )
            ),
        )

    def _sample_terms(self, st: FtsStorage) -> list[str]:
        """One term from each of a sample of the generation's own documents.

        Sampled from the documents rather than from the dictionary so the sample follows the
        corpus: a term drawn this way is one a real query could plausibly use, and a document
        that failed to be indexed is more likely to be caught by a term of its own.
        """
        rows = self.db.execute(
            f"SELECT list_filter({tokens_sql('content')}, x -> x <> '') AS terms "
            f"FROM {st.source} USING SAMPLE {int(self.validate_sample) * 4} ROWS"
        ).fetchall()
        rng = random.Random(0)
        out: list[str] = []
        for (terms,) in rows:
            if terms:
                out.append(str(rng.choice(list(terms))))
        return sorted(set(out))[: int(self.validate_sample)]

    def _oracle_postings(self, terms: Sequence[str]) -> set[tuple[str, int, int]]:
        """``(term, tenant_id, memory_id)`` the SQL path finds, with no index at all."""
        marks = ", ".join("?" for _ in terms)
        sql = (
            f"WITH q(term) AS (SELECT unnest([{marks}])), "
            f"src AS (SELECT tenant_id, memory_id, content FROM memories {_NEWEST_VERSION}), "
            f"tok AS (SELECT tenant_id, memory_id, "
            f"        unnest({tokens_sql('content')}) AS term FROM src) "
            f"SELECT q.term, tok.tenant_id, tok.memory_id FROM tok JOIN q ON q.term = tok.term "
            f"GROUP BY 1, 2, 3"
        )
        return {
            (str(r[0]), int(r[1]), int(r[2]))
            for r in self.db.execute(sql, [str(t) for t in terms]).fetchall()
        }

    def _base_postings(self, st: FtsStorage, terms: Sequence[str]) -> set[tuple[str, int, int]]:
        """``(term, tenant_id, memory_id)`` this generation's postings return."""
        marks = ", ".join("?" for _ in terms)
        sql = (
            f"SELECT d.term, dm.tenant_id, dm.memory_id "
            f"FROM {st.index_schema}.dict d "
            f"JOIN {st.index_schema}.terms t ON t.termid = d.termid "
            f"JOIN {st.docmap} dm ON dm.docid = t.docid "
            f"WHERE d.term IN ({marks}) GROUP BY 1, 2, 3"
        )
        return {
            (str(r[0]), int(r[1]), int(r[2]))
            for r in self.db.execute(sql, [str(t) for t in terms]).fetchall()
        }

    # ------------------------------------------------------------------ health and reads

    def health(
        self,
        tenant: Any = None,
        *,
        as_of: AsOf | _dt.datetime | None = None,
        policy: MaintenancePolicy | None = None,
        now: _dt.datetime | None = None,
    ) -> HealthReport:
        """The generation's health.  Two differences from the base implementation.

        ``pending_rows`` and ``tombstone_rows`` are counted over EVERY tenant, whatever
        ``tenant`` is, because the generation is file-wide and a rebuild absorbs every tenant's
        journal rows: the maintenance policy has to see the work a rebuild would do.  Per-tenant
        journal sizes are on :class:`~anatid.types.FtsStatus` (:func:`status`), which is the
        report about one caller's reads.

        An ``as_of`` read does not make the generation unusable (:data:`ANSWERS_HISTORICAL`);
        the detail says so rather than the reason changing, so that :func:`resolve` and this
        agree about whether a historical read uses the index.
        """
        report = super().health(None, policy=policy, now=now)
        scope = AsOf.coerce(as_of)
        if scope.is_current:
            return report
        if not self.answers_historical:  # pragma: no cover - the constant is True
            return super().health(None, as_of=scope, policy=policy, now=now)
        if not report.usable:
            return report
        return replace(
            report,
            detail=(
                f"{report.detail} (an as_of read uses this generation: its base holds document "
                f"identity and content, and the time predicate is applied to the canonical rows)"
            ).strip(),
        )

    def base_damage(self, generation: Generation) -> str | None:
        """:func:`_base_damage` for this generation, so ``index_health()`` agrees with a read."""
        st = self.storage(generation)
        con = self._con()
        if not _objects_present(con, st):
            return f"the full-text objects at {st.source} are not in this database"
        try:
            return _base_damage(con, generation, st)
        except duckdb.Error as exc:
            return f"the base could not be counted: {exc}"

    def plan(self, tenant: Any = None, *, as_of: AsOf | _dt.datetime | None = None) -> FtsPlan:
        """:func:`resolve` for this index, on this handle's connection.

        ``tenant`` of ``None`` counts the journal over every tenant, which is what a file-wide
        report wants; a search always names its tenant.
        """
        return resolve(
            self._con(),
            tenant_id=None if tenant is None else _tenant_of(tenant),
            as_of=as_of or CURRENT,
            index=self,
        )

    def search(
        self,
        *,
        tenant_id: int,
        query_text: str,
        topn: int = DEFAULT_TOPN,
        as_of: AsOf | _dt.datetime | None = CURRENT,
        kinds: Sequence[str] | None = None,
        on_unusable: str = "scan",
    ) -> FtsSearch:
        """:func:`search`, with the generation PINNED for the duration of the read.

        The module function resolves a generation and reads it; this one holds a
        :meth:`~anatid.derived.DerivedIndex.pin` around both, so a rebuild running on another
        thread cannot retire and drop the generation between the two.  Use it when you hold the
        index object; ``recall()`` uses the module function, which survives the same race by
        catching the error and answering with the scan.
        """
        scope = AsOf.coerce(as_of)
        pin_scope = None if self.answers_historical else scope
        with self.pin(tenant_id, as_of=pin_scope):
            return search(
                self._con(),
                tenant_id=tenant_id,
                query_text=query_text,
                topn=topn,
                as_of=scope,
                kinds=kinds,
                index=self,
                on_unusable=on_unusable,
            )

    def status(self, *, tenant_id: int | None = None, deep: bool = False) -> FtsStatus:
        """:func:`status` for this index, on this handle's connection."""
        return status(self._con(), tenant_id=tenant_id, deep=deep, index=self)

    def rebuild(self, *, now: _dt.datetime | None = None, validate: bool = True) -> FtsStatus:
        """:func:`rebuild` for this index."""
        return rebuild(self, now=now, validate=validate)


def _objects_present(con, st: FtsStorage) -> bool:
    """True when ``PRAGMA create_fts_index`` has produced this generation's schema."""
    row = con.execute(_OBJECTS_PRESENT_SQL, [st.index_schema]).fetchone()
    return bool(row and int(row[0]))


def _count(result: Any) -> int:
    row = result.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


# --------------------------------------------------------------------------- lifecycle


def attach(db: Any, *, terms_index: bool = True, build: bool = False) -> FtsIndex:
    """Move ``db``'s full-text search onto the derived-index framework.

    Registers an :class:`FtsIndex` on the handle and writes its definition into the FILE, which
    is the half that matters: from that moment every handle on the file journals every write on
    ``memories`` inside the writing transaction, whether or not it holds this class.  Idempotent.

    Nothing is built here.  Until a generation is published a text search answers exactly by
    scanning (:func:`scan`) and says so; ``build=True`` builds and publishes the first generation
    at once, and :func:`rebuild` or :meth:`anatid.Anatid.maintain_indexes` does it later.

    A machine with no ``fts`` extension attaches anyway, with the failure recorded as
    ``load_error``: journalling and the exact scan need no extension at all, and reporting a
    load failure on every read is more useful than refusing to start.  A build does need it,
    and says so then.

    :meth:`anatid.Anatid.open` calls this itself for ``accelerators=True`` and ``fts=True``,
    which are the defaults, so an ordinary database is already on the framework and a write on
    it is searchable with no rebuild.  Call this directly on a handle opened with
    ``accelerators=False``, or on one whose file predates the derived-index catalog and has
    since been migrated.
    """
    existing = index_of(db)
    if existing is None:
        existing = FtsIndex(db, terms_index=terms_index)
        try:
            ensure_fts_extension(db.connection)
        except Exception as exc:  # noqa: BLE001 - reported, not raised: the scan still answers
            existing.load_error = f"the fts extension could not be loaded: {exc}"
            log.warning("full-text index: %s; searches will scan", existing.load_error)
        db.indexes.register(existing)
        _ATTACHED.insert(0, weakref.ref(existing))
        _bump_definitions()
    if build and existing.current_generation() is None:
        rebuild(existing)
    return existing


def detach(db: Any, *, retire: bool = True) -> None:
    """Take the full-text accelerator off ``db`` and stop the file journalling for it.

    ``retire=True`` (the default) also drops every generation's storage, which is what an index
    holding the indexed CONTENT needs: left behind, a generation is a verbatim copy of every
    document it ever saw, and once no handle holds the implementation nothing can delete one
    document from it, so a later ``forget(hard=True)`` could only report it as unclean.  It needs
    this handle to hold the implementation, so a handle that never attached detaches without it
    and leaves the storage to the handle that built it.  Searches fall back to the legacy index,
    or to the scan when there is none.
    """
    registry = getattr(db, "indexes", None)
    if registry is None:
        return
    registry.unregister(FTS_INDEX_NAME, retire=retire and index_of(db) is not None)
    _bump_definitions()


def index_of(db: Any) -> FtsIndex | None:
    """This handle's :class:`FtsIndex`, or ``None`` when it holds none."""
    registry = getattr(db, "indexes", None)
    if registry is None:
        return None
    found = registry.get(FTS_INDEX_NAME)
    return found if isinstance(found, FtsIndex) else None


def _index_on_connection(con) -> FtsIndex | None:
    """The attached index whose handle owns ``con``, or ``None``.

    Only the connection-only WRITE entry point needs this (:func:`rebuild_fts_index`, which is
    what ``Anatid.rebuild_fts_index`` calls): a build needs the object, and a read does not.
    """
    alive: list[Any] = []
    found: FtsIndex | None = None
    for ref in _ATTACHED:
        idx = ref()
        if idx is None:
            continue
        alive.append(ref)
        if found is None:
            with contextlib.suppress(Exception):
                if idx.db.connection is con:
                    found = idx
    _ATTACHED[:] = alive
    return found


def rebuild(
    target: Any,
    *,
    now: _dt.datetime | None = None,
    validate: bool = True,
    terms_index: bool = True,
) -> FtsStatus:
    """Build the next generation, validate it against the oracle, publish it.

    The framework replacement for 0.1.1's ``rebuild_fts_index()``: the new generation is built
    BESIDE the live one, so reads keep answering from the old one throughout, and publication is
    one metadata row.  A generation that fails validation is retired and the previous one stays
    published, with :class:`~anatid.errors.IndexValidationError` saying what did not match.
    ``validate=False`` publishes with ``force``, which the framework records as
    ``published_unvalidated`` and :class:`~anatid.derived.HealthReason` reports as
    ``UNVALIDATED`` -- usable, and never mistaken for a validated generation.

    ``target`` is an :class:`FtsIndex`, an ``Anatid`` handle, or a connection.  With no framework
    index this is 0.1.1's drop-and-recreate rebuild, unchanged.
    """
    idx = target if isinstance(target, FtsIndex) else index_of(target)
    con = getattr(target, "connection", target)
    if idx is None:
        idx = _index_on_connection(con)
    if idx is None:
        return legacy_rebuild(con, now=now, terms_index=terms_index)
    generation = idx.build_next(now=now)
    if validate:
        report = idx.validate(generation)
        if not report.ok:
            idx.retire(report.generation)
            raise IndexValidationError(
                f"full-text generation {generation.generation} failed validation: {report.detail}",
                report=report,
                index=idx.name,
                generation=generation.generation,
            )
        generation = report.generation
    published = idx.publish(generation, force=not validate)
    _record_meta_watermark(idx, published)
    return idx.status()


def _record_meta_watermark(index: "FtsIndex", generation: Generation) -> None:
    """Copy the published generation's watermark into ``anatid_meta``.

    Those three columns are what :meth:`anatid.Anatid.info` and the MCP ``health`` tool report as
    "when was the full-text index last built, over how many documents, up to which id".  They
    were 0.1.1's only watermark; a generation carries its own, and keeping the meta row in step
    means the two reports do not contradict each other.  Nothing READS them on the framework
    path: :func:`status` takes every number from the generation.
    """
    rows = generation.stats.get("rows")
    with contextlib.suppress(Exception):
        index.db.execute(
            "UPDATE anatid_meta SET fts_indexed_rows = ?, fts_indexed_max_id = ?, "
            "fts_indexed_at = ?",
            [
                int(rows) if isinstance(rows, (int, float)) else None,
                generation.watermark_id,
                generation.built_at,
            ],
        )


# --------------------------------------------------------------------------- compatibility
#
# The three names anatid.recall re-exports and anatid.Anatid calls.  Each one picks a half and
# keeps 0.1.1's signature, so nothing outside this module has to know which is in play.


def fts_index_present(con) -> bool:
    """True when a BM25 index has been BUILT on this database (either half).

    Not the same question as "can the text arm answer": with the framework attached a search is
    answered by scanning until the first generation is published, and
    :attr:`anatid.types.FtsStatus.available` is the field that says so.  This one is about built
    storage.

    A schema-v2 file's ``fts_main_memories`` does not count: it is keyed on ``memory_id`` and
    leaks across tenants, and the 2->3 migration drops it, so a migrated file reports "no index"
    until a rebuild.  Reporting no index is the fail-safe answer; serving BM25 from the old one
    would not be.
    """
    if _definition(con) is not None:
        gen, _building, _storage = _generation_row(con)
        return gen is not None
    return _schema.fts_objects_present(con)


def bm25_arm(
    con,
    *,
    tenant_id: int,
    query_text: str,
    topn: int = DEFAULT_TOPN,
    as_of: AsOf = CURRENT,
    kinds: Sequence[str] | None = None,
    available: bool | None = None,
) -> list[tuple[int, float]]:
    """Okapi BM25 top-N over **this tenant's** documents: the text arm of ``recall()``.

    A thin adapter over :func:`search` returning what :func:`anatid.recall.rrf_fuse` wants.
    Which path answered, and whether the answer is complete, is on the :class:`FtsSearch` that
    :func:`search` returns; this signature is 0.1.1's and is kept for callers that want the
    ranked list alone.
    """
    return search(
        con,
        tenant_id=tenant_id,
        query_text=query_text,
        topn=topn,
        as_of=as_of,
        kinds=kinds,
        available=available,
    ).hits


def fts_status(con, *, deep: bool = False, tenant_id: int | None = None) -> FtsStatus:
    """:func:`status` under 0.1.1's name, for :mod:`anatid.recall` and
    :meth:`anatid.Anatid.fts_status`."""
    return status(con, tenant_id=tenant_id, deep=deep)


def rebuild_fts_index(
    con, *, now: _dt.datetime | None = None, terms_index: bool = True
) -> FtsStatus:
    """:func:`rebuild` under 0.1.1's name, for :meth:`anatid.Anatid.rebuild_fts_index`.

    Given only a connection, the object that owns a build is found among the indexes
    :func:`attach` has created in this process (:func:`_index_on_connection`).  Passing the
    handle or the index itself to :func:`rebuild` is the direct route and is what
    ``Anatid.rebuild_fts_index`` should do once it is wired.
    """
    return rebuild(con, now=now, terms_index=terms_index)


def ensure_fts_extension(con) -> None:
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
    row = con.execute(
        "SELECT count(*) FROM duckdb_functions() WHERE function_name = 'create_fts_index'"
    ).fetchone()
    if row and int(row[0]):
        return
    con.execute("INSTALL fts")
    con.execute("LOAD fts")


# --------------------------------------------------------------------------- the legacy index
#
# 0.1.1's full-text index, moved here from anatid.recall unchanged.  One file-wide
# `PRAGMA create_fts_index` over anatid_fts_documents, rebuilt wholesale, with per-tenant
# document maps, dictionaries and statistics derived from the postings it produces.  A database
# that has never called attach() gets exactly this, and behaves exactly as it did in 0.1.1.

#: BM25 over the schema-v3, per-tenant legacy index.  ``{where}`` is the time predicate on ``m``
#: and ``{kinds}`` the optional kind filter; the rest is fixed.  The four tenant predicates are
#: rendered into the template by :func:`anatid.visibility.tenant_sql` when this module loads, and
#: the time predicate by :meth:`anatid.visibility.Visibility.temporal` in
#: :func:`legacy_bm25_arm`, so no predicate text is written here.  Parameters, in order:
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
#:   tenants -- the same reason :func:`anatid.recall.hydrate` takes a tenant.
#:
#: ``fts_main_*.dict`` is still read, but only to map a query term to its ``termid``; the ``df``
#: column of that table is never selected.
_BM25_SQL = f"""
        WITH q AS (
            SELECT DISTINCT term FROM (
                SELECT unnest({tokens_sql("?")}) AS term)
            WHERE term <> ''
        ), dt AS (
            SELECT docid, memory_id, len FROM {FTS_DOCS_TABLE} WHERE {tenant_sql()}
        ), qt AS (
            SELECT d.termid, td.df
            FROM {FTS_INDEX_SCHEMA}.dict d
            JOIN q ON d.term = q.term
            JOIN {FTS_DICT_TABLE} td ON td.termid = d.termid AND {tenant_sql("td")}
        ), st AS (
            SELECT num_docs, avgdl FROM {FTS_STATS_TABLE} WHERE {tenant_sql()}
        ), tf AS (
            SELECT t.docid, t.termid, count(*)::DOUBLE AS tf
            FROM {FTS_INDEX_SCHEMA}.terms t
            JOIN qt ON t.termid = qt.termid
            JOIN dt ON dt.docid = t.docid
            GROUP BY t.docid, t.termid
        ), sc AS (
            SELECT tf.docid,
                   sum({_bm25_term("tf.tf", "dt.len", "qt.df")}) AS score
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
        WHERE {tenant_sql("m")} AND {{where}}{{kinds}}
        ORDER BY sc.score DESC, m.memory_id ASC
        LIMIT ?
"""


def legacy_bm25_arm(
    con,
    *,
    tenant_id: int,
    query_text: str,
    topn: int = DEFAULT_TOPN,
    as_of: AsOf = CURRENT,
    kinds: Sequence[str] | None = None,
    available: bool | None = None,
) -> list[tuple[int, float]]:
    """0.1.1's BM25 arm: the single file-wide index, scored with per-tenant statistics.

    Query text is tokenised the way the index was built (lower-cased, ``(\\.|[^a-z])+`` treated
    as a separator -- :data:`anatid.schema.FTS_TOKENIZER`), so query terms and index terms agree.
    Only memories matching at least one term score.  Ties break on ``memory_id ASC``.

    Nothing outside ``tenant_id`` can influence the result: the candidate documents are pruned to
    the tenant before scoring and the corpus statistics are the tenant's own.  See
    :data:`_BM25_SQL` for which predicate does what.

    Returns ``[]`` when no fts index exists.  Rows written since the last index build cannot
    appear here at all, which is exactly what the framework half fixes.

    One artifact worth stating: the rebuild keeps a single document per ``(tenant_id,
    memory_id)``, but if ``memories`` holds two *current* rows under one id -- an integrity fault
    ``Anatid.doctor()`` reports -- the final join matches both and the id is returned twice, so
    RRF scores it twice and it costs two candidate slots.  Deduplicating here would put a hash
    aggregate over the whole matching set on every query to compensate for a broken file, so this
    is left to :func:`~anatid.Anatid.doctor` to find and to the write path to prevent.
    """
    if not (_schema.fts_objects_present(con) if available is None else available):
        return []
    vis = Visibility.at(tenant_id, as_of)
    w, wp = vis.temporal("m")
    kf, kp = _kinds("m", kinds)
    sql = _BM25_SQL.format(where=w, kinds=kf)
    t = vis.tenant_id
    params = [str(query_text), t, t, t, t] + wp + kp + [int(topn)]
    return [(int(r[0]), float(r[1])) for r in con.execute(sql, params).fetchall()]


def legacy_fts_status(con, *, deep: bool = False, tenant_id: int | None = None) -> FtsStatus:
    """0.1.1's report of how far the non-incremental BM25 index has fallen behind ``memories``.

    With ``tenant_id`` the report covers **only that tenant's** rows, and comes from the
    per-tenant tables the last rebuild wrote rather than from the file-wide watermark:
    ``indexed_rows`` is the tenant's ``num_docs`` in :data:`~anatid.schema.FTS_STATS_TABLE` --
    the corpus size its BM25 scores are computed with -- and ``indexed_max_id`` the largest
    ``memory_id`` among its :data:`~anatid.schema.FTS_DOCS_TABLE` rows.  ``num_docs`` rather than
    a count of those rows on purpose: a hard purge removes the document from the index but not
    from the statistics, so a purge with no rebuild after it reports ``pending_rows = -1`` here
    exactly as the file-wide report does, because the scores are now computed over a corpus that
    no longer exists.

    ``indexed_rows`` is the document count recorded by the last rebuild; ``current_rows`` is the
    number of live memory versions now, one per logical memory (a correction adds a version row
    but no document, so it is not counted).  ``pending_rows`` is the difference: rows the BM25
    arm cannot see.  A negative difference (rows deleted since the build) also counts as stale,
    because the index still holds documents that no longer exist -- they are filtered out by the
    tenant/validity join, but the corpus statistics behind the scores are off.

    A row *count* on its own is not a watermark: one insert plus one hard purge leaves the count
    where it was while the new document is invisible to BM25.  So the cheap path also compares
    ``max(memory_id)`` with ``anatid_meta.fts_indexed_max_id``, the largest id present at the
    last rebuild.  ids from :func:`anatid.ids.new_id` are time-ordered, so any insert raises it.
    The two checks are OR'ed; either one alone can miss, together they catch every write anatid's
    own verbs can perform.

    ``deep=True`` additionally reads ``max(tx_from)`` (a second column scan) to catch rows
    back-dated into the table without changing the count.

    What no watermark here can see: a raw SQL ``UPDATE memories SET content = ...``.  No anatid
    verb rewrites content in place (``supersede`` inserts a new row), so this is reachable only
    through :attr:`anatid.Anatid.connection`; if you do it, call ``rebuild_fts_index()``
    yourself.  Stated rather than hidden, like the rest of the fts contract.
    """
    # One statement rather than several: every extra round trip costs a duckdb-python parameter
    # marshalling pass, and this runs on every recall() that has a text query.  The counts were
    # metadata-only before schema v4; the tx_to predicate that counts live versions makes each a
    # one-column scan, the same order of cost as the max(memory_id) that was measured at
    # +0.04 ms on 100k rows and +0.22 ms on 1M, inside a status call that costs ~1.5 ms either
    # way because of the information_schema probe.  That is the price of not lying about
    # staleness.  Rows are counted per logical memory, i.e. live versions only (schema v4).
    live = live_row_sql()
    if tenant_id is None:
        sql = (
            "SELECT (SELECT count(*) FROM information_schema.tables "
            "        WHERE table_schema = ? AND table_name = 'docs'),"
            f"       (SELECT count(*) FROM memories WHERE {live}),"
            "       (SELECT max(memory_id) FROM memories),"
            "       (SELECT fts_indexed_rows FROM anatid_meta LIMIT 1),"
            "       (SELECT fts_indexed_max_id FROM anatid_meta LIMIT 1),"
            "       (SELECT fts_indexed_at FROM anatid_meta LIMIT 1)"
        )
        params: list = [FTS_INDEX_SCHEMA]
        deep_sql, deep_params = "SELECT max(tx_from) FROM memories", []
    else:
        t_sql, t_p = Visibility(int(tenant_id)).tenant()
        l_sql, l_p = Visibility(int(tenant_id)).live()
        sql = (
            "SELECT (SELECT count(*) FROM information_schema.tables "
            "        WHERE table_schema = ? AND table_name = 'docs'),"
            f"       (SELECT count(*) FROM memories WHERE {l_sql}),"
            f"       (SELECT max(memory_id) FROM memories WHERE {t_sql}),"
            f"       (SELECT max(num_docs) FROM {FTS_STATS_TABLE} WHERE {t_sql}),"
            f"       (SELECT max(memory_id) FROM {FTS_DOCS_TABLE} WHERE {t_sql}),"
            "       (SELECT fts_indexed_at FROM anatid_meta LIMIT 1)"
        )
        params = [FTS_INDEX_SCHEMA] + l_p + t_p * 3
        deep_sql = f"SELECT max(tx_from) FROM memories WHERE {t_sql}"
        deep_params = list(t_p)
    present, current, current_max, indexed_rows, indexed_max, indexed_at = con.execute(
        sql, params
    ).fetchone()
    present = bool(present)
    current = int(current)
    current_max = None if current_max is None else int(current_max)
    indexed_rows = None if indexed_rows is None else int(indexed_rows)
    indexed_max = None if indexed_max is None else int(indexed_max)
    newest = con.execute(deep_sql, deep_params).fetchone()[0] if deep else None
    if not present:
        return FtsStatus(
            available=False,
            stale=True,
            indexed_rows=None,
            current_rows=current,
            pending_rows=current,
            indexed_at=None,
            newest_row_at=newest,
            policy=FTS_STALENESS_POLICY,
            indexed_max_id=indexed_max,
            current_max_id=current_max,
        )
    pending = current - (indexed_rows if indexed_rows is not None else 0)
    moved = current_max != indexed_max
    stale = (
        pending != 0
        or moved
        or (indexed_at is not None and newest is not None and newest > indexed_at)
    )
    return FtsStatus(
        available=True,
        stale=bool(stale),
        indexed_rows=indexed_rows,
        current_rows=current,
        pending_rows=pending,
        indexed_at=indexed_at,
        newest_row_at=newest,
        policy=FTS_STALENESS_POLICY,
        indexed_max_id=indexed_max,
        current_max_id=current_max,
    )


def legacy_rebuild(con, *, now: _dt.datetime | None = None, terms_index: bool = True) -> FtsStatus:
    """0.1.1's rebuild: drop and recreate the file-wide index, record the watermark.

    Runs :func:`anatid.schema.fts_rebuild_statements` in order: refill
    :data:`~anatid.schema.FTS_SOURCE_TABLE` from ``memories`` keyed ``'<tenant>:<memory>'``,
    ``PRAGMA create_fts_index(..., overwrite=1)`` over it, then re-derive the per-tenant
    ``docid -> (tenant, memory, len)`` map, ``df`` dictionary and ``(num_docs, avgdl)`` stats
    from the postings that build produced.  Cost O(corpus).  It runs as **one transaction**,
    joining the caller's if one is open, so a rebuild that fails half-way leaves the previous
    index, its sidecar tables and the ``anatid_meta`` watermark exactly as they were.

    It is never called implicitly by a read: an implicit rebuild would turn an unlucky
    ``recall()`` into a multi-second stall.  The index covers every row in the file, all tenants;
    per-tenant filtering happens in the query.

    ``terms_index`` also rebuilds the ART index on the postings table, which the spike measured
    at BM25 p50 8.57 ms with versus 12.51 ms without, for a 0.7 s build at 100k memories.
    ``create_fts_index`` drops the whole fts schema, so it has to be recreated every time.
    """
    at = to_utc_naive(now) or utcnow()
    ensure_fts_extension(con)  # INSTALL/LOAD are not transactional: keep them outside
    # schema's own transaction wrapper: joins an open transaction (Anatid.rebuild_fts_index opens
    # one) instead of issuing a nested BEGIN, which in DuckDB would abort the caller's.
    with _schema._transaction(con):  # noqa: SLF001 -- shared with ensure_schema on purpose
        for stmt in _schema.fts_rebuild_statements(terms_index=terms_index, con=con):
            con.execute(stmt)
        # Live versions only: one document per logical memory, the same count fts_status reads.
        rows, top = con.execute(
            f"SELECT count(*), max(memory_id) FROM memories WHERE {live_row_sql()}"
        ).fetchone()
        con.execute(
            "UPDATE anatid_meta SET fts_indexed_rows = ?, fts_indexed_max_id = ?, "
            "fts_indexed_at = ?",
            [int(rows), None if top is None else int(top), at],
        )
    return legacy_fts_status(con)
