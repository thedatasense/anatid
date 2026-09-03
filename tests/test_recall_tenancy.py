"""Defect 1, the retrieval half: BM25 must not cross a tenant, in rows *or* in scores.

The reported bug had two halves, and this file pins both.

**The rows.**  Schema v2 built the fts index over ``memories`` keyed on ``memory_id``, which is
unique only *within* a tenant.  Two tenants each holding ``memory_id`` 4242 gave the extension one
document key for two documents, so a BM25 hit on tenant 2's text came back joined to tenant 1's
row: ``recall("swordfish", tenant=1)`` returned "apples are crisp".  A tenant learned that another
tenant's corpus contained a term, and got a document that does not match its query.

**The scores.**  Even with the rows fixed, ``df``/``idf`` and ``(num_docs, avgdl)`` read from
``fts_main_*.dict`` / ``fts_main_*.stats`` are computed over every tenant's text.  That is a
weaker leak (a term's rarity in the whole file is observable through the score) and it skews the
ranking a tenant sees for reasons inside a corpus it cannot read.

Schema v3 fixes both below the query: the index is built over ``anatid_fts_documents`` keyed
``'<tenant_id>:<memory_id>'``, and ``anatid.recall.bm25_arm`` prunes candidates to the tenant
through ``anatid_fts_docmap`` before scoring while reading every corpus statistic from the
per-tenant ``anatid_fts_dict`` / ``anatid_fts_stats``.

Tests are named for the guarantee, not the bug, so a rewrite that reintroduces it fails loudly.
The first test is the reviewer's reproduction verbatim.
"""

from __future__ import annotations

import datetime as _dt
import math
import re

import duckdb
import pytest

from anatid import Anatid
from anatid import fts
from anatid import recall as _recall
from anatid import schema as S

from conftest import DIM, T0

MINUTE = _dt.timedelta(minutes=1)


@pytest.fixture
def db(legacy_db):
    """0.1.1's file-wide BM25 index, which is what this file is about.

    ``Anatid.open(accelerators=False)``.  The derived full-text index answers the same tenancy
    questions -- ``tests/test_fts_framework.py`` asks them of it, memory 4242 in two tenants and
    all -- but it answers the STALENESS questions here differently, because on the framework a
    write is searchable at once and "stale" stops meaning "rows the arm cannot see".  Pinning
    the 0.1.1 configuration is what keeps these assertions about the statement they are about.
    """
    return legacy_db


@pytest.fixture
def file_db(legacy_file_db):
    """:func:`db` on disk.  Same reason."""
    return legacy_file_db


#: Enough other-tenant documents that a leaked corpus statistic could not hide in rounding: with
#: 10k documents containing the query term, a global ``df`` drives ``idf`` to ~0 while the
#: querying tenant's own ``df`` stays at 1.
OTHER_TENANT_DOCS = 10_000


def bulk_insert(
    con, *, tenant_id: int, count: int, content: str, first_id: int = 100_000, at: _dt.datetime = T0
) -> None:
    """``count`` memories for ``tenant_id``, written straight to the table.

    Not ``remember()``: the point of these rows is corpus *mass*, and 10k verb calls would make
    the test a benchmark of the write path.  Column list is explicit so a schema change that
    drops a NOT NULL column fails here rather than silently writing NULLs.
    """
    con.execute(
        "INSERT INTO memories (memory_id, tenant_id, content, kind, created_at, valid_from, "
        "tx_from, confidence, access_count) "
        "SELECT ? + i, ?, ? || ' ' || i, 'fact', ?, ?, ?, 1.0, 0 FROM range(?) t(i)",
        [int(first_id), int(tenant_id), str(content), at, at, at, int(count)],
    )


# ===================================================================== the reviewer's repro


def test_a_bm25_hit_never_crosses_a_tenant_even_on_a_colliding_memory_id(file_db):
    """The reported reproduction, both directions, through the public API.

    ``memory_id`` 4242 exists in tenant 1 ("apples are crisp") and in tenant 2 ("swordfish are
    large").  Searching either tenant for the *other* tenant's word must return nothing at all --
    not the local row that happens to share the id, which is what v2 returned.
    """
    file_db.remember("apples are crisp", memory_id=4242, tenant=1, now=T0)
    file_db.remember("swordfish are large", memory_id=4242, tenant=2, now=T0)
    file_db.rebuild_fts_index(now=T0)
    con = file_db.connection

    # the index really does hold both documents, under distinct composite keys
    assert [
        r[0]
        for r in con.execute(f"SELECT fts_doc_id FROM {S.FTS_SOURCE_TABLE} ORDER BY 1").fetchall()
    ] == ["1:4242", "2:4242"]

    # neither tenant can see the other's term ...
    assert _recall.bm25_arm(con, tenant_id=1, query_text="swordfish") == []
    assert _recall.bm25_arm(con, tenant_id=2, query_text="apples") == []

    leaked = file_db.recall("swordfish", tenant=1, k=5)
    assert leaked.bm25_available is True  # the arm ran; it simply matched nothing
    assert "text" in leaked.arms
    assert list(leaked) == []
    assert list(file_db.recall("apples", tenant=2, k=5)) == []

    # ... and each still finds its own, so the fix is not "return nothing"
    assert [m for m, _s in _recall.bm25_arm(con, tenant_id=1, query_text="apples")] == [4242]
    assert [m for m, _s in _recall.bm25_arm(con, tenant_id=2, query_text="swordfish")] == [4242]
    mine = file_db.recall("apples", tenant=1, k=5)
    assert [(h.memory.tenant_id, h.memory.content) for h in mine] == [(1, "apples are crisp")]
    theirs = file_db.recall("swordfish", tenant=2, k=5)
    assert [(h.memory.tenant_id, h.memory.content) for h in theirs] == [(2, "swordfish are large")]


def test_a_colliding_id_cannot_smuggle_a_row_into_a_hit_it_did_not_match(file_db):
    """The leak's payload was *content*: the joined row is what the caller reads.

    Distinct text under a shared id in three tenants; every tenant must see only its own, and the
    hydrated content must be its own too (the join carries ``tenant_id``, so a hit and its row
    cannot come from different tenants).
    """
    for tenant, text in (
        (1, "alpha alpha alpha"),
        (2, "bravo bravo bravo"),
        (3, "charlie charlie charlie"),
    ):
        file_db.remember(text, memory_id=99, tenant=tenant, now=T0)
    file_db.rebuild_fts_index(now=T0)

    for tenant, term, text in (
        (1, "alpha", "alpha alpha alpha"),
        (2, "bravo", "bravo bravo bravo"),
        (3, "charlie", "charlie charlie charlie"),
    ):
        hits = file_db.recall(term, tenant=tenant, k=5)
        assert [(h.memory.tenant_id, h.memory.content) for h in hits] == [(tenant, text)]
        for other in (t for t in (1, 2, 3) if t != tenant):
            assert list(file_db.recall(term, tenant=other, k=5)) == [], (term, other)


# ===================================================================== the corpus statistics


def test_another_tenants_corpus_changes_neither_this_tenants_rows_nor_its_scores(file_db):
    """The score half of the defect: ``idf`` and ``avgdl`` must be the tenant's own.

    Tenant 1's BM25 result is captured, then tenant 2 gains 10,000 documents containing the query
    term and the index is rebuilt.  Tenant 1's ids *and* its scores must be unchanged, while the
    file-wide statistics the v2 query read move by four orders of magnitude -- which is the
    control: had the query kept reading them, this test would fail loudly rather than subtly.
    """
    file_db.remember("swordfish are large", memory_id=4242, tenant=1, now=T0)
    file_db.remember("swordfish swim fast and swordfish are fish", memory_id=7, tenant=1, now=T0)
    file_db.remember("apples are crisp", memory_id=8, tenant=1, now=T0)
    file_db.remember("swordfish are large", memory_id=4242, tenant=2, now=T0)
    file_db.rebuild_fts_index(now=T0)
    con = file_db.connection

    # A two-term query on purpose: "are" is in every document, so the ranking depends on df and
    # avgdl, not only on which documents match.
    before = _recall.bm25_arm(con, tenant_id=1, query_text="swordfish are")
    assert [m for m, _s in before] == [4242, 7, 8]
    termid = con.execute(
        f"SELECT termid FROM {S.FTS_INDEX_SCHEMA}.dict WHERE term = 'swordfish'"
    ).fetchone()[0]
    global_df_before = con.execute(
        f"SELECT df FROM {S.FTS_INDEX_SCHEMA}.dict WHERE termid = ?", [termid]
    ).fetchone()[0]

    bulk_insert(
        con,
        tenant_id=2,
        count=OTHER_TENANT_DOCS,
        content="swordfish are large and swordfish are many",
    )
    file_db.rebuild_fts_index(now=T0)

    after = _recall.bm25_arm(con, tenant_id=1, query_text="swordfish are")
    assert [m for m, _s in after] == [m for m, _s in before]
    # Bit-identical on this machine; the tolerance only allows for the order a parallel hash
    # aggregate may sum a group in, which is ~15 orders of magnitude below any leak.
    assert [s for _m, s in after] == pytest.approx([s for _m, s in before], rel=1e-12, abs=1e-15)

    # the control: the statistics v2 scored with really did move, and the per-tenant ones did not
    termid = con.execute(
        f"SELECT termid FROM {S.FTS_INDEX_SCHEMA}.dict WHERE term = 'swordfish'"
    ).fetchone()[0]
    global_df_after = con.execute(
        f"SELECT df FROM {S.FTS_INDEX_SCHEMA}.dict WHERE termid = ?", [termid]
    ).fetchone()[0]
    assert global_df_before == 3 and global_df_after == OTHER_TENANT_DOCS + 3
    df = dict(
        con.execute(
            f"SELECT tenant_id, df FROM {S.FTS_DICT_TABLE} WHERE termid = ?", [termid]
        ).fetchall()
    )
    assert df == {1: 2, 2: OTHER_TENANT_DOCS + 1}
    stats = {
        t: (n, avg)
        for t, n, avg in con.execute(
            f"SELECT tenant_id, num_docs, avgdl FROM {S.FTS_STATS_TABLE}"
        ).fetchall()
    }
    assert stats[1][0] == 3 and stats[2][0] == OTHER_TENANT_DOCS + 1
    assert stats[1][1] != stats[2][1]  # avgdl is per tenant too, not one file-wide average


def test_the_bm25_statement_reads_no_file_wide_corpus_statistic():
    """A source-level guard: the two columns that leak must not appear in the query at all.

    ``fts_main_*.dict`` is still read -- but only to map a query term to its ``termid``; the ``df``
    beside it, and ``fts_main_*.stats``, count every tenant's documents.  This is cheap to assert
    and catches a "simplification" that reintroduces the leak without changing any behaviour a
    single-tenant test would notice.
    """
    sql = _recall._BM25_SQL.format(where="1=1", kinds="")
    assert f"{S.FTS_INDEX_SCHEMA}.stats" not in sql
    # the fts dictionary is aliased `d` and joined for termid only: `d.df` must never be read
    # (`td.df`, the per-tenant dictionary, is the one that may be)
    assert re.search(r"(?<![\w.])d\.df\b", sql) is None
    assert "td.df" in sql
    assert "fts_main_memories" not in sql  # the v2 index, keyed on memory_id
    for table in (S.FTS_DOCS_TABLE, S.FTS_DICT_TABLE, S.FTS_STATS_TABLE):
        assert table in sql
    # the parameter contract: query, four tenant predicates, limit
    assert sql.count("?") == 6


# ===================================================================== filters and time travel


def test_kind_and_as_of_filters_still_bind_in_the_right_order(file_db):
    """The tenant predicates moved the parameter order; the filters after them must still line up.

    ``bm25_arm`` binds ``(query, tenant, tenant, tenant, tenant, *temporal, *kinds, topn)``.  A
    mis-ordered list does not raise -- DuckDB happily compares a kind against a timestamp
    parameter -- it silently returns the wrong rows, so this pins it with filters that would each
    fail differently.
    """
    file_db.remember("swordfish are large", kind="fact", memory_id=1, tenant=1, now=T0)
    file_db.remember("swordfish are tasty", kind="note", memory_id=2, tenant=1, now=T0)
    file_db.remember("swordfish are large", kind="fact", memory_id=1, tenant=2, now=T0)
    file_db.rebuild_fts_index(now=T0)
    con = file_db.connection

    assert [m for m, _s in _recall.bm25_arm(con, tenant_id=1, query_text="swordfish")] == [1, 2]
    assert [
        m for m, _s in _recall.bm25_arm(con, tenant_id=1, query_text="swordfish", kinds=["note"])
    ] == [2]
    assert [
        m for m, _s in _recall.bm25_arm(con, tenant_id=2, query_text="swordfish", kinds=["note"])
    ] == []
    assert [m for m, _s in _recall.bm25_arm(con, tenant_id=1, query_text="swordfish", topn=1)] == [
        1
    ]

    hits = file_db.recall("swordfish", tenant=1, kinds=["note"], k=5)
    assert [h.memory.content for h in hits] == ["swordfish are tasty"]


def test_as_of_bm25_stays_tenant_scoped_through_a_supersede(file_db):
    """Superseded rows stay in the index so ``as_of`` BM25 works; the tenant filter still holds."""
    old = file_db.remember("swordfish are large", tenant=1, now=T0)
    file_db.remember("swordfish are enormous", tenant=2, now=T0)
    new = file_db.supersede(old.memory_id, "swordfish are enormous", tenant=1, now=T0 + MINUTE)
    file_db.rebuild_fts_index(now=T0 + MINUTE)
    con = file_db.connection

    assert [m for m, _s in _recall.bm25_arm(con, tenant_id=1, query_text="swordfish")] == [
        new.memory_id
    ]
    past = file_db.as_of(T0 + _dt.timedelta(seconds=1))
    hits = past.recall("swordfish", tenant=1, k=5)
    assert [(h.memory.tenant_id, h.memory.content) for h in hits] == [(1, "swordfish are large")]


# ===================================================================== staleness, per tenant


def test_staleness_is_reported_per_tenant_not_per_file(file_db):
    """Tenant 1 must not be told its index is stale because tenant 2 wrote a row.

    A rebuild is file-wide, but visibility is not: since v3 nothing tenant 2 writes can reach
    tenant 1's BM25 rows or scores.  Reporting tenant 2's pending rows to tenant 1 would be a
    false alarm *and* a side channel -- it is a count of another tenant's writes.  The file-wide
    ``fts_status()`` still reports the whole file, because that is what an operator deciding when
    to rebuild needs.
    """
    file_db.remember("apples are crisp", tenant=1, now=T0)
    file_db.remember("swordfish are large", tenant=2, now=T0)
    file_db.rebuild_fts_index(now=T0)
    con = file_db.connection
    assert _recall.fts_status(con, tenant_id=1).stale is False
    assert _recall.fts_status(con, tenant_id=2).stale is False

    file_db.remember("swordfish are also fast", tenant=2, now=T0)

    quiet = _recall.fts_status(con, tenant_id=1)
    assert (quiet.stale, quiet.pending_rows, quiet.indexed_rows, quiet.current_rows) == (
        False,
        0,
        1,
        1,
    )
    busy = _recall.fts_status(con, tenant_id=2)
    assert (busy.stale, busy.pending_rows, busy.indexed_rows, busy.current_rows) == (True, 1, 1, 2)
    whole_file = _recall.fts_status(con)
    assert (whole_file.stale, whole_file.pending_rows) == (True, 1)
    assert file_db.fts_status().stale is True  # the operator-facing report is unchanged

    calm = file_db.recall("apples", tenant=1, k=5)
    assert calm.bm25_stale is False and calm.pending_fts_rows == 0
    assert [h.memory.content for h in calm] == ["apples are crisp"]
    alarmed = file_db.recall("swordfish", tenant=2, k=5)
    assert alarmed.bm25_stale is True and alarmed.pending_fts_rows == 1
    assert any("stale" in n for n in alarmed.notes)


def test_a_tenant_the_index_has_never_seen_gets_nothing_and_is_told_why(file_db):
    """A tenant created after the last rebuild has no corpus statistics of its own.

    It must get an empty text arm and a staleness report -- not a crash, and not somebody else's
    ``num_docs`` / ``avgdl`` standing in for the ones it does not have.
    """
    file_db.remember("apples are crisp", tenant=1, now=T0)
    file_db.rebuild_fts_index(now=T0)
    file_db.remember("swordfish are large", tenant=5, now=T0)

    con = file_db.connection
    assert (
        con.execute(f"SELECT count(*) FROM {S.FTS_STATS_TABLE} WHERE tenant_id = 5").fetchone()[0]
        == 0
    )
    assert _recall.bm25_arm(con, tenant_id=5, query_text="swordfish") == []
    newcomer = file_db.recall("swordfish", tenant=5, k=5)
    assert list(newcomer) == []
    assert newcomer.bm25_stale is True and newcomer.pending_fts_rows == 1
    assert _recall.fts_status(con, tenant_id=1).stale is False

    file_db.rebuild_fts_index(now=T0)
    assert [h.memory.content for h in file_db.recall("swordfish", tenant=5, k=5)] == [
        "swordfish are large"
    ]


def test_per_tenant_staleness_still_catches_an_insert_cancelled_out_by_a_purge(file_db):
    """The id watermark survives the move to per-tenant counting."""
    a = file_db.remember("alpha alpha", tenant=1, now=T0)
    file_db.remember("bravo bravo", tenant=1, now=T0)
    file_db.remember("noise from another tenant", tenant=2, now=T0)
    file_db.rebuild_fts_index(now=T0)
    con = file_db.connection
    assert _recall.fts_status(con, tenant_id=1).stale is False

    file_db.remember("charlie charlie", tenant=1, now=T0)
    file_db.forget(a.memory_id, tenant=1, hard=True, now=T0)

    status = _recall.fts_status(con, tenant_id=1)
    assert status.current_rows == 2  # count(*) is back where it started ...
    assert status.indexed_max_id != status.current_max_id
    assert status.stale is True
    hits = file_db.recall("charlie", tenant=1, k=5)
    assert hits.bm25_stale is True

    file_db.rebuild_fts_index(now=T0)
    assert _recall.fts_status(con, tenant_id=1).stale is False
    assert [h.memory.content for h in file_db.recall("charlie", tenant=1, k=5)] == [
        "charlie charlie"
    ]


def test_a_purge_alone_makes_the_purged_tenant_stale_and_nobody_else(file_db):
    """``indexed_rows`` per tenant is the ``num_docs`` its scores are computed with, not a count.

    A hard forget removes the document from the index (``fts_purge``) but not from
    ``anatid_fts_stats``/``anatid_fts_dict``, so until the next rebuild the tenant's scores are
    computed over a corpus that no longer exists.  The file-wide report has always called that
    stale (``pending_rows < 0``); the per-tenant one must say the same -- and only for the tenant
    that purged.  Counting the tenant's ``docmap`` rows instead would report "fresh" here, which
    is the one answer that is wrong for both tenants' reasons: the scores are off, and the
    file-wide operator report disagrees with what ``recall()`` tells the caller.
    """
    a = file_db.remember("alpha alpha", tenant=1, now=T0)
    file_db.remember("bravo bravo", tenant=1, now=T0)
    file_db.remember("charlie charlie", tenant=2, now=T0)
    file_db.rebuild_fts_index(now=T0)
    con = file_db.connection

    file_db.forget(a.memory_id, tenant=1, hard=True, now=T0)

    # the document itself is gone; only its count survives until the next rebuild
    assert (
        con.execute(f"SELECT count(*) FROM {S.FTS_DOCS_TABLE} WHERE tenant_id = 1").fetchone()[0]
        == 1
    )
    assert (
        con.execute(f"SELECT num_docs FROM {S.FTS_STATS_TABLE} WHERE tenant_id = 1").fetchone()[0]
        == 2
    )
    purged = _recall.fts_status(con, tenant_id=1)
    assert (purged.stale, purged.pending_rows, purged.indexed_rows, purged.current_rows) == (
        True,
        -1,
        2,
        1,
    )
    neighbour = _recall.fts_status(con, tenant_id=2)
    assert (neighbour.stale, neighbour.pending_rows, neighbour.indexed_rows) == (False, 0, 1)
    assert _recall.fts_status(con).pending_rows == -1  # the file-wide report agrees

    alarmed = file_db.recall("bravo", tenant=1, k=5)
    assert alarmed.bm25_stale is True and alarmed.pending_fts_rows == -1
    assert any("statistics" in n for n in alarmed.notes)
    assert [h.memory.content for h in alarmed] == ["bravo bravo"]  # still answered, and right
    calm = file_db.recall("charlie", tenant=2, k=5)
    assert calm.bm25_stale is False and calm.pending_fts_rows == 0

    file_db.rebuild_fts_index(now=T0)
    assert _recall.fts_status(con, tenant_id=1).stale is False
    assert _recall.fts_status(con).stale is False


def test_a_hard_forget_leaves_no_token_of_the_erased_text_in_the_fts_dictionary(file_db):
    """The extension's ``dict`` table spells out every token in the corpus.

    A token that occurred only in the erased document is a fragment of the erased text (for a
    one-word memory, the whole text), and the first 0.1.1 draft left it there until the next
    rebuild -- readable through raw SQL, and by the catalog-wide scan the review asked for.
    ``fts_purge`` now drops the dict rows whose every posting belonged to the erased document,
    and keeps the ones another document also uses: those are that document's words too.
    """
    con = file_db.connection
    lone = file_db.remember("zqxjkvbnmw", tenant=1, now=T0)  # a token nobody else has
    shared = file_db.remember("pangolin zqxjkvbnmw", tenant=1, now=T0)
    file_db.remember("pangolin", tenant=2, now=T0)
    file_db.rebuild_fts_index(now=T0)
    dict_terms = lambda: {  # noqa: E731 - a tiny local probe
        r[0] for r in con.execute(f"SELECT term FROM {S.FTS_INDEX_SCHEMA}.dict").fetchall()
    }
    assert {"zqxjkvbnmw", "pangolin"} <= dict_terms()

    file_db.forget(lone.memory_id, tenant=1, hard=True, now=T0)
    # still used by `shared`: the token stays and the other document still matches on it
    assert "zqxjkvbnmw" in dict_terms()
    assert [h.memory_id for h in file_db.recall("zqxjkvbnmw", tenant=1)] == [shared.memory_id]

    file_db.forget(shared.memory_id, tenant=1, hard=True, now=T0)
    # now nothing uses it: the token is gone; "pangolin" survives because tenant 2 uses it
    assert "zqxjkvbnmw" not in dict_terms()
    assert "pangolin" in dict_terms()
    assert file_db.recall("zqxjkvbnmw", tenant=1).arms == ("text",)
    assert list(file_db.recall("zqxjkvbnmw", tenant=1)) == []
    assert [h.memory.content for h in file_db.recall("pangolin", tenant=2)] == ["pangolin"]
    # every table in the fts schema, every column: no trace of the erased one-word text
    for table in ("dict", "terms", "docs", "stats"):
        cols = [
            r[1]
            for r in con.execute(f"PRAGMA table_info('{S.FTS_INDEX_SCHEMA}.{table}')").fetchall()
        ]
        for col in cols:
            hits = con.execute(
                f"SELECT count(*) FROM {S.FTS_INDEX_SCHEMA}.{table} "
                f"WHERE contains(coalesce(CAST({col} AS VARCHAR), ''), 'zqxjkvbnmw')"
            ).fetchone()[0]
            assert hits == 0, f"{table}.{col} still holds the erased token"
    # and the purge is idempotent on the dict as on everything else
    assert S.fts_purge(con, 1, shared.memory_id) == 0


def test_a_hard_forget_leaves_nothing_the_text_arm_can_return(file_db):
    """The index this module reads holds the memory's content verbatim, so erasure must reach it.

    Two guarantees meet here.  ``recall``'s: the BM25 join goes through ``memories``, so a purged
    row is out of the answer the moment it is purged, whatever the index still holds.  And
    ``forget(hard=True)``'s (defect 4, in ``verbs.py``): it calls ``anatid.schema.fts_purge``, so
    the verbatim copy in ``anatid_fts_documents.content`` goes at purge time rather than at the
    next rebuild.  A file where only the first held would still hand the erased text to anyone
    with a connection.
    """
    secret = file_db.remember("swordfish are extremely secret", tenant=1, now=T0)
    file_db.remember("swordfish are large", tenant=2, now=T0)
    file_db.rebuild_fts_index(now=T0)
    con = file_db.connection
    assert file_db.recall("secret", tenant=1, k=5)[0].memory_id == secret.memory_id

    file_db.forget(secret.memory_id, tenant=1, hard=True, now=T0)

    assert list(file_db.recall("secret", tenant=1, k=5)) == []
    assert list(file_db.recall("swordfish", tenant=1, k=5)) == []
    assert _recall.bm25_arm(con, tenant_id=1, query_text="secret swordfish") == []
    assert (
        con.execute(
            f"SELECT count(*) FROM {S.FTS_SOURCE_TABLE} WHERE content LIKE '%extremely secret%'"
        ).fetchone()[0]
        == 0
    )
    assert (
        con.execute(
            f"SELECT count(*) FROM {S.FTS_DOCS_TABLE} WHERE tenant_id = 1 AND memory_id = ?",
            [secret.memory_id],
        ).fetchone()[0]
        == 0
    )

    # the other tenant's document is untouched, and still found
    assert [h.memory.content for h in file_db.recall("swordfish", tenant=2, k=5)] == [
        "swordfish are large"
    ]


# ===================================================================== the v2 file it replaces

#: A schema-v2 file, written with the v2 DDL, so the migration test runs against a shape this
#: build did not produce.  Kept here (rather than imported) so this file stands alone.
V2_DDL = [
    """CREATE TABLE anatid_meta (
    schema_version   INTEGER   NOT NULL,
    created_at       TIMESTAMP NOT NULL,
    embedding_dim    INTEGER   NOT NULL,
    anatid_version   VARCHAR   NOT NULL,
    duckdb_version   VARCHAR   NOT NULL,
    system_columns   BOOLEAN   NOT NULL,
    fts_indexed_at   TIMESTAMP,
    fts_indexed_rows BIGINT,
    fts_indexed_max_id BIGINT,
    contract         VARCHAR   NOT NULL)""",
    """CREATE TABLE anatid_audit (
    audit_id   BIGINT NOT NULL, tenant_id INTEGER NOT NULL, memory_id BIGINT,
    related_memory_id BIGINT, action VARCHAR NOT NULL, reason VARCHAR, writer VARCHAR,
    happened_at TIMESTAMP NOT NULL)""",
    """CREATE TABLE entities (
    entity_id BIGINT NOT NULL, tenant_id INTEGER NOT NULL, kind VARCHAR, name VARCHAR,
    valid_from TIMESTAMP, valid_to TIMESTAMP, tx_from TIMESTAMP, tx_to TIMESTAMP,
    writer VARCHAR, episode_id BIGINT, confidence FLOAT)""",
    """CREATE TABLE memories (
    memory_id BIGINT NOT NULL, tenant_id INTEGER NOT NULL, content VARCHAR, kind VARCHAR,
    embedding FLOAT[8], created_at TIMESTAMP NOT NULL,
    valid_from TIMESTAMP, valid_to TIMESTAMP, tx_from TIMESTAMP, tx_to TIMESTAMP,
    writer VARCHAR, episode_id BIGINT, confidence FLOAT,
    access_count INTEGER DEFAULT 0, last_access_at TIMESTAMP)""",
    """CREATE TABLE episodes (
    episode_id BIGINT NOT NULL, tenant_id INTEGER NOT NULL, source VARCHAR, content VARCHAR,
    kind VARCHAR, created_at TIMESTAMP NOT NULL, valid_from TIMESTAMP, valid_to TIMESTAMP,
    tx_from TIMESTAMP, tx_to TIMESTAMP, writer VARCHAR)""",
    """CREATE TABLE edges_about (
    edge_id BIGINT NOT NULL, src BIGINT NOT NULL, dst BIGINT NOT NULL, tenant_id INTEGER NOT NULL,
    weight FLOAT, valid_from TIMESTAMP, valid_to TIMESTAMP, tx_from TIMESTAMP, tx_to TIMESTAMP,
    writer VARCHAR, episode_id BIGINT, confidence FLOAT)""",
    """CREATE TABLE edges_relates (
    edge_id BIGINT NOT NULL, src BIGINT NOT NULL, dst BIGINT NOT NULL, tenant_id INTEGER NOT NULL,
    rel_kind VARCHAR, valid_from TIMESTAMP, valid_to TIMESTAMP, tx_from TIMESTAMP, tx_to TIMESTAMP,
    writer VARCHAR, episode_id BIGINT, confidence FLOAT)""",
    """CREATE TABLE edges_supersedes (
    edge_id BIGINT NOT NULL, src BIGINT NOT NULL, dst BIGINT NOT NULL, tenant_id INTEGER NOT NULL,
    tx_from TIMESTAMP, writer VARCHAR)""",
]

#: The v2 fts index: one document key per ``memory_id``, which is the defect.
V2_FTS_SQL = (
    r"PRAGMA create_fts_index('memories', 'memory_id', 'content', stemmer='none', "
    r"stopwords='none', ignore='(\.|[^a-z])+', strip_accents=0, lower=1, overwrite=1)"
)


def write_v2_file(path) -> None:
    """A v2 database holding the reviewer's two colliding rows and a v2 fts index over them."""
    con = duckdb.connect(str(path))
    try:
        con.execute("INSTALL fts")
        con.execute("LOAD fts")
        for stmt in V2_DDL:
            con.execute(stmt)
        con.execute(
            "INSERT INTO anatid_meta VALUES (2, ?, 8, '0.1.0', ?, TRUE, ?, 2, 4242, 'v2')",
            [T0, duckdb.__version__, T0],
        )
        for tenant, text in ((1, "apples are crisp"), (2, "swordfish are large")):
            con.execute(
                "INSERT INTO memories (memory_id, tenant_id, content, kind, created_at, "
                "valid_from, tx_from, confidence, access_count) "
                "VALUES (4242, ?, ?, 'fact', ?, ?, ?, 1.0, 0)",
                [tenant, text, T0, T0, T0],
            )
        con.execute(V2_FTS_SQL)
        # The file really is broken before the migration.  Two documents, and the join the v2
        # BM25 arm did -- `memories.memory_id = fts_main_memories.docs.name` -- matches each of
        # them to BOTH tenants' rows: 2 documents x 2 rows = 4.  That cross product is the leak.
        assert con.execute("SELECT count(*) FROM fts_main_memories.docs").fetchone()[0] == 2
        assert (
            con.execute(
                "SELECT count(*) FROM fts_main_memories.docs d "
                "JOIN memories m ON m.memory_id = d.name"
            ).fetchone()[0]
            == 4
        )
    finally:
        con.close()


def test_a_migrated_v2_file_never_serves_bm25_from_the_memory_id_index(tmp_path):
    """Opening a leaking v2 file must fail safe, then rebuild into the scoped index.

    The 2->3 migration drops ``fts_main_memories`` outright rather than leaving a half-migrated
    file serving from an index keyed on a colliding id, so BM25 reports itself absent -- an empty
    text arm with a note saying so -- until ``rebuild_fts_index()`` runs.  Then the same query
    that leaked in v2 returns nothing, and each tenant finds its own row.
    """
    path = tmp_path / "v2.anatid"
    write_v2_file(path)

    with Anatid.open(path, tenant=1, embedding_dim=DIM, accelerators=False) as db:
        con = db.connection
        assert db.info().schema_version == S.SCHEMA_VERSION
        assert (
            con.execute(
                "SELECT count(*) FROM duckdb_schemas() WHERE schema_name = 'fts_main_memories'"
            ).fetchone()[0]
            == 0
        )
        assert _recall.fts_index_present(con) is False

        blind = db.recall("swordfish", tenant=1, k=5)
        assert blind.bm25_available is False
        assert "text" not in blind.arms and list(blind) == []
        assert any("no fts index" in n for n in blind.notes)

        db.rebuild_fts_index(now=T0)
        assert _recall.fts_index_present(con) is True
        assert list(db.recall("swordfish", tenant=1, k=5)) == []
        assert list(db.recall("apples", tenant=2, k=5)) == []
        assert [
            (h.memory.tenant_id, h.memory.content) for h in db.recall("apples", tenant=1, k=5)
        ] == [(1, "apples are crisp")]
        assert [
            (h.memory.tenant_id, h.memory.content) for h in db.recall("swordfish", tenant=2, k=5)
        ] == [(2, "swordfish are large")]


# ===================================================================== the rebuild is one unit


def test_rebuild_on_a_bare_connection_is_atomic(file_db, monkeypatch):
    """A rebuild that fails half-way leaves the previous index, its sidecars and the watermark.

    ``Anatid.rebuild_fts_index`` opens a transaction; ``anatid.recall.rebuild_fts_index`` on a
    raw connection did not, so a failure after ``create_fts_index`` had already dropped the old
    schema left a file with no index, a ``docmap`` from the old build and a watermark claiming
    freshness.  The index and the four tables that scope it are only meaningful together.
    """
    file_db.remember("apples are crisp", tenant=1, now=T0)
    file_db.rebuild_fts_index(now=T0)
    file_db.remember("swordfish are large", tenant=2, now=T0)
    con = file_db.connection
    docs_before = con.execute(f"SELECT * FROM {S.FTS_DOCS_TABLE} ORDER BY 1").fetchall()
    meta_before = con.execute(
        "SELECT fts_indexed_rows, fts_indexed_max_id, fts_indexed_at FROM anatid_meta"
    ).fetchone()
    assert meta_before == (1, docs_before[0][2], T0)

    real = S.fts_rebuild_statements
    monkeypatch.setattr(
        S,
        "fts_rebuild_statements",
        lambda **kw: real(**kw) + ["SELECT * FROM anatid_no_such_table"],
    )
    with pytest.raises(duckdb.Error):
        _recall.rebuild_fts_index(con, now=T0 + MINUTE)  # the bare-connection entry point
    monkeypatch.undo()

    assert _recall.fts_index_present(con) is True
    assert con.execute(f"SELECT * FROM {S.FTS_DOCS_TABLE} ORDER BY 1").fetchall() == docs_before
    assert (
        con.execute(
            "SELECT fts_indexed_rows, fts_indexed_max_id, fts_indexed_at FROM anatid_meta"
        ).fetchone()
        == meta_before
    )
    assert [h.memory.content for h in file_db.recall("apples", tenant=1, k=5)] == [
        "apples are crisp"
    ]
    assert _recall.fts_status(con, tenant_id=2).stale is True  # still pending, honestly

    # and the connection was left usable: no aborted transaction behind it
    _recall.rebuild_fts_index(con, now=T0 + MINUTE)
    assert [h.memory.content for h in file_db.recall("swordfish", tenant=2, k=5)] == [
        "swordfish are large"
    ]
    assert con.execute("SELECT fts_indexed_rows, fts_indexed_at FROM anatid_meta").fetchone() == (
        2,
        T0 + MINUTE,
    )


# ===================================================================== the spike oracle

#: The fts tokenizer, mirrored in Python: lower-case, then ``(\.|[^a-z])+`` is a separator.  It
#: is what ``PRAGMA create_fts_index(..., ignore='(\.|[^a-z])+', lower=1)`` compiles to
#: (``fts_main_*.tokenize``) and what ``_BM25_SQL``'s ``q`` CTE applies to the query text.  The
#: oracle checks its own mirror against the index's document lengths before trusting it.
_SEPARATOR = re.compile(r"(\.|[^a-z])+")
K1, B = _recall.BM25_K1, _recall.BM25_B


def tokenize(text: str) -> list[str]:
    return _SEPARATOR.sub(" ", text.lower()).split()


class PerTenantBm25:
    """Brute-force Okapi BM25 over one tenant's corpus alone, in numpy, sharing nothing.

    Built straight from ``memories`` rows.  Every row of the tenant -- live and superseded, as
    the index holds them -- counts towards ``df``, ``N`` and ``avgdl``; only current rows are
    candidates, as ``bm25_arm``'s temporal predicate demands.  ``tenant_scoped=False`` scores
    the same candidates with file-wide statistics, which is what schema v2 did and what this
    test must be able to tell apart from the fix.
    """

    def __init__(self, rows):
        import numpy as np

        self.np = np
        n = len(rows)
        self.ids = np.array([r[0] for r in rows], dtype=np.int64)
        self.tenant = np.array([r[1] for r in rows], dtype=np.int64)
        self.current = np.array([bool(r[3]) for r in rows])
        self.vocab: dict[str, int] = {}
        term_of_tok: list[int] = []
        doc_of_tok: list[int] = []
        doclen = np.zeros(n, dtype=np.int64)
        for i, r in enumerate(rows):
            toks = tokenize(r[2] or "")
            doclen[i] = len(toks)
            term_of_tok.extend(self.vocab.setdefault(t, len(self.vocab)) for t in toks)
            doc_of_tok.extend([i] * len(toks))
        key, tf = np.unique(
            np.asarray(term_of_tok, dtype=np.int64) * n + np.asarray(doc_of_tok, dtype=np.int64),
            return_counts=True,
        )
        self.term_sorted, self.doc_sorted = key // n, key % n
        self.tf = tf.astype(np.float64)
        self.doclen, self.n = doclen, n
        v = len(self.vocab)
        tkey, tdf = np.unique(
            self.tenant[self.doc_sorted] * v + self.term_sorted, return_counts=True
        )
        self.df_tenant = {(int(k // v), int(k % v)): int(c) for k, c in zip(tkey, tdf, strict=True)}
        self.df_global = np.bincount(self.term_sorted, minlength=v)
        self.n_tenant = {
            int(t): int(c) for t, c in zip(*np.unique(self.tenant, return_counts=True), strict=True)
        }
        self.avgdl_tenant = {t: float(doclen[self.tenant == t].mean()) for t in self.n_tenant}
        self.avgdl_global = float(doclen.mean())

    def rank(self, tenant_id: int, query: str, *, tenant_scoped: bool = True):
        """Every current document of the tenant matching a term, as ``[(memory_id, score)]``
        ordered ``score DESC, memory_id ASC``, plus ``{memory_id: score}`` for the same set."""
        np = self.np
        scores = np.zeros(self.n)
        in_tenant = self.tenant == tenant_id
        if tenant_scoped:
            n_docs = self.n_tenant.get(tenant_id, 0)
            avgdl = self.avgdl_tenant.get(tenant_id, 1.0)
        else:
            n_docs, avgdl = self.n, self.avgdl_global
        for term in dict.fromkeys(tokenize(query)):
            tid = self.vocab.get(term)
            if tid is None:
                continue
            lo = int(np.searchsorted(self.term_sorted, tid, "left"))
            hi = int(np.searchsorted(self.term_sorted, tid, "right"))
            docs, tf = self.doc_sorted[lo:hi], self.tf[lo:hi]
            if tenant_scoped:
                keep = in_tenant[docs]
                docs, tf = docs[keep], tf[keep]
                df = self.df_tenant.get((tenant_id, tid), 0)
                if df == 0:
                    continue
            else:
                df = int(self.df_global[tid])
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            scores[docs] += (
                idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * self.doclen[docs] / avgdl))
            )
        cand = np.flatnonzero(in_tenant & self.current & (scores > 0))
        cand = cand[np.lexsort((self.ids[cand], -scores[cand]))]
        ranked = [(int(self.ids[d]), float(scores[d])) for d in cand]
        return ranked, dict(ranked)


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def ranking_agrees(got, ranked, score_of, *, topn: int) -> str:
    """``"exact"`` when anatid's top-N is the oracle's, ``"ties"`` when the two differ only in
    the order of documents whose scores tie, else a message naming the first disagreement.

    Ties are real, not a loophole: two documents with the same ``tf`` and length score
    identically, and a three-term sum can land an ulp apart in DuckDB's hash aggregate and in
    numpy, so a strict id-list comparison would fail on floating-point noise rather than on a
    defect.  The fallback is still sound: every returned id must carry the oracle's score *for
    that id*, the score at every rank must be the oracle's score at that rank, and the ids must
    be distinct -- together that means no document scoring strictly above the N-th can be
    missing and no document scoring strictly below it can be present.
    """
    want = ranked[:topn]
    if [m for m, _s in got] == [m for m, _s in want] and all(
        _close(a, b) for (_m, a), (_n, b) in zip(got, want, strict=True)
    ):
        return "exact"
    if len(got) != len(want):
        return f"{len(got)} rows, reference has {len(want)}"
    if len({m for m, _s in got}) != len(got):
        return "an id is returned twice"
    for rank, ((gid, gscore), (_wid, wscore)) in enumerate(zip(got, want, strict=True), start=1):
        if gid not in score_of:
            return f"rank {rank}: id {gid} matches no query term in this tenant"
        if not _close(gscore, score_of[gid]):
            return f"rank {rank}: id {gid} scored {gscore!r}, reference {score_of[gid]!r}"
        if not _close(gscore, wscore):
            return f"rank {rank}: score {gscore!r}, reference has {wscore!r} at this rank"
    return "ties"


@pytest.mark.slow
@pytest.mark.oracle
def test_bm25_arm_matches_a_per_tenant_reference_on_the_spike_dataset(
    spike_db, spike_common, spike_queries
):
    """``bm25_arm`` == a per-tenant brute-force BM25 over 100k memories / 10 tenants, 500 queries.

    The composite-key index shares one postings table and one ``fts_main_*.dict`` across
    tenants; this proves that what comes out is what a private per-tenant index would return
    -- ids, order and scores -- on real data, for the 200 verify queries and the 300 R2 queries
    the spike benchmarked.  Zero mismatches; a tie-order difference is accepted only under
    :func:`ranking_agrees`'s conditions and is counted.  The control is the same reference with
    file-wide statistics, i.e. schema v2's scoring: it must disagree with anatid on the scores,
    or this test could not tell the fix from the bug.
    """
    # The 0.1.1 statement over the 0.1.1 tables, by name.  `spike_db` is a default handle, so
    # `rebuild_fts_index()` would build a generation and leave the legacy tables this test reads
    # empty; the derived index's answer to the same question is in tests/test_fts_framework.py.
    con = spike_db.connection
    fts.legacy_rebuild(con)
    rows = con.execute(
        "SELECT memory_id, tenant_id, content, valid_to IS NULL AND tx_to IS NULL "
        "FROM memories ORDER BY memory_id"
    ).fetchall()
    oracle = PerTenantBm25(rows)

    # the tokenizer mirror agrees with the index on every document's length, all 100k
    lens = con.execute(
        f"SELECT memory_id, len FROM {S.FTS_DOCS_TABLE} ORDER BY memory_id"
    ).fetchall()
    assert [m for m, _n in lens] == oracle.ids.tolist()
    assert [n for _m, n in lens] == oracle.doclen.tolist()
    # ... and on the per-tenant statistics the query reads
    stats = {
        t: (n, a)
        for t, n, a in con.execute(
            f"SELECT tenant_id, num_docs, avgdl FROM {S.FTS_STATS_TABLE}"
        ).fetchall()
    }
    assert {t: n for t, (n, _a) in stats.items()} == oracle.n_tenant
    for t, (_n, avgdl) in stats.items():
        assert avgdl == pytest.approx(oracle.avgdl_tenant[t], rel=1e-12)

    qids = list(spike_common.VERIFY_QUERY_IDS) + list(spike_common.R2_QUERY_IDS)
    assert len(qids) == 500
    topn = _recall.DEFAULT_CANDIDATES
    mismatches, ties, control_differs = [], [], 0
    for qid in qids:
        q = spike_queries[qid]
        tenant, text = int(q["tenant_id"]), str(q["query_text"])
        got = fts.legacy_bm25_arm(con, tenant_id=tenant, query_text=text, topn=topn)
        ranked, score_of = oracle.rank(tenant, text)
        verdict = ranking_agrees(got, ranked, score_of, topn=topn)
        if verdict == "ties":
            ties.append(qid)
        elif verdict != "exact":
            mismatches.append((qid, tenant, text, verdict))
        assert got, (qid, text)  # every benchmark query matches something
        # the control: file-wide statistics give different scores for the same documents
        v2_ranked, v2_score_of = oracle.rank(tenant, text, tenant_scoped=False)
        assert [m for m, _s in v2_ranked] and set(v2_score_of) == set(score_of)
        if any(not _close(s, v2_score_of[m]) for m, s in got):
            control_differs += 1

    assert mismatches == [], (
        f"{len(mismatches)}/{len(qids)} queries differ from the per-tenant reference "
        f"({len(ties)} tie-order only): {mismatches[:3]}"
    )
    assert control_differs == len(qids), (
        f"only {control_differs}/{len(qids)} queries score differently under file-wide "
        f"statistics; the control cannot tell the fix from the bug"
    )
