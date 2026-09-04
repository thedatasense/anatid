"""What the text arm can see, and what ``rebuild_fts_index()`` is actually for.

The package, the ``Anatid`` class, ``recall()`` and the MCP instruction text all used to say
that a write is invisible to BM25 until :meth:`anatid.Anatid.rebuild_fts_index` runs.  That was
true of 0.1.1's single file-wide index and it stopped being true of an ordinary anatid database
in 0.2, where :func:`anatid.fts.attach` is wired into ``Anatid.open(accelerators=True)`` (the
default) and every write is journalled inside the transaction that makes it.  This file pins the
behaviour the corrected prose describes, so the prose cannot drift back.

Every test here is written the way a reader of those docs would write it: **no test in this file
calls** ``rebuild_fts_index()`` **or** ``maintain_indexes()`` on the database under test, except
:func:`test_a_rebuild_changes_no_answer`, whose whole subject is that rebuilding is a latency
decision.  ``tests/test_fts_framework.py`` covers the mechanism; this file covers the promise.
"""

from __future__ import annotations

import pytest

from anatid import Anatid
from anatid.fts import SCAN_CEILING

from conftest import DIM


def _contents(hits):
    return [h.memory.content for h in hits]


def _ids(hits):
    return [h.memory.memory_id for h in hits]


# --------------------------------------------------------------------------- the promise


def test_a_write_is_searchable_by_the_very_next_recall_with_no_rebuild(db):
    """The claim the docs make first: nothing is built, and the row is found."""
    m = db.remember("the quokka prefers eucalyptus at dusk", entities=["Quokka"])

    hits = db.recall("quokka")

    assert _ids(hits) == [m.memory_id]
    assert "text" in hits.arms
    assert hits.bm25_stale is False
    assert hits.bm25_available is True


def test_the_write_is_visible_to_another_handle_on_the_same_file_with_no_rebuild(tmp_path):
    """The journal row is written in the file, not held on the handle that wrote it."""
    path = tmp_path / "shared.anatid"
    with Anatid.open(path, tenant=1, embedding_dim=DIM) as writer:
        written = writer.remember("the numbat eats termites by daylight")

    with Anatid.open(path, tenant=1, embedding_dim=DIM) as reader:
        hits = reader.recall("numbat")

    assert _ids(hits) == [written.memory_id]
    assert hits.bm25_stale is False


def test_pending_fts_rows_counts_documents_the_search_read_not_documents_it_missed(db):
    """``pending_fts_rows`` is a rescan count.  Every one of those documents was searched."""
    for i in range(5):
        db.remember(f"survey transect {i} recorded a bettong at dawn")

    hits = db.recall("bettong")

    assert db.fts_status().pending_rows == 5
    assert hits.pending_fts_rows == 5
    assert len(hits) == 5  # all five pending documents came back
    assert hits.bm25_stale is False


# --------------------------------------------------------------------------- corrections


def test_a_superseded_memory_leaves_the_text_results_with_no_rebuild(db):
    """A correction is journalled by the same transaction, so the next search has it."""
    old = db.remember("the quokka prefers eucalyptus at dusk")
    new = db.supersede(old.memory_id, "the quokka prefers wattle at dawn")

    assert _ids(db.recall("eucalyptus")) == []
    assert _ids(db.recall("wattle")) == [new.memory_id]
    assert _ids(db.recall("quokka")) == [new.memory_id]  # shared term, only the current one


def test_a_soft_forgotten_memory_leaves_the_text_results_with_no_rebuild(db):
    """``forget()`` closes the memory; the text arm stops returning it on the next read."""
    kept = db.remember("the numbat eats termites by daylight")
    dropped = db.remember("the potoroo forages under banksia")

    assert _ids(db.recall("potoroo")) == [dropped.memory_id]
    db.forget(dropped.memory_id)

    assert _ids(db.recall("potoroo")) == []
    assert _ids(db.recall("numbat")) == [kept.memory_id]  # the neighbour is untouched


def test_a_soft_forgotten_memory_is_still_reachable_through_as_of_with_no_rebuild(db):
    """Closing is a correction rather than a rewrite, and the text arm honours the time axes."""
    m = db.remember("the potoroo forages under banksia")
    before = db.recall("potoroo")[0].memory.tx_from
    db.forget(m.memory_id)

    assert _ids(db.recall("potoroo")) == []
    assert _ids(db.as_of(before).recall("potoroo")) == [m.memory_id]


def test_a_hard_forgotten_memory_is_gone_from_the_text_results_with_no_rebuild(db):
    """Erasure has to beat the index, not wait for one: the purge runs in the same transaction."""
    kept = db.remember("the numbat eats termites by daylight")
    erased = db.remember("the bilby digs burrows nightly")

    assert _ids(db.recall("bilby")) == [erased.memory_id]
    receipt = db.forget(erased.memory_id, hard=True)

    assert receipt.hard is True
    assert receipt.memories_deleted == 1
    assert _ids(db.recall("bilby")) == []
    assert db.get(erased.memory_id) is None
    assert _ids(db.recall("numbat")) == [kept.memory_id]
    # and not by an as-of route either: a purge is gone from every view.
    assert _contents(db.recall("bilby", as_of=erased.tx_from)) == []


# --------------------------------------------------------------------------- what a rebuild is for


def test_a_rebuild_changes_no_answer(db):
    """The one test here that rebuilds.  Rebuilding buys latency; the answer is already right."""
    for i in range(40):
        db.remember(f"transect {i} recorded a bettong and a potoroo near the banksia")
    query = "bettong banksia"

    before = _ids(db.recall(query, k=10))
    assert len(before) == 10
    assert db.fts_status().pending_rows == 40

    db.rebuild_fts_index()

    assert db.fts_status().pending_rows == 0
    assert _ids(db.recall(query, k=10)) == before


def test_a_write_after_a_rebuild_is_searchable_again_with_no_second_rebuild(db):
    """A rebuild does not close the window, because there is no window to close."""
    db.remember("the numbat eats termites by daylight")
    db.rebuild_fts_index()

    later = db.remember("the bilby digs burrows nightly")

    assert _ids(db.recall("bilby")) == [later.memory_id]
    assert db.fts_status().pending_rows == 1
    assert db.recall("bilby").bm25_stale is False


def test_bm25_stale_is_about_exactness_and_not_about_freshness(db):
    """On the derived index ``stale`` needs no usable generation AND a corpus over the ceiling.

    Nothing here is near :data:`anatid.fts.SCAN_CEILING`, so a database with no generation at all
    answers exactly by scanning and reports itself fresh.  The docs say the flag means the answer
    would not be exact; this is the half of that claim a small database can check.
    """
    db.remember("the quokka prefers eucalyptus at dusk")
    status = db.fts_status()

    assert status.indexed_rows is None  # no generation has ever been published
    assert status.current_rows < SCAN_CEILING
    assert status.stale is False
    assert db.recall("quokka").bm25_stale is False


# --------------------------------------------------------------------------- the other half


def test_the_legacy_index_still_cannot_see_a_write_until_a_rebuild(legacy_db):
    """The replacement prose has to stay accurate about 0.1.1, which is still shipped.

    ``Anatid.open(accelerators=False)`` attaches no derived index, so the file journals nothing
    and 0.1.1's file-wide BM25 index answers.  There a write really is invisible until a rebuild,
    and ``bm25_stale`` keeps its original meaning.
    """
    legacy_db.remember("the quokka prefers eucalyptus at dusk")

    missed = legacy_db.recall("quokka")
    assert _ids(missed) == []
    assert missed.bm25_stale is True
    assert missed.pending_fts_rows == 1

    legacy_db.rebuild_fts_index()
    found = legacy_db.recall("quokka")
    assert len(found) == 1
    assert found.bm25_stale is False

    legacy_db.remember("the bilby digs burrows nightly")
    assert _ids(legacy_db.recall("bilby")) == []  # invisible again, until the next rebuild
    assert legacy_db.recall("bilby").bm25_stale is True


# --------------------------------------------------------------------------- the instruction text


def test_the_mcp_instructions_do_not_tell_an_agent_that_a_write_is_invisible():
    """``anatid-mcp`` opens a default handle, so its instruction text describes this file.

    The MCP text is the one place where a wrong claim is executed rather than read: it is handed
    to a model as instructions, and an agent told its writes are invisible will call
    ``rebuild_fts_index`` after every ``remember``.
    """
    pytest.importorskip("mcp.server.mcpserver", reason="the MCP server needs 'mcp>=2.1'")
    from anatid.integrations.mcp.server import INSTRUCTIONS

    lowered = INSTRUCTIONS.lower()
    assert "invisible" not in lowered, INSTRUCTIONS
    assert "not incremental" not in lowered, INSTRUCTIONS
    assert "rebuild_fts_index" in lowered  # still named, for what it IS for
