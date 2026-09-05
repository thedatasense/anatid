"""The ingestion pipeline: text in, one reviewed patch applied as one transaction.

What these tests hold the package to:

* a patch is applied whole or not at all, and the raw text is stored as an episode before any
  belief derived from it, so every created row carries the episode id;
* the pipeline says what it did: a duplicate the graph already holds is dropped with a note, an
  alias rewrites references and is reported, a correction that names its memory by text is
  resolved to an id or downgraded with a note;
* the review hook can edit the patch and can decline it, and declining writes nothing;
* every id in the JSON form is a decimal string, checked through a real ``node`` when one is on
  PATH, because that is the parser the contract exists for;
* the offline example runs to completion with no key in the environment.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from anatid import Anatid
from anatid.errors import NotFoundError, ValidationError
from anatid.ingest import (
    ALIAS_REL_KIND,
    EPISODE_KIND,
    PATCH_JSON_SCHEMA,
    AddFact,
    Alias,
    Correction,
    ExtractionError,
    KnownFact,
    MemoryPatch,
    OpenAICompatibleExtractor,
    Relation,
    ScriptedExtractor,
    Span,
    dedupe,
    existing_context,
    fold_text,
    ingest,
    parse_json_object,
    prepare,
    propose,
    resolve_corrections,
    resolve_entities,
)
from anatid.visibility import current_row_sql

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "examples" / "ingest_notes.py"

NODE = shutil.which("node")
requires_node = pytest.mark.skipif(
    NODE is None, reason="node is not on PATH; the JSON round trip needs a real JavaScript parser"
)

T1 = _dt.datetime(2026, 3, 2, 9, 0)
T2 = _dt.datetime(2026, 6, 15, 9, 0)
T3 = _dt.datetime(2026, 8, 20, 9, 0)
APRIL = _dt.datetime(2026, 4, 1)

NOTE1 = (
    "Ada leads the Kestrel team. Kestrel owns the ingest service, and Bo maintains it day to day."
)
NOTE2 = "Bo moved to the platform group. Cy took over the ingest service from Bo this week."
NOTE3 = (
    "The ingest service must stay on Python 3.10 until the Kestrel team finishes the migration. "
    "Ada leads Kestrel."
)


def patch1() -> MemoryPatch:
    return MemoryPatch(
        add_facts=(
            AddFact("Ada leads Kestrel", ("Ada", "Kestrel"), span=Span(0, 26)),
            AddFact("Kestrel owns the ingest service", ("Kestrel", "ingest service")),
            AddFact("Bo maintains the ingest service", ("Bo", "ingest service")),
        ),
        add_relations=(
            Relation("Ada", "Kestrel", "leads"),
            Relation("Kestrel", "ingest service", "owns"),
            Relation("Bo", "ingest service", "maintains"),
        ),
    )


def patch2() -> MemoryPatch:
    return MemoryPatch(
        add_facts=(AddFact("Bo works in the platform group", ("Bo", "platform group")),),
        corrections=(
            Correction(
                "Cy maintains the ingest service",
                old_text="Bo maintains the ingest service",
                entities=("Cy", "ingest service"),
            ),
        ),
        remove_relations=(Relation("Bo", "ingest service", "maintains"),),
        add_relations=(
            Relation("Cy", "ingest service", "maintains"),
            Relation("Bo", "platform group", "member_of"),
        ),
    )


def patch3() -> MemoryPatch:
    return MemoryPatch(
        add_facts=(
            AddFact(
                "The ingest service must stay on Python 3.10 until the Kestrel migration finishes",
                ("ingest service", "Kestrel"),
                kind="constraint",
            ),
            AddFact("Ada leads Kestrel", ("Ada", "the Kestrel team")),
        ),
        entity_aliases=(Alias("the Kestrel team", "Kestrel"),),
    )


def ingest_notes(db, *patches, notes=(NOTE1, NOTE2, NOTE3), times=(T1, T2, T3), review=None):
    extractor = ScriptedExtractor(patches)
    receipts = []
    for note, when in zip(notes[: len(patches)], times):
        receipts.append(
            ingest(db, note, extractor=extractor, writer="notes-bot", now=when, review=review)
        )
    return extractor, receipts


def current_edges(db) -> list[tuple[str, str, str | None]]:
    rows = db.execute(
        f"SELECT s.name, d.name, r.rel_kind FROM edges_relates r "
        f"JOIN entities s ON s.entity_id = r.src JOIN entities d ON d.entity_id = r.dst "
        f"WHERE {current_row_sql('r')} ORDER BY 1, 2, 3"
    ).fetchall()
    return [tuple(r) for r in rows]


def current_contents(db) -> list[str]:
    rows = db.execute(f"SELECT content FROM memories WHERE {current_row_sql()}").fetchall()
    return sorted(r[0] for r in rows)


def count(db, table: str) -> int:
    return db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


# --------------------------------------------------------------------------------------
# Evidence before belief, and one transaction.
# --------------------------------------------------------------------------------------


def test_every_created_row_points_at_the_episode_stored_first(db):
    _, (receipt,) = ingest_notes(db, patch1())

    episode = db.get_episode(receipt.episode_id)
    assert episode is not None
    assert episode.content == NOTE1 and episode.kind == EPISODE_KIND
    assert episode.writer == "notes-bot"
    assert len(receipt.memories_created) == 3
    for mid in receipt.memories_created:
        memory = db.get(mid)
        assert memory is not None and memory.episode_id == receipt.episode_id
        assert memory.valid_from == T1 and memory.writer == "notes-bot"
    edge_episodes = db.execute("SELECT DISTINCT episode_id FROM edges_relates").fetchall()
    assert edge_episodes == [(receipt.episode_id,)]
    about_episodes = db.execute("SELECT DISTINCT episode_id FROM edges_about").fetchall()
    assert about_episodes == [(receipt.episode_id,)]


def test_a_correction_carries_the_new_episode_and_provenance_walks_to_both_notes(db):
    _, (_, receipt) = ingest_notes(db, patch1(), patch2())

    ((old_id, new_id),) = receipt.corrections
    new = db.get(new_id)
    assert new is not None and new.episode_id == receipt.episode_id
    prov = db.provenance(new_id)
    assert [m.content for m in prov.chain] == [
        "Cy maintains the ingest service",
        "Bo maintains the ingest service",
    ]
    assert [e.content for e in prov.episodes] == [NOTE2, NOTE1]
    assert prov.source_text == NOTE1
    old = db.get(old_id)
    assert old is not None and not old.is_current and old.valid_to == T2


def test_a_failure_at_the_last_step_leaves_nothing_not_even_the_episode(db, monkeypatch):
    ingest_notes(db, patch1())
    edges_before = current_edges(db)
    contents_before = current_contents(db)
    episodes_before = count(db, "episodes")

    calls = {"n": 0}
    real_relate = Anatid.relate

    def relate_then_fail(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # the second edge of the second patch
            raise RuntimeError("the edge write failed")
        return real_relate(self, *args, **kwargs)

    monkeypatch.setattr(Anatid, "relate", relate_then_fail)
    with pytest.raises(RuntimeError, match="the edge write failed"):
        patch2().replace(source_text=NOTE2).apply(db, writer="notes-bot", now=T2)

    assert current_edges(db) == edges_before
    assert current_contents(db) == contents_before
    assert count(db, "episodes") == episodes_before
    assert db.get_entity("Cy") is None
    assert db.get_entity("platform group") is None
    old = [m for m in db.context("ingest service") if m.content.startswith("Bo")]
    assert len(old) == 1 and old[0].is_current

    # Nothing was consumed by the failure: the same patch still applies cleanly.
    monkeypatch.undo()
    receipt = patch2().replace(source_text=NOTE2).apply(db, writer="notes-bot", now=T2)
    assert ("Cy", "ingest service", "maintains") in current_edges(db)
    assert ("Bo", "ingest service", "maintains") not in current_edges(db)
    assert len(receipt.relations_closed) == 1 and len(receipt.relations_opened) == 2


def test_a_failure_in_the_middle_rolls_back_the_facts_already_written(db, monkeypatch):
    def boom(self, *args, **kwargs):
        raise RuntimeError("supersede failed")

    ingest_notes(db, patch1())
    before = count(db, "memories")
    monkeypatch.setattr(Anatid, "supersede", boom)
    with pytest.raises(RuntimeError, match="supersede failed"):
        patch2().replace(source_text=NOTE2).apply(db, writer="notes-bot", now=T2)
    assert count(db, "memories") == before
    assert "Bo works in the platform group" not in current_contents(db)


def test_apply_refuses_a_patch_with_no_text_to_store(db):
    with pytest.raises(ValidationError, match="source text"):
        MemoryPatch(add_facts=(AddFact("x", ("A",)),)).apply(db, writer="w")
    assert count(db, "episodes") == 0


def test_an_empty_patch_still_stores_the_episode(db):
    receipt = MemoryPatch(source_text="nothing durable here").apply(db, writer="w", now=T1)
    assert receipt.changes == 0
    assert db.get_episode(receipt.episode_id).content == "nothing durable here"
    assert receipt.describe() == f"episode {receipt.episode_id} stored."


# --------------------------------------------------------------------------------------
# The correction closes the edge it contradicts, and time travel still sees the old owner.
# --------------------------------------------------------------------------------------


def test_the_correction_moves_the_maintainer_and_april_still_says_bo(db):
    _, (r1, r2) = ingest_notes(db, patch1(), patch2())

    maintainers = [e for e in current_edges(db) if e[1] == "ingest service" and e[2] == "maintains"]
    assert maintainers == [("Cy", "ingest service", "maintains")]
    now = [m.content for m in db.context("ingest service")]
    assert "Cy maintains the ingest service" in now
    assert "Bo maintains the ingest service" not in now
    then = [m.content for m in db.as_of(APRIL).context("ingest service")]
    assert "Bo maintains the ingest service" in then
    assert "Cy maintains the ingest service" not in then
    assert r2.memories_closed == (r1.memories_created[2],)
    assert r2.relations_closed == (r1.relations_opened[2],)


def test_facts_stored_through_the_pipeline_are_reachable_two_hops_from_ada(db):
    """G3 met on the G1 shape: three facts named by entity alone need the edges the patch
    carries for recall_2hop("Ada") to return all three."""
    ingest_notes(db, patch1(), patch2(), patch3())
    reached = [m.content for m in db.recall_2hop("Ada", limit=20)]
    assert "Ada leads Kestrel" in reached
    assert "Kestrel owns the ingest service" in reached
    assert "Cy maintains the ingest service" in reached
    assert any(m.startswith("The ingest service must stay") for m in reached)
    assert "Bo maintains the ingest service" not in reached


def test_a_correction_by_id_supersedes_that_memory(db):
    _, (r1,) = ingest_notes(db, patch1())
    old_id = r1.memories_created[2]
    patch = MemoryPatch(
        source_text=NOTE2,
        corrections=(Correction("Cy maintains the ingest service", old_id=old_id),),
    )
    receipt = patch.apply(db, writer="w", now=T2)
    assert receipt.corrections[0][0] == old_id
    new = db.get(receipt.corrections[0][1])
    assert new is not None and new.content == "Cy maintains the ingest service"
    # entities=None inherits the old memory's entities.
    assert sorted(e.name for e in db.entities_of(new.memory_id)) == ["Bo", "ingest service"]


def test_apply_without_prepare_needs_exactly_one_match_for_a_text_correction(db):
    ingest_notes(db, patch1())
    missing = MemoryPatch(
        source_text="x", corrections=(Correction("new", old_text="never stored"),)
    )
    with pytest.raises(NotFoundError, match="no current memory reads"):
        missing.apply(db, writer="w")
    db.remember("Bo maintains the ingest service", entities=["Bo"], now=T1)
    ambiguous = MemoryPatch(
        source_text="x",
        corrections=(Correction("new", old_text="Bo maintains the ingest service."),),
    )
    with pytest.raises(ValidationError, match="2 current memories"):
        ambiguous.apply(db, writer="w")
    assert count(db, "episodes") == 1, "the failed applies stored nothing"


def test_resolve_corrections_gives_a_text_correction_its_id_and_downgrades_a_missing_one(db):
    _, (r1,) = ingest_notes(db, patch1())
    patch = MemoryPatch(
        source_text=NOTE2,
        corrections=(
            Correction(
                "Cy maintains the ingest service", old_text="Bo maintains the ingest service."
            ),
            Correction(
                "Di runs the build farm", old_text="Bo runs the build farm", entities=("Di",)
            ),
            Correction("Ada leads Kestrel", old_id=1234),
        ),
    )
    resolved = resolve_corrections(patch, db)
    assert [c.old_id for c in resolved.corrections] == [r1.memories_created[2]]
    assert [f.content for f in resolved.add_facts] == [
        "Di runs the build farm",
        "Ada leads Kestrel",
    ]
    assert resolved.add_facts[0].entities == ("Di",)
    assert any("no current memory reads that way" in n for n in resolved.notes)
    assert any("memory 1234: not found" in n for n in resolved.notes)


def test_resolve_corrections_prefers_the_match_about_the_corrections_entities(db):
    a = db.remember("The service is on call", entities=["Ada"], now=T1)
    b = db.remember("The service is on call", entities=["Bo"], now=T2)
    patch = MemoryPatch(
        source_text="x",
        corrections=(
            Correction("Bo is off call", old_text="The service is on call", entities=("Bo",)),
        ),
    )
    resolved = resolve_corrections(patch, db)
    assert resolved.corrections[0].old_id == b.memory_id
    assert a.memory_id != b.memory_id
    assert any("2 current memories match" in n for n in resolved.notes)


def test_a_correction_of_a_memory_that_is_no_longer_current_is_downgraded(db):
    old = db.remember("Bo maintains the ingest service", entities=["Bo"], now=T1)
    db.supersede(old.memory_id, "Cy maintains the ingest service", now=T2)
    patch = MemoryPatch(
        source_text="x",
        corrections=(Correction("Di maintains the ingest service", old_id=old.memory_id),),
    )
    resolved = resolve_corrections(patch, db)
    assert resolved.corrections == ()
    assert any("no longer current" in n for n in resolved.notes)


# --------------------------------------------------------------------------------------
# Dedupe and aliases.
# --------------------------------------------------------------------------------------


def test_dedupe_drops_a_fact_the_graph_holds_with_the_same_entities_and_says_so(db):
    _, (r1,) = ingest_notes(db, patch1())
    patch = MemoryPatch(
        source_text="again",
        add_facts=(
            AddFact("ada leads kestrel.", ("ada", "KESTREL")),  # same fact, folded
            AddFact("Ada leads Kestrel", ("Ada",)),  # same words, different entities
            AddFact("Bo works in the platform group", ("Bo", "platform group")),
            AddFact("Bo works in the platform group", ("Bo", "platform group")),
        ),
        add_relations=(
            Relation("Ada", "Kestrel", "leads"),  # held
            Relation("Kestrel", "Ada", "leads"),  # held, other direction
            Relation("Ada", "Kestrel", "advises"),  # a different kind is new
        ),
        remove_relations=(Relation("Ada", "ingest service", "owns"),),  # no such edge
    )
    deduped = dedupe(patch, db)
    assert [f.content for f in deduped.add_facts] == [
        "Ada leads Kestrel",
        "Bo works in the platform group",
    ]
    assert [str(r) for r in deduped.add_relations] == ["Ada -advises-> Kestrel"]
    assert deduped.remove_relations == ()
    notes = "\n".join(deduped.notes)
    assert f"memory {r1.memories_created[0]} is current with the same entities" in notes
    assert "proposed twice in this patch" in notes
    assert "a current edge already holds it" in notes
    assert "no current edge to close" in notes


def test_ingesting_the_same_note_twice_adds_no_memory_the_second_time(db):
    extractor = ScriptedExtractor([patch1(), patch1()])
    first = ingest(db, NOTE1, extractor=extractor, writer="w", now=T1)
    second = ingest(db, NOTE1, extractor=extractor, writer="w", now=T2)
    assert first.changes == 6
    assert second.changes == 0
    assert len(second.patch.notes) == 6
    assert count(db, "episodes") == 2, "the repeat is still recorded as evidence"
    assert len(current_contents(db)) == 3


def test_an_alias_rewrites_references_and_the_stored_spelling_wins(db):
    _, (_, _, r3) = ingest_notes(db, patch1(), patch2(), patch3())

    assert r3.aliases == (("the Kestrel team", "Kestrel"),)
    assert db.get_entity("the Kestrel team") is None, "the alias did not become an entity"
    constraint = db.get(r3.memories_created[0])
    assert sorted(e.name for e in db.entities_of(constraint.memory_id)) == [
        "Kestrel",
        "ingest service",
    ]
    notes = "\n".join(r3.patch.notes)
    assert "alias 'the Kestrel team' -> 'Kestrel': 1 reference rewritten" in notes
    assert "dedupe: dropped 'Ada leads Kestrel'" in notes
    assert len(r3.memories_created) == 1


def test_an_alias_of_an_existing_entity_links_the_two_so_recall_crosses(db):
    db.remember("The Kestrel team meets on Mondays", entities=["the Kestrel team"], now=T1)
    db.remember("Kestrel owns the ingest service", entities=["Kestrel", "ingest service"], now=T1)
    assert [m.content for m in db.recall_2hop("Kestrel")] == ["Kestrel owns the ingest service"]

    patch = MemoryPatch(
        source_text="The Kestrel team is Kestrel.",
        entity_aliases=(Alias("the Kestrel team", "Kestrel"),),
    )
    prepared = prepare(patch, db)
    assert any("is already an entity; apply links it" in n for n in prepared.notes)
    receipt = prepared.apply(db, writer="w", now=T2)
    assert len(receipt.relations_opened) == 1
    assert ("the Kestrel team", "Kestrel", ALIAS_REL_KIND) in current_edges(db)
    assert "The Kestrel team meets on Mondays" in [m.content for m in db.recall_2hop("Kestrel")]

    again = prepared.apply(db, writer="w", now=T3)
    assert again.relations_opened == (), "the alias edge is written once"


def test_resolve_entities_spells_names_the_way_the_graph_does(db):
    db.upsert_entity("Ingest Service")
    patch = MemoryPatch(
        source_text="x",
        add_facts=(AddFact("It is down", ("ingest   service",)),),
        add_relations=(Relation("bo", "INGEST SERVICE", "maintains"),),
    )
    resolved = resolve_entities(patch, db)
    assert resolved.add_facts[0].entities == ("Ingest Service",)
    assert str(resolved.add_relations[0]) == "bo -maintains-> Ingest Service"


# --------------------------------------------------------------------------------------
# The review hook.
# --------------------------------------------------------------------------------------


def test_the_review_hook_can_edit_the_patch(db):
    seen = []

    def drop_bo(patch):
        seen.append(patch)
        return patch.replace(add_facts=patch.add_facts[:2], add_relations=patch.add_relations[:2])

    _, (receipt,) = ingest_notes(db, patch1(), review=drop_bo)
    assert len(seen) == 1 and seen[0].source_text == NOTE1
    assert len(receipt.memories_created) == 2
    assert db.get_entity("Bo") is None
    assert ("Bo", "ingest service", "maintains") not in current_edges(db)


def test_the_review_hook_can_decline_and_nothing_is_written(db):
    _, (receipt,) = ingest_notes(db, patch1(), review=lambda patch: None)
    assert receipt is None
    assert count(db, "episodes") == 0
    assert count(db, "memories") == 0
    assert count(db, "entities") == 0


def test_the_review_hook_sees_the_prepared_patch_not_the_raw_proposal(db):
    ingest_notes(db, patch1())
    seen = []
    extractor = ScriptedExtractor([patch1()])
    ingest(
        db, NOTE1, extractor=extractor, writer="w", now=T2, review=lambda p: seen.append(p) or None
    )
    assert seen[0].add_facts == (), "dedupe ran before the review"
    assert len(seen[0].notes) == 6


def test_a_review_hook_that_returns_the_wrong_thing_is_an_error(db):
    with pytest.raises(TypeError, match="MemoryPatch or None"):
        ingest_notes(db, patch1(), review=lambda patch: "yes")
    assert count(db, "episodes") == 0


# --------------------------------------------------------------------------------------
# The context the extractor sees.
# --------------------------------------------------------------------------------------


def test_the_extractor_is_handed_current_facts_about_the_entities_the_note_names(db):
    extractor, _ = ingest_notes(db, patch1(), patch2())
    first, second = extractor.calls
    assert first.existing == ()
    contents = {m.content: m.about for m in second.existing}
    assert set(contents["Bo maintains the ingest service"]) == {"Bo", "ingest service"}
    assert "Kestrel owns the ingest service" in contents
    assert all(isinstance(m, KnownFact) for m in second.existing)


def test_existing_context_matches_whole_words_only(db):
    db.remember("Bo maintains the ingest service", entities=["Bo"], now=T1)
    db.remember("Bob runs the build farm", entities=["Bob"], now=T1)
    about_bob = existing_context(db, "Bob is out this week", limit=10)
    assert [m.content for m in about_bob] == ["Bob runs the build farm"]
    about_bo = existing_context(db, "Bo is out this week", limit=10)
    assert [m.content for m in about_bo] == ["Bo maintains the ingest service"]
    assert existing_context(db, "   ") == []


def test_scripted_extractor_fills_in_the_text_and_runs_out(db):
    extractor = ScriptedExtractor([MemoryPatch(add_facts=(AddFact("x", ("A",)),))])
    patch = extractor.extract("the note", existing=[])
    assert patch.source_text == "the note"
    assert extractor.remaining == 0
    with pytest.raises(LookupError, match="no patch left"):
        extractor.extract("another", existing=[])
    assert [c.text for c in extractor.calls] == ["the note", "another"]


# --------------------------------------------------------------------------------------
# JSON: ids are strings, parsing is tolerant, and notes say what was skipped.
# --------------------------------------------------------------------------------------


def test_to_json_writes_ids_as_decimal_strings_and_round_trips(db):
    old_id = 883768514279557120
    patch = MemoryPatch(
        source_text=NOTE2,
        add_facts=(
            AddFact("Bo works in the platform group", ("Bo", "platform group"), span=Span(0, 30)),
        ),
        corrections=(
            Correction("Cy maintains the ingest service", old_id=old_id, entities=("Cy",)),
        ),
        add_relations=(Relation("Cy", "ingest service", "maintains"),),
        remove_relations=(Relation("Bo", "ingest service", "maintains"),),
        entity_aliases=(Alias("the ingest svc", "ingest service"),),
        notes=("a note",),
    )
    text = patch.to_json()
    data = json.loads(text)
    assert data["corrections"][0]["old_memory_id"] == "883768514279557120"
    assert isinstance(data["corrections"][0]["old_memory_id"], str)
    assert data["add_facts"][0]["span"] == {"start": 0, "end": 30}
    back = MemoryPatch.from_json(text)
    assert back == patch


@requires_node
def test_a_patch_survives_a_real_javascript_parser(db):
    patch = MemoryPatch(
        source_text="x",
        corrections=(Correction("new", old_id=883768514279557120),),
    )
    done = subprocess.run(
        [
            str(NODE),
            "-e",
            "let r='';process.stdin.on('data',c=>r+=c);process.stdin.on('end',()=>process.stdout.write(JSON.stringify(JSON.parse(r))))",
        ],
        input=patch.to_json(indent=None),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    back = MemoryPatch.from_json(done.stdout)
    assert back.corrections[0].old_id == 883768514279557120
    assert '"883768514279557120"' in done.stdout


def test_receipt_to_dict_writes_every_id_as_a_string(db):
    _, (r1, r2) = ingest_notes(db, patch1(), patch2())
    data = r2.to_dict()
    assert data["episode_id"] == str(r2.episode_id)
    assert data["memories_closed"] == [str(r1.memories_created[2])]
    assert data["corrections"] == [
        {"old_id": str(r1.memories_created[2]), "new_id": str(r2.memories_created[1])}
    ]
    for key in ("memories_created", "relations_opened", "relations_closed"):
        assert all(isinstance(v, str) for v in data[key]) and data[key]
    json.dumps(data)


def test_from_dict_reads_the_model_shape_with_evidence_quotes():
    data = {
        "facts": [
            {
                "content": "Bo works in the platform group",
                "entities": "Bo",
                "evidence": "Bo moved to the platform group",
            },
            {"content": "", "entities": ["x"]},
            "not an object",
        ],
        "corrections": [
            {"old_memory_id": 883768514279557120, "new_content": "Cy maintains the ingest service"},
            {
                "old_id": "not-an-id",
                "old_content": "Bo maintains it",
                "new_content": "Cy maintains it",
            },
            {"new_content": "orphan"},
        ],
        "relations": [
            {
                "source": "Cy",
                "target": "ingest service",
                "rel_kind": "maintains",
                "evidence": "Cy took over the ingest service",
            }
        ],
        "remove_relations": [{"src": "Bo"}],
        "aliases": [
            {"alias": "the platform team", "entity": "platform group"},
            {"name": "Kestrel", "canonical": "kestrel"},
        ],
    }
    patch = MemoryPatch.from_dict(data, source_text=NOTE2)
    assert patch.source_text == NOTE2
    assert [f.content for f in patch.add_facts] == ["Bo works in the platform group"]
    assert patch.add_facts[0].entities == ("Bo",)
    assert patch.add_facts[0].span == Span(0, 30)
    assert [c.old_id for c in patch.corrections] == [883768514279557120, None]
    assert patch.corrections[1].old_text == "Bo maintains it"
    assert str(patch.add_relations[0]) == "Cy -maintains-> ingest service"
    assert patch.add_relations[0].span is not None
    assert patch.remove_relations == ()
    assert patch.entity_aliases == (Alias("the platform team", "platform group"),)
    notes = "\n".join(patch.notes)
    assert "skipped a fact without content" in notes
    assert "skipped a non-object entry in 'facts'" in notes
    assert "is not an id; matching by text" in notes
    assert "names no memory" in notes
    assert "missing an endpoint" in notes


def test_from_dict_clamps_a_confidence_outside_the_unit_interval_and_notes_it():
    patch = MemoryPatch.from_dict(
        {
            "add_facts": [
                {"content": "x", "entities": ["A"], "confidence": 1.7},
                {"content": "y", "confidence": "nan"},
            ]
        },
        source_text="t",
    )
    assert patch.add_facts[0].confidence == 1.0
    assert patch.add_facts[1].confidence == 1.0
    assert sum("confidence" in n for n in patch.notes) == 2


def test_from_json_rejects_what_is_not_a_patch():
    with pytest.raises(ValidationError, match="not valid JSON"):
        MemoryPatch.from_json("{")
    with pytest.raises(ValidationError, match="JSON object"):
        MemoryPatch.from_dict(["a list"])  # type: ignore[arg-type]


def test_operations_validate_their_arguments():
    with pytest.raises(ValidationError):
        AddFact("   ")
    with pytest.raises(ValidationError):
        Correction("new")
    with pytest.raises(ValidationError):
        Relation("", "b")
    with pytest.raises(ValidationError):
        Alias("a", " ")
    with pytest.raises(ValidationError):
        Span(3, 1)
    assert fold_text("  Ada   leads Kestrel. ") == "ada leads kestrel"
    assert fold_text(None) is None


def test_describe_renders_every_operation_with_its_evidence():
    patch = MemoryPatch(
        source_text=NOTE2,
        add_facts=(
            AddFact(
                "Bo works in the platform group",
                ("Bo", "platform group"),
                kind="fact",
                confidence=0.8,
                span=Span(0, 30),
            ),
        ),
        corrections=(
            Correction(
                "Cy maintains the ingest service",
                old_id=42,
                old_text="Bo maintains the ingest service",
                entities=("Cy", "ingest service"),
            ),
        ),
        remove_relations=(Relation("Bo", "ingest service", "maintains"),),
        add_relations=(Relation("Cy", "ingest service", "maintains"),),
        entity_aliases=(Alias("the ingest svc", "ingest service", span=Span(32, 34)),),
        notes=("dedupe: dropped something",),
    )
    text = patch.describe()
    assert (
        text.splitlines()[0]
        == "memory patch: 1 fact, 1 correction, 1 relation(s) added, 1 relation(s) removed, 1 alias"
    )
    assert (
        '+ fact        "Bo works in the platform group"  about: Bo, platform group  confidence: 0.8  [0:30] "Bo moved to the platform group"'
        in text
    )
    assert '~ correction  memory 42 "Bo maintains the ingest service"' in text
    assert '-> "Cy maintains the ingest service"  about: Cy, ingest service' in text
    assert "- relation    Bo -maintains-> ingest service" in text
    assert "+ relation    Cy -maintains-> ingest service" in text
    assert '= alias       "the ingest svc" -> "ingest service"  [32:34] "Cy"' in text
    assert "note          dedupe: dropped something" in text
    assert "—" not in text
    assert MemoryPatch().describe() == "memory patch: no operations"


# --------------------------------------------------------------------------------------
# The OpenAI-compatible extractor, with a fake client.
# --------------------------------------------------------------------------------------


class FakeClient:
    def __init__(self, reply: str) -> None:
        self.requests: list[dict] = []
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.requests.append(kwargs)
                message = SimpleNamespace(content=reply, reasoning="thinking...")
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        self.chat = SimpleNamespace(completions=Completions())


MODEL_REPLY = """Here is the patch you asked for:
```json
{
  "add_facts": [{"content": "Bo works in the platform group", "entities": ["Bo", "platform group"],
                 "evidence": "Bo moved to the platform group"}],
  "corrections": [{"old_memory_id": "883768514279557120", "new_content": "Cy maintains the ingest service",
                   "entities": ["Cy", "ingest service"], "evidence": "Cy took over the ingest service from Bo"}],
  "add_relations": [{"src": "Cy", "dst": "ingest service", "rel_kind": "maintains"}],
  "remove_relations": [{"src": "Bo", "dst": "ingest service", "rel_kind": "maintains"}],
  "entity_aliases": []
}
```
Let me know if you need anything else."""


def test_the_openai_compatible_extractor_prompts_with_string_ids_and_parses_a_fenced_reply(db):
    client = FakeClient(MODEL_REPLY)
    extractor = OpenAICompatibleExtractor(
        model="test-model",
        client=client,
        extra_body={"reasoning": {"enabled": True}},
        response_format="json_object",
    )
    existing = [
        KnownFact.of(
            db.remember(
                "Bo maintains the ingest service",
                entities=["Bo", "ingest service"],
                memory_id=883768514279557120,
            ),
            ("Bo", "ingest service"),
        )
    ]
    patch = extractor.extract(NOTE2, existing=existing)

    (request,) = client.requests
    assert request["model"] == "test-model"
    assert request["response_format"] == {"type": "json_object"}
    assert request["extra_body"] == {"reasoning": {"enabled": True}}
    assert request["temperature"] == 0.0
    user = request["messages"][1]["content"]
    assert (
        'memory_id "883768514279557120": Bo maintains the ingest service  (about: Bo, ingest service)'
        in user
    )
    assert "Existing entity names: Bo, ingest service" in user
    assert user.endswith(NOTE2)
    assert "additionalProperties" in request["messages"][0]["content"], (
        "the schema is in the prompt"
    )

    assert patch.source_text == NOTE2
    assert patch.corrections[0].old_id == 883768514279557120
    assert patch.add_facts[0].span == Span(0, 30)
    assert str(patch.remove_relations[0]) == "Bo -maintains-> ingest service"
    assert extractor.last_reply == MODEL_REPLY
    assert patch.notes == ()


def test_the_openai_compatible_extractor_can_ask_for_a_strict_schema_or_nothing():
    strict = OpenAICompatibleExtractor(
        model="m", client=FakeClient("{}"), response_format="json_schema"
    )
    assert strict.request("t", [])["response_format"]["json_schema"]["schema"] is PATCH_JSON_SCHEMA
    plain = OpenAICompatibleExtractor(
        model="m", client=FakeClient("{}"), response_format=None, temperature=None
    )
    request = plain.request("t", [])
    assert "response_format" not in request and "temperature" not in request
    assert plain.extract("t", existing=[]) == MemoryPatch(source_text="t")
    with pytest.raises(ValueError, match="response_format"):
        OpenAICompatibleExtractor(model="m", client=FakeClient("{}"), response_format="xml")


def test_the_extractor_reports_a_reply_it_cannot_read_with_the_raw_text():
    extractor = OpenAICompatibleExtractor(model="m", client=FakeClient("I cannot help with that."))
    with pytest.raises(ExtractionError, match="no JSON object") as info:
        extractor.extract("note", existing=[])
    assert info.value.raw == "I cannot help with that."
    assert parse_json_object('prose {"a": 1} more prose') == {"a": 1}
    with pytest.raises(ExtractionError, match="not an object"):
        parse_json_object("[1, 2]")
    with pytest.raises(ExtractionError, match="no content"):
        parse_json_object("   ")


def test_propose_runs_the_model_extractor_through_preparation_without_writing(db):
    _, (r1,) = ingest_notes(db, patch1())
    reply = MODEL_REPLY.replace("883768514279557120", str(r1.memories_created[2]))
    extractor = OpenAICompatibleExtractor(model="m", client=FakeClient(reply))
    before = count(db, "memories")
    patch = propose(db, NOTE2, extractor=extractor)
    assert count(db, "memories") == before
    assert patch.corrections[0].old_id == r1.memories_created[2]
    assert patch.remove_relations == (Relation("Bo", "ingest service", "maintains"),)
    receipt = patch.apply(db, writer="w", now=T2)
    assert ("Cy", "ingest service", "maintains") in current_edges(db)
    assert receipt.relations_closed == (r1.relations_opened[2],)


# --------------------------------------------------------------------------------------
# Embeddings, and the example.
# --------------------------------------------------------------------------------------


def test_an_embedder_gives_every_new_memory_a_vector(db):
    calls = []

    def embed(text: str):
        calls.append(text)
        return [1.0, 0, 0, 0, 0, 0, 0, 0]

    _, (r1, _) = ingest_notes(db, patch1(), patch2())
    receipt = patch3().replace(source_text=NOTE3).apply(db, writer="w", now=T3, embedder=embed)
    assert calls == [receipt.patch.add_facts[0].content, "Ada leads Kestrel"]
    hits = db.recall(embedding=[1.0, 0, 0, 0, 0, 0, 0, 0], k=5)
    assert receipt.memories_created[0] in hits.memory_ids
    assert db.get(r1.memories_created[0], with_embedding=True).embedding is None


def test_the_offline_example_runs_with_no_key(tmp_path):
    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper() not in ("OPEN_ROUTER_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")
    }
    env["PYTHONIOENCODING"] = "utf-8"
    done = subprocess.run(
        [sys.executable, str(EXAMPLE), "--db", str(tmp_path / "demo.anatid")],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        cwd=str(tmp_path),
    )
    assert done.returncode == 0, done.stderr
    out = done.stdout
    assert "~ correction  memory" in out
    assert "note          dedupe: dropped 'Ada leads Kestrel'" in out
    assert "now:     Cy maintains the ingest service" in out
    assert "before:  Bo maintains the ingest service  (valid until 2026-06-15)" in out
    april = out.split("4. as_of(2026-04-01)")[1].split("5. The same read today")[0]
    assert "Bo maintains the ingest service" in april
    assert "Cy maintains" not in april
    assert (tmp_path / "demo.anatid").exists()
