"""The ingestion pipeline: text in, one reviewed and applied patch out.

::

    from anatid.ingest import ingest, OpenAICompatibleExtractor

    extractor = OpenAICompatibleExtractor(model="gpt-4o-mini", api_key=key)
    receipt = ingest(db, note, extractor=extractor, writer="notes-bot", source="notes/03-02.md")

:func:`ingest` runs four steps, each a function you can call on its own:

1. :func:`existing_context` finds the current facts about the entities the text names, so
   the extractor can propose a correction instead of a duplicate.
2. The extractor proposes a :class:`~anatid.ingest.patch.MemoryPatch`.
3. :func:`prepare` resolves the proposal against the database: aliases are applied and names
   are spelled the way the graph already spells them (:func:`resolve_entities`), a correction
   that names its memory by text gets the memory's id (:func:`resolve_corrections`), and a fact
   or relation the graph already holds is dropped (:func:`dedupe`).  Everything a step changes
   is written into the patch's ``notes``.
4. The optional ``review`` callable sees the prepared patch and returns the patch to apply
   (edited or not) or ``None`` to decline.  Then :meth:`~anatid.ingest.patch.MemoryPatch.apply`
   commits it in one transaction.

:func:`propose` is steps 1 to 3 without the apply, for a caller that wants to show the patch,
wait for approval elsewhere, and apply it later (``MemoryPatch.from_json(...).apply(db, ...)``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from ..schema import entity_key
from ..types import Memory
from ..visibility import current_row_sql, tenant_sql
from .extract import Extractor, KnownFact
from .patch import (
    ALIAS_REL_KIND,
    AddFact,
    Correction,
    MemoryPatch,
    PatchReceipt,
    Relation,
    current_relation_ids,
    find_current_by_text,
    fold_text,
)

if TYPE_CHECKING:
    from ..database import Anatid

__all__ = [
    "EXISTING_LIMIT",
    "ENTITY_SCAN_LIMIT",
    "ReviewHook",
    "existing_context",
    "propose",
    "prepare",
    "resolve_entities",
    "resolve_corrections",
    "dedupe",
    "ingest",
]

#: How many existing facts :func:`existing_context` returns at most.
EXISTING_LIMIT = 40

#: How many known entities named in the text are expanded into facts.
ENTITY_SCAN_LIMIT = 25

#: The review hook: receives the prepared patch, returns the patch to apply or ``None``.
ReviewHook = Callable[[MemoryPatch], "MemoryPatch | None"]


# --------------------------------------------------------------------------- 1. context


def existing_context(
    db: Anatid,
    text: str,
    *,
    tenant: Any = None,
    limit: int = EXISTING_LIMIT,
    entity_limit: int = ENTITY_SCAN_LIMIT,
) -> list[KnownFact]:
    """Current facts about the entities the text names, plus the text-search hits.

    An entity counts as named when its canonical key occurs in the text as a whole word, so
    "Bo" does not match "Bob".  Each fact is a :class:`~anatid.ingest.extract.KnownFact`
    carrying the names it is about.  Newest first within an entity; at most ``limit`` in all.
    """
    ns = db.resolve_tenant(tenant)
    folded = entity_key(text) or ""
    if not folded:
        return []
    rows = db.execute(
        f"SELECT name, entity_key FROM entities WHERE {tenant_sql()} AND {current_row_sql()} "
        f"AND length(entity_key) >= 2 AND contains(?, entity_key) "
        f"ORDER BY length(entity_key) DESC, entity_id",
        [ns.tenant_id, folded],
    ).fetchall()
    named: list[str] = []
    for name, key in rows:
        if re.search(rf"(?<!\w){re.escape(key)}(?!\w)", folded):
            named.append(name)
        if len(named) >= entity_limit:
            break

    out: list[KnownFact] = []
    seen: set[int] = set()

    def take(memory: Memory) -> bool:
        if memory.memory_id in seen or not memory.is_current:
            return False
        seen.add(memory.memory_id)
        about = tuple(e.name for e in db.entities_of(memory.memory_id, tenant=ns))
        out.append(KnownFact.of(memory, about))
        return len(out) >= limit

    for name in named:
        for memory in db.context(name, tenant=ns, limit=limit):
            if take(memory):
                return out
    for hit in db.recall(text, tenant=ns, k=min(limit, 20), on_stale_fts="ignore"):
        if take(hit.memory):
            break
    return out


# --------------------------------------------------------------------------- 3. prepare


def resolve_entities(patch: MemoryPatch, db: Anatid, *, tenant: Any = None) -> MemoryPatch:
    """Apply the patch's aliases and spell every name the way the graph already does.

    A name that resolves to an existing entity takes that entity's stored spelling, so the
    patch reads the way the graph will.  The database folds case and whitespace on its own
    (:func:`anatid.schema.entity_key`), so this changes what a reviewer sees rather than which
    entity is written.  Each alias that rewrote a reference is noted.
    """
    ns = db.resolve_tenant(tenant)
    alias_map = patch.alias_map()
    rewrites: dict[str, int] = {}
    spelling: dict[str, str] = {}

    def resolve(name: str) -> str:
        key = entity_key(name) or ""
        target = alias_map.get(key)
        if target is not None:
            rewrites[key] = rewrites.get(key, 0) + 1
            name = target
            key = entity_key(name) or ""
        if key not in spelling:
            existing = db.get_entity(name, tenant=ns)
            spelling[key] = existing.name if existing is not None else name
        return spelling[key]

    facts = tuple(
        f.__class__(f.content, tuple(resolve(e) for e in f.entities), f.kind, f.confidence, f.span)
        for f in patch.add_facts
    )
    corrections = tuple(
        Correction(
            c.new_content,
            old_id=c.old_id,
            old_text=c.old_text,
            entities=None if c.entities is None else tuple(resolve(e) for e in c.entities),
            kind=c.kind,
            confidence=c.confidence,
            span=c.span,
        )
        for c in patch.corrections
    )
    add = tuple(
        Relation(resolve(r.src), resolve(r.dst), rel_kind=r.rel_kind, span=r.span)
        for r in patch.add_relations
    )
    remove = tuple(
        Relation(resolve(r.src), resolve(r.dst), rel_kind=r.rel_kind, span=r.span)
        for r in patch.remove_relations
    )
    notes: list[str] = []
    for alias in patch.entity_aliases:
        count = rewrites.get(entity_key(alias.name) or "", 0)
        existing = db.get_entity(alias.name, tenant=ns)
        canonical = db.get_entity(alias.canonical, tenant=ns)
        parts = []
        if count:
            parts.append(f"{count} reference{'s' if count != 1 else ''} rewritten")
        if existing is not None:
            parts.append(
                f"{alias.name!r} is already an entity; apply links it to {alias.canonical!r} "
                f"with an alias_of edge"
            )
        if canonical is None:
            parts.append(f"{alias.canonical!r} is not an entity yet")
        if parts:
            notes.append(f"alias {alias.name!r} -> {alias.canonical!r}: " + "; ".join(parts))
    return patch.replace(
        add_facts=facts,
        corrections=corrections,
        add_relations=add,
        remove_relations=remove,
        notes=patch.notes + tuple(notes),
    )


def resolve_corrections(patch: MemoryPatch, db: Anatid, *, tenant: Any = None) -> MemoryPatch:
    """Give every correction the id of the current memory it replaces.

    A correction by ``old_text`` becomes a correction by ``old_id`` when exactly one current
    memory reads that way; when several do, the one about the correction's entities is chosen,
    and failing that the newest, with a note.  A correction whose memory cannot be found, or is
    no longer current, is downgraded to a new fact and noted: nothing in the graph contradicts
    it, and the note's content is still worth keeping.
    """
    ns = db.resolve_tenant(tenant)
    kept: list[Correction] = []
    facts: list[AddFact] = list(patch.add_facts)
    notes: list[str] = []

    def downgrade(corr: Correction, why: str) -> None:
        facts.append(
            AddFact(
                corr.new_content,
                entities=corr.entities or (),
                kind=corr.kind or "fact",
                confidence=corr.confidence,
                span=corr.span,
            )
        )
        notes.append(
            f"correction of {corr.target}: {why}; added {corr.new_content!r} as a new fact"
        )

    for corr in patch.corrections:
        if corr.old_id is not None:
            memory = db.get(corr.old_id, tenant=ns, with_embedding=False)
            if memory is None:
                downgrade(corr, "not found")
            elif not memory.is_current:
                downgrade(corr, "no longer current")
            else:
                kept.append(corr)
            continue
        assert corr.old_text is not None
        matches = find_current_by_text(db, corr.old_text, tenant=ns)
        if not matches:
            downgrade(corr, "no current memory reads that way")
            continue
        chosen = matches[0]
        if len(matches) > 1:
            wanted = {entity_key(e) for e in corr.entities or ()}
            about = [
                m
                for m in matches
                if wanted and {entity_key(e.name) for e in db.entities_of(m, tenant=ns)} >= wanted
            ]
            chosen = about[0] if about else matches[0]
            notes.append(
                f"correction of {corr.target}: {len(matches)} current memories match; "
                f"chose memory {chosen}"
            )
        kept.append(
            Correction(
                corr.new_content,
                old_id=chosen,
                old_text=corr.old_text,
                entities=corr.entities,
                kind=corr.kind,
                confidence=corr.confidence,
                span=corr.span,
            )
        )
    return patch.replace(
        add_facts=tuple(facts), corrections=tuple(kept), notes=patch.notes + tuple(notes)
    )


def move_relations(patch: MemoryPatch, db: Anatid, *, tenant: Any = None) -> MemoryPatch:
    """Carry a corrected fact's edges over to its replacement.

    A correction that keeps some of the old memory's entities and swaps exactly one of them for
    exactly one new entity is a handover: "Atlas owns the ledger" corrected to "Cinder owns the
    ledger", "Priya is on call for Atlas" to "Tomasz is on call for Atlas", "Tomasz is a member
    of Dune" to "Tomasz is a member of Cinder".  Every current ``RELATES_TO`` edge between the
    replaced entity and a kept one that the fact's own note opened is closed, and the same edge,
    same kind and same direction, is opened with the new entity in the replaced one's place,
    unless the patch already says so.  Each move is a note.

    Three things keep this from guessing.  Only edges the fact's own note opened move: an edge
    another note stated ("Diego reports to Oskar") is not what a correction of "Diego wrote the
    doc for Oskar" is about.  "The fact's own note" follows the fact through its corrections: a
    wording correction gives the fact a new episode while the edge keeps the episode of the note
    that first stated it, so the lookup spans the fact's whole supersede chain.  And when those
    notes state more than one fact about the pair ("Ada reports to Bo" and "Ada mentors Bo" in
    one standup), an edge moves only when the correction's wording names its kind: correcting
    the manager moves ``reports_to`` and leaves ``mentors`` alone, with a note saying so.  A
    correction that swaps two entities, or none, or keeps none, moves nothing.

    The answer-quality benchmark (docs/quality.md) is where the need showed: the extraction
    model corrected the fact and left the old edge open in most handovers, so the graph arm kept
    walking through the previous owner.
    """
    ns = db.resolve_tenant(tenant)
    notes: list[str] = []
    remove: list[Relation] = list(patch.remove_relations)
    add: list[Relation] = list(patch.add_relations)

    def key(name: str) -> str:
        return entity_key(name) or ""

    def stated(rels: Sequence[Relation], src: str, dst: str, kind: str | None) -> bool:
        for rel in rels:
            if rel.rel_kind != kind:
                continue
            pair = {key(rel.src), key(rel.dst)}
            if pair == {key(src), key(dst)}:
                return True
        return False

    for corr in patch.corrections:
        if corr.old_id is None or corr.entities is None:
            continue
        old = db.get(corr.old_id, tenant=ns, with_embedding=False)
        if old is None:
            continue
        chain = db.provenance(corr.old_id, tenant=ns).chain
        episodes = sorted({int(m.episode_id) for m in chain if m.episode_id is not None})
        if not episodes:
            continue
        old_names = {key(e.name): e.name for e in db.entities_of(corr.old_id, tenant=ns)}
        new_names = {key(n): n for n in corr.entities if key(n)}
        replaced = [n for k, n in old_names.items() if k not in new_names]
        added = [n for k, n in new_names.items() if k not in old_names]
        kept = [n for k, n in old_names.items() if k in new_names]
        if len(replaced) != 1 or len(added) != 1 or not kept:
            continue
        gone, comes = replaced[0], added[0]
        gone_entity = db.get_entity(gone, tenant=ns)
        if gone_entity is None:
            continue
        wording = _words(old.content) | _words(corr.new_content)
        episode_list = ", ".join(str(e) for e in episodes)
        for other in kept:
            other_entity = db.get_entity(other, tenant=ns)
            if other_entity is None:
                continue
            rows = db.execute(
                f"SELECT src, dst, rel_kind FROM edges_relates "
                f"WHERE ((src = ? AND dst = ?) OR (src = ? AND dst = ?)) "
                f"AND episode_id IN ({episode_list}) "
                f"AND {tenant_sql()} AND {current_row_sql()} ORDER BY edge_id",
                [
                    gone_entity.entity_id,
                    other_entity.entity_id,
                    other_entity.entity_id,
                    gone_entity.entity_id,
                    ns.tenant_id,
                ],
            ).fetchall()
            if not rows:
                continue
            # How many current facts from those notes are about this pair?  One, and the edge
            # can only belong to the fact being corrected.  More, and the note said several
            # things about the two: an edge then moves only when the correction names its kind.
            siblings = db.execute(
                f"SELECT count(*) FROM memories m "
                f"WHERE m.episode_id IN ({episode_list}) AND {tenant_sql('m')} "
                f"AND {current_row_sql('m')} "
                f"AND EXISTS (SELECT 1 FROM edges_about a WHERE a.src = m.memory_id "
                f"            AND a.dst = ? AND {current_row_sql('a')}) "
                f"AND EXISTS (SELECT 1 FROM edges_about a WHERE a.src = m.memory_id "
                f"            AND a.dst = ? AND {current_row_sql('a')})",
                [ns.tenant_id, gone_entity.entity_id, other_entity.entity_id],
            ).fetchone()
            siblings = int(siblings[0]) if siblings else 0
            for src_id, _dst_id, kind in rows:
                if kind == ALIAS_REL_KIND:
                    continue
                if siblings > 1 and not _names_relation(kind, wording):
                    notes.append(
                        f"handover: left {kind} between {gone!r} and {other!r} alone; the note "
                        f"states {siblings} facts about them and the correction of memory "
                        f"{corr.old_id} does not name that relation"
                    )
                    continue
                gone_is_src = int(src_id) == gone_entity.entity_id
                src, dst = (gone, other) if gone_is_src else (other, gone)
                new_src, new_dst = (comes, other) if gone_is_src else (other, comes)
                moved = []
                if not stated(remove, src, dst, kind):
                    remove.append(Relation(src, dst, kind))
                    moved.append("closed")
                if not stated(add, new_src, new_dst, kind):
                    add.append(Relation(new_src, new_dst, kind))
                    moved.append("opened")
                if moved:
                    notes.append(
                        f"handover: {' and '.join(moved)} {kind} between {gone!r} and "
                        f"{other!r} for {comes!r}, from the correction of memory {corr.old_id}"
                    )
    if not notes:
        return patch
    return patch.replace(
        remove_relations=tuple(remove), add_relations=tuple(add), notes=patch.notes + tuple(notes)
    )


def _words(text: str | None) -> set[str]:
    """The lower-cased words of ``text``, each also without a trailing ``s``, so a relation
    kind matches the fact's wording across ``reports``/``report`` and ``mentors``/``mentor``."""
    out: set[str] = set()
    for word in re.findall(r"[a-z0-9]+", (text or "").lower()):
        out.add(word)
        if len(word) > 3 and word.endswith("s"):
            out.add(word[:-1])
    return out


def _names_relation(rel_kind: str | None, wording: set[str]) -> bool:
    """Whether every word of ``rel_kind`` (``reports_to``, ``on_call_for``) is in ``wording``."""
    parts = re.findall(r"[a-z0-9]+", (rel_kind or "").lower())
    if not parts:
        return False
    return all(p in wording or (len(p) > 3 and p.rstrip("s") in wording) for p in parts)


def dedupe(patch: MemoryPatch, db: Anatid, *, tenant: Any = None) -> MemoryPatch:
    """Drop what the graph already holds, and say so.

    A fact is a duplicate when a current memory has the same folded content
    (:func:`~anatid.ingest.patch.fold_text`) and the same entities, or when the patch itself
    already proposes it.  A relation is a duplicate when a current ``RELATES_TO`` edge of the
    same kind already joins the two entities in either direction.  A relation removal that
    matches no current edge closes nothing and is dropped too.  Every drop is a note.
    """
    ns = db.resolve_tenant(tenant)
    notes: list[str] = []

    def keys(names: Sequence[str]) -> frozenset[str]:
        return frozenset(entity_key(n) or "" for n in names)

    seen: set[tuple[str, frozenset[str]]] = set()
    facts: list[AddFact] = []
    for fact in patch.add_facts:
        signature = (fold_text(fact.content) or "", keys(fact.entities))
        if signature in seen:
            notes.append(f"dedupe: dropped {fact.content!r}, proposed twice in this patch")
            continue
        seen.add(signature)
        held = [
            m
            for m in find_current_by_text(db, fact.content, tenant=ns)
            if keys([e.name for e in db.entities_of(m, tenant=ns)]) == signature[1]
        ]
        if held:
            notes.append(
                f"dedupe: dropped {fact.content!r}; memory {held[0]} is current with the same "
                f"entities"
            )
            continue
        facts.append(fact)

    def endpoint_ids(rel: Relation) -> tuple[int, int] | None:
        s = db.get_entity(rel.src, tenant=ns)
        d = db.get_entity(rel.dst, tenant=ns)
        if s is None or d is None:
            return None
        return s.entity_id, d.entity_id

    seen_rel: set[tuple[str, str, str | None]] = set()
    add: list[Relation] = []
    for rel in patch.add_relations:
        signature_rel = (entity_key(rel.src) or "", entity_key(rel.dst) or "", rel.rel_kind)
        if signature_rel in seen_rel:
            notes.append(f"dedupe: dropped relation {rel}, proposed twice in this patch")
            continue
        seen_rel.add(signature_rel)
        ids = endpoint_ids(rel)
        if ids is not None and current_relation_ids(
            db, ids[0], ids[1], rel_kind=rel.rel_kind, tenant=ns
        ):
            notes.append(f"dedupe: dropped relation {rel}; a current edge already holds it")
            continue
        add.append(rel)

    remove: list[Relation] = []
    for rel in patch.remove_relations:
        ids = endpoint_ids(rel)
        if ids is None or not current_relation_ids(
            db, ids[0], ids[1], rel_kind=rel.rel_kind, tenant=ns
        ):
            notes.append(f"dedupe: dropped removal of {rel}; no current edge to close")
            continue
        remove.append(rel)

    return patch.replace(
        add_facts=tuple(facts),
        add_relations=tuple(add),
        remove_relations=tuple(remove),
        notes=patch.notes + tuple(notes),
    )


def prepare(patch: MemoryPatch, db: Anatid, *, tenant: Any = None) -> MemoryPatch:
    """:func:`resolve_entities`, :func:`resolve_corrections`, :func:`move_relations`, then
    :func:`dedupe`."""
    patch = resolve_entities(patch, db, tenant=tenant)
    patch = resolve_corrections(patch, db, tenant=tenant)
    patch = move_relations(patch, db, tenant=tenant)
    return dedupe(patch, db, tenant=tenant)


# --------------------------------------------------------------------------- 2 + 3. propose


def propose(
    db: Anatid,
    text: str,
    *,
    extractor: Extractor,
    tenant: Any = None,
    context_limit: int = EXISTING_LIMIT,
) -> MemoryPatch:
    """Extract a patch for ``text`` and prepare it against the database, without applying it."""
    existing = existing_context(db, text, tenant=tenant, limit=context_limit)
    proposed = extractor.extract(text, existing=existing)
    if not proposed.source_text:
        proposed = proposed.replace(source_text=text)
    return prepare(proposed, db, tenant=tenant)


# --------------------------------------------------------------------------- 4. ingest


def ingest(
    db: Anatid,
    text: str,
    *,
    extractor: Extractor,
    writer: str | None,
    source: str | None = None,
    review: ReviewHook | None = None,
    now: Any = None,
    tenant: Any = None,
    embedder: Callable[[str], Sequence[float]] | None = None,
    context_limit: int = EXISTING_LIMIT,
) -> PatchReceipt | None:
    """Turn ``text`` into memory: propose, review, apply.

    ``review`` receives the prepared patch and returns the patch to apply, which may be an
    edited copy (:meth:`~anatid.ingest.patch.MemoryPatch.replace`), or ``None`` to decline, in
    which case nothing is written and ``None`` is returned.  Without ``review`` the prepared
    patch is applied as proposed.  ``source`` labels the stored episode; ``now`` stamps every
    row the patch writes; ``embedder`` gives each new memory a vector (see
    :meth:`~anatid.ingest.patch.MemoryPatch.apply`).  The receipt says what landed and carries
    the patch it came from.

    A patch with no operations is still applied: the text is stored as an episode, so the
    record shows the note was read even when it changed nothing.
    """
    patch = propose(db, text, extractor=extractor, tenant=tenant, context_limit=context_limit)
    if review is not None:
        reviewed = review(patch)
        if reviewed is None:
            return None
        if not isinstance(reviewed, MemoryPatch):
            raise TypeError(
                f"the review hook must return a MemoryPatch or None, got {type(reviewed).__name__}"
            )
        patch = reviewed
    return patch.apply(db, writer=writer, source=source, now=now, tenant=tenant, embedder=embedder)
