"""Ingestion: from a note, a message or a document to reviewed changes in the memory graph.

The core verbs take facts and edges one at a time.  This package takes text.  An
:class:`Extractor` proposes a :class:`MemoryPatch` (facts to add, facts to correct, edges to
open and close, names that mean an existing entity), the pipeline resolves it against what the
graph already holds and says what it changed, a reviewer may edit or decline it, and
:meth:`MemoryPatch.apply` commits the whole patch in one transaction with the raw text stored
as an episode first.

::

    from anatid import Anatid
    from anatid.ingest import OpenAICompatibleExtractor, ingest

    extractor = OpenAICompatibleExtractor(model="gpt-4o-mini", api_key=key)
    with Anatid.open("team.anatid", tenant=1) as db:
        receipt = ingest(
            db,
            "Cy took over the ingest service from Bo this week.",
            extractor=extractor,
            writer="notes-bot",
            source="standup/2026-06-15",
            review=lambda patch: patch if input(patch.describe() + "\\napply? ") == "y" else None,
        )

Nothing here is imported by ``import anatid``.  The ``openai`` package is needed only for
:class:`OpenAICompatibleExtractor`, and only when it is asked to build its own client.
"""

from __future__ import annotations

from .extract import (
    SYSTEM_PROMPT,
    ExtractCall,
    ExtractionError,
    Extractor,
    KnownFact,
    OpenAICompatibleExtractor,
    ScriptedExtractor,
    parse_json_object,
    render_existing,
)
from .patch import (
    ALIAS_REL_KIND,
    EPISODE_KIND,
    PATCH_JSON_SCHEMA,
    AddFact,
    Alias,
    Correction,
    MemoryPatch,
    PatchReceipt,
    Relation,
    Span,
    current_relation_ids,
    find_current_by_text,
    fold_text,
    fold_text_sql,
)
from .pipeline import (
    ENTITY_SCAN_LIMIT,
    EXISTING_LIMIT,
    ReviewHook,
    dedupe,
    existing_context,
    move_relations,
    ingest,
    prepare,
    propose,
    resolve_corrections,
    resolve_entities,
)

__all__ = [
    # the patch
    "MemoryPatch",
    "PatchReceipt",
    "AddFact",
    "Correction",
    "Relation",
    "Alias",
    "Span",
    "PATCH_JSON_SCHEMA",
    "EPISODE_KIND",
    "ALIAS_REL_KIND",
    "fold_text",
    "fold_text_sql",
    "find_current_by_text",
    "current_relation_ids",
    # extractors
    "Extractor",
    "ExtractionError",
    "KnownFact",
    "ExtractCall",
    "ScriptedExtractor",
    "OpenAICompatibleExtractor",
    "SYSTEM_PROMPT",
    "parse_json_object",
    "render_existing",
    # the pipeline
    "ingest",
    "propose",
    "prepare",
    "existing_context",
    "resolve_entities",
    "resolve_corrections",
    "move_relations",
    "dedupe",
    "ReviewHook",
    "EXISTING_LIMIT",
    "ENTITY_SCAN_LIMIT",
]
