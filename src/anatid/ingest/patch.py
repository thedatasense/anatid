"""The memory patch: proposed changes to the graph, applied as one transaction.

A :class:`MemoryPatch` is a frozen description of what a piece of source text says the memory
should now hold: facts to add, facts to correct, relationships to open and close, and names
that mean an existing entity.  It is data.  Nothing happens until :meth:`MemoryPatch.apply`,
and everything that happens there happens in one transaction: the raw text is stored as an
episode first, then every fact, correction and edge is written carrying that episode's id, so
:meth:`anatid.Anatid.provenance` can always walk from a belief back to the note it came from.
A failure anywhere leaves the database as it was.

Between proposal and application a patch can be shown to a person (:meth:`MemoryPatch.describe`
renders it as a diff), edited (:meth:`MemoryPatch.replace`), or carried across a process
boundary (:meth:`MemoryPatch.to_json` / :meth:`MemoryPatch.from_json`).  On the wire every id
is a decimal string, for the reason :mod:`anatid.integrations.wire` gives: a 63-bit id does not
survive a JavaScript JSON parser as a number.

Each operation may carry the :class:`Span` of source text it came from, so a reviewer can see
the sentence behind a proposed fact.  A span is optional; an extractor that cannot locate its
evidence leaves it ``None``.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ..errors import NotFoundError, ValidationError
from ..integrations.wire import coerce_id, wire_id, wire_ids
from ..schema import entity_key, entity_key_sql
from ..types import to_utc_naive, utcnow
from ..visibility import current_row_sql, tenant_sql

if TYPE_CHECKING:
    from ..database import Anatid

__all__ = [
    "ALIAS_REL_KIND",
    "EPISODE_KIND",
    "PATCH_JSON_SCHEMA",
    "Span",
    "AddFact",
    "Correction",
    "Relation",
    "Alias",
    "MemoryPatch",
    "PatchReceipt",
    "fold_text",
    "fold_text_sql",
    "find_current_by_text",
    "current_relation_ids",
]

#: ``episodes.kind`` for the raw text a patch stores before deriving anything from it.
EPISODE_KIND = "ingest"

#: ``rel_kind`` of the edge :meth:`MemoryPatch.apply` writes between an alias that already exists
#: as its own entity and the entity it turns out to name, so graph recall crosses between them.
ALIAS_REL_KIND = "alias_of"


# --------------------------------------------------------------------------- text folding


def fold_text(text: str | None) -> str | None:
    """Canonical form of a memory's content for equality: lower case, one space per whitespace
    run, no surrounding space, no trailing full stop.

    The same folding as an entity's canonical key (:func:`anatid.schema.entity_key`), with the
    trailing full stop removed as well, because "Ada leads Kestrel" and "Ada leads Kestrel." are
    one fact.  :func:`fold_text_sql` is the SQL twin; a lookup binds the raw text against that.
    """
    folded = entity_key(text)
    return None if folded is None else folded.rstrip(".").rstrip(" ")


def fold_text_sql(expr: str = "content") -> str:
    """The SQL for :func:`fold_text` over ``expr`` (a column reference or a ``?`` placeholder)."""
    return f"rtrim(rtrim({entity_key_sql(expr)}, '.'), ' ')"


# --------------------------------------------------------------------------- spans


@dataclass(frozen=True, slots=True)
class Span:
    """Half-open character range ``[start, end)`` into a patch's source text."""

    start: int
    end: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "end", int(self.end))
        if self.start < 0 or self.end < self.start:
            raise ValidationError(f"span [{self.start}, {self.end}) is not a valid range")

    def text(self, source: str) -> str:
        """The characters the span covers in ``source``."""
        return source[self.start : self.end]

    @classmethod
    def locate(cls, quote: str | None, source: str) -> Span | None:
        """Find ``quote`` in ``source``, exactly first and then ignoring case; ``None`` if absent."""
        if not quote or not source:
            return None
        at = source.find(quote)
        if at < 0:
            at = source.casefold().find(quote.casefold())
        if at < 0:
            return None
        return cls(at, at + len(quote))

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}


# --------------------------------------------------------------------------- operations


def _names(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = (value,)
    out: list[str] = []
    for item in value:
        name = str(item).strip()
        if name and name not in out:
            out.append(name)
    return tuple(out)


def _unit(value: Any, default: float = 1.0) -> float:
    if value is None:
        return default
    return float(value)


@dataclass(frozen=True, slots=True)
class AddFact:
    """A new memory: ``content`` about ``entities``."""

    content: str
    entities: tuple[str, ...] = ()
    kind: str = "fact"
    confidence: float = 1.0
    span: Span | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", str(self.content).strip())
        object.__setattr__(self, "entities", _names(self.entities))
        object.__setattr__(self, "kind", str(self.kind or "fact"))
        object.__setattr__(self, "confidence", _unit(self.confidence))
        if not self.content:
            raise ValidationError("a fact needs content")


@dataclass(frozen=True, slots=True)
class Correction:
    """Replace one current memory with ``new_content``.

    The memory is named by ``old_id`` or, when the proposer only has the sentence, by
    ``old_text`` (matched against current memories with :func:`fold_text`).  ``entities=None``
    inherits the old memory's entities; a tuple replaces them.  ``kind=None`` inherits the kind.
    """

    new_content: str
    old_id: int | None = None
    old_text: str | None = None
    entities: tuple[str, ...] | None = None
    kind: str | None = None
    confidence: float = 1.0
    span: Span | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "new_content", str(self.new_content).strip())
        if self.old_id is not None:
            object.__setattr__(self, "old_id", coerce_id(self.old_id))
        if self.old_text is not None:
            text = str(self.old_text).strip()
            object.__setattr__(self, "old_text", text or None)
        if self.entities is not None:
            object.__setattr__(self, "entities", _names(self.entities))
        object.__setattr__(self, "confidence", _unit(self.confidence))
        if not self.new_content:
            raise ValidationError("a correction needs new content")
        if self.old_id is None and self.old_text is None:
            raise ValidationError(
                f"correction to {self.new_content!r} names no memory: give old_id or old_text"
            )

    @property
    def target(self) -> str:
        """How the correction names the memory it replaces, for messages."""
        if self.old_id is not None:
            return f"memory {self.old_id}"
        return f"memory matching {self.old_text!r}"


@dataclass(frozen=True, slots=True)
class Relation:
    """A ``RELATES_TO`` edge between two entities, named by their names."""

    src: str
    dst: str
    rel_kind: str | None = None
    span: Span | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "src", str(self.src).strip())
        object.__setattr__(self, "dst", str(self.dst).strip())
        if self.rel_kind is not None:
            kind = str(self.rel_kind).strip()
            object.__setattr__(self, "rel_kind", kind or None)
        if not self.src or not self.dst:
            raise ValidationError("a relation needs both a source and a destination entity")

    def __str__(self) -> str:
        kind = self.rel_kind or "relates_to"
        return f"{self.src} -{kind}-> {self.dst}"


@dataclass(frozen=True, slots=True)
class Alias:
    """``name`` in the source text means the entity ``canonical``."""

    name: str
    canonical: str
    span: Span | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "canonical", str(self.canonical).strip())
        if not self.name or not self.canonical:
            raise ValidationError("an alias needs both a name and the canonical entity")


# --------------------------------------------------------------------------- lookups


def find_current_by_text(db: Anatid, text: str, *, tenant: Any = None) -> list[int]:
    """Ids of current memories whose folded content equals ``fold_text(text)``, newest first."""
    ns = db.resolve_tenant(tenant)
    rows = db.execute(
        f"SELECT memory_id FROM memories WHERE {tenant_sql()} AND {current_row_sql()} "
        f"AND {fold_text_sql('content')} = {fold_text_sql('?')} "
        f"ORDER BY created_at DESC, memory_id DESC",
        [ns.tenant_id, text],
    ).fetchall()
    return [int(r[0]) for r in rows]


def current_relation_ids(
    db: Anatid, src_id: int, dst_id: int, *, rel_kind: str | None = None, tenant: Any = None
) -> list[int]:
    """Ids of the current ``RELATES_TO`` edges between two entities, in either direction."""
    ns = db.resolve_tenant(tenant)
    where = f"((src = ? AND dst = ?) OR (src = ? AND dst = ?)) AND {tenant_sql()} AND {current_row_sql()}"
    params: list[Any] = [int(src_id), int(dst_id), int(dst_id), int(src_id), ns.tenant_id]
    if rel_kind is not None:
        where += " AND rel_kind = ?"
        params.append(str(rel_kind))
    rows = db.execute(
        f"SELECT edge_id FROM edges_relates WHERE {where} ORDER BY edge_id", params
    ).fetchall()
    return [int(r[0]) for r in rows]


# --------------------------------------------------------------------------- the receipt


@dataclass(frozen=True, slots=True)
class PatchReceipt:
    """What :meth:`MemoryPatch.apply` wrote, all of it in one transaction.

    ``memories_created`` are the new memories, facts first and then the replacements the
    corrections wrote; ``memories_closed`` the memories those corrections superseded, and
    ``corrections`` pairs them ``(old_id, new_id)``.  ``relations_opened`` and
    ``relations_closed`` are ``RELATES_TO`` edge ids.  ``aliases`` are the ``(name, canonical)``
    pairs whose references were rewritten.  Every created row carries ``episode_id``.
    """

    episode_id: int
    tenant_id: int
    writer: str | None
    at: _dt.datetime
    patch: MemoryPatch
    memories_created: tuple[int, ...] = ()
    memories_closed: tuple[int, ...] = ()
    corrections: tuple[tuple[int, int], ...] = ()
    relations_opened: tuple[int, ...] = ()
    relations_closed: tuple[int, ...] = ()
    aliases: tuple[tuple[str, str], ...] = ()

    @property
    def changes(self) -> int:
        """Rows the patch created or closed, the episode excluded."""
        return (
            len(self.memories_created)
            + len(self.memories_closed)
            + len(self.relations_opened)
            + len(self.relations_closed)
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form; every id is a decimal string."""
        return {
            "episode_id": wire_id(self.episode_id),
            "tenant_id": self.tenant_id,
            "writer": self.writer,
            "at": self.at.isoformat(),
            "memories_created": wire_ids(self.memories_created),
            "memories_closed": wire_ids(self.memories_closed),
            "corrections": [
                {"old_id": wire_id(old), "new_id": wire_id(new)} for old, new in self.corrections
            ],
            "relations_opened": wire_ids(self.relations_opened),
            "relations_closed": wire_ids(self.relations_closed),
            "aliases": [{"name": n, "canonical": c} for n, c in self.aliases],
            "patch": self.patch.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def describe(self) -> str:
        """One paragraph of what landed, with ids."""
        parts = [f"episode {self.episode_id} stored"]
        if self.memories_created:
            parts.append(
                f"{len(self.memories_created)} memories created "
                f"({', '.join(str(i) for i in self.memories_created)})"
            )
        if self.memories_closed:
            parts.append(
                f"{len(self.memories_closed)} superseded "
                f"({', '.join(str(i) for i in self.memories_closed)})"
            )
        if self.relations_opened:
            parts.append(f"{len(self.relations_opened)} relations opened")
        if self.relations_closed:
            parts.append(f"{len(self.relations_closed)} relations closed")
        if self.aliases:
            parts.append("aliases " + ", ".join(f"{n!r} -> {c!r}" for n, c in self.aliases))
        return "; ".join(parts) + "."


# --------------------------------------------------------------------------- the patch


def _quote(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class MemoryPatch:
    """Proposed operations on the memory graph, derived from ``source_text``.

    ``notes`` is where anything that shaped the patch says so: a fact the dedupe step dropped,
    a correction whose target was not found, an entry the parser could not read.  They are
    rendered by :meth:`describe` and travel with the patch, and :meth:`apply` does not act on
    them.
    """

    source_text: str = ""
    add_facts: tuple[AddFact, ...] = ()
    corrections: tuple[Correction, ...] = ()
    add_relations: tuple[Relation, ...] = ()
    remove_relations: tuple[Relation, ...] = ()
    entity_aliases: tuple[Alias, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_text", str(self.source_text or ""))
        object.__setattr__(self, "add_facts", tuple(self.add_facts))
        object.__setattr__(self, "corrections", tuple(self.corrections))
        object.__setattr__(self, "add_relations", tuple(self.add_relations))
        object.__setattr__(self, "remove_relations", tuple(self.remove_relations))
        object.__setattr__(self, "entity_aliases", tuple(self.entity_aliases))
        object.__setattr__(self, "notes", tuple(str(n) for n in self.notes))

    # ------------------------------------------------------------------ shape

    @property
    def operations(self) -> int:
        """How many operations the patch proposes (notes and aliases not counted)."""
        return (
            len(self.add_facts)
            + len(self.corrections)
            + len(self.add_relations)
            + len(self.remove_relations)
        )

    @property
    def is_empty(self) -> bool:
        """True when applying would store the episode and nothing else."""
        return self.operations == 0 and not self.entity_aliases

    def replace(self, **changes: Any) -> MemoryPatch:
        """A copy with the given fields replaced (the editing step of a review)."""
        return replace(self, **changes)

    def with_notes(self, *notes: str) -> MemoryPatch:
        return self.replace(notes=self.notes + tuple(notes)) if notes else self

    def entity_names(self) -> tuple[str, ...]:
        """Every entity name the operations mention, in first-seen order."""
        seen: list[str] = []
        for name in self._mentioned():
            if name not in seen:
                seen.append(name)
        return tuple(seen)

    def _mentioned(self) -> Iterable[str]:
        for fact in self.add_facts:
            yield from fact.entities
        for corr in self.corrections:
            yield from corr.entities or ()
        for rel in (*self.add_relations, *self.remove_relations):
            yield rel.src
            yield rel.dst

    def alias_map(self) -> dict[str, str]:
        """``entity_key(alias name) -> canonical name`` for every alias in the patch."""
        out: dict[str, str] = {}
        for alias in self.entity_aliases:
            key = entity_key(alias.name)
            if key and entity_key(alias.canonical) != key:
                out[key] = alias.canonical
        return out

    def resolve_name(self, name: str) -> str:
        """``name`` with the patch's aliases applied."""
        key = entity_key(name)
        return self.alias_map().get(key or "", name)

    # ------------------------------------------------------------------ rendering

    def describe(self) -> str:
        """The patch as a diff a person can review.

        ``+`` adds, ``~`` corrects, ``-`` closes, ``=`` declares an alias.  The evidence span
        follows each line when the operation has one.
        """
        counts = []
        if self.add_facts:
            counts.append(f"{len(self.add_facts)} fact{'s' if len(self.add_facts) != 1 else ''}")
        if self.corrections:
            n = len(self.corrections)
            counts.append(f"{n} correction{'s' if n != 1 else ''}")
        if self.add_relations:
            counts.append(f"{len(self.add_relations)} relation(s) added")
        if self.remove_relations:
            counts.append(f"{len(self.remove_relations)} relation(s) removed")
        if self.entity_aliases:
            n = len(self.entity_aliases)
            counts.append(f"{n} alias{'es' if n != 1 else ''}")
        head = "memory patch: " + (", ".join(counts) if counts else "no operations")
        lines = [head]
        if self.source_text:
            preview = " ".join(self.source_text.split())
            if len(preview) > 90:
                preview = preview[:87] + "..."
            lines.append(f"source: {_quote(preview)} ({len(self.source_text)} chars)")

        def evidence(span: Span | None) -> str:
            if span is None or not self.source_text:
                return ""
            return f"  [{span.start}:{span.end}] {_quote(span.text(self.source_text))}"

        def about(names: Sequence[str] | None) -> str:
            if names is None:
                return "  about: (inherited)"
            return f"  about: {', '.join(names)}" if names else ""

        for fact in self.add_facts:
            extra = "" if fact.kind == "fact" else f"  kind: {fact.kind}"
            if fact.confidence < 1.0:
                extra += f"  confidence: {fact.confidence:g}"
            lines.append(
                f"  + fact        {_quote(fact.content)}{about(fact.entities)}{extra}"
                f"{evidence(fact.span)}"
            )
        for corr in self.corrections:
            target = f"memory {corr.old_id}" if corr.old_id is not None else "memory (by text)"
            was = f" {_quote(corr.old_text)}" if corr.old_text else ""
            extra = "" if corr.kind is None else f"  kind: {corr.kind}"
            if corr.confidence < 1.0:
                extra += f"  confidence: {corr.confidence:g}"
            lines.append(f"  ~ correction  {target}{was}")
            lines.append(
                f"                -> {_quote(corr.new_content)}{about(corr.entities)}{extra}"
                f"{evidence(corr.span)}"
            )
        for rel in self.remove_relations:
            lines.append(f"  - relation    {rel}{evidence(rel.span)}")
        for rel in self.add_relations:
            lines.append(f"  + relation    {rel}{evidence(rel.span)}")
        for alias in self.entity_aliases:
            lines.append(
                f"  = alias       {_quote(alias.name)} -> {_quote(alias.canonical)}"
                f"{evidence(alias.span)}"
            )
        for note in self.notes:
            lines.append(f"  note          {note}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ JSON

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form.  Ids are decimal strings; spans are ``{"start", "end"}``."""

        def span(s: Span | None) -> dict[str, int] | None:
            return None if s is None else s.to_dict()

        return {
            "source_text": self.source_text,
            "add_facts": [
                {
                    "content": f.content,
                    "entities": list(f.entities),
                    "kind": f.kind,
                    "confidence": f.confidence,
                    "span": span(f.span),
                }
                for f in self.add_facts
            ],
            "corrections": [
                {
                    "old_memory_id": wire_id(c.old_id),
                    "old_text": c.old_text,
                    "new_content": c.new_content,
                    "entities": None if c.entities is None else list(c.entities),
                    "kind": c.kind,
                    "confidence": c.confidence,
                    "span": span(c.span),
                }
                for c in self.corrections
            ],
            "add_relations": [
                {"src": r.src, "dst": r.dst, "rel_kind": r.rel_kind, "span": span(r.span)}
                for r in self.add_relations
            ],
            "remove_relations": [
                {"src": r.src, "dst": r.dst, "rel_kind": r.rel_kind, "span": span(r.span)}
                for r in self.remove_relations
            ],
            "entity_aliases": [
                {"name": a.name, "canonical": a.canonical, "span": span(a.span)}
                for a in self.entity_aliases
            ],
            "notes": list(self.notes),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, source_text: str | None = None) -> MemoryPatch:
        """Build a patch from the JSON shape, tolerantly.

        Accepts what :meth:`to_dict` writes and what a model writes against
        :data:`PATCH_JSON_SCHEMA`: ``facts`` for ``add_facts``, ``aliases`` for
        ``entity_aliases``, ``old_id`` for ``old_memory_id``, an ``evidence`` quote in place of
        a ``span`` (located in the source text), a single string where a list of entities is
        expected, and an id as a decimal string or an integer.  An entry that cannot be read is
        skipped and the reason is recorded in ``notes``, so nothing is dropped silently.
        """
        if not isinstance(data, Mapping):
            raise ValidationError(f"a memory patch is a JSON object, got {type(data).__name__}")
        text = source_text if source_text is not None else str(data.get("source_text") or "")
        notes: list[str] = [str(n) for n in _as_list(data.get("notes"))]

        def span_of(item: Mapping[str, Any]) -> Span | None:
            raw = item.get("span")
            if isinstance(raw, Mapping) and "start" in raw and "end" in raw:
                try:
                    return Span(int(raw["start"]), int(raw["end"]))
                except (TypeError, ValueError, ValidationError):
                    return None
            return Span.locate(_str_or_none(item.get("evidence")), text)

        def confidence_of(item: Mapping[str, Any], what: str) -> float:
            raw = item.get("confidence")
            if raw is None:
                return 1.0
            try:
                value = float(raw)
            except (TypeError, ValueError):
                notes.append(f"parser: {what} had confidence {raw!r}; used 1.0")
                return 1.0
            if math.isnan(value):
                notes.append(f"parser: {what} had confidence {raw!r}; used 1.0")
                return 1.0
            if value < 0.0 or value > 1.0:
                clamped = min(1.0, max(0.0, value))
                notes.append(f"parser: {what} had confidence {raw!r}; clamped to {clamped:g}")
                return clamped
            return value

        facts: list[AddFact] = []
        for item in _entries(data, "add_facts", "facts", notes=notes):
            content = _str_or_none(item.get("content"))
            if not content:
                notes.append("parser: skipped a fact without content")
                continue
            facts.append(
                AddFact(
                    content,
                    entities=_names(item.get("entities")),
                    kind=_str_or_none(item.get("kind")) or "fact",
                    confidence=confidence_of(item, f"fact {content!r}"),
                    span=span_of(item),
                )
            )

        corrections: list[Correction] = []
        for item in _entries(data, "corrections", notes=notes):
            new_content = _str_or_none(item.get("new_content")) or _str_or_none(item.get("content"))
            if not new_content:
                notes.append("parser: skipped a correction without new content")
                continue
            raw_id = item.get("old_memory_id", item.get("old_id"))
            old_id: int | None = None
            if raw_id is not None and raw_id != "":
                try:
                    old_id = coerce_id(raw_id)
                except ValueError:
                    notes.append(
                        f"parser: correction to {new_content!r} named memory {raw_id!r}, which "
                        f"is not an id; matching by text instead"
                    )
            old_text = _str_or_none(item.get("old_text")) or _str_or_none(item.get("old_content"))
            if old_id is None and not old_text:
                notes.append(
                    f"parser: correction to {new_content!r} names no memory (no old_memory_id, "
                    f"no old_text); skipped"
                )
                continue
            raw_entities = item.get("entities")
            corrections.append(
                Correction(
                    new_content,
                    old_id=old_id,
                    old_text=old_text,
                    entities=None if raw_entities is None else _names(raw_entities),
                    kind=_str_or_none(item.get("kind")),
                    confidence=confidence_of(item, f"correction {new_content!r}"),
                    span=span_of(item),
                )
            )

        def relations(*keys: str) -> list[Relation]:
            out: list[Relation] = []
            for item in _entries(data, *keys, notes=notes):
                src = _str_or_none(item.get("src")) or _str_or_none(item.get("source"))
                dst = _str_or_none(item.get("dst")) or _str_or_none(item.get("target"))
                if not src or not dst:
                    notes.append(f"parser: skipped a relation missing an endpoint ({dict(item)!r})")
                    continue
                out.append(
                    Relation(
                        src, dst, rel_kind=_str_or_none(item.get("rel_kind")), span=span_of(item)
                    )
                )
            return out

        aliases: list[Alias] = []
        for item in _entries(data, "entity_aliases", "aliases", notes=notes):
            name = _str_or_none(item.get("name")) or _str_or_none(item.get("alias"))
            canonical = _str_or_none(item.get("canonical")) or _str_or_none(item.get("entity"))
            if not name or not canonical:
                notes.append(f"parser: skipped an alias missing a name ({dict(item)!r})")
                continue
            if entity_key(name) == entity_key(canonical):
                continue
            aliases.append(Alias(name, canonical, span=span_of(item)))

        return cls(
            source_text=text,
            add_facts=tuple(facts),
            corrections=tuple(corrections),
            add_relations=tuple(relations("add_relations", "relations")),
            remove_relations=tuple(relations("remove_relations")),
            entity_aliases=tuple(aliases),
            notes=tuple(notes),
        )

    @classmethod
    def from_json(cls, text: str, *, source_text: str | None = None) -> MemoryPatch:
        """:meth:`from_dict` over a JSON document."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"memory patch is not valid JSON: {exc}") from exc
        return cls.from_dict(data, source_text=source_text)

    # ------------------------------------------------------------------ apply

    def apply(
        self,
        db: Anatid,
        *,
        writer: str | None,
        episode: str | None = None,
        source: str | None = None,
        now: _dt.datetime | None = None,
        tenant: Any = None,
        embedder: Callable[[str], Sequence[float]] | None = None,
    ) -> PatchReceipt:
        """Commit the whole patch in one transaction and return what was written.

        ``episode`` is the raw text to store first (default: the patch's ``source_text``) and
        ``source`` labels where it came from (a file name, a channel).  Every memory and edge
        the patch writes carries the new episode's id.  The operations run in this order:
        aliases, new facts, corrections, relations removed, relations added.  Any error rolls
        back all of it, the episode included.  ``embedder`` is called with each new memory's
        content and its vector is stored with the memory, so the vector arm of
        :meth:`anatid.Anatid.recall` can find it.  Without one the verbs decide: a handle
        opened with ``Anatid.open(embedder=...)`` embeds every memory it is not given a vector
        for, and a handle without one writes no embedding.

        Aliases rewrite every entity reference in the patch to the canonical name.  When the
        alias is already an entity of its own, an ``alias_of`` edge (:data:`ALIAS_REL_KIND`)
        is written between it and the canonical entity, so graph recall crosses between the two
        spellings; the receipt lists it under ``relations_opened``.

        A correction that names its memory by ``old_text`` must match exactly one current
        memory; none raises :class:`~anatid.errors.NotFoundError` and several raise
        :class:`~anatid.errors.ValidationError`.  :func:`anatid.ingest.prepare` resolves these
        ahead of time so a reviewer sees the resolved id.  Removing a relation whose entities
        or edge do not exist closes nothing and is not an error.
        """
        text = episode if episode is not None else self.source_text
        if not text or not text.strip():
            raise ValidationError(
                "a patch needs source text to store as its episode: set source_text or pass "
                "episode="
            )
        ns = db.resolve_tenant(tenant)
        at = to_utc_naive(now) or utcnow()
        resolve = self.resolve_name

        def embed(content: str) -> list[float] | None:
            return None if embedder is None else [float(x) for x in embedder(content)]

        created: list[int] = []
        closed: list[int] = []
        pairs: list[tuple[int, int]] = []
        opened: list[int] = []
        closed_edges: list[int] = []
        aliases: list[tuple[str, str]] = []

        with db.transaction():
            ep = db.episode(
                text, source=source, kind=EPISODE_KIND, writer=writer, tenant=ns, now=at
            )
            eid = ep.episode_id

            for alias in self.entity_aliases:
                if entity_key(alias.name) == entity_key(alias.canonical):
                    continue
                aliases.append((alias.name, alias.canonical))
                existing = db.get_entity(alias.name, tenant=ns)
                if existing is None:
                    continue
                canonical_id = db.entity_id(
                    alias.canonical, tenant=ns, create=True, now=at, writer=writer, episode_id=eid
                )
                if current_relation_ids(
                    db, existing.entity_id, canonical_id, rel_kind=ALIAS_REL_KIND, tenant=ns
                ):
                    continue
                edge = db.relate(
                    existing.entity_id,
                    canonical_id,
                    rel_kind=ALIAS_REL_KIND,
                    writer=writer,
                    episode_id=eid,
                    now=at,
                    tenant=ns,
                )
                opened.append(edge.edge_id)

            for fact in self.add_facts:
                memory = db.remember(
                    fact.content,
                    entities=[resolve(e) for e in fact.entities],
                    kind=fact.kind,
                    embedding=embed(fact.content),
                    confidence=fact.confidence,
                    writer=writer,
                    episode_id=eid,
                    now=at,
                    tenant=ns,
                )
                created.append(memory.memory_id)

            for corr in self.corrections:
                old_id = self._target_of(db, corr, ns)
                replacement = db.supersede(
                    old_id,
                    corr.new_content,
                    entities=None if corr.entities is None else [resolve(e) for e in corr.entities],
                    kind=corr.kind,
                    embedding=embed(corr.new_content),
                    confidence=corr.confidence,
                    writer=writer,
                    episode_id=eid,
                    now=at,
                    tenant=ns,
                )
                created.append(replacement.memory_id)
                closed.append(old_id)
                pairs.append((old_id, replacement.memory_id))

            for rel in self.remove_relations:
                s = db.get_entity(resolve(rel.src), tenant=ns)
                d = db.get_entity(resolve(rel.dst), tenant=ns)
                if s is None or d is None:
                    continue
                ids = current_relation_ids(
                    db, s.entity_id, d.entity_id, rel_kind=rel.rel_kind, tenant=ns
                )
                if not ids:
                    continue
                db.unrelate(s.entity_id, d.entity_id, rel_kind=rel.rel_kind, now=at, tenant=ns)
                closed_edges.extend(ids)

            for rel in self.add_relations:
                edge = db.relate(
                    resolve(rel.src),
                    resolve(rel.dst),
                    rel_kind=rel.rel_kind,
                    writer=writer,
                    episode_id=eid,
                    now=at,
                    tenant=ns,
                )
                opened.append(edge.edge_id)

        return PatchReceipt(
            episode_id=eid,
            tenant_id=ns.tenant_id,
            writer=writer,
            at=at,
            patch=self,
            memories_created=tuple(created),
            memories_closed=tuple(closed),
            corrections=tuple(pairs),
            relations_opened=tuple(opened),
            relations_closed=tuple(closed_edges),
            aliases=tuple(aliases),
        )

    @staticmethod
    def _target_of(db: Anatid, corr: Correction, ns: Any) -> int:
        if corr.old_id is not None:
            return int(corr.old_id)
        assert corr.old_text is not None
        matches = find_current_by_text(db, corr.old_text, tenant=ns)
        if not matches:
            raise NotFoundError(
                f"correction to {corr.new_content!r}: no current memory reads "
                f"{corr.old_text!r} in tenant {ns.tenant_id}"
            )
        if len(matches) > 1:
            raise ValidationError(
                f"correction to {corr.new_content!r}: {len(matches)} current memories read "
                f"{corr.old_text!r} ({', '.join(str(m) for m in matches)}); name one by old_id"
            )
        return matches[0]


# --------------------------------------------------------------------------- parsing helpers


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _entries(data: Mapping[str, Any], *keys: str, notes: list[str]) -> list[Mapping[str, Any]]:
    """The list of object entries under the first of ``keys`` that is present."""
    for key in keys:
        if key in data and data[key] is not None:
            raw = data[key]
            if isinstance(raw, Mapping):
                raw = [raw]
            if not isinstance(raw, (list, tuple)):
                notes.append(f"parser: {key!r} is not a list; ignored")
                return []
            out: list[Mapping[str, Any]] = []
            for item in raw:
                if isinstance(item, Mapping):
                    out.append(item)
                else:
                    notes.append(f"parser: skipped a non-object entry in {key!r}: {item!r}")
            return out
    return []


#: JSON Schema for the model-facing patch shape.  It is what :meth:`MemoryPatch.from_dict` reads
#: and what :class:`anatid.ingest.extract.OpenAICompatibleExtractor` puts in its prompt.  Ids
#: are strings and the schema says why.  ``evidence`` is the quoted span of the note that
#: supports an entry; the parser locates it in the text.
PATCH_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "add_facts": {
            "type": "array",
            "description": "New durable facts the note states, one standalone sentence each.",
            "items": {
                "type": "object",
                "required": ["content", "entities"],
                "additionalProperties": False,
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The fact as a declarative sentence that names its subjects.",
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Every entity the fact is about, by name.",
                    },
                    "kind": {
                        "type": "string",
                        "description": 'A short category: "fact", "constraint", "decision", "preference".',
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {
                        "type": "string",
                        "description": "The shortest quote from the note that states this.",
                    },
                },
            },
        },
        "corrections": {
            "type": "array",
            "description": (
                "Existing facts the note changes. Use these instead of adding a contradicting fact."
            ),
            "items": {
                "type": "object",
                "required": ["new_content"],
                "additionalProperties": False,
                "properties": {
                    "old_memory_id": {
                        "type": "string",
                        "description": (
                            "The id of the existing memory as given, a decimal string such as "
                            '"883768514279557120". Ids are strings because JSON numbers lose '
                            "precision above 2**53."
                        ),
                    },
                    "old_text": {
                        "type": "string",
                        "description": "The existing fact's content, when its id is not known.",
                    },
                    "new_content": {"type": "string"},
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Entities of the corrected fact; omit to keep the old ones.",
                    },
                    "kind": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string"},
                },
            },
        },
        "add_relations": {
            "type": "array",
            "description": "Relationships between two entities the note establishes.",
            "items": {
                "type": "object",
                "required": ["src", "dst", "rel_kind"],
                "additionalProperties": False,
                "properties": {
                    "src": {"type": "string"},
                    "dst": {"type": "string"},
                    "rel_kind": {
                        "type": "string",
                        "description": "A short verb: leads, maintains, owns, depends_on, member_of.",
                    },
                    "evidence": {"type": "string"},
                },
            },
        },
        "remove_relations": {
            "type": "array",
            "description": "Relationships the note says no longer hold.",
            "items": {
                "type": "object",
                "required": ["src", "dst"],
                "additionalProperties": False,
                "properties": {
                    "src": {"type": "string"},
                    "dst": {"type": "string"},
                    "rel_kind": {"type": "string"},
                    "evidence": {"type": "string"},
                },
            },
        },
        "entity_aliases": {
            "type": "array",
            "description": "A name the note uses for an entity that already exists under another name.",
            "items": {
                "type": "object",
                "required": ["name", "canonical"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "canonical": {"type": "string"},
                    "evidence": {"type": "string"},
                },
            },
        },
    },
}
