"""The memory systems the answer-quality benchmark compares, and what they share.

Every system in this package answers the same call: it is built once from the corpus notes in
arrival order, then asked for a memory context for one question under one token budget.  The
harness wraps that context in the same prompt for every system, so what differs between the
columns of the report is only what each system chose to put in front of the model.

The pieces that every system must share, so the comparison is a comparison, live here:

* :func:`count_tokens` is the one token counter.  It is tiktoken's ``cl100k_base`` when the
  package is importable and its vocabulary loads, otherwise a documented estimate of four
  characters per token.  :func:`tokenizer_name` says which one a run used; the harness records
  it next to every context.
* :func:`render_note` is the one rendering of a note: a Markdown bullet holding the note's
  ``rendered`` string, ``- [date] source: text``.  The date and the source label are in the
  string because a system that drops them cannot answer temporal or provenance questions, and
  a system that gets more than that is not being compared fairly.
* :func:`pack` is the one packing rule.  A system hands over its ranking; ``pack`` walks it in
  rank order, keeps every hit whose line fits in what is left of the budget, and renders the
  kept hits oldest first so the model sees a handover after the one it replaced.  The anatid
  systems should call it with their own hits so the rule is identical there too.

Concrete systems are one module each: :mod:`.markdown` (S1), :mod:`.bm25` (S2) and
:mod:`.vector` (S3).  Nothing here is imported by ``import anatid``.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from bench.quality import NOTES_PATH

__all__ = [
    "DEFAULT_BUDGET",
    "Hit",
    "Note",
    "Retrieval",
    "System",
    "count_tokens",
    "load_notes",
    "pack",
    "render_note",
    "set_token_counter",
    "tokenizer_name",
]

#: The memory-context budget every system answers under, in tokens of :func:`count_tokens`.
DEFAULT_BUDGET = 1200


# ------------------------------------------------------------------------------------ notes


@dataclass(frozen=True)
class Note:
    """One line of ``data/notes.jsonl``.  ``rendered`` is what every system receives."""

    note_id: str
    seq: int
    date: str
    source: str
    source_kind: str
    team: str
    author: str
    text: str
    rendered: str
    event_ids: tuple[int, ...] = ()
    event_kinds: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Note:
        return cls(
            note_id=str(row["note_id"]),
            seq=int(row["seq"]),
            date=str(row["date"]),
            source=str(row["source"]),
            source_kind=str(row.get("source_kind", "")),
            team=str(row.get("team", "")),
            author=str(row.get("author", "")),
            text=str(row["text"]),
            rendered=str(row["rendered"]),
            event_ids=tuple(int(x) for x in row.get("event_ids", ())),
            event_kinds=tuple(str(x) for x in row.get("event_kinds", ())),
        )

    @property
    def when(self) -> _dt.datetime:
        """The note's date as a naive midnight, the clock anatid's verbs take for ``now=``."""
        return _dt.datetime.fromisoformat(self.date)


def load_notes(path: Path | str = NOTES_PATH) -> list[Note]:
    """Read ``notes.jsonl`` in file order, which is arrival order."""
    notes: list[Note] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                notes.append(Note.from_dict(json.loads(line)))
    return notes


def render_note(note: Note) -> str:
    """The dated Markdown bullet every system shows the model for one note."""
    return f"- {note.rendered}"


# ----------------------------------------------------------------------------- token counting

#: Characters per token when tiktoken is not available.  Applied identically to every system.
CHARS_PER_TOKEN = 4

_ENCODER: Any = None
_TOKENIZER_NAME: str | None = None
_COUNT: Callable[[str], int] | None = None


def _resolve_tokenizer() -> None:
    global _ENCODER, _TOKENIZER_NAME
    if _TOKENIZER_NAME is not None:
        return
    try:
        import tiktoken  # type: ignore[import-not-found]  # optional; resolved once

        _ENCODER = tiktoken.get_encoding("cl100k_base")
        _TOKENIZER_NAME = "tiktoken/cl100k_base"
    except Exception:  # noqa: BLE001 - any failure (no package, no vocabulary) means the estimate
        _ENCODER = None
        _TOKENIZER_NAME = f"estimate/{CHARS_PER_TOKEN}-chars-per-token"


def set_token_counter(count: Callable[[str], int], *, name: str) -> None:
    """Install the harness's counter so the systems and the harness count identically.

    Call it before any system is built.  ``name`` is what :func:`tokenizer_name` reports from
    then on.  Without it the package resolves its own counter, by the same rule.
    """
    global _ENCODER, _TOKENIZER_NAME, _COUNT
    _COUNT = count
    _ENCODER = None
    _TOKENIZER_NAME = name


def tokenizer_name() -> str:
    """Which counter :func:`count_tokens` uses in this process.  Record it with the results."""
    _resolve_tokenizer()
    assert _TOKENIZER_NAME is not None
    return _TOKENIZER_NAME


def count_tokens(text: str) -> int:
    """Tokens in ``text`` under the run's one counter.  The empty string counts zero."""
    _resolve_tokenizer()
    if not text:
        return 0
    if _COUNT is not None:
        return int(_COUNT(text))
    if _ENCODER is not None:
        return len(_ENCODER.encode(text, disallowed_special=()))
    return math.ceil(len(text) / CHARS_PER_TOKEN)


# ----------------------------------------------------------------------------------- results


@dataclass(frozen=True)
class Hit:
    """One ranked candidate for the context.

    ``ref`` names the thing retrieved (a note id for S1 to S3, a memory id for anatid),
    ``rank`` is its 1-based position in the system's ranking, ``score`` the system's own
    number for it (``None`` when the ranking has no score, as for recency), ``date`` and
    ``seq`` order it chronologically, and ``line`` is exactly what the model sees for it.
    """

    ref: str
    rank: int
    score: float | None
    date: str
    seq: int
    line: str

    @classmethod
    def of_note(cls, note: Note, *, rank: int, score: float | None) -> Hit:
        return cls(
            ref=note.note_id,
            rank=rank,
            score=score,
            date=note.date,
            seq=note.seq,
            line=render_note(note),
        )


@dataclass
class Retrieval:
    """What a system hands the harness for one question: the context and how it was chosen.

    ``text`` is the memory block for the prompt (empty when nothing was retrieved), ``hits``
    the items in the order they appear in it, ``tokens`` the block's size under
    :func:`count_tokens`, ``budget`` the limit it was packed to (``None`` for unbudgeted),
    ``considered`` how many candidates the ranking held, ``skipped`` how many of them did not
    fit, ``seconds`` the wall time of the retrieval, and ``meta`` whatever else the system
    wants on record (its parameters, cache statistics).
    """

    system: str
    question: str
    budget: int | None
    text: str
    hits: list[Hit] = field(default_factory=list)
    tokens: int = 0
    considered: int = 0
    skipped: int = 0
    seconds: float = 0.0
    tokenizer: str = field(default_factory=tokenizer_name)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def refs(self) -> list[str]:
        """The ``ref`` of every hit in context order; what retrieval recall is computed on."""
        return [hit.ref for hit in self.hits]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _chronological(hits: Iterable[Hit]) -> list[Hit]:
    return sorted(hits, key=lambda hit: (hit.date, hit.seq, hit.rank))


def pack(
    ranked: Sequence[Hit],
    *,
    budget: int | None,
    order: str = "chronological",
) -> tuple[str, list[Hit], int]:
    """Choose the hits that fit ``budget`` and render them as one block.

    The walk is in rank order.  A hit is kept when its line, plus the newline that joins it to
    the block, fits in what is left of the budget; otherwise it is skipped and the walk goes
    on, so a long note low in the ranking does not shut out a short one below it.  The kept
    hits are then laid out oldest first (``order="chronological"``) or as ranked
    (``order="rank"``).  Because a tokenizer can merge or split at line joins, the finished
    block is counted once more and the lowest-ranked kept hit is dropped until it fits.

    ``budget=None`` keeps everything.  Returns ``(text, kept_hits_in_context_order, skipped)``.
    """
    if order not in ("chronological", "rank"):
        raise ValueError(f"order must be 'chronological' or 'rank', got {order!r}")
    kept: list[Hit] = []
    skipped = 0
    if budget is None:
        kept = list(ranked)
    else:
        remaining = int(budget)
        for hit in ranked:
            cost = count_tokens(hit.line + "\n")
            if cost <= remaining:
                kept.append(hit)
                remaining -= cost
            else:
                skipped += 1

    def render(items: Sequence[Hit]) -> tuple[str, list[Hit]]:
        laid_out = _chronological(items) if order == "chronological" else list(items)
        return "\n".join(hit.line for hit in laid_out), laid_out

    text, laid_out = render(kept)
    if budget is not None:
        while kept and count_tokens(text) > budget:
            kept.pop()  # `kept` is in rank order, so this drops the weakest hit
            skipped += 1
            text, laid_out = render(kept)
    return text, laid_out, skipped


# ---------------------------------------------------------------------------------- protocol


@runtime_checkable
class System(Protocol):
    """What the harness asks of a memory system.

    ``build`` receives the corpus notes in arrival order, once.  ``retrieve`` returns the
    memory context for one question under ``budget`` tokens (``None`` for no budget).  The
    question string is all a system gets: no category, no gold, no ``as_of``.  ``close``
    releases whatever the system holds.
    """

    name: str

    def build(self, notes: Sequence[Note]) -> None: ...

    def retrieve(self, question: str, *, budget: int | None = DEFAULT_BUDGET) -> Retrieval: ...

    def close(self) -> None: ...
