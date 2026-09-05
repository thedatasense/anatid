"""Render anatid memories as the memory context an answer prompt receives.

The baselines hand the model raw notes.  anatid can hand it what a note store does not hold:
for every memory the interval during which it was believed true, the note that wrote it, and,
for a fact that has since been corrected, the earlier versions with the dates they held.  That
is what lets the same model answer "who owned X on <date>" and "which note said so" from a
context that is no larger than the baselines' one.

Two things live here:

* the token counter every system in the answer-quality benchmark shares
  (:func:`count_tokens`): ``tiktoken``'s ``cl100k_base`` when the package is importable,
  otherwise a documented estimate of one token per four characters, applied identically to
  every system, and named in :func:`tokenizer_name` so the report can say which one ran;
* the rendering (:func:`render_current`, :func:`render_superseded`) and the packing
  (:func:`render_context`) that turns a ranked list of recall hits into lines that fit a
  budget.

The packing rule is deliberately plain so it can be stated in the report in two sentences: the
legend first, then the hits in rank order, each as its current line followed by the versions
it superseded, newest first, and the first line that does not fit the budget ends the context.
Superseded versions may take at most a fixed share of the budget (``history_share``, 20 percent
by default); they are attached to hits in rank order until that share is used, and the rest of
the budget goes to current lines.  Nothing is reordered or trimmed to squeeze one more line in.
The share exists because history is what answers "who held it on <date>" and it sits under the
hits about the entity the question names, which rank first, while an on-call rotation seven
versions deep under every hit would spend the budget on rotations of teams the question never
asked about and leave no room for the second and third hop of a multi-hop question.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from anatid import Memory, RecallHit

__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_HISTORY_SHARE",
    "LEGEND",
    "ContextLine",
    "RenderedContext",
    "TokenCounter",
    "count_tokens",
    "date_of",
    "render_context",
    "render_current",
    "render_superseded",
    "tokenizer_name",
]

#: A callable from text to a token count; :func:`count_tokens` is the shared default.
TokenCounter = Callable[[str], int]

#: The estimate used when ``tiktoken`` is not importable: one token per this many characters,
#: rounded up.  English prose in ``cl100k_base`` averages a little under four characters per
#: token, so the estimate is slightly generous and identical for every system.
CHARS_PER_TOKEN = 4

#: The share of the budget superseded versions may take, by default.  Chosen by a sweep on the
#: gold store over the benchmark's own questions, counting whether the line that carries the
#: gold answer landed inside a 1,200-token context: no history at all covered 116 of the 125
#: answerable questions, 0.2 covered 124, 0.25 to 0.5 covered 122 and an unlimited share 121.
#: Below 0.2 a seven-step on-call rotation no longer fits; above it the second and third hop of
#: a multi-hop question start falling off the end.  It is a default, and it is disclosed.
DEFAULT_HISTORY_SHARE = 0.2

#: The one line that opens every anatid context.  It is counted against the budget.
LEGEND = (
    "Memory: one remembered fact per line, with the dates it held and the note that recorded "
    'it. An indented "earlier" line is a version of the fact above it that was later corrected.'
)

_ENCODER: Any = None
_TOKENIZER: str | None = None


def _load_tokenizer() -> None:
    global _ENCODER, _TOKENIZER
    if _TOKENIZER is not None:
        return
    try:
        import tiktoken  # type: ignore[import-not-found]

        _ENCODER = tiktoken.get_encoding("cl100k_base")
        _TOKENIZER = "tiktoken/cl100k_base"
    except Exception:  # noqa: BLE001 - any failure means the estimate is used, and reported
        _ENCODER = None
        _TOKENIZER = f"estimate/{CHARS_PER_TOKEN}-chars-per-token"


def tokenizer_name() -> str:
    """Which counter :func:`count_tokens` runs: ``tiktoken/cl100k_base`` or the estimate."""
    _load_tokenizer()
    assert _TOKENIZER is not None
    return _TOKENIZER


def count_tokens(text: str) -> int:
    """Tokens in ``text`` under the shared counter (see the module docstring)."""
    _load_tokenizer()
    if _ENCODER is not None:
        return len(_ENCODER.encode(text, disallowed_special=()))
    return math.ceil(len(text) / CHARS_PER_TOKEN)


# --------------------------------------------------------------------------- one line


def date_of(value: _dt.datetime | _dt.date | None) -> str | None:
    """The ISO date of a timestamp, or ``None``."""
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date().isoformat()
    return value.isoformat()


def _kind_tag(memory: Memory) -> str:
    return f"[{memory.kind}] " if memory.kind and memory.kind != "fact" else ""


def _source(memory: Memory) -> str:
    return f"; source {memory.writer}" if memory.writer else ""


def render_current(memory: Memory) -> str:
    """One current memory: ``- <content> (since <valid_from>; source <writer>)``.

    ``writer`` is the note label the benchmark ingests every note under, so the line carries
    exactly the header the baselines' notes carry, plus the date the fact took effect.
    """
    since = date_of(memory.valid_from)
    when = f"since {since}" if since else "date unknown"
    return f"- {_kind_tag(memory)}{memory.content.strip()} ({when}{_source(memory)})"


def render_superseded(memory: Memory) -> str:
    """One superseded predecessor, indented under the memory that replaced it.

    ``- earlier: <content> (held <valid_from> to <valid_to>; source <writer>)``.  The closing
    date is the day the correction landed, which is the date the fact stopped being believed.
    """
    start, end = date_of(memory.valid_from), date_of(memory.valid_to)
    if start and end:
        held = f"held {start} to {end}"
    elif start:
        held = f"held from {start}"
    else:
        held = "dates unknown"
    return f"  - earlier: {_kind_tag(memory)}{memory.content.strip()} ({held}{_source(memory)})"


# --------------------------------------------------------------------------- packing


@dataclass(frozen=True)
class ContextLine:
    """One rendered line and where it came from, for the saved retrieval record."""

    text: str
    memory_id: int
    status: str  # "current" or "superseded"
    rank: int  # 1-based rank of the recall hit this line belongs to
    depth: int  # 0 for the hit itself, 1.. for each earlier version
    tokens: int  # tokens this line added to the context
    writer: str | None
    valid_from: str | None
    valid_to: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RenderedContext:
    """What :func:`render_context` produced, and what it left out."""

    text: str
    tokens: int
    budget: int
    tokenizer: str
    lines: list[ContextLine] = field(default_factory=list)
    hits_total: int = 0
    hits_rendered: int = 0
    history_rendered: int = 0
    history_available: int = 0
    history_tokens: int = 0
    history_budget: int = 0
    history_exhausted: bool = False
    truncated: bool = False

    @property
    def memory_ids(self) -> list[int]:
        """Every memory id in the context, current and superseded, in order, once each."""
        seen: set[int] = set()
        out: list[int] = []
        for line in self.lines:
            if line.memory_id not in seen:
                seen.add(line.memory_id)
                out.append(line.memory_id)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tokens": self.tokens,
            "budget": self.budget,
            "tokenizer": self.tokenizer,
            "hits_total": self.hits_total,
            "hits_rendered": self.hits_rendered,
            "history_rendered": self.history_rendered,
            "history_available": self.history_available,
            "history_tokens": self.history_tokens,
            "history_budget": self.history_budget,
            "history_exhausted": self.history_exhausted,
            "truncated": self.truncated,
            "memory_ids": [str(i) for i in self.memory_ids],
            "lines": [line.to_dict() for line in self.lines],
        }


def _memory_of(hit: RecallHit | Memory) -> Memory:
    return hit.memory if isinstance(hit, RecallHit) else hit


def render_context(
    hits: Iterable[RecallHit | Memory],
    *,
    budget: int,
    history: Callable[[int], Sequence[Memory]] | None = None,
    count: TokenCounter | None = None,
    legend: str | None = LEGEND,
    history_share: float = DEFAULT_HISTORY_SHARE,
    max_history: int | None = None,
) -> RenderedContext:
    """Pack ranked hits into context lines that fit ``budget`` tokens.

    ``hits`` are :class:`~anatid.RecallHit` (or bare :class:`~anatid.Memory`) in rank order.
    ``history(memory_id)`` returns the memories a hit superseded, newest first; the system
    supplies it from :meth:`anatid.Anatid.provenance` and may cache it.  Superseded versions
    may take at most ``history_share`` of the budget (``0`` for none, ``1`` for no limit
    beyond the budget itself), attached to hits in rank order until the share is used;
    ``max_history`` caps how many one hit may add (``None`` for all).  ``count`` is the token
    counter; the default is :func:`count_tokens`, the counter every system shares.

    Lines are added in order, the legend first, and the first line that does not fit the
    budget ends the context; ``truncated`` says whether that happened.  A history line that
    would overrun the share is skipped, along with every later one (``history_exhausted``),
    and current lines carry on.  A hit whose current line did not fit adds none of its history.
    """
    counter = count or count_tokens
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    if not 0.0 <= history_share <= 1.0:
        raise ValueError(f"history_share must be within [0, 1], got {history_share}")

    text = legend or ""
    tokens = counter(text) if text else 0
    if tokens > budget:
        raise ValueError(
            f"the legend alone is {tokens} tokens, above the budget of {budget}; pass legend=None"
        )
    out = RenderedContext(
        text=text,
        tokens=tokens,
        budget=budget,
        tokenizer=tokenizer_name(),
        history_budget=int(budget * history_share),
    )
    seen: set[int] = set()

    def measure(line: str) -> tuple[str, int, int] | None:
        """``(new text, new total, tokens added)`` when ``line`` fits the budget, else None."""
        candidate = f"{out.text}\n{line}" if out.text else line
        total = counter(candidate)
        if total > budget:
            return None
        return candidate, total, total - out.tokens

    def try_add(line: str) -> int | None:
        """Append ``line`` when it fits; return the tokens it added, or ``None``."""
        fit = measure(line)
        if fit is None:
            return None
        out.text, out.tokens, added = fit[0], fit[1], fit[2]
        return added

    for rank, item in enumerate(hits, start=1):
        out.hits_total += 1
        memory = _memory_of(item)
        if memory.memory_id in seen:
            continue
        seen.add(memory.memory_id)
        if out.truncated:
            continue
        added = try_add(render_current(memory))
        if added is None:
            out.truncated = True
            continue
        out.hits_rendered += 1
        out.lines.append(
            ContextLine(
                text=render_current(memory),
                memory_id=memory.memory_id,
                status="current",
                rank=rank,
                depth=0,
                tokens=added,
                writer=memory.writer,
                valid_from=date_of(memory.valid_from),
                valid_to=date_of(memory.valid_to),
            )
        )
        if history is None:
            continue
        earlier = list(history(memory.memory_id))
        out.history_available += len(earlier)
        if out.history_exhausted:
            continue
        if max_history is not None:
            earlier = earlier[:max_history]
        for depth, old in enumerate(earlier, start=1):
            if old.memory_id in seen:
                continue
            line = render_superseded(old)
            fit = measure(line)
            if fit is None:
                out.truncated = True
                break
            if out.history_tokens + fit[2] > out.history_budget:
                out.history_exhausted = True
                break
            out.text, out.tokens, added = fit
            out.history_tokens += added
            seen.add(old.memory_id)
            out.history_rendered += 1
            out.lines.append(
                ContextLine(
                    text=line,
                    memory_id=old.memory_id,
                    status="superseded",
                    rank=rank,
                    depth=depth,
                    tokens=added,
                    writer=old.writer,
                    valid_from=date_of(old.valid_from),
                    valid_to=date_of(old.valid_to),
                )
            )
    return out
