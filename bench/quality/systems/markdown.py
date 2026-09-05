"""S1: one Markdown file, every note appended in arrival order, the most recent notes that fit.

This is the memory system most agents actually have: a notes file that grows at the bottom.
The prompt receives the tail of the file, because that is what fits.  Nothing is indexed and
nothing is searched, so the system is right whenever the answer is recent and wrong whenever
it is not, and how often each happens is what the benchmark measures.

Two variants share the code:

* ``MarkdownSystem()`` packs the budget with the newest notes.  The tail is contiguous: it
  walks from the newest note backwards and stops at the first note that does not fit, which
  is what truncating the file from the top gives.  The kept notes are shown oldest first, in
  file order.
* ``MarkdownSystem(whole_file=True)`` hands over the entire file whatever the budget, as the
  upper bound of "just put it in the prompt".  Its retrievals record ``budget=None``.

:meth:`MarkdownSystem.write` puts the file on disk so a run directory holds the actual
artefact the system answered from.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from bench.quality.systems import (
    DEFAULT_BUDGET,
    Hit,
    Note,
    Retrieval,
    count_tokens,
)

__all__ = ["MarkdownSystem"]

#: The heading at the top of the file.  It is part of the file, not of the context: the
#: prompt receives bullets only, the same shape every other system produces.
HEADING = "# Notes\n\n"


class MarkdownSystem:
    """Recency-ordered Markdown notes.  See the module docstring."""

    def __init__(self, *, whole_file: bool = False, name: str | None = None) -> None:
        self.whole_file = bool(whole_file)
        self.name = name or ("markdown-full" if self.whole_file else "markdown")
        self._notes: list[Note] = []
        self._lines: list[str] = []
        self.build_seconds = 0.0

    def __repr__(self) -> str:
        return f"MarkdownSystem(name={self.name!r}, whole_file={self.whole_file}, notes={len(self._notes)})"

    # ------------------------------------------------------------------------------- build

    def build(self, notes: Sequence[Note]) -> None:
        started = time.perf_counter()
        self._notes = list(notes)
        self._lines = [f"- {note.rendered}" for note in self._notes]
        self.build_seconds = time.perf_counter() - started

    @property
    def document(self) -> str:
        """The whole file: a heading, then one bullet per note in arrival order."""
        return HEADING + "\n".join(self._lines) + ("\n" if self._lines else "")

    def write(self, path: Path | str) -> Path:
        """Write the file to ``path`` and return it."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.document, encoding="utf-8")
        return target

    # ---------------------------------------------------------------------------- retrieve

    def retrieve(self, question: str, *, budget: int | None = DEFAULT_BUDGET) -> Retrieval:
        started = time.perf_counter()
        total = len(self._notes)
        # Rank 1 is the newest note: recency is the ranking this system has.
        ranked = [
            Hit.of_note(note, rank=total - index, score=None)
            for index, note in enumerate(self._notes)
        ]
        effective_budget = None if self.whole_file else budget
        if effective_budget is None:
            kept = ranked
        else:
            remaining = int(effective_budget)
            start = total
            # Walk backwards from the newest note; stop at the first that does not fit.
            while start > 0:
                cost = count_tokens(ranked[start - 1].line + "\n")
                if cost > remaining:
                    break
                remaining -= cost
                start -= 1
            kept = ranked[start:]
            text = "\n".join(hit.line for hit in kept)
            # Line joins can merge tokens; make the block itself honour the budget.
            while kept and count_tokens(text) > effective_budget:
                kept = kept[1:]
                text = "\n".join(hit.line for hit in kept)
        text = "\n".join(hit.line for hit in kept)
        return Retrieval(
            system=self.name,
            question=question,
            budget=effective_budget,
            text=text,
            hits=list(kept),
            tokens=count_tokens(text),
            considered=total,
            skipped=total - len(kept),
            seconds=time.perf_counter() - started,
            meta={
                "whole_file": self.whole_file,
                "order": "arrival",
                "selection": "whole file" if self.whole_file else "contiguous tail",
                "file_tokens": count_tokens("\n".join(self._lines)),
                "build_seconds": self.build_seconds,
            },
        )

    def close(self) -> None:
        """Nothing to release."""
