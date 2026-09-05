"""S2+S3 and S3 with feedback: the two fused baselines over the raw notes.

anatid fuses three retrieval arms over its memories with Reciprocal Rank Fusion (RRF).  A
comparison against single-arm baselines alone cannot say whether the fusion or the memory made
the difference, so this module gives the raw notes the same fusion:

* :class:`HybridSystem` ranks the notes with :class:`~bench.quality.systems.bm25.Bm25System`
  and with :class:`~bench.quality.systems.vector.VectorSystem`, then fuses the two rankings with
  :func:`fuse_hits` (``score = sum 1 / (k + rank)``, k = 60, the constant anatid uses); ties
  keep the cosine order.
* :class:`VectorPrfSystem` is the vector baseline with one round of pseudo-relevance feedback:
  the text of the top :data:`PRF_FEEDBACK_NOTES` notes is appended to the question, the notes
  are ranked once more against that, and the two rankings are fused; ties keep the first
  ranking.  A second retrieval round is the usual cheap way to reach a note that shares no
  words with the question, which is what a multi-hop question needs.

Neither system is tuned.  The kept hits are shown oldest first, as every system here shows
them.  ``bench.quality.harness`` carries the same two systems in its built-in family.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from anatid.embed import Embedder
from bench.quality.systems import (
    DEFAULT_BUDGET,
    Hit,
    Note,
    Retrieval,
    count_tokens,
    pack,
)
from bench.quality.systems.bm25 import Bm25System
from bench.quality.systems.vector import VectorSystem

__all__ = [
    "FUSION_DECIMALS",
    "FUSION_K",
    "PRF_FEEDBACK_NOTES",
    "HybridSystem",
    "VectorPrfSystem",
    "fuse_hits",
]

#: Reciprocal Rank Fusion constant; anatid's own is :data:`anatid.recall.RRF_K`.
FUSION_K = 60
#: Fused scores are compared at this many decimals; the rest is a tie.
FUSION_DECIMALS = 9
#: Notes of the first ranking whose text is appended to the question for the second.
PRF_FEEDBACK_NOTES = 3


def fuse_hits(
    rankings: Sequence[Sequence[Hit]],
    *,
    k: int = FUSION_K,
    tie_break: Sequence[Hit] | None = None,
) -> list[Hit]:
    """Reciprocal Rank Fusion of several rankings of the same notes, by ``ref``.

    A note's fused score is ``sum 1 / (k + rank)`` over the rankings that hold it, ranks
    1-based.  Scores are compared at :data:`FUSION_DECIMALS`; ties keep the order of
    ``tie_break`` (the first ranking when none is given), with notes absent from it last.  The
    returned hits carry the fused score and their fused rank.
    """
    score: dict[str, float] = {}
    first_seen: dict[str, Hit] = {}
    for ranking in rankings:
        for hit in ranking:
            score[hit.ref] = score.get(hit.ref, 0.0) + 1.0 / (k + hit.rank)
            first_seen.setdefault(hit.ref, hit)
    basis = tie_break if tie_break is not None else (rankings[0] if rankings else ())
    order = {hit.ref: i for i, hit in enumerate(basis)}
    refs = sorted(
        score, key=lambda ref: (-round(score[ref], FUSION_DECIMALS), order.get(ref, len(order)))
    )
    return [
        replace(first_seen[ref], rank=position, score=score[ref])
        for position, ref in enumerate(refs, start=1)
    ]


class HybridSystem:
    """BM25 and cosine over the raw notes, fused with RRF.  See the module docstring."""

    name: str

    def __init__(
        self,
        embedder: Embedder,
        *,
        k: int = FUSION_K,
        order: str = "chronological",
        name: str | None = None,
    ) -> None:
        self.bm25 = Bm25System(order=order)
        self.vector = VectorSystem(embedder, order=order)
        self.k = int(k)
        self.order = order
        self.name = name or "hybrid"
        self.build_seconds = 0.0

    def __repr__(self) -> str:
        return f"HybridSystem(name={self.name!r}, k={self.k}, bm25={self.bm25!r}, vector={self.vector!r})"

    def build(self, notes: Sequence[Note]) -> None:
        started = time.perf_counter()
        self.bm25.build(notes)
        self.vector.build(notes)
        self.build_seconds = time.perf_counter() - started

    def rank(self, question: str) -> list[Hit]:
        """Every note with a BM25 or a cosine rank, in fused order; ties keep the cosine order."""
        by_text = self.bm25.rank(question)
        by_vector = self.vector.rank(question)
        return fuse_hits([by_text, by_vector], k=self.k, tie_break=by_vector)

    def retrieve(self, question: str, *, budget: int | None = DEFAULT_BUDGET) -> Retrieval:
        started = time.perf_counter()
        ranked = self.rank(question)
        text, kept, skipped = pack(ranked, budget=budget, order=self.order)
        meta: dict[str, Any] = {
            "fusion": f"rrf k={self.k}",
            "arms": ["bm25", "vector"],
            "tie_break": "cosine rank",
            "order": self.order,
            "build_seconds": self.build_seconds,
        }
        return Retrieval(
            system=self.name,
            question=question,
            budget=budget,
            text=text,
            hits=kept,
            tokens=count_tokens(text),
            considered=len(ranked),
            skipped=skipped,
            seconds=time.perf_counter() - started,
            meta=meta,
        )

    def close(self) -> None:
        self.bm25.close()
        self.vector.close()


class VectorPrfSystem(VectorSystem):
    """The vector baseline with one round of pseudo-relevance feedback.  See the module
    docstring."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        feedback: int = PRF_FEEDBACK_NOTES,
        k: int = FUSION_K,
        order: str = "chronological",
        name: str | None = None,
    ) -> None:
        super().__init__(embedder, order=order, name=name or "vector-prf")
        self.feedback = int(feedback)
        self.k = int(k)

    def rank(self, question: str) -> list[Hit]:
        first = super().rank(question)
        by_id = {note.note_id: note for note in self._notes}
        feedback = " ".join(by_id[hit.ref].rendered for hit in first[: self.feedback])
        second = super().rank(question + "\n" + feedback)
        return fuse_hits([first, second], k=self.k, tie_break=first)

    def retrieve(self, question: str, *, budget: int | None = DEFAULT_BUDGET) -> Retrieval:
        result = super().retrieve(question, budget=budget)
        result.meta.update(
            {"feedback_notes": self.feedback, "fusion": f"rrf k={self.k}", "rounds": 2}
        )
        return result
