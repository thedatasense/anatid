"""The memory systems, one prompt, one budget, one runner.

Every system here is asked the same 150 questions with the same model, the same prompt
template and the same memory budget.  The only thing a system decides is *which* text goes
into the prompt.  Everything that could favour one system over another is outside the
system's control:

* :func:`fit_budget` is the one budget cut.  A system returns its ranked candidates
  (:meth:`BaseSystem.candidates`) and the harness keeps the longest prefix whose token count
  fits.  Tokens are counted by one :class:`~bench.quality.llm.TokenCounter` for every system.
* :func:`build_messages` is the one prompt.  It tells the model to answer from the memory
  alone and to say exactly ``I don't know`` when the memory does not hold the answer.
* :func:`run_system` ingests once, asks every question in order, and saves every prompt,
  every retrieved context and every answer to ``results/<run_id>/<system>/``.

Two families of systems plug in.  The built-in ones below are self-contained and are what the
tests exercise.  The ``bench.quality.systems`` package, when present, provides the same systems
through its own ``build`` / ``retrieve`` protocol, and :class:`ExternalSystem` adapts them so
the runner, the prompt, the token counter and the budget are the same either way.  A run never
mixes the two families: :func:`default_systems` picks one and says which.

The systems:

``markdown``
    Every note appended to one Markdown file in arrival order; the prompt gets the most recent
    notes that fit.  ``markdown-full`` is the same file with no budget, the upper bound of
    "just put everything in the prompt" at this corpus size.
``bm25``
    DuckDB's ``fts`` extension over the raw notes; the top notes by BM25.
``vector``
    ``text-embedding-3-small`` over the raw notes; the top notes by cosine similarity.
``hybrid``
    The BM25 and the cosine rankings of the raw notes fused with Reciprocal Rank Fusion, the
    fusion anatid itself uses.  It is here so the comparison can tell fusion from memory: a
    system that fuses arms should be compared with a baseline that fuses arms.
``vector-prf``
    The vector baseline with one round of pseudo-relevance feedback: the top notes of the
    first ranking are appended to the question and the notes are ranked once more.  A second
    retrieval round is the cheap remedy for a question whose answer shares no words with it,
    which is what the multi-hop questions are.
``anatid``
    Notes ingested through :func:`anatid.ingest.ingest` with the same chat model as the
    extractor and the same embedder on the handle; ``recall(question)`` with every arm; each
    memory rendered with its validity interval, its writer and the versions it superseded.
    ``anatid-text``, ``anatid-vector`` and ``anatid-graph`` query the same store with one arm.
``anatid-gold``
    The same store built from the corpus's gold patches (:class:`anatid.ingest.ScriptedExtractor`)
    instead of the model's.  It is an oracle for extraction, so the S4 to S5 gap is the cost of
    extraction and the S5 number is the ceiling of anatid's retrieval on this corpus.  It is
    not the product.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import itertools
import json
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from . import GOLD_PATCHES_PATH, NOTES_PATH, QUESTIONS_PATH
from .llm import LLMClient, TokenCounter

__all__ = [
    "ANSWER_MAX_TOKENS",
    "ANSWER_MAX_TOKENS_RETRY",
    "BUDGET_TOKENS",
    "IDK",
    "NOW",
    "RECALL_K",
    "RESULTS_DIR",
    "SYSTEM_NAMES",
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
    "AnatidAblation",
    "AnatidSystem",
    "BaseSystem",
    "Bm25System",
    "ExternalSystem",
    "HybridSystem",
    "MarkdownSystem",
    "Note",
    "Question",
    "QuestionRun",
    "SortedExistingExtractor",
    "System",
    "SystemRun",
    "VectorPrfSystem",
    "VectorSystem",
    "build_messages",
    "builtin_systems",
    "context_tokens",
    "default_systems",
    "deterministic_ids",
    "external_systems",
    "existing_facts",
    "fuse_rankings",
    "fused_recall",
    "pin_ties",
    "fit_budget",
    "load_gold_patches",
    "load_notes",
    "load_questions",
    "render_memory",
    "run_system",
]

#: The memory budget every system answers within, in tokens of the counter in use.
BUDGET_TOKENS = 1200
#: "Now" for every question, as the corpus defines it.
NOW = "2026-06-30"
#: The exact refusal every system is told to give.
IDK = "I don't know"
#: How many memories anatid's recall returns before the budget cut.
RECALL_K = 20
#: Candidates per retrieval arm before fusion.  anatid's default is 50; the benchmark uses a
#: list long enough to hold every memory the store has (about 170 current memories), so a tie at
#: the end of an arm's list can never change which memories take part in the fusion.
RECALL_CANDIDATES = 200
#: Scores are compared at this many decimals when a ranking is sorted; two BM25 scores that
#: differ only in the last floating-point bits are one tie, broken by recency or memory id.
SCORE_DECIMALS = 6
#: Reciprocal Rank Fusion constant for the fused baselines: ``score = sum 1 / (k + rank)`` with
#: 1-based ranks.  It is the constant anatid's own fusion uses (:data:`anatid.recall.RRF_K`).
FUSION_K = 60
#: Fused scores are compared at this many decimals; two notes whose fused scores differ only in
#: the last bits are one tie, broken by the tie-break ranking the caller names.
FUSION_DECIMALS = 9
#: How many top notes of the first ranking the pseudo-relevance-feedback baseline appends to the
#: question before ranking a second time.
PRF_FEEDBACK_NOTES = 3
#: How many existing facts the extractor is shown, and how many named entities are expanded:
#: anatid's own ``EXISTING_LIMIT`` and ``ENTITY_SCAN_LIMIT``.
EXISTING_LIMIT = 40
ENTITY_SCAN_LIMIT = 25
#: Completion cap for an answer.  The chat model reasons before it answers and those tokens
#: count against the cap, so this is generous on purpose; the instruction keeps answers short.
#: An answer that comes back empty because the reasoning used the whole cap is asked once more
#: with :data:`ANSWER_MAX_TOKENS_RETRY`; one that is still empty is recorded as truncated.
ANSWER_MAX_TOKENS = 4000
ANSWER_MAX_TOKENS_RETRY = 12000

#: The eleven system names, in the order they run.  Ablations follow the store they share.
SYSTEM_NAMES = (
    "markdown",
    "markdown-full",
    "bm25",
    "vector",
    "hybrid",
    "vector-prf",
    "anatid",
    "anatid-text",
    "anatid-vector",
    "anatid-graph",
    "anatid-gold",
)

#: Where the counter allocator starts for each anatid store, so an id says which store it
#: belongs to.  Ids must be the same on every run: the extractor's prompt lists existing
#: memories by id, and a changed id is a changed prompt and a cache miss.
ID_BASE_LLM = 1_000_000
ID_BASE_GOLD = 2_000_000

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# --------------------------------------------------------------------------- the corpus


@dataclass(frozen=True)
class Note:
    """One line of ``notes.jsonl``.  ``raw`` keeps the row for adapters that need other fields."""

    note_id: str
    seq: int
    date: _dt.date
    source: str
    source_kind: str
    team: str
    author: str
    text: str
    rendered: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def when(self) -> _dt.datetime:
        """The instant anatid stamps this note with: 09:00 UTC on its date plus ``seq``
        minutes, so two notes on one day keep their arrival order (as ``gen_corpus`` does)."""
        return _dt.datetime.combine(
            self.date, _dt.time(9, 0, tzinfo=_dt.timezone.utc)
        ) + _dt.timedelta(minutes=self.seq)

    def to_dict(self) -> dict[str, Any]:
        row = dict(self.raw)
        row.update(
            {
                "note_id": self.note_id,
                "seq": self.seq,
                "date": self.date.isoformat(),
                "source": self.source,
                "source_kind": self.source_kind,
                "team": self.team,
                "author": self.author,
                "text": self.text,
                "rendered": self.rendered,
            }
        )
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Note:
        return cls(
            note_id=str(row["note_id"]),
            seq=int(row["seq"]),
            date=_dt.date.fromisoformat(row["date"]),
            source=str(row["source"]),
            source_kind=str(row.get("source_kind", "")),
            team=str(row.get("team", "")),
            author=str(row.get("author", "")),
            text=str(row.get("text", "")),
            rendered=str(row["rendered"]),
            raw=dict(row),
        )


@dataclass(frozen=True)
class Question:
    """One line of ``questions.jsonl``."""

    qid: str
    category: str
    subtype: str
    question: str
    gold: str
    aliases: tuple[str, ...] = ()
    support: tuple[str, ...] = ()
    as_of: str | None = None
    hops: int | None = None
    distractors: tuple[str, ...] = ()
    rubric: str = ""
    chain: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "category": self.category,
            "subtype": self.subtype,
            "question": self.question,
            "gold": self.gold,
            "aliases": list(self.aliases),
            "distractors": list(self.distractors),
            "support": list(self.support),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Question:
        return cls(
            qid=str(row["qid"]),
            category=str(row["category"]),
            subtype=str(row.get("subtype", "")),
            question=str(row["question"]),
            gold=str(row["gold"]),
            aliases=tuple(str(a) for a in row.get("aliases") or ()),
            support=tuple(str(s) for s in row.get("support") or ()),
            as_of=row.get("as_of"),
            hops=row.get("hops"),
            distractors=tuple(str(d) for d in row.get("distractors") or ()),
            rubric=str(row.get("rubric", "")),
            chain=tuple(str(c) for c in row.get("chain") or ()),
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_notes(path: Path = NOTES_PATH) -> list[Note]:
    notes = [Note.from_dict(r) for r in _read_jsonl(path)]
    notes.sort(key=lambda n: n.seq)
    return notes


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    return [Question.from_dict(r) for r in _read_jsonl(path)]


def load_gold_patches(path: Path = GOLD_PATCHES_PATH) -> list[dict[str, Any]]:
    return _read_jsonl(path)


# --------------------------------------------------------------------------- the prompt

SYSTEM_PROMPT = (
    "You answer questions for an engineering team from a memory store. The user message holds "
    "the memory you may use, then the question.\n"
    "Rules:\n"
    "- Answer only from the memory provided. Do not use outside knowledge and do not guess.\n"
    f"- If the memory does not contain the answer, reply exactly: {IDK}\n"
    "- When the memory does contain it, reply in one short line with just the value asked for "
    "(a name, a team, a service, a date, a note label, a reason) and nothing else.\n"
    "- When a value changed over time, answer for the moment the question asks about; without "
    f"a stated moment, answer for now. Today is {NOW}.\n"
    "- Memory entries carry the date they were recorded. Treat a later entry as superseding an "
    "earlier one on the same subject."
)

USER_TEMPLATE = "Memory:\n{memory}\n\nQuestion: {question}"


def render_memory(contexts: Sequence[str]) -> str:
    """The memory block exactly as the prompt shows it: one context per line."""
    return "\n".join(contexts) if contexts else "(the memory is empty)"


def build_messages(question: str, contexts: Sequence[str]) -> list[dict[str, str]]:
    """The one prompt every system uses."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(memory=render_memory(contexts), question=question),
        },
    ]


# --------------------------------------------------------------------------- the budget


def context_tokens(contexts: Sequence[str], counter: TokenCounter) -> int:
    """Tokens of the memory block as the prompt will carry it (separators included)."""
    return counter.count(render_memory(contexts)) if contexts else 0


def fit_budget(candidates: Iterable[str], budget: int | None, counter: TokenCounter) -> list[str]:
    """The longest prefix of ``candidates`` whose rendered memory block fits in ``budget`` tokens.

    Candidates are taken in the order given (best first), whole, until the next one would push
    the block over the budget; the cut stops there rather than skipping ahead, so the result
    is always a prefix of the ranking.  ``budget=None`` means no budget.  Empty strings are
    dropped.  This is the only place the budget is applied, for every system.
    """
    kept: list[str] = []
    for text in candidates:
        if not text or not text.strip():
            continue
        if budget is None:
            kept.append(text)
            continue
        if context_tokens([*kept, text], counter) > budget:
            break
        kept.append(text)
    return kept


# --------------------------------------------------------------------------- systems


@runtime_checkable
class System(Protocol):
    """What the runner needs from a memory system."""

    name: str

    def ingest(self, notes: Sequence[Note]) -> None: ...

    def retrieve(self, question: str, budget: int | None) -> list[str]: ...

    def answer(self, question: str, contexts: Sequence[str]) -> str: ...


class BaseSystem:
    """Shared machinery: the budget cut and the answer prompt are final here.

    A subclass implements :meth:`ingest` and :meth:`candidates` (ranked contexts, best first,
    unbudgeted) and may override :meth:`arrange` to reorder the *selected* contexts for the
    prompt (the Markdown system shows the tail of its file in reading order).  ``unbudgeted``
    marks the one variant the runner deliberately runs without a budget.
    """

    name: str = "base"
    description: str = ""
    unbudgeted: bool = False
    shares_store_with: str | None = None

    def __init__(self, llm: LLMClient, *, counter: TokenCounter | None = None) -> None:
        self.llm = llm
        self.counter = counter or llm.counter
        self.workdir: Path | None = None
        self.ingest_report: dict[str, Any] = {}
        self.last_retrieval: dict[str, Any] = {}
        self.last_answer: dict[str, Any] = {}

    def ingest(self, notes: Sequence[Note]) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def candidates(self, question: str) -> list[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def arrange(self, selected: list[str]) -> list[str]:
        return selected

    def retrieve(self, question: str, budget: int | None) -> list[str]:
        self.last_retrieval = {}
        ranked = self.candidates(question)  # may record how the ranking was made
        selected = fit_budget(ranked, budget, self.counter)
        self.last_retrieval = {
            **self.last_retrieval,
            "candidates": len(ranked),
            "selected": len(selected),
            "candidate_tokens": context_tokens(ranked, self.counter),
        }
        return self.arrange(selected)

    def answer(self, question: str, contexts: Sequence[str]) -> str:
        messages = build_messages(question, contexts)
        result = self.llm.chat(messages, temperature=0.0, max_tokens=ANSWER_MAX_TOKENS)
        escalated = False
        if not result.content and result.finish_reason == "length":
            escalated = True
            result = self.llm.chat(messages, temperature=0.0, max_tokens=ANSWER_MAX_TOKENS_RETRY)
        self.last_answer = {
            "finish_reason": result.finish_reason,
            "escalated": escalated,
            "truncated": not result.content and result.finish_reason == "length",
        }
        return result.content

    def close(self) -> None:
        return None


# ---------------------------------------------------------------- S1 markdown


class MarkdownSystem(BaseSystem):
    """One Markdown file, notes appended in arrival order, the most recent ones in the prompt."""

    def __init__(self, llm: LLMClient, *, budgeted: bool = True, **kw: Any) -> None:
        super().__init__(llm, **kw)
        self.name = "markdown" if budgeted else "markdown-full"
        self.unbudgeted = not budgeted
        self.description = (
            "one Markdown file in arrival order; the most recent notes that fit the budget"
            if budgeted
            else "the same Markdown file, whole, with no budget (upper bound of the plain prompt)"
        )
        self.lines: list[str] = []

    def ingest(self, notes: Sequence[Note]) -> None:
        self.lines = [f"- {n.rendered}" for n in notes]
        if self.workdir is not None:
            self.workdir.mkdir(parents=True, exist_ok=True)
            (self.workdir / "notes.md").write_text(
                "# Team notes\n\n" + "\n".join(self.lines) + "\n", encoding="utf-8"
            )
        self.ingest_report = {"notes": len(self.lines)}

    def candidates(self, question: str) -> list[str]:
        return list(reversed(self.lines))

    def arrange(self, selected: list[str]) -> list[str]:
        order = {line: i for i, line in enumerate(self.lines)}
        return sorted(selected, key=lambda line: order.get(line, 0))


# ---------------------------------------------------------------- S2 bm25


class Bm25System(BaseSystem):
    """DuckDB ``fts`` over the raw notes, Okapi BM25, top notes first.

    The index keeps digits (``ignore='(\\.|[^a-z0-9])+'``) so a date or an incident id in a
    question can match; DuckDB's default pattern drops them.  Porter stemming and English stop
    words otherwise as the extension ships them.  Ties break newest first.
    """

    name = "bm25"
    description = "DuckDB fts over raw notes; the top notes by BM25 that fit the budget"

    def __init__(self, llm: LLMClient, **kw: Any) -> None:
        super().__init__(llm, **kw)
        self._con: Any = None

    def ingest(self, notes: Sequence[Note]) -> None:
        import duckdb

        con = duckdb.connect(":memory:")
        try:
            con.execute("LOAD fts")
        except duckdb.Error:
            con.execute("INSTALL fts")
            con.execute("LOAD fts")
        con.execute(
            "CREATE TABLE notes (note_id VARCHAR PRIMARY KEY, seq INTEGER, rendered VARCHAR)"
        )
        con.executemany(
            "INSERT INTO notes VALUES (?, ?, ?)", [(n.note_id, n.seq, n.rendered) for n in notes]
        )
        con.execute(
            "PRAGMA create_fts_index('notes', 'note_id', 'rendered', stemmer='porter', "
            "stopwords='english', ignore='(\\.|[^a-z0-9])+', strip_accents=1, lower=1, overwrite=1)"
        )
        self._con = con
        self.ingest_report = {
            "notes": len(notes),
            "index": "duckdb fts, porter stemmer, english stopwords, digits kept",
        }

    def candidates(self, question: str) -> list[str]:
        # The score is rounded before ordering: DuckDB sums the per-term contributions in
        # parallel, so two notes with the same BM25 score can differ in the last bit from one
        # call to the next, and the tie-break on recency would otherwise not be reached.
        rows = self._con.execute(
            "SELECT rendered, score FROM (SELECT rendered, seq, "
            "fts_main_notes.match_bm25(note_id, ?) AS score FROM notes) "
            f"WHERE score IS NOT NULL ORDER BY round(score, {SCORE_DECIMALS}) DESC, seq DESC",
            [question],
        ).fetchall()
        return [f"- {r[0]}" for r in rows]

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None


# ---------------------------------------------------------------- S3 vector


class VectorSystem(BaseSystem):
    """The embedding model over the raw notes; top notes by cosine similarity, newest first
    on ties."""

    name = "vector"
    description = (
        "text-embedding-3-small over raw notes; the top notes by cosine that fit the budget"
    )

    def __init__(self, llm: LLMClient, **kw: Any) -> None:
        super().__init__(llm, **kw)
        self._matrix: Any = None
        self._texts: list[str] = []
        self._seq: list[int] = []

    def ingest(self, notes: Sequence[Note]) -> None:
        import numpy as np

        self._texts = [f"- {n.rendered}" for n in notes]
        self._seq = [n.seq for n in notes]
        vectors = self.llm.embed([n.rendered for n in notes])
        matrix = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._matrix = matrix / norms
        self.ingest_report = {"notes": len(notes), "dim": int(matrix.shape[1])}

    def rank(self, text: str) -> list[str]:
        """Every note by cosine to ``text``, best first, newest first on ties."""
        import numpy as np

        q = np.asarray(self.llm.embed([text])[0], dtype=np.float64)
        norm = float(np.linalg.norm(q)) or 1.0
        scores = self._matrix @ (q / norm)
        order = sorted(
            range(len(self._texts)),
            key=lambda i: (-round(float(scores[i]), SCORE_DECIMALS), -self._seq[i]),
        )
        return [self._texts[i] for i in order]

    def candidates(self, question: str) -> list[str]:
        return self.rank(question)


# ---------------------------------------------------------------- fused baselines


def fuse_rankings(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = FUSION_K,
    tie_break: Sequence[str] | None = None,
) -> list[str]:
    """Reciprocal Rank Fusion of several rankings of the same lines.

    A line's fused score is ``sum 1 / (k + rank)`` over the rankings that hold it, ranks
    1-based, so a line that two arms rank high beats one that a single arm ranks first.
    Scores are compared at :data:`FUSION_DECIMALS`; ties keep the order of ``tie_break`` (the
    first ranking when none is given), with lines absent from it last.  This is the rule
    :func:`anatid.recall.rrf_fuse` applies to memory ids, applied to prompt lines.
    """
    score: dict[str, float] = {}
    for ranking in rankings:
        for i, line in enumerate(ranking):
            score[line] = score.get(line, 0.0) + 1.0 / (k + i + 1)
    order = {line: i for i, line in enumerate(tie_break if tie_break is not None else rankings[0])}
    return sorted(
        score, key=lambda line: (-round(score[line], FUSION_DECIMALS), order.get(line, len(order)))
    )


class HybridSystem(BaseSystem):
    """BM25 and cosine over the raw notes, fused with Reciprocal Rank Fusion.

    The two single-arm baselines each rank every note; :func:`fuse_rankings` combines the two
    lists with ``k`` = :data:`FUSION_K`, ties broken by the cosine rank.  anatid fuses three
    arms over its memories, so a fair question is whether the fusion or the memory earns the
    difference; this system answers it for the raw notes.
    """

    name = "hybrid"
    description = (
        "BM25 and cosine over raw notes fused with reciprocal rank fusion; the top notes that "
        "fit the budget"
    )

    def __init__(self, llm: LLMClient, *, k: int = FUSION_K, **kw: Any) -> None:
        super().__init__(llm, **kw)
        self.fusion_k = int(k)
        self.bm25 = Bm25System(llm, counter=self.counter)
        self.vector = VectorSystem(llm, counter=self.counter)

    def ingest(self, notes: Sequence[Note]) -> None:
        self.bm25.ingest(notes)
        self.vector.ingest(notes)
        self.ingest_report = {
            "notes": len(notes),
            "arms": ["bm25", "vector"],
            "fusion": f"rrf k={self.fusion_k}, ties by cosine rank",
            "bm25": self.bm25.ingest_report,
            "vector": self.vector.ingest_report,
        }

    def candidates(self, question: str) -> list[str]:
        by_text = self.bm25.candidates(question)
        by_vector = self.vector.candidates(question)
        self.last_retrieval = {"bm25_candidates": len(by_text), "vector_candidates": len(by_vector)}
        return fuse_rankings([by_text, by_vector], k=self.fusion_k, tie_break=by_vector)

    def close(self) -> None:
        self.bm25.close()
        self.vector.close()


class VectorPrfSystem(VectorSystem):
    """The vector baseline with one round of pseudo-relevance feedback.

    The question is ranked as ``vector`` ranks it; the text of the top
    :data:`PRF_FEEDBACK_NOTES` notes is appended to the question and the notes are ranked once
    more against that; the two rankings are fused with :func:`fuse_rankings`, ties broken by
    the first ranking.  A second round is the usual cheap way to reach a note that shares no
    words with the question, so this is the baseline anatid's multi-hop number should be read
    against.  The feedback embedding is one more cached model call per question.
    """

    name = "vector-prf"
    description = (
        "the vector baseline with one round of pseudo-relevance feedback (top notes appended to "
        "the question, ranked again, both rankings fused)"
    )

    def __init__(
        self, llm: LLMClient, *, feedback: int = PRF_FEEDBACK_NOTES, k: int = FUSION_K, **kw: Any
    ) -> None:
        super().__init__(llm, **kw)
        self.feedback = int(feedback)
        self.fusion_k = int(k)

    def candidates(self, question: str) -> list[str]:
        first = self.rank(question)
        # a candidate line is "- <rendered>"; the feedback is the notes' text without the bullet
        feedback = " ".join(line[2:] for line in first[: self.feedback])
        second = self.rank(question + "\n" + feedback)
        self.last_retrieval = {"feedback_notes": min(self.feedback, len(first))}
        return fuse_rankings([first, second], k=self.fusion_k, tie_break=first)


# ---------------------------------------------------------------- S4 / S5 anatid

ALL_ARMS = frozenset({"text", "vector", "graph"})


@contextlib.contextmanager
def deterministic_ids(start: int) -> Iterator[Callable[[], int]]:
    """Make :func:`anatid.ids.new_id` a counter from ``start`` for the duration of the block.

    Installed through the public :func:`anatid.ids.set_allocator` and restored on exit.  It is
    process wide, so nothing else should write to an anatid database while a build runs.
    """
    from anatid.ids import reset_allocator, set_allocator

    counter = itertools.count(int(start))
    lock = threading.Lock()

    def allocate() -> int:
        with lock:
            return next(counter)

    set_allocator(allocate)
    try:
        yield allocate
    finally:
        reset_allocator()


def _day(value: _dt.datetime | None) -> str:
    return value.date().isoformat() if value is not None else "?"


def pin_ties(rows: Sequence[tuple[int, float]]) -> list[tuple[int, float]]:
    """One arm's ``[(memory_id, score)]`` in an order that does not change between runs.

    anatid's BM25 arm sums term contributions inside DuckDB, and the order of that floating point
    sum varies between executions, so two memories with the same BM25 score can come back with
    their ranks swapped from one call to the next; the fused order then varies too, and with it
    the prompt.  Sorting by the score rounded to :data:`SCORE_DECIMALS`, then by memory id, pins
    the tie order and changes nothing else about the ranking.
    """
    return sorted(rows, key=lambda r: (-round(float(r[1]), SCORE_DECIMALS), int(r[0])))


def fused_recall(
    db: Any,
    query: str,
    *,
    arms: Iterable[str] = ALL_ARMS,
    embedding: Sequence[float] | None = None,
    k: int = RECALL_K,
    candidates: int = RECALL_CANDIDATES,
    tenant_id: int = 1,
) -> tuple[list[Any], dict[str, Any]]:
    """anatid's three retrieval arms, fused with anatid's Reciprocal Rank Fusion, tie order pinned.

    This is :func:`anatid.recall.hybrid_recall` assembled in the open: the same
    :func:`~anatid.recall.bm25_arm`, :func:`~anatid.recall.vector_arm`, graph expansion from the
    entities the query names (:func:`~anatid.recall.auto_seeds`), the same
    :func:`~anatid.recall.rrf_fuse` with ``k = 60`` and ties on memory id, and the same
    :func:`~anatid.recall.hydrate`.  Two things differ, both for reproducibility: each arm's list
    goes through :func:`pin_ties`, and the lists are long enough (``candidates``) to hold the
    whole store, so an arm's cut-off never decides membership.  Returns the hydrated memories in
    fused order and a record of which arms ran and from which seeds.
    """
    from anatid.recall import (
        RRF_K,
        auto_seeds,
        bm25_arm,
        hybrid_recall,
        hydrate,
        rrf_fuse,
        vector_arm,
    )

    wanted = frozenset(arms)
    con = db.connection
    lists: dict[str, list[tuple[int, float]]] = {}
    seeds: list[str] = []
    if "vector" in wanted:
        if embedding is None:
            if db.embedder is None:
                raise RuntimeError("the vector arm needs an embedding or an embedder on the handle")
            embedding = db.embedder.embed_one(query)
        lists["vector"] = pin_ties(
            vector_arm(
                con,
                tenant_id=tenant_id,
                embedding=embedding,
                dim=db.config.embedding_dim,
                topn=candidates,
            )
        )
    if "text" in wanted:
        lists["text"] = pin_ties(
            bm25_arm(con, tenant_id=tenant_id, query_text=query, topn=candidates)
        )
    if "graph" in wanted:
        found = auto_seeds(con, tenant_id=tenant_id, query=query)
        seeds = [name for _sid, name in found]
        if found:
            # The graph arm orders by creation time and memory id, so it needs no pinning; with
            # one arm the fused order is the arm's order.
            graph_hits = hybrid_recall(
                con,
                tenant_id=tenant_id,
                query=None,
                embedding=None,
                dim=db.config.embedding_dim,
                k=candidates,
                candidates=candidates,
                seed_entity=[sid for sid, _name in found],
                backend=db.csr,
                on_stale_fts="ignore",
            )
            lists["graph"] = [
                (h.memory_id, float(len(graph_hits) - i)) for i, h in enumerate(graph_hits)
            ]
    fused = rrf_fuse(lists, k=RRF_K, top=k)
    ids = [mid for mid, _score, _ranks, _scores in fused]
    rows = hydrate(con, ids, tenant_id=tenant_id)
    memories = [rows[mid] for mid in ids if mid in rows]
    return memories, {"arms_ran": list(lists), "seeds": seeds, "hits": len(memories)}


def existing_facts(db: Any, text: str, *, tenant_id: int = 1) -> list[Any]:
    """The current facts about the entities ``text`` names, plus the text-search hits, as
    :func:`anatid.ingest.existing_context` assembles them, with :func:`fused_recall` in place of
    ``db.recall`` so the list is the same on every run.

    An entity counts as named when its canonical key occurs in the text as a whole word; its
    facts come newest first; at most :data:`EXISTING_LIMIT` facts in all.  Each is a
    :class:`anatid.ingest.KnownFact`.
    """
    import re

    from anatid.ingest import KnownFact
    from anatid.schema import entity_key
    from anatid.visibility import current_row_sql, tenant_sql

    folded = entity_key(text) or ""
    if not folded:
        return []
    rows = db.execute(
        f"SELECT name, entity_key FROM entities WHERE {tenant_sql()} AND {current_row_sql()} "
        f"AND length(entity_key) >= 2 AND contains(?, entity_key) "
        f"ORDER BY length(entity_key) DESC, entity_id",
        [tenant_id, folded],
    ).fetchall()
    named: list[str] = []
    for name, key in rows:
        if re.search(rf"(?<!\w){re.escape(key)}(?!\w)", folded):
            named.append(name)
        if len(named) >= ENTITY_SCAN_LIMIT:
            break

    out: list[Any] = []
    seen: set[int] = set()

    def take(memory: Any) -> bool:
        if memory.memory_id in seen or not memory.is_current:
            return False
        seen.add(memory.memory_id)
        about = tuple(e.name for e in db.entities_of(memory.memory_id))
        out.append(KnownFact.of(memory, about))
        return len(out) >= EXISTING_LIMIT

    for name in named:
        for memory in db.context(name, limit=EXISTING_LIMIT):
            if take(memory):
                return out
    hits, _info = fused_recall(db, text, k=min(EXISTING_LIMIT, 20), tenant_id=tenant_id)
    for memory in hits:
        if take(memory):
            break
    return out


class SortedExistingExtractor:
    """An extractor whose ``existing`` facts arrive newest first by memory id.

    The pipeline hands the extractor the current facts about the entities a note names, in the
    order its own recall produced them, and that order is not stable between runs (see
    :func:`stable_order`).  The prompt the model sees lists those facts, so an unstable order is
    a different prompt and a cache miss.  Sorting by memory id, which the deterministic allocator
    makes a creation order, pins it.  ``inner`` is any :class:`anatid.ingest.Extractor`.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def extract(self, text: str, *, existing: Sequence[Any]) -> Any:
        ordered = sorted(existing, key=lambda m: -int(m.memory_id))
        return self.inner.extract(text, existing=ordered)


class AnatidSystem(BaseSystem):
    """Notes ingested through ``anatid.ingest``; ``recall`` with the given arms.

    ``extractor="llm"`` uses :class:`anatid.ingest.OpenAICompatibleExtractor` with the
    benchmark's chat model (the product path, S4); ``extractor="gold"`` replays the corpus's
    gold patches through :class:`anatid.ingest.ScriptedExtractor` (the oracle, S5).  Either way
    the handle carries the benchmark's embedder, so every memory has a vector and the vector
    arm runs.  Retrieval is :func:`fused_recall`: anatid's arms and anatid's fusion, assembled
    in the harness so tie order is pinned and a rerun asks the same prompts.  Each recalled
    memory is rendered with its kind, its validity interval and its writer (the note label),
    followed by the versions it superseded, newest first, so temporal and provenance questions
    can be answered from the rendering alone.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        extractor: str = "llm",
        name: str | None = None,
        arms: Iterable[str] = ALL_ARMS,
        k: int = RECALL_K,
        gold_patches: Sequence[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(llm, **kw)
        if extractor not in ("llm", "gold"):
            raise ValueError(f"extractor must be 'llm' or 'gold', got {extractor!r}")
        self.extractor_kind = extractor
        self.arms = frozenset(arms)
        if not self.arms <= ALL_ARMS:
            raise ValueError(f"unknown arms {sorted(self.arms - ALL_ARMS)}")
        self.k = int(k)
        self.name = name or ("anatid" if extractor == "llm" else "anatid-gold")
        self.description = (
            "notes ingested with anatid.ingest and the chat model as extractor; recall with "
            "every arm, memories rendered with validity interval, writer and superseded versions"
            if extractor == "llm"
            else "the same store built from the corpus's gold patches (ScriptedExtractor): an "
            "oracle for extraction, not the product"
        )
        self._gold_patches = list(gold_patches) if gold_patches is not None else None
        self.db: Any = None
        self.tenant_id: int = 1

    # ------------------------------------------------------------------ ingest

    def _open(self) -> Any:
        from anatid import Anatid

        path: str | Path = ":memory:"
        if self.workdir is not None:
            self.workdir.mkdir(parents=True, exist_ok=True)
            path = self.workdir / "memory.anatid"
            for suffix in ("", ".wal"):
                p = Path(str(path) + suffix)
                if p.exists():
                    p.unlink()
        return Anatid.open(
            path,
            tenant=self.tenant_id,
            embedding_dim=self.llm.embed_dim,
            embedder=self.llm.embedder(),
        )

    def _make_extractor(self, notes: Sequence[Note]) -> Any:
        from anatid.ingest import MemoryPatch, OpenAICompatibleExtractor, ScriptedExtractor

        if self.extractor_kind == "gold":
            patches = self._gold_patches if self._gold_patches is not None else load_gold_patches()
            by_note = {p["note_id"]: p for p in patches}
            ordered = [MemoryPatch.from_dict(by_note[n.note_id]["patch"]) for n in notes]
            return ScriptedExtractor(ordered)
        return SortedExistingExtractor(
            OpenAICompatibleExtractor(
                model=self.llm.chat_model,
                client=self.llm.chat_client(),
                temperature=0.0,
                response_format="json_object",
            )
        )

    def _ingest_one(self, extractor: Any, note: Note) -> Any:
        """:func:`anatid.ingest.ingest` step by step, with :func:`existing_facts` as the context.

        The pipeline's own context step ranks with ``db.recall``, whose tie order varies between
        runs (see :func:`pin_ties`); a varying list of existing facts is a varying extractor
        prompt, a cache miss, and a store that differs from the last run's.  Everything else is
        the pipeline unchanged: the extractor proposes, :func:`anatid.ingest.prepare` resolves,
        the review hook drops a second correction of one memory, ``apply`` commits.
        """
        from anatid.ingest import prepare

        existing = existing_facts(self.db, note.rendered, tenant_id=self.tenant_id)
        proposed = extractor.extract(note.rendered, existing=existing)
        if not proposed.source_text:
            proposed = proposed.replace(source_text=note.rendered)
        patch = _one_correction_per_memory(prepare(proposed, self.db))
        return patch.apply(self.db, writer=note.source, source=note.source, now=note.when)

    def ingest(self, notes: Sequence[Note]) -> None:
        from anatid.errors import AnatidError
        from anatid.ingest import MemoryPatch

        self.db = self._open()
        extractor = self._make_extractor(notes)
        report: dict[str, Any] = {
            "notes": len(notes),
            "extractor": self.extractor_kind,
            "memories_created": 0,
            "memories_closed": 0,
            "relations_opened": 0,
            "relations_closed": 0,
            "dedupe_drops": 0,
            "downgraded_corrections": 0,
            "parser_notes": 0,
            "empty_patches": 0,
            "extraction_failures": 0,
            "failed_notes": [],
        }
        base = ID_BASE_LLM if self.extractor_kind == "llm" else ID_BASE_GOLD
        with deterministic_ids(base):
            for note in notes:
                try:
                    receipt = self._ingest_one(extractor, note)
                except AnatidError as exc:
                    # The note is still recorded as read: an empty patch stores the episode.
                    report["extraction_failures"] += 1
                    report["failed_notes"].append(
                        {"note_id": note.note_id, "error": str(exc)[:200]}
                    )
                    MemoryPatch(source_text=note.rendered).apply(
                        self.db, writer=note.source, source=note.source, now=note.when
                    )
                    continue
                assert receipt is not None
                report["memories_created"] += len(receipt.memories_created)
                report["memories_closed"] += len(receipt.memories_closed)
                report["relations_opened"] += len(receipt.relations_opened)
                report["relations_closed"] += len(receipt.relations_closed)
                if receipt.changes == 0:
                    report["empty_patches"] += 1
                for line in receipt.patch.notes:
                    if line.startswith("dedupe:"):
                        report["dedupe_drops"] += 1
                    elif line.startswith("correction of"):
                        report["downgraded_corrections"] += 1
                    elif line.startswith("parser:"):
                        report["parser_notes"] += 1
        stats = self.db.stats()
        report["store"] = {k: v for k, v in stats.items() if isinstance(v, (int, float, str))}
        self.ingest_report = report

    # ------------------------------------------------------------------ recall

    def render_hit(self, memory: Any) -> str:
        """One memory as the prompt shows it, with the versions it superseded."""
        head = "- " + _render_memory_line(memory, current=True)
        chain = self.db.provenance(memory.memory_id).chain[1:]
        tail = [f"    earlier: {_render_memory_line(old, current=False)}" for old in chain]
        return "\n".join([head, *tail])

    def candidates_for(self, question: str, arms: frozenset[str]) -> list[str]:
        if self.db is None:
            raise RuntimeError(f"{self.name}: ingest() has not run")
        embedding = self.llm.embed([question])[0] if "vector" in arms else None
        memories, info = fused_recall(
            self.db,
            question,
            arms=arms,
            embedding=embedding,
            k=self.k,
            tenant_id=self.tenant_id,
        )
        self.last_retrieval = info
        return [self.render_hit(m) for m in memories]

    def candidates(self, question: str) -> list[str]:
        return self.candidates_for(question, self.arms)

    def ablation(self, name: str, arms: Iterable[str]) -> AnatidAblation:
        return AnatidAblation(self, name=name, arms=arms)

    def close(self) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None


def _one_correction_per_memory(patch: Any) -> Any:
    """Review hook for the model's patches: a second correction of the same memory in one
    patch would make ``supersede`` refuse to fork the chain and roll the whole patch back, so
    only the first is kept and the drop is noted."""
    seen: set[int] = set()
    kept = []
    notes: list[str] = []
    for corr in patch.corrections:
        if corr.old_id is not None:
            if corr.old_id in seen:
                notes.append(
                    f"bench: dropped a second correction of memory {corr.old_id} in one patch "
                    f"({corr.new_content!r})"
                )
                continue
            seen.add(corr.old_id)
        kept.append(corr)
    if len(kept) == len(patch.corrections):
        return patch
    return patch.replace(corrections=tuple(kept), notes=patch.notes + tuple(notes))


def _render_memory_line(memory: Any, *, current: bool) -> str:
    kind = f"{memory.kind}; " if memory.kind and memory.kind != "fact" else ""
    until = "now" if current and memory.valid_to is None else _day(memory.valid_to)
    writer = memory.writer or "unknown"
    return (
        f"{memory.content} [{kind}valid {_day(memory.valid_from)} to {until}; recorded by {writer}]"
    )


class AnatidAblation(BaseSystem):
    """One recall arm over a parent :class:`AnatidSystem`'s store; ingests nothing itself."""

    def __init__(self, parent: AnatidSystem, *, name: str, arms: Iterable[str]) -> None:
        super().__init__(parent.llm, counter=parent.counter)
        self.parent = parent
        self.name = name
        self.arms = frozenset(arms)
        if not self.arms <= ALL_ARMS:
            raise ValueError(f"unknown arms {sorted(self.arms - ALL_ARMS)}")
        self.shares_store_with = parent.name
        self.description = (
            f"the {parent.name} store queried with the {'+'.join(sorted(self.arms))} arm only"
        )

    def ingest(self, notes: Sequence[Note]) -> None:
        if self.parent.db is None:
            raise RuntimeError(f"{self.name}: run {self.parent.name} first; it owns the store")
        self.ingest_report = {"shares_store_with": self.parent.name}

    def candidates(self, question: str) -> list[str]:
        out = self.parent.candidates_for(question, self.arms)
        self.last_retrieval = dict(self.parent.last_retrieval)
        return out


# --------------------------------------------------------------------------- external systems


class ExternalSystem(BaseSystem):
    """A system from ``bench.quality.systems`` behind this module's protocol.

    Those systems are built with ``build(notes)`` and answer ``retrieve(question, budget=...)``
    with a record whose ``text`` (or ``context``) is the memory block.  The adapter converts the
    notes, calls them, splits the block into lines, and passes the lines through
    :func:`fit_budget` so the harness's budget holds for them too (a no-op when their own
    packing already fits, which it does when they share the counter).  ``factory`` builds the
    inner system lazily so its construction cost lands in the ingest phase.
    """

    def __init__(
        self,
        llm: LLMClient,
        factory: Callable[[Sequence[Note]], Any],
        *,
        name: str,
        description: str = "",
        unbudgeted: bool = False,
        shares_store_with: str | None = None,
        note_type: Any | None = None,
    ) -> None:
        super().__init__(llm)
        self.name = name
        self.description = description
        self.unbudgeted = unbudgeted
        self.shares_store_with = shares_store_with
        self._factory = factory
        self._note_type = note_type
        self.inner: Any = None

    def ingest(self, notes: Sequence[Note]) -> None:
        self.inner = self._factory(notes)
        build = getattr(self.inner, "build", None)
        if callable(build):
            converted = (
                [self._note_type.from_dict(n.to_dict()) for n in notes]
                if self._note_type is not None
                else list(notes)
            )
            build(converted)
        report: dict[str, Any] = {"notes": len(notes), "external": type(self.inner).__name__}
        store = getattr(self.inner, "store", None)
        stats = getattr(store, "stats", None)
        if stats is not None and hasattr(stats, "to_dict"):
            report["store_stats"] = stats.to_dict()
        if self.workdir is not None and callable(getattr(self.inner, "write", None)):
            self.inner.write(self.workdir / "notes.md")
        self.ingest_report = report

    def candidates(self, question: str) -> list[str]:  # pragma: no cover - retrieve is overridden
        raise NotImplementedError

    def retrieve(self, question: str, budget: int | None) -> list[str]:
        if self.inner is None:
            raise RuntimeError(f"{self.name}: ingest() has not run")
        record = self.inner.retrieve(question, budget=budget)
        text = getattr(record, "text", None)
        if text is None:
            text = getattr(record, "context", "")
        lines = [line for line in str(text or "").split("\n") if line.strip()]
        selected = fit_budget(lines, budget, self.counter)
        meta: dict[str, Any] = {"lines": len(lines), "selected": len(selected)}
        for attr in ("considered", "skipped", "hits_total", "hits_rendered", "arms", "seeds"):
            value = getattr(record, attr, None)
            if value is not None:
                meta[attr] = list(value) if isinstance(value, tuple) else value
        self.last_retrieval = meta
        return selected

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()


def external_systems(
    llm: LLMClient,
    notes: Sequence[Note],
    gold_patches: Sequence[dict[str, Any]],
    *,
    store_dir: Path | None = None,
    k: int = RECALL_K,
    rebuild: bool = False,
) -> list[BaseSystem]:
    """The systems of ``bench.quality.systems`` behind :class:`ExternalSystem`.

    Raises ``ImportError`` when the package is absent and ``AttributeError`` when its surface
    differs from what the adapter expects; :func:`default_systems` falls back to the built-ins.
    The package's token counter is replaced by the harness's so every system counts alike.
    """
    from . import systems as ext
    from .systems import anatid_sys, bm25, hybrid, markdown, vector

    ext.set_token_counter(llm.counter.count, name=llm.counter.name)
    embedder = llm.embedder()
    chat = llm.chat_client()
    by_note = {p["note_id"]: p for p in gold_patches}
    patches = [by_note[n.note_id] for n in notes]
    store_dir = store_dir or (RESULTS_DIR / "stores")

    groups: dict[str, dict[str, Any]] = {}

    def anatid_factory(
        name: str, group: str, include: list[str]
    ) -> Callable[[Sequence[Note]], Any]:
        def make(my_notes: Sequence[Note]) -> Any:
            if group not in groups:
                converted = [anatid_sys.Note.from_dict(n.to_dict()) for n in my_notes]
                groups[group] = anatid_sys.make_systems(
                    notes=converted,
                    patches=patches,
                    include=include,
                    store_dir=store_dir,
                    cache_dir=llm.cache.root,
                    embedder=embedder,
                    chat=chat,
                    rebuild=rebuild,
                    k=k,
                )
            return groups[group][name]

        return make

    glm = ["anatid", "anatid-text", "anatid-vector", "anatid-graph"]
    return [
        ExternalSystem(
            llm,
            lambda _n: markdown.MarkdownSystem(),
            name="markdown",
            description="external: one Markdown file in arrival order; the most recent notes "
            "that fit the budget",
            note_type=ext.Note,
        ),
        ExternalSystem(
            llm,
            lambda _n: markdown.MarkdownSystem(whole_file=True),
            name="markdown-full",
            description="external: the same Markdown file, whole, with no budget",
            unbudgeted=True,
            note_type=ext.Note,
        ),
        ExternalSystem(
            llm,
            lambda _n: bm25.Bm25System(),
            name="bm25",
            description="external: DuckDB fts over raw notes; the top notes by BM25",
            note_type=ext.Note,
        ),
        ExternalSystem(
            llm,
            lambda _n: vector.VectorSystem(embedder),
            name="vector",
            description="external: text-embedding-3-small over raw notes; the top notes by cosine",
            note_type=ext.Note,
        ),
        ExternalSystem(
            llm,
            lambda _n: hybrid.HybridSystem(embedder),
            name="hybrid",
            description="external: BM25 and cosine over raw notes fused with reciprocal rank fusion",
            note_type=ext.Note,
        ),
        ExternalSystem(
            llm,
            lambda _n: hybrid.VectorPrfSystem(embedder),
            name="vector-prf",
            description="external: the vector baseline with one round of pseudo-relevance feedback",
            note_type=ext.Note,
        ),
        ExternalSystem(
            llm,
            anatid_factory("anatid", "glm", glm),
            name="anatid",
            description="external: anatid.ingest with the chat model as extractor; recall with "
            "every arm",
        ),
        ExternalSystem(
            llm,
            anatid_factory("anatid-text", "glm", glm),
            name="anatid-text",
            description="external: the anatid store, text arm only",
            shares_store_with="anatid",
        ),
        ExternalSystem(
            llm,
            anatid_factory("anatid-vector", "glm", glm),
            name="anatid-vector",
            description="external: the anatid store, vector arm only",
            shares_store_with="anatid",
        ),
        ExternalSystem(
            llm,
            anatid_factory("anatid-graph", "glm", glm),
            name="anatid-graph",
            description="external: the anatid store, graph arm only",
            shares_store_with="anatid",
        ),
        ExternalSystem(
            llm,
            anatid_factory("anatid-gold", "gold", ["anatid-gold"]),
            name="anatid-gold",
            description="external: the store built from the gold patches (ScriptedExtractor), "
            "an oracle for extraction",
        ),
    ]


# --------------------------------------------------------------------------- the runner


@dataclass
class QuestionRun:
    """One answered question.  ``answer_s`` is the wall time of the answer call in this run;
    ``answer_network_s`` is the network latency the call had when it was first made, which a
    replay from the cache reports unchanged."""

    qid: str
    category: str
    question: str
    contexts: list[str]
    n_contexts: int
    context_tokens: int
    answer: str
    answer_cached: bool
    retrieve_s: float
    answer_s: float
    answer_network_s: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    retrieval: dict[str, Any] = field(default_factory=dict)
    escalated: bool = False
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SystemRun:
    name: str
    description: str
    budget: int | None
    ingest_s: float
    ingest_stats: dict[str, Any]
    ingest_report: dict[str, Any]
    answers: list[QuestionRun]
    answer_stats: dict[str, Any]
    shares_store_with: str | None = None

    @property
    def answers_by_qid(self) -> dict[str, str]:
        return {a.qid: a.answer for a in self.answers}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "budget": self.budget,
            "shares_store_with": self.shares_store_with,
            "ingest_s": self.ingest_s,
            "ingest_stats": self.ingest_stats,
            "ingest_report": self.ingest_report,
            "answer_stats": self.answer_stats,
            "answers": [a.to_dict() for a in self.answers],
        }


def run_system(
    system: BaseSystem,
    notes: Sequence[Note],
    questions: Sequence[Question],
    *,
    budget: int | None = BUDGET_TOKENS,
    out_dir: Path,
    llm: LLMClient,
    progress: Callable[[str, int, int], None] | None = None,
) -> SystemRun:
    """Ingest once, answer every question, save everything under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    system.workdir = out_dir
    counter = llm.counter

    before = llm.snapshot()
    t0 = time.perf_counter()
    system.ingest(notes)
    ingest_s = time.perf_counter() - t0
    ingest_stats = llm.stats.diff(before).to_dict()
    (out_dir / "ingest.json").write_text(
        json.dumps(
            {"seconds": ingest_s, "llm": ingest_stats, "report": system.ingest_report},
            indent=1,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    effective_budget = None if system.unbudgeted else budget
    runs: list[QuestionRun] = []
    before_answers = llm.snapshot()
    for i, q in enumerate(questions):
        t1 = time.perf_counter()
        contexts = system.retrieve(q.question, effective_budget)
        retrieve_s = time.perf_counter() - t1
        messages = build_messages(q.question, contexts)
        snap = llm.snapshot()
        t2 = time.perf_counter()
        answer = system.answer(q.question, contexts)
        answer_s = time.perf_counter() - t2
        d = llm.stats.diff(snap)
        run = QuestionRun(
            qid=q.qid,
            category=q.category,
            question=q.question,
            contexts=list(contexts),
            n_contexts=len(contexts),
            context_tokens=context_tokens(contexts, counter),
            answer=answer,
            answer_cached=d.network_calls == 0,
            retrieve_s=retrieve_s,
            answer_s=answer_s,
            answer_network_s=d.recorded_wall_s,
            prompt_tokens=d.prompt_tokens,
            completion_tokens=d.completion_tokens,
            cost_usd=d.cost_usd if d.cost_known else None,
            retrieval=dict(system.last_retrieval),
            escalated=bool(system.last_answer.get("escalated", False)),
            truncated=bool(system.last_answer.get("truncated", False)),
        )
        runs.append(run)
        (prompts_dir / f"{q.qid}.json").write_text(
            json.dumps(
                {
                    "qid": q.qid,
                    "system": system.name,
                    "messages": messages,
                    "contexts": contexts,
                    "answer": answer,
                },
                indent=1,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if progress is not None:
            progress(system.name, i + 1, len(questions))
    answer_stats = llm.stats.diff(before_answers).to_dict()
    with (out_dir / "answers.jsonl").open("w", encoding="utf-8") as fh:
        for run in runs:
            fh.write(json.dumps({"qid": run.qid, "answer": run.answer}, ensure_ascii=False) + "\n")
    with (out_dir / "questions.jsonl").open("w", encoding="utf-8") as fh:
        for run in runs:
            fh.write(json.dumps(run.to_dict(), ensure_ascii=False) + "\n")
    return SystemRun(
        name=system.name,
        description=system.description,
        budget=effective_budget,
        ingest_s=ingest_s,
        ingest_stats=ingest_stats,
        ingest_report=system.ingest_report,
        answers=runs,
        answer_stats=answer_stats,
        shares_store_with=system.shares_store_with,
    )


def builtin_systems(
    llm: LLMClient,
    *,
    k: int = RECALL_K,
    gold_patches: Sequence[dict[str, Any]] | None = None,
) -> list[BaseSystem]:
    """The built-in systems, in the order they run.  Ablations follow the store they share."""
    anatid = AnatidSystem(llm, extractor="llm", k=k)
    return [
        MarkdownSystem(llm, budgeted=True),
        MarkdownSystem(llm, budgeted=False),
        Bm25System(llm),
        VectorSystem(llm),
        HybridSystem(llm),
        VectorPrfSystem(llm),
        anatid,
        anatid.ablation("anatid-text", ["text"]),
        anatid.ablation("anatid-vector", ["vector"]),
        anatid.ablation("anatid-graph", ["graph"]),
        AnatidSystem(llm, extractor="gold", k=k, gold_patches=gold_patches),
    ]


def default_systems(
    llm: LLMClient,
    *,
    family: str = "auto",
    notes: Sequence[Note] | None = None,
    gold_patches: Sequence[dict[str, Any]] | None = None,
    store_dir: Path | None = None,
    k: int = RECALL_K,
    rebuild: bool = False,
) -> tuple[list[BaseSystem], dict[str, str]]:
    """The registered systems and a note on which family they come from.

    ``family`` is ``"builtin"``, ``"external"`` (the ``bench.quality.systems`` package) or
    ``"auto"`` (external when it imports and constructs, otherwise built-in).  The second value
    records the choice and, on a fallback, why.
    """
    if family not in ("auto", "builtin", "external"):
        raise ValueError(f"family must be 'auto', 'builtin' or 'external', got {family!r}")
    if family in ("auto", "external"):
        try:
            systems = external_systems(
                llm,
                notes if notes is not None else load_notes(),
                gold_patches if gold_patches is not None else load_gold_patches(),
                store_dir=store_dir,
                k=k,
                rebuild=rebuild,
            )
        except (ImportError, AttributeError, TypeError, ValueError, KeyError) as exc:
            if family == "external":
                raise
            note = f"external systems unavailable ({type(exc).__name__}: {exc}); built-in used"
            return builtin_systems(llm, k=k, gold_patches=gold_patches), {
                "family": "builtin",
                "note": note,
            }
        return systems, {"family": "external", "note": "bench.quality.systems"}
    return builtin_systems(llm, k=k, gold_patches=gold_patches), {
        "family": "builtin",
        "note": "bench.quality.harness built-ins",
    }
