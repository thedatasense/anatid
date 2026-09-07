"""Support coverage: an offline proxy for the retrieval half of the answer-quality benchmark.

A question is *covered* when the memory block a system would hand the model holds, for every
note the gold answer rests on (``Question.support``), at least one memory extracted from that
note, as a head line or in its ``earlier:`` chain.  On the runs behind ``docs/quality.md`` the
LLM judge marked 97 to 98% of covered questions correct and 11 to 29% of uncovered ones, so
coverage ranks retrieval variants without a model call: the store comes from a finished run,
the question embeddings from the cache, and nothing is charged.  It is how the fusion weights
in :func:`anatid.recall.default_arm_weights` and the graph arm's ordering were chosen, and how
they were checked on a world the choice never saw.

    python -m bench.quality.coverage --run s20260905-b1200                # the anatid store
    python -m bench.quality.coverage --run s7-b1200 --system anatid-gold
    python -m bench.quality.coverage --run s11-b1200 --weights vector=1,text=0.5,graph=0.5

The variants: ``product`` is what ``recall()`` does now (:func:`~bench.quality.harness.fused_recall`,
the graph arm ranked by the query's signal, the default weights); ``equal`` is the fusion 0.4.1
shipped (equal votes, the graph arm newest first); ``vector``, ``text`` and ``graph`` are one arm
each; ``--weights`` adds a variant with those weights and the product's graph ordering.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .__init__ import DATA_DIR
from .harness import (
    BUDGET_TOKENS,
    RECALL_CANDIDATES,
    RECALL_K,
    RESULTS_DIR,
    Question,
    _render_memory_line,
    fit_budget,
    load_notes,
    load_questions,
    pin_ties,
)
from .llm import LLMClient, TokenCounter

CATEGORIES = ("single_fact", "knowledge_update", "temporal", "multi_hop", "provenance")
BUILTIN_VARIANTS = ("product", "equal", "vector", "text", "graph")


def parse_weights(text: str) -> dict[str, float]:
    """``vector=1,text=0.25,graph=0.5`` as a mapping."""
    out: dict[str, float] = {}
    for part in text.split(","):
        name, _, value = part.strip().partition("=")
        if not name or not value:
            raise ValueError(f"weights must look like vector=1,text=0.25,graph=0.5, got {text!r}")
        out[name] = float(value)
    return out


class CoverageLab:
    """The arms of one finished run's store, and the coverage of any fusion over them.

    ``run_dir`` is ``bench/quality/results/<run-id>``; ``system`` names the store to open
    (``anatid`` or ``anatid-gold``).  The store is copied to a temporary file first, so a run
    still in progress on the same directory is not disturbed and nothing is written back.
    ``llm`` defaults to an offline client over the benchmark's cache; pass an online one for a
    world whose questions were never embedded.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        system: str = "anatid",
        llm: LLMClient | None = None,
        data_dir: Path | None = None,
    ) -> None:
        from anatid import Anatid

        self.run_dir = Path(run_dir)
        self.system = system
        data = data_dir or (self.run_dir / "data" if (self.run_dir / "data").exists() else DATA_DIR)
        self.notes = load_notes(data / "notes.jsonl")
        self.questions = load_questions(data / "questions.jsonl")
        self.llm = llm or LLMClient(offline=True, counter=TokenCounter(mode="chars4"))
        self.counter = self.llm.counter
        source = self.run_dir / system / "memory.anatid"
        if not source.exists():
            raise FileNotFoundError(
                f"no store at {source}; run the benchmark for {run_dir.name} first"
            )
        self._tmp = Path(tempfile.mkdtemp(prefix="anatid-coverage-"))
        store = self._tmp / "memory.anatid"
        shutil.copy(source, store)
        wal = Path(str(source) + ".wal")
        if wal.exists():
            # A run that is still open has not checkpointed: the writes since the last
            # checkpoint are in the write-ahead log, which DuckDB replays on open.
            shutil.copy(wal, Path(str(store) + ".wal"))
        self.db = Anatid.open(store, tenant=1, embedding_dim=self.llm.embed_dim)
        self.con = self.db.connection
        self.source_of = {n.note_id: n.source for n in self.notes}
        self.memories_of_source: dict[str, set[int]] = defaultdict(set)
        for src, mid in self.con.execute(
            "SELECT e.source, m.memory_id FROM memories m "
            "JOIN episodes e ON e.episode_id = m.episode_id WHERE m.tenant_id = 1"
        ).fetchall():
            self.memories_of_source[src].add(int(mid))
        self._arms: dict[str, dict[str, Any]] = {}
        self._rendered: dict[int, tuple[str, frozenset[int]]] = {}

    def close(self) -> None:
        self.db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def __enter__(self) -> CoverageLab:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ arms

    def arms(self, question: Question) -> dict[str, Any]:
        """The three raw arms for a question: vector and text pinned, graph newest first."""
        if question.qid in self._arms:
            return self._arms[question.qid]
        from anatid.recall import auto_seeds, bm25_arm, hybrid_recall, vector_arm

        dim = self.db.config.embedding_dim
        embedding = self.llm.embed([question.question])[0]
        vector = pin_ties(
            vector_arm(self.con, tenant_id=1, embedding=embedding, dim=dim, topn=RECALL_CANDIDATES)
        )
        text = pin_ties(
            bm25_arm(self.con, tenant_id=1, query_text=question.question, topn=RECALL_CANDIDATES)
        )
        seeds = auto_seeds(self.con, tenant_id=1, query=question.question)
        graph: list[tuple[int, float]] = []
        if seeds:
            hits = hybrid_recall(
                self.con,
                tenant_id=1,
                query=None,
                embedding=None,
                dim=dim,
                k=RECALL_CANDIDATES,
                candidates=RECALL_CANDIDATES,
                seed_entity=[sid for sid, _ in seeds],
                backend=self.db.csr,
                on_stale_fts="ignore",
            )
            graph = [(h.memory_id, float(len(hits) - i)) for i, h in enumerate(hits)]
        out = {
            "vector": vector,
            "text": text,
            "graph": graph,
            "embedding": embedding,
            "seeds": seeds,
        }
        self._arms[question.qid] = out
        return out

    def ranking(
        self,
        question: Question,
        variant: str = "product",
        *,
        weights: Mapping[str, float] | None = None,
        k: int = RECALL_K,
    ) -> list[int]:
        """Memory ids in fused order for one variant (see the module docstring)."""
        from anatid.recall import RRF_K, default_arm_weights, rank_graph_candidates, rrf_fuse

        a = self.arms(question)
        if variant in ("vector", "text", "graph"):
            return [mid for mid, _ in a[variant][:k]]
        lists = {"vector": a["vector"], "text": a["text"], "graph": a["graph"]}
        lists = {name: rows for name, rows in lists.items() if rows}
        if variant == "equal":
            fused = rrf_fuse(lists, k=RRF_K, top=k)
        elif variant in ("product", "weighted"):
            if "graph" in lists:
                lists["graph"] = rank_graph_candidates(
                    self.con,
                    lists["graph"],
                    tenant_id=1,
                    embedding=a["embedding"],
                    dim=self.db.config.embedding_dim,
                )
            used = default_arm_weights(lists)
            if weights:
                used.update({name: float(w) for name, w in weights.items() if name in lists})
            fused = rrf_fuse(lists, k=RRF_K, top=k, weights=used)
        else:
            raise ValueError(
                f"unknown variant {variant!r}; choose from {BUILTIN_VARIANTS} or 'weighted'"
            )
        return [mid for mid, _score, _ranks, _scores in fused]

    # ------------------------------------------------------------------ the block

    def rendered(self, memory_id: int) -> tuple[str, frozenset[int]]:
        """One memory as the anatid system renders it, and every memory id the text shows."""
        if memory_id not in self._rendered:
            memory = self.db.get(memory_id)
            if memory is None:
                self._rendered[memory_id] = ("", frozenset())
            else:
                chain = self.db.provenance(memory_id).chain
                head = "- " + _render_memory_line(memory, current=True)
                tail = [
                    f"    earlier: {_render_memory_line(old, current=False)}" for old in chain[1:]
                ]
                self._rendered[memory_id] = (
                    "\n".join([head, *tail]),
                    frozenset(m.memory_id for m in chain),
                )
        return self._rendered[memory_id]

    def block(
        self, ranking: Sequence[int], *, budget: int | None = BUDGET_TOKENS
    ) -> tuple[set[int], int]:
        """The memory ids the budgeted block shows, and how many lines it kept."""
        items = [self.rendered(mid) for mid in ranking]
        kept = fit_budget([text for text, _ in items if text], budget, self.counter)
        shown: set[int] = set()
        for _text, ids in items[: len(kept)]:
            shown |= ids
        return shown, len(kept)

    def covered(self, question: Question, shown: Iterable[int]) -> tuple[bool, float]:
        """Whether every supporting note has a memory in ``shown``, and the fraction that do."""
        if not question.support:
            return True, 1.0
        ids = set(shown)
        hits = [bool(self.memories_of_source[self.source_of[n]] & ids) for n in question.support]
        return all(hits), sum(hits) / len(hits)

    # ------------------------------------------------------------------ tables

    def evaluate(
        self,
        variant: str = "product",
        *,
        weights: Mapping[str, float] | None = None,
        budget: int | None = BUDGET_TOKENS,
    ) -> dict[str, tuple[bool, float, int]]:
        """``{qid: (covered, fraction, lines kept)}`` for every question."""
        out: dict[str, tuple[bool, float, int]] = {}
        for q in self.questions:
            shown, lines = self.block(self.ranking(q, variant, weights=weights), budget=budget)
            full, fraction = self.covered(q, shown)
            out[q.qid] = (full, fraction, lines)
        return out

    def summary(self, results: Mapping[str, tuple[bool, float, int]]) -> dict[str, float]:
        """Coverage over the questions with support notes, overall and per category."""
        answerable = [q for q in self.questions if q.support]
        out = {"all": _rate(results, answerable), "lines": _mean_lines(results, self.questions)}
        for category in CATEGORIES:
            out[category] = _rate(results, [q for q in answerable if q.category == category])
        return out

    def judge_verdicts(self, system: str | None = None) -> dict[str, bool] | None:
        """The LLM judge's verdict per question for ``system`` in this run, when scored."""
        path = self.run_dir / (system or self.system) / "scores.jsonl"
        if not path.exists():
            return None
        out: dict[str, bool] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["qid"]] = bool((row.get("judge") or {}).get("correct"))
        return out


def _rate(results: Mapping[str, tuple[bool, float, int]], questions: Sequence[Question]) -> float:
    if not questions:
        return float("nan")
    return sum(results[q.qid][0] for q in questions) / len(questions)


def _mean_lines(
    results: Mapping[str, tuple[bool, float, int]], questions: Sequence[Question]
) -> float:
    return sum(results[q.qid][2] for q in questions) / len(questions) if questions else 0.0


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def render_table(rows: Mapping[str, Mapping[str, float]]) -> str:
    head = (
        f"{'variant':<28} {'all':>5} " + " ".join(f"{c[:10]:>10}" for c in CATEGORIES) + "  lines"
    )
    lines = [head]
    for name, summary in rows.items():
        cells = " ".join(f"{100 * summary[c]:>9.0f}%" for c in CATEGORIES)
        lines.append(f"{name:<28} {100 * summary['all']:>4.0f}% {cells}  {summary['lines']:5.1f}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="a run id under bench/quality/results/")
    parser.add_argument("--system", default="anatid", help="anatid or anatid-gold")
    parser.add_argument("--variants", default=",".join(BUILTIN_VARIANTS))
    parser.add_argument("--weights", default=None, help="e.g. vector=1,text=0.25,graph=0.5")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument(
        "--cache-dir", default=None, help="the model cache (default: the benchmark's)"
    )
    parser.add_argument("--embed-dim", type=int, default=None, help="the store's embedding width")
    parser.add_argument("--online", action="store_true", help="embed uncached questions")
    parser.add_argument("--json", default=None, help="write per-question results here")
    args = parser.parse_args(argv)
    kwargs: dict[str, Any] = {"offline": not args.online, "counter": TokenCounter(mode="chars4")}
    if args.cache_dir:
        kwargs["cache_dir"] = Path(args.cache_dir)
    if args.embed_dim:
        kwargs["embed_dim"] = int(args.embed_dim)
    llm = LLMClient(**kwargs)
    run_dir = Path(args.results_dir) / args.run
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    with CoverageLab(run_dir, system=args.system, llm=llm) as lab:
        results: dict[str, dict[str, tuple[bool, float, int]]] = {}
        for variant in variants:
            results[variant] = lab.evaluate(variant)
        if args.weights:
            weights = parse_weights(args.weights)
            label = "weighted " + ",".join(f"{k}={v:g}" for k, v in weights.items())
            results[label] = lab.evaluate("weighted", weights=weights)
        answerable = sum(1 for q in lab.questions if q.support)
        _say(
            f"{args.run} / {args.system}: {len(lab.questions)} questions, {answerable} with "
            f"support notes; coverage = every supporting note has a memory in the block"
        )
        _say(render_table({name: lab.summary(rows) for name, rows in results.items()}))
        for sysname, variant in ((args.system, "product"), (f"{args.system}-vector", "vector")):
            verdict = lab.judge_verdicts(sysname)
            if verdict is None or variant not in results:
                continue
            rows = results[variant]
            covered = [q for q in lab.questions if q.support and rows[q.qid][0]]
            missed = [q for q in lab.questions if q.support and not rows[q.qid][0]]
            pc = (
                sum(verdict.get(q.qid, False) for q in covered) / len(covered)
                if covered
                else float("nan")
            )
            pm = (
                sum(verdict.get(q.qid, False) for q in missed) / len(missed)
                if missed
                else float("nan")
            )
            _say(
                f"{sysname}: judge-correct {100 * pc:.0f}% of the {len(covered)} covered "
                f"questions and {100 * pm:.0f}% of the {len(missed)} uncovered ones"
            )
        if args.json:
            Path(args.json).write_text(
                json.dumps(
                    {v: {q: list(r) for q, r in rows.items()} for v, rows in results.items()},
                    indent=1,
                ),
                encoding="utf-8",
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
