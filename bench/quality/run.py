"""One command runs the answer-quality benchmark end to end.

::

    python -m bench.quality.run                      # every system, default seed and budget
    python -m bench.quality.run --offline            # replay from the cache; no network
    python -m bench.quality.run --limit 10 --systems markdown,bm25   # a smoke run
    python -m bench.quality.run --seed 7                             # another world

``--seed`` names the world.  The committed ``data/`` is used when it was generated with that
seed; for any other seed the corpus is generated (and verified) into ``results/<run_id>/data``
first, so the run is about that world and its report says so.

The systems run one after another, never concurrently, so their timings do not disturb each
other.  Each is ingested once and asked every question; every prompt, context and answer is
saved under ``results/<run_id>/<system>/``.  Then the blind judge scores every answer (it sees
question, gold, rubric and answer only), ``results/<run_id>/summary.json`` is written, and
``results/REPORT.md`` is rendered from it.  With a warm cache the whole run costs nothing and
gives the same numbers; ``--offline`` refuses any network call to prove it.
"""

from __future__ import annotations

if __package__ in (None, ""):  # pragma: no cover - invoked as `python bench/quality/run.py`
    # Run as a file rather than as a module: make the repository root importable and take the
    # package's name, so the relative imports below resolve the same way they do under ``-m``.
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "bench.quality"

import argparse
import concurrent.futures
import datetime as _dt
import json
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import DATA_DIR
from .harness import (
    BUDGET_TOKENS,
    RECALL_K,
    RESULTS_DIR,
    SYSTEM_NAMES,
    BaseSystem,
    Question,
    SystemRun,
    default_systems,
    load_gold_patches,
    load_notes,
    load_questions,
    run_system,
)
from .judge import ScoredAnswer, lexical_score, llm_judge, summarise
from .llm import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_DIM,
    DEFAULT_EMBED_MODEL,
    LLMClient,
    OfflineCacheMiss,
    Stats,
    TokenCounter,
    load_api_key,
)
from .report import write_report

__all__ = ["DEFAULT_SEED", "assemble_summary", "corpus_for_seed", "main", "score_runs"]

DEFAULT_SEED = 20260905


def _say(text: str) -> None:
    sys.stderr.write(text + "\n")
    sys.stderr.flush()


# --------------------------------------------------------------------------- the corpus


def _committed_seed(data_dir: Path) -> int | None:
    """The seed the corpus in ``data_dir`` was generated with, from its ``counts.json``."""
    try:
        return int(json.loads((data_dir / "counts.json").read_text(encoding="utf-8"))["seed"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def corpus_for_seed(seed: int, run_dir: Path, *, verify: bool = True) -> tuple[Path, bool]:
    """The data directory holding the corpus for ``seed``, and whether it was generated now.

    The seed names the world (see ``bench.quality.gen_corpus``): the committed ``data/`` is
    used when it was generated with ``seed``, otherwise the corpus for ``seed`` is generated
    into ``run_dir/data`` with ``gen_corpus`` (and, with ``verify``, checked against anatid the
    way ``gen_corpus --verify`` does), so a run labelled with a seed answers questions about
    that seed's world and nothing else.  Generation is deterministic, so a rerun writes the same
    bytes.
    """
    if _committed_seed(DATA_DIR) == int(seed):
        return DATA_DIR, False
    from .gen_corpus import main as gen_main

    out = run_dir / "data"
    argv = ["--seed", str(int(seed)), "--out", str(out)]
    if verify:
        argv.append("--verify")
    if gen_main(argv) != 0:
        raise RuntimeError(f"gen_corpus could not produce a verified corpus for seed {seed}")
    if _committed_seed(out) != int(seed):
        raise RuntimeError(f"gen_corpus wrote {out} without recording seed {seed}")
    return out, True


# --------------------------------------------------------------------------- scoring


def score_runs(
    runs: Sequence[SystemRun],
    questions: Sequence[Question],
    *,
    llm: LLMClient | None,
    workers: int = 4,
    progress: bool = True,
) -> dict[str, list[ScoredAnswer]]:
    """Score every system's answers with both scorers; the judge sees no system name.

    The judge calls are independent and untimed, so they run in a thread pool; the systems
    themselves ran one at a time.  Returns ``{system name: [ScoredAnswer, ...]}`` in question
    order.
    """
    by_qid = {q.qid: q for q in questions}
    lexical: dict[tuple[int, str], Any] = {}
    jobs: list[tuple[int, str, Question, str]] = []
    for i, run in enumerate(runs):
        answers = run.answers_by_qid
        for q in questions:
            answer = str(answers.get(q.qid, ""))
            lexical[(i, q.qid)] = lexical_score(q, answer)
            jobs.append((i, q.qid, q, answer))

    verdicts: dict[tuple[int, str], Any] = {}
    if llm is not None and jobs:
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(llm_judge, llm, q, answer): (i, qid) for i, qid, q, answer in jobs
            }
            for future in concurrent.futures.as_completed(futures):
                verdicts[futures[future]] = future.result()
                done += 1
                if progress and (done % 100 == 0 or done == len(jobs)):
                    _say(f"  judge {done}/{len(jobs)}")

    out: dict[str, list[ScoredAnswer]] = {}
    for i, run in enumerate(runs):
        answers = run.answers_by_qid
        scored = [
            ScoredAnswer(
                qid=q.qid,
                category=q.category,
                subtype=q.subtype,
                answer=str(answers.get(q.qid, "")),
                lexical=lexical[(i, q.qid)],
                judge=verdicts.get((i, q.qid)),
            )
            for q in questions
            if q.qid in by_qid
        ]
        out[run.name] = scored
    return out


# --------------------------------------------------------------------------- the summary


def _system_entry(
    run: SystemRun, scored: Sequence[ScoredAnswer], questions: Sequence[Question]
) -> dict[str, Any]:
    by_qid = {q.qid: q for q in questions}
    answers = {a.qid: a for a in run.answers}
    per_question = []
    for s in scored:
        q = by_qid[s.qid]
        a = answers.get(s.qid)
        entry = s.to_dict()
        entry.update(
            {
                "question": q.question,
                "gold": q.gold,
                "context_tokens": a.context_tokens if a else 0,
                "n_contexts": a.n_contexts if a else 0,
                "retrieve_s": a.retrieve_s if a else 0.0,
                "answer_s": a.answer_s if a else 0.0,
                "answer_network_s": a.answer_network_s if a else 0.0,
                "answer_cached": a.answer_cached if a else True,
                "cost_usd": a.cost_usd if a else None,
                "retrieval": a.retrieval if a else {},
            }
        )
        per_question.append(entry)
    entry = run.to_dict()
    entry.pop("answers", None)
    entry.update(
        {
            "scores": summarise(scored),
            "per_question": per_question,
            "mean_context_tokens": _mean([a.context_tokens for a in run.answers]),
            "mean_contexts": _mean([a.n_contexts for a in run.answers]),
            "mean_answer_s": _mean([a.answer_network_s for a in run.answers]),
            "mean_answer_wall_s": _mean([a.answer_s for a in run.answers]),
            "answers_from_cache": sum(a.answer_cached for a in run.answers),
            "escalated_answers": sum(a.escalated for a in run.answers),
            "truncated_answers": sum(a.truncated for a in run.answers),
            "retrieve_s_total": sum(a.retrieve_s for a in run.answers),
            "answer_s_total": sum(a.answer_network_s for a in run.answers),
            "answer_wall_s_total": sum(a.answer_s for a in run.answers),
            "ingest_network_s": run.ingest_stats.get("recorded_wall_s"),
            "cost_usd_answers": run.answer_stats.get("cost_usd"),
            "cost_usd_ingest": run.ingest_stats.get("cost_usd"),
        }
    )
    return entry


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _relative(path: Path) -> Path:
    """``path`` relative to the repository root when it is inside it (for the summary)."""
    root = DATA_DIR.parents[2]
    try:
        return path.resolve().relative_to(root)
    except ValueError:
        return path


def assemble_summary(
    runs: Sequence[SystemRun],
    scored: Mapping[str, Sequence[ScoredAnswer]],
    questions: Sequence[Question],
    *,
    run_id: str,
    seed: int,
    budget: int,
    llm: LLMClient,
    family: Mapping[str, str],
    judge_stats: Stats | None,
    judge_workers: int,
    started_at: str,
    wall_s: float,
    notes_count: int,
    offline: bool,
    limit: int | None,
    corpus: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The one JSON document the report is rendered from.  ``corpus`` may carry where the
    data came from (``seed``, ``data_dir``, ``generated``)."""
    by_category: dict[str, int] = {}
    for q in questions:
        by_category[q.category] = by_category.get(q.category, 0) + 1
    return {
        "run_id": run_id,
        "seed": seed,
        "budget": budget,
        "tokenizer": llm.counter.name,
        "family": dict(family),
        "chat_model": llm.chat_model,
        "embed_model": llm.embed_model,
        "embed_dim": llm.embed_dim,
        "started_at": started_at,
        "finished_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "wall_s": wall_s,
        "offline": offline,
        "limit": limit,
        "corpus": {
            **dict(corpus or {}),
            "notes": notes_count,
            "questions": len(questions),
            "by_category": by_category,
        },
        "llm_totals": llm.stats.to_dict(),
        "judge_stats": judge_stats.to_dict() if judge_stats is not None else {},
        "judge_workers": judge_workers,
        "systems": [_system_entry(run, scored.get(run.name, []), questions) for run in runs],
        "questions": [q.to_dict() for q in questions],
    }


# --------------------------------------------------------------------------- the command


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-id", default=None, help="results/<run-id>/; default seed-budget")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--budget", type=int, default=BUDGET_TOKENS, help="memory tokens")
    parser.add_argument(
        "--counter",
        choices=TokenCounter.MODES,
        default="auto",
        help="how tokens are counted against the budget: tiktoken cl100k_base when importable "
        "(auto), always the four-characters-per-token estimate (chars4), or tiktoken and fail "
        "without it; the report's reproduce command pins the one the run used",
    )
    parser.add_argument(
        "--systems",
        default=",".join(SYSTEM_NAMES),
        help="comma-separated subset of " + ", ".join(SYSTEM_NAMES),
    )
    parser.add_argument(
        "--family",
        choices=("auto", "builtin", "external"),
        default="auto",
        help="built-in systems, the bench.quality.systems package, or whichever is available",
    )
    parser.add_argument("--k", type=int, default=RECALL_K, help="anatid recall hits before the cut")
    parser.add_argument("--limit", type=int, default=None, help="first N questions only (smoke)")
    parser.add_argument("--offline", action="store_true", help="cache only; no network")
    parser.add_argument("--no-judge", action="store_true", help="lexical scorer only")
    parser.add_argument("--judge-workers", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true", help="rebuild persisted anatid stores")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    random.seed(args.seed)

    run_id = args.run_id or f"s{args.seed}-b{args.budget}" + (
        f"-first{args.limit}" if args.limit else ""
    )
    results_dir = Path(args.results_dir)
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.offline:
        load_api_key()
    kwargs: dict[str, Any] = {"offline": args.offline, "counter": TokenCounter(mode=args.counter)}
    if args.cache_dir:
        kwargs["cache_dir"] = Path(args.cache_dir)
    llm = LLMClient(
        chat_model=DEFAULT_CHAT_MODEL,
        embed_model=DEFAULT_EMBED_MODEL,
        embed_dim=DEFAULT_EMBED_DIM,
        **kwargs,
    )

    try:
        data_dir, generated = corpus_for_seed(args.seed, run_dir)
    except RuntimeError as exc:
        _say(str(exc))
        return 2
    if generated:
        _say(f"seed {args.seed} is not the committed corpus; generated and verified {data_dir}")
    notes = load_notes(data_dir / "notes.jsonl")
    questions = load_questions(data_dir / "questions.jsonl")
    gold = load_gold_patches(data_dir / "gold_patches.jsonl")
    corpus_info = {
        "seed": args.seed,
        "data_dir": str(_relative(data_dir)),
        "generated": generated,
    }
    if args.limit:
        questions = questions[: args.limit]
    wanted = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = sorted(set(wanted) - set(SYSTEM_NAMES))
    if unknown:
        _say(f"unknown systems {unknown}; choose from {', '.join(SYSTEM_NAMES)}")
        return 2

    systems, family = default_systems(
        llm,
        family=args.family,
        notes=notes,
        gold_patches=gold,
        store_dir=run_dir / "stores",
        k=args.k,
        rebuild=args.rebuild,
    )
    chosen: list[BaseSystem] = [s for s in systems if s.name in wanted]
    # An ablation needs the store it shares; pull its owner in ahead of it when it was left out.
    for s in list(chosen):
        owner = s.shares_store_with
        if owner and owner not in [c.name for c in chosen]:
            owner_sys = next(x for x in systems if x.name == owner)
            chosen.insert(chosen.index(s), owner_sys)
            _say(f"{s.name} shares {owner}'s store; {owner} runs as well")
    chosen.sort(key=lambda s: SYSTEM_NAMES.index(s.name))

    _say(
        f"run {run_id}: {len(notes)} notes, {len(questions)} questions, budget {args.budget}, "
        f"counter {llm.counter.name}, systems {family['family']} ({family['note']})"
    )
    started_at = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    runs: list[SystemRun] = []

    def progress(name: str, i: int, n: int) -> None:
        if i % 25 == 0 or i == n:
            _say(f"  {name}: {i}/{n}")

    try:
        for system in chosen:
            _say(f"[{system.name}] ingest")
            run = run_system(
                system,
                notes,
                questions,
                budget=args.budget,
                out_dir=run_dir / system.name,
                llm=llm,
                progress=progress,
            )
            runs.append(run)
            _say(
                f"[{system.name}] ingest {run.ingest_s:.1f}s; answers "
                f"{run.answer_stats.get('network_calls', 0)} network calls, "
                f"{run.answer_stats.get('cache_hits', 0)} cached"
            )
        _say("judging" if not args.no_judge else "scoring (lexical only)")
        judge_before = llm.snapshot()
        scored = score_runs(
            runs,
            questions,
            llm=None if args.no_judge else llm,
            workers=args.judge_workers,
        )
        judge_stats = None if args.no_judge else llm.stats.diff(judge_before)
    except OfflineCacheMiss as exc:
        _say(f"offline run stopped: {exc}")
        return 3
    finally:
        for system in chosen:
            try:
                system.close()
            except Exception as exc:  # noqa: BLE001 - closing is best effort
                _say(f"{system.name}: close failed: {exc}")

    summary = assemble_summary(
        runs,
        scored,
        questions,
        run_id=run_id,
        seed=args.seed,
        budget=args.budget,
        llm=llm,
        family=family,
        judge_stats=judge_stats,
        judge_workers=args.judge_workers,
        started_at=started_at,
        wall_s=time.perf_counter() - t0,
        notes_count=len(notes),
        offline=args.offline,
        limit=args.limit,
        corpus=corpus_info,
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
    )
    for name, rows in scored.items():
        with (run_dir / name / "scores.jsonl").open("w", encoding="utf-8") as fh:
            for s in rows:
                fh.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    report = write_report(summary, results_dir / "REPORT.md", copy_to=run_dir / "REPORT.md")
    totals = llm.stats
    _say(
        f"done in {summary['wall_s']:.0f}s: {totals.calls} model calls, {totals.cache_hits} from "
        f"the cache ({100 * totals.hit_rate:.0f}%), charged ${totals.cost_usd_charged:.4f}; "
        f"summary {run_dir / 'summary.json'}; report {report}"
    )
    for s in summary["systems"]:
        overall = s["scores"]["by_category"].get("all", {})
        llm_acc = overall.get("llm")
        lex = overall.get("lexical")
        _say(
            f"  {s['name']:<14} judge {'n/a' if llm_acc is None else f'{100 * llm_acc:.0f}%':>4}  "
            f"lexical {'n/a' if lex is None else f'{100 * lex:.0f}%':>4}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
