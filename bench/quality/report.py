"""Render ``results/REPORT.md`` from a run's ``summary.json``.

The report states what was measured and how, leads with its limitations, and then gives the
numbers: accuracy per category per system under both scorers, abstention precision, stale
answers, context tokens, latency and cost, the gap between the model-extracted store and the
gold-extracted one as the cost of extraction, and the questions anatid lost next to the ones
it won.  Every number comes from ``summary.json``; nothing is typed in by hand.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .harness import BUDGET_TOKENS, RESULTS_DIR

__all__ = ["CODES", "CATEGORY_ORDER", "render_report", "write_report"]

#: The short codes the benchmark plan gave the systems.
CODES: dict[str, str] = {
    "markdown": "S1",
    "markdown-full": "S1, no budget",
    "bm25": "S2",
    "vector": "S3",
    "hybrid": "S2+S3",
    "vector-prf": "S3, feedback",
    "anatid": "S4",
    "anatid-text": "S4t",
    "anatid-vector": "S4v",
    "anatid-graph": "S4g",
    "anatid-gold": "S5, oracle",
}

CATEGORY_ORDER = (
    "single_fact",
    "knowledge_update",
    "temporal",
    "multi_hop",
    "provenance",
    "abstention",
)

#: The budgeted baselines anatid is compared with question by question.
BASELINES = ("markdown", "bm25", "vector", "hybrid", "vector-prf")


# --------------------------------------------------------------------------- helpers


def _pct(value: float | None, *, digits: int = 0) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.{digits}f}%"


def _num(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _usd(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "$0.00"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in header) + " |"]
    lines.append("|" + "|".join(" ---: " if i else " --- " for i in range(len(header))) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _label(name: str) -> str:
    code = CODES.get(name)
    return f"{name} ({code})" if code else name


def _by_name(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {s["name"]: s for s in summary.get("systems", [])}


def _cat_score(system: Mapping[str, Any], category: str, scorer: str) -> float | None:
    by_cat = system.get("scores", {}).get("by_category", {})
    return by_cat.get(category, {}).get(scorer)


def _categories(summary: Mapping[str, Any]) -> list[str]:
    present: set[str] = set()
    for s in summary.get("systems", []):
        present.update(k for k in s.get("scores", {}).get("by_category", {}) if k != "all")
    ordered = [c for c in CATEGORY_ORDER if c in present]
    ordered += sorted(present - set(CATEGORY_ORDER))
    return ordered


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


# --------------------------------------------------------------------------- sections


def _intro(summary: Mapping[str, Any]) -> str:
    corpus = summary.get("corpus", {})
    return "\n".join(
        [
            "# Answer-quality benchmark",
            "",
            (
                "Does an agent answer better from anatid than from the obvious alternatives, with the "
                "same model, the same prompt, the same questions and the same memory budget? The Phase "
                "0 benchmark (`docs/benchmarks.md`) measures storage speed and structural correctness "
                "and says nothing about answers. This report measures answers. The losses are "
                "published next to the wins."
            ),
            "",
            (
                f"Run `{summary.get('run_id', '?')}`, seed {summary.get('seed', '?')}"
                + (
                    f" (corpus generated for this seed into `{corpus.get('data_dir')}`)"
                    if corpus.get("generated")
                    else ""
                )
                + f", {corpus.get('notes', '?')} notes, {corpus.get('questions', '?')} questions, memory "
                f"budget {summary.get('budget', BUDGET_TOKENS)} tokens "
                f"(counter: `{summary.get('tokenizer', '?')}`). Answering and judging model "
                f"`{summary.get('chat_model', '?')}` at temperature 0; embeddings "
                f"`{summary.get('embed_model', '?')}` ({summary.get('embed_dim', '?')} dimensions). "
                f"Systems: {summary.get('family', {}).get('family', '?')} "
                f"({summary.get('family', {}).get('note', '')})."
            ),
        ]
    )


def _limitations(summary: Mapping[str, Any]) -> str:
    corpus = summary.get("corpus", {})
    n_q = corpus.get("questions", 150)
    n_n = corpus.get("notes", 178)
    by_cat = corpus.get("by_category") or {}
    per_cat = min(by_cat.values()) if by_cat else 0
    return "\n".join(
        [
            "## Limitations, first",
            "",
            (
                "1. The corpus is synthetic and generated by us (`bench/quality/gen_corpus.py`). Its "
                "phrasing comes from a few dozen templates, its entities are few, and its notes are "
                "short and clean. Real notes are longer, messier and less consistent about names."
            ),
            (
                "2. The judge is the same model family as the answerer, and the same model. A judge "
                "that shares the answerer's blind spots can agree with a wrong answer. The lexical "
                "scorer is reported next to it so the two can be compared, and their agreement rate is "
                "given per system."
            ),
            (
                f"3. n is small: {n_q} questions, {per_cat or 'a handful'} per category, over "
                f"{n_n} notes. A difference of one question in a category is "
                f"{_pct(1 / per_cat if per_cat else None)} of that category. Differences of a "
                "few questions between systems are noise."
            ),
            (
                "4. The budget is what makes the comparison a comparison. At this corpus size the whole "
                "file fits in a prompt, so the unbudgeted Markdown variant is the upper bound of "
                '"just put it in the prompt" and is expected to do well. That variant would not '
                "exist at ten times the corpus, and nothing here says how any system behaves there."
            ),
            (
                "5. The anatid systems render each memory with its validity interval, its writer and "
                "the versions it superseded. The baselines get the raw dated notes. Both carry the "
                "date and the source label, so both can answer temporal and provenance questions; "
                "what differs is how much of the history fits in the budget."
            ),
            (
                "6. Every model call is cached on disk and the rerun is deterministic, so every number "
                "here is reproducible from one command. The model itself is not deterministic across "
                "provider changes: a cold rerun on a later day may answer differently."
            ),
            (
                "7. The gold-extracted store (S5) is an oracle. It says what anatid's retrieval can do "
                "when extraction is perfect. It is not the product; the product is S4."
            ),
            (
                "8. The anatid systems call anatid's three retrieval arms and anatid's rank fusion "
                "through the harness (`fused_recall`) rather than `db.recall` directly, because the "
                "BM25 arm returns documents with equal scores in an order that varies between calls "
                "(the floating point sum inside DuckDB), and a varying order is a varying prompt. The "
                "harness rounds arm scores to six decimals and breaks ties by memory id, and does the "
                "same for the BM25 and vector baselines. Ranking is otherwise unchanged; the "
                "extractor's view of existing facts is built the same way for the same reason."
            ),
            (
                "9. One seed is one world. The generator writes a different organisation for every "
                "seed, with the same shape, and the order of the systems is not the same in every "
                "world we ran. A difference seen in one run is a hypothesis until another seed "
                "shows it too; `docs/quality.md` reports two."
            ),
            (
                "10. The fused baselines (`hybrid`, `vector-prf`) exist so that anatid's fusion of "
                "three arms is compared with a baseline that also fuses, and so that a second "
                "retrieval round, the cheap remedy for multi-hop questions, is on the table next to "
                "the graph arm. Both use the same fusion constant anatid uses; neither is tuned."
            ),
        ]
    )


def _headline(summary: Mapping[str, Any]) -> str:
    rows = []
    for s in summary.get("systems", []):
        scores = s.get("scores", {})
        overall = scores.get("by_category", {}).get("all", {})
        abst = scores.get("abstention", {})
        stale = scores.get("stale", {})
        rows.append(
            [
                _label(s["name"]),
                _pct(overall.get("llm")),
                _pct(overall.get("lexical")),
                _pct(overall.get("agreement")),
                _pct(abst.get("precision")),
                _pct(abst.get("false_refusal_rate")),
                _pct(stale.get("stale_rate")),
                _num(s.get("mean_context_tokens"), 0),
                _num(s.get("mean_answer_s"), 2),
                _usd(s.get("cost_usd_answers")),
            ]
        )
    return "\n".join(
        [
            "## Headline",
            "",
            (
                "Accuracy over all questions under both scorers, with the abstention and staleness "
                'measures that the accuracy alone hides. "Stale" is an answer to a knowledge-update '
                "or temporal question that names a value that was once true instead of the gold."
            ),
            "",
            _table(
                [
                    "system",
                    "LLM judge",
                    "lexical",
                    "agreement",
                    "abstention precision",
                    "false refusals",
                    "stale answers",
                    "context tokens",
                    "answer latency s",
                    "answer cost",
                ],
                rows,
            ),
            "",
            (
                "Context tokens and latency are means per question. Latency is the network time of "
                "the answer call, recorded when the call was first made, so a replay from the cache "
                "reports the same number. Answer cost is the sum over the questions, at the "
                "provider's reported price."
            ),
        ]
    )


def _per_category(summary: Mapping[str, Any], scorer: str, title: str) -> str:
    cats = _categories(summary)
    header = ["system", *cats, "all"]
    rows = []
    for s in summary.get("systems", []):
        rows.append(
            [_label(s["name"])]
            + [_pct(_cat_score(s, c, scorer)) for c in cats]
            + [_pct(_cat_score(s, "all", scorer))]
        )
    return "\n".join([f"### {title}", "", _table(header, rows)])


def _abstention(summary: Mapping[str, Any]) -> str:
    rows = []
    for s in summary.get("systems", []):
        a = s.get("scores", {}).get("abstention", {})
        rows.append(
            [
                _label(s["name"]),
                a.get("refusals_total", 0),
                _pct(a.get("precision")),
                _pct(a.get("recall")),
                a.get("false_refusals", 0),
                _pct(a.get("false_refusal_rate")),
            ]
        )
    return "\n".join(
        [
            "## Knowing when to stop",
            "",
            (
                "A quarter of the questions have no answer in the notes. A system that always answers "
                "scores zero there; a system that refuses too readily loses everywhere else. Precision "
                "is the share of refusals that were right to refuse; recall is the share of "
                "unanswerable questions that were refused; false refusals are refusals on questions "
                "the notes do answer."
            ),
            "",
            _table(
                [
                    "system",
                    "refusals",
                    "precision",
                    "recall",
                    "false refusals",
                    "false refusal rate",
                ],
                rows,
            ),
        ]
    )


def _extraction_cost(summary: Mapping[str, Any]) -> str:
    systems = _by_name(summary)
    s4, s5 = systems.get("anatid"), systems.get("anatid-gold")
    if s4 is None or s5 is None:
        return ""
    cats = _categories(summary)
    rows = []
    for c in [*cats, "all"]:
        a, b = _cat_score(s4, c, "llm"), _cat_score(s5, c, "llm")
        gap = None if a is None or b is None else b - a
        rows.append([c, _pct(a), _pct(b), ("+" if gap and gap > 0 else "") + _pct(gap)])
    r4, r5 = s4.get("ingest_report", {}), s5.get("ingest_report", {})

    def field(report: Mapping[str, Any], key: str) -> Any:
        if key in report:
            return report[key]
        store = report.get("store_stats", {})
        return store.get(key, "n/a")

    build = _table(
        [
            "store",
            "memories created",
            "superseded",
            "downgraded corrections",
            "dedupe drops",
            "failures",
        ],
        [
            [
                _label("anatid"),
                field(r4, "memories_created"),
                field(r4, "memories_closed"),
                field(r4, "downgraded_corrections"),
                field(r4, "dedupe_drops"),
                field(r4, "extraction_failures"),
            ],
            [
                _label("anatid-gold"),
                field(r5, "memories_created"),
                field(r5, "memories_closed"),
                field(r5, "downgraded_corrections"),
                field(r5, "dedupe_drops"),
                field(r5, "extraction_failures"),
            ],
        ],
    )
    return "\n".join(
        [
            "## The cost of extraction: S4 against the S5 oracle",
            "",
            (
                "S4 and S5 share the pipeline, the embedder and the retrieval; they differ only in "
                "who wrote the patches. The model wrote S4's; the corpus generator wrote S5's, so S5 "
                "is what anatid's retrieval does when extraction is perfect. The gap is what the "
                "model's extraction costs on this corpus, under the LLM judge."
            ),
            "",
            _table(["category", "S4 anatid", "S5 anatid-gold (oracle)", "gap"], rows),
            "",
            "What the two builds did:",
            "",
            build,
            "",
            (
                "A downgraded correction is one the model proposed against a memory the pipeline could "
                "not find; it lands as a new fact next to the old one instead of replacing it, which is "
                "the usual way a stale value survives into an answer."
            ),
        ]
    )


def _arms(summary: Mapping[str, Any]) -> str:
    systems = _by_name(summary)
    names = [n for n in ("anatid", "anatid-text", "anatid-vector", "anatid-graph") if n in systems]
    if len(names) < 2:
        return ""
    cats = _categories(summary)
    rows = [
        [_label(n)]
        + [_pct(_cat_score(systems[n], c, "llm")) for c in cats]
        + [_pct(_cat_score(systems[n], "all", "llm"))]
        for n in names
    ]
    return "\n".join(
        [
            "## Which arm earns its place",
            "",
            (
                "The same S4 store read through one recall arm at a time, LLM judge. The fused "
                "system is the first row."
            ),
            "",
            _table(["system", *cats, "all"], rows),
        ]
    )


def _wins_and_losses(summary: Mapping[str, Any]) -> str:
    systems = _by_name(summary)
    s4 = systems.get("anatid")
    baselines = [systems[b] for b in BASELINES if b in systems]
    if s4 is None or not baselines:
        return ""
    questions = {q["qid"]: q for q in summary.get("questions", [])}

    def verdict(system: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        return {p["qid"]: p for p in system.get("per_question", [])}

    v4 = verdict(s4)
    vb = {b["name"]: verdict(b) for b in baselines}

    def correct(p: Mapping[str, Any] | None) -> bool:
        if p is None:
            return False
        j = p.get("judge")
        return bool(j["correct"]) if j else bool(p.get("lexical", {}).get("correct"))

    wins, losses = [], []
    for qid, p4 in v4.items():
        c4 = correct(p4)
        cb = {name: correct(v.get(qid)) for name, v in vb.items()}
        if c4 and not any(cb.values()):
            wins.append(qid)
        elif not c4 and any(cb.values()):
            losses.append(qid)

    # categories where S4 trails the best budgeted baseline
    cats = _categories(summary)
    trailing = []
    for c in cats:
        a = _cat_score(s4, c, "llm")
        best_name, best = None, None
        for b in baselines:
            v = _cat_score(b, c, "llm")
            if v is not None and (best is None or v > best):
                best_name, best = b["name"], v
        if a is not None and best is not None and best > a:
            trailing.append(
                f"{c}: {_label('anatid')} {_pct(a)} against {_label(best_name or '')} {_pct(best)}"
            )

    def rows(qids: Sequence[str], limit: int = 12) -> str:
        out = []
        for qid in qids[:limit]:
            q = questions.get(qid, {})
            p4 = v4.get(qid, {})
            others = "; ".join(
                f"{CODES.get(n, n)}: {vb[n].get(qid, {}).get('answer', '')[:60]}" for n in vb
            )
            out.append(
                [
                    qid,
                    q.get("category", ""),
                    q.get("question", "")[:80],
                    q.get("gold", "")[:50],
                    p4.get("answer", "")[:60],
                    others,
                ]
            )
        return _table(["qid", "category", "question", "gold", "S4 answer", "baselines"], out)

    present = [b for b in BASELINES if b in systems]
    parts = [
        "## Where anatid lost, and where it won",
        "",
        (
            f"Against the {len(present)} budgeted baselines ({', '.join(_label(b) for b in present)}), "
            f"under the LLM judge: S4 alone right on {len(wins)} question{'s' if len(wins) != 1 else ''}; "
            f"S4 wrong where at least one baseline was right on {len(losses)} "
            f"question{'s' if len(losses) != 1 else ''}. A win is a question no baseline got; a "
            "loss is one at least one baseline got. Both lists are complete in `summary.json`."
        ),
        "",
    ]
    if trailing:
        parts += ["Categories where S4 trails the best budgeted baseline:", ""]
        parts += [f"- {t}" for t in trailing]
        parts.append("")
    else:
        parts += ["S4 is at or above the best budgeted baseline in every category.", ""]
    parts += [f"### Losses ({len(losses)}, first {min(12, len(losses))} shown)", ""]
    parts.append(rows(losses) if losses else "None.")
    parts += ["", f"### Wins ({len(wins)}, first {min(12, len(wins))} shown)", ""]
    parts.append(rows(wins) if wins else "None.")
    return "\n".join(parts)


def _cost_and_time(summary: Mapping[str, Any]) -> str:
    rows = []
    for s in summary.get("systems", []):
        ing = s.get("ingest_stats", {})
        ans = s.get("answer_stats", {})
        shared = s.get("shares_store_with")
        rows.append(
            [
                _label(s["name"]),
                "shares " + shared if shared else _num(s.get("ingest_s"), 1),
                "" if shared else _num(s.get("ingest_network_s"), 1),
                "" if shared else _usd(ing.get("cost_usd")),
                "" if shared else ing.get("calls", 0),
                _num(s.get("retrieve_s_total"), 2),
                _num(s.get("answer_s_total"), 1),
                _usd(ans.get("cost_usd")),
                f"{s.get('answers_from_cache', 0)}/{len(s.get('per_question', []))}",
                s.get("truncated_answers", 0),
            ]
        )
    totals = summary.get("llm_totals", {})
    judge = summary.get("judge_stats", {})
    return "\n".join(
        [
            "## Cost and time",
            "",
            (
                "Ingest is the one-off cost of building the memory (extraction calls and embeddings "
                "for anatid, embeddings for the vector baseline, nothing for Markdown and BM25). "
                "Retrieval is the time to pick the context for all questions; for the systems that "
                "embed the question (vector and the anatid systems with a vector arm) it includes "
                "that embedding call. Answering is the network time of the answer calls. Network "
                "time and cost are recorded with each call when it is first made, so a replay from "
                "the cache reports the same numbers and charges nothing. The model reasons "
                "before it answers; an answer that came back empty because the reasoning used the "
                "whole completion cap was asked once more with a cap three times larger, and "
                '"truncated answers" counts those that were still empty (scored as wrong).'
            ),
            "",
            _table(
                [
                    "system",
                    "ingest s (this run)",
                    "ingest network s (recorded)",
                    "ingest cost",
                    "ingest model calls",
                    "retrieval s (all questions)",
                    "answering network s (recorded)",
                    "answer cost",
                    "answers from cache",
                    "truncated answers",
                ],
                rows,
            ),
            "",
            (
                f"Whole run: {_num(summary.get('wall_s'), 0)} s wall time; {totals.get('calls', 0)} model "
                f"calls of which {totals.get('cache_hits', 0)} came from the cache "
                f"({_pct(totals.get('hit_rate'))}); recorded cost {_usd(totals.get('cost_usd'))}, of "
                f"which {_usd(totals.get('cost_usd_charged'))} was charged in this run. The judge made "
                f"{judge.get('chat_calls', 0)} calls costing {_usd(judge.get('cost_usd'))}."
            ),
        ]
    )


def _reproduce(summary: Mapping[str, Any]) -> str:
    corpus = summary.get("corpus", {})
    seed = summary.get("seed", 20260905)
    run_id = summary.get("run_id", "run")
    generated = bool(corpus.get("generated"))
    tokenizer = str(summary.get("tokenizer", "chars/4"))
    counter = "tiktoken" if tokenizer.startswith("tiktoken") else "chars4"
    run_cmd = (
        f"python -m bench.quality.run --run-id {run_id} --seed {seed} "
        f"--budget {summary.get('budget', BUDGET_TOKENS)} --counter {counter} "
        f"--family {summary.get('family', {}).get('family', 'auto')}"
    )
    if generated:
        commands = [run_cmd]
        corpus_note = (
            f"Seed {seed} is not the committed corpus, so the command first generates the "
            f"corpus for that seed into `{corpus.get('data_dir', f'bench/quality/results/{run_id}/data')}` "
            "and checks it against anatid the way `gen_corpus --verify` does (`--seed` names "
            "the world; the committed `data/` is used only when it was generated with the seed "
            "asked for). Generation is deterministic, so a rerun writes the same corpus."
        )
        which = "The command"
    else:
        commands = ["python -m bench.quality.gen_corpus --verify", run_cmd]
        corpus_note = ""
        which = "The second command"
    return "\n".join(
        [
            "## Reproduce",
            "",
            "```",
            *commands,
            "```",
            "",
            *([corpus_note, ""] if corpus_note else []),
            (
                f"{which} reads the key from `OPEN_ROUTER_KEY` or `.env`, caches every model "
                "call under `bench/quality/cache/`, writes every prompt, context and answer under "
                f"`bench/quality/results/{run_id}/<system>/`, and renders this "
                "file. Add `--offline` to prove a rerun needs no network: with a warm cache it "
                f"completes with zero charged cost and the same numbers. `--counter {counter}` "
                f"pins the token counter this run used (`{tokenizer}`); a different counter is a "
                "different budget, different prompts and a cold cache."
            ),
        ]
    )


# --------------------------------------------------------------------------- the report


def render_report(summary: Mapping[str, Any]) -> str:
    """The whole report as Markdown."""
    sections = [
        _intro(summary),
        _limitations(summary),
        _headline(summary),
        "## Accuracy per category",
        "",
        (
            "Two scorers. The LLM judge sees the question, the gold, the accepted aliases, the "
            "category's rubric and the answer, and nothing about where the answer came from. The "
            "lexical scorer is a normalised whole-word match of the gold or an alias, with a stale "
            "value named before the gold counted wrong and a refusal counted right only for "
            "abstention questions."
        ),
        "",
        _per_category(summary, "llm", "LLM judge"),
        "",
        _per_category(summary, "lexical", "Lexical match"),
        "",
        _per_category(summary, "agreement", "Agreement between the two scorers"),
        _abstention(summary),
        _extraction_cost(summary),
        _arms(summary),
        _wins_and_losses(summary),
        _cost_and_time(summary),
        _reproduce(summary),
    ]
    return "\n\n".join(s for s in sections if s) + "\n"


def write_report(
    summary: Mapping[str, Any], path: Path | None = None, *, copy_to: Path | None = None
) -> Path:
    """Render and write the report; ``copy_to`` writes a second copy (the run directory)."""
    target = path or (RESULTS_DIR / "REPORT.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    text = render_report(summary)
    target.write_text(text, encoding="utf-8")
    if copy_to is not None:
        copy_to.parent.mkdir(parents=True, exist_ok=True)
        copy_to.write_text(text, encoding="utf-8")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m bench.quality.report <summary.json> [<REPORT.md>]``."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        sys.stderr.write("usage: python -m bench.quality.report <summary.json> [<REPORT.md>]\n")
        return 2
    summary = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    out = write_report(summary, Path(args[1]) if len(args) > 1 else None)
    sys.stderr.write(f"wrote {out}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
