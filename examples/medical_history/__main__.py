"""Run from the repository root: python -m examples.medical_history --help."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from urllib.parse import urlsplit

from anatid import HashEmbedder, OpenAICompatibleEmbedder
from anatid.embed import Embedder

from .evaluate import assess, interpret
from .systems import SYSTEMS, RetrievalSystems, token_estimate
from .world import DISCLAIMER, build_world


def json_lines(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for name in SYSTEMS:
        selected = [r for r in rows if r["system"] == name]
        summary[name] = {
            "questions": len(selected),
            **{
                metric: mean(r[metric] for r in selected)
                for metric in (
                    "exact_answer",
                    "evidence_recall",
                    "complete_evidence",
                    "supported_correct",
                    "false_support",
                    "context_tokens",
                )
            },
            "visibility_violations": sum(r["visibility_violations"] for r in selected),
        }
    return summary


def report_text(metadata: dict, rows: list[dict], summary: dict, demo: dict) -> str:
    lines = [
        "# Fictional medical-device history: evidence retrieval",
        "",
        DISCLAIMER,
        "",
        (
            f"Embedding mode: **{metadata['embedding']}**. "
            "Shared deterministic source interpreter; no LLM judge or answer generation."
        ),
        "",
        (
            f"Seeds: {metadata['seeds']}; {metadata['documents']} documents and "
            f"{metadata['questions']} questions per seed; "
            f"{metadata['budget']} approximate tokens maximum per context. "
            "Whole records, same source fields, same embeddings and date rules for every system."
        ),
        "",
        "| System | Exact answer | Evidence recall | Complete evidence | Supported correct | False support |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in summary.items():
        values = " | ".join(
            f"{metrics[m]:.1%}"
            for m in (
                "exact_answer",
                "evidence_recall",
                "complete_evidence",
                "supported_correct",
                "false_support",
            )
        )
        lines.append(f"| {name} | {values} |")
    lines.extend(
        [
            "",
            (
                "Supported correct requires an exact answer AND retrieval and citation of the "
                "full scenario evidence checklist. Exact abstentions without that evidence do not count. "
                "This checklist is intentionally broader than the minimum proof of an individual pass: "
                "it includes the misleading candidate reports and their exclusions. An unanswerable "
                "question has no required evidence and must return unknown."
            ),
            "",
            "False support is the fraction of questions that incorrectly accept at least one report. "
            "Visibility violations: "
            + str(sum(m["visibility_violations"] for m in summary.values()))
            + ".",
            "",
            "## Worked example",
            "",
            demo["question"],
            "",
            "Expected: `" + json.dumps(demo["expected"], sort_keys=True) + "`",
            "",
        ]
    )
    for row in demo["results"]:
        answer = row["answer"]
        lines.extend(
            [
                f"### {row['system']}",
                "",
                "Answer: `" + json.dumps(answer["value"], sort_keys=True) + "`",
                "",
                "Citations: " + (", ".join(answer["citations"]) or "none") + ".",
                "",
                "Missing checklist evidence: "
                + (", ".join(row["missing_evidence"]) or "none")
                + ".",
                "",
            ]
        )
        for rid, reason in sorted(answer["exclusions"].items()):
            lines.append(f"- {rid}: {reason}.")
        lines.append("")
    lines.extend(
        [
            "## By question family",
            "",
            "| Family | System | Supported correct |",
            "| --- | --- | ---: |",
        ]
    )
    grouped: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in rows:
        grouped[row["category"], row["system"]].append(row["supported_correct"])
    for (category, name), values in sorted(grouped.items()):
        lines.append(f"| {category} | {name} | {mean(values):.1%} |")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            (
                "- Offline hash vectors measure word overlap, not semantic embedding quality. "
                "Even a semantic-embedding run here uses a deterministic structured answerer, not an LLM."
            ),
            (
                "- The source exports include explicit links and structured fields for ALL systems. "
                "This is not a prose extraction benchmark. SQL traceability can match anatid."
            ),
            (
                "- anatid-hybrid is existing library recall; anatid-trace is the application's explicit "
                "two-hop trace expansion followed by the same hybrid ranking used by trace-sql. "
                "Do not attribute the trace strategy's score to default recall."
            ),
            (
                "- Seeds vary outcomes, owners, phrasing and background notes, but share one event "
                "template. Repeated questions within a case are correlated; these are not independent "
                "real-world validations or confidence intervals."
            ),
            (
                "- At fixed question count, increasing documents adds background notes, not deeper "
                "revision chains. This isolates distraction, not all effects of product-history growth."
            ),
            (
                "- No regulatory, safety, release-readiness or clinical conclusions follow. "
                "Original controlled records and authorized human review remain authoritative."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(
    output: Path,
    *,
    seeds: list[int],
    documents: int,
    questions: int,
    budget: int,
    candidates: int,
    embedder: Embedder,
    embedding_label: str,
) -> dict:
    """The caller supplies a NEW directory. Never overwrite an existing run or database."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide at least one seed, without duplicates")
    if budget < 1 or candidates < 1:
        raise ValueError("budget and candidates must be positive")
    # Validate dimensions of the experiment before creating output or calling a model.
    worlds = [build_world(seed, documents=documents, questions=questions) for seed in seeds]
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1,
        "disclaimer": DISCLAIMER,
        "seeds": seeds,
        "documents": documents,
        "questions": questions,
        "budget": budget,
        "candidates": candidates,
        "embedding": embedding_label,
        "embedding_dimension": embedder.dim,
        "anatid_version": version("anatid"),
        "duckdb_version": version("duckdb"),
        "duckdb_threads": 1,
        "source_sha256": {},
        "answerer": "deterministic source interpreter; no LLM",
        "systems": list(SYSTEMS),
    }
    all_rows = []
    demo = None
    start = time.perf_counter()
    for world in worlds:
        print(f"Seed {world.seed}: indexing {len(world.records)} synthetic records...", flush=True)
        directory = output / f"seed-{world.seed}"
        directory.mkdir()
        source_path = directory / "documents.jsonl"
        json_lines(source_path, (r.to_dict() for r in world.records))
        json_lines(directory / "questions.jsonl", (q.to_dict() for q in world.questions))
        metadata["source_sha256"][str(world.seed)] = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
        systems = RetrievalSystems(world.records, embedder, directory / "history.anatid")
        rows = []
        try:
            for number, question in enumerate(world.questions, 1):
                contexts = systems.retrieve(question.query, budget=budget, candidates=candidates)
                question_rows = []
                for name, context in contexts.items():
                    row = {
                        "seed": world.seed,
                        "system": name,
                        **assess(question, context, interpret(question.query, context)),
                        "context_tokens": sum(token_estimate(r) for r in context),
                    }
                    rows.append(row)
                    question_rows.append(row)
                if demo is None and (question.query.category == "evidence_gap" or questions < 3):
                    demo = {
                        "question": question.query.question,
                        "expected": question.expected,
                        "results": question_rows,
                    }
                if number % 20 == 0 or number == len(world.questions):
                    print(
                        f"Seed {world.seed}: {number}/{len(world.questions)} questions evaluated",
                        flush=True,
                    )
        finally:
            systems.close()
        json_lines(directory / "results.jsonl", rows)
        all_rows.extend(rows)
    metadata["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    if demo is None:
        raise RuntimeError("no worked example was evaluated")
    summary = summarize(all_rows)
    (output / "summary.json").write_text(
        json.dumps({"metadata": metadata, "systems": summary}, indent=2) + "\n", encoding="utf-8"
    )
    (output / "REPORT.md").write_text(
        report_text(metadata, all_rows, summary, demo), encoding="utf-8"
    )
    print(f"Report: {output / 'REPORT.md'}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    parser.add_argument("--documents", type=int, default=500)
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument(
        "--budget", type=int, default=2000, help="approximate context tokens (characters / 4)"
    )
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument(
        "--out", type=Path, help="NEW output directory; default: a fresh temporary directory"
    )
    parser.add_argument(
        "--embedding-url",
        help="opt-in: send synthetic sources to a compatible /embeddings endpoint",
    )
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument(
        "--embedding-key-env",
        default="ANATID_EMBEDDING_KEY",
        help="environment variable name, NOT a key",
    )
    parser.add_argument(
        "--request-dimensions",
        action="store_true",
        help="include dimensions in embedding API request",
    )
    args = parser.parse_args(argv)
    if args.embedding_dim < 1:
        parser.error("--embedding-dim must be positive")
    embedder: Embedder
    if args.embedding_url:
        parsed = urlsplit(args.embedding_url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or any((parsed.username, parsed.password, parsed.query, parsed.fragment))
        ):
            parser.error(
                "use an HTTP(S) endpoint without credentials, query parameters or fragments"
            )
        if not args.embedding_model:
            parser.error("--embedding-url requires --embedding-model")
        embedder = OpenAICompatibleEmbedder(
            args.embedding_url,
            os.environ.get(args.embedding_key_env),
            args.embedding_model,
            args.embedding_dim,
            request_dimensions=args.request_dimensions,
        )
        label = f"endpoint embeddings: {args.embedding_model} (not an LLM answer benchmark)"
        print(
            "Opt-in endpoint mode: sending ONLY generated synthetic records and questions. "
            "Your configured provider may charge for embeddings.",
            flush=True,
        )
    else:
        if args.embedding_model or args.request_dimensions:
            parser.error("--embedding-model and --request-dimensions require --embedding-url")
        embedder = HashEmbedder(dim=args.embedding_dim)
        label = "offline lexical hash (NOT semantic embeddings)"
    output = (
        args.out.resolve()
        if args.out
        else Path(tempfile.mkdtemp(prefix="anatid-medical-history-")) / "run"
    )
    print(DISCLAIMER, flush=True)
    try:
        run(
            output,
            seeds=args.seeds,
            documents=args.documents,
            questions=args.questions,
            budget=args.budget,
            candidates=args.candidates,
            embedder=embedder,
            embedding_label=label,
        )
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
