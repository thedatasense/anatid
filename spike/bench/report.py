#!/usr/bin/env python
"""anatid spike report: merge results/*.<scale>.json -> results/REPORT.md (SPEC.md, section "report.py").

    .venv/bin/python bench/report.py --scale full [--results-dir DIR] [--out FILE] [--no-reference]

Emits: the per-engine R1/R2/W1/W2 table, the kill-criterion lines (duckdb_sql and duckdb_ext vs ladybug,
p50 and p95, PASS if ratio <= 5), the verify section (query_ids whose R1 id lists differ between
engines; engines lacking verify data are named), load time / DB size / peak RSS, concurrent-phase
numbers, and every engine's notes and errors. Exit status: 0 valid, 1 INVALID (verify mismatches > 0),
2 no result files. Output defaults to results/REPORT.md for --scale full, results/REPORT.<scale>.md otherwise.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

SPIKE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = SPIKE_DIR / "results"
KILL_RATIO = 5.0
DUCKDB_ENGINES = ("duckdb_sql", "duckdb_ext")
BASELINE = "ladybug"
ENGINE_ORDER = ("ladybug", "duckdb_sql", "duckdb_ext", "grafeo")
PHASE_KEYS = {"count", "p50_ms", "p95_ms", "p99_ms", "mean_ms", "ops_per_s", "wall_s", "raw_ms"}


# ----------------------------------------------------------------------------- helpers

def load_results(results_dir: Path, scale: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(results_dir.glob(f"*.{scale}.json")):
        engine = p.name.split(".")[0]
        try:
            d = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            d = {"engine": engine, "_load_error": f"{type(e).__name__}: {e}"}
        if not isinstance(d, dict):
            d = {"engine": engine, "_load_error": "top-level JSON is not an object"}
        d.setdefault("engine", engine)
        d["_path"] = str(p)
        out[d["engine"]] = d
    return out


def ordered(engines) -> list[str]:
    return sorted(engines, key=lambda e: (ENGINE_ORDER.index(e) if e in ENGINE_ORDER else 99, e))


def get(d: dict | None, *keys):
    """Nested lookup; returns None when any level is missing or not a dict."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def phase_status(ph) -> str | None:
    """None if the phase holds metrics, else a short status string (missing / ERROR / unsupported)."""
    if ph is None:
        return "missing"
    if isinstance(ph, str):
        return ph[:24]
    if isinstance(ph, dict):
        if "error" in ph:
            return "ERROR"
        if ph.get("unsupported") or ph.get("status") == "unsupported":
            return "unsupported"
        return None
    return "?"


def metric(ph, key: str):
    return ph.get(key) if isinstance(ph, dict) and phase_status(ph) is None else None


def fmt(v, nd: int = 2, unit: str = "") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (v != v):  # NaN
            return "n/a"
        return f"{v:,.{nd}f}{unit}" if isinstance(v, float) or nd else f"{v:,d}{unit}"
    return str(v)


def cell(ph, key: str, nd: int = 2) -> str:
    st = phase_status(ph)
    if st is not None:
        return st
    return fmt(ph.get(key), nd)


def md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def tidy(s) -> str:
    return " ".join(str(s).split()).replace("|", "\\|")


# ----------------------------------------------------------------------------- sections

def section_table(results: dict[str, dict]) -> list[str]:
    header = ["engine", "version", "R1 p50 ms", "R1 p95 ms", "R1 ops/s", "R2 p50 ms", "R2 p95 ms", "R2 ops/s",
              "W1 p50 ms", "W1 p95 ms", "W1 ops/s", "W2 p50 ms", "W2 p95 ms", "W2 ops/s"]
    rows = []
    for e in ordered(results):
        d = results[e]
        r1, r2 = get(d, "phases", "r1_only"), get(d, "phases", "r2_only")
        w1, w2 = get(d, "phases", "mixed", "W1"), get(d, "phases", "mixed", "W2")
        if phase_status(get(d, "phases", "mixed")) is not None:
            w1 = w2 = get(d, "phases", "mixed")
        rows.append([e, tidy(d.get("engine_version", "?")),
                     cell(r1, "p50_ms"), cell(r1, "p95_ms"), cell(r1, "ops_per_s", 1),
                     cell(r2, "p50_ms"), cell(r2, "p95_ms"), cell(r2, "ops_per_s", 1),
                     cell(w1, "p50_ms"), cell(w1, "p95_ms"), cell(w1, "ops_per_s", 1),
                     cell(w2, "p50_ms"), cell(w2, "p95_ms"), cell(w2, "ops_per_s", 1)])
    out = ["## Latency and throughput", "",
           "R1/R2 from the single-threaded `r1_only` (1,000 queries) / `r2_only` (300 queries) phases; "
           "W1/W2 from the `mixed` phase. Percentiles are per-op wall time including result materialization.", ""]
    out += md_table(header, rows)
    # mixed-phase detail
    out += ["", "### Mixed phase (70% writes / 30% reads, single thread)", ""]
    header2 = ["engine", "ops", "wall s", "W1 n", "W1 p50", "W1 p95", "W2 n", "W2 p50", "W2 p95",
               "R1 n", "R1 p50", "R1 p95", "R2 n", "R2 p50", "R2 p95"]
    rows2 = []
    for e in ordered(results):
        m = get(results[e], "phases", "mixed")
        st = phase_status(m)
        if st is not None:
            rows2.append([e, st] + [""] * (len(header2) - 2))
            continue
        cols = [e]
        n_total = sum((metric(m.get(k), "count") or 0) for k in ("W1", "W2", "R1", "R2"))
        cols += [fmt(n_total, 0), fmt(m.get("wall_s"), 1)]
        for k in ("W1", "W2", "R1", "R2"):
            ph = m.get(k)
            cols += [cell(ph, "count", 0), cell(ph, "p50_ms"), cell(ph, "p95_ms")]
        rows2.append(cols)
    out += md_table(header2, rows2)
    # r2 extras (recall@20 etc.)
    extras = []
    for e in ordered(results):
        r2 = get(results[e], "phases", "r2_only")
        if isinstance(r2, dict):
            ex = {k: v for k, v in r2.items() if k not in PHASE_KEYS and k not in ("traceback",)}
            if ex:
                extras.append(f"- **{e}**: " + ", ".join(f"{k}={tidy(v) if not isinstance(v, float) else f'{v:.3f}'}" for k, v in ex.items()))
    if extras:
        out += ["", "### R2 details (recall@20 vs brute-force truth, index used, ...)", ""] + extras
    return out


def section_kill(results: dict[str, dict]) -> list[str]:
    out = ["## Kill criterion (R1 2-hop recall, DuckDB layer vs LadybugDB)", "",
           f"`ratio = duckdb_x.R1.p50 / ladybug.R1.p50` (and p95). PASS if ratio <= {KILL_RATIO:g}.", ""]
    base = get(results.get(BASELINE), "phases", "r1_only")
    b50, b95 = metric(base, "p50_ms"), metric(base, "p95_ms")
    if b50 is None:
        why = "no result file" if BASELINE not in results else f"r1_only is {phase_status(base) or 'incomplete'}"
        out.append(f"- baseline `{BASELINE}` R1 not available ({why}); the kill criterion cannot be evaluated.")
    for e in DUCKDB_ENGINES:
        ph = get(results.get(e), "phases", "r1_only")
        p50, p95 = metric(ph, "p50_ms"), metric(ph, "p95_ms")
        if e not in results:
            out.append(f"- **{e}**: NOT AVAILABLE (no result file)")
            continue
        if p50 is None or b50 is None:
            out.append(f"- **{e}**: NOT AVAILABLE ({e} r1_only = {phase_status(ph) or 'ok'}, {BASELINE} r1_only = {phase_status(base) or 'ok'})")
            continue
        r50 = p50 / b50 if b50 > 0 else float("inf")
        r95 = p95 / b95 if (p95 is not None and b95) else None
        v50 = "PASS" if r50 <= KILL_RATIO else "FAIL"
        v95 = ("PASS" if r95 <= KILL_RATIO else "FAIL") if r95 is not None else "n/a"
        overall = "PASS" if (v50 == "PASS" and v95 in ("PASS", "n/a")) else "FAIL"
        out.append(f"- **{e}**: ratio_p50 = {r50:.2f} ({p50:.3f} ms / {b50:.3f} ms) -> {v50}; "
                   f"ratio_p95 = {fmt(r95)} ({fmt(p95, 3)} ms / {fmt(b95, 3)} ms) -> {v95}  =>  **{overall}**")
    return out


def reference_verify(scale: str, qids: list[str]) -> dict[str, list[int]] | None:
    """Reference R1 id lists (state after the mixed phase) for the verify query ids, or None if unavailable."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import common  # noqa: PLC0415
        queries = common.load_queries(scale)
        out = {}
        for q in qids:
            qq = queries[int(q)]
            out[q] = [m for m, _ in common.reference_r1(scale, qq["tenant_id"], qq["seed_entity_id"], after_mixed=True)]
        return out
    except Exception as e:  # noqa: BLE001
        print(f"reference check skipped: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def section_verify(results: dict[str, dict], scale: str, use_reference: bool) -> tuple[list[str], int]:
    out = ["## Verify (R1 for query_ids 0..199 after the mixed phase must be identical across engines)", ""]
    maps: dict[str, dict] = {}
    lacking = []
    for e in ordered(results):
        v = get(results[e], "verify", "r1")
        if isinstance(v, dict) and v:
            maps[e] = {str(k): (list(ids) if isinstance(ids, list) else ids) for k, ids in v.items()}
        else:
            lacking.append(e)
    if lacking:
        out.append(f"- engines lacking verify data: {', '.join(f'`{e}`' for e in lacking)} (excluded from the comparison)")
    mismatches: list[str] = []
    if len(maps) < 2:
        out.append(f"- only {len(maps)} engine(s) with verify data ({', '.join(maps) or 'none'}): cross-engine agreement cannot be checked")
    else:
        qids = sorted({q for m in maps.values() for q in m}, key=lambda s: int(s) if s.isdigit() else s)
        for q in qids:
            variants = {json.dumps(m.get(q)) for m in maps.values()}
            if len(variants) > 1:
                mismatches.append(q)
        out.append(f"- engines compared: {', '.join(f'`{e}`' for e in maps)}; query_ids compared: {len(qids)}")
        out.append(f"- **query_ids with differing R1 id lists: {len(mismatches)}**")
        if mismatches:
            out.append("- **BENCHMARK INVALID**: the engines do not return the same answers; fix the runners before reading any numbers.")
            for q in mismatches[:10]:
                out.append(f"  - query_id {q}: " + "; ".join(
                    f"{e}={'missing' if m.get(q) is None else str(m[q][:6]) + ('...' if len(m[q]) > 6 else '') + f' (n={len(m[q])})'}"
                    for e, m in maps.items()))
            if len(mismatches) > 10:
                out.append(f"  - ... and {len(mismatches) - 10} more")
        else:
            out.append("- all compared engines agree exactly.")
    if use_reference and maps:
        qids_all = sorted({q for m in maps.values() for q in m}, key=lambda s: int(s) if s.isdigit() else s)
        ref = reference_verify(scale, qids_all)
        if ref is None:
            out.append("- reference check: skipped (dataset not available or common.py failed; see stderr)")
        else:
            for e, m in maps.items():
                bad = [q for q in qids_all if m.get(q) != ref.get(q)]
                flag = "" if not bad else "  **WARNING: disagrees with the pure-numpy reference (schedule writes applied)**"
                out.append(f"- reference check `{e}`: {len(bad)} / {len(qids_all)} query_ids differ from common.reference_r1(after_mixed=True){flag}"
                           + (f" e.g. {bad[:8]}" if bad else ""))
    return out, len(mismatches)


def section_load(results: dict[str, dict]) -> list[str]:
    out = ["## Load, size, memory", ""]
    rows = []
    for e in ordered(results):
        d = results[e]
        ld = d.get("load") or {}
        db = ld.get("db_bytes") if isinstance(ld, dict) else None
        rows.append([e, fmt(ld.get("seconds") if isinstance(ld, dict) else None, 1),
                     fmt(db / 1e6, 1) if isinstance(db, (int, float)) else "n/a",
                     fmt(d.get("peak_rss_mb"), 0), tidy(ld.get("index_notes", "") if isinstance(ld, dict) else ld)])
    out += md_table(["engine", "load s", "DB MB", "peak RSS MB", "index notes"], rows)
    return out


def section_concurrent(results: dict[str, dict]) -> list[str]:
    out = ["## Concurrent phase (4 W1 writer threads + 2 R1 reader threads, own connection each)", ""]
    rows = []
    for e in ordered(results):
        c = get(results[e], "phases", "concurrent")
        st = phase_status(c)
        if st is not None:
            rows.append([e, st, "", "", "", ""])
            continue
        rows.append([e, fmt(c.get("seconds") or c.get("elapsed_s"), 0), fmt(c.get("W1_ops_per_s"), 1),
                     fmt(c.get("R1_ops_per_s"), 1), fmt(c.get("errors"), 0), tidy(c.get("notes", ""))])
    out += md_table(["engine", "seconds", "W1 ops/s", "R1 ops/s", "errors", "notes"], rows)
    return out


def section_notes(results: dict[str, dict]) -> list[str]:
    out = ["## Notes and errors per engine", ""]
    for e in ordered(results):
        d = results[e]
        out.append(f"### {e} ({tidy(d.get('engine_version', '?'))})")
        out.append("")
        out.append(f"- result file: `{Path(d['_path']).name}`, host: {tidy(d.get('host', '?'))}, written: {tidy(d.get('written_at', '?'))}")
        if d.get("_load_error"):
            out.append(f"- **could not parse result file**: {tidy(d['_load_error'])}")
        phases = d.get("phases") if isinstance(d.get("phases"), dict) else {}
        for name, ph in phases.items():
            if isinstance(ph, dict) and "error" in ph:
                out.append(f"- **phase `{name}` ERROR**: {tidy(ph['error'])}")
            elif isinstance(ph, dict) and name == "mixed":
                for k, sub in ph.items():
                    if isinstance(sub, dict) and "error" in sub:
                        out.append(f"- **phase `mixed/{k}` ERROR**: {tidy(sub['error'])}")
            elif ph is None:
                out.append(f"- phase `{name}`: missing")
            elif isinstance(ph, str):
                out.append(f"- phase `{name}`: {tidy(ph)}")
        notes = d.get("notes") or []
        if isinstance(notes, str):
            notes = [notes]
        for n in notes:
            out.append(f"- {tidy(n)}")
        if not notes:
            out.append("- (no notes)")
        out.append("")
    return out


# ----------------------------------------------------------------------------- main

def build_report(results: dict[str, dict], scale: str, use_reference: bool) -> tuple[str, int]:
    lines = [f"# anatid Phase 0 spike report -- scale `{scale}`", "",
             f"Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from "
             f"{', '.join(f'`{Path(results[e]['_path']).name}`' for e in ordered(results))}. Contract: `SPEC.md`.", ""]
    verify_lines, n_mismatch = section_verify(results, scale, use_reference)
    lines += section_kill(results) + [""]
    lines += section_table(results) + [""]
    lines += verify_lines + [""]
    lines += section_load(results) + [""]
    lines += section_concurrent(results) + [""]
    lines += section_notes(results)
    if n_mismatch > 0:
        lines.insert(3, f"> **INVALID**: {n_mismatch} verify query_ids differ between engines. Numbers below must not be compared.")
        lines.insert(4, "")
    return "\n".join(lines).rstrip() + "\n", n_mismatch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=("small", "full"), default="full")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--out", type=Path, default=None, help="default results/REPORT.md (full) or results/REPORT.<scale>.md")
    ap.add_argument("--no-reference", action="store_true", help="skip comparing verify lists with common.reference_r1")
    args = ap.parse_args(argv)
    results = load_results(args.results_dir, args.scale)
    if not results:
        print(f"no result files matching *.{args.scale}.json in {args.results_dir}", file=sys.stderr)
        return 2
    report, n_mismatch = build_report(results, args.scale, not args.no_reference)
    out = args.out or (args.results_dir / ("REPORT.md" if args.scale == "full" else f"REPORT.{args.scale}.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)
    print(f"wrote {out}")
    if n_mismatch > 0:
        print(f"INVALID: {n_mismatch} verify query_ids differ between engines")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
