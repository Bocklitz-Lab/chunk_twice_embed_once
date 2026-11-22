"""Everything that is computed AFTER the runs, in one step: run table, identical-input check, statistics (bootstrap CIs at
query and paper level, paired tests with Holm correction, lexical-vs-span agreement, ranking agreement), coverage diagnostics,
lexical-overlap diagnostic, truncation summary and the paper tables.

It runs automatically at the end of ``make rerun`` / ``make eval`` / ``make pilot`` (stages/rerun.py, ``params.report``) and
by hand with ``make report`` / ``make collect``. Nothing here needs a GPU or an embedding.

Two places are written:
  <artifacts root>/report/   everything, including the large per-question files (not tracked by git)
  --out-dir                  (tracked by git, e.g. results/v2-revision/) the SMALL summary CSVs used in the paper, a
                             README.md that lists them, and report_manifest.json (git commit, package versions, seeds,
                             hashes of the files). A file above --track-max-mb stays in <artifacts root>/report/ only.

Statistics families per configuration (chunker, size, overlap): every dense run against BM25, and every hybrid against its
dense run and against BM25; Holm correction within (level, metric) over these pairs.

  python -m tools.make_report --artifacts-root artifacts --out-dir results/v2-revision
  python -m tools.make_report --artifacts-root artifacts/pilot_v2 --out-dir results/v2-revision/pilot
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib.dataset_pin import sha256_file  # noqa: E402
from pipeline_lib.run_manifest import git_info, package_versions  # noqa: E402
from tools import check_identical, collect_runs, hybrid, lexical_overlap, stats  # noqa: E402

BM25 = "bm25"

DESCRIPTIONS = {
    "table_main.csv": "Main table: one row per configuration and model/hybrid: Geom@10, nDCG@10, Recall@10 under the span labels with 95% "
                      "paper-level bootstrap CI of Geom@10, Geom@10 under the lexical labels, span coverage@5, % chunks truncated, number of chunks.",
    "runs_overview.csv": "One row per run: metrics under both label sets, span coverage, truncation, timing, prompts, chunking diagnostics, git commit.",
    "stats_ci.csv": "Bootstrap 95% CIs (1000 resamples) of Geom@10, nDCG@10, Recall@10 per run: level=query and level=paper.",
    "stats_compare.csv": "Paired comparisons (dense vs BM25, hybrid vs its dense run and vs BM25): difference, CI, bootstrap and Wilcoxon p, Holm-adjusted.",
    "stats_label_agreement.csv": "Lexical vs span labels per configuration: % questions with the same top relevant chunk, Jaccard of the relevant sets.",
    "stats_rankings.csv": "Kendall tau-b between the rankings of runs by the lexical and by the span labels (all runs, within model, within configuration).",
    "coverage_diagnostics.csv": "Per configuration: number of chunks, chunk length with the reference tokenizer, % evidence spans inside one chunk, relevant chunks per question.",
    "table_truncation.csv": "Per run: model max sequence length, number and % of chunks truncated by the model's own tokenizer.",
    "overlap_distribution.json": "Distribution of the question/evidence-span word overlap over the 952 questions.",
    "overlap_questions.csv": "Per question: overlap share (question content words found in the evidence span), span length.",
    "overlap_advantage.csv": "BM25 minus dense per question metric by overlap bin, with paper-level bootstrap CI, Spearman rho and slope.",
    "overlap_rank_by_bin.csv": "Rank of the evidence chunk (% rank 1, % in top 10, % outside top 100, median) by overlap bin, per run.",
    "identical_check.csv": "Were corpus, queries, chunks and qrels byte-identical for all models of a configuration (and equal to the prepared folder)?",
}


def _run_paths(tests_root: Path) -> Tuple[Dict[tuple, Dict[str, Path]], Dict[str, Dict[str, Any]]]:
    """({(chunker, size, overlap): {run name: per_query_metrics.csv}}, {run name: run_info}) from the run folders."""
    cells: Dict[tuple, Dict[str, Path]] = {}
    infos: Dict[str, Dict[str, Any]] = {}
    for d in sorted(Path(tests_root).iterdir()):
        info_p = d / "run_info.json"
        if not (d.is_dir() and info_p.is_file()):
            continue
        pq = list((d / "results").glob("*/per_query_metrics.csv"))
        if len(pq) != 1:
            continue
        info = json.loads(info_p.read_text(encoding="utf-8"))
        cells.setdefault((info["chunker"], int(info["size"]), int(info["overlap"])), {})[info["model"]] = pq[0]
        infos[f"{info['model']}|{info['chunker']}|{info['size']}|{info['overlap']}"] = info
    return cells, infos


def comparison_pairs(names: List[str], infos: Dict[str, Dict[str, Any]], cell: tuple) -> List[tuple]:
    pairs = []
    for n in names:
        info = infos.get(f"{n}|{cell[0]}|{cell[1]}|{cell[2]}", {})
        h = info.get("hybrid")
        if h:
            pairs += [(n, h["dense"]), (n, h["sparse"])]
        elif not n.startswith(BM25) and BM25 in names:
            pairs.append((n, BM25))
    return [p for p in pairs if p[0] in names and p[1] in names]


def _csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def table_main(overview: List[Dict[str, Any]], ci_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ci = {(r["chunker"], r["size"], r["overlap"], r["run"]): r for r in ci_rows if r["level"] == "paper" and r["metric"] == "geom_at_10"}
    out = []
    for r in overview:
        c = ci.get((r["chunker"], int(r["size"]), int(r["overlap"]), r["model"]), {})
        out.append({"chunker": r["chunker"], "size": r["size"], "overlap": r["overlap"], "model": r["model"], "n_chunks": r.get("n_chunks"),
                    "span_geom_at_10": r.get("span_geom_at_10"), "geom_ci95_low_paper": c.get("ci_low"), "geom_ci95_high_paper": c.get("ci_high"),
                    "span_ndcg_at_10": r.get("span_ndcg_at_10"), "span_recall_at_10": r.get("span_recall_at_10"),
                    "lexical_geom_at_10": r.get("lexical_geom_at_10"), "span_coverage_at_5": r.get("span_coverage_at_5"),
                    "pct_chunks_truncated": r.get("pct_chunks_truncated")})
    return sorted(out, key=lambda r: (r["chunker"], int(r["size"]), int(r["overlap"]), r["model"]))


def build_report(artifacts_root: str | Path, out_dir: str | Path, qa_jsonl: Optional[str | Path] = None, *, resamples: int = 1000,
                 seed: int = stats.DEFAULT_SEED, overlap: bool = True, track_max_mb: float = 5.0, tests_root: Optional[str | Path] = None,
                 prepared_root: Optional[str | Path] = None, verbose: bool = True) -> Dict[str, Any]:
    t_start = time.time()
    root = Path(artifacts_root)
    tests = Path(tests_root) if tests_root else root / "tests"
    prepared = Path(prepared_root) if prepared_root else root / "prepared"
    qa = Path(qa_jsonl) if qa_jsonl else root / "chemquests_hf_db" / "qa.jsonl"
    heavy = root / "report"
    heavy.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: print("[report]", *a, flush=True)) if verbose else (lambda *a: None)
    steps: Dict[str, Any] = {}

    log("run table")
    overview = collect_runs.collect(tests)
    collect_runs.write_csv(heavy / "runs_overview.csv", overview)
    overview_rows = _csv_rows(heavy / "runs_overview.csv")
    steps["n_runs"] = len(overview)

    log("identical-input check")
    steps["check_identical_exit_code"] = check_identical.main(["--tests-root", str(tests), "--out", str(heavy / "identical_check.csv")])

    cells, infos = _run_paths(tests)
    log(f"statistics: {len(cells)} configurations")
    ci_all: List[Dict[str, Any]] = []
    cmp_all: List[Dict[str, Any]] = []
    for (ch, s, o), runs in sorted(cells.items()):
        label = {"chunker": ch, "size": s, "overlap": o}
        paths = {n: str(p) for n, p in runs.items()}
        ci_all += [{**label, **r} for r in stats.ci_rows(paths, qa, resamples, seed)]
        pairs = comparison_pairs(list(paths), infos, (ch, s, o))
        if pairs:
            cmp_all += [{**label, **r} for r in stats.compare_rows(paths, pairs, qa, resamples, seed)]
    stats._write(heavy / "stats_ci.csv", ci_all)
    stats._write(heavy / "stats_compare.csv", cmp_all)

    label_rows = []
    for d in sorted(p for p in prepared.glob("*") if (p / "manifest.json").exists()):
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        label_rows.append({"config": d.name, **stats.label_agreement(d / "mteb_task", man.get("span_qrels_file", "qrels_span_c090.jsonl"))})
    stats._write(heavy / "stats_label_agreement.csv", label_rows)
    try:
        stats._write(heavy / "stats_rankings.csv", stats.ranking_agreement(heavy / "runs_overview.csv"))
    except Exception as e:                                           # too few runs for a rank correlation
        steps["rankings_skipped"] = str(e)

    log("coverage diagnostics")
    cov: List[Dict[str, Any]] = []
    for d in sorted(prepared.glob("*")):
        p = d / "coverage_diagnostics.csv"
        if p.exists():
            cov += [{"prepared": d.name, **r} for r in _csv_rows(p)]
    stats._write(heavy / "coverage_diagnostics.csv", cov)

    trunc = [{k: r.get(k) for k in ("chunker", "size", "overlap", "model", "max_seq_length", "n_chunks", "n_chunks_truncated", "pct_chunks_truncated",
                                    "mean_tokens_own_tokenizer")} for r in overview_rows if r.get("max_seq_length") not in (None, "")]
    stats._write(heavy / "table_truncation.csv", trunc)

    if overlap:
        dense = sorted({i["model"] for i in infos.values() if not i["model"].startswith(BM25) and not i.get("hybrid")})
        if dense and any(i["model"].startswith(BM25) for i in infos.values()):
            log(f"lexical-overlap diagnostic: {len(dense)} dense models")
            lexical_overlap.analyse(tests, qa, dense, BM25, heavy, seed)
        else:
            steps["overlap_skipped"] = "needs at least one dense run and the BM25 run"

    stats._write(heavy / "table_main.csv", table_main([{k: _num(v) for k, v in r.items()} for r in overview_rows], ci_all))

    # ---- the small files, tracked by git
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tracked, kept_back = {}, []
    for name in DESCRIPTIONS:
        src = heavy / name
        if not src.exists():
            continue
        if src.stat().st_size > track_max_mb * 1024 * 1024:
            kept_back.append(name)
            continue
        shutil.copyfile(src, out / name)
        tracked[name] = {"bytes": src.stat().st_size, "sha256": sha256_file(src)}
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t_start, 1),
        "git": git_info(_REPO_ROOT), "packages": package_versions(["numpy", "scipy", "pytrec_eval", "scikit-learn", "mteb", "sentence-transformers", "transformers", "torch"]),
        "artifacts_root": str(root), "qa_jsonl": str(qa), "qa_sha256": sha256_file(qa) if qa.exists() else None,
        "bootstrap": {"resamples": resamples, "seed": seed, "levels": ["query", "paper"]},
        "n_configurations": len(cells), "models": sorted({m for r in cells.values() for m in r}),
        "steps": steps, "files": tracked, "kept_in_artifacts_only": kept_back, "full_report_dir": str(heavy),
    }
    (out / "report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "README.md").write_text(_readme(tracked, kept_back, manifest), encoding="utf-8")
    log(f"{len(tracked)} files -> {out}")
    return manifest


def _num(v):
    try:
        return int(v) if str(v).lstrip("-").isdigit() else float(v)
    except (TypeError, ValueError):
        return v


def _readme(tracked: Dict[str, Any], kept_back: List[str], m: Dict[str, Any]) -> str:
    lines = ["# Summary tables (written by `tools/make_report.py`; do not edit by hand)", "",
             f"Created {m['created_utc']} from commit `{(m['git'] or {}).get('commit')}`"
             f"{' (dirty working tree)' if (m['git'] or {}).get('dirty') else ''}. Artifacts root: `{m['artifacts_root']}`.",
             f"Bootstrap: {m['bootstrap']['resamples']} resamples, seed {m['bootstrap']['seed']}, query level and paper level "
             "(the 952 questions come from 149 papers). Labels: span labels, cover level c=0.90 (primary); the lexical labels are reported next to them.", "",
             "| file | content |", "|---|---|"]
    lines += [f"| `{n}` | {DESCRIPTIONS[n]} |" for n in tracked]
    if kept_back:
        lines += ["", "Too large to track, only in the artifacts folder: " + ", ".join(f"`{n}`" for n in kept_back) + "."]
    lines += ["", "Large files (rankings, chunks, per-question rank tables) stay in `artifacts/`.", "`report_manifest.json` lists package versions and SHA-256 of every file."]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-root", required=True)
    ap.add_argument("--out-dir", required=True, help="tracked folder for the small CSV tables, e.g. results/v2-revision")
    ap.add_argument("--qa")
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=stats.DEFAULT_SEED)
    ap.add_argument("--no-overlap", action="store_true")
    ap.add_argument("--track-max-mb", type=float, default=5.0)
    args = ap.parse_args(argv)
    m = build_report(args.artifacts_root, args.out_dir, args.qa, resamples=args.resamples, seed=args.seed, overlap=not args.no_overlap,
                     track_max_mb=args.track_max_mb)
    return 0 if m["steps"].get("check_identical_exit_code", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
