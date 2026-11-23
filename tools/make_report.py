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
dense run and against BM25, and every primary rerank run against its dense run; Holm correction within (level, metric) over
these pairs.

Summary tables written next to the statistics (CSV and Markdown; the grid is read from the runs, nothing is hard-coded):
  table_retrievers_span / table_retrievers_lexical   rows = retriever (each dense model, bm25, each dense+bm25 hybrid, each
                                                    dense+rerank at the primary depth), columns = chunker x size x overlap,
                                                    values = Geom@10, nDCG@10, Recall@10, evaluation_time
  table_rerank_ablation                              rerank depth ablation (dense run without rerank and every depth that was run)
  table_external                                     BM25 next to the dense Phase 1 results on the external MTEB tasks

  python -m tools.make_report --artifacts-root artifacts --out-dir results/v2-revision
  python -m tools.make_report --artifacts-root artifacts/pilot_v2 --out-dir results/v2-revision/pilot
"""
from __future__ import annotations

import argparse
import csv
import itertools
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
from pipeline_lib.grid import CHUNKERS as CHUNKER_ORDER  # noqa: E402     # only the display order of the chunkers; sizes / overlaps come from the runs
from pipeline_lib.run_manifest import git_info, package_versions  # noqa: E402
from pipeline_lib.truncation import PHASE1_COLUMNS, collect_phase1  # noqa: E402
from tools import check_identical, collect_runs, hybrid, label_stats_table, lexical_overlap, stats  # noqa: E402

BM25 = "bm25"
METRIC_COLS = (("geom_at_10", "Geom@10"), ("ndcg_at_10", "nDCG@10"), ("recall_at_10", "Recall@10"))
SPLIT_PREFERENCE = ("test", "validation", "train")

DESCRIPTIONS = {
    "table_main.csv": "Main table: one row per configuration and model/hybrid: Geom@10, nDCG@10, Recall@10 under the span labels with 95% "
                      "paper-level bootstrap CI of Geom@10, Geom@10 under the lexical labels, span coverage@5, % chunks truncated, number of chunks.",
    "runs_overview.csv": "One row per run: metrics under both label sets, span coverage, truncation, timing, prompts, chunking diagnostics, git commit.",
    "stats_ci.csv": "Bootstrap 95% CIs of Geom@10, nDCG@10, Recall@10 per run: level=query and level=paper (number of resamples: report_manifest.json).",
    "stats_compare.csv": "Paired comparisons per configuration (dense vs BM25, dense vs dense, hybrid vs its dense run and vs BM25, rerank vs its dense run, vs BM25 and vs the hybrid): difference, CI, bootstrap and Wilcoxon p, Holm-adjusted within `family` x level x metric.",
    "stats_label_agreement.csv": "Lexical vs span labels per configuration: % questions with the same top relevant chunk, Jaccard of the relevant sets.",
    "stats_rankings.csv": "Kendall tau-b between the rankings of runs by the lexical and by the span labels (all runs, within model, within configuration).",
    "coverage_diagnostics.csv": "Per configuration: number of chunks, chunk length with the reference tokenizer, % evidence spans inside one chunk, relevant chunks per question.",
    "table_label_stats.csv": "Per configuration and cover level: chunks, chunk length (median / mean / max reference tokens), questions labelled in Step 1 / 2 / 3, % evidence split over several chunks, relevant chunks per question (mean / max).",
    "table_truncation.csv": "Per run: model max sequence length, number and % of chunks truncated by the model's own tokenizer.",
    "table_truncation_phase1.csv": "Phase 1: per model, external task and split, the number and % of passages and queries longer than the model's max sequence length (own tokenizer, title + text + prompt).",
    "overlap_distribution.json": "Distribution of the question/evidence-span word overlap over the 952 questions.",
    "overlap_questions.csv": "Per question: overlap share (question content words found in the evidence span), span length.",
    "overlap_advantage.csv": "BM25 minus dense per question metric by overlap bin, with paper-level bootstrap CI, Spearman rho and slope.",
    "overlap_rank_by_bin.csv": "Rank of the evidence chunk (% rank 1, % in top 10, % outside top 100, median) by overlap bin, per run.",
    "table_retrievers_span.csv": "Retrievers x configurations under the span labels (primary): Geom@10, nDCG@10, Recall@10 and evaluation_time per chunker x size x overlap (dense, bm25, hybrid RRF, dense + rerank).",
    "table_retrievers_span.md": "Same table as Markdown, one block per metric.",
    "table_retrievers_lexical.csv": "The same table under the lexical labels (evaluation_time is identical).",
    "table_retrievers_lexical.md": "Same table as Markdown, one block per metric.",
    "table_rerank_ablation.csv": "Rerank depth ablation: the dense run (depth 0) and every rerank depth that was run for the same dense model and configuration, both label sets, evaluation_time and rerank_s.",
    "table_rerank_ablation.md": "Same table as Markdown.",
    "table_external.csv": "External MTEB tasks (Phase 1): BM25 next to the dense models, Geom@10, nDCG@10, Recall@10, evaluation_time per task.",
    "table_external.md": "Same table as Markdown.",
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


def run_kind(name: str, info: Dict[str, Any]) -> str:
    return "hybrid" if info.get("hybrid") else "rerank" if info.get("rerank") else "bm25" if name.startswith(BM25) else "dense"


def comparison_pairs(names: List[str], infos: Dict[str, Dict[str, Any]], cell: tuple, rerank_top_n: Optional[int] = None) -> List[tuple]:
    """(A, B) pairs of one configuration. ``rerank_top_n``: only the rerank runs of that depth are compared (None: every rerank depth).

      dense   vs BM25                                  every dense run
      hybrid  vs its dense run, vs BM25                every RRF hybrid
      rerank  vs its dense run, vs BM25, vs the hybrid of the same dense run (when that hybrid exists)
      dense   vs dense                                 every pair of dense runs (models) on the same configuration, alphabetical order
    """
    infos_c = {n: infos.get(f"{n}|{cell[0]}|{cell[1]}|{cell[2]}", {}) for n in names}
    hybrid_of = {i["hybrid"]["dense"]: n for n, i in infos_c.items() if i.get("hybrid")}
    pairs = []
    for n in names:
        info = infos_c[n]
        h, rr = info.get("hybrid"), info.get("rerank")
        if h:
            pairs += [(n, h["dense"]), (n, h["sparse"])]
        elif rr:
            if rerank_top_n is None or int(rr["top_n"]) == int(rerank_top_n):
                pairs.append((n, rr["dense"]))
                if BM25 in names:
                    pairs.append((n, BM25))
                if rr["dense"] in hybrid_of:
                    pairs.append((n, hybrid_of[rr["dense"]]))
        elif not n.startswith(BM25) and BM25 in names:
            pairs.append((n, BM25))
    dense = sorted(n for n in names if run_kind(n, infos_c[n]) == "dense")
    pairs += list(itertools.combinations(dense, 2))
    return [p for p in pairs if p[0] in names and p[1] in names]


def pair_family(pair: tuple, infos: Dict[str, Dict[str, Any]], cell: tuple) -> str:
    """Holm family of a pair: hybrid (vs its parts), rerank (vs dense / BM25 / hybrid), dense_vs_bm25, dense_vs_dense. Holm's correction is applied
    within each family (and level and metric) of one configuration, so that the 78 dense pairs of 13 models do not weaken the baseline tests."""
    a, b = pair
    info = infos.get(f"{a}|{cell[0]}|{cell[1]}|{cell[2]}", {})
    kind = run_kind(a, info)
    if kind in ("hybrid", "rerank"):
        return kind
    return "dense_vs_bm25" if b.startswith(BM25) else "dense_vs_dense"


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


# ------------------------------------------------------------------------------------------------
# summary tables: one per label type, rerank ablation, external tasks
# ------------------------------------------------------------------------------------------------
def _cfg_name(r: Dict[str, Any]) -> str:
    return f"{r['chunker']}_c{r['size']}_o{r['overlap']}"


def _cfg_sort(cfg: tuple):
    ch = cfg[0]
    return (CHUNKER_ORDER.index(ch) if ch in CHUNKER_ORDER else len(CHUNKER_ORDER), ch, cfg[1], cfg[2])


def retriever_label(r: Dict[str, Any]) -> str:
    kind = r.get("kind") or "dense"
    if kind == "hybrid":
        return f"hybrid RRF: {r.get('source_dense')} + bm25"
    if kind == "rerank":
        return f"rerank@{r.get('rerank_top_n')}: {r.get('source_dense')} -> {str(r.get('rerank_model') or '').split('/')[-1]}"
    return r["model"]


def table_retrievers(overview: List[Dict[str, Any]], label: str, rerank_top_n: Optional[int] = 50) -> Tuple[List[Dict[str, Any]], List[tuple]]:
    """Wide table of one label type: one row per retriever, columns ``<chunker>_c<L>_o<O>__<metric>``.
    Rerank runs of other depths than ``rerank_top_n`` belong to the ablation table. Returns (rows, configurations in order)."""
    order = {"dense": 0, "bm25": 1, "hybrid": 2, "rerank": 3}
    runs = [r for r in overview if not (r.get("kind") == "rerank" and rerank_top_n is not None and _f(r.get("rerank_top_n")) != float(rerank_top_n))]
    cfgs = sorted({(r["chunker"], int(r["size"]), int(r["overlap"])) for r in runs}, key=_cfg_sort)
    by: Dict[str, Dict[tuple, Dict[str, Any]]] = {}
    kind: Dict[str, str] = {}
    for r in runs:
        lab = retriever_label(r)
        by.setdefault(lab, {})[(r["chunker"], int(r["size"]), int(r["overlap"]))] = r
        kind[lab] = r.get("kind") or "dense"
    rows = []
    for lab in sorted(by, key=lambda l: (order.get(kind[l], 9), l)):
        row: Dict[str, Any] = {"kind": kind[lab], "retriever": lab}
        for c in cfgs:
            r = by[lab].get(c)
            name = f"{c[0]}_c{c[1]}_o{c[2]}"
            for key, _ in METRIC_COLS:
                row[f"{name}__{key}"] = _f(r.get(f"{label}_{key}")) if r else None
            row[f"{name}__evaluation_time_s"] = _f(r.get("evaluation_time_s")) if r else None
        rows.append(row)
    return rows, cfgs


def _fmt(v, nd=3) -> str:
    if v is None or v == "":
        return ""
    return str(v) if isinstance(v, int) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _retrievers_md(label: str, rows: List[Dict[str, Any]], cfgs: List[tuple], rerank_top_n: int) -> str:
    out = [f"# Retrievers x configurations, {label} labels", "",
           f"Rows: dense models, bm25, dense+bm25 hybrids (RRF), dense+rerank@{rerank_top_n}. Columns: chunker_c<size>_o<overlap>. "
           "Empty cell = not run. evaluation_time in seconds (dense: MTEB; bm25: index + search + metrics + dump; hybrid: dense + bm25 + fusion; "
           "rerank: dense + rerank_s)."]
    for key, title in METRIC_COLS + (("evaluation_time_s", "evaluation_time (s)"),):
        nd = 1 if key == "evaluation_time_s" else 3
        head = ["retriever"] + [f"{c[0]}_c{c[1]}_o{c[2]}" for c in cfgs]
        out += ["", f"## {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for r in rows:
            out.append("| " + " | ".join([r["retriever"]] + [_fmt(r.get(f"{c[0]}_c{c[1]}_o{c[2]}__{key}"), nd) for c in cfgs]) + " |")
    return "\n".join(out) + "\n"


def _md_table(title: str, rows: List[Dict[str, Any]]) -> str:
    cols: List[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    out = [f"# {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out) + "\n"


def rerank_ablation_rows(overview: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per (dense model, configuration, depth) for every (dense model, configuration) with at least two rerank depths; depth 0 is
    the dense run itself."""
    dense = {(r["model"], r["chunker"], int(r["size"]), int(r["overlap"])): r for r in overview if (r.get("kind") or "dense") == "dense"}
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in overview:
        if r.get("kind") == "rerank":
            groups.setdefault((r["source_dense"], r["chunker"], int(r["size"]), int(r["overlap"])), []).append(r)
    rows = []
    for key in sorted(groups, key=lambda k: (k[0], _cfg_sort(k[1:]))):
        runs = groups[key]
        if len({_f(r.get("rerank_top_n")) for r in runs}) < 2:
            continue
        entries = ([("0", dense[key])] if key in dense else []) + [(r["rerank_top_n"], r) for r in sorted(runs, key=lambda r: _f(r["rerank_top_n"]))]
        for depth, r in entries:
            rows.append({"dense": key[0], "chunker": key[1], "size": key[2], "overlap": key[3], "rerank_top_n": int(float(depth)),
                         **{f"span_{k}": _f(r.get(f"span_{k}")) for k, _ in METRIC_COLS}, **{f"lexical_{k}": _f(r.get(f"lexical_{k}")) for k, _ in METRIC_COLS},
                         "evaluation_time_s": _f(r.get("evaluation_time_s")), "rerank_s": _f(r.get("rerank_s"))})
    return rows


def _split_scores(js: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    scores = js.get("scores") or {}
    for sp in SPLIT_PREFERENCE:
        if scores.get(sp):
            return scores[sp][0]
    return next((v[0] for v in scores.values() if v), None)


def external_rows(roots: List[Path]) -> List[Dict[str, Any]]:
    """One row per model found under ``roots`` (a leaf folder holds model_meta.json and <Task>.json files, the layout of Phase 1 and of
    stages/run_bm25_mteb.py): Geom@10 = sqrt(nDCG@10 x Recall@10), nDCG@10, Recall@10, evaluation_time per task. The first root that has
    a (model, revision, task) wins. BM25 rows come first."""
    seen: Dict[tuple, Dict[str, Any]] = {}
    for root in roots:
        if not Path(root).is_dir():
            continue
        for meta_p in sorted(Path(root).glob("**/model_meta.json")):
            leaf = meta_p.parent
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            for fp in sorted(leaf.glob("*.json")):
                if fp.name == "model_meta.json":
                    continue
                try:
                    js = json.loads(fp.read_text(encoding="utf-8"))
                except ValueError:
                    continue
                sc = _split_scores(js) if isinstance(js, dict) else None
                if not sc or "ndcg_at_10" not in sc or "recall_at_10" not in sc:
                    continue
                task = js.get("task_name") or fp.stem
                key = (meta.get("name"), meta.get("revision"), task)
                if key not in seen:
                    seen[key] = {"nd": float(sc["ndcg_at_10"]), "rc": float(sc["recall_at_10"]), "t": js.get("evaluation_time")}
    models = sorted({(k[0], k[1]) for k in seen}, key=lambda m: (not str(m[0]).startswith(BM25), str(m[0])))
    tasks = sorted({k[2] for k in seen})
    rows = []
    for name, rev in models:
        row: Dict[str, Any] = {"kind": "bm25" if str(name).startswith(BM25) else "dense", "model": name, "revision": rev}
        for t in tasks:
            v = seen.get((name, rev, t))
            row[f"{t}__geom_at_10"] = (v["nd"] * v["rc"]) ** 0.5 if v else None
            row[f"{t}__ndcg_at_10"] = v["nd"] if v else None
            row[f"{t}__recall_at_10"] = v["rc"] if v else None
            row[f"{t}__evaluation_time_s"] = v["t"] if v else None
        rows.append(row)
    return rows


def build_report(artifacts_root: str | Path, out_dir: str | Path, qa_jsonl: Optional[str | Path] = None, *, resamples: int = stats.DEFAULT_RESAMPLES,
                 seed: int = stats.DEFAULT_SEED, overlap: bool = True, track_max_mb: float = 5.0, tests_root: Optional[str | Path] = None,
                 prepared_root: Optional[str | Path] = None, verbose: bool = True, rerank_top_n: int = 50,
                 external_roots: Optional[List[str | Path]] = None) -> Dict[str, Any]:
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
        cache: Dict[tuple, tuple] = {}                                  # each run is resampled once per level, for the CIs and for the pairs
        ci_all += [{**label, **r} for r in stats.ci_rows(paths, qa, resamples, seed, cache=cache)]
        pairs = comparison_pairs(list(paths), infos, (ch, s, o), rerank_top_n)
        if pairs:
            fams = {p: pair_family(p, infos, (ch, s, o)) for p in pairs}
            cmp_all += [{**label, **r} for r in stats.compare_rows(paths, pairs, qa, resamples, seed, families=fams, cache=cache)]
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
    ls_rows = label_stats_table.build_table(prepared)
    if ls_rows:
        label_stats_table.write_table(heavy / "table_label_stats.csv", ls_rows)
    steps["label_stats_rows"] = len(ls_rows)

    trunc = [{k: r.get(k) for k in ("chunker", "size", "overlap", "model", "max_seq_length", "n_chunks", "n_chunks_truncated", "pct_chunks_truncated",
                                    "mean_tokens_own_tokenizer")} for r in overview_rows if r.get("max_seq_length") not in (None, "")]
    stats._write(heavy / "table_truncation.csv", trunc)
    p1_trunc = collect_phase1([Path(r) for r in (external_roots or [])])
    if p1_trunc:
        stats._write(heavy / "table_truncation_phase1.csv", [{k: r.get(k) for k in PHASE1_COLUMNS} for r in p1_trunc])
    steps["phase1_truncation_rows"] = len(p1_trunc)

    if overlap:
        dense = sorted({i["model"] for i in infos.values() if not i["model"].startswith(BM25) and not i.get("hybrid") and not i.get("rerank")})
        if dense and any(i["model"].startswith(BM25) for i in infos.values()):
            log(f"lexical-overlap diagnostic: {len(dense)} dense models")
            lexical_overlap.analyse(tests, qa, dense, BM25, heavy, seed, resamples)
        else:
            steps["overlap_skipped"] = "needs at least one dense run and the BM25 run"

    stats._write(heavy / "table_main.csv", table_main([{k: _num(v) for k, v in r.items()} for r in overview_rows], ci_all))

    log("retriever tables")
    for label in ("span", "lexical"):
        rows, blocks = table_retrievers(overview_rows, label, rerank_top_n)
        stats._write(heavy / f"table_retrievers_{label}.csv", rows)
        (heavy / f"table_retrievers_{label}.md").write_text(_retrievers_md(label, rows, blocks, rerank_top_n), encoding="utf-8")
    abl = rerank_ablation_rows(overview_rows)
    if abl:
        stats._write(heavy / "table_rerank_ablation.csv", abl)
        (heavy / "table_rerank_ablation.md").write_text(_md_table("Rerank depth ablation (depth 0 = the dense run without reranking)", abl), encoding="utf-8")
    ext_rows = external_rows([Path(r) for r in (external_roots or [])])
    if ext_rows:
        stats._write(heavy / "table_external.csv", ext_rows)
        (heavy / "table_external.md").write_text(_md_table("External MTEB tasks: BM25 and the dense Phase 1 models", ext_rows), encoding="utf-8")
    steps["external_rows"] = len(ext_rows)

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
    lines += ["", "evaluation_time (seconds): dense = MTEB (encode corpus and queries, search, metrics, dump; model load not counted); bm25 = index + search + "
              "metrics + dump; hybrid = dense + bm25 + fusion (read + fuse + metrics + dump); rerank = dense + rerank_s (read + score pairs + reorder + metrics + dump).", ""]
    lines += ["Large files (rankings, chunks, per-question rank tables) stay in `artifacts/`.", "`report_manifest.json` lists package versions and SHA-256 of every file."]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-root", required=True)
    ap.add_argument("--out-dir", required=True, help="tracked folder for the small CSV tables, e.g. results/v2-revision")
    ap.add_argument("--qa")
    ap.add_argument("--resamples", type=int, default=stats.DEFAULT_RESAMPLES, help="paper / query bootstrap resamples (default: params.report.resamples)")
    ap.add_argument("--seed", type=int, default=stats.DEFAULT_SEED)
    ap.add_argument("--no-overlap", action="store_true")
    ap.add_argument("--track-max-mb", type=float, default=5.0)
    ap.add_argument("--main-config", default="configs/main_config.yaml", help="source of params.rerank.top_n and params.external (when the flags below are absent)")
    ap.add_argument("--rerank-top-n", type=int, help="depth of the rerank runs in the main tables, the others go to the ablation table (default params.rerank.top_n, else 50)")
    ap.add_argument("--external-root", action="append", default=[], help="folder with Phase 1 / BM25 results of the external tasks (repeatable; default: params.external)")
    args = ap.parse_args(argv)
    cfg: Dict[str, Any] = {}
    if Path(args.main_config).is_file():
        import yaml
        cfg = (yaml.safe_load(Path(args.main_config).read_text(encoding="utf-8")) or {}).get("params", {})
    if args.rerank_top_n is None:
        args.rerank_top_n = int((cfg.get("rerank") or {}).get("top_n", 50))
    if not args.external_root and cfg.get("external"):
        ext, base = cfg["external"], Path((cfg.get("paths") or {}).get("artifacts_root", "artifacts"))
        args.external_root = [str(base / sub) for sub in [*(ext.get("phase1_subdirs") or []), ext.get("output_subdir", "")] if sub]
    m = build_report(args.artifacts_root, args.out_dir, args.qa, resamples=args.resamples, seed=args.seed, overlap=not args.no_overlap,
                     track_max_mb=args.track_max_mb, rerank_top_n=args.rerank_top_n, external_roots=args.external_root)
    return 0 if m["steps"].get("check_identical_exit_code", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
