"""Does fragmented evidence explain the small-chunk penalty? (``make analysis``; Reviewer 2 item 26)

The discussion says small chunks lose because they "may divide evidence". The label statistics count fragmented questions per configuration; this tool
links the two at the level of the QUESTION. For every strategy, overlap and pair of sizes (small, large) (``params.analysis.fragmentation``), for every
dense model:

  fragmented(q)   at the SMALL size no single chunk holds >= c of the span of question q (c = labeling.cover_level; the question is in Step 2 or 3 of the
                  span rule, or has no chunk at all). Read from ``overlap_table.jsonl`` of the prepared folder of the small configuration, exact integer arithmetic.
  gain(q)         metric(run at the large size)[q] - metric(run at the small size)[q] of the SAME model, for nDCG@10, Recall@10 and span coverage@5.

Output ``fragmentation.csv``: per strategy, pair of sizes, model (and "all dense models": the gain of a question is the mean over the models) and metric:
the number and share of fragmented questions, the mean gain of the fragmented and of the not fragmented questions, and the DIFFERENCE of the two with a
paper-level bootstrap interval (papers redrawn, all questions of a paper kept; ``params.report.resamples``) and a two-sided bootstrap p. A positive
difference says: the questions whose evidence the small size splits gain more from the large size than the others, which is what the explanation predicts.
This is an association over questions, not an experiment: fragmented questions may also be the harder ones.

  python -m tools.fragmentation --roots artifacts --qa artifacts/chemquests_hf_db/qa.jsonl --out-dir results/analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import config_params  # noqa: E402
from pipeline_lib.span_labels import parse_cover_level  # noqa: E402
from tools import analysis_data as AD  # noqa: E402

METRICS = ("ndcg_at_10", "recall_at_10", "span_coverage_at_5")


def settings(main_config: Optional[str] = None) -> Dict[str, Any]:
    s = config_params.section("analysis", main_config).get("fragmentation")
    if not s:
        raise ValueError("params.analysis.fragmentation is missing in the main config")
    return s


def fragmented_questions(overlap_table: Any, qids: Sequence[str], cover: Fraction) -> np.ndarray:
    """Boolean array over ``qids``: no chunk holds >= c of the span (exact: overlap_chars * den >= num * span_len). Questions without any row count as fragmented."""
    best: Dict[str, Tuple[int, int]] = {}                       # query -> (largest overlap in characters, span length)
    for line in open(overlap_table, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            q = str(r["query_id"])
            n = int(r["overlap_chars"])
            if q not in best or n > best[q][0]:
                best[q] = (n, int(r["span_len"]))
    return np.array([not (q in best and best[q][0] * cover.denominator >= cover.numerator * best[q][1]) for q in qids])


def read_span_coverage(path: Any, qids: Sequence[str], column: str = "span_coverage_at_5") -> np.ndarray:
    with open(path, newline="", encoding="utf-8") as f:
        rows = {str(r["query_id"]): float(r[column]) for r in csv.DictReader(f)}
    return np.array([rows[q] for q in qids])


def group_difference(gain: np.ndarray, frag: np.ndarray, boot: AD.PaperBootstrap) -> Dict[str, Any]:
    """Mean gain of the fragmented and of the other questions and their difference, point and paper-level percentile intervals."""
    out: Dict[str, Any] = {}
    stats_: Dict[str, Tuple[float, np.ndarray]] = {}
    for name, mask in (("fragmented", frag), ("not_fragmented", ~frag)):
        w = mask.astype(float)
        den = boot.counts @ (boot.paper_sums(w[None, :])[0])                     # fragmented questions in each resample
        num = boot.counts @ boot.paper_sums((gain * w)[None, :])[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            res = num / den
        stats_[name] = (float(gain[mask].mean()) if mask.any() else float("nan"), res)
        v = res[np.isfinite(res)]
        lo, hi = (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)
        out.update({f"mean_gain_{name}": stats_[name][0], f"gain_{name}_ci_low": float(lo), f"gain_{name}_ci_high": float(hi)})
    d = stats_["fragmented"][1] - stats_["not_fragmented"][1]
    d = d[np.isfinite(d)]
    point = stats_["fragmented"][0] - stats_["not_fragmented"][0]
    if len(d):
        le, ge = int((d <= 0).sum()), int((d >= 0).sum())
        out.update({"difference": point, "difference_ci_low": float(np.percentile(d, 2.5)), "difference_ci_high": float(np.percentile(d, 97.5)),
                    "p_bootstrap": min(1.0, 2 * (1 + min(le, ge)) / (len(d) + 1))})
    else:
        out.update({"difference": point, "difference_ci_low": np.nan, "difference_ci_high": np.nan, "p_bootstrap": np.nan})
    return out


def analyse(runs: AD.RunSet, cover: Fraction, pairs: Sequence[Sequence[int]], overlap: int, *, resamples: int, seed: int) -> List[Dict[str, Any]]:
    boot = AD.PaperBootstrap(runs.groups, resamples, seed)
    by_key = {(m["model"], m["chunker"], m["size"], m["overlap"]): i for i, m in enumerate(runs.meta) if m["kind"] == "dense"}
    models = sorted({k[0] for k in by_key})
    strategies = sorted({k[1] for k in by_key})
    frag_cache: Dict[str, np.ndarray] = {}
    cov_cache: Dict[int, np.ndarray] = {}

    def coverage(i: int) -> np.ndarray:
        if i not in cov_cache:
            cov_cache[i] = read_span_coverage(Path(runs.meta[i]["out_dir"]) / "span_coverage.csv", runs.qids)
        return cov_cache[i]

    def metric_values(i: int, metric: str) -> np.ndarray:
        return coverage(i) if metric == "span_coverage_at_5" else runs.metrics[metric][i]

    rows: List[Dict[str, Any]] = []
    for strategy in strategies:
        for small, large in pairs:
            have = [m for m in models if all((m, strategy, s, overlap) in by_key for s in (small, large))]
            if not have:
                continue
            i0 = by_key[(have[0], strategy, small, overlap)]
            prep = Path(runs.meta[i0]["prepared_dir"])
            if str(prep) not in frag_cache:
                frag_cache[str(prep)] = fragmented_questions(prep / "mteb_task" / "overlap_table.jsonl", runs.qids, cover)
            frag = frag_cache[str(prep)]
            for metric in METRICS:
                gains = {m: metric_values(by_key[(m, strategy, large, overlap)], metric) - metric_values(by_key[(m, strategy, small, overlap)], metric) for m in have}
                for who, g in [*gains.items(), ("ALL dense models (mean gain per question)", np.mean(list(gains.values()), axis=0))]:
                    rows.append({"strategy": strategy, "overlap": overlap, "small": small, "large": large, "model": who, "metric": metric,
                                 "n_fragmented": int(frag.sum()), "n_not_fragmented": int((~frag).sum()), "pct_fragmented": 100.0 * float(frag.mean()),
                                 **group_difference(g, frag, boot), "resamples": resamples})
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    AD.add_common_args(ap)
    args = ap.parse_args(argv)
    resamples, seed = AD.resolve(args)
    st = settings(args.main_config)
    cover = parse_cover_level(AD.defaults(args.main_config)["cover_level"])
    runs = AD.load_runs(args.roots, args.qa, kinds=["dense"], label_set=args.label_set)
    rows = analyse(runs, cover, st["pairs"], int(st["overlap"]), resamples=resamples, seed=seed)
    AD.write_rows(Path(args.out_dir) / "fragmentation.csv", rows)
    print(f"{len(rows)} rows -> {Path(args.out_dir) / 'fragmentation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
