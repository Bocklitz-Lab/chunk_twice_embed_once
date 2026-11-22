"""Compare two runs of the same configuration (repeat-run check): are rankings, scores and metrics identical?

Takes two result folders (the folder with run_top100.trec.gz, per_query_metrics.csv, ChemQuest.json) or two run folders
(artifacts/.../tests/<run>, then the single results/<model>/ folder inside is used; ``--results-subdir`` picks another).
Reports: queries with a different top-k ranking, the largest absolute score difference, the largest per-query metric
difference, the aggregate metrics of MTEB's ChemQuest.json. Exit code 0 only if every ranking and every per-query
metric is identical.

  python -m tools.compare_runs --a artifacts/pilot/tests/RUN/results_old_20261122T101500/m --b artifacts/pilot/tests/RUN/results/m
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools import metrics as M  # noqa: E402


def result_folder(p: Path, subdir: str = "results") -> Path:
    p = Path(p)
    if (p / "run_top100.trec.gz").exists():
        return p
    cands = [d for d in (p / subdir).glob("*") if d.is_dir() and (d / "per_query_metrics.csv").exists()]
    if len(cands) != 1:
        raise SystemExit(f"ERROR: expected one result folder with per_query_metrics.csv under {p / subdir}, found {len(cands)}")
    return cands[0]


def compare(a: Path, b: Path) -> Dict[str, Any]:
    ra, rb = M.load_run(a / "run_top100.trec.gz"), M.load_run(b / "run_top100.trec.gz")
    pa, pb = M.load_per_query_csv(a / "per_query_metrics.csv"), M.load_per_query_csv(b / "per_query_metrics.csv")
    if set(ra) != set(rb):
        raise SystemExit("ERROR: the runs cover different queries")
    diff_rank, max_score = [], 0.0
    for q in ra:
        ta, tb = M.ranked(ra[q]), M.ranked(rb[q])
        if [d for d, _ in ta] != [d for d, _ in tb]:
            diff_rank.append(q)
        else:
            max_score = max(max_score, max(abs(x[1] - y[1]) for x, y in zip(ta, tb)))
    max_metric = max((abs(pa[q][k] - pb[q][k]) for q in pa for k in pa[q]), default=0.0)
    out: Dict[str, Any] = {"n_queries": len(ra), "n_queries_with_different_ranking": len(diff_rank), "first_different": diff_rank[:5],
                           "max_abs_score_diff_where_ranking_equal": max_score, "max_abs_per_query_metric_diff": max_metric,
                           "run_files_identical": gzip.open(a / "run_top100.trec.gz").read() == gzip.open(b / "run_top100.trec.gz").read()}
    ca, cb = list(a.rglob("ChemQuest.json")), list(b.rglob("ChemQuest.json"))
    if ca and cb:
        sa, sb = M.mteb_stored_scores(ca[0]), M.mteb_stored_scores(cb[0])
        out["max_abs_stored_metric_diff"] = max(abs(float(sa[k]) - float(sb[k])) for k in sa if isinstance(sa[k], (int, float)) and k in sb)
    out["identical"] = not diff_rank and max_metric == 0.0
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--results-subdir", default="results")
    args = ap.parse_args(argv)
    res = compare(result_folder(Path(args.a), args.results_subdir), result_folder(Path(args.b), args.results_subdir))
    print(json.dumps(res, indent=2))
    return 0 if res["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
