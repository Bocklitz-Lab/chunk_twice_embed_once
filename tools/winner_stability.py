"""Winner stability (``make analysis``; Reviewer 2 item 19): how often does a model, a configuration or a chunking setting come out on top?

A table of "the best configuration is X" rests on one sample of 952 questions from 149 papers. In every paper-level resample (``analysis_data.PaperBootstrap``;
params.report.resamples, default 10,000; the papers are redrawn, all questions of a drawn paper are kept) Geom@10 of EVERY run is recomputed from the
per-query values, and the run that is first is counted. Shares of resamples:

  winner_by_configuration.csv   for each configuration (strategy, size, overlap): the share of resamples in which each model is first  (which model wins here?)
  winner_by_model.csv           for each model: the share of resamples in which each configuration is its best                          (which setting is best for it?)
  winner_overall.csv            over all runs of the pool: the share of resamples won by each run (model + configuration), and the summed share
                                of each strategy, size, overlap and of each (strategy, size, overlap) of the best run
  Each row also says whether the run / setting is the winner in the actual data (``winner_in_data``).

POOLS: ``dense`` = the dense models only (the design of the paper); ``all`` = dense, BM25, hybrid and rerank runs together (a hybrid or a reranked run
can win). ``--pools`` chooses; the default is both. A run that never wins in any resample is not listed.
Ties (equal Geom@10 in a resample) go to the first run in the table order, which is by strategy, size, overlap, kind and model name.

  python -m tools.winner_stability --roots artifacts --qa artifacts/chemquests_hf_db/qa.jsonl --out-dir results/analysis
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools import analysis_data as AD  # noqa: E402

POOLS = {"dense": ["dense"], "all": ["dense", "bm25", "hybrid", "rerank"]}


def cell_of(m: Dict[str, Any]) -> tuple:
    return (m["chunker"], m["size"], m["overlap"])


def analyse(runs: AD.RunSet, pool_kinds: Sequence[str], *, resamples: int, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    """The three tables of one pool (see the module docstring)."""
    idx = runs.select(kind=list(pool_kinds))
    if not idx:
        return {"configuration": [], "model": [], "overall": []}
    meta = [runs.meta[i] for i in idx]
    boot = AD.PaperBootstrap(runs.groups, resamples, seed)
    G = boot.geom_resamples(runs.metrics["ndcg_at_10"][idx], runs.metrics["recall_at_10"][idx])          # (B, R)
    point = runs.geom(idx)
    B = G.shape[0]

    def shares(cols: List[int]) -> Dict[int, float]:
        """cols (positions in ``meta``): the share of resamples in which each is the maximum among them (first wins ties)."""
        win = np.array(cols)[np.argmax(G[:, cols], axis=1)]
        return {int(c): float(n) / B for c, n in zip(*np.unique(win, return_counts=True))}

    def point_winner(cols: List[int]) -> int:
        return int(np.array(cols)[np.argmax(point[cols])])

    by_cell: Dict[tuple, List[int]] = defaultdict(list)
    by_model: Dict[str, List[int]] = defaultdict(list)
    for j, m in enumerate(meta):
        by_cell[cell_of(m)].append(j)
        by_model[m["model"]].append(j)
    conf_rows, model_rows = [], []
    for cell, cols in sorted(by_cell.items()):
        win, pw = shares(cols), point_winner(cols)
        for c in sorted(win, key=lambda c: -win[c]):
            conf_rows.append({"chunker": cell[0], "size": cell[1], "overlap": cell[2], "model": meta[c]["model"], "kind": meta[c]["kind"], "share_first": win[c],
                              "winner_in_data": c == pw, "geom_at_10": float(point[c]), "n_runs_compared": len(cols)})
    for model, cols in sorted(by_model.items()):
        win, pw = shares(cols), point_winner(cols)
        for c in sorted(win, key=lambda c: -win[c]):
            m = meta[c]
            model_rows.append({"model": model, "kind": m["kind"], "chunker": m["chunker"], "size": m["size"], "overlap": m["overlap"], "share_best": win[c],
                               "winner_in_data": c == pw, "geom_at_10": float(point[c]), "n_configurations": len(cols)})
    allcols = list(range(len(meta)))
    win, pw = shares(allcols), point_winner(allcols)
    overall = []
    for c in sorted(win, key=lambda c: -win[c]):
        m = meta[c]
        overall.append({"level": "run", "value": f"{m['model']} | {m['chunker']} {m['size']}/{m['overlap']}", "share_first": win[c], "winner_in_data": c == pw})
    for level, key in (("model", lambda m: m["model"]), ("strategy", lambda m: m["chunker"]), ("size", lambda m: m["size"]), ("overlap", lambda m: m["overlap"]),
                       ("configuration", lambda m: f"{m['chunker']} {m['size']}/{m['overlap']}")):
        agg: Dict[Any, float] = defaultdict(float)
        for c, s in win.items():
            agg[key(meta[c])] += s
        pwin = key(meta[pw])
        for v, s in sorted(agg.items(), key=lambda kv: -kv[1]):
            overall.append({"level": level, "value": v, "share_first": s, "winner_in_data": v == pwin})
    return {"configuration": conf_rows, "model": model_rows, "overall": overall}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    AD.add_common_args(ap)
    ap.add_argument("--pools", nargs="+", choices=list(POOLS), default=list(POOLS))
    args = ap.parse_args(argv)
    resamples, seed = AD.resolve(args)
    runs = AD.load_runs(args.roots, args.qa, label_set=args.label_set)
    out = Path(args.out_dir)
    for pool in args.pools:
        res = analyse(runs, POOLS[pool], resamples=resamples, seed=seed)
        for name in ("configuration", "model", "overall"):
            AD.write_rows(out / f"winner_{pool}_by_{name}.csv" if name != "overall" else out / f"winner_{pool}_overall.csv", res[name])
        top = next((r for r in res["overall"] if r["level"] == "run"), None)
        print(f"pool {pool}: {len(runs.select(kind=POOLS[pool]))} runs; most frequent winner: {top['value'] if top else '-'} ({top['share_first']:.1%} of {resamples} resamples)"
              if top else f"pool {pool}: no runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
