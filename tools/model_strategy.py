"""Model x chunking-strategy table and figure, with means and paper-level intervals (replaces the heatmap of MAXIMA; Reviewer 2: conclusions must not rest on
maximum scores or post-hoc top configurations).

For every (model, strategy) of the DENSE runs: the MEAN Geom@10 over all valid size / overlap settings of the grid that were run (not the maximum), with a 95 %
paper-level percentile interval. In every paper-level resample (``analysis_data.PaperBootstrap``, params.report.resamples, params.seed: the draws behind
``stats_ci.csv``) the papers are redrawn, Geom@10 of every run is recomputed from its per-query nDCG@10 and Recall@10, and the runs of the cell are averaged.
Also written per cell:
  share_best_model_in_strategy   share of resamples in which the model has the highest mean of its strategy (which model is first FOR THIS STRATEGY)
  winner_share_overall           the share of resamples in which the single best run of ALL dense runs belongs to this (model, strategy):
                                 the sum of the ``run`` shares of ``tools/winner_stability.py`` (same resamples)
  rank_in_strategy               rank of the mean among the models of the strategy
Outputs ``table_model_strategy.csv``, ``fig_model_strategy.png/.pdf`` and ``fig_model_strategy.json`` (inputs, resamples, seed, number of runs per cell).

  python -m tools.model_strategy --roots artifacts --qa artifacts/chemquests_hf_db/qa.jsonl --out-dir results/v2-revision
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools import analysis_data as AD  # noqa: E402
from tools import winner_stability  # noqa: E402


def analyse(runs: AD.RunSet, *, resamples: int, seed: int) -> List[Dict[str, Any]]:
    dense = runs.select(kind="dense")
    if not dense:
        raise ValueError("no dense runs")
    meta = [runs.meta[i] for i in dense]
    boot = AD.PaperBootstrap(runs.groups, resamples, seed)
    G = boot.geom_resamples(runs.metrics["ndcg_at_10"][dense], runs.metrics["recall_at_10"][dense])        # (B, R)
    point = runs.geom(dense)
    cells: Dict[tuple, List[int]] = defaultdict(list)
    for j, m in enumerate(meta):
        cells[(m["model"], m["chunker"])].append(j)
    strategies = sorted({k[1] for k in cells})
    mean_b = {k: G[:, cols].mean(axis=1) for k, cols in cells.items()}                                       # (B,) mean over the cell's runs in each resample
    mean_p = {k: float(point[cols].mean()) for k, cols in cells.items()}
    win = winner_stability.analyse(runs, ["dense"], resamples=resamples, seed=seed)["overall"]
    overall: Dict[tuple, float] = defaultdict(float)
    for r in win:
        if r["level"] == "run":
            model, rest = r["value"].split(" | ")
            overall[(model, rest.split(" ")[0])] += r["share_first"]
    rows = []
    for s in strategies:
        models = sorted(m for (m, st) in cells if st == s)
        stack = np.vstack([mean_b[(m, s)] for m in models])                                                # (M, B)
        first = np.argmax(stack, axis=0)
        ranks = {m: 1 + sum(mean_p[(o, s)] > mean_p[(m, s)] for o in models) for m in models}
        for i, m in enumerate(models):
            lo, hi = np.quantile(mean_b[(m, s)], [0.025, 0.975])
            rows.append({"model": m, "strategy": s, "n_runs": len(cells[(m, s)]), "settings": "; ".join(sorted({f"{meta[j]['size']}/{meta[j]['overlap']}" for j in cells[(m, s)]}, key=lambda x: tuple(map(int, x.split("/"))))), "mean_geom_at_10": mean_p[(m, s)], "ci_low": float(lo), "ci_high": float(hi),
                         "rank_in_strategy": ranks[m], "share_best_model_in_strategy": float((first == i).mean()), "winner_share_overall": overall.get((m, s.split(" ")[0]), 0.0),
                         "resamples": resamples, "seed": seed})
    return rows


def draw(rows: List[Dict[str, Any]], out_dir: Path, meta: Dict[str, Any]) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools import figures as F
    F.style()
    models = sorted({r["model"] for r in rows}, key=lambda m: -np.mean([r["mean_geom_at_10"] for r in rows if r["model"] == m]))
    strat = [s for s in F.CHUNKER_ORDER if s in {r["strategy"] for r in rows}] + sorted({r["strategy"] for r in rows} - set(F.CHUNKER_ORDER))
    by = {(r["model"], r["strategy"]): r for r in rows}
    A = np.array([[by[(m, s)]["mean_geom_at_10"] if (m, s) in by else np.nan for s in strat] for m in models])
    W = np.array([[by[(m, s)]["winner_share_overall"] if (m, s) in by else np.nan for s in strat] for m in models])
    h = (34 + 7.2 * len(models)) * F.MM
    fig, axes = plt.subplots(1, 2, figsize=(F.FULL_WIDTH_MM * F.MM, h), gridspec_kw={"width_ratios": [1.25, 0.75], "wspace": 0.12}, sharey=True)
    for ax, M, title, cmap, fmt in ((axes[0], A, "mean Geom@10 over the settings run\n[95 % paper-level CI]", "viridis", "ci"),
                                    (axes[1], W, "share of resamples in which\nthe best run is of this cell", "Greys", "share")):
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=np.nanmin(M) if fmt == "ci" else 0, vmax=np.nanmax(M))
        ax.set_xticks(range(len(strat)))
        ax.set_xticklabels([s.replace("_", " ") for s in strat], fontsize=F.FONT, rotation=30, ha="right")
        ax.set_title(title, fontsize=F.FONT, loc="left")
        ax.tick_params(length=0)
        for i, m in enumerate(models):
            for j, s in enumerate(strat):
                r = by.get((m, s))
                if r is None:
                    continue
                v = M[i, j]
                dark = (v - np.nanmin(M)) / ((np.nanmax(M) - np.nanmin(M)) or 1) < (0.55 if fmt == "ci" else 0.5)
                txt = f"{r['mean_geom_at_10']:.3f}\n[{r['ci_low']:.3f}, {r['ci_high']:.3f}]" if fmt == "ci" else f"{r['winner_share_overall']:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=F.FONT - 1.5, color="white" if (dark if fmt == "ci" else not dark) else "black")
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels([m.split("/")[-1] for m in models], fontsize=F.FONT)
    return F.save(fig, out_dir, "fig_model_strategy", {**meta, "models": models, "strategies": strat, "cell": "mean over the valid size / overlap settings (not the maximum)",
                                                     "right_panel": "winner_share_overall (tools/winner_stability.py)"})


def run(roots, qa, out_dir, *, resamples: int, seed: int, label_set: str = "span", figure: bool = True) -> Dict[str, Any]:
    runs = AD.load_runs(roots, qa, kinds=["dense"], label_set=label_set)
    rows = analyse(runs, resamples=resamples, seed=seed)
    out = Path(out_dir)
    AD.write_rows(out / "table_model_strategy.csv", rows)
    files = []
    if figure:
        files = draw(rows, out, {"inputs": [str(r) for r in roots], "n_dense_runs": len(runs.select(kind="dense")), "resamples": resamples, "seed": seed,
                                 "runs_per_cell": sorted({r["n_runs"] for r in rows})})
    return {"rows": len(rows), "files": files}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    AD.add_common_args(ap)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    resamples, seed = AD.resolve(args)
    res = run(args.roots, args.qa, args.out_dir, resamples=resamples, seed=seed, label_set=args.label_set, figure=not args.no_figure)
    print(f"{res['rows']} model x strategy cells -> {args.out_dir}/table_model_strategy.csv ({resamples} paper-level resamples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
