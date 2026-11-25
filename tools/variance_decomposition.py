"""Variance decomposition of Geom@10 over the dense runs (``make analysis``; Reviewer 2 item 20): how much of the differences between the
retrieval results is due to the embedding model, and how much to the chunking?

DATA: one value per dense run (model x strategy x size x overlap): Geom@10 = sqrt(mean nDCG@10 x mean Recall@10) over the 952 questions.
The grid is not a full factorial (28 valid size/overlap pairs of 35: size 512 only with overlap 0, size 128 only up to overlap 32), so the sums of
squares are Type II on an unbalanced layout, computed with least squares (numpy; no statsmodels).

MODEL: main effects of model, strategy (chunker), size and overlap (all categorical) and the interaction model x strategy.
  Type II sum of squares of a main effect f: RSS(main effects without f) - RSS(all main effects)
  of the interaction:                         RSS(all main effects) - RSS(main effects + model x strategy)
  eta^2_f = SS_f / SS_total, SS_total = sum (y - mean y)^2; the rest is the residual (size x overlap and the other interactions, not modelled).
In an unbalanced design the eta^2 do not add up to 1 minus the residual exactly (Type II sums of squares are not additive).

INTERVALS: paper-level bootstrap (``analysis_data.PaperBootstrap``; params.report.resamples, default 10,000): in every resample the papers are redrawn,
Geom@10 of every run is recomputed from the per-query values, the same design is fitted again; percentile intervals of every eta^2 and of
eta^2_model - eta^2_strategy. ``--balanced`` repeats everything on the largest balanced sub-grid (the same sizes x overlaps for every model and
strategy; with the published grid: sizes 256-448 x overlaps 0-64, 20 cells; sizes 192-448 x overlaps 0-48 is as large, ties prefer more overlaps).

SIMPLE RANGES (``variance_ranges.csv``): range_models = for each strategy the spread (max - min) of the model means over the strategy's
configurations, averaged over the strategies; range_strategies = for each model the spread of the strategy means, averaged over the models; and their
difference, each with the same paper-level interval. "How much does the choice of model move the result at a fixed chunker, compared with the choice of chunker at a
fixed model?" in units of Geom@10.

  python -m tools.variance_decomposition --roots artifacts --qa artifacts/chemquests_hf_db/qa.jsonl --out-dir results/analysis [--balanced]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import grid  # noqa: E402
from tools import analysis_data as AD  # noqa: E402

FACTORS = ("model", "strategy", "size", "overlap")
INTERACTION = ("model", "strategy")


# ------------------------------------------------------------------------------------------------------------------
# linear algebra
# ------------------------------------------------------------------------------------------------------------------
def dummies(codes: np.ndarray) -> np.ndarray:
    """Treatment-coded dummy columns (the first level is dropped) of integer level codes."""
    levels = np.unique(codes)
    return (codes[:, None] == levels[None, 1:]).astype(float)


def column_basis(X: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the column space of X (rank-revealing SVD): projections do not depend on the coding of the factors."""
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    tol = max(X.shape) * np.finfo(float).eps * (s[0] if len(s) else 0.0)
    return u[:, : int((s > tol).sum())]


def design_bases(factors: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Column-space bases of the intercept, the full main-effects model, each 'main effects without f' model and the model with the interaction."""
    n = len(next(iter(factors.values())))
    D = {f: dummies(factors[f]) for f in FACTORS}
    one = np.ones((n, 1))
    full = np.hstack([one] + [D[f] for f in FACTORS])
    bases = {"intercept": column_basis(one), "main": column_basis(full)}
    for f in FACTORS:
        bases[f"without_{f}"] = column_basis(np.hstack([one] + [D[g] for g in FACTORS if g != f]))
    a, b = D[INTERACTION[0]], D[INTERACTION[1]]
    inter = np.hstack([a[:, [i]] * b[:, [j]] for i in range(a.shape[1]) for j in range(b.shape[1])]) if a.size and b.size else np.zeros((n, 0))
    bases["with_interaction"] = column_basis(np.hstack([full, inter]))
    return bases


def explained(bases: Dict[str, np.ndarray], Y: np.ndarray) -> Dict[str, np.ndarray]:
    """||Q' y||^2 for every basis; ``Y`` is (B, N) (rows = resamples or one row = the data): the part of the sum of squares each model explains."""
    return {k: ((Y @ Q) ** 2).sum(axis=1) for k, Q in bases.items()}


def type2(bases: Dict[str, np.ndarray], Y: np.ndarray) -> Dict[str, np.ndarray]:
    """{name: array over the rows of Y}: ss_<factor>, ss_model_x_strategy, ss_residual, ss_total, eta2_<factor>, eta2_model_x_strategy, eta2_residual."""
    ex = explained(bases, Y)
    total = (Y ** 2).sum(axis=1) - Y.shape[1] * Y.mean(axis=1) ** 2
    out: Dict[str, np.ndarray] = {f"ss_{f}": ex["main"] - ex[f"without_{f}"] for f in FACTORS}
    out["ss_model_x_strategy"] = ex["with_interaction"] - ex["main"]
    out["ss_residual"] = (Y ** 2).sum(axis=1) - ex["with_interaction"]
    out["ss_total"] = total
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in [f"ss_{f}" for f in FACTORS] + ["ss_model_x_strategy", "ss_residual"]:
            out["eta2_" + k[3:]] = out[k] / total
    return out


# ------------------------------------------------------------------------------------------------------------------
# the data
# ------------------------------------------------------------------------------------------------------------------
def factor_codes(meta: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    def codes(vals):
        u = sorted(set(vals))
        return np.array([u.index(v) for v in vals])
    return {"model": codes([m["model"] for m in meta]), "strategy": codes([m["chunker"] for m in meta]),
            "size": codes([m["size"] for m in meta]), "overlap": codes([m["overlap"] for m in meta])}


def balanced_rows(meta: Sequence[Dict[str, Any]]) -> Tuple[List[int], List[int], List[int]]:
    """(row indices, sizes, overlaps) of the largest balanced sub-grid: sizes x overlaps present for EVERY (model, strategy); ties prefer more overlaps."""
    cells_by = {}
    for i, m in enumerate(meta):
        cells_by.setdefault((m["model"], m["chunker"]), set()).add((m["size"], m["overlap"]))
    common = set.intersection(*cells_by.values()) if cells_by else set()
    overlaps = sorted({o for _, o in common})
    best: Tuple[int, int, List[int], List[int]] = (0, 0, [], [])
    for r in range(1, len(overlaps) + 1):
        for sub in itertools.combinations(overlaps, r):
            sizes = sorted(s for s in {s for s, _ in common} if all((s, o) in common for o in sub))
            cand = (len(sizes) * len(sub), len(sub), sizes, list(sub))
            if cand[:2] > best[:2]:
                best = cand
    sizes, ovs = best[2], best[3]
    return [i for i, m in enumerate(meta) if m["size"] in sizes and m["overlap"] in ovs], sizes, ovs


def range_statistics(G: np.ndarray, model_codes: np.ndarray, strategy_codes: np.ndarray) -> Dict[str, np.ndarray]:
    """Simple ranges from Geom@10 values ``G`` of shape (B, N): see the module docstring. Returns arrays over the B rows."""
    M, S = model_codes.max() + 1, strategy_codes.max() + 1
    cell = model_codes * S + strategy_codes
    A = np.zeros((len(cell), M * S))
    A[np.arange(len(cell)), cell] = 1.0
    cnt = A.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = ((G @ A) / np.where(cnt > 0, cnt, np.nan)).reshape(G.shape[0], M, S)
    r_models = (np.nanmax(means, axis=1) - np.nanmin(means, axis=1)).mean(axis=1)             # spread over models at a fixed strategy, mean over strategies
    r_strats = (np.nanmax(means, axis=2) - np.nanmin(means, axis=2)).mean(axis=1)             # spread over strategies at a fixed model, mean over models
    return {"range_models": r_models, "range_strategies": r_strats, "range_models_minus_strategies": r_models - r_strats}


def ci(arr: np.ndarray) -> Tuple[float, float]:
    a = arr[np.isfinite(arr)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (float("nan"), float("nan"))


def analyse(runs: AD.RunSet, *, resamples: int, seed: int, balanced: bool = False) -> Dict[str, Any]:
    # the control strategies (grid.CONTROL_CELLS: a few cells only) are not part of the model x strategy x size x overlap design
    dense = [i for i in runs.select(kind="dense") if runs.meta[i]["chunker"] not in grid.CONTROL_CELLS]
    if balanced:
        keep, sizes, ovs = balanced_rows([runs.meta[i] for i in dense])
        dense = [dense[k] for k in keep]
    else:
        sizes = sorted({runs.meta[i]["size"] for i in dense})
        ovs = sorted({runs.meta[i]["overlap"] for i in dense})
    meta = [runs.meta[i] for i in dense]
    if len({m["model"] for m in meta}) < 2 or len({m["chunker"] for m in meta}) < 2:
        raise ValueError("variance decomposition needs at least two models and two strategies among the dense runs")
    f = factor_codes(meta)
    bases = design_bases(f)
    nd, rc = runs.metrics["ndcg_at_10"][dense], runs.metrics["recall_at_10"][dense]
    y = np.sqrt(nd.mean(axis=1) * rc.mean(axis=1))[None, :]
    point = type2(bases, y)
    point_r = range_statistics(y, f["model"], f["strategy"])
    boot = AD.PaperBootstrap(runs.groups, resamples, seed)
    Gs = boot.geom_resamples(nd, rc)                                   # (B, N)
    res = type2(bases, Gs)
    res_r = range_statistics(Gs, f["model"], f["strategy"])
    names = list(FACTORS) + ["model_x_strategy", "residual"]
    df = {n: int(np.linalg.matrix_rank(bases["main"]) - np.linalg.matrix_rank(bases[f"without_{n}"])) if n in FACTORS else
          (int(np.linalg.matrix_rank(bases["with_interaction"]) - np.linalg.matrix_rank(bases["main"])) if n == "model_x_strategy" else
           len(dense) - int(np.linalg.matrix_rank(bases["with_interaction"]))) for n in names}
    rows = []
    for n in names:
        lo, hi = ci(res[f"eta2_{n}"])
        rows.append({"term": n, "df": df[n], "ss": float(point[f"ss_{n}"][0]), "eta2": float(point[f"eta2_{n}"][0]), "eta2_ci_low": lo, "eta2_ci_high": hi})
    d = res["eta2_model"] - res["eta2_strategy"]
    lo, hi = ci(d)
    rows.append({"term": "model_minus_strategy", "df": "", "ss": "", "eta2": float(point["eta2_model"][0] - point["eta2_strategy"][0]), "eta2_ci_low": lo, "eta2_ci_high": hi})
    rrows = []
    for k in ("range_models", "range_strategies", "range_models_minus_strategies"):
        lo, hi = ci(res_r[k])
        rrows.append({"statistic": k, "value": float(point_r[k][0]), "ci_low": lo, "ci_high": hi})
    info = {"n_runs": len(dense), "n_models": int(f["model"].max() + 1), "n_strategies": int(f["strategy"].max() + 1), "sizes": sizes, "overlaps": ovs,
            "balanced": balanced, "resamples": resamples, "seed": seed, "ss_total": float(point["ss_total"][0])}
    return {"rows": rows, "ranges": rrows, "info": info}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    AD.add_common_args(ap)
    ap.add_argument("--balanced", action="store_true", help="also repeat on the largest balanced sub-grid")
    args = ap.parse_args(argv)
    resamples, seed = AD.resolve(args)
    runs = AD.load_runs(args.roots, args.qa, kinds=["dense"], label_set=args.label_set)
    out = Path(args.out_dir)
    for balanced in ([False, True] if args.balanced else [False]):
        res = analyse(runs, resamples=resamples, seed=seed, balanced=balanced)
        tag = "_balanced" if balanced else ""
        AD.write_rows(out / f"variance_decomposition{tag}.csv", res["rows"])
        AD.write_rows(out / f"variance_ranges{tag}.csv", res["ranges"])
        (out / f"variance_decomposition{tag}.json").write_text(json.dumps(res["info"], indent=2), encoding="utf-8")
        eta = {r["term"]: r["eta2"] for r in res["rows"]}
        print(f"{'balanced' if balanced else 'all'}: {res['info']['n_runs']} runs; eta2 model {eta['model']:.3f}, strategy {eta['strategy']:.3f}, size {eta['size']:.3f}, "
              f"overlap {eta['overlap']:.3f}, model x strategy {eta['model_x_strategy']:.3f}, residual {eta['residual']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
