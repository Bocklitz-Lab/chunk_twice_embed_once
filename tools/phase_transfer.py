"""Does Phase 1 predict Phase 2? (``make analysis``; Reviewer 2 item 16, ported from the Reviewer 1 experiment).

Phase 1 screens the models on two external retrieval tasks (ChemNQRetrieval, ChemHotpotQARetrieval); only the top 20% per task go to Phase 2
(ChemQuests). A correlation between the two phases among the 13 shortlisted models alone has a restricted range and says little; it needs
models that were NOT shortlisted. They are run by ``make transfer`` (``params.transfer``: models, configurations C1-C5, the extra C6, the pairs P1-P4
and the groups a-e of the Reviewer 1 experiment).

INPUTS
  --phase1-root     a Phase 1 results folder: the published run (no prompts, ``validation/``) or the run with prompts (``validation_prompts/``);
                    layout <root>/<model>_<revision>/<inner>/<revision>/<Task>.json; split test, else validation, else train (as stage 2)
  --shortlist       the stage 2 file of THAT Phase 1 run (passed_per_dataset*.csv or passed_models_union*.csv): which models were shortlisted
  --runs-overview   runs_overview.csv of the main root, then of the transfer root (first file wins for a model and configuration); dense runs only
Phase 2 score of a model = mean Geom@10 (span labels) over the transfer configurations C1-C5 (``params.transfer.configurations`` without extra);
C6 is reported separately and never enters the mean. Models with a configuration missing get no mean.

OUTPUT
  phase_transfer_correlations.csv  Spearman rho and Kendall tau-b (with p) and 95% bootstrap intervals over MODELS (models are resampled with
                                   replacement, ``params.report.resamples`` times; all tested models, shortlisted only, excluded only; Phase 1 per task
                                   and the mean of the two; Phase 2 mean C1-C5 and each configuration)
  phase_transfer_models.csv        one row per model: shortlisted, group, Phase 1 per task, Phase 2 per configuration and mean
  phase_transfer_pairs.csv         tuned - base per pair (P1-P4) and setting: Phase 2 C1-C6 and mean, Phase 1 per task and mean (nDCG@10, Recall@10, Geom@10)
  phase_transfer_groups.csv        mean / sd / min / max of Geom@10 per model group a-e (Phase 2 and Phase 1)
  phase_transfer_beats.csv         per configuration: best non-shortlisted vs best shortlisted model, how many non-shortlisted > shortlisted pairs, largest margin
  fig_phase_transfer.png / .pdf    Phase 1 (mean of both tasks) against Phase 2 (mean C1-C5), shortlisted vs excluded models

  python -m tools.phase_transfer --phase1-root artifacts/embedding_models_screening/validation_prompts --shortlist artifacts/embedding_models_screening/passed_per_dataset__validation_prompts.csv \\
      --runs-overview results/v2-revision/runs_overview.csv results/v2-revision/transfer/runs_overview.csv --out-dir results/analysis
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import config_params  # noqa: E402
from tools import analysis_data as AD  # noqa: E402

TASKS = ("ChemNQRetrieval", "ChemHotpotQARetrieval")
SPLITS = ("test", "validation", "train")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
# colours: slots 1 and 2 of the validated reference palette (references/palette.md of the dataviz skill); the marker shape is the second encoding
C_SHORT, C_EXCL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def geom(ndcg: float, recall: float) -> float:
    return float((ndcg * recall) ** 0.5)


# ------------------------------------------------------------------------------------------------------------------
# inputs
# ------------------------------------------------------------------------------------------------------------------
def load_phase1(root: Any, models: Sequence[Tuple[str, Optional[str]]], tasks: Sequence[str] = TASKS) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{model: {task: {ndcg_at_10, recall_at_10, geom_at_10, split}}} for the models that have every task in ``root`` (the others are left out)."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, rev in models:
        safe = name.replace("/", "__").replace(":", "_")
        found: Dict[str, Dict[str, float]] = {}
        for t in tasks:
            hits = [h for h in glob.glob(str(Path(root) / f"{safe}_*" / "*" / "*" / f"{t}.json"))
                    if re.fullmatch(re.escape(safe) + r"_(main|[0-9a-f]{40})", Path(h).parents[2].name)]
            if rev:
                pinned = [h for h in hits if Path(h).parents[2].name == f"{safe}_{rev}"]
                hits = pinned or hits
            if len(hits) != 1:
                break
            sc = json.loads(Path(hits[0]).read_text(encoding="utf-8"))["scores"]
            split = next((s for s in SPLITS if s in sc), None)
            if split is None:
                break
            s = sc[split][0]
            found[t] = {"ndcg_at_10": s["ndcg_at_10"], "recall_at_10": s["recall_at_10"], "geom_at_10": geom(s["ndcg_at_10"], s["recall_at_10"]), "split": split}
        if len(found) == len(tasks):
            out[name] = found
    return out


def read_shortlist(path: Any) -> set:
    """Model names of a stage 2 file (``model_name`` column, else ``model_id`` = name@revision)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["model_name"] if r.get("model_name") else r["model_id"].split("@")[0] for r in rows}


def read_phase2(overviews: Sequence[Any], label: str = "span") -> Dict[Tuple[str, str, int, int], Dict[str, float]]:
    """{(model, chunker, size, overlap): {geom, ndcg, recall}} of the dense runs of the overview files; the first file wins."""
    out: Dict[Tuple[str, str, int, int], Dict[str, float]] = {}
    for p in overviews:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                kind = r.get("kind") or ("dense" if not r["model"].startswith(("bm25", "hybrid_", "rerank_")) else "other")      # older tables have no kind column
                if kind != "dense":
                    continue
                key = (r["model"], r["chunker"], int(r["size"]), int(r["overlap"]))
                if key not in out and r.get(f"{label}_geom_at_10") not in (None, ""):
                    out[key] = {"geom_at_10": float(r[f"{label}_geom_at_10"]), "ndcg_at_10": float(r[f"{label}_ndcg_at_10"]), "recall_at_10": float(r[f"{label}_recall_at_10"])}
    return out


# ------------------------------------------------------------------------------------------------------------------
# rank correlations with a bootstrap over models
# ------------------------------------------------------------------------------------------------------------------
def spearman_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Spearman rho of each row pair of X, Y (B, n): Pearson correlation of the average ranks."""
    from scipy.stats import rankdata
    rx, ry = rankdata(X, axis=1), rankdata(Y, axis=1)
    rx, ry = rx - rx.mean(axis=1, keepdims=True), ry - ry.mean(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (rx * ry).sum(axis=1) / np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))


def kendall_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Kendall tau-b of each row pair (B, n): sum of sign products over the pairs / sqrt(untied pairs in x * untied pairs in y)."""
    sx = np.sign(X[:, :, None] - X[:, None, :])
    sy = np.sign(Y[:, :, None] - Y[:, None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        return (sx * sy).sum(axis=(1, 2)) / np.sqrt((sx ** 2).sum(axis=(1, 2)) * (sy ** 2).sum(axis=(1, 2)))


def correlation(x: Sequence[float], y: Sequence[float], resamples: int, seed: int) -> Dict[str, Any]:
    """Spearman and Kendall with p and a percentile interval over models; NaN below four models or with a constant variable."""
    from scipy.stats import kendalltau, spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    nan = {"n_models": n, "spearman": np.nan, "spearman_p": np.nan, "spearman_ci_low": np.nan, "spearman_ci_high": np.nan,
           "kendall": np.nan, "kendall_p": np.nan, "kendall_ci_low": np.nan, "kendall_ci_high": np.nan}
    if n < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return nan
    sp, kt = spearmanr(x, y), kendalltau(x, y, variant="b")
    idx = np.random.default_rng(seed).integers(0, n, size=(resamples, n))
    out = dict(nan, spearman=float(sp.statistic), spearman_p=float(sp.pvalue), kendall=float(kt.statistic), kendall_p=float(kt.pvalue))
    for name, fn in (("spearman", spearman_rows), ("kendall", kendall_rows)):
        v = fn(x[idx], y[idx])
        v = v[np.isfinite(v)]
        if len(v):
            out[f"{name}_ci_low"], out[f"{name}_ci_high"] = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    return out


# ------------------------------------------------------------------------------------------------------------------
# tables
# ------------------------------------------------------------------------------------------------------------------
def analyse(p1: Dict[str, Dict[str, Dict[str, float]]], p2: Dict[Tuple[str, str, int, int], Dict[str, float]], shortlist: set, transfer: Dict[str, Any],
            resamples: int, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    cfgs = transfer["configurations"]
    main_cfgs = [c for c in cfgs if not c.get("extra")]
    cid = {c["id"]: (c["chunker"], int(c["size"]), int(c["overlap"])) for c in cfgs}
    reg = {m["name"]: m for m in transfer["models"]}
    names = sorted({k[0] for k in p2} & set(p1))

    def p2v(m: str, c: str, k: str = "geom_at_10") -> float:
        v = p2.get((m,) + cid[c])
        return v[k] if v else np.nan

    def p2mean(m: str, k: str = "geom_at_10") -> float:
        vals = [p2v(m, c["id"], k) for c in main_cfgs]
        return float(np.mean(vals)) if not np.any(np.isnan(vals)) else np.nan

    def p1v(m: str, t: str, k: str = "geom_at_10") -> float:
        return float(np.mean([p1[m][x][k] for x in TASKS])) if t == "mean" else p1[m][t][k]

    models = []
    for m in names:
        row = {"model": m, "shortlisted": m in shortlist, "group": (reg.get(m) or {}).get("group", "d" if m in shortlist else "?"), "role": (reg.get(m) or {}).get("role", "")}
        for t in (*TASKS, "mean"):
            row[f"phase1_geom_{t}"] = p1v(m, t)
        for c in cfgs:
            row[f"phase2_geom_{c['id']}"] = p2v(m, c["id"])
        row["phase2_geom_mean_main"] = p2mean(m)
        models.append(row)
    corr = []
    pops = {"all models": names, "shortlisted only": [m for m in names if m in shortlist], "excluded only": [m for m in names if m not in shortlist]}
    ys = [("mean_main", lambda m: p2mean(m))] + [(c["id"], (lambda m, c=c: p2v(m, c["id"]))) for c in cfgs]
    for pop, ms in pops.items():
        for t in (*TASKS, "mean"):
            for yname, yfn in ys:
                ok = [m for m in ms if not np.isnan(yfn(m))]
                corr.append({"population": pop, "phase1": t, "phase2": yname, **correlation([p1v(m, t) for m in ok], [yfn(m) for m in ok], resamples, seed),
                             "resamples": resamples, "seed": seed})
    pair_rows = []
    for pr in transfer.get("pairs") or []:
        for tuned in pr["tuned"]:
            base = pr["base"]
            settings = [(f"Phase 2 {c['id']}" + (" (extra)" if c.get("extra") else ""), (lambda m, k, cid=c["id"]: p2v(m, cid, k))) for c in cfgs]
            settings.append(("Phase 2 mean main configurations", lambda m, k: p2mean(m, k)))
            settings += [(f"Phase 1 {t}", (lambda m, k, t=t: p1v(m, t, k) if m in p1 else np.nan)) for t in (*TASKS, "mean")]
            for label, fn in settings:
                row = {"pair": pr["id"], "base": base, "tuned": tuned, "setting": label}
                for k in ("geom_at_10", "ndcg_at_10", "recall_at_10"):
                    b_, t_ = fn(base, k), fn(tuned, k)
                    row[f"base_{k}"], row[f"tuned_{k}"], row[f"delta_{k}"] = b_, t_, t_ - b_
                pair_rows.append(row)
    group_rows = []
    gnames = transfer.get("groups") or {}
    for g, desc in gnames.items():
        mem = [m for m in names if (reg.get(m) or {}).get("group") == g and (reg.get(m) or {}).get("role") != "extra"]
        for label, fn in [(f"Phase 2 {c['id']}", (lambda m, c=c: p2v(m, c["id"]))) for c in cfgs] + [("Phase 2 mean main", p2mean)] + \
                         [(f"Phase 1 {t}", (lambda m, t=t: p1v(m, t))) for t in (*TASKS, "mean")]:
            v = [x for x in (fn(m) for m in mem) if not np.isnan(x)]
            group_rows.append({"group": g, "description": desc, "setting": label, "n_models": len(v), "mean_geom_at_10": float(np.mean(v)) if v else np.nan,
                               "sd_geom_at_10": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan, "min": float(np.min(v)) if v else np.nan,
                               "max": float(np.max(v)) if v else np.nan, "members": "; ".join(x.split("/")[-1] for x in mem)})
    beats = []
    short_m = [m for m in names if m in shortlist and (reg.get(m) or {}).get("role") != "extra"]
    other_m = [m for m in names if m not in shortlist and (reg.get(m) or {}).get("role") != "extra"]
    for label, fn in [(c["id"], (lambda m, c=c: p2v(m, c["id"]))) for c in cfgs] + [("mean main", p2mean)]:
        S = {m: fn(m) for m in short_m if not np.isnan(fn(m))}
        O = {m: fn(m) for m in other_m if not np.isnan(fn(m))}
        if not S or not O:
            continue
        bo, bs = max(O, key=O.get), max(S, key=S.get)
        won = [(o, s, O[o] - S[s]) for o in O for s in S if O[o] > S[s]]
        top = max(won, key=lambda w: w[2]) if won else None
        beats.append({"setting": label, "best_excluded": bo, "best_excluded_geom": O[bo], "best_shortlisted": bs, "best_shortlisted_geom": S[bs],
                      "gap_best_excluded_minus_best_shortlisted": O[bo] - S[bs], "n_pairs_excluded_beats_shortlisted": len(won), "n_pairs": len(O) * len(S),
                      "largest_margin": top[2] if top else np.nan, "largest_margin_pair": f"{top[0]} > {top[1]}" if top else "",
                      "excluded_models_that_beat_any_shortlisted": "; ".join(sorted({w[0] for w in won}))})
    return {"correlations": corr, "models": models, "pairs": pair_rows, "groups": group_rows, "beats": beats}


# ------------------------------------------------------------------------------------------------------------------
# figure
# ------------------------------------------------------------------------------------------------------------------
def scatter(models: List[Dict[str, Any]], corr: List[Dict[str, Any]], out_base: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = [m for m in models if not np.isnan(m["phase2_geom_mean_main"])]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for short, color, marker, label in ((True, C_SHORT, "o", "shortlisted (Phase 2)"), (False, C_EXCL, "s", "not shortlisted")):
        sel = [m for m in pts if m["shortlisted"] == short]
        ax.scatter([m["phase1_geom_mean"] for m in sel], [m["phase2_geom_mean_main"] for m in sel], s=46, c=color, marker=marker, edgecolors="white", linewidths=1.2,
                   label=f"{label} (n={len(sel)})", zorder=3)
        for m in sel:
            ax.annotate(m["model"].split("/")[-1], (m["phase1_geom_mean"], m["phase2_geom_mean_main"]), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=MUTED)
    texts = []
    for pop in ("all models", "shortlisted only", "excluded only"):
        r = next((c for c in corr if c["population"] == pop and c["phase1"] == "mean" and c["phase2"] == "mean_main"), None)
        if r and not np.isnan(r["spearman"]):
            texts.append(f"{pop}: Spearman {r['spearman']:.2f} [{r['spearman_ci_low']:.2f}, {r['spearman_ci_high']:.2f}], n={r['n_models']}")
    ax.text(0.02, 0.98, "\n".join(texts), transform=ax.transAxes, va="top", fontsize=7.5, color=INK)
    ax.set_xlabel("Phase 1: Geom@10, mean of ChemNQ and ChemHotpotQA", color=INK)
    ax.set_ylabel("Phase 2: Geom@10 on ChemQuests, mean of C1-C5", color=INK)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase1-root", required=True)
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--runs-overview", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--main-config", default=str(config_params.DEFAULT_MAIN_CONFIG))
    ap.add_argument("--label-set", choices=["span", "lexical"], default="span")
    ap.add_argument("--resamples", type=int)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args(argv)
    d = AD.defaults(args.main_config)
    resamples, seed = args.resamples or d["resamples"], args.seed if args.seed is not None else d["seed"]
    transfer = config_params.section("transfer", args.main_config)
    cand = {m["name"]: m["revision"] for m in transfer["models"]}
    shortlist = read_shortlist(args.shortlist)
    stage1 = {m["name"]: m["revision"] for m in __import__("yaml").safe_load(Path(args.main_config).read_text(encoding="utf-8"))["stage1"]["models"]}
    p1 = load_phase1(args.phase1_root, [(n, stage1.get(n) or cand.get(n)) for n in sorted(set(stage1) | set(cand) | shortlist)])
    p2 = read_phase2(args.runs_overview, args.label_set)
    res = analyse(p1, p2, shortlist, transfer, resamples, seed)
    out = Path(args.out_dir)
    for name, fname in (("correlations", "phase_transfer_correlations.csv"), ("models", "phase_transfer_models.csv"), ("pairs", "phase_transfer_pairs.csv"),
                        ("groups", "phase_transfer_groups.csv"), ("beats", "phase_transfer_beats.csv")):
        AD.write_rows(out / fname, res[name])
    scatter(res["models"], res["correlations"], out / "fig_phase_transfer")
    n_both = len(res["models"])
    allm = next((c for c in res["correlations"] if c["population"] == "all models" and c["phase1"] == "mean" and c["phase2"] == "mean_main"), {})
    print(f"{n_both} models with Phase 1 and Phase 2 -> {out}; all models: Spearman {allm.get('spearman', float('nan')):.3f}, n={allm.get('n_models')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
