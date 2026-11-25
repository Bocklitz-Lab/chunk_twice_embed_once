"""Efficiency tables and plots for the revision (reads efficiency_phase{1,2}.csv and the existing score table).

  python -m tools.efficiency_analysis --efficiency artifacts/tests/efficiency_phase2.csv \\
      --scores results/v2-revision/runs_overview.csv --out results/efficiency

Writes to --out:
  (a) table_efficiency_models.{csv,tex}   one row per model: params, weight size, dim, peak GPU memory, latency, best Geom@10,
                                          index size (memory / latency / index columns at the model's best-Geom@10 configuration;
                                          if latency was not measured there: the median over the model's measured configurations,
                                          flagged in `latency_basis`)
  (b) table_index_size.{csv,tex}          index size (MB, fp32) by chunk size x overlap, mean over chunkers and models;
      table_index_chunks.csv              the number of chunks (model independent), long form per chunker in table_index_size_long.csv
  (c) fig_geom_vs_latency.{png,pdf}       Geom@10 vs total per-query latency, one point per configuration, Pareto front
  (d) fig_geom_vs_index_size.{png,pdf}    Geom@10 vs index size (fp32), Pareto front
  pareto_latency.csv / pareto_index_size.csv   the points on each front
  table_efficiency_phase1.{csv,tex}       Phase 1 models (if --phase1 is given)

Geom@10 = ``span_geom_at_10`` of the score table (tools/collect_runs.py output, e.g. results/v2-revision/runs_overview.csv);
change with --score-col.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import phase1_split  # noqa: E402

KEY = ["model", "chunker", "size", "overlap"]
# categorical slots 1-5 of the reference palette (validated, light surface), fixed order; marker shape is the second encoding
CHUNKERS = ["recursive_token", "fixed_token", "semantic_recursive", "semantic_fixed", "hierarchical_section"]
COLORS = dict(zip(CHUNKERS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARKERS = dict(zip(CHUNKERS, ["o", "s", "^", "D", "v"]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------------------------------------- LaTeX
def _esc(s) -> str:
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def to_latex(df: pd.DataFrame, path: Path, caption: str, label: str, fmt: Optional[dict] = None, index: bool = False) -> None:
    """booktabs table without a jinja2 dependency. ``fmt``: {column: format string}; NaN prints as --."""
    fmt = fmt or {}
    d = df.reset_index() if index else df
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{_esc(caption)}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + "l" + "r" * (d.shape[1] - 1) + "}", r"\toprule",
             " & ".join(_esc(c) for c in d.columns) + r" \\", r"\midrule"]
    for _, r in d.iterrows():
        cells = []
        for c in d.columns:
            v = r[c]
            cells.append("--" if (v is None or (isinstance(v, float) and np.isnan(v))) else (fmt[c].format(v) if c in fmt else _esc(v)))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------- data
def load(eff_csv: Path, scores_csv: Path, score_col: str) -> pd.DataFrame:
    e = pd.read_csv(eff_csv)
    s = pd.read_csv(scores_csv)
    if "kind" in s:
        s = s[s["kind"] == "dense"]
    s = s[KEY + [score_col]].drop_duplicates(KEY)
    d = e.merge(s, on=KEY, how="left")
    miss = d[score_col].isna().sum()
    if miss:
        print(f"[WARN] {miss} efficiency rows have no score in {scores_csv}; they are left out of the plots")
    return d


def pareto(df: pd.DataFrame, cost: str, gain: str) -> pd.DataFrame:
    """Points not dominated by any other (lower cost AND higher gain), sorted by cost."""
    d = df.dropna(subset=[cost, gain]).sort_values([cost, gain], ascending=[True, False])
    best, keep = -np.inf, []
    for i, r in d.iterrows():
        if r[gain] > best:
            keep.append(i)
            best = r[gain]
    return d.loc[keep]


# ---------------------------------------------------------------------------------------------- tables
def _at_best_or_median(g: pd.DataFrame, best, col: str):
    """Value at the best configuration, else the median over the model's configurations that have one."""
    if col not in g:
        return np.nan
    v = best.get(col)
    return v if not pd.isna(v) else (g[col].median() if g[col].notna().any() else np.nan)


def model_table(d: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for model, g in d.groupby("model"):
        sc = g.dropna(subset=[score_col])
        if sc.empty:
            continue
        b = sc.loc[sc[score_col].idxmax()]
        lat = g["latency_total_ms_median"].dropna() if "latency_total_ms_median" in g else pd.Series(dtype=float)
        has_best = "latency_total_ms_median" in g and not pd.isna(b.get("latency_total_ms_median"))
        rows.append({
            "model": model, "params_M": b["n_params"] / 1e6, "weights_MB": b["weights_mb"], "dim": int(b["embedding_dim"]),
            "peak_gpu_MB": _at_best_or_median(g, b, "peak_gpu_mem_mb"), "encode_batch": b.get("encode_batch_size"),
            "chunks_per_s": _at_best_or_median(g, b, "corpus_chunks_per_s"),
            "gpu_basis": "best" if ("peak_gpu_mem_mb" in g and not pd.isna(b.get("peak_gpu_mem_mb"))) else
                         ("median-of-measured" if "peak_gpu_mem_mb" in g and g["peak_gpu_mem_mb"].notna().any() else "not measured"),
            "latency_ms_median": b["latency_total_ms_median"] if has_best else (lat.median() if len(lat) else np.nan),
            "latency_ms_p95": b["latency_total_ms_p95"] if has_best else (g["latency_total_ms_p95"].median() if len(lat) else np.nan),
            "latency_basis": "best" if has_best else ("median-of-measured" if len(lat) else "not measured"),
            "best_geom10": b[score_col], "best_config": f"{b['chunker']} {int(b['size'])}/{int(b['overlap'])}",
            "index_MB_fp32": b["index_mb_fp32"], "index_MB_fp16": b["index_mb_fp16"], "n_chunks": int(b["n_chunks"]),
        })
    return pd.DataFrame(rows).sort_values("best_geom10", ascending=False).reset_index(drop=True)


def index_tables(d: pd.DataFrame):
    long = d.groupby(["chunker", "size", "overlap"]).agg(n_chunks=("n_chunks", "mean"), index_mb_fp32=("index_mb_fp32", "mean"),
                                                         index_mb_fp16=("index_mb_fp16", "mean"), n_models=("model", "nunique")).reset_index()
    mb = d.groupby(["size", "overlap"])["index_mb_fp32"].mean().unstack("overlap")
    ch = d.groupby(["size", "overlap"])["n_chunks"].mean().unstack("overlap")
    return long, mb, ch


# ---------------------------------------------------------------------------------------------- plots
def _style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def scatter_pareto(d: pd.DataFrame, x: str, score_col: str, xlabel: str, title: str, out: Path, logx: bool = False) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = d.dropna(subset=[x, score_col])
    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=160, facecolor="#fcfcfb")
    _style(ax)
    for ch in CHUNKERS:
        g = d[d["chunker"] == ch]
        if len(g):
            ax.scatter(g[x], g[score_col], s=34, c=COLORS[ch], marker=MARKERS[ch], alpha=0.8, edgecolors="#fcfcfb", linewidths=0.8, label=ch.replace("_", " "))
    front = pareto(d, x, score_col)
    ax.step(front[x], front[score_col], where="post", color=INK, lw=1.6, label="Pareto front", zorder=3)
    ax.scatter(front[x], front[score_col], s=70, facecolors="none", edgecolors=INK, linewidths=1.4, zorder=4)
    # direct labels on the front only (model, chunk size/overlap), alternating above/below to limit collisions
    for k, (_, r) in enumerate(front.iterrows()):
        ax.annotate(f"{str(r['model']).split('/')[-1]} {int(r['size'])}/{int(r['overlap'])}", (r[x], r[score_col]), fontsize=7, color=INK,
                    xytext=(5, 6 if k % 2 == 0 else -11), textcoords="offset points")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel("Geom@10 (span labels)", color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    leg = ax.legend(fontsize=8, frameon=False, labelcolor=MUTED, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), facecolor=fig.get_facecolor())
    plt.close(fig)
    return front


# ---------------------------------------------------------------------------------------------- main
def phase1_table(p1: Path) -> pd.DataFrame:
    e = pd.read_csv(p1)
    agg = e.groupby("model").agg(params_M=("n_params", lambda s: s.iloc[0] / 1e6), weights_MB=("weights_mb", "first"), dim=("embedding_dim", "first"),
                                 peak_gpu_MB=("peak_gpu_mem_mb", "max") if "peak_gpu_mem_mb" in e else ("weights_mb", "first"),
                                 chunks_per_s=("corpus_chunks_per_s", "mean") if "corpus_chunks_per_s" in e else ("weights_mb", "first"),
                                 index_MB_fp32_max=("index_mb_fp32", "max")).reset_index()
    if "latency_total_ms_median" in e:
        agg = agg.merge(e.groupby("model")["latency_total_ms_median"].mean().rename("latency_ms_median").reset_index(), on="model", how="left")
    agg = agg.sort_values("params_M").reset_index(drop=True)
    # The Phase 1 efficiency row of a task is the LAST evaluation pass MTEB makes (for ChemHotpotQARetrieval: train, dev, test in this order, so test).
    # The split that the quality columns of the paper use is params.phase1.eval_split; both are written so that the table says which split it belongs to.
    es, nq = phase1_split.eval_splits(), phase1_split.n_queries_table()
    agg["split"] = "; ".join(f"{t}={sp}" for t, sp in sorted(es.items()))
    agg["n_queries"] = "; ".join(f"{t}={phase1_split.n_queries_of(t, sp, nq)}" for t, sp in sorted(es.items()))
    return agg


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--efficiency", required=True, help="efficiency_phase2.csv")
    ap.add_argument("--scores", required=True, help="score table with model, chunker, size, overlap and the Geom@10 column")
    ap.add_argument("--score-col", default="span_geom_at_10")
    ap.add_argument("--phase1", help="efficiency_phase1.csv (optional)")
    ap.add_argument("--out", default="results/efficiency")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load(Path(a.efficiency), Path(a.scores), a.score_col)

    t = model_table(d, a.score_col)
    t.to_csv(out / "table_efficiency_models.csv", index=False, float_format="%.4f")
    tex = t.copy()
    tex["model"] = tex["model"].str.split("/").str[-1]
    to_latex(tex[["model", "params_M", "weights_MB", "dim", "peak_gpu_MB", "latency_ms_median", "latency_ms_p95", "best_geom10", "best_config", "index_MB_fp32"]],
             out / "table_efficiency_models.tex", "Efficiency per model at its best Geom@10 configuration (latency: batch size 1, exact cosine top-10; see latency\\_basis in the CSV).",
             "tab:efficiency-models", {"params_M": "{:.0f}", "weights_MB": "{:.0f}", "dim": "{:d}", "peak_gpu_MB": "{:.0f}", "latency_ms_median": "{:.1f}",
                                       "latency_ms_p95": "{:.1f}", "best_geom10": "{:.3f}", "index_MB_fp32": "{:.1f}"})

    long, mb, ch = index_tables(d)
    long.to_csv(out / "table_index_size_long.csv", index=False, float_format="%.4f")
    mb.to_csv(out / "table_index_size.csv", float_format="%.3f")
    ch.to_csv(out / "table_index_chunks.csv", float_format="%.1f")
    mbt = mb.copy()
    mbt.columns = [f"overlap {c}" for c in mbt.columns]
    to_latex(mbt, out / "table_index_size.tex", "Index size in MB (fp32), mean over chunkers and models, by chunk size and overlap.", "tab:index-size",
             {c: "{:.1f}" for c in mbt.columns}, index=True)

    if "latency_total_ms_median" in d and d["latency_total_ms_median"].notna().any():
        f = scatter_pareto(d, "latency_total_ms_median", a.score_col, "Per-query latency, median (ms; query encoding + exact cosine top-10, batch size 1)",
                           "Geom@10 vs query latency", out / "fig_geom_vs_latency")
        f[KEY + ["latency_total_ms_median", a.score_col]].to_csv(out / "pareto_latency.csv", index=False)
    else:
        print("[WARN] no latency measured: fig_geom_vs_latency not written")
    f = scatter_pareto(d, "index_mb_fp32", a.score_col, "Index size (MB, fp32, log scale)", "Geom@10 vs index size", out / "fig_geom_vs_index_size", logx=True)
    f[KEY + ["index_mb_fp32", a.score_col]].to_csv(out / "pareto_index_size.csv", index=False)

    if a.phase1:
        p = phase1_table(Path(a.phase1))
        p.to_csv(out / "table_efficiency_phase1.csv", index=False, float_format="%.4f")
        p2 = p.copy()
        p2["model"] = p2["model"].str.split("/").str[-1]
        to_latex(p2, out / "table_efficiency_phase1.tex", "Efficiency of the Phase 1 models.", "tab:efficiency-phase1",
                 {c: "{:.1f}" for c in p2.columns if c not in ("model", "dim", "split", "n_queries")} | {"dim": "{:d}"})
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
