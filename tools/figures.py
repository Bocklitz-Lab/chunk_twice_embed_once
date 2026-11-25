"""Publication figures drawn from the run table of ``make report`` (``runs_overview.csv``): the quality-time trade-off in small multiples (B1) and the effect
of chunk size (B2) and of overlap (B3). They replace the old figure code of ``pipeline_lib/generate_figures.py`` (kept, legacy).

  python -m tools.figures --runs results/v2-revision/runs_overview.csv --out results/v2-revision/figures
  python -m tools.figures --runs ... --out ... --roots artifacts --qa artifacts/chemquests_hf_db/qa.jsonl    # adds the paper-level bands of B2 / B3

Files (PNG at 300 dpi and PDF; each with a sidecar ``<name>.json``: SHA-256 of the input, number of runs, threshold and how it was computed, run kinds, panels,
git commit, so that the caption can be written from facts):

  fig_tradeoff_by_model            B1. One panel per embedding model (shared axes). All dense runs in light grey, the model's runs on top, coloured by chunking
                                   strategy. x = evaluation time (log), y = Geom@10, a horizontal line at the top-1 % threshold (99th percentile).
  fig_tradeoff_by_strategy         B1, compact variant: one panel per chunking strategy, runs coloured by model FAMILY (``params.figures.model_families``).
  fig_chunk_size_by_strategy       B2. One panel per strategy: x = chunk size, y = Geom@10 at overlap 0; line = mean over the dense models with a 95 % paper-level
  fig_chunk_size_lines             bootstrap band of that mean (the same resampled papers for every run; every run's Geom@10 is recomputed from its per-query nDCG@10 and
                                   Recall@10 in each resample, then averaged over the models); the second file has all strategies in one panel.
  fig_overlap_by_size              B3. x = overlap, one panel per chunk size that has more than one overlap, one line per strategy. Sizes with a single overlap
                                   (512 in the current grid) get no panel; the sidecar lists them.

Rules. Dense runs only; ``--include-baselines`` adds BM25, hybrid and rerank runs as a separate, marked layer (hybrid and rerank at their cumulative
``evaluation_time_s``); the top-1 % threshold is computed over the dense runs unless ``--threshold-over all``. Colours: Okabe-Ito (colour-blind safe),
at most 6 per panel, never a marker shape alone. Fonts are 7 pt or more at the printed size (``FULL_WIDTH_MM`` = 174, ``COLUMN_MM`` = 84).
Strategies, sizes, overlaps and models are read from the data. Without per-query files (``--roots``) B2 / B3 show the means without bands and the sidecar says so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pipeline_lib import config_params  # noqa: E402
from pipeline_lib.grid import ALL_CHUNKERS as CHUNKER_ORDER  # noqa: E402     # display order only; the strategies drawn are the ones in the data

FULL_WIDTH_MM, COLUMN_MM = 174.0, 84.0
MM = 1 / 25.4
FONT = 7.5
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]      # blue, orange, green, pink, vermillion, sky blue
GREY, INK, MUTED = "#cfcfcf", "#111111", "#555555"
BASELINE_STYLE = {"bm25": ("#000000", "D"), "hybrid": ("#6a3d9a", "^"), "rerank": ("#b2182b", "s")}      # colour AND marker; open markers = baseline layer
KINDS = ("dense", "bm25", "hybrid", "rerank")


# ------------------------------------------------------------------------------------------------------------------
# input
# ------------------------------------------------------------------------------------------------------------------
def kind_of(row: Dict[str, Any]) -> str:
    k = row.get("kind")
    if k:
        return k
    m = str(row["model"])
    return "bm25" if m.startswith("bm25") else "hybrid" if m.startswith("hybrid_") else "rerank" if m.startswith("rerank_") else "dense"


def read_runs(path: Any, label: str = "span") -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Runs of a ``runs_overview.csv``: model, kind, chunker, size, overlap, geom, time. Returns (rows, which columns were used).
    Time is ``evaluation_time_s`` (cumulative for hybrid / rerank) or, in older tables, ``mteb_evaluation_time_s``."""
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise ValueError(f"{path}: no rows")
    cols = set(raw[0])
    gcol = next((c for c in (f"{label}_geom_at_10", "geom_at_10") if c in cols), None)
    tcol = next((c for c in ("evaluation_time_s", "mteb_evaluation_time_s") if c in cols), None)
    if gcol is None or tcol is None:
        raise ValueError(f"{path}: needs a Geom@10 column ({label}_geom_at_10) and evaluation_time_s / mteb_evaluation_time_s; has {sorted(cols)[:12]}...")
    rows = []
    for r in raw:
        def num(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return float("nan")
        rows.append({"model": r["model"], "kind": kind_of(r), "chunker": r["chunker"], "size": int(r["size"]), "overlap": int(r["overlap"]),
                     "geom": num(r[gcol]), "time": num(r[tcol])})
    return rows, {"geom_column": gcol, "time_column": tcol}


def strategies_in(rows: Sequence[Dict[str, Any]]) -> List[str]:
    found = {r["chunker"] for r in rows}
    return sorted(found, key=lambda c: (CHUNKER_ORDER.index(c) if c in CHUNKER_ORDER else len(CHUNKER_ORDER), c))


def families(rows: Sequence[Dict[str, Any]], main_config: Optional[str] = None) -> Dict[str, str]:
    """{model: family} from ``params.figures.model_families``; a dense model that is in no list is "Other"."""
    table = (config_params.section("figures", main_config).get("model_families") or {})
    of = {m: fam for fam, ms in table.items() for m in ms}
    return {m: of.get(m, "Other") for m in sorted({r["model"] for r in rows if r["kind"] == "dense"})}


def _slug(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", s)


def source_model(row: Dict[str, Any], dense_models: Sequence[str]) -> Optional[str]:
    """The dense model a hybrid / rerank run was built from: the one whose file-safe name is inside the run's name (names of derived runs are built that way)."""
    hits = [m for m in dense_models if _slug(m) in _slug(row["model"])]
    return max(hits, key=len) if hits else None


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def sha256(path: Any) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------------------
# bands: mean over models of Geom@10, paper-level bootstrap
# ------------------------------------------------------------------------------------------------------------------
class Bands:
    """Paper-level bootstrap of the mean over a set of runs of Geom@10 (the same resampled papers for every run, ``tools/analysis_data.PaperBootstrap``)."""

    def __init__(self, roots: Sequence[Any], qa: Any, *, resamples: int, seed: int, kinds: Sequence[str]):
        from tools import analysis_data as AD
        self.AD = AD
        self.runs = AD.load_runs(roots, qa, kinds=list(kinds))
        self.pb = AD.PaperBootstrap(self.runs.groups, resamples, seed)
        self.resamples, self.seed = resamples, seed
        self._geom: Optional[np.ndarray] = None
        self.index = {(m["model"], m["chunker"], m["size"], m["overlap"]): i for i, m in enumerate(self.runs.meta)}

    def geom_resamples(self) -> np.ndarray:
        """(B, R) Geom@10 of every run in every resample, from the per-query nDCG@10 and Recall@10."""
        if self._geom is None:
            self._geom = self.pb.geom_resamples(self.runs.metrics["ndcg_at_10"], self.runs.metrics["recall_at_10"])
        return self._geom

    def band(self, keys: Sequence[Tuple[str, str, int, int]]) -> Optional[Dict[str, Any]]:
        cols = [self.index[k] for k in keys if k in self.index]
        if not cols:
            return None
        g = self.geom_resamples()[:, cols].mean(axis=1)
        point = float(self.runs.geom(cols).mean())
        lo, hi = np.quantile(g, [0.025, 0.975])
        return {"value": point, "ci_low": float(lo), "ci_high": float(hi), "n_runs": len(cols)}


def mean_point(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    v = [r["geom"] for r in rows if not math.isnan(r["geom"])]
    return {"value": float(np.mean(v)), "ci_low": None, "ci_high": None, "n_runs": len(v)}


def effect_series(rows: Sequence[Dict[str, Any]], bands: Optional[Bands], group_key, x_key, kinds: Sequence[str]) -> Dict[Any, List[Dict[str, Any]]]:
    """{group: [{x, value, ci_low, ci_high, n_runs, models}]} : mean over the runs of ``kinds`` that share (group, x)."""
    cell: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["kind"] in kinds:
            cell[(group_key(r), x_key(r))].append(r)
    out: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for (g, x), rs in cell.items():
        keys = [(r["model"], r["chunker"], r["size"], r["overlap"]) for r in rs]
        b = bands.band(keys) if bands is not None else None
        out[g].append({"x": x, **(b or mean_point(rs)), "models": sorted({r["model"] for r in rs})})
    return {g: sorted(v, key=lambda p: p["x"]) for g, v in sorted(out.items(), key=lambda kv: str(kv[0]))}


# ------------------------------------------------------------------------------------------------------------------
# drawing helpers
# ------------------------------------------------------------------------------------------------------------------
def style() -> None:
    plt.rcParams.update({"font.size": FONT, "axes.labelsize": FONT, "axes.titlesize": FONT, "xtick.labelsize": FONT, "ytick.labelsize": FONT,
                         "legend.fontsize": FONT, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "figure.dpi": 100})


def save(fig, out_dir: Path, name: str, sidecar: Dict[str, Any]) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        files.append(p.name)
    plt.close(fig)
    (out_dir / f"{name}.json").write_text(json.dumps({**sidecar, "figure": name, "files": files}, indent=2, default=str), encoding="utf-8")
    return files


def palette(names: Sequence[str]) -> Dict[str, str]:
    if len(names) > len(OKABE_ITO):
        raise ValueError(f"{len(names)} categories need more than the {len(OKABE_ITO)} colours of a panel ({list(names)}); group them (params.figures.model_families)")
    return dict(zip(names, OKABE_ITO))


def legend_below(fig, handles, labels=None, ncol: int = 5) -> None:
    """Legend under the panels with its height RESERVED in inches (a fixed fraction of a short figure let the legend cover the x labels)."""
    ncol = max(1, min(ncol, len(handles)))
    nrows = math.ceil(len(handles) / ncol)
    reserve = nrows * FONT * 2.0 / 72 + 0.14                       # inches: rows x line height + margin
    if labels is None:
        fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.0))
    else:
        fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, reserve / fig.get_figheight(), 1, 1))


def panel_columns(n: int, wide: int) -> int:
    """Columns of a panel grid: ``wide`` for up to 12 panels, one more for 13 or more (13-18 models stay readable at 174 mm)."""
    return wide if n <= 12 else wide + 1


def grid_shape(n: int, max_cols: int) -> Tuple[int, int]:
    cols = min(max_cols, n)
    return math.ceil(n / cols), cols


def threshold(rows: Sequence[Dict[str, Any]], over: str) -> Tuple[float, Dict[str, Any]]:
    """99th percentile of Geom@10 over the dense runs (``over = dense``) or over every run (``all``)."""
    sel = [r for r in rows if (over == "all" or r["kind"] == "dense") and not math.isnan(r["geom"])]
    if not sel:
        raise ValueError("no run to compute the threshold from")
    v = float(np.percentile([r["geom"] for r in sel], 99))
    return v, {"threshold_value": v, "threshold_rule": "99th percentile of Geom@10 (the top 1 %), numpy.percentile with linear interpolation",
               "threshold_over": over, "threshold_n_runs": len(sel), "threshold_kinds": sorted({r["kind"] for r in sel})}


def _scatter_base(ax, rows):
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))      # 1-3-10-30-100: labels do not touch in a 5-column grid
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.minorticks_off()
    ax.grid(True, color="#eeeeee", linewidth=0.5, zorder=0)


# ------------------------------------------------------------------------------------------------------------------
# B1
# ------------------------------------------------------------------------------------------------------------------
def _usable(rows):
    return [r for r in rows if not (math.isnan(r["geom"]) or math.isnan(r["time"]) or r["time"] <= 0)]


def fig_tradeoff_by_model(rows, out_dir: Path, meta: Dict[str, Any], *, include_baselines: bool, over: str) -> Dict[str, Any]:
    dense = _usable([r for r in rows if r["kind"] == "dense"])
    models = sorted({r["model"] for r in dense})
    strat = strategies_in(dense)
    col = palette(strat)
    thr, thr_info = threshold(rows, over)
    base = _usable([r for r in rows if r["kind"] != "dense"]) if include_baselines else []
    nrow, ncol = grid_shape(len(models), panel_columns(len(models), 4))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL_WIDTH_MM * MM, (14 + 42 * nrow) * MM), sharex=True, sharey=True, squeeze=False)
    dense_models = models
    for ax, m in zip(axes.ravel(), models):
        _scatter_base(ax, rows)
        ax.scatter([r["time"] for r in dense], [r["geom"] for r in dense], s=3, c=GREY, linewidths=0, zorder=1, rasterized=True)
        for s in strat:
            sel = [r for r in dense if r["model"] == m and r["chunker"] == s]
            ax.scatter([r["time"] for r in sel], [r["geom"] for r in sel], s=7, c=col[s], linewidths=0, zorder=3, rasterized=True)
        if base:
            for kind in ("bm25", "hybrid", "rerank"):
                sel = [r for r in base if r["kind"] == kind and (kind == "bm25" or source_model(r, dense_models) == m)]
                c, mk = BASELINE_STYLE[kind]
                ax.scatter([r["time"] for r in sel], [r["geom"] for r in sel], s=14, marker=mk, facecolors="none", edgecolors=c, linewidths=0.8, zorder=4)
        ax.axhline(thr, color=INK, lw=0.7, ls="--", zorder=2)
        ax.set_title(m.split("/")[-1], fontsize=FONT, loc="left", pad=2)
    for ax in axes.ravel()[len(models):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("evaluation time (s, log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Geom@10")
    handles = [plt.Line2D([], [], marker="o", ls="", color=col[s], markersize=4, label=s.replace("_", " ")) for s in strat]
    handles += [plt.Line2D([], [], color=GREY, marker="o", ls="", markersize=3, label="all dense runs"), plt.Line2D([], [], color=INK, ls="--", lw=0.7, label=f"top 1 % ({thr:.3f})")]
    if base:
        handles += [plt.Line2D([], [], marker=BASELINE_STYLE[k][1], ls="", markerfacecolor="none", markeredgecolor=BASELINE_STYLE[k][0], markersize=4, label=f"{k} (baseline layer)")
                    for k in ("bm25", "hybrid", "rerank")]
    legend_below(fig, handles, ncol=5)
    side = {**meta, **thr_info, "n_runs_plotted": len(dense) + len(base), "n_dense_runs": len(dense), "panels": models, "panel_by": "model", "colour_by": "chunking strategy",
            "strategies": strat, "kinds_included": ["dense"] + (["bm25", "hybrid", "rerank"] if base else []), "x": "evaluation_time_s (log)", "y": "Geom@10"}
    save(fig, out_dir, "fig_tradeoff_by_model", side)
    return side


def fig_tradeoff_by_strategy(rows, out_dir: Path, meta: Dict[str, Any], *, include_baselines: bool, over: str, fam: Dict[str, str]) -> Dict[str, Any]:
    dense = _usable([r for r in rows if r["kind"] == "dense"])
    strat = strategies_in(dense)
    fams = sorted(set(fam.get(r["model"], "Other") for r in dense))
    col = palette(fams)
    thr, thr_info = threshold(rows, over)
    base = _usable([r for r in rows if r["kind"] != "dense"]) if include_baselines else []
    nrow, ncol = grid_shape(len(strat), 3)
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL_WIDTH_MM * MM, (28 + 50 * nrow) * MM), sharex=True, sharey=True, squeeze=False)
    for ax, s in zip(axes.ravel(), strat):
        _scatter_base(ax, rows)
        ax.scatter([r["time"] for r in dense if r["chunker"] != s], [r["geom"] for r in dense if r["chunker"] != s], s=3, c=GREY, linewidths=0, zorder=1, rasterized=True)
        for f in fams:
            sel = [r for r in dense if r["chunker"] == s and fam.get(r["model"], "Other") == f]
            ax.scatter([r["time"] for r in sel], [r["geom"] for r in sel], s=7, c=col[f], linewidths=0, zorder=3, rasterized=True)
        for kind in ("bm25", "hybrid", "rerank"):
            sel = [r for r in base if r["kind"] == kind and r["chunker"] == s]
            c, mk = BASELINE_STYLE[kind]
            ax.scatter([r["time"] for r in sel], [r["geom"] for r in sel], s=14, marker=mk, facecolors="none", edgecolors=c, linewidths=0.8, zorder=4)
        ax.axhline(thr, color=INK, lw=0.7, ls="--", zorder=2)
        ax.set_title(s.replace("_", " "), fontsize=FONT, loc="left", pad=2)
    for ax in axes.ravel()[len(strat):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("evaluation time (s, log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Geom@10")
    handles = [plt.Line2D([], [], marker="o", ls="", color=col[f], markersize=4, label=f) for f in fams]
    handles += [plt.Line2D([], [], color=GREY, marker="o", ls="", markersize=3, label="other strategies"), plt.Line2D([], [], color=INK, ls="--", lw=0.7, label=f"top 1 % ({thr:.3f})")]
    if base:
        handles += [plt.Line2D([], [], marker=BASELINE_STYLE[k][1], ls="", markerfacecolor="none", markeredgecolor=BASELINE_STYLE[k][0], markersize=4, label=f"{k} (baseline layer)")
                    for k in ("bm25", "hybrid", "rerank")]
    legend_below(fig, handles, ncol=5)
    side = {**meta, **thr_info, "n_runs_plotted": len(dense) + len(base), "n_dense_runs": len(dense), "panels": strat, "panel_by": "chunking strategy", "colour_by": "model family",
            "families": fams, "family_of_model": fam, "kinds_included": ["dense"] + (["bm25", "hybrid", "rerank"] if base else []), "x": "evaluation_time_s (log)", "y": "Geom@10"}
    save(fig, out_dir, "fig_tradeoff_by_strategy", side)
    return side


# ------------------------------------------------------------------------------------------------------------------
# B2, B3
# ------------------------------------------------------------------------------------------------------------------
def _draw_line(ax, pts, color, label=None, ls="-", lw=1.4, band=True, marker="o", ms=3, xmap=None):
    """``xmap``: {x value: plotted position}, for a categorical axis with equally spaced ticks (chunk sizes)."""
    x = [xmap[p["x"]] if xmap else p["x"] for p in pts]
    y = [p["value"] for p in pts]
    if band and pts and pts[0]["ci_low"] is not None:
        ax.fill_between(x, [p["ci_low"] for p in pts], [p["ci_high"] for p in pts], color=color, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(x, y, color=color, ls=ls, lw=lw, marker=marker, ms=ms, label=label, zorder=3)


def best_model(rows, kinds=("dense",)) -> Optional[str]:
    by = defaultdict(list)
    for r in rows:
        if r["kind"] in kinds and not math.isnan(r["geom"]):
            by[r["model"]].append(r["geom"])
    return max(by, key=lambda m: np.mean(by[m])) if by else None


def _baseline_series(rows, bands, group_key, x_key, include):
    return effect_series(rows, bands, group_key, x_key, ("bm25",)) if include else {}


def fig_chunk_size(rows, out_dir: Path, meta: Dict[str, Any], bands: Optional[Bands], *, include_baselines: bool, show_best: bool) -> Dict[str, Any]:
    strat = strategies_in([r for r in rows if r["kind"] == "dense"])
    at0 = [r for r in rows if r["overlap"] == 0]
    dense_series = effect_series(at0, bands, lambda r: r["chunker"], lambda r: r["size"], ("dense",))
    bm = _baseline_series(at0, bands, lambda r: r["chunker"], lambda r: r["size"], include_baselines)
    best = best_model(at0) if show_best else None
    best_series = effect_series([r for r in at0 if r["model"] == best], bands, lambda r: r["chunker"], lambda r: r["size"], ("dense",)) if best else {}
    col = palette(strat)
    sizes = sorted({r["size"] for r in at0})
    xmap = {s: i for i, s in enumerate(sizes)}                          # equally spaced categorical axis: 384 / 448 / 512 do not touch
    nrow, ncol = grid_shape(len(strat), 3)
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL_WIDTH_MM * MM, (22 + 48 * nrow) * MM), sharex=True, sharey=True, squeeze=False)
    for ax, s in zip(axes.ravel(), strat):
        ax.grid(True, color="#eeeeee", linewidth=0.5, zorder=0)
        if s in best_series:
            _draw_line(ax, best_series[s], MUTED, label=f"best model: {best}", lw=0.7, band=False, ms=2, xmap=xmap)
        if s in bm:
            _draw_line(ax, bm[s], INK, label="BM25", ls="--", lw=1.0, ms=2, xmap=xmap)
        _draw_line(ax, dense_series.get(s, []), col[s], label="mean of dense models", xmap=xmap)
        ax.set_title(s.replace("_", " "), fontsize=FONT, loc="left", pad=2)
    for ax in axes.ravel()[len(strat):]:
        ax.set_visible(False)
    for ax in axes.ravel():
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([str(s) for s in sizes])
    for ax in axes[-1]:
        ax.set_xlabel("chunk size (tokens)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Geom@10 (overlap 0)")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    legend_below(fig, h, l, ncol=3)
    n_models = sorted({len(p["models"]) for pts in dense_series.values() for p in pts})
    side = {**meta, "panels": strat, "panel_by": "chunking strategy", "x": "chunk size (overlap 0), equally spaced categorical axis", "y": "Geom@10, mean over dense models",
            "band": "95 % paper-level bootstrap of the mean over models" if bands else "none: no per-query files were given (--roots)",
            "bootstrap_resamples": bands.resamples if bands else None, "bootstrap_seed": bands.seed if bands else None, "models_per_point": n_models,
            "best_model": best, "baseline_layer": include_baselines, "kinds_included": ["dense"] + (["bm25"] if include_baselines else []), "sizes": sizes}
    save(fig, out_dir, "fig_chunk_size_by_strategy", side)
    # all strategies in one panel (the second variant)
    fig2, ax = plt.subplots(figsize=(COLUMN_MM * MM * 1.6, 62 * MM))
    ax.grid(True, color="#eeeeee", linewidth=0.5, zorder=0)
    for s in strat:
        _draw_line(ax, dense_series.get(s, []), col[s], label=s.replace("_", " "), xmap=xmap)
    if include_baselines:
        allbm = effect_series(at0, bands, lambda r: "bm25", lambda r: r["size"], ("bm25",)).get("bm25")
        if allbm:
            _draw_line(ax, allbm, INK, label="BM25 (mean of strategies' runs)", ls="--", lw=1.0, ms=2, xmap=xmap)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("chunk size (tokens)")
    ax.set_ylabel("Geom@10 (overlap 0)")
    ax.legend(frameon=False, loc="best")
    fig2.tight_layout()
    save(fig2, out_dir, "fig_chunk_size_lines", {**side, "panels": ["all strategies"], "panel_by": "none (one panel)", "colour_by": "chunking strategy"})
    return side


def fig_overlap(rows, out_dir: Path, meta: Dict[str, Any], bands: Optional[Bands], *, include_baselines: bool) -> Dict[str, Any]:
    dense = [r for r in rows if r["kind"] == "dense"]
    strat = strategies_in(dense)
    overlaps_of_size = defaultdict(set)
    for r in dense:
        overlaps_of_size[r["size"]].add(r["overlap"])
    sizes = sorted(s for s, o in overlaps_of_size.items() if len(o) > 1)
    excluded = {str(s): sorted(o) for s, o in overlaps_of_size.items() if len(o) <= 1}
    side = {**meta, "panels": [str(s) for s in sizes], "panel_by": "chunk size (only sizes with more than one overlap)", "sizes_without_panel": excluded,
            "x": "overlap (tokens)", "y": "Geom@10, mean over dense models", "baseline_layer": include_baselines, "kinds_included": ["dense"] + (["bm25"] if include_baselines else []),
            "band": "95 % paper-level bootstrap of the mean over models" if bands else "none: no per-query files were given (--roots)",
            "bootstrap_resamples": bands.resamples if bands else None, "bootstrap_seed": bands.seed if bands else None}
    if not sizes:
        side["note"] = "no chunk size has more than one overlap: no figure"
        return side
    col = palette(strat)
    nrow, ncol = grid_shape(len(sizes), 3)
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL_WIDTH_MM * MM, (22 + 48 * nrow) * MM), sharey=True, squeeze=False)
    for ax, size in zip(axes.ravel(), sizes):
        ax.grid(True, color="#eeeeee", linewidth=0.5, zorder=0)
        sub = [r for r in rows if r["size"] == size]
        ser = effect_series(sub, bands, lambda r: r["chunker"], lambda r: r["overlap"], ("dense",))
        bm = _baseline_series(sub, bands, lambda r: "bm25", lambda r: r["overlap"], include_baselines).get("bm25")
        if bm and len(bm) > 1:
            _draw_line(ax, bm, INK, label="BM25", ls="--", lw=1.0, ms=2)
        for s in strat:
            if len(ser.get(s, [])) > 1:                                  # a line needs two overlaps
                _draw_line(ax, ser[s], col[s], label=s.replace("_", " "))
        ax.set_title(f"chunk size {size}", fontsize=FONT, loc="left", pad=2)
        ax.set_xticks(sorted(overlaps_of_size[size]))
        ax.set_xlabel("overlap (tokens)")
    for ax in axes.ravel()[len(sizes):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Geom@10")
    h, l = {}, {}
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            h.setdefault(ll, hh)
    legend_below(fig, list(h.values()), list(h.keys()), ncol=6)
    save(fig, out_dir, "fig_overlap_by_size", side)
    return side


# ------------------------------------------------------------------------------------------------------------------
# driver
# ------------------------------------------------------------------------------------------------------------------
def draw_all(runs_csv: Any, out_dir: Any, *, roots: Optional[Sequence[Any]] = None, qa: Optional[Any] = None, include_baselines: bool = False, threshold_over: str = "dense",
             main_config: Optional[str] = None, show_best: bool = False, label: str = "span", resamples: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    style()
    out_dir = Path(out_dir)
    rows, used = read_runs(runs_csv, label)
    usable = [r for r in rows if not math.isnan(r["geom"])]
    meta = {"input": str(runs_csv), "input_sha256": sha256(runs_csv), "n_runs_in_table": len(rows), "run_kinds_in_table": sorted({r["kind"] for r in rows}), **used,
            "git_commit": git_commit(), "include_baselines": include_baselines}
    fam = families(rows, main_config)
    bands = None
    if roots and qa:
        from tools import analysis_data as AD
        d = AD.defaults(main_config)
        kinds = ["dense"] + (["bm25"] if include_baselines else [])
        try:
            bands = Bands(roots, qa, resamples=resamples or d["resamples"], seed=seed if seed is not None else d["seed"], kinds=kinds)
        except FileNotFoundError:
            bands = None
    out = {"tradeoff_by_model": fig_tradeoff_by_model(usable, out_dir, meta, include_baselines=include_baselines, over=threshold_over),
           "tradeoff_by_strategy": fig_tradeoff_by_strategy(usable, out_dir, meta, include_baselines=include_baselines, over=threshold_over, fam=fam),
           "chunk_size": fig_chunk_size(usable, out_dir, meta, bands, include_baselines=include_baselines, show_best=show_best),
           "overlap": fig_overlap(usable, out_dir, meta, bands, include_baselines=include_baselines)}
    return out


def adapt_published(canonical_csv: Any, tests_root: Any, out_csv: Any) -> Path:
    """The published Phase 2 table (``canonical_phase2_corrected_1820.csv``, one row per dense run) as a run table for the figures: the old layout has one
    ``ChemQuest.json`` per configuration (``result_file``, whose ``artifacts/tests/`` prefix is replaced by ``tests_root``) and no per-query metrics, so
    ``evaluation_time_s`` is MTEB's ``evaluation_time`` and B2 / B3 have no bands. Runs whose result file is not found are listed on stderr and left out."""
    import csv
    rows, missing = [], []
    for r in csv.DictReader(open(canonical_csv, newline="", encoding="utf-8")):
        f = Path(tests_root) / re.sub(r"^.*?artifacts/tests/", "", r["result_file"])
        if not f.is_file():
            missing.append(r["configuration_id"])
            continue
        t = json.loads(f.read_text(encoding="utf-8")).get("evaluation_time")
        rows.append({"model": r["model_hf"], "kind": "dense", "chunker": r["chunker"], "size": r["token_size"], "overlap": r["overlap"],
                     "span_geom_at_10": r["geom_at_10"], "evaluation_time_s": t if t is not None else ""})
    if missing:
        print(f"{len(missing)} published runs without a result file (left out): {missing[:3]}...", file=sys.stderr)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return out_csv


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", help="runs_overview.csv of make report")
    ap.add_argument("--from-published", help="instead of --runs: canonical_phase2_corrected_1820.csv of the published Phase 2 (needs --tests-root)")
    ap.add_argument("--tests-root", help="folder with the published run folders (artifacts/tests of the old layout)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", nargs="+", help="artifacts roots with tests/ (per-query metrics) for the paper-level bands of B2 / B3")
    ap.add_argument("--qa", help="qa.jsonl (with --roots)")
    ap.add_argument("--include-baselines", action="store_true", help="add BM25, hybrid and rerank runs as a separate, marked layer")
    ap.add_argument("--threshold-over", choices=["dense", "all"], default="dense", help="which runs the top-1 %% threshold is computed over (default: dense)")
    ap.add_argument("--best-model", action="store_true", help="B2: also draw the best dense model as a thin line")
    ap.add_argument("--label-set", choices=["span", "lexical"], default="span")
    ap.add_argument("--main-config", default=str(config_params.DEFAULT_MAIN_CONFIG))
    ap.add_argument("--resamples", type=int)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args(argv)
    if args.from_published:
        if not args.tests_root:
            ap.error("--from-published needs --tests-root")
        args.runs = str(adapt_published(args.from_published, args.tests_root, Path(args.out) / "published_run_table.csv"))
    if not args.runs:
        ap.error("--runs (or --from-published) is required")
    res = draw_all(args.runs, args.out, roots=args.roots, qa=args.qa, include_baselines=args.include_baselines, threshold_over=args.threshold_over, main_config=args.main_config,
                   show_best=args.best_model, label=args.label_set, resamples=args.resamples, seed=args.seed)
    print(f"figures -> {args.out}: " + ", ".join(sorted(k for k, v in res.items() if "note" not in v)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
