"""Time estimate of the full rerun for the CURRENT Phase 2 model list (stage 2 file), from the measurements of the subset run and the audit.

Everything is derived from files; nothing is typed in:
  * per-run evaluation time of a dense model on the full 140-configuration grid:
      M  measured on the subset (``time_estimate.csv`` of the subset: "measured per run, scaled to 140 configurations": bge-m3, e5-large-v2, nomic-v1.5, e5-small)
      E  every other model: slope x its Phase 1 evaluation_time (least squares through the origin on the measured models; Phase 1 files of the prompted run)
      M  hkunlp/instructor-xl: the audit measurement (one configuration, fixed-token 256/0, 451.5 s) x the e5-large-v2 time of the grid / e5-large-v2 at that cell
  * cell factor f(c) = e5-large-v2 time at cell c / its mean over the grid (the shape of the time over chunk size, overlap and strategy is taken from the
    measured e5-large-v2 runs; fixed-token 384/32 is not in the subset: fixed 384/0 x (1 + 32 / 384));
  * wall time = evaluation_time x ``wall_ratio`` (default 1.07, measured on the long instructor-xl run: 481 s wall / 451.5 s evaluation_time; the 1.49 of the
    subset comes from short runs, where model load, probes and latency dominate: ``--wall-ratio 1.49`` gives that bound);
  * BM25, hybrid, rerank (dense first stage), prepare, report: per-run means measured on the subset, multiplied by the number of runs;
  * rerank of the BM25 top-N: per-run seconds given by ``--rerank-bm25-seconds`` (measured by the B7 check of the pre-rerun branch);
  * the control strategy fixed_token_nomatter (grid.CONTROL_CELLS): fixed-token time at the same cell x the share of the text it keeps (0.832, measured).
Output: ``time_estimate.csv`` (stage, item, seconds, hours, basis M / E, note), a Markdown summary, totals for one and two GPUs (longest-processing-time split of the GPU jobs;
CPU work is added serially where it cannot overlap: prepare before, report after).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import grid  # noqa: E402

KEPT_SHARE = 0.832                    # share of the characters that hierarchical-section / fixed_token_nomatter keep (audit check 12, prepared folders)
N_GRID = len(grid.CHUNKERS) * len(grid.grid())            # 140
N_CONTROL = sum(len(grid.cells_for(c, grid.grid())) for c in grid.CONTROL_CELLS)      # 3


def cell_factors(runs: pd.DataFrame, ref: str, ref_per_run_grid: float) -> Dict[Tuple[str, int, int], float]:
    d = runs[(runs["kind"] == "dense") & (runs["model"] == ref)]
    t = {(r["chunker"], int(r["size"]), int(r["overlap"])): float(r["evaluation_time_s"]) for _, r in d.iterrows()}
    f = {k: v / ref_per_run_grid for k, v in t.items()}
    k384 = ("fixed_token", 384, 0)
    if k384 in t:
        f[("fixed_token", 384, 32)] = t[k384] * (1 + 32 / 384) / ref_per_run_grid
    return f


def estimate(subset_est: pd.DataFrame, external: pd.DataFrame, runs: pd.DataFrame, models: List[str], transfer: Dict[str, Any], *, wall_ratio: float,
             instructor_seconds: Optional[float], instructor_cell: Tuple[str, int, int], rerank_bm25_seconds: Optional[float], n_gpus: int = 2,
             ref: str = "intfloat/e5-large-v2", prepare_control_seconds: float = 0.0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add(stage, item, seconds, basis, note=""):
        rows.append({"stage": stage, "item": item, "seconds": float(seconds), "hours": float(seconds) / 3600, "basis": basis, "note": note})

    ext = external[external["kind"] == "dense"].set_index("model")
    p1 = ext[[c for c in ext.columns if c.endswith("evaluation_time_s")]].sum(axis=1)
    # measured per-run grid means (rows "x 140 configurations" with basis "measured")
    g = subset_est[subset_est["item"].astype(str).str.contains(" x 140 configurations") & subset_est["basis"].astype(str).str.startswith("measured")]
    meas = {r["item"].split(" x ")[0]: r["seconds"] / N_GRID for _, r in g.iterrows()}
    xs = np.array([p1[m] for m in meas if m in p1.index])
    ys = np.array([meas[m] for m in meas if m in p1.index])
    slope = float((xs * ys).sum() / (xs * xs).sum())
    per_run: Dict[str, Tuple[float, str]] = {m: (v, "M") for m, v in meas.items()}
    f = cell_factors(runs, ref, meas[ref])
    if instructor_seconds:
        fk = f.get(instructor_cell)
        per_run["hkunlp/instructor-xl"] = (instructor_seconds / fk, "M")          # measured at one cell, scaled to the grid mean by the e5-large-v2 cell factor
    wanted = sorted(set(models) | {m["name"] for m in transfer["models"]})
    for m in wanted:
        if m not in per_run:
            if m in p1.index:
                per_run[m] = (slope * float(p1[m]), "E")
    add("parameter (a ratio, not seconds)", f"slope: grid seconds per run / Phase 1 evaluation_time (least squares through the origin on {len(xs)} measured models)", slope, "M")
    gpu_jobs: List[Tuple[str, float]] = []
    n_cfg = N_GRID + N_CONTROL
    total_dense = 0.0
    for m in sorted(models):
        if m not in per_run:
            add("full rerun: dense", f"{m}", float("nan"), "none", "no Phase 1 time")
            continue
        t, b = per_run[m]
        main = t * N_GRID
        ctrl = sum(t * f[("fixed_token", s, o)] * KEPT_SHARE for s, o in grid.CONTROL_CELLS["fixed_token_nomatter"])
        sec = (main + ctrl) * wall_ratio
        add("full rerun: dense", f"{m} x {N_GRID} + {N_CONTROL} control configurations", sec, b, f"{t:.1f} s per run x {n_cfg} (control: x{KEPT_SHARE}) x wall ratio {wall_ratio}")
        gpu_jobs.append((f"dense {m}", sec))
        total_dense += sec
    # baselines
    def row(prefix):
        r = subset_est[subset_est["item"].astype(str).str.startswith(prefix)]
        return float(r["seconds"].iloc[0]) if len(r) else float("nan")
    bm25_run = row("BM25 x 140") / N_GRID
    hyb_item = subset_est[subset_est["item"].astype(str).str.startswith("hybrid x")]["item"].iloc[0]
    hyb_run = row("hybrid x") / int(hyb_item.split(" x ")[1])
    add("full rerun: bm25 (CPU)", f"BM25 x {n_cfg}", bm25_run * n_cfg, "M", f"{bm25_run:.1f} s per run")
    add("full rerun: hybrid (CPU)", f"hybrid x {len(models) * n_cfg}", hyb_run * len(models) * n_cfg, "M", f"{hyb_run:.2f} s per run (fusion only; inputs counted above)")
    rr_run = row("rerank (primary depth) x 140") / N_GRID
    rr_models = 1
    add("full rerun: rerank (GPU)", f"rerank of the dense top-50, e5-large-v2 x {n_cfg}", rr_run * n_cfg * rr_models, "M", f"{rr_run:.0f} s per run")
    gpu_jobs.append(("rerank dense", rr_run * n_cfg))
    add("full rerun: rerank (GPU)", "depth ablation (20 and 100) at fixed-token 256/0", rr_run * 3.5, "E", "depth 20 ~0.5 x, depth 100 ~2 x the depth-50 run")
    gpu_jobs.append(("rerank ablation", rr_run * 3.5))
    rb = rerank_bm25_seconds if rerank_bm25_seconds else rr_run
    add("full rerun: rerank (GPU)", f"rerank of the BM25 top-50 x {n_cfg}", rb * n_cfg, "M" if rerank_bm25_seconds else "E", f"{rb:.0f} s per run" + ("" if rerank_bm25_seconds else " (assumed = dense rerank)"))
    gpu_jobs.append(("rerank bm25", rb * n_cfg))
    # transfer
    cfgs = transfer["configurations"]
    t_total = 0.0
    for m in transfer["models"]:
        if m["name"] not in per_run:
            continue
        t, b = per_run[m["name"]]
        key = lambda c: (c["chunker"], int(c["size"]), int(c["overlap"]))
        sec = sum(t * f.get(key(c), 1.0) for c in cfgs) * wall_ratio
        t_total += sec
        gpu_jobs.append((f"transfer {m['name']}", sec))
    add("full rerun: transfer (GPU)", f"{len(transfer['models'])} models x {len(cfgs)} configurations", t_total, "E", "per-model grid mean x cell factor of the configuration x wall ratio")
    # prepare / report
    prep = row("full rerun: prepare") if False else float(subset_est[subset_est["stage"].astype(str).str.contains("full rerun: prepare")]["seconds"].iloc[0])
    add("full rerun: prepare (CPU)", f"chunk + label {N_GRID} + {N_CONTROL} configurations", prep + prepare_control_seconds, "M", "")
    n_runs = len(models) * n_cfg * 2 + n_cfg * 3 + 3 * n_cfg
    scale = n_runs / 4060.0
    rep = 0.0
    for key in ("make report", "make analysis", "figures"):
        r = subset_est[subset_est["stage"].astype(str).str.contains(f"full rerun: {key}")]
        if len(r):
            rep += float(r["seconds"].iloc[0]) * scale
    add("full rerun: report + analysis + figures (CPU)", f"scaled by the number of runs ({n_runs})", rep, "E", "")
    ph = float(subset_est[subset_est["stage"].astype(str).str.contains("full rerun: phase1")]["seconds"].iloc[0])
    add("already done", "Phase 1, 41 models (prompted run; not part of the rerun)", ph, "M", "only the pooling-override models are re-run (B9)")
    # totals
    cpu = bm25_run * n_cfg + hyb_run * len(models) * n_cfg
    gpu_total = sum(s for _, s in gpu_jobs)
    loads = [0.0] * n_gpus
    for _, s in sorted(gpu_jobs, key=lambda j: -j[1]):
        loads[loads.index(min(loads))] += s
    serial_cpu = prep + prepare_control_seconds + rep
    add("TOTAL", "one GPU: all GPU jobs + CPU work (nothing overlapped)", gpu_total + cpu + serial_cpu, "M+E", f"GPU {gpu_total / 3600:.1f} h + CPU {(cpu + serial_cpu) / 3600:.1f} h")
    add("TOTAL", f"{n_gpus} GPUs: prepare, then the longest GPU load, then hybrid and report (BM25 overlaps the GPU jobs)",
        prep + prepare_control_seconds + max(loads) + hyb_run * len(models) * n_cfg + rep, "M+E",
        "GPU loads " + ", ".join(f"{x / 3600:.1f} h" for x in loads) + f"; wall ratio {wall_ratio}")
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset-dir", default="results/v2-revision/subset")
    ap.add_argument("--shortlist", default="artifacts/subset_v1/embedding_models_screening/passed_per_dataset__validation_prompts.csv")
    ap.add_argument("--main-config", default="configs/main_config.yaml")
    ap.add_argument("--out", default="results/v2-revision/time_estimate.csv")
    ap.add_argument("--wall-ratio", type=float, default=1.07)
    ap.add_argument("--instructor-seconds", type=float, default=451.5, help="audit: evaluation_time of instructor-xl at fixed-token 256/0 (0 = none)")
    ap.add_argument("--rerank-bm25-seconds", type=float, help="seconds per run of reranking the BM25 top-50 (default: same as the dense rerank)")
    ap.add_argument("--prepare-control-seconds", type=float, default=80.0, help="prepare of the 3 control configurations (measured on the subset data)")
    a = ap.parse_args(argv)
    sd = Path(a.subset_dir)
    est = pd.read_csv(sd / "time_estimate.csv")
    ext = pd.read_csv(sd / "table_external.csv")
    runs = pd.read_csv(sd / "runs_overview.csv")
    models = sorted({r["model_name"] for r in csv.DictReader(open(a.shortlist, newline="", encoding="utf-8"))})
    transfer = yaml.safe_load(Path(a.main_config).read_text(encoding="utf-8"))["params"]["transfer"]
    rows = estimate(est, ext, runs, models, transfer, wall_ratio=a.wall_ratio, instructor_seconds=a.instructor_seconds or None, instructor_cell=("fixed_token", 256, 0),
                    rerank_bm25_seconds=a.rerank_bm25_seconds, prepare_control_seconds=a.prepare_control_seconds)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.2f")
    lines = ["| stage | item | hours | basis | note |", "|---|---|---|---|---|"] + [f"| {r.stage} | {r.item} | {r.hours:.2f} | {r.basis} | {r.note} |" for r in df.itertuples()]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for r in df[df["stage"] == "TOTAL"].itertuples():
        print(f"{r.item}: {r.hours:.1f} h ({r.note})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
