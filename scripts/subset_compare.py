#!/usr/bin/env python
"""Numbers for the subset check (Parts B, C, D, E). Writes small tables; assigns NO verdicts (that is Session 2).

  compare_published_vs_new.csv   one row per run of the 4 models x 35 configurations: published (old code), new code with lexical labels, new code with span labels (+ paper-level CI)
  compare_correlations.csv       Spearman / Kendall between published and new Geom@10: overall, within each model, within each strategy
  claims_numbers.md              one block per claim of scripts/subset_claims.yaml: published statement, published / new numbers, intervals, winner shares
  reviewer_outputs.csv           Part D: does each output file exist, is it non-empty, how many rows
  time_estimate.csv              Part E: measured times of the subset and the extrapolation to the full rerun (measured vs extrapolated is a column)

  python scripts/subset_compare.py --root artifacts/subset_v1 --out results/v2-revision/subset
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import analysis_data as AD  # noqa: E402

PUB_DIR = REPO / "temp_files" / "artifacts_old" / "tests"
PUB_P1 = REPO / "temp_files" / "artifacts_old" / "embedding_models_screening"
KEY = ["model", "chunker", "size", "overlap"]


# ------------------------------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------------------------------
def md(df: pd.DataFrame, floatfmt: str = "{:.3f}", maxrows: int = 60) -> str:
    if df is None or len(df) == 0:
        return "_(no rows)_\n"
    d = df.head(maxrows)
    cols = list(d.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and not math.isnan(v) else ("" if isinstance(v, float) and math.isnan(v) else str(v)) for v in r) + " |")
    if len(df) > maxrows:
        out.append(f"| ... {len(df) - maxrows} more rows |" + " |" * (len(cols) - 1))
    return "\n".join(out) + "\n"


def ci_str(v, lo, hi) -> str:
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]" if lo is not None and not (isinstance(lo, float) and math.isnan(lo)) else f"{v:.3f}"


def short(m: str) -> str:
    return m.split("/")[-1]


def read_csv(p: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(p) if Path(p).is_file() and Path(p).stat().st_size > 0 else None


class Boot:
    """Paper-level bootstrap of means of Geom@10 over groups of runs, with the same resampled papers for every run (tools/analysis_data.PaperBootstrap)."""

    def __init__(self, roots: Sequence[Any], qa: Any, label_set: str, resamples: int, seed: int):
        self.runs = AD.load_runs(roots, qa, label_set=label_set)
        self.pb = AD.PaperBootstrap(self.runs.groups, resamples, seed)
        self.g = self.pb.geom_resamples(self.runs.metrics["ndcg_at_10"], self.runs.metrics["recall_at_10"])        # (B, R)
        self.pt = self.runs.geom()
        self.meta = pd.DataFrame(self.runs.meta)[["key", "model", "kind", "chunker", "size", "overlap"]]

    def keys(self, **where) -> List[int]:
        m = self.meta
        for k, v in where.items():
            m = m[m[k].isin(v if isinstance(v, (list, tuple, set)) else [v])]
        return list(m.index)

    def mean(self, idx: Sequence[int]) -> Tuple[float, float, float]:
        if not len(idx):
            return (float("nan"),) * 3
        r = self.g[:, list(idx)].mean(axis=1)
        lo, hi = np.quantile(r, [0.025, 0.975])
        return float(self.pt[list(idx)].mean()), float(lo), float(hi)

    def diff(self, a: Sequence[int], b: Sequence[int]) -> Tuple[float, float, float, float]:
        """mean(a) - mean(b): estimate, 95 % interval, share of resamples with the opposite sign (two-sided bootstrap p)."""
        if not len(a) or not len(b):
            return (float("nan"),) * 4
        r = self.g[:, list(a)].mean(axis=1) - self.g[:, list(b)].mean(axis=1)
        lo, hi = np.quantile(r, [0.025, 0.975])
        B = len(r)
        p = min(1.0, 2 * (1 + min((r <= 0).sum(), (r >= 0).sum())) / (B + 1))
        return float(self.pt[list(a)].mean() - self.pt[list(b)].mean()), float(lo), float(hi), float(p)


class Ctx:
    def __init__(self, a):
        self.a = a
        self.root, self.out = Path(a.root), Path(a.out)
        self.cl = yaml.safe_load(open(a.claims))
        self.models = a.models or self.cl["subset"]["models"]
        self.report = self.root / "report"
        self.analysis = self.out / "analysis"
        self.ov = read_csv(self.report / "runs_overview.csv")
        if self.ov is None:
            raise SystemExit(f"{self.report}/runs_overview.csv missing: run `make report` first")
        if "kind" not in self.ov:
            self.ov["kind"] = self.ov["model"].map(lambda m: "bm25" if str(m).startswith("bm25") else "hybrid" if str(m).startswith("hybrid_") else "rerank" if str(m).startswith("rerank_") else "dense")
        if "evaluation_time_s" not in self.ov and "mteb_evaluation_time_s" in self.ov:
            self.ov["evaluation_time_s"] = self.ov["mteb_evaluation_time_s"]
        self.qa = Path(a.qa) if a.qa else self.root / "chemquests_hf_db" / "qa.jsonl"
        self._boot: Dict[str, Boot] = {}
        self.table = self.build_table()

    def boot(self, label: str = "span") -> Boot:
        if label not in self._boot:
            self._boot[label] = Boot([self.root], self.qa, label, self.a.resamples, self.a.seed)
        return self._boot[label]

    # ---- published / new / lexical for the same runs
    def published(self) -> pd.DataFrame:
        f = PUB_DIR / "canonical_phase2_corrected_1820.csv"
        d = pd.read_csv(f)
        d = d.rename(columns={"model_hf": "model", "token_size": "size", "geom_at_10": "pub_geom", "ndcg_at_10": "pub_ndcg", "recall_at_10": "pub_recall"})
        d["chunker"] = d["chunker"].astype(str)
        times = []
        for r in d["result_file"]:
            p = PUB_DIR / re.sub(r"^.*?artifacts/tests/", "", r)
            try:
                times.append(json.loads(p.read_text()).get("evaluation_time"))
            except Exception:
                times.append(np.nan)
        d["pub_time_s"] = times
        return d[KEY + ["pub_geom", "pub_ndcg", "pub_recall", "pub_time_s"]]

    def build_table(self) -> pd.DataFrame:
        dense = self.ov[self.ov["kind"] == "dense"].copy()
        new = dense.rename(columns={"span_geom_at_10": "span_geom", "span_ndcg_at_10": "span_ndcg", "span_recall_at_10": "span_recall", "lexical_geom_at_10": "lex_geom",
                                    "lexical_ndcg_at_10": "lex_ndcg", "lexical_recall_at_10": "lex_recall", "evaluation_time_s": "time_s"})
        keep = KEY + [c for c in ("span_geom", "span_ndcg", "span_recall", "lex_geom", "lex_ndcg", "lex_recall", "time_s", "n_chunks", "pct_chunks_truncated") if c in new]
        new = new[keep]
        pub = self.published()
        t = new.merge(pub, on=KEY, how="left")
        return t[t["model"].isin(self.models)] if len(t[t["model"].isin(self.models)]) else t

    def idx(self, b: Boot, **where) -> List[int]:
        return b.keys(**where)


def blk(ctx: Ctx, title: str, body: str) -> str:
    return f"\n## {title}\n\n{body}\n"


# ------------------------------------------------------------------------------------------------------------------
# Part B
# ------------------------------------------------------------------------------------------------------------------
def part_b(ctx: Ctx) -> None:
    from scipy.stats import kendalltau, spearmanr
    t = ctx.table.copy()
    ci = read_csv(ctx.report / "stats_ci.csv")
    if ci is not None:
        c = ci[(ci["level"] == "paper") & (ci["metric"] == "geom_at_10")].rename(columns={"run": "model", "estimate": "span_geom_ci_est", "ci_low": "span_geom_ci_low", "ci_high": "span_geom_ci_high"})
        t = t.merge(c[["chunker", "size", "overlap", "model", "span_geom_ci_low", "span_geom_ci_high"]], on=KEY, how="left")
    t.to_csv(ctx.out / "compare_published_vs_new.csv", index=False, float_format="%.6f")
    rows = []

    def corr(d, scope, a, b):
        d = d.dropna(subset=[a, b])
        if len(d) >= 3:
            rows.append({"scope": scope, "pair": f"{a} vs {b}", "n": len(d), "spearman": spearmanr(d[a], d[b])[0], "kendall": kendalltau(d[a], d[b])[0]})
    for a, b in (("pub_geom", "span_geom"), ("pub_geom", "lex_geom"), ("lex_geom", "span_geom")):
        corr(t, "all", a, b)
        for m, d in t.groupby("model"):
            corr(d, f"model {m}", a, b)
        for s, d in t.groupby("chunker"):
            corr(d, f"strategy {s}", a, b)
    pd.DataFrame(rows).to_csv(ctx.out / "compare_correlations.csv", index=False, float_format="%.4f")


# ------------------------------------------------------------------------------------------------------------------
# Part C: one function per claim
# ------------------------------------------------------------------------------------------------------------------
def mean_by(t: pd.DataFrame, by: Sequence[str], cols=("pub_geom", "lex_geom", "span_geom")) -> pd.DataFrame:
    return t.groupby(list(by))[[c for c in cols if c in t]].mean().reset_index()


def p1_table(root: Path, newer: bool) -> pd.DataFrame:
    from tools import make_report as R
    rows = R.external_rows([root])
    return pd.DataFrame(rows)


def phase1_dirs(ctx: Ctx) -> Tuple[Path, Path]:
    return ctx.root / "embedding_models_screening" / "validation_prompts", PUB_P1 / "validation"


def phase1_family(ctx: Ctx) -> str:
    import pipeline_lib.config_params as CP
    fam = {m: f for f, ms in (CP.section("figures").get("model_families") or {}).items() for m in ms}
    dom = set(ctx.cl["domain_models"])
    out = ""
    new_dir, pub_dir = phase1_dirs(ctx)
    for label, d in (("published (no prompts)", pub_dir), ("new (prompts)", new_dir)):
        t = p1_table(d, label.startswith("new"))
        if t.empty:
            out += f"**{label}**: no Phase 1 results found in {d}\n\n"
            continue
        t = t[t["kind"] == "dense"]
        for task in ("ChemNQRetrieval", "ChemHotpotQARetrieval"):
            g = t[["model", f"{task}__geom_at_10"]].rename(columns={f"{task}__geom_at_10": "geom"}).dropna()
            tuned = g[g["model"].isin(fam)]
            dm = g[g["model"].isin(dom)]
            med = tuned["geom"].median()
            for excl in (False, True):
                d2 = dm[dm["model"] != ctx.cl["smiles_model"]] if excl else dm
                if len(d2) == 0 or len(tuned) == 0:
                    continue
                out += (f"- {label}, {task}{', without ChemBERTa-77M-MTR' if excl else ''}: best domain model {short(d2.sort_values('geom').iloc[-1]['model'])} {d2['geom'].max():.3f}; "
                        f"worst tuned family model {short(tuned.sort_values('geom').iloc[0]['model'])} {tuned['geom'].min():.3f}; best tuned {tuned['geom'].max():.3f}; median tuned {med:.3f}; "
                        f"domain models above the tuned median: {int((d2['geom'] > med).sum())} of {len(d2)}; domain models above the worst tuned: {int((d2['geom'] > tuned['geom'].min()).sum())}\n")
    ci = read_csv(ctx.report / "table_phase1_ci.csv")
    if ci is not None:
        c = ci[ci["model"].isin(dom) & (ci["task"] != "union")][["model", "task", "geom_at_10", "ci_low", "ci_high", "rank"]]
        out += "\nNew Phase 1, domain-pretrained models with query-level intervals:\n\n" + md(c)
    return out


def phase1_top5(ctx: Ctx) -> str:
    out = ""
    new_dir, pub_dir = phase1_dirs(ctx)
    for task in ("ChemHotpotQARetrieval", "ChemNQRetrieval"):
        parts = []
        for label, d in (("published", pub_dir), ("new", new_dir)):
            t = p1_table(d, label == "new")
            if t.empty:
                continue
            g = t[t["kind"] == "dense"][["model", f"{task}__geom_at_10"]].dropna().sort_values(f"{task}__geom_at_10", ascending=False).reset_index(drop=True)
            g.index += 1
            g = g.rename(columns={f"{task}__geom_at_10": label})
            parts.append(g.head(8).reset_index().rename(columns={"index": f"rank_{label}"}))
            n15 = g[g["model"] == "nomic-ai/nomic-embed-text-v1.5"]
            out += f"- {task} {label}: nomic-v1.5 rank {int(n15.index[0]) if len(n15) else 'n/a'}\n"
        out += f"\n{task} top 8 (published / new):\n\n" + "".join(md(p) for p in parts)
    return out


def phase1_shortlist(ctx: Ctx) -> str:
    new = read_csv(ctx.root / "embedding_models_screening" / "passed_per_dataset__validation_prompts.csv")
    pub = read_csv(PUB_P1 / "passed_per_dataset.csv")
    if new is None or pub is None:
        return "stage 2 output of the new Phase 1 or the published shortlist is missing\n"
    n, p = set(new["model_name"]), set(pub["model_name"])
    out = f"- new shortlist {len(n)} models, published {len(p)}; in both: {len(n & p)}; only new: {sorted(n - p)}; only published: {sorted(p - n)}\n"
    for lab, d in (("published", pub), ("new", new)):
        sets = [set(d[d["dataset"] == t]["model_name"]) for t in d["dataset"].unique()]
        out += f"- {lab}: models in both top-eight sets: {len(set.intersection(*sets)) if sets else 'n/a'} {sorted(set.intersection(*sets)) if sets else ''}\n"
    ci = read_csv(ctx.report / "table_phase1_ci.csv")
    if ci is not None:
        u = ci[ci["task"] == "union"].sort_values("top_share", ascending=False)
        out += f"\nShortlist stability (share of query-level resamples in which the model is in the top 20 % of at least one task); union size mean {u['union_size_mean'].iloc[0]:.1f} [{u['union_size_low'].iloc[0]:.0f}, {u['union_size_high'].iloc[0]:.0f}]:\n\n"
        out += md(u[["model", "in_top", "top_share"]].head(25))
    return out


def model_vs_strategy(ctx: Ctx) -> str:
    t = ctx.table
    out = "Mean Geom@10 ranges (max - min of the group means), subset: published / new lexical / new span\n\n"
    rows = []
    for col in ("pub_geom", "lex_geom", "span_geom"):
        if col in t:
            rows.append({"labels": col, "range over models": np.ptp(t.groupby("model")[col].mean()), "range over strategies": np.ptp(t.groupby("chunker")[col].mean()),
                         "range over sizes": np.ptp(t.groupby("size")[col].mean()), "range over overlaps": np.ptp(t.groupby("overlap")[col].mean())})
    out += md(pd.DataFrame(rows))
    for f, title in (("variance_decomposition.csv", "eta^2 (Type II, all dense runs of the subset, paper-level intervals)"), ("variance_ranges.csv", "ranges with paper-level intervals")):
        d = read_csv(ctx.analysis / f)
        out += f"\n{title}:\n\n" + md(d) if d is not None else f"\n{f}: missing\n"
    return out


def winners_by_strategy(ctx: Ctx) -> str:
    t = ctx.table
    rows = []
    for col in ("pub_geom", "lex_geom", "span_geom"):
        if col not in t:
            continue
        for s, d in t.groupby("chunker"):
            m = d.groupby("model")[col].mean().sort_values(ascending=False)
            rows.append({"labels": col, "strategy": s, "best model (mean over cells)": short(m.index[0]), "mean": m.iloc[0], "second": f"{short(m.index[1])} {m.iloc[1]:.3f}" if len(m) > 1 else ""})
    out = md(pd.DataFrame(rows))
    w = read_csv(ctx.analysis / "winner_dense_by_configuration.csv")
    if w is not None:
        g = w.groupby(["chunker", "model"])["share_first"].mean().reset_index().sort_values(["chunker", "share_first"], ascending=[True, False])
        out += "\nWinner stability (new, span): mean share of resamples in which the model is first, averaged over the configurations of the strategy:\n\n" + md(g.groupby("chunker").head(3))
    wo = read_csv(ctx.analysis / "winner_dense_by_model.csv")
    if wo is not None:
        out += "\nOverall winner share by model:\n\n" + md(wo.head(10))
    return out


def strategy_means(ctx: Ctx) -> str:
    b = ctx.boot("span")
    out, rows = "", []
    dense = b.keys(kind="dense")
    for s in ctx.cl["subset"]["strategies"]:
        i = [j for j in dense if b.meta.loc[j, "chunker"] == s]
        v, lo, hi = b.mean(i)
        ref = [j for j in dense if b.meta.loc[j, "chunker"] == "fixed_token"]
        d, dlo, dhi, p = b.diff(i, ref) if s != "fixed_token" else (0, 0, 0, 1)
        t = ctx.table[ctx.table["chunker"] == s]
        rows.append({"strategy": s, "span mean [CI]": ci_str(v, lo, hi), "diff vs fixed [CI]": ci_str(d, dlo, dhi), "p": p, "lexical mean": t["lex_geom"].mean(), "published mean": t["pub_geom"].mean()})
    return md(pd.DataFrame(rows)) + "\n(means over the 4 models x 7 cells; paper-level CI, paired over the same resampled papers)\n"


def size_curve(ctx: Ctx) -> str:
    b = ctx.boot("span")
    sizes = sorted({s for s, o in ctx.cl["subset"]["cells_size_sweep_overlap0"]})
    dense0 = b.keys(kind="dense", overlap=0)
    rows = []
    for s in sizes:
        i = [j for j in dense0 if b.meta.loc[j, "size"] == s]
        v, lo, hi = b.mean(i)
        t = ctx.table[(ctx.table["size"] == s) & (ctx.table["overlap"] == 0)]
        rows.append({"size": s, "span mean [CI]": ci_str(v, lo, hi), "lexical": t["lex_geom"].mean(), "published": t["pub_geom"].mean()})
    out = "Overlap 0, mean over the 4 models x 5 strategies:\n\n" + md(pd.DataFrame(rows))
    drows = []
    for a, c in ((256, 128), (384, 256), (448, 384), (512, 448), (512, 128), (512, 256)):
        ia = [j for j in dense0 if b.meta.loc[j, "size"] == a]
        ic = [j for j in dense0 if b.meta.loc[j, "size"] == c]
        d, lo, hi, p = b.diff(ia, ic)
        drows.append({"difference": f"{a} - {c}", "span [CI]": ci_str(d, lo, hi), "p": p})
    out += "\nDifferences (paired bootstrap):\n\n" + md(pd.DataFrame(drows))
    t0 = ctx.table[ctx.table["overlap"] == 0]
    out += "\nBy model (span / published):\n\n" + md(t0.pivot_table(index="model", columns="size", values="span_geom").reset_index())
    out += md(t0.pivot_table(index="model", columns="size", values="pub_geom").reset_index())
    return out


def overlap_curve(ctx: Ctx) -> str:
    b = ctx.boot("span")
    dense256 = b.keys(kind="dense", size=256)
    rows = []
    for o in (0, 32, 64):
        i = [j for j in dense256 if b.meta.loc[j, "overlap"] == o]
        v, lo, hi = b.mean(i)
        t = ctx.table[(ctx.table["size"] == 256) & (ctx.table["overlap"] == o)]
        rows.append({"overlap (size 256)": o, "span mean [CI]": ci_str(v, lo, hi), "lexical": t["lex_geom"].mean(), "published": t["pub_geom"].mean()})
    out = md(pd.DataFrame(rows))
    drows = []
    for o in (32, 64):
        ia = [j for j in dense256 if b.meta.loc[j, "overlap"] == o]
        i0 = [j for j in dense256 if b.meta.loc[j, "overlap"] == 0]
        d, lo, hi, p = b.diff(ia, i0)
        drows.append({"difference": f"overlap {o} - 0", "span [CI]": ci_str(d, lo, hi), "p": p})
        for s in ctx.cl["subset"]["strategies"]:
            sa = [j for j in ia if b.meta.loc[j, "chunker"] == s]
            s0 = [j for j in i0 if b.meta.loc[j, "chunker"] == s]
            d, lo, hi, p = b.diff(sa, s0)
            drows.append({"difference": f"  {s}", "span [CI]": ci_str(d, lo, hi), "p": p})
    return out + "\n" + md(pd.DataFrame(drows)) + "\nLocal optima at 16 or 32: NOT TESTABLE (only overlaps 0, 32, 64 at size 256 in the subset).\n"


def best_runs(ctx: Ctx) -> str:
    t = ctx.table
    out = ""
    for col in ("pub_geom", "lex_geom", "span_geom"):
        if col in t:
            top = t.sort_values(col, ascending=False).head(5)[["model", "chunker", "size", "overlap", col]]
            out += f"\nTop 5 runs ({col}):\n\n" + md(top)
    return out


def top_region(ctx: Ctx) -> str:
    t = ctx.table
    frac = ctx.cl["subset"]["top_fraction_for_top1pct_claims"]
    n = max(1, int(round(len(t) * frac)))
    rows = []
    for col in ("pub_geom", "lex_geom", "span_geom"):
        if col not in t:
            continue
        top = t.sort_values(col, ascending=False).head(n)
        thr = top[col].min()
        rows.append({"labels": col, "n runs": n, "threshold": thr, "only e5-large-v2": bool((top["model"] == "intfloat/e5-large-v2").all()), "share e5-large-v2": (top["model"] == "intfloat/e5-large-v2").mean(),
                     "share sizes 384-512": top["size"].between(384, 512).mean(), "share overlap<=32": (top["overlap"] <= 32).mean(),
                     "strategies": dict(top["chunker"].value_counts()), "models": {short(k): int(v) for k, v in top["model"].value_counts().items()}})
    return f"Top {int(frac * 100)} % of the {len(t)} dense runs of the subset (the paper's top 1 % of 1,820 runs is not comparable with 140 runs):\n\n" + md(pd.DataFrame(rows))


def top10_configs(ctx: Ctx) -> str:
    t = ctx.table
    rows = []
    for col in ("pub_geom", "lex_geom", "span_geom"):
        if col not in t:
            continue
        top = t.sort_values(col, ascending=False).head(10)
        rows.append({"labels": col, "all e5-large-v2": bool((top["model"] == "intfloat/e5-large-v2").all()), "n e5-large-v2": int((top["model"] == "intfloat/e5-large-v2").sum()),
                     "n size 448/512": int(top["size"].isin([448, 512]).sum()), "n overlap<=32": int((top["overlap"] <= 32).sum()), "n overlap 0": int((top["overlap"] == 0).sum()),
                     "strategies": dict(top["chunker"].value_counts())})
    return md(pd.DataFrame(rows))


def time_structure(ctx: Ctx) -> str:
    t = ctx.table
    out = "Evaluation time per run (s): new code `evaluation_time_s` (dense: MTEB's time; chunking and labelling are done once per configuration and are NOT in it) vs published `evaluation_time` (chunking included)\n\n"
    for by in ("model", "chunker", "size", "overlap"):
        g = t.groupby(by)[[c for c in ("time_s", "pub_time_s") if c in t]].mean().reset_index()
        out += f"By {by}:\n\n" + md(g, "{:.1f}")
    ov = ctx.ov[ctx.ov["kind"] == "dense"]
    parts = [c for c in ("chunking_s", "qrels_span_s", "qrels_lexical_s", "encode_passage_s", "encode_query_s", "search_metrics_and_dump_s", "model_load_s") if c in ov]
    if parts:
        out += "\nParts of the new dense run (mean over runs; chunking_s / qrels_* are per configuration, shared by all models):\n\n" + md(ov.groupby("chunker")[parts].mean().reset_index(), "{:.1f}")
    rng = {by: float(np.ptp(t.groupby(by)["time_s"].mean())) for by in ("model", "chunker", "size", "overlap") if "time_s" in t}
    out += f"\nRanges of the group means of time_s: {rng}\n"
    return out


def time_vs_quality(ctx: Ctx) -> str:
    from scipy.stats import spearmanr
    t = ctx.table.dropna(subset=["time_s"])
    rng = np.random.default_rng(1)
    rows = []
    for lab, col, tcol in (("new span", "span_geom", "time_s"), ("published", "pub_geom", "pub_time_s")):
        for scope, d in [("all", t)] + [(short(m), g) for m, g in t.groupby("model")]:
            d = d.dropna(subset=[col, tcol])
            if len(d) < 4:
                continue
            rho = spearmanr(d[tcol], d[col])[0]
            bs = [spearmanr(d[tcol].values[i], d[col].values[i])[0] for i in (rng.integers(0, len(d), len(d)) for _ in range(500))]
            rows.append({"labels": lab, "scope": scope, "n": len(d), "spearman(time, Geom@10)": rho, "CI low": np.nanquantile(bs, 0.025), "CI high": np.nanquantile(bs, 0.975)})
    return md(pd.DataFrame(rows)) + "\n(bootstrap over runs, 500 resamples; runs of one model are not independent: read as descriptive)\n"


def recommendation(ctx: Ctx) -> str:
    return "See claims 5, 6, 7, 8 and 15: the recommendation combines them. Best run per model (span):\n\n" + md(
        ctx.table.sort_values("span_geom", ascending=False).groupby("model").head(1)[["model", "chunker", "size", "overlap", "span_geom", "lex_geom", "pub_geom"]])


def baselines(ctx: Ctx) -> str:
    b = ctx.boot("span")
    kinds = sorted(set(b.meta["kind"]))
    sizes = sorted({s for s, o in ctx.cl["subset"]["cells_size_sweep_overlap0"]})
    rows = []
    for s in sizes:
        r = {"size": s}
        idx = {k: b.keys(kind=k, size=s, overlap=0) for k in kinds}
        for k in kinds:
            v, lo, hi = b.mean(idx[k])
            r[k] = ci_str(v, lo, hi)
        if idx.get("bm25") and idx.get("dense"):
            d, lo, hi, p = b.diff(idx["bm25"], idx["dense"])
            r["bm25 - dense"] = ci_str(d, lo, hi)
        if idx.get("hybrid") and idx.get("dense"):
            d, lo, hi, p = b.diff(idx["hybrid"], idx["dense"])
            r["hybrid - dense"] = ci_str(d, lo, hi)
        if idx.get("hybrid") and idx.get("bm25"):
            d, lo, hi, p = b.diff(idx["hybrid"], idx["bm25"])
            r["hybrid - bm25"] = ci_str(d, lo, hi)
        e5 = [j for j in idx.get("dense", []) if b.meta.loc[j, "model"] == "intfloat/e5-large-v2"]
        if idx.get("rerank") and e5:
            d, lo, hi, p = b.diff(idx["rerank"], e5)
            r["rerank - e5-large-v2"] = ci_str(d, lo, hi)
        rows.append(r)
    out = "Overlap 0, mean Geom@10 over strategies (and over the 4 models for dense and hybrid), span labels, paper-level CI:\n\n" + md(pd.DataFrame(rows))
    sl = ctx.table
    out += "\nBM25 is better than the mean dense run (CI above 0) at sizes: see the column `bm25 - dense`.\n"
    st = read_csv(ctx.report / "stats_compare.csv")
    if st is not None:
        s = st[(st["level"] == "paper") & (st["metric"] == "geom_at_10") & (st["B"] == "bm25")]
        out += f"\nstats_compare.csv, dense vs BM25 per run (paper level): {int((s['diff'] < 0).sum())} of {len(s)} comparisons have the dense run below BM25; significant (Holm) BM25 wins: " \
               f"{int(((s['diff'] < 0) & (s.get('p_bootstrap_holm', 1) < 0.05)).sum())}\n"
    return out


def token_budget(ctx: Ctx) -> str:
    d = read_csv(ctx.analysis / "token_budget.csv")
    if d is None:
        return "token_budget.csv missing\n"
    d = d[d["overlap"] == 0]
    g = d.groupby(["kind", "budget_tokens", "size"])[["hit_rate", "mean_span_coverage"]].mean().reset_index()
    out = "Mean over models and strategies (overlap 0), hit rate = share of questions with >= c of the span inside the retrieved budget, by chunk size:\n\n"
    for k in ("dense", "bm25", "hybrid", "rerank"):
        gk = g[g["kind"] == k]
        if len(gk):
            out += f"{k}:\n\n" + md(gk.pivot_table(index="budget_tokens", columns="size", values="hit_rate").reset_index())
    return out + "\n(means of run-level values; run-level paper-level intervals are in token_budget.csv)\n"


def fragmentation(ctx: Ctx) -> str:
    d = read_csv(ctx.analysis / "fragmentation.csv")
    if d is None:
        return "fragmentation.csv missing\n"
    d = d[d["metric"] == "ndcg_at_10"] if "metric" in d else d
    s = d.groupby(["strategy", "small", "large"]).agg(pct_fragmented=("pct_fragmented", "mean"), gain_fragmented=("mean_gain_fragmented", "mean"), gain_not=("mean_gain_not_fragmented", "mean"),
                                                      diff=("difference", "mean"), rows=("difference", "size"), ci_above_0=("difference_ci_low", lambda x: int((x > 0).sum()))).reset_index()
    return md(s) + "\n(difference = gain of fragmented questions - gain of the others; ci_above_0 = number of model rows whose interval is above 0)\n"


def breakdowns(ctx: Ctx) -> str:
    from scipy.stats import kendalltau
    out = ""
    for f in ("breakdown_question_type", "breakdown_subfield", "breakdown_question_features"):
        d = read_csv(ctx.analysis / f"{f}.csv")
        if d is None:
            out += f"{f}.csv missing\n"
            continue
        d = d[(d["metric"] == "geom_at_10") & (d["row_kind"] == "group") & (d["kind"] == "dense")]
        d = d[~d["group"].str.contains(r"\|", regex=True)]
        key = ["label_rule"] if "label_rule" in d else []
        rows = []
        for lr, dd in (d.assign(label_rule=d["label_rule"].fillna("")).groupby(key) if key else [("", d)]):
            om = dd.groupby("model")["value"].mean()
            osz = dd[dd["overlap"] == 0].groupby("size")["value"].mean()
            ost = dd.groupby("chunker")["value"].mean()
            for g, gg in dd.groupby("group"):
                gm, gs, gst = gg.groupby("model")["value"].mean(), gg[gg["overlap"] == 0].groupby("size")["value"].mean(), gg.groupby("chunker")["value"].mean()
                rows.append({"group set": f"{f} {lr}".strip(), "group": g, "n_questions": int(gg["n_questions"].iloc[0]), "best model": short(gm.idxmax()), "tau models": kendalltau(om.reindex(gm.index), gm)[0],
                             "best size": int(gs.idxmax()) if len(gs) else "", "tau sizes": kendalltau(osz.reindex(gs.index), gs)[0] if len(gs) > 2 else float("nan"),
                             "best strategy": gst.idxmax(), "tau strategies": kendalltau(ost.reindex(gst.index), gst)[0] if len(gst) > 2 else float("nan"),
                             "overall best model": short(om.idxmax())})
        r = pd.DataFrame(rows)
        if r.empty:
            out += f"\n{f}: no dense group rows\n"
            continue
        flag = r[(r["best model"] != r["overall best model"]) | (r["tau models"] < 0.5) | (r["tau sizes"] < 0.5)]
        out += f"\n{f}: {len(r)} groups; groups where the best model differs from the overall best or a ranking correlates with tau < 0.5 with the overall ranking: {len(flag)}\n\n" + md(flag, maxrows=40)
    return out


def cover_levels(ctx: Ctx) -> str:
    d = read_csv(ctx.report / "table_cover_sensitivity.csv")
    if d is None:
        return "table_cover_sensitivity.csv missing\n"
    gcol = next((c for c in d.columns if "geom" in c.lower()), None)
    if gcol is None:
        return f"columns: {list(d.columns)}\n"
    d = d[d["kind"] == "dense"] if "kind" in d else d
    from scipy.stats import spearmanr
    p = d.pivot_table(index=["model", "chunker", "size", "overlap"], columns="cover_level", values=gcol)
    out = f"Geom@10 column `{gcol}`; dense runs; Spearman between cover levels:\n\n" + md(p.corr(method="spearman").reset_index())
    rows = []
    for c, g in d.groupby("cover_level"):
        top = g.sort_values(gcol, ascending=False).iloc[0]
        rows.append({"c": c, "best run": f"{short(top['model'])} {top['chunker']} {top['size']}/{top['overlap']}", "best": top[gcol],
                     "best model (mean)": short(g.groupby("model")[gcol].mean().idxmax()), "best strategy": g.groupby("chunker")[gcol].mean().idxmax(),
                     "best size at o=0": int(g[g["overlap"] == 0].groupby("size")[gcol].mean().idxmax())})
    return out + "\n" + md(pd.DataFrame(rows))


def transfer(ctx: Ctx) -> str:
    out = ""
    for f in ("phase_transfer_correlations.csv", "phase_transfer_beats.csv", "phase_transfer_groups.csv"):
        d = read_csv(ctx.analysis / f)
        out += f"\n{f}:\n\n" + (md(d, maxrows=25) if d is not None else "missing (the transfer subset did not run, or phase_transfer was skipped)\n")
    return out


CLAIM_FUNCS = {n: f for n, f in list(globals().items()) if callable(f)}


def part_c(ctx: Ctx) -> None:
    out = ["# Claim numbers of the subset check (numbers only; verdicts are assigned in Session 2)\n",
           f"Subset: {len(ctx.models)} models {ctx.models}; {len(ctx.table)} dense runs with published numbers; resamples {ctx.a.resamples}. "
           "Published = old code (model tokenizer, lexical labels, no prompts, chunk title), restricted to the same runs. "
           "lexical = new code, lexical qrels; span = new code, span qrels (primary).\n"]
    for c in ctx.cl["claims"]:
        fn = CLAIM_FUNCS.get(c["calc"])
        head = f"Claim {c['id']} (main:{c['lines']}): {c['text']}"
        try:
            body = fn(ctx)
        except Exception as e:                                          # noqa: BLE001 - one failing claim must not stop the others
            body = f"**FAILED**: {type(e).__name__}: {e}\n\n```\n{traceback.format_exc(limit=3)[-600:]}\n```\n"
        out.append(blk(ctx, head, body))
    (ctx.out / "claims_numbers.md").write_text("".join(out), encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------------
# Part D
# ------------------------------------------------------------------------------------------------------------------
REVIEWER_OUTPUTS = [
    # (comment, pattern relative to the out dir unless it starts with @root/ or @repo/)
    ("Editor: declarations / title (text only)", None),
    ("R1.1 uncertainty: bootstrap CIs", "@root/report/stats_ci.csv"),
    ("R1.1 uncertainty: paired tests", "@root/report/stats_compare.csv"),
    ("R1.2 labels: label statistics", "@root/report/table_label_stats.csv"),
    ("R1.2 labels: manual validation sheets", "validation/validation_sheet_annotator_1.csv"),
    ("R1.3 prompts / models: model properties", "table_model_properties.csv"),
    ("R1.4 baselines: BM25 / hybrid / rerank tables", "@root/report/table_retrievers_span.csv"),
    ("R1.4 baselines: lexical labels", "@root/report/table_retrievers_lexical.csv"),
    ("R1.5 subfield breakdown", "analysis/breakdown_subfield.csv"),
    ("R1.5 question types", "analysis/breakdown_question_type.csv"),
    ("R1.5 automatic features", "analysis/breakdown_question_features.csv"),
    ("R1.6 external validity: transfer runs", "@root/../subset_v1_transfer/report/runs_overview.csv"),
    ("R1.7 evaluation time / efficiency", "@root/tests/efficiency_phase2.csv"),
    ("R2 methods: prompts used", "@root/report/runs_overview.csv"),
    ("R2 methods: pooling / dimension / max length", "table_model_properties.csv"),
    ("R2 methods: truncation Phase 2", "@root/report/table_truncation.csv"),
    ("R2 methods: truncation Phase 1", "@root/report/table_truncation_phase1.csv"),
    ("R2 methods: dataset splits (Phase 1 table with split, n_queries)", "@root/report/table_external.csv"),
    ("R2 methods: Phase 1 per-query CIs and shortlist stability", "@root/report/table_phase1_ci.csv"),
    ("R2 methods: computational settings (manifest)", "report_manifest.json"),
    ("R2 major 1 fixed token budget", "analysis/token_budget.csv"),
    ("R2 major 2 cover-level sensitivity", "@root/report/table_cover_sensitivity.csv"),
    ("R2 major 3 Phase 1 -> Phase 2 transfer", "analysis/phase_transfer_correlations.csv"),
    ("R2 major 4 variance decomposition", "analysis/variance_decomposition.csv"),
    ("R2 major 4 winner stability", "analysis/winner_dense_overall.csv"),
    ("R2 major 5 stats: families / Holm", "@root/report/stats_compare.csv"),
    ("R2 major 5 hybrid / rerank vs BM25", "@root/report/table_retrievers_span.csv"),
    ("R2 major 5 rerank depth ablation", "@root/report/table_rerank_ablation.csv"),
    ("R2 major 6 fragmentation", "analysis/fragmentation.csv"),
    ("R2 major 6 examples", "analysis/examples.csv"),
    ("R2 minor 1 efficiency tables", "@root/report/../tests/efficiency_phase2.csv"),
    ("R2 minor 3 Figure 3 replacement", "figures/fig_tradeoff_by_model.png"),
    ("R2 presentation: chunk-size figure", "figures/fig_chunk_size_by_strategy.png"),
    ("R2 presentation: overlap figure", "figures/fig_overlap_by_size.png"),
    ("manual annotation of the automatic features", "validation/question_features_check.csv"),
]


def part_d(ctx: Ctx) -> None:
    rows = []
    for comment, pat in REVIEWER_OUTPUTS:
        if pat is None:
            rows.append({"comment": comment, "file": "", "exists": "n/a", "non_empty": "n/a", "rows": ""})
            continue
        p = Path(pat.replace("@root", str(ctx.root))) if pat.startswith("@root") else ctx.out / pat
        ex = p.exists()
        ne = ex and p.stat().st_size > 0
        n = ""
        if ne and p.suffix == ".csv":
            try:
                n = sum(1 for _ in open(p, encoding="utf-8")) - 1
            except Exception:
                n = "?"
        rows.append({"comment": comment, "file": str(p), "exists": ex, "non_empty": ne, "rows": n})
    pd.DataFrame(rows).to_csv(ctx.out / "reviewer_outputs.csv", index=False)


# ------------------------------------------------------------------------------------------------------------------
# Part E
# ------------------------------------------------------------------------------------------------------------------
def part_e(ctx: Ctx) -> None:
    rows: List[Dict[str, Any]] = []

    def add(stage, item, seconds, basis, note=""):
        rows.append({"stage": stage, "item": item, "seconds": None if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)) else float(seconds), "hours": None if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)) else float(seconds) / 3600, "basis": basis, "note": note})
    steps = read_csv(ctx.out / "steps.csv")
    ov = ctx.ov
    dense = ov[ov["kind"] == "dense"]
    pub = ctx.published()
    n_cells_full, n_sub = 140, 35
    # ---- measured steps of the driver
    if steps is not None:
        for _, r in steps.iterrows():
            add("driver step (measured)", r["step"], r["seconds"], "measured", str(r.get("note", "")))
    # ---- dense: wall time per run per model from the driver (includes model load, probes, latency)
    per_model_wall: Dict[str, float] = {}
    if steps is not None:
        for _, r in steps.iterrows():
            if str(r["step"]).startswith("dense:") and r["seconds"] and r.get("runs", 0):
                per_model_wall[str(r["step"]).split(":", 1)[1].split(":")[0]] = per_model_wall.get(str(r["step"]).split(":", 1)[1].split(":")[0], 0) + float(r["seconds"])
    sub_cfg = dense[["chunker", "size", "overlap"]].drop_duplicates()
    for m in ctx.models:
        d = dense[dense["model"] == m]
        if d.empty:
            continue
        add("dense", f"{m}: mean evaluation_time_s per run", d["evaluation_time_s"].mean(), "measured", f"{len(d)} runs")
    # scale factor of the full grid vs the subset cells for the same model, from the published times (old code, same hardware family)
    pm = pub[pub["model"].isin(ctx.models)]
    p_all = pm.groupby("model")["pub_time_s"].mean()
    p_sub = pm.merge(sub_cfg, on=["chunker", "size", "overlap"]).groupby("model")["pub_time_s"].mean()
    cellfac = (p_all / p_sub).replace([np.inf, -np.inf], np.nan)
    new_t = dense.groupby("model")["evaluation_time_s"].mean()
    ratio = (new_t / p_sub).dropna()
    # Phase 1 times of the new run (evaluation_time per task), used to scale the models that were not run
    from tools import make_report as R
    p1 = pd.DataFrame(R.external_rows([ctx.root / "embedding_models_screening" / "validation_prompts"]))
    tcols = [c for c in p1.columns if c.endswith("__evaluation_time_s")]
    p1_time = p1[p1["kind"] == "dense"].set_index("model")[tcols].sum(axis=1) if len(p1) else pd.Series(dtype=float)
    full_models = ctx.a.full_models or []
    meas = {m: new_t[m] * (cellfac.get(m, 1.0) if not np.isnan(cellfac.get(m, np.nan)) else 1.0) for m in new_t.index}
    if len(p1_time) and len(meas) >= 2:
        xs = np.array([p1_time[m] for m in meas if m in p1_time.index])
        ys = np.array([meas[m] for m in meas if m in p1_time.index])
        slope = float((xs * ys).sum() / (xs * xs).sum()) if len(xs) else float("nan")
    else:
        slope = float("nan")
    total_dense = 0.0
    per_model_full: Dict[str, float] = {}
    for m in full_models:
        if m in meas:
            per_model_full[m] = meas[m] * n_cells_full
            basis = "measured per run, scaled to 140 configurations"
        elif m in p1_time.index and not math.isnan(slope):
            per_model_full[m] = slope * p1_time[m] * n_cells_full
            basis = "extrapolated from the model's Phase 1 time (least-squares through the origin on the measured models)"
        else:
            per_model_full[m] = float("nan")
            basis = "no Phase 1 time"
        add("full rerun: dense evaluation_time", f"{m} x 140 configurations", per_model_full[m], basis)
        total_dense += 0 if math.isnan(per_model_full[m]) else per_model_full[m]
    # driver wall time overhead per run (load, probes, latency, writing) vs evaluation_time
    if per_model_wall:
        over = []
        for m, w in per_model_wall.items():
            d = dense[dense["model"] == m]
            if len(d):
                over.append(w / max(1, len(d)) / d["evaluation_time_s"].mean())
        if over:
            f = float(np.mean(over))
            add("full rerun: dense", "wall time / evaluation_time (model load, probes, latency, disk), measured on the subset", f, "measured", "ratio, not seconds")
            total_dense *= f
    add("full rerun: dense", "total wall time of all dense runs, one GPU", total_dense, "mixed (see rows above)")
    # LPT split over two GPUs
    loads = sorted([v for v in per_model_full.values() if not math.isnan(v)], reverse=True)
    g = [0.0, 0.0]
    for v in loads:
        g[g.index(min(g))] += v
    add("full rerun: dense", "wall time with two GPUs (largest model first on the emptier GPU)", max(g) * (total_dense / sum(loads) if sum(loads) else 1), "mixed")
    # ---- BM25, hybrid, rerank
    for kind, n_full, label in (("bm25", 140, "BM25 x 140"), ("hybrid", 140 * max(1, len(full_models)), f"hybrid x {140 * max(1, len(full_models))}"), ("rerank", 140, "rerank (primary depth) x 140")):
        d = ov[ov["kind"] == kind]
        col = {"bm25": "evaluation_time_s", "hybrid": "fusion_s", "rerank": "rerank_s"}[kind]
        if col in d and d[col].notna().any():
            add("full rerun: " + kind, label, d[col].mean() * n_full, "measured per run, scaled")
    # ---- prepare (chunking + labelling once per configuration)
    part_cols = [c for c in ("chunking_s", "qrels_span_s", "qrels_lexical_s") if c in dense]
    if part_cols:
        per_cfg = dense.drop_duplicates(["chunker", "size", "overlap"])[part_cols].sum(axis=1)
        by_s = dense.drop_duplicates(["chunker", "size", "overlap"]).assign(t=per_cfg).groupby("chunker")["t"].mean()
        add("full rerun: prepare", "chunk + label 140 configurations (per-strategy mean x 28 per strategy)", float(by_s.sum() * 28), "measured per configuration, scaled", "excludes coverage diagnostics")
    # ---- report / analysis / figures: scale with the number of runs
    n_sub_runs = len(ov)
    n_full_runs = 13 * 140 * 2 + 140 * 2 + 140
    if steps is not None:
        for key in ("make report", "make analysis", "figures"):
            s = steps[steps["step"].astype(str).str.startswith(key)]
            if len(s):
                add("full rerun: " + key, f"scaled by runs ({n_sub_runs} -> {n_full_runs})", float(s["seconds"].sum()) * n_full_runs / n_sub_runs, "extrapolated (linear in the number of runs)")
        for key in ("phase1", "transfer"):
            s = steps[steps["step"].astype(str).str.startswith(key)]
            if len(s):
                f = 1.0 if key == "phase1" else 6 / 1
                add("full rerun: " + key, "Phase 1 (41 models, measured)" if key == "phase1" else "transfer: 6 configurations (measured 1 x 6; the extra configuration C6 is cheap)", float(s["seconds"].sum()) * f, "measured" if key == "phase1" else "extrapolated")
    df = pd.DataFrame(rows)
    # totals
    full = df[df["stage"].astype(str).str.startswith("full rerun")]
    tot = full[~full["item"].astype(str).str.contains("two GPUs|ratio|wall time / ")]["seconds"].sum()
    df = pd.concat([df, pd.DataFrame([{"stage": "TOTAL", "item": "full rerun, one GPU (sum of the rows above; dense counted once)", "seconds": tot - 0, "hours": tot / 3600, "basis": "mixed", "note": "rows marked two GPUs / ratio excluded"}])], ignore_index=True)
    df.to_csv(ctx.out / "time_estimate.csv", index=False, float_format="%.2f")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="artifacts/subset_v1")
    ap.add_argument("--out", default="results/v2-revision/subset")
    ap.add_argument("--claims", default=str(REPO / "scripts" / "subset_claims.yaml"))
    ap.add_argument("--qa")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--full-models", nargs="+", help="the models of the full rerun (default: the published shortlist)")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--only", nargs="+", choices=["b", "c", "d", "e"])
    a = ap.parse_args()
    if not a.full_models:
        pub = read_csv(PUB_P1 / "passed_per_dataset.csv")
        a.full_models = sorted(set(pub["model_name"])) if pub is not None else []
    Path(a.out).mkdir(parents=True, exist_ok=True)
    ctx = Ctx(a)
    for name, fn in (("b", part_b), ("c", part_c), ("d", part_d), ("e", part_e)):
        if a.only and name not in a.only:
            continue
        try:
            fn(ctx)
            print(f"part {name}: ok")
        except Exception as e:                                            # noqa: BLE001
            print(f"part {name}: FAILED {type(e).__name__}: {e}")
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
