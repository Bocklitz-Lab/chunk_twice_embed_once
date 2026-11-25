"""Supplement table of the Phase 2 model selection: per model the Geom@10 and rank per task, the inclusion probability per task, the selection under the four rules
(published top 20 %, P >= 0.4 / 0.5 / 0.6) and whether the model is on the PUBLISHED shortlist (no-prompt Phase 1) and on the new one (primary rule).
Also the SPECTER pooling comparison (mean vs CLS) when its two Phase 1 folders are given.

  python -m tools.shortlist_table --selection <shortlist_selection__*.csv> --published artifacts/embedding_models_screening/passed_per_dataset.csv --out-dir results/v2-revision
Writes shortlist_selection.csv (all 41 models), table_shortlist_supplement.tex (models selected by any rule or with P >= 0.2) and, with --specter-mean / --specter-cls, table_specter_pooling.csv.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

TASKS = ("ChemHotpotQARetrieval", "ChemNQRetrieval")


def build(selection_csv: Any, published_csv: Any) -> pd.DataFrame:
    s = pd.read_csv(selection_csv)
    pub = {r["model_name"] for r in csv.DictReader(open(published_csv, newline="", encoding="utf-8"))}
    s["published_shortlist"] = s["model"].isin(pub)
    s["new_shortlist"] = s["selected_primary"].astype(bool)
    s["change"] = [("stays" if p and n else "leaves" if p else "enters" if n else "") for p, n in zip(s["published_shortlist"], s["new_shortlist"])]
    return s


def to_latex(df: pd.DataFrame, path: Path) -> None:
    d = df[df["new_shortlist"] | df["published_shortlist"] | (df["max_p_top"] >= 0.2)]
    yn = lambda b: "yes" if b else "--"
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Phase 2 model selection from the prompted Phase 1 (test splits: ChemHotpotQARetrieval 18 queries, ChemNQRetrieval 27 queries). $P$ is the share of 10{,}000 query-level resamples in which the model is in the top 20\,\% (top eight of 41) of the task. Rules: published top 20\,\% (point estimate), $P\ge0.4$, $P\ge0.5$ (primary), $P\ge0.6$; the observed top eight of each task are always kept. Last columns: membership in the published shortlist (no prompts) and in the new one.}",
             r"\label{tab:supp_shortlist_selection}", r"\begin{tabular}{lrrrrccccc}", r"\toprule",
             r"Model & rank H & $P$ H & rank NQ & $P$ NQ & top 20\,\% & $\ge$0.4 & $\ge$0.5 & $\ge$0.6 & published $\to$ new \\", r"\midrule"]
    for _, r in d.sort_values(["max_p_top"], ascending=False).iterrows():
        lines.append(" & ".join([r["model"].replace("_", r"\_"), str(int(r[f"rank_{TASKS[0]}"])), f"{r[f'p_top_{TASKS[0]}']:.2f}", str(int(r[f"rank_{TASKS[1]}"])), f"{r[f'p_top_{TASKS[1]}']:.2f}",
                                 yn(r["selected_top_fraction"]), yn(r["selected_p0.4"]), yn(r["selected_p0.5"]), yn(r["selected_p0.6"]), r["change"] or "--"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def specter(mean_root: Any, cls_root: Any) -> pd.DataFrame:
    def g(root):
        out = {}
        for t in TASKS:
            p = glob.glob(f"{root}/*specter*/*/*/{t}.json") or glob.glob(f"{root}/*/*/*/{t}.json")
            sc = json.load(open(p[0]))["scores"]["test"][0]
            out[t] = (math.sqrt(sc["ndcg_at_10"] * sc["recall_at_10"]), sc["ndcg_at_10"], sc["recall_at_10"])
        return out
    a, b = g(mean_root), g(cls_root)
    rows = [{"task": t, "pooling": name, "geom_at_10": v[t][0], "ndcg_at_10": v[t][1], "recall_at_10": v[t][2]} for t in TASKS for name, v in (("mean (library default)", a), ("CLS (params.model_pooling)", b))]
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--published", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--specter-mean")
    ap.add_argument("--specter-cls")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = build(a.selection, a.published)
    df.to_csv(out / "shortlist_selection.csv", index=False, float_format="%.4f")
    to_latex(df, out / "table_shortlist_supplement.tex")
    if a.specter_mean and a.specter_cls:
        specter(a.specter_mean, a.specter_cls).to_csv(out / "table_specter_pooling.csv", index=False, float_format="%.4f")
    print(f"{len(df)} models; published {int(df.published_shortlist.sum())}, new {int(df.new_shortlist.sum())}; stays {int((df.change == 'stays').sum())}, leaves {int((df.change == 'leaves').sum())}, enters {int((df.change == 'enters').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
