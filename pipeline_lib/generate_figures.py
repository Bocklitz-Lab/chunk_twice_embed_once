#!/usr/bin/env python3
"""
make_rag_figs.py

Generate figures and tables for a RAG chunking+embedding evaluation from the CSV
produced by your composite-loss script.

Outputs:
- Figure 1: (optional) pipeline schematic (Graphviz DOT + PNG if graphviz available)
- Figure 2: Leaderboard heatmap (model x chunker) using min-loss over c,o
- Figure 3: Loss vs overlap (lines) at fixed chunk size (default c=448)
- Figure 4: Loss vs chunk size (lines) at o=0
- Figure 5: Pareto frontier (loss vs cost_norm)
- Figure 6: Precision decay vs nDCG@10 scatter
- Figure 7: Robustness bars (NAUC with 0.5*std error bars) and/or loss_rob
- Figure 8: Radar (spider) chart for Top-3 configs

Tables:
- Table 2: Overall leaderboard (top-k) — LaTeX + CSV subset
- Table 3: Marginal means by model and by chunker — LaTeX + CSV

Author: you 🧪
"""

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import dedent

sns.set_context("talk")
sns.set_style("whitegrid")

# ----------------------------- helpers -----------------------------

def ensure_cols(df: pd.DataFrame):
    # Backfill helpful columns if missing.
    if 'precision_at_1' in df and 'precision_at_10' in df:
        with np.errstate(divide='ignore', invalid='ignore'):
            decay = 1 - (df['precision_at_10'] / (df['precision_at_1'].replace(0, np.nan)))
        df['loss_decay'] = np.clip(decay.fillna(0), 0, 1)
    if 'nauc_ndcg_at_10_max' in df and 'nauc_ndcg_at_10_std' in df:
        r_rob = np.maximum(0.0, df['nauc_ndcg_at_10_max'] - 0.5 * df['nauc_ndcg_at_10_std'].abs())
        r_rob_norm = np.minimum(1.0, r_rob / 0.2)
        df['loss_rob'] = 1.0 - r_rob_norm
    if 'cost_norm' not in df:
        df['cost_norm'] = 1.0
    # Normalize chunker strings (optional)
    df['chunker'] = df['chunker'].fillna('unknown')
    # Friendly model short names
    df['model_short'] = df['model_name'].str.replace(r'[_\-]+', ' ', regex=True)

    # Coerce numeric
    for c in ['token_size','overlap']:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def topk_configs(df, k=20):
    return df.sort_values('loss_total', ascending=True).head(k).copy()

def reduce_min_loss_by(df, keys=('model_name','chunker')):
    agg = (df.groupby(list(keys))['loss_total']
             .min()
             .reset_index())
    return agg

def pareto_frontier(df, x='cost_norm', y='loss_total'):
    """Return boolean mask of non-dominated points (minimize both x and y)."""
    d = df[[x, y]].to_numpy()
    order = np.lexsort((d[:,1], d[:,0]))  # sort by x then y
    best_y = math.inf
    mask = np.zeros(len(df), dtype=bool)
    for idx in order:
        xi, yi = d[idx]
        if yi < best_y - 1e-12:
            mask[idx] = True
            best_y = yi
    return mask

def savefig(fig, outdir: Path, name: str, tight=True):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in (".png",".pdf"):
        fig.savefig(outdir / f"{name}{ext}", bbox_inches='tight' if tight else None, dpi=300)
    plt.close(fig)

def latex_table(df, caption, label, floatfmt="{:.3f}", bold_cols=()):
    df2 = df.copy()

    # Bold best values for columns listed in bold_cols
    for col in bold_cols:
        if col in df2:
            if df2[col].dtype.kind in "fiu":
                best = df2[col].min() if col.startswith('loss') else df2[col].max()
            else:
                best = None

            def fmt_bold(v):
                if isinstance(v, (float, int, np.floating, np.integer)):
                    s = floatfmt.format(v)
                else:
                    s = str(v)
                if best is not None and v == best:
                    return r"\textbf{" + s + "}"
                return s

            df2[col] = df2[col].apply(fmt_bold)

    # Format other numeric columns
    for col in df2.columns:
        if col in bold_cols:
            continue
        if pd.api.types.is_numeric_dtype(df2[col]):
            df2[col] = df2[col].apply(lambda v: floatfmt.format(v))

    header = " & ".join(df2.columns) + r" \\"
    rows = [" & ".join(map(str, r)) + r" \\" for r in df2.itertuples(index=False, name=None)]
    body = "\n".join(rows)

    # Build LaTeX without f-strings to avoid { } interpolation issues
    colspec = " ".join(["l"] * len(df2.columns))
    latex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{" + colspec + "}\n"
        "\\toprule\n"
        + header + "\n"
        "\\midrule\n"
        + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{" + str(caption) + "}\n"
        "\\label{" + str(label) + "}\n"
        "\\end{table}"
    )
    return latex


# ----------------------------- plotting -----------------------------

def fig_leaderboard_heatmap(df, outdir: Path):
    # Heatmap of min loss per (model, chunker)
    agg = reduce_min_loss_by(df, keys=('model_short','chunker'))
    pivot = agg.pivot(index='model_short', columns='chunker', values='loss_total')
    # Order by best loss
    order = pivot.min(axis=1).sort_values().index
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(1.2 + 0.6*pivot.shape[1], 0.6 + 0.45*pivot.shape[0]))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="mako_r", ax=ax, cbar_kws={'label':'Loss (lower is better)'})
    ax.set_title("Composite Loss by Embedding Model × Chunker (min over c,o)")
    ax.set_xlabel("Chunker")
    ax.set_ylabel("Model")
    savefig(fig, outdir, "fig_leaderboard_heatmap")

def fig_overlap_lines(df, outdir: Path, fixed_c=448, n_models=3):
    sub = df[df['token_size'] == fixed_c].copy()
    if sub.empty:
        return
    # Choose top n models by min loss at this c
    pick = (sub.groupby('model_short')['loss_total'].min()
              .sort_values().head(n_models).index)
    sub = sub[sub['model_short'].isin(pick)]
    fig, ax = plt.subplots(figsize=(8,5))
    sns.lineplot(data=sub, x='overlap', y='loss_total', hue='model_short', style='chunker', marker='o', ax=ax)
    ax.set_title(f"Loss vs Overlap (o) at c={fixed_c}")
    ax.set_xlabel("Overlap (tokens)")
    ax.set_ylabel("Composite Loss")
    ax.legend(title="Model / Chunker", bbox_to_anchor=(1.02, 1), loc='upper left')
    savefig(fig, outdir, f"fig_loss_vs_overlap_c{fixed_c}")

def fig_chunksize_lines(df, outdir: Path, fixed_o=0, n_models=5):
    sub = df[df['overlap'] == fixed_o].copy()
    if sub.empty:
        return
    pick = (sub.groupby('model_short')['loss_total'].min()
              .sort_values().head(n_models).index)
    sub = sub[sub['model_short'].isin(pick)]
    fig, ax = plt.subplots(figsize=(8,5))
    sns.lineplot(data=sub, x='token_size', y='loss_total', hue='model_short', style='chunker', marker='o', ax=ax)
    ax.set_title(f"Loss vs Chunk Size (c) at o={fixed_o}")
    ax.set_xlabel("Chunk Size (tokens)")
    ax.set_ylabel("Composite Loss")
    ax.legend(title="Model / Chunker", bbox_to_anchor=(1.02, 1), loc='upper left')
    savefig(fig, outdir, f"fig_loss_vs_chunksize_o{fixed_o}")

def fig_pareto(df, outdir: Path):
    sub = df.copy()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=sub, x='cost_norm', y='loss_total', hue='chunker', style='model_short', ax=ax, s=60)
    mask = pareto_frontier(sub, x='cost_norm', y='loss_total')
    frontier = sub.loc[mask].sort_values(['cost_norm','loss_total'])
    ax.plot(frontier['cost_norm'], frontier['loss_total'], color='black', linewidth=2, label='Pareto frontier')
    ax.set_title("Quality–Cost Trade-off (Loss vs Cost Norm)")
    ax.set_xlabel("Cost Norm (relative chunk count)")
    ax.set_ylabel("Composite Loss (lower is better)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    savefig(fig, outdir, "fig_pareto_loss_cost")

def fig_decay_vs_ndcg(df, outdir: Path):
    sub = df.copy()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=sub, x='ndcg_at_10', y='loss_decay', hue='chunker', style='model_short', ax=ax, s=60)
    ax.set_title("Tail vs Global: Precision Decay vs nDCG@10")
    ax.set_xlabel("nDCG@10 (higher is better)")
    ax.set_ylabel("Precision Decay (1 - P@10/P@1; lower is better)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    savefig(fig, outdir, "fig_decay_vs_ndcg")

def fig_robustness(df, outdir: Path):
    # Use best-per-(model,chunker) by min loss; plot NAUC with 0.5*std error bars
    cols = ['model_short','chunker','nauc_ndcg_at_10_max','nauc_ndcg_at_10_std','loss_rob']
    have = [c for c in cols if c in df.columns]
    if len(have) < 4:
        return
    best = (df.sort_values('loss_total')
              .groupby(['model_short','chunker'], as_index=False)
              .first())
    sub = best[cols].dropna(subset=['nauc_ndcg_at_10_max','nauc_ndcg_at_10_std'])
    if sub.empty:
        return
    sub = sub.sort_values('nauc_ndcg_at_10_max', ascending=False)
    fig, ax = plt.subplots(figsize=(10,6))
    y = np.arange(len(sub))
    ax.barh(
        y,
        sub['nauc_ndcg_at_10_max'],
        xerr=0.5 * sub['nauc_ndcg_at_10_std'].abs(),  # abs() ensures non-negative
        color="#4c72b0",
        alpha=0.8
    )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m} · {c}" for m,c in zip(sub['model_short'], sub['chunker'])])
    ax.invert_yaxis()
    ax.set_xlabel("NAUC nDCG@10 (higher is better) ± 0.5×std")
    ax.set_title("Robustness Across Test Items")
    savefig(fig, outdir, "fig_robustness_nauc")

def fig_radar_top3(df, outdir: Path):
    # Radar for top-3 configs over selected metrics; normalize to [0,1] with "higher=better"
    metrics = [
        ('ndcg_at_10', True),
        ('map_at_10', True),
        ('mrr_at_10', True),
        ('recall_at_20', True),
        ('precision_at_1', True),
        ('precision_at_10', True),
        ('loss_rob', False),   # lower better -> invert
        ('loss_total', False), # lower better -> invert
    ]
    sub = df.sort_values('loss_total').head(3).copy()
    if sub.empty:
        return
    # Normalize each metric to [0,1]
    norm_vals = []
    for name, higher_better in metrics:
        if name not in df.columns:
            sub[name] = np.nan
            norm_vals.append(np.ones(len(sub))*0.5)
            continue
        col = df[name].astype(float)
        lo, hi = np.nanmin(col), np.nanmax(col)
        rng = hi - lo if hi > lo else 1.0
        vals = (sub[name].astype(float) - lo) / rng
        if not higher_better:
            vals = 1.0 - vals
        norm_vals.append(vals.values)
    data = np.vstack(norm_vals)  # shape (M, K)
    labels = [f"{row.model_short} · {row.chunker} · c={int(row.token_size) if pd.notna(row.token_size) else '?'} o={int(row.overlap) if pd.notna(row.overlap) else '?'}"
              for _, row in sub.iterrows()]
    # Plot radar
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)
    for i in range(len(sub)):
        vals = data[:,i].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=labels[i])
        ax.fill(angles, vals, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m for m,_ in metrics])
    ax.set_yticks([0.2,0.4,0.6,0.8])
    ax.set_ylim(0,1)
    ax.set_title("Top-3 Configurations (Normalized Metrics)")
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    savefig(fig, outdir, "fig_radar_top3")

def fig_pipeline_schematic(outdir: Path):
    """Writes a Graphviz DOT file and tries to render PNG if graphviz is installed."""
    dot = dedent(r"""
    digraph G {
      rankdir=LR;
      node [shape=box, style="rounded,filled", color="#333333", fillcolor="#f5f5f5"];
      Corpus -> Chunker -> Embedder -> Index -> Retriever -> "Scorer/Evaluator";
      Chunker [label="Chunker\n(fixed/recursive/semantic/hybrid)\n(c, o)"];
      Embedder [label="Embedding Model"];
      Index [label="Vector Index (FAISS/ANN)"];
      Retriever [label="Top-k Retrieval"];
      "Scorer/Evaluator" [label="Metrics:\nnDCG@10, MAP@10, MRR@10,\nRecall@20, P@1, P@10,\nNAUC + Composite Loss"];
    }
    """).strip()
    outdir.mkdir(parents=True, exist_ok=True)
    dot_path = outdir / "fig_pipeline.dot"
    dot_path.write_text(dot, encoding="utf-8")
    # Try to render if graphviz present
    try:
        import subprocess
        subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(outdir / "fig_pipeline.png")], check=True)
        subprocess.run(["dot", "-Tpdf", str(dot_path), "-o", str(outdir / "fig_pipeline.pdf")], check=True)
    except Exception:
        # dot not installed — leave DOT file for later rendering
        pass

# ----------------------------- tables -----------------------------

def table_leaderboard(df, outdir: Path, topk=20):
    cols = [
        'model_short','chunker','token_size','overlap','loss_total',
        'ndcg_at_10','map_at_10','mrr_at_10','recall_at_20',
        'precision_at_1','precision_at_10','loss_decay','loss_rob','cost_norm'
    ]
    sub = topk_configs(df, topk)[cols].copy()
    sub.to_csv(outdir / "table_leaderboard_topk.csv", index=False)
    tex = latex_table(
        sub.head(min(10, len(sub))),
        caption="Top configurations by composite loss (lower is better). Best per metric in bold.",
        label="tab:leaderboard",
        floatfmt="{:.3f}",
        bold_cols=('loss_total','ndcg_at_10','map_at_10','mrr_at_10','recall_at_20','precision_at_1','precision_at_10')
    )
    (outdir / "table_leaderboard_top10.tex").write_text(tex, encoding="utf-8")

def table_marginal_means(df, outdir: Path):
    by_model = (df.groupby('model_short')['loss_total']
                  .mean().reset_index().sort_values('loss_total'))
    by_chunker = (df.groupby('chunker')['loss_total']
                    .mean().reset_index().sort_values('loss_total'))
    by_model.to_csv(outdir / "table_marginal_means_by_model.csv", index=False)
    by_chunker.to_csv(outdir / "table_marginal_means_by_chunker.csv", index=False)

    tex1 = latex_table(by_model, caption="Marginal mean loss by model (lower is better).",
                       label="tab:marginal_model", floatfmt="{:.3f}", bold_cols=('loss_total',))
    tex2 = latex_table(by_chunker, caption="Marginal mean loss by chunker (lower is better).",
                       label="tab:marginal_chunker", floatfmt="{:.3f}", bold_cols=('loss_total',))
    (outdir / "table_marginal_means_by_model.tex").write_text(tex1, encoding="utf-8")
    (outdir / "table_marginal_means_by_chunker.tex").write_text(tex2, encoding="utf-8")

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="CSV produced by your evaluator script")
    ap.add_argument("--out", type=Path, default=Path("figs"), help="Output directory for figures/tables")
    ap.add_argument("--topk", type=int, default=20, help="Top-K rows for leaderboard")
    ap.add_argument("--fixed_c", type=int, default=448, help="Chunk size for overlap line plot")
    ap.add_argument("--n_models_overlap", type=int, default=3, help="#models to show in overlap plot")
    ap.add_argument("--n_models_c", type=int, default=5, help="#models to show in chunk-size sweep")
    args = ap.parse_args()

    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = ensure_cols(df)

    # Figures
    fig_pipeline_schematic(outdir)
    fig_leaderboard_heatmap(df, outdir)
    fig_overlap_lines(df, outdir, fixed_c=args.fixed_c, n_models=args.n_models_overlap)
    fig_chunksize_lines(df, outdir, fixed_o=0, n_models=args.n_models_c)
    fig_pareto(df, outdir)
    fig_decay_vs_ndcg(df, outdir)
    fig_robustness(df, outdir)
    fig_radar_top3(df, outdir)

    # Tables
    table_leaderboard(df, outdir, topk=args.topk)
    table_marginal_means(df, outdir)

    print(f"Done. Figures and tables written to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
