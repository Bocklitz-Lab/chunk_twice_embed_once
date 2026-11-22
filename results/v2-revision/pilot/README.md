# Summary tables (written by `tools/make_report.py`; do not edit by hand)

Created 2026-11-22T13:42:09Z from commit `7d814672179c1e0be4069a39395c941421b939f1` (dirty working tree). Artifacts root: `artifacts/pilot_v2`.
Bootstrap: 1000 resamples, seed 20261121, query level and paper level (the 952 questions come from 149 papers). Labels: span labels, cover level c=0.90 (primary); the lexical labels are reported next to them.

| file | content |
|---|---|
| `table_main.csv` | Main table: one row per configuration and model/hybrid: Geom@10, nDCG@10, Recall@10 under the span labels with 95% paper-level bootstrap CI of Geom@10, Geom@10 under the lexical labels, span coverage@5, % chunks truncated, number of chunks. |
| `runs_overview.csv` | One row per run: metrics under both label sets, span coverage, truncation, timing, prompts, chunking diagnostics, git commit. |
| `stats_ci.csv` | Bootstrap 95% CIs (1000 resamples) of Geom@10, nDCG@10, Recall@10 per run: level=query and level=paper. |
| `stats_compare.csv` | Paired comparisons (dense vs BM25, hybrid vs its dense run and vs BM25): difference, CI, bootstrap and Wilcoxon p, Holm-adjusted. |
| `stats_label_agreement.csv` | Lexical vs span labels per configuration: % questions with the same top relevant chunk, Jaccard of the relevant sets. |
| `stats_rankings.csv` | Kendall tau-b between the rankings of runs by the lexical and by the span labels (all runs, within model, within configuration). |
| `coverage_diagnostics.csv` | Per configuration: number of chunks, chunk length with the reference tokenizer, % evidence spans inside one chunk, relevant chunks per question. |
| `table_truncation.csv` | Per run: model max sequence length, number and % of chunks truncated by the model's own tokenizer. |
| `overlap_distribution.json` | Distribution of the question/evidence-span word overlap over the 952 questions. |
| `overlap_questions.csv` | Per question: overlap share (question content words found in the evidence span), span length. |
| `overlap_advantage.csv` | BM25 minus dense per question metric by overlap bin, with paper-level bootstrap CI, Spearman rho and slope. |
| `overlap_rank_by_bin.csv` | Rank of the evidence chunk (% rank 1, % in top 10, % outside top 100, median) by overlap bin, per run. |
| `identical_check.csv` | Were corpus, queries, chunks and qrels byte-identical for all models of a configuration (and equal to the prepared folder)? |

Large files (rankings, chunks, per-question rank tables) stay in `artifacts/`.
`report_manifest.json` lists package versions and SHA-256 of every file.
