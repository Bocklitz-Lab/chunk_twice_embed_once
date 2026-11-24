# Paper notes: which split Phase 1 was evaluated on

For the authors (Reviewer 2, Methods: "the dataset splits actually evaluated"). Checked on the published no-prompt Phase 1 files
(`temp_files/artifacts_old/embedding_models_screening/validation/`, 42 model folders, and `passed_per_dataset.csv`) and on the dataset files of
`BASF-AI/ChemNQRetrieval` and `BASF-AI/ChemHotpotQARetrieval` (queries counted in the qrels files).

## What was evaluated

| task | split | queries | passages | used for Table 1 and the shortlist |
|---|---|---|---|---|
| ChemNQRetrieval | test | 27 | 22,933 | yes (the only split) |
| ChemHotpotQARetrieval | train | 180 | 10,069 | no |
| ChemHotpotQARetrieval | dev | 8 | 10,069 | no |
| ChemHotpotQARetrieval | test | 18 | 10,069 | **yes** |

MTEB scores every split of a task and writes all of them into `<Task>.json` (HotpotQA: `train`, `dev`, `test`; the corpus is encoded once per split, so the
`evaluation_time` of HotpotQA covers three passes). Stage 2 used the first of `test`, `validation`, `train` that is present. HotpotQA has no `validation` split
(it is called `dev`), and `dev` was never in that list; the order picked `test` for both tasks.

## Which split reproduces Table 1 (Geom@10, three decimals)

| task | model | Table 1 | train | dev | **test** |
|---|---|---|---|---|---|
| ChemHotpotQARetrieval | nomic-embed-text-v1 | 0.890 | 0.914 | 0.875 | **0.890** |
| ChemHotpotQARetrieval | bge-large-en-v1.5 | 0.873 | 0.932 | 0.953 | **0.873** |
| ChemHotpotQARetrieval | nomic-embed-text-v1.5 | 0.859 | 0.903 | 0.875 | **0.859** |
| ChemHotpotQARetrieval | bge-base-en-v1.5 | 0.855 | 0.913 | 0.955 | **0.855** |
| ChemHotpotQARetrieval | e5-small | 0.840 | 0.782 | 0.803 | **0.840** |
| ChemNQRetrieval | e5-large | 0.737 | n/a | n/a | **0.737** |
| ChemNQRetrieval | e5-base | 0.730 | n/a | n/a | **0.730** |
| ChemNQRetrieval | nomic-embed-text-v1.5 | 0.722 | n/a | n/a | **0.722** |
| ChemNQRetrieval | multilingual-e5-base | 0.713 | n/a | n/a | **0.713** |
| ChemNQRetrieval | multilingual-e5-small | 0.708 | n/a | n/a | **0.708** |

All ten rows match on `test` (and only on `test`). Re-running the stage 2 selection on the published files with HotpotQA `test` gives exactly the published 13 models
(`split_used = test` in `passed_per_dataset.csv`); exactly three models are in both top-eight sets (multilingual-e5-small, nomic-embed-text-v1, nomic-embed-text-v1.5), as the text says.

## Sensitivity: the shortlist under each HotpotQA split (NQ always `test`; stage 2 logic, top 8 of 42 per task)

| HotpotQA split | queries | HotpotQA ∩ NQ top-8 | union | overlap with the published 13 | not in the published 13 | published models lost |
|---|---|---|---|---|---|---|
| train | 180 | 2 | 14 | 11 | bge-base-en, bge-large-en, bge-small-en | bge-m3, e5-small |
| dev | 8 | 1 | 15 | 10 | bge-base-en, bge-large-en, e5-base-v2, e5-small-v2, gtr-t5-xl | bge-m3, bge-small-en-v1.5, e5-small |
| **test (published)** | 18 | 3 | 13 | 13 | none | none |

With 18 or 8 queries a top-8 cut is fragile; `make report` now writes `table_phase1_ci.csv` (query-level 95 % intervals and the share of resamples in which
each model is in the top 20 % of its task and of the union) for the prompted rerun. These are query-level intervals (Phase 1 queries have no paper grouping).

## What the code does now

* One key, `params.phase1.eval_split` (default `{ChemNQRetrieval: test, ChemHotpotQARetrieval: test}`), is read by stage 2, `table_external` of `make report`,
  `tools/phase_transfer.py`, `stages/run_bm25_mteb.py`, the Phase 1 truncation reports and the efficiency Phase 1 table. There is no fallback order; a result file without the
  configured split stops the run. `params.phase1.n_queries` holds the query counts above.
* Phase 1 outputs carry `split` and `n_queries` (shortlist file, `table_external.csv` as `<task>__split` / `<task>__n_queries`, the phase-transfer tables, `table_phase1_ci.csv`,
  `report_manifest.json`).
* Stage 1 saves per-query results (`params.phase1.save_per_query`); MTEB's predictions are those of the last split evaluated (HotpotQA: `test`), and the code checks them against
  the stored scores. The `<Task>.json` scores are identical with and without saving (checked on all-MiniLM-L6-v2).

## What the paper must say

* Supplementary Section 1.3 and Algorithm S1: replace "the first available split in the order test, validation, train" by "the `test` split of each task: ChemNQRetrieval has only
  `test` (27 queries); ChemHotpotQARetrieval has `train` (180), `dev` (8) and `test` (18), and `test` (18 queries) was used". Say that all three HotpotQA splits share the same 10,069 passages
  and that MTEB scores all of them, but only `test` enters Table 1 and the shortlist. HotpotQA has no split called `validation`.
* Table 1: state the query counts (27 and 18) in the caption.
* Report the sensitivity table (or a sentence: with `train` the union has 14 models and 11 of the 13; with `dev` it has 15 and 10), and the intervals / selection shares from `table_phase1_ci.csv`
  once the prompted Phase 1 is run.

## Decisions for the authors

* Keep `test` for the prompted rerun (the default, comparable with the paper), or use `train` (180 queries, more stable, but 11 of the 13 models) or the union of splits. The code does not change it on its own.
* The published folder holds 42 models, the rerun 41 (PubMedBERT-abstract is not in the rerun list); the top-8 cut is the same for 41 and 42 models.
