# Plan for the full rerun (Reviewer 2 revision)

All commands run from the repository root with the repository's environment (`myenv`), in this order. GPU times are **estimates**: the pilot measured
85-120 s per dense run for e5-large-v2 and bge-m3 on one RTX 3090 (encoding is about 23,000 tokens/s for a large model), BM25 26-54 s, a hybrid 2 s,
reranking e5-large-v2 at depth 50 193 s (size 256); the times of the base, small and instructor-xl models are scaled from their size, not measured. The published
Phase 1 took 2.1 h of `evaluation_time` for 42 models x 2 tasks. Two GPUs halve the dense steps if the models are split (see step 7).

Nothing in this plan has been run. The code was tested on the pilot data (`artifacts/pilot_v2`) and on synthetic data only.

## 0. Before anything

```bash
git checkout reviewer2-code                  # after review; this branch is not merged or pushed
export PYTHONHASHSEED=0 HF_HUB_OFFLINE=1     # offline: all models and the external datasets must be in the Hugging Face cache (ChemQuests is gated: token or cache)
```
Decide before starting (they change what the rerun does): the Phase 1 prompts question (this plan runs Phase 1 **with** prompts), the latency subset
(`params.efficiency` in `configs/main_config.yaml`: fixed-token 128/0, 256/0, 512/0 and hierarchical-section 512/0, 200 queries, CPU search), the number of
resamples (`params.report.resamples: 10000`), the rerank model (`params.rerank.models`: e5-large-v2 only; change it, and the ablation entry, if the new
Phase 1 puts another model first).

## 1. Restore the published Phase 1 results and shortlist (read-only reference)

The repository does not create these files. They are kept untracked in `temp_files/artifacts_old/`:
```bash
cp -a temp_files/artifacts_old/embedding_models_screening artifacts/
ls artifacts/embedding_models_screening     # validation/ (published, no prompts), passed_per_dataset.csv, passed_models_union.csv
```
These files are never overwritten: Phase 1 with prompts writes to `validation_prompts/` and stage 2 to `passed_per_dataset__validation_prompts.csv`
(one key, `params.paths.phase1_results_subdir`). They are the input of `tools/phase_transfer.py` (published run) and of the comparison of the two shortlists.

## 2. Phase 1 with prompts (about 3 h on one GPU)

The split that Phase 1 is scored on is `params.phase1.eval_split` (default `test` for both tasks = the published Table 1 and shortlist; see `docs/PAPER_NOTES_phase1_split.md`). Do not change it before the run unless the authors decide so: stage 2 follows it. Stage 1 also writes `per_query_<Task>.csv` and `run_top100_<Task>.trec.gz` (`params.phase1.save_per_query`); `make report` then writes `table_phase1_ci.csv`.

```bash
make stage0                  # regenerates configs/stages/stage1-3.yaml from the main config (they are tracked; tests/test_phase1_paths.py checks them)
make stage1                  # 41 models x 2 tasks, prompts on, --efficiency on; writes artifacts/embedding_models_screening/validation_prompts/
make stage2                  # top 20% per task -> passed_per_dataset__validation_prompts.csv (the model list of Phase 2)
```
* `make stage1` refuses a folder that already holds results; to resume an interrupted run (finished tasks are kept by MTEB): `make stage1 STAGE1_ARGS="--efficiency --force"`.
* Each model also writes `truncation_report_<Task>.json` and `run_manifest.json` (pooling, dimension, maximum length, tokenizer, CPU, RAM).
* Compare with the published shortlist: `diff <(cut -d, -f2 artifacts/embedding_models_screening/passed_per_dataset.csv | sort -u) <(cut -d, -f2 artifacts/embedding_models_screening/passed_per_dataset__validation_prompts.csv | sort -u)`.
  If the shortlist changes, check `params.rerank.models`, the roles in `params.transfer` and the text before going on.
* Published-run truncation (no prompts, CPU, a few minutes): `python -m tools.phase1_truncation --out-dir results/v2-revision/published --prompts none`.
* Model table for Table S1 (CPU, seconds): `python -m tools.model_properties --out-dir results/v2-revision --shortlist artifacts/embedding_models_screening/passed_per_dataset__validation_prompts.csv`.

## 3. Prepare: chunk and label every configuration once (CPU, about 2-3 h)

```bash
make prepare                 # 140 configurations; ChemQuests is downloaded (stage 3) if artifacts/chemquests_hf_db is missing
make stage3                  # also needed once if artifacts/chemquests_hf_db exists from an earlier run: downloads qa/question_metadata.jsonl (needs an HF token)
```
The question labels (`question_metadata.jsonl`: official question type and subfields) are NOT in the pinned release `5ab7dc67...`; stage 3 fetches them from their own pinned revision
`790d2bc6829ccd7e1e480a6a4610149650b79e9f` (a dataset commit that differs from the release only by this file) and stops if the SHA-256 is not `4e1f83fb...`. Existing files are only verified, so qa.jsonl and
full_text.jsonl stay byte-identical. Checks to read after the download: the loader (`python -c "from tools import analysis_data as AD; m = AD.load_question_metadata('artifacts/chemquests_hf_db/qa.jsonl'); print(m.type_counts(), m.subfield_counts())"`)
raises unless there are 952 rows, `q%04d` equals the row index of `qa.jsonl`, `corpus_id` and the question text equal `qa.jsonl`, the four types and exactly 17 subfield names are present. The revision and the hash are recorded in
`manifest.json` of every prepared folder, in `run_manifest.json` and in `report_manifest.json`.
The semantic chunker's first configuration is cold (about 8 min); the others are cached. Check afterwards: 140 folders in `artifacts/prepared/`, each with a `manifest.json`.

## 4. Manual validation of the labels (start now; takes human time, no GPU)

```bash
python -m tools.validation_sample --qa artifacts/chemquests_hf_db/qa.jsonl --prepared-root artifacts/prepared --out-dir results/validation
#   -> validation_sheet_annotator_1.csv/.md and validation_sheet_annotator_2.csv/.md, the same 100 questions in the same order
```
Each annotator fills `span_correct`, `span_sufficient` and `chunks_sufficient_<configuration>` with `yes`, `partial` or `no` in the CSV (the Markdown file is for reading) and
works alone. When both are done:
```bash
python -m tools.kappa --sheets results/validation/validation_sheet_annotator_1.csv results/validation/validation_sheet_annotator_2.csv --out-dir results/validation
#   -> kappa.csv (agreement, kappa, weighted kappa with paper-level intervals), kappa_disagreements.csv (discuss these items)
```
Keep the filled sheets and `validation_sample_summary.json` (seed, questions per type and subfield, hashes of the prepared folders) with the paper material.
The sample is stratified by the official question type and every one of the 17 subfields has at least 2 questions in it.

Second hand check, also human time: the two automatic question features (`auto_has_number`, `auto_has_formula`; rule-based, not annotations):
```bash
python -m tools.question_features_check draw --qa artifacts/chemquests_hf_db/qa.jsonl --out-dir results/validation    # 50 questions, manual_* columns empty
python -m tools.question_features_check score --sheet results/validation/question_features_check.csv --qa artifacts/chemquests_hf_db/qa.jsonl
```
Do not quote any breakdown by these features as final before the sheet is filled in and precision / recall are in the paper.

## 5. Main run: every shortlisted model and BM25 on all 140 configurations

```bash
make eval                    # efficiency + latency on (EFFICIENCY_ARGS), prompts on, per-model batch sizes, then hybrids, rerank and the report
```
Expected: about 55-75 h on one RTX 3090 (dense encoding about 40 h, retrieval/metrics/dump 2.5 h, efficiency probes and latency 3-5 h, BM25 1.5 h on the CPU,
1,820 hybrids 1 h, 140 reranks of e5-large-v2 7.5 h, report about 1 h). Finished runs are skipped, so an interrupted `make eval` can simply be started again.
Peak GPU memory, throughput and latency cannot be recorded afterwards: do not start the main run with `EFFICIENCY_ARGS=` empty.

Two GPUs (dense steps in about half the time): run the dense models in two shells with disjoint model lists and without the steps that need all runs, then finish once:
```bash
# shell 1
CUDA_VISIBLE_DEVICES=0 make eval RERUN_ARGS="--models <name>@<rev> ... --no-bm25 --no-hybrid --no-rerank --no-report"
# shell 2 (the other models; CPU job for BM25: --models bm25@rank_bm25)
CUDA_VISIBLE_DEVICES=1 make eval RERUN_ARGS="--models <name>@<rev> ... --no-bm25 --no-hybrid --no-rerank --no-report"
# then, once
make hybrid && make rerank && make report
```
(the model names and revisions are in `artifacts/embedding_models_screening/passed_per_dataset__validation_prompts.csv`).

Batch sizes: each model uses its Phase 1 batch size, else 32. Compared with the earlier Phase 2 configuration (32 for all) these shortlisted models change: bge-base-en-v1.5,
bge-small-en-v1.5, multilingual-e5-base, multilingual-e5-small, e5-base, e5-small, nomic-embed-text-v1, nomic-embed-text-v1.5 (64) and instructor-xl (16). Unchanged: bge-large-en-v1.5,
bge-m3, e5-large, e5-large-v2 (32). The batch size of every run is in `run_info.json` and the manifest.

## 6. Transfer runs: models that Phase 1 did not shortlist (about 2.5-3 GPU hours)

```bash
make transfer                # params.transfer: 14 non-shortlisted + 3 shortlisted reference models + bert-base-uncased, configurations C1-C5 and C6 -> artifacts/transfer/
make transfer TRANSFER_ARGS="--dry-run"      # prints the plan and checks params.transfer first
```
Needs `make prepare` of step 3 (the transfer run reads the same prepared chunks and labels and the same ChemQuests files). `--no-extras` skips C6 and bert-base-uncased, `--no-references` skips the
three models that are also in the main run. The report of this root goes to `results/v2-revision/transfer/`.

## 7. Report, then the analysis tools (CPU)

```bash
make report                  # main root -> results/v2-revision/ (tracked small CSVs; large files stay in artifacts/report/): about 45-60 min at 10,000 resamples for ~3,900 runs (extrapolated from the pilot: 18 s for 20 runs)
make report ARTIFACTS_ROOT=artifacts/transfer REPORT_DIR=results/v2-revision/transfer       # (already written by make transfer)
python -m tools.efficiency_analysis --efficiency artifacts/tests/efficiency_phase2.csv --scores results/v2-revision/runs_overview.csv --out results/efficiency --phase1 artifacts/embedding_models_screening/validation_prompts/efficiency_phase1.csv
make analysis                # token budget, question type / automatic features / subfield, variance decomposition, winner stability, Phase 1 -> Phase 2 transfer, examples, fragmentation -> results/v2-revision/analysis/
```
`make analysis ANALYSIS_ARGS="--only token_budget examples"` runs some tools; `ANALYSIS_ARGS="--resamples 1000"` is a quick test. Time: `token_budget` about 0.5 s per run (about 30 min), the others minutes.
`params.analysis.enabled_in_report: true` chains `make analysis` at the end of `make report` (main root only). `phase_transfer` needs `make report` of both roots first (it reads their `runs_overview.csv`)
and is skipped with a message if the Phase 1 folder or the shortlist is missing. For the published (no prompts) Phase 1 add a second run:
`python -m tools.phase_transfer --phase1-root artifacts/embedding_models_screening/validation --shortlist artifacts/embedding_models_screening/passed_per_dataset.csv --runs-overview artifacts/report/runs_overview.csv artifacts/transfer/report/runs_overview.csv --out-dir results/v2-revision/analysis_published`.

## 8. What comes out, per Reviewer 2 item

| item | output |
|---|---|
| 3 label statistics | `table_label_stats.csv` (report) |
| 4 cover-level sensitivity | `table_cover_sensitivity.csv` (report) |
| 6 manual validation | `results/validation/` (sheets, `kappa.csv`) |
| 7 token budget | `analysis/token_budget.csv` |
| 9, 10 pooling, dimension, maximum length | `run_manifest.json: model_properties` of every run, `table_model_properties.csv` |
| 11 truncation | `table_truncation.csv` (Phase 2), `table_truncation_phase1.csv` (Phase 1) |
| 14, 16 transfer, Phase 1 -> Phase 2 | `artifacts/transfer/`, `analysis/phase_transfer_*.csv`, `fig_phase_transfer.png` |
| 17, 18 statistics | `stats_ci.csv`, `stats_compare.csv` (10,000 resamples, families) |
| 19, 20 winner stability, variance | `analysis/winner_*.csv`, `analysis/variance_*.csv` |
| 23, 24 subfield, question type | `analysis/breakdown_question_type*.csv`, `breakdown_subfield*.csv` (official labels; multi-label and fractional), `breakdown_question_features*.csv` (automatic features), `breakdown_groups.csv`, `question_features.csv` |
| 25, 26 examples, fragmentation | `analysis/examples.csv/.md`, `analysis/fragmentation.csv` |
| 27 hardware | `run_manifest.json: hardware` (CPU, RAM, GPU, threads, seeds) of dense, BM25, hybrid and rerank runs |
| 29 efficiency | `artifacts/tests/efficiency_phase2.csv`, `efficiency_phase1.csv`, `results/efficiency/` |

## 9. Checks before the numbers are used

* `make check-identical` (also part of the report): `identical_check.csv` must say OK for every configuration.
* `git status` is clean before `make eval` (the manifests record the commit and whether the tree was dirty).
* The top-level files of `results/v2-revision/` are now real results of this run (the earlier ones were test output and were removed).
* Do not tag or push before the authors have looked at the numbers.
