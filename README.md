# Chunk Twice, Embed Once

Code, configurations, data, and result artifacts for:

> **Chunk Twice, Embed Once: A Systematic Study of Segmentation and Representation Trade-offs in Chemistry-Aware Retrieval-Augmented Generation**  
> Mahmoud Amiri and Thomas Bocklitz (2025)

## Study design

Phase 1 evaluates 41 distinct embedding models (the published run listed 42 names; `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` is the old name of `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`, same commit, and is no longer listed) with complete results on the two external retrieval tasks `ChemNQRetrieval` and `ChemHotpotQARetrieval`. The union of the top 20% from each task gives the verified 13-model Phase 2 shortlist.

Phase 2 evaluates those 13 models with five chunking strategies, seven nominal chunk sizes (`128`, `192`, `256`, `320`, `384`, `448`, and `512` tokens), and five nominal overlaps (`0`, `16`, `32`, `48`, and `64` tokens). The grid constraints leave 28 valid size-overlap combinations, or 140 configurations per model and 1,820 canonical configurations in total.

## Repository layout

```text
configs/                 Pipeline and stage configurations
pipeline_lib/            Shared task, chunking, and evaluation code
stages/                  Phase 1 and Phase 2 pipeline stages
artifacts/
  chemquests_hf_db/      ChemQuests metadata, full text, and QA input
  embedding_models_screening/
                          Phase 1 results and shortlist files
  tests/                  Historical Phase 2 run directories and summaries
  canonical_phase2/      Canonical 13-model, 1,820-run analysis package
notebooks/                Analysis and figure-generation notebooks
Makefile                  Supported pipeline targets
test.mk                   Stage 4-6 execution targets
requirements.txt          Python dependencies
```

Each generated local retrieval task contains:

- `corpus.jsonl`: configuration-specific document chunks;
- `queries.jsonl`: ChemQuests questions;
- `qrels.jsonl`: generated query-relevance records.

## Installation

Python 3.10 was used for the audited experiments.

```bash
git clone https://github.com/Bocklitz-Lab/chunk_twice_embed_once.git
cd chunk_twice_embed_once
python3.10 -m venv myenv
source myenv/bin/activate
python -m pip install -r requirements.txt
```

The canonical analysis was verified with the following core environment: Python 3.10.16, MTEB 1.36.8, Sentence Transformers 3.0.0, Datasets 2.19.0, NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.7.2, pandas 2.3.2, PyTorch 2.8.0, Transformers 4.56.1, rank-bm25 0.2.2, RapidFuzz 3.14.1, tiktoken 0.11.0, LangChain 0.3.27, and regex 2025.9.1. The historical dependency specification was not a complete lockfile, so exact bit-for-bit environment reproduction is not claimed.

## Pipeline targets

The Makefile provides these targets:

```bash
make stage0      # split the main configuration into stage configurations
make stage1      # run Phase 1 model screening
make stage2      # select the top 20% per external task and write shortlist files
make stage3      # download/export ChemQuests inputs
make stage4-6    # chunk documents, build local tasks, and evaluate Phase 2
make stage7      # run the configured post-processing stage
make show-configs
```

The full screening and Phase 2 grid require model downloads and substantial compute. Existing raw outputs are retained under `artifacts/` so analyses can be inspected without rerunning every model.

## Reviewer-revision workflow (consistent chunks, prompts, per-query results)

The legacy `make stage4-6` path chunks every configuration once per model with that model's own tokenizer. The
revision workflow separates model effects from segmentation effects:

```bash
make prepare           # chunk + label each of the 140 configurations ONCE (reference tokenizer, fixed breakpoint model)
make eval              # every model (+ BM25) on the prepared folders; finished runs are skipped; efficiency measures on (EFFICIENCY_ARGS)
make rerun             # prepare, then eval
make pilot             # 2 models + BM25 + their RRF hybrids, fixed-token and hierarchical-section, sizes 128/512, overlap 0 (artifacts/pilot_v2)
make check-identical TESTS_ROOT=artifacts/tests   # corpus / qrels byte-identical across models?
make report            # statistics, diagnostics, overlap diagnostic, paper tables -> results/v2-revision/ (tracked); also run by eval/rerun/pilot
make transfer          # models that Phase 1 did not shortlist on C1-C5 (+C6) in the new pipeline -> artifacts/transfer (params.transfer)
make analysis          # token budget, subfield / question type, variance decomposition, winner stability, Phase 1 -> 2 transfer, examples, fragmentation
```

- `params.chunking_tokenizer` (bert-base-uncased, pinned) counts all chunk sizes; each model still embeds with its own
  tokenizer. `chunking_tokenizer=model` restores the legacy per-model chunking.
- `params.semantic_breakpoint_model` (all-MiniLM-L6-v2, pinned) finds the semantic breakpoints for all models.
- `params.use_prompts` / `params.model_prompts`: query and document prompts per model (checked against the model cards),
  applied at encoding time only. `--param use_prompts=false` reproduces the published behaviour (no prompts).
- Prepared folders (`artifacts/prepared/<strategy>_c<L>_o<O>__<key>/`) hold the chunks, `corpus.jsonl`, `queries.jsonl`,
  `qrels_lexical.jsonl`, `qrels_span_c<c>.jsonl`, the overlap table, coverage diagnostics and a manifest with file hashes.
- Every run writes `run_top100.trec.gz`, `per_query_metrics*.csv`, `span_coverage.csv`, `truncation_report.json`,
  `metrics_summary.json` (with a cross-check against MTEB's own numbers) and `run_manifest.json` (config, git commit,
  package versions, GPU, prompts, seeds, data hashes, timing breakdown).
- Tools: `tools/metrics.py` (all metrics from a saved ranking for any qrels file), `tools/stats.py` (bootstrap CIs at query
  and paper level, paired tests with Holm correction, lexical-vs-span agreement), `tools/coverage_diagnostics.py`,
  `tools/compare_runs.py` (repeat-run check).
- `stages/run_bm25.py` is the BM25 baseline (`bm25` as a model name).
- `params.embed_chunk_title` (default `false`): the corpus title is empty, so MTEB embeds the chunk text only. `true` is the
  old behaviour (`"<doc> - Chunk <n> <text>"` was embedded).
- Stage 3 builds `full_text.jsonl` from `full_text/<id>.txt` + `metadata.jsonl` of the pinned release (it has no
  `full_text.jsonl`), byte-identical to the file of the published runs, and checks all 952 evidence spans.
- `tools/hybrid.py`: reciprocal rank fusion (k=60) of saved dense and BM25 rankings, same outputs as a normal run, no
  embedding (`params.hybrid`, run automatically by `make eval` / `make pilot`).
- `tools/rerank.py` (`make rerank`, `params.rerank`): cross-encoder reranking (`BAAI/bge-reranker-v2-m3`, pinned commit,
  `sentence_transformers.CrossEncoder`) of the saved top-N of a dense run; chunks below rank N keep their order. Post-hoc: no new
  chunking, labels or embeddings. Primary depth `top_n` (default 50) for the dense models in `params.rerank.models`, plus the
  `ablation` entries (model / chunker / size / overlap / extra depths; nothing is tied to a size). The study reranks
  e5-large-v2 only (`params.rerank.models`). `stages/rerun.py` aborts before any run starts if an ablation entry does not fit the
  grid of the invocation (`--chunkers/--sizes/--overlaps`), so update `params.rerank.ablation` together with the grid. Resumable.
- `stages/run_bm25_mteb.py` (`make external-bm25`, `params.external`): BM25 on `ChemNQRetrieval` / `ChemHotpotQARetrieval`, written
  in the layout of a Phase 1 model (`model_meta.json` + `<Task>.json`) to `artifacts/embedding_models_screening/baselines/`.
  It is deliberately NOT written to the folder stage 2 screens (and stage 2 skips `bm25*` unless `include_baselines: true`), so BM25
  can never change the Phase 2 shortlist. Dense models are not re-run on the external tasks.
- `make hybrid` / `make rerank` run only that step on finished runs (`python -m stages.rerun hybrid|rerank`, same
  `--models/--chunkers/--sizes/--overlaps` flags); `--no-hybrid` / `--no-rerank` skip them in `eval` / `all`.
- The report (`make report`, last step of `eval`) writes one table per label type, `table_retrievers_span.{csv,md}` and
  `table_retrievers_lexical.{csv,md}` (rows: dense models, bm25, dense+bm25 hybrids, dense+rerank@N; columns: chunker x size x
  overlap read from the runs; Geom@10, nDCG@10, Recall@10, evaluation_time), `table_rerank_ablation.{csv,md}` and
  `table_external.{csv,md}` (BM25 next to the Phase 1 dense results).
- `tools/lexical_overlap.py`: question/evidence-span word overlap, BM25 rank of the evidence chunk, BM25 advantage over a
  dense model by overlap bin.
- All 41 Phase 1 models are pinned to commit hashes in `stage1.models`; `rerun` replaces a moving `main` by the pin.
- `params.instructor_mask_instruction` (default `true`; only acts with `use_prompts: true`): INSTRUCTOR masking (instruction tokens left out of the mean
  pooling) for instructor-xl, implemented without the `InstructorEmbedding` package.

### Reviewer 2 revision: what was added (branch `reviewer2-code`)

The exact order of the full rerun with times is in [`docs/RERUN_PLAN.md`](docs/RERUN_PLAN.md). Every default below is read from `configs/main_config.yaml`.

**Phase 1 and the model list**

- `params.paths.phase1_results_subdir` (default `validation_prompts`) is the ONE key for the folder that stage 1 writes and stage 2 reads; the stage 2 shortlist files carry its name
  (`passed_per_dataset__<subdir>.csv`), so the published Phase 1 (`validation/`) and its shortlist are never overwritten. `make stage1` refuses a folder that already holds result files
  unless `--force` is given (`make stage1 STAGE1_ARGS="--efficiency --force"` resumes an interrupted run). `configs/stages/stage1-3.yaml` are generated by `make stage0` and tracked
  (`tests/test_phase1_paths.py` fails if they are stale).
- `MODELS_CSV` (default: the stage 2 shortlist of that folder) is checked before `make eval`, `rerun`, `hybrid` and `rerank` start; the message says how to restore the published one.
- 41 models: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` is the old name of `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` (same commit) and is listed once.
- `tools/model_properties.py`: one row per Phase 1 model (pooling, library-default pooling, dimension, maximum length, tokenizer, revision, prompt) from the pinned local snapshots, offline.
- `tools/phase1_truncation.py`: Phase 1 truncation (share of passages and queries over the model's maximum length) for existing runs from the cached datasets; stage 1 now writes
  `truncation_report_<Task>.json` per model, collected by `make report` into `table_truncation_phase1.csv`.
- Every run manifest has `model_properties` (modules, pooling mode, library default, dimension, maximum length, tokenizer) and `hardware` with CPU model, cores, RAM, torch threads and the GPU
  driver; hybrid and rerank manifests have the hardware, seeds and device blocks too.

**Efficiency (on by default)**: `make stage1` passes `STAGE1_ARGS = --efficiency`, `make eval` / `rerun` / `transfer` pass `EFFICIENCY_ARGS = --efficiency --latency` (set them empty to switch off).
`params.efficiency` holds the latency settings: 200 queries, 10 warm-up, search on the CPU, all models, exactly these configurations: fixed-token 128/0, 256/0, 512/0 and hierarchical-section 512/0
(`--latency-cells CHUNKER:SIZE:OVERLAP`; any `--latency-*` flag on the command line replaces the configured subset). The reference tokenizer of the token statistics is `params.chunking_tokenizer`.

**Batch size**: each Phase 2 model uses its `stage1.models` batch size, else `params.model.batch_size` (32); `--param model.batch_size=N` forces one value. It is recorded in `run_info.json`
(`batch_size`, `batch_size_source`) and the manifest.

**Transfer runs** (`make transfer`, `stages/transfer.py`, `params.transfer`): the 14 non-shortlisted and 3 shortlisted reference models of the Reviewer 1 experiment (+ `bert-base-uncased`) on
C1-C5 (+ the extra C6) in the new pipeline, own root `artifacts/transfer`, same chunks and labels as the main run (the chunk cache and the ChemQuests files of the main root are reused),
no BM25 / hybrid / rerank, so the rerank ablation check cannot fire. `make transfer TRANSFER_ARGS="--dry-run"` validates the block and prints the plan.

**Statistics**: `params.report.resamples: 10000`. Pairs per configuration: dense vs BM25, dense vs dense, hybrid vs its dense run and vs BM25, rerank vs its dense run, vs BM25 and vs its hybrid;
Holm within a `family` (`dense_vs_bm25`, `dense_vs_dense`, `hybrid`, `rerank`) x level x metric (`stats_compare.csv`). Each run is resampled once per level; with the pilot copy (20 runs, 4 configurations) `make report`
takes 18 s at 10,000 resamples (6 s of it without the cover-level table). `tools/lexical_overlap.py` uses the same setting.

**New tables in `make report`**: `table_label_stats.csv` (chunks, chunk length median/mean/max, questions in Step 1/2/3, % split evidence, relevant chunks per question),
`table_cover_sensitivity.csv` (every run rescored at c = 0.8 / 0.9 / 1.0 = `params.analysis.cover_levels`, paper-level intervals; `python -m tools.rescore --sensitivity <tests> --qa <qa.jsonl> --output <csv>`),
`table_truncation_phase1.csv`.

**Manual validation of the labels**: `python -m tools.validation_sample --qa ... --prepared-root artifacts/prepared --out-dir results/validation` draws `params.validation.n_questions` (100) questions,
seeded, stratified by the first listed subfield of the paper (at least 2 per subfield), and writes one CSV + Markdown sheet per annotator (question, span with 300 characters of context, labelled chunks of three
configurations; answers `yes` / `partial` / `no`). `python -m tools.kappa --sheets A.csv B.csv --out-dir ...` gives percent agreement, Cohen's and linearly weighted kappa with paper-level intervals and the disagreements.

**Analysis tools on the rerun outputs** (`make analysis` = `python -m tools.run_analysis`; `ANALYSIS_ARGS="--only ..."`; all share `tools/analysis_data.py`: a runs x queries matrix and the paper-level resampling of
`tools/stats.py`; output folder `params.analysis.out_dir`):

| tool | output | what it answers |
|---|---|---|
| `tools/token_budget.py` | `token_budget.csv` | share of the evidence span and hit rate (>= c) when the reader gets B = 512 / 1024 / 2048 reference tokens: the ranking is walked, overlapping chunks are counted as the union of their characters, the last chunk is cut at the token that still fits |
| `tools/subfield_breakdown.py` | `breakdown_subfield.csv`, `breakdown_tag.csv`, `*_paired.csv`, `breakdown_groups.csv` | nDCG@10 / Recall@10 / Geom@10 per subfield (a paper counts in every listed subfield, "unlabelled" for none) and per rule-based question tag (question word, yes/no, number or unit, chemical formula), paper-level intervals, dense - BM25 per group, questions and papers per group |
| `tools/variance_decomposition.py` | `variance_decomposition[_balanced].csv`, `variance_ranges*.csv` | Type II eta^2 of model, strategy, size, overlap, model x strategy for Geom@10 of the dense runs, intervals by paper-level bootstrap, difference model - strategy, balanced sub-grid, simple ranges |
| `tools/winner_stability.py` | `winner_<pool>_by_configuration.csv`, `_by_model.csv`, `_overall.csv` | share of resamples in which a run is first (pools: dense, all kinds) |
| `tools/phase_transfer.py` | `phase_transfer_*.csv`, `fig_phase_transfer.png` | Spearman and Kendall (bootstrap over models) between Phase 1 and Phase 2 for all, shortlisted and excluded models; tuned - base pairs; group means; excluded vs shortlisted |
| `tools/examples.py` | `examples.csv`, `examples.md` | named cases (fails at 128 / works at 512, dense wins, BM25 wins, all fail, rerank rescues) with question, span excerpt and top chunks of each retriever |
| `tools/fragmentation.py` | `fragmentation.csv` | per question: fragmented at the small size vs gain at the large size, difference with an interval |

### Efficiency measures (deployment cost)

`pipeline_lib/efficiency.py` records what a deployment needs to know, per configuration, in `efficiency_phase2.csv`
(`<artifacts>/tests/`, key `model, revision, chunker, size, overlap`) and `efficiency_phase1.csv` (key `model, revision, task`),
plus an `efficiency.json` (rows + GPU / CUDA / library versions) next to each run's results. The library default is off, the Makefile
switches it on (`STAGE1_ARGS`, `EFFICIENCY_ARGS`) and it does not touch retrieval, chunking, qrels, metrics, seeds or the grid: the probe wraps the model handed to MTEB and only watches
`encode()`. Its timings are **not** part of `evaluation_time` or `run_manifest.json: timing_s` (the latency runs happen after the
manifest is written, and what they add to `STNoPrompt.timings` is removed again).
Checked on two pilot configurations (`BAAI/bge-m3` fixed-token 128/0, `e5-large-v2` hierarchical-section 512/0) and on `e5-small`
(Phase 1): `ChemQuest.json` scores, per-query metrics and the saved rankings are byte-identical with and without the flags.

| column(s) | meaning |
|---|---|
| `n_chunks`, `embedding_dim`, `index_mb_fp32`, `index_mb_fp16` | corpus size, dimension, `n_chunks x dim x 4 or 2 bytes` (MB = MiB) |
| `chunk_tokens_ref_{mean,median,max}` | chunk length with the chunking tokenizer (bert-base-uncased, no special tokens) |
| `chunk_tokens_own_{mean,median,max}` | chunk length with the model's own tokenizer (special tokens, no prompt, no truncation) |
| `n_params`, `weights_mb`, `weights_fp16_mb`, `weight_dtype` | parameter count, size of the loaded weights, size as fp16 |
| `peak_gpu_mem_mb`, `encode_batch_size` | `torch.cuda.max_memory_allocated` during the corpus encode (weights included; `weights_gpu_mem_mb` = weights only), batch size used |
| `corpus_encode_s`, `corpus_chunks_per_s` | time of the corpus `encode()` calls and throughput (the last eval split for Phase 1 tasks with several, `n_eval_passes`) |
| `latency_{query_encode,search,total}_ms_{median,p95,mean}` | batch size 1 per query after a warm-up; search = exact cosine against the whole corpus + top-10 (on the CPU as in MTEB; `--latency-device cuda` for the GPU). The index is normalised once; `latency_top10_agree` compares the top-10 with MTEB's own `cos_sim` |

```bash
# new runs (finished runs are skipped; use a new --artifacts-root or --force)
python -m stages.rerun eval --efficiency                        # index, model, memory, encoding time; no latency
python -m stages.rerun eval --efficiency --latency              # + latency (200 evenly spaced queries, 10 warm-up)
python -m stages.rerun eval --latency --latency-sizes 256 512 --latency-overlaps 0 --latency-chunkers fixed_token   # latency for a subset only
#   also: --latency-models bge e5 --latency-queries 0 (all queries) --latency-device cuda
python -m stages.run_eval --config <stage6.yaml> --efficiency --latency --eff-chunker fixed_token --eff-size 256 --eff-overlap 0
python -m stages.model_screening --config configs/stages/stage1.yaml --efficiency [--latency]   # Phase 1

# existing runs, nothing re-encoded: index and model columns from the chunk files (loads each model on the CPU once)
python -m tools.efficiency_backfill --tests-root artifacts/tests [--no-model] [--models bge-m3]
python -m tools.efficiency_backfill --phase 1 --config configs/stages/stage1.yaml --out artifacts/embedding_models_screening/validation

# tables and figures (Geom@10 from the score table of tools/collect_runs.py / make report)
python -m tools.efficiency_analysis --efficiency artifacts/tests/efficiency_phase2.csv \
    --scores results/v2-revision/runs_overview.csv --out results/efficiency [--phase1 .../efficiency_phase1.csv]
```

`tools/efficiency_analysis.py` writes `table_efficiency_models.{csv,tex}` (per model, at its best Geom@10 configuration; memory and
latency fall back to the median over the measured configurations, flagged in `latency_basis` / `gpu_basis`),
`table_index_size.{csv,tex}` (chunk size x overlap), `fig_geom_vs_latency.{png,pdf}` and `fig_geom_vs_index_size.{png,pdf}` with the
Pareto front (`pareto_*.csv`). The backfill and the live runs write the same CSV and complete each other (merged by key).
Latency and GPU numbers belong to the machine named in `efficiency.json` and are comparable only within one run of the study.

### `evaluation_time` (seconds) of the four kinds of runs

`ChemQuest.json: evaluation_time` and `run_manifest.json: timing_s.mteb_evaluation_time_s` (also the `evaluation_time_s` column of
the run table). Model loading is never counted. The parts of a derived run are stored separately in its `run_manifest.json`.

| run | evaluation_time | what it covers |
|---|---|---|
| dense | MTEB's `evaluation_time` | encode corpus and queries, exact search, metrics, dump of the ranking |
| bm25 | index + search + metrics + dump | `bm25_index_s + bm25_search_s + metrics_s + dump_s`; reading the corpus / queries files is not counted (MTEB loads the dataset before it starts the clock). On the external tasks: index + search + metrics (the Phase 1 dense runs saved no predictions) |
| hybrid (RRF) | dense + bm25 + `fusion_s` | `dense_evaluation_time_s`, `bm25_evaluation_time_s`, `fusion_s` = read the two rankings + fuse + metrics + dump |
| rerank | dense + `rerank_s` | `dense_evaluation_time_s`, `rerank_s` = read corpus / queries + score the pairs + reorder + metrics + dump (`score_pairs_s`: the cross-encoder forward passes only); `model_load_s` stored, not counted |

A hybrid and a rerank run need their input runs, so their cost includes the cost of those inputs. If an input run has no time the
sum is null and a warning is printed. All four kinds are written with both label sets (span primary, lexical next to it).

## Results

- Historical Phase 1 results: `artifacts/embedding_models_screening/validation/`
- Historical Phase 2 runs: `artifacts/tests/`
- Canonical Phase 2 package: `artifacts/canonical_phase2/`

The canonical Phase 2 table contains exactly 13 intended models × 140 configurations = 1,820 rows. Historical `intfloat/e5-base-v2` Phase 2 runs remain preserved under `artifacts/tests/` as provenance but are not part of the canonical analysis.

The audited results show that embedding-model choice is associated with the largest retrieval-effectiveness differences. Retrieval-oriented E5, BGE, and Nomic models are among the strongest evaluated encoders. Within the evaluated grid, large chunks—especially 448–512 tokens—with low overlap are generally strong, while semantic chunking adds computational cost without consistently improving retrieval.

## Citation

```bibtex
@article{amiri2025chunktwice,
  title   = {Chunk Twice, Embed Once: A Systematic Study of Segmentation and Representation Trade-offs in Chemistry-Aware Retrieval-Augmented Generation},
  author  = {Amiri, Mahmoud and Bocklitz, Thomas},
  journal = {arXiv preprint arXiv:2506.17277},
  year    = {2025},
  doi     = {10.48550/arXiv.2506.17277}
}
```

## License

[MIT License](LICENSE)
