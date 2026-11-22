# Chunk Twice, Embed Once

Code, configurations, data, and result artifacts for:

> **Chunk Twice, Embed Once: A Systematic Study of Segmentation and Representation Trade-offs in Chemistry-Aware Retrieval-Augmented Generation**  
> Mahmoud Amiri and Thomas Bocklitz (2025)

## Study design

Phase 1 evaluates 41 distinct embedding models with complete results on the two external retrieval tasks `ChemNQRetrieval` and `ChemHotpotQARetrieval`. The union of the top 20% from each task gives the verified 13-model Phase 2 shortlist.

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
make eval              # every model (+ BM25) on the prepared folders; finished runs are skipped
make rerun             # prepare, then eval
make pilot             # 2 models + BM25 + their RRF hybrids, fixed-token and hierarchical-section, sizes 128/512, overlap 0 (artifacts/pilot_v2)
make check-identical TESTS_ROOT=artifacts/tests   # corpus / qrels byte-identical across models?
make collect TESTS_ROOT=artifacts/tests           # one CSV row per run
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
- `tools/lexical_overlap.py`: question/evidence-span word overlap, BM25 rank of the evidence chunk, BM25 advantage over a
  dense model by overlap bin.
- All 42 Phase 1 models are pinned to commit hashes in `stage1.models`; `rerun` replaces a moving `main` by the pin.
- `params.mask_instruction_tokens` (default `false`): INSTRUCTOR masking (instruction tokens left out of the mean
  pooling) for instructor-xl, implemented without the `InstructorEmbedding` package.

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
