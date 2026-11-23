SHELL := /bin/bash
.PHONY: all stage0 stage1 stage2 stage3 stage4-6 stage7 show-configs prepare eval rerun pilot hybrid rerank external-bm25 check-identical collect collect-csv report check-models-csv transfer

OVERLAP = 0 16 32 48 64
SIZE    = 128 384 192 256 320  448 512
TOTAL_LIMIT ?= 512
TOKEN_LIMIT ?= 512
MIN_STRIDE ?= 96
MAX_OVERLAP_ABS ?= 64
MAX_OVERLAP_PCT ?= 30

# Shortlist of stage 2 = the Phase 2 model list. Default = stage2.output_per_dataset_csv of configs/main_config.yaml (it carries the name of the Phase 1
# results folder, params.paths.phase1_results_subdir; tests/test_phase1_paths.py keeps the two equal). The PUBLISHED shortlist is
# artifacts/embedding_models_screening/passed_per_dataset.csv (restore it first, see docs/RERUN_PLAN.md) and is selected with MODELS_CSV=...
MODELS_CSV ?= artifacts/embedding_models_screening/passed_per_dataset__validation_prompts.csv

MODELS = $(shell awk -F, 'NR>1 {printf "%s%s@%s", sep,$$2,$$3; sep=" "}' $(MODELS_CSV) | tr -d "\r")

CHUNKER = recursive_token fixed_token semantic_recursive semantic_fixed  hierarchical_section 

INPUT_CONFIG = configs/main_config.yaml
OUTPUT_CONFIG = configs/stages/

LOSS_CSV ?= artifacts/loss.csv
RESULTS ?= artifacts/tests/
DURATION_CSV ?= $(RESULTS)/durations.csv

# A configuration counts as finished only when the Stage 6 summary exists. run_eval.py writes it
# last, after MTEB finished and the predictions were saved. A non-empty folder is not enough:
# a run that failed after stage 4 or 5 leaves one behind.
FINAL_RESULT ?= results/summary.json

# Extra "--param KEY=VALUE" options for split_config (e.g. a pilot with other paths or eval.top_k=100).
# Labels (stage 5, see params.labeling in the config): method span (default) or lexical, cover level c:
#   make stage4-6 EXTRA_PARAMS="--param labeling.cover_level=0.8"
#   make stage4-6 EXTRA_PARAMS="--param labeling.method=lexical"
# Stage 5 runs stages/make_span_qrels.py for method span and stages/make_mteb_task.py for method lexical.
EXTRA_PARAMS ?=

# Build directory name
STAGE1_CONFIG ?= configs/stages/stage1.yaml
STAGE2_CONFIG ?= configs/stages/stage2.yaml
STAGE3_CONFIG ?= configs/stages/stage3.yaml


all: stage1 stage2 stage3 stage4-6
	@echo "✅ Pipeline complete."

show-configs:
	@echo "Using:"
	@echo "  stage0"
	@echo "  stage1: $(STAGE1_CONFIG)"
	@echo "  stage2: $(STAGE2_CONFIG)"
	@echo "  stage3: $(STAGE3_CONFIG)"

stage0:
	@echo "▶️ Stage0: generate_configs"
	python -m pipeline_lib.split_config --input  $(INPUT_CONFIG) --output-dir $(OUTPUT_CONFIG) --section stage1 stage2 stage3
	# rm -rf configs/stages/stage4.yaml configs/stages/stage5.yaml configs/stages/stage6.yaml

# Efficiency measures (params.efficiency in the main config) are ON by default; make them empty to switch off, e.g. make eval EFFICIENCY_ARGS=
STAGE1_ARGS ?= --efficiency
EFFICIENCY_ARGS ?= --efficiency --latency

stage1:
	@echo "▶️ Stage1: model_screening_validation"
	python -m stages.model_screening --config $(STAGE1_CONFIG) $(STAGE1_ARGS)

stage2:
	@echo "▶️ Stage2: model_screening_selection"
	python -m stages.screen_top_k_models --config $(STAGE2_CONFIG)

stage3:
	@echo "▶️ Stage3: download_chemquests"
	python -m stages.download_chemquests --config $(STAGE3_CONFIG)

stage4-6:
	@test -s "$(MODELS_CSV)" || { echo "❌ MODELS_CSV not found or empty: $(MODELS_CSV). Did stage2 run and write it?"; exit 1; }
	@echo "📄 Using models from $(MODELS_CSV)"
	@echo "🧪 Models: $(MODELS)"
	@# Ensure durations CSV exists with header
	@[ -f "$(DURATION_CSV)" ] || { \
		mkdir -p "$(RESULTS)"; \
		echo "started_at,ended_at,duration_s,status,stage,model,revision,chunker,size,overlap,outdir" > "$(DURATION_CSV)"; \
	}
	@for m in $(MODELS); do \
		name="$${m%@*}"; \
		rev="$$(printf '%s' "$$m" | cut -d@ -f2-)"; \
		dir="$${name//\//_}"; dir="$${dir//-/_}"; \
		for c in $(CHUNKER); do \
			for s in $(SIZE); do \
			for o in $(OVERLAP); do \
				if (( s > $(TOKEN_LIMIT) )); then \
					echo "⏭️  Skip: L=$$s > TOKEN_LIMIT=$(TOKEN_LIMIT)"; \
					continue; \
				fi; \
				pct_cap=$$(( s * $(MAX_OVERLAP_PCT) / 100 )); \
				max_o=$$pct_cap; \
				if (( max_o > $(MAX_OVERLAP_ABS) )); then max_o=$(MAX_OVERLAP_ABS); fi; \
				if (( o > max_o )); then \
					echo "⏭️  Skip: O=$$o > max_allowed=$$max_o for L=$$s"; \
					continue; \
				fi; \
				if (( s - o < $(MIN_STRIDE) )); then \
					echo "⏭️  Skip: stride=$$((s-o)) < $(MIN_STRIDE) (L=$$s, O=$$o)"; \
					continue; \
				fi; \
				if (( s + o > $(TOTAL_LIMIT) )); then \
					echo "⏭️  Skip: L+O=$$((s+o)) > $(TOTAL_LIMIT) (L=$$s, O=$$o)"; \
					continue; \
				fi; \
				outdir=artifacts/tests/$${dir}_$${c}_c$${s}_o$${o}; \
				if [ -s "$$outdir/$(FINAL_RESULT)" ]; then \
					echo "⏭️  Skip: finished ($(FINAL_RESULT) exists) -> $$outdir"; \
					continue; \
				fi; \
				if [ -d "$$outdir" ]; then \
					echo "♻️  Earlier run incomplete (no $(FINAL_RESULT)); running again -> $$outdir"; \
				fi; \
				config_dir=$$outdir/config; \
				echo "▶️ model=$$name@$$rev chunker=$$c size=$$s overlap=$$o -> $$config_dir"; \
				mkdir -p $$config_dir; \
				python -m pipeline_lib.split_config \
				--input $(INPUT_CONFIG) \
				--output-dir $$config_dir \
				--sections stage4 stage5 stage6 \
				--param model.name=$${name} \
				--param model.revision=$${rev} \
				--param model.dir=$${dir} \
				--param chunk.chunker=$${c} \
				--param chunk.size=$${s} \
				--param chunk.overlap=$${o} $(EXTRA_PARAMS); \
				rm -f $$config_dir/stage1.yaml $$config_dir/stage2.yaml $$config_dir/stage3.yaml; \
				\
				# ---------- Run stage4 (chunking) ---------- \
				stage="stage4"; \
				start_iso=$$(date -u +"%Y-%m-%dT%H:%M:%SZ"); \
				start_sec=$$(date +%s); \
				$(MAKE) -f test.mk stage4 \
					MODELS=$$m CHUNKER=$$c SIZE=$$s OVERLAP=$$o \
					OUTDIR=$$outdir \
					STAGE4_CONFIG=$$config_dir/stage4.yaml; \
				status=$$?; \
				end_sec=$$(date +%s); \
				end_iso=$$(date -u +"%Y-%m-%dT%H:%M:%SZ"); \
				dur=$$((end_sec - start_sec)); \
				echo $$dur > "$$outdir/duration_$${stage}_s.txt"; \
				printf "%s,%s,%d,%d,%s,%s,%s,%s,%s,%s,%s\n" \
					"$$start_iso" "$$end_iso" "$$dur" "$$status" "$$stage" \
					"$$name" "$$rev" "$$c" "$$s" "$$o" "$$outdir" >> "$(DURATION_CSV)"; \
				if [ $$status -ne 0 ]; then \
					echo "❌ $$stage failed (model=$$name@$$rev, chunker=$$c, size=$$s, overlap=$$o) in $$dur s"; \
					exit $$status; \
				else \
					echo "⏱️  $$stage done in $$dur s -> $$outdir"; \
				fi; \
				\
				# ---------- Run stage5 (MTEB task) ---------- \
				stage="stage5"; \
				start_iso=$$(date -u +"%Y-%m-%dT%H:%M:%SZ"); \
				start_sec=$$(date +%s); \
				$(MAKE) -f test.mk stage5 \
					MODELS=$$m CHUNKER=$$c SIZE=$$s OVERLAP=$$o \
					OUTDIR=$$outdir \
					STAGE5_CONFIG=$$config_dir/stage5.yaml; \
				status=$$?; \
				end_sec=$$(date +%s); \
				end_iso=$$(date -u +"%Y-%m-%dT%H:%M:%SZ"); \
				dur=$$((end_sec - start_sec)); \
				echo $$dur > "$$outdir/duration_$${stage}_s.txt"; \
				printf "%s,%s,%d,%d,%s,%s,%s,%s,%s,%s,%s\n" \
					"$$start_iso" "$$end_iso" "$$dur" "$$status" "$$stage" \
					"$$name" "$$rev" "$$c" "$$s" "$$o" "$$outdir" >> "$(DURATION_CSV)"; \
				if [ $$status -ne 0 ]; then \
					echo "❌ $$stage failed (model=$$name@$$rev, chunker=$$c, size=$$s, overlap=$$o) in $$dur s"; \
					exit $$status; \
				else \
					echo "⏱️  $$stage done in $$dur s -> $$outdir"; \
				fi; \
				\
				# ---------- Run stage6 (custom eval) ---------- \
				stage="stage6"; \
				start_iso=$$(date -u +"%Y-%m-%dT%H:%M:%SZ"); \
				start_sec=$$(date +%s); \
				$(MAKE) -f test.mk stage6 \
					MODELS=$$m CHUNKER=$$c SIZE=$$s OVERLAP=$$o \
					OUTDIR=$$outdir \
					STAGE6_CONFIG=$$config_dir/stage6.yaml; \
				status=$$?; \
				end_sec=$$(date +%s); \
				end_iso=$$(date -u +"%Y-%m-%dT%H:%M:%SZ"); \
				dur=$$((end_sec - start_sec)); \
				echo $$dur > "$$outdir/duration_$${stage}_s.txt"; \
				printf "%s,%s,%d,%d,%s,%s,%s,%s,%s,%s,%s\n" \
					"$$start_iso" "$$end_iso" "$$dur" "$$status" "$$stage" \
					"$$name" "$$rev" "$$c" "$$s" "$$o" "$$outdir" >> "$(DURATION_CSV)"; \
				if [ $$status -ne 0 ]; then \
					echo "❌ $$stage failed (model=$$name@$$rev, chunker=$$c, size=$$s, overlap=$$o) in $$dur s"; \
					exit $$status; \
				else \
					echo "⏱️  $$stage done in $$dur s -> $$outdir"; \
				fi; \
			done; \
			done; \
		done; \
		done

stage7:
	@echo "▶️ Stage7: calculate loss"
	python -m pipeline_lib.single_value_scores --root $(RESULTS) --w-ndcg 0.5 --w-recall 0.5 --w-time 0.0 --time-seconds-key evaluation_time --time-transform inverse --reference-seconds 300.0 --allow-time-missing-as-one --csv $(RESULTS)/loss.csv
	python -m pipeline_lib.merge_runs --csv $(RESULTS)/durations.csv --save $(RESULTS)/merged_runs.csv
	python -m pipeline_lib.merge_csvs --loss $(RESULTS)/loss.csv --runs $(RESULTS)/merged_runs.csv --out $(RESULTS)/combined.csv

# clean:
# 	rm -rf artifacts
# 	@echo "🧹 Cleaned artifacts/"


# ---------------------------------------------------------------------------------------------------------------
# Reviewer revisions: prepare / eval split (stages/rerun.py). stage4-6 above is the legacy per-model path.
#   make prepare            chunk + label every configuration ONCE (reference tokenizer; 140 configurations)
#   make eval               every model (+ BM25) on the prepared folders; finished runs are skipped
#   make rerun              prepare, then eval
#   make pilot              the pilot of params.pilot in the main config (own artifacts root, nothing else touched)
#   make hybrid             only the RRF hybrids (dense + BM25) of the finished runs (also part of eval / rerun / pilot; resumable)
#   make rerank             only the cross-encoder rerank runs, params.rerank: primary depth + ablation (also part of eval / rerun / pilot; resumable)
#   make external-bm25      BM25 on ChemNQRetrieval / ChemHotpotQARetrieval (params.external) -> artifacts/embedding_models_screening/baselines
#   make check-identical    hash the corpus / qrels each model read: byte-identical across models?
#   make report             statistics, diagnostics, lexical-overlap, paper tables -> results/v2-revision/ (tracked); also the last step of
#                           eval / rerun / pilot (params.report). Extra options: REPORT_ARGS="--external-root <Phase 1 results> --rerank-top-n 50"
#   make collect            same as report
#   make collect-csv        only the run table, one CSV row per run
# Extra options:  make rerun RERUN_ARGS="--no-bm25 --param use_prompts=false"
# Legacy per-model chunking (old behaviour):  RERUN_ARGS="--param chunking_tokenizer=model --param semantic_breakpoint_model=model"
# ---------------------------------------------------------------------------------------------------------------
RERUN_ARGS ?=
TESTS_ROOT ?= artifacts/tests
COLLECT_CSV ?= artifacts/runs_overview.csv

# eval / rerun / hybrid / rerank need the model list: stop at once, with a message, if MODELS_CSV is missing (nothing is created or copied here)
check-models-csv:
	@test -s "$(MODELS_CSV)" || { \
		echo "❌ MODELS_CSV not found or empty: $(MODELS_CSV)"; \
		echo "   It is written by stage 2 (make stage1 stage2: Phase 1 with prompts)."; \
		echo "   To use the PUBLISHED shortlist, restore the old Phase 1 results first:"; \
		echo "     cp -a temp_files/artifacts_old/embedding_models_screening artifacts/"; \
		echo "     make eval MODELS_CSV=artifacts/embedding_models_screening/passed_per_dataset.csv"; \
		exit 1; }

prepare:
	PYTHONHASHSEED=0 python -m stages.rerun prepare --models-csv $(MODELS_CSV) $(RERUN_ARGS)

eval: check-models-csv
	PYTHONHASHSEED=0 python -m stages.rerun eval --models-csv $(MODELS_CSV) $(EFFICIENCY_ARGS) $(RERUN_ARGS)

rerun: check-models-csv
	PYTHONHASHSEED=0 python -m stages.rerun all --models-csv $(MODELS_CSV) $(EFFICIENCY_ARGS) $(RERUN_ARGS)

pilot:
	PYTHONHASHSEED=0 python -m stages.rerun all --pilot $(RERUN_ARGS)

hybrid: check-models-csv
	PYTHONHASHSEED=0 python -m stages.rerun hybrid --models-csv $(MODELS_CSV) $(RERUN_ARGS)

rerank: check-models-csv
	PYTHONHASHSEED=0 python -m stages.rerun rerank --models-csv $(MODELS_CSV) $(RERUN_ARGS)

# Transfer runs (params.transfer in the main config): models Phase 1 did not shortlist, on a few ChemQuests configurations, own artifacts root
# (artifacts/transfer), same prepared chunks and labels as the main run (run `make prepare` first). TRANSFER_ARGS e.g. --dry-run, --no-extras.
TRANSFER_ARGS ?=
transfer:
	PYTHONHASHSEED=0 python -m stages.transfer $(EFFICIENCY_ARGS) $(TRANSFER_ARGS)

EXTERNAL_ARGS ?=
external-bm25:
	PYTHONHASHSEED=0 python -m stages.run_bm25_mteb --main-config $(INPUT_CONFIG) $(EXTERNAL_ARGS)

check-identical:
	python -m tools.check_identical --tests-root $(TESTS_ROOT) --out $(TESTS_ROOT)/../identical_check.csv

ARTIFACTS_ROOT ?= artifacts
REPORT_DIR ?= results/v2-revision

REPORT_ARGS ?=
report:
	python -m tools.make_report --artifacts-root $(ARTIFACTS_ROOT) --out-dir $(REPORT_DIR) $(REPORT_ARGS)

collect: report

collect-csv:
	python -m tools.collect_runs --tests-root $(TESTS_ROOT) --out $(COLLECT_CSV)
