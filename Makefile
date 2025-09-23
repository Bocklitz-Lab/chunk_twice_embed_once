SHELL := /bin/bash
.PHONY: all stage1 stage2 stage3 stage4 clean show-configs

OVERLAP = 0 32 #64
SIZE    = 128 384 #192 256 320  448
CSV ?= artifacts/embedding_models_screening/passed_per_dataset.csv

MODELS = $(shell awk -F, 'NR>1 {printf "%s%s@%s", sep,$$2,$$3; sep=" "}' $(CSV) | tr -d "\r")

CHUNKER =  recursive_token fixed_token semantic_fixed semantic_recursive hierarchical_section hybrid_multi

INPUT_CONFIG = configs/main_config.yaml
OUTPUT_CONFIG = configs/stages/


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

stage1:
	@echo "▶️ Stage1: model_screening_validation"
	python -m stages.model_screening --config $(STAGE1_CONFIG)

stage2:
	@echo "▶️ Stage2: model_screening_selection"
	python -m stages.screen_top_k_models --config $(STAGE2_CONFIG)

stage3:
	@echo "▶️ Stage3: download_chemquests"
	python -m stages.download_chemquests --config $(STAGE3_CONFIG)

stage4-6:
	@test -s "$(CSV)" || { echo "❌ CSV not found or empty: $(CSV). Did stage2 run and write it?"; exit 1; }
	@echo "📄 Using models from $(CSV)"
	@echo "🧪 Models: $(MODELS)"
	@for m in $(MODELS); do \
		name="$${m%@*}"; \
		rev="$$(printf '%s' "$$m" | cut -d@ -f2-)"; \
		dir="$${name//\//_}"; dir="$${dir//-/_}"; \
		for c in $(CHUNKER); do \
			for s in $(SIZE); do \
			for o in $(OVERLAP); do \
				outdir=artifacts/tests/$${dir}_$${c}_c$${s}_o$${o}; \
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
				--param chunk.overlap=$${o}; \
				rm -f $$config_dir/stage1.yaml $$config_dir/stage2.yaml $$config_dir/stage3.yaml; \
				$(MAKE) -f test.mk all \
				MODELS=$$m CHUNKER=$$c CHUNK_SIZE=$$s CHUNK_OVERLAP=$$o \
				OUTDIR=$$outdir \
				STAGE4_CONFIG=$$config_dir/stage4.yaml \
				STAGE5_CONFIG=$$config_dir/stage5.yaml \
				STAGE6_CONFIG=$$config_dir/stage6.yaml; \
			done; \
			done; \
		done; \
		done


# clean:
# 	rm -rf artifacts
# 	@echo "🧹 Cleaned artifacts/"
