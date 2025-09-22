.PHONY: all stage1 stage2 stage3 stage4 clean show-configs

OVERLAP = 0 32 64
SIZE    = 128 192 256 320 384 448
MODEL   = all_MiniLM_L6_v2
CHUNKER = fixed_token

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
	python -m pipeline_lib.split_config -i $(INPUT_CONFIG) -o $(OUTPUT_CONFIG)
	rm -rf configs/stages/stage4.yaml configs/stages/stage5.yaml configs/stages/stage6.yaml

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
	@for m in $(MODEL); do \
	  for c in $(CHUNKER); do \
	    for s in $(SIZE); do \
	      for o in $(OVERLAP); do \
	        outdir=artifacts/tests/$${m}_$${c}_c$${s}_o$${o}; \
	        config_dir=$$outdir/config; \
	        echo "▶️ model=$$m chunker=$$c size=$$s overlap=$$o -> $$config_dir"; \
	        mkdir -p $$config_dir; \
	        python -m pipeline_lib.split_config -i $(INPUT_CONFIG) -o $$config_dir; \
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
