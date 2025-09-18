.PHONY: all stage1 stage2 stage3 stage4 clean show-configs

# Per-stage configs (override on CLI if you want)
STAGE1_CONFIG ?= configs/embedding_models_fast_validation_config.yaml
STAGE2_CONFIG ?= configs/screen_top_models_config.yaml
STAGE3_CONFIG ?= configs/download_chemquests_config.yaml


# all: stage1 stage2 stage3 stage4
# 	@echo "✅ Pipeline complete. See artifacts/stage4 for queries/qrels and artifacts/stage1 for corpus."

show-configs:
	@echo "Using:"
	@echo "  stage1: $(STAGE1_CONFIG)"
	@echo "  stage2: $(STAGE2_CONFIG)"
	@echo "  stage3: $(STAGE3_CONFIG)"

stage1:
	@echo "▶️ Stage1: model_screening_validation"
	python -m stages.model_screening --config $(STAGE1_CONFIG)

stage2:
	@echo "▶️ Stage2: model_screening_selection"
	python -m stages.screen_top_k_models --config $(STAGE2_CONFIG)

stage3:
	@echo "▶️ Stage3: download_chemquests"
	python -m stages.download_chemquests --config $(STAGE3_CONFIG)

# clean:
# 	rm -rf artifacts
# 	@echo "🧹 Cleaned artifacts/"
