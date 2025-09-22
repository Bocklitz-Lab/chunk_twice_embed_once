.PHONY: all stage1 stage2 stage3 stage4 clean show-configs

MODELS ?= all_MiniLM_L6_v2
CHUNKER ?= fixed_token
CHUNK_SIZE ?= 256
CHUNK_OVERLAP ?= 20

# Build directory name
OUTDIR := artifacts/$(MODELS)_$(CHUNKER)_c$(CHUNK_SIZE)_o$(CHUNK_OVERLAP)
STAGE1_CONFIG ?= $(OUTDIR)/config/stage1.yaml
STAGE2_CONFIG ?= $(OUTDIR)/config/stage2.yaml
STAGE3_CONFIG ?= $(OUTDIR)/config/stage3.yaml
STAGE4_CONFIG ?= $(OUTDIR)/config/stage4.yaml
STAGE5_CONFIG ?= $(OUTDIR)/config/stage5.yaml
STAGE6_CONFIG ?= $(OUTDIR)/config/stage6.yaml


# all: stage1 stage2 stage3 stage4
# 	@echo "✅ Pipeline complete. See artifacts/stage4 for queries/qrels and artifacts/stage1 for corpus."

show-configs:
	@echo "Using:"
	@echo "  stage1: $(STAGE1_CONFIG)"
	@echo "  stage2: $(STAGE2_CONFIG)"
	@echo "  stage3: $(STAGE3_CONFIG)"
	@echo "  stage4: $(STAGE4_CONFIG)"
	@echo "  stage5: $(STAGE5_CONFIG)"
	@echo "  stage6: $(STAGE6_CONFIG)"

stage1:
	@echo "▶️ Stage1: model_screening_validation"
	python -m stages.model_screening --config $(STAGE1_CONFIG)

stage2:
	@echo "▶️ Stage2: model_screening_selection"
	python -m stages.screen_top_k_models --config $(STAGE2_CONFIG)

stage3:
	@echo "▶️ Stage3: download_chemquests"
	python -m stages.download_chemquests --config $(STAGE3_CONFIG)

stage4:
	@echo "▶️ Stage4: chunking"
	python -m stages.chunking --config $(STAGE4_CONFIG)

stage5:
	@echo "▶️ Stage5: MTEB task"
	python -m stages.make_mteb_task --config $(STAGE5_CONFIG)

stage6:
	@echo "▶️ Stage6: test custom task"
	python -m stages.run_eval --config $(STAGE6_CONFIG)
# clean:
# 	rm -rf artifacts
# 	@echo "🧹 Cleaned artifacts/"
