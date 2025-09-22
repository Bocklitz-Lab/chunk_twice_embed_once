.PHONY: all stage1 stage2 stage3 stage4 clean show-configs

MODELS ?= all_MiniLM_L6_v2
CHUNKER ?= fixed_token

SIZE ?= 256
OVERLAP ?= 20


INPUT_CONFIG = configs/main_config.yaml
OUTPUT_CONFIG = configs/stages/
SCRIPT = pipeline_lib/split_config.py


# Build directory name
OUTDIR := artifacts/tests/$(MODELS)_$(CHUNKER)_c$(SIZE)_o$(OVERLAP)
STAGE4_CONFIG ?= $(OUTDIR)/config/stage4.yaml
STAGE5_CONFIG ?= $(OUTDIR)/config/stage5.yaml
STAGE6_CONFIG ?= $(OUTDIR)/config/stage6.yaml


all: stage4 stage5 stage6
	@echo "✅ Pipeline complete."

show-configs:
	@echo "Using:"
	@echo "  stage4: $(STAGE4_CONFIG)"
	@echo "  stage5: $(STAGE5_CONFIG)"
	@echo "  stage6: $(STAGE6_CONFIG)"


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
