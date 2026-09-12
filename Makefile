SHELL := /bin/sh
APP_DIR := techcombank_rag
PYTHON_BIN ?= python3
VENV_DIR := $(APP_DIR)/.venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
STAMP := $(VENV_DIR)/.requirements-installed
QUESTIONS ?= $(APP_DIR)/data/evaluation/public.json
OUTPUT_DIR ?= $(APP_DIR)/data/evaluation/results

PADDLE_PYTHON ?= python3
PADDLE_DEVICE ?= cpu
A1_PDF ?= $(APP_DIR)/data/raw/techcombank-bao-cao-thuong-nien-2025-vie-update.pdf
A1A_INDEX := $(APP_DIR)/data/index/a1a_paddleocr_vl_1_6_plain_fixed
A1B_INDEX := $(APP_DIR)/data/index/a1b_paddleocr_vl_1_6_markdown_fixed
A2_INDEX := $(APP_DIR)/data/index/a2_paddleocr_vl_1_6_layout_aware
A3_INDEX := $(APP_DIR)/data/index/a3_paddleocr_vl_1_6_multigranularity
A31_INDEX_ROOT := $(APP_DIR)/data/index/a31_semantic_multirepr
A32_INDEX_ROOT := $(APP_DIR)/data/index/a32_metric_aware
A4_INDEX := $(APP_DIR)/data/index/a4_selective_cascade

.PHONY: setup run evaluate baseline test verify-index verify-baseline verify-b4-index ingest ingest-a1 ingest-a2-a4 build-b1a-index build-a31 build-a32 build-bm25 benchmark-metric-aware benchmark-metric-multiturn benchmark-a1a benchmark-a1b benchmark-a2 benchmark-a3 benchmark-a4 benchmark-a31 benchmark-a31-dev ablate-hyde ablate-hyde-dev ablate-b4-retrieval ablate-b4 ablate-b4-dev ablate-b4-holdout ablate-b5 ablate-b5-dev ablate-b5-holdout router-proxy evaluate-router-b1b ablate-b1a ablate-b1a-dev ablate-b1b ablate-b1b-dev ablate-b1b-controlled-a2 ablate-b1b-controlled-a2-dev ablate-b2 ablate-b2-dev clean-generated

setup: $(STAMP)

$(STAMP): $(APP_DIR)/requirements.txt
	@test -x "$(PYTHON)" || "$(PYTHON_BIN)" -m venv "$(VENV_DIR)"
	@"$(PYTHON)" -m pip install --upgrade pip
	@"$(PIP)" install -r "$(APP_DIR)/requirements.txt"
	@touch "$(STAMP)"

run: setup
	@$(MAKE) --no-print-directory verify-index
	@"$(PYTHON)" "$(APP_DIR)/scripts/chat.py"

evaluate: setup
	@$(MAKE) --no-print-directory verify-index
	@"$(PYTHON)" "$(APP_DIR)/scripts/evaluate.py" "$(QUESTIONS)" --output-dir "$(OUTPUT_DIR)"

baseline: setup
	@$(MAKE) --no-print-directory verify-baseline
	@INDEX_DIR="$(APP_DIR)/data/index" B4_RETRIEVAL_MODE=off \
		XROUTER_ENABLED=false B2_ENABLED=false ENABLE_HYDE=false \
		B5_AGENT_ENABLED=false \
		"$(PYTHON)" "$(APP_DIR)/scripts/evaluate.py" "$(APP_DIR)/data/evaluation/public.json" \
		--output "$(APP_DIR)/data/evaluation/baseline/baseline-a0-b0.jsonl" \
		--run-name baseline-a0-b0

test: setup
	@cd "$(APP_DIR)" && ".venv/bin/python" -m pytest -q

verify-index:
	@"$(PYTHON)" "$(APP_DIR)/scripts/verify_artifacts.py" \
		--index-dir "$(A31_INDEX_ROOT)/all"

verify-baseline:
	@"$(PYTHON)" "$(APP_DIR)/scripts/verify_artifacts.py" \
		--index-dir "$(APP_DIR)/data/index"

verify-b4-index:
	@"$(PYTHON)" "$(APP_DIR)/scripts/verify_artifacts.py" \
		--index-dir "$(A31_INDEX_ROOT)/all"

# Graders do not need this target: the built index is shipped. It exists only
# to document the reproducible build command used by the author.
ingest: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/ingest.py" \
		--pdf "$(APP_DIR)/data/raw/techcombank-bao-cao-thuong-nien-2025-vie-update.pdf"

# Author-only research target. It is not part of clean-machine serving because
# the assessment requires the prebuilt indexes to be shipped.
ingest-a1:
	@"$(PADDLE_PYTHON)" "$(APP_DIR)/scripts/ingest_paddleocr_vl.py" \
		--pdf "$(A1_PDF)" --device "$(PADDLE_DEVICE)"

# Builds A2/A3/A4 from the shared 393-fragment OCR checkpoint. No OCR/API call.
ingest-a2-a4: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/build_track_a_indexes.py"

# Deterministically projects A2 paragraph/section vectors; no OCR or embedding call.
build-b1a-index: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/build_b1a_narrative_index.py"

# Reuses the shipped A3 chunks and PaddleOCR-VL checkpoint; no OCR/API call.
build-a31: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/build_a31_index.py" \
		--output-root "$(A31_INDEX_ROOT)"

# Derives the metric-aware representation from the frozen A3.1 chunks. It does
# not rerun PaddleOCR-VL; outputs are written to a different directory.
build-a32: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/build_a32_index.py" \
		--source-index "$(A31_INDEX_ROOT)/all" \
		--output-dir "$(A32_INDEX_ROOT)/all"

benchmark-metric-aware: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/compare_metric_aware.py" \
		"$(APP_DIR)/data/evaluation/metric_confusion.json" \
		--run-label metric-confusion --run-type full

benchmark-metric-multiturn: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/evaluate_metric_multiturn.py"

# Deterministic lexical artifact built from shipped A3.1 chunks; no OCR/embedding/API.
build-bm25: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/build_bm25_index.py" \
		--index-dir "$(A31_INDEX_ROOT)/all"

benchmark-a1a: setup
	@INDEX_DIR="$(A1A_INDEX)" "$(PYTHON)" "$(APP_DIR)/scripts/evaluate.py" \
		"$(APP_DIR)/data/evaluation/public.json" \
		--output "$(APP_DIR)/data/evaluation/experiments/a1a-public.jsonl" \
		--run-name a1a-paddleocr-vl-1.6-plain-fixed

benchmark-a1b: setup
	@INDEX_DIR="$(A1B_INDEX)" "$(PYTHON)" "$(APP_DIR)/scripts/evaluate.py" \
		"$(APP_DIR)/data/evaluation/public.json" \
		--output "$(APP_DIR)/data/evaluation/experiments/a1b-public.jsonl" \
		--run-name a1b-paddleocr-vl-1.6-markdown-fixed

benchmark-a2: setup
	@INDEX_DIR="$(A2_INDEX)" "$(PYTHON)" "$(APP_DIR)/scripts/evaluate.py" \
		"$(APP_DIR)/data/evaluation/public.json" \
		--output "$(APP_DIR)/data/evaluation/experiments/a2-public.jsonl" \
		--run-name a2-paddleocr-vl-1.6-layout-aware

benchmark-a3: setup
	@INDEX_DIR="$(A3_INDEX)" "$(PYTHON)" "$(APP_DIR)/scripts/evaluate.py" \
		"$(APP_DIR)/data/evaluation/public.json" \
		--output "$(APP_DIR)/data/evaluation/experiments/a3-public.jsonl" \
		--run-name a3-paddleocr-vl-1.6-multigranularity

benchmark-a4: setup
	@INDEX_DIR="$(A4_INDEX)" "$(PYTHON)" "$(APP_DIR)/scripts/evaluate.py" \
		"$(APP_DIR)/data/evaluation/public.json" \
		--output "$(APP_DIR)/data/evaluation/experiments/a4-public.jsonl" \
		--run-name a4-selective-cascade

benchmark-a31: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/benchmark_a31_controllers.py" \
		"$(APP_DIR)/data/evaluation/public.json" --run-label public \
		--index-root "$(A31_INDEX_ROOT)"

benchmark-a31-dev: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/benchmark_a31_controllers.py" \
		"$(APP_DIR)/data/evaluation/dev.json" --run-label dev \
		--index-root "$(A31_INDEX_ROOT)"

ablate-hyde: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/ablate_hyde.py" \
		"$(APP_DIR)/data/evaluation/public.json" --run-label public \
		--index-dir "$(A31_INDEX_ROOT)/all"

ablate-hyde-dev: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/ablate_hyde.py" \
		"$(APP_DIR)/data/evaluation/dev.json" --run-label dev \
		--index-dir "$(A31_INDEX_ROOT)/all"

ablate-b4-retrieval: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/ablate_b4.py" \
		"$(APP_DIR)/data/evaluation/dev.json" --run-label dev \
		--run-type retrieval_only --index-dir "$(A31_INDEX_ROOT)/all"

ablate-b4: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/ablate_b4.py" \
		"$(APP_DIR)/data/evaluation/public.json" --run-label public \
		--run-type full --index-dir "$(A31_INDEX_ROOT)/all"

ablate-b4-dev: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/ablate_b4.py" \
		"$(APP_DIR)/data/evaluation/dev.json" --run-label dev \
		--run-type full --index-dir "$(A31_INDEX_ROOT)/all"

ablate-b4-holdout: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/ablate_b4.py" \
		"$(APP_DIR)/data/evaluation/holdout.json" --run-label holdout \
		--run-type full --index-dir "$(A31_INDEX_ROOT)/all"

ablate-b5: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/ablate_b5.py" \
		"$(APP_DIR)/data/evaluation/public.json" --run-label public \
		--index-dir "$(A31_INDEX_ROOT)/all"

ablate-b5-dev: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/ablate_b5.py" \
		"$(APP_DIR)/data/evaluation/dev.json" --run-label dev \
		--index-dir "$(A31_INDEX_ROOT)/all"

ablate-b5-holdout: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/ablate_b5.py" \
		"$(APP_DIR)/data/evaluation/holdout.json" --run-label holdout \
		--index-dir "$(A31_INDEX_ROOT)/all"

ablate-b1a: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/ablate_xrouter_b1a.py" \
		"$(APP_DIR)/data/evaluation/public.json" --run-label public

ablate-b1a-dev: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/ablate_xrouter_b1a.py" \
		"$(APP_DIR)/data/evaluation/dev.json" --run-label dev

# Optional logical port split: answer calls use :8000, B1b routing uses :8001.
# This shares one Qwen model process and therefore does not duplicate VRAM.
router-proxy: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/serve_router_proxy.py" \
		--host 127.0.0.1 --port 8001 --backend http://127.0.0.1:8000

evaluate-router-b1b: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/evaluate_router_b1b.py"

ablate-b1b: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/ablate_xrouter_b1b.py" \
		"$(APP_DIR)/data/evaluation/public.json" --run-label public

ablate-b1b-dev: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/ablate_xrouter_b1b.py" \
		"$(APP_DIR)/data/evaluation/dev.json" --run-label dev

ablate-b1b-controlled-a2: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/ablate_xrouter_b1b.py" \
		"$(APP_DIR)/data/evaluation/public.json" --run-label public \
		--retrieval-config "$(APP_DIR)/config/retrieval_controlled_a2.json"

ablate-b1b-controlled-a2-dev: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/ablate_xrouter_b1b.py" \
		"$(APP_DIR)/data/evaluation/dev.json" --run-label dev \
		--retrieval-config "$(APP_DIR)/config/retrieval_controlled_a2.json"

ablate-b2: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/ablate_b2.py" \
		"$(APP_DIR)/data/evaluation/public.json" --run-label public

ablate-b2-dev: setup
	@"$(PYTHON)" "$(APP_DIR)/scripts/ablate_b2.py" \
		"$(APP_DIR)/data/evaluation/dev.json" --run-label dev

clean-generated:
	@echo "Generated artifacts are retained by default for auditability."
