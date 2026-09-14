#!/usr/bin/env bash
set -euo pipefail

# Author-side overnight helper. Graders never run ingestion; they consume the
# two prebuilt indexes produced here.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
APP_ROOT="$REPO_ROOT/techcombank_rag"
RAW_DIR="$APP_ROOT/data/processed/paddleocr_vl_1_6/raw"
PDF="$APP_ROOT/data/raw/techcombank-bao-cao-thuong-nien-2025-vie-update.pdf"
PYTHON="$APP_ROOT/.venv/bin/python"
EXPECTED=393

raw_count() {
  find "$RAW_DIR" -name 'fragment_*.json' -type f | wc -l | tr -d ' '
}

echo "Waiting for $EXPECTED atomic PaddleOCR-VL checkpoints"
while [ "$(raw_count)" -lt "$EXPECTED" ]; do
  printf '%s raw=%s/%s\n' "$(date -u +%FT%TZ)" "$(raw_count)" "$EXPECTED"
  sleep 30
done

# Allow parse-only workers to finish their final manifest write.
while pgrep -f 'ingest_paddleocr_vl.py.*--parse-only' >/dev/null; do
  sleep 10
done

cd "$REPO_ROOT"
"$PYTHON" "$APP_ROOT/scripts/ingest_paddleocr_vl.py" \
  --pdf "$PDF" \
  --raw-dir "$RAW_DIR" \
  --skip-parse --device cpu

"$PYTHON" "$APP_ROOT/scripts/verify_artifacts.py" \
  --index-dir "$APP_ROOT/data/index/a1a_paddleocr_vl_1_6_plain_fixed"
"$PYTHON" "$APP_ROOT/scripts/verify_artifacts.py" \
  --index-dir "$APP_ROOT/data/index/a1b_paddleocr_vl_1_6_markdown_fixed"

mkdir -p "$APP_ROOT/data/evaluation/experiments"
INDEX_DIR=techcombank_rag/data/index/a1a_paddleocr_vl_1_6_plain_fixed \
  "$PYTHON" "$APP_ROOT/scripts/evaluate.py" \
  "$APP_ROOT/data/evaluation/public.json" \
  --output "$APP_ROOT/data/evaluation/experiments/a1a-public.jsonl" \
  --run-name a1a-paddleocr-vl-1.6-plain-fixed

INDEX_DIR=techcombank_rag/data/index/a1b_paddleocr_vl_1_6_markdown_fixed \
  "$PYTHON" "$APP_ROOT/scripts/evaluate.py" \
  "$APP_ROOT/data/evaluation/public.json" \
  --output "$APP_ROOT/data/evaluation/experiments/a1b-public.jsonl" \
  --run-name a1b-paddleocr-vl-1.6-markdown-fixed

"$PYTHON" "$APP_ROOT/scripts/compare_document_intelligence.py" \
  --a0 "$APP_ROOT/data/evaluation/baseline/baseline-a0-b0.summary.json" \
  --a1a "$APP_ROOT/data/evaluation/experiments/a1a-public.summary.json" \
  --a1b "$APP_ROOT/data/evaluation/experiments/a1b-public.summary.json" \
  --a0-rows "$APP_ROOT/data/evaluation/baseline/baseline-a0-b0.jsonl" \
  --a1a-rows "$APP_ROOT/data/evaluation/experiments/a1a-public.jsonl" \
  --a1b-rows "$APP_ROOT/data/evaluation/experiments/a1b-public.jsonl" \
  --output "$APP_ROOT/data/evaluation/experiments/a1-comparison.json"

date -u +%FT%TZ > "$APP_ROOT/data/evaluation/experiments/A1_AUTOMATIC_DONE"
echo "A1 automatic finalization complete"
