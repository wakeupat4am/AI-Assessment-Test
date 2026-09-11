#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/home/ubuntu/TCB_Test}"
APP_ROOT="$REPO_ROOT/techcombank_rag"
PYTHON="$APP_ROOT/.venv/bin/python"
BUILD_PID_FILE="$REPO_ROOT/a2-a4-build.pid"
RESULTS="$APP_ROOT/data/evaluation/experiments"

while [ -f "$BUILD_PID_FILE" ] && \
  ps -p "$(cat "$BUILD_PID_FILE")" -o args= 2>/dev/null | grep -q 'build_track_a_indexes.py'; do
  sleep 30
done

for spec in \
  "a2:a2_paddleocr_vl_1_6_layout_aware" \
  "a3:a3_paddleocr_vl_1_6_multigranularity" \
  "a4:a4_selective_cascade"; do
  name="${spec%%:*}"
  directory="${spec##*:}"
  "$PYTHON" "$APP_ROOT/scripts/verify_artifacts.py" \
    --index-dir "$APP_ROOT/data/index/$directory"
  for split in public dev; do
    INDEX_DIR="techcombank_rag/data/index/$directory" \
      "$PYTHON" "$APP_ROOT/scripts/evaluate.py" \
      "$APP_ROOT/data/evaluation/$split.json" \
      --output "$RESULTS/$name-$split.jsonl" \
      --run-name "$name-$split"
  done
done

date -u +%FT%TZ > "$RESULTS/A2_A4_AUTOMATIC_DONE"
echo "A2/A3/A4 verification and automatic evaluation complete"
