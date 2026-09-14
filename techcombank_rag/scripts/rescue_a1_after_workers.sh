#!/usr/bin/env bash
set -euo pipefail

# Wait for the sharded GGUF pass, then fill only any failed/missing checkpoints
# with the official native Paddle backend. This makes an overnight author run
# self-healing without changing the A1 representation or index settings.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
APP_ROOT="$REPO_ROOT/techcombank_rag"
RAW_DIR="$APP_ROOT/data/processed/paddleocr_vl_1_6/raw"
PDF="$APP_ROOT/data/raw/techcombank-bao-cao-thuong-nien-2025-vie-update.pdf"
PADDLE_PYTHON="${PADDLE_PYTHON:-python3}"
EXPECTED=393
RESIDUES="2 10 3 11 18 26 19 27"

worker_alive() {
  for residue in $RESIDUES; do
    pid_file="$REPO_ROOT/a1-r32-$residue.pid"
    if [ -f "$pid_file" ]; then
      pid="$(cat "$pid_file")"
      if ps -p "$pid" -o args= 2>/dev/null | grep -q 'ingest_paddleocr_vl.py'; then
        return 0
      fi
    fi
  done
  return 1
}

while worker_alive; do
  sleep 30
done

count="$(find "$RAW_DIR" -name 'fragment_*.json' -type f | wc -l | tr -d ' ')"
echo "Sharded pass ended with raw=$count/$EXPECTED"
if [ "$count" -lt "$EXPECTED" ]; then
  "$PADDLE_PYTHON" "$APP_ROOT/scripts/ingest_paddleocr_vl.py" \
    --pdf "$PDF" \
    --raw-dir "$RAW_DIR" \
    --device cpu --dpi 160 --num-shards 1 --shard-index 0 \
    --max-attempts 3 --parse-only
fi

final_count="$(find "$RAW_DIR" -name 'fragment_*.json' -type f | wc -l | tr -d ' ')"
test "$final_count" -eq "$EXPECTED"
echo "A1 rescue complete: raw=$final_count/$EXPECTED"
