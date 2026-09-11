#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
qwen_python="${QWEN_PYTHON:-python3}"
qwen_deps="${QWEN_DEPS:-}"
qwen_model_path="${QWEN_MODEL_PATH:-}"
qwen_model_name="${LLM_MODEL:-}"

if [[ -z "$qwen_model_path" ]]; then
  echo "QWEN_MODEL_PATH is required; no machine-specific path is assumed." >&2
  exit 2
fi
if [[ -z "$qwen_model_name" ]]; then
  echo "LLM_MODEL is required; no model name is hard-coded." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${QWEN_GPU_IDS:-2,3}"
if [[ -n "$qwen_deps" ]]; then
  export PYTHONPATH="$qwen_deps:$project_root:${PYTHONPATH:-}"
else
  export PYTHONPATH="$project_root:${PYTHONPATH:-}"
fi
cd "$project_root"

exec "$qwen_python" scripts/serve_qwen.py \
  --model-path "$qwen_model_path" \
  --model-name "$qwen_model_name" \
  --host "${LLM_HOST:-127.0.0.1}" \
  --port "${LLM_PORT:-8000}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
  --max-memory-gib "${QWEN_MAX_MEMORY_GIB:-22}"
