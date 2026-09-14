#!/bin/sh
set -eu

REPOSITORY_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${PYTHON_BIN:-}" ]; then
  LAUNCHER_PYTHON=$PYTHON_BIN
elif [ -x "$REPOSITORY_ROOT/techcombank_rag/.venv/bin/python" ]; then
  LAUNCHER_PYTHON="$REPOSITORY_ROOT/techcombank_rag/.venv/bin/python"
else
  LAUNCHER_PYTHON=python3
fi

exec "$LAUNCHER_PYTHON" "$REPOSITORY_ROOT/techcombank_rag/scripts/portable_run.py" "$@"
