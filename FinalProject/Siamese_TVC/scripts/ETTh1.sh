#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
PYTHON="$ROOT_DIR/.venv/Scripts/python.exe"
if [[ ! -x "$PYTHON" ]]; then PYTHON=python; fi
cd "$ROOT_DIR"
exec "$PYTHON" -u "$PROJECT_DIR/benchmark.py" --datasets ETTh1 "$@"
