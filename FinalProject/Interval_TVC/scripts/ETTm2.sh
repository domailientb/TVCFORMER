#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
PROJECT_ROOT="$REPO_ROOT/FinalProject/Interval_TVC"
PYTHON="$REPO_ROOT/.venv/Scripts/python.exe"

"$PYTHON" -u "$PROJECT_ROOT/run_full_experiment.py" \
  --dataset ETTm2 \
  --seeds 2021 2023 2025 \
  --pred_lens 96 192 336 720
