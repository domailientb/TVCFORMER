#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
failures=0
for dataset_script in ETTh1.sh ETTh2.sh ETTm1.sh ETTm2.sh Weather.sh; do
  if ! bash "$SCRIPT_DIR/$dataset_script" "$@"; then
    printf 'Dataset script failed: %s\n' "$dataset_script" >&2
    failures=$((failures + 1))
  fi
done
if (( failures > 0 )); then
  printf '%s dataset script(s) failed. Inspect logs and benchmark_full_runs.csv.\n' "$failures" >&2
  exit 1
fi
