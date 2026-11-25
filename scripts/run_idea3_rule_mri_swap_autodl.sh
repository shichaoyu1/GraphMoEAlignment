#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash $0 AUTHORITY_ARCHIVE RESIDUAL_ARCHIVE [OUTPUT_DIR]" >&2
  exit 2
fi

repo=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
output=${3:-"$repo/output/idea3_rule_mri_swap"}
mkdir -p -- "$output"
exec 9>"$output/.run.lock"
flock -n 9 || { echo "An MRI swap run is already using $output" >&2; exit 1; }

exec "${IDEA3_PYTHON:-python}" -u "$repo/scripts/run_idea3_module.py" glioma.cli.idea3_rule_mri_swap \
  --authority-source "$1" --residual-source "$2" --output "$output" \
  --device "${IDEA3_DEVICE:-cuda}" --swap-batch-size "${IDEA3_SWAP_BATCH_SIZE:-8}"
