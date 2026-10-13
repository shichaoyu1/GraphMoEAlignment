#!/usr/bin/env bash
set -Eeuo pipefail

# Supplemental entry point only. Existing scripts and run roots are not modified.
# Defaults to a plan; set EXECUTE=1 to run the selected group.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/UTSW-Glioma}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/output/paper2_revision_20261013}"
GROUP="${GROUP:-cheap}"
EXECUTE="${EXECUTE:-0}"
cd "$WORKSPACE_ROOT"
export PYTHONPATH="$WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
command=("$PYTHON_BIN" -m glioma.cli.run_paper2_revision
  --data_root "$DATA_ROOT" --out_root "$RUN_ROOT" --group "$GROUP"
  --cv_folds "${CV_FOLDS:-5}" --cv_seed "${CV_SEED:-2026}"
  --epochs "${DIRECT_EPOCHS:-40}" --router_epochs "${ROUTER_EPOCHS:-15}"
  --joint_epochs "${JOINT_EPOCHS:-10}" --batch_size "${BATCH_SIZE:-4}"
  --num_workers "${NUM_WORKERS:-4}"
  --segmentation_source "${SEGMENTATION_SOURCE:-not_documented}")
read -r -a seeds <<< "${SEEDS:-42}"
command+=(--seeds "${seeds[@]}")
if [[ -n "${METADATA_TSV:-}" ]]; then command+=(--metadata_tsv "$METADATA_TSV"); fi
if [[ -n "${REFERENCE_ROOT:-}" ]]; then command+=(--reference_root "$REFERENCE_ROOT"); fi
if [[ "$EXECUTE" == "1" ]]; then command+=(--execute); fi
if [[ "${SKIP_COMPLETED:-0}" == "1" ]]; then command+=(--skip_completed); fi
"${command[@]}"
