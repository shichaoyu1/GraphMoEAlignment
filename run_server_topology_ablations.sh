#!/usr/bin/env bash
set -Eeuo pipefail

# Re-aggregate an existing hybrid baseline, then run the three remaining
# topology ablations with matched seeds and training settings.
#
# Example:
#   BASELINE_RUN_ROOT=/root/autodl-tmp/glioma/output/server_runs/paper2_topomoe_full \
#   DATA_ROOT=/root/autodl-tmp/UTSW-Glioma \
#   METADATA_TSV=/root/autodl-tmp/UTSW_Glioma_Metadata-2-1.tsv \
#   nohup bash run_server_topology_ablations.sh > topology_ablations.log 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

BASELINE_RUN_ROOT="${BASELINE_RUN_ROOT:-}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/UTSW-Glioma}"
METADATA_TSV="${METADATA_TSV:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output/server_runs}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs/server_runs}"
ABLATION_PREFIX="${ABLATION_PREFIX:-paper2_topology_ablation}"
SEEDS="${SEEDS:-42 43 44}"
PAPER2_EPOCHS="${PAPER2_EPOCHS:-50}"
PAPER2_NUM_WORKERS="${PAPER2_NUM_WORKERS:-4}"
GPU_LOG_INTERVAL="${GPU_LOG_INTERVAL:-30}"
DRY_RUN="${DRY_RUN:-0}"

is_true() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

[[ -n "$BASELINE_RUN_ROOT" ]] || {
  echo "[ERROR] BASELINE_RUN_ROOT must point to the existing hybrid run root." >&2
  exit 1
}
[[ -d "$BASELINE_RUN_ROOT" ]] || {
  echo "[ERROR] Baseline run root does not exist: $BASELINE_RUN_ROOT" >&2
  exit 1
}
[[ -d "$DATA_ROOT" ]] || {
  echo "[ERROR] DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 1
}
if [[ -n "$METADATA_TSV" && ! -f "$METADATA_TSV" ]]; then
  echo "[ERROR] METADATA_TSV does not exist: $METADATA_TSV" >&2
  exit 1
fi

cd "$WORKSPACE_ROOT"
export PYTHONPATH="$WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

aggregate_cmd=(
  "$PYTHON_BIN" -m glioma.cli.aggregate_topomoe_runs
  --run_root "$BASELINE_RUN_ROOT"
  --out_dir "$BASELINE_RUN_ROOT/aggregate"
)
echo "[INFO] Re-aggregating existing hybrid baseline"
printf '[INFO] Command:'
printf ' %q' "${aggregate_cmd[@]}"
printf '\n'
if ! is_true "$DRY_RUN"; then
  "${aggregate_cmd[@]}"
fi

run_ablation() {
  local suffix="$1"
  local topo_mode="$2"
  local disable_refinement="$3"
  local run_name="${ABLATION_PREFIX}_${suffix}"

  echo
  echo "[INFO] Launching topology ablation: $run_name"
  env \
    DATA_ROOT="$DATA_ROOT" \
    METADATA_TSV="$METADATA_TSV" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    LOG_ROOT="$LOG_ROOT" \
    RUN_NAME="$run_name" \
    RUN_OUTPUT_ROOT= \
    RUN_LOG_ROOT= \
    VALIDATION_OUTPUT_ROOT= \
    PAPER_CONFIGS=paper2 \
    SEEDS="$SEEDS" \
    PAPER2_EPOCHS="$PAPER2_EPOCHS" \
    PAPER2_NUM_WORKERS="$PAPER2_NUM_WORKERS" \
    PAPER2_TOPOMOE_VERSION=v2 \
    PAPER2_TOPO_MODE="$topo_mode" \
    PAPER2_DISABLE_TOPOLOGY_REFINEMENT="$disable_refinement" \
    MAX_CASES= \
    PAPER2_ALIGN_MAX_CASES= \
    AGGREGATE_TOPOMOE=1 \
    GPU_LOG_INTERVAL="$GPU_LOG_INTERVAL" \
    DRY_RUN="$DRY_RUN" \
    bash "$SCRIPT_DIR/run_server_papers.sh"
}

run_ablation prior_only prior_only 0
run_ablation learned_only learned_only 0
run_ablation hybrid_no_refinement prior_plus_learned 1

echo "[INFO] All requested topology ablations completed."
