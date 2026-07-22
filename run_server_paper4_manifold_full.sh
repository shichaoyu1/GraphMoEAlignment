#!/usr/bin/env bash
set -Eeuo pipefail

# Full Paper 4 hierarchical-manifold protocol: five matched variants, three seeds.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/UTSW-Glioma}"
METADATA_TSV="${METADATA_TSV:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output/server_runs}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs/server_runs}"
GROUP_NAME="${GROUP_NAME:-paper4_hierarchical_spd_v1}"
SEEDS="${SEEDS:-42 43 44}"
PAPER4_EPOCHS="${PAPER4_EPOCHS:-50}"
PAPER4_NUM_WORKERS="${PAPER4_NUM_WORKERS:-4}"
DRY_RUN="${DRY_RUN:-0}"

[[ -d "$DATA_ROOT" ]] || { echo "[ERROR] DATA_ROOT does not exist: $DATA_ROOT" >&2; exit 1; }
if [[ -n "$METADATA_TSV" && ! -f "$METADATA_TSV" ]]; then
  echo "[ERROR] METADATA_TSV does not exist: $METADATA_TSV" >&2
  exit 1
fi

GROUP_OUTPUT_ROOT="$OUTPUT_ROOT/$GROUP_NAME"
GROUP_LOG_ROOT="$LOG_ROOT/$GROUP_NAME"

run_variant() {
  local variant="$1"
  local backend="$2"
  local geometry="$3"
  local disable_upper="$4"
  local disable_families="$5"
  local fusion_mode="$6"

  echo "[INFO] Paper 4 manifold variant: $variant"
  env \
    DATA_ROOT="$DATA_ROOT" \
    METADATA_TSV="$METADATA_TSV" \
    RUN_NAME="${GROUP_NAME}_${variant}" \
    RUN_OUTPUT_ROOT="$GROUP_OUTPUT_ROOT/$variant" \
    RUN_LOG_ROOT="$GROUP_LOG_ROOT/$variant" \
    VALIDATION_OUTPUT_ROOT="$GROUP_OUTPUT_ROOT/$variant/validation" \
    PAPER_CONFIGS=paper4 \
    PAPER4_PAPER_CONFIG=paper4 \
    PAPER4_FUSION_BACKEND="$backend" \
    PAPER4_SPD_GEOMETRY="$geometry" \
    PAPER4_DISABLE_SPD_UPPER_GRAPH="$disable_upper" \
    PAPER4_DISABLE_SPD_ANCHOR_FAMILIES="$disable_families" \
    PAPER4_FUSION_MODE="$fusion_mode" \
    PAPER4_EPOCHS="$PAPER4_EPOCHS" \
    PAPER4_NUM_WORKERS="$PAPER4_NUM_WORKERS" \
    SEEDS="$SEEDS" \
    MAX_CASES= \
    PAPER4_ALIGN_MAX_CASES= \
    AGGREGATE_TOPOMOE=0 \
    DRY_RUN="$DRY_RUN" \
    bash "$SCRIPT_DIR/run_server_papers.sh"
}

run_variant hierarchical_spd_graph spd_hierarchical spd 0 0 geodesic
run_variant euclidean_hierarchical_graph spd_hierarchical euclidean 0 0 geodesic
run_variant spd_local_only spd_hierarchical spd 1 0 geodesic
run_variant spd_no_anchor_family spd_hierarchical spd 0 1 geodesic
run_variant latent_concat vector_geodesic spd 1 1 concat

case "$DRY_RUN" in
  1|true|TRUE|yes|YES|on|ON) echo "[INFO] Dry run complete; aggregation skipped." ;;
  *)
    cd "$WORKSPACE_ROOT"
    export PYTHONPATH="$WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    "$PYTHON_BIN" -m glioma.cli.aggregate_manifold_runs --run_root "$GROUP_OUTPUT_ROOT"
    ;;
esac
