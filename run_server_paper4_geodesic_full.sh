#!/usr/bin/env bash
set -Eeuo pipefail

# Full Paper 4 protocol: five matched ablations, three seeds each.
# Use an outer nohup to keep the complete protocol alive after disconnecting.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/UTSW-Glioma}"
METADATA_TSV="${METADATA_TSV:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output/server_runs}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs/server_runs}"
GROUP_NAME="${GROUP_NAME:-paper4_geodesic_full_$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-42 43 44}"
PAPER4_EPOCHS="${PAPER4_EPOCHS:-50}"
PAPER4_NUM_WORKERS="${PAPER4_NUM_WORKERS:-4}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "$METADATA_TSV" ]]; then
  for candidate in \
    "$DATA_ROOT/../UTSW_Glioma_Metadata-2-1.tsv" \
    "/root/autodl-tmp/UTSW_Glioma_Metadata-2-1.tsv"; do
    if [[ -f "$candidate" ]]; then
      METADATA_TSV="$candidate"
      break
    fi
  done
fi

[[ -d "$DATA_ROOT" ]] || {
  echo "[ERROR] DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 1
}
if [[ -n "$METADATA_TSV" && ! -f "$METADATA_TSV" ]]; then
  echo "[ERROR] METADATA_TSV does not exist: $METADATA_TSV" >&2
  exit 1
fi

GROUP_OUTPUT_ROOT="$OUTPUT_ROOT/$GROUP_NAME"
GROUP_LOG_ROOT="$LOG_ROOT/$GROUP_NAME"

run_variant() {
  local variant="$1"
  local fusion_mode="$2"
  local metric_support="$3"
  local disable_graph="$4"

  echo
  echo "[INFO] Paper 4 variant: $variant"
  env \
    DATA_ROOT="$DATA_ROOT" \
    METADATA_TSV="$METADATA_TSV" \
    RUN_NAME="${GROUP_NAME}_${variant}" \
    RUN_OUTPUT_ROOT="$GROUP_OUTPUT_ROOT/$variant" \
    RUN_LOG_ROOT="$GROUP_LOG_ROOT/$variant" \
    VALIDATION_OUTPUT_ROOT="$GROUP_OUTPUT_ROOT/$variant/validation" \
    PAPER_CONFIGS=paper4 \
    PAPER4_PAPER_CONFIG=paper4 \
    PAPER4_FUSION_MODE="$fusion_mode" \
    PAPER4_GEO_METRIC_SUPPORT="$metric_support" \
    PAPER4_DISABLE_FUSION_GRAPH="$disable_graph" \
    PAPER4_EPOCHS="$PAPER4_EPOCHS" \
    PAPER4_NUM_WORKERS="$PAPER4_NUM_WORKERS" \
    SEEDS="$SEEDS" \
    MAX_CASES= \
    PAPER4_ALIGN_MAX_CASES= \
    AGGREGATE_TOPOMOE=0 \
    DRY_RUN="$DRY_RUN" \
    bash "$SCRIPT_DIR/run_server_papers.sh"
}

echo "[INFO] Paper 4 group: $GROUP_NAME"
echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] METADATA_TSV=${METADATA_TSV:-<auto-discovery in code>}"
echo "[INFO] seeds=$SEEDS epochs=$PAPER4_EPOCHS"

run_variant full_geodesic_graph geodesic case_and_anchors 0
run_variant euclidean_graph euclidean case_and_anchors 0
run_variant case_only_metric geodesic case_only 0
run_variant geodesic_no_graph geodesic case_and_anchors 1
run_variant latent_concat concat case_and_anchors 1

case "$DRY_RUN" in
  1|true|TRUE|yes|YES|on|ON)
    echo "[INFO] Dry run complete; aggregation skipped."
    ;;
  *)
    cd "$WORKSPACE_ROOT"
    export PYTHONPATH="$WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    "$PYTHON_BIN" -m glioma.cli.aggregate_geodesic_runs --run_root "$GROUP_OUTPUT_ROOT"
    echo "[INFO] Paper 4 protocol complete: $GROUP_OUTPUT_ROOT"
    ;;
esac
