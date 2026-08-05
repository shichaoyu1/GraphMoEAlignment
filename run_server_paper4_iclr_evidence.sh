#!/usr/bin/env bash
set -Eeuo pipefail

# Factorial Paper 4 protocol: split seeds and model seeds are never conflated.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/UTSW-Glioma}"
METADATA_TSV="${METADATA_TSV:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output/server_runs}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs/server_runs}"
GROUP_NAME="${GROUP_NAME:-paper4_iclr_factorial_v1}"
STAGE="${STAGE:-screen}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42 43 44}"
SCREEN_MODEL_SEEDS="${SCREEN_MODEL_SEEDS:-101}"
CONFIRM_NEW_MODEL_SEEDS="${CONFIRM_NEW_MODEL_SEEDS:-102 103}"
PAPER4_EPOCHS="${PAPER4_EPOCHS:-50}"
PAPER4_NUM_WORKERS="${PAPER4_NUM_WORKERS:-4}"
BEST_PUBLISHED_BASELINE="${BEST_PUBLISHED_BASELINE:-}"
DRY_RUN="${DRY_RUN:-0}"

[[ -d "$DATA_ROOT" ]] || { echo "[ERROR] DATA_ROOT does not exist: $DATA_ROOT" >&2; exit 1; }
case "$STAGE" in screen|confirm) ;; *) echo "[ERROR] STAGE must be screen or confirm" >&2; exit 1 ;; esac
if [[ "$STAGE" == "confirm" && -z "$BEST_PUBLISHED_BASELINE" ]]; then
  echo "[ERROR] Confirmation requires BEST_PUBLISHED_BASELINE selected by screening validation mAP." >&2
  exit 1
fi
if [[ -n "$BEST_PUBLISHED_BASELINE" ]]; then
  case "$BEST_PUBLISHED_BASELINE" in
    latent_concat|hemis|gmu|mbt_style) ;;
    *) echo "[ERROR] Invalid BEST_PUBLISHED_BASELINE: $BEST_PUBLISHED_BASELINE" >&2; exit 1 ;;
  esac
fi

GROUP_OUTPUT_ROOT="$OUTPUT_ROOT/$GROUP_NAME"
GROUP_LOG_ROOT="$LOG_ROOT/$GROUP_NAME"

run_variant() {
  local variant="$1"
  local backend="$2"
  local geometry="$3"
  local local_topology="$4"
  local upper_topology="$5"
  local disable_families="$6"
  local baseline="$7"
  local model_seeds="$8"
  local skip_interventions="$9"
  echo "[INFO] $variant: split seeds [$SPLIT_SEEDS], model seeds [$model_seeds]"
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
    PAPER4_BASELINE="$baseline" \
    PAPER4_SPD_GEOMETRY="$geometry" \
    PAPER4_SPD_GRAPH_POLICY=cross_budget \
    PAPER4_SPD_LOCAL_CROSS_MASS=0.35 \
    PAPER4_SPD_UPPER_CROSS_MASS=0.40 \
    PAPER4_SPD_REGION_FAMILY_FRACTION=0.625 \
    PAPER4_SPD_LOCAL_TOPOLOGY="$local_topology" \
    PAPER4_SPD_UPPER_TOPOLOGY="$upper_topology" \
    PAPER4_GRAPH_INTERVENTION=none \
    PAPER4_DISABLE_SPD_ANCHOR_FAMILIES="$disable_families" \
    PAPER4_SKIP_INTERVENTIONS="$skip_interventions" \
    PAPER4_EPOCHS="$PAPER4_EPOCHS" \
    PAPER4_NUM_WORKERS="$PAPER4_NUM_WORKERS" \
    SPLIT_SEEDS="$SPLIT_SEEDS" \
    MODEL_SEEDS="$model_seeds" \
    DRY_RUN="$DRY_RUN" \
    bash "$SCRIPT_DIR/run_server_papers.sh"
}

run_core() {
  local model_seeds="$1"
  run_variant spd_cross_graph spd_hierarchical spd learned learned 0 latent_concat "$model_seeds" 0
  run_variant spd_identity_graph spd_hierarchical spd identity identity 0 latent_concat "$model_seeds" 1
  run_variant spd_uniform_graph spd_hierarchical spd uniform uniform 0 latent_concat "$model_seeds" 1
  run_variant spd_upper_only spd_hierarchical spd identity learned 0 latent_concat "$model_seeds" 1
  run_variant spd_ub spd_hierarchical spd identity uniform 0 latent_concat "$model_seeds" 1
  run_variant spd_no_anchor_family spd_hierarchical spd learned learned 1 latent_concat "$model_seeds" 1
  run_variant euclidean_cross_graph spd_hierarchical euclidean learned learned 0 latent_concat "$model_seeds" 1
}

run_baseline() {
  local baseline="$1"
  local model_seeds="$2"
  run_variant "$baseline" published_baseline spd identity identity 1 "$baseline" "$model_seeds" 1
}

cd "$WORKSPACE_ROOT"
export PYTHONPATH="$WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$STAGE" == "screen" ]]; then
  run_core "$SCREEN_MODEL_SEEDS"
  for baseline in latent_concat hemis gmu mbt_style; do
    run_baseline "$baseline" "$SCREEN_MODEL_SEEDS"
  done
  expected=(spd_cross_graph spd_identity_graph spd_uniform_graph spd_upper_only spd_ub spd_no_anchor_family euclidean_cross_graph latent_concat hemis gmu mbt_style)
  read -r -a aggregate_models <<< "$SCREEN_MODEL_SEEDS"
else
  run_core "$CONFIRM_NEW_MODEL_SEEDS"
  run_baseline "$BEST_PUBLISHED_BASELINE" "$CONFIRM_NEW_MODEL_SEEDS"
  expected=(spd_cross_graph spd_identity_graph spd_uniform_graph spd_upper_only spd_ub spd_no_anchor_family euclidean_cross_graph "$BEST_PUBLISHED_BASELINE")
  read -r -a aggregate_models <<< "$SCREEN_MODEL_SEEDS $CONFIRM_NEW_MODEL_SEEDS"
fi

case "$DRY_RUN" in
  1|true|TRUE|yes|YES|on|ON) echo "[INFO] Dry run complete; aggregation skipped." ;;
  *) "$PYTHON_BIN" -m glioma.cli.aggregate_paper4_diagnostics \
       --run_root "$GROUP_OUTPUT_ROOT" \
       --metric map \
       --expected_variants "${expected[@]}" \
       --expected_split_seeds $SPLIT_SEEDS \
       --expected_model_seeds "${aggregate_models[@]}" ;;
esac
