#!/usr/bin/env bash
set -Eeuo pipefail

# Two-stage Paper 4 graph/manifold evidence protocol.
# Stage screen: 10 variants x seeds 42/43/44.
# Stage confirm: selected variants add seeds 45/46, then aggregate all five seeds.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/UTSW-Glioma}"
METADATA_TSV="${METADATA_TSV:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output/server_runs}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs/server_runs}"
GROUP_NAME="${GROUP_NAME:-paper4_graph_evidence_v2}"
STAGE="${STAGE:-screen}"
SCREEN_SEEDS="${SCREEN_SEEDS:-42 43 44}"
CONFIRM_NEW_SEEDS="${CONFIRM_NEW_SEEDS:-45 46}"
PAPER4_EPOCHS="${PAPER4_EPOCHS:-50}"
PAPER4_NUM_WORKERS="${PAPER4_NUM_WORKERS:-4}"
DRY_RUN="${DRY_RUN:-0}"
BEST_PUBLISHED_BASELINE="${BEST_PUBLISHED_BASELINE:-}"

[[ -d "$DATA_ROOT" ]] || { echo "[ERROR] DATA_ROOT does not exist: $DATA_ROOT" >&2; exit 1; }
if [[ -n "$METADATA_TSV" && ! -f "$METADATA_TSV" ]]; then
  echo "[ERROR] METADATA_TSV does not exist: $METADATA_TSV" >&2
  exit 1
fi
case "$STAGE" in screen|confirm) ;; *) echo "[ERROR] STAGE must be screen or confirm" >&2; exit 1 ;; esac

GROUP_OUTPUT_ROOT="$OUTPUT_ROOT/$GROUP_NAME"
GROUP_LOG_ROOT="$LOG_ROOT/$GROUP_NAME"

run_variant() {
  local variant="$1"
  local backend="$2"
  local geometry="$3"
  local intervention="$4"
  local disable_upper="$5"
  local disable_families="$6"
  local baseline="$7"
  local seeds="$8"
  local skip_interventions="$9"
  local local_topology="${10:-learned}"
  local upper_topology="${11:-learned}"

  echo "[INFO] Paper 4 graph-evidence variant: $variant (seeds: $seeds)"
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
    PAPER4_GRAPH_INTERVENTION="$intervention" \
    PAPER4_DISABLE_SPD_UPPER_GRAPH="$disable_upper" \
    PAPER4_DISABLE_SPD_ANCHOR_FAMILIES="$disable_families" \
    PAPER4_SKIP_INTERVENTIONS="$skip_interventions" \
    PAPER4_EPOCHS="$PAPER4_EPOCHS" \
    PAPER4_NUM_WORKERS="$PAPER4_NUM_WORKERS" \
    SEEDS="$seeds" \
    MAX_CASES= \
    PAPER4_ALIGN_MAX_CASES= \
    AGGREGATE_TOPOMOE=0 \
    DRY_RUN="$DRY_RUN" \
    bash "$SCRIPT_DIR/run_server_papers.sh"
}

run_spd_set() {
  local seeds="$1"
  run_variant spd_cross_graph spd_hierarchical spd none 0 0 latent_concat "$seeds" 0 learned learned
  run_variant spd_identity_graph spd_hierarchical spd none 0 0 latent_concat "$seeds" 1 identity identity
  run_variant spd_uniform_graph spd_hierarchical spd none 0 0 latent_concat "$seeds" 1 uniform uniform
  run_variant spd_local_only spd_hierarchical spd none 1 0 latent_concat "$seeds" 1 learned identity
  run_variant spd_no_anchor_family spd_hierarchical spd none 0 1 latent_concat "$seeds" 1 learned learned
  run_variant euclidean_cross_graph spd_hierarchical euclidean none 0 0 latent_concat "$seeds" 1 learned learned
}

run_baseline() {
  local baseline="$1"
  local seeds="$2"
  run_variant "$baseline" published_baseline spd none 1 1 "$baseline" "$seeds" 1
}

cd "$WORKSPACE_ROOT"
export PYTHONPATH="$WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$STAGE" == "screen" ]]; then
  run_spd_set "$SCREEN_SEEDS"
  for baseline in latent_concat hemis gmu mbt_style; do
    run_baseline "$baseline" "$SCREEN_SEEDS"
  done
  case "$DRY_RUN" in
    1|true|TRUE|yes|YES|on|ON) echo "[INFO] Screening dry run complete; aggregation skipped." ;;
    *) "$PYTHON_BIN" -m glioma.cli.aggregate_manifold_runs --run_root "$GROUP_OUTPUT_ROOT" --stage screen --expected_seeds 42 43 44 ;;
  esac
else
  if [[ -z "$BEST_PUBLISHED_BASELINE" ]]; then
    aggregate_json="$GROUP_OUTPUT_ROOT/aggregate/aggregate_manifold.json"
    if [[ ! -f "$aggregate_json" && "$DRY_RUN" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
      BEST_PUBLISHED_BASELINE=mbt_style
      echo "[INFO] Confirmation dry run uses placeholder baseline: $BEST_PUBLISHED_BASELINE"
    elif [[ ! -f "$aggregate_json" ]]; then
      "$PYTHON_BIN" -m glioma.cli.aggregate_manifold_runs --run_root "$GROUP_OUTPUT_ROOT" --stage screen --expected_seeds 42 43 44
    fi
    if [[ -z "$BEST_PUBLISHED_BASELINE" ]]; then
      BEST_PUBLISHED_BASELINE="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["best_published_baseline"])' "$aggregate_json")"
    fi
  fi
  case "$BEST_PUBLISHED_BASELINE" in
    latent_concat|hemis|gmu|mbt_style) ;;
    *) echo "[ERROR] Invalid BEST_PUBLISHED_BASELINE: $BEST_PUBLISHED_BASELINE" >&2; exit 1 ;;
  esac
  run_spd_set "$CONFIRM_NEW_SEEDS"
  run_baseline "$BEST_PUBLISHED_BASELINE" "$CONFIRM_NEW_SEEDS"
  case "$DRY_RUN" in
    1|true|TRUE|yes|YES|on|ON) echo "[INFO] Confirmation dry run complete; aggregation skipped." ;;
    *) "$PYTHON_BIN" -m glioma.cli.aggregate_manifold_runs --run_root "$GROUP_OUTPUT_ROOT" --stage confirm --expected_seeds 42 43 44 45 46 ;;
  esac
fi
