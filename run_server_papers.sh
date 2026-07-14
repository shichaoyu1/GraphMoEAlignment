#!/usr/bin/env bash
set -Eeuo pipefail

# Server launcher for glioma semantic-alignment paper runs.
#
# Recommended Paper 2 tuning run:
#   DATA_ROOT=/root/autodl-tmp/UTSW-Glioma \
#   METADATA_TSV=/root/autodl-tmp/UTSW_Glioma_Metadata-2-1.tsv \
#   bash run_server_papers.sh
#
# The safe default runs Paper 2 only. Select other papers explicitly:
#   PAPER_CONFIGS="paper1 paper2 paper3 paper4" bash run_server_papers.sh
#
# Per-paper overrides:
#   PAPER2_EPOCHS=30 PAPER2_TOPO_MODE=prior_only bash run_server_papers.sh
#
# Confirmation run after tuning:
#   SEEDS="42 43 44" PAPER2_EPOCHS=30 bash run_server_papers.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$PROJECT_ROOT/.." && pwd)}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-paper2_topomoe_tune_$(date +%Y%m%d_%H%M%S)}"
PAPER_CONFIGS="${PAPER_CONFIGS:-paper2}"
SEEDS="${SEEDS:-42}"

DATA_ROOT="${DATA_ROOT:-}"
METADATA_TSV="${METADATA_TSV:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/output/server_runs}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/server_runs}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-$OUTPUT_ROOT/$RUN_NAME}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-$LOG_ROOT/$RUN_NAME}"
VALIDATION_OUTPUT_ROOT="${VALIDATION_OUTPUT_ROOT:-$RUN_OUTPUT_ROOT/validation}"

VARIANT="${VARIANT:-full}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
ROI_SIZE="${ROI_SIZE:-96}"
Z_SLICES="${Z_SLICES:-7}"
MAX_CASES="${MAX_CASES:-}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
VAL_RATIO="${VAL_RATIO:-0.1}"
FEAT_DIM="${FEAT_DIM:-256}"
SHARED_DIM="${SHARED_DIM:-128}"
PRIVATE_DIM="${PRIVATE_DIM:-128}"
DIFFUSION_T="${DIFFUSION_T:-20}"
GRAPH_TYPE="${GRAPH_TYPE:-learnable}"
NODE_MODE="${NODE_MODE:-regions}"
TARGET_POLICY="${TARGET_POLICY:-region_rules}"
ALIGNMENT_OBJECTIVE="${ALIGNMENT_OBJECTIVE:-clip}"
MOE_MODULE="${MOE_MODULE:-none}"
TOPO_MODE="${TOPO_MODE:-prior_plus_learned}"
TOPOMOE_VERSION="${TOPOMOE_VERSION:-v2}"
TOPO_EPSILON="${TOPO_EPSILON:-1e-4}"
TOPO_TEMPERATURE="${TOPO_TEMPERATURE:-1.0}"
TOPO_BETA_INIT="${TOPO_BETA_INIT:-0.1}"
ROUTE_MIXTURE="${ROUTE_MIXTURE:-log_prob}"
SPECIALIZE_MARGIN="${SPECIALIZE_MARGIN:-0.05}"
FUSION_MODE="${FUSION_MODE:-geodesic}"
GEO_METRIC_SUPPORT="${GEO_METRIC_SUPPORT:-case_and_anchors}"
GEO_PATH_STEPS="${GEO_PATH_STEPS:-5}"
GEO_GAMMA="${GEO_GAMMA:-0.5}"
GEO_RHO="${GEO_RHO:-0.001}"
GEO_METRIC_ALPHA="${GEO_METRIC_ALPHA:-1.0}"
GEO_GRAPH_TEMPERATURE="${GEO_GRAPH_TEMPERATURE:-1.0}"
GEO_BEND_INIT="${GEO_BEND_INIT:-0.1}"
GEO_WARMUP_EPOCHS="${GEO_WARMUP_EPOCHS:-5}"
GRAPH_TOP_K="${GRAPH_TOP_K:-3}"
ALIGN_MAX_CASES="${ALIGN_MAX_CASES:-}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TEMPERATURE="${TEMPERATURE:-0.07}"

LAMBDA_ROUTE="${LAMBDA_ROUTE:-1.0}"
LAMBDA_TOPO="${LAMBDA_TOPO:-0.05}"
LAMBDA_SPECIALIZE="${LAMBDA_SPECIALIZE:-0.02}"
LAMBDA_ROUTE_BALANCE="${LAMBDA_ROUTE_BALANCE:-0.01}"
LAMBDA_ROUTE_SPARSE="${LAMBDA_ROUTE_SPARSE:-0.0}"
LAMBDA_FAMILY_ROUTE="${LAMBDA_FAMILY_ROUTE:-0.3}"
LAMBDA_WITHIN_ANCHOR="${LAMBDA_WITHIN_ANCHOR:-0.3}"
LAMBDA_TOPO_PRIOR="${LAMBDA_TOPO_PRIOR:-0.05}"
LAMBDA_TOPO_DELTA="${LAMBDA_TOPO_DELTA:-0.001}"
LAMBDA_ANCHOR_FAMILY_BALANCE="${LAMBDA_ANCHOR_FAMILY_BALANCE:-0.05}"
LAMBDA_GEO_ENERGY="${LAMBDA_GEO_ENERGY:-0.1}"
LAMBDA_PATH_SEMANTIC="${LAMBDA_PATH_SEMANTIC:-0.1}"

AUGMENT="${AUGMENT:-1}"
CACHE="${CACHE:-0}"
CPU="${CPU:-0}"
PREFER_REGISTERED="${PREFER_REGISTERED:-0}"
INCLUDE_CLINICAL_ANCHORS="${INCLUDE_CLINICAL_ANCHORS:-0}"
EXCLUDE_PATHOLOGY_ANCHORS="${EXCLUDE_PATHOLOGY_ANCHORS:-0}"
EXCLUDE_MOLECULAR_ANCHORS="${EXCLUDE_MOLECULAR_ANCHORS:-0}"
NO_PRIVATE="${NO_PRIVATE:-0}"
NO_DIFFUSION="${NO_DIFFUSION:-0}"
DISABLE_TOPOLOGY_REFINEMENT="${DISABLE_TOPOLOGY_REFINEMENT:-0}"
DISABLE_FAMILY_BALANCED_ROUTE="${DISABLE_FAMILY_BALANCED_ROUTE:-0}"
DISABLE_FUSION_GRAPH="${DISABLE_FUSION_GRAPH:-0}"
SKIP_INTERVENTIONS="${SKIP_INTERVENTIONS:-0}"

# Paper 2 tuning defaults based on the first full TopoMoE run. Environment
# variables supplied by the caller still take precedence.
PAPER2_EPOCHS="${PAPER2_EPOCHS:-20}"
PAPER2_ALIGN_MAX_CASES="${PAPER2_ALIGN_MAX_CASES:-}"
PAPER2_NUM_WORKERS="${PAPER2_NUM_WORKERS:-4}"
PAPER2_AUGMENT="${PAPER2_AUGMENT:-1}"
PAPER2_TOPO_MODE="${PAPER2_TOPO_MODE:-prior_plus_learned}"
PAPER2_TOPOMOE_VERSION="${PAPER2_TOPOMOE_VERSION:-v2}"
PAPER2_TOPO_EPSILON="${PAPER2_TOPO_EPSILON:-1e-4}"
PAPER2_TOPO_TEMPERATURE="${PAPER2_TOPO_TEMPERATURE:-1.0}"
PAPER2_TOPO_BETA_INIT="${PAPER2_TOPO_BETA_INIT:-0.1}"
PAPER2_ROUTE_MIXTURE="${PAPER2_ROUTE_MIXTURE:-log_prob}"
PAPER2_SPECIALIZE_MARGIN="${PAPER2_SPECIALIZE_MARGIN:-0.05}"
PAPER2_LAMBDA_ROUTE="${PAPER2_LAMBDA_ROUTE:-0.3}"
PAPER2_LAMBDA_TOPO="${PAPER2_LAMBDA_TOPO:-0.05}"
PAPER2_LAMBDA_SPECIALIZE="${PAPER2_LAMBDA_SPECIALIZE:-0.1}"
PAPER2_LAMBDA_ROUTE_BALANCE="${PAPER2_LAMBDA_ROUTE_BALANCE:-0.05}"
PAPER2_LAMBDA_ROUTE_SPARSE="${PAPER2_LAMBDA_ROUTE_SPARSE:-0.0}"
PAPER2_LAMBDA_FAMILY_ROUTE="${PAPER2_LAMBDA_FAMILY_ROUTE:-0.3}"
PAPER2_LAMBDA_WITHIN_ANCHOR="${PAPER2_LAMBDA_WITHIN_ANCHOR:-0.3}"
PAPER2_LAMBDA_TOPO_PRIOR="${PAPER2_LAMBDA_TOPO_PRIOR:-0.05}"
PAPER2_LAMBDA_TOPO_DELTA="${PAPER2_LAMBDA_TOPO_DELTA:-0.001}"
PAPER2_LAMBDA_ANCHOR_FAMILY_BALANCE="${PAPER2_LAMBDA_ANCHOR_FAMILY_BALANCE:-0.05}"

COMMON_EXTRA_ARGS="${COMMON_EXTRA_ARGS:-}"
GPU_LOG_INTERVAL="${GPU_LOG_INTERVAL:-30}"
DRY_RUN="${DRY_RUN:-0}"
AGGREGATE_TOPOMOE="${AGGREGATE_TOPOMOE:-1}"

usage() {
  cat <<'USAGE'
glioma server paper launcher

Required:
  DATA_ROOT=/path/to/UTSW-Glioma

Common optional environment variables:
  METADATA_TSV=/path/to/metadata.tsv
  PAPER_CONFIGS="paper2"
  SEEDS="42 43 44"
  RUN_NAME=my_run
  OUTPUT_ROOT=/root/autodl-tmp/glioma_output
  LOG_ROOT=/root/autodl-tmp/glioma_logs
  EPOCHS=30
  BATCH_SIZE=4
  MAX_CASES=8
  CPU=1
  AUGMENT=0
  DRY_RUN=1

Per-paper override pattern:
  PAPER1_EPOCHS=50
  PAPER2_ENABLED=0
  PAPER3_BATCH_SIZE=2
  PAPER4_FUSION_MODE=geodesic
  PAPER1_EXTRA_ARGS="--temperature 0.05"

Paper 2 TopoMoE controls:
  PAPER2_TOPOMOE_VERSION=v2
  PAPER2_TOPO_MODE=prior_plus_learned
  PAPER2_TOPO_EPSILON=1e-4
  PAPER2_TOPO_TEMPERATURE=1.0
  PAPER2_TOPO_BETA_INIT=0.1
  PAPER2_ROUTE_MIXTURE=log_prob
  PAPER2_LAMBDA_FAMILY_ROUTE=0.3
  PAPER2_LAMBDA_WITHIN_ANCHOR=0.3
  PAPER2_LAMBDA_TOPO_PRIOR=0.05
  PAPER2_LAMBDA_TOPO_DELTA=0.001
  PAPER2_LAMBDA_SPECIALIZE=0.1
  PAPER2_LAMBDA_ANCHOR_FAMILY_BALANCE=0.05
  PAPER2_ALIGN_MAX_CASES=50  # optional debug/evaluation cap; unset means full test split
  PAPER2_VARIANT=graph_shared_only  # MRI graph ablation only

Topology ablations should use separate RUN_NAME values:
  RUN_NAME=p2_prior PAPER2_TOPO_MODE=prior_only bash run_server_papers.sh
  RUN_NAME=p2_learned PAPER2_TOPO_MODE=learned_only bash run_server_papers.sh
  RUN_NAME=p2_hybrid PAPER2_TOPO_MODE=prior_plus_learned bash run_server_papers.sh

Paper profile override:
  PAPER1_PAPER_CONFIG=paper1  # default
  PAPER2_PAPER_CONFIG=paper2  # default
  PAPER3_PAPER_CONFIG=paper3  # default
  PAPER4_PAPER_CONFIG=paper4  # default

Set PAPER*_PAPER_CONFIG=none to fully control graph/MoE/diffusion flags from
the launcher:
  PAPER2_PAPER_CONFIG=none PAPER2_GRAPH_TYPE=learnable PAPER2_MOE_MODULE=graph_moe PAPER2_NO_DIFFUSION=1
USAGE
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

is_true() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

upper_name() {
  echo "$1" | tr '[:lower:]-' '[:upper:]_'
}

paper_cfg() {
  local paper="$1"
  local key="$2"
  local default_value="$3"
  local var_name
  var_name="$(upper_name "${paper}_${key}")"
  printf '%s' "${!var_name:-$default_value}"
}

append_words() {
  local words="$1"
  local -n target_ref="$2"
  if [[ -n "$words" ]]; then
    read -r -a split_words <<< "$words"
    target_ref+=("${split_words[@]}")
  fi
}

append_flag_if_true() {
  local value="$1"
  local flag="$2"
  local -n target_ref="$3"
  if is_true "$value"; then
    target_ref+=("$flag")
  fi
}

start_gpu_logger() {
  if [[ "${GPU_LOG_INTERVAL}" == "0" ]]; then
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] nvidia-smi not found; GPU logging disabled."
    return 0
  fi

  local gpu_log="$RUN_LOG_ROOT/gpu_${RUN_NAME}.csv"
  {
    echo "timestamp,index,name,utilization_gpu,utilization_memory,memory_used_mb,memory_total_mb,power_draw_w,temperature_gpu"
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits || true
      sleep "$GPU_LOG_INTERVAL"
    done
  } >> "$gpu_log" &
  GPU_LOG_PID=$!
  echo "[INFO] GPU logger started: $gpu_log (pid=$GPU_LOG_PID)"
}

stop_gpu_logger() {
  if [[ -n "${GPU_LOG_PID:-}" ]]; then
    kill "$GPU_LOG_PID" >/dev/null 2>&1 || true
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ -n "$DATA_ROOT" ]] || die "DATA_ROOT is required. Example: DATA_ROOT=/root/autodl-tmp/UTSW-Glioma bash run_server_papers.sh"
[[ -d "$DATA_ROOT" ]] || die "DATA_ROOT does not exist: $DATA_ROOT"
if [[ -n "$METADATA_TSV" && ! -f "$METADATA_TSV" ]]; then
  die "METADATA_TSV does not exist: $METADATA_TSV"
fi

mkdir -p "$RUN_OUTPUT_ROOT" "$RUN_LOG_ROOT" "$VALIDATION_OUTPUT_ROOT"
cd "$WORKSPACE_ROOT"
export PYTHONPATH="$WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

SUMMARY_FILE="$RUN_LOG_ROOT/runs.tsv"
echo -e "paper\tseed\tstatus\tout_dir\tlog_file" > "$SUMMARY_FILE"

echo "[INFO] Workspace root: $WORKSPACE_ROOT"
echo "[INFO] Project root:   $PROJECT_ROOT"
echo "[INFO] Output root:    $RUN_OUTPUT_ROOT"
echo "[INFO] Log root:       $RUN_LOG_ROOT"
echo "[INFO] Papers:         $PAPER_CONFIGS"
echo "[INFO] Seeds:          $SEEDS"

trap stop_gpu_logger EXIT
start_gpu_logger

failures=()

for paper in $PAPER_CONFIGS; do
  case "$paper" in
    paper1|paper2|paper3|paper4) ;;
    *) die "Unsupported paper in PAPER_CONFIGS: $paper. Use paper1, paper2, paper3, paper4." ;;
  esac

  enabled="$(paper_cfg "$paper" ENABLED 1)"
  if ! is_true "$enabled"; then
    echo "[INFO] Skip $paper because ${paper^^}_ENABLED=$enabled"
    continue
  fi

  paper_config="$(paper_cfg "$paper" PAPER_CONFIG "$paper")"
  variant="$(paper_cfg "$paper" VARIANT "$VARIANT")"
  epochs="$(paper_cfg "$paper" EPOCHS "$EPOCHS")"
  batch_size="$(paper_cfg "$paper" BATCH_SIZE "$BATCH_SIZE")"
  lr="$(paper_cfg "$paper" LR "$LR")"
  weight_decay="$(paper_cfg "$paper" WEIGHT_DECAY "$WEIGHT_DECAY")"
  roi_size="$(paper_cfg "$paper" ROI_SIZE "$ROI_SIZE")"
  z_slices="$(paper_cfg "$paper" Z_SLICES "$Z_SLICES")"
  max_cases="$(paper_cfg "$paper" MAX_CASES "$MAX_CASES")"
  train_ratio="$(paper_cfg "$paper" TRAIN_RATIO "$TRAIN_RATIO")"
  val_ratio="$(paper_cfg "$paper" VAL_RATIO "$VAL_RATIO")"
  feat_dim="$(paper_cfg "$paper" FEAT_DIM "$FEAT_DIM")"
  shared_dim="$(paper_cfg "$paper" SHARED_DIM "$SHARED_DIM")"
  private_dim="$(paper_cfg "$paper" PRIVATE_DIM "$PRIVATE_DIM")"
  diffusion_t="$(paper_cfg "$paper" DIFFUSION_T "$DIFFUSION_T")"
  graph_type="$(paper_cfg "$paper" GRAPH_TYPE "$GRAPH_TYPE")"
  node_mode="$(paper_cfg "$paper" NODE_MODE "$NODE_MODE")"
  target_policy="$(paper_cfg "$paper" TARGET_POLICY "$TARGET_POLICY")"
  alignment_objective="$(paper_cfg "$paper" ALIGNMENT_OBJECTIVE "$ALIGNMENT_OBJECTIVE")"
  moe_module="$(paper_cfg "$paper" MOE_MODULE "$MOE_MODULE")"
  topo_mode="$(paper_cfg "$paper" TOPO_MODE "$TOPO_MODE")"
  topomoe_version="$(paper_cfg "$paper" TOPOMOE_VERSION "$TOPOMOE_VERSION")"
  topo_epsilon="$(paper_cfg "$paper" TOPO_EPSILON "$TOPO_EPSILON")"
  topo_temperature="$(paper_cfg "$paper" TOPO_TEMPERATURE "$TOPO_TEMPERATURE")"
  topo_beta_init="$(paper_cfg "$paper" TOPO_BETA_INIT "$TOPO_BETA_INIT")"
  route_mixture="$(paper_cfg "$paper" ROUTE_MIXTURE "$ROUTE_MIXTURE")"
  specialize_margin="$(paper_cfg "$paper" SPECIALIZE_MARGIN "$SPECIALIZE_MARGIN")"
  fusion_mode="$(paper_cfg "$paper" FUSION_MODE "$FUSION_MODE")"
  geo_metric_support="$(paper_cfg "$paper" GEO_METRIC_SUPPORT "$GEO_METRIC_SUPPORT")"
  geo_path_steps="$(paper_cfg "$paper" GEO_PATH_STEPS "$GEO_PATH_STEPS")"
  geo_gamma="$(paper_cfg "$paper" GEO_GAMMA "$GEO_GAMMA")"
  geo_rho="$(paper_cfg "$paper" GEO_RHO "$GEO_RHO")"
  geo_metric_alpha="$(paper_cfg "$paper" GEO_METRIC_ALPHA "$GEO_METRIC_ALPHA")"
  geo_graph_temperature="$(paper_cfg "$paper" GEO_GRAPH_TEMPERATURE "$GEO_GRAPH_TEMPERATURE")"
  geo_bend_init="$(paper_cfg "$paper" GEO_BEND_INIT "$GEO_BEND_INIT")"
  geo_warmup_epochs="$(paper_cfg "$paper" GEO_WARMUP_EPOCHS "$GEO_WARMUP_EPOCHS")"
  graph_top_k="$(paper_cfg "$paper" GRAPH_TOP_K "$GRAPH_TOP_K")"
  align_max_cases="$(paper_cfg "$paper" ALIGN_MAX_CASES "$ALIGN_MAX_CASES")"
  num_workers="$(paper_cfg "$paper" NUM_WORKERS "$NUM_WORKERS")"
  grad_clip="$(paper_cfg "$paper" GRAD_CLIP "$GRAD_CLIP")"
  temperature="$(paper_cfg "$paper" TEMPERATURE "$TEMPERATURE")"
  lambda_route="$(paper_cfg "$paper" LAMBDA_ROUTE "$LAMBDA_ROUTE")"
  lambda_topo="$(paper_cfg "$paper" LAMBDA_TOPO "$LAMBDA_TOPO")"
  lambda_specialize="$(paper_cfg "$paper" LAMBDA_SPECIALIZE "$LAMBDA_SPECIALIZE")"
  lambda_route_balance="$(paper_cfg "$paper" LAMBDA_ROUTE_BALANCE "$LAMBDA_ROUTE_BALANCE")"
  lambda_route_sparse="$(paper_cfg "$paper" LAMBDA_ROUTE_SPARSE "$LAMBDA_ROUTE_SPARSE")"
  lambda_family_route="$(paper_cfg "$paper" LAMBDA_FAMILY_ROUTE "$LAMBDA_FAMILY_ROUTE")"
  lambda_within_anchor="$(paper_cfg "$paper" LAMBDA_WITHIN_ANCHOR "$LAMBDA_WITHIN_ANCHOR")"
  lambda_topo_prior="$(paper_cfg "$paper" LAMBDA_TOPO_PRIOR "$LAMBDA_TOPO_PRIOR")"
  lambda_topo_delta="$(paper_cfg "$paper" LAMBDA_TOPO_DELTA "$LAMBDA_TOPO_DELTA")"
  lambda_anchor_family_balance="$(paper_cfg "$paper" LAMBDA_ANCHOR_FAMILY_BALANCE "$LAMBDA_ANCHOR_FAMILY_BALANCE")"
  lambda_geo_energy="$(paper_cfg "$paper" LAMBDA_GEO_ENERGY "$LAMBDA_GEO_ENERGY")"
  lambda_path_semantic="$(paper_cfg "$paper" LAMBDA_PATH_SEMANTIC "$LAMBDA_PATH_SEMANTIC")"

  augment="$(paper_cfg "$paper" AUGMENT "$AUGMENT")"
  cache="$(paper_cfg "$paper" CACHE "$CACHE")"
  cpu="$(paper_cfg "$paper" CPU "$CPU")"
  prefer_registered="$(paper_cfg "$paper" PREFER_REGISTERED "$PREFER_REGISTERED")"
  include_clinical="$(paper_cfg "$paper" INCLUDE_CLINICAL_ANCHORS "$INCLUDE_CLINICAL_ANCHORS")"
  exclude_pathology="$(paper_cfg "$paper" EXCLUDE_PATHOLOGY_ANCHORS "$EXCLUDE_PATHOLOGY_ANCHORS")"
  exclude_molecular="$(paper_cfg "$paper" EXCLUDE_MOLECULAR_ANCHORS "$EXCLUDE_MOLECULAR_ANCHORS")"
  no_private="$(paper_cfg "$paper" NO_PRIVATE "$NO_PRIVATE")"
  no_diffusion="$(paper_cfg "$paper" NO_DIFFUSION "$NO_DIFFUSION")"
  disable_topology_refinement="$(paper_cfg "$paper" DISABLE_TOPOLOGY_REFINEMENT "$DISABLE_TOPOLOGY_REFINEMENT")"
  disable_family_balanced_route="$(paper_cfg "$paper" DISABLE_FAMILY_BALANCED_ROUTE "$DISABLE_FAMILY_BALANCED_ROUTE")"
  disable_fusion_graph="$(paper_cfg "$paper" DISABLE_FUSION_GRAPH "$DISABLE_FUSION_GRAPH")"
  skip_interventions="$(paper_cfg "$paper" SKIP_INTERVENTIONS "$SKIP_INTERVENTIONS")"
  paper_extra_args="$(paper_cfg "$paper" EXTRA_ARGS "")"

  for seed in $SEEDS; do
    out_dir="$RUN_OUTPUT_ROOT/$paper/seed_${seed}"
    log_file="$RUN_LOG_ROOT/${paper}_seed_${seed}.log"
    mkdir -p "$out_dir"

    cmd=(
      "$PYTHON_BIN" -m glioma.cli.train_semantic_alignment
      --data_root "$DATA_ROOT"
      --variant "$variant"
      --paper_config "$paper_config"
      --out_dir "$out_dir"
      --validation_output_root "$VALIDATION_OUTPUT_ROOT"
      --roi_size "$roi_size"
      --z_slices "$z_slices"
      --train_ratio "$train_ratio"
      --val_ratio "$val_ratio"
      --feat_dim "$feat_dim"
      --shared_dim "$shared_dim"
      --private_dim "$private_dim"
      --diffusion_T "$diffusion_t"
      --node_mode "$node_mode"
      --graph_type "$graph_type"
      --target_policy "$target_policy"
      --alignment_objective "$alignment_objective"
      --moe_module "$moe_module"
      --topomoe_version "$topomoe_version"
      --topo_mode "$topo_mode"
      --topo_epsilon "$topo_epsilon"
      --topo_temperature "$topo_temperature"
      --topo_beta_init "$topo_beta_init"
      --route_mixture "$route_mixture"
      --specialize_margin "$specialize_margin"
      --fusion_mode "$fusion_mode"
      --geo_metric_support "$geo_metric_support"
      --geo_path_steps "$geo_path_steps"
      --geo_gamma "$geo_gamma"
      --geo_rho "$geo_rho"
      --geo_metric_alpha "$geo_metric_alpha"
      --geo_graph_temperature "$geo_graph_temperature"
      --geo_bend_init "$geo_bend_init"
      --geo_warmup_epochs "$geo_warmup_epochs"
      --graph_top_k "$graph_top_k"
      --num_workers "$num_workers"
      --epochs "$epochs"
      --batch_size "$batch_size"
      --lr "$lr"
      --weight_decay "$weight_decay"
      --temperature "$temperature"
      --lambda_route "$lambda_route"
      --lambda_topo "$lambda_topo"
      --lambda_specialize "$lambda_specialize"
      --lambda_route_balance "$lambda_route_balance"
      --lambda_route_sparse "$lambda_route_sparse"
      --lambda_family_route "$lambda_family_route"
      --lambda_within_anchor "$lambda_within_anchor"
      --lambda_topo_prior "$lambda_topo_prior"
      --lambda_topo_delta "$lambda_topo_delta"
      --lambda_anchor_family_balance "$lambda_anchor_family_balance"
      --lambda_geo_energy "$lambda_geo_energy"
      --lambda_path_semantic "$lambda_path_semantic"
      --grad_clip "$grad_clip"
      --seed "$seed"
    )

    if [[ -n "$METADATA_TSV" ]]; then
      cmd+=(--metadata_tsv "$METADATA_TSV")
    fi
    if [[ -n "$max_cases" ]]; then
      cmd+=(--max_cases "$max_cases")
    fi
    if [[ -n "$align_max_cases" ]]; then
      cmd+=(--align_max_cases "$align_max_cases")
    fi

    append_flag_if_true "$augment" --augment cmd
    append_flag_if_true "$cache" --cache cmd
    append_flag_if_true "$cpu" --cpu cmd
    append_flag_if_true "$prefer_registered" --prefer_registered cmd
    append_flag_if_true "$include_clinical" --include_clinical_anchors cmd
    append_flag_if_true "$exclude_pathology" --exclude_pathology_anchors cmd
    append_flag_if_true "$exclude_molecular" --exclude_molecular_anchors cmd
    append_flag_if_true "$no_private" --no_private cmd
    append_flag_if_true "$no_diffusion" --no_diffusion cmd
    append_flag_if_true "$disable_topology_refinement" --disable_topology_refinement cmd
    append_flag_if_true "$disable_family_balanced_route" --disable_family_balanced_route cmd
    append_flag_if_true "$disable_fusion_graph" --disable_fusion_graph cmd
    append_flag_if_true "$skip_interventions" --skip_interventions cmd
    append_words "$COMMON_EXTRA_ARGS" cmd
    append_words "$paper_extra_args" cmd

    echo
    echo "[INFO] Starting $paper seed=$seed"
    printf '[INFO] Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    if is_true "$DRY_RUN"; then
      echo -e "$paper\t$seed\tDRY_RUN\t$out_dir\t$log_file" >> "$SUMMARY_FILE"
      continue
    fi

    set +e
    "${cmd[@]}" 2>&1 | tee "$log_file"
    status="${PIPESTATUS[0]}"
    set -e

    if [[ "$status" -eq 0 ]]; then
      echo "[INFO] Finished $paper seed=$seed"
      echo -e "$paper\t$seed\tOK\t$out_dir\t$log_file" >> "$SUMMARY_FILE"
    else
      echo "[ERROR] Failed $paper seed=$seed status=$status"
      echo -e "$paper\t$seed\tFAIL:$status\t$out_dir\t$log_file" >> "$SUMMARY_FILE"
      failures+=("${paper}:seed_${seed}:status_${status}")
    fi
  done
done

echo
echo "[INFO] Run summary: $SUMMARY_FILE"
cat "$SUMMARY_FILE"

if [[ "${#failures[@]}" -gt 0 ]]; then
  echo "[ERROR] Failed runs: ${failures[*]}" >&2
  exit 1
fi

if ! is_true "$DRY_RUN" && is_true "$AGGREGATE_TOPOMOE" && [[ " $PAPER_CONFIGS " == *" paper2 "* ]]; then
  seed_count="$(find "$RUN_OUTPUT_ROOT/paper2" -maxdepth 1 -type d -name 'seed_*' 2>/dev/null | wc -l)"
  if [[ "$seed_count" -ge 2 ]]; then
    echo "[INFO] Aggregating $seed_count Paper 2 seeds"
    "$PYTHON_BIN" -m glioma.cli.aggregate_topomoe_runs --run_root "$RUN_OUTPUT_ROOT"
  else
    echo "[INFO] Multi-seed aggregation skipped; found $seed_count completed seed directory."
  fi
fi

echo "[INFO] All requested paper runs completed."
