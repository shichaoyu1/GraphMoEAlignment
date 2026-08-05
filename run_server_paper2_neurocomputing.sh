#!/usr/bin/env bash
set -Eeuo pipefail

# Five-fold patient-level Idea2 matrix for the Neurocomputing revision.
# Main models use five folds x three initialization seeds. Set FULL_ABLATIONS=1
# to add topology controls, target-policy/supervision audits, and the safe
# direct-logit residual fallback.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/UTSW-Glioma}"
METADATA_TSV="${METADATA_TSV:-}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/output/paper2_neurocomputing}"
SEEDS="${SEEDS:-42 43 44}"
ABLATION_SEEDS="${ABLATION_SEEDS:-42}"
CV_FOLDS="${CV_FOLDS:-5}"
CV_SEED="${CV_SEED:-2026}"
DIRECT_EPOCHS="${DIRECT_EPOCHS:-40}"
ROUTER_EPOCHS="${ROUTER_EPOCHS:-15}"
JOINT_EPOCHS="${JOINT_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FULL_ABLATIONS="${FULL_ABLATIONS:-0}"

[[ -d "$DATA_ROOT" ]] || { echo "[ERROR] Missing DATA_ROOT: $DATA_ROOT" >&2; exit 1; }
if [[ -n "$METADATA_TSV" && ! -f "$METADATA_TSV" ]]; then
  echo "[ERROR] Missing METADATA_TSV: $METADATA_TSV" >&2
  exit 1
fi

cd "$WORKSPACE_ROOT"
export PYTHONPATH="$WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

run_one() {
  local run_name="$1"
  local profile="$2"
  local fold="$3"
  local seed="$4"
  local training_prior="$5"
  local target_policy="$6"
  local disable_supervision="$7"
  local score_mode="$8"
  local out_dir="$RUN_ROOT/$run_name/fold_${fold}/seed_${seed}"
  local command=(
    "$PYTHON_BIN" -m glioma.cli.train_semantic_alignment
    --data_root "$DATA_ROOT"
    --paper_config paper2
    --experiment_profile "$profile"
    --cv_folds "$CV_FOLDS"
    --cv_fold "$fold"
    --cv_seed "$CV_SEED"
    --seed "$seed"
    --training_prior "$training_prior"
    --target_policy "$target_policy"
    --score_mode "$score_mode"
    --direct_stage_epochs "$DIRECT_EPOCHS"
    --router_stage_epochs "$ROUTER_EPOCHS"
    --joint_stage_epochs "$JOINT_EPOCHS"
    --stage_patience 8
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --out_dir "$out_dir"
  )
  if [[ -n "$METADATA_TSV" ]]; then
    command+=(--metadata_tsv "$METADATA_TSV")
  fi
  if [[ "$disable_supervision" == "1" ]]; then
    command+=(--disable_family_balanced_route)
  fi
  echo "[INFO] $run_name fold=$fold seed=$seed"
  "${command[@]}"
}

run_matrix() {
  local run_name="$1"
  local profile="$2"
  local seed_list="$3"
  local training_prior="$4"
  local target_policy="$5"
  local disable_supervision="$6"
  local score_mode="$7"
  for fold in $(seq 0 $((CV_FOLDS - 1))); do
    for seed in $seed_list; do
      run_one "$run_name" "$profile" "$fold" "$seed" "$training_prior" "$target_policy" "$disable_supervision" "$score_mode"
    done
  done
}

run_matrix direct_only direct_only "$SEEDS" empirical region_rules 0 conditional
run_matrix unstructured_family_moe unstructured_family_moe "$SEEDS" empirical region_rules 0 conditional
run_matrix prior_guided_router prior_guided_router "$SEEDS" empirical region_rules 0 conditional

if [[ "$FULL_ABLATIONS" == "1" ]]; then
  run_matrix prior_plus_learned prior_plus_learned "$ABLATION_SEEDS" empirical region_rules 0 conditional
  run_matrix uniform_prior_router prior_guided_router "$ABLATION_SEEDS" uniform region_rules 0 conditional
  run_matrix random_prior_router prior_guided_router "$ABLATION_SEEDS" random region_rules 0 conditional
  run_matrix all_patient_anchors prior_guided_router "$ABLATION_SEEDS" empirical all_patient_anchors 0 conditional
  run_matrix family_supervision_off prior_guided_router "$ABLATION_SEEDS" empirical region_rules 1 conditional
  run_matrix direct_logit_residual prior_guided_router "$ABLATION_SEEDS" empirical region_rules 0 direct_residual
fi

audit_command=(
  "$PYTHON_BIN" -m glioma.cli.audit_paper2_neurocomputing
  --method "direct_only=direct_scores|$RUN_ROOT/direct_only"
  --method "unstructured_family_moe=routed_scores|$RUN_ROOT/unstructured_family_moe"
  --method "prior_guided_router=routed_scores|$RUN_ROOT/prior_guided_router"
  --baseline direct_only
  --routing "region_rules=$RUN_ROOT/prior_guided_router"
  --out_dir "$RUN_ROOT/audit"
  --bootstrap 10000
)
if [[ "$FULL_ABLATIONS" == "1" ]]; then
  audit_command+=(
    --method "prior_plus_learned=routed_scores|$RUN_ROOT/prior_plus_learned"
    --method "uniform_prior_router=routed_scores|$RUN_ROOT/uniform_prior_router"
    --method "random_prior_router=routed_scores|$RUN_ROOT/random_prior_router"
    --routing "all_patient_anchors=$RUN_ROOT/all_patient_anchors"
    --routing "family_supervision_off=$RUN_ROOT/family_supervision_off"
  )
fi
"${audit_command[@]}"

echo "[INFO] Completed. Audit: $RUN_ROOT/audit/SUBMISSION_GATE_REPORT.md"
