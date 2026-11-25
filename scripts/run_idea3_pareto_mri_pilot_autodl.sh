#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
output=${IDEA3_PARETO_OUTPUT:-"$repo/output/idea3_pareto_mri_pilot"}
seed=${IDEA3_SEED:-42}
fold=${IDEA3_FOLD:-0}
authority=${IDEA3_AUTHORITY:-"$repo/output/idea3_rule_mri_swap/utsw_authority_grade"}
data_root=${IDEA3_DATA_ROOT:-/root/autodl-tmp/dataset/UTSW-Glioma}
metadata=${IDEA3_METADATA:-/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv}
reader_epochs=${IDEA3_READER_EPOCHS:-30}
supervised_steps=${IDEA3_SUPERVISED_UPDATES:-4000}
post_steps=${IDEA3_POST_UPDATES:-1000}
for value in "$reader_epochs" "$supervised_steps" "$post_steps"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo 'Training budgets must be positive integers' >&2; exit 2; }
done
(( reader_epochs > 0 && supervised_steps >= 2 && post_steps > 0 )) || exit 2
[[ -f "$authority/manifest.json" ]] || { echo "Missing authority archive: $authority/manifest.json" >&2; exit 1; }
prepared=${IDEA3_PREPARED:-"$output/prepared_images/experiment1"}
if [[ ! -f "$prepared/manifest.json" ]]; then
  [[ -d "$data_root" ]] || { echo "Missing MRI directory: $data_root" >&2; exit 1; }
  [[ -f "$metadata" ]] || { echo "Missing metadata: $metadata" >&2; exit 1; }
fi
mkdir -p "$output/logs"
exec 7>"$output/.pilot.lock"
flock -n 7 || { echo 'An MRI pilot is already using this output directory' >&2; exit 1; }
export IDEA3_PARETO_OUTPUT="$output" IDEA3_AUTHORITY="$authority"
export IDEA3_DATA_ROOT="$data_root" IDEA3_METADATA="$metadata"
export IDEA3_FOLD="$fold" IDEA3_SEED="$seed"
unset IDEA3_FIXED_WEIGHT IDEA3_RUN_TAG
launcher="$repo/scripts/run_idea3_pareto_autodl.sh"
bash "$launcher" prepare --epochs "$reader_epochs"
warmup=$((supervised_steps / 2))
updates=$((supervised_steps - warmup))
for method in direct_sft dense_sft moe_sft; do
  IDEA3_METHOD="$method" bash "$launcher" train --warmup "$warmup" --updates "$updates" --teacher-mode top8
done
for method in dense_nft moe_nft moe_pg; do
  case "$method" in
    dense_nft) initial_method=dense_sft ;;
    *) initial_method=moe_sft ;;
  esac
  initial="$output/main/$initial_method/seed$seed/fold$fold/best.pt"
  IDEA3_METHOD="$method" bash "$launcher" train --init-policy "$initial" --warmup 0 --updates "$post_steps"
done
initial="$output/main/dense_sft/seed$seed/fold$fold/best.pt"
for weight in 0 .5 1; do
  IDEA3_METHOD=dense_pg IDEA3_FIXED_WEIGHT="$weight" bash "$launcher" train \
    --init-policy "$initial" --warmup 0 --updates "$post_steps"
done
bash "$launcher" analyze --bootstrap "${IDEA3_BOOTSTRAP:-10000}"
echo "MRI pilot complete: $output/analysis/summary.json"
