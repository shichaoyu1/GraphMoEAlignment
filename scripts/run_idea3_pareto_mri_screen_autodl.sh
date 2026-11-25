#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
mode=${1:-screen}
[[ "$mode" == screen || "$mode" == posttrain ]] || { echo 'Use screen or posttrain' >&2; exit 2; }
output=${IDEA3_PARETO_OUTPUT:-"$repo/output/idea3_pareto_mri_screen"}
prior=${IDEA3_MRI_PILOT_ROOT:-"$repo/output/idea3_pareto_mri_pilot_poolfix"}
authority=${IDEA3_AUTHORITY:-"$repo/output/idea3_rule_mri_swap/utsw_authority_grade"}
python_bin=${IDEA3_PYTHON:-python}
device=${IDEA3_DEVICE:-cuda}
seed=${IDEA3_SEED:-42}
bootstrap=${IDEA3_BOOTSTRAP:-10000}
epochs=${IDEA3_READER_EPOCHS:-30}
proposals=${IDEA3_PROPOSALS:-256}
local_steps=${IDEA3_LOCAL_STEPS:-128}
repeats=${IDEA3_RANDOM_REPEATS:-256}
post_steps=${IDEA3_POST_UPDATES:-200}
dry_run=${IDEA3_DRY_RUN:-0}
for value in "$seed" "$bootstrap" "$epochs" "$proposals" "$local_steps" "$repeats" "$post_steps"; do
  [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'Budgets and seed must be nonnegative decimal integers' >&2; exit 2; }
done
(( bootstrap > 0 && epochs > 0 && proposals > 0 && repeats > 0 && post_steps > 0 )) || exit 2
[[ "$dry_run" == 0 || "$dry_run" == 1 ]] || exit 2
[[ "$device" == cpu || "$device" == cuda || "$device" =~ ^cuda:[0-9]+$ ]] || exit 2
read -r -a folds <<< "${IDEA3_FOLDS:-0 1 2 3 4}"
declare -A seen=()
for fold in "${folds[@]}"; do
  [[ "$fold" =~ ^[0-4]$ && -z "${seen[$fold]:-}" ]] || { echo 'Folds must be unique integers in 0..4' >&2; exit 2; }
  seen[$fold]=1
done
mkdir -p "$output/logs"
if [[ "$dry_run" == 0 ]]; then
  command -v flock >/dev/null || { echo 'This server launcher requires flock (util-linux)' >&2; exit 1; }
  exec 6>"$output/.screen.lock"
  flock -n 6 || { echo 'Another screening job is using this output root' >&2; exit 1; }
fi
run() {
  if [[ "$dry_run" == 1 ]]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
module() { run "$python_bin" -u "$repo/scripts/run_idea3_module.py" "$@"; }
token_path() {
  local fold=$1
  local previous="$prior/prepared/seed$seed/fold$fold/tokens.pt"
  if [[ -f "$previous" ]]; then printf '%s' "$previous";
  else printf '%s' "$output/prepared/seed$seed/fold$fold/tokens.pt"; fi
}

if [[ "$mode" == screen ]]; then
  [[ -f "$authority/manifest.json" ]] || { echo "Missing authority manifest: $authority" >&2; exit 1; }
  for fold in "${folds[@]}"; do
    [[ -f "$authority/splits/fold$fold.json" ]] || { echo "Missing authority fold split: $authority/splits/fold$fold.json" >&2; exit 1; }
  done
  if [[ -z "${IDEA3_PREPARED:-}" ]]; then
    for cache in "$prior/prepared_images/experiment1" "$repo/output/idea3_pareto_mri_pilot/prepared_images/experiment1" "$output/prepared_images/experiment1"; do
      if [[ -f "$cache/manifest.json" ]]; then export IDEA3_PREPARED="$cache"; break; fi
    done
  fi
  if [[ -n "${IDEA3_PREPARED:-}" && ! -f "$IDEA3_PREPARED/manifest.json" ]]; then
    echo "Missing prepared images: $IDEA3_PREPARED/manifest.json" >&2; exit 1;
  fi
  # Check existing frozen archives before starting any new reader training.
  ordered=()
  missing=()
  for fold in "${folds[@]}"; do
    if [[ -f "$(token_path "$fold")" ]]; then ordered+=("$fold"); else missing+=("$fold"); fi
  done
  if (( ${#missing[@]} > 0 )) && [[ -z "${IDEA3_PREPARED:-}" ]]; then
    data_root=${IDEA3_DATA_ROOT:-/root/autodl-tmp/dataset/UTSW-Glioma}
    metadata=${IDEA3_METADATA:-/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv}
    [[ -d "$data_root" && -f "$metadata" ]] || {
      echo 'No prepared image cache; set IDEA3_PREPARED or provide IDEA3_DATA_ROOT and IDEA3_METADATA' >&2; exit 1;
    }
    module glioma.cli.idea3 prepare --data-root "$data_root" --metadata "$metadata" --output "$output/prepared_images"
    export IDEA3_PREPARED="$output/prepared_images/experiment1"
  fi
  ordered+=("${missing[@]}")
  # Reuse frozen tokens directly. Calling prepare in an old output would mix source hashes.
  for fold in "${ordered[@]}"; do
    tokens=$(token_path "$fold")
    if [[ ! -f "$tokens" ]]; then
      [[ -f "$authority/manifest.json" && -f "$authority/splits/fold$fold.json" ]] || {
        echo "Missing authority manifest or fold split: $authority (fold $fold)" >&2; exit 1;
      }
      if [[ -n "${IDEA3_PREPARED:-}" ]]; then
        [[ "$dry_run" == 1 || -f "$IDEA3_PREPARED/manifest.json" ]] || { echo "Missing prepared images: $IDEA3_PREPARED" >&2; exit 1; }
      elif [[ ! -f "$output/prepared_images/experiment1/manifest.json" ]]; then
        [[ -d "${IDEA3_DATA_ROOT:-/root/autodl-tmp/dataset/UTSW-Glioma}" &&
           -f "${IDEA3_METADATA:-/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv}" ]] || {
          echo 'No prepared image cache; set IDEA3_PREPARED or provide IDEA3_DATA_ROOT and IDEA3_METADATA' >&2; exit 1;
        }
      fi
      echo "Preparing reader: seed=$seed fold=$fold output=$output/prepared/seed$seed/fold$fold"
      IDEA3_PARETO_OUTPUT="$output" IDEA3_AUTHORITY="$authority" IDEA3_FOLD="$fold" IDEA3_SEED="$seed" \
        run bash "$repo/scripts/run_idea3_pareto_autodl.sh" prepare --epochs "$epochs" --device "$device"
    fi
    flags=()
    image_manifest="${IDEA3_PREPARED:-$output/prepared_images/experiment1}/manifest.json"
    if [[ -f "$image_manifest" ]]; then flags+=(--image-manifest "$image_manifest"); fi
    echo "Evaluating frozen screen: seed=$seed fold=$fold tokens=$tokens"
    module glioma.cli.idea3_pareto_screen evaluate --tokens "$tokens" \
      --output "$output/screen/seed$seed/fold$fold" --device "$device" --seed 2026 \
      --proposals "$proposals" --local-steps "$local_steps" --random-repeats "$repeats" --bootstrap "$bootstrap" \
      --authority-source "$authority" --fold "$fold" --training-seed "$seed" "${flags[@]}"
  done
  module glioma.cli.idea3_pareto_screen aggregate --source "$output/screen/seed$seed" \
    --output "$output/analysis/seed$seed" --bootstrap "$bootstrap" --expected-folds "${folds[@]}"
  echo "Screen finished (dry_run=$dry_run): $output/analysis/seed$seed/summary.json"
  echo 'Diagnostic only: no NFT training or full matrix is started by screen.'
else
  # Optional separate command, after reviewing the screen. One fold and one seed only.
  fold=${IDEA3_FOLD:-0}
  [[ "$fold" =~ ^[0-4]$ ]] || exit 2
  tokens=$(token_path "$fold")
  [[ -f "$tokens" ]] || { echo "Missing frozen tokens: $tokens" >&2; exit 1; }
  read -r -a families <<< "${IDEA3_FAMILIES:-dense moe}"
  for family in "${families[@]}"; do
    [[ "$family" == dense || "$family" == moe ]] || exit 2
    initial="$prior/main/${family}_sft/seed$seed/fold$fold/best.pt"
    [[ -f "$initial" && -f "$(dirname "$initial")/config.json" ]] || { echo "Missing SFT initializer/config: $initial" >&2; exit 1; }
    for suffix in sft pg nft; do
      flags=()
      if [[ "$suffix" == nft ]]; then flags+=(--nft-objective population_stch); fi
      echo "Matched continuation: method=${family}_$suffix updates=$post_steps initial=$initial"
      module glioma.cli.idea3_pareto train --tokens "$tokens" --init-policy "$initial" \
        --output "$output/posttrain/main/${family}_$suffix/seed$seed/fold$fold" \
        --method "${family}_$suffix" --device "$device" --seed "$seed" --fold "$fold" \
        --warmup 0 --updates "$post_steps" --validate-every 50 "${flags[@]}"
    done
  done
  module glioma.cli.analyze_idea3_pareto --source "$output/posttrain/main" \
    --output "$output/posttrain/analysis" --bootstrap "$bootstrap"
  echo "Matched continuation finished (dry_run=$dry_run): $output/posttrain/analysis/summary.json"
fi
