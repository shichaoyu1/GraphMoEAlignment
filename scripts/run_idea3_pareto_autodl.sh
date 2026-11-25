#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
mode=${1:?Use controlled, controlled-grid, prepare, prepare-grid, train, ablation, reference, matrix, authority, or analyze}
shift
output=${IDEA3_PARETO_OUTPUT:-"$repo/output/idea3_pareto_v1"}
python_bin=${IDEA3_PYTHON:-python}
mkdir -p "$output/logs"
run_module() { "$python_bin" -u "$repo/scripts/run_idea3_module.py" glioma.cli.idea3_pareto "$@"; }
case "$mode" in
  controlled)
    task=${IDEA3_TASK:-sensors}
    method=${IDEA3_METHOD:-moe_pg}
    seed=${IDEA3_SEED:-42}
    run="$output/controlled/$task/$method/seed$seed"
    mkdir -p "$run"
    exec 9>"$run/.lock"
    flock -n 9 || exit 1
    run_module controlled --output "$run" --task "$task" --method "$method" --seed "$seed" "$@" 2>&1 | tee -a "$run/run.log"
    ;;
  prepare)
    prepared=${IDEA3_PREPARED:-"$output/prepared_images/experiment1"}
    authority=${IDEA3_AUTHORITY:?Set IDEA3_AUTHORITY to original authority archive}
    [[ -f "$authority/manifest.json" ]] || { echo "Missing authority manifest: $authority" >&2; exit 1; }
    if [[ ! -f "$prepared/manifest.json" ]]; then
      [[ -z "${IDEA3_PREPARED:-}" ]] || { echo "Missing prepared manifest: $prepared" >&2; exit 1; }
      mkdir -p "$output/prepared_images"
      exec 8>"$output/prepared_images/.lock"
      flock 8
      if [[ ! -f "$prepared/manifest.json" ]]; then
        "$python_bin" -u "$repo/scripts/run_idea3_module.py" glioma.cli.idea3 prepare \
          --data-root "${IDEA3_DATA_ROOT:-/root/autodl-tmp/dataset/UTSW-Glioma}" \
          --metadata "${IDEA3_METADATA:-/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv}" \
          --output "$output/prepared_images"
      fi
      flock -u 8
    fi
    fold=${IDEA3_FOLD:-0}
    seed=${IDEA3_SEED:-42}
    run="$output/prepared/seed$seed/fold$fold"
    mkdir -p "$run"
    exec 9>"$run/.lock"
    flock -n 9 || exit 1
    run_module prepare --prepared "$prepared" --authority-source "$authority" --output "$run" --fold "$fold" --seed "$seed" "$@" 2>&1 | tee -a "$run/run.log"
    ;;
  train)
    fold=${IDEA3_FOLD:-0}
    seed=${IDEA3_SEED:-42}
    method=${IDEA3_METHOD:-moe_pg}
    tokens=${IDEA3_TOKENS:-"$output/prepared/seed$seed/fold$fold/tokens.pt"}
    run="$output/main/$method/seed$seed/fold$fold"
    if [[ -n "${IDEA3_RUN_TAG:-}" ]]; then
      run="$run/$IDEA3_RUN_TAG"
    fi
    if [[ -n "${IDEA3_FIXED_WEIGHT:-}" ]]; then
      run="$run/fixed_${IDEA3_FIXED_WEIGHT}"
      set -- --fixed-weight "$IDEA3_FIXED_WEIGHT" "$@"
    fi
    mkdir -p "$run"
    exec 9>"$run/.lock"
    flock -n 9 || exit 1
    run_module train --tokens "$tokens" --output "$run" --method "$method" --seed "$seed" --fold "$fold" "$@" 2>&1 | tee -a "$run/run.log"
    ;;
  ablation)
    for method in direct_sft direct_pg dense_sft moe_sft dense_pg moe_pg dense_nft moe_nft moe_linear; do
      IDEA3_METHOD="$method" bash "$0" train "$@"
    done
    ;;
  reference)
    for weight in 0 .25 .5 .75 1; do
      IDEA3_METHOD=dense_pg IDEA3_FIXED_WEIGHT="$weight" bash "$0" train "$@"
    done
    ;;
  controlled-grid)
    for task in nonconvex sensors common; do
      for method in direct_sft direct_pg dense_sft moe_sft dense_pg moe_pg dense_nft moe_nft moe_linear; do
        IDEA3_TASK="$task" IDEA3_METHOD="$method" bash "$0" controlled "$@"
      done
    done
    ;;
  matrix)
    for seed in ${IDEA3_SEEDS:-42}; do
      for fold in ${IDEA3_FOLDS:-0 1 2 3 4}; do
        IDEA3_SEED="$seed" IDEA3_FOLD="$fold" bash "$0" ablation "$@"
      done
    done
    ;;
  prepare-grid)
    for seed in ${IDEA3_SEEDS:-42}; do
      for fold in ${IDEA3_FOLDS:-0 1 2 3 4}; do
        IDEA3_SEED="$seed" IDEA3_FOLD="$fold" bash "$0" prepare "$@"
      done
    done
    ;;
  authority)
    fold=${IDEA3_FOLD:-0}
    seed=${IDEA3_SEED:-42}
    method=${IDEA3_METHOD:-moe_pg}
    run="$output/authority/$method/seed$seed/fold$fold"
    mkdir -p "$run"
    exec 9>"$run/.lock"
    flock -n 9 || exit 1
    "$python_bin" -u "$repo/scripts/run_idea3_module.py" glioma.cli.idea3_pareto_authority \
      --tokens "$output/prepared/seed$seed/fold$fold/tokens.pt" \
      --policy "$output/main/$method/seed$seed/fold$fold" \
      --authority-source "${IDEA3_AUTHORITY:?Set IDEA3_AUTHORITY}" \
      --data-root "${IDEA3_DATA_ROOT:?Set IDEA3_DATA_ROOT}" \
      --image-cache "$output/prepared_images/authority_full_view" \
      --output "$run" --fold "$fold" --seed "$seed" "$@" 2>&1 | tee -a "$run/run.log"
    ;;
  analyze)
    "$python_bin" -u "$repo/scripts/run_idea3_module.py" glioma.cli.analyze_idea3_pareto \
      --source "$output/main" --output "$output/analysis" "$@"
    ;;
  *) echo 'Use controlled, controlled-grid, prepare, train, ablation, reference, or matrix' >&2; exit 2 ;;
esac
