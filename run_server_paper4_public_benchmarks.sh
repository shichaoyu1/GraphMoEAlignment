#!/usr/bin/env bash
set -Eeuo pipefail

# Token-level synthetic, AV-MNIST, and CMU-MOSEI diagnostic launcher.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-synthetic}"
DATA_PATH="${DATA_PATH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output/paper4_public}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42}"
MODEL_SEEDS="${MODEL_SEEDS:-101 102 103 104 105}"
GRAPH_TYPES="${GRAPH_TYPES:-chain}"
REGIMES="${REGIMES:-topology_relevant}"
MISSING_RATES="${MISSING_RATES:-0.0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DRY_RUN="${DRY_RUN:-0}"

case "$DATASET" in synthetic|avmnist|mosei) ;; *) echo "[ERROR] Unsupported DATASET: $DATASET" >&2; exit 1 ;; esac
if [[ "$DATASET" != "synthetic" && ! -f "$DATA_PATH" ]]; then
  echo "[ERROR] DATA_PATH must point to a pretokenized NPZ file for $DATASET" >&2
  exit 1
fi

variants=(spd_learned spd_uniform spd_identity euclidean_learned)

run_one() {
  local variant="$1" split_seed="$2" model_seed="$3" graph="$4" regime="$5" missing="$6"
  local geometry=spd topology=learned
  case "$variant" in
    spd_uniform) topology=uniform ;;
    spd_identity) topology=identity ;;
    euclidean_learned) geometry=euclidean ;;
  esac
  local condition="${DATASET}_${graph}_${regime}_missing_${missing}"
  local out_dir="$OUTPUT_ROOT/$condition/$variant/split_${split_seed}_model_${model_seed}"
  cmd=(
    "$PYTHON_BIN" -m glioma.cli.train_paper4_benchmark
    --dataset "$DATASET"
    --out_dir "$out_dir"
    --geometry "$geometry"
    --local_topology "$topology"
    --split_seed "$split_seed"
    --model_seed "$model_seed"
    --planted_graph "$graph"
    --synthetic_regime "$regime"
    --missing_rate "$missing"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
  )
  if [[ -n "$DATA_PATH" ]]; then cmd+=(--data_path "$DATA_PATH"); fi
  printf '[CMD]'; printf ' %q' "${cmd[@]}"; printf '\n'
  if [[ ! "$DRY_RUN" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then "${cmd[@]}"; fi
}

cd "$WORKSPACE_ROOT"
export PYTHONPATH="$WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
for graph in $GRAPH_TYPES; do
  for regime in $REGIMES; do
    for missing in $MISSING_RATES; do
      for variant in "${variants[@]}"; do
        for split_seed in $SPLIT_SEEDS; do
          for model_seed in $MODEL_SEEDS; do
            run_one "$variant" "$split_seed" "$model_seed" "$graph" "$regime" "$missing"
          done
        done
      done
      condition="${DATASET}_${graph}_${regime}_missing_${missing}"
      if [[ ! "$DRY_RUN" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        "$PYTHON_BIN" -m glioma.cli.aggregate_paper4_diagnostics \
          --run_root "$OUTPUT_ROOT/$condition" \
          --expected_variants "${variants[@]}" \
          --expected_split_seeds $SPLIT_SEEDS \
          --expected_model_seeds $MODEL_SEEDS \
          --learned_variant spd_learned \
          --uniform_variant spd_uniform \
          --identity_variant spd_identity \
          --euclidean_variant euclidean_learned
      fi
    done
  done
done
