#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-pilot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output/idea4_5_v3/$MODE}"
VALIDATION_ROOT="${VALIDATION_ROOT:-$SCRIPT_DIR/output/idea4_5_v3/validation/$MODE}"
RUN_TESTS="${RUN_TESTS:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-6000}"
EPOCHS="${EPOCHS:-30}"
AUDIT_SOURCE_BATCHES="${AUDIT_SOURCE_BATCHES:-20}"
AUDIT_BATCH_SIZE="${AUDIT_BATCH_SIZE:-64}"
TRAIN_SEEDS="${TRAIN_SEEDS:-101 102}"
FULL_DATA_SEEDS="${FULL_DATA_SEEDS:-2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020}"
FULL_EDGE_NORMALIZATIONS="${FULL_EDGE_NORMALIZATIONS:-dynamic fixed}"
RUN_ARCH_CONTROLS="${RUN_ARCH_CONTROLS:-1}"
DATA_PATH="${DATA_PATH:-}"
LABEL_MODE="${LABEL_MODE:-sign}"

cd "$WORKSPACE_ROOT"
export PYTHONPATH="$WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_ROOT"

run_tests() {
  mkdir -p "$VALIDATION_ROOT"
  "$PYTHON_BIN" -m unittest glioma.tests.test_idea4_5 -v
  "$PYTHON_BIN" -m glioma.cli.idea4_5 \
    --dataset synthetic --out_dir "$VALIDATION_ROOT/smoke" \
    --num_samples 120 --epochs 2 --batch_size 16 \
    --audit_source_batches 2 --audit_batch_size 8 \
    --audit_checkpoints 0.5 1.0 --audit_optimizers mechanism_sgd \
    --information_condition exact_copy --data_seed 991 --train_seed 991 --cpu
}

if [[ "$RUN_TESTS" == "1" ]]; then
  run_tests
fi
if [[ "$MODE" == "test" ]]; then
  exit 0
fi

PLAN="$OUTPUT_ROOT/run_plan.tsv"
FAILURES="$OUTPUT_ROOT/failed_runs.tsv"
printf 'condition\trich_fraction\tdata_seed\ttrain_seed\tedge_normalization\ttopology\treadout\tout_dir\n' > "$PLAN"
printf 'out_dir\texit_code\n' > "$FAILURES"
failure_count=0

run_one() {
  local condition="$1" fraction="$2" data_seed="$3" train_seed="$4" normalization="$5" dataset="$6"
  local topology="$7" readout="$8"
  local run_name="${condition}_rich_${fraction}_${normalization}_${topology}_${readout}/data_${data_seed}_train_${train_seed}"
  local out_dir="$OUTPUT_ROOT/runs/$run_name"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$condition" "$fraction" "$data_seed" "$train_seed" "$normalization" \
    "$topology" "$readout" "$out_dir" >> "$PLAN"
  if [[ -f "$out_dir/run_summary.json" ]]; then
    printf '[SKIP completed] %s\n' "$run_name"
    return
  fi
  mkdir -p "$out_dir"
  local cmd=(
    "$PYTHON_BIN" -m glioma.cli.idea4_5
    --dataset "$dataset" --out_dir "$out_dir"
    --rich_fraction "$fraction" --edge_normalization "$normalization"
    --data_seed "$data_seed" --split_seed 42 --train_seed "$train_seed"
    --epochs "$EPOCHS" --batch_size 64
    --audit_source_batches "$AUDIT_SOURCE_BATCHES" --audit_batch_size "$AUDIT_BATCH_SIZE"
    --audit_checkpoints 0.1 0.5 0.9 --audit_optimizers mechanism_sgd stateful
    --geometry euclidean --local_topology "$topology" --readout_mode "$readout"
  )
  if [[ "$dataset" == "synthetic" ]]; then
    cmd+=(--information_condition "$condition" --signal_form mean --num_samples "$NUM_SAMPLES")
  else
    cmd+=(--data_path "$DATA_PATH" --label_mode "$LABEL_MODE")
  fi
  printf '[RUN] %s\n' "$run_name"
  if "${cmd[@]}" > "$out_dir/stdout.log" 2>&1; then
    printf '[DONE] %s\n' "$run_name"
  else
    local code=$?
    printf '{"status":"failed","exit_code":%s}\n' "$code" > "$out_dir/run_status.json"
    printf '%s\t%s\n' "$out_dir" "$code" >> "$FAILURES"
    failure_count=$((failure_count + 1))
    printf '[FAILED %s] %s\n' "$code" "$run_name"
  fi
}

if [[ "$MODE" == "pilot" ]]; then
  for data_seed in 1001 1002 1003; do
    for train_seed in $TRAIN_SEEDS; do
      run_one novel_signal 0.5 "$data_seed" "$train_seed" dynamic synthetic uniform nodewise
      run_one novel_signal 0.8 "$data_seed" "$train_seed" dynamic synthetic uniform nodewise
      run_one exact_copy 0.8 "$data_seed" "$train_seed" dynamic synthetic uniform nodewise
      run_one null_signal 0.8 "$data_seed" "$train_seed" dynamic synthetic uniform nodewise
    done
  done
elif [[ "$MODE" == "full" ]]; then
  for data_seed in $FULL_DATA_SEEDS; do
    for train_seed in $TRAIN_SEEDS; do
      for normalization in $FULL_EDGE_NORMALIZATIONS; do
        run_one novel_signal 0.5 "$data_seed" "$train_seed" "$normalization" synthetic uniform nodewise
        run_one novel_signal 0.8 "$data_seed" "$train_seed" "$normalization" synthetic uniform nodewise
        run_one exact_copy 0.8 "$data_seed" "$train_seed" "$normalization" synthetic uniform nodewise
        run_one null_signal 0.8 "$data_seed" "$train_seed" "$normalization" synthetic uniform nodewise
      done
      if [[ "$RUN_ARCH_CONTROLS" == "1" ]]; then
        run_one novel_signal 0.8 "$data_seed" "$train_seed" dynamic synthetic identity nodewise
        run_one novel_signal 0.8 "$data_seed" "$train_seed" dynamic synthetic uniform pooled
      fi
    done
  done
elif [[ "$MODE" == "public" ]]; then
  if [[ -z "$DATA_PATH" ]]; then
    printf 'DATA_PATH is required for public mode\n' >&2
    exit 2
  fi
  for train_seed in $TRAIN_SEEDS; do
    for normalization in $FULL_EDGE_NORMALIZATIONS; do
      run_one observed 0.5 0 "$train_seed" "$normalization" npz uniform nodewise
      run_one observed 0.8 0 "$train_seed" "$normalization" npz uniform nodewise
    done
  done
else
  printf 'Usage: %s {test|pilot|full|public}\n' "$0" >&2
  exit 2
fi

"$PYTHON_BIN" -m glioma.cli.aggregate_idea4_5 --root "$OUTPUT_ROOT" --out_dir "$OUTPUT_ROOT/aggregate"
if (( failure_count > 0 )); then
  printf '%s run(s) failed; inspect %s\n' "$failure_count" "$FAILURES" >&2
  exit 1
fi
