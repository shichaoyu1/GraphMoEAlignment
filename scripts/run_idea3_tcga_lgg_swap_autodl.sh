#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
base="$repo/output/idea3_tcga_lgg_swap"
archive="$repo/output/idea3_rule_mri_swap"
authority=${IDEA3_AUTHORITY_SOURCE:-"$archive/utsw_authority_grade"}
residual=${IDEA3_RESIDUAL_SOURCE:-"$archive/experiment2"}
inputs=${IDEA3_TCGA_INPUTS:-"$repo/output/tcga_lgg_13_inputs.zip"}
python_bin=${IDEA3_PYTHON:-python}

if (( $# > 1 )) || [[ $# == 1 && $1 != --smoke ]]; then
  echo "Usage: bash $0 [--smoke]" >&2
  exit 2
fi
command -v "$python_bin" >/dev/null
for file in "$inputs" "$authority/manifest.json" "$residual"/fold{0..4}/best.pt \
            "$authority"/fold{0..4}/{learned_joint,acf}/best.pt; do
  [[ -s "$file" ]] || { echo "Missing input: $file" >&2; exit 1; }
done

mkdir -p -- "$base"
run=$(mktemp -d "$base/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")
nohup "$python_bin" -u "$repo/scripts/run_idea3_module.py" glioma.cli.idea3_tcga_lgg_swap \
  --inputs "$inputs" --authority-source "$authority" --residual-source "$residual" \
  --output "$run" --device "${IDEA3_DEVICE:-cuda}" --batch-size "${IDEA3_BATCH_SIZE:-8}" "$@" \
  </dev/null >"$run/run.log" 2>&1 &
pid=$!
echo "$pid" >"$run/run.pid"
printf 'PID %s\nLog: %s/run.log\nResults: %s/summary.json\n' "$pid" "$run" "$run"
