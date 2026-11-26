#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
archive="$repo/output/idea3_rule_mri_swap"
authority=${IDEA3_AUTHORITY_SOURCE:-"$archive/utsw_authority_grade"}
residual=${IDEA3_RESIDUAL_SOURCE:-"$archive/experiment2"}
base="$repo/output/idea3_role_transition"
mkdir -p -- "$base"
run=$(mktemp -d "$base/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")
log="$run/run.log"
python_bin=${IDEA3_PYTHON:-python}

for file in "$authority/manifest.json" "$authority/image_audit.json" "$residual/authority_oof_predictions.csv"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing input: $file. Set IDEA3_AUTHORITY_SOURCE and IDEA3_RESIDUAL_SOURCE to the archived directories." | tee -a "$log" >&2
    exit 1
  fi
done
if ! command -v "$python_bin" >/dev/null; then
  echo "Python unavailable: $python_bin" | tee -a "$log" >&2
  exit 1
fi
nohup "$python_bin" -u "$repo/scripts/run_idea3_module.py" glioma.cli.idea3_role_transition \
  --authority-source "$authority" --residual-source "$residual" --output "$run" \
  --device "${IDEA3_DEVICE:-cuda}" --batch-size "${IDEA3_BATCH_SIZE:-8}" \
  </dev/null >"$log" 2>&1 &
pid=$!
echo "$pid" >"$run/run.pid"
printf 'PID %s\nLog: %s\nResults: %s\n' "$pid" "$log" "$run"
