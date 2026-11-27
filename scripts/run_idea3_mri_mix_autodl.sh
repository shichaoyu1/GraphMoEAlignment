#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
archive="$repo/output/idea3_rule_mri_swap"
authority=${IDEA3_AUTHORITY_SOURCE:-"$archive/utsw_authority_grade"}
residual=${IDEA3_RESIDUAL_SOURCE:-"$archive/experiment2"}
base="$repo/output/idea3_mri_mix"
python_bin=${IDEA3_PYTHON:-python}
role_file="idea3_role_transition/run_20261126_101228_7zvSUO/mri_pair_predictions.csv"

if (( $# > 1 )) || [[ $# == 1 && $1 != --smoke ]]; then
  echo "Usage: bash $0 [--smoke]" >&2
  exit 2
fi
mkdir -p -- "$base"
run=$(mktemp -d "$base/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")
log="$run/run.log"
command -v "$python_bin" >/dev/null || { echo "Python unavailable: $python_bin" | tee "$log" >&2; exit 1; }
for file in "$authority/manifest.json" "$authority/image_audit.json" "$residual/fold0/best.pt"; do
  [[ -s "$file" ]] || { echo "Missing input: $file. Check IDEA3_AUTHORITY_SOURCE and IDEA3_RESIDUAL_SOURCE." | tee "$log" >&2; exit 1; }
done

pairs=()
if [[ -n "${IDEA3_OOF_PAIRS:-}" ]]; then
  pairs=(--oof-pairs "$IDEA3_OOF_PAIRS")
else
  for candidate in "$repo/output/$role_file" "$repo/output/idea3_pilot_20261124/$role_file"; do
    if [[ -f "$candidate" ]]; then
      pairs=(--oof-pairs "$candidate")
      break
    fi
  done
fi
nohup "$python_bin" -u "$repo/scripts/run_idea3_module.py" glioma.cli.idea3_mri_mix \
  --authority-source "$authority" --residual-source "$residual" --output "$run" \
  --device "${IDEA3_DEVICE:-cuda}" --batch-size "${IDEA3_BATCH_SIZE:-8}" "${pairs[@]}" "$@" \
  </dev/null >"$log" 2>&1 &
pid=$!
echo "$pid" >"$run/run.pid"
printf 'PID %s\nLog: %s\nResults: %s/summary.json\n' "$pid" "$log" "$run"
