#!/usr/bin/env bash
set -euo pipefail

matrix=${1:-verify}
case "$matrix" in
  verify|screen|full|replay) ;;
  *) echo "Usage: bash $0 [verify|screen|full|replay]" >&2; exit 2 ;;
esac

repo=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
output=${IDEA3_OUTPUT:-/root/autodl-tmp/idea3_matched_v2}
data_root=${IDEA3_DATA_ROOT:-/root/autodl-tmp/dataset/UTSW-Glioma}
metadata=${IDEA3_METADATA:-/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv}
python_bin=${IDEA3_PYTHON:-python}
mkdir -p -- "$output"
output=$(cd -- "$output" && pwd -P)
exec 9>"$output/.run.lock"
flock -n 9 || { echo "An Idea 3 run is already using $output" >&2; exit 1; }

# Remove only the link created by the previous launcher, while holding its run lock.
legacy_link="$output/import_root/glioma"
if [[ ! -L "$output/import_root" && -L "$legacy_link" && "$legacy_link" -ef "$repo" ]]; then
  unlink -- "$legacy_link"
  rmdir -- "$output/import_root" 2>/dev/null || true
fi

run_module() {
  "$python_bin" -u "$repo/scripts/run_idea3_module.py" "$@"
}

if [[ "$matrix" == replay ]]; then
  pilot=${IDEA3_PILOT:?Set IDEA3_PILOT to the archived pilot experiment1 directory}
  replay_root="$output/replay"
  if [[ ! -f "$replay_root/prepared/experiment1/manifest.json" ]]; then
    run_module glioma.cli.idea3 prepare --data-root "$data_root" --metadata "$metadata" \
      --output "$replay_root/prepared"
  fi
  run_module glioma.cli.replay_idea3_anchor --pilot "$pilot" --prepared "$replay_root/prepared/experiment1" \
    --output "$replay_root/replay" --device cuda
  for method in probe refit_raw; do
    run_module glioma.cli.diagnose_idea3_anchor --predictions "$replay_root/replay/replay_predictions.csv" \
      --candidate-method "$method" --output "$replay_root/diagnosis_$method"
  done
else
  if [[ ! -f "$output/prepared/experiment1/manifest.json" ]]; then
    run_module glioma.cli.idea3 prepare --data-root "$data_root" --metadata "$metadata" \
      --output "$output/prepared"
  else
    "$python_bin" -c 'import hashlib, json, sys; from pathlib import Path
manifest, root, metadata = map(Path, sys.argv[1:])
inputs = json.loads(manifest.read_bytes())["inputs"]
if inputs["data_root"] != str(root.resolve()) or inputs["metadata_sha256"] != hashlib.sha256(metadata.read_bytes()).hexdigest():
    sys.exit("Prepared inputs changed. Choose a fresh IDEA3_OUTPUT directory.")' \
      "$output/prepared/experiment1/manifest.json" "$data_root" "$metadata"
  fi
  if [[ "$matrix" == screen ]]; then
    run_module glioma.cli.screen_idea3_matched --prepared "$output/prepared/experiment1" --output "$output"
  else
    run_module glioma.cli.train_idea3_matched --prepared "$output/prepared/experiment1" \
      --output "$output/$matrix" --matrix "$matrix" --device cuda
  fi
fi
