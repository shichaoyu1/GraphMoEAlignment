#!/usr/bin/env bash
set -euo pipefail

matrix=${1:-verify}
case "$matrix" in
  verify|full|replay) ;;
  *) echo "Usage: bash $0 [verify|full|replay]" >&2; exit 2 ;;
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

# The source is importable as glioma even when the checkout is named GraphMoEAlignment.
mkdir -p -- "$output/import_root"
if [[ ! -e "$output/import_root/glioma" && ! -L "$output/import_root/glioma" ]]; then
  ln -s -- "$repo" "$output/import_root/glioma"
fi
if [[ ! "$output/import_root/glioma" -ef "$repo" ]]; then
  echo "Use an output directory associated with this checkout" >&2
  exit 1
fi
export PYTHONPATH="$output/import_root${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$matrix" == replay ]]; then
  "$python_bin" -u -m glioma.cli.run_idea3_autodl_replay --data-root "$data_root" --metadata "$metadata" \
    --pilot "${IDEA3_PILOT:?Set IDEA3_PILOT to the archived pilot experiment1 directory}" --output "$output/replay"
else
  if [[ ! -f "$output/prepared/experiment1/manifest.json" ]]; then
    "$python_bin" -u -m glioma.cli.idea3 prepare --data-root "$data_root" --metadata "$metadata" \
      --output "$output/prepared"
  else
    "$python_bin" -c 'import hashlib, json, sys; from pathlib import Path
manifest, root, metadata = map(Path, sys.argv[1:])
inputs = json.loads(manifest.read_bytes())["inputs"]
if inputs["data_root"] != str(root.resolve()) or inputs["metadata_sha256"] != hashlib.sha256(metadata.read_bytes()).hexdigest():
    sys.exit("Prepared inputs changed. Choose a fresh IDEA3_OUTPUT directory.")' \
      "$output/prepared/experiment1/manifest.json" "$data_root" "$metadata"
  fi
  "$python_bin" -u -m glioma.cli.train_idea3_matched --prepared "$output/prepared/experiment1" \
    --output "$output/$matrix" --matrix "$matrix" --device cuda
fi
