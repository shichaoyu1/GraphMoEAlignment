#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
base="$repo/output/idea3_waw_swap"
archive="$repo/output/idea3_rule_mri_swap"
authority=${IDEA3_AUTHORITY_SOURCE:-"$archive/utsw_authority_grade"}
residual=${IDEA3_RESIDUAL_SOURCE:-"$archive/experiment2"}
inputs=${WAW_INPUT_DIR:-"$base/inputs_v3"}

if [[ ${1:-} != --worker ]]; then
  mkdir -p -- "$base"
  run=$(mktemp -d "$base/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")
  nohup bash "$0" --worker "$run" </dev/null >"$run/run.log" 2>&1 &
  pid=$!
  echo "$pid" >"$run/run.pid"
  printf 'PID %s\nLog: %s/run.log\nResults: %s\n' "$pid" "$run" "$run"
  exit 0
fi

run=$2
python_bin=${IDEA3_PYTHON:-python}
command -v "$python_bin" >/dev/null
for file in "$authority/manifest.json" "$residual"/fold{0..4}/best.pt \
            "$authority"/fold{0..4}/{learned_joint,acf}/best.pt; do
  [[ -s "$file" ]] || { echo "Missing UTSW archive file: $file" >&2; exit 1; }
done

# The pinned files can also be supplied in WAW_INPUT_DIR for servers without Zenodo access.
mkdir -p -- "$inputs"
(
  flock 9
  cd -- "$inputs"
  while read -r checksum file; do
    if printf '%s  %s\n' "$checksum" "$file" | md5sum --check --status 2>/dev/null; then
      echo "Verified $file"
      continue
    fi
    echo "Fetching WAW v3 $file"
    url="https://zenodo.org/records/22205631/files/$file?download=1"
    if ! curl -fsSL --retry 4 --retry-delay 3 --continue-at - --output "$file.part" "$url" ||
       ! printf '%s  %s.part\n' "$checksum" "$file" | md5sum --check --status; then
      echo "Could not verify resumed download; fetching a fresh copy"
      curl -fsSL --retry 3 --output "$file.part" "$url"
    fi
    printf '%s  %s.part\n' "$checksum" "$file" | md5sum --check
    mv -- "$file.part" "$file"
  done <<'FILES'
7dd4f66797b768853898e77de76cda44 1_0_clinical_molecular_data.csv
cca9e9da6d7cc4a06e0c31bebb86b592 3_1_preop_masks.zip
0d24a3a2c76bc6fa9b5b6468679676b1 2_1_preop_scans_coreg.zip
FILES
) 9>"$base/download.lock"

exec "$python_bin" -u "$repo/scripts/run_idea3_module.py" glioma.cli.idea3_waw_swap \
  --waw-inputs "$inputs" --authority-source "$authority" --residual-source "$residual" \
  --output "$run" --device "${IDEA3_DEVICE:-cuda}" --batch-size "${IDEA3_BATCH_SIZE:-8}"
