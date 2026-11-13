#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$(dirname -- "$PROJECT_ROOT")${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/dataset/UTSW-Glioma}"
METADATA="${METADATA:-$(dirname -- "$DATA_ROOT")/UTSW_Glioma_Metadata-2-1.tsv}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/output/utsw_authority_grade}"
mkdir -p "$OUTPUT"
exec > >(tee -a "$OUTPUT/console.log") 2>&1
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m pytest tests/test_utsw_authority.py -q
"$PYTHON_BIN" -m glioma.cli.utsw_authority prepare --data-root "$DATA_ROOT" --metadata "$METADATA" --output "$OUTPUT"
"$PYTHON_BIN" -m glioma.cli.utsw_authority smoke --data-root "$DATA_ROOT" --output "$OUTPUT" --device cuda
"$PYTHON_BIN" -m glioma.cli.utsw_authority controls --output "$OUTPUT"
# Both fold-0 runs must pass their validation gate before proceeding to folds 1-4.
for fold in 0 1 2 3 4; do
  for variant in acf learned_joint; do
    "$PYTHON_BIN" -m glioma.cli.utsw_authority train --output "$OUTPUT" --fold "$fold" --variant "$variant" --device cuda
  done
done
"$PYTHON_BIN" -m glioma.cli.utsw_authority aggregate --output "$OUTPUT"
"$PYTHON_BIN" -m glioma.cli.package_utsw_authority results --output "$OUTPUT"
echo "Complete: $OUTPUT/results_for_paper.zip"
