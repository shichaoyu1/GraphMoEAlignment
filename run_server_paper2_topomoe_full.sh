#!/usr/bin/env bash
set -Eeuo pipefail

# Full server run for paper2 TopoMoE-GDI.
#
# Minimal usage on the server:
#   DATA_ROOT=/root/autodl-tmp/UTSW-Glioma bash run_server_paper2_topomoe_full.sh
#
# Optional overrides:
#   PAPER2_EPOCHS=80 BATCH_SIZE=8 bash run_server_paper2_topomoe_full.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/UTSW-Glioma}"

if [[ -z "${METADATA_TSV:-}" ]]; then
  for candidate in \
    "$DATA_ROOT/../UTSW_Glioma_Metadata-2-1.tsv" \
    "/root/autodl-tmp/UTSW_Glioma_Metadata-2-1.tsv"; do
    if [[ -f "$candidate" ]]; then
      export METADATA_TSV="$candidate"
      break
    fi
  done
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
export RUN_NAME="${RUN_NAME:-paper2_topomoe_full_${timestamp}}"
export PAPER_CONFIGS="paper2"

# Full-data run: leave both data and evaluation caps empty.
export MAX_CASES="${MAX_CASES:-}"
export PAPER2_ALIGN_MAX_CASES="${PAPER2_ALIGN_MAX_CASES-}"

export PAPER2_EPOCHS="${PAPER2_EPOCHS:-50}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export SEEDS="${SEEDS:-42 43 44}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output/server_runs}"
export LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs/server_runs}"

# Keep paper2 on the v2 profile: no MRI graph, no private branch, no diffusion.
export PAPER2_PAPER_CONFIG="paper2"
export PAPER2_TOPOMOE_VERSION="${PAPER2_TOPOMOE_VERSION:-v2}"
export PAPER2_TOPO_MODE="${PAPER2_TOPO_MODE:-prior_plus_learned}"
export AGGREGATE_TOPOMOE="${AGGREGATE_TOPOMOE:-1}"

echo "[INFO] Launching full paper2 TopoMoE run"
echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] METADATA_TSV=${METADATA_TSV:-<auto-discovery in code>}"
echo "[INFO] RUN_NAME=$RUN_NAME"
echo "[INFO] PAPER2_EPOCHS=$PAPER2_EPOCHS BATCH_SIZE=$BATCH_SIZE SEEDS=$SEEDS"
echo "[INFO] TopoMoE=$PAPER2_TOPOMOE_VERSION mode=$PAPER2_TOPO_MODE"

bash "$SCRIPT_DIR/run_server_papers.sh"
