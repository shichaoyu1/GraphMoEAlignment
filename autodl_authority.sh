#!/usr/bin/env bash
# AutoDL container-instance launcher. Run setup once, then start all.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
action="${1:-help}"
default_output="${AUTHORITY_OUTPUT:-/root/autodl-tmp/paper4_authority_v2_summary}"
python_bin="${AUTHORITY_PYTHON:-python}"
threads="${AUTHORITY_THREADS:-4}"
artifact_level="${AUTHORITY_ARTIFACT_LEVEL:-summary}"
checkpoint_retention="${AUTHORITY_CHECKPOINT_RETENTION:-none}"
minimum_free_gb="${AUTHORITY_MIN_FREE_GB:-2}"
if [[ ! "$threads" =~ ^[1-9][0-9]*$ ]]; then
  echo 'Invalid AUTHORITY_THREADS; using 4.' >&2
  threads=4
fi
case "$artifact_level" in summary|audit|full) ;; *) echo 'Invalid AUTHORITY_ARTIFACT_LEVEL; expected summary, audit, or full.' >&2; exit 2;; esac
case "$checkpoint_retention" in all|best|none) ;; *) echo 'Invalid AUTHORITY_CHECKPOINT_RETENTION; expected all, best, or none.' >&2; exit 2;; esac
if [[ ! "$minimum_free_gb" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo 'Invalid AUTHORITY_MIN_FREE_GB; expected a non-negative number.' >&2
  exit 2
fi
# Container images may inherit empty or non-integer OpenMP settings.
export AUTHORITY_THREADS="$threads" OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="$threads" NUMEXPR_NUM_THREADS="$threads"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONPATH="$(dirname -- "$repo")${PYTHONPATH:+:$PYTHONPATH}"

check_runtime() {
  "$python_bin" - <<'PY'
import importlib.util
import sys
print("Python:", sys.executable, sys.version.split()[0], flush=True)
if sys.version_info < (3, 11):
    raise SystemExit("Activate Python 3.11/3.12, then run: bash autodl_authority.sh setup")
missing = [name for name in ("torch", "numpy", "scipy", "matplotlib", "pytest", "cvxpy", "cvxpylayers", "diffcp", "scs")
           if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing packages in this interpreter: " + ", ".join(missing) +
                     ". Run: bash autodl_authority.sh setup (in this same environment).")
import torch
from cvxpylayers.torch import CvxpyLayer
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable in this interpreter; install a driver-compatible CUDA PyTorch build.")
print("GPU:", torch.cuda.get_device_name(0), "Torch:", torch.__version__, flush=True)
PY
}

case "$action" in
  setup)
    "$python_bin" -c 'import sys, torch; assert sys.version_info >= (3,11), "Use Python 3.11+"; assert torch.cuda.is_available(), "Install a driver-compatible CUDA PyTorch build first"; print(sys.executable, torch.__version__, torch.cuda.get_device_name(0))'
    "$python_bin" -m pip install -r "$repo/requirements-authority-server.txt"
    check_runtime
    ;;
  start)
    stage="${2:-all}"
    output="${3:-$default_output}"
    case "$stage" in all|smoke|development|main|attribution|rule-only) ;; *) echo "Unknown stage: $stage" >&2; exit 2;; esac
    [[ "$(basename -- "$repo")" == glioma ]] || { echo 'Clone this repository into a directory named glioma (see AUTODL_AUTHORITY.md).' >&2; exit 2; }
    command -v flock >/dev/null || { echo 'flock is required (Ubuntu util-linux).' >&2; exit 2; }
    python_bin="$("$python_bin" -c 'import sys; print(sys.executable)')"
    mkdir -p -- "$output"
    output="$(cd -- "$output" && pwd)"
    exec 9>"$output/.pipeline.lock"
    flock -n 9 || { echo "A pipeline already owns $output. Use status or logs." >&2; exit 1; }
    # Report missing dependencies before submitting a background process.
    check_runtime
    export AUTHORITY_PYTHON="$python_bin" AUTHORITY_BACKGROUND=1
    export AUTHORITY_ARTIFACT_LEVEL="$artifact_level" AUTHORITY_CHECKPOINT_RETENTION="$checkpoint_retention"
    export AUTHORITY_MIN_FREE_GB="$minimum_free_gb"
    printf 'starting\n' > "$output/pipeline.status"
    nohup bash "$repo/autodl_authority.sh" _run "$stage" "$output" >> "$output/pipeline.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$output/pipeline.pid"
    echo "Background process submitted: $(cat "$output/pipeline.pid")"
    echo "Log: $output/pipeline.log"
    echo "Status: bash '$repo/autodl_authority.sh' status '$output'"
    # FD 9 is inherited by the worker; closing this terminal releases no worker lock.
    ;;
  _run)
    [[ "${AUTHORITY_BACKGROUND:-}" == 1 ]] || { echo 'Use start instead of _run.' >&2; exit 2; }
    stage="$2"; output="$3"
    phase="initializing"
    finish() {
      rc=$?
      if [[ $rc -eq 0 ]]; then printf 'completed: %s\n' "$stage" > "$output/pipeline.status";
      else printf 'failed: phase=%s exit=%s (see pipeline.log)\n' "$phase" "$rc" > "$output/pipeline.status"; fi
      echo "[$(date -Is)] finished stage=$stage exit=$rc"
    }
    trap finish EXIT
    cd -- "$repo"
    export MPLCONFIGDIR="$output/.matplotlib"
    echo "[$(date -Is)] starting stage=$stage python=$python_bin"
    git rev-parse HEAD > "$output/checkout_commit.txt"
    "$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0), "Torch:", torch.__version__)'
    phase=check
    printf 'running: %s\n' "$phase" > "$output/pipeline.status"
    "$python_bin" -m glioma.cli.check_authority --output "$output/preflight"
    if [[ "$stage" == all ]]; then phases=(plan development freeze main attribution rule-only);
    elif [[ "$stage" == development ]]; then phases=(plan development freeze);
    else phases=("$stage"); fi
    for phase in "${phases[@]}"; do
      printf 'running: %s\n' "$phase" > "$output/pipeline.status"
      echo "[$(date -Is)] phase=$phase"
      "$python_bin" -m glioma.cli.run_authority_protocol --phase "$phase" --root "$output" --device cuda \
        --threads "${AUTHORITY_THREADS:-4}" --artifact-level "${AUTHORITY_ARTIFACT_LEVEL:-summary}" \
        --checkpoint-retention "${AUTHORITY_CHECKPOINT_RETENTION:-none}" \
        --minimum-free-gb "${AUTHORITY_MIN_FREE_GB:-2}"
    done
    if [[ "$stage" == all || "$stage" == smoke ]]; then
      phase=aggregate
      printf 'running: %s\n' "$phase" > "$output/pipeline.status"
      if [[ "$stage" == smoke ]]; then
        "$python_bin" -m glioma.cli.aggregate_authority --root "$output" --phases smoke
      else
        "$python_bin" -m glioma.cli.aggregate_authority --root "$output"
      fi
    fi
    ;;
  status)
    output="${2:-$default_output}"
    if [[ ! -d "$output" ]]; then echo "No run directory: $output"; exit 0; fi
    cat "$output/pipeline.status" 2>/dev/null || true
    if [[ -f "$output/.pipeline.lock" ]]; then
      if flock -n "$output/.pipeline.lock" true; then echo 'No active pipeline lock (completed, failed, or interrupted).';
      else echo 'Pipeline lock is active.'; fi
    fi
    ;;
  logs)
    tail -n 60 -f "${2:-$default_output}/pipeline.log"
    ;;
  *)
    echo 'Usage: bash autodl_authority.sh setup'
    echo '       bash autodl_authority.sh start [all|smoke|development|main|attribution|rule-only] [OUTPUT]'
    echo '       bash autodl_authority.sh {status|logs} [OUTPUT]'
    ;;
esac
