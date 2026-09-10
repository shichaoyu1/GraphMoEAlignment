#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$(dirname -- "$package_dir")${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
authority_python="${AUTHORITY_PYTHON:-python}"
phase="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$phase" in
  check)
    "$authority_python" -m glioma.cli.check_authority "$@"
    ;;
  plan|smoke|development|freeze|main|attribution|rule-only)
    "$authority_python" -m glioma.cli.run_authority_protocol --phase "$phase" "$@"
    ;;
  aggregate)
    "$authority_python" -m glioma.cli.aggregate_authority "$@"
    ;;
  prune)
    "$authority_python" -m glioma.cli.prune_authority_outputs "$@"
    ;;
  *)
    echo 'Usage: bash glioma/authority_server.sh {check|plan|smoke|development|freeze|main|attribution|rule-only|aggregate|prune} [options]'
    echo 'Read glioma/PAPER4_AUTHORITY_GUIDE.md for installation, sequential runs and GPU sharding.'
    exit 2
    ;;
esac
