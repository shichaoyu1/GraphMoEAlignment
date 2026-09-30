#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-full}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/output/idea4_5_v3/logs}"
mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/${MODE}_${STAMP}.log"
PID_FILE="$LOG_ROOT/${MODE}_${STAMP}.pid"

nohup bash "$SCRIPT_DIR/run_server_idea4_5.sh" "$MODE" > "$LOG_FILE" 2>&1 &
job_pid=$!
printf '%s\n' "$job_pid" > "$PID_FILE"
printf 'Idea4.5 %s started: PID=%s\nlog=%s\npid_file=%s\n' "$MODE" "$job_pid" "$LOG_FILE" "$PID_FILE"
