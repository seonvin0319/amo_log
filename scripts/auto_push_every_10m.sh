#!/usr/bin/env bash
# Run amo_log ingest + commit + push to origin/shchoi every 10 minutes.
set -uo pipefail

ROOT="${AMO_LOG_ROOT:-/home/shchoi/amo_log}"
PUSH="${ROOT}/scripts/auto_push.sh"
LOG="${AMO_LOG_10M_LOG:-${ROOT}/.auto_push_10m.log}"
INTERVAL_SEC="${AMO_LOG_PUSH_INTERVAL_SEC:-600}"

mkdir -p "$(dirname "$LOG")"
echo "==== $(TZ=Asia/Seoul date -Is) loop start interval=${INTERVAL_SEC}s pid=$$ ====" >>"$LOG"

while true; do
  echo "==== $(TZ=Asia/Seoul date -Is) tick ====" >>"$LOG"
  if ! bash "$PUSH" >>"$LOG" 2>&1; then
    echo "[$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z')] ERROR: auto_push exit=$?" >>"$LOG"
  fi
  sleep "$INTERVAL_SEC"
done
