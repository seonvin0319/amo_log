#!/usr/bin/env bash
# Periodic amo_log ingest+push on ext_csh (crontab unavailable on this host).
set -uo pipefail

RUNTIME_DIR="${AMO_LOG_PERIODIC_DIR:-/home/ext_csh/logs/amo_log_periodic}"
INTERVAL_SEC="${INTERVAL_SEC:-300}"
PID_FILE="${RUNTIME_DIR}/pid"
RUN_LOG="${RUNTIME_DIR}/periodic.log"
PUSH_SCRIPT="${AMO_LOG_ROOT:-/home/ext_csh/amo_log}/scripts/auto_push_ext_csh.sh"

mkdir -p "$RUNTIME_DIR"

if [[ -f "$PID_FILE" ]]; then
  old="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    printf '[%s] already running pid=%s\n' "$(TZ=Asia/Seoul date '+%F %T %Z')" "$old" >>"$RUN_LOG"
    exit 0
  fi
fi

echo $$ >"$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

printf '[%s] periodic amo_log watcher start interval=%ss\n' \
  "$(TZ=Asia/Seoul date '+%F %T %Z')" "$INTERVAL_SEC" >>"$RUN_LOG"

while true; do
  if bash "$PUSH_SCRIPT" >>"$RUN_LOG" 2>&1; then
    printf '[%s] amo_log push ok\n' "$(TZ=Asia/Seoul date '+%F %T %Z')" >>"$RUN_LOG"
  else
    printf '[%s] amo_log push failed\n' "$(TZ=Asia/Seoul date '+%F %T %Z')" >>"$RUN_LOG"
  fi
  sleep "$INTERVAL_SEC"
done
