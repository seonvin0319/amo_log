#!/usr/bin/env bash
# Forever loop: run auto_push.sh on an interval (ext_csv has no crontab).
# Default 600s (10m). Override with AMO_LOG_PUSH_INTERVAL_SEC.
# Does not skip validation or force-push; failed ticks are logged and retried next interval.
set -uo pipefail
ROOT="${AMO_LOG_ROOT:-/home/ext_csv/amo_log}"
INTERVAL="${AMO_LOG_PUSH_INTERVAL_SEC:-600}"
LOG="${AMO_LOG_PUSH_LOOP_LOG:-/home/ext_csv/logs/amo_log_auto_push_loop.nohup}"
PIDFILE="${AMO_LOG_PUSH_LOOP_PID:-/home/ext_csv/logs/amo_log_auto_push_loop.pid}"

mkdir -p "$(dirname "$LOG")"
echo $$ >"$PIDFILE"
echo "==== $(TZ=Asia/Seoul date -Is) loop start pid=$$ interval=${INTERVAL}s ====" >>"$LOG"

# First tick immediately, then every INTERVAL.
while true; do
  echo "---- $(TZ=Asia/Seoul date -Is) tick start ----" >>"$LOG"
  if bash "$ROOT/scripts/auto_push.sh" >>"$LOG" 2>&1; then
    echo "---- $(TZ=Asia/Seoul date -Is) tick ok ----" >>"$LOG"
  else
    rc=$?
    echo "---- $(TZ=Asia/Seoul date -Is) tick FAIL rc=${rc} (will retry next interval; no force-push) ----" >>"$LOG"
  fi
  sleep "$INTERVAL"
done
