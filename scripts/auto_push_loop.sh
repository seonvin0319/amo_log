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

# Replay a local collect commit if another push landed during collection.
recover_push() {
  local git="${GIT_BIN:-git}"
  (
    cd "$ROOT"
    if [[ -f scripts/host_config.sh ]]; then source scripts/host_config.sh; fi
    "$git" fetch --filter=blob:none origin "refs/heads/${AMO_LOG_BRANCH}:refs/remotes/origin/${AMO_LOG_BRANCH}"
    if ! "$git" -c user.name='amo-log collector' -c user.email='amo-log@localhost' rebase "origin/${AMO_LOG_BRANCH}"; then
      "$git" rebase --abort || true
      return 1
    fi
    "$git" push origin "HEAD:refs/heads/${AMO_LOG_BRANCH}"
  )
}

# First tick immediately, then every INTERVAL.
while true; do
  echo "---- $(TZ=Asia/Seoul date -Is) tick start ----" >>"$LOG"
  if bash "$ROOT/scripts/auto_push.sh" >>"$LOG" 2>&1; then
    echo "---- $(TZ=Asia/Seoul date -Is) tick ok ----" >>"$LOG"
  else
    rc=$?
    echo "---- $(TZ=Asia/Seoul date -Is) tick FAIL rc=${rc}; trying rebase push ----" >>"$LOG"
    if recover_push >>"$LOG" 2>&1; then
      echo "---- $(TZ=Asia/Seoul date -Is) tick ok after rebase ----" >>"$LOG"
    else
      echo "---- $(TZ=Asia/Seoul date -Is) tick FAIL rc=${rc} (will retry next interval; no force-push) ----" >>"$LOG"
    fi
  fi
  sleep "$INTERVAL"
done
