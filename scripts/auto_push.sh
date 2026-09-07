#!/usr/bin/env bash
# Ingest local AMO/APART runs into amo_log, commit if needed, push origin/main.
# Intended for host cron every 2 hours on choi.
set -uo pipefail

export GIT_EXEC_PATH="${GIT_EXEC_PATH:-/usr/lib/git-core}"
export GIT_TEMPLATE_DIR="${GIT_TEMPLATE_DIR:-/usr/share/git-core/templates}"
GIT="${GIT_BIN:-/usr/bin/git}"
ROOT="${AMO_LOG_ROOT:-/home/choi/amo_log}"
LOG="${AMO_LOG_PUSH_LOG:-/home/choi/logs/amo_log_auto_push.log}"
LOCK="${AMO_LOG_PUSH_LOCK:-/home/choi/.amo_log_auto_push.lock}"
PY="${PYTHON_BIN:-/usr/bin/python3}"

mkdir -p /home/choi/logs
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "==== $(TZ=Asia/Seoul date -Is) SKIP (lock busy) ====" >>"$LOG"
  exit 0
fi

log() {
  printf '[%s] %s\n' "$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$LOG"
}

cd "$ROOT" || {
  log "ERROR: cannot cd $ROOT"
  exit 1
}

log "start"

if ! "$PY" scripts/ingest_runs.py >>"$LOG" 2>&1; then
  log "ERROR: ingest_runs.py failed"
  exit 1
fi

if ! "$PY" scripts/build_catalog.py >>"$LOG" 2>&1; then
  log "ERROR: build_catalog.py failed"
  exit 1
fi

if ! "$GIT" pull --rebase --autostash origin main >>"$LOG" 2>&1; then
  log "ERROR: git pull --rebase failed"
  "$GIT" rebase --abort >>"$LOG" 2>&1 || true
  exit 1
fi

"$GIT" add runs catalog docs scripts README.md .gitignore 2>/dev/null || true

if "$GIT" diff --cached --quiet; then
  log "nothing to commit"
  exit 0
fi

MSG="collect: auto ingest $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M %Z')"
export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-$("$GIT" config user.name 2>/dev/null || echo amo_log-cron)}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-$("$GIT" config user.email 2>/dev/null || echo amo_log-cron@choi)}"
export GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-$GIT_AUTHOR_NAME}"
export GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-$GIT_AUTHOR_EMAIL}"

if ! "$GIT" commit -m "$MSG" >>"$LOG" 2>&1; then
  log "ERROR: git commit failed"
  exit 1
fi
log "committed: $MSG"

if ! "$GIT" push origin HEAD:main >>"$LOG" 2>&1; then
  log "push failed; retrying after pull --rebase"
  if ! "$GIT" pull --rebase --autostash origin main >>"$LOG" 2>&1; then
    log "ERROR: git pull --rebase failed on retry"
    "$GIT" rebase --abort >>"$LOG" 2>&1 || true
    exit 1
  fi
  "$GIT" add runs catalog docs scripts README.md .gitignore 2>/dev/null || true
  if ! "$GIT" diff --cached --quiet; then
    "$GIT" commit -m "collect: after rebase $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M %Z')" >>"$LOG" 2>&1 || true
  fi
  if ! "$GIT" push origin HEAD:main >>"$LOG" 2>&1; then
    log "ERROR: git push failed after retry"
    exit 1
  fi
fi

log "DONE ok"
exit 0
