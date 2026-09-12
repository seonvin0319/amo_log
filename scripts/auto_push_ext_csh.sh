#!/usr/bin/env bash
# Ingest local AMO/APART runs on ext_csh into amo_log and push origin/ext_csh.
# Periodic: every 10 minutes via scripts/run_amo_log_auto_push_periodic.sh.
set -uo pipefail

export GIT_EXEC_PATH="${GIT_EXEC_PATH:-/usr/lib/git-core}"
export GIT_TEMPLATE_DIR="${GIT_TEMPLATE_DIR:-/usr/share/git-core/templates}"
GIT="${GIT_BIN:-/usr/bin/git}"
ROOT="${AMO_LOG_ROOT:-/home/ext_csh/amo_log}"
LOG="${AMO_LOG_PUSH_LOG:-/home/ext_csh/logs/amo_log_auto_push_ext_csh.log}"
LOCK="${AMO_LOG_PUSH_LOCK:-/home/ext_csh/.amo_log_auto_push_ext_csh.lock}"
PY="${PYTHON_BIN:-/usr/bin/python3}"
BRANCH="${AMO_LOG_BRANCH:-ext_csh}"

mkdir -p "$(dirname "$LOG")"
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

log "start branch=$BRANCH"

# Ensure we are on the target branch (detached/rebase leftovers are fatal).
if ! "$GIT" rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  log "ERROR: local branch $BRANCH missing"
  exit 1
fi
"$GIT" rebase --abort >>"$LOG" 2>&1 || true
if ! "$GIT" checkout "$BRANCH" >>"$LOG" 2>&1; then
  log "ERROR: checkout $BRANCH failed"
  exit 1
fi

if ! "$GIT" fetch origin "$BRANCH" >>"$LOG" 2>&1; then
  log "WARN: fetch origin/$BRANCH failed (continuing with local)"
fi

if "$GIT" rev-parse --verify "origin/$BRANCH" >/dev/null 2>&1; then
  if ! "$GIT" pull --rebase --autostash origin "$BRANCH" >>"$LOG" 2>&1; then
    log "ERROR: git pull --rebase origin/$BRANCH failed"
    "$GIT" rebase --abort >>"$LOG" 2>&1 || true
    exit 1
  fi
fi

if ! "$PY" scripts/ingest_runs.py --host=ext_csh >>"$LOG" 2>&1; then
  log "ERROR: ingest_runs.py failed"
  exit 1
fi

if ! "$PY" scripts/build_catalog.py >>"$LOG" 2>&1; then
  log "ERROR: build_catalog.py failed"
  exit 1
fi

"$GIT" add runs catalog docs scripts README.md .gitignore 2>/dev/null || true

if "$GIT" diff --cached --quiet; then
  log "nothing to commit"
  exit 0
fi

MSG="collect(ext_csh): auto ingest $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M %Z')"
# One-shot identity; do not write git config.
export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-seonvin0319}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-seonvin0319@users.noreply.github.com}"
export GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-$GIT_AUTHOR_NAME}"
export GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-$GIT_AUTHOR_EMAIL}"

if ! "$GIT" -c user.name="$GIT_AUTHOR_NAME" -c user.email="$GIT_AUTHOR_EMAIL" \
  commit -m "$MSG" >>"$LOG" 2>&1; then
  log "ERROR: git commit failed"
  exit 1
fi
log "committed: $MSG"

if ! "$GIT" push -u origin "HEAD:$BRANCH" >>"$LOG" 2>&1; then
  log "push failed; retrying after pull --rebase"
  if ! "$GIT" pull --rebase --autostash origin "$BRANCH" >>"$LOG" 2>&1; then
    log "ERROR: git pull --rebase failed on retry"
    "$GIT" rebase --abort >>"$LOG" 2>&1 || true
    exit 1
  fi
  "$GIT" add runs catalog docs scripts README.md .gitignore 2>/dev/null || true
  if ! "$GIT" diff --cached --quiet; then
    "$GIT" -c user.name="$GIT_AUTHOR_NAME" -c user.email="$GIT_AUTHOR_EMAIL" \
      commit -m "collect(ext_csh): after rebase $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M %Z')" \
      >>"$LOG" 2>&1 || true
  fi
  if ! "$GIT" push -u origin "HEAD:$BRANCH" >>"$LOG" 2>&1; then
    log "ERROR: git push failed after retry"
    exit 1
  fi
fi

log "pushed origin/$BRANCH"
exit 0
