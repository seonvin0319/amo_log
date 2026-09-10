#!/usr/bin/env bash
# Ingest local AMO runs into amo_log, commit if needed, push origin/ext_csv.
# Host: ext_csv. Intended every 2 hours (cron or loop wrapper).
set -uo pipefail

export GIT_EXEC_PATH="${GIT_EXEC_PATH:-/usr/lib/git-core}"
export GIT_TEMPLATE_DIR="${GIT_TEMPLATE_DIR:-/usr/share/git-core/templates}"
GIT="${GIT_BIN:-/usr/bin/git}"
ROOT="${AMO_LOG_ROOT:-/home/ext_csv/amo_log}"
LOG="${AMO_LOG_PUSH_LOG:-/home/ext_csv/logs/amo_log_auto_push.log}"
LOCK="${AMO_LOG_PUSH_LOCK:-/home/ext_csv/.amo_log_auto_push.lock}"
PY="${PYTHON_BIN:-/home/ext_csv/miniconda3/bin/python3}"
BRANCH="${AMO_LOG_BRANCH:-ext_csv}"
# Dedicated deploy key (write access must be granted on GitHub).
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -i /home/ext_csv/.ssh/id_ed25519_amo_log -o IdentitiesOnly=yes -o BatchMode=yes}"
export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-SChoish}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-petersun0221@hanyang.ac.kr}"
export GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-$GIT_AUTHOR_NAME}"
export GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-$GIT_AUTHOR_EMAIL}"


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

if ! "$GIT" checkout "$BRANCH" >>"$LOG" 2>&1; then
  log "ERROR: git checkout $BRANCH failed"
  exit 1
fi

if ! "$PY" scripts/ingest_runs.py --host=ext_csv >>"$LOG" 2>&1; then
  log "ERROR: ingest_runs.py failed"
  exit 1
fi

if ! "$PY" scripts/build_catalog.py >>"$LOG" 2>&1; then
  log "ERROR: build_catalog.py failed"
  exit 1
fi

# First push may have no remote branch yet.
if "$GIT" ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  if ! "$GIT" pull --rebase --autostash origin "$BRANCH" >>"$LOG" 2>&1; then
    log "ERROR: git pull --rebase failed"
    "$GIT" rebase --abort >>"$LOG" 2>&1 || true
    exit 1
  fi
else
  log "remote branch origin/$BRANCH missing; will create on push"
fi

"$GIT" add runs catalog docs scripts README.md .gitignore 2>/dev/null || true

if "$GIT" diff --cached --quiet; then
  log "nothing to commit"
  # still try push if local is ahead of a newly created tracking setup
  if "$GIT" rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    if ! "$GIT" push origin "HEAD:$BRANCH" >>"$LOG" 2>&1; then
      log "WARN: push of no-op commit state failed (may already be up to date)"
    fi
  fi
  log "DONE ok (no commit)"
  exit 0
fi

MSG="collect(ext_csv): auto ingest $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M %Z')"

if ! "$GIT" commit -m "$MSG" >>"$LOG" 2>&1; then
  log "ERROR: git commit failed"
  exit 1
fi
log "committed: $MSG"

if ! "$GIT" push -u origin "HEAD:$BRANCH" >>"$LOG" 2>&1; then
  log "push failed; retrying after pull --rebase"
  if "$GIT" ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
    if ! "$GIT" pull --rebase --autostash origin "$BRANCH" >>"$LOG" 2>&1; then
      log "ERROR: git pull --rebase failed on retry"
      "$GIT" rebase --abort >>"$LOG" 2>&1 || true
      exit 1
    fi
    "$GIT" add runs catalog docs scripts README.md .gitignore 2>/dev/null || true
    if ! "$GIT" diff --cached --quiet; then
      "$GIT" commit -m "collect(ext_csv): after rebase $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M %Z')" >>"$LOG" 2>&1 || true
    fi
  fi
  if ! "$GIT" push -u origin "HEAD:$BRANCH" >>"$LOG" 2>&1; then
    log "ERROR: git push failed after retry (add write deploy key id_ed25519_amo_log.pub on seonvin0319/amo_log)"
    exit 1
  fi
fi

log "DONE ok"
exit 0
