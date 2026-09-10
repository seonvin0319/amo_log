#!/usr/bin/env bash
# Ingest local AMO/APART runs into amo_log, commit if needed, push <alias> branch.
#
# Multi-host + low disk: NEVER pull/fetch. Each host owns refs/heads/<alias>
# only (choi, offrl, svcho, ...). Do not push to main from cron.
set -uo pipefail

export GIT_EXEC_PATH="${GIT_EXEC_PATH:-/usr/lib/git-core}"
export GIT_TEMPLATE_DIR="${GIT_TEMPLATE_DIR:-/usr/share/git-core/templates}"
GIT="${GIT_BIN:-/usr/bin/git}"
ROOT="${AMO_LOG_ROOT:-/home/choi/amo_log}"
LOG="${AMO_LOG_PUSH_LOG:-/home/choi/logs/amo_log_auto_push.log}"
LOCK="${AMO_LOG_PUSH_LOCK:-/home/choi/.amo_log_auto_push.lock}"
PY="${PYTHON_BIN:-/usr/bin/python3}"
# Canonical aliases: choi | offrl | svcho | ext_csv | ext_csh
HOST_ALIAS="${AMO_LOG_HOST_ALIAS:-choi}"
BRANCH="${HOST_ALIAS}"

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

log "start (push-only → origin ${BRANCH}; no pull/fetch)"

# Stay on this host's branch so we never need to rebase onto others' objects.
cur="$("$GIT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
if [[ "$cur" != "$BRANCH" ]]; then
  if "$GIT" show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    if ! "$GIT" checkout "$BRANCH" >>"$LOG" 2>&1; then
      log "ERROR: checkout ${BRANCH} failed"
      exit 1
    fi
  else
    if ! "$GIT" checkout -b "$BRANCH" >>"$LOG" 2>&1; then
      log "ERROR: create ${BRANCH} failed"
      exit 1
    fi
  fi
fi

if ! "$PY" scripts/ingest_runs.py >>"$LOG" 2>&1; then
  log "ERROR: ingest_runs.py failed"
  exit 1
fi

if ! "$PY" scripts/build_catalog.py >>"$LOG" 2>&1; then
  log "ERROR: build_catalog.py failed"
  exit 1
fi

"$GIT" add runs catalog docs scripts README.md .gitignore 2>/dev/null || true

export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-$("$GIT" config user.name 2>/dev/null || echo amo_log-cron)}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-$("$GIT" config user.email 2>/dev/null || echo amo_log-cron@${HOST_ALIAS})}"
export GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-$GIT_AUTHOR_NAME}"
export GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-$GIT_AUTHOR_EMAIL}"

if ! "$GIT" diff --cached --quiet; then
  MSG="collect(${HOST_ALIAS}): auto ingest $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M %Z')"
  if ! "$GIT" commit -m "$MSG" >>"$LOG" 2>&1; then
    log "ERROR: git commit failed"
    exit 1
  fi
  log "committed: $MSG"
else
  log "nothing to commit"
fi

# Push only this host ref. No pull/fetch retry (disk-safe for multi-host).
# Prefer seonvin0319 for seonvin0319/amo_log (host may have multiple gh accounts).
if command -v gh >/dev/null 2>&1; then
  gh auth switch --user seonvin0319 >>"$LOG" 2>&1 || true
  gh auth setup-git >>"$LOG" 2>&1 || true
fi
if ! "$GIT" push -u origin "HEAD:refs/heads/${BRANCH}" >>"$LOG" 2>&1; then
  log "ERROR: git push origin ${BRANCH} failed (no pull retry; fix remote or auth)"
  exit 1
fi

log "DONE ok (origin/${BRANCH})"
exit 0
