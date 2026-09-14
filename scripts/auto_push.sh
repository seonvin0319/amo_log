#!/usr/bin/env bash
# Common entry point on all branches. Machine-specific defaults live in host_config.sh.
set -euo pipefail
ROOT="${AMO_LOG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
if [[ -f scripts/host_config.sh ]]; then source scripts/host_config.sh; fi
GIT="${GIT_BIN:-git}"
PY="${PYTHON_BIN:-python3}"
BRANCH="$("$GIT" branch --show-current)"
case "$BRANCH" in choi|ext_csh|ext_csv|offrl|shchoi|svcho) ;; *) echo 'ERROR: checkout the correct machine branch first' >&2; exit 1;; esac
EXPECTED="${AMO_LOG_BRANCH:-${AMO_LOG_HOST_ALIAS:-$BRANCH}}"
[[ "$BRANCH" == "$EXPECTED" ]] || { echo 'ERROR: machine/branch mismatch' >&2; exit 1; }
LOCK="${AMO_LOG_PUSH_LOCK:-$("$GIT" rev-parse --git-path amo-log-push.lock)}"
mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
flock -n 9 || { echo 'SKIP: collector already running'; exit 0; }
"$GIT" diff --quiet && "$GIT" diff --cached --quiet || { echo 'ERROR: preserve/commit existing changes before collection' >&2; exit 1; }
"$GIT" fetch --filter=blob:none origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" "refs/heads/main:refs/remotes/origin/main"
"$GIT" merge --ff-only "origin/$BRANCH"
# Start the freshly fetched Python implementation, not a pre-pull loaded module.
exec "$PY" scripts/collect_logs.py --branch "$BRANCH" "$@"
