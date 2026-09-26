#!/usr/bin/env bash
# ext_csh entry. The shared auto_push.sh is unchanged.
# normalize() can delete a staging run_meta after the path list is taken.
# That FileNotFoundError leaves main/ablation/catalog dirty, and the next
# auto_push then refuses to start. Rebuild the catalog once and finish.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set +e
out="$(bash "$ROOT/scripts/auto_push.sh" "$@" 2>&1)"
rc=$?
set -e
printf '%s\n' "$out"
if [[ "$rc" -eq 0 ]]; then
  exit 0
fi
if [[ "$out" != *"FileNotFoundError"* ]]; then
  exit "$rc"
fi
echo "retry catalog after normalize FileNotFoundError" >&2
if [[ -f scripts/host_config.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/host_config.sh
fi
PY="${PYTHON_BIN:-python3}"
"$PY" scripts/build_catalog.py
git add -A -- main ablation catalog
"$PY" scripts/validate_logs.py --index --branch ext_csh --common-ref origin/main
if ! git diff --cached --quiet; then
  name="$(git config --get user.name || true)"
  email="$(git config --get user.email || true)"
  git -c user.name="${name:-amo-log collector}" -c user.email="${email:-amo-log@localhost}" \
    commit -m "collect(ext_csh): refresh standardized logs"
fi
git push origin "HEAD:refs/heads/ext_csh"
