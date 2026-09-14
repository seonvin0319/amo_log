#!/usr/bin/env bash
# Machine-only defaults. Common behavior belongs in auto_push.sh/collect_logs.py.
if [[ -x /home/shchoi/miniconda3/bin/python ]]; then export PYTHON_BIN="${PYTHON_BIN:-/home/shchoi/miniconda3/bin/python}"; fi
export AMO_LOG_BRANCH="${AMO_LOG_BRANCH:-shchoi}"
