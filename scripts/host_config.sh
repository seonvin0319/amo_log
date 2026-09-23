#!/usr/bin/env bash
# Machine-only defaults. Common behavior belongs in auto_push.sh/collect_logs.py.
export AMO_LOG_BRANCH="${AMO_LOG_BRANCH:-ext_csh}"
export PYTHON_BIN="${PYTHON_BIN:-/home/ext_csh/miniconda3/envs/capo_jax/bin/python}"
