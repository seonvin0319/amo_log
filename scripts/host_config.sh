#!/usr/bin/env bash
# Machine-only defaults. Common behavior belongs in auto_push.sh/collect_logs.py.
if [[ -x /home/ext_csv/miniconda3/envs/amo-jax/bin/python ]]; then
 export PYTHON_BIN="${PYTHON_BIN:-/home/ext_csv/miniconda3/envs/amo-jax/bin/python}"
fi
if [[ -f /home/ext_csv/.ssh/id_ed25519_amo_log ]]; then
 export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -i /home/ext_csv/.ssh/id_ed25519_amo_log -o IdentitiesOnly=yes -o BatchMode=yes}"
fi
export AMO_LOG_BRANCH="${AMO_LOG_BRANCH:-ext_csv}"
