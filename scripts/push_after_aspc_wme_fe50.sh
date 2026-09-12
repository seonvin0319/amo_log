#!/usr/bin/env bash
# Wait for ASPC wme_s0 to finish 1M + final_eval_50, then ingest+push amo_log.
set -euo pipefail
WME_DIR=/home/shchoi/ASPC/results_pi_l3/wme_s0_aspc-walker2d-medium-expert-v2-52616add
LOG=/home/shchoi/amo_log/.push_after_aspc_fe50.log
PUSH=/home/shchoi/amo_log/scripts/auto_push.sh

log() { printf '[%s] %s\n' "$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$LOG"; }

log "waiting for wme_s0 1M + final_eval_50"

while true; do
  # training process gone?
  if pgrep -af 'algorithms/offline/aspc.py.*wme_s0_aspc' | grep -v pgrep >/dev/null; then
    step="?"
    if [[ -f "$WME_DIR/metrics.jsonl" ]]; then
      step="$(python3 -c "import json;print(json.loads(open('$WME_DIR/metrics.jsonl').read().strip().splitlines()[-1]).get('step'))")"
    fi
    log "wme still training step=$step"
    sleep 120
    continue
  fi
  # need 1M metrics / eval and fe50
  ready="$(python3 - <<'PY'
import json
from pathlib import Path
d=Path("/home/shchoi/ASPC/results_pi_l3/wme_s0_aspc-walker2d-medium-expert-v2-52616add")
ev=d/"eval.jsonl"
fe=d/"final_eval_50.jsonl"
step=None
if ev.exists() and ev.stat().st_size:
    step=int(json.loads(ev.read_text().strip().splitlines()[-1]).get("step") or 0)
ck=d/"checkpoint_999999.pt"
ok_train = (step is not None and step >= 999000) or (ck.exists() and ck.stat().st_size > 0)
ok_fe = fe.exists() and fe.stat().st_size > 0
print("1" if (ok_train and ok_fe) else "0")
print(f"step={step} fe={ok_fe} ck999={ck.exists() and ck.stat().st_size>0}")
PY
)"
  ok="$(echo "$ready" | head -1)"
  detail="$(echo "$ready" | tail -1)"
  if [[ "$ok" == "1" ]]; then
    log "ready: $detail"
    break
  fi
  log "train done but waiting fe/ckpt: $detail"
  # if train finished without fe, the resume script should still run fe; wait
  sleep 60
done

log "running auto_push (ingest final_eval_50 + wme complete)"
bash "$PUSH" >>"$LOG" 2>&1
log "DONE auto_push exit=$?"
