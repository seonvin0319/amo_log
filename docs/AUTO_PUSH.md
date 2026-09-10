# 자동 Push

호스트 `shchoi` (`iisl-server04`)에서 로컬 실험 로그를 ingest → catalog → commit → `origin/shchoi` push 합니다.
`main`에는 올리지 않습니다.

## 동작

1. `python scripts/ingest_runs.py` — `DEFAULT_SOURCES`에서 새 런 수집
2. `python scripts/build_catalog.py`
3. `git pull --rebase --autostash origin shchoi`
4. 변경 있으면 commit (`collect: auto ingest …`) 후 `git push origin HEAD:shchoi`

checkpoint / wandb 는 ingest 규칙상 제외됩니다.

로그: `/home/shchoi/amo_log/.auto_push.log`
락: `/home/shchoi/amo_log/.auto_push.lock`

## 수동 실행

```bash
/home/shchoi/amo_log/scripts/auto_push.sh
tail -n 50 /home/shchoi/amo_log/.auto_push.log
```
