# 자동 Push (cron)

호스트 `choi`에서 **2시간마다** 로컬 실험 로그를 ingest → catalog → commit → `origin/main` push 합니다.

## 스케줄

```cron
11 */2 * * * /home/choi/amo_log/scripts/auto_push.sh
```

- 매 짝수시 `:11` (KST 시스템 시각 기준; cron은 보통 local time = Asia/Seoul)
- 로그: `/home/choi/logs/amo_log_auto_push.log`
- 락: `/home/choi/.amo_log_auto_push.lock` (`flock` — 겹치면 skip)

## 동작

1. `python scripts/ingest_runs.py` — `DEFAULT_SOURCES`에서 새 런 수집
2. `python scripts/build_catalog.py`
3. `git pull --rebase --autostash`
4. 변경 있으면 commit (`collect: auto ingest …`) 후 `git push`

checkpoint / wandb 는 ingest 규칙상 여전히 제외됩니다.

## 수동 실행

```bash
/home/choi/amo_log/scripts/auto_push.sh
tail -n 50 /home/choi/logs/amo_log_auto_push.log
```

## 끄기

```bash
crontab -e   # 해당 줄 삭제
# 또는
crontab -l | grep -v amo_log/scripts/auto_push.sh | crontab -
```
