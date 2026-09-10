# 자동 Push (ext_csv)

호스트 `ext_csv`에서 **2시간마다** 로컬 실험 로그를 ingest → catalog → commit → `origin/ext_csv` push 합니다.

## GitHub write access (one-time)

This host has no personal GitHub credential. Push uses deploy key:

- private: `/home/ext_csv/.ssh/id_ed25519_amo_log`
- public: `/home/ext_csv/.ssh/id_ed25519_amo_log.pub`
- SSH host alias: `github-amo-log` (see `~/.ssh/config`)
- remote push URL: `git@github-amo-log:seonvin0319/amo_log.git`

Add the **public** key as a **read/write deploy key** on
https://github.com/seonvin0319/amo_log/settings/keys

Then:

```bash
cd /home/ext_csv/amo_log && bash scripts/auto_push.sh
```

## 스케줄


`crontab`이 없는 환경이라 **루프 래퍼**를 씁니다:

```bash
nohup bash /home/ext_csv/amo_log/scripts/auto_push_loop.sh \
  >>/home/ext_csv/logs/amo_log_auto_push_loop.nohup 2>&1 &
```

- 주기: 2시간 (`AMO_LOG_PUSH_INTERVAL_SEC`, 기본 7200)
- 로그: `/home/ext_csv/logs/amo_log_auto_push.log`
- 락: `/home/ext_csv/.amo_log_auto_push.lock` (`flock` — 겹치면 skip)
- 루프 pid: `/home/ext_csv/logs/amo_log_auto_push_loop.pid`

`crontab`을 쓸 수 있으면:

```cron
11 */2 * * * /home/ext_csv/amo_log/scripts/auto_push.sh
```

## 동작

1. `python scripts/ingest_runs.py --host=ext_csv`
2. `python scripts/build_catalog.py`
3. `git pull --rebase --autostash origin ext_csv` (원격 브랜치 있을 때)
4. 변경 있으면 commit (`collect(ext_csv): auto ingest …`) 후 `git push origin HEAD:ext_csv`

checkpoint / wandb 는 ingest 규칙상 제외됩니다.

## 수동 실행

```bash
/home/ext_csv/amo_log/scripts/auto_push.sh
tail -n 50 /home/ext_csv/logs/amo_log_auto_push.log
```

## 끄기

```bash
kill "$(cat /home/ext_csv/logs/amo_log_auto_push_loop.pid)"
```
