# 자동 Push (cron) — multi-host, pull 없음

여러 머신이 같은 `amo_log`에 올리되, **디스크를 아끼기 위해 pull/fetch 하지 않습니다.**

## 모델

| 역할 | ref |
|------|-----|
| 각 호스트 cron | `<alias>` 브랜치만 push (`choi`, `offrl`, `svcho`, …) |
| `main` | 초기 스냅샷 / 디스크 여유 있는 곳에서 merge |

같은 `main`에 여러 호스트가 pull 없이 push하면 non-fast-forward로 막힙니다.  
그래서 **호스트당 브랜치 1개**(`choi` 등)를 소유하고, 그 브랜치만 fast-forward push 합니다.

## 스케줄 (choi)

```cron
11 */2 * * * /home/choi/amo_log/scripts/auto_push.sh
```

다른 머신:

```bash
export AMO_LOG_HOST_ALIAS=offrl   # 또는 svcho / ext_csv / ext_csh
# 동일 스크립트 + cron → origin/offrl 등으로 push
```

- 로그: `/home/choi/logs/amo_log_auto_push.log`
- 락: `/home/choi/.amo_log_auto_push.lock`
- **pull/fetch/rebase 절대 안 함**

## 동작

1. `checkout` → 로컬 `choi` (없으면 생성; alias에 따라 다름)
2. `ingest_runs.py` → `build_catalog.py`
3. 변경 있으면 commit
4. `git push origin HEAD:refs/heads/choi`

## `main`으로 합치기 (디스크 여유 있는 머신에서만)

```bash
git fetch origin
git checkout main
git merge --no-ff origin/choi origin/offrl   # 필요한 호스트만
git push origin main
```

## 수동 실행

```bash
AMO_LOG_HOST_ALIAS=choi /home/choi/amo_log/scripts/auto_push.sh
tail -n 50 /home/choi/logs/amo_log_auto_push.log
```

## 끄기

```bash
crontab -l | grep -v amo_log/scripts/auto_push.sh | crontab -
```
