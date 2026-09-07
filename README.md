# amo_log

AMO / APART (및 관련 offline RL) **실험 로그**를 머신별로 모아 두는 공통 저장소입니다.

## 브랜치 = 머신

| 규칙 | 설명 |
|------|------|
| **`main`** | 이 README만 둡니다. 실험 로그·카탈로그는 올리지 않습니다. |
| **머신 브랜치** | 브랜치 이름 = 호스트 이름 (`ext_csh`, `svcho`, `choi`, …). **그 머신에서 돌린 로그만** 그 브랜치에 둡니다. |

로그는 **각 머신 브랜치에만** 있습니다. `main`에는 없습니다.

## 포함 / 제외 (머신 브랜치)

- **포함:** `config.yaml`, `metrics.jsonl`, `eval.jsonl`, `run_meta.json`
- **제외:** checkpoint(`*.pt`), wandb 바이너리, 대용량 버퍼

상세 규칙·ingest 스크립트·`runs/`·`catalog/`는 각 머신 브랜치를 보세요.

## 머신에서 쓰기

```bash
git clone git@github.com:seonvin0319/amo_log.git
cd amo_log

HOST="$(hostname -s)"   # 예: ext_csh
git fetch origin
git checkout -B "$HOST" "origin/$HOST" 2>/dev/null \
  || git checkout -B "$HOST"

# (첫 머신) 다른 호스트 브랜치에서 scripts/docs 복사해 부트스트랩한 뒤
# SOURCES에 로컬 results 경로를 넣고 ingest → commit → push
git push -u origin "$HOST"
```

다른 머신 로그를 볼 때는 해당 브랜치를 checkout 하면 됩니다.

```bash
git fetch origin
git checkout svcho    # 예: svcho 머신 로그만
```

## 레이아웃 (머신 브랜치)

```text
amo_log/                 # branch == hostname
  docs/                  # COLLECTION_RULES, NAMING, SOURCES, AUTO_PUSH
  scripts/               # ingest_runs.py, build_catalog.py, auto_push_*.sh
  runs/<algo>/<family>/<run_id>/
  catalog/
```

원격 `amo_log`만 공유 인터페이스로 쓰고, 원본 머신 경로의 checkpoint는 올리지 않습니다.
