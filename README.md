# amo_log

AMO / APART (및 관련 offline RL) 실험 로그 아카이브입니다.

- **포함:** `config.yaml`, `metrics.jsonl`, `eval.jsonl`, `run_meta.json`
- **제외:** checkpoint(`*.pt`), wandb 바이너리, 대용량 버퍼
- **규칙:** [docs/COLLECTION_RULES.md](docs/COLLECTION_RULES.md), [docs/NAMING.md](docs/NAMING.md), [docs/SOURCES.md](docs/SOURCES.md)
- **자동 push:** [docs/AUTO_PUSH.md](docs/AUTO_PUSH.md) (choi cron, 2시간마다)
- **카탈로그:** [catalog/INDEX.md](catalog/INDEX.md), [catalog/catalog.json](catalog/catalog.json)

## 빠른 사용

```bash
# 새 런을 수집 (로컬 소스 → runs/)
python scripts/ingest_runs.py --dry-run
python scripts/ingest_runs.py

# 카탈로그만 재생성
python scripts/build_catalog.py
```

## 레이아웃

```text
amo_log/
  docs/                 # 수집/이름 규칙
  scripts/              # ingest / catalog
  runs/
    <algo>/             # apart | amo
      <family>/         # dual_proximal | chain | pi_only_xfit | ...
        <run_id>/       # 정규화된 런 디렉터리
          config.yaml
          metrics.jsonl
          eval.jsonl
          run_meta.json
  catalog/
    catalog.json
    INDEX.md
```

## 출처

| 소스 | 설명 |
|------|------|
| `/home/choi/APART/results_apart` | APART dual/chain locomotion·expert 스윕 |
| `/home/choi/APART/results_pi_only_xfit_target` | APART π-only xfit target baseline |
| `/home/choi/APART/results_pi_only_xfit_target_mpi_nstep` | MPI N-step / actor-lr ablations |
| `/home/choi/amo/results/segment_interval` | AMO segment-interval B_PI smoke |

원격: 이 저장소(`amo_log`)만 공유 인터페이스로 쓰고, 원본 머신 경로의 checkpoint는 올리지 않습니다.
