# amo_log (`ext_csv` 브랜치)

이 브랜치는 **ext_csv 머신**에서 돌린 실험 로그만 둡니다.
공통 규칙(브랜치=호스트 이름, `main`은 README만)은 [`main` README](https://github.com/seonvin0319/amo_log/blob/main/README.md)를 보세요.

- **포함:** `config.yaml`, `metrics.jsonl`, `eval.jsonl`, `run_meta.json`
- **제외:** checkpoint(`*.pt`), wandb 바이너리, 대용량 버퍼
- **규칙:** [docs/COLLECTION_RULES.md](docs/COLLECTION_RULES.md), [docs/NAMING.md](docs/NAMING.md), [docs/SOURCES.md](docs/SOURCES.md)
- **카탈로그:** [catalog/INDEX.md](catalog/INDEX.md), [catalog/catalog.json](catalog/catalog.json)

## 빠른 사용

```bash
python scripts/ingest_runs.py --dry-run --host=ext_csv
python scripts/ingest_runs.py --host=ext_csv
python scripts/build_catalog.py
```

## 레이아웃

```text
amo_log/
  docs/
  scripts/
  runs/
    amo/
      behavior_bc_l1_joint/
        <run_id>/
          config.yaml
          metrics.jsonl
          eval.jsonl
          run_meta.json
  catalog/
    catalog.json
    INDEX.md
```

원격: 이 저장소(`amo_log`)만 공유 인터페이스로 쓰고, 원본 경로의 checkpoint는 올리지 않습니다.
