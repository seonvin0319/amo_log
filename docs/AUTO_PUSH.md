# 자동 수집 (ext_csh)

최신 origin/ext_csh 변경을 먼저 반영한 뒤 기존 수집 스크립트를 실행합니다. main 브랜치에 로그를 올리지 않습니다.

```bash
python scripts/ingest_runs.py
python scripts/build_catalog.py
git add -A -- main ablation catalog docs scripts README.md .gitignore
git commit -m "collect(ext_csh): refresh experiment logs"
git push origin HEAD:ext_csh
```

기존 JAX 제거 목록은 특정 실행·코드 버전에만 적용됩니다.
