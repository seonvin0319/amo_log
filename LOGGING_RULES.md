# amo_log 공통 로그 규칙

이 문서는 모든 브랜치에서 동일하게 유지한다. 기준본은 Git `main` 브랜치이며, 규칙 변경은 모든 머신 브랜치에 함께 반영한다.

## 브랜치와 경로

- Git `main` 브랜치: 전체 위치 README, 공통 규칙과 도구. 실험 로그를 저장하지 않는다.
- 머신 브랜치: `choi`, `ext_csh`, `ext_csv`, `offrl`, `shchoi`, `svcho`. 각 서버는 자기 브랜치에만 push한다.
- 본 실험 디렉터리: `main/<method>/<env>/<meta_lr>/seed_<seed>/<run_id>/`.
- Ablation 디렉터리: `ablation/<method>/<env>/<meta_lr>/seed_<seed>/<run_id>/`.
- `<method>`: `td3_amo`, `iql_amo`, `td3bc+rc`, `iql`, `a2pr`, `wpc`, `aspc`만 사용한다.
- 환경은 `walker2d-medium-replay-v2` 같은 전체 이름과 버전을 사용한다. 약어·unknown을 쓰지 않는다.
- AMO meta lr 폴더는 `T_lr` 또는 `rho_lr`(=beta_lr)를 뜻한다. actor/critic lr를 이 폴더로 대신하지 않는다.
- AMO 본 실험의 lr 이름은 `1e-3`, `2e-3`, `3e-4`. 다른 lr 실험은 실제 lr 이름으로 ablation에 저장한다.
- baseline에는 meta_lr 폴더를 만들지 않는다. 예: `main/iql/hopper-medium-v2/seed_0/<run_id>/`.
- 실제 로그가 있는 경로만 만든다. 폴더가 있다는 이유로 완료된 실행이라고 판단하지 않는다.

## 본 실험과 ablation

- AMO 본 실험: **초기값** T_E=T_B=1 또는 beta_initial=1. 학습 중 T/beta가 1이어야 한다는 뜻이 아니다.
- 5, 10뿐 아니라 다른 초기값(예: beta_initial=3)도 ablation이다.
- 기존 loss·critic 구조·N·T 비율/스케줄 등 변형 실험은 ablation으로 보존한다. 현재 분류 기준은 `scripts/log_layout.py`와 `classification_reasons`를 따른다.
- baseline의 고유 beta 설정은 AMO 초기 beta가 아니므로 이 초기값 제한을 적용하지 않는다.
- 폴더 이름이나 높은 점수로 실험 설정을 추정하지 않는다. 설정 파일과 원본 metadata를 먼저 확인한다.

## 실행 식별과 원본 보존

- seed마다 run_id를 보존한다. 다른 실행은 seed가 같아도 덮어쓰거나 점수로 골라 삭제하지 않는다.
- `run_meta.json`에 `layout_version=2`, method, section, env, seed, backend, meta_lr, rel_path, run_id, source_path, settings, git.code_commit을 기록한다.
- 새로운 실행은 backend(`torch`/`jax`)와 실제 코드 commit을 명시한다. 과거 commit 미기록 로그는 null을 유지하고 값을 지어내지 않는다.
- source_host는 수집 머신을 기록한다. TD3-AMO는 `amo_exp`의 `td3-amo`, IQL-AMO는 `iql-amo` 브랜치의 실행 provenance를 남긴다.
- `config.yaml`, 평가·학습 파일의 원본 내용을 보존한다. 원본 경로는 metadata에 남긴다.
- 같은 실행의 별칭은 `is_alias`와 `alias_of`로 기록하고 카탈로그에서 중복 집계하지 않는다.
- 서로 다른 backend의 실행을 같은 run_id로 덮어쓰지 않는다. 수정된 코드/새 실행은 별도의 식별자를 쓴다.

## 기존 JAX 제거와 이후 JAX

2026-09-14 정리에서 제거한 기존 JAX 로그는 복원하지 않는다. `fail/jax` 폴더도 만들지 않는다.
`catalog/removed_legacy_jax.json`은 제거한 실행의 식별자만 담고, 점수·평가 로그는 담지 않는다.
제외는 기존 source_path + code_commit 조합에 한정한다. 이후 수정된 JAX 실행 전체를 영구 금지하지 않는다.
새 JAX 실행은 backend와 수정된 코드 commit을 기록한다. 구현이 다르면 집계에서도 backend와 코드 버전을 구분한다.

## 업로드 절차

각 머신에서 최신 해당 브랜치를 받은 후, 수집에 사용할 Python 환경에 `python3 -m pip install -r requirements-log-tools.txt`를 한 번 실행한다. YAML 파서가 없으면 설정을 추측해 분류하지 않고 중단한다. `PYTHON_BIN`으로 Python 실행 파일을 지정할 수 있다. 다음 공통 명령을 사용한다.

```bash
bash scripts/auto_push.sh
```

공통 흐름: 머신 브랜치 확인 → 변경 중인 작업 보호 → 해당 브랜치와 main fetch → 해당 브랜치 fast-forward → 공통 파일 일치 확인 → ingest → catalog 재생성 → stage → staged 파일 검증 → commit → 자기 브랜치 push.
자동 rebase, force push, 임의 계정 전환, 오류 무시는 하지 않는다. 충돌·원격 변경·규칙 불일치는 중단하고 먼저 해결한다.
원본 탐색과 머신별 실행환경은 `scripts/ingest_runs.py`, `scripts/host_config.sh`에서만 다를 수 있다. 기존 주기 실행기는 공통 auto_push.sh를 호출한다.

수동 commit 전에도 staged 상태를 검사한다.

```bash
python3 scripts/validate_logs.py --index --branch "$(git branch --show-current)" --common-ref origin/main
```

## 규칙·도구 변경

공통 파일 목록은 `scripts/validate_logs.py`의 SHARED에 있다. 모든 브랜치에서 바이트 단위로 같아야 한다.
공통 변경은 main과 머신 6개 브랜치에 함께 반영하고 각 브랜치를 검증한다. 머신에서만 분류 로직을 임의 변경하지 않는다.
GitHub Actions `Validate log conventions`는 push/PR 때 구조와 공통 파일 일치를 검사한다. 실패한 실행은 정상 업로드로 취급하지 않는다.
수동 push 자체를 서버에서 차단하는 branch protection은 별도 저장소 설정이며, 이 작업에서 강제 설정하지 않는다.
전체 실험 위치 README는 main의 `scripts/refresh_index.py`가 각 머신 catalog로 재생성한다.
