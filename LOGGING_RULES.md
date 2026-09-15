# amo_log 공통 로그 규칙

이 문서는 모든 브랜치에서 동일하게 유지한다. 기준본은 Git `main` 브랜치이며, 규칙 변경은 모든 머신 브랜치에 함께 반영한다.

## 브랜치와 경로

- Git `main` 브랜치: 전체 위치 README, 공통 규칙과 도구. 실험 로그를 저장하지 않는다.
- 머신 브랜치: `choi`, `ext_csh`, `ext_csv`, `offrl`, `shchoi`, `svcho`. 각 서버는 자기 브랜치에만 push한다.
- 본 실험 디렉터리: `main/<method>/<env>/<meta_lr>/seed_<seed>/<run_id>/`.
- Ablation 디렉터리: `ablation/<method>/<env>/<meta_lr>/seed_<seed>/<run_id>/`.
- `<method>`: `td3_amo`, `iql_amo`, `td3bc+rc`, `iql`, `a2pr`, `wpc`, `aspc`만 사용한다.
- 환경은 `walker2d-medium-replay-v2` 같은 전체 이름과 버전을 사용한다. 약어·unknown을 쓰지 않는다.
- AMO meta lr 폴더는 `alpha_lr`(기존 `T_lr`와 같은 학습률) 또는 `rho_lr`(=beta_lr)를 뜻한다. actor/critic lr를 이 폴더로 대신하지 않는다.
- AMO 본 실험의 lr 이름은 `1e-3`, `2e-3`, `3e-4`. 다른 meta lr 실험은 실제 lr 이름으로 ablation에 저장한다.
- baseline에는 meta_lr 폴더를 만들지 않는다. 예: `main/iql/hopper-medium-v2/seed_0/<run_id>/`.
- 실제 로그가 있는 경로만 만든다. 폴더가 있다는 이유로 완료된 실행이라고 판단하지 않는다.

## 본 실험과 ablation

- **Adroit는 모든 방법에서 ablation**이다. `door-*`, `hammer-*`, `pen-*`, `relocate-*`의 human/cloned/expert 등 모든 데이터셋에 적용하며, baseline도 예외가 아니다.
- AMO 본 실험의 **초기값**은 아래 표를 따른다. 학습 중 alpha/beta가 이 값으로 고정되어야 한다는 뜻이 아니다.

  | 방법 | main 허용 초기값 | 실제 설정 기준 |
  |---|---|---|
  | TD3-AMO | alpha = 1, 2, 5 | alpha_E=alpha_B = 1, 2, 5 |
  | IQL-AMO | beta = 1, 2, 5 | beta_initial = 1, 2, 5 |

- TD3의 alpha_E와 alpha_B는 같은 허용값으로 시작해야 한다. 서로 다른 초기값은 ablation이다. alpha_B 생략 시 alpha_E를 따르는 기존 기본값을 적용한다.
- TD3의 **T=5는 alpha=10**이므로 main에 포함하지 않는다. T=1.25, 5, 10 및 IQL beta=3, 10 등 허용 목록 밖의 초기값은 ablation이다.
- TD3는 config의 alpha_E/alpha_B(또는 alpha_init), IQL은 beta_initial을 읽는다. 기존 T 설정은 아래 단위 변환 후 읽는다. 폴더명이나 baseline 고유 alpha를 AMO 스케일로 추정하지 않는다.
- 허용 초기값과 함께 meta lr `1e-3`, `2e-3`, `3e-4` 및 기존 구조/loss 기준도 만족해야 main이다. actor/critic lr는 이 meta lr와 별개다.
- 기존 loss·critic 구조·N·alpha 비율/스케줄 등 변형 실험은 ablation으로 보존한다. 현재 분류 기준은 `scripts/log_layout.py`와 `classification_reasons`를 따른다.
- baseline의 고유 beta 설정은 AMO 초기 beta가 아니므로 이 초기값 제한을 적용하지 않는다.
- 폴더 이름이나 높은 점수로 실험 설정을 추정하지 않는다. 설정 파일과 원본 metadata를 먼저 확인한다.
- `adroit`, `initial_scale_outside_main`, `initial_scale_mismatch`, `initial_beta_outside_main` 등의 사유를 metadata에 기록한다. 초기값만 허용 범위에 들어와도 다른 ablation 사유가 남으면 승격하지 않는다.

## 잘못 변경한 네트워크 학습률

- 사용자 지시에 따라 기본 네트워크 lr를 바꾼 실행은 잘못 돌린 로그로 취급한다. main과 ablation 모두 실행 디렉터리 전체와 카탈로그 항목을 삭제하며 ablation으로 보관하지 않는다. 과거 원본은 Git 이력에 남는다.
- 지원 방법의 `actor_lr`, `critic_lr`와 그 별칭 `qf_lr`는 **3e-4**여야 한다. IQL/IQL-AMO/wPC/A2PR의 `value_lr`·`vf_lr`, TD3-AMO의 별도 behavior 네트워크 `behavior_lr`도 **3e-4**를 적용한다.
- 실제 config와 metadata settings에 명시된 값 모두 확인한다. 생략/null은 과거 implicit default로 두며 폴더 이름만으로 변경을 추정하지 않는다. `3e-4` 문자열과 `0.0003`은 같다.
- `alpha_lr`(과거 `T_lr`), `rho_lr`/`beta_lr`, `scale_lr`, `lambda_lr` 등 meta lr 튜닝은 이 삭제 조건이 아니다. A2PR의 `vae_lr=1e-3` 및 IQL config에 남은 미사용 `cql_policy_lr` 같은 별도/미사용 항목에도 actor lr 기준을 적용하지 않는다.
- `catalog/removed_invalid_network_lr.json`에는 삭제 사유, 위반 필드, run_id, source_path, code_commit과 기존 위치만 남긴다. 원본 점수나 로그는 복사하지 않는다. 같은 source_path+code_commit(경로 미기록 시 run_id+commit)의 재수집은 차단한다. 올바른 설정으로 다시 실행할 때는 새 실행 식별자를 사용한다.
- 공통 정규화기는 신규 `runs/`와 기존 `main/`·`ablation/`에서 위반 실행을 제거한 뒤 카탈로그를 재생성한다. 검증기는 실제 config까지 읽어 위반 및 삭제 실행의 재유입을 거부한다.

## 기존 로그 재분류

- `python3 scripts/build_catalog.py`는 `runs/`의 신규 로그와 기존 `main/`·`ablation/` 로그를 현재 규칙으로 재분류한다.
- 실제 실행 디렉터리와 `run_meta.json`의 section/rel_path/classification_reasons를 함께 갱신하고, 카탈로그도 같은 위치를 가리키게 한다.
- 평가 결과·학습 관측값·run_id와 원본 provenance는 보존한다. T 스케일의 키와 단위는 아래 규칙으로 정규화한다. 초기값별 실행은 기존 run_id로 구별하므로 서로 덮어쓰지 않는다.
- 이 변경으로 Adroit는 ablation으로 이동한다. 초기값 사유만 있던 IQL beta=5 등은 나머지 main 조건을 만족할 때 main으로 이동한다.

## 실행 식별과 원본 보존

- seed마다 run_id를 보존한다. 다른 실행은 seed가 같아도 덮어쓰거나 점수로 골라 삭제하지 않는다.
- `run_meta.json`에 `layout_version=2`, method, section, env, seed, backend, meta_lr, rel_path, run_id, source_path, settings, git.code_commit을 기록한다.
- 새로운 실행은 backend(`torch`/`jax`)와 실제 코드 commit을 명시한다. 과거 commit 미기록 로그는 null을 유지하고 값을 지어내지 않는다.
- source_host는 수집 머신을 기록한다. TD3-AMO는 `amo_exp`의 `td3-amo`, IQL-AMO는 `iql-amo` 브랜치의 실행 provenance를 남긴다.
- `config.yaml`, 평가·학습 파일은 아래 T→alpha 변환만 적용한다. 원본 경로와 코드 commit은 metadata에 남기고, 변환 전 파일은 Git 이력 및 해시로 추적한다.
- 같은 실행의 별칭은 `is_alias`와 `alias_of`로 기록하고 카탈로그에서 중복 집계하지 않는다.
- 서로 다른 backend의 실행을 같은 run_id로 덮어쓰지 않는다. 수정된 코드/새 실행은 별도의 식별자를 쓴다.

## T 로그의 alpha 단위 정규화

- TD3-AMO의 main과 ablation 모두 `alpha = 2T`로 저장한다. `scripts/alpha_logs.py`가 유일한 변환 규칙이며 수집·카탈로그 생성과 기존 로그 마이그레이션이 함께 사용한다.
- 실제 스케일은 키를 바꾸고 값을 2배 한다: `T → alpha`, `T_E → alpha_E`, `T_B → alpha_B`, `T_init/min/max → alpha_init/min/max`. raw/effective/used/next 및 스케줄 시작·끝·현재 스케일도 동일하다. `amo/`, `apart/` 등의 지표 접두사는 유지한다.
- `T_lr → alpha_lr`, `T_freq → alpha_freq`는 이름만 바꾼다. 학습률, 주기, 비율, 스케줄 step, projection flag, loss, 평가 점수는 2배 하지 않는다. 예: `T_B_over_T_E → alpha_B_over_alpha_E`, `L_T_E → L_alpha_E`의 숫자는 그대로다.
- 과거 `grad_T_*`/`T_grad`는 작성기마다 rho/T 미분 좌표가 다를 수 있어 원래 수치를 `legacy_T/<기존 키>`로 보존한다. 이를 alpha gradient로 취급하거나 임의로 2배/절반으로 바꾸지 않는다. rho·h·tau 등 별도 내부 좌표도 재계산하지 않는다.
- 기존 alpha 필드는 다시 2배 하지 않는다. 동일 레코드의 T와 alpha가 공존하면 `alpha=2T` 일치 여부를 확인하고, 불일치는 업로드를 중단한다. IQL의 beta와 baseline의 고유 alpha/beta는 변환 대상이 아니다.
- 저장 중 끊겨 JSON으로 읽을 수 없는 원본 행은 `legacy_unparsed_record`에 원문·행 번호·오류를 그대로 보존하고 파일별 `unparsed_source_lines`에 기록한다. 해당 행의 값은 추정하거나 변환하지 않으며 관측값 집계에서 제외한다. 변환 및 검증 출력에 보존한 행 수를 표시한다.
- `run_meta.json`의 `scale_conversion.schema=alpha_v1`와 파일별 변환 전 `legacy_sha256`·변환 후 `sha256`을 남긴다. 파일별 해시로 이미 변환한 파일을 확인하므로, 원본 T 로그를 다시 수집하거나 T/alpha 레코드가 섞여 추가되어도 중복 변환하지 않는다.
- run_id, 과거 경로·variant 문자열, 코드 commit은 원래 실행 식별 정보다. 이름 속 T 숫자는 고치지 않으며 초기값 집계는 정규화한 settings/config와 카탈로그의 alpha 표시를 사용한다.
- 기존 로그 변환 및 전체 파일 검증: `python3 scripts/migrate_alpha_logs.py`. 검증만 수행: `python3 scripts/migrate_alpha_logs.py --check`.
- GitHub의 `Normalize alpha log units`는 변환 도구 변경 시 머신 브랜치에서 기존 로그를 변환·검증 후 fast-forward push한다. 원격이 먼저 바뀌면 push가 실패하며 force push하지 않는다.

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
