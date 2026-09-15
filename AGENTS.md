# amo_log 작업 지침

작업 전에 루트 `LOGGING_RULES.md`를 읽고 따른다. Git main은 공통 규칙·전체 인덱스용이며 머신 브랜치의 main/ 디렉터리와 다르다.

- 관측값과 provenance를 보존하고 config와 metadata를 기준으로 분류한다. 승인된 T→alpha 변환은 LOGGING_RULES.md에 따라 실제 스케일만 2배 하고, 이미 alpha인 값은 다시 변환하지 않는다. 점수로 실행을 삭제/교체하지 않는다.
- 사용자 지시에 따라 actor/critic/value 등 기본 네트워크 lr를 바꾼 오실행은 main/ablation 양쪽에서 삭제한다. 고정 기준 3e-4, 적용 필드와 예외는 LOGGING_RULES.md를 따른다. 삭제 실행의 식별자는 catalog/removed_invalid_network_lr.json에 남겨 재수집을 막는다. alpha_lr/rho_lr 등 meta lr 튜닝은 삭제 사유가 아니다.
- Adroit는 baseline을 포함한 모든 방법에서 ablation이다. AMO main 초기값은 TD3 alpha_E=alpha_B=1/2/5, IQL beta_initial=1/2/5이며, meta lr와 구조/loss 기준도 함께 적용한다.
- 기준을 바꿀 때 기존 main/ablation 실행 경로·metadata·카탈로그까지 재분류한다. 문서만 바꾸거나 기존 ablation 사유를 무조건 지우지 않는다.
- 새 backend/설정 이름은 확인 없이 기존 방법으로 추정하지 않는다.
- 로그는 공통 수집기로 정규화하고 commit 전에 `python3 scripts/validate_logs.py --index --branch <현재브랜치> --common-ref origin/main`을 실행한다.
- 공통 규칙·분류기·카탈로그·검증기·업로드 도구는 main을 기준으로 7개 브랜치 모두 동일하게 반영한다. SHARED 목록을 따른다.
- 머신별 source 탐색, host_config, 주기 실행 간격만 다를 수 있다.
- 자동 rebase, force push, 진행 중 작업 폐기, 임의 GitHub 계정 전환을 하지 않는다.
- 완료 보고에 변경 브랜치, 검증 결과, 서버 pull 필요 여부를 명시한다.
