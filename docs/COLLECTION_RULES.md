# 로그 저장 규칙

- 본 실험은 `main/<method>/<env>/<meta_lr>/seed_<seed>/<run_id>/`.
- Ablation은 `ablation/<method>/<env>/<meta_lr>/seed_<seed>/<run_id>/`.
- 방법 이름은 td3_amo, iql_amo, td3bc+rc, iql, a2pr, wpc, aspc.
- meta_lr는 TD3의 T_lr, IQL+AMO의 rho_lr(beta_lr). baseline은 이 폴더를 생략.
- AMO 본 실험: 초기 T_E=T_B=1 또는 beta_initial=1, meta lr=1e-3/2e-3/3e-4.
- Baseline 고유 beta=3/10 등은 AMO 초기 beta와 다르므로 baseline을 잘못 ablation으로 옮기지 않는다.
- 다른 초기값, lr, loss/critic/비율/스케줄 실험은 ablation. 이유는 run_meta.json의 classification_reasons에 기록.
- seed마다 별도 run_id를 보존한다. 환경·lr·seed가 같아도 다른 실행을 덮어쓰지 않는다.
- config.yaml과 원본 metrics/eval 파일은 이름과 내용을 그대로 보존한다. backend와 코드 commit을 반드시 기록.
- run_meta.json: source_path, git.code_commit, method, section, env, seed, backend, meta_lr, rel_path, original_rel_path, layout_version=2.
- 현재 제거된 JAX 실행은 catalog/removed_legacy_jax.json의 식별자 목록만 유지한다. 점수·평가 로그는 남기지 않는다.
- 제외는 기존 source_path와 code_commit 조합에만 적용한다. 이후 수정된 JAX 버전은 새 code_commit과 backend=jax를 명시해 수집할 수 있다.
- 불완전한 실행도 보존한다. 완료 여부는 평가 step으로 확인하고 디렉터리 존재만으로 완료 판정하지 않는다.
- 자동 수집: ingest_runs.py → log_layout.normalize → build_catalog.py → 머신 브랜치 commit/push.
- 새 분류가 필요한 경우 scripts/log_layout.py의 기준을 갱신하고 동일 기준을 모든 머신 브랜치에 반영한다.
