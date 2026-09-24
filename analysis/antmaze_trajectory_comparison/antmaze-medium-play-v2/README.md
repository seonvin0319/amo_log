# antmaze-medium-play-v2 trajectory comparison

Shared actor와 TD3+RAPO (`L_E=-B_π`, `L_B=L2_RMS`)의 step 1,000,000 체크포인트를 CPU에서 평가한 궤적이다. 재학습하지 않았고, 각 run의 `eval.jsonl`에는 쓰지 않았다.

## 결과

- Shared actor: s0 3/25 (12.0%), s1 7/25 (28.0%); seed 평균 20.0% ± 11.3% (sample SD, ddof=1)
- TD3+RAPO: s0 2/25 (8.0%), s1 0/25 (0.0%); seed 평균 4.0% ± 5.7% (sample SD, ddof=1)

성공률의 평균은 포함된 학습 seed의 에피소드 성공률에 대한 값이다. seed가 2개 이상일 때만 sample SD(ddof=1)를 붙인다. 에피소드 수를 seed 수로 세지 않는다.

## 체크포인트

- Shared actor 예: `/home/svcho/amo/results/td3_amo_execonly_main_seeds03/jobs/a1_am-mp_r3e4_s0/run/checkpoints/step_1000000.npz`
  - `execution_only: true`. 평가 정책은 `state['p']['actor']` (공유 actor).
- TD3+RAPO 예: `/home/svcho/amo/results/td3_amo_bootrms_maincand_seeds03/jobs/a1_am-mp_r3e4_s0/run/checkpoints/step_1000000.npz`
  - `execution_score: bpi` (`L_E=-B_π`), `bootstrap_loss: l2_rms` (`L_B=L2_RMS`), `execution_only: false`.
  - 평가 정책은 실행 actor `π_E` = `state['p']['actor']`. bootstrap actor는 평가에 쓰지 않았다.
- 포함된 학습 seed: [0, 1]

각 파일의 sha256, config, probe action은 `summary.json`과 `trajectories/*.json`에 있다.
관측 정규화는 같은 run의 `normalization.npz`를 `(obs - mean) / std`로 적용했다.
그림 좌표는 정규화 관측이 아니라 `get_xy().copy()`의 환경 XY다.

## 평가

- 환경: `antmaze-medium-play-v2` (D4RL, horizon 1000, sparse eval).
- 학습 seed `s`, 에피소드 `e`의 평가 seed: `10000 + 100*s + e` (`e=0..24`). 두 방법이 같은 seed를 쓴다.
- seed마다 `AntMazeEnv.seed`와 전역 `numpy.random`을 그 값으로 다시 고정한 뒤 `reset`한다. `NormalizedBoxEnv.seed`는 안쪽 미로 RNG로 전달되지 않으므로 초기 qpos를 만드는 `AntMazeEnv.np_random`을 직접 시드했다. 포함된 seed에서 두 방법의 초기 `qpos`/`qvel`, 시작 XY, 목표가 일치했다.
- 행동 선택은 `BaseAgent.act`의 deterministic tanh mean이다. 탐색 노이즈는 없다.
- 종료는 기존 평가와 같이 목표 거리 0.5 이하(성공, return 1) 또는 Gym TimeLimit 1000(timeout)이다.
- 코드 revision: `d9be263443ffc1ca8834e277418826cff4ac6dc0` (dirty=True). 로더는 체크포인트 트리와 키가 정확히 같을 때만 로드하고, 로드 후 배열이 파일과 다르면 중단한다.

## 다시 실행

기존 GPU/CPU 학습 평가와 겹치지 않게 프로세스 2개, 스레드 1개로 제한한다.

```bash
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH=/home/svcho/.mujoco/mujoco210
export LD_LIBRARY_PATH=/home/svcho/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

PY=/home/svcho/anaconda3/envs/offrl/bin/python
OUT=/home/svcho/amo/results/antmaze_trajectory_comparison/antmaze-medium-play-v2
$PY /home/svcho/amo/scripts/eval_antmaze_trajectory_comparison.py --smoke \
  --env antmaze-medium-play-v2 --seeds 0,1 --alpha 1 \
  --shared-jobs /home/svcho/amo/results/td3_amo_execonly_main_seeds03 --rapo-jobs /home/svcho/amo/results/td3_amo_bootrms_maincand_seeds03 \
  --shared-template 'jobs_s{seed}' --rapo-template 'jobs_s{seed}' \
  --out "$OUT/trajectories"
$PY /home/svcho/amo/scripts/eval_antmaze_trajectory_comparison.py --all --jobs 2 \
  --env antmaze-medium-play-v2 --seeds 0,1 --alpha 1 \
  --shared-jobs /home/svcho/amo/results/td3_amo_execonly_main_seeds03 --rapo-jobs /home/svcho/amo/results/td3_amo_bootrms_maincand_seeds03 \
  --shared-template 'jobs_s{seed}' --rapo-template 'jobs_s{seed}' \
  --out "$OUT/trajectories"
$PY /home/svcho/amo/scripts/plot_antmaze_trajectory_comparison.py --root "$OUT"
```

## 파일

- `trajectories/{shared,rapo}_s{0,1,2,3}.npz` 와 같은 이름의 `.json`
- `summary.json`
- `figures/trajectories_combined.{pdf,png}`
- `figures/trajectories_by_seed.{pdf,png}`
- `smoke/` 한 에피소드 확인 결과
