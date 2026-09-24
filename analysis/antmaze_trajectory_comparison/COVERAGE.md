# AntMaze trajectory comparison coverage

medium-diverse 그림에 쓴 α=5 design-ablation 체크포인트는 이 세 환경에 없다. 두 방법의 α와 α_lr을 맞추기 위해, 둘 다 step 1,000,000이 있는 α=1, α_lr=3e-4 (`a1_*_r3e4`)만 평가했다. α=5 TD3+RAPO(`a5_*_r3e4`)는 seed 0–3이 있지만, 같은 α의 Shared actor가 없어 비교에 넣지 않았다.

| 환경 | 평가한 seed | 없는 Shared actor `step_1000000.npz` |
|---|---|---|
| antmaze-medium-play-v2 | 0, 1 | `td3_amo_execonly_main_seeds03/jobs/a1_am-mp_r3e4_s2`, `..._s3` |
| antmaze-large-play-v2 | 0 | `.../a1_am-lp_r3e4_s1`, `s2`, `s3` (디렉터리 없음) |
| antmaze-large-diverse-v2 | 0 | `.../a1_am-ld_r3e4_s1`, `s2`, `s3` (디렉터리 없음) |

공통 루트: `/home/svcho/amo/results/`. TD3+RAPO는 `td3_amo_bootrms_maincand_seeds03/jobs/a1_<env>_r3e4_s{seed}`이고 seed 0–3 체크포인트는 있다. 없는 seed를 다른 α나 중간 step으로 대체하지 않았다.
