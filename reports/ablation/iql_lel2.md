# IQL-AMO L_E+L2_RMS ablation

기존 IQL 설정과 다른 실행이라 main에서 내렸습니다. 로그와 평가는 ablation으로 보존합니다.

## IQL-AMO π_E · L_E+L2_RMS · ablation

IQL-AMO의 β_E meta-loss를 **L_E (BPI) + L2_RMS**로 둔 ablation입니다. 벨만은 V, 평가 정책은 π_E입니다. 초기 **beta=1, 2, 5**, **rho_lr=2e-3, 1e-3, 3e-4**, **seed 0~3**.

- 초기 `beta_initial`로 묶으며 표의 `rho_lr`는 beta의 meta 학습률입니다.
- **1M checkpoint 평가**를 사용하며, 같은 checkpoint에 반복평가 평균이 있으면 우선합니다. 없으면 마지막 일반 평가를 사용합니다.
- `†300k`처럼 표시한 값은 1M 평가가 없는 실행의 최신 점수입니다. `대기`는 실행은 있으나 평가가 없고, `—`는 업로드된 실행이 없습니다.
- **평균±표준편차는 seed 0~3 모두 1M 평가가 있을 때만** 계산합니다. 표준편차는 네 시드 점수의 population std(ddof=0)입니다. `(n/4)`는 1M 평가를 확보한 시드 수입니다.
- 같은 설정·시드의 재실행은 1M 결과 중 높은 점수를 표시합니다. 1M 결과가 없으면 가장 진행된 step을 우선합니다. `⁺n`은 후보 실행 수이며, 모든 후보와 채택 여부는 [실행별 CSV](iql_lel2_runs.csv)에 남깁니다.
- `T`=Torch, `J`=JAX. 평가 episode 수·집계 유형·코드 버전은 [실행별 CSV](iql_lel2_runs.csv)에서 확인할 수 있습니다. 각 학습률은 별도 행이며 서로 섞어 평균내지 않습니다.
- family `iql_amo_lel2`만 집계하며 기존 IQL-AMO 표와 칸을 나누지 않습니다.

| 방법 | 초기값 | 평가 있는 시드 칸 | 1M 평가 시드 칸 | 4시드 완료 환경×lr |
|---|---:|---:|---:|---:|
| [IQL-AMO](#iql_lel2-iql_amo-1) | beta=1 | 54/180 | 54/180 | 0/45 |
| [IQL-AMO](#iql_lel2-iql_amo-2) | beta=2 | 0/180 | 0/180 | 0/45 |
| [IQL-AMO](#iql_lel2-iql_amo-5) | beta=5 | 0/180 | 0/180 | 0/45 |

## IQL-AMO L_E+L2_RMS

<a id="iql_lel2-iql_amo-1"></a>

<details open>
<summary><strong>beta = 1</strong></summary>

### IQL-AMO L_E+L2_RMS · beta=1 · Locomotion

| 환경 | rho_lr | seed 0 | seed 1 | seed 2 | seed 3 | 1M 평균 ± std |
|---|---|---|---|---|---|---|
| halfcheetah-medium-v2 | 2e-3 | [43.66 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/halfcheetah-medium-v2/2e-3/seed_0/hcm_s0_beta1__513f09fe) | — | — | — | — (1/4) |
| halfcheetah-medium-v2 | 1e-3 | [43.49 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/halfcheetah-medium-v2/1e-3/seed_0/hcm_s0_beta1_rlr0p001__dcf3a2d9) | — | — | — | — (1/4) |
| halfcheetah-medium-v2 | 3e-4 | [42.82 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/halfcheetah-medium-v2/3e-4/seed_0/hcm_s0_beta1_rlr0p0003__2b27ef52) | — | — | — | — (1/4) |
| halfcheetah-medium-replay-v2 | 2e-3 | [35.37 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/halfcheetah-medium-replay-v2/2e-3/seed_0/hcmr_s0_beta1__3f103bbc) | — | — | — | — (1/4) |
| halfcheetah-medium-replay-v2 | 1e-3 | [39.14 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/halfcheetah-medium-replay-v2/1e-3/seed_0/hcmr_s0_beta1_rlr0p001__9bf804ca) | — | — | — | — (1/4) |
| halfcheetah-medium-replay-v2 | 3e-4 | [40.60 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/halfcheetah-medium-replay-v2/3e-4/seed_0/hcmr_s0_beta1_rlr0p0003__1e58e369) | — | — | — | — (1/4) |
| halfcheetah-medium-expert-v2 | 2e-3 | [66.77 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/halfcheetah-medium-expert-v2/2e-3/seed_0/hme_s0_beta1__93411d3a) | — | — | — | — (1/4) |
| halfcheetah-medium-expert-v2 | 1e-3 | [82.76 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/halfcheetah-medium-expert-v2/1e-3/seed_0/hme_s0_beta1_rlr0p001__d86669e9) | — | — | — | — (1/4) |
| halfcheetah-medium-expert-v2 | 3e-4 | [87.13 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/halfcheetah-medium-expert-v2/3e-4/seed_0/hme_s0_beta1_rlr0p0003__62ff513d) | — | — | — | — (1/4) |
| hopper-medium-v2 | 2e-3 | [54.94 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/hopper-medium-v2/2e-3/seed_0/hopm_s0_beta1__05373d51) | [62.23 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/hopper-medium-v2/2e-3/seed_1/hopm_s1_beta1__ff8c9e5b) | — | — | — (2/4) |
| hopper-medium-v2 | 1e-3 | [53.09 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/hopper-medium-v2/1e-3/seed_0/hopm_s0_beta1_rlr0p001__0fd1df03) | [61.73 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/hopper-medium-v2/1e-3/seed_1/hopm_s1_beta1_rlr0p001__d131af40) | — | — | — (2/4) |
| hopper-medium-v2 | 3e-4 | [68.21 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/hopper-medium-v2/3e-4/seed_0/hopm_s0_beta1_rlr0p0003__77be9fd4) | [65.90 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/hopper-medium-v2/3e-4/seed_1/hopm_s1_beta1_rlr0p0003__f54fbadb) | — | — | — (2/4) |
| hopper-medium-replay-v2 | 2e-3 | [46.98 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/hopper-medium-replay-v2/2e-3/seed_0/hopmr_s0_beta1__f3dd2b83) | [38.77 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/hopper-medium-replay-v2/2e-3/seed_1/hopmr_s1_beta1__b2c7876e) | — | — | — (2/4) |
| hopper-medium-replay-v2 | 1e-3 | [43.51 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/hopper-medium-replay-v2/1e-3/seed_0/hopmr_s0_beta1_rlr0p001__7e412f89) | [41.46 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/hopper-medium-replay-v2/1e-3/seed_1/hopmr_s1_beta1_rlr0p001__e8918963) | — | — | — (2/4) |
| hopper-medium-replay-v2 | 3e-4 | [37.80 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/hopper-medium-replay-v2/3e-4/seed_0/hopmr_s0_beta1_rlr0p0003__032bff4a) | [44.15 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/hopper-medium-replay-v2/3e-4/seed_1/hopmr_s1_beta1_rlr0p0003__f4ee5d59) | — | — | — (2/4) |
| hopper-medium-expert-v2 | 2e-3 | [53.01 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/hopper-medium-expert-v2/2e-3/seed_0/hopme_s0_beta1__ff27ed67) | [64.42 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/hopper-medium-expert-v2/2e-3/seed_1/hopme_s1_beta1__af0e5e99) | — | — | — (2/4) |
| hopper-medium-expert-v2 | 1e-3 | [56.15 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/hopper-medium-expert-v2/1e-3/seed_0/hopme_s0_beta1_rlr0p001__18a1fa20) | [70.43 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/hopper-medium-expert-v2/1e-3/seed_1/hopme_s1_beta1_rlr0p001__f0cb248f) | — | — | — (2/4) |
| hopper-medium-expert-v2 | 3e-4 | [99.28 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/hopper-medium-expert-v2/3e-4/seed_0/hopme_s0_beta1_rlr0p0003__47810a4a) | [111.32 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/hopper-medium-expert-v2/3e-4/seed_1/hopme_s1_beta1_rlr0p0003__8604e90d) | — | — | — (2/4) |
| walker2d-medium-v2 | 2e-3 | [73.34 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/walker2d-medium-v2/2e-3/seed_0/wm_s0_beta1__63847c7b) | — | — | — | — (1/4) |
| walker2d-medium-v2 | 1e-3 | [70.89 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/walker2d-medium-v2/1e-3/seed_0/wm_s0_beta1_rlr0p001__3de165d4) | — | — | — | — (1/4) |
| walker2d-medium-v2 | 3e-4 | [82.10 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/walker2d-medium-v2/3e-4/seed_0/wm_s0_beta1_rlr0p0003__3bf7ea36) | [대기 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/walker2d-medium-v2/3e-4/seed_1/wm_s1_beta1_rlr0p0003__d48777b9) | — | — | — (1/4) |
| walker2d-medium-replay-v2 | 2e-3 | [27.98 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/walker2d-medium-replay-v2/2e-3/seed_0/wmr_s0_beta1__bd961829) | — | — | — | — (1/4) |
| walker2d-medium-replay-v2 | 1e-3 | [21.09 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/walker2d-medium-replay-v2/1e-3/seed_0/wmr_s0_beta1_rlr0p001__5cd1dc57) | — | — | — | — (1/4) |
| walker2d-medium-replay-v2 | 3e-4 | [21.18 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/walker2d-medium-replay-v2/3e-4/seed_0/wmr_s0_beta1_rlr0p0003__f8538279) | — | — | — | — (1/4) |
| walker2d-medium-expert-v2 | 2e-3 | [108.59 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/walker2d-medium-expert-v2/2e-3/seed_0/wme_s0_beta1__896900f0) | — | — | — | — (1/4) |
| walker2d-medium-expert-v2 | 1e-3 | [108.55 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/walker2d-medium-expert-v2/1e-3/seed_0/wme_s0_beta1_rlr0p001__77f53bb0) | — | — | — | — (1/4) |
| walker2d-medium-expert-v2 | 3e-4 | [108.66 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/walker2d-medium-expert-v2/3e-4/seed_0/wme_s0_beta1_rlr0p0003__cc6c09a7) | — | — | — | — (1/4) |
### IQL-AMO L_E+L2_RMS · beta=1 · AntMaze

| 환경 | rho_lr | seed 0 | seed 1 | seed 2 | seed 3 | 1M 평균 ± std |
|---|---|---|---|---|---|---|
| antmaze-umaze-v2 | 2e-3 | [76.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/antmaze-umaze-v2/2e-3/seed_0/amu_s0_beta1__d0f2f968) | — | — | — | — (1/4) |
| antmaze-umaze-v2 | 1e-3 | [66.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/antmaze-umaze-v2/1e-3/seed_0/amu_s0_beta1_rlr0p001__f3e9f58c) | — | — | — | — (1/4) |
| antmaze-umaze-v2 | 3e-4 | [74.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/antmaze-umaze-v2/3e-4/seed_0/amu_s0_beta1_rlr0p0003__a388fa36) | — | — | — | — (1/4) |
| antmaze-umaze-diverse-v2 | 2e-3 | [32.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/antmaze-umaze-diverse-v2/2e-3/seed_0/amud_s0_beta1__08681e97) | — | — | — | — (1/4) |
| antmaze-umaze-diverse-v2 | 1e-3 | [50.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/antmaze-umaze-diverse-v2/1e-3/seed_0/amud_s0_beta1_rlr0p001__69367842) | — | — | — | — (1/4) |
| antmaze-umaze-diverse-v2 | 3e-4 | [42.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/antmaze-umaze-diverse-v2/3e-4/seed_0/amud_s0_beta1_rlr0p0003__6c71ff4e) | — | — | — | — (1/4) |
| antmaze-medium-play-v2 | 2e-3 | [0.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/antmaze-medium-play-v2/2e-3/seed_0/ammp_s0_beta1__0b2f1366) | — | — | — | — (1/4) |
| antmaze-medium-play-v2 | 1e-3 | [0.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/antmaze-medium-play-v2/1e-3/seed_0/ammp_s0_beta1_rlr0p001__b3dbf06a) | — | — | — | — (1/4) |
| antmaze-medium-play-v2 | 3e-4 | [0.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/antmaze-medium-play-v2/3e-4/seed_0/ammp_s0_beta1_rlr0p0003__288aa994) | — | — | — | — (1/4) |
| antmaze-medium-diverse-v2 | 2e-3 | [0.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/antmaze-medium-diverse-v2/2e-3/seed_0/ammd_s0_beta1__8fe45916) | — | — | — | — (1/4) |
| antmaze-medium-diverse-v2 | 1e-3 | [0.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/antmaze-medium-diverse-v2/1e-3/seed_0/ammd_s0_beta1_rlr0p001__4432ad57) | — | — | — | — (1/4) |
| antmaze-medium-diverse-v2 | 3e-4 | [4.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/antmaze-medium-diverse-v2/3e-4/seed_0/ammd_s0_beta1_rlr0p0003__a1bdf094) | — | — | — | — (1/4) |
| antmaze-large-play-v2 | 2e-3 | [0.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/antmaze-large-play-v2/2e-3/seed_0/amlp_s0_beta1__2cd69014) | — | — | — | — (1/4) |
| antmaze-large-play-v2 | 1e-3 | [0.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/antmaze-large-play-v2/1e-3/seed_0/amlp_s0_beta1_rlr0p001__35556e8b) | — | — | — | — (1/4) |
| antmaze-large-play-v2 | 3e-4 | [0.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/antmaze-large-play-v2/3e-4/seed_0/amlp_s0_beta1_rlr0p0003__632d740e) | — | — | — | — (1/4) |
| antmaze-large-diverse-v2 | 2e-3 | [0.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/antmaze-large-diverse-v2/2e-3/seed_0/amld_s0_beta1__bf35e96d) | — | — | — | — (1/4) |
| antmaze-large-diverse-v2 | 1e-3 | [0.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/antmaze-large-diverse-v2/1e-3/seed_0/amld_s0_beta1_rlr0p001__192f9975) | — | — | — | — (1/4) |
| antmaze-large-diverse-v2 | 3e-4 | [0.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/a40e030d41289d8d48474ed621dfc38cda75a4a8/ablation/iql_amo/antmaze-large-diverse-v2/3e-4/seed_0/amld_s0_beta1_rlr0p0003__9f6332c8) | — | — | — | — (1/4) |

</details>

<a id="iql_lel2-iql_amo-2"></a>

<details open>
<summary><strong>beta = 2</strong></summary>

### IQL-AMO L_E+L2_RMS · beta=2 · Locomotion

| 환경 | rho_lr | seed 0 | seed 1 | seed 2 | seed 3 | 1M 평균 ± std |
|---|---|---|---|---|---|---|
| halfcheetah-medium-v2 | 2e-3 | — | — | — | — | — (0/4) |
| halfcheetah-medium-v2 | 1e-3 | — | — | — | — | — (0/4) |
| halfcheetah-medium-v2 | 3e-4 | — | — | — | — | — (0/4) |
| halfcheetah-medium-replay-v2 | 2e-3 | — | — | — | — | — (0/4) |
| halfcheetah-medium-replay-v2 | 1e-3 | — | — | — | — | — (0/4) |
| halfcheetah-medium-replay-v2 | 3e-4 | — | — | — | — | — (0/4) |
| halfcheetah-medium-expert-v2 | 2e-3 | — | — | — | — | — (0/4) |
| halfcheetah-medium-expert-v2 | 1e-3 | — | — | — | — | — (0/4) |
| halfcheetah-medium-expert-v2 | 3e-4 | — | — | — | — | — (0/4) |
| hopper-medium-v2 | 2e-3 | — | — | — | — | — (0/4) |
| hopper-medium-v2 | 1e-3 | — | — | — | — | — (0/4) |
| hopper-medium-v2 | 3e-4 | — | — | — | — | — (0/4) |
| hopper-medium-replay-v2 | 2e-3 | — | — | — | — | — (0/4) |
| hopper-medium-replay-v2 | 1e-3 | — | — | — | — | — (0/4) |
| hopper-medium-replay-v2 | 3e-4 | — | — | — | — | — (0/4) |
| hopper-medium-expert-v2 | 2e-3 | — | — | — | — | — (0/4) |
| hopper-medium-expert-v2 | 1e-3 | — | — | — | — | — (0/4) |
| hopper-medium-expert-v2 | 3e-4 | — | — | — | — | — (0/4) |
| walker2d-medium-v2 | 2e-3 | — | — | — | — | — (0/4) |
| walker2d-medium-v2 | 1e-3 | — | — | — | — | — (0/4) |
| walker2d-medium-v2 | 3e-4 | — | — | — | — | — (0/4) |
| walker2d-medium-replay-v2 | 2e-3 | — | — | — | — | — (0/4) |
| walker2d-medium-replay-v2 | 1e-3 | — | — | — | — | — (0/4) |
| walker2d-medium-replay-v2 | 3e-4 | — | — | — | — | — (0/4) |
| walker2d-medium-expert-v2 | 2e-3 | — | — | — | — | — (0/4) |
| walker2d-medium-expert-v2 | 1e-3 | — | — | — | — | — (0/4) |
| walker2d-medium-expert-v2 | 3e-4 | — | — | — | — | — (0/4) |
### IQL-AMO L_E+L2_RMS · beta=2 · AntMaze

| 환경 | rho_lr | seed 0 | seed 1 | seed 2 | seed 3 | 1M 평균 ± std |
|---|---|---|---|---|---|---|
| antmaze-umaze-v2 | 2e-3 | — | — | — | — | — (0/4) |
| antmaze-umaze-v2 | 1e-3 | — | — | — | — | — (0/4) |
| antmaze-umaze-v2 | 3e-4 | — | — | — | — | — (0/4) |
| antmaze-umaze-diverse-v2 | 2e-3 | — | — | — | — | — (0/4) |
| antmaze-umaze-diverse-v2 | 1e-3 | — | — | — | — | — (0/4) |
| antmaze-umaze-diverse-v2 | 3e-4 | — | — | — | — | — (0/4) |
| antmaze-medium-play-v2 | 2e-3 | — | — | — | — | — (0/4) |
| antmaze-medium-play-v2 | 1e-3 | — | — | — | — | — (0/4) |
| antmaze-medium-play-v2 | 3e-4 | — | — | — | — | — (0/4) |
| antmaze-medium-diverse-v2 | 2e-3 | — | — | — | — | — (0/4) |
| antmaze-medium-diverse-v2 | 1e-3 | — | — | — | — | — (0/4) |
| antmaze-medium-diverse-v2 | 3e-4 | — | — | — | — | — (0/4) |
| antmaze-large-play-v2 | 2e-3 | — | — | — | — | — (0/4) |
| antmaze-large-play-v2 | 1e-3 | — | — | — | — | — (0/4) |
| antmaze-large-play-v2 | 3e-4 | — | — | — | — | — (0/4) |
| antmaze-large-diverse-v2 | 2e-3 | — | — | — | — | — (0/4) |
| antmaze-large-diverse-v2 | 1e-3 | — | — | — | — | — (0/4) |
| antmaze-large-diverse-v2 | 3e-4 | — | — | — | — | — (0/4) |

</details>

<a id="iql_lel2-iql_amo-5"></a>

<details open>
<summary><strong>beta = 5</strong></summary>

### IQL-AMO L_E+L2_RMS · beta=5 · Locomotion

| 환경 | rho_lr | seed 0 | seed 1 | seed 2 | seed 3 | 1M 평균 ± std |
|---|---|---|---|---|---|---|
| halfcheetah-medium-v2 | 2e-3 | — | — | — | — | — (0/4) |
| halfcheetah-medium-v2 | 1e-3 | — | — | — | — | — (0/4) |
| halfcheetah-medium-v2 | 3e-4 | — | — | — | — | — (0/4) |
| halfcheetah-medium-replay-v2 | 2e-3 | — | — | — | — | — (0/4) |
| halfcheetah-medium-replay-v2 | 1e-3 | — | — | — | — | — (0/4) |
| halfcheetah-medium-replay-v2 | 3e-4 | — | — | — | — | — (0/4) |
| halfcheetah-medium-expert-v2 | 2e-3 | — | — | — | — | — (0/4) |
| halfcheetah-medium-expert-v2 | 1e-3 | — | — | — | — | — (0/4) |
| halfcheetah-medium-expert-v2 | 3e-4 | — | — | — | — | — (0/4) |
| hopper-medium-v2 | 2e-3 | — | — | — | — | — (0/4) |
| hopper-medium-v2 | 1e-3 | — | — | — | — | — (0/4) |
| hopper-medium-v2 | 3e-4 | — | — | — | — | — (0/4) |
| hopper-medium-replay-v2 | 2e-3 | — | — | — | — | — (0/4) |
| hopper-medium-replay-v2 | 1e-3 | — | — | — | — | — (0/4) |
| hopper-medium-replay-v2 | 3e-4 | — | — | — | — | — (0/4) |
| hopper-medium-expert-v2 | 2e-3 | — | — | — | — | — (0/4) |
| hopper-medium-expert-v2 | 1e-3 | — | — | — | — | — (0/4) |
| hopper-medium-expert-v2 | 3e-4 | — | — | — | — | — (0/4) |
| walker2d-medium-v2 | 2e-3 | — | — | — | — | — (0/4) |
| walker2d-medium-v2 | 1e-3 | — | — | — | — | — (0/4) |
| walker2d-medium-v2 | 3e-4 | — | — | — | — | — (0/4) |
| walker2d-medium-replay-v2 | 2e-3 | — | — | — | — | — (0/4) |
| walker2d-medium-replay-v2 | 1e-3 | — | — | — | — | — (0/4) |
| walker2d-medium-replay-v2 | 3e-4 | — | — | — | — | — (0/4) |
| walker2d-medium-expert-v2 | 2e-3 | — | — | — | — | — (0/4) |
| walker2d-medium-expert-v2 | 1e-3 | — | — | — | — | — (0/4) |
| walker2d-medium-expert-v2 | 3e-4 | — | — | — | — | — (0/4) |
### IQL-AMO L_E+L2_RMS · beta=5 · AntMaze

| 환경 | rho_lr | seed 0 | seed 1 | seed 2 | seed 3 | 1M 평균 ± std |
|---|---|---|---|---|---|---|
| antmaze-umaze-v2 | 2e-3 | — | — | — | — | — (0/4) |
| antmaze-umaze-v2 | 1e-3 | — | — | — | — | — (0/4) |
| antmaze-umaze-v2 | 3e-4 | — | — | — | — | — (0/4) |
| antmaze-umaze-diverse-v2 | 2e-3 | — | — | — | — | — (0/4) |
| antmaze-umaze-diverse-v2 | 1e-3 | — | — | — | — | — (0/4) |
| antmaze-umaze-diverse-v2 | 3e-4 | — | — | — | — | — (0/4) |
| antmaze-medium-play-v2 | 2e-3 | — | — | — | — | — (0/4) |
| antmaze-medium-play-v2 | 1e-3 | — | — | — | — | — (0/4) |
| antmaze-medium-play-v2 | 3e-4 | — | — | — | — | — (0/4) |
| antmaze-medium-diverse-v2 | 2e-3 | — | — | — | — | — (0/4) |
| antmaze-medium-diverse-v2 | 1e-3 | — | — | — | — | — (0/4) |
| antmaze-medium-diverse-v2 | 3e-4 | — | — | — | — | — (0/4) |
| antmaze-large-play-v2 | 2e-3 | — | — | — | — | — (0/4) |
| antmaze-large-play-v2 | 1e-3 | — | — | — | — | — (0/4) |
| antmaze-large-play-v2 | 3e-4 | — | — | — | — | — (0/4) |
| antmaze-large-diverse-v2 | 2e-3 | — | — | — | — | — (0/4) |
| antmaze-large-diverse-v2 | 1e-3 | — | — | — | — | — (0/4) |
| antmaze-large-diverse-v2 | 3e-4 | — | — | — | — | — (0/4) |

</details>

## IQL L_E+L2_RMS 집계 브랜치

| 브랜치 | 로그 snapshot | IQL L_E+L2_RMS 실행 |
|---|---|---:|
| choi | [77632f62](https://github.com/seonvin0319/amo_log/commit/77632f62205b32266edb06cf137e9830ea5b4236) | 0 |
| ext_csh | [20d124f1](https://github.com/seonvin0319/amo_log/commit/20d124f13b399ce157604b0bf269ca6e7d36a91b) | 0 |
| ext_csv | [93c05076](https://github.com/seonvin0319/amo_log/commit/93c0507622bb235eac004caf472f947931abc83d) | 0 |
| offrl | [b0732b4d](https://github.com/seonvin0319/amo_log/commit/b0732b4d115128740cd7309b2c96d86a6af07a58) | 0 |
| shchoi | [7b71e19c](https://github.com/seonvin0319/amo_log/commit/7b71e19cb5fbc8c2f54499139787ea07153faa7d) | 0 |
| svcho | [a40e030d](https://github.com/seonvin0319/amo_log/commit/a40e030d41289d8d48474ed621dfc38cda75a4a8) | 55 |

머신 브랜치의 로그 검증이 성공하면 이 표를 자동 갱신합니다. 30분 주기의 보완 갱신과 [수동 갱신](https://github.com/seonvin0319/amo_log/actions/workflows/refresh-index.yml)도 지원합니다.
