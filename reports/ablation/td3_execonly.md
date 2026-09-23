## TD3-AMO π_E-only · L_E+L2_RMS · ablation

TD3-AMO π_E-only (`execution_only=true`) ablation입니다. α_E meta-loss는 **L_E (BPI) + L2_RMS**이며 π_E가 환경과 벨만 타깃을 모두 맡습니다. 초기 **alpha=1, 2, 5**, **alpha_lr=2e-3, 1e-3, 3e-4**, **seed 0~3**.

- 초기 `alpha_E`로 묶습니다. π_B/α_B는 없습니다.
- **1M checkpoint 평가**를 사용하며, 같은 checkpoint에 반복평가 평균이 있으면 우선합니다. 없으면 마지막 일반 평가를 사용합니다.
- `†300k`처럼 표시한 값은 1M 평가가 없는 실행의 최신 점수입니다. `대기`는 실행은 있으나 평가가 없고, `—`는 업로드된 실행이 없습니다.
- **평균±표준편차는 seed 0~3 모두 1M 평가가 있을 때만** 계산합니다. 표준편차는 네 시드 점수의 population std(ddof=0)입니다. `(n/4)`는 1M 평가를 확보한 시드 수입니다.
- 같은 설정·시드의 재실행은 1M 결과 중 높은 점수를 표시합니다. 1M 결과가 없으면 가장 진행된 step을 우선합니다. `⁺n`은 후보 실행 수이며, 모든 후보와 채택 여부는 [실행별 CSV](execonly_runs.csv)에 남깁니다.
- `T`=Torch, `J`=JAX. 평가 episode 수·집계 유형·코드 버전은 [실행별 CSV](execonly_runs.csv)에서 확인할 수 있습니다. 각 학습률은 별도 행이며 서로 섞어 평균내지 않습니다.
- family `td3_amo_execonly_main`만 집계하며 기존 BootRMS dual-actor 표와 칸을 나누지 않습니다.

| 방법 | 초기값 | 평가 있는 시드 칸 | 1M 평가 시드 칸 | 4시드 완료 환경×lr |
|---|---:|---:|---:|---:|
| [TD3-AMO](#execonly-td3_amo-1) | alpha=1 | 70/180 | 70/180 | 0/45 |
| [TD3-AMO](#execonly-td3_amo-2) | alpha=2 | 0/180 | 0/180 | 0/45 |
| [TD3-AMO](#execonly-td3_amo-5) | alpha=5 | 0/180 | 0/180 | 0/45 |

## TD3-AMO π_E-only

<a id="execonly-td3_amo-1"></a>

<details open>
<summary><strong>alpha = 1</strong></summary>

### TD3-AMO π_E-only · alpha=1 · Locomotion

| 환경 | alpha_lr | seed 0 | seed 1 | seed 2 | seed 3 | 1M 평균 ± std |
|---|---|---|---|---|---|---|
| halfcheetah-medium-v2 | 2e-3 | [56.45 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/halfcheetah-medium-v2/2e-3/seed_0/hcm_s0_alr0p002__530cac95) | — | — | — | — (1/4) |
| halfcheetah-medium-v2 | 1e-3 | [53.43 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/halfcheetah-medium-v2/1e-3/seed_0/hcm_s0_alr0p001__f43cf145) | — | — | — | — (1/4) |
| halfcheetah-medium-v2 | 3e-4 | [48.90 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/halfcheetah-medium-v2/3e-4/seed_0/hcm_s0_alr0p0003__1687401a) | — | — | — | — (1/4) |
| halfcheetah-medium-replay-v2 | 2e-3 | [49.96 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/halfcheetah-medium-replay-v2/2e-3/seed_0/hcmr_s0_alr0p002__4b9ef4b3) | — | — | — | — (1/4) |
| halfcheetah-medium-replay-v2 | 1e-3 | [46.98 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/halfcheetah-medium-replay-v2/1e-3/seed_0/hcmr_s0_alr0p001__1b6534bc) | — | — | — | — (1/4) |
| halfcheetah-medium-replay-v2 | 3e-4 | [45.79 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/halfcheetah-medium-replay-v2/3e-4/seed_0/hcmr_s0_alr0p0003__7f527b09) | — | — | — | — (1/4) |
| halfcheetah-medium-expert-v2 | 2e-3 | [102.20 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/halfcheetah-medium-expert-v2/2e-3/seed_0/hme_s0_alr0p002__c50ef0df) | — | — | — | — (1/4) |
| halfcheetah-medium-expert-v2 | 1e-3 | [98.30 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/halfcheetah-medium-expert-v2/1e-3/seed_0/hme_s0_alr0p001__031001da) | — | — | — | — (1/4) |
| halfcheetah-medium-expert-v2 | 3e-4 | [97.70 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/halfcheetah-medium-expert-v2/3e-4/seed_0/hme_s0_alr0p0003__6636626f) | — | — | — | — (1/4) |
| hopper-medium-v2 | 2e-3 | [92.59 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/hopper-medium-v2/2e-3/seed_0/hopm_s0_alr0p002__494c20d9) | [101.01 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/hopper-medium-v2/2e-3/seed_1/hopm_s1_alr0p002__be2ffb0f) | — | — | — (2/4) |
| hopper-medium-v2 | 1e-3 | [70.96 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/hopper-medium-v2/1e-3/seed_0/hopm_s0_alr0p001__a2061d4d) | [92.50 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/hopper-medium-v2/1e-3/seed_1/hopm_s1_alr0p001__e6123647) | — | — | — (2/4) |
| hopper-medium-v2 | 3e-4 | [60.04 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/hopper-medium-v2/3e-4/seed_0/hopm_s0_alr0p0003__56ffae25) | [61.18 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/hopper-medium-v2/3e-4/seed_1/hopm_s1_alr0p0003__2b00012f) | — | — | — (2/4) |
| hopper-medium-replay-v2 | 2e-3 | [102.14 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/hopper-medium-replay-v2/2e-3/seed_0/hopmr_s0_alr0p002__ec144868) | [100.64 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/hopper-medium-replay-v2/2e-3/seed_1/hopmr_s1_alr0p002__0c401f49) | — | — | — (2/4) |
| hopper-medium-replay-v2 | 1e-3 | [101.25 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/hopper-medium-replay-v2/1e-3/seed_0/hopmr_s0_alr0p001__f01d68cb) | [94.59 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/hopper-medium-replay-v2/1e-3/seed_1/hopmr_s1_alr0p001__0297a27a) | — | — | — (2/4) |
| hopper-medium-replay-v2 | 3e-4 | [91.84 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/hopper-medium-replay-v2/3e-4/seed_0/hopmr_s0_alr0p0003__06c28a69) | [22.73 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/hopper-medium-replay-v2/3e-4/seed_1/hopmr_s1_alr0p0003__205ac285) | — | — | — (2/4) |
| hopper-medium-expert-v2 | 2e-3 | [112.22 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/hopper-medium-expert-v2/2e-3/seed_0/hopme_s0_alr0p002__f14c0d85) | [106.36 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/hopper-medium-expert-v2/2e-3/seed_1/hopme_s1_alr0p002__54b2f1c6) | — | — | — (2/4) |
| hopper-medium-expert-v2 | 1e-3 | [104.65 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/hopper-medium-expert-v2/1e-3/seed_0/hopme_s0_alr0p001__3e2501d4) | [104.68 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/hopper-medium-expert-v2/1e-3/seed_1/hopme_s1_alr0p001__cc06a245) | — | — | — (2/4) |
| hopper-medium-expert-v2 | 3e-4 | [98.97 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/hopper-medium-expert-v2/3e-4/seed_0/hopme_s0_alr0p0003__b92d1ab5) | [101.45 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/hopper-medium-expert-v2/3e-4/seed_1/hopme_s1_alr0p0003__780a8d95) | — | — | — (2/4) |
| walker2d-medium-v2 | 2e-3 | [96.42 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/walker2d-medium-v2/2e-3/seed_0/wm_s0_alr0p002__6c29edbb) | [94.73 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/walker2d-medium-v2/2e-3/seed_1/wm_s1_alr0p002__586b3521) | — | — | — (2/4) |
| walker2d-medium-v2 | 1e-3 | [87.37 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/walker2d-medium-v2/1e-3/seed_0/wm_s0_alr0p001__af718e9a) | [87.95 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/walker2d-medium-v2/1e-3/seed_1/wm_s1_alr0p001__448f1f94) | — | — | — (2/4) |
| walker2d-medium-v2 | 3e-4 | [83.81 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/walker2d-medium-v2/3e-4/seed_0/wm_s0_alr0p0003__c230cf5d) | [84.13 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/walker2d-medium-v2/3e-4/seed_1/wm_s1_alr0p0003__42493313) | — | — | — (2/4) |
| walker2d-medium-replay-v2 | 2e-3 | [95.87 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/walker2d-medium-replay-v2/2e-3/seed_0/wmr_s0_alr0p002__5636e26f) | [94.61 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/walker2d-medium-replay-v2/2e-3/seed_1/wmr_s1_alr0p002__bdda96e9) | — | — | — (2/4) |
| walker2d-medium-replay-v2 | 1e-3 | [91.21 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/walker2d-medium-replay-v2/1e-3/seed_0/wmr_s0_alr0p001__23a4497a) | [96.95 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/walker2d-medium-replay-v2/1e-3/seed_1/wmr_s1_alr0p001__5d384c04) | — | — | — (2/4) |
| walker2d-medium-replay-v2 | 3e-4 | [87.35 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/walker2d-medium-replay-v2/3e-4/seed_0/wmr_s0_alr0p0003__9a6f7d3e) | [84.26 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/walker2d-medium-replay-v2/3e-4/seed_1/wmr_s1_alr0p0003__96eaee37) | — | — | — (2/4) |
| walker2d-medium-expert-v2 | 2e-3 | [110.54 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/walker2d-medium-expert-v2/2e-3/seed_0/wme_s0_alr0p002__b8142482) | [110.43 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/walker2d-medium-expert-v2/2e-3/seed_1/wme_s1_alr0p002__a1018eee) | — | — | — (2/4) |
| walker2d-medium-expert-v2 | 1e-3 | [109.69 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/walker2d-medium-expert-v2/1e-3/seed_0/wme_s0_alr0p001__5719d17d) | [109.64 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/walker2d-medium-expert-v2/1e-3/seed_1/wme_s1_alr0p001__2550e75c) | — | — | — (2/4) |
| walker2d-medium-expert-v2 | 3e-4 | [108.99 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/walker2d-medium-expert-v2/3e-4/seed_0/wme_s0_alr0p0003__3bffe451) | [108.89 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/walker2d-medium-expert-v2/3e-4/seed_1/wme_s1_alr0p0003__7ce3341f) | — | — | — (2/4) |
### TD3-AMO π_E-only · alpha=1 · AntMaze

| 환경 | alpha_lr | seed 0 | seed 1 | seed 2 | seed 3 | 1M 평균 ± std |
|---|---|---|---|---|---|---|
| antmaze-umaze-v2 | 2e-3 | [96.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-umaze-v2/2e-3/seed_0/amu_s0_alr0p002__c52052e7) | [92.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-umaze-v2/2e-3/seed_1/amu_s1_alr0p002__4ebe6636) | — | — | — (2/4) |
| antmaze-umaze-v2 | 1e-3 | [80.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-umaze-v2/1e-3/seed_0/amu_s0_alr0p001__1980b3bb) | [92.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-umaze-v2/1e-3/seed_1/amu_s1_alr0p001__1198a634) | — | — | — (2/4) |
| antmaze-umaze-v2 | 3e-4 | [94.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-umaze-v2/3e-4/seed_0/amu_s0_alr0p0003__7ee45fb6) | [94.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-umaze-v2/3e-4/seed_1/amu_s1_alr0p0003__62d732c6) | — | — | — (2/4) |
| antmaze-umaze-diverse-v2 | 2e-3 | [82.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-umaze-diverse-v2/2e-3/seed_0/amud_s0_alr0p002__abc73738) | [88.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-umaze-diverse-v2/2e-3/seed_1/amud_s1_alr0p002__85645d22) | — | — | — (2/4) |
| antmaze-umaze-diverse-v2 | 1e-3 | [90.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-umaze-diverse-v2/1e-3/seed_0/amud_s0_alr0p001__03c5bc7f) | [82.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-umaze-diverse-v2/1e-3/seed_1/amud_s1_alr0p001__7bd07efd) | — | — | — (2/4) |
| antmaze-umaze-diverse-v2 | 3e-4 | [84.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-umaze-diverse-v2/3e-4/seed_0/amud_s0_alr0p0003__21354926) | [74.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-umaze-diverse-v2/3e-4/seed_1/amud_s1_alr0p0003__447b4cc7) | — | — | — (2/4) |
| antmaze-medium-play-v2 | 2e-3 | [86.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-medium-play-v2/2e-3/seed_0/ammp_s0_alr0p002__1ee58866) | — | — | — | — (1/4) |
| antmaze-medium-play-v2 | 1e-3 | [84.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-medium-play-v2/1e-3/seed_0/ammp_s0_alr0p001__b4adaab4) | [대기 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-medium-play-v2/1e-3/seed_1/ammp_s1_alr0p001__8ab996a0) | — | — | — (1/4) |
| antmaze-medium-play-v2 | 3e-4 | [12.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-medium-play-v2/3e-4/seed_0/ammp_s0_alr0p0003__7c16c060) | [16.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-medium-play-v2/3e-4/seed_1/ammp_s1_alr0p0003__b2512e47) | — | — | — (2/4) |
| antmaze-medium-diverse-v2 | 2e-3 | [40.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-medium-diverse-v2/2e-3/seed_0/ammd_s0_alr0p002__a8a5783c) | — | — | — | — (1/4) |
| antmaze-medium-diverse-v2 | 1e-3 | [80.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-medium-diverse-v2/1e-3/seed_0/ammd_s0_alr0p001__f00d2b16) | — | — | — | — (1/4) |
| antmaze-medium-diverse-v2 | 3e-4 | [16.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-medium-diverse-v2/3e-4/seed_0/ammd_s0_alr0p0003__4fad2ec8) | — | — | — | — (1/4) |
| antmaze-large-play-v2 | 2e-3 | [52.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-large-play-v2/2e-3/seed_0/amlp_s0_alr0p002__6c4bd017) | — | — | — | — (1/4) |
| antmaze-large-play-v2 | 1e-3 | [38.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-large-play-v2/1e-3/seed_0/amlp_s0_alr0p001__bf427238) | — | — | — | — (1/4) |
| antmaze-large-play-v2 | 3e-4 | [0.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-large-play-v2/3e-4/seed_0/amlp_s0_alr0p0003__634e19c6) | — | — | — | — (1/4) |
| antmaze-large-diverse-v2 | 2e-3 | [62.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-large-diverse-v2/2e-3/seed_0/amld_s0_alr0p002__b5fc0d85) | — | — | — | — (1/4) |
| antmaze-large-diverse-v2 | 1e-3 | [12.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-large-diverse-v2/1e-3/seed_0/amld_s0_alr0p001__d6acb016) | — | — | — | — (1/4) |
| antmaze-large-diverse-v2 | 3e-4 | [0.00 svcho/J](https://github.com/seonvin0319/amo_log/tree/77030bba0c665054fb4889db9a22c906b9b6943b/ablation/td3_amo/antmaze-large-diverse-v2/3e-4/seed_0/amld_s0_alr0p0003__46aa9de3) | — | — | — | — (1/4) |

</details>

<a id="execonly-td3_amo-2"></a>

<details open>
<summary><strong>alpha = 2</strong></summary>

### TD3-AMO π_E-only · alpha=2 · Locomotion

| 환경 | alpha_lr | seed 0 | seed 1 | seed 2 | seed 3 | 1M 평균 ± std |
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
### TD3-AMO π_E-only · alpha=2 · AntMaze

| 환경 | alpha_lr | seed 0 | seed 1 | seed 2 | seed 3 | 1M 평균 ± std |
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

<a id="execonly-td3_amo-5"></a>

<details open>
<summary><strong>alpha = 5</strong></summary>

### TD3-AMO π_E-only · alpha=5 · Locomotion

| 환경 | alpha_lr | seed 0 | seed 1 | seed 2 | seed 3 | 1M 평균 ± std |
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
### TD3-AMO π_E-only · alpha=5 · AntMaze

| 환경 | alpha_lr | seed 0 | seed 1 | seed 2 | seed 3 | 1M 평균 ± std |
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

## TD3 π_E-only 집계 브랜치

| 브랜치 | 로그 snapshot | TD3 π_E-only 실행 |
|---|---|---:|
| choi | [77632f62](https://github.com/seonvin0319/amo_log/commit/77632f62205b32266edb06cf137e9830ea5b4236) | 0 |
| ext_csh | [9bf1225a](https://github.com/seonvin0319/amo_log/commit/9bf1225aad756fa56c83264928dc817209dc96eb) | 0 |
| ext_csv | [b49c45be](https://github.com/seonvin0319/amo_log/commit/b49c45be3a24b36f1ecf8d9d09e0e28d8ad6ceca) | 0 |
| offrl | [b0732b4d](https://github.com/seonvin0319/amo_log/commit/b0732b4d115128740cd7309b2c96d86a6af07a58) | 0 |
| shchoi | [4bf9f721](https://github.com/seonvin0319/amo_log/commit/4bf9f72126d238106039cc07c3d2d4559ac92f0c) | 0 |
| svcho | [77030bba](https://github.com/seonvin0319/amo_log/commit/77030bba0c665054fb4889db9a22c906b9b6943b) | 71 |

머신 브랜치의 로그 검증이 성공하면 이 표를 자동 갱신합니다. 30분 주기의 보완 갱신과 [수동 갱신](https://github.com/seonvin0319/amo_log/actions/workflows/refresh-index.yml)도 지원합니다.
