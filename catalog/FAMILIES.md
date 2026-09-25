# Parsed experiment families

Derived index only. Canonical data remain `run_meta.json`, `config.yaml`, and raw evaluation files.

| section | method | family | runs | scored | complete@1M | environments | initializations |
|---|---|---|---:|---:|---:|---|---|
| ablation | fql_amo | fql_amo_jax | 49 | 8 | 8 | halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;walker2d-medium-expert-v2;walker2d-medium-replay-v2;walker2d-medium-v2 | alpha_E/B=10/10 |
| ablation | iql_amo | iql_amo_jax_adroit_beta1_rho | 3 | 0 | 0 | relocate-expert-v1 | beta=1 |
| ablation | iql_amo | iql_amo_jax_antmaze_beta5_rho | 30 | 24 | 24 | antmaze-large-diverse-v2;antmaze-large-play-v2;antmaze-medium-diverse-v2;antmaze-medium-play-v2;antmaze-umaze-diverse-v2;antmaze-umaze-v2 | beta=5 |
| ablation | iql_amo | iql_amo_jax_loco_beta1_rho | 2 | 0 | 0 | walker2d-medium-replay-v2;walker2d-medium-v2 | beta=1 |
| ablation | iql_amo | iql_amo_qweight_jax | 140 | 122 | 122 | antmaze-large-diverse-v2;antmaze-large-play-v2;antmaze-medium-diverse-v2;antmaze-medium-play-v2;antmaze-umaze-diverse-v2;antmaze-umaze-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;walker2d-medium-expert-v2;walker2d-medium-replay-v2;walker2d-medium-v2 | beta=1;beta=2;beta=5 |
| ablation | iql_amo | iql_ddpgbc_amo_fixed_alpha_E | 1 | 1 | 1 | hopper-medium-v2 |  |
| ablation | td3_amo | adaptive_multiscale | 165 | 162 | 156 | antmaze-large-diverse-v2;antmaze-large-play-v2;antmaze-medium-diverse-v2;antmaze-medium-play-v2;antmaze-umaze-diverse-v2;antmaze-umaze-v2;halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;walker2d-medium-expert-v2;walker2d-medium-replay-v2;walker2d-medium-v2 | alpha_E/B=10/10;alpha_E/B=10/2;alpha_E/B=2/0.5;alpha_E/B=2/2;alpha_E/B=20/20 |
| ablation | td3_amo | behavior_bc_l1_joint | 6 | 0 | 0 | antmaze-medium-play-v2;antmaze-umaze-diverse-v2;halfcheetah-medium-expert-v2;halfcheetah-medium-v2;hopper-medium-expert-v2;hopper-medium-v2 | alpha_E/B=8/8 |
| ablation | td3_amo | td3_amo_directq_l2rms | 12 | 12 | 12 | antmaze-medium-diverse-v2;hopper-medium-replay-v2;walker2d-medium-replay-v2 | alpha_E/B=5/5 |
| ablation | td3_amo | td3_amo_firstorder_l2rms | 12 | 12 | 12 | antmaze-medium-diverse-v2;hopper-medium-replay-v2;walker2d-medium-replay-v2 | alpha_E/B=5/5 |
| ablation | td3_amo | td3_amo_fixedB5_lrgrid | 12 | 2 | 2 | halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;hopper-medium-replay-v2;walker2d-medium-expert-v2 | alpha_E/B=5/5 |
| ablation | td3_amo | td3_amo_fixedB_lrmatch | 24 | 24 | 24 | antmaze-large-diverse-v2;antmaze-large-play-v2;antmaze-medium-play-v2;halfcheetah-medium-expert-v2;hopper-medium-expert-v2;walker2d-medium-v2 | alpha_E/B=5/5 |
| ablation | td3_amo | td3_amo_fixed_alpha_B | 4 | 3 | 3 | hopper-medium-replay-v2;walker2d-medium-replay-v2 | alpha_E/B=5/5 |
| ablation | td3_amo | td3_amo_jax | 243 | 243 | 243 | antmaze-large-diverse-v2;antmaze-large-play-v2;antmaze-medium-diverse-v2;antmaze-medium-play-v2;antmaze-umaze-diverse-v2;antmaze-umaze-v2;halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;walker2d-medium-expert-v2;walker2d-medium-replay-v2;walker2d-medium-v2 | alpha_E/B=1/1;alpha_E/B=2/2 |
| ablation | td3_amo | td3_amo_le_bel2_a5 | 114 | 97 | 97 | antmaze-large-diverse-v2;antmaze-large-play-v2;antmaze-medium-diverse-v2;antmaze-medium-play-v2;antmaze-umaze-diverse-v2;antmaze-umaze-v2;halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;walker2d-medium-expert-v2;walker2d-medium-replay-v2;walker2d-medium-v2 | alpha_E/B=5/5 |
| ablation | td3_amo | td3_amo_split_meta_lr | 16 | 16 | 16 | antmaze-umaze-diverse-v2;hopper-medium-expert-v2 | alpha_E/B=5/5 |
| main | fql_amo | fql_amo_jax | 12 | 12 | 12 | antmaze-large-diverse-v2;halfcheetah-medium-replay-v2;hopper-medium-expert-v2 | alpha_E/B=5/5 |

Machine-readable views: `runs_flat.csv`, `evaluations_flat.csv`, `families.csv`.
