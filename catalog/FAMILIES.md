# Parsed experiment families

Derived index only. Canonical data remain `run_meta.json`, `config.yaml`, and raw evaluation files.

| section | method | family | runs | scored | complete@1M | environments | initializations |
|---|---|---|---:|---:|---:|---|---|
| ablation | aspc | benchmark | 46 | 46 | 34 | halfcheetah-expert-v2;halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;hopper-expert-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;walker2d-expert-v2;walker2d-medium-expert-v2;walker2d-medium-replay-v2;walker2d-medium-v2 |  |
| ablation | iql_amo | iql_adaptive_beta | 8 | 6 | 2 | halfcheetah-medium-replay-v2;halfcheetah-medium-v2 | beta=3 |
| ablation | iql_amo | qweight | 110 | 108 | 108 | antmaze-large-diverse-v2;antmaze-large-play-v2;antmaze-medium-diverse-v2;antmaze-medium-play-v2;antmaze-umaze-diverse-v2;antmaze-umaze-v2;halfcheetah-medium-v2 | beta=1;beta=2;beta=5 |
| ablation | td3_amo | adaptive_multiscale | 105 | 105 | 99 | antmaze-large-diverse-v2;antmaze-large-play-v2;antmaze-medium-diverse-v2;antmaze-medium-play-v2;antmaze-umaze-diverse-v2;antmaze-umaze-v2 | alpha_E/B=10/10;alpha_E/B=2/2;alpha_E/B=20/20 |
| ablation | td3_amo | amo_td3bc | 9 | 9 | 9 | halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;walker2d-medium-expert-v2;walker2d-medium-replay-v2;walker2d-medium-v2 | alpha_E/B=2.5/2.5 |
| ablation | td3_amo | rebrac_amo | 9 | 9 | 9 | halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;walker2d-medium-expert-v2;walker2d-medium-replay-v2;walker2d-medium-v2 | alpha_E/B=2/2 |
| ablation | td3_amo | segment_interval | 1 | 0 | 0 | hopper-medium-expert-v2 | alpha_E/B=2.5/2.5 |
| ablation | td3_amo | td3_amo_fixed1 | 42 | 39 | 39 | antmaze-large-diverse-v2;antmaze-large-play-v2;antmaze-medium-diverse-v2;antmaze-medium-play-v2;antmaze-umaze-diverse-v2;antmaze-umaze-v2 | alpha_E/B=5/1 |
| ablation | td3_amo | td3_amo_jax | 120 | 118 | 118 | antmaze-large-diverse-v2;antmaze-large-play-v2;antmaze-medium-diverse-v2;antmaze-medium-play-v2;antmaze-umaze-diverse-v2;antmaze-umaze-v2;halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;walker2d-medium-expert-v2;walker2d-medium-replay-v2;walker2d-medium-v2 | alpha_E/B=1/1 |
| main | aspc | benchmark | 76 | 76 | 72 | antmaze-large-diverse-v2;antmaze-large-play-v2;antmaze-medium-diverse-v2;antmaze-medium-play-v2;antmaze-umaze-diverse-v2;antmaze-umaze-v2;halfcheetah-expert-v2;halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;hopper-expert-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;walker2d-expert-v2;walker2d-medium-expert-v2;walker2d-medium-replay-v2;walker2d-medium-v2 |  |
| main | iql_amo | amo_bpi | 390 | 388 | 388 | antmaze-large-diverse-v2;antmaze-large-play-v2;antmaze-medium-diverse-v2;antmaze-medium-play-v2;antmaze-umaze-diverse-v2;antmaze-umaze-v2;halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;walker2d-medium-expert-v2;walker2d-medium-replay-v2;walker2d-medium-v2 | beta=1;beta=2;beta=5 |
| main | wpc | benchmark | 48 | 48 | 48 | antmaze-large-diverse-v2;antmaze-large-play-v2;antmaze-medium-diverse-v2;antmaze-medium-play-v2;antmaze-umaze-diverse-v2;antmaze-umaze-v2;halfcheetah-expert-v2;halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;hopper-expert-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;walker2d-expert-v2;walker2d-medium-expert-v2;walker2d-medium-replay-v2;walker2d-medium-v2 |  |

Machine-readable views: `runs_flat.csv`, `evaluations_flat.csv`, `families.csv`.
