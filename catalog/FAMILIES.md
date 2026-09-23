# Parsed experiment families

Derived index only. Canonical data remain `run_meta.json`, `config.yaml`, and raw evaluation files.

| section | method | family | runs | scored | complete@1M | environments | initializations |
|---|---|---|---:|---:|---:|---|---|
| ablation | iql_amo | adaptive_beta | 8 | 8 | 8 | halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;walker2d-medium-expert-v2;walker2d-medium-v2 | beta=3;beta=6 |
| ablation | iql_amo | amo_qweight | 283 | 18 | 18 | halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;walker2d-medium-expert-v2;walker2d-medium-replay-v2;walker2d-medium-v2 | beta=1;beta=2;beta=5 |
| ablation | iql_amo | iql_ddpgbc | 360 | 360 | 360 | antmaze-large-diverse-v2;antmaze-large-play-v2;antmaze-medium-diverse-v2;antmaze-medium-play-v2;antmaze-umaze-diverse-v2;antmaze-umaze-v2;halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;walker2d-medium-expert-v2;walker2d-medium-replay-v2;walker2d-medium-v2 |  |
| ablation | td3_amo | adaptive_multiscale | 171 | 164 | 75 | antmaze-large-diverse-v2;antmaze-large-play-v2;antmaze-medium-diverse-v2;antmaze-medium-play-v2;antmaze-umaze-diverse-v2;antmaze-umaze-v2;halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;halfcheetah-random-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;hopper-random-v2;walker2d-medium-expert-v2;walker2d-medium-replay-v2;walker2d-medium-v2 | alpha_E/B=0.01/0.01;alpha_E/B=0.2/0.2;alpha_E/B=10/10;alpha_E/B=2/2;alpha_E/B=20/20 |
| ablation | td3_amo | segment_interval | 1 | 0 | 0 | hopper-medium-expert-v2 | alpha_E/B=2.5/2.5 |
| ablation | td3_amo | td3_amo_execonly_main | 96 | 0 | 0 | antmaze-large-diverse-v2;antmaze-large-play-v2;antmaze-medium-diverse-v2;antmaze-medium-play-v2;antmaze-umaze-diverse-v2;antmaze-umaze-v2;halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;walker2d-medium-expert-v2;walker2d-medium-replay-v2;walker2d-medium-v2 | alpha_E/B=5/5 |
| ablation | wpc | paper_benchmark | 48 | 48 | 32 | halfcheetah-expert-v2;halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;hopper-expert-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;walker2d-medium-expert-v2;walker2d-medium-v2 |  |
| main | iql_amo | amo_bpi | 33 | 33 | 27 | antmaze-large-diverse-v2;antmaze-large-play-v2;antmaze-medium-diverse-v2;antmaze-medium-play-v2;antmaze-umaze-diverse-v2;antmaze-umaze-v2;halfcheetah-medium-expert-v2;halfcheetah-medium-replay-v2;halfcheetah-medium-v2;hopper-medium-expert-v2;hopper-medium-replay-v2;hopper-medium-v2;walker2d-medium-expert-v2;walker2d-medium-replay-v2;walker2d-medium-v2 | beta=1;beta=5 |

Machine-readable views: `runs_flat.csv`, `evaluations_flat.csv`, `families.csv`.
