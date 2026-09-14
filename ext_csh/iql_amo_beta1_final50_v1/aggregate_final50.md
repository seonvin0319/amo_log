# IQL+AMO β=1 final50_singlepass_v1 (ext_csh)

- Generated: 2026-09-14T10:11:24.556946+09:00
- Protocol: `final50_singlepass_v1`
- actor/critic/value LR: **3e-4** (not svcho 1e-3)
- Eval policy: π_E (`agent.act` → main actor)
- Seeds: 0–3 where JAX seed0 exists; else `missing_seed0`

## Dual-actor IQL semantics (reference)

Critic target is V-based IQL Bellman (`target_q` from target critic on dataset actions; value expectile). Bootstrap actor enters AMO meta/proxy terms only — **not** the critic Bellman target (unlike TD3 critic coupling).

## missing_seed0 (JAX β=1 compatible)

| rho_lr | env |
|--------|-----|
| 0.0003 | halfcheetah-medium-v2 |
| 0.0003 | halfcheetah-medium-replay-v2 |
| 0.0003 | halfcheetah-medium-expert-v2 |
| 0.0003 | hopper-medium-v2 |
| 0.0003 | hopper-medium-replay-v2 |
| 0.0003 | hopper-medium-expert-v2 |
| 0.0003 | walker2d-medium-v2 |
| 0.0003 | walker2d-medium-replay-v2 |
| 0.0003 | walker2d-medium-expert-v2 |
| 0.002 | halfcheetah-medium-v2 |
| 0.002 | halfcheetah-medium-replay-v2 |
| 0.002 | halfcheetah-medium-expert-v2 |
| 0.002 | hopper-medium-v2 |
| 0.002 | hopper-medium-replay-v2 |
| 0.002 | hopper-medium-expert-v2 |
| 0.002 | walker2d-medium-v2 |
| 0.002 | walker2d-medium-replay-v2 |
| 0.002 | walker2d-medium-expert-v2 |
| 0.002 | antmaze-umaze-v2 |
| 0.002 | antmaze-umaze-diverse-v2 |
| 0.002 | antmaze-medium-play-v2 |
| 0.002 | antmaze-medium-diverse-v2 |
| 0.002 | antmaze-large-play-v2 |
| 0.002 | antmaze-large-diverse-v2 |

## rho_lr = 0.0003

### Locomotion (mean ± sample std)

| env | n | missing | mean±std (ddof=1) |
|-----|---|---------|-------------------|
| halfcheetah-medium-v2 | 3 | 0 | -- |
| halfcheetah-medium-replay-v2 | 3 | 0 | -- |
| halfcheetah-medium-expert-v2 | 3 | 0 | -- |
| hopper-medium-v2 | 3 | 0 | -- |
| hopper-medium-replay-v2 | 3 | 0 | -- |
| hopper-medium-expert-v2 | 3 | 0 | -- |
| walker2d-medium-v2 | 3 | 0 | -- |
| walker2d-medium-replay-v2 | 3 | 0 | -- |
| walker2d-medium-expert-v2 | 3 | 0 | -- |

### AntMaze (mean±std and median; paper uses median)

| env | n | missing | mean±std | median (paper) |
|-----|---|---------|----------|----------------|
| antmaze-umaze-v2 | 4 | — | 69.50 ± 7.55 | 70.00 |
| antmaze-umaze-diverse-v2 | 4 | — | 55.00 ± 11.14 | 53.00 |
| antmaze-medium-play-v2 | 4 | — | 65.50 ± 4.12 | 65.00 |
| antmaze-medium-diverse-v2 | 4 | — | 63.50 ± 7.72 | 63.00 |
| antmaze-large-play-v2 | 4 | — | 30.00 ± 5.42 | 32.00 |
| antmaze-large-diverse-v2 | 4 | — | 19.50 ± 3.79 | 21.00 |

## rho_lr = 0.002

### Locomotion (mean ± sample std)

| env | n | missing | mean±std (ddof=1) |
|-----|---|---------|-------------------|
| halfcheetah-medium-v2 | 3 | 0 | -- |
| halfcheetah-medium-replay-v2 | 3 | 0 | -- |
| halfcheetah-medium-expert-v2 | 3 | 0 | -- |
| hopper-medium-v2 | 3 | 0 | -- |
| hopper-medium-replay-v2 | 3 | 0 | -- |
| hopper-medium-expert-v2 | 3 | 0 | -- |
| walker2d-medium-v2 | 3 | 0 | -- |
| walker2d-medium-replay-v2 | 3 | 0 | -- |
| walker2d-medium-expert-v2 | 3 | 0 | -- |

### AntMaze (mean±std and median; paper uses median)

| env | n | missing | mean±std | median (paper) |
|-----|---|---------|----------|----------------|
| antmaze-umaze-v2 | 3 | 0 | -- | -- |
| antmaze-umaze-diverse-v2 | 3 | 0 | -- | -- |
| antmaze-medium-play-v2 | 3 | 0 | -- | -- |
| antmaze-medium-diverse-v2 | 3 | 0 | -- | -- |
| antmaze-large-play-v2 | 3 | 0 | -- | -- |
| antmaze-large-diverse-v2 | 3 | 0 | -- | -- |

