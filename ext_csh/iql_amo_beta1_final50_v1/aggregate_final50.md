# IQL+AMO β=1 · seeds 0–3 (ext_csh)

- Generated: 2026-09-14T10:23:51.391408+09:00
- Seeds 1–3: `final50_singlepass_v1` (JAX)
- Seed0: JAX `final50` when present; else **CORL Torch** `adaptive_beta` / `mean_normalized` @ 1M
- actor/critic/value LR: **3e-4** (not svcho 1e-3)
- Eval policy: π_E (JAX `agent.act` / CORL `adaptive_beta`)

## Dual-actor IQL semantics (reference)

Critic target is V-based IQL Bellman (`target_q` from target critic on dataset actions; value expectile). Bootstrap actor enters AMO meta/proxy terms only — **not** the critic Bellman target (unlike TD3 critic coupling).

## CORL seed0 filled (24)

| rho_lr | env | score | protocol |
|--------|-----|-------|----------|
| 0.0003 | halfcheetah-medium-v2 | 48.61 | final_10x5_r4 |
| 0.0003 | halfcheetah-medium-replay-v2 | 43.88 | final_10x5_r4 |
| 0.0003 | halfcheetah-medium-expert-v2 | 86.67 | final_10x5_r4 |
| 0.0003 | hopper-medium-v2 | 54.92 | final_10x5_r4 |
| 0.0003 | hopper-medium-replay-v2 | 97.49 | final_10x5_r4 |
| 0.0003 | hopper-medium-expert-v2 | 57.90 | final_10x5_r4 |
| 0.0003 | walker2d-medium-v2 | 87.18 | final_10x5_r4 |
| 0.0003 | walker2d-medium-replay-v2 | 79.97 | final_10x5_r4 |
| 0.0003 | walker2d-medium-expert-v2 | 110.54 | final_10x5_r4 |
| 0.002 | halfcheetah-medium-v2 | 48.43 | locomotion_10 |
| 0.002 | halfcheetah-medium-replay-v2 | 40.90 | locomotion_10 |
| 0.002 | halfcheetah-medium-expert-v2 | 92.57 | locomotion_10 |
| 0.002 | hopper-medium-v2 | 55.68 | locomotion_10 |
| 0.002 | hopper-medium-replay-v2 | 101.81 | locomotion_10 |
| 0.002 | hopper-medium-expert-v2 | 111.65 | locomotion_10 |
| 0.002 | walker2d-medium-v2 | 81.32 | locomotion_10 |
| 0.002 | walker2d-medium-replay-v2 | 89.37 | locomotion_10 |
| 0.002 | walker2d-medium-expert-v2 | 111.28 | locomotion_10 |
| 0.002 | antmaze-umaze-v2 | 38.00 | antmaze_100 |
| 0.002 | antmaze-umaze-diverse-v2 | 61.00 | antmaze_100 |
| 0.002 | antmaze-medium-play-v2 | 77.00 | antmaze_100 |
| 0.002 | antmaze-medium-diverse-v2 | 68.00 | antmaze_100 |
| 0.002 | antmaze-large-play-v2 | 31.00 | antmaze_100 |
| 0.002 | antmaze-large-diverse-v2 | 17.00 | antmaze_100 |

## Still missing seed0

_none_

## rho_lr = 0.0003

### Locomotion (mean ± sample std)

| env | n | missing | mean±std (ddof=1) |
|-----|---|---------|-------------------|
| halfcheetah-medium-v2 | 4 | — | 48.54 ± 0.11 |
| halfcheetah-medium-replay-v2 | 4 | — | 44.07 ± 0.59 |
| halfcheetah-medium-expert-v2 | 4 | — | 89.49 ± 1.98 |
| hopper-medium-v2 | 4 | — | 57.83 ± 1.96 |
| hopper-medium-replay-v2 | 4 | — | 97.52 ± 2.84 |
| hopper-medium-expert-v2 | 4 | — | 83.20 ± 31.37 |
| walker2d-medium-v2 | 4 | — | 82.65 ± 4.51 |
| walker2d-medium-replay-v2 | 4 | — | 76.84 ± 4.64 |
| walker2d-medium-expert-v2 | 4 | — | 110.98 ± 0.50 |

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
| halfcheetah-medium-v2 | 4 | — | 48.44 ± 0.08 |
| halfcheetah-medium-replay-v2 | 4 | — | 43.50 ± 1.77 |
| halfcheetah-medium-expert-v2 | 4 | — | 91.32 ± 1.29 |
| hopper-medium-v2 | 4 | — | 57.24 ± 1.14 |
| hopper-medium-replay-v2 | 4 | — | 100.86 ± 0.98 |
| hopper-medium-expert-v2 | 4 | — | 98.74 ± 21.73 |
| walker2d-medium-v2 | 4 | — | 79.85 ± 3.87 |
| walker2d-medium-replay-v2 | 4 | — | 80.23 ± 6.34 |
| walker2d-medium-expert-v2 | 4 | — | 111.32 ± 1.39 |

### AntMaze (mean±std and median; paper uses median)

| env | n | missing | mean±std | median (paper) |
|-----|---|---------|----------|----------------|
| antmaze-umaze-v2 | 4 | — | 70.50 ± 22.05 | 79.00 |
| antmaze-umaze-diverse-v2 | 4 | — | 62.25 ± 6.65 | 62.50 |
| antmaze-medium-play-v2 | 4 | — | 74.25 ± 2.63 | 74.00 |
| antmaze-medium-diverse-v2 | 4 | — | 64.50 ± 3.00 | 64.00 |
| antmaze-large-play-v2 | 4 | — | 33.75 ± 6.45 | 34.50 |
| antmaze-large-diverse-v2 | 4 | — | 13.75 ± 2.87 | 14.00 |

