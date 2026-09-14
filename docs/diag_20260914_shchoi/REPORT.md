# TD3+AMO AntMaze T_init=1 zero-score diagnosis (shchoi, 2026-09-14)

## Code / env
- Train checkout: `/home/shchoi/AMO-jax-upstream` @ `b9338d9815525482d2cf34d6fc6315ea4d2f93a6` (matches requested AMO main ref)
- Results live under `/home/shchoi/amo/results/...` (different repo); do not treat `amo` as the JAX train tree
- Eval worktree: `/home/shchoi/AMO-log-audit-20260914` branch `codex/log-audit-20260914-shchoi` commit `bc9a768`
- Conda: `/home/shchoi/miniconda3/envs/offrl`; `D4RL_DATASET_DIR=/home/shchoi/.d4rl/datasets`; MuJoCo `~/.mujoco/mujoco210`

## Inventory (T_E=T_B=1, T_lr∈{3e-4,1e-3,2e-3}, seed 0–3)
- Grid size 72; see `inventory_t1.json`
- `eval_done` (legacy posthoc): 30 — all medium/large were 0 under 5×10
- `missing` / `empty_dir`: 42 (FAILED jobs / never trained); **no new 1M sweep started**
- Torch / T∈{5,10} kept in separate groups (`inventory_other_tinit.json`, `inventory_torch.json`)

## Protocol
`final50_singlepass_v1`: 50 episodes, repeats=1, seed once; writes `eval_final50_v1.jsonl` + `posthoc_final50_v1/` (does not delete checkpoints; does not use stock 5×10 reaper as bulk).

## Priority re-eval + positive controls
| run | final50 | note |
|-----|---------|------|
| amd / T_lr=2e-3 / seed2 / T=1 | **0.0** | 50/50 timeout |
| ald / T_lr=3e-4 / seed1 / T=1 | **0.0** | 50/50 timeout |
| ald / T_lr=3e-4 / seed1 / **T=5** | **42.0** | control only |
| ald / T_lr=3e-4 / seed1 / **T=10** | **34.0** | control only |

## Cause classification
**`train_update_policy_quality`** (not evaluator / metadata / norm / π_E load).

Evidence:
1. Same evaluator + loader: T=5→42, T=10→34, T=1→0
2. ckpt env/seed/step + normalization match; reload fixed-obs Δaction=0; actions finite, near-bound rare
3. Umaze T=1 final50 non-zero (e.g. 78/82) → stack not globally broken
4. amd T=1 metrics: T_E→~14, T_B→~0.002 (scale collapse) vs healthy T=5 scales
5. `reward_transform=none` matches launcher; left unchanged

No train-code patch: no confirmed JAX≠reference bug; zeros reproduce under new protocol.

## AntMaze 4-seed table (final50)
All cells are `--` (n<4). Per-seed scores in `antmaze_t1_final50_table.json`. Missing seeds listed there (do not retrain in this task).

## Adroit
48 runs present under `amo_jax_td3amo_adroit_t1_tlr2e3`; final50 bulk in progress / see report update.

## Adroit (T_lr=2e-3, T_init=1) — final50_singlepass_v1
- 48/48 runs trained; no duplicate (env,seed) cells
- All 48 evaluated under final50; table in `adroit_final50_table.json`

| env | 4-seed mean±std | s0 | s1 | s2 | s3 |
|-----|-----------------|----|----|----|----|
| door-cloned-v1 | -0.3±0.1 | -0.34 | -0.12 | -0.38 | -0.31 |
| door-expert-v1 | 0.8±0.4 | 0.21 | 0.99 | 1.22 | 0.76 |
| door-human-v1 | -0.3±0.0 | -0.34 | -0.37 | -0.34 | -0.29 |
| hammer-cloned-v1 | -0.4±0.6 | -0.10 | 0.12 | -1.18 | -0.25 |
| hammer-expert-v1 | 22.1±24.7 | 10.07 | 0.00 | 21.66 | 56.72 |
| hammer-human-v1 | -0.2±0.2 | -0.52 | 0.04 | -0.20 | -0.26 |
| pen-cloned-v1 | 1.5±3.9 | 3.02 | 6.11 | -0.25 | -2.92 |
| pen-expert-v1 | 7.6±7.8 | 14.30 | 14.27 | -0.93 | 2.88 |
| pen-human-v1 | -1.9±1.4 | 0.13 | -2.30 | -2.40 | -2.89 |
| relocate-cloned-v1 | -0.2±0.0 | -0.22 | -0.24 | -0.21 | -0.24 |
| relocate-expert-v1 | -0.1±0.1 | -0.10 | -0.11 | -0.13 | 0.07 |
| relocate-human-v1 | -0.2±0.1 | -0.16 | -0.24 | -0.22 | -0.08 |
