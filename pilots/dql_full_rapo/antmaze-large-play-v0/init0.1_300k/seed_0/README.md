# DQL full RAPO pilot — antmaze-large-play-v0, T_E_init = T_B_init = 0.1, 1M updates (ext_csv; started as 300k)

New-environment pilot. Not part of `main/` or `ablation/`; `catalog/` and the
paper tables are untouched. Single run, seed 0, started from scratch.

## Code

- Repo: `https://github.com/seonvin0319/AMO.git`, branch `dql-full-rapo-share-20261110`
- Pinned commit: `e3c042423bc0eaeecb072516db2175af12e9437f`
  ("Add official kitchen-mixed and antmaze-large settings for a 300k init-0.1 run."),
  the ext_csh update that adds `antmaze-large-play-v0` / `kitchen-mixed-v0` to
  `scripts/train_dql_pilot.py` `OFFICIAL`. Working tree clean (`code_dirty: false`).
- Checkout on host: `/home/ext_csv/dql-full-rapo-largeplay` (detached at the pinned SHA).
- Trainer: `scripts/train_dql_pilot.py --algo full_rapo`; agent `agents/ql_diffusion_full_rapo.py`.
  Algorithm, losses, optimizers and normalization unchanged.

## Effective config (`config.yaml`, written by the trainer)

| item | value |
|---|---|
| env / seed | antmaze-large-play-v0 / 0 |
| actor lr / critic lr | 3e-4 / 3e-4 |
| official eta | 4.5 (not multiplied into T_E / T_B) |
| T_E_init / T_B_init | 0.1 / 0.1 → rho = softplus⁻¹(0.1) = −2.2522 (observed in metrics step 1) |
| max_q_backup / reward_tune / grad norm | true / cql_antmaze / 10.0 |
| T (diffusion steps) / beta_schedule | 5 / vp |
| batch / discount / tau | 256 / 0.99 / 0.005 |
| lr_decay / lr_maxt / scheduler_every | true / 1000 / 50000 updates (1000-epoch cosine kept, not compressed) |
| meta_interval / meta_warmup | 20 / 0 |
| meta lr initial / final ratio / horizon | 3e-4 / 0.01 / 1,000,000 (kept) |
| max_steps / eval_every / eval_episodes | 1,000,000 (extended from 300,000 at step 170,092, see below) / 100,000 / 50 (E = reported, B = diagnostic) |
| synthetic / stub_eval | false / false |

## Dataset

- D4RL 1.1 `antmaze-large-play-v0` → `Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5`
  (230,880,846 bytes, sha256 in `dataset_file_sha256.txt`).
- `d4rl.qlearning_dataset`: 999,059 transitions, obs_dim 29, act_dim 8, 12,517 terminals
  (`dataset.json` has the trainer's own record and obs-head sha256).

## Environment

Separate conda env `/raid/ext_csv/miniconda3/envs/dql` (see `env_packages.txt`):
python 3.10.22, torch 2.6.0+cu124 (CUDA 12.4, H200), numpy 1.26.4, gym 0.23.0,
D4RL 1.1, mujoco-py 2.1.2.14 on mujoco210. `LD_LIBRARY_PATH` holds
`~/.mujoco/mujoco210/bin`, `/usr/lib/x86_64-linux-gnu`, and the env `lib`.
Evaluation is the real `gym.make('antmaze-large-play-v0')` (EGL-built cymj).

## Launch record

See `launch.json`. Output / checkpoints / logs:
`/home/ext_csv/checkpoints/dql_largeplay_init01_300k/antmaze-large-play-v0_s0/`
(`train.log`, `metrics.jsonl`, `eval.jsonl`, `checkpoint.pt`, `checkpoints/step_*.pt`).
Watcher queue pair registered in `server_management/config/ext_csv.json`
(`/raid/ext_csv/dql_env_setup/run_dql_largeplay_init01_300k_{gpu,cpu}.sh`).

Pre-launch validation: 40-step real-data smoke (2 meta steps, 2-episode real eval
for E and B, checkpoint) in `/raid/ext_csv/dql_smoke/largeplay_s0` — not reused.
A first launch at 05:22Z was stopped at step 4193 (SIGTERM, emergency checkpoint)
because the trainer was invoked with a relative script path that the watcher's
trainer pattern did not match; its directory was archived to
`/raid/ext_csv/dql_smoke/aborted_launch1_largeplay_s0_relpath` and the run was
restarted from scratch at 05:26Z. Step-1 metrics of both launches and the smoke
are identical (deterministic seeding).

## Budget extension to 1M (2026-11-10 06:41Z)

User request: run to 1M instead of stopping at 300k. The wrapper was changed to
`--max-steps 1000000` (nothing else), the trainer was stopped with SIGTERM at
step 170,092 (checkpoint written), and relaunched in the same run dir; it resumed
from `checkpoint.pt` (`[dql] resume step=170092`). The checkpoint restores all
networks, optimizers, rho_E/rho_B, meta optimizers, LR schedulers and every RNG,
so the trajectory is continuous (T_E 0.11979 → 0.11980, meta lr 1.4e-4 unchanged
across the boundary). LR schedule (lr_maxt 1000 / 1M) and meta horizon (1M) were
already on the 1M grid, so nothing changes in the schedule. The off-grid
`checkpoints/step_170092.pt` was removed after the resume was verified; the
directory name `init0.1_300k` / `dql_largeplay_init01_300k` is kept for continuity.

## Evaluations

At 100k (06:13Z): T_E 0.1068, T_B 0.0998; critic_loss median (80k–100k) 7.1, critic_grad_norm median 546 (clip 10 active).

Filled in at 100k / 200k / 300k from `eval.jsonl` (E = `evaluation`, B = `evaluation_B`).

| step | E normalized score | B normalized score | episodes |
|---|---|---|---|
| 100000 | 0.10 (return_mean 0.10, std 0.30) | 0.04 (std 0.20) | 50 |
| 200000 | pending | pending | 50 |
| 300000 | pending | pending | 50 |
| 400000 | pending | pending | 50 |
| 500000 | pending | pending | 50 |
| 600000 | pending | pending | 50 |
| 700000 | pending | pending | 50 |
| 800000 | pending | pending | 50 |
| 900000 | pending | pending | 50 |
| 1000000 | pending | pending | 50 |
