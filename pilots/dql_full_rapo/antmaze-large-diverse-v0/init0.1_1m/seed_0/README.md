# DQL full RAPO pilot — antmaze-large-diverse-v0, T_E_init = T_B_init = 0.1, 1M updates (ext_csv)

Companion to the antmaze-large-play-v0 pilot (`../../antmaze-large-play-v0/init0.1_300k/seed_0/`),
launched in parallel on the same GPU. Not part of `main/` or `ablation/`; `catalog/` and the
paper tables are untouched. Single run, seed 0, from scratch.

## Code

- Base: ext_csh share commit `e3c042423bc0eaeecb072516db2175af12e9437f`
  (branch `dql-full-rapo-share-20261110`).
- Run commit: `196e9e0eca4d9a6fd3d3c625aded62afb66d60a9` = base + one 5-line addition to
  `scripts/train_dql_pilot.py` `OFFICIAL` for `antmaze-large-diverse-v0`, **copied verbatim
  from the official hyperparameter table in `main.py`** of the same repo
  (lr 3e-4, eta 3.5, max_q_backup True, reward_tune cql_antmaze, gn 7.0, 1000 epochs).
  No algorithm/loss/optimizer change. Patch: `code_delta_vs_e3c04242.patch`.
  Local-only commit (not pushed), worktree `/home/ext_csv/dql-full-rapo-largediverse`.
  The share branch did not contain this env; ext_csh can replace this with its own
  entry if it differs.

## Effective config (`config.yaml`)

Identical to the large-play pilot except the env-specific official values:
eta **3.5**, grad norm **7.0** (large-play: 4.5 / 10.0). actor/critic lr 3e-4,
T_E_init = T_B_init = 0.1 (rho = −2.2522, observed at step 1), max_q_backup true,
cql_antmaze, T 5, batch 256, lr_maxt 1000 (scheduler_every 50,000), meta_interval 20,
meta_warmup 0, meta lr 3e-4 / 0.01 / 1M, max_steps 1,000,000, eval_every 100,000,
eval_episodes 50 (E reported, B diagnostic), synthetic/stub_eval false.

## Dataset

- D4RL 1.1 `antmaze-large-diverse-v0` → `Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse.hdf5`
  (downloaded 2026-11-10 from rail.eecs.berkeley.edu; sha256 in `dataset_file_sha256.txt`).
- `qlearning_dataset`: 999,048 transitions, obs 29 / act 8, 6,189 terminals.

## Environment

Same conda env as the large-play pilot (`env_packages.txt`).

## Launch record

- 40-step real-data smoke in `/raid/ext_csv/dql_smoke/largediverse_s0` (2 meta steps,
  2-episode real E/B eval, checkpoint): passed; not reused.
- Launched 2026-11-10 06:49:14Z via `/raid/ext_csv/dql_env_setup/run_dql_env_init01_1m_gpu.sh`
  (wrapper PID 60401, trainer PID 60408, setsid nohup). Watcher GPU queue
  `run_dql_largeplay_init01_300k_gpu.sh` now launches both runs (flock-guarded, resume).
- Output: `/home/ext_csv/checkpoints/dql_largediverse_init01_1m/antmaze-large-diverse-v0_s0/`
  (`train.log`, `metrics.jsonl`, `eval.jsonl`, `checkpoint.pt`, `checkpoints/step_*.pt`).
- Step 1: critic_loss 9.047, critic_grad_norm 43.4, L_E −4.2e-4, T_E 0.1000, T_B 0.09997.
  ~31.7 updates/s while sharing the GPU with the large-play run.

## Evaluations

| step | E normalized score | B normalized score | episodes |
|---|---|---|---|
| 100000 | 0.18 | 0.16 | 50 |
| 200000 | 0.04 | 0.18 | 50 |
| 300000 | 0.20 | 0.30 | 50 |
| 400000 | 0.08 | 0.12 | 50 |
| 500000 | 0.00 | 0.00 | 50 |
| 600000 | 0.00 | 0.00 | 50 |
| 700000 | 0.00 | 0.00 | 50 |
| 800000 | 0.00 | 0.00 | 50 |
| 900000 | 0.00 | 0.00 | 50 |
| 1000000 | 0.00 | 0.00 | 50 |

## Final state

Finished 1M at 2026-11-10 18:04Z (DONE marker; reload resumes at 300k, 600k, 900k). E 0.18 / 0.04 / 0.20 / 0.08 at 100k–400k, then 0.00 from 500k to 1M (B likewise 0.00). **Critic diverged**: critic_loss median 6.5–9.3 up to 400k, then 1.4e3 (400k–500k), 2.2e4, …, 9.5e6 (900k–1M); critic grad norm up to 2.7e8 (first critic_loss > 1e3 at step 367,941; > 1e5 at 414,581). T_E 0.100 → 0.393, T_B ≈0.10. Results after 400k are not meaningful.
