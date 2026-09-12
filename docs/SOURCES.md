# Collected sources (initial snapshot)

Collected on host `choi` at ingest time. Checkpoint weights were **not** copied.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/choi/APART/results_apart` | apart | `dual_proximal` / `chain` | ~53 |
| `/home/choi/APART/results_pi_only_xfit_target` | apart | `pi_only_xfit_target` | 9 |
| `/home/choi/APART/results_pi_only_xfit_target_mpi_nstep` | apart | `pi_only_xfit_mpi_nstep` | 8 |
| `/home/choi/amo/results/segment_interval` | amo | `segment_interval` | 1 (smoke) |

Not collected (empty or logs-only):

- `/home/choi/APART/results_pi_only_xfit_target_mpi_nstep_antmaze` (no completed run dirs)

## Host `shchoi` (`iisl-server04`)

Collected 2026-09-07 (JAX) and 2026-09-10 (PyTorch). **No APART tree on this host.** Checkpoints (`*.pkl`, `*.pt`) and per-update `amo_*_metrics.jsonl` (tens–hundreds of MB) were **not** copied. JAX: `config.json` → `config.yaml`, eval rows extracted into `eval.jsonl`.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/shchoi/AMO/results_amo_pilot` | amo | `jax_v1` | 3 (1 incomplete) |
| `/home/shchoi/AMO/results_amo_v2_pilot` | amo | `jax_v2` / `jax_td3bc` | 17 (2 incomplete; skip s2 T025 no metrics) |
| `/home/shchoi/AMO/results_amo_v3_pilot` | amo | `jax_v3a` / `jax_v3b` | 6 (v3b 3 incomplete) |
| `/home/shchoi/AMO/results_amo_v2_1m` | amo | `jax_v2` | 1 incomplete (hopper-medium) |
| `/home/shchoi/amo_td3bc/results/amo_td3bc_locomotion9_seed0/runs` | amo | `amo_td3bc` | locomotion-9 seed 0 (live) |
| `/home/shchoi/amo_lambda0_fork_diag/results/rebrac_amo_locomotion9_seed0/runs` | amo | `rebrac_amo` | locomotion-9 seed 0 |
| `/home/shchoi/amo/results/amo_adaptive_multiscale_antmaze6_sweep_seed0/runs` | amo | `adaptive_multiscale` | AntMaze-6 B_PI sweep: T_lr×T_E=T_B init (live) |
| `/home/shchoi/ASPC/results_aspc` | aspc | `benchmark` | ASPC D4RL full sweep (`_aspc-*`, 1M) |
| `/home/shchoi/ASPC/results_wpc` | wpc | `benchmark` | WPC baseline; eval from `_logs/*.log` |
| `/home/shchoi/ASPC/results_pi_l3` | aspc | `benchmark` | ASPC-repo L3 modes (`aspc`/`pi_*`; incomplete tagged) |
| `/home/shchoi/ASPC/results_multiscale` | aspc | `benchmark` | `l3_mode=multiscale` seed0 locomotion |

Not collected:

- `/home/shchoi/AMO/results_wmr_geometry`, `results_wmr_mild_robustness`, `results_wmr_trajectory_diagnosis` (진단 런, TrainConfig 런 디렉터리 아님)
- APART: 이 호스트에 `/home/shchoi/APART` 없음

To refresh after new experiments:

```bash
cd /home/shchoi/amo_log   # or /home/choi/amo_log
python scripts/ingest_runs.py
git add -A && git commit -m "collect: refresh from local sources" && git push
```
