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

Collected 2026-09-07. JAX AMO logs only. **No APART tree on this host.** Checkpoints (`*.pkl`) and per-update `amo_*_metrics.jsonl` (tens–hundreds of MB) were **not** copied. `config.json` → `config.yaml`, eval rows extracted into `eval.jsonl`.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/shchoi/AMO/results_amo_pilot` | amo | `jax_v1` | 3 (1 incomplete) |
| `/home/shchoi/AMO/results_amo_v2_pilot` | amo | `jax_v2` / `jax_td3bc` | 17 (2 incomplete; skip s2 T025 no metrics) |
| `/home/shchoi/AMO/results_amo_v3_pilot` | amo | `jax_v3a` / `jax_v3b` | 6 (v3b 3 incomplete) |
| `/home/shchoi/AMO/results_amo_v2_1m` | amo | `jax_v2` | 1 incomplete (hopper-medium) |

Not collected:

- `/home/shchoi/AMO/results_wmr_geometry`, `results_wmr_mild_robustness`, `results_wmr_trajectory_diagnosis` (진단 런, TrainConfig 런 디렉터리 아님)
- APART: 이 호스트에 `/home/shchoi/APART` 없음

To refresh after new experiments:

```bash
cd /home/shchoi/amo_log   # or /home/choi/amo_log
python scripts/ingest_runs.py
git add -A && git commit -m "collect: refresh from local sources" && git push
```
