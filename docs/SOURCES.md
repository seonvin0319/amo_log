# Collected sources (initial snapshot)

Collected on host `choi` at ingest time. Checkpoint weights were **not** copied.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/choi/APART/results_apart` | apart | `dual_proximal` / `chain` | ~53 |
| `/home/choi/APART/results_pi_only_xfit_target` | apart | `pi_only_xfit_target` | 9 |
| `/home/choi/APART/results_pi_only_xfit_target_mpi_nstep` | apart | `pi_only_xfit_mpi_nstep` | 8 |
| `/home/choi/amo/results/segment_interval` | amo | `segment_interval` | 1 (smoke) |
| `/home/choi/amo/results/amo_antmaze_t_init_tune_seed0` | amo | `antmaze_t_init_tune` | live (5m cron) |

Not collected (empty or logs-only):

- `/home/choi/APART/results_pi_only_xfit_target_mpi_nstep_antmaze` (no completed run dirs)

To refresh after new experiments:

```bash
cd /home/choi/amo_log
python scripts/ingest_runs.py
git add -A && git commit -m "collect: refresh from local sources" && git push
```
