# Collected sources

Checkpoint weights were **not** copied. Logs only: `config.yaml`, `metrics.jsonl`, `eval.jsonl`, `run_meta.json`.

## Host `choi` (initial snapshot)

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/choi/APART/results_apart` | apart | `dual_proximal` / `chain` | ~53 |
| `/home/choi/APART/results_pi_only_xfit_target` | apart | `pi_only_xfit_target` | 9 |
| `/home/choi/APART/results_pi_only_xfit_target_mpi_nstep` | apart | `pi_only_xfit_mpi_nstep` | 8 |
| `/home/choi/amo/results/segment_interval` | amo | `segment_interval` | 1 (smoke) |

Not collected on choi (empty or logs-only):

- `/home/choi/APART/results_pi_only_xfit_target_mpi_nstep_antmaze` (no completed run dirs)

## Host `ext_csh`

| Source path | algo | family | notes |
|-------------|------|--------|-------|
| `/home/ext_csh/AMO/results/amo_loco9_s0_*` | amo | `adaptive_multiscale` | B_PI / L1_E / td3bc / qraw T_lr×T_init sweeps |
| `/home/ext_csh/AMO/results/amo_adaptive_multiscale_locomotion9_seed0` | amo | `adaptive_multiscale` | early loco-9 |
| `/home/ext_csh/AMO-antmaze-bpi-tlr-init-sweep/results/amo_antmaze6_s0_bpi_tlr_te_tb_init_sweep/cells/*` | amo | `adaptive_multiscale` | antmaze6 B_PI T_lr×(T_E=T_B) init sweep |
| `/home/ext_csh/AMO-a8c1e48-te1tb1/results/amo_antmaze6_s0_tlr1e-3_te1_tb_div2` | amo | `adaptive_multiscale` | antmaze6 parent T_B=T_E/2 |
| `/home/ext_csh/APART/results/adaptive_multiscale_*` | amo | `adaptive_multiscale` | code lived in APART tree; archived as amo |
| `/home/ext_csh/APART/results/adaptive_bootstrap_*` | amo | `adaptive_multiscale` | bootstrap variant tag `boot` |
| `/home/ext_csh/APART/results/dual_n24_tlr` | apart | `dual_proximal` | Adroit dual N=2/4 T_lr sweep |
| `/home/ext_csh/APART/results/dual_n24_adroit` | apart | `dual_proximal` | Adroit dual |
| `/home/ext_csh/APART/results/n1_pen_cloned` | apart | `chain` | pen-cloned N=1 |
| `/home/ext_csh/CORL-iql-adaptive-beta-v1/results/iql_beta_locomotion4_s0` | iql | `adaptive_beta` | CORL IQL adaptive-β loco4 seed0 |
| `/home/ext_csh/CORL-iql-adaptive-beta-v1/results/iql_beta_remaining7_jp2_s0` | iql | `adaptive_beta` | remaining (walker/antmaze/…) seed0 |

To refresh on ext_csh:

```bash
cd /home/ext_csh/amo_log
python scripts/ingest_runs.py --host=ext_csh
git add -A && git commit -m "collect(ext_csh): refresh AMO/APART logs" && git push
```

## Auto update (ext_csh)

Cron every 5 minutes (`INTERVAL_SEC=300`) runs `scripts/run_amo_log_auto_push_periodic.sh`
→ `scripts/auto_push_ext_csh.sh`:

- `python scripts/ingest_runs.py --host=ext_csh`
- rebuild catalog
- commit + push to `origin/ext_csh` only (not `main`)

Log: `/home/ext_csh/logs/amo_log_auto_push_ext_csh.log`
Periodic: `/home/ext_csh/logs/amo_log_periodic/periodic.log`
