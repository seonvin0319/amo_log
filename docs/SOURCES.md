# Collected sources (initial snapshot)

Collected on host `choi` at ingest time. Checkpoint weights were **not** copied.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/choi/APART/results_apart` | apart | `dual_proximal` / `chain` | ~53 |
| `/home/choi/APART/results_pi_only_xfit_target` | apart | `pi_only_xfit_target` | 9 |
| `/home/choi/APART/results_pi_only_xfit_target_mpi_nstep` | apart | `pi_only_xfit_mpi_nstep` | 8 |
| `/home/choi/amo/results/segment_interval` | amo | `segment_interval` | 1 (smoke) |
| `/home/choi/amo/results/amo_antmaze_t_init_tune_seed0` | amo | `antmaze_t_init_tune` | live (cron) |
| `/home/choi/ASPC/results/td3bc_aspc_table6/runs` | td3bc | `aspc_rc` | 60 (eval from stdout logs) |
| `/home/choi/amo/benchmark/iql` | iql | `benchmark` | 15 Actor0/pi_base (env/seed layout) |
| `/home/choi/ASPC/results_wpc` | wpc | `benchmark` | choi wPC cohort (policy_noise=0.2) |
| `/home/choi/ASPC/results_aspc` | aspc | `benchmark` | choi ASPC cohort (`l3_mode=aspc`) |
| `/home/choi/amo_jax/results/iql_vanilla_antmaze` | iql | `vanilla` | JAX IQL antmaze seed0–3 (paper τ=0.9, β=10) |

MPI-IQL Actor0/pi_base는 `amo/benchmark/iql/{env}/seed_N/`로 추출해 둔다
(원래 IQL loss/`beta`; W2/FB 제외). ingest는 이 트리를 `runs/iql/benchmark/`로 수집한다.

ASPC식 TD3+BC + Robust Critic(RC, 3×256+LayerNorm, Table 6)는 run dir에 `eval.jsonl`이 없어 `logs/*.log`의 Evaluation 줄을 ingest 시 `eval.jsonl`로 합성한다. checkpoint는 올리지 않는다.

Not collected (empty or logs-only):

- `/home/choi/APART/results_pi_only_xfit_target_mpi_nstep_antmaze` (no completed run dirs)

To refresh after new experiments:

```bash
cd /home/choi/amo_log
python scripts/ingest_runs.py
git add -A && git commit -m "collect: refresh from local sources" && git push
```
