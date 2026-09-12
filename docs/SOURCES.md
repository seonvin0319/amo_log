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
| `/home/choi/MPI/results/iql` | iql | `vanilla` | 15 (Actor0/pi_base from MPI-IQL logs) |

ASPC식 TD3+BC + Robust Critic(RC, 3×256+LayerNorm, Table 6)는 run dir에 `eval.jsonl`이 없어 `logs/*.log`의 Evaluation 줄을 ingest 시 `eval.jsonl`로 합성한다. checkpoint는 올리지 않는다.

MPI-IQL은 multi-actor(`num_actors=3`) + W2/FB인데, **Actor0(pi_base)는 W2/FB term이 없어 vanilla IQL과 동일 업데이트**다. ingest는 stdout 로그의 `Actor 0 - Raw/D4RL score`만 `eval.jsonl`로 합성하고, config에서는 `num_actors`/`w2_weights`/`use_fb`/`fb_tau`를 제거한다.

Not collected (empty or logs-only):

- `/home/choi/APART/results_pi_only_xfit_target_mpi_nstep_antmaze` (no completed run dirs)

To refresh after new experiments:

```bash
cd /home/choi/amo_log
python scripts/ingest_runs.py
git add -A && git commit -m "collect: refresh from local sources" && git push
```
