# Collected sources (`ext_csv`)

Checkpoint weights were **not** copied. Logs only: `config.yaml` (from `effective_config.yaml` when that is the trainer output), `metrics.jsonl`, `eval.jsonl`, `run_meta.json`. Small extras: `train_done.json`, `run_manifest.json`.

## Host `ext_csv`

| Source path | algo | family | notes |
|-------------|------|--------|-------|
| `/home/ext_csv/AMO-behavior-l1-joint-v2/results/behavior_l1_joint_suite6_s0` | amo | `behavior_bc_l1_joint` | suite6 seed0, 1M, 6 envs |
| `/home/ext_csv/AMO/results/amo_adaptive_multiscale_antmaze6_tlr_te1tb1_seed0/runs` | amo | `adaptive_multiscale` | antmaze-v2 x6, Te=Tb=1, T_lr 1e-3/5e-4, seed0 |
| `/home/ext_csv/AMO/results/amo_adaptive_multiscale_antmaze6_scale_sweep_seed0/runs` | amo | `adaptive_multiscale` | antmaze-v2 x6, (Te,Tb) grid, T_lr=2e-3, seed0 |

Code:

- `AMO-behavior-l1-joint-v2` @ `9710464` (`td3-amo`)
- `AMO` wandb commit `a8c1e48` (launch_manifest HEAD was `32a1028`, one commit later)

Catalog `final_score` for joint-v2 is last **execution** (`π_E`) score. Adaptive-multiscale uses last `eval.jsonl` `d4rl_normalized_score` (train-time 10 ep). Posthoc 50-ep JSON stays in the original pack, not amo_log.

Catalog `final_score` is the last **execution** (`π_E`) D4RL normalized score, not BC.

To refresh on ext_csv:

```bash
cd /home/ext_csv/amo_log
git checkout ext_csv
python scripts/ingest_runs.py --host=ext_csv
git add -A && git commit -m "collect(ext_csv): refresh AMO logs" && git push origin ext_csv
```
