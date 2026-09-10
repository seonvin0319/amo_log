# Collected sources (`ext_csv`)

Checkpoint weights were **not** copied. Logs only: `config.yaml` (from `effective_config.yaml` when that is the trainer output), `metrics.jsonl`, `eval.jsonl`, `run_meta.json`. Small extras: `train_done.json`, `run_manifest.json`.

## Host `ext_csv`

| Source path | algo | family | notes |
|-------------|------|--------|-------|
| `/home/ext_csv/AMO-behavior-l1-joint-v2/results/behavior_l1_joint_suite6_s0` | amo | `behavior_bc_l1_joint` | suite6 seed0, 1M, 6 envs |
| `/raid/ext_csv/AMO_store/amo_adaptive_multiscale_antmaze6_tlr_te1tb1_seed0/runs` | amo | `adaptive_multiscale` | antmaze×6, Te=Tb=1, T_lr 1e-3/5e-4, seed0 |
| `/raid/ext_csv/AMO_store/amo_adaptive_multiscale_antmaze6_tlr_te1tb1_1em3_seeds123/runs` | amo | `adaptive_multiscale` | antmaze×6, Te=Tb=1, T_lr=1e-3, seeds 1–3 |
| `/raid/ext_csv/AMO_store/amo_adaptive_multiscale_antmaze6_scale_sweep_seed0/runs` | amo | `adaptive_multiscale` | antmaze×6, (Te,Tb)∈{(1,1),(5,5),(5,1)}, T_lr=2e-3, seed0 |
| `/raid/ext_csv/AMO_store/amo_adaptive_multiscale_antmaze6_te1_tb025_seed0/runs` | amo | `adaptive_multiscale` | antmaze×6, Te=1 Tb=0.25, T_lr=1e-3, seed0 |
| `/raid/ext_csv/AMO_store/amo_adaptive_multiscale_loco9_te1_tb025_seed0/runs` | amo | `adaptive_multiscale` | loco×9, Te=1 Tb=0.25, T_lr=1e-3, seed0 |
| `/raid/ext_csv/AMO_store/amo_adaptive_multiscale_loco9_te1_tb1_seeds1to3/runs` | amo | `adaptive_multiscale` | loco×9, Te=Tb=1, T_lr=1e-3, seeds 1–3 (live ok) |
| `/raid/ext_csv/AMO_store/amo_adaptive_multiscale_antmaze6_tb_bc_qimprove_1em3_seed0/runs` | amo | `adaptive_multiscale` | antmaze tb/bc q-improve, T_lr=1e-3, seed0 |

Code:

- `AMO-behavior-l1-joint-v2` @ `9710464`
- `AMO` @ `c45671c` (store manifests may pin older HEADs)
- `AMO-lambda0-te-tb` @ `30abcfc` (Te1/Tb0.25 and Te1=Tb1 loco)

Catalog `final_score` for joint-v2 is last **execution** (`π_E`) score. Adaptive-multiscale uses last `eval.jsonl` `d4rl_normalized_score` (train-time 10 ep).

To refresh on ext_csv:

```bash
cd /home/ext_csv/amo_log
git checkout ext_csv
python scripts/ingest_runs.py --host=ext_csv
git add -A && git commit -m "collect(ext_csv): refresh AMO logs" && git push origin ext_csv
```

Auto: see [AUTO_PUSH.md](AUTO_PUSH.md).
