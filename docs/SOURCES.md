# Collected sources (`ext_csv`)

Checkpoint weights were **not** copied. Logs only: `config.yaml` (from `effective_config.yaml` when that is the trainer output), `metrics.jsonl`, `eval.jsonl`, `run_meta.json`. Small extras: `train_done.json`, `run_manifest.json`.

## Host `ext_csv`

| Source path | algo | family | notes |
|-------------|------|--------|-------|
| `/home/ext_csv/AMO-behavior-l1-joint-v2/results/behavior_l1_joint_suite6_s0` | amo | `behavior_bc_l1_joint` | suite6 seed0, 1M, 6 envs |

Code: `AMO-behavior-l1-joint-v2` @ `9710464` (`td3-amo`).

Catalog `final_score` is the last **execution** (`π_E`) D4RL normalized score, not BC.

To refresh on ext_csv:

```bash
cd /home/ext_csv/amo_log
git checkout ext_csv
python scripts/ingest_runs.py --host=ext_csv
git add -A && git commit -m "collect(ext_csv): refresh AMO logs" && git push origin ext_csv
```
