# Collected sources (offrl / iisl-server01)

Collected on host `offrl` (hostname `iisl-server01`) without cloning checkpoint weights.

| Source path | algo | family | #runs |
|-------------|------|--------|------:|
| `/home/offrl/CAPO/results/iql/**/ *baseline_iql*` | iql | `vanilla` | 10 |

Notes:

- These are **vanilla IQL** controls from CAPO (`configs/baseline_iql.yaml`, `use_capo=false`, `n_critics=4`, seed 0, 1M steps).
- CAPO `metrics.jsonl` already stores eval scores; ingest writes a catalog-friendly `eval.jsonl` (keys include `d4rl_score` / `d4rl_normalized_score`).
- Checkpoints (`*.pt`), plots, and `train.log` are **not** uploaded.

To refresh:

```bash
cd /home/offrl/amo_log
python scripts/ingest_runs.py
git add -A && git commit -m "collect(iql/vanilla): CAPO baseline IQL refresh" && git push -u origin offrl
```
