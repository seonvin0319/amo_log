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

To refresh after new experiments:

```bash
cd /home/choi/amo_log
python scripts/ingest_runs.py
git add -A && git commit -m "collect: refresh from local sources" && git push
```


## svcho refresh (2026-09-07)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_local_antmaze` | amo | `pi_local` | 3 |
| `/home/svcho/amo/results_pi_local_antmaze_aspc` | amo | `pi_local_aspc` | 6 |
| `/home/svcho/amo/results_pi_only_antmaze` | amo | `pi_only` | 3 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 7 |

Not found on svcho: APART `results_apart*` (no local APART tree).

## svcho refresh (2026-09-07 14:57 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 7 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 19:48 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 2 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 19:51 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 2 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 20:01 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 4 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 20:06 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 4 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 20:11 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 4 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 20:16 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 4 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 20:21 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 4 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 20:26 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 4 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 20:31 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 4 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 20:36 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 4 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 20:41 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 4 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 20:46 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 4 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 20:51 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 4 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 20:56 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 4 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 21:01 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 4 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 21:06 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 4 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 21:11 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 4 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 21:16 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 4 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-10 21:21 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 4 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-11 00:41 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 10 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-11 10:00 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 18 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-12 11:18 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/AMO-t-path-*/results` | amo | `t_path_schedule` | 54 |
| `/home/svcho/AMO-adaptive-t-four-eval-v1/results` | amo | `four_eval_fixed_n` | 10 |

Not found on svcho: APART `results_apart*`.

## svcho refresh (2026-09-12 14:22 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/A2PR/results_aspc_table1` | a2pr | `aspc_table1` | 103 |

Incomplete A2PR cells (no `.npy` / <199 evals) skipped. Checkpoints/tfevents not uploaded.

## svcho refresh (2026-09-12 21:02 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/PORL/training_curve/unified_logs/rebrac/wasserstein` | rebrac | `pogo_w2` | 300 |

Note: these are **not** vanilla ReBRAC. They are POGO multi-actor ReBRAC+W2 logs; catalog scores use **Actor0** (original ReBRAC loss only). Full train logs not uploaded.

## svcho refresh (2026-09-12 21:13 KST)

Removed mistaken `rebrac/pogo_w2` upload (POGO multi-actor + W2, not vanilla ReBRAC).
- deleted 300 runs from `runs/rebrac/pogo_w2/`
- catalog regenerated without those entries

## svcho refresh (2026-09-12 22:59 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/amo/results_t_path` | amo | `t_path_schedule` | 66 |
| `/home/svcho/amo/results_four_eval` | amo | `four_eval_fixed_n` | 10 |
| `/home/svcho/A2PR/results_aspc_table1` | a2pr | `aspc_table1` | 103 |
| `/home/svcho/ASPC/results_wpc` | wpc | `benchmark` | 0 |

Not found on svcho: APART `results_apart*`. Incomplete A2PR cells (no `.npy`) skipped.
wPC partial runs (eval.jsonl present) are uploaded every refresh.

## svcho refresh (2026-09-12 23:39 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/amo/results_t_path` | amo | `t_path_schedule` | 66 |
| `/home/svcho/amo/results_four_eval` | amo | `four_eval_fixed_n` | 10 |
| `/home/svcho/A2PR/results_aspc_table1` | a2pr | `aspc_table1` | 103 |
| `/home/svcho/ASPC/results_wpc` | wpc | `benchmark` | 1 |

Not found on svcho: APART `results_apart*`. Incomplete A2PR cells (no `.npy`) skipped.
wPC partial runs (eval.jsonl present) are uploaded every refresh.

## svcho refresh (2026-09-12 23:49 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/amo/results_t_path` | amo | `t_path_schedule` | 66 |
| `/home/svcho/amo/results_four_eval` | amo | `four_eval_fixed_n` | 10 |
| `/home/svcho/A2PR/results_aspc_table1` | a2pr | `aspc_table1` | 103 |
| `/home/svcho/ASPC/results_wpc` | wpc | `benchmark` | 3 |

Not found on svcho: APART `results_apart*`. Incomplete A2PR cells (no `.npy`) skipped.
wPC partial runs (eval.jsonl present) are uploaded every refresh.

## svcho refresh (2026-09-12 23:59 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/amo/results_t_path` | amo | `t_path_schedule` | 66 |
| `/home/svcho/amo/results_four_eval` | amo | `four_eval_fixed_n` | 10 |
| `/home/svcho/A2PR/results_aspc_table1` | a2pr | `aspc_table1` | 103 |
| `/home/svcho/ASPC/results_wpc` | wpc | `benchmark` | 3 |

Not found on svcho: APART `results_apart*`. Incomplete A2PR cells (no `.npy`) skipped.
wPC partial runs (eval.jsonl present) are uploaded every refresh.

## svcho refresh (2026-09-13 00:10 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 12 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/amo/results_t_path` | amo | `t_path_schedule` | 66 |
| `/home/svcho/amo/results_four_eval` | amo | `four_eval_fixed_n` | 10 |
| `/home/svcho/A2PR/results_aspc_table1` | a2pr | `aspc_table1` | 103 |
| `/home/svcho/ASPC/results_wpc` | wpc | `benchmark` | 3 |

Not found on svcho: APART `results_apart*`. Incomplete A2PR cells (no `.npy`) skipped.
wPC partial runs (eval.jsonl present) are uploaded every refresh.

