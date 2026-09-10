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

