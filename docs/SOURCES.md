# 수집 원본

각 실행의 run_meta.json에 원본 source_path와 코드 commit을 보존했습니다. [현재 카탈로그](../catalog/INDEX.md)를 기준으로 확인하세요.

## svcho refresh (2026-09-15 00:43 KST)

Collected on host `svcho` without cloning checkpoint weights.

| Source path | algo | family (assigned) | #runs |
|-------------|------|-------------------|------:|
| `/home/svcho/amo/results_pi_*` | amo | `pi_local`/`pi_local_aspc`/`pi_only` | 0 |
| `/home/svcho/CaPO/results_amo_*` | amo | `capo_td3bc` | 0 |
| `/home/svcho/amo/results_t_path` | amo | `t_path_schedule` | 0 |
| `/home/svcho/amo/results_four_eval` | amo | `four_eval_fixed_n` | 0 |
| `/home/svcho/A2PR/results_aspc_table1` | a2pr | `aspc_table1` | 103 |
| `/home/svcho/A2PR/results_aspc_table1_final50` | a2pr | `aspc_table1_final50` | 0 |
| `/home/svcho/ASPC/results_wpc` | wpc | `benchmark` | 10 |
| `/home/svcho/amo_iql_sweep/results/iql_amo_lr1e3_beta_sweep_seed0` | iql_amo | `lr1e3_beta_sweep` | 90 |
| `/home/svcho/amo_iql_sweep/results/td3_amo_adroit_T1_Tlr1e3_seeds03` | td3_amo | `adroit_T1_Tlr1e3` | 48 |

Not found on svcho: APART `results_apart*`. Incomplete A2PR cells (no `.npy`) skipped.
wPC / IQL+AMO / TD3+AMO Adroit partial runs (eval.jsonl present) are uploaded every refresh.
IQL+AMO preferred final score protocol: `final50_singlepass_v1` (legacy posthoc 10×5 seed_stride=0 retained in eval.jsonl but not selected).

