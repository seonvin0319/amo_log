# choi audit 2026-09-14 — TD3+AMO Adroit T_lr=3e-4

- Generated: 2026-09-14T10:00:51+09:00
- Code ref: `b9338d9815525482d2cf34d6fc6315ea4d2f93a6`
- Worktree: `codex/log-audit-20260914-choi` @ `/home/choi/amo_jax_wt_log_audit`
- Log branch: `choi`
- Profile: T_E=T_B=1, T_lr=3e-4, actor/critic_lr=3e-4
- Eval: **final50_singlepass_v1**

## Dedup
- Adroit: canonical `*_jax_te1_tb1_Tlr0p0003__*`; stale variant exports aliased.
- ingest_runs.py: source_path index + catalog skips is_alias; prefers latest source refresh (not high-score).

## 48-cell final50 table
| env | s0 | s1 | s2 | s3 | mean±std |
|---|---:|---:|---:|---:|---|
| door-human-v1 | -0.26 | -0.16 | -0.22 | -0.26 | -0.23±0.05 |
| door-cloned-v1 | -0.13 | -0.07 | -0.33 | 0.04 | -0.12±0.16 |
| door-expert-v1 | 12.17 | 41.43 | 11.55 | 17.71 | 20.71±14.08 |
| hammer-human-v1 | -0.23 | -0.45 | 0.21 | -0.16 | -0.16±0.27 |
| hammer-cloned-v1 | 1.38 | 0.36 | 2.58 | @425k | — |
| hammer-expert-v1 | — | — | — | — | — |
| pen-human-v1 | — | — | — | — | — |
| pen-cloned-v1 | — | — | — | — | — |
| pen-expert-v1 | — | — | — | — | — |
| relocate-human-v1 | — | — | — | — | — |
| relocate-cloned-v1 | — | — | — | — | — |
| relocate-expert-v1 | — | — | — | — | — |

## Door-expert
- final50: **20.71±14.08** (legacy ref ≈20.07±11.26)

## Incomplete / not started (no retrain)
- `hammer-cloned-v1` seed3: training step=425000
- `hammer-expert-v1` seed0: not_started step=0
- `hammer-expert-v1` seed1: not_started step=0
- `hammer-expert-v1` seed2: not_started step=0
- `hammer-expert-v1` seed3: not_started step=0
- `pen-human-v1` seed0: not_started step=0
- `pen-human-v1` seed1: not_started step=0
- `pen-human-v1` seed2: not_started step=0
- `pen-human-v1` seed3: not_started step=0
- `pen-cloned-v1` seed0: not_started step=0
- `pen-cloned-v1` seed1: not_started step=0
- `pen-cloned-v1` seed2: not_started step=0
- `pen-cloned-v1` seed3: not_started step=0
- `pen-expert-v1` seed0: not_started step=0
- `pen-expert-v1` seed1: not_started step=0
- `pen-expert-v1` seed2: not_started step=0
- `pen-expert-v1` seed3: not_started step=0
- `relocate-human-v1` seed0: not_started step=0
- `relocate-human-v1` seed1: not_started step=0
- `relocate-human-v1` seed2: not_started step=0
- `relocate-human-v1` seed3: not_started step=0
- `relocate-cloned-v1` seed0: not_started step=0
- `relocate-cloned-v1` seed1: not_started step=0
- `relocate-cloned-v1` seed2: not_started step=0
- `relocate-cloned-v1` seed3: not_started step=0
- `relocate-expert-v1` seed0: not_started step=0
- `relocate-expert-v1` seed1: not_started step=0
- `relocate-expert-v1` seed2: not_started step=0
- `relocate-expert-v1` seed3: not_started step=0

## IQL seed0
- vanilla vs benchmark: different source trees; keep separate cohorts.
- benchmark internal default vs pi_base same-source dups aliased to default.

