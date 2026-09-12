# 이름 / 경로 규칙 (NAMING)

## 1. 런 디렉터리 ID (`run_id`)

형식:

```text
{env_short}_s{seed}_{variant}__{uuid8}
```

예:

- `hme_s0_n4_dual__1fb26a1c`
- `hopme_s0_pi_only_xfit__68128a42`
- `hme_s0_mpi_n4_alr1e4__93ad3672`
- `hme_s0_seg4_smoke__c3cff423`

### 구성요소

| 조각 | 규칙 |
|------|------|
| `env_short` | 아래 표의 짧은 코드 |
| `seed` | 정수, `s0`, `s1`, … |
| `variant` | 세팅을 snake_case로 압축 (`n4_dual`, `n2`, `n4_Ng`, `mpi_n2_alr1e4`, `seg4_smoke`) |
| `uuid8` | 원본 디렉터리명 끝 8-hex (충돌 방지). 없으면 ingest가 내용 해시 8자 생성 |

전체 경로:

```text
runs/{algo}/{family}/{run_id}/
```

## 2. 환경 짧은 이름 (`env_short`)

| env | short |
|-----|-------|
| halfcheetah-medium-v2 | `hcm` |
| halfcheetah-medium-replay-v2 | `hcmr` |
| halfcheetah-medium-expert-v2 | `hme` |
| halfcheetah-expert-v2 | `hce` |
| hopper-medium-v2 | `hopm` |
| hopper-medium-replay-v2 | `hopmr` |
| hopper-medium-expert-v2 | `hopme` |
| hopper-expert-v2 | `hope` |
| walker2d-medium-v2 | `wm` |
| walker2d-medium-replay-v2 | `wmr` |
| walker2d-medium-expert-v2 | `wme` |
| walker2d-expert-v2 | `we` |
| antmaze-umaze-v2 | `amu` |
| antmaze-umaze-diverse-v2 | `amud` |
| antmaze-medium-play-v2 | `ammp` |
| antmaze-medium-diverse-v2 | `ammd` |
| antmaze-large-play-v2 | `amlp` |
| antmaze-large-diverse-v2 | `amld` |

새 env는 이 표에 추가한 뒤 사용한다. 임시로 풀네임을 쓰지 않는다.

> 참고: 과거 APART 일부 런은 halfcheetah를 `cm`/`cme`로 썼다. ingest는 이를 `hcm`/`hme`로 **정규화**하고, `run_meta.json`의 `legacy_name`에 원본을 남긴다.

## 3. variant 토큰 규칙

여러 토큰은 `_`로 연결, **의미 있는 순서는** `n{N}` → `dual|chain` → `Ng` → `mpi` → `alr…` → `seg{M}` → `smoke|1m` → 기타.

| 토큰 | 의미 |
|------|------|
| `n1`/`n2`/`n4` | `proximal_n_steps` |
| `dual` | `dual_proximal=true` |
| `Ng` | outer loss에 N배율 없음 (레거시 태그) |
| `mpi_n{k}` | MPI / multi-step actor 실험의 N |
| `alr1e4` | actor lr = 1e-4 등 |
| `seg{M}` | `pi_bound_segments=M` |
| `secant` / `segint` | `pi_bound_method` |
| `smoke` | 짧은 검증 런 |
| `Tlr2e3` | `T_lr=2e-3` |
| `adaptT` | JAX AMO v2 `amo_v2_adapt_t=true` |
| `T025`/`T050`/`T075` | JAX AMO v2 고정 `amo_v2_t_init` |
| `inner{k}` | JAX AMO v2 `amo_v2_inner_steps=k` (기본 1이면 생략) |
| `v3a`/`v3b` | JAX AMO v3 family marker |
| `td3bc` | JAX TD3+BC baseline |
| `rebrac` | ReBRAC-AMO critic BC |
| `qouter` | T_E outer = Q_1_target(s, π+) |
| `TE{k}` / `TB{k}` | adaptive_multiscale init (`T_E`/`T_B`; TB는 TE와 다를 때만) |
| `incomplete` | 목표 step 미달 (summary complete가 아닐 때) |

## 4. family 디렉터리

family는 **실험 프로토콜** 단위다. 하이퍼 하나 바뀐 정도는 variant로 넣고 family를 쪼개지 않는다.

| family | 포함 조건 |
|--------|-----------|
| `dual_proximal` | APART dual_proximal sweeps |
| `chain` | APART non-dual proximal chain |
| `pi_only_xfit_target` | π-only + xfit target, default N=1 |
| `pi_only_xfit_mpi_nstep` | π-only + MPI/N-step/alr ablations |
| `segment_interval` | AMO segment-interval B_PI |
| `adaptive_multiscale` | AMO T_E/T_B 분리 |
| `jax_v1` | JAX AMO v1 (`use_amo`) |
| `jax_v2` | JAX AMO v2 |
| `jax_v3a` | JAX AMO v3a |
| `jax_v3b` | JAX AMO v3b |
| `jax_td3bc` | JAX TD3+BC baseline (AMO 꺼짐) |
| `rebrac_amo` | PyTorch AMO + ReBRAC critic BC |
| `amo_td3bc` | PyTorch AMO, T_E maximizes Q_1_target(s, π+) |
| `benchmark` | ASPC/WPC D4RL benchmark (`algo=aspc|wpc`; variant = `l3_mode` / `wpc`) |
| `misc` | 위 어디에도 안 들어가면 임시. 곧 family를 승격할 것 |

## 5. 카탈로그 표기

`catalog/INDEX.md` 한 줄 예:

```text
| apart | dual_proximal | hme_s0_n4_dual__1fb26a1c | halfcheetah-medium-expert-v2 | 0 | 1e6 | 98.2 | ... |
```

점수는 `eval.jsonl` **마지막** `d4rl_normalized_score` (또는 동등 키). 없으면 `—`.

## 6. 금지하는 이름

- 공백, `/`, 대문자 남발 (`HME_S0` 금지 → `hme_s0`)
- `final`, `best`, `new`, `test2` 같은 모호한 variant
- 호스트 이름을 run_id에 넣기 (호스트는 `run_meta.source_host`)
