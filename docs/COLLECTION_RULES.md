# 수집 규칙 (COLLECTION_RULES)

이 문서는 `amo_log`에 실험 로그를 넣을 때 반드시 따르는 규칙이다.

## 1. 무엇을 모으는가

각 **런(run)** 디렉터리에 아래만 넣는다.

| 파일 | 필수 | 설명 |
|------|------|------|
| `config.yaml` | 예 | 학습에 사용한 전체 TrainConfig (원본 복사) |
| `eval.jsonl` | 가능하면 | step별 evaluation / D4RL normalized score |
| `metrics.jsonl` | 가능하면 | step별 train metric (T, loss, bound 등) |
| `run_meta.json` | 예 | ingest가 생성: 출처 경로, algo/family, 핵심 하이퍼, 수집 시각 |

선택적으로 작은 텍스트만 추가 가능:

- `launch_cmd.txt`, `notes.md` (수 KB 수준)

## 2. 무엇을 넣지 않는가

- `checkpoint_*.pt`, `*.pth`, `*.pkl` (모델 가중치)
- `wandb/` 런 디렉터리 전체
- replay buffer / dataset 캐시
- 수십 MB 초과 단일 바이너리

필요하면 checkpoint는 **별도 스토리지**에 두고 `run_meta.json`의 `source_path` / `checkpoint_hint`로만 가리킨다.

## 3. 한 런 = 한 디렉터리

- UUID가 다른 재시작/재실행은 **별도 run_id**로 보관한다 (덮어쓰지 않음).
- 같은 env·seed·세팅이라도 resume으로 이어진 연속 학습이면, 최종 디렉터리 하나를 남기고 `run_meta.json`에 `resume_of`를 적는다.
- incomplete / smoke / full 은 **family 또는 tag**로 구분한다 (`smoke`, `1m`, `n4_dual` 등).

## 4. 알고리즘 / 패밀리 분류

`runs/<algo>/<family>/<run_id>/`

| algo | family 예 | 의미 |
|------|-----------|------|
| `apart` | `dual_proximal` | `dual_proximal=true`, 공유 T, N-hop actor bank |
| `apart` | `chain` | dual 아님, `proximal_n_steps=N` chain |
| `apart` | `pi_only_xfit_target` | π-only outer, xfit target critic |
| `apart` | `pi_only_xfit_mpi_nstep` | 위 + MPI / N-step / actor-lr 실험 |
| `amo` | `adaptive_multiscale` | 독립 T_E / T_B |
| `amo` | `segment_interval` | T_E B_PI = segment directional interval |
| `amo` | `secant` | 기본 endpoint-secant B_PI |
| `amo` | `jax_v1` | JAX AMO v1 (`use_amo`, N-hop) |
| `amo` | `jax_v2` | JAX AMO v2 (고정 T 또는 `adaptT`) |
| `amo` | `jax_v3a` | JAX AMO v3a (ratio controller) |
| `amo` | `jax_v3b` | JAX AMO v3b (horizon-ratio) |
| `amo` | `jax_td3bc` | JAX TD3+BC baseline (AMO off) |

새 family를 만들 때는 `docs/NAMING.md`에 한 줄 추가하고, ingest 매핑 테이블을 갱신한다.

## 5. 메타데이터 (`run_meta.json`) 최소 필드

```json
{
  "algo": "apart",
  "family": "dual_proximal",
  "run_id": "hme_s0_n4_dual__1fb26a1c",
  "env": "halfcheetah-medium-expert-v2",
  "env_short": "hme",
  "seed": 0,
  "source_path": "/absolute/path/to/original/run",
  "source_host": "choi",
  "collected_at": "2026-09-07T14:40:00+09:00",
  "settings": {
    "max_timesteps": 1000000,
    "proximal_n_steps": 4,
    "dual_proximal": true,
    "T": 1.25,
    "T_lr": 0.0002
  },
  "artifacts": ["config.yaml", "metrics.jsonl", "eval.jsonl"],
  "git": {
    "code_repo": "APART",
    "code_commit": null
  }
}
```

`settings`에는 **그 실험을 재현하는 데 필요한 키만** 넣는다. 전체 값은 `config.yaml`에 있다.

## 6. 수집 절차

1. 원본 런이 끝났는지 확인 (`eval.jsonl` 마지막 step, 또는 checkpoint 유무).
2. `python scripts/ingest_runs.py` 실행 (기본 소스 목록은 스크립트 내 `DEFAULT_SOURCES`).
3. `catalog/INDEX.md` / `catalog/catalog.json` 자동 갱신 확인.
4. `git add` → commit → `git push` (이 레포만).

## 7. 커밋 메시지

```text
collect(<algo>/<family>): <env_short> s<seed> [<tag>]

- N runs added/updated
- source: <host>:<path>
```

## 8. 금지

- 다른 호스트의 stale row를 베껴 “완료”로 표시하지 않는다.
- metrics를 임의로 잘라 “보기 좋게” 만들지 않는다 (요약은 catalog에서).
- 서로 다른 method/세팅의 로그를 같은 run_id에 섞지 않는다.
