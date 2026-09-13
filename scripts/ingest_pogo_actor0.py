#!/usr/bin/env python3
"""Extract vanilla IQL / ReBRAC from POGO multi-actor stdout logs (Actor 0 only).

Actor0 uses the original algorithm loss (no W2 / Fisher-Rao / Sinkhorn term).
One completed log is kept per (algo, env, seed); W2 weight variants are not
separate vanilla runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

ENV_SHORT = {
    "halfcheetah-medium-v2": "hcm",
    "halfcheetah-medium-replay-v2": "hcmr",
    "halfcheetah-medium-expert-v2": "hme",
    "halfcheetah-expert-v2": "hce",
    "hopper-medium-v2": "hopm",
    "hopper-medium-replay-v2": "hopmr",
    "hopper-medium-expert-v2": "hopme",
    "hopper-expert-v2": "hope",
    "walker2d-medium-v2": "wm",
    "walker2d-medium-replay-v2": "wmr",
    "walker2d-medium-expert-v2": "wme",
    "walker2d-expert-v2": "we",
    "antmaze-umaze-v2": "amu",
    "antmaze-umaze-diverse-v2": "amud",
    "antmaze-medium-play-v2": "ammp",
    "antmaze-medium-diverse-v2": "ammd",
    "antmaze-large-play-v2": "amlp",
    "antmaze-large-diverse-v2": "amld",
}

IQL_LOCO_DEFAULTS = {
    "iql_tau": 0.7,
    "beta": 3.0,
    "vf_lr": 0.0003,
    "qf_lr": 0.0003,
    "actor_lr": 0.0003,
    "iql_deterministic": False,
    "batch_size": 256,
    "discount": 0.99,
    "tau": 0.005,
    "buffer_size": 2000000,
    "normalize": True,
    "normalize_reward": False,
}

IQL_ANTMAZE_DEFAULTS = {
    **IQL_LOCO_DEFAULTS,
    "iql_tau": 0.9,
    "beta": 10.0,
    "normalize_reward": True,
}


def env_short(env: str) -> str:
    if env in ENV_SHORT:
        return ENV_SHORT[env]
    return (
        env.replace("halfcheetah", "hc")
        .replace("hopper", "hop")
        .replace("walker2d", "w")
        .replace("antmaze", "am")
        .replace("medium-replay", "mr")
        .replace("medium-expert", "me")
        .replace("medium-play", "mp")
        .replace("medium-diverse", "md")
        .replace("large-play", "lp")
        .replace("large-diverse", "ld")
        .replace("umaze-diverse", "ud")
        .replace("umaze", "u")
        .replace("medium", "m")
        .replace("expert", "e")
        .replace("-v2", "")
        .replace("-", "")
    )


def yaml_value(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


def normalize_env(raw: str) -> str:
    return raw.strip().replace("_", "-")


def parse_log(path: Path) -> Optional[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "Actor 0 - Raw:" not in text and "Actor 0 - Raw:" not in text:
        return None

    header: Dict[str, Any] = {}
    m = re.search(r"Algorithm:\s*([A-Za-z0-9_+-]+)", text)
    if m:
        header["algorithm"] = m.group(1).lower()
    m = re.search(r"Environment:\s*([^\s,]+)", text)
    if m:
        header["env"] = normalize_env(m.group(1))
    if "env" not in header:
        m = re.search(r"Env:\s*([^\s,]+)", text)
        if m:
            header["env"] = normalize_env(m.group(1))
    m = re.search(r"Seed:\s*(\d+)", text)
    if m:
        header["seed"] = int(m.group(1))
    if "seed" not in header:
        m = re.search(r"seed(\d+)", path.name)
        if m:
            header["seed"] = int(m.group(1))
    m = re.search(r"Actors:\s*(\d+)", text)
    if m:
        header["num_actors"] = int(m.group(1))
    m = re.search(r"W2 weights \(Actor1\+\):\s*(\[[^\]]+\])", text)
    if m:
        header["w2_weights"] = m.group(1)
    m = re.search(r"policy_freq:\s*(\d+)", text)
    if m:
        header["policy_freq"] = int(m.group(1))
    m = re.search(r"Actor types:\s*(\[[^\]]+\])", text)
    if m:
        header["actor_types"] = m.group(1)
    m = re.search(r"/home/([A-Za-z0-9_]+)/", text)
    if m:
        header["source_user"] = m.group(1)

    rows: List[Dict[str, Any]] = []
    step: Optional[int] = None
    n_episodes = 10
    for line in text.splitlines():
        tm = re.search(r"Time steps:\s*(\d+)", line)
        if tm:
            step = int(tm.group(1))
            continue
        em = re.search(r"Evaluation over\s+(\d+)\s+episodes:", line)
        if em:
            n_episodes = int(em.group(1))
            continue
        am = re.search(
            r"Actor 0 - Raw:\s*([-\d.]+),\s*D4RL score:\s*([-\d.]+)",
            line,
        )
        if not am:
            continue
        if step is None:
            step = (len(rows) + 1) * 5000
        rows.append(
            {
                "step": step,
                "n_episodes": n_episodes,
                "return_mean": float(am.group(1)),
                "d4rl_normalized_score": float(am.group(2)),
                "actor": 0,
            }
        )

    if not rows:
        return None

    final: Dict[str, Any] = {}
    # Isolate Actor 0 final block so later actors are not mixed in.
    fm = re.search(
        r"======== Final Evaluation: Actor 0 ========.*?======== Final Evaluation: Actor 1",
        text,
        re.S,
    )
    block = fm.group(0) if fm else text
    dm = re.search(
        r"\[FINAL\] Deterministic:\s*mean=([-\d.]+),\s*std=([-\d.]+)",
        block,
    )
    if dm:
        final["final_det_mean"] = float(dm.group(1))
        final["final_det_std"] = float(dm.group(2))
    sm = re.search(
        r"\[FINAL\] Stochastic:\s*mean=([-\d.]+),\s*std=([-\d.]+)",
        block,
    )
    if sm:
        final["final_stoch_mean"] = float(sm.group(1))
        final["final_stoch_std"] = float(sm.group(2))

    completed = (
        "Final Evaluation (all actors)" in text
        or "Training completed" in text
        or "model_step" in text and "_final" in text
    )
    header["n_episodes"] = n_episodes
    header["eval_freq"] = 5000
    if rows:
        last_step = int(rows[-1]["step"])
        header["max_timesteps"] = 1000000 if last_step >= 1_000_000 else last_step
    return {
        "header": header,
        "rows": rows,
        "final": final,
        "completed": completed,
        "path": path,
    }


def infer_algo(path: Path, parsed: Dict[str, Any]) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    if "iql" in parts:
        return "iql"
    if "rebrac" in parts:
        return "rebrac"
    algo = parsed["header"].get("algorithm")
    if algo in ("iql", "rebrac"):
        return algo
    return None


def infer_metric(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for name in ("wasserstein", "fishrao", "sinkhorn"):
        if name in parts:
            return name
    return "unknown"


def infer_weight(path: Path, parsed: Dict[str, Any]) -> Optional[float]:
    for part in path.parts:
        m = re.fullmatch(r"w(\d+(?:\.\d+)?)", part)
        if m:
            return float(m.group(1))
    raw = parsed["header"].get("w2_weights")
    if isinstance(raw, str):
        nums = re.findall(r"[\d.]+", raw)
        if nums:
            return float(nums[0])
    return None


def pick_best(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    def key(p: Dict[str, Any]) -> Tuple:
        metric = infer_metric(p["path"])
        weight = infer_weight(p["path"], p)
        last_step = int(p["rows"][-1]["step"])
        return (
            1 if p["completed"] else 0,
            last_step,
            len(p["rows"]),
            1 if metric == "wasserstein" else 0,
            -(weight if weight is not None else 10**9),
        )

    return max(cands, key=key)


def iql_defaults(env: str) -> Dict[str, Any]:
    if env.startswith("antmaze"):
        return dict(IQL_ANTMAZE_DEFAULTS)
    return dict(IQL_LOCO_DEFAULTS)


def write_config(algo: str, env: str, seed: int, parsed: Dict[str, Any]) -> str:
    header = parsed["header"]
    lines = [
        f"# Vanilla {algo.upper()} (pi_base / Actor0 extracted from POGO multi-actor).",
        "# Actor1+ W2/Fisher-Rao/Sinkhorn fields were dropped; Actor0 uses the original loss only.",
        f"algo: {algo}",
        "family: vanilla",
        "extracted_from: pogo_actor0",
        f"env: {env}",
        f"seed: {seed}",
        f"eval_freq: {int(header.get('eval_freq', 5000))}",
        f"n_episodes: {int(header.get('n_episodes', 10))}",
        f"max_timesteps: {int(header.get('max_timesteps', 1000000))}",
    ]
    if algo == "iql":
        for key, val in iql_defaults(env).items():
            lines.append(f"{key}: {yaml_value(val)}")
    if "policy_freq" in header:
        lines.append(f"policy_freq: {int(header['policy_freq'])}")
    return "\n".join(lines) + "\n"


def settings_summary(algo: str, env: str, seed: int, parsed: Dict[str, Any]) -> Dict[str, Any]:
    header = parsed["header"]
    out: Dict[str, Any] = {
        "env": env,
        "seed": seed,
        "eval_freq": int(header.get("eval_freq", 5000)),
        "n_episodes": int(header.get("n_episodes", 10)),
        "max_timesteps": int(header.get("max_timesteps", 1000000)),
    }
    if algo == "iql":
        out.update(iql_defaults(env))
    if "policy_freq" in header:
        out["policy_freq"] = int(header["policy_freq"])
    return out


def write_run(algo: str, parsed: Dict[str, Any], dest_root: Path, dry_run: bool) -> Optional[Dict[str, Any]]:
    header = parsed["header"]
    env = str(header.get("env") or "unknown")
    if env == "unknown":
        return None
    seed = int(header.get("seed", 0) or 0)
    family = "vanilla"
    variant = "pi_base"
    last_step = int(parsed["rows"][-1]["step"])
    if last_step < 100_000:
        variant = "pi_base_smoke"
    uuid8 = hashlib.sha1(str(parsed["path"].resolve()).encode()).hexdigest()[:8]
    short = env_short(env)
    run_id = f"{short}_s{seed}_{variant}__{uuid8}"
    dest = dest_root / "runs" / algo / family / run_id

    source_host = "choi"
    meta = {
        "algo": algo,
        "family": family,
        "run_id": run_id,
        "env": env,
        "env_short": short,
        "seed": seed,
        "variant": variant,
        "legacy_name": parsed["path"].name,
        "source_path": str(parsed["path"].parent.resolve()),
        "source_host": source_host,
        "source_log": str(parsed["path"].resolve()),
        "source_user": header.get("source_user"),
        "collected_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "settings": settings_summary(algo, env, seed, parsed),
        "artifacts": ["config.yaml", "eval.jsonl"],
        "git": {"code_repo": "PORL", "code_commit": None},
        "eval_source": "synthesized_from_pogo_actor0_log",
        "notes": (
            f"Actor0/pi_base scores from POGO multi-actor {algo.upper()}. "
            "Actor0 has no W2/Fisher-Rao/Sinkhorn term and is vanilla algorithm performance."
        ),
        "pogo_source": {
            "num_actors": header.get("num_actors"),
            "w2_weights": header.get("w2_weights"),
            "metric": infer_metric(parsed["path"]),
            "weight": infer_weight(parsed["path"], parsed),
            "actor_types": header.get("actor_types"),
            "completed": parsed["completed"],
            **parsed["final"],
        },
    }

    if dry_run:
        print(
            f"DRY {parsed['path']} -> {dest.relative_to(dest_root)} "
            f"evals={len(parsed['rows'])} last={last_step} completed={parsed['completed']}"
        )
        return meta

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.yaml").write_text(write_config(algo, env, seed, parsed), encoding="utf-8")
    (dest / "eval.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in parsed["rows"]),
        encoding="utf-8",
    )
    (dest / "run_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"OK {dest.relative_to(dest_root)}")
    return meta


def discover(src: Path) -> List[Path]:
    out: List[Path] = []
    for algo in ("iql", "rebrac"):
        root = src / algo
        if not root.exists():
            continue
        for log in root.rglob("*.log"):
            if "combined" in {p.lower() for p in log.parts}:
                continue
            out.append(log)
    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="unified_logs root")
    parser.add_argument("--dest", type=Path, default=ROOT, help="amo_log checkout")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-catalog", action="store_true", default=True)
    args = parser.parse_args()

    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    skipped = 0
    for log in discover(args.src):
        parsed = parse_log(log)
        if parsed is None:
            skipped += 1
            continue
        algo = infer_algo(log, parsed)
        if algo not in ("iql", "rebrac"):
            skipped += 1
            continue
        env = parsed["header"].get("env")
        seed = parsed["header"].get("seed")
        if not env or seed is None:
            skipped += 1
            continue
        grouped.setdefault((algo, env, int(seed)), []).append(parsed)

    collected: List[Dict[str, Any]] = []
    for key, cands in sorted(grouped.items()):
        best = pick_best(cands)
        meta = write_run(key[0], best, args.dest, args.dry_run)
        if meta:
            collected.append(meta)

    print(
        f"# selected {len(collected)} vanilla runs "
        f"from {sum(len(v) for v in grouped.values())} actor0 logs "
        f"(skipped {skipped})"
    )
    if args.rebuild_catalog and not args.dry_run:
        import sys

        sys.path.insert(0, str((args.dest / "scripts").resolve()))
        from build_catalog import build

        # build_catalog uses its own ROOT; run only if dest is that checkout
        if args.dest.resolve() == ROOT.resolve():
            build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
