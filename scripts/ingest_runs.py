#!/usr/bin/env python3
"""Ingest offrl CAPO vanilla-IQL run artifacts into amo_log/runs (no checkpoints)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


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
}

DEFAULT_SOURCES: List[Dict[str, Any]] = [
    {
        "algo": "iql",
        "root": Path("/home/offrl/CAPO/results/iql"),
        "host": "offrl",
        "code_repo": "CAPO",
        "family_force": "vanilla",
        "kind": "capo_baseline_iql",
    },
]


def env_short(env: str) -> str:
    if env in ENV_SHORT:
        return ENV_SHORT[env]
    return (
        env.replace("halfcheetah", "hc")
        .replace("hopper", "hop")
        .replace("walker2d", "w")
        .replace("medium-replay", "mr")
        .replace("medium-expert", "me")
        .replace("medium", "m")
        .replace("expert", "e")
        .replace("-v2", "")
        .replace("-", "")
    )


def extract_uuid8(dirname: str) -> str:
    m = re.search(r"([0-9a-fA-F]{8})$", dirname)
    if m:
        return m.group(1).lower()
    return hashlib.sha1(dirname.encode()).hexdigest()[:8]


def json_to_yamlish(cfg: Dict[str, Any]) -> str:
    """Serialize config.json to a readable YAML-ish file (no PyYAML dep)."""
    lines = ["# converted from CAPO config.json at ingest time"]
    for k in sorted(cfg.keys()):
        v = cfg[k]
        if isinstance(v, bool):
            lines.append(f"{k}: {str(v).lower()}")
        elif v is None:
            lines.append(f"{k}: null")
        elif isinstance(v, (int, float)):
            lines.append(f"{k}: {v}")
        elif isinstance(v, str):
            lines.append(f'{k}: "{v}"')
        elif isinstance(v, list):
            lines.append(f"{k}: {json.dumps(v)}")
        elif isinstance(v, dict):
            lines.append(f"{k}: {json.dumps(v)}")
        else:
            lines.append(f"{k}: {json.dumps(v)}")
    return "\n".join(lines) + "\n"


def settings_summary(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "env",
        "seed",
        "algorithm",
        "max_timesteps",
        "eval_freq",
        "batch_size",
        "discount",
        "normalize",
        "normalize_reward",
        "n_critics",
        "actor_lr",
        "critic_lr",
        "vf_lr",
        "iql_tau",
        "iql_beta",
        "use_capo",
        "run_tag",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        if k in cfg and cfg[k] is not None:
            out[k] = cfg[k]
    return out


def discover_capo_baseline_iql(root: Path) -> List[Path]:
    """Find completed CAPO baseline (vanilla) IQL run dirs under results/iql/."""
    if not root.exists():
        return []
    hits: List[Path] = []
    # layout: results/iql/<env>/s<seed>/<stamp>_baseline_iql_<env>_s<seed>/config.json
    for cfg in root.glob("*/*/*baseline*/config.json"):
        run = cfg.parent
        name = run.name
        if "baseline_iql" not in name:
            continue
        # completed: summary.json and/or final checkpoint
        if not (run / "summary.json").exists() and not (run / "checkpoint_1000000.pt").exists():
            continue
        if not (run / "metrics.jsonl").exists():
            continue
        hits.append(run)
    return sorted(set(hits))


def metrics_to_eval_jsonl(metrics_path: Path) -> str:
    """CAPO stores eval scores inside metrics.jsonl; emit catalog-friendly eval.jsonl."""
    out_lines: List[str] = []
    for line in metrics_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = obj.get("step")
        score = obj.get("d4rl_score", obj.get("student_d4rl_score", obj.get("base_d4rl_score")))
        if step is None or score is None:
            continue
        row = {
            "step": int(step),
            "d4rl_score": float(score),
            "d4rl_normalized_score": float(score),
            "return_mean": obj.get("return_mean"),
            "return_std": obj.get("return_std"),
            "student_d4rl_score": obj.get("student_d4rl_score"),
            "base_d4rl_score": obj.get("base_d4rl_score"),
        }
        out_lines.append(json.dumps(row, sort_keys=True))
    return "\n".join(out_lines) + ("\n" if out_lines else "")


def ingest_capo_baseline_iql(
    src: Path,
    *,
    host: str,
    code_repo: str,
    dry_run: bool,
) -> Optional[Dict[str, Any]]:
    cfg = json.loads((src / "config.json").read_text())
    if cfg.get("use_capo") is True:
        # safety: only vanilla / baseline
        return None
    env = str(cfg.get("env") or "unknown")
    seed = int(cfg.get("seed", 0) or 0)
    algo = "iql"
    family = "vanilla"
    # distinguish CAPO-repo baseline IQL from other hosts' MPI/CORL vanilla logs
    variant = "capo_baseline"
    if int(cfg.get("max_timesteps", 0) or 0) >= 1_000_000:
        variant = "capo_baseline_1m"
    uuid8 = extract_uuid8(src.name)
    short = env_short(env)
    run_id = f"{short}_s{seed}_{variant}__{uuid8}"
    dest = RUNS / algo / family / run_id

    artifacts = ["config.yaml", "metrics.jsonl", "eval.jsonl"]
    meta = {
        "algo": algo,
        "family": family,
        "run_id": run_id,
        "env": env,
        "env_short": short,
        "seed": seed,
        "variant": variant,
        "legacy_name": src.name,
        "source_path": str(src.resolve()),
        "source_host": host,
        "collected_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "settings": settings_summary(cfg),
        "artifacts": artifacts,
        "protocol": "capo_baseline_iql",
        "git": {"code_repo": code_repo, "code_commit": None},
        "notes": "Vanilla IQL (use_capo=false) from CAPO configs/baseline_iql.yaml; checkpoints not uploaded.",
    }

    if dry_run:
        print(f"DRY {src} -> {dest.relative_to(ROOT)}")
        return meta

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.yaml").write_text(json_to_yamlish(cfg))
    (dest / "metrics.jsonl").write_text((src / "metrics.jsonl").read_text(errors="ignore"))
    (dest / "eval.jsonl").write_text(metrics_to_eval_jsonl(src / "metrics.jsonl"))
    # tiny extras
    for extra in ("summary.json", "training_curve_meta.json"):
        if (src / extra).exists():
            dest.joinpath(extra).write_text((src / extra).read_text(errors="ignore"))
            meta["artifacts"].append(extra)
    (dest / "run_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"OK  {dest.relative_to(ROOT)}")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-catalog", action="store_true", default=True)
    args = parser.parse_args()

    collected: List[Dict[str, Any]] = []
    for src_spec in DEFAULT_SOURCES:
        root: Path = src_spec["root"]
        kind = src_spec.get("kind")
        if kind == "capo_baseline_iql":
            for run_dir in discover_capo_baseline_iql(root):
                meta = ingest_capo_baseline_iql(
                    run_dir,
                    host=src_spec["host"],
                    code_repo=src_spec["code_repo"],
                    dry_run=args.dry_run,
                )
                if meta:
                    collected.append(meta)
        else:
            print(f"WARN: unknown kind={kind} for {root}")

    print(f"# collected {len(collected)} runs")
    if args.rebuild_catalog and not args.dry_run:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_catalog import build

        build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
