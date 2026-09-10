#!/usr/bin/env python3
"""Ingest AMO/APART run artifacts into amo_log/runs with canonical names."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

KEEP_FILES = ("config.yaml", "metrics.jsonl", "eval.jsonl")

ENV_SHORT = {
    "halfcheetah-medium-v2": "hcm",
    "halfcheetah-medium-replay-v2": "hcmr",
    "halfcheetah-medium-expert-v2": "hme",
    "halfcheetah-expert-v2": "hce",
    "halfcheetah-random-v2": "hcrand",
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

# legacy short codes that appeared in older APART directory names
LEGACY_ENV_PREFIX = {
    "cm": "hcm",
    "cmr": "hcmr",
    "cme": "hme",
    "hm": "hopm",  # careful: some used hm for hopper-medium
    "hmr": "hopmr",
    "hme": "hopme",  # conflict: hme also means halfcheetah-medium-expert
}

DEFAULT_SOURCES: List[Dict[str, Any]] = [
    {
        "algo": "apart",
        "root": Path("/home/choi/APART/results_apart"),
        "host": "choi",
        "code_repo": "APART",
    },
    {
        "algo": "apart",
        "root": Path("/home/choi/APART/results_pi_only_xfit_target"),
        "host": "choi",
        "code_repo": "APART",
        "family_force": "pi_only_xfit_target",
    },
    {
        "algo": "apart",
        "root": Path("/home/choi/APART/results_pi_only_xfit_target_mpi_nstep"),
        "host": "choi",
        "code_repo": "APART",
        "family_force": "pi_only_xfit_mpi_nstep",
    },
    {
        "algo": "amo",
        "root": Path("/home/choi/amo/results/segment_interval"),
        "host": "choi",
        "code_repo": "amo",
        "family_force": "segment_interval",
        "nested": True,
    },
    {
        "algo": "amo",
        "root": Path("/home/choi/amo/results/amo_antmaze_t_init_tune_seed0"),
        "host": "choi",
        "code_repo": "AMO",
        "family_force": "antmaze_t_init_tune",
        "nested": True,
    },
]


def load_yaml_lite(path: Path) -> Dict[str, Any]:
    """Minimal YAML subset reader (key: value) without PyYAML dependency."""
    out: Dict[str, Any] = {}
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        raw = raw.strip().strip("'\"")
        if raw.lower() in ("true", "false"):
            out[key] = raw.lower() == "true"
        elif raw.lower() in ("null", "none", ""):
            out[key] = None
        else:
            try:
                if "." in raw or "e" in raw.lower():
                    out[key] = float(raw)
                else:
                    out[key] = int(raw)
            except ValueError:
                out[key] = raw
    return out


def env_short(env: str) -> str:
    if env in ENV_SHORT:
        return ENV_SHORT[env]
    # fallback: compress
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


def extract_uuid8(dirname: str) -> str:
    m = re.search(r"([0-9a-fA-F]{8})$", dirname)
    if m:
        return m.group(1).lower()
    return hashlib.sha1(dirname.encode()).hexdigest()[:8]


def classify_family(algo: str, cfg: Dict[str, Any], force: Optional[str]) -> str:
    if force:
        return force
    if algo == "amo":
        method = str(cfg.get("pi_bound_method", "secant"))
        if method == "segment_interval":
            return "segment_interval"
        if cfg.get("adaptive_multiscale"):
            return "adaptive_multiscale"
        return "secant"
    # apart
    if cfg.get("dual_proximal"):
        return "dual_proximal"
    name = str(cfg.get("name", ""))
    if "pi_only_xfit" in name and "mpi" in name:
        return "pi_only_xfit_mpi_nstep"
    if "pi_only_xfit" in name:
        return "pi_only_xfit_target"
    if int(cfg.get("proximal_n_steps", 1) or 1) > 1:
        return "chain"
    return "misc"


def build_variant(algo: str, family: str, cfg: Dict[str, Any], dirname: str) -> str:
    tokens: List[str] = []
    n = int(cfg.get("proximal_n_steps", 1) or 1)
    if family in ("dual_proximal", "chain") or "apart_n" in dirname:
        tokens.append(f"n{n}")
    if cfg.get("dual_proximal") or "_dual" in dirname:
        tokens.append("dual")
    if "_Ng" in dirname or dirname.endswith("_Ng") or "_Ng-" in dirname:
        tokens.append("Ng")
    if "mpi_n" in dirname:
        m = re.search(r"mpi_n(\d+)", dirname)
        if m:
            tokens.append(f"mpi_n{m.group(1)}")
    if "alr1e4" in dirname:
        tokens.append("alr1e4")
    if family == "segment_interval":
        segs = int(cfg.get("pi_bound_segments", 4) or 4)
        tokens.append(f"seg{segs}")
    if family in ("adaptive_multiscale", "antmaze_t_init_tune") or cfg.get(
        "adaptive_multiscale"
    ):
        te = cfg.get("T_E")
        tb = cfg.get("T_B")
        if te is not None:
            te_f = float(te)
            tokens.append(f"te{int(te_f) if te_f.is_integer() else te_f}")
        if tb is not None:
            tb_f = float(tb)
            tokens.append(f"tb{int(tb_f) if tb_f.is_integer() else tb_f}")
    if "smoke" in dirname or int(cfg.get("max_timesteps", 0) or 0) < 100_000:
        if "smoke" in dirname or int(cfg.get("max_timesteps", 0) or 0) <= 20_000:
            tokens.append("smoke")
    t_lr = cfg.get("T_lr")
    if t_lr is not None and float(t_lr) not in (2e-4, 0.0002):
        # compact scientific-ish
        tokens.append("Tlr" + f"{float(t_lr):g}".replace(".", "p").replace("-", "m"))
    if family.startswith("pi_only") and not tokens:
        tokens.append("pi_only_xfit")
    if not tokens:
        tokens.append("default")
    # de-dup while preserving order
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return "_".join(out)


def settings_summary(algo: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "env",
        "seed",
        "max_timesteps",
        "eval_freq",
        "batch_size",
        "discount",
        "tau",
        "policy_freq",
        "policy_noise",
        "noise_clip",
        "normalize",
        "normalize_reward",
        "T",
        "T_E",
        "T_B",
        "T_lr",
        "T_freq",
        "proximal_n_steps",
        "dual_proximal",
        "adaptive_multiscale",
        "pi_bound_method",
        "pi_bound_segments",
        "smoothness_eps",
        "smoothness_max",
        "actor_lr",
        "alpha",
    ]
    out = {}
    for k in keys:
        if k in cfg and cfg[k] is not None:
            out[k] = cfg[k]
    return out


def discover_run_dirs(root: Path, nested: bool) -> List[Path]:
    if not root.exists():
        return []
    if nested:
        # parent/*/run_dir/config.yaml
        return sorted({p.parent for p in root.glob("*/*/config.yaml")})
    return sorted({p.parent for p in root.glob("*/config.yaml")})


def ingest_one(
    src: Path,
    algo: str,
    host: str,
    code_repo: str,
    family_force: Optional[str],
    dry_run: bool,
) -> Optional[Dict[str, Any]]:
    cfg_path = src / "config.yaml"
    if not cfg_path.exists():
        return None
    cfg = load_yaml_lite(cfg_path)
    env = str(cfg.get("env") or "unknown")
    seed = int(cfg.get("seed", 0) or 0)
    family = classify_family(algo, cfg, family_force)
    variant = build_variant(algo, family, cfg, src.name)
    uuid8 = extract_uuid8(src.name)
    short = env_short(env)
    run_id = f"{short}_s{seed}_{variant}__{uuid8}"
    dest = RUNS / algo / family / run_id

    artifacts = [f for f in KEEP_FILES if (src / f).exists()]
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
        "settings": settings_summary(algo, cfg),
        "artifacts": artifacts,
        "git": {"code_repo": code_repo, "code_commit": None},
    }

    if dry_run:
        print(f"DRY {src} -> {dest.relative_to(ROOT)}")
        return meta

    dest.mkdir(parents=True, exist_ok=True)
    for name in artifacts:
        shutil.copy2(src / name, dest / name)
    # optional tiny extras
    for extra in ("launch_cmd.txt", "notes.md"):
        if (src / extra).exists():
            shutil.copy2(src / extra, dest / extra)
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
        for run_dir in discover_run_dirs(root, bool(src_spec.get("nested"))):
            meta = ingest_one(
                run_dir,
                algo=src_spec["algo"],
                host=src_spec["host"],
                code_repo=src_spec["code_repo"],
                family_force=src_spec.get("family_force"),
                dry_run=args.dry_run,
            )
            if meta:
                collected.append(meta)

    print(f"# collected {len(collected)} runs")
    if args.rebuild_catalog and not args.dry_run:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_catalog import build

        build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
