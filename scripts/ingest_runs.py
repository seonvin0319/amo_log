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
from typing import Any, Dict, List, Optional


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
    "pen-cloned-v1": "pencl",
    "pen-human-v1": "penh",
    "pen-expert-v1": "pene",
    "door-cloned-v1": "doorcl",
    "door-human-v1": "doorh",
    "door-expert-v1": "doore",
    "hammer-cloned-v1": "hamcl",
    "hammer-human-v1": "hamh",
    "hammer-expert-v1": "hame",
    "relocate-cloned-v1": "relcl",
    "relocate-human-v1": "relh",
    "relocate-expert-v1": "rele",
}

# Host ext_csv only. Other machines keep their own branches/sources.
# Prefer /raid/ext_csv/AMO_store when that is the canonical store (home may symlink).
_LAMBDA0 = "30abcfcfbc62b892c6c0a7d0763c1c8323154d11"
_AMO = "c45671c47cfae89154331e9c9dab59b9cdbc9c40"
_AMO_MAIN = "3fdd895194b2ca8bb6c2777567f3deccd0b60822"
DEFAULT_SOURCES: List[Dict[str, Any]] = [
    {
        "algo": "amo",
        "root": Path(
            "/home/ext_csv/AMO-behavior-l1-joint-v2/results/behavior_l1_joint_suite6_s0"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-behavior-l1-joint-v2",
        "code_commit": "9710464f82c4db33b95ec9d6a68d7a822228e598",
        "family_force": "behavior_bc_l1_joint",
        "nested": False,
    },
    {
        "algo": "amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/amo_adaptive_multiscale_antmaze6_tlr_te1tb1_seed0/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO",
        "code_commit": _AMO,
        "family_force": "adaptive_multiscale",
        "nested": False,
    },
    {
        "algo": "amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/amo_adaptive_multiscale_antmaze6_tlr_te1tb1_1em3_seeds123/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO",
        "code_commit": _AMO,
        "family_force": "adaptive_multiscale",
        "nested": False,
    },
    {
        "algo": "amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/amo_adaptive_multiscale_antmaze6_scale_sweep_seed0/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO",
        "code_commit": _AMO,
        "family_force": "adaptive_multiscale",
        "nested": False,
    },
    {
        "algo": "amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/amo_adaptive_multiscale_antmaze6_te1_tb025_seed0/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-lambda0-te-tb",
        "code_commit": _LAMBDA0,
        "family_force": "adaptive_multiscale",
        "nested": False,
    },
    {
        "algo": "amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/amo_adaptive_multiscale_loco9_te1_tb025_seed0/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-lambda0-te-tb",
        "code_commit": _LAMBDA0,
        "family_force": "adaptive_multiscale",
        "nested": False,
    },
    {
        "algo": "amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/amo_adaptive_multiscale_loco9_te1_tb1_seeds1to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-lambda0-te-tb",
        "code_commit": _LAMBDA0,
        "family_force": "adaptive_multiscale",
        "nested": False,
    },
    {
        "algo": "amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/amo_adaptive_multiscale_loco9_te_tb_tlr_grid_seed0/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-lambda0-te-tb",
        "code_commit": _LAMBDA0,
        "family_force": "adaptive_multiscale",
        "nested": False,
    },
    {
        "algo": "amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/amo_adaptive_multiscale_loco9_tlr3e4_band95_seeds1to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-lambda0-te-tb",
        "code_commit": _LAMBDA0,
        "family_force": "adaptive_multiscale",
        "nested": False,
    },
    {
        "algo": "amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/amo_adaptive_multiscale_antmaze6_tb_bc_qimprove_1em3_seed0/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO",
        "code_commit": _AMO,
        "family_force": "adaptive_multiscale",
        "nested": False,
    },
    {
        "algo": "td3_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/td3_amo_jax_loco9_default_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-main",
        "code_commit": _AMO_MAIN,
        "family_force": "td3_amo_jax",
        "nested": False,
    },
]

CFG_NAMES = ("config.yaml", "effective_config.yaml")
EXTRA_FILES = ("train_done.json", "run_manifest.json", "launch_cmd.txt", "notes.md")


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
    return (
        env.replace("halfcheetah", "hc")
        .replace("hopper", "hop")
        .replace("walker2d", "w")
        .replace("antmaze", "am")
        .replace("relocate", "rel")
        .replace("hammer", "ham")
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
        .replace("cloned", "cl")
        .replace("human", "h")
        .replace("-v2", "")
        .replace("-v1", "")
        .replace("-", "")
    )


def extract_uuid8(dirname: str) -> str:
    m = re.search(r"([0-9a-fA-F]{8})$", dirname)
    if m:
        return m.group(1).lower()
    return hashlib.sha1(dirname.encode()).hexdigest()[:8]


def fmt_num(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def classify_family(algo: str, cfg: Dict[str, Any], force: Optional[str]) -> str:
    if force:
        return force
    if algo == "amo":
        algo_id = str(cfg.get("algorithm_id") or cfg.get("protocol_version") or "")
        if algo_id.startswith("behavior_bc_l1_joint"):
            return "behavior_bc_l1_joint"
        method = str(cfg.get("pi_bound_method", "secant"))
        if method == "segment_interval":
            return "segment_interval"
        if cfg.get("adaptive_multiscale"):
            return "adaptive_multiscale"
        return "secant"
    if cfg.get("adaptive_multiscale"):
        return "adaptive_multiscale"
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


def build_variant(
    algo: str,
    family: str,
    cfg: Dict[str, Any],
    dirname: str,
    source_root: Optional[Path] = None,
) -> str:
    tokens: List[str] = []
    root_name = source_root.name if source_root is not None else ""
    blob = f"{dirname} {root_name}"
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

    if family == "behavior_bc_l1_joint":
        tokens.append("joint")
        te = cfg.get("T_init", cfg.get("T_E", cfg.get("T")))
        if te is not None:
            tokens.append(f"te{fmt_num(float(te))}")
        tokens.append("l1e")
        max_t = int(cfg.get("max_timesteps") or cfg.get("max_steps") or 0)
        if max_t >= 1_000_000:
            tokens.append("1m")

    if family == "adaptive_multiscale":
        te = cfg.get("T_E", cfg.get("T"))
        tb = cfg.get("T_B", te)
        if te is not None:
            tokens.append(f"te{fmt_num(float(te))}")
        if tb is not None:
            tokens.append(f"tb{fmt_num(float(tb))}")
        if cfg.get("execution_l1") is True or "l1e" in blob:
            tokens.append("l1e")
        crit_h = cfg.get("critic_n_hiddens")
        crit_ln = cfg.get("critic_layernorm")
        if crit_h == 2 and crit_ln is False:
            tokens.append("td3bc")
        if cfg.get("normalize_q") is False or "qraw" in blob:
            tokens.append("qraw")
        if "unconstrained" in blob or "unconst" in blob:
            tokens.append("unconst")
        if "bootstrap" in blob and "adaptive_bootstrap" in root_name:
            tokens.append("boot")

    if family in ("dual_proximal", "chain", "misc"):
        t_init = cfg.get("T")
        if t_init is not None and family != "adaptive_multiscale":
            tokens.append(f"T{fmt_num(float(t_init))}")

    max_t = int(cfg.get("max_timesteps") or cfg.get("max_steps") or 0)
    if "smoke" in blob or (0 < max_t <= 20_000):
        tokens.append("smoke")

    t_lr = cfg.get("T_lr", cfg.get("rho_lr"))
    if t_lr is not None and float(t_lr) not in (2e-4, 0.0002):
        tokens.append("Tlr" + fmt_num(float(t_lr)))

    if family.startswith("pi_only") and not tokens:
        tokens.append("pi_only_xfit")
    if not tokens:
        tokens.append("default")

    seen = set()
    out = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            out.append(token)
    return "_".join(out)


def settings_summary(algo: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "env",
        "seed",
        "max_timesteps",
        "max_steps",
        "algorithm_id",
        "eval_freq",
        "batch_size",
        "discount",
        "tau",
        "policy_freq",
        "policy_noise",
        "noise_clip",
        "normalize",
        "normalize_reward",
        "normalize_q",
        "T",
        "T_E",
        "T_B",
        "T_init",
        "T_lr",
        "rho_lr",
        "T_freq",
        "meta_warmup_steps",
        "eval_policy_id",
        "proximal_n_steps",
        "dual_proximal",
        "adaptive_multiscale",
        "execution_l1",
        "pi_bound_method",
        "pi_bound_segments",
        "smoothness_eps",
        "smoothness_max",
        "actor_lr",
        "alpha",
        "critic_n_hiddens",
        "critic_hidden_dim",
        "critic_layernorm",
    ]
    out = {}
    for key in keys:
        if key in cfg and cfg[key] is not None:
            out[key] = cfg[key]
    if "max_timesteps" not in out and out.get("max_steps") is not None:
        out["max_timesteps"] = out["max_steps"]
    return out


def discover_run_dirs(root: Path, nested: bool) -> List[Path]:
    if not root.exists():
        return []
    found = set()
    prefixes = ("*/*/",) if nested else ("*/",)
    for prefix in prefixes:
        for name in CFG_NAMES:
            found.update(p.parent for p in root.glob(prefix + name))
    return sorted(found)


def ingest_one(
    src: Path,
    algo: str,
    host: str,
    code_repo: str,
    family_force: Optional[str],
    dry_run: bool,
    source_root: Optional[Path] = None,
    code_commit: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    cfg_path = None
    cfg_src_name = None
    for name in CFG_NAMES:
        candidate = src / name
        if candidate.exists():
            cfg_path = candidate
            cfg_src_name = name
            break
    if cfg_path is None:
        return None
    cfg = load_yaml_lite(cfg_path)
    env = str(cfg.get("env") or "unknown")
    seed = int(cfg.get("seed", 0) or 0)
    family = classify_family(algo, cfg, family_force)
    # Adaptive-multiscale always archives under amo/, even if code lived in APART/.
    if family == "adaptive_multiscale":
        algo = "amo"
    variant = build_variant(algo, family, cfg, src.name, source_root=source_root)
    uuid8 = extract_uuid8(src.name)
    short = env_short(env)
    run_id = f"{short}_s{seed}_{variant}__{uuid8}"
    dest = RUNS / algo / family / run_id

    artifacts = ["config.yaml"]
    for name in KEEP_FILES:
        if name == "config.yaml":
            continue
        if (src / name).exists():
            artifacts.append(name)
    extras = [name for name in EXTRA_FILES if (src / name).exists()]
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
        "source_config": cfg_src_name,
        "checkpoint_hint": str(src.resolve()),
        "collected_at": datetime.now(timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds"),
        "settings": settings_summary(algo, cfg),
        "artifacts": artifacts + extras,
        "git": {"code_repo": code_repo, "code_commit": code_commit},
    }

    if dry_run:
        print(f"DRY {src} -> {dest.relative_to(ROOT)}")
        return meta

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, dest / "config.yaml")
    for name in artifacts:
        if name == "config.yaml":
            continue
        shutil.copy2(src / name, dest / name)
    for extra in extras:
        shutil.copy2(src / extra, dest / extra)
    (dest / "run_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )
    print(f"OK  {dest.relative_to(ROOT)}")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-catalog", action="store_true", default=True)
    parser.add_argument(
        "--host",
        default=None,
        help="Only ingest sources for this source_host (e.g. ext_csv).",
    )
    args = parser.parse_args()

    collected: List[Dict[str, Any]] = []
    for src_spec in DEFAULT_SOURCES:
        if args.host and src_spec.get("host") != args.host:
            continue
        root: Path = src_spec["root"]
        for run_dir in discover_run_dirs(root, bool(src_spec.get("nested"))):
            meta = ingest_one(
                run_dir,
                algo=src_spec["algo"],
                host=src_spec["host"],
                code_repo=src_spec["code_repo"],
                family_force=src_spec.get("family_force"),
                dry_run=args.dry_run,
                source_root=root,
                code_commit=src_spec.get("code_commit"),
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
