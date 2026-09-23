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

KEEP_FILES = ("config.yaml", "metrics.jsonl", "eval.jsonl", "eval_final50_v1.jsonl")

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
_LAMBDA0_GAPFILL = "eee3d486fac0c9f5ecbbea411424e6691b715265"
_AMO = "c45671c47cfae89154331e9c9dab59b9cdbc9c40"
_AMO_MAIN = "b9338d9815525482d2cf34d6fc6315ea4d2f93a6"
# AMO-main after T→alpha rename (alpha := 2T).
_AMO_MAIN_ALPHA = "1e34514ddf70bfc8a78757d9a78b82306627164c"
_AMO_FQL = "3f4401279175f33858c4fbf5993bd1fc2f3dfc84"
_AMO_EXECONLY = "4e3203f35bcc6048201ae3cfe75d932830ccd5f9"
DEFAULT_SOURCES: List[Dict[str, Any]] = [
    {
        "algo": "amo",
        "root": Path(
            "/home/ext_csv/AMO_results/behavior_l1_joint/behavior_l1_joint_suite6_s0"
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
        "algo": "amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/amo_adaptive_multiscale_antmaze6_te_tb_tlr_grid_seed0/runs"
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
            "/raid/ext_csv/AMO_store/amo_te1_tb1_tlr_gapfill_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-lambda0-te-tb",
        "code_commit": _LAMBDA0_GAPFILL,
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
    {
        "algo": "iql_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/iql_amo_jax_adroit_beta1_rho_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-main",
        "code_commit": _AMO_MAIN,
        "family_force": "iql_amo_jax_adroit_beta1_rho",
        "nested": False,
    },
    {
        "algo": "td3_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/td3_amo_jax_loco_antmaze_alpha1_alr_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-main",
        "code_commit": _AMO_MAIN_ALPHA,
        "family_force": "td3_amo_jax",
        "nested": False,
    },
    {
        "algo": "td3_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/td3_amo_jax_loco9_alpha2_alr1em3_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-main",
        "code_commit": _AMO_MAIN_ALPHA,
        "family_force": "td3_amo_jax",
        "nested": False,
    },
    {
        "algo": "iql_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/iql_amo_jax_antmaze_beta5_rho2em3_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-main",
        "code_commit": _AMO_MAIN,
        "family_force": "iql_amo_jax_antmaze_beta5_rho",
        "nested": False,
    },
    {
        "algo": "iql_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/iql_amo_jax_antmaze_beta5_rho_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-main",
        "code_commit": _AMO_MAIN,
        "family_force": "iql_amo_jax_antmaze_beta5_rho",
        "nested": False,
    },
    {
        "algo": "iql_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/iql_amo_jax_loco_beta1_rho2em3_wm_wmr_s3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-main",
        "code_commit": _AMO_MAIN,
        "family_force": "iql_amo_jax_loco_beta1_rho",
        "nested": False,
    },
    {
        "algo": "iql_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/iql_amo_qweight_jax_beta125_rho_loco_antmaze_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-main",
        "code_commit": "ceffa5373681e09b20d5b513091bccd4d301a4eb",
        "family_force": "iql_amo_qweight_jax",
        "nested": False,
    },
    {
        "algo": "fql_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/fql_amo_jax_loco9_tinit5_alr3e-4_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-fql",
        "code_commit": _AMO_FQL,
        "family_force": "fql_amo_jax",
        "nested": False,
    },
    {
        "algo": "fql_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/fql_amo_jax_loco9_tinit5_alr1e-3_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-fql",
        "code_commit": _AMO_FQL,
        "family_force": "fql_amo_jax",
        "nested": False,
    },
    {
        "algo": "fql_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/fql_amo_jax_loco9_tinit5_alr2e-3_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-fql",
        "code_commit": _AMO_FQL,
        "family_force": "fql_amo_jax",
        "nested": False,
    },
    {
        "algo": "fql_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/fql_rapo_main_a5_3env_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "fql",
        "code_commit": "e8cd16eb490332924dfa2492097219f181765933",
        "family_force": "fql_amo_jax",
        "nested": False,
    },
    {
        "algo": "td3_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/td3_amo_execonly_a5_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-execonly",
        "code_commit": _AMO_EXECONLY,
        "family_force": "td3_amo_execonly_main",
        "nested": False,
    },
    {
        "algo": "td3_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/td3_amo_bootrms_a5_cmp4_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-execonly",
        "code_commit": _AMO_EXECONLY,
        "family_force": "td3_amo_bootrms_cmp4",
        "nested": False,
    },
    {
        "algo": "td3_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/td3_amo_el2_bootrms_a5_cmp4_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-execonly",
        "code_commit": _AMO_EXECONLY,
        "family_force": "td3_amo_el2_bootrms_cmp4",
        "nested": False,
    },
    {
        "algo": "td3_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/td3_amo_le_bel2_a5_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-execonly",
        "code_commit": _AMO_EXECONLY,
        "family_force": "td3_amo_le_bel2_a5",
        "nested": False,
    },
    {
        "algo": "iql_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/iql_amo_lel2_b5_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-execonly",
        "code_commit": _AMO_EXECONLY,
        "family_force": "iql_amo_lel2",
        "nested": False,
    },
    {
        "algo": "td3_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/td3_amo_fixed_alpha_B5_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-fixed-alpha",
        "code_commit": None,
        "family_force": "td3_amo_fixed_alpha_B",
        "nested": False,
    },
    {
        "algo": "td3_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/td3_amo_fixed_alpha_B1_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-fixed-alpha",
        "code_commit": None,
        "family_force": "td3_amo_fixed_alpha_B",
        "nested": False,
    },
    {
        "algo": "iql_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/iql_ddpgbc_fixed_alpha_E5_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-fixed-alpha",
        "code_commit": None,
        "family_force": "iql_ddpgbc_fixed_alpha_E",
        "nested": False,
    },
    {
        "algo": "iql_amo",
        "root": Path(
            "/raid/ext_csv/AMO_store/iql_ddpgbc_fixed_alpha_E1_seeds0to3/runs"
        ),
        "host": "ext_csv",
        "code_repo": "AMO-fixed-alpha",
        "code_commit": None,
        "family_force": "iql_ddpgbc_fixed_alpha_E",
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


def load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(errors="ignore"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_checkpoint_metadata(src: Path) -> Dict[str, Any]:
    """Read __metadata__ from the best available checkpoint without mutating it."""
    candidates: List[Path] = []
    final = src / "checkpoints" / "step_1000000.npz"
    if final.exists():
        candidates.append(final)
    ckpt_dir = src / "checkpoints"
    if ckpt_dir.is_dir():
        stepped = []
        for path in ckpt_dir.iterdir():
            match = re.match(r"^step_(\d+)\.npz$", path.name)
            if match:
                stepped.append((int(match.group(1)), path))
        stepped.sort(key=lambda item: item[0], reverse=True)
        candidates.extend(path for _, path in stepped)
    root_ckpt = src / "checkpoint.npz"
    if root_ckpt.exists():
        candidates.append(root_ckpt)
    seen = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            import numpy as np  # local import: ingest may run without numpy

            with np.load(path, allow_pickle=False) as data:
                if "__metadata__" not in data:
                    continue
                meta = json.loads(str(data["__metadata__"]))
            if isinstance(meta, dict):
                meta = dict(meta)
                meta["_checkpoint_path"] = str(path)
                return meta
        except Exception:
            continue
    return {}


def infer_env_seed_from_dirname(dirname: str) -> Dict[str, Any]:
    """Best-effort dirname parse. Never treat as verified truth alone."""
    out: Dict[str, Any] = {}
    seed_m = re.search(r"_s(\d+)(?:_|$)", dirname)
    if seed_m:
        out["seed"] = int(seed_m.group(1))
    # td3amo_jax_w-mr_s0 / iqlamo_jax_door-cln_beta1_rho3em4_s2
    env_map = {
        "hcm": "halfcheetah-medium-v2",
        "hc-m": "halfcheetah-medium-v2",
        "hcmr": "halfcheetah-medium-replay-v2",
        "hc-mr": "halfcheetah-medium-replay-v2",
        "hcme": "halfcheetah-medium-expert-v2",
        "hc-me": "halfcheetah-medium-expert-v2",
        "hopm": "hopper-medium-v2",
        "h-m": "hopper-medium-v2",
        "hopmr": "hopper-medium-replay-v2",
        "h-mr": "hopper-medium-replay-v2",
        "hopme": "hopper-medium-expert-v2",
        "h-me": "hopper-medium-expert-v2",
        "wm": "walker2d-medium-v2",
        "w-m": "walker2d-medium-v2",
        "wmr": "walker2d-medium-replay-v2",
        "w-mr": "walker2d-medium-replay-v2",
        "wme": "walker2d-medium-expert-v2",
        "w-me": "walker2d-medium-expert-v2",
        "door-hum": "door-human-v1",
        "door-cln": "door-cloned-v1",
        "door-exp": "door-expert-v1",
        "hammer-hum": "hammer-human-v1",
        "hammer-cln": "hammer-cloned-v1",
        "hammer-exp": "hammer-expert-v1",
        "pen-hum": "pen-human-v1",
        "pen-cln": "pen-cloned-v1",
        "pen-exp": "pen-expert-v1",
        "relocate-hum": "relocate-human-v1",
        "relocate-cln": "relocate-cloned-v1",
        "relocate-exp": "relocate-expert-v1",
        "am-u": "antmaze-umaze-v2",
        "am-m-p": "antmaze-medium-play-v2",
        "am-l-p": "antmaze-large-play-v2",
        "am-u-div": "antmaze-umaze-diverse-v2",
        "am-m-div": "antmaze-medium-diverse-v2",
        "am-l-div": "antmaze-large-diverse-v2",
    }
    for key, env in env_map.items():
        if f"_{key}_" in f"_{dirname}_" or f"_{key}_s" in dirname or dirname.endswith(f"_{key}"):
            # prefer more specific keys already ordered loosely by length in dict insertion
            if "env" not in out or len(key) > 3:
                out["env"] = env
    # more reliable: look for explicit tokens in iql / td3 names
    m = re.search(
        r"(?:td3amo_jax_|iqlamo_jax_)([a-z0-9-]+?)(?:_beta1|_s\d|$)",
        dirname,
    )
    if m:
        token = m.group(1)
        if token in env_map:
            out["env"] = env_map[token]
    return out


def resolve_env_seed(
    src: Path, cfg: Dict[str, Any]
) -> tuple[str, int, Dict[str, Any], Dict[str, Any]]:
    """Merge env/seed: source run_meta → checkpoint extra → config → dirname (inferred)."""
    source_run_meta = load_json_dict(src / "run_meta.json")
    ckpt_meta = load_checkpoint_metadata(src)
    extra = ckpt_meta.get("extra") if isinstance(ckpt_meta.get("extra"), dict) else {}
    provenance: Dict[str, Any] = {
        "env_source": None,
        "seed_source": None,
        "inferred": False,
        "verified_against_checkpoint": False,
        "checkpoint_path": ckpt_meta.get("_checkpoint_path"),
    }
    env: Optional[str] = None
    seed: Optional[int] = None

    def take_env(value: Any, source: str) -> None:
        nonlocal env
        if env is None and value not in (None, "", "unknown"):
            env = str(value)
            provenance["env_source"] = source

    def take_seed(value: Any, source: str) -> None:
        nonlocal seed
        if seed is None and value is not None and str(value) != "":
            try:
                seed = int(value)
                provenance["seed_source"] = source
            except (TypeError, ValueError):
                pass

    take_env(source_run_meta.get("env"), "source_run_meta.json")
    take_seed(source_run_meta.get("seed"), "source_run_meta.json")
    take_env(extra.get("env"), "checkpoint.extra")
    take_seed(extra.get("seed"), "checkpoint.extra")
    take_env(ckpt_meta.get("env"), "checkpoint.meta")
    take_seed(ckpt_meta.get("seed"), "checkpoint.meta")
    take_env(cfg.get("env"), "config.yaml")
    take_seed(cfg.get("seed"), "config.yaml")

    inferred = infer_env_seed_from_dirname(src.name)
    if env is None and inferred.get("env"):
        env = str(inferred["env"])
        provenance["env_source"] = "dirname_inferred"
        provenance["inferred"] = True
    if seed is None and inferred.get("seed") is not None:
        seed = int(inferred["seed"])
        provenance["seed_source"] = "dirname_inferred"
        provenance["inferred"] = True

    # If dirname was used, require checkpoint extra agreement when available.
    if provenance["inferred"] and extra.get("env") and env and str(extra["env"]) != env:
        env = str(extra["env"])
        provenance["env_source"] = "checkpoint.extra_overrides_inferred"
        provenance["inferred"] = False
    if provenance["inferred"] and extra.get("seed") is not None and seed is not None:
        if int(extra["seed"]) != int(seed):
            seed = int(extra["seed"])
            provenance["seed_source"] = "checkpoint.extra_overrides_inferred"
            provenance["inferred"] = False
    if extra.get("env") and env and str(extra["env"]) == env:
        if extra.get("seed") is None or seed is None or int(extra["seed"]) == int(seed):
            provenance["verified_against_checkpoint"] = True
    if source_run_meta.get("env") and env and str(source_run_meta["env"]) == env:
        if source_run_meta.get("seed") is None or seed is None or int(source_run_meta["seed"]) == int(seed):
            provenance["verified_against_checkpoint"] = (
                provenance["verified_against_checkpoint"] or bool(extra)
            )

    if env is None:
        env = "unknown"
    if seed is None:
        seed = 0
    return env, int(seed), provenance, source_run_meta


def remap_existing_by_uuid(
    family_dir: Path, run_id: str, uuid8: str, dry_run: bool
) -> Optional[Path]:
    """Preserve identity: move unknown_*__uuid8 → corrected run_id if needed."""
    if not family_dir.is_dir():
        return None
    matches = sorted(family_dir.glob(f"*__{uuid8}"))
    if not matches:
        return None
    dest = family_dir / run_id
    # Prefer exact dest if already present.
    if dest.exists():
        for old in matches:
            if old.resolve() == dest.resolve():
                continue
            if dry_run:
                print(f"DRY remove-dup {old} (keep {dest})")
            else:
                shutil.rmtree(old, ignore_errors=True)
        return dest
    primary = matches[0]
    if primary.name == run_id:
        return primary
    if dry_run:
        print(f"DRY remap {primary.name} -> {run_id}")
        return dest
    primary.rename(dest)
    for old in matches[1:]:
        if old.exists() and old.resolve() != dest.resolve():
            shutil.rmtree(old, ignore_errors=True)
    return dest


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

    if family in ("td3_amo_jax", "fql_amo_jax"):
        # Prefer alpha_* when present (AMO-main rename); else legacy T_*.
        if cfg.get("alpha_E") is not None or cfg.get("alpha_B") is not None:
            ae = cfg.get("alpha_E")
            ab = cfg.get("alpha_B", ae)
            if ae is not None:
                tokens.append(f"ae{fmt_num(float(ae))}")
            if ab is not None:
                tokens.append(f"ab{fmt_num(float(ab))}")
        else:
            te = cfg.get("T_E", cfg.get("T"))
            tb = cfg.get("T_B", te)
            if te is not None:
                tokens.append(f"te{fmt_num(float(te))}")
            if tb is not None:
                tokens.append(f"tb{fmt_num(float(tb))}")
    if family == "fql_amo_jax":
        alr = cfg.get("alpha_lr", cfg.get("T_lr"))
        if alr is not None:
            tokens.append("alr" + fmt_num(float(alr)))
    if family == "iql_amo_qweight_jax":
        beta = cfg.get("beta_initial")
        if beta is not None:
            tokens.append("beta" + fmt_num(float(beta)))

    if family in ("dual_proximal", "chain", "misc"):
        t_init = cfg.get("T")
        if t_init is not None and family != "adaptive_multiscale":
            tokens.append(f"T{fmt_num(float(t_init))}")

    max_t = int(cfg.get("max_timesteps") or cfg.get("max_steps") or 0)
    if "smoke" in blob or (0 < max_t <= 20_000):
        tokens.append("smoke")

    t_lr = cfg.get("T_lr", cfg.get("rho_lr"))
    if family != "fql_amo_jax" and t_lr is not None and float(t_lr) not in (2e-4, 0.0002):
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


def alias_alpha_to_legacy_t(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Keep alpha_* as source of truth; only fill T_lr when missing (shared layout reads alpha_*)."""
    out = dict(cfg)
    # Do not invent T_E=alpha/2 — that mislabels run_ids as te0.5 for alpha_E=1.
    if out.get("alpha_lr") is not None and out.get("T_lr") is None:
        out["T_lr"] = out["alpha_lr"]
    return out


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
        "alpha_E",
        "alpha_B",
        "alpha_lr",
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
        "beta_initial",
        "rho_lr",
        "expectile",
        "algorithm",
        "backend",
        "freeze_scale_B",
        "freeze_scale_E",
        "bootstrap_loss",
        "execution_only",
        "execution_meta_loss",
        "execution_score",
        "const_std",
        "gaussian",
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
    env, seed, provenance, source_run_meta = resolve_env_seed(src, cfg)
    if not code_commit:
        code_commit = source_run_meta.get("code_commit")
    family = classify_family(algo, cfg, family_force)
    # Adaptive-multiscale always archives under amo/, even if code lived in APART/.
    if family == "adaptive_multiscale":
        algo = "amo"
    # Merge recovered env/seed into settings view without mutating source config.yaml.
    # alpha_* → T_* aliases so shared log_layout (T_E=T_B=1 main) still classifies.
    cfg_view = alias_alpha_to_legacy_t(cfg)
    cfg_view["env"] = env
    cfg_view["seed"] = seed
    if source_run_meta.get("algorithm") and "algorithm" not in cfg_view:
        cfg_view["algorithm"] = source_run_meta.get("algorithm")
    if source_run_meta.get("backend") and "backend" not in cfg_view:
        cfg_view["backend"] = source_run_meta.get("backend")
    variant = build_variant(algo, family, cfg_view, src.name, source_root=source_root)
    uuid8 = extract_uuid8(src.name)
    short = env_short(env)
    run_id = f"{short}_s{seed}_{variant}__{uuid8}"
    family_dir = RUNS / algo / family
    remapped = remap_existing_by_uuid(family_dir, run_id, uuid8, dry_run=dry_run)
    dest = remapped if remapped is not None else family_dir / run_id

    artifacts = ["config.yaml"]
    for name in KEEP_FILES:
        if name == "config.yaml":
            continue
        if (src / name).exists():
            artifacts.append(name)
    extras = [name for name in EXTRA_FILES if (src / name).exists()]
    if source_run_meta:
        artifacts.append("source_run_meta.json")
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
        "settings": settings_summary(algo, cfg_view),
        "artifacts": artifacts + extras,
        "git": {"code_repo": code_repo, "code_commit": code_commit},
        "metadata_provenance": provenance,
        "identity_uuid8": uuid8,
    }

    if dry_run:
        print(f"DRY {src} -> {dest.relative_to(ROOT)} env={env} seed={seed}")
        return meta

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, dest / "config.yaml")
    for name in artifacts:
        if name in ("config.yaml", "source_run_meta.json"):
            continue
        shutil.copy2(src / name, dest / name)
    for extra in extras:
        shutil.copy2(src / extra, dest / extra)
    if source_run_meta:
        (dest / "source_run_meta.json").write_text(
            json.dumps(source_run_meta, indent=2, sort_keys=True) + "\n"
        )
    (dest / "run_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )
    print(f"OK  {dest.relative_to(ROOT)} env={env} seed={seed}")
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
    if not args.dry_run:
        from log_layout import normalize
        normalize(ROOT)

    if args.rebuild_catalog and not args.dry_run:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_catalog import build

        build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
