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

KEEP_FILES = ("config.yaml", "metrics.jsonl", "eval.jsonl", "final_eval_50.jsonl")

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
    "door-human-v1": "dh",
    "door-cloned-v1": "dc",
    "door-expert-v1": "de",
    "hammer-human-v1": "hh",
    "hammer-cloned-v1": "hc",
    "hammer-expert-v1": "he",
    "pen-human-v1": "ph",
    "pen-cloned-v1": "pc",
    "pen-expert-v1": "pe",
    "relocate-human-v1": "rh",
    "relocate-cloned-v1": "rc",
    "relocate-expert-v1": "re",
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
    # JAX AMO on iisl-server04 / shchoi (config.json; no APART tree on this host)
    {
        "algo": "amo",
        "root": Path("/home/shchoi/AMO/results_amo_pilot"),
        "host": "shchoi",
        "code_repo": "AMO",
        "kind": "jax",
        "nested": True,
    },
    {
        "algo": "amo",
        "root": Path("/home/shchoi/AMO/results_amo_v2_pilot"),
        "host": "shchoi",
        "code_repo": "AMO",
        "kind": "jax",
        "nested": True,
    },
    {
        "algo": "amo",
        "root": Path("/home/shchoi/AMO/results_amo_v3_pilot"),
        "host": "shchoi",
        "code_repo": "AMO",
        "kind": "jax",
        "nested": True,
    },
    {
        "algo": "amo",
        "root": Path("/home/shchoi/AMO/results_amo_v2_1m"),
        "host": "shchoi",
        "code_repo": "AMO",
        "kind": "jax",
        "nested": True,
    },
    {
        "algo": "amo",
        "root": Path(
            "/home/shchoi/amo_td3bc/results/amo_td3bc_locomotion9_seed0/runs"
        ),
        "host": "shchoi",
        "code_repo": "AMO",
        "family_force": "amo_td3bc",
    },
    {
        "algo": "amo",
        "root": Path(
            "/home/shchoi/amo_lambda0_fork_diag/results/"
            "rebrac_amo_locomotion9_seed0/runs"
        ),
        "host": "shchoi",
        "code_repo": "AMO",
        "family_force": "rebrac_amo",
    },
    {
        "algo": "amo",
        "root": Path(
            "/home/shchoi/amo/results/"
            "amo_adaptive_multiscale_antmaze6_sweep_seed0/runs"
        ),
        "host": "shchoi",
        "code_repo": "amo",
        "family_force": "adaptive_multiscale",
    },
    {
        "algo": "amo",
        "root": Path(
            "/home/shchoi/amo/results/"
            "amo_adaptive_multiscale_antmaze6_sweep_seeds1-2-3/runs"
        ),
        "host": "shchoi",
        "code_repo": "amo",
        "family_force": "adaptive_multiscale",
    },
    # JAX AMS AntMaze-6 fill (config.yaml + run_meta.json; final in posthoc_eval_cpu/)
    {
        "algo": "amo",
        "root": Path(
            "/home/shchoi/amo/results/"
            "amo_jax_ams_antmaze6_sweep_seeds1-2-3/runs"
        ),
        "host": "shchoi",
        "code_repo": "AMO-jax-upstream",
        "family_force": "adaptive_multiscale",
    },
    # JAX TD3+AMO Adroit-12 · T=1 init · T_lr=2e-3
    {
        "algo": "amo",
        "root": Path(
            "/home/shchoi/amo/results/"
            "amo_jax_td3amo_adroit_t1_tlr2e3/runs"
        ),
        "host": "shchoi",
        "code_repo": "AMO-jax-upstream",
        "family_force": "adaptive_multiscale",
    },
    # IQL+AMO adaptive-beta (AMO_EXP iql-amo) · shchoi 69-job matrix
    {
        "algo": "iql",
        "root": Path(
            "/home/shchoi/AMO_EXP_iql-amo/results/iql_amo_shchoi69/runs"
        ),
        "host": "shchoi",
        "code_repo": "AMO_EXP_iql-amo",
        "family_force": "iql_adaptive_beta",
    },
    # IQL+AMO BPI main · JAX · beta_initial=1 · loco9 (AMO release)
    {
        "algo": "iql_amo",
        "root": Path("/home/shchoi/AMO/results/iql_amo_jax_main_loco9/runs"),
        "host": "shchoi",
        "code_repo": "AMO",
        "family_force": "amo_bpi",
    },
    # IQL+AMO BPI ablation · JAX · beta_initial=5 · loco9
    {
        "algo": "iql_amo",
        "root": Path("/home/shchoi/AMO/results/iql_amo_jax_b5_loco9/runs"),
        "host": "shchoi",
        "code_repo": "AMO",
        "family_force": "amo_bpi",
    },
    # ASPC D4RL benchmark (ASPC_WPC_FULL phase 1) on iisl-server04
    {
        "algo": "aspc",
        "root": Path("/home/shchoi/ASPC/results_aspc"),
        "host": "shchoi",
        "code_repo": "ASPC",
        "family_force": "benchmark",
        "dirname_contains": "_aspc-",
    },
    # WPC baseline (same ASPC_WPC_FULL sweep); scores recovered from _logs/
    {
        "algo": "wpc",
        "root": Path("/home/shchoi/ASPC/results_wpc"),
        "host": "shchoi",
        "code_repo": "ASPC",
        "family_force": "benchmark",
        "dirname_contains": "_wpc-",
    },
    # ASPC-repo L3 / multiscale / ablation runs (aspc.py modes)
    {
        "algo": "aspc",
        "root": Path("/home/shchoi/ASPC/results_pi_l3"),
        "host": "shchoi",
        "code_repo": "ASPC",
        "family_force": "benchmark",
    },
    {
        "algo": "aspc",
        "root": Path("/home/shchoi/ASPC/results_multiscale"),
        "host": "shchoi",
        "code_repo": "ASPC",
        "family_force": "benchmark",
        "dirname_contains": "_multiscale-",
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


def load_json_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(errors="ignore")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        patched = (
            text.replace(": Infinity", ": null")
            .replace(":-Infinity", ": null")
            .replace(": NaN", ": null")
        )
        return json.loads(patched)


def dump_yaml(obj: Any, indent: int = 0) -> str:
    """Write a JSON-serializable object as indentation-based YAML."""
    sp = "  " * indent
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, (int, float)):
        return repr(obj) if isinstance(obj, float) else str(obj)
    if isinstance(obj, str):
        if obj == "" or any(c in obj for c in ":#{}[]&*!|>'\"%@`\n"):
            return json.dumps(obj, ensure_ascii=False)
        return obj
    if isinstance(obj, list):
        if not obj:
            return "[]"
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                nested = dump_yaml(item, indent + 1)
                if "\n" in nested:
                    lines.append(f"{sp}-")
                    for nline in nested.splitlines():
                        lines.append(nline if nline.startswith(" ") else ("  " * (indent + 1) + nline))
                else:
                    lines.append(f"{sp}- {nested}")
            else:
                lines.append(f"{sp}- {dump_yaml(item, indent + 1)}")
        return "\n".join(lines)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines = []
        for k, v in obj.items():
            key = str(k)
            if isinstance(v, dict):
                if not v:
                    lines.append(f"{sp}{key}: {{}}")
                else:
                    lines.append(f"{sp}{key}:")
                    lines.append(dump_yaml(v, indent + 1))
            elif isinstance(v, list):
                if not v:
                    lines.append(f"{sp}{key}: []")
                else:
                    lines.append(f"{sp}{key}:")
                    lines.append(dump_yaml(v, indent + 1))
            else:
                lines.append(f"{sp}{key}: {dump_yaml(v, indent + 1)}")
        return "\n".join(lines)
    return json.dumps(obj, ensure_ascii=False)


def last_metrics_step(metrics_path: Path) -> Optional[int]:
    if not metrics_path.exists() or not metrics_path.stat().st_size:
        return None
    last = None
    for line in metrics_path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in obj:
            try:
                last = int(obj["step"])
            except (TypeError, ValueError):
                pass
    return last


def eval_score_from_row(row: Dict[str, Any]) -> Optional[float]:
    for key in (
        "eval/amo_v2_score",
        "teacher_d4rl_score",
        "d4rl_normalized_score",
        "d4rl_score",
        "student_d4rl_score",
    ):
        if key in row and row[key] is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return None


def write_eval_jsonl(metrics_path: Path, dest: Path) -> bool:
    if not metrics_path.exists():
        return False
    rows = []
    for line in metrics_path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        score = eval_score_from_row(obj)
        if score is None or "step" not in obj:
            continue
        rec: Dict[str, Any] = {
            "step": int(obj["step"]),
            "d4rl_normalized_score": score,
        }
        if "return_mean" in obj:
            rec["eval_return"] = obj["return_mean"]
        if "eval/base_score" in obj:
            rec["d4rl_normalized_score_base"] = obj["eval/base_score"]
        if "eval/amo_v2_score" in obj:
            rec["d4rl_normalized_score_refined"] = obj["eval/amo_v2_score"]
        elif "teacher_d4rl_score" in obj:
            rec["d4rl_normalized_score_refined"] = obj["teacher_d4rl_score"]
        rows.append(rec)
    if not rows:
        return False
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return True


def classify_family(algo: str, cfg: Dict[str, Any], force: Optional[str]) -> str:
    if force:
        return force
    if algo in ("aspc", "wpc"):
        return "benchmark"
    if algo == "amo":
        if cfg.get("use_amo_v3b"):
            return "jax_v3b"
        if cfg.get("use_amo_v3a"):
            return "jax_v3a"
        if cfg.get("use_amo_v2"):
            return "jax_v2"
        if cfg.get("use_amo"):
            return "jax_v1"
        if cfg.get("algorithm") == "td3_bc" and "amo_v2_t_init" in cfg:
            return "jax_td3bc"
        method = str(cfg.get("pi_bound_method", "secant"))
        if method == "segment_interval":
            return "segment_interval"
        if cfg.get("rebrac_critic_bc"):
            return "rebrac_amo"
        name = str(cfg.get("name", ""))
        if cfg.get("adaptive_multiscale") and "td3bc" in name:
            return "amo_td3bc"
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
    if family == "jax_v1":
        tokens.append(f"n{int(cfg.get('amo_n_steps', 1) or 1)}")
    if family == "jax_v2":
        if cfg.get("amo_v2_adapt_t"):
            tokens.append("adaptT")
        else:
            t_init = float(cfg.get("amo_v2_t_init", 0) or 0)
            frac = f"{t_init:.2f}".split(".")[-1]
            tokens.append(f"T0{frac}" if t_init < 1 else f"T{int(t_init)}")
        inner = int(cfg.get("amo_v2_inner_steps", 1) or 1)
        if inner != 1:
            tokens.append(f"inner{inner}")
    if family == "jax_v3a":
        tokens.append("v3a")
    if family == "jax_v3b":
        tokens.append("v3b")
    if family == "jax_td3bc":
        tokens.append("td3bc")
    if family == "rebrac_amo":
        tokens.append("rebrac")
    if family == "amo_td3bc":
        tokens.append("qouter")
    if family == "iql_adaptive_beta":
        rho = cfg.get("rho_lr", cfg.get("T_lr"))
        if rho is not None:
            tokens.append(
                "rho"
                + f"{float(rho):g}".replace(".", "p").replace("-", "m").replace("+", "")
            )
        else:
            tokens.append("adaptive_beta")
    if family == "amo_bpi":
        beta = cfg.get("beta_initial")
        if beta is not None:
            tokens.append(
                "b"
                + f"{float(beta):g}".replace(".", "p").replace("-", "m").replace("+", "")
            )
        rho = cfg.get("rho_lr")
        if rho is not None:
            tokens.append(
                "rho"
                + f"{float(rho):g}".replace(".", "p").replace("-", "m").replace("+", "")
            )
        if cfg.get("backend") == "jax":
            tokens.append("jax")
        if cfg.get("gaussian") is False:
            tokens.append("det")
    if family == "adaptive_multiscale":
        te = cfg.get("T_E")
        if te is not None:
            te_f = float(te)
            if float(te_f).is_integer():
                tokens.append(f"TE{int(te_f)}")
            else:
                tokens.append(
                    "TE" + f"{te_f:g}".replace(".", "p").replace("-", "m")
                )
        tb = cfg.get("T_B")
        if tb is not None and (
            te is None or abs(float(tb) - float(te)) > 1e-12
        ):
            tb_f = float(tb)
            if float(tb_f).is_integer():
                tokens.append(f"TB{int(tb_f)}")
            else:
                tokens.append(
                    "TB" + f"{tb_f:g}".replace(".", "p").replace("-", "m")
                )
    if family == "benchmark" and algo in ("aspc", "wpc"):
        # Prefer dirname tag (distinguishes pi_no_l2 from pi_local).
        m = re.search(
            r"_(aspc|wpc|multiscale|pi_local|pi_only|pi_raw|pi_no_l2)-",
            dirname,
        )
        if m:
            tokens.append(m.group(1))
        else:
            mode = str(cfg.get("l3_mode") or "").strip()
            if mode:
                tokens.append(
                    re.sub(r"[^0-9a-zA-Z]+", "_", mode).strip("_") or algo
                )
            else:
                tokens.append(algo)
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
        "rho_lr",
        "beta",
        "beta_initial",
        "gaussian",
        "expectile",
        "adaptive_enabled",
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
        "use_amo",
        "use_amo_v2",
        "use_amo_v3a",
        "use_amo_v3b",
        "amo_n_steps",
        "amo_v2_adapt_t",
        "amo_v2_t_init",
        "amo_v2_inner_steps",
        "amo_v3a_target_ratio",
        "amo_v3b_horizon_ratio",
        "rebrac_critic_bc",
        "actor_n_hiddens",
        "run_id",
        "sweep_name",
        "l3_mode",
        "loss_function",
        "ema_alpha",
        "project",
        "group",
    ]
    out = {}
    for k in keys:
        if k in cfg and cfg[k] is not None:
            out[k] = cfg[k]
    return out


def discover_run_dirs(
    root: Path,
    nested: bool,
    kind: str = "yaml",
    dirname_contains: Optional[str] = None,
) -> List[Path]:
    if not root.exists():
        return []
    name = "config.json" if kind == "jax" else "config.yaml"
    if nested:
        found = set(root.glob(f"*/*/{name}")) | set(root.glob(f"*/{name}"))
        dirs = sorted({p.parent for p in found})
    else:
        dirs = sorted({p.parent for p in root.glob(f"*/{name}")})
    if dirname_contains:
        dirs = [d for d in dirs if dirname_contains in d.name]
    # skip backup / crash leftovers
    dirs = [
        d
        for d in dirs
        if ".bak" not in d.name
        and "extcsh" not in d.name
        and not d.name.endswith("~")
    ]
    return dirs


def wpc_log_path(src: Path) -> Optional[Path]:
    m = re.match(r"(.+_wpc)-", src.name)
    if not m:
        return None
    candidate = src.parent / "_logs" / f"{m.group(1)}.log"
    return candidate if candidate.exists() else None


def parse_wpc_eval_log(log_path: Path) -> List[Dict[str, Any]]:
    """Recover eval curve from WPC stdout logs (no metrics.jsonl on disk)."""
    step_re = re.compile(r"Time steps:\s*(\d+)")
    # Often one line: "Evaluation over 10 episodes: 1.000 , D4RL score: 100.000"
    eval_line_re = re.compile(
        r"Evaluation over\s+\d+\s+episodes:\s*([0-9.]+)\s*,\s*D4RL score:\s*([0-9.]+)"
    )
    score_re = re.compile(r"D4RL score:\s*([0-9.]+)")
    ret_re = re.compile(r"Evaluation over\s+\d+\s+episodes:\s*([0-9.]+)")
    by_step: Dict[int, Dict[str, Any]] = {}
    pending_step: Optional[int] = None
    pending_return: Optional[float] = None
    for line in log_path.read_text(errors="ignore").splitlines():
        m = step_re.search(line)
        if m:
            pending_step = int(m.group(1))
            pending_return = None
            continue
        m = eval_line_re.search(line)
        if m and pending_step is not None:
            by_step[pending_step] = {
                "step": pending_step,
                "t": pending_step,
                "d4rl_normalized_score": float(m.group(2)),
                "eval_return": float(m.group(1)),
            }
            pending_step = None
            pending_return = None
            continue
        m = ret_re.search(line)
        if m:
            pending_return = float(m.group(1))
            continue
        m = score_re.search(line)
        if m and pending_step is not None:
            rec: Dict[str, Any] = {
                "step": pending_step,
                "t": pending_step,
                "d4rl_normalized_score": float(m.group(1)),
            }
            if pending_return is not None:
                rec["eval_return"] = pending_return
            by_step[pending_step] = rec
            pending_step = None
            pending_return = None
    return [by_step[k] for k in sorted(by_step)]


def last_eval_step(path: Path) -> Optional[int]:
    rows = []
    if path.exists():
        for line in path.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        return None
    try:
        return int(rows[-1].get("step", 0) or 0)
    except (TypeError, ValueError):
        return None


def merge_run_meta(cfg: Dict[str, Any], src: Path) -> Dict[str, Any]:
    """Fill env/seed/backend from run_meta.json when YAML omits them (JAX AMS)."""
    meta_path = src / "run_meta.json"
    if not meta_path.exists():
        return cfg
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return cfg
    out = dict(cfg)
    for key in ("env", "seed", "backend", "algorithm", "device"):
        if out.get(key) in (None, "", "unknown") and meta.get(key) is not None:
            out[key] = meta[key]
    if "max_timesteps" not in out and "max_steps" in out:
        out["max_timesteps"] = out["max_steps"]
    return out


def read_final_eval_50_row(src: Path) -> Optional[Dict[str, Any]]:
    """Prefer final50_singlepass_v1, then final_eval_50.jsonl, then posthoc 5x10.

    Protocol priority (do not mix as extra seeds):
      1) eval_final50_v1.jsonl / posthoc_final50_v1/final.json  (50 eps, repeats=1)
      2) final_eval_50.jsonl
      3) posthoc_eval_cpu/final.json (legacy episodes=10, final_repeats=5)
    Older rows are preserved in source run dirs; ingest records protocol/tag.
    """
    # 1) final50_singlepass_v1
    for path in (
        src / "eval_final50_v1.jsonl",
        src / "posthoc_final50_v1" / "final.json",
    ):
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            if path.suffix == ".jsonl":
                row = json.loads(path.read_text().strip().splitlines()[0])
            else:
                payload = json.loads(path.read_text())
                row = payload.get("row") if isinstance(payload, dict) else None
                if not isinstance(row, dict):
                    row = payload if isinstance(payload, dict) else None
            if not isinstance(row, dict):
                continue
            if row.get("status") not in (None, "done"):
                continue
            protocol = row.get("protocol") or row.get("tag") or "final50_singlepass_v1"
            return {
                "step": int(row.get("step") or 1_000_000),
                "d4rl_normalized_score": row.get(
                    "normalized_score", row.get("d4rl_normalized_score")
                ),
                "eval_return": row.get("return_mean", row.get("eval_return")),
                "n_episodes": row.get("episodes", row.get("n_episodes")),
                "episodes_per_repeat": row.get("episodes_per_repeat"),
                "repeats": row.get("repeats"),
                "device": row.get("device", "cpu"),
                "evaluated_at": row.get("evaluated_at"),
                "tag": row.get("tag", "final50_singlepass_v1"),
                "protocol": protocol,
                "backend": row.get("backend"),
                "eval_seed": row.get("eval_seed"),
                "checkpoint_sha256": row.get("checkpoint_sha256"),
                "success_rate": row.get("success_rate"),
            }
        except Exception:
            pass

    dest = src / "final_eval_50.jsonl"
    if dest.exists() and dest.stat().st_size > 0:
        try:
            row = json.loads(dest.read_text().strip().splitlines()[0])
            row = dict(row)
            row.setdefault("protocol", row.get("tag") or "final_eval_50")
            return row
        except Exception:
            pass
    posthoc = src / "posthoc_eval_cpu" / "final.json"
    if not posthoc.exists() or posthoc.stat().st_size == 0:
        return None
    try:
        payload = json.loads(posthoc.read_text())
        row = payload.get("row") if isinstance(payload, dict) else None
        if not isinstance(row, dict):
            row = payload if isinstance(payload, dict) else None
        if not isinstance(row, dict):
            return None
        return {
            "step": int(row.get("step") or 1_000_000),
            "d4rl_normalized_score": row.get(
                "normalized_score", row.get("d4rl_normalized_score")
            ),
            "eval_return": row.get("return_mean", row.get("eval_return")),
            "n_episodes": row.get("episodes", row.get("n_episodes")),
            "episodes_per_repeat": row.get("episodes_per_repeat"),
            "repeats": row.get("repeats"),
            "device": row.get("device", "cpu"),
            "evaluated_at": row.get("evaluated_at"),
            "tag": row.get("tag", "posthoc_cpu_final"),
            "protocol": "posthoc_cpu_final_5x10",
            "backend": row.get("backend"),
        }
    except Exception:
        return None


def resolve_code_commit(src: Path) -> Optional[str]:
    """Use only the commit recorded at training time; never invent from current HEAD.

    launch_manifest fallback is JAX-only: historical torch logs keep null so
    re-ingest does not collide with already-normalized main/ablation entries.
    """
    meta_path = src / "run_meta.json"
    meta: Optional[Dict[str, Any]] = None
    if meta_path.exists():
        try:
            loaded = json.loads(meta_path.read_text())
            if isinstance(loaded, dict):
                meta = loaded
        except Exception:
            meta = None
    if isinstance(meta, dict):
        git = meta.get("git") if isinstance(meta.get("git"), dict) else {}
        for key in ("code_commit", "commit", "git_commit"):
            val = git.get(key)
            if val:
                return str(val)
        for key in ("code_commit", "git_commit"):
            val = meta.get(key)
            if val:
                return str(val)
    is_jax = "jax" in str(src).lower()
    if isinstance(meta, dict) and str(meta.get("backend", "")).lower() == "jax":
        is_jax = True
    if not is_jax:
        return None
    # Launcher matrix manifests keep git even when train.py rewrites run_meta.
    for parent in (src, *src.parents):
        manifest = parent / "launch_manifest.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text())
            except Exception:
                data = None
            if isinstance(data, dict):
                git = data.get("git") if isinstance(data.get("git"), dict) else {}
                if git.get("commit"):
                    return str(git["commit"])
                if git.get("code_commit"):
                    return str(git["code_commit"])
        if parent.name in ("results", "home") or str(parent) == "/":
            break
    return None


def ingest_one(
    src: Path,
    algo: str,
    host: str,
    code_repo: str,
    family_force: Optional[str],
    dry_run: bool,
    kind: str = "yaml",
) -> Optional[Dict[str, Any]]:
    if kind == "jax":
        cfg_path = src / "config.json"
        if not cfg_path.exists():
            return None
        if not (src / "metrics.jsonl").exists() or not (src / "metrics.jsonl").stat().st_size:
            return None
        cfg = load_json_config(cfg_path)
    else:
        cfg_path = src / "config.yaml"
        if not cfg_path.exists():
            return None
        cfg = merge_run_meta(load_yaml_lite(cfg_path), src)
    if family_force == "adaptive_multiscale":
        cfg.setdefault("adaptive_multiscale", True)
    if "max_timesteps" not in cfg and cfg.get("max_steps") is not None:
        cfg["max_timesteps"] = cfg["max_steps"]
    fe_row = read_final_eval_50_row(src)
    env = str(cfg.get("env") or "unknown")
    seed = int(cfg.get("seed", 0) or 0)
    family = classify_family(algo, cfg, family_force)
    variant = build_variant(algo, family, cfg, src.name)
    last_step = last_metrics_step(src / "metrics.jsonl")
    if last_step is None and (src / "eval.jsonl").exists():
        last_step = last_eval_step(src / "eval.jsonl")
    synthesized_eval: Optional[List[Dict[str, Any]]] = None
    log_path = wpc_log_path(src) if algo == "wpc" else None
    if algo == "wpc" and not (src / "eval.jsonl").exists() and log_path is not None:
        synthesized_eval = parse_wpc_eval_log(log_path)
        if synthesized_eval and last_step is None:
            last_step = int(synthesized_eval[-1]["step"])
    if last_step is None and (src / "checkpoint_999999.pt").exists():
        last_step = 1_000_000
    summary_status = None
    summary_path = src / "summary.json"
    if summary_path.exists():
        try:
            summary_status = json.loads(summary_path.read_text()).get("status")
        except Exception:
            pass
    max_t = int(cfg.get("max_timesteps", 0) or 0)
    eval_freq = int(cfg.get("eval_freq", 5000) or 5000)
    incomplete = (
        last_step is not None
        and max_t > 0
        and summary_status != "complete"
        and last_step < max_t - eval_freq
        and (kind == "jax" or algo in ("aspc", "wpc"))
    )
    if incomplete and "incomplete" not in variant.split("_"):
        variant = f"{variant}_incomplete"
    uuid8 = extract_uuid8(src.name)
    if (kind == "jax" or cfg.get("backend") == "jax") and not re.search(
        r"[0-9a-fA-F]{8}$", src.name
    ):
        uuid8 = hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:8]
    short = env_short(env)
    run_id = f"{short}_s{seed}_{variant}__{uuid8}"
    dest = RUNS / algo / family / run_id
    src_metrics = src / "metrics.jsonl"
    dest_metrics = dest / "metrics.jsonl"
    src_fe = src / "final_eval_50.jsonl"
    dest_fe = dest / "final_eval_50.jsonl"
    src_final50 = src / "eval_final50_v1.jsonl"
    src_final50_marker = src / "posthoc_final50_v1" / "final.json"

    def _mtime(path: Path) -> float:
        return path.stat().st_mtime if path.exists() else 0.0

    newest_src_final = max(
        _mtime(src_fe),
        _mtime(src / "posthoc_eval_cpu" / "final.json"),
        _mtime(src_final50),
        _mtime(src_final50_marker),
    )
    need_final_eval = bool(fe_row) and (
        (not dest_fe.exists())
        or dest_fe.stat().st_size == 0
        or (
            src_fe.exists()
            and dest_fe.exists()
            and (
                dest_fe.stat().st_mtime < src_fe.stat().st_mtime
                or dest_fe.stat().st_size != src_fe.stat().st_size
            )
        )
        or (
            dest_fe.exists()
            and newest_src_final > dest_fe.stat().st_mtime
        )
    )
    if dest_metrics.exists() and src_metrics.exists():
        dest_step = last_metrics_step(dest_metrics)
        if (
            dest_step is not None
            and last_step is not None
            and dest_step >= last_step
            and dest_metrics.stat().st_size >= src_metrics.stat().st_size
            and not need_final_eval
        ):
            print(f"SKIP {dest.relative_to(ROOT)} step={dest_step}", flush=True)
            return None
    # WPC/config-only: skip if dest already has equal-or-newer eval
    if (
        not src_metrics.exists()
        and (dest / "eval.jsonl").exists()
        and last_step is not None
        and not need_final_eval
    ):
        dest_eval_step = last_eval_step(dest / "eval.jsonl")
        if dest_eval_step is not None and dest_eval_step >= last_step:
            print(f"SKIP {dest.relative_to(ROOT)} eval_step={dest_eval_step}", flush=True)
            return None

    artifacts = [
        f for f in KEEP_FILES if (src / f).exists() and (src / f).stat().st_size > 0
    ]
    if fe_row and "final_eval_50.jsonl" not in artifacts:
        artifacts.append("final_eval_50.jsonl")
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
        "git": {
            "code_repo": code_repo,
            "code_commit": resolve_code_commit(src),
        },
    }
    if last_step is not None:
        meta["settings"]["last_metrics_step"] = last_step
    if summary_status:
        meta["settings"]["source_status"] = summary_status
    if fe_row:
        meta["settings"]["final_eval_50"] = {
            "d4rl_normalized_score": fe_row.get("d4rl_normalized_score"),
            "eval_return": fe_row.get("eval_return"),
            "n_episodes": fe_row.get("n_episodes"),
            "device": fe_row.get("device"),
            "evaluated_at": fe_row.get("evaluated_at"),
        }
    if kind == "jax" or cfg.get("backend") == "jax":
        meta["checkpoint_hint"] = str(src.resolve())
        meta["backend"] = "jax"
    elif cfg.get("backend") in ("torch", "pytorch"):
        meta["backend"] = "torch"
    if log_path is not None:
        meta["settings"]["score_source"] = str(log_path.resolve())

    if dry_run:
        print(f"DRY {src} -> {dest.relative_to(ROOT)}")
        return meta

    dest.mkdir(parents=True, exist_ok=True)
    if kind == "jax":
        (dest / "config.yaml").write_text(dump_yaml(cfg) + "\n")
        shutil.copy2(src / "metrics.jsonl", dest / "metrics.jsonl")
        artifacts = ["config.yaml", "metrics.jsonl"]
        if write_eval_jsonl(src / "metrics.jsonl", dest / "eval.jsonl"):
            artifacts.append("eval.jsonl")
        meta["artifacts"] = artifacts
    else:
        for name in artifacts:
            if name == "final_eval_50.jsonl" and not (
                src / "final_eval_50.jsonl"
            ).exists():
                continue
            shutil.copy2(src / name, dest / name)
        if synthesized_eval and "eval.jsonl" not in artifacts:
            (dest / "eval.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in synthesized_eval)
            )
            meta["artifacts"].append("eval.jsonl")
    if fe_row:
        (dest / "final_eval_50.jsonl").write_text(
            json.dumps(fe_row, ensure_ascii=False) + "\n"
        )
        if "final_eval_50.jsonl" not in meta["artifacts"]:
            meta["artifacts"].append("final_eval_50.jsonl")
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
        kind = src_spec.get("kind", "yaml")
        for run_dir in discover_run_dirs(
            root,
            bool(src_spec.get("nested")),
            kind=kind,
            dirname_contains=src_spec.get("dirname_contains"),
        ):
            meta = ingest_one(
                run_dir,
                algo=src_spec["algo"],
                host=src_spec["host"],
                code_repo=src_spec["code_repo"],
                family_force=src_spec.get("family_force"),
                dry_run=args.dry_run,
                kind=kind,
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
