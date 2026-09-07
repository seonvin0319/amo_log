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
        "run_id",
        "sweep_name",
    ]
    out = {}
    for k in keys:
        if k in cfg and cfg[k] is not None:
            out[k] = cfg[k]
    return out


def discover_run_dirs(root: Path, nested: bool, kind: str = "yaml") -> List[Path]:
    if not root.exists():
        return []
    name = "config.json" if kind == "jax" else "config.yaml"
    if nested:
        found = set(root.glob(f"*/*/{name}")) | set(root.glob(f"*/{name}"))
        return sorted({p.parent for p in found})
    return sorted({p.parent for p in root.glob(f"*/{name}")})


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
        cfg = load_yaml_lite(cfg_path)
    env = str(cfg.get("env") or "unknown")
    seed = int(cfg.get("seed", 0) or 0)
    family = classify_family(algo, cfg, family_force)
    variant = build_variant(algo, family, cfg, src.name)
    last_step = last_metrics_step(src / "metrics.jsonl")
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
        kind == "jax"
        and last_step is not None
        and max_t > 0
        and summary_status != "complete"
        and last_step < max_t - eval_freq
    )
    if incomplete and "incomplete" not in variant.split("_"):
        variant = f"{variant}_incomplete"
    uuid8 = extract_uuid8(src.name)
    if kind == "jax" and not re.search(r"[0-9a-fA-F]{8}$", src.name):
        uuid8 = hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:8]
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
    if last_step is not None:
        meta["settings"]["last_metrics_step"] = last_step
    if summary_status:
        meta["settings"]["source_status"] = summary_status
    if kind == "jax":
        meta["checkpoint_hint"] = str(src.resolve())

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
            shutil.copy2(src / name, dest / name)
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
        for run_dir in discover_run_dirs(root, bool(src_spec.get("nested")), kind=kind):
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
    if args.rebuild_catalog and not args.dry_run:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_catalog import build

        build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
