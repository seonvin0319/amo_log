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
    {
        # ASPC-repo TD3+BC with Table 6 hparams + robust critic (3×256 + LayerNorm).
        "algo": "td3bc",
        "root": Path("/home/choi/ASPC/results/td3bc_aspc_table6/runs"),
        "host": "choi",
        "code_repo": "ASPC",
        "family_force": "aspc_rc",
        "log_dir": Path("/home/choi/ASPC/results/td3bc_aspc_table6/logs"),
    },
    {
        # MPI-IQL Actor0 / pi_base == vanilla IQL update (W2/FB only on Actor1+).
        # Source layout has no per-run config.yaml; special-cased in main().
        "algo": "iql",
        "root": Path("/home/choi/MPI/results/iql"),
        "host": "choi",
        "code_repo": "MPI",
        "family_force": "vanilla",
        "kind": "mpi_iql_actor0",
        "config_root": Path("/home/choi/MPI/configs/offline/mpi"),
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
    if algo == "td3bc":
        name = str(cfg.get("name", "")).lower()
        if "aspc" in name or "td3bc_aspc" in name:
            return "aspc_rc"
        return "misc"
    if algo == "aspc":
        name = str(cfg.get("name", "")).lower()
        if "td3bc" in name or "td3_bc" in name or "alpha" in cfg:
            return "aspc_rc"
        return "misc"
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
    if family == "aspc_rc" or family == "td3bc_table6" or algo in ("td3bc", "aspc"):
        # ASPC-style robust critic TD3+BC (Table 6).
        tokens.append("aspc_rc")
        alpha = cfg.get("alpha")
        if alpha is not None:
            a = float(alpha)
            tokens.append(f"a{int(a) if a.is_integer() else str(a).replace('.', 'p')}")
    if family == "vanilla" and algo == "iql":
        # Scores extracted from MPI multi-actor Actor0 (pi_base).
        tokens.append("pi_base")
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
        "beta",
        "iql_tau",
        "iql_deterministic",
        "vf_lr",
        "qf_lr",
        "n_episodes",
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


def aspc_log_path(src: Path, log_dir: Optional[Path]) -> Optional[Path]:
    if log_dir is None or not log_dir.exists():
        return None
    m = re.match(r"(td3bc_aspc_[a-z0-9]+_s\d+)-", src.name)
    if not m:
        # fall back to config name prefix before first '-' env chunk
        cfg = src / "config.yaml"
        if cfg.exists():
            name = str(load_yaml_lite(cfg).get("name") or "")
            # name may already include env-uuid suffix
            m2 = re.match(r"(td3bc_aspc_[a-z0-9]+_s\d+)", name)
            if m2:
                candidate = log_dir / f"{m2.group(1)}.log"
                return candidate if candidate.exists() else None
        return None
    candidate = log_dir / f"{m.group(1)}.log"
    return candidate if candidate.exists() else None


def synthesize_eval_jsonl_from_aspc_log(
    log_path: Path, eval_freq: int = 5000
) -> List[Dict[str, Any]]:
    """Parse CORL/ASPC stdout logs into eval records.

    Training prints `Time steps: N` then the Evaluation block. Fallback: assume
    evals at eval_freq, 2*eval_freq, ...
    """
    text = log_path.read_text(errors="ignore")
    rows: List[Dict[str, Any]] = []
    step: Optional[int] = None
    for line in text.splitlines():
        tm = re.search(r"Time steps:\s*(\d+)", line)
        if tm:
            step = int(tm.group(1))
            continue
        em = re.search(
            r"Evaluation over\s+(\d+)\s+episodes:\s*([-\d.]+)\s*,\s*D4RL score:\s*([-\d.]+)",
            line,
        )
        if not em:
            continue
        if step is None:
            step = (len(rows) + 1) * int(eval_freq)
        rows.append(
            {
                "step": step,
                "n_episodes": int(em.group(1)),
                "return_mean": float(em.group(2)),
                "d4rl_normalized_score": float(em.group(3)),
            }
        )
        step = None
    return rows


MPI_IQL_DROP_KEYS = {
    "num_actors",
    "w2_weights",
    "use_fb",
    "fb_tau",
    "sinkhorn_K",
    "sinkhorn_blur",
    "sinkhorn_backend",
    "algorithm",  # kept implicitly via algo=iql
}


def discover_mpi_iql_actor0_logs(root: Path) -> List[Path]:
    """Return completed MPI-IQL stdout logs under results/iql/.../logs/*.log."""
    if not root.exists():
        return []
    out: List[Path] = []
    for log in sorted(root.glob("*/*/seed_*/logs/*.log")):
        text = log.read_text(errors="ignore")
        if "Training completed" not in text:
            continue
        if "Actor 0 - Raw:" not in text:
            continue
        out.append(log)
    return out


def mpi_iql_config_path(log_path: Path, config_root: Path) -> Optional[Path]:
    # .../results/iql/{domain}/{dataset}/seed_N/logs/run_....log
    parts = log_path.parts
    try:
        i = parts.index("iql")
        domain, dataset = parts[i + 1], parts[i + 2]
    except (ValueError, IndexError):
        return None
    for name in (f"{dataset}_iql_fb.yaml", f"{dataset}_iql.yaml"):
        cand = config_root / domain / name
        if cand.exists():
            return cand
    return None


def synthesize_eval_jsonl_from_mpi_iql_actor0(
    log_path: Path, eval_freq: int = 5000
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse MPI multi-actor IQL logs; keep Actor 0 (pi_base) only."""
    text = log_path.read_text(errors="ignore")
    rows: List[Dict[str, Any]] = []
    step: Optional[int] = None
    header: Dict[str, Any] = {}
    m = re.search(r"Env:\s*([^\s,]+)", text)
    if m:
        header["env"] = m.group(1)
    m = re.search(r"seed:\s*(\d+)", text)
    if m:
        header["seed"] = int(m.group(1))
    m = re.search(r"num_actors:\s*(\d+)", text)
    if m:
        header["num_actors"] = int(m.group(1))
    m = re.search(r"w2_weights:\s*(\[[^\]]+\])", text)
    if m:
        header["w2_weights"] = m.group(1)
    m = re.search(r"use_fb:\s*(True|False)", text)
    if m:
        header["use_fb"] = m.group(1) == "True"
    for line in text.splitlines():
        tm = re.search(r"Time steps:\s*(\d+)", line)
        if tm:
            step = int(tm.group(1))
            continue
        am = re.search(
            r"Actor 0 - Raw:\s*([-\d.]+),\s*D4RL score:\s*([-\d.]+)",
            line,
        )
        if not am:
            continue
        if step is None:
            step = (len(rows) + 1) * int(eval_freq)
        rows.append(
            {
                "step": step,
                "return_mean": float(am.group(1)),
                "d4rl_normalized_score": float(am.group(2)),
                "actor": 0,
            }
        )
        step = None
    fm = re.search(
        r"Actor 0 final - det_mean:\s*([-\d.]+),\s*stoch_mean:\s*([-\d.]+)",
        text,
    )
    if fm:
        header["final_det_mean"] = float(fm.group(1))
        header["final_stoch_mean"] = float(fm.group(2))
    return rows, header


def write_vanilla_iql_config_from_mpi(cfg: Dict[str, Any]) -> str:
    """Emit a flat YAML for Actor0 / vanilla IQL (MPI multi-actor fields dropped)."""
    keep_order = [
        "env",
        "seed",
        "eval_freq",
        "n_episodes",
        "max_timesteps",
        "iql_tau",
        "beta",
        "vf_lr",
        "qf_lr",
        "actor_lr",
        "iql_deterministic",
        "actor_dropout",
        "batch_size",
        "discount",
        "tau",
        "buffer_size",
        "normalize",
        "normalize_reward",
        "final_eval_runs",
        "final_eval_episodes",
        "project",
        "group",
        "name",
    ]
    lines = [
        "# Vanilla IQL (pi_base / Actor0 extracted from MPI-IQL).",
        "# Multi-actor W2/FB fields were dropped; Actor0 update matches standalone IQL.",
        "algo: iql",
        "family: vanilla",
        "extracted_from: mpi_iql_actor0",
    ]
    seen = set()
    for key in keep_order:
        if key not in cfg or cfg[key] is None:
            continue
        seen.add(key)
        val = cfg[key]
        if isinstance(val, bool):
            raw = "true" if val else "false"
        else:
            raw = str(val)
        lines.append(f"{key}: {raw}")
    for key, val in cfg.items():
        if key in seen or key in MPI_IQL_DROP_KEYS or val is None:
            continue
        if isinstance(val, (list, dict)):
            continue
        if isinstance(val, bool):
            raw = "true" if val else "false"
        else:
            raw = str(val)
        lines.append(f"{key}: {raw}")
    return "\n".join(lines) + "\n"


def ingest_mpi_iql_actor0(
    log_path: Path,
    *,
    host: str,
    code_repo: str,
    config_root: Path,
    dry_run: bool,
) -> Optional[Dict[str, Any]]:
    cfg_path = mpi_iql_config_path(log_path, config_root)
    if cfg_path is None:
        print(f"SKIP no config for {log_path}")
        return None
    cfg = load_yaml_lite(cfg_path)
    eval_rows, header = synthesize_eval_jsonl_from_mpi_iql_actor0(
        log_path, int(cfg.get("eval_freq", 5000) or 5000)
    )
    if not eval_rows:
        print(f"SKIP no Actor0 evals in {log_path}")
        return None

    env = str(header.get("env") or cfg.get("env") or "unknown")
    seed = int(header.get("seed", cfg.get("seed", 0)) or 0)
    cfg["env"] = env
    cfg["seed"] = seed
    family = "vanilla"
    algo = "iql"
    dirname = log_path.parent.parent.name + "_" + log_path.stem
    variant = build_variant(algo, family, cfg, dirname)
    uuid8 = hashlib.sha1(str(log_path.resolve()).encode()).hexdigest()[:8]
    short = env_short(env)
    run_id = f"{short}_s{seed}_{variant}__{uuid8}"
    dest = RUNS / algo / family / run_id

    artifacts = ["config.yaml", "eval.jsonl"]
    meta = {
        "algo": algo,
        "family": family,
        "run_id": run_id,
        "env": env,
        "env_short": short,
        "seed": seed,
        "variant": variant,
        "legacy_name": log_path.name,
        "source_path": str(log_path.parent.parent.resolve()),  # seed_N dir
        "source_host": host,
        "source_log": str(log_path.resolve()),
        "source_config": str(cfg_path.resolve()),
        "collected_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "settings": settings_summary(algo, cfg),
        "artifacts": artifacts,
        "git": {"code_repo": code_repo, "code_commit": None},
        "eval_source": "synthesized_from_mpi_actor0_log",
        "notes": (
            "Actor0/pi_base scores from MPI-IQL (num_actors>1). "
            "Actor0 has no W2/FB term and matches vanilla IQL under the same IQL hparams."
        ),
        "mpi_source": {
            "num_actors": header.get("num_actors"),
            "w2_weights": header.get("w2_weights"),
            "use_fb": header.get("use_fb"),
            "final_det_mean": header.get("final_det_mean"),
            "final_stoch_mean": header.get("final_stoch_mean"),
        },
    }

    if dry_run:
        print(f"DRY {log_path} -> {dest.relative_to(ROOT)} eval_rows={len(eval_rows)}")
        return meta

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.yaml").write_text(write_vanilla_iql_config_from_mpi(cfg))
    (dest / "eval.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in eval_rows)
    )
    (dest / "run_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"OK  {dest.relative_to(ROOT)}")
    return meta


def ingest_one(
    src: Path,
    algo: str,
    host: str,
    code_repo: str,
    family_force: Optional[str],
    dry_run: bool,
    log_dir: Optional[Path] = None,
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

    # Materialize eval.jsonl for ASPC TD3+BC (stdout-only logging).
    synthesized_eval: Optional[List[Dict[str, Any]]] = None
    log_path = aspc_log_path(src, log_dir)
    if not (src / "eval.jsonl").exists() and log_path is not None:
        synthesized_eval = synthesize_eval_jsonl_from_aspc_log(
            log_path, int(cfg.get("eval_freq", 5000) or 5000)
        )

    artifacts = [f for f in KEEP_FILES if (src / f).exists()]
    if synthesized_eval and "eval.jsonl" not in artifacts:
        artifacts.append("eval.jsonl")
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
    if log_path is not None:
        meta["source_log"] = str(log_path.resolve())
    if synthesized_eval is not None:
        meta["eval_source"] = "synthesized_from_stdout_log"

    if dry_run:
        print(
            f"DRY {src} -> {dest.relative_to(ROOT)}"
            + (f" eval_rows={len(synthesized_eval)}" if synthesized_eval else "")
        )
        return meta

    dest.mkdir(parents=True, exist_ok=True)
    for name in artifacts:
        if name == "eval.jsonl" and synthesized_eval is not None and not (src / name).exists():
            (dest / name).write_text(
                "".join(json.dumps(row) + "\n" for row in synthesized_eval)
            )
        else:
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
        if src_spec.get("kind") == "mpi_iql_actor0":
            config_root: Path = src_spec["config_root"]
            for log_path in discover_mpi_iql_actor0_logs(root):
                meta = ingest_mpi_iql_actor0(
                    log_path,
                    host=src_spec["host"],
                    code_repo=src_spec["code_repo"],
                    config_root=config_root,
                    dry_run=args.dry_run,
                )
                if meta:
                    collected.append(meta)
            continue
        for run_dir in discover_run_dirs(root, bool(src_spec.get("nested"))):
            meta = ingest_one(
                run_dir,
                algo=src_spec["algo"],
                host=src_spec["host"],
                code_repo=src_spec["code_repo"],
                family_force=src_spec.get("family_force"),
                dry_run=args.dry_run,
                log_dir=src_spec.get("log_dir"),
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
