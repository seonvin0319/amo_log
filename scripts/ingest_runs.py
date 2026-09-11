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

# Host choi (original snapshot) + host ext_csh (this machine).
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
]

# AMO loco-9 packs on ext_csh (layout: <pack>/runs/<run>/config.yaml).
_AMO_EXT_PACKS = [
    "amo_adaptive_multiscale_locomotion9_seed0",
    "amo_loco9_s0_tlr2e-3_te5_tb5",
    "amo_loco9_s0_tlr2e-3_te1_tb1",
    "amo_loco9_s0_tlr2e-3_te10_tb10",
    "amo_loco9_s0_tlr1e-3_te1_tb1",
    "amo_loco9_s0_tlr5e-4_te1_tb1",
    "amo_loco9_s0_tlr2e-3_te1_tb1_l1e",
    "amo_loco9_s0_tlr1e-3_te1_tb1_l1e",
    "amo_loco9_s0_tlr2e-3_te1_tb1_td3bc_critic",
    "amo_loco9_s0_tlr2e-3_te0.1_tb0.1_td3bc_critic",
    "amo_loco9_s0_tlr5e-4_te1_tb1_td3bc_critic",
    "amo_loco9_s0_tlr2e-3_te0.005_tb0.005_td3bc_critic_qraw",
]
for _name in _AMO_EXT_PACKS:
    DEFAULT_SOURCES.append(
        {
            "algo": "amo",
            "root": Path("/home/ext_csh/AMO/results") / _name,
            "host": "ext_csh",
            "code_repo": "AMO",
            "family_force": "adaptive_multiscale",
            "nested": True,
        }
    )

# Antmaze-6 B_PI T_lr × (T_E=T_B) init sweep: cells/<tag>/runs/<run>/config.yaml
_ANTMAZE_BPI_CELLS = Path(
    "/home/ext_csh/AMO-antmaze-bpi-tlr-init-sweep/results/"
    "amo_antmaze6_s0_bpi_tlr_te_tb_init_sweep/cells"
)
if _ANTMAZE_BPI_CELLS.is_dir():
    for _cell in sorted(p for p in _ANTMAZE_BPI_CELLS.iterdir() if p.is_dir()):
        DEFAULT_SOURCES.append(
            {
                "algo": "amo",
                "root": _cell,
                "host": "ext_csh",
                "code_repo": "AMO",
                "family_force": "adaptive_multiscale",
                "nested": True,
            }
        )

# Historical antmaze6 parent (T_B=T_E/2 fixed-ratio).
DEFAULT_SOURCES.append(
    {
        "algo": "amo",
        "root": Path(
            "/home/ext_csh/AMO-a8c1e48-te1tb1/results/"
            "amo_antmaze6_s0_tlr1e-3_te1_tb_div2"
        ),
        "host": "ext_csh",
        "code_repo": "AMO",
        "family_force": "adaptive_multiscale",
        "nested": True,
    }
)

# Early adaptive-multiscale runs lived under APART/results but are AMO family.
for _name in (
    "adaptive_multiscale_locomotion9_seed0",
    "adaptive_multiscale_locomotion9_unconstrained_seed0",
    "adaptive_multiscale_locomotion12_seed0",
    "adaptive_bootstrap_locomotion12_seed0",
    "adaptive_bootstrap_locomotion12_seed0_unconstrained",
    "adaptive_multiscale_smoke",
    "adaptive_multiscale_equal_init_smoke",
    "adaptive_bootstrap_smoke",
):
    DEFAULT_SOURCES.append(
        {
            "algo": "amo",
            "root": Path("/home/ext_csh/APART/results") / _name,
            "host": "ext_csh",
            "code_repo": "APART",
            "family_force": "adaptive_multiscale",
            "nested": True,
        }
    )

# APART dual / n1 Adroit packs on ext_csh.
DEFAULT_SOURCES.extend(
    [
        {
            "algo": "apart",
            "root": Path("/home/ext_csh/APART/results/dual_n24_tlr"),
            "host": "ext_csh",
            "code_repo": "APART",
            "family_force": "dual_proximal",
            "nested": True,
        },
        {
            "algo": "apart",
            "root": Path("/home/ext_csh/APART/results/dual_n24_adroit"),
            "host": "ext_csh",
            "code_repo": "APART",
            "family_force": "dual_proximal",
            "nested": True,
        },
        {
            "algo": "apart",
            "root": Path("/home/ext_csh/APART/results/n1_pen_cloned"),
            "host": "ext_csh",
            "code_repo": "APART",
            "family_force": "chain",
            "nested": True,
        },
    ]
)

# CORL IQL adaptive-β (corl_iql_adaptive_beta_v1) on ext_csh.
# Layout: <suite>/<env-v2>/{resolved_config.yaml,metrics.jsonl,eval.jsonl}
for _name, _tag in (
    ("iql_beta_locomotion4_s0", "loco4"),
    ("iql_beta_remaining7_jp2_s0", "rem7"),
):
    DEFAULT_SOURCES.append(
        {
            "algo": "iql",
            "root": Path("/home/ext_csh/CORL-iql-adaptive-beta-v1/results") / _name,
            "host": "ext_csh",
            "code_repo": "CORL-iql-adaptive-beta-v1",
            "family_force": "adaptive_beta",
            "config_file": "resolved_config.yaml",
            "variant_tag": _tag,
        }
    )

# CORL IQL+AMO dual B_PI sweep (corl_iql_amo_bpi_v1).
# Layout: <suite>/cells/<rlrX_bY>/<env-v2>/{resolved_config.yaml,...}
DEFAULT_SOURCES.append(
    {
        "algo": "iql",
        "root": Path(
            "/home/ext_csh/CORL-iql-adaptive-beta-v1/results/"
            "iql_amo_bpi_s0_rlr_binit_loco9_antmaze6/cells"
        ),
        "host": "ext_csh",
        "code_repo": "CORL-iql-adaptive-beta-v1",
        "family_force": "amo_bpi",
        "config_file": "resolved_config.yaml",
        "nested": True,
        "variant_tag": "bpi_sweep",
    }
)


def _parse_yaml_scalar(raw: str) -> Any:
    raw = raw.strip().strip("'\"")
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    if raw.lower() in ("null", "none", ""):
        return None
    try:
        if "." in raw or "e" in raw.lower():
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def load_yaml_lite(path: Path) -> Dict[str, Any]:
    """Minimal YAML subset reader (key: value) without PyYAML dependency."""
    out: Dict[str, Any] = {}
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        out[key.strip()] = _parse_yaml_scalar(raw)
    return out


def load_iql_resolved_config(path: Path) -> Dict[str, Any]:
    """Flatten CORL resolved_config.yaml (corl:/meta: sections) for ingest."""
    corl: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}
    section: Optional[str] = None
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if re.match(r"^[A-Za-z0-9_]+:\s*$", line):
            section = line.split(":", 1)[0].strip()
            continue
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        # skip list items under meta (rho_adam_betas)
        if key.startswith("-"):
            continue
        val = _parse_yaml_scalar(raw)
        if section == "corl":
            corl[key] = val
        elif section == "meta":
            meta[key] = val
    flat: Dict[str, Any] = {}
    flat.update(corl)
    flat.update(meta)
    # Prefer explicit meta betas when present.
    if meta.get("beta_fixed") is not None:
        flat["beta_fixed"] = meta["beta_fixed"]
    if meta.get("beta_initial") is not None:
        flat["beta_initial"] = meta["beta_initial"]
    flat["_raw_sections"] = {"corl": corl, "meta": meta}
    return flat


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

    if "smoke" in blob or int(cfg.get("max_timesteps", 0) or 0) <= 20_000:
        tokens.append("smoke")

    t_lr = cfg.get("T_lr")
    if t_lr is not None and float(t_lr) not in (2e-4, 0.0002):
        tokens.append("Tlr" + fmt_num(float(t_lr)))

    if family.startswith("pi_only") and not tokens:
        tokens.append("pi_only_xfit")
    if family == "adaptive_beta":
        tokens.append("iql_ab")
        if cfg.get("_variant_tag"):
            tokens.append(str(cfg["_variant_tag"]))
        # Record CORL beta0 (fixed / adaptive init) when non-default lore.
        b0 = cfg.get("beta_fixed", cfg.get("beta_initial", cfg.get("beta")))
        if b0 is not None:
            tokens.append(f"b{fmt_num(float(b0))}")
        tau = cfg.get("iql_tau")
        if tau is not None:
            tokens.append(f"t{fmt_num(float(tau))}")
    if family == "amo_bpi":
        tokens.append("iql_amo_bpi")
        if cfg.get("_variant_tag"):
            tokens.append(str(cfg["_variant_tag"]))
        rlr = cfg.get("rho_lr")
        if rlr is not None:
            tokens.append("rlr" + fmt_num(float(rlr)))
        b0 = cfg.get("beta_initial", cfg.get("beta_init", cfg.get("beta")))
        if b0 is not None:
            tokens.append(f"b{fmt_num(float(b0))}")
        tau = cfg.get("iql_tau")
        if tau is not None:
            tokens.append(f"t{fmt_num(float(tau))}")
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
        "T_lr",
        "T_freq",
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
        "beta",
        "beta_fixed",
        "beta_initial",
        "iql_tau",
        "adaptive_enabled",
        "meta_warmup_steps",
        "meta_interval",
        "rho_lr",
        "weight_cap",
        "amo_dual_bpi",
    ]
    out = {}
    for key in keys:
        if key.startswith("_"):
            continue
        if key in cfg and cfg[key] is not None:
            out[key] = cfg[key]
    return out


def discover_run_dirs(
    root: Path, nested: bool, config_file: str = "config.yaml"
) -> List[Path]:
    if not root.exists():
        return []
    if nested:
        # <pack>/runs/<run>/config.yaml  or  <pack>/<group>/<run>/config.yaml
        return sorted({p.parent for p in root.glob(f"*/*/{config_file}")})
    return sorted({p.parent for p in root.glob(f"*/{config_file}")})


def ingest_one(
    src: Path,
    algo: str,
    host: str,
    code_repo: str,
    family_force: Optional[str],
    dry_run: bool,
    source_root: Optional[Path] = None,
    config_file: str = "config.yaml",
    variant_tag: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    cfg_path = src / config_file
    if not cfg_path.exists() and config_file != "config.yaml":
        cfg_path = src / "config.yaml"
    if not cfg_path.exists():
        return None

    if cfg_path.name == "resolved_config.yaml" or family_force in (
        "adaptive_beta",
        "amo_bpi",
    ):
        cfg = load_iql_resolved_config(cfg_path)
    else:
        cfg = load_yaml_lite(cfg_path)
    if variant_tag:
        cfg["_variant_tag"] = variant_tag
    # Nested cell tag (e.g. rlr2e-3_b1) for amo_bpi uniqueness.
    if family_force == "amo_bpi" and src.parent is not None:
        cfg["_cell"] = src.parent.name

    env = str(cfg.get("env") or src.name or "unknown")
    seed = int(cfg.get("seed", 0) or 0)
    family = classify_family(algo, cfg, family_force)
    # Adaptive-multiscale always archives under amo/, even if code lived in APART/.
    if family == "adaptive_multiscale":
        algo = "amo"
    # Include parent cell dir in dirname blob so uuid/variant stay unique per cell.
    dirname_for_variant = (
        f"{src.parent.name}_{src.name}" if family == "amo_bpi" else src.name
    )
    variant = build_variant(
        algo, family, cfg, dirname_for_variant, source_root=source_root
    )
    uuid8 = extract_uuid8(dirname_for_variant)
    short = env_short(env)
    run_id = f"{short}_s{seed}_{variant}__{uuid8}"
    dest = RUNS / algo / family / run_id

    present = [f for f in ("metrics.jsonl", "eval.jsonl") if (src / f).exists()]
    if not present:
        return None

    meta: Dict[str, Any] = {
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
        "checkpoint_hint": str(src.resolve()),
        "collected_at": datetime.now(timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds"),
        "settings": settings_summary(algo, cfg),
        "artifacts": ["config.yaml"] + present,
        "git": {"code_repo": code_repo, "code_commit": None},
    }
    if family == "adaptive_beta":
        meta["protocol"] = "corl_iql_adaptive_beta_v1"
    if family == "amo_bpi":
        meta["protocol"] = "corl_iql_amo_bpi_v1"
        if cfg.get("_cell"):
            meta["cell"] = cfg["_cell"]

    if dry_run:
        print(f"DRY {src} -> {dest.relative_to(ROOT)}")
        return meta

    dest.mkdir(parents=True, exist_ok=True)
    # Archive as config.yaml even when source used resolved_config.yaml.
    shutil.copy2(cfg_path, dest / "config.yaml")
    for name in present:
        shutil.copy2(src / name, dest / name)
    for extra in ("launch_cmd.txt", "notes.md"):
        if (src / extra).exists():
            shutil.copy2(src / extra, dest / extra)
            meta["artifacts"].append(extra)
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
        help="Only ingest sources for this source_host (e.g. ext_csh).",
    )
    args = parser.parse_args()

    collected: List[Dict[str, Any]] = []
    for src_spec in DEFAULT_SOURCES:
        if args.host and src_spec.get("host") != args.host:
            continue
        root: Path = src_spec["root"]
        config_file = str(src_spec.get("config_file") or "config.yaml")
        for run_dir in discover_run_dirs(
            root, bool(src_spec.get("nested")), config_file=config_file
        ):
            meta = ingest_one(
                run_dir,
                algo=src_spec["algo"],
                host=src_spec["host"],
                code_repo=src_spec["code_repo"],
                family_force=src_spec.get("family_force"),
                dry_run=args.dry_run,
                source_root=root,
                config_file=config_file,
                variant_tag=src_spec.get("variant_tag"),
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
