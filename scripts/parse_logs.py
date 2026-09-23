#!/usr/bin/env python3
"""Flat, analysis-friendly views over canonical amo_log runs.

The canonical source of truth remains run_meta.json/config.yaml/raw evaluation files.
This module only derives stable CSV/Markdown views; it never selects/deletes runs.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
import sys
from pathlib import Path

from log_layout import initial_alphas, number

TARGET_STEP = 1_000_000
EVAL_FILES = ("eval.jsonl", "final_eval_50.json", "final_eval_50.jsonl")
SCORE_FIELDS = ("d4rl_normalized_score", "normalized_score", "mean_normalized")
DIAGNOSTIC_POLICY_TOKENS = ("minus", "diff", "delta")


def finite(value):
    n = number(value)
    if n is None or isinstance(value, bool) or not math.isfinite(n):
        return None
    return n


def domain(env):
    env = str(env or "")
    if env.startswith(("halfcheetah-", "hopper-", "walker2d-")):
        return "locomotion"
    if env.startswith("antmaze-"):
        return "antmaze"
    if env.startswith(("door-", "hammer-", "pen-", "relocate-")):
        return "adroit"
    return "other"


def record_step(record):
    for key in ("step", "t", "timestep", "main_step"):
        n = finite(record.get(key))
        if n is not None and n >= 0 and float(n).is_integer():
            return int(n)
    checkpoint = str(record.get("checkpoint", ""))
    match = re.search(r"(?:^|/)checkpoint_(\d+)\.pt$", checkpoint)
    if match:
        return int(match.group(1)) + 1
    match = re.search(r"(?:^|/)step_(\d+)\.npz$", checkpoint)
    return int(match.group(1)) if match else None


def episode_count(record, settings):
    for key in ("episode_count", "episodes", "n_episodes_total"):
        n = finite(record.get(key))
        if n is not None:
            return int(n)
    if finite(record.get("episodes_per_repeat")) is not None and finite(record.get("repeats")) is not None:
        return int(record["episodes_per_repeat"]) * int(record["repeats"])
    if finite(record.get("n_eval_episodes")) is not None:
        return int(record["n_eval_episodes"])
    if finite(record.get("n_episodes")) is not None:
        repeats = record.get("final_eval_repeats", 1) if record.get("final_eval_aggregate") else 1
        return int(record["n_episodes"]) * int(repeats)
    fallback = settings.get("n_episodes", settings.get("eval_episodes"))
    n = finite(fallback)
    return int(n) if n is not None else None


def normalized_score(record):
    values = [(key, finite(record.get(key))) for key in SCORE_FIELDS]
    values = [(key, value) for key, value in values if value is not None]
    if not values:
        return None, None
    key, score = values[0]
    if any(not math.isclose(score, value, rel_tol=1e-9, abs_tol=1e-9) for _, value in values[1:]):
        raise ValueError("conflicting normalized score fields")
    return key, score


def _unwrap_record(record):
    if not isinstance(record, dict) or "legacy_unparsed_record" in record:
        return None
    if "ok" in record:
        if record.get("ok") is not True:
            return None
        record = record.get("row")
        if not isinstance(record, dict):
            return None
    return record


def evaluation_rows(run_dir, meta):
    """Return scored evaluation rows and non-fatal parse errors for one run."""
    rows, errors = [], []
    settings = meta.get("settings", {})
    for file_rank, filename in enumerate(EVAL_FILES):
        path = run_dir / filename
        if not path.exists():
            continue
        raw = path.read_text(errors="replace")
        records = [raw] if filename.endswith(".json") else raw.splitlines()
        for line_no, line in enumerate(records, 1):
            if not line.strip():
                continue
            try:
                record = _unwrap_record(json.loads(line))
                if record is None or "final_eval_repeat" in record:
                    continue
                field, score = normalized_score(record)
                if score is None:
                    continue
                step = record_step(record)
                if step is None:
                    raise ValueError("missing evaluation step")
                aggregate = bool(
                    record.get("final_eval_aggregate")
                    or record.get("groups")
                    or (finite(record.get("repeats")) or 0) > 1
                    or (finite(record.get("n_eval_repeats")) or 0) > 1
                    or filename.startswith("final_eval_50")
                )
                rows.append({
                    "rel_path": meta.get("rel_path"),
                    "run_id": meta.get("run_id"),
                    "method": meta.get("method"),
                    "family": meta.get("family") or "",
                    "env": meta.get("env"),
                    "seed": meta.get("seed"),
                    "policy_id": record.get("policy_id"),
                    "step": step,
                    "score": score,
                    "score_field": field,
                    "aggregate": aggregate,
                    "episodes": episode_count(record, settings),
                    "evaluated_at": record.get("evaluated_at") or "",
                    "eval_file": filename,
                    "eval_line": line_no,
                    "_file_rank": file_rank,
                })
            except (ValueError, TypeError) as exc:
                errors.append(f"{filename}:{line_no}:{exc}")
    return rows, errors


def policy_priority(meta):
    settings = meta.get("settings", {})
    method = meta.get("method")
    if method == "iql_amo" and (
        settings.get("freeze_scale_E") is True
        or settings.get("adaptive_scale_E") is False
        or "fixed_beta_E" in str(meta.get("family", ""))
    ):
        return ("fixed_beta", "adaptive_beta", "adaptive", None, "execution")
    if method in ("td3_amo", "td3bc+rc", "fql_amo"):
        return ("execution", None, "adaptive", "adaptive_beta", "fixed_beta")
    return ("adaptive_beta", "adaptive", None, "fixed_beta", "execution")


def primary_evaluation(meta, rows):
    if not rows:
        return None
    clean = [
        row for row in rows
        if not any(token in str(row.get("policy_id") or "").lower()
                   for token in DIAGNOSTIC_POLICY_TOKENS)
        and str(row.get("policy_id") or "").lower() != "bootstrap"
    ]
    pool = clean or rows
    priority = policy_priority(meta)
    selected_policy = next(
        (policy for policy in priority if any(row.get("policy_id") == policy for row in pool)),
        None,
    )
    if any(row.get("policy_id") == selected_policy for row in pool):
        pool = [row for row in pool if row.get("policy_id") == selected_policy]
    target = [row for row in pool if row["step"] == TARGET_STEP]
    pool = target or pool
    return max(
        pool,
        key=lambda row: (
            row["step"],
            bool(row["aggregate"]),
            row["episodes"] or -1,
            row["evaluated_at"],
            row["_file_rank"],
            row["eval_line"],
        ),
    )


def _initial_fields(meta):
    settings = meta.get("settings", {})
    if meta.get("method") in ("td3_amo", "fql_amo"):
        alpha_e, alpha_b = initial_alphas(settings)
        return alpha_e, alpha_b, None
    if meta.get("method") == "iql_amo":
        return None, None, number(settings.get("beta_initial"))
    return None, None, None


def flat_run(meta, primary=None, errors=()):
    settings = meta.get("settings", {})
    alpha_e, alpha_b, beta = _initial_fields(meta)
    reasons = meta.get("classification_reasons") or []
    git = meta.get("git") or {}
    row = {
        "section": meta.get("section"),
        "method": meta.get("method"),
        "family": meta.get("family") or "",
        "variant": meta.get("variant") or "",
        "domain": domain(meta.get("env")),
        "env": meta.get("env"),
        "seed": meta.get("seed"),
        "backend": meta.get("backend"),
        "meta_lr": meta.get("meta_lr"),
        "alpha_E_initial": alpha_e,
        "alpha_B_initial": alpha_b,
        "beta_initial": beta,
        "adaptive_scale_E": settings.get("adaptive_scale_E"),
        "adaptive_scale_B": settings.get("adaptive_scale_B"),
        "freeze_scale_E": settings.get("freeze_scale_E"),
        "freeze_scale_B": settings.get("freeze_scale_B"),
        "execution_only": bool(settings.get("execution_only", False)),
        "bootstrap_loss": settings.get("bootstrap_loss"),
        "execution_meta_loss": settings.get("execution_meta_loss"),
        "execution_score": settings.get("execution_score"),
        "actor_lr": settings.get("actor_lr"),
        "critic_lr": settings.get("critic_lr", settings.get("qf_lr")),
        "value_lr": settings.get("value_lr", settings.get("vf_lr")),
        "classification_reasons": ";".join(reasons),
        "last_eval_step": meta.get("last_eval_step"),
        "primary_policy_id": None,
        "primary_eval_step": None,
        "primary_score": None,
        "primary_eval_episodes": None,
        "complete_1m": False,
        "eval_parse_errors": ";".join(errors),
        "run_id": meta.get("run_id"),
        "rel_path": meta.get("rel_path"),
        "source_path": meta.get("source_path"),
        "source_host": meta.get("source_host"),
        "code_commit": git.get("code_commit"),
    }
    if primary:
        row.update(
            primary_policy_id=primary.get("policy_id"),
            primary_eval_step=primary.get("step"),
            primary_score=primary.get("score"),
            primary_eval_episodes=primary.get("episodes"),
            complete_1m=primary.get("step") == TARGET_STEP,
        )
    return row


RUN_COLUMNS = (
    "section", "method", "family", "variant", "domain", "env", "seed", "backend", "meta_lr",
    "alpha_E_initial", "alpha_B_initial", "beta_initial",
    "adaptive_scale_E", "adaptive_scale_B", "freeze_scale_E", "freeze_scale_B",
    "execution_only", "bootstrap_loss", "execution_meta_loss", "execution_score",
    "actor_lr", "critic_lr", "value_lr", "classification_reasons",
    "last_eval_step", "primary_policy_id", "primary_eval_step", "primary_score",
    "primary_eval_episodes", "complete_1m", "eval_parse_errors",
    "run_id", "rel_path", "source_path", "source_host", "code_commit",
)

EVAL_COLUMNS = (
    "rel_path", "run_id", "method", "family", "env", "seed", "policy_id", "step", "score",
    "score_field", "aggregate", "episodes", "evaluated_at", "eval_file", "eval_line",
)

FAMILY_COLUMNS = (
    "section", "method", "family", "runs", "scored_runs", "complete_1m_runs",
    "environments", "seeds", "meta_lrs", "backends", "initializations",
    "classification_reasons",
)


def collect(root, metas=None):
    root = Path(root)
    if metas is None:
        catalog_path = root / "catalog" / "catalog.json"
        if catalog_path.exists():
            metas = json.loads(catalog_path.read_text()).get("runs", [])
        else:
            metas = []
            for section in ("main", "ablation"):
                for path in (root / section).rglob("run_meta.json"):
                    meta = json.loads(path.read_text())
                    if not meta.get("is_alias"):
                        metas.append(meta)
    runs, evaluations = [], []
    for meta in metas:
        if meta.get("is_alias"):
            continue
        run_dir = root / meta["rel_path"]
        evals, errors = evaluation_rows(run_dir, meta)
        evaluations.extend(evals)
        runs.append(flat_run(meta, primary_evaluation(meta, evals), errors))
    runs.sort(key=lambda r: (
        r["section"] or "", r["method"] or "", r["family"] or "", r["env"] or "",
        r["meta_lr"] if r["meta_lr"] is not None else -1, r["seed"] if r["seed"] is not None else -1,
        r["run_id"] or "",
    ))
    evaluations.sort(key=lambda r: (
        r["method"] or "", r["family"] or "", r["env"] or "",
        r["seed"] if r["seed"] is not None else -1, r["run_id"] or "",
        r["step"], str(r["policy_id"]), r["eval_file"], r["eval_line"],
    ))
    return runs, evaluations


def family_rows(runs):
    grouped = collections.defaultdict(list)
    for run in runs:
        grouped[(run["section"], run["method"], run["family"])].append(run)
    out = []
    for (section, method, family), items in sorted(grouped.items()):
        initials = set()
        for row in items:
            if row["beta_initial"] is not None:
                initials.add(f"beta={row['beta_initial']:g}")
            elif row["alpha_E_initial"] is not None:
                initials.add(f"alpha_E/B={row['alpha_E_initial']:g}/{row['alpha_B_initial']:g}")
        reasons = sorted({
            reason
            for row in items
            for reason in str(row["classification_reasons"] or "").split(";")
            if reason
        })
        out.append({
            "section": section,
            "method": method,
            "family": family,
            "runs": len(items),
            "scored_runs": sum(row["primary_score"] is not None for row in items),
            "complete_1m_runs": sum(bool(row["complete_1m"]) for row in items),
            "environments": ";".join(sorted({str(row["env"]) for row in items})),
            "seeds": ";".join(str(x) for x in sorted({row["seed"] for row in items if row["seed"] is not None})),
            "meta_lrs": ";".join(format(x, "g") for x in sorted({row["meta_lr"] for row in items if row["meta_lr"] is not None})),
            "backends": ";".join(sorted({str(row["backend"]) for row in items if row["backend"]})),
            "initializations": ";".join(sorted(initials)),
            "classification_reasons": ";".join(reasons),
        })
    return out


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_exports(root, metas=None):
    root = Path(root)
    runs, evaluations = collect(root, metas)
    families = family_rows(runs)
    catalog = root / "catalog"
    _write_csv(catalog / "runs_flat.csv", runs, RUN_COLUMNS)
    _write_csv(catalog / "evaluations_flat.csv", evaluations, EVAL_COLUMNS)
    _write_csv(catalog / "families.csv", families, FAMILY_COLUMNS)
    lines = [
        "# Parsed experiment families",
        "",
        "Derived index only. Canonical data remain `run_meta.json`, `config.yaml`, and raw evaluation files.",
        "",
        "| section | method | family | runs | scored | complete@1M | environments | initializations |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in families:
        lines.append(
            f"| {row['section']} | {row['method']} | {row['family'] or '—'} | "
            f"{row['runs']} | {row['scored_runs']} | {row['complete_1m_runs']} | "
            f"{row['environments']} | {row['initializations']} |"
        )
    lines += [
        "",
        "Machine-readable views: `runs_flat.csv`, `evaluations_flat.csv`, `families.csv`.",
        "",
    ]
    (catalog / "FAMILIES.md").write_text("\n".join(lines))
    return runs, evaluations, families


def _matches(row, args):
    if args.section and row["section"] != args.section:
        return False
    if args.method and row["method"] != args.method:
        return False
    if args.family and row["family"] != args.family:
        return False
    if args.env and row["env"] != args.env:
        return False
    if args.completed_only and not row["complete_1m"]:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--write", action="store_true",
                        help="write catalog/runs_flat.csv, evaluations_flat.csv, families.csv, FAMILIES.md")
    parser.add_argument("--section", choices=("main", "ablation"))
    parser.add_argument("--method")
    parser.add_argument("--family")
    parser.add_argument("--env")
    parser.add_argument("--completed-only", action="store_true")
    parser.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    args = parser.parse_args()

    runs, evaluations = collect(args.root)
    if args.write:
        write_exports(args.root)
    selected = [row for row in runs if _matches(row, args)]
    if args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=RUN_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    else:
        for row in selected:
            print(json.dumps(row, sort_keys=True, ensure_ascii=False))
    if args.write:
        print(json.dumps({
            "runs": len(runs),
            "evaluations": len(evaluations),
            "families": len(family_rows(runs)),
        }, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
