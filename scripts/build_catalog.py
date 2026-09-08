#!/usr/bin/env python3
"""Build catalog/INDEX.md and catalog/catalog.json from runs/*/run_meta.json."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CATALOG = ROOT / "catalog"


SCORE_KEYS = (
    "d4rl_normalized_score",
    "normalized_score",
    "eval/d4rl_normalized_score",
    "score",
)


def _score_from_obj(obj: Dict[str, Any]) -> Optional[float]:
    for key in SCORE_KEYS:
        if key in obj:
            try:
                return float(obj[key])
            except (TypeError, ValueError):
                pass
    results = obj.get("results")
    if isinstance(results, dict):
        for key in SCORE_KEYS:
            if key in results:
                try:
                    return float(results[key])
                except (TypeError, ValueError):
                    pass
    return None


def _step_from_obj(obj: Dict[str, Any]) -> Optional[int]:
    for key in ("step", "timestep", "t", "main_step"):
        if key in obj:
            try:
                return int(obj[key])
            except (TypeError, ValueError):
                pass
    return None


def last_eval_score(eval_path: Path) -> Optional[float]:
    """Prefer last execution-policy score; fall back to last any score."""
    if not eval_path.exists() or eval_path.stat().st_size == 0:
        return None
    last_any = None
    last_exec = None
    for line in eval_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        score = _score_from_obj(obj)
        if score is None:
            continue
        last_any = score
        policy = obj.get("policy_id")
        if policy in (None, "execution"):
            last_exec = score
    return last_exec if last_exec is not None else last_any


def last_eval_step(eval_path: Path) -> Optional[int]:
    if not eval_path.exists():
        return None
    last_any = None
    last_exec = None
    for line in eval_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = _step_from_obj(obj)
        if step is None:
            continue
        last_any = step
        policy = obj.get("policy_id")
        if policy in (None, "execution"):
            last_exec = step
    return last_exec if last_exec is not None else last_any


def build() -> None:
    CATALOG.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for meta_path in sorted(RUNS.glob("*/*/*/run_meta.json")):
        meta = json.loads(meta_path.read_text())
        run_dir = meta_path.parent
        score = last_eval_score(run_dir / "eval.jsonl")
        step = last_eval_step(run_dir / "eval.jsonl")
        settings = meta.get("settings", {})
        rows.append(
            {
                **meta,
                "rel_path": str(run_dir.relative_to(ROOT)),
                "final_score": score,
                "last_eval_step": step,
                "max_timesteps": settings.get("max_timesteps"),
            }
        )

    (CATALOG / "catalog.json").write_text(
        json.dumps({"n_runs": len(rows), "runs": rows}, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Experiment catalog",
        "",
        f"Total runs: **{len(rows)}**",
        "",
        "| algo | family | run_id | env | seed | max_steps | last_eval_step | final_score | path |",
        "|------|--------|--------|-----|------|-----------|----------------|-------------|------|",
    ]
    for r in sorted(rows, key=lambda x: (x["algo"], x["family"], x["env"], x["seed"], x["run_id"])):
        score = "—" if r["final_score"] is None else f"{r['final_score']:.2f}"
        step = "—" if r["last_eval_step"] is None else str(r["last_eval_step"])
        mstep = "—" if r.get("max_timesteps") is None else str(r["max_timesteps"])
        lines.append(
            f"| {r['algo']} | {r['family']} | `{r['run_id']}` | {r['env']} | {r['seed']} | "
            f"{mstep} | {step} | {score} | `{r['rel_path']}` |"
        )
    lines.append("")
    (CATALOG / "INDEX.md").write_text("\n".join(lines))
    print(f"catalog: {len(rows)} runs -> catalog/INDEX.md, catalog/catalog.json")


if __name__ == "__main__":
    build()
