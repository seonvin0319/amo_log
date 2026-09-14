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
    "d4rl",
    "score",
)
PREFERRED_EVAL_FILE = "eval_final50_v1.jsonl"
PREFERRED_EVAL_PROTOCOL = "final50_singlepass_v1"


def _score_from_obj(obj: Dict[str, Any]) -> Optional[float]:
    for key in SCORE_KEYS:
        if key in obj and obj[key] is not None:
            try:
                return float(obj[key])
            except (TypeError, ValueError):
                pass
    return None


def last_eval_score(eval_path: Path) -> Optional[float]:
    if not eval_path.exists() or eval_path.stat().st_size == 0:
        return None
    last = None
    for line in eval_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        sc = _score_from_obj(obj)
        if sc is not None:
            last = sc
    return last


def last_eval_step(eval_path: Path) -> Optional[int]:
    if not eval_path.exists():
        return None
    last = None
    for line in eval_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in ("step", "timestep", "t"):
            if key in obj:
                try:
                    last = int(obj[key])
                except (TypeError, ValueError):
                    pass
    return last


def preferred_final50(run_dir: Path) -> tuple[Optional[float], Optional[int], Optional[str]]:
    """Return (score, step, source_label) from final50_singlepass_v1 when valid."""
    path = run_dir / PREFERRED_EVAL_FILE
    if not path.is_file() or path.stat().st_size == 0:
        return None, None, None
    last_score = None
    last_step = None
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("protocol") != PREFERRED_EVAL_PROTOCOL:
            continue
        if int(obj.get("episodes") or 0) != 50:
            continue
        if int(obj.get("repeats") or 0) != 1:
            continue
        if int(obj.get("episodes_per_repeat") or 0) != 50:
            continue
        sc = _score_from_obj(obj)
        if sc is None:
            continue
        last_score = sc
        try:
            last_step = int(obj.get("step") or 0) or None
        except (TypeError, ValueError):
            last_step = None
    if last_score is None:
        return None, None, None
    return last_score, last_step, f"{PREFERRED_EVAL_FILE}:{PREFERRED_EVAL_PROTOCOL}"


def build() -> None:
    CATALOG.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for meta_path in sorted(RUNS.glob("*/*/*/run_meta.json")):
        meta = json.loads(meta_path.read_text())
        run_dir = meta_path.parent
        pref_score, pref_step, pref_src = preferred_final50(run_dir)
        legacy_score = last_eval_score(run_dir / "eval.jsonl")
        legacy_step = last_eval_step(run_dir / "eval.jsonl")
        if pref_score is not None:
            score, step, score_source = pref_score, pref_step, pref_src
        else:
            score, step, score_source = legacy_score, legacy_step, "eval.jsonl"
        settings = meta.get("settings", {})
        rows.append(
            {
                **meta,
                "rel_path": str(run_dir.relative_to(ROOT)),
                "final_score": score,
                "last_eval_step": step,
                "score_source": score_source,
                "legacy_final_score": legacy_score,
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
