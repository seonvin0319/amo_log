"""Tests for generic flat log parser exports."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from parse_logs import collect, write_exports


def meta(method="td3_amo", family="td3_amo_fixed_alpha_B", run_id="run", **settings):
    defaults = dict(
        alpha_E=5, alpha_B=5, actor_lr=.0003, critic_lr=.0003,
        adaptive_scale_E=True, adaptive_scale_B=False,
        freeze_scale_E=False, freeze_scale_B=True,
        bootstrap_loss="l2_rms", execution_score="bpi",
    )
    defaults.update(settings)
    if method == "iql_amo":
        defaults.pop("alpha_E", None)
        defaults.pop("alpha_B", None)
        defaults.setdefault("beta_initial", 5)
        defaults.setdefault("value_lr", .0003)
    return {
        "section": "ablation",
        "method": method,
        "family": family,
        "env": "hopper-medium-v2",
        "seed": 0,
        "backend": "jax",
        "meta_lr": .002,
        "classification_reasons": ["method_variant:"+family],
        "run_id": run_id,
        "rel_path": f"ablation/{method}/hopper-medium-v2/2e-3/seed_0/{run_id}",
        "source_path": "/source/"+run_id,
        "source_host": "svcho",
        "git": {"code_commit": "abc"},
        "settings": defaults,
        "last_eval_step": 1_000_000,
    }


def write_eval(root, m, *rows):
    path = root / m["rel_path"]
    path.mkdir(parents=True)
    (path / "eval.jsonl").write_text("\n".join(json.dumps(row) for row in rows)+"\n")


class ParseLogTests(unittest.TestCase):
    def test_td3_prefers_execution_and_keeps_all_policy_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = meta()
            write_eval(
                root, m,
                dict(step=1_000_000, policy_id="bootstrap", normalized_score=99),
                dict(step=1_000_000, policy_id="execution", normalized_score=88,
                     episodes=50),
            )
            runs, evaluations = collect(root, [m])
            self.assertEqual(len(evaluations), 2)
            self.assertEqual(runs[0]["primary_policy_id"], "execution")
            self.assertEqual(runs[0]["primary_score"], 88)
            self.assertTrue(runs[0]["complete_1m"])

    def test_fixed_iql_prefers_fixed_beta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = meta(
                "iql_amo", "iql_amo_bpi_fixed_beta_E", "fixed",
                adaptive_scale_E=False, freeze_scale_E=True,
            )
            write_eval(
                root, m,
                dict(step=1_000_000, policy_id="adaptive_beta", normalized_score=77),
                dict(step=1_000_000, policy_id="fixed_beta", normalized_score=66),
            )
            runs, _ = collect(root, [m])
            self.assertEqual((runs[0]["primary_policy_id"], runs[0]["primary_score"]),
                             ("fixed_beta", 66))

    def test_malformed_eval_is_reported_without_dropping_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = meta(run_id="broken")
            path = root / m["rel_path"]
            path.mkdir(parents=True)
            (path / "eval.jsonl").write_text(
                '{"step": 1000000, "policy_id": "execution", "normalized_score": 42}\n'
                '{"step":'
            )
            runs, evaluations = collect(root, [m])
            self.assertEqual(len(runs), 1)
            self.assertEqual(len(evaluations), 1)
            self.assertEqual(runs[0]["primary_score"], 42)
            self.assertIn("eval.jsonl:2", runs[0]["eval_parse_errors"])

    def test_exports_and_family_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "catalog").mkdir()
            rows = [meta(run_id="a"), meta(run_id="b")]
            rows[1]["seed"] = 1
            rows[1]["rel_path"] = "ablation/td3_amo/hopper-medium-v2/2e-3/seed_1/b"
            for i, m in enumerate(rows):
                write_eval(root, m, dict(step=1_000_000, policy_id="execution",
                                         normalized_score=80+i))
            (root / "catalog" / "catalog.json").write_text(
                json.dumps({"layout_version": 2, "runs": rows})
            )
            runs, evaluations, families = write_exports(root)
            self.assertEqual((len(runs), len(evaluations), len(families)), (2, 2, 1))
            self.assertEqual(families[0]["complete_1m_runs"], 2)
            for name in ("runs_flat.csv", "evaluations_flat.csv", "families.csv", "FAMILIES.md"):
                self.assertTrue((root / "catalog" / name).exists(), name)
            with (root / "catalog" / "runs_flat.csv").open() as handle:
                parsed = list(csv.DictReader(handle))
            self.assertEqual({row["run_id"] for row in parsed}, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
