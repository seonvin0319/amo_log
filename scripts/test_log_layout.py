#!/usr/bin/env python3
"""Regression checks for main initializations, Adroit and stored-run migration."""
import json
import tempfile
import unittest
from pathlib import Path

import build_catalog
from log_layout import classify, normalize
from validate_logs import SHARED, validate
from alpha_logs import convert


def run(algo='amo', env='hopper-medium-v2', **settings):
    family = 'adaptive_multiscale' if algo == 'amo' else 'amo_bpi' if algo == 'iql_amo' else 'fql_amo_jax' if algo == 'fql_amo' else 'benchmark'
    cfg = dict(env=env, seed=0)
    if algo == 'amo':
        cfg.update(T_E=1, T_B=1, T_lr=0.001)
    elif algo == 'iql_amo':
        cfg.update(beta_initial=1, rho_lr=0.001)
    cfg.update(settings)
    meta = dict(algo=algo, family=family, run_id='trial', source_path='/source/trial',
                git={'code_commit': None}, settings=cfg, backend='torch')
    return meta, cfg


class ClassificationTests(unittest.TestCase):
    def test_td3_initializations_and_alpha_conversion(self):
        for alpha, t in ((1, .5), (2, 1), (5, 2.5)):
            for lr in (.001, .002, .0003):
                with self.subTest(alpha=alpha, t=t, lr=lr):
                    m, c = run(T_E=t, T_B=t, T_lr=lr)
                    self.assertEqual(classify(m, c)['section'], 'main')
        for t in (0, 1.25, 2, 5, 10, None):
            with self.subTest(rejected_t=t):
                m, c = run(T_E=t, T_B=t)
                self.assertIn('initial_scale_outside_main', classify(m, c)['classification_reasons'])

    def test_iql_initializations(self):
        for beta in (1, 2, 5, '5'):
            m, c = run('iql_amo', beta_initial=beta)
            self.assertEqual(classify(m, c)['section'], 'main')
        for beta in (.5, 2.5, 3, 10, None):
            m, c = run('iql_amo', beta_initial=beta)
            self.assertIn('initial_beta_outside_main', classify(m, c)['classification_reasons'])

    def test_unequal_scales_and_default_tb(self):
        m, c = run(T_E=.5, T_B=2.5)
        self.assertIn('initial_scale_mismatch', classify(m, c)['classification_reasons'])
        m, c = run(T_E=2.5, T_B=None)
        self.assertEqual(classify(m, c)['section'], 'main')
        c.pop('T_E')
        c['T_init'] = .5
        self.assertEqual(classify(m, c)['section'], 'main')

    def test_fql_initializations(self):
        for alpha in (1, 2, 5):
            for lr in (.001, .002, .0003):
                m, c = run('fql_amo', alpha_E=alpha, alpha_B=alpha, alpha_lr=lr)
                m['family'] = 'fql_amo_jax'
                self.assertEqual(classify(m, c)['section'], 'main')
        m, c = run('fql_amo', alpha_E=10, alpha_B=10, alpha_lr=0.001)
        m['family'] = 'fql_amo_jax'
        self.assertIn('initial_scale_outside_main', classify(m, c)['classification_reasons'])
        m, c = run('fql_amo', alpha_E=5, alpha_B=5, alpha_lr=0.001)
        m['family'] = 'other_fql'
        self.assertIn('method_variant:other_fql', classify(m, c)['classification_reasons'])

    def test_adroit_all_methods_and_datasets(self):
        for algo in ('amo', 'iql_amo', 'fql_amo', 'td3bc', 'iql', 'a2pr', 'wpc', 'aspc'):
            for task in ('door', 'hammer', 'pen', 'relocate'):
                for dataset in ('human', 'cloned', 'expert'):
                    m, c = run(algo, task+'-'+dataset+'-v1')
                    if algo == 'iql':
                        m['family'] = 'vanilla'
                    if algo == 'fql_amo':
                        m['family'] = 'fql_amo_jax'
                        c.update(alpha_E=5, alpha_B=5, alpha_lr=0.001)
                    result = classify(m, c)
                    self.assertEqual(result['section'], 'ablation')
                    self.assertIn('adroit', result['classification_reasons'])

    def test_other_ablation_rules_still_apply(self):
        cases = [dict(T_lr=.0002), dict(critic_layernorm=False),
                 dict(critic_depth=2), dict(proximal_n_steps=4),
                 dict(execution_l1=True), dict(T_schedule='linear'),
                 dict(T_B_from_T_E_divisor=2)]
        for changes in cases:
            m, c = run(T_E=2.5, T_B=2.5, **changes)
            self.assertEqual(classify(m, c)['section'], 'ablation')
        m, c = run('iql_amo', beta_initial=5)
        m['family'] = 'adaptive_beta'
        self.assertEqual(classify(m, c)['section'], 'ablation')
        # Baseline alpha/beta values are not AMO initialization rules.
        m, c = run('aspc', alpha=2.5, beta_initial=3)
        self.assertEqual(classify(m, c)['section'], 'main')


class MigrationTests(unittest.TestCase):
    def test_existing_and_new_runs_move_with_original_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [
                ('main', run('aspc', 'door-human-v1')),
                ('ablation', run('iql_amo', beta_initial=5)),
                ('ablation', run(T_E=2.5, T_B=2.5)),
                ('ablation', run(T_E=.5, T_B=.5, critic_depth=2)),
                ('runs', run('a2pr', 'pen-expert-v1')),
            ]
            expected = []
            for i, (old_section, (m, c)) in enumerate(cases):
                m['run_id'] = 'trial_'+str(i)
                m['source_path'] += str(i)
                layout = classify(m, c)
                old = '/'.join([old_section] + layout['rel_path'].split('/')[1:])
                m.update(layout, section=old_section, rel_path=old, layout_version=2)
                m['classification_reasons'] = ['old_rule']
                src = root/old
                src.mkdir(parents=True)
                original = {'config.yaml': json.dumps(c).encode(),
                            'eval.jsonl': b'{"step": 1000000, "score": 42}\n',
                            'metrics.jsonl': b'{"step": 1000000, "loss": 0.25}\n'}
                for name, content in original.items():
                    (src/name).write_bytes(content)
                (src/'run_meta.json').write_text(json.dumps(m))
                expected.append((layout, original, m['source_path']))
            self.assertEqual(normalize(root), len(cases))
            for layout, originals, source in expected:
                dst = root/layout['rel_path']
                for name, content in originals.items():
                    if name=='config.yaml' and layout['method']=='td3_amo':
                        import yaml
                        self.assertEqual(yaml.safe_load((dst/name).read_text()), convert(json.loads(content),config=True))
                    else:
                        self.assertEqual((dst/name).read_bytes(), content)
                m = json.loads((dst/'run_meta.json').read_text())
                for key, value in layout.items():
                    self.assertEqual(m[key], value)
                self.assertEqual(m['source_path'], source)
                self.assertEqual(m['run_id'], dst.name)
            self.assertFalse((root/'runs').exists())
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            normalize(root)
            after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            saved_root = build_catalog.ROOT
            try:
                build_catalog.ROOT = root
                build_catalog.build()
            finally:
                build_catalog.ROOT = saved_root
            rows = json.loads((root/'catalog/catalog.json').read_text())['runs']
            self.assertEqual(len(rows), len(cases))
            paths = {str(p.relative_to(root)): '' for p in root.rglob('*') if p.is_file()}
            paths.update({p: '' for p in SHARED})
            contents = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(validate(paths, contents, 'ext_csv'), [])

    def test_validator_rejects_old_main_adroit_and_wrong_initialization(self):
        for m, c in (run('aspc', 'door-human-v1'), run(T_E=5, T_B=5), run('iql_amo', beta_initial=3)):
            layout = classify(m, c)
            m.update(layout, layout_version=2)
            m['section'] = 'main'
            m['classification_reasons'] = []
            m['rel_path'] = 'main/' + layout['rel_path'].split('/', 1)[1]
            parent = m['rel_path']
            paths = {p: '' for p in SHARED}
            paths.update({parent+'/config.yaml': '', parent+'/run_meta.json': '', 'catalog/catalog.json': ''})
            contents = {parent+'/run_meta.json': json.dumps(m).encode(),
                        'catalog/catalog.json': json.dumps({'layout_version': 2, 'runs': [m]}).encode()}
            self.assertTrue(validate(paths, contents, 'ext_csv'))


if __name__ == '__main__':
    unittest.main()
