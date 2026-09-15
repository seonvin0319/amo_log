#!/usr/bin/env python3
"""Unit semantics, mixed schemas, fresh ingestion and repeat-conversion checks."""
import json
import tempfile
import unittest
from pathlib import Path
import yaml
from alpha_logs import convert, normalize_run, legacy_keys, rewrite
from log_layout import classify, initial_alphas


class AlphaLogsTests(unittest.TestCase):
    def test_scales_and_non_scales(self):
        old={'T_E':.5,'T_B':2.5,'T_lr':.001,'T_freq':10,
             'T_schedule_start':1,'T_schedule_end':10,'T_schedule_steps':1000000,
             'T_B_from_T_E_divisor':2,'project_T_B_to_T_E':False}
        new=convert(old,config=True)
        self.assertEqual(new,{'alpha_E':1.,'alpha_B':5.,'alpha_lr':.001,'alpha_freq':10,
                             'alpha_schedule_start':2,'alpha_schedule_end':20,'alpha_schedule_steps':1000000,
                             'alpha_B_from_alpha_E_divisor':2,'project_alpha_B_to_alpha_E':False})
        self.assertEqual(convert(new,config=True),new)

    def test_log_scales_gradients_losses_and_scores(self):
        old={'step':200,'amo/T':1.25,'amo/T_E':1,'amo/T_B_raw':2.5,
             'amo/T_B_over_T_E':2.5,'amo/L_T_B':3.5,'amo/grad_T_E':-4,
             'amo/delta_log_T_E':.01,'return':123.4,'alpha':2.5,
             'nested':{'T_used':2,'T_next':3,'rho_next':.25},'config':{'T_init':5}}
        new=convert(old)
        self.assertEqual(new,{'step':200,'amo/alpha':2.5,'amo/alpha_E':2,'amo/alpha_B_raw':5.,
                             'amo/alpha_B_over_alpha_E':2.5,'amo/L_alpha_B':3.5,
                             'legacy_T/amo/grad_T_E':-4,'amo/delta_log_alpha_E':.01,
                             'return':123.4,'alpha':2.5,'nested':{'alpha_used':4,'alpha_next':6,'rho_next':.25},
                             'config':{'alpha_init':10}})
        self.assertEqual(convert(new),new)
        self.assertEqual(list(legacy_keys(new)),[])

    def test_conflicts_and_native_alpha(self):
        self.assertEqual(convert({'T_used':4,'T_projected':False}),{'alpha_used':8,'alpha_projected':False})
        self.assertEqual(convert({'apart/T_B_pre_projection':1.25}),{'apart/alpha_B_pre_projection':2.5})
        self.assertEqual(convert({'T_E':1,'alpha_E':2}),{'alpha_E':2})
        for old in ({'T_E':1,'alpha_E':5},{'alpha_E':5,'T_E':1}):
            with self.assertRaises(ValueError):convert(old)
        with self.assertRaises(ValueError):convert({'T_surprise':2},config=True)
        new={'alpha_E':5,'alpha_B':5,'alpha_lr':.001,'beta_initial':5}
        self.assertEqual(convert(new),new)
        self.assertEqual(initial_alphas(new),(5,5))
        m={'algo':'td3_amo','family':'td3_amo_jax','run_id':'native','backend':'jax'}
        self.assertEqual(classify(m,{**new,'env':'hopper-medium-v2','seed':0})['section'],'main')

    def test_ingestion_and_idempotence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cfg=root/'config.yaml';log=root/'metrics.jsonl'
            cfg.write_text('T_E: 1\nT_B: 1\nT_lr: 0.001\n')
            raw='{"step":200,"T_E":1.5,"score":42}\n'
            log.write_text(raw)
            meta={'algo':'amo','settings':{'T_E':1,'T_B':1,'T_lr':.001},
                  'run_id':'original_T1','source_path':'/original/T1','git':{'code_commit':'abc'}}
            first=normalize_run(root,meta)
            self.assertEqual(yaml.safe_load(cfg.read_text()),{'alpha_E':2,'alpha_B':2,'alpha_lr':.001})
            self.assertEqual(json.loads(log.read_text()),{'step':200,'alpha_E':3.,'score':42})
            self.assertEqual(first['run_id'],meta['run_id'])
            self.assertEqual(first['git'],meta['git'])
            self.assertEqual(normalize_run(root,first),first)
            # The uploader may overwrite a previously converted file with old T.
            log.write_text(raw+'{"step":400,"T_E":2,"score":43}\n')
            second=normalize_run(root,first)
            self.assertEqual([r['alpha_E'] for r in map(json.loads,log.read_text().splitlines())],[3.,4])
            self.assertEqual(normalize_run(root,second),second)
            # Native alpha and legacy T may coexist in a resumed JSONL stream.
            log.write_text(log.read_text()+'{"step":600,"alpha_E":5,"score":44}\n')
            third=normalize_run(root,second)
            self.assertEqual([r['alpha_E'] for r in map(json.loads,log.read_text().splitlines())],[3.,4,5])
            self.assertEqual(normalize_run(root,third),third)

    def test_other_methods_and_invalid_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'metrics.jsonl'
            raw='{"alpha":2.5,"beta":5,"T":3}\n';p.write_text(raw)
            for algo in ('aspc','iql','iql_amo','wpc','a2pr','td3bc'):
                m={'algo':algo,'settings':{'alpha':2.5,'beta_initial':5}}
                self.assertEqual(normalize_run(root,m),m)
                self.assertEqual(p.read_text(),raw)
            p.write_text('{"T_E":1}\n{"T_E":')
            rewrite(p)
            records=list(map(json.loads,p.read_text().splitlines()))
            self.assertEqual(records[0],{'alpha_E':2})
            self.assertEqual(records[1]['legacy_unparsed_record']['raw_line'],'{"T_E":')
            self.assertEqual(records[1]['legacy_unparsed_record']['source_line'],2)
            self.assertFalse(rewrite(p))
            self.assertEqual(list(root.glob('.alpha-*')),[])
            # Semantic conflicts still stop and leave the original file intact.
            p.write_text('{"T_E":1,"alpha_E":5}\n');before=p.read_bytes()
            with self.assertRaises(ValueError):rewrite(p)
            self.assertEqual(p.read_bytes(),before)


if __name__=='__main__':unittest.main()
