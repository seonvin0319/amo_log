#!/usr/bin/env python3
"""Regression checks for deleting misconfigured runs and rejecting re-ingestion."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from test_log_layout import run
from log_layout import (classify, normalize, invalid_network_lrs,
                        NETWORK_LR_EXCLUSIONS, network_lr_exclusion)
from validate_logs import SHARED, validate


class NetworkLrTests(unittest.TestCase):
    def test_active_networks_and_aliases(self):
        for algo in ('amo','iql_amo','iql','td3bc','a2pr','wpc','aspc'):
            for key in ('actor_lr','critic_lr','qf_lr'):
                m,c=run(algo,**{key:.001})
                self.assertIn(key,invalid_network_lrs(m,c))
        for algo in ('iql_amo','iql','wpc','a2pr'):
            for key in ('value_lr','vf_lr'):
                m,c=run(algo,**{key:.001})
                self.assertIn(key,invalid_network_lrs(m,c))
        m,c=run(behavior_lr=.001)
        self.assertIn('behavior_lr',invalid_network_lrs(m,c))

    def test_meta_rates_and_separate_defaults_are_preserved(self):
        for algo in ('amo','iql_amo','iql','a2pr','aspc'):
            m,c=run(algo,actor_lr='3e-4',critic_lr=.0003,value_lr=None,
                    T_lr=.002,alpha_lr=.002,rho_lr=.002,beta_lr=.001,
                    scale_lr=.002,lambda_lr=0,vae_lr=.001,cql_policy_lr='3e-05')
            self.assertEqual(invalid_network_lrs(m,c),{})
        m,c=run('iql_amo')
        self.assertEqual(invalid_network_lrs(m,c),{})

    def test_delete_main_ablation_and_incoming_without_changing_good_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def put(ident,section,actor_lr,commit='old'):
                m,c=run('iql_amo',actor_lr=actor_lr,critic_lr=.0003,value_lr=.0003)
                m.update(run_id=ident,source_path='/source/'+ident,git={'code_commit':commit})
                layout=classify(m,c)
                path='/'.join([section]+layout['rel_path'].split('/')[1:])
                m.update(layout,section=section,rel_path=path,layout_version=2)
                dst=root/path;dst.mkdir(parents=True)
                for name,data in {'run_meta.json':json.dumps(m),'config.yaml':json.dumps(c),
                                  'metrics.jsonl':'{"loss":0.25}\n','eval.jsonl':'{"score":42}\n'}.items():
                    (dst/name).write_text(data)
                return dst
            good=put('good','main',.0003)
            before={p.name:p.read_bytes() for p in good.iterdir() if p.name!='run_meta.json'}
            bad=[put('bad_'+s,s,.001) for s in ('main','ablation','runs')]
            self.assertEqual(normalize(root),1)
            self.assertTrue(all(not p.exists() for p in bad))
            self.assertEqual(before,{p.name:p.read_bytes() for p in good.iterdir() if p.name!='run_meta.json'})
            ledger=(root/NETWORK_LR_EXCLUSIONS).read_bytes()
            self.assertEqual(len(json.loads(ledger)['runs']),3)
            # Changing recorded rates cannot bring the same experiment back.
            reintroduced=put('bad_main','runs',.0003)
            self.assertEqual(normalize(root),1)
            self.assertFalse(reintroduced.exists())
            self.assertEqual((root/NETWORK_LR_EXCLUSIONS).read_bytes(),ledger)
            put('corrected_rerun','runs',.0003,commit='new')
            self.assertEqual(normalize(root),2)
            snapshot={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            normalize(root)
            self.assertEqual(snapshot,{str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_validator_reads_both_config_and_metadata_and_exclusions(self):
        m,c=run('iql_amo',actor_lr=.0003,critic_lr=.0003,value_lr=.0003)
        m.update(classify(m,c),layout_version=2)
        parent=m['rel_path']
        def errors(meta,original,exclusions=None):
            content={parent+'/run_meta.json':json.dumps(meta).encode(),
                     parent+'/config.yaml':json.dumps(original).encode(),
                     'catalog/catalog.json':json.dumps({'layout_version':2,'runs':[meta]}).encode()}
            if exclusions is not None:content[NETWORK_LR_EXCLUSIONS]=json.dumps({'runs':exclusions}).encode()
            return validate({**dict.fromkeys(SHARED,''),**dict.fromkeys(content,'')},content,'svcho')
        self.assertEqual(errors(m,c),[])
        bad=copy.deepcopy(c);bad['actor_lr']=.001
        self.assertTrue(any('Invalid network lr' in e for e in errors(m,bad)))
        self.assertTrue(any('Invalid network lr' in e for e in errors(m,{'corl':bad,'meta':{}})))
        bad_meta=copy.deepcopy(m);bad_meta['settings']['value_lr']=.001
        self.assertTrue(any('Invalid network lr' in e for e in errors(bad_meta,c)))
        exclusion=network_lr_exclusion(m,bad,parent)
        self.assertTrue(any('reintroduced' in e for e in errors(m,c,[exclusion])))


if __name__=='__main__':
    unittest.main()
