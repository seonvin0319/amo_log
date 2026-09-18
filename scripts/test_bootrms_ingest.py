import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('collector', Path(__file__).with_name('ingest_runs.py'))
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)

class BootrmsIngestTests(unittest.TestCase):
    def test_new_and_refreshed_final_eval_keep_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'export'
            src = Path(tmp) / 'source' / 'a5_h-m_r3e4_s0' / 'run'
            src.mkdir(parents=True)
            (src / 'config.yaml').write_text('algorithm: td3_amo\nbackend: jax\nalpha_E: 5\nalpha_B: 5\nalpha_lr: 0.0003\nbootstrap_loss: l2_rms\nexecution_score: bpi\nactor_lr: 0.0003\ncritic_lr: 0.0003\n')
            source_meta = {'env':'hopper-medium-v2','seed':0,'git':{'code_commit':'a'*40}}
            (src / 'run_meta.json').write_text(json.dumps(source_meta))
            final = src / 'posthoc_eval_cpu' / 'final.json'
            final.parent.mkdir()
            final.write_text(json.dumps({'step':1000000,'normalized_score':0,'episodes':50}))
            with patch.object(collector,'ROOT',root), patch.object(collector,'RUNS',root/'runs'):
                kwargs = dict(algo='td3_amo',host='svcho',code_repo='AMO',family_force='td3_amo_bootrms_maincand',dry_run=False)
                meta = collector.ingest_one(src,**kwargs)
                dest = root/'runs'/'td3_amo'/'td3_amo_bootrms_maincand'/meta['run_id']
                self.assertEqual(meta['git']['code_commit'],'a'*40)
                self.assertEqual(json.loads((dest/'source_run_meta.json').read_text()),source_meta)
                self.assertEqual((dest/'final_eval_50.json').read_text(),final.read_text())
                final.write_text(json.dumps({'step':1000000,'normalized_score':80,'episodes':50}))
                refreshed = collector.ingest_one(src,**kwargs)
                self.assertEqual(refreshed['run_id'],meta['run_id'])
                self.assertEqual((dest/'final_eval_50.json').read_text(),final.read_text())
                source_meta.pop('git')
                (src/'run_meta.json').write_text(json.dumps(source_meta))
                self.assertIsNone(collector.ingest_one(src,source_index={},**kwargs))

if __name__ == '__main__':
    unittest.main()
