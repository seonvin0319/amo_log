"""Result-selection tests: policies, checkpoints, repeats, missing seeds and provenance."""
import json
import unittest

from test_log_layout import run
from log_layout import classify, network_lr_exclusion
from readme_results import (collect, check_source, eligible, record_step, render_results,
                            result_cell, row_stats, summarize)


def meta(algo='amo', seed=0, run_id='test', **settings):
    m,c = run(algo, seed=seed, **settings)
    m['run_id']=run_id
    m.update(classify(m,c),layout_version=2)
    return m


def jsonl(*rows):
    return '\n'.join(json.dumps(row) for row in rows)+'\n'


class ResultTests(unittest.TestCase):
    def test_checkpoint_selection_does_not_select_peak_or_repeat(self):
        rows=[dict(step=100000,d4rl_normalized_score=120),
              dict(step=1000000,d4rl_normalized_score=70),
              dict(step=1000000,d4rl_normalized_score=80,final_eval_aggregate=True,
                   final_eval_repeats=5,n_episodes=10),
              dict(step=1000000,d4rl_normalized_score=99,final_eval_repeat=4),
              dict(step=1100000,d4rl_normalized_score=200)]
        score=summarize(meta(),{'eval.jsonl':jsonl(*rows)})
        self.assertEqual((score['score'],score['step'],score['episodes']),(80,1000000,50))

    def test_adaptive_policy_and_zero_score(self):
        rows=[dict(step=1000000,policy_id='fixed_beta',mean_normalized=99),
              dict(step=1000000,policy_id='adaptive_beta',mean_normalized=0),
              dict(step=1000000,policy_id='adaptive_minus_fixed',mean_normalized_diff=-99)]
        self.assertEqual(summarize(meta('iql_amo'),{'eval.jsonl':jsonl(*rows)})['score'],0)
        td3=[dict(step=1000000,policy_id='bootstrap',normalized_score=100),
             dict(step=1000000,policy_id='execution',normalized_score=42)]
        self.assertEqual(summarize(meta(),{'eval.jsonl':jsonl(*td3)})['score'],42)

    def test_final_file_wrappers_dates_and_checkpoint_conventions(self):
        base=dict(step=1000000,normalized_score=60,episodes=50,repeats=5,evaluated_at='2026-09-15T00:00:00Z')
        later={**base,'normalized_score':61,'evaluated_at':'2026-09-15T01:00:00Z'}
        files={'eval.jsonl':jsonl(later), 'final_eval_50.json':json.dumps({'ok':True,'row':base})}
        self.assertEqual(summarize(meta(),files)['score'],61)
        legacy=dict(checkpoint='/run/checkpoint_999999.pt',d4rl_normalized_score=62,n_episodes_total=50)
        score=summarize(meta(),{'final_eval_50.json':json.dumps(legacy)})
        self.assertEqual((score['step'],score['score'],score['episodes']),(1000000,62,50))
        self.assertEqual(record_step({'checkpoint':'/run/step_1000000.npz'}),1000000)
        total=dict(step=1000000,d4rl_normalized_score=76,n_eval_episodes=50,n_eval_repeats=5,
                   eval_group_d4rl=[60,70,90,80,80])
        self.assertEqual(summarize(meta(),{'eval.jsonl':jsonl(total)})['episodes'],50)

    def test_duplicate_runs_and_missing_seeds_do_not_inflate_statistics(self):
        catalogs={};evaluations={};revisions={}
        def add(branch,seed,score,step=1000000,run_id='test'):
            m=meta(seed=seed,run_id=run_id)
            catalogs.setdefault(branch,[]).append(m);revisions[branch]=branch+'_sha'
            evaluations[(branch,m['rel_path']+'/eval.jsonl')]=jsonl(dict(step=step,normalized_score=score))
        add('ext_csh',0,1);add('ext_csv',0,2);add('choi',0,99,step=100000)
        add('ext_csv',1,4)
        runs,chosen=collect(catalogs,evaluations,revisions)
        cells=[chosen.get(('td3_amo',2,.001,'hopper-medium-v2',seed)) for seed in range(4)]
        self.assertEqual(cells[0]['score'],2)
        self.assertEqual(cells[0]['branch'],'ext_csv')
        self.assertEqual(row_stats(cells),(None,None,2))
        add('svcho',2,6);add('svcho',3,8,run_id='test3')
        runs,chosen=collect(catalogs,evaluations,revisions)
        cells=[chosen[('td3_amo',2,.001,'hopper-medium-v2',seed)] for seed in range(4)]
        mean,std,n=row_stats(cells)
        self.assertEqual((mean,n),(5,4));self.assertAlmostEqual(std,5**.5)
        self.assertEqual(sum(r['selected'] for r in runs),4)
        self.assertIn('ext_csv_sha',result_cell(cells[0]))
        self.assertIn('⁺3',result_cell(cells[0]))

    def test_raw_config_and_initialization_are_checked(self):
        m=meta('iql_amo',actor_lr=.0003)
        check_source(m,m['settings'],[])
        with self.assertRaises(ValueError):check_source(m,{'actor_lr':.001},[])
        with self.assertRaises(ValueError):check_source(m,{'beta_initial':5},[])
        with self.assertRaises(ValueError):check_source(m,m['settings'],[network_lr_exclusion(m,m['settings'],m['rel_path'])])
        self.assertFalse(eligible(meta('amo',env='door-human-v1')))

    def test_incomplete_missing_and_malformed_evaluation(self):
        m=meta()
        self.assertIsNone(summarize(m,{})['score'])
        for raw in ('not json',jsonl(dict(step=1000000,eval_return=100)),
                    jsonl(dict(step=1000000,normalized_score=float('nan')))):
            with self.assertRaises(ValueError):summarize(m,{'eval.jsonl':raw})
        score=summarize(m,{'eval.jsonl':jsonl({'legacy_unparsed_record':{'raw':'truncated'}},
                                            dict(step=300000,normalized_score=42))})
        self.assertEqual((score['score'],score['step']),(42,300000))

    def test_all_requested_grid_rows_are_rendered(self):
        text=render_results([],{}, {})
        rows=[line for line in text.splitlines() if line.startswith('| hopper-medium-v2 |')]
        self.assertEqual(len(rows),18) # 2 methods × 3 initializations × 3 lrs
        self.assertIn('| 2e-3 |',rows[0]);self.assertIn('| 1e-3 |',rows[1]);self.assertIn('| 3e-4 |',rows[2])
        self.assertIn('— (0/4)',rows[0])

    def test_bootrms_is_separate_from_original_for_all_initializations(self):
        catalogs={'svcho':[]}; evaluations={}; revisions={'svcho':'snapshot'}
        for init in (1,2,5):
            for seed in range(4):
                for loss,score in (('l1_l2_rms',40+seed),('l2_rms',70+seed)):
                    m=meta(seed=seed,run_id=f'{loss}-{init}-{seed}',
                           alpha_E=init,alpha_B=init,alpha_lr=.001,bootstrap_loss=loss,
                           execution_score='bpi')
                    if loss=='l2_rms':
                        m['family']='td3_amo_bootrms_maincand'
                        m.update(classify(m,m['settings']))
                    catalogs['svcho'].append(m)
                    evaluations[('svcho',m['rel_path']+'/eval.jsonl')]=jsonl(
                        dict(step=1000000,normalized_score=score))
        original,old=collect(catalogs,evaluations,revisions)
        bootrms,new=collect(catalogs,evaluations,revisions,'bootrms')
        self.assertEqual((len(original),len(bootrms)),(12,12))
        key=('td3_amo',5,.001,'hopper-medium-v2',0)
        self.assertEqual((old[key]['score'],new[key]['score']),(40,70))
        text=render_results(bootrms,new,revisions,'bootrms')
        self.assertIn('**71.50 ± 1.12**',text)
        self.assertIn('id="bootrms-td3_amo-5"',text)
        self.assertNotIn('id="td3_amo-5"',text)
        self.assertIn('reports/bootrms_runs.csv',text)
        self.assertEqual(sum(line.startswith('| hopper-medium-v2 |')
                             for line in text.splitlines()),9)

    def test_bootrms_requires_recorded_loss_and_rejects_other_ablations(self):
        m=meta(bootstrap_loss='l2_rms')
        self.assertFalse(eligible(m))
        self.assertTrue(eligible(m,'bootrms'))
        for overrides in ({'bootstrap_loss':'l1'}, {'critic_depth':2},
                          {'execution_score':'direct_q'}, {'alpha_B':5},
                          {'env':'door-human-v1'}, {'alpha_E':10,'alpha_B':10}):
            candidate=meta(bootstrap_loss='l2_rms')
            candidate['settings'].update(overrides)
            candidate.update(classify(candidate,candidate['settings']))
            self.assertFalse(eligible(candidate,'bootrms'),overrides)
        with self.assertRaises(ValueError):
            check_source(m,{'bootstrap_loss':'l1_l2_rms'},[])
        unknown=meta(bootstrap_loss='l2_rms')
        unknown['family']='unreviewed_variant'
        unknown.update(classify(unknown,unknown['settings']))
        self.assertFalse(eligible(unknown,'bootrms'))


if __name__=='__main__':
    unittest.main()
