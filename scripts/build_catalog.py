#!/usr/bin/env python3
"""Refresh canonical catalog without mixing policies, runs, or backends."""
import json
from pathlib import Path
from log_layout import normalize, initial_label
ROOT=Path(__file__).resolve().parents[1]

def latest_eval(run):
    path=run/'eval.jsonl';last=None
    if not path.exists():return None
    for line in path.read_text(errors='replace').splitlines():
        try:r=json.loads(line)
        except (ValueError,TypeError):continue
        if r.get('policy_id') not in (None,'execution','adaptive_beta','adaptive'):continue
        step=r.get('step',r.get('t',r.get('timestep',r.get('main_step'))))
        if isinstance(step,(int,float)) and (last is None or step>=last):last=step
    return last

def write_catalog(root, rows):
    cat=root/'catalog';cat.mkdir(exist_ok=True)
    rows=sorted(rows,key=lambda r:(r['section'],r['method'],r['env'],r.get('meta_lr') or 0,str(r['seed']),r['run_id']))
    (cat/'catalog.json').write_text(json.dumps({'layout_version':2,'runs':rows},indent=2,sort_keys=True)+'\n')
    lines=['# 실험 로그','', '| 구분 | 방법 | 환경 | 초기값 | lr | seed | backend | 마지막 평가 step | 로그 |','|---|---|---|---|---|---:|---|---:|---|']
    for r in rows:
        lr=r.get('meta_lr');lr='—' if lr is None else str(lr)
        lines.append(f"| {r['section']} | {r['method']} | {r['env']} | {initial_label(r)} | {lr} | {r['seed']} | {r['backend']} | {r.get('last_eval_step') or '—'} | [{r['run_id']}](../{r['rel_path']}/) |")
    (cat/'INDEX.md').write_text('\n'.join(lines)+'\n')

def build():
    normalize(ROOT);rows=[]
    for section in ('main','ablation'):
        for p in (ROOT/section).rglob('run_meta.json'):
            m=json.loads(p.read_text())
            if m.get('is_alias'):continue
            m['last_eval_step']=latest_eval(p.parent)
            rows.append(m)
    write_catalog(ROOT,rows)
    print('catalog:',len(rows),'runs')
if __name__=='__main__':build()
