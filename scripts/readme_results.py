#!/usr/bin/env python3
"""Main-branch AMO result tables, read exclusively from recorded evaluations."""
import collections
import csv
import json
import math
import re
import statistics
from pathlib import Path

from log_layout import (classify, config, excluded_network_lr, initial_alphas,
                        invalid_network_lrs, number)

BASE = 'https://github.com/seonvin0319/amo_log'
METHODS = ('td3_amo', 'iql_amo')
INITIALS = (1, 2, 5)
LRS = (.002, .001, .0003)
LR_LABELS = ('2e-3', '1e-3', '3e-4')
SEEDS = (0, 1, 2, 3)
TARGET_STEP = 1_000_000
EVAL_FILES = ('eval.jsonl', 'final_eval_50.json', 'final_eval_50.jsonl')
LOCOMOTION = tuple(f'{agent}-{dataset}-v2' for agent in ('halfcheetah','hopper','walker2d')
                   for dataset in ('medium','medium-replay','medium-expert'))
ANTMAZE = tuple(f'antmaze-{task}-v2' for task in ('umaze','umaze-diverse','medium-play',
                                               'medium-diverse','large-play','large-diverse'))
ENVIRONMENTS = LOCOMOTION + ANTMAZE


def finite(value):
    n = number(value)
    return n if n is not None and math.isfinite(n) and not isinstance(value, bool) else None


def eligible(meta):
    if meta.get('is_alias') or meta.get('method') not in METHODS or meta.get('section') != 'main':
        return False
    c = meta['settings']
    if invalid_network_lrs(meta, c):
        raise ValueError('Invalid network lr in main catalog: '+meta['rel_path'])
    layout = classify(meta, c)
    if any(layout[k] != meta[k] for k in ('section','method','env','seed','meta_lr','rel_path')):
        raise ValueError('Stale main catalog classification: '+meta['rel_path'])
    return meta['env'] in ENVIRONMENTS and meta['seed'] in SEEDS and meta['meta_lr'] in LRS


def initial(meta):
    return initial_alphas(meta['settings'])[0] if meta['method']=='td3_amo' else number(meta['settings']['beta_initial'])


def check_source(meta, original, exclusions):
    actual = config(meta, original)
    if invalid_network_lrs(meta, actual) or excluded_network_lr(meta, exclusions):
        raise ValueError('Excluded/misconfigured source run: '+meta['rel_path'])
    layout = classify(meta, actual)
    if any(layout[k] != meta[k] for k in ('section','method','env','seed','meta_lr','rel_path')):
        raise ValueError('Original config disagrees with catalog: '+meta['rel_path'])
    actual_initial = initial_alphas(actual)[0] if meta['method']=='td3_amo' else number(actual.get('beta_initial'))
    if actual_initial != initial(meta):
        raise ValueError('Original initialization disagrees with catalog: '+meta['rel_path'])


def records(text, filename):
    lines = [text] if filename.endswith('.json') else text.splitlines()
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise ValueError(f'Malformed evaluation {filename}:{line_no}') from exc
        if not isinstance(record, dict):
            raise ValueError(f'Expected evaluation object {filename}:{line_no}')
        if 'legacy_unparsed_record' in record:
            continue
        if 'ok' in record:
            if record['ok'] is not True:
                continue
            record = record.get('row', {})
        yield line_no, record


def record_step(record):
    for key in ('step','t','timestep','main_step'):
        n = finite(record.get(key))
        if n is not None and n >= 0 and n.is_integer():
            return int(n)
    # The legacy Torch writer stores checkpoint_{step-1}; JAX stores step_{step}.
    checkpoint = str(record.get('checkpoint', ''))
    match = re.search(r'(?:^|/)checkpoint_(\d+)\.pt$', checkpoint)
    if match:
        return int(match[1]) + 1
    match = re.search(r'(?:^|/)step_(\d+)\.npz$', checkpoint)
    return int(match[1]) if match else None


def episode_count(record, settings):
    for key in ('episode_count','episodes','n_episodes_total'):
        if finite(record.get(key)) is not None:
            return int(record[key])
    if record.get('episodes_per_repeat') is not None and record.get('repeats') is not None:
        return int(record['episodes_per_repeat']) * int(record['repeats'])
    if record.get('n_eval_episodes') is not None:
        # This writer records the total across eval_group_* repeats (5 x 10 = 50).
        return int(record['n_eval_episodes'])
    if record.get('n_episodes') is not None:
        repeats = record.get('final_eval_repeats', 1) if record.get('final_eval_aggregate') else 1
        return int(record['n_episodes']) * int(repeats)
    n = settings.get('n_episodes', settings.get('eval_episodes'))
    return int(n) if finite(n) is not None else None


def summarize(meta, files):
    candidates = []
    permitted = (None,'execution') if meta['method']=='td3_amo' else (None,'adaptive_beta','adaptive')
    for file_rank, filename in enumerate(EVAL_FILES):
        if filename not in files:
            continue
        for line_no, record in records(files[filename], filename):
            if record.get('policy_id') not in permitted or 'final_eval_repeat' in record:
                continue
            values = [(key, finite(record.get(key))) for key in
                      ('d4rl_normalized_score','normalized_score','mean_normalized')]
            values = [(k,v) for k,v in values if v is not None]
            if not values:
                # Do not silently treat returns, differences or corrupt scores as D4RL scores.
                raise ValueError(f'No finite normalized score: {meta["rel_path"]}/{filename}:{line_no}')
            key, score = values[0]
            if any(not math.isclose(score,v,rel_tol=1e-9,abs_tol=1e-9) for _,v in values[1:]):
                raise ValueError('Conflicting normalized score fields: '+meta['rel_path'])
            step = record_step(record)
            if step is None:
                raise ValueError('Missing evaluation step: '+meta['rel_path']+'/'+filename)
            aggregate = bool(record.get('final_eval_aggregate') or record.get('groups') or
                             (finite(record.get('repeats')) or 0) > 1 or
                             (finite(record.get('n_eval_repeats')) or 0) > 1 or
                             filename.startswith('final_eval_50'))
            candidates.append(dict(score=score,step=step,aggregate=aggregate,
                                   episodes=episode_count(record,meta['settings']),
                                   score_field=key,eval_file=filename,eval_line=line_no,
                                   evaluated_at=record.get('evaluated_at') or '',file_rank=file_rank))
    if not candidates:
        return dict(score=None,step=None,aggregate=False,episodes=None,
                    score_field=None,eval_file=None,eval_line=None,evaluated_at='')
    # Fixed 1M comparison; otherwise show the most advanced available checkpoint.
    at_target = [r for r in candidates if r['step']==TARGET_STEP]
    pool = at_target or candidates
    return max(pool,key=lambda r:(r['step'],r['aggregate'],r['evaluated_at'],r['file_rank'],r['eval_line']))


def collect(catalogs, evaluations, revisions):
    runs = []
    for branch, rows in catalogs.items():
        for meta in rows:
            if not eligible(meta):
                continue
            score = summarize(meta, {name:evaluations[(branch,meta['rel_path']+'/'+name)]
                                    for name in EVAL_FILES if (branch,meta['rel_path']+'/'+name) in evaluations})
            runs.append(dict(method=meta['method'],initial=int(initial(meta)),meta_lr=meta['meta_lr'],
                             env=meta['env'],seed=meta['seed'],branch=branch,backend=meta['backend'],
                             run_id=meta['run_id'],rel_path=meta['rel_path'],source_path=meta['source_path'],
                             code_commit=meta.get('git',{}).get('code_commit'),log_commit=revisions[branch],
                             **score))
    grouped = collections.defaultdict(list)
    for run in runs:
        grouped[cell_key(run)].append(run)
    selected = {}
    for key, candidates in grouped.items():
        # Re-runs share a seed, never count as extra independent seeds.
        ranked = sorted(candidates,key=lambda r:(r['branch'],r['run_id']))
        winner = max(ranked,key=lambda r:(r['step']==TARGET_STEP,r['step'] if r['step'] is not None else -1,
                                          r['score'] if r['score'] is not None else -math.inf))
        selected[key] = winner
        for candidate in candidates:
            candidate['selected'] = candidate is winner
            candidate['candidate_runs'] = len(candidates)
    return runs, selected


def cell_key(run):
    return (run['method'],run['initial'],run['meta_lr'],run['env'],run['seed'])


def row_stats(cells):
    complete = [c['score'] for c in cells if c and c['step']==TARGET_STEP and c['score'] is not None]
    if len(complete)!=len(SEEDS):
        return None, None, len(complete)
    return statistics.mean(complete), statistics.pstdev(complete), len(complete)


def result_cell(run):
    if run is None:
        return '—'
    label = '대기' if run['score'] is None else f'{run["score"]:.2f}'
    backend = {'torch':'T','jax':'J'}[run['backend']]
    label += f' {run["branch"]}/{backend}'
    # Pin links to the snapshot used for the numbers, even as a branch keeps updating.
    url = f'{BASE}/tree/{run["log_commit"]}/{run["rel_path"]}'
    cell = f'[{label}]({url})'
    if run['score'] is not None and run['step']!=TARGET_STEP:
        cell += f' †{run["step"]/1000:g}k'
    if run['candidate_runs']>1:
        cell += f' ⁺{run["candidate_runs"]}'
    return cell


def render_results(runs, selected, revisions):
    lines = ['## AMO 결과', '',
             '초기 **alpha/beta = 1, 2, 5**, **alpha_lr/beta_lr = 2e-3, 1e-3, 3e-4**별 결과입니다. '
             '각 셀은 **정규화 점수 · 출처 브랜치/backend**이며 클릭하면 해당 실행으로 이동합니다.', '',
             '- TD3는 초기 `alpha_E=alpha_B`, IQL은 `beta_initial`로 묶습니다. IQL의 `beta_lr`는 로그의 `rho_lr`입니다.',
             '- **1M checkpoint 평가**를 사용하며, 같은 checkpoint에 반복평가 평균이 있으면 우선합니다. 없으면 마지막 일반 평가를 사용합니다.',
             '- `†300k`처럼 표시한 값은 1M 평가가 없는 실행의 최신 점수입니다. `대기`는 실행은 있으나 평가가 없고, `—`는 업로드된 실행이 없습니다.',
             '- **평균±표준편차는 seed 0~3 모두 1M 평가가 있을 때만** 계산합니다. 표준편차는 네 시드 점수의 population std(ddof=0)입니다. `(n/4)`는 1M 평가를 확보한 시드 수입니다.',
             '- 같은 설정·시드의 재실행은 1M 결과 중 높은 점수를 표시합니다. 1M 결과가 없으면 가장 진행된 step을 우선합니다. `⁺n`은 후보 실행 수이며, 모든 후보와 채택 여부는 [실행별 CSV](reports/amo_runs.csv)에 남깁니다.',
             '- `T`=Torch, `J`=JAX. 평가 episode 수·집계 유형·코드 버전은 [실행별 CSV](reports/amo_runs.csv)에서 확인할 수 있습니다. 각 학습률은 별도 행이며 서로 섞어 평균내지 않습니다.',
             '- 현재 분류 규칙의 `main/`만 집계합니다. Adroit와 ablation, 기본 네트워크 lr가 잘못된 실행은 이 표에 포함하지 않습니다.', '',
             '| 방법 | 초기값 | 평가 있는 시드 칸 | 1M 평가 시드 칸 | 4시드 완료 환경×lr |',
             '|---|---:|---:|---:|---:|']
    for method in METHODS:
        for init in INITIALS:
            cells = [r for key,r in selected.items() if key[:2]==(method,init)]
            full = sum(row_stats([selected.get((method,init,lr,env,seed)) for seed in SEEDS])[2]==4
                       for lr in LRS for env in ENVIRONMENTS)
            title = 'TD3-AMO' if method=='td3_amo' else 'IQL-AMO'
            scale = 'alpha' if method=='td3_amo' else 'beta'
            anchor = f'{method}-{init}'
            lines.append(f'| [{title}](#{anchor}) | {scale}={init} | {sum(r["score"] is not None for r in cells)}/180 | '
                         f'{sum(r["step"]==TARGET_STEP for r in cells)}/180 | {full}/45 |')
    for method in METHODS:
        title, scale, lr_title = ('TD3-AMO','alpha','alpha_lr') if method=='td3_amo' else ('IQL-AMO','beta','beta_lr')
        lines += ['', f'## {title}', '']
        for init in INITIALS:
            lines += [f'<a id="{method}-{init}"></a>', '', '<details open>',
                      f'<summary><strong>{scale} = {init}</strong></summary>', '']
            for domain, environments in (('Locomotion',LOCOMOTION),('AntMaze',ANTMAZE)):
                lines += [f'### {title} · {scale}={init} · {domain}', '',
                          f'| 환경 | {lr_title} | seed 0 | seed 1 | seed 2 | seed 3 | 1M 평균 ± std |',
                          '|---|---|---|---|---|---|---|']
                for env in environments:
                    for lr, label in zip(LRS, LR_LABELS):
                        cells = [selected.get((method,init,lr,env,seed)) for seed in SEEDS]
                        mean,std,n = row_stats(cells)
                        summary = f'**{mean:.2f} ± {std:.2f}**' if mean is not None else f'— ({n}/4)'
                        lines.append('| '+' | '.join([env,label,*map(result_cell,cells),summary])+' |')
            lines += ['', '</details>', '']
    lines += ['## 집계한 브랜치', '', '| 브랜치 | 로그 snapshot | AMO main 실행 |', '|---|---|---:|']
    for branch,revision in revisions.items():
        lines.append(f'| {branch} | [{revision[:8]}]({BASE}/commit/{revision}) | {sum(r["branch"]==branch for r in runs)} |')
    lines += ['', '머신 브랜치의 로그 검증이 성공하면 이 표를 자동 갱신합니다. 30분 주기의 보완 갱신과 '
              f'[수동 갱신]({BASE}/actions/workflows/refresh-index.yml)도 지원합니다.', '']
    return '\n'.join(lines)


def write_csv(root, runs):
    path = Path(root)/'reports/amo_runs.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    fields = ('method','initial','meta_lr','env','seed','score','step','aggregate','episodes','selected',
              'candidate_runs','branch','backend','run_id','code_commit','log_commit','rel_path',
              'source_path','eval_file','eval_line','score_field','evaluated_at')
    with path.open('w',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
        writer.writeheader()
        for row in sorted(runs,key=lambda r:(METHODS.index(r['method']),INITIALS.index(r['initial']),
                                            ENVIRONMENTS.index(r['env']),LRS.index(r['meta_lr']),r['seed'],r['branch'],r['run_id'])):
            writer.writerow(row)
