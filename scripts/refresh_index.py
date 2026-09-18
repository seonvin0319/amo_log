#!/usr/bin/env python3
"""Build main-branch README from machine catalogs at fetched origin refs."""
import argparse,collections,json,subprocess
import yaml
from pathlib import Path
from log_layout import initial_label, NETWORK_LR_EXCLUSIONS
from validate_logs import entries, blobs
from readme_results import EVAL_FILES, eligible, check_source, collect, render_results, write_csv
BRANCHES=('choi','ext_csh','ext_csv','offrl','shchoi','svcho')
BASE='https://github.com/seonvin0319/amo_log'

def make_readme(catalogs, result_text=''):
    lines=['# amo_log','', '[공통 로그 규칙](LOGGING_RULES.md) · [에이전트 작업 지침](AGENTS.md)', '', '실험 로그는 각 머신 브랜치에 저장합니다. 이 `main` 브랜치는 AMO 결과와 전체 실험 위치를 안내합니다.','', '## 저장 규칙','', '- 본 실험: `main/<방법>/<환경>/<meta_lr>/seed_<seed>/<run_id>/`.','- Ablation: `ablation/<방법>/<환경>/<meta_lr>/seed_<seed>/<run_id>/`.','- Meta lr가 없는 baseline은 해당 경로 단계를 생략합니다.','- 방법: `td3_amo`, `iql_amo`, `td3bc+rc`, `iql`, `a2pr`, `wpc`, `aspc`.','- Adroit(door/hammer/pen/relocate)는 baseline을 포함한 모든 방법에서 ablation입니다.','- AMO 본 실험 초기값: TD3 alpha_E=alpha_B=1/2/5, IQL beta=1/2/5. Meta lr는 `1e-3`, `2e-3`, `3e-4`입니다. 그 외 초기값·meta 학습률·구조/loss 변형은 ablation입니다. Baseline 고유 beta는 이 초기값 제한 대상이 아닙니다.','- actor/critic/value 등 기본 네트워크 lr를 바꾼 오실행은 삭제하며 재수집을 차단합니다. 고정 기준과 적용 필드는 공통 로그 규칙을 따릅니다.', '- 기존 JAX 실행은 2026-09-14 정리에서 제거했습니다. Git 이력은 유지합니다. 이후 수정된 JAX 실행을 일괄 차단하지 않습니다.','- run_id를 보존해 같은 seed의 별도 실행을 덮어쓰지 않습니다.','- 설정·평가 방식은 각 실행의 `config.yaml`, `run_meta.json`을 확인합니다. 아래 실행 위치 표는 위치 인덱스입니다. 점수 집계 기준은 위 AMO 결과를 따릅니다.','', '## 머신별 카탈로그','', '| 브랜치 | main 실행 | ablation 실행 | 목록 |','|---|---:|---:|---|']
    for br,rows in catalogs.items():
        n=collections.Counter(r['section'] for r in rows)
        lines.append(f"| [{br}]({BASE}/tree/{br}) | {n['main']} | {n['ablation']} | [전체 로그]({BASE}/blob/{br}/catalog/INDEX.md) |")
    for section in ('main','ablation'):
        lines += ['',f'## {section} 실험 위치','', '| 방법 | 환경 | meta lr | 설정 | 브랜치·시드·로그 위치 |','|---|---|---|---|---|']
        groups=collections.defaultdict(lambda:collections.defaultdict(set))
        for br,rows in catalogs.items():
            for r in rows:
                if r['section']!=section:continue
                c=r.get('settings',{});settings=[]
                if r['method']=='td3_amo':settings += [initial_label(r)]
                if r['method']=='iql_amo':settings += ['beta0='+str(c.get('beta_initial','?'))]
                settings += ['actor_lr='+str(c.get('actor_lr',.0003))]
                if section=='ablation':settings += [r.get('family',''),','.join(r.get('classification_reasons',[]))]
                key=(r['method'],r['env'],str(r.get('meta_lr') or '—'),'; '.join(settings))
                parent=r['rel_path'].split('/seed_')[0]
                groups[key][(br,parent)].add(str(r['seed']))
        for key,locations in sorted(groups.items()):
            links=[f"[{br}: seed {','.join(sorted(seeds))}]({BASE}/tree/{br}/{p})" for (br,p),seeds in sorted(locations.items())]
            lines.append('| '+' | '.join([*key,'; '.join(links)])+' |')
    lines += ['', '## 새 로그 반영','', '머신에서 최신 브랜치를 받은 뒤 `python scripts/ingest_runs.py`와 `python scripts/build_catalog.py`를 실행합니다. 카탈로그 생성기는 신규 `runs/`와 기존 `main/`·`ablation/`를 현재 분류 규칙으로 정규화합니다. 기존 제외 목록은 source_path + code_commit으로 한정하며 새 코드 commit의 JAX 로그는 허용합니다. commit이 없는 이전 경로를 재사용하려면 새 실행 ID/경로와 코드 commit을 기록하세요.','', '이 인덱스는 `.github/workflows/refresh-index.yml`에서 주기적으로 갱신하며 수동 실행도 가능합니다.','']
    if result_text:
        lines = lines[:6] + [result_text, '', '<details>',
                            '<summary>저장 규칙 · 전체 카탈로그 · main/ablation 실행 위치</summary>', ''] + lines[6:] + ['', '</details>', '']
    return '\n'.join(lines)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ref-prefix',default='refs/remotes/origin')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];catalogs={};revisions={};trees={}
    for br in BRANCHES:
        revisions[br]=subprocess.check_output(['git','rev-parse',f'{args.ref_prefix}/{br}'],cwd=root,text=True).strip()
        trees[br]=entries(root,revisions[br])
    cat_blobs=blobs(root,[tree['catalog/catalog.json'] for tree in trees.values()])
    for br in BRANCHES:
        catalog=json.loads(cat_blobs[trees[br]['catalog/catalog.json']])
        if catalog.get('layout_version')!=2:raise ValueError('Unmigrated branch '+br)
        catalogs[br]=catalog['runs']
    wanted={};source_files={}
    for br,rows in catalogs.items():
        if NETWORK_LR_EXCLUSIONS in trees[br]:source_files[(br,NETWORK_LR_EXCLUSIONS)]=trees[br][NETWORK_LR_EXCLUSIONS]
        for row in rows:
            if not (eligible(row) or eligible(row,'bootrms')):continue
            config_path=row['rel_path']+'/config.yaml'
            source_files[(br,config_path)]=trees[br][config_path]
            for name in EVAL_FILES:
                path=row['rel_path']+'/'+name
                if path in trees[br]:wanted[(br,path)]=trees[br][path]
    source_blobs=blobs(root,source_files.values())
    for br,rows in catalogs.items():
        ex_oid=source_files.get((br,NETWORK_LR_EXCLUSIONS))
        exclusions=json.loads(source_blobs[ex_oid])['runs'] if ex_oid else []
        for row in rows:
            if eligible(row) or eligible(row,'bootrms'):
                original=yaml.safe_load(source_blobs[source_files[(br,row['rel_path']+'/config.yaml')]]) or {}
                check_source(row,original,exclusions)
    eval_blobs=blobs(root,wanted.values())
    evaluations={key:eval_blobs[oid].decode() for key,oid in wanted.items()}
    runs,selected=collect(catalogs,evaluations,revisions)
    boot_runs,boot_selected=collect(catalogs,evaluations,revisions,'bootrms')
    result_text = (render_results(boot_runs,boot_selected,revisions,'bootrms')+'\n\n'
                   +render_results(runs,selected,revisions))
    (root/'README.md').write_text(make_readme(catalogs,result_text))
    write_csv(root,runs)
    write_csv(root,boot_runs,'bootrms_runs.csv')
    print(json.dumps({'amo_runs':len(runs),'seed_cells':len(selected),
                      'bootrms_runs':len(boot_runs),'bootrms_seed_cells':len(boot_selected),
                      'scored_cells':sum(r['score'] is not None for r in selected.values()),
                      'completed_1m_cells':sum(r['step']==1000000 for r in selected.values()),
                      'evaluation_bytes':sum(len(raw) for raw in eval_blobs.values())}))
if __name__=='__main__':main()
