#!/usr/bin/env python3
"""Build main-branch README from machine catalogs at fetched origin refs."""
import collections,json,subprocess
from pathlib import Path
BRANCHES=('choi','ext_csh','ext_csv','offrl','shchoi','svcho')
BASE='https://github.com/seonvin0319/amo_log'

def make_readme(catalogs):
    lines=['# amo_log','', '실험 로그는 각 머신 브랜치에 저장합니다. 이 `main` 브랜치는 전체 실험 위치를 안내합니다.','', '## 저장 규칙','', '- 본 실험: `main/<방법>/<환경>/<meta_lr>/seed_<seed>/<run_id>/`.','- Ablation: `ablation/<방법>/<환경>/<meta_lr>/seed_<seed>/<run_id>/`.','- Meta lr가 없는 baseline은 해당 경로 단계를 생략합니다.','- 방법: `td3_amo`, `iql_amo`, `td3bc+rc`, `iql`, `a2pr`, `wpc`, `aspc`.','- AMO 본 실험은 초기 T_E=T_B=1 또는 beta=1, meta lr `1e-3`, `2e-3`, `3e-4`입니다. 그 외 초기값·학습률·구조/loss 변형은 ablation입니다. Baseline 고유 beta는 이 초기값 제한 대상이 아닙니다.','- 기존 JAX 실행은 2026-09-14 정리에서 제거했습니다. Git 이력은 유지합니다. 이후 수정된 JAX 실행을 일괄 차단하지 않습니다.','- run_id를 보존해 같은 seed의 별도 실행을 덮어쓰지 않습니다.','- 설정·평가 방식은 각 실행의 `config.yaml`, `run_meta.json`을 확인합니다. 표는 위치 인덱스이며 점수를 선택/평균하지 않습니다.','', '## 머신별 카탈로그','', '| 브랜치 | main 실행 | ablation 실행 | 목록 |','|---|---:|---:|---|']
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
                if r['method']=='td3_amo':settings += ['TE='+str(c.get('T_E',c.get('T_init','?'))),'TB='+str(c.get('T_B') if c.get('T_B') is not None else c.get('T_E','?'))]
                if r['method']=='iql_amo':settings += ['beta0='+str(c.get('beta_initial','?'))]
                settings += ['actor_lr='+str(c.get('actor_lr',.0003))]
                if section=='ablation':settings += [r.get('family',''),','.join(r.get('classification_reasons',[]))]
                key=(r['method'],r['env'],str(r.get('meta_lr') or '—'),'; '.join(settings))
                parent=r['rel_path'].split('/seed_')[0]
                groups[key][(br,parent)].add(str(r['seed']))
        for key,locations in sorted(groups.items()):
            links=[f"[{br}: seed {','.join(sorted(seeds))}]({BASE}/tree/{br}/{p})" for (br,p),seeds in sorted(locations.items())]
            lines.append('| '+' | '.join([*key,'; '.join(links)])+' |')
    lines += ['', '## 새 로그 반영','', '머신에서 최신 브랜치를 받은 뒤 `python scripts/ingest_runs.py`와 `python scripts/build_catalog.py`를 실행합니다. 수집기는 임시 `runs/`를 표준 경로로 정규화합니다. 기존 제외 목록은 source_path + code_commit으로 한정하며 새 코드 commit의 JAX 로그는 허용합니다. commit이 없는 이전 경로를 재사용하려면 새 실행 ID/경로와 코드 commit을 기록하세요.','', '이 인덱스는 `.github/workflows/refresh-index.yml`에서 주기적으로 갱신하며 수동 실행도 가능합니다.','']
    return '\n'.join(lines)

def main():
    root=Path(__file__).resolve().parents[1];catalogs={}
    for br in BRANCHES:
        content=subprocess.check_output(['git','show',f'origin/{br}:catalog/catalog.json'],cwd=root,text=True)
        catalog=json.loads(content)
        if catalog.get('layout_version')!=2:raise ValueError('Unmigrated branch '+br)
        catalogs[br]=catalog['runs']
    (root/'README.md').write_text(make_readme(catalogs))
if __name__=='__main__':main()
