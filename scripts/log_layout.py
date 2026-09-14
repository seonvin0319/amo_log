#!/usr/bin/env python3
"""Canonical amo_log layout. Historical exclusions identify runs, never a backend."""
import hashlib
import json
import re
import shutil
from pathlib import Path

METHODS = ('td3_amo', 'iql_amo', 'td3bc+rc', 'iql', 'a2pr', 'wpc', 'aspc')
MAIN_LRS = (0.001, 0.002, 0.0003)

def number(x):
    try: return float(x)
    except (ValueError, TypeError): return None

def slug(x):
    return re.sub(r'[^A-Za-z0-9_.+\-]', '_', str(x))

def lr_name(x):
    n = number(x)
    if n is None: return 'unspecified'
    a,b = format(n, '.10e').split('e')
    return a.rstrip('0').rstrip('.') + 'e' + str(int(b))

def config(meta, original=None):
    original = original or {}
    if 'corl' in original:
        original = {**original.get('corl', {}), **original.get('meta', {})}
    return {**meta.get('settings', {}), **original}

def backend(m, c):
    explicit = str(c.get('backend', m.get('backend', ''))).lower()
    if explicit in ('torch', 'pytorch'): return 'torch'
    if explicit == 'jax': return 'jax'
    labels = ' '.join(str(m.get(k, '')) for k in ('family','variant','legacy_name','source_path'))
    labels += ' ' + str(m.get('git', {}).get('code_repo', ''))
    if 'jax' in labels.lower() or c.get('algorithm') in ('td3_amo','iql_amo'):
        return 'jax'
    return 'torch'

def identity(m, c):
    env = c.get('env', c.get('env_name', c.get('env_id', m.get('env'))))
    seed = c.get('seed', m.get('seed'))
    if env and env != 'unknown': return slug(env), seed
    name = ' '.join(str(m.get(k,'')) for k in ('legacy_name','source_path','run_id'))
    match = re.search(r'(halfcheetah|hopper|walker2d|antmaze|door|hammer|pen|relocate)-[a-z-]+-v[012]', name)
    if match: env = match[0]
    else:
        match = re.search(r'(?:_|/)(hc|h|w)-(mr|me|m)_s', name)
        if match:
            env = {'hc':'halfcheetah','h':'hopper','w':'walker2d'}[match[1]]+'-'+{'mr':'medium-replay','me':'medium-expert','m':'medium'}[match[2]]+'-v2'
        else:
            match = re.search(r'(door|hammer|pen|relocate)-(hum|cln|exp)', name)
            if match: env=match[1]+'-'+{'hum':'human','cln':'cloned','exp':'expert'}[match[2]]+'-v1'
    sm = re.search(r'_s(\d+)(?:_|\b)', name)
    if sm: seed = int(sm[1])
    return slug(env or 'unknown'), seed

def classify(m, c):
    algo, fam = m.get('algo',''), m.get('family','')
    reasons=[]
    if algo == 'iql_amo' or algo=='iql' and fam not in ('benchmark','vanilla'): method='iql_amo'
    elif algo in ('amo','apart','td3_amo'): method='td3_amo'
    elif algo=='td3bc': method='td3bc+rc'
    elif algo in METHODS: method=algo
    else: raise ValueError('Unmapped method: '+repr((algo,fam)))
    lr = c.get('T_lr') if method=='td3_amo' else c.get('rho_lr', c.get('beta_lr')) if method=='iql_amo' else None
    if method=='td3_amo':
        if algo=='apart' or fam not in ('adaptive_multiscale','td3_amo_jax','adroit','adroit_T1_Tlr1e3','antmaze_t_init_tune'):
            reasons.append('method_variant:'+fam)
        te=number(c.get('T_E',c.get('T_init')));tb=number(c.get('T_B'))
        if tb is None: tb=te  # reference implementation defaults T_B to T_E
        if te!=1 or tb!=1: reasons.append('initial_scale_not_1')
        if c.get('T_B_from_T_E_divisor') not in (None,1,1.0):reasons.append('scale_ratio')
        if c.get('T_schedule') not in (None,'','none','learned'):reasons.append('scale_schedule')
        if c.get('proximal_n_steps',1)!=1:reasons.append('multi_step')
        if c.get('critic_layernorm',True) is False or c.get('critic_n_hiddens',c.get('critic_depth',3))!=3:reasons.append('critic_architecture')
        if c.get('normalize_q',True) is False:reasons.append('q_normalization')
        if c.get('execution_l1',False) or c.get('execution_outer_loss_version') in ('l1','l1e') or 'l1e' in m.get('variant',''):reasons.append('execution_loss')
        if c.get('bootstrap_outer_loss_version') not in (None,'tq_detached_rms_target_v1'):reasons.append('bootstrap_loss')
    elif method=='iql_amo':
        if number(c.get('beta_initial'))!=1:reasons.append('initial_beta_not_1')
        if fam not in ('amo_bpi','iql_amo_jax_adroit_beta1_rho','lr1e3_beta_sweep'):reasons.append('method_variant:'+fam)
    elif method=='aspc' and c.get('l3_mode','aspc')!='aspc': reasons.append('l3_variant')
    elif method=='wpc' and number(c.get('policy_noise',.2))!=.2:reasons.append('policy_noise')
    if method in ('td3_amo','iql_amo') and number(lr) not in MAIN_LRS:reasons.append('meta_lr_outside_main')
    env, seed=identity(m,c)
    if env=='unknown' or seed is None:reasons.append('identity_unresolved')
    section='ablation' if reasons else 'main'
    parts=[section,method,env]
    if method in ('td3_amo','iql_amo'):parts.append(lr_name(lr))
    parts += ['seed_'+str(seed) if seed is not None else 'seed_unknown', slug(m.get('run_id') or 'run')]
    return dict(section=section,method=method,env=env,seed=seed,meta_lr=number(lr),classification_reasons=reasons,rel_path='/'.join(parts),backend=backend(m,c))

def fingerprint(m):
    return {'source_path':m.get('source_path'), 'code_commit':m.get('git',{}).get('code_commit'), 'run_id':m.get('run_id')}

def excluded(m, c, entries):
    if backend(m,c)!='jax':return False
    f=fingerprint(m)
    return any(e.get('source_path')==f['source_path'] and e.get('code_commit')==f['code_commit'] and (f['source_path'] or e.get('run_id')==f['run_id']) for e in entries)

def load_config(path,m):
    if not path.exists():return config(m)
    try:
        import yaml
        return config(m,yaml.safe_load(path.read_text()) or {})
    except ImportError:
        return config(m)

def normalize(root=None):
    root=Path(root or Path(__file__).resolve().parents[1]);legacy=root/'runs'
    ef=root/'catalog/removed_legacy_jax.json'
    entries=json.loads(ef.read_text()).get('runs',[]) if ef.exists() else []
    paths=sorted(legacy.rglob('run_meta.json')) if legacy.exists() else []
    for mp in paths:
        m=json.loads(mp.read_text());src=mp.parent
        if m.get('is_alias') and not m.get('algo'):continue
        c=load_config(src/'config.yaml',m)
        if excluded(m,c,entries):shutil.rmtree(src);continue
        layout=classify(m,c);dest=root/layout['rel_path']
        m.update(layout);m.setdefault('original_rel_path',src.relative_to(root).as_posix());m['layout_version']=2
        if dest.exists():
            old=json.loads((dest/'run_meta.json').read_text())
            if old.get('source_path')!=m.get('source_path'):
                raise ValueError('Run-id collision at '+str(dest))
        dest.mkdir(parents=True,exist_ok=True)
        for f in src.iterdir():
            if f.name=='run_meta.json':continue
            if not f.is_file():raise ValueError('Unexpected nested artifact: '+str(f))
            shutil.copy2(f,dest/f.name)
        (dest/'run_meta.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
        shutil.rmtree(src)
    # Alias stubs carry no observations; canonical metadata retains alias names.
    for mp in sorted(legacy.rglob('run_meta.json')) if legacy.exists() else []:
        if json.loads(mp.read_text()).get('is_alias') and not json.loads(mp.read_text()).get('algo'):shutil.rmtree(mp.parent)
    if legacy.exists():
        for p in sorted(legacy.rglob('*'),key=lambda p:len(p.parts),reverse=True):
            if p.is_dir() and not any(p.iterdir()):p.rmdir()
        if not any(legacy.iterdir()):legacy.rmdir()
    return sum(1 for section in ('main','ablation') for _ in (root/section).rglob('run_meta.json'))

if __name__=='__main__':
    print('Canonical runs:',normalize())
