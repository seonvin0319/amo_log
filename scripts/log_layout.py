#!/usr/bin/env python3
"""Canonical amo_log layout. Historical exclusions identify runs, never a backend."""
import hashlib
import json
import re
import shutil
from pathlib import Path
from alpha_logs import normalize_run

METHODS = ('td3_amo', 'iql_amo', 'td3bc+rc', 'iql', 'a2pr', 'wpc', 'aspc')
MAIN_LRS = (0.001, 0.002, 0.0003)
MAIN_ALPHAS = (1.0, 2.0, 5.0)
MAIN_TS = tuple(alpha / 2 for alpha in MAIN_ALPHAS)
MAIN_BETAS = (1.0, 2.0, 5.0)
ADROIT_TASKS = ('door', 'hammer', 'pen', 'relocate')

def number(x):
    try: return float(x)
    except (ValueError, TypeError): return None

def initial_ts(c):
    te = number(c.get('T_E', c.get('T_init')))
    tb = number(c.get('T_B'))
    return te, te if tb is None else tb

def initial_alphas(c):
    te, tb = initial_ts(c)
    ae = number(c.get('alpha_E', c.get('alpha_init')))
    if ae is None and te is not None: ae = 2 * te
    ab = number(c.get('alpha_B'))
    if ab is None: ab = 2 * tb if tb is not None else ae
    return ae, ab

def is_adroit(env):
    return str(env).split('-', 1)[0] in ADROIT_TASKS

def initial_label(m):
    c = m.get('settings', {})
    if m['method'] == 'td3_amo':
        te, tb = initial_alphas(c)
        label = lambda x: '?' if x is None else format(x, 'g')
        return 'alpha_E/alpha_B=' + label(te) + '/' + label(tb)
    if m['method'] == 'iql_amo':
        return 'beta=' + str(c.get('beta_initial', '?'))
    return '—'

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

def method_name(m):
    algo, fam = m.get('algo',''), m.get('family','')
    if algo == 'iql_amo' or algo=='iql' and fam not in ('benchmark','vanilla'): method='iql_amo'
    elif algo in ('amo','apart','td3_amo'): method='td3_amo'
    elif algo=='td3bc': method='td3bc+rc'
    elif algo in METHODS: method=algo
    else: raise ValueError('Unmapped method: '+repr((algo,fam)))
    return method

def classify(m, c):
    algo, fam = m.get('algo',''), m.get('family','')
    reasons=[]
    method=method_name(m)
    lr = c.get('T_lr') if method=='td3_amo' else c.get('rho_lr', c.get('beta_lr')) if method=='iql_amo' else None
    if method=='td3_amo' and ('alpha_E' in c or 'alpha_init' in c or
                             m.get('scale_conversion',{}).get('meta_lr_source')=='T_lr'):
        lr = c.get('alpha_lr', lr)
    if method=='td3_amo':
        if algo=='apart' or fam not in ('adaptive_multiscale','td3_amo_jax','adroit','adroit_T1_Tlr1e3','antmaze_t_init_tune'):
            reasons.append('method_variant:'+fam)
        te,tb=initial_alphas(c)
        if te not in MAIN_ALPHAS or tb not in MAIN_ALPHAS: reasons.append('initial_scale_outside_main')
        if te != tb: reasons.append('initial_scale_mismatch')
        if c.get('alpha_B_from_alpha_E_divisor',c.get('T_B_from_T_E_divisor')) not in (None,1,1.0):reasons.append('scale_ratio')
        if c.get('alpha_schedule',c.get('T_schedule')) not in (None,'','none','learned'):reasons.append('scale_schedule')
        if c.get('proximal_n_steps',1)!=1:reasons.append('multi_step')
        if c.get('critic_layernorm',True) is False or c.get('critic_n_hiddens',c.get('critic_depth',3))!=3:reasons.append('critic_architecture')
        if c.get('normalize_q',True) is False:reasons.append('q_normalization')
        if c.get('execution_l1',False) or c.get('execution_outer_loss_version') in ('l1','l1e') or 'l1e' in m.get('variant',''):reasons.append('execution_loss')
        if c.get('bootstrap_outer_loss_version') not in (None,'tq_detached_rms_target_v1'):reasons.append('bootstrap_loss')
    elif method=='iql_amo':
        if number(c.get('beta_initial')) not in MAIN_BETAS:reasons.append('initial_beta_outside_main')
        if fam not in ('amo_bpi','iql_amo_jax_adroit_beta1_rho','lr1e3_beta_sweep'):reasons.append('method_variant:'+fam)
    elif method=='aspc' and c.get('l3_mode','aspc')!='aspc': reasons.append('l3_variant')
    elif method=='wpc' and number(c.get('policy_noise',.2))!=.2:reasons.append('policy_noise')
    if method in ('td3_amo','iql_amo') and number(lr) not in MAIN_LRS:reasons.append('meta_lr_outside_main')
    env, seed=identity(m,c)
    if is_adroit(env):reasons.append('adroit')
    if env=='unknown' or seed is None:reasons.append('identity_unresolved')
    section='ablation' if reasons else 'main'
    parts=[section,method,env]
    if method in ('td3_amo','iql_amo'):parts.append(lr_name(lr))
    parts += ['seed_'+str(seed) if seed is not None else 'seed_unknown', slug(m.get('run_id') or 'run')]
    return dict(section=section,method=method,env=env,seed=seed,meta_lr=number(lr),classification_reasons=reasons,rel_path='/'.join(parts),backend=backend(m,c))

def fingerprint(m):
    return {'source_path':m.get('source_path'), 'code_commit':m.get('git',{}).get('code_commit'), 'run_id':m.get('run_id')}

NETWORK_LR_EXCLUSIONS = 'catalog/removed_invalid_network_lr.json'

def invalid_network_lrs(m, c):
    """Check explicit active-network rates, never meta rates or unused config knobs."""
    method=method_name(m)
    keys=['actor_lr','critic_lr','qf_lr']
    if method in ('iql','iql_amo','wpc','a2pr'):keys += ['value_lr','vf_lr']
    if method=='td3_amo':keys.append('behavior_lr')
    bad={}
    # Check both sources: neither an edited metadata field nor a config override
    # may hide a recorded nonstandard rate. Missing rates mean implicit defaults.
    for source, settings in (('settings',m.get('settings',{})),('config',c)):
        for key in keys:
            value=settings.get(key)
            if value is not None and number(value)!=0.0003:
                bad.setdefault(key,{'expected':0.0003})[source]=value
    return bad

def excluded_network_lr(m, entries):
    f=fingerprint(m)
    return any(e.get('code_commit')==f['code_commit'] and
               (e.get('source_path')==f['source_path'] if e.get('source_path') and f['source_path']
                else bool(f['run_id']) and e.get('run_id')==f['run_id']) for e in entries)

def network_lr_exclusion(m, c, rel_path):
    env,seed=identity(m,c)
    return dict(**fingerprint(m),method=method_name(m),env=env,seed=seed,
                rel_path=rel_path,reason='invalid_network_lr',invalid_network_lrs=invalid_network_lrs(m,c))

def write_network_lr_exclusions(root, entries):
    path=Path(root)/NETWORK_LR_EXCLUSIONS
    path.parent.mkdir(parents=True,exist_ok=True)
    data={'schema_version':1,'runs':sorted(entries,key=lambda e:(e.get('source_path') or '',e.get('code_commit') or '',e.get('run_id') or ''))}
    encoded=json.dumps(data,indent=2,sort_keys=True)+'\n'
    if not path.exists() or path.read_text()!=encoded:
        temp=path.with_suffix('.tmp');temp.write_text(encoded);temp.replace(path)

def excluded(m, c, entries):
    if backend(m,c)!='jax':return False
    f=fingerprint(m)
    return any(e.get('source_path')==f['source_path'] and e.get('code_commit')==f['code_commit'] and (f['source_path'] or e.get('run_id')==f['run_id']) for e in entries)

def load_config(path,m):
    if not path.exists():return config(m)
    try:
        import yaml
        return config(m,yaml.safe_load(path.read_text()) or {})
    except ImportError as exc:
        raise RuntimeError('Install log dependencies: python -m pip install -r requirements-log-tools.txt') from exc

def normalize(root=None):
    root=Path(root or Path(__file__).resolve().parents[1]);legacy=root/'runs'
    ef=root/'catalog/removed_legacy_jax.json'
    entries=json.loads(ef.read_text()).get('runs',[]) if ef.exists() else []
    nf=root/NETWORK_LR_EXCLUSIONS
    network_exclusions=json.loads(nf.read_text()).get('runs',[]) if nf.exists() else []
    # Reclassify stored runs as well as newly ingested runs after a rule change.
    # Legacy snapshots are applied last, retaining the existing ingestion order.
    paths=[mp for section in ('main','ablation','runs')
           for mp in sorted((root/section).rglob('run_meta.json'))]
    for mp in paths:
        m=json.loads(mp.read_text());src=mp.parent
        if m.get('is_alias') and not m.get('algo'):continue
        c=load_config(src/'config.yaml',m)
        if excluded(m,c,entries):
            if src.is_relative_to(legacy):shutil.rmtree(src);continue
            raise ValueError('Removed historical JAX present at '+str(src))
        bad=invalid_network_lrs(m,c)
        known=excluded_network_lr(m,network_exclusions)
        if bad or known:
            if not known:
                network_exclusions.append(network_lr_exclusion(m,c,src.relative_to(root).as_posix()))
                write_network_lr_exclusions(root,network_exclusions)
            print('Removed invalid network-lr run:',src.relative_to(root),sorted(bad) or 'previously excluded')
            shutil.rmtree(src)
            continue
        m=normalize_run(src,m)
        c=load_config(src/'config.yaml',m)
        layout=classify(m,c);dest=root/layout['rel_path']
        m.update(layout);m.setdefault('original_rel_path',src.relative_to(root).as_posix());m['layout_version']=2; m['settings']=c
        if dest != src and dest.exists():
            old=json.loads((dest/'run_meta.json').read_text())
            if (old.get('source_path'), old.get('backend'), old.get('git',{}).get('code_commit')) != (m.get('source_path'), m.get('backend'), m.get('git',{}).get('code_commit')):
                raise ValueError('Run-id collision at '+str(dest))
        if dest != src:
            dest.mkdir(parents=True,exist_ok=True)
            for f in src.iterdir():
                if f.name=='run_meta.json':continue
                if not f.is_file():raise ValueError('Unexpected nested artifact: '+str(f))
                shutil.copy2(f,dest/f.name)
        encoded=json.dumps(m,indent=2,sort_keys=True)+'\n'
        target=dest/'run_meta.json'
        if not target.exists() or target.read_text()!=encoded:target.write_text(encoded)
        if dest != src:shutil.rmtree(src)
    # Alias stubs carry no observations; canonical metadata retains alias names.
    for mp in sorted(legacy.rglob('run_meta.json')) if legacy.exists() else []:
        if json.loads(mp.read_text()).get('is_alias') and not json.loads(mp.read_text()).get('algo'):shutil.rmtree(mp.parent)
    for section in ('main','ablation','runs'):
        base=root/section
        if base.exists():
            for p in sorted(base.rglob('*'),key=lambda p:len(p.parts),reverse=True):
                if p.is_dir() and not any(p.iterdir()):p.rmdir()
            if not any(base.iterdir()):base.rmdir()
    return sum(1 for section in ('main','ablation') for _ in (root/section).rglob('run_meta.json'))

if __name__=='__main__':
    print('Canonical runs:',normalize())
