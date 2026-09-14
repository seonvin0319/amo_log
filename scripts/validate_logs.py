#!/usr/bin/env python3
"""Validate a Git snapshot; raw evaluation files are never downloaded/read."""
import argparse,json,subprocess,os
from pathlib import Path
METHODS={'td3_amo','iql_amo','td3bc+rc','iql','a2pr','wpc','aspc'}
BRANCHES={'main','choi','ext_csh','ext_csv','offrl','shchoi','svcho'}
SHARED=('requirements-log-tools.txt','LOGGING_RULES.md','AGENTS.md','docs/COLLECTION_RULES.md','docs/NAMING.md','docs/AUTO_PUSH.md','scripts/log_layout.py','scripts/build_catalog.py','scripts/validate_logs.py','scripts/collect_logs.py','scripts/auto_push.sh','.github/workflows/validate-logs.yml')
def git(root,*args,input=None):
 return subprocess.check_output(['git',*args],cwd=root,input=input)
def entries(root,ref):
 out={}
 if ref==':':
  lines=git(root,'ls-files','--stage').decode().splitlines()
  for l in lines:
   h,p=l.split('\t',1);mode,sha,stage=h.split()
   if stage!='0':raise ValueError('Unmerged index: '+p)
   out[p]=sha
 else:
  for l in git(root,'ls-tree','-r',ref).decode().splitlines():
   h,p=l.split('\t',1);out[p]=h.split()[2]
 return out
def blobs(root,oids):
 oids=list(dict.fromkeys(oids))
 if not oids:return {}
 request=('\n'.join(oids)+'\n').encode()
 check=subprocess.check_output(['git','cat-file','--batch-check'],cwd=root,input=request,env={**os.environ,'GIT_NO_LAZY_FETCH':'1'}).decode()
 missing=[line.split()[0] for line in check.splitlines() if line.endswith('missing')]
 if missing:
  git(root,'-c','fetch.negotiationAlgorithm=noop','fetch','origin','--no-tags','--no-write-fetch-head','--filter=blob:none','--stdin',input=('\n'.join(missing)+'\n').encode())
 raw=git(root,'cat-file','--batch',input=request);pos=0;out={}
 for oid in oids:
  end=raw.index(b'\n',pos);h=raw[pos:end].split()
  if h[-1]==b'missing':raise ValueError('Missing object '+oid)
  size=int(h[2]);out[oid]=raw[end+1:end+size+1];pos=end+size+2
 return out

def validate(paths,contents,branch):
 errors=[];metas={}
 def fail(msg):errors.append(msg)
 for p in SHARED:
  if p not in paths:fail('Missing common file: '+p)
 for p in paths:
  if p.startswith(('runs/','fail/')):fail('Forbidden legacy path: '+p)
 if branch=='main':
  if any(p.startswith(('main/','ablation/')) for p in paths):fail('Git main must not store experiment logs')
  return errors
 if branch not in BRANCHES:fail('Unknown machine branch: '+branch)
 exclusion_path='catalog/removed_legacy_jax.json'
 excluded=json.loads(contents.get(exclusion_path,b'{"runs":[]}')).get('runs',[])
 for p in paths:
  if not p.endswith('/run_meta.json'):continue
  try:m=json.loads(contents[p])
  except (ValueError,KeyError):fail('Invalid metadata: '+p);continue
  parent=p.rsplit('/',1)[0];metas[parent]=m
  required=('layout_version','method','section','env','seed','backend','meta_lr','rel_path','run_id','source_path','settings','git')
  if any(k not in m for k in required):fail('Missing metadata fields: '+p);continue
  if m['layout_version']!=2:fail('Unsupported layout version: '+p)
  if m['method'] not in METHODS or m['section'] not in ('main','ablation'):fail('Invalid method/section: '+p);continue
  if m['backend'] not in ('torch','jax'):fail('Missing explicit backend: '+p)
  if not isinstance(m['seed'],int) or m['seed']<0:fail('Invalid seed: '+p)
  if m['env']=='unknown' or not m['env'].endswith(('-v0','-v1','-v2')):fail('Invalid environment: '+p)
  parts=[m['section'],m['method'],m['env']]
  if m['method'] in ('td3_amo','iql_amo'):
   lr=m['meta_lr']
   if lr is None:label='unspecified'
   else:
    try:
     a,b=format(float(lr),'.10e').split('e');label=a.rstrip('0').rstrip('.')+'e'+str(int(b))
    except (ValueError,TypeError):fail('Invalid lr: '+p);continue
   parts.append(label)
   if m['section']=='main':
    if lr not in (.001,.002,.0003):fail('Main meta lr: '+p)
    c=m['settings']
    if m['method']=='td3_amo':
     te=c.get('T_E',c.get('T_init'));tb=c.get('T_B');tb=te if tb is None else tb
     if te!=1 or tb!=1:fail('Main requires initial TE=TB=1: '+p)
    elif c.get('beta_initial')!=1:fail('Main requires initial beta=1: '+p)
    if m.get('classification_reasons'):fail('Ablation reasons in main: '+p)
  elif m['meta_lr'] is not None:fail('Baseline must not have meta lr: '+p)
  parts+=['seed_'+str(m['seed']),m['run_id']]
  if '/'.join(parts)!=parent or m['rel_path']!=parent:fail('Path/metadata mismatch: '+p)
  if parent+'/config.yaml' not in paths:fail('Missing original config: '+p)
  if m['backend']=='jax':
   if any(e.get('source_path')==m['source_path'] and e.get('code_commit')==m['git'].get('code_commit') for e in excluded):fail('Removed historical JAX reintroduced: '+p)
   if not m['git'].get('code_commit'):fail('New JAX requires code commit: '+p)
 for p in paths:
  if p.startswith(('main/','ablation/')) and p not in ('main/README.md','ablation/README.md'):
   if p.rsplit('/',1)[0] not in metas:fail('Artifact outside a run: '+p)
 try:catalog=json.loads(contents['catalog/catalog.json'])
 except (ValueError,KeyError):fail('Missing/invalid catalog');return errors
 if catalog.get('layout_version')!=2:fail('Catalog layout version')
 rows=catalog.get('runs',[]);by_path={r['rel_path']:r for r in rows}
 if len(rows)!=len(by_path):fail('Duplicate catalog paths')
 expected={p for p,m in metas.items() if not m.get('is_alias')}
 if set(by_path)!=expected:fail('Catalog run set differs from metadata')
 for p in expected & set(by_path):
  for key in ('method','section','env','seed','backend','meta_lr','run_id'):
   if by_path[p].get(key)!=metas[p].get(key):fail('Catalog mismatch '+key+': '+p)
 return errors

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',default='.');parser.add_argument('--ref',default='HEAD');parser.add_argument('--index',action='store_true');parser.add_argument('--branch');parser.add_argument('--common-ref');a=parser.parse_args()
 root=Path(a.root);ref=':' if a.index else a.ref;paths=entries(root,ref)
 wanted={p:o for p,o in paths.items() if p.endswith('/run_meta.json') or p in ('catalog/catalog.json','catalog/removed_legacy_jax.json')}
 data=blobs(root,wanted.values());contents={p:data[o] for p,o in wanted.items()}
 branch=a.branch or git(root,'branch','--show-current').decode().strip();errors=validate(paths,contents,branch)
 if a.common_ref:
  common=entries(root,a.common_ref)
  for p in SHARED:
   if paths.get(p)!=common.get(p):errors.append('Common file differs from '+a.common_ref+': '+p)
 if errors:
  print('\n'.join('ERROR: '+e for e in errors));raise SystemExit(1)
 print('PASS',branch,ref,'— layout, metadata, catalog and common rules')
if __name__=='__main__':main()
