#!/usr/bin/env python3
"""Collect on the current machine branch, validate the exact index, then push."""
import argparse,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
GIT=os.environ.get('GIT_BIN','git')
def run(*args):subprocess.run(args,cwd=ROOT,check=True)
def identity(key,fallback):
 r=subprocess.run([GIT,'config','--get',key],cwd=ROOT,text=True,capture_output=True)
 return r.stdout.strip() or fallback
def main():
 p=argparse.ArgumentParser();p.add_argument('--branch',required=True);a=p.parse_args()
 current=subprocess.check_output([GIT,'branch','--show-current'],cwd=ROOT,text=True).strip()
 if current!=a.branch or current not in ('choi','ext_csh','ext_csv','offrl','shchoi','svcho'):raise SystemExit('Wrong machine branch')
 run(sys.executable,'scripts/validate_logs.py','--branch',current,'--common-ref','origin/main')
 run(sys.executable,'scripts/ingest_runs.py')
 run(sys.executable,'scripts/build_catalog.py')
 run(GIT,'add','-A','--','main','ablation','catalog')
 run(sys.executable,'scripts/validate_logs.py','--index','--branch',current,'--common-ref','origin/main')
 if subprocess.run([GIT,'diff','--cached','--quiet'],cwd=ROOT).returncode:
  run(GIT,'-c','user.name='+identity('user.name','amo-log collector'),'-c','user.email='+identity('user.email','amo-log@localhost'),'commit','-m',f'collect({current}): refresh standardized logs')
 run(GIT,'push','origin',f'HEAD:refs/heads/{current}')
if __name__=='__main__':main()
