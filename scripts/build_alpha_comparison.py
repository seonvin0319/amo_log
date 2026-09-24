#!/usr/bin/env python3
"""Build alpha-learning comparisons from machine-branch flat catalogs and metrics.

For each environment/condition/initialization, choose the meta learning rate
with maximum seed-mean 1M normalized score, preferring broader seed coverage
first. Duplicate seed cells keep the higher 1M score.
"""
from __future__ import annotations

import csv, io, json, math, statistics, subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ("choi","ext_csh","ext_csv","offrl","shchoi","svcho")
TARGET_STEP = 1_000_000
STEP_STRIDE = 25_000
PURE_LE_COMMIT = "d9be263443ffc1ca8834e277418826cff4ac6dc0"
CONDITIONS = ("single_le","main_dual","dual_le_fixed_b")
INITS = (1.0,5.0)

def finite(value):
    if value in (None,""): return None
    try: x=float(value)
    except (TypeError,ValueError): return None
    return x if math.isfinite(x) else None

def truthy(value):
    return str(value).strip().lower() in {"1","true","yes"}

def git_show(ref,path):
    p=subprocess.run(["git","show",f"{ref}:{path}"],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    return None if p.returncode else p.stdout.decode(errors="replace")

def csv_rows(text):
    return list(csv.DictReader(io.StringIO(text or "")))

def load_flat_runs():
    rows=[]
    for branch in BRANCHES:
        text=git_show(f"origin/{branch}","catalog/runs_flat.csv")
        if not text: continue
        for row in csv_rows(text):
            row=dict(row); row["branch"]=branch; rows.append(row)
    return rows

def domain(env):
    env=str(env)
    if env.startswith("antmaze-"): return "antmaze"
    if env.startswith(("halfcheetah-","hopper-","walker2d-")): return "locomotion"
    return "other"

def is_pure_single(row,init):
    source=row.get("source_path",""); family=row.get("family",""); branch=row.get("branch","")
    allowed=(
        branch=="ext_csh" and family=="td3_amo_execonly_main"
        and "td3_amo_execonly_a51_seeds03" in source
    ) or (
        branch=="svcho" and family=="td3_amo_execonly_le"
        and "td3_amo_execonly_le_a1_back85" in source
    )
    return (
        row.get("method")=="td3_amo" and allowed
        and row.get("code_commit")==PURE_LE_COMMIT
        and truthy(row.get("execution_only"))
        and row.get("execution_meta_loss")=="le"
        and finite(row.get("alpha_E_initial"))==init
        and finite(row.get("alpha_B_initial"))==init
    )

def is_main_dual(row,init):
    # Historical main runs predate explicit adaptive_scale_* metadata. The
    # canonical family + section identifies the dual adaptive BootRMS design.
    return (
        row.get("section")=="main" and row.get("method")=="td3_amo"
        and row.get("family")=="td3_amo_bootrms_maincand"
        and not truthy(row.get("execution_only"))
        and finite(row.get("alpha_E_initial"))==init
        and finite(row.get("alpha_B_initial"))==init
        and row.get("bootstrap_loss")=="l2_rms"
    )

def is_fixed_b(row,init):
    return (
        row.get("method")=="td3_amo"
        and row.get("family")=="td3_amo_fixed_alpha_B"
        and not truthy(row.get("execution_only"))
        and finite(row.get("alpha_E_initial"))==5.0
        and finite(row.get("alpha_B_initial"))==init
        and truthy(row.get("adaptive_scale_E"))
        and truthy(row.get("freeze_scale_B"))
        and row.get("execution_meta_loss")=="le"
        and row.get("bootstrap_loss")=="l2_rms"
    )

def candidates(rows,condition,init):
    pred={"single_le":is_pure_single,"main_dual":is_main_dual,"dual_le_fixed_b":is_fixed_b}[condition]
    out=[]
    for row in rows:
        if not pred(row,init): continue
        score=finite(row.get("primary_score")); step=finite(row.get("primary_eval_step"))
        lr=finite(row.get("meta_lr")); seed=finite(row.get("seed"))
        if score is None or step!=TARGET_STEP or lr is None or seed is None: continue
        x=dict(row); x["score"]=score; x["lr"]=lr; x["seed_num"]=int(seed); out.append(x)
    return out

def choose_best(rows):
    cell={}
    for row in rows:
        k=(row["env"],row["lr"],row["seed_num"])
        old=cell.get(k)
        if old is None or row["score"]>old["score"]: cell[k]=row
    grouped=defaultdict(list)
    for (env,lr,_),row in cell.items(): grouped[(env,lr)].append(row)
    selected=[]; summary=[]
    for env in sorted({e for e,_ in grouped}):
        choices=[]
        for (e,lr),items in grouped.items():
            if e!=env: continue
            vals=[x["score"] for x in items]
            choices.append((len(items),statistics.fmean(vals),lr,items))
        if not choices: continue
        n,avg,lr,items=max(choices,key=lambda x:(x[0],x[1],x[2]))
        selected.extend(items)
        summary.append({
            "env":env,"best_lr":lr,"n_seeds":n,"mean_score":avg,
            "seeds":";".join(str(x["seed_num"]) for x in sorted(items,key=lambda z:z["seed_num"]))
        })
    return selected,summary

def parse_metrics(text):
    out={}
    for line in (text or "").splitlines():
        if not line.strip(): continue
        try: rec=json.loads(line)
        except json.JSONDecodeError: continue
        step=finite(rec.get("step"))
        if step is None or step<0 or step>TARGET_STEP or int(step)%STEP_STRIDE: continue
        a_e=finite(rec.get("alpha_E")); a_b=finite(rec.get("alpha_B"))
        if a_e is None and a_b is None: continue
        out[int(step)]=(a_e,a_b)
    return out

def initial_value(condition,init,role):
    if condition=="dual_le_fixed_b": return 5.0 if role=="alpha_E" else init
    if role=="alpha_B" and condition=="single_le": return None
    return init

def aggregate(selected,condition,init):
    by_env=defaultdict(list); failures=[]
    for row in selected:
        text=git_show(f"origin/{row['branch']}",row["rel_path"]+"/metrics.jsonl")
        if not text:
            failures.append(f"{row['branch']}:{row['rel_path']}"); continue
        by_env[row["env"]].append(parse_metrics(text))
    records=[]; steps=range(0,TARGET_STEP+1,STEP_STRIDE)
    roles=("alpha_E",) if condition=="single_le" else ("alpha_E","alpha_B")
    for scope in ("all","locomotion","antmaze"):
        envs=[e for e in sorted(by_env) if scope=="all" or domain(e)==scope]
        for role in roles:
            idx=0 if role=="alpha_E" else 1
            for step in steps:
                env_means=[]; run_count=0
                for env in envs:
                    vals=[]
                    for metrics in by_env[env]:
                        if step==0:
                            value=initial_value(condition,init,role)
                        else:
                            pair=metrics.get(step)
                            value=pair[idx] if pair else None
                            if value is None and condition=="dual_le_fixed_b" and role=="alpha_B":
                                value=init
                        if value is not None: vals.append(value)
                    if vals:
                        env_means.append(statistics.fmean(vals)); run_count+=len(vals)
                mean=statistics.fmean(env_means) if env_means else None
                std=statistics.pstdev(env_means) if len(env_means)>1 else (0.0 if env_means else None)
                records.append({
                    "init":init,"condition":condition,"domain":scope,"alpha_role":role,
                    "step":step,"mean":mean,"std_across_envs":std,
                    "n_env":len(env_means),"n_runs":run_count,
                })
    return records,failures

def write_csv(path,rows,columns):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def main():
    flat=load_flat_runs()
    curves=[]; selections=[]; finals=[]; failures=[]
    for init in INITS:
        for condition in CONDITIONS:
            chosen,lr_summary=choose_best(candidates(flat,condition,init))
            for row in lr_summary: selections.append({"init":init,"condition":condition,**row})
            rows,missing=aggregate(chosen,condition,init)
            curves.extend(rows)
            failures.extend({"init":init,"condition":condition,"path":p} for p in missing)
            roles=("alpha_E",) if condition=="single_le" else ("alpha_E","alpha_B")
            for role in roles:
                hit=[r for r in rows if r["domain"]=="all" and r["alpha_role"]==role and r["step"]==TARGET_STEP]
                if hit:
                    r=hit[0]
                    finals.append({
                        "init":init,"condition":condition,"alpha_role":role,
                        "final_mean":r["mean"],"final_std_across_envs":r["std_across_envs"],
                        "n_env":r["n_env"],"n_runs":r["n_runs"],
                    })
    reports=ROOT/"reports"
    write_csv(reports/"alpha_comparison.csv",curves,
              ("init","condition","domain","alpha_role","step","mean","std_across_envs","n_env","n_runs"))
    write_csv(reports/"alpha_comparison_selection.csv",selections,
              ("init","condition","env","best_lr","n_seeds","mean_score","seeds"))
    write_csv(reports/"alpha_comparison_final.csv",finals,
              ("init","condition","alpha_role","final_mean","final_std_across_envs","n_env","n_runs"))
    write_csv(reports/"alpha_comparison_missing.csv",failures,("init","condition","path"))
    print(json.dumps({"curves":len(curves),"selections":len(selections),"finals":len(finals),"missing_metrics":len(failures)},sort_keys=True))

if __name__=="__main__":
    main()
