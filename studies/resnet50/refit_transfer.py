"""Repeat the CNN fits using the exact 999 paired FMD and Lytro draws of Supplement S11."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from cnn_models import PENALTIES, classify, kernels, load_npz, load_task, save_csv

STATE={}
SEED=20261121
REPLICATES=999

def initialize(root,features,standard_results):
    ctl=threadpool_limits(1)
    _,y,raw,_,folds=load_task(root,features,"Lytro")
    _,training,test,inner=folds[0]
    fit,valid=inner[0]
    strata=[[indices[y[indices]==c] for c in range(10)] for indices in (fit,valid,test)]
    if [[len(x) for x in section] for section in strata] != [[50]*10,[50]*10,[100]*10]:
        raise ValueError("Unexpected source roles or target classes")
    base=root/"studies/source_refit/results/bootstrap_draws.csv"
    with base.open(newline="") as f:old=list(csv.DictReader(f))
    if [int(r["replicate"]) for r in old]!=list(range(REPLICATES)):
        raise ValueError("The historical paired bootstrap is incomplete")
    reference=load_npz(standard_results/"Lytro_decisions.npz")
    STATE.update(control=ctl,raw=raw,y=y,roles=(fit,valid,test),strata=strata,reference=reference,old=old)

def draw(index):
    if index<0:return STATE["roles"]
    rng=np.random.default_rng([SEED,index])
    return [np.concatenate([rng.choice(indices,len(indices),replace=True) for indices in section])
            for section in STATE["strata"]]

def compress(indices,literal):
    if literal:return indices,np.ones(len(indices),dtype=int)
    return np.unique(indices,return_counts=True)

def evaluate(index,literal=False,keep=False):
    fit,val,test=draw(index)
    train,ct=compress(fit,literal);valid,cv=compress(val,literal)
    combined=np.r_[fit,val];pooled,cp=compress(combined,literal);ev,ce=compress(test,literal)
    raw,y=STATE["raw"],STATE["y"]
    inner=kernels(raw,fit,train,valid)
    outer=kernels(raw,combined,pooled,ev)
    row={"replicate":index};labels={}
    for suffix,kernel,final in zip(("S","Q"),inner,outer,strict=True):
        hits=[int(cv @ (classify(kernel,y[train],10,p,ct)==y[valid]).astype(int)) for p in PENALTIES]
        chosen=PENALTIES[int(np.argmax(hits))]
        row["lambda_R_"+suffix]=chosen;cache={}
        for regime,penalty in (("fixed",1.),("selected",chosen)):
            if penalty not in cache:cache[penalty]=classify(final,y[pooled],10,penalty,cp)
            row["R_"+regime+"_"+suffix]=100*int(ce @ (cache[penalty]==y[ev]).astype(int))/len(test)
            if keep:
                lookup=dict(zip(ev,cache[penalty],strict=True))
                labels[regime+"_"+suffix]=np.array([lookup[j] for j in test])
    if index>=0:
        base=STATE["old"][index]
        for comparison in ("H","D"):
            contrast=[]
            for regime in ("fixed","selected"):
                def gain(data,representation):
                    return float(data[representation+"_"+regime+"_Q"])-float(data[representation+"_"+regime+"_S"])
                contrast.append(gain(row,"R")-gain(base,comparison))
            row["T_R_minus_"+comparison]=round(contrast[1]-contrast[0],10)
    return row,labels

def worker(index):
    return evaluate(index)[0]

def run(root,features,standard,out,jobs):
    initialize(root,features,standard)
    observed,preds=evaluate(-1,keep=True)
    ref=STATE["reference"]
    for key,value in preds.items():
        regime,suffix=key.split("_")
        name=("selected_" if regime=="selected" else "")+suffix
        np.testing.assert_array_equal(value,ref[name][0])
    literal_checks=0
    for index in (0,1):
        a,l=evaluate(index,keep=True);b,r=evaluate(index,literal=True,keep=True)
        if a!=b:raise ValueError("Literal-row coefficients disagree on a draw")
        for key in l:
            np.testing.assert_array_equal(l[key],r[key]);literal_checks+=len(l[key])
    with ProcessPoolExecutor(max_workers=jobs,initializer=initialize,initargs=(root,features,standard)) as pool:
        rows=[]
        for i,row in enumerate(pool.map(worker,range(REPLICATES),chunksize=3),1):
            rows.append(row)
            if i%100==0:print("Paired CNN bootstrap",i,flush=True)
    oldref=load_npz(root/"studies/CVIU_Experience_couleur/results/Lytro_predictions.npz")
    observed_shifts={}
    y=oldref["y"]
    for family in ("H","D"):
        contrasts=[]
        for regime in ("fixed","selected"):
            conv_gain=observed["R_"+regime+"_Q"]-observed["R_"+regime+"_S"]
            other_gain=np.mean((oldref[family+"_"+regime+"_Q"]==y).astype(int)-(oldref[family+"_"+regime+"_S"]==y).astype(int))*100
            contrasts.append(conv_gain-other_gain)
        observed_shifts["T_R_minus_"+family]=round(contrasts[1]-contrasts[0],10)
    info={"replicates":len(rows),"seed":SEED,"observed":observed,"observed_shifts_pp":observed_shifts,
          "source_target_percentiles_pp":{k:np.quantile([r[k] for r in rows],[.025,.975]).tolist() for k in observed_shifts},
          "penalty_counts":{k:{str(p):sum(row[k]==p for row in rows) for p in PENALTIES} for k in ("lambda_R_S","lambda_R_Q")},
          "reference_labels_checked":sum(map(len,preds.values())),"literal_row_labels_checked":literal_checks}
    out.mkdir(exist_ok=True,parents=True);save_csv(out/"resnet_bootstrap_draws.csv",rows)
    (out/"resnet_bootstrap.json").write_text(json.dumps(info,indent=2)+"\n")
    print(info,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("root","features","standard","output"):p.add_argument(k,type=Path)
    p.add_argument("--jobs",type=int,default=2)
    a=p.parse_args();run(a.root,a.features,a.standard,a.output,a.jobs)
