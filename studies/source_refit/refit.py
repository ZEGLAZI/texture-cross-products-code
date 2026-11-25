"""Paired full-refit bootstrap for the FMD to Lytro experiment."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'CVIU_Experience_couleur' / 'code'))
from fit import PENALTIES, kernels, get
SEED = 20261121
N_BOOTSTRAP = 999
STATE = {}

def initialize(cache, reference):
    ys, hashes, files = [], [], []
    blocks = {k: [] for k in ('F7','F8','rgb','patch3','patch6','patch12')}
    for ds in ('FMD','Lytro'):
        d,p,h = [get(cache / (ds+s)) for s in ('_design.npz','_palettes.npz','_features.npz')]
        np.testing.assert_array_equal(d['ids'],p['ids'])
        np.testing.assert_array_equal(d['ids'],h['ids'])
        np.testing.assert_array_equal(p['hash224'],h['hash224'])
        ys.append(d['y']);hashes.extend(p['hash224']);files.extend(p['file_hash'])
        for k in blocks:
            blocks[k].append(p['values'][:,0] if k=='rgb' else h[k])
    if len(set(hashes))!=2000 or len(set(files))!=2000:
        raise ValueError('Expected disjoint source and target image identities')
    raw={k:np.concatenate(v).astype(np.float64) for k,v in blocks.items()}
    for k,dim in [('F7',9),('F8',3)]:
        r,c=np.triu_indices(dim)
        raw[k][...,np.flatnonzero(r!=c)]*=np.sqrt(2.)
    original=get(reference/'Lytro_predictions.npz')
    roles=[original['fit'],original['val'],np.arange(1000,2000)]
    assert len(np.intersect1d(*roles[:2]))==0
    y=np.concatenate(ys)
    strata=[[part[y[part]==c] for c in range(10)] for part in roles]
    assert [[len(s) for s in role] for role in strata]==[[50]*10,[50]*10,[100]*10]
    STATE.update(raw=raw,original=original,roles=roles,y=y,onehot=np.eye(10)[y],strata=strata)

def draws(index):
    if index<0:
        return STATE['roles']
    rng=np.random.default_rng([SEED,index])
    return [np.concatenate([rng.choice(s,len(s),replace=True) for s in groups]) for groups in STATE['strata']]

def pair(family, repeated, train, evaluation):
    ix=np.r_[train,evaluation]; grams=[]
    keys=('F7','F8','rgb') if family=='H' else ('patch3','patch6','patch12')
    for key in keys:
        raw=STATE['raw'][key]
        x=raw[ix]
        if key!='rgb':
            sample=raw[repeated]
            axes=(0,2) if key.startswith('F') else 0
            x-=sample.mean(axis=axes,keepdims=True)
            x/=np.maximum(sample.std(axis=axes,keepdims=True),1.e-4)
            np.clip(x,-12.,12.,out=x)
            x=x.reshape(len(ix),-1)
            norm=np.linalg.norm(x,axis=1,keepdims=True)
            np.divide(x,norm,out=x,where=norm!=0.)
        grams.append(x@x[:len(train)].T)
    return kernels(*grams,both=True)

def predict(kernel,train,count,penalty):
    n=len(train)
    labels=STATE['onehot'][train]
    prior=np.average(labels,axis=0,weights=count)
    matrix=kernel[:n].copy(order='F')
    matrix.flat[::n+1]+=penalty/count
    coef=solve(matrix,labels-prior,assume_a='pos',overwrite_a=True,check_finite=False)
    return (kernel[n:]@coef+prior).argmax(axis=1).astype(np.int16)

def compress(ids,literal):
    return (ids,np.ones(len(ids),dtype=int)) if literal else np.unique(ids,return_counts=True)

def evaluate(index,literal=False,check=False):
    fit,val,test=draws(index)
    tr,ct=compress(fit,literal);va,cv=compress(val,literal)
    pool=np.r_[fit,val];outer,co=compress(pool,literal);te,ce=compress(test,literal)
    row={'replicate':index}; archive={};y=STATE['y']
    for family in ('H','D'):
        validation=pair(family,fit,tr,va)
        target=pair(family,pool,outer,te)
        for suffix,v,k in zip(('S','Q'),validation,target,strict=True):
            hits=[int(cv@(predict(v,tr,ct,p)==y[va]).astype(int)) for p in PENALTIES]
            chosen=PENALTIES[int(np.argmax(hits))]
            row['lambda_'+family+'_'+suffix]=chosen
            fitted={}
            for regime,pen in [('fixed',1.),('selected',chosen)]:
                if pen not in fitted:fitted[pen]=predict(k,outer,co,pen)
                pred=fitted[pen];key=family+'_'+regime+'_'+suffix
                row[key]=100*int(ce@(pred==y[te]).astype(int))/len(test)
                if check:
                    lookup=dict(zip(te,pred,strict=True))
                    archive[key]=np.array([lookup[i] for i in test])
    for regime in ('fixed','selected'):
        row['D_'+regime]=round((row['D_'+regime+'_Q']-row['D_'+regime+'_S'])-(row['H_'+regime+'_Q']-row['H_'+regime+'_S']),10)
    row['T']=round(row['D_selected']-row['D_fixed'],10)
    return row,archive

def worker(index):
    return evaluate(index)[0]

def run(cache,reference,out,jobs,replicates):
    out.mkdir(exist_ok=True,parents=True)
    initialize(cache,reference)
    point,labels=evaluate(-1,check=True)
    for k,p in labels.items():np.testing.assert_array_equal(p,STATE['original'][k])
    probes=0
    for index in (0,1):
        a,la=evaluate(index,check=True)
        b,lb=evaluate(index,literal=True,check=True)
        assert a==b
        for k in la:
            np.testing.assert_array_equal(la[k],lb[k]);probes+=len(la[k])
    print('Reference fit and literal bootstrap checks passed.',flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=jobs,initializer=initialize,initargs=(cache,reference)) as pool:
        for n,row in enumerate(pool.map(worker,range(replicates),chunksize=2),1):
            rows.append(row)
            if n%50==0:print(n,'/',replicates,flush=True)
    with (out/'bootstrap_draws.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    keys=('H_S','H_Q','D_S','D_Q')
    t=np.array([r['T'] for r in rows])
    summary=dict(replicates=len(rows),seed=SEED,original_T_pp=point['T'],
        refit_percentile95_pp=np.quantile(t,[.025,.975]).tolist(),bootstrap_median_pp=float(np.median(t)),
        all_four_choose_one=sum(all(r['lambda_'+k]==1. for k in keys) for r in rows),
        penalty_counts={k:{str(p):sum(r['lambda_'+k]==p for r in rows) for p in PENALTIES} for k in keys},
        literal_prediction_comparisons=probes,reference_prediction_comparisons=sum(map(len,labels.values())))
    (out/'bootstrap_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('cache','reference','output'):parser.add_argument(k,type=Path)
    parser.add_argument('--jobs',type=int,default=2);parser.add_argument('--replicates',type=int,default=N_BOOTSTRAP)
    a=parser.parse_args()
    for var in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[var]='1'
    with threadpool_limits(1):run(a.cache,a.reference,a.output,a.jobs,a.replicates)
