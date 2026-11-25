"""Reconstruct all tables from full held-out predictions. Intervals condition on trained models."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from fit import NAMES, get
from scipy.stats import binom

SEED=20261120
DRAWS=100000
def correctness(pred,y):
    return (pred==y).astype(np.int16)
def macro(value,y):
    return float(np.mean([value[y==c].mean() for c in np.unique(y)])*100)
def ci_dtd(d,y,groups,m):
    # Exact Rademacher simulation distribution by aggregating groups with identical integer weights.
    _,idx=np.unique(groups,return_inverse=True);idx=idx.reshape(y.shape)
    influence=np.zeros(idx.max()+1,dtype=np.int64)
    assert y.shape==(10,1880)
    for j in range(10):
        np.testing.assert_array_equal(np.bincount(y[j]),np.full(47,40))
        totals=np.bincount(y[j],weights=d[j],minlength=47).astype(int)
        np.add.at(influence,idx[j],40*d[j]-totals[y[j]])
    w,num=np.unique(np.abs(influence[influence!=0]),return_counts=True)
    rng=np.random.default_rng(SEED);sim=np.zeros(DRAWS)
    for v,n in zip(w,num):sim+=v*(2*rng.binomial(n,.5,DRAWS)-n)
    effect=macro(d,y);sim=effect+sim*100/(d.size*40)
    return [effect,*np.quantile(sim,[.025/m,1-.025/m]).tolist()]

def ci_external(d,y,groups,m):
    # Resample complete groups within class. Use sums and counts so that varying group sizes
    # preserve the per-photo recall estimand. All methods see the same resampled observations.
    rng=np.random.default_rng(SEED)
    simulations=np.zeros(DRAWS);classes=np.unique(y)
    for c in classes:
        mask=y==c;g,inv=np.unique(groups[mask],return_inverse=True);inv=inv.ravel()
        totals=np.bincount(inv,weights=d[mask]);counts=np.bincount(inv)
        pairs,counts_pairs=np.unique(np.stack((totals,counts),axis=1),axis=0,return_counts=True)
        draw=rng.multinomial(len(g),counts_pairs/len(g),DRAWS)
        simulations+=(draw@pairs[:,0])/(draw@pairs[:,1])
    return [macro(d,y),*np.quantile(simulations*100/len(classes),[.025/m,1-.025/m]).tolist()]

def csv_out(path,items):
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=items[0]);w.writeheader();w.writerows(items)
def analyse(cache,output):
    tab=[];classes=[];folds=[];external=[];interactions=[]
    for corpus in ('DTD','KTH','Lytro'):
        z=get(output/(corpus+'_predictions.npz'))
        design=get(cache/(corpus+'_design.npz'))
        y=z['y'];groups=z['groups'];correct={k:correctness(z[k],y) for k in NAMES}
        for key,value in correct.items():
            diff=value-correct['RGB_hard'];planned=key in ('Lab_soft','Lab_abs_rel')
            point=macro(diff,y)
            interval=(ci_dtd if corpus=='DTD' else ci_external)(diff,y,groups,6) if planned else [point,None,None]
            tab.append(dict(corpus=corpus,palette=key,recall_pct=macro(value,y),gain_pp=point,
                            lower_adjusted_pp=interval[1],upper_adjusted_pp=interval[2]))
            for j,label in enumerate(design['classes']):
                mask=y==j;classes.append(dict(corpus=corpus,palette=key,label=label,n_appearances=int(mask.sum()),
                    correct=int(value[mask].sum()),recall_pct=float(value[mask].mean()*100),gain_pp=float(diff[mask].mean()*100)))
            if corpus!='Lytro':
                for j in range(len(y)):folds.append(dict(corpus=corpus,palette=key,fold=j+1,recall_pct=macro(value[j],y[j]),gain_pp=macro(diff[j],y[j])))
        if corpus!='Lytro':continue
        for g in np.unique(groups):assert len(np.unique(y[groups==g]))==1
        seq=np.array([int(s.rsplit('/',1)[1].split('.')[0])-1 for s in z['ids']])
        blocks=(y*1000+seq//5);gains={}
        for regime in ('fixed','matched','selected'):
            for f in ('H','D'):
                ref=f+'_'+('fixed' if regime=='matched' else regime)+'_S'
                alternative=f+'_'+regime+'_Q'
                s=correctness(z[ref],y);q=correctness(z[alternative],y);gains[f,regime]=q-s
                ci=ci_external(q-s,y,groups,1)
                external.append(dict(regime=regime,representation=f,S_pct=macro(s,y),Q_pct=macro(q,y),
                    gain_pp=ci[0],pointwise_low=ci[1],pointwise_high=ci[2]))
            d=gains['D',regime]-gains['H',regime]
            ci=ci_external(d,y,groups,1)
            interactions.append(dict(contrast='DINO_gain_minus_handcrafted_'+regime,gain_pp=ci[0],low=ci[1],high=ci[2]))
        delta=(gains['D','selected']-gains['H','selected'])-(gains['D','fixed']-gains['H','fixed'])
        for label,g in [('primary_captures',groups),('sensitivity_blocks_of_5',blocks)]:
            ci=ci_external(delta,y,g,1);interactions.append(dict(contrast=label,gain_pp=ci[0],low=ci[1],high=ci[2]))
    for k,rows in [('palettes',tab),('by_class',classes),('by_fold',folds),('external_kernels',external),('external_contrasts',interactions)]:
        csv_out(output/(k+'.csv'),rows)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();analyse(a.cache,a.output)
