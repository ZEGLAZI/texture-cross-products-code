"""Reconstruct the external primary contrast and eight paired secondary comparisons."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from garments import write_csv
from fit_garments import read_csv,MODELS

REGIMES={'fixed':('S','Q'),'matched':('S','matched_Q'),'selected':('selected_S','selected_Q')}

def direct(delta):
    move={m:delta[m,'selected']-delta[m,'fixed'] for m in ('R','D','DR','DV') if (m,'selected') in delta}
    pairs={'A':('DR','R'),'B':('DR','DV')}
    if 'D' in move:pairs['T']=('R','D')
    return {q:move[left]-move[right] for q,(left,right) in pairs.items()}

def garment_interval(values,labels,family):
    classes=np.unique(labels);samples=np.zeros(100000);estimate=0.
    rng=np.random.default_rng(20261120)
    for c in classes:
        x=values[labels==c];v,n=np.unique(x,return_counts=True)
        estimate+=x.mean();samples+=(rng.multinomial(len(x),n/n.sum(),size=len(samples))@v)/len(x)
    factor=100/len(classes)
    return [factor*estimate,*np.quantile(factor*samples,[.025/family,1-.025/family])]

def append_test(tests,name,key,estimate,low,high):
    tests.append(dict(dataset=name,contrast=key,estimate_pp=estimate,low_pp=low,high_pp=high,
                      family_size=1 if key=='T' else 8))

def analyze(root,results,out):
    sys.path.insert(0,str(root.resolve()))
    from summarize_pixel_check import macro,paired_interval
    sys.path.insert(0,str((root/'studies/CVIU_Experience_couleur/code').resolve()))
    from analyse import ci_external
    gains=[];tests=[];class_rows=[]
    for ds in ('DTD','KTH','Lytro'):
        def load(folder):
            with np.load(folder/(ds+'_decisions.npz')) as z:return dict(z)
        previous=load(root/'studies/resnet50/results');y=previous['labels'];groups=previous['groups'];delta={}
        def interval(value,family):
            if ds=='Lytro':return ci_external(value.ravel(),y.ravel(),groups.ravel(),family)
            return [macro(value,y),*paired_interval(ds,value,y,groups,family)]
        for m,folder in [('R',None),('DR','dino_r50'),('DV','dino_vits16')]:
            item=load(results/folder) if folder else previous
            for key in ('ids','labels','groups','test'):np.testing.assert_array_equal(previous[key],item[key])
            for regime,(before,after) in REGIMES.items():
                s=(item[before]==y).astype(int);q=(item[after]==y).astype(int);diff=q-s
                delta[m,regime]=diff
                if folder:
                    est,lo,hi=interval(diff,1)
                    gains.append(dict(dataset=ds,representation=m,regime=regime,square_pct=macro(s,y),full_pct=macro(q,y),
                                      gain_pp=est,point_low_pp=lo,point_high_pp=hi))
        for label,x in direct(delta).items():append_test(tests,ds,label,*interval(x,8))
    table=read_csv(results/'garment_test_evidence.csv')
    labels=np.array([r['motif'] for r in table]);views=np.array([int(r['views']) for r in table])
    if np.any((views<1)|(views>2)) or len({r['test_product'] for r in table})!=len(views):raise ValueError('Invalid test groups')
    delta={};photo=[]
    def score(x):return 100*np.mean([np.mean(x[labels==c]) for c in np.unique(labels)])
    def image(x):return 100*np.mean([np.average(x[labels==c],weights=views[labels==c]) for c in np.unique(labels)])
    for m in MODELS:
        for regime,(before,after) in REGIMES.items():
            s,q=[np.array([int(r[m+'_'+method]) for r in table])/views for method in (before,after)]
            if np.any((s<0)|(s>1)|(q<0)|(q>1)):raise ValueError('Invalid correctness count')
            diff=q-s;delta[m,regime]=diff;est,lo,hi=garment_interval(diff,labels,1)
            gains.append(dict(dataset='Garments',representation=m,regime=regime,square_pct=score(s),full_pct=score(q),
                              gain_pp=est,point_low_pp=lo,point_high_pp=hi))
            photo.append(dict(representation=m,regime=regime,square_pct=image(s),full_pct=image(q),gain_pp=image(diff)))
            for c in np.unique(labels):
                ok=labels==c
                class_rows.append(dict(motif=c,products=int(ok.sum()),views=int(views[ok].sum()),representation=m,regime=regime,
                         square_pct=100*np.mean(s[ok]),full_pct=100*np.mean(q[ok]),gain_pp=100*np.mean(diff[ok])))
    for label,x in direct(delta).items():
        append_test(tests,'Garments',label,*garment_interval(x,labels,1 if label=='T' else 8))
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('gains',gains),('contrasts',tests),('garment_image_macro',photo),('garment_classes',class_rows)]:
        write_csv(out/(name+'.csv'),rows)
    print('Computed',len(gains),'gain rows and',len(tests),'contrasts.')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('package','results','out'):p.add_argument(field,type=Path)
    a=p.parse_args()
    with threadpool_limits(1):analyze(a.package,a.results,a.out)
