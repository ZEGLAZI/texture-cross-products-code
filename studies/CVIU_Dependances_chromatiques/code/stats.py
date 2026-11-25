"""Paired inference and classification metrics from the ordered held-out decisions."""
import argparse
import csv
from pathlib import Path
import numpy as np
from fit import ABLATIONS, LIGHTS, read_npz
SEED=20261120
DRAWS=100000
TASKS=('DTD','KTH','Lytro','omit_ambient')
META={'ids','y','groups','indices','classes','light','pose'}
DIMS={'RGB':(64,64),'RGB_common':(64,65),'Lab6':(6,7),'Lab9':(9,10),'P':(30,31),'J':(30,31)}
for form in ('P','J'):
    for lab in ('','_perm','_mean'):DIMS['Lab_'+form+lab]=(36,38)
    for lab in ('','_perm'):DIMS['RGB_'+form+lab]=(94,95)

def model_dims(name):
    name=name.split('__')[0]
    if name in ('A','AW'):return (945,945) if name=='A' else (1449,1449)
    base,colour=name.split('_',1);offset={'A':945,'AW':1449,'only':0}[base]
    return tuple(offset+v for v in DIMS[colour])

def macro(vals,y):return float(np.mean([vals[y==i].mean() for i in np.unique(y)])*100)
def correct(x,y):return (x==y).astype(np.int16)

def inference(d,y,groups,task,family):
    delta=macro(d,y);rng=np.random.default_rng(SEED)
    q=[.025/family,1-.025/family]
    if task=='DTD':
        unique,inv=np.unique(groups,return_inverse=True);inv=inv.reshape(y.shape)
        total=np.zeros(len(unique),dtype=np.int64)
        assert d.shape==(10,1880)
        for i in range(10):
            np.testing.assert_array_equal(np.bincount(y[i]),np.full(47,40))
            sums=np.bincount(y[i],weights=d[i]).astype(np.int64)
            np.add.at(total,inv[i],40*d[i]-sums[y[i]])
        values,counts=np.unique(abs(total[total!=0]),return_counts=True)
        sim=np.zeros(DRAWS)
        for value,count in zip(values,counts):sim+=value*(2*rng.binomial(count,.5,DRAWS)-count)
        draws=delta+sim*100/(40*d.size)
    else:
        labels=np.unique(y);draws=np.zeros(DRAWS)
        for c in labels:
            keep=y==c;ids,inv=np.unique(groups[keep],return_inverse=True);inv=inv.ravel()
            counts=np.bincount(inv);sums=np.bincount(inv,weights=d[keep])
            # All held-out groups have the same number of observations in each of these designs.
            assert np.min(counts)==np.max(counts)
            values,freq=np.unique(sums/counts,return_counts=True)
            samples=rng.multinomial(len(ids),freq/len(ids),DRAWS)
            draws+=(samples@values)/len(ids)
        draws*=100/len(labels)
    low,high=np.quantile(draws,q)
    return float(delta),float(low),float(high)

def write(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def calc(out):
    scores=[];primary=[];augment=[];ablations=[];light_rows=[];direct=[];by_class=[];folds=[]
    light_cases={k:read_npz(out/('omit_'+k+'_predictions.npz')) for k in LIGHTS}
    for task in TASKS:
        z=read_npz(out/(task+'_predictions.npz'));y=z['y'];groups=z['groups']
        methods=[k for k in z if k not in META];ind={k:correct(z[k],y) for k in methods}
        def contrast(field,ref,family=1):
            d=ind[field]-ind[ref]
            mean,lo,hi=inference(d,y,groups,task,family)
            return dict(task=task,method=field,reference=ref,effect_pp=mean,low_pp=lo,high_pp=hi,bonferroni_family=family)
        for k in methods:
            v=ind[k];pred=z[k]
            precisions=[]
            for label,name in enumerate(z['classes']):
                keep=y==label;den=int((pred==label).sum());tp=int(v[keep].sum())
                prec=tp/den if den else 0.
                precisions.append(prec)
                by_class.append(dict(task=task,method=k,label=name,test_appearances=int(keep.sum()),correct=tp,
                                     recall_pct=100*v[keep].mean(),precision_pct=100*prec))
            raw,explicit=model_dims(k)
            scores.append(dict(task=task,method=k,raw_dimensions=raw,explicit_block_dimensions=explicit,
                test_appearances=len(y.ravel()),macro_recall_pct=macro(v,y),top1_accuracy_pct=100*v.mean(),
                macro_precision_pct=100*np.mean(precisions)))
            for j in range(len(y)):
                folds.append(dict(task=task,fold=j+1,method=k,n=len(y[j]),recall_pct=macro(v[j],y[j])))
        for form in ('P','J'):
            cand='AW_Lab_'+form
            for key in ('AW_RGB','AW_Lab6',cand+'_perm',cand+'_mean'):primary.append(contrast(cand,key,32))
            enriched='AW_RGB_'+form
            for ref in ('AW_RGB',enriched+'_perm'):augment.append(contrast(enriched,ref,16))
        for form in ABLATIONS:
            for base in ('A','AW'):
                k=base+'_'+form
                ablations.append(contrast(k,base))
                kc=(k+'__native') if base=='A' else k
                ablations.append(contrast(kc,base+'__native'))
        if task=='KTH':
            for m in methods:
                if m.endswith('__native'):continue
                for ell,name in enumerate(LIGHTS):
                    keep=z['light']==ell
                    light_rows.append(dict(protocol='original_split',light=name,method=m,n=int(keep.sum()),recall_pct=macro(ind[m][keep],y[keep])))
        if task=='omit_ambient':
            controls=[light_cases[a] for a in LIGHTS[:3]];keys=[k for k in controls[0] if k not in META]
            for c in controls:
                for key in ('ids','y','groups','indices','light','pose'):np.testing.assert_array_equal(c[key],z[key])
            def response(k):
                cc=np.mean([correct(c[k],y) for c in controls],axis=0)
                return ind[k]-cc,cc
            base,_=response('AW_RGB')
            for m in keys:
                for name in LIGHTS:
                    x=light_cases[name];v=correct(x[m],y)
                    light_rows.append(dict(protocol='train_without_'+name,light='ambient',method=m,n=y.size,recall_pct=macro(v,y)))
                diff,ctrl=response(m)
                gap,lo,hi=inference(diff,y,groups,task,1)
                delta,dlo,dhi=inference(diff-base,y,groups,task,1)
                direct.append(dict(method=m,ambient_unseen_pct=macro(ind[m],y),ambient_seen_size_matched_pct=macro(ctrl,y),
                    exposure_gap_pp=gap,gap_low=lo,gap_high=hi,gap_minus_RGB_gap_pp=delta,difference_low=dlo,difference_high=dhi))
    for name,rows in [('scores',scores),('primary',primary),('enriched',augment),('complementarity',ablations),
                      ('illumination',light_rows),('matched_illumination',direct),('by_class',by_class),('by_fold',folds)]:
        write(out/(name+'.csv'),rows)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);args=p.parse_args();calc(args.results)
