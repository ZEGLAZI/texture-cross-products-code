"""Fit colour controls on manuscript folds or train on FMD and test on held-out Lytro captures."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

PENALTIES=(1.,.1,10.,.01,100.,.001)
NAMES=("RGB_hard","RGB_soft","Lab_hard","Lab_soft","Lab_abs_rel")

def get(path):
    with np.load(path,allow_pickle=False) as z:return dict(z)

def standardized(array,fit,dim=None):
    x=array.astype(np.float64)
    if dim is not None:
        row,col=np.triu_indices(dim);x[...,np.flatnonzero(row!=col)]*=np.sqrt(2.)
    axes=0 if dim is None else (0,2)
    training=x[fit]
    x-=training.mean(axis=axes,keepdims=True)
    x/=np.maximum(training.std(axis=axes,keepdims=True),1e-4)
    np.clip(x,-12.,12.,out=x);x=x.reshape(len(x),-1)
    norm=np.linalg.norm(x,axis=1,keepdims=True)
    return np.divide(x,norm,out=np.zeros_like(x),where=norm!=0.)

def palettes(values):
    return (values[:,0],values[:,1],values[:,2],values[:,3],
            np.concatenate((values[:,3],values[:,4]),axis=1)/np.sqrt(2.))

def gram(a,tr,ev):
    return a[np.r_[tr,ev]] @ a[tr].T

def kernels(a,b,c,both=False):
    linear=a+b+c
    # Coefficients match equation (4), with equal weights, including the outer 1/13.
    Q=((8./3.)*linear+linear**2/3.)/13.
    if not both:return Q
    S=((8./3.)*linear+(a*a+b*b+c*c)/3.)/13.
    return S,Q

def classify(k,n,labels,nclass,penalty):
    train=k[:n].copy(order='F');train.flat[::n+1]+=penalty
    Y=np.eye(nclass)[labels];prior=Y.mean(axis=0)
    coef=solve(train,Y-prior,assume_a='pos',overwrite_a=True,check_finite=False)
    return (k[n:]@coef+prior).argmax(axis=1).astype(np.int16)

def macro(pred,labels):
    return float(100*np.mean([(pred[labels==c]==c).mean() for c in np.unique(labels)]))

def previous(name,cache,out):
    z=get(cache/(name+'_design.npz'));hb=get(cache/(name+'_original.npz'));p=get(cache/(name+'_palettes.npz'))
    for m in (hb,p):np.testing.assert_array_equal(m['ids'],z['ids'])
    np.testing.assert_array_equal(p['values'][:,0],hb['rgb'])
    results={k:[] for k in NAMES};y=z['y'];tests=[]
    for i,truth in enumerate(z['baseline'],1):
        tr,te=z['train'+str(i)],z['test'+str(i)]
        a=gram(standardized(hb['F7'],tr,9),tr,te)
        b=gram(standardized(hb['F8'],tr,3),tr,te)
        for key,block in zip(NAMES,palettes(p['values']),strict=True):
            k=kernels(a,b,gram(block,tr,te));labels=classify(k,len(tr),y[tr],len(z['classes']),1.)
            if key=='RGB_hard':np.testing.assert_array_equal(labels,truth)
            results[key].append(labels);del k
        tests.append(te);print(name,'fold',i,'RGB decisions verified',len(te),flush=True)
    tests=np.stack(tests)
    np.savez_compressed(out/(name+'_predictions.npz'),indices=tests,ids=z['ids'][tests],y=y[tests],
        groups=z['occurrence_group'],**{k:np.stack(v) for k,v in results.items()})

def external(cache,out):
    designs=[get(cache/(c+'_design.npz')) for c in ('FMD','Lytro')]
    ps=[get(cache/(c+'_palettes.npz')) for c in ('FMD','Lytro')]
    features=[get(cache/(c+'_features.npz')) for c in ('FMD','Lytro')]
    for d,p,h in zip(designs,ps,features,strict=True):
        np.testing.assert_array_equal(d['ids'],p['ids'])
        np.testing.assert_array_equal(d['ids'],h['ids'])
        np.testing.assert_array_equal(h['hash224'],p['hash224'])
    np.testing.assert_array_equal(designs[0]['classes'],designs[1]['classes'])
    y=np.concatenate([d['y'] for d in designs]);hashes=np.concatenate([p['hash224'] for p in ps])
    rawhash=np.concatenate([p['file_hash'] for p in ps]);n=len(designs[0]['y']);nc=len(designs[0]['classes'])
    pool=np.arange(n);te=np.arange(n,len(y))
    def keep(ix,other):
        return ix[~(np.isin(hashes[ix],hashes[other])|np.isin(rawhash[ix],rawhash[other]))]
    tr=keep(pool,te);va=keep(designs[0]['val'],te);fit=keep(keep(designs[0]['fit'],te),va)
    raw={k:np.concatenate([h[k] for h in features]) for k in ('F7','F8','patch3','patch6','patch12')}
    palette_values=palettes(np.concatenate([p['values'] for p in ps]))
    def blocks(family,fit,ev):
        if family=='H':
            return (gram(standardized(raw['F7'],fit,9),fit,ev),
                    gram(standardized(raw['F8'],fit,3),fit,ev),gram(palette_values[0],fit,ev))
        return tuple(gram(standardized(raw[k],fit),fit,ev) for k in ('patch3','patch6','patch12'))
    config={'classes':designs[0]['classes'].tolist(),'source_all':n,'source_fit':len(fit),'source_val':len(va),
        'source_refit':len(tr),'target':len(te),'target_unique_hash':len(np.unique(hashes[te])),
        'native_target_sizes':np.unique(ps[1]['sizes'],axis=0).tolist(),'penalties':PENALTIES,'families':{}}
    preds={}
    for family in ('H','D'):
        fitted={}
        for mode,k in zip(('S','Q'),kernels(*blocks(family,fit,va),both=True),strict=True):
            grid=[macro(classify(k,len(fit),y[fit],nc,pen),y[va]) for pen in PENALTIES]
            sel=PENALTIES[np.flatnonzero(np.array(grid)>=max(grid)-1e-11)[0]]
            fitted[mode]={'inner_recall_pct':grid,'lambda':sel}
        a,b,c=blocks(family,tr,te);s,q=kernels(a,b,c,both=True)
        factor=float(np.trace(s[:len(tr)])/np.trace(q[:len(tr)]))
        fitted['matched_scale']=factor
        for mode,k in (('S',s),('Q',q)):
            for key,pen in (('fixed',1.),('selected',fitted[mode]['lambda'])):
                preds[family+'_'+key+'_'+mode]=classify(k,len(tr),y[tr],nc,pen)
        preds[family+'_matched_Q']=classify(q,len(tr),y[tr],nc,1./factor)
        if family=='H':
            for key,block in zip(NAMES,palette_values,strict=True):
                preds[key]=classify(kernels(a,b,gram(block,tr,te)),len(tr),y[tr],nc,1.)
            np.testing.assert_array_equal(preds['RGB_hard'],preds['H_fixed_Q'])
        config['families'][family]=fitted
    _,groups=np.unique(hashes[te],return_inverse=True)
    np.savez_compressed(out/'Lytro_predictions.npz',ids=designs[1]['ids'],y=y[te],groups=groups,
                        fit=fit,val=va,refit=tr,**preds)
    (out/'Lytro_settings.json').write_text(json.dumps(config,indent=2)+'\n')
    print('FMD fit / validation / refit; external',len(fit),len(va),len(tr),len(te),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('corpus',choices=('DTD','KTH','Lytro'))
    p.add_argument('cache',type=Path);p.add_argument('output',type=Path);p.add_argument('--threads',type=int,default=3)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(a.threads):
        if a.corpus=='Lytro':external(a.cache,a.output)
        else:previous(a.corpus,a.cache,a.output)
