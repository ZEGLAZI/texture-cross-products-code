"""Fixed-ridge evaluation on published splits and real held-out KTH lighting conditions."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

CANDIDATES=('RGB','Lab6','Lab9','RGB_common','P','J','Lab_P','Lab_J','Lab_P_perm','Lab_J_perm',
            'Lab_P_mean','Lab_J_mean','RGB_P','RGB_J','RGB_P_perm','RGB_J_perm')
ABLATIONS=('RGB','Lab6','P','J','Lab_P','Lab_J','Lab_P_mean','Lab_J_mean')
LIGHTS=('front','top','side','ambient')

def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return dict(z)

def rows(data,c):
    b=read_npz(data/(c+'_inputs.npz'));v=read_npz(data/(c+'_new.npz'))
    np.testing.assert_array_equal(b['ids'],v.pop('ids'))
    return dict(b,**v)

def light_and_pose(ids):
    numbers=np.array([int(re.search(r'_im_(\d+)_col[.]',key).group(1)) for key in ids])
    if np.any((numbers<1)|(numbers>12)):raise ValueError('Expected KTH indices 1–12')
    return np.where(numbers<=9,(numbers-1)%3,3),np.where(numbers<=9,(numbers-1)//3,numbers-10)

def load_design(data,task):
    if task=='Lytro':
        src,target=rows(data,'FMD'),rows(data,'Lytro');plan=read_npz(data/'transfer_design.npz')
        np.testing.assert_array_equal(src['classes'],target['classes'])
        np.testing.assert_array_equal(target['ids'],plan['target_ids'])
        rec={k:np.concatenate((src[k],target[k])) for k in ('ids','y','F7','F8','rgb','lab','edges','hash224')}
        tr=plan['refit'];te=np.arange(len(src['y']),len(rec['y']))
        assert not np.isin(rec['hash224'][tr],rec['hash224'][te]).any()
        return rec,[(tr,te)],target['classes'],plan['groups'][None,:],plan['baseline'][None,:]
    name='DTD' if task=='DTD' else 'KTH';rec=rows(data,name)
    pairs=[];groups=[]
    for j in range(1,1+len(rec['baseline'])):
        tr,te=rec['train'+str(j)],rec['test'+str(j)]
        mask=np.ones(len(te),dtype=bool)
        if task.startswith('omit_'):
            light,_=light_and_pose(rec['ids'])
            tr=tr[light[tr]!=LIGHTS.index(task[5:])]
            mask=light[te]==3
            assert len(te[mask])==297 and set(rec['specimens'][tr]).isdisjoint(rec['specimens'][te[mask]])
        pairs.append((tr,te[mask]));groups.append(rec['occurrence_group'][j-1][mask])
    expected=rec['baseline'] if task==name else None
    return rec,pairs,rec['classes'],np.stack(groups),expected

def standard_wavelet(x,fit,dim):
    x=x.astype(np.float64)
    i,j=np.triu_indices(dim);x[...,np.flatnonzero(i!=j)]*=np.sqrt(2.)
    sample=x[fit]
    x-=sample.mean(axis=(0,2),keepdims=True);x/=np.maximum(sample.std(axis=(0,2),keepdims=True),1e-4)
    np.clip(x,-12.,12.,out=x);x=x.reshape(len(x),-1)
    norm=np.linalg.norm(x,axis=1,keepdims=True)
    return np.divide(x,norm,out=np.zeros_like(x),where=norm!=0)

def compact_map(x,fit):
    z=(x-x[fit].mean(axis=0))/np.maximum(x[fit].std(axis=0),1.e-6)
    z=np.clip(z,-8.,8.)/np.sqrt(z.shape[1])
    explicit=np.column_stack((np.ones(len(x)),z))
    return explicit/np.linalg.norm(explicit,axis=1,keepdims=True)

def colour_maps(rec,tr):
    e=rec['edges'];maps={'RGB':rec['rgb'],'RGB_common':compact_map(rec['rgb'],tr),
        'Lab6':compact_map(rec['lab'][:,3:],tr),'Lab9':compact_map(rec['lab'],tr)}
    for j,key in enumerate(('P','J')):
        for i,stat in enumerate(('','_mean','_perm')):
            maps[key+stat]=compact_map(e[:,j,i,:],tr)
        for base,mapkey in [('Lab','Lab6'),('RGB','RGB')]:
            for stat in (('','_mean','_perm') if base=='Lab' else ('','_perm')):
                maps[base+'_'+key+stat]=np.concatenate((maps[mapkey],maps[key+stat]),axis=1)/np.sqrt(2.)
    return maps

def gram(x,ix,tr):
    return x[ix]@x[tr].T

def polynomial(sums):
    return (8.*sums+sums*sums)/39.

def classify(k,n,centred,prior,penalty):
    coeff=k[:n].copy(order='F')
    coeff.flat[::n+1]+=penalty
    alpha=solve(coeff,centred,assume_a='pos',overwrite_a=True,check_finite=False)
    return (k[n:]@alpha+prior).argmax(axis=1).astype(np.int16)

def compute(task,data,out,public_reference=True):
    rec,pairs,classes,groups,expected=load_design(data,task)
    check=not task.startswith('omit_') or task=='omit_ambient'
    allpred={};diagnostics=[];indices=[]
    for j,(tr,te) in enumerate(pairs):
        y=rec['y'];ix=np.r_[tr,te];nt=len(tr)
        a=gram(standard_wavelet(rec['F7'],tr,9),ix,tr);w=gram(standard_wavelet(rec['F8'],tr,3),ix,tr)
        aw=a+w;del w
        maps=colour_maps(rec,tr);target=np.eye(len(classes))[y[tr]];prior=target.mean(0);centred=target-prior
        predictions={};scale_info={}
        def measure(key,sim,native=False):
            k=polynomial(sim);trace=float(np.diag(k[:nt]).mean())
            factor=(11/13)/trace
            predictions[key]=classify(k,nt,centred,prior,1./factor)
            if native:
                predictions[key+'__native']=(predictions[key] if abs(factor-1)<1e-13 else classify(k,nt,centred,prior,1.))
            scale_info[key]=factor
        measure('A',a,check);measure('AW',aw,check)
        for key in CANDIDATES:
            c=gram(maps[key],ix,tr)
            measure('AW_'+key,aw+c)
            if check and key in ABLATIONS:
                measure('A_'+key,a+c,True);measure('only_'+key,c,True)
            del c
        if expected is not None and public_reference:
            np.testing.assert_array_equal(predictions['AW_RGB'],expected[j])
        for key,val in predictions.items():allpred.setdefault(key,[]).append(val)
        indices.append(te)
        diagnostics.append(dict(fold=j+1,train=len(tr),test=len(te),scale=scale_info))
        print(task,'fold',j+1,'train',len(tr),'test',len(te),'fits',len(predictions),flush=True)
    indices=np.stack(indices)
    signal={'ids':rec['ids'][indices],'y':rec['y'][indices],'groups':groups,'indices':indices,'classes':classes}
    if task.startswith(('KTH','omit_')):
        illum,pose=light_and_pose(rec['ids'])
        signal.update(light=illum[indices],pose=pose[indices])
    path=out/(task+'_predictions.npz')
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**signal,**{k:np.stack(v) for k,v in allpred.items()});tmp.replace(path)
    (out/(task+'_fits.json')).write_text(json.dumps(diagnostics,indent=1)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',choices=('DTD','KTH','Lytro',*['omit_'+k for k in LIGHTS]))
    p.add_argument('data',type=Path);p.add_argument('out',type=Path);p.add_argument('--threads',type=int,default=3)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(a.threads):compute(a.task,a.data,a.out)
