"""Train the fixed models on the locked cohort, without using external folds for tuning."""
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/frozen'))
from calcul_historique import compact
from degree_two import decomposition
from nested import PEN, PROFILES, fit, kernel, macro

METHODS=('squares','crosses','additive','clbp','dino','dino_squares','dino_additive')
SINGLE={'clbp','dino'}

def representations():
    with np.load(ROOT/'data/features.npz') as f:
        raw={k:f[k] for k in ('F7','F8','rgb')}
        keys,y,classes=f['keys'],f['y'],f['classes']
    with np.load(ROOT/'data/reference_features.npz') as f:
        assert np.array_equal(keys,f['keys'])
        counts=f['clbp_counts'].astype(float)
        token=f['dino_cls'].astype(float)
    clbp=np.sqrt(counts/counts.sum(1,keepdims=True))
    dino=token/np.linalg.norm(token,axis=1,keepdims=True)
    return raw, {'clbp':clbp, 'dino':dino}, keys, y, classes

def make_stage(train, evaluation, raw, fixed):
    """Kernels are always n_train+n_evaluation rows by n_train columns."""
    order=np.r_[train,evaluation]
    parts,_=compact(raw,training=train)
    parts*=np.sqrt(2.)
    blocks=(parts[:, :945],parts[:, 945:],raw['rgb'])
    grams=[block[order]@block[train].T for block in blocks]
    diagonal=[np.einsum('ij,ij->i',block[order],block[order]) for block in blocks]
    external={name:block[order]@block[train].T for name,block in fixed.items()}
    return grams,diagonal,external

def make_kernel(method,profile,stage):
    grams,diagonal,external=stage
    if method in SINGLE:
        return external[method]
    component=method.removeprefix('dino_')
    if component=='additive':
        value=kernel(grams,profile,'Add')
    else:
        value=decomposition(grams,profile,component)
    if method.startswith('dino_'):
        diag=(kernel(diagonal,profile,'Add') if component=='additive'
              else decomposition(diagonal,profile,component))
        norm=np.sqrt(np.maximum(diag,0))
        denominator=norm[:,None]*norm[:value.shape[1]]
        value=np.divide(value,denominator,out=np.zeros_like(value),where=denominator>0)
        value=(value+external['dino'])*0.5
    return value

def outer_fold(assignment,k,raw,fixed,y,methods=METHODS):
    test=np.flatnonzero(assignment==k)
    valid=np.flatnonzero(assignment==(k+1)%5)
    train=np.flatnonzero((assignment!=k)&(assignment!=(k+1)%5))
    refit=np.flatnonzero(assignment!=k)
    assert np.intersect1d(test,refit).size==0
    assert np.array_equal(np.sort(np.r_[train,valid]),refit)
    assert all(len(np.unique(y[t]))==6 for t in (test,valid,train))
    val_stage=make_stage(train,valid,raw,fixed)
    test_stage=make_stage(refit,test,raw,fixed)
    grid=np.full((len(methods),len(PROFILES),len(PEN)),np.nan)
    choices=np.zeros((len(methods),2),np.int8)
    decisions=np.full((len(methods),len(test)),-1,np.int8)
    for index,method in enumerate(methods):
        profiles=PROFILES[:1] if method in SINGLE else PROFILES
        for j,p in enumerate(profiles):
            gram=make_kernel(method,p,val_stage)
            for h,penalty in enumerate(PEN):
                scores=fit(gram,len(train),y[train],penalty,6)
                grid[index,j,h]=macro(scores.argmax(1),y[valid])
        j,h=np.unravel_index(np.nanargmax(grid[index]),grid[index].shape)
        choices[index]=[j,h]
        gram=make_kernel(method,PROFILES[j],test_stage)
        decisions[index]=fit(gram,len(refit),y[refit],PEN[h],6).argmax(1)
    return test,decisions,choices,grid

def run():
    raw,fixed,keys,y,classes=representations()
    with np.load(ROOT/'data/partitions.npz') as split:
        assert np.array_equal(keys,split['keys']) and np.array_equal(y,split['y'])
        folds,seeds=split['folds'],split['seeds']
    n=len(folds)
    predictions=np.full((n,len(METHODS),len(keys)),-1,np.int8)
    choices=np.zeros((n,5,len(METHODS),2),np.int8)
    validation=np.full((n,5,len(METHODS),len(PROFILES),len(PEN)),np.nan)
    for r,assignment in enumerate(folds):
        for k in range(5):
            test,decisions,choice,grid=outer_fold(assignment,k,raw,fixed,y)
            assert np.all(predictions[r][:,test]==-1)
            predictions[r][:,test]=decisions
            choices[r,k]=choice
            validation[r,k]=grid
        assert np.all((predictions[r]>=0)&(predictions[r]<6))
        if (r+1)%5==0:
            print(f'Completed {r+1}/{n} repetitions',flush=True)
    np.savez_compressed(ROOT/'results/predictions.npz',pred=predictions,
                        validation=validation,choices=choices,methods=METHODS,
                        keys=keys,y=y,classes=classes,folds=folds,seeds=seeds)

if __name__=='__main__':
    with threadpool_limits(2):
        run()
