"""Independent feature-level check of both quadratic margins on one outer fold."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

def run(data,fold,out):
    with np.load(data) as f:z={k:f[k] for k in ('F7','F8','rgb','y')}
    with np.load(fold) as f:g={k:f[k] for k in ('train','test','fixed','unit')}
    it,iv=g['train'],g['test'];order=np.r_[it,iv];n=len(it);blocks=[]
    for k,d in [('F7',9),('F8',3)]:
        x=z[k].astype(float);i,j=np.triu_indices(d);x[...,np.flatnonzero(i!=j)]*=np.sqrt(2.)
        train=x[it].transpose(1,0,2,3).reshape(x.shape[1],-1,x.shape[-1])
        mean=train.mean(axis=1)[:,None,:];std=np.maximum(train.std(axis=1),1e-4)[:,None,:]
        x=np.clip((x-mean)/std,-12,12).reshape(len(x),-1)
        norm=np.linalg.norm(x,axis=1,keepdims=True)
        blocks.append(np.divide(x,norm,out=np.zeros_like(x),where=norm>0))
    blocks.append(z['rgb']);p=np.array([.25,.25,.5])
    # Build square and mixed monomials explicitly on whole-image inner products.
    gram=[b[order]@b[order].T for b in blocks]
    weighted=[w*k for w,k in zip(p,gram)]
    l=(2/3.25)*sum(weighted);ss=sum(k*k for k in weighted)*(.75/3.25)
    xx=sum(weighted[i]*weighted[j] for i,j in ((0,1),(0,2),(1,2)))*(1.5/3.25)
    q=(2*sum(weighted)+.75*sum(weighted)**2)/3.25
    err=np.max(np.abs(q-l-ss-xx));assert err<1e-12
    target=np.eye(len(np.unique(z['y'])))[z['y'][it]];avg=target.mean(0)
    mis={}
    for m,k in enumerate([l,l+ss,l+xx,q]):
        dd=np.sqrt(np.maximum(np.diag(k),0));den=np.outer(dd,dd)
        for key,mat in [('fixed',k),('unit',np.divide(k,den,out=np.zeros_like(k),where=den>0))]:
            system=mat[:n,:n]+np.eye(n)
            coeff=solve(system,target-avg,assume_a='sym')
            yp=(mat[n:,:n]@coeff+avg).argmax(1)
            mis[f'{key}_{m}']=int((yp!=g[key][m,0]).sum())
    assert not any(mis.values()),mis
    return dict(test_count=len(iv),max_gram_identity_residual=float(err),decision_mismatches=mis)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('data','fold','out'):p.add_argument(n,type=Path)
    a=p.parse_args()
    with threadpool_limits(1):res=run(a.data,a.fold,a.out)
    a.out.write_text(json.dumps(res,indent=2)+'\n')
