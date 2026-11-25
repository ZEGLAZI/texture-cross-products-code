"""Finite-sample radial moment recovery for independent or replicated isotropic vectors."""
from pathlib import Path
import argparse
import csv
import numpy as np
from scipy.special import gammaln
from radial_assess import theory, nearest

def experiment(out):
    rng=np.random.default_rng(19112026)
    rows=[]
    for beta in (.2,.3,.5,1.,2.):
        # R^2 = c T^(1/beta); c sets E[R^2]=3 in dimension three.
        shape=3/(2*beta)
        logc=np.log(3)+gammaln(shape)-gammaln(shape+1/beta)
        for n in (294,1176,4704):
            for replicas in (1,6):
                ratios=[]
                for _ in range(100):
                    p=n//replicas
                    v=rng.normal(size=(p,3));v/=np.linalg.norm(v,axis=1,keepdims=True)
                    rad=np.exp(.5*(logc+np.log(rng.gamma(shape,size=p))/beta))
                    x=v*rad[:,None];x-=x.mean(axis=0)
                    cov=(x.T@x)*replicas/(n-1)+1e-4*np.eye(3)
                    q=np.einsum('ij,jk,ik->i',x,np.linalg.inv(cov),x).clip(1e-8)
                    ratios.append((q*q).mean()/max(1e-8,q.mean()**2))
                rr=np.array(ratios);orig=nearest(rr,False);ext=nearest(rr,True)
                rows.append(dict(beta_true=beta,vectors=n,replicas=replicas,independent_draws=p,runs=100,
                    population_ratio=theory(beta),median_ratio=np.median(rr),native_median=np.median(orig),
                    expanded_median=np.median(ext),native_lower_pct=100*(orig==.3).mean(),
                    out_of_range_high_pct=100*(rr>theory(.3)).mean(),expanded_lower_pct=100*(ext==.12).mean()))
    out.parent.mkdir(exist_ok=True,parents=True)
    with out.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);experiment(p.parse_args().output)
