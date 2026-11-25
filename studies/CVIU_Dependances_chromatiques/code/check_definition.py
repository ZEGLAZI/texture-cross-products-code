"""Independent, direct rectangle sums and analytical tests for the proposed measurements."""
import argparse
import json
from pathlib import Path
import numpy as np
from extract import routine

class Shuffle:
    """Portable mirror of the published uint64 SplitMix/Fisher–Yates stream."""
    def __init__(self,seed):self.state=seed;self.mask=(1<<64)-1
    def next(self):
        self.state=(self.state+0x9e3779b97f4a7c15)&self.mask;v=self.state
        v=((v^(v>>30))*0xbf58476d1ce4e5b9)&self.mask
        v=((v^(v>>27))*0x94d049bb133111eb)&self.mask
        return v^(v>>31)
    def permute(self,flag):
        for n in range(len(flag),1,-1):
            lim=(1<<64)%n;x=self.next()
            while x<lim:x=self.next()
            j=x%n;flag[n-1],flag[j]=int(flag[j]),int(flag[n-1])

def literal(logs,seed):
    u=logs.mean(axis=2)
    col=np.stack(((logs[:,:,0]-logs[:,:,1])/np.sqrt(2),
        (logs[:,:,0]+logs[:,:,1]-2*logs[:,:,2])/np.sqrt(6)),axis=-1)
    mu=col.mean(axis=(0,1))
    joint=np.dstack((u,col));nh,nw=u.shape
    vals=np.zeros((2,3,6,5));diag=np.zeros((6,6))
    prng=Shuffle(seed)
    def haar(y,x,r,axis):
        tile=joint[y-r:y+r,x-r:x+r]
        left,right=(tile[:,:r],tile[:,r:]) if axis==0 else (tile[:r],tile[r:])
        return right.mean(axis=(0,1))-left.mean(axis=(0,1))
    for ic,(r,axis) in enumerate((r,axis) for r in (2,4,8) for axis in (0,1)):
        rec=[]
        for y in range(2*r,nh-2*r+1,2):
            for x in range(2*r,nw-2*r+1,2):
                h=haar(y,x,r,axis);parent=haar(y,x,r*2,axis)[0]
                pal=col[y-r:y+r,x-r:x+r].mean(axis=(0,1))-mu
                rec.append([h[0],parent,*pal,*(np.sign(h[0])*h[1:])])
        rr=np.asarray(rec);f=np.abs(rr[:,0]);c=np.abs(rr[:,1])
        selected=(f>=np.median(f))&(f>1e-8)&(c>1e-8);rr=rr[selected]
        if not len(rr):continue
        f,c=np.abs(rr[:,:2]).T
        strata=2*(f>np.median(f)).astype(int)+(c>np.median(c))
        B=(rr[:,0]>0)==(rr[:,1]>0)
        diag[ic,:2]=len(rr),B.mean()
        for g in range(4):
            take=strata==g;diag[ic,2+g]=take.sum()
            if not np.any(take):continue
            flag=B[take].astype(int);permuted=flag.copy();average=np.zeros(len(flag))
            for draw in range(4):prng.permute(permuted);average+=permuted/4
            for branch,s in enumerate((slice(2,4),slice(4,6))):
                v=rr[take,s];t=v/(1+np.linalg.norm(v,axis=1,keepdims=True));a,b=t.T
                phi=np.column_stack((a,b,a*a,a*b,b*b))
                vals[branch,0,ic]+=4*phi.T@(flag-flag.mean())/len(rr)
                vals[branch,1,ic]+=phi.sum(axis=0)/len(rr)
                vals[branch,2,ic]+=4*phi.T@(average-flag.mean())/len(rr)
    return vals.reshape(2,3,30),diag

def compiled(lib,logs,seed):
    out=np.zeros((2,3,30));diag=np.zeros((6,6));v=np.ascontiguousarray(logs,dtype=np.float64)
    lib.from_log_rgb(v,*v.shape[:2],seed,out,diag);return out,diag

def lab_reference(im):
    x=im.reshape(-1,3).astype(float)/255.
    lin=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
    m=np.array([[.4124564,.2126729,.0193339],[.3575761,.7151522,.1191920],[.1804375,.0721750,.9503041]])
    xyz=lin@m/np.array([.95047,1.,1.08883]);d=6/29
    v=np.where(xyz>d**3,np.cbrt(xyz),xyz/(3*d*d)+4/29)
    l=np.column_stack((116*v[:,1]-16,500*(v[:,0]-v[:,1]),200*(v[:,1]-v[:,2])))
    cen=l-l.mean(0)
    return np.stack((l.mean(0),l.std(0),np.cbrt((cen**3).mean(0))),axis=1).ravel()
def run(binary):
    lib=routine(binary);rng=np.random.default_rng(32861);err=0.;inv=0.;chrom_ach=0.;laberr=0.
    for seed in (13,71,614,55732,234490):
        x=rng.uniform(-5.,-1.,(47,53,3))
        fast,d=compiled(lib,x,seed);slow,rd=literal(x,seed)
        np.testing.assert_allclose(d,rd,atol=4e-13,rtol=0)
        err=max(err,float(abs(fast-slow).max()))
        shifted,_=compiled(lib,x+np.log([1.4,.7,1.1]),seed)
        inv=max(inv,float(abs(fast-shifted).max()))
        gray=np.repeat(x[:,:,:1],3,axis=2)
        ach,_=compiled(lib,gray,seed);chrom_ach=max(chrom_ach,float(abs(ach).max()))
        rgb=rng.integers(0,256,x.shape,dtype=np.uint8)
        L=np.empty(9);edge=np.empty((2,3,30));debug=np.empty((6,6))
        lib.colour_describe(rgb,*rgb.shape[:2],seed,L,edge,debug)
        laberr=max(laberr,float(abs(L-lab_reference(rgb)).max()))
    assert max(err,inv,chrom_ach)<2e-11 and laberr<2e-8,(err,inv,chrom_ach,laberr)
    # Identical marginal samples, opposite alignment, in a single balanced stratum.
    indicator=np.array([1.,1.,0.,0.]);vals=np.array([.6,.6,-.2,-.2])
    covariance=lambda v: float(4*np.mean((indicator-indicator.mean())*v))
    np.testing.assert_allclose([covariance(vals),covariance(vals[::-1])],[.8,-.8])
    return dict(synthetic_images=5,max_direct_reference_error=err,max_continuous_diagonal_gain_error=inv,
                max_achromatic_response=chrom_ach,max_Lab_against_vectorized_reference=laberr,
                site_level_marginal_collision=[covariance(vals),covariance(vals[::-1])])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('binary',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();ans=run(a.binary);print(json.dumps(ans,indent=2))
    if a.output:a.output.write_text(json.dumps(ans,indent=2)+'\n')
