"""Independent vectorized reference, to check linear interpolation, rank ties and histogram layout."""
import argparse
import numpy as np
from palettes import load

def soft(coords,centres):
    bins=[];weights=[]
    for coord,knots in zip(coords,centres,strict=True):
        knots=np.asarray(knots)
        low=np.clip(np.searchsorted(knots,coord,side='right')-1,0,2)
        weight=np.clip((coord-knots[low])/(knots[low+1]-knots[low]),0.,1.)
        bins.append(low);weights.append(weight)
    hist=np.zeros(64)
    for i,j,k in np.ndindex(2,2,2):
        indices=(bins[0]+i)*16+(bins[1]+j)*4+bins[2]+k
        mass=np.prod([w if hi else 1-w for w,hi in zip(weights,(i,j,k))],axis=0)
        hist+=np.bincount(indices,weights=mass,minlength=64)
    return np.sqrt(hist/len(coords[0]))

def reference(image):
    x=image.reshape(-1,3).astype(float);rgb=np.arange(4)*64.+31.5
    linear=x/255.;linear=np.where(linear<=.04045,linear/12.92,((linear+.055)/1.055)**2.4)
    xyz=(linear@np.array([[.4124564,.2126729,.0193339],[.3575761,.7151522,.1191920],
                        [.1804375,.0721750,.9503041]]))/[.95047,1.,1.08883]
    delta=6/29;f=np.where(xyz>delta**3,np.cbrt(xyz),xyz/(3*delta**2)+4/29)
    l=116*f[:,1]-16;a=500*(f[:,0]-f[:,1]);b=200*(f[:,1]-f[:,2])
    k=np.arange(4)*25.+12.5;c=np.array([-45.,-10.,10.,45.])
    idx=lambda m,points:np.searchsorted((points[:-1]+points[1:])/2,m,side='right')
    h=lambda ix:np.sqrt(np.bincount(ix,minlength=64)/len(l))
    level=np.clip(np.floor(2.55*l+.5).astype(int),0,255);counts=np.bincount(level,minlength=256)
    ranked=100*(counts.cumsum()-.5*counts)[level]/len(level)
    xr=x.astype(int)//64
    return np.stack([h(16*xr[:,0]+4*xr[:,1]+xr[:,2]),soft(x.T,[rgb]*3),
        h(16*idx(l,k)+4*idx(a,c)+idx(b,c)),soft((l,a,b),(k,c,c)),soft((ranked,a,b),(k,c,c))])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('binary');q=p.parse_args()
    fast=load(q.binary);rng=np.random.default_rng(20261120)
    sample=[rng.integers(0,256,(28,31,3),dtype=np.uint8) for _ in range(24)]
    sample.extend(np.full((9,12,3),v,dtype=np.uint8) for v in (0,255,(40,171,48),(127,127,127)))
    for im in sample:np.testing.assert_allclose(fast(im),reference(im),atol=2e-13,rtol=0)
    print(len(sample),'images checked against independent vectorized reference')
