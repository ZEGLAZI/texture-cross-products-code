"""Nested fixed-coefficient trials, redistributed-budget controls and RBF reference."""
import argparse,csv,os,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from calcul_historique import compact
from nested import PROFILES,PEN,fit,macro

ROOT=Path(__file__).resolve().parents[1]
COEFF=('L','S','X','Q','RS','RX','RBF')
GAMMA=(.25,1.,4.,16.)
def stage(raw,tr,ev):
    order=np.r_[tr,ev]
    features,_=compact(raw,training=tr);features*=np.sqrt(2.)
    data=[features[:,:945],features[:,945:],raw['rgb']]
    return ([b[order]@b[tr].T for b in data],[np.einsum('ij,ij->i',b[order],b[order]) for b in data])
def forms(gs,p):
    a,b,c=p;s,w,h=gs
    q=a*s+b*w+c*h
    quad=(a*s)**2+(b*w)**2+(c*h)**2
    cross=(a*b)*s*w+(a*c)*s*h+(b*c)*w*h
    # Preserve each family's original coefficients before adding either one.
    l=2*q/3.25
    yield 'L',l
    yield 'S',l+(.75/3.25)*quad
    yield 'X',l+(1.5/3.25)*cross
    yield 'Q',l+(.75/3.25)*q*q
    yield 'RS',l+(.75/(3.25*(a*a+b*b+c*c)))*quad
    yield 'RX',l+(.75/(3.25*(a*b+a*c+b*c)))*cross
def kernels(gs,ds,j):
    p=PROFILES[j];a,b,c=p;pg=PROFILES[0]
    n=gs[0].shape[1]
    diag=sum(pg[i]*ds[i] for i in range(3))
    for name in COEFF:
        out=np.empty_like(gs[0])
        for start in range(0,len(out),64):
            ix=slice(start,start+64)
            s,w,h=(k[ix] for k in gs)
            q=a*s+b*w+c*h
            if name=='RBF':
                weighted=pg[0]*s+pg[1]*w+pg[2]*h
                out[ix]=np.exp(-GAMMA[j]*np.maximum(0,diag[ix,None]+diag[:n]-2*weighted))
            elif name=='L':out[ix]=2*q/3.25
            elif name=='Q':out[ix]=(2*q+.75*q*q)/3.25
            elif name in ('S','RS'):
                scale=a*a+b*b+c*c if name=='RS' else 1.
                out[ix]=(2*q+.75*((a*s)**2+(b*w)**2+(c*h)**2)/scale)/3.25
            else:
                sc=(a*b+a*c+b*c) if name=='RX' else .5
                out[ix]=(2*q+.75*((a*b)*s*w+(a*c)*s*h+(b*c)*w*h)/sc)/3.25
        yield name,out
        del out
def run_fold(file,raw,y,tr,te,inner):
    if file.exists():return
    nc=len(np.unique(y))
    assert all(np.intersect1d(t,v).size==0 and np.isin(np.r_[t,v],tr).all() for t,v in inner)
    assert len(np.intersect1d(tr,te))==0
    grid=np.zeros((len(COEFF),4,6));store={'test':te,'train':tr,'labels':y[te],'modes':np.array(COEFF)}
    for h,(it,iv) in enumerate(inner):
        store['inner_train_'+str(h)]=it;store['inner_val_'+str(h)]=iv
        gs,ds=stage(raw,it,iv)
        for j in range(4):
            for idx,(name,kernel) in enumerate(kernels(gs,ds,j)):
                for n,lam in enumerate(PEN):
                    pred=fit(kernel,len(it),y[it],lam,nc).argmax(1)
                    grid[idx,j,n]+=macro(pred,y[iv])/len(inner)
                del kernel
        del gs,ds
    gs,ds=stage(raw,tr,te)
    outer=np.zeros((len(COEFF),4,len(te)),np.int16)
    fixed=np.zeros((4,4,len(te)),np.int16);unit=fixed.copy()
    for j in range(4):
        d=dict(forms(ds,PROFILES[j]))
        for idx,(name,kernel) in enumerate(kernels(gs,ds,j)):
            best=np.argmax(grid[idx,j]);lam=PEN[best]
            def decision(mat,reg):return fit(mat,len(tr),y[tr],reg,nc).argmax(1)
            outer[idx,j]=decision(kernel,lam)
            if idx<4:
                fixed[idx,j]=outer[idx,j] if lam==1 else decision(kernel,1.)
                dd=np.sqrt(np.maximum(d[name],0))
                for start in range(0,len(dd),64):
                    ix=slice(start,start+64);den=dd[ix,None]*dd[:len(tr)]
                    np.divide(kernel[ix],den,out=kernel[ix],where=den>0)
                    kernel[ix][den==0]=0
                unit[idx,j]=decision(kernel,1.)
            del kernel
    store.update(grid=grid,tuned=outer,fixed=fixed,unit=unit)
    np.savez_compressed(file,**store)
    print(file.stem,flush=True)
def select(arr):
    return np.asarray([arr['tuned'][i,np.unravel_index(np.argmax(g),g.shape)[0]] for i,g in enumerate(arr['grid'])])
def main(a):
    dest=a.out;dest.mkdir(exist_ok=True,parents=True)
    def blocks(path):return {k:np.load(path)[k] for k in ('F7','F8','rgb')}
    if a.corpus=='DTD':
        sys.path.insert(0,str(a.supp/'dtd'))
        from predire import partitions,load_blocks
        ids=np.load(a.supp/'dtd/results/identites.npz');y=ids['y'];g=ids['groupes']
        raw=load_blocks(a.cache/'hsmt',ids['ids'],['F7','F8','rgb'])
        for k,tr,te in partitions(ids,g):
            it,iv=ids['train'+str(k)],ids['val'+str(k)]
            iv=iv[np.isin(iv,tr)];it=it[np.isin(it,tr)&~np.isin(g[it],g[iv])]
            run_fold(dest/f'DTD_{k:02}.npz',raw,y,tr,te,[(it,iv)])
    elif a.corpus=='KTH':
        d=a.supp/'external/data';r=list(csv.DictReader((d/'kth_images.csv').open()))
        y=np.unique([x['materiau'] for x in r],return_inverse=True)[1]
        g=np.array([x['specimen'] for x in r]);sha=np.array([x['sha_pixels'] for x in r]);raw=blocks(d/'kth.npz')
        for k in 'abcd':
            te=np.flatnonzero(g==k);tr=np.flatnonzero((g!=k)&~np.isin(sha,sha[te]));inn=[]
            for h in sorted(set('abcd')-{k}):
                iv=tr[g[tr]==h];it=tr[(g[tr]!=h)&~np.isin(sha[tr],sha[iv])];inn.append((it,iv))
            run_fold(dest/f'KTH_{k}.npz',raw,y,tr,te,inn)
    elif a.corpus=='Flower':
        d=a.supp/'external/data'
        paths=[d/f'{n}.npz' for n in ('tf','plantnet','nouveau_2026')]+[a.cohort/'features.npz']
        parts=[np.load(p) for p in paths]
        raw={k:np.concatenate([f[k] for f in parts]) for k in ('F7','F8','rgb')}
        y=(np.concatenate([f['classe'] for f in parts])[:,None]==[10,12,9]).argmax(1)
        groups=np.array([int(r['credit_group']) for r in csv.DictReader((d/'tf_manifest.csv').open())])
        tr=np.arange(len(groups));te=np.arange(len(tr),len(y));g=np.random.default_rng(19112026).permutation(np.unique(groups))
        inn=[(tr[~np.isin(groups,g[j::3])],tr[np.isin(groups,g[j::3])]) for j in range(3)]
        run_fold(dest/'Flower.npz',raw,y,tr,te,inn)
    elif a.corpus=='Textile':
        z=np.load(a.textile/'features.npz');part=np.load(a.textile/'partitions.npz');y=z['y']
        assert np.array_equal(z['keys'],part['keys'])
        raw={k:z[k] for k in ('F7','F8','rgb')}
        for r,assignment in enumerate(part['folds']):
            for k in range(5):
                tr=np.flatnonzero(assignment!=k);te=np.flatnonzero(assignment==k)
                val=(k+1)%5;iv=tr[assignment[tr]==val];it=tr[assignment[tr]!=val]
                run_fold(dest/f'Textile_{r:02}_{k}.npz',raw,y,tr,te,[(it,iv)])
    else:
        z=np.load(a.input);raw={k:z[k] for k in ('F7','F8','rgb')}
        run_fold(dest/f'{a.corpus}.npz',raw,z['y'],z['train'],z['test'],[(z['in_train'],z['val'])])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('corpus',choices=['DTD','KTH','Flower','Textile','FMD','Outex'])
    p.add_argument('--supp',type=Path,default=ROOT/'historical');p.add_argument('--cache',type=Path,default=ROOT/'dtd_cache');p.add_argument('--cohort',type=Path,default=ROOT/'historical/revision/results/cohort')
    p.add_argument('--textile',type=Path,default=ROOT/'clothing/data');p.add_argument('--input',type=Path);p.add_argument('--out',type=Path,default=ROOT/'results/kernels');p.add_argument('--threads',type=int,default=3)
    a=p.parse_args()
    with threadpool_limits(a.threads):main(a)
