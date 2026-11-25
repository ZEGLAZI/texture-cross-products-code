"""Equal-budget, training-only kernel selection with complete per-fold records."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'protocol.json').read_text())
PROFILES=np.array(CFG['profiles']);PEN=CFG['ridge_grid'];METHODS=CFG['methods']
def macro(pred,y):
    return 100*np.mean([np.mean(pred[y==c]==c) for c in np.unique(y)])
def kernel(gs,p,method):
    s,w,c=gs
    q=p[0]*s+p[1]*w+p[2]*c
    if method=='Q':return (2*q+.75*q*q)/3.25
    out=2*q
    if method=='W':out+=.25*s*w+.5*w*c
    if method=='S':out+=.25*s*w+.5*s*c
    if method=='C':out+=.25*s*c+.5*w*c
    if method=='U':out+=.25*(s*w+s*c+w*c)
    return out/3.25
def fit(k,n,lab,pen,dim):
    targets=np.eye(dim)[lab];prior=targets.mean(0);targets-=prior
    sysmat=np.array(k[:n],dtype='float64',order='F',copy=True);sysmat.flat[::n+1]+=pen
    coef=cho_solve(cho_factor(sysmat,lower=True,check_finite=False,overwrite_a=True),targets,check_finite=False)
    return k[n:]@coef+prior
def makegrams(raw,tr,va,compact):
    x,_=compact(raw,training=tr);x*=np.sqrt(2)
    order=np.r_[tr,va]
    return [m[order]@m[tr].T for m in (x[:,:945],x[:,945:],raw['rgb'])]
def runfold(name,tr,te,inner,y,raw,dim,compact,out,group):
    start=time.time()
    grid=np.zeros((len(METHODS),len(PROFILES),len(PEN))); mindex={m:i for i,m in enumerate(METHODS)}
    for it,iv in inner:
        assert not np.intersect1d(group[it],group[iv]).size
        if not set(it).issubset(set(tr)) or not set(iv).issubset(set(tr)):raise ValueError('test leakage')
        g=makegrams(raw,it,iv,compact)
        for j,p in enumerate(PROFILES):
            for m in METHODS:
                k=kernel(g,p,m)
                for l,pen in enumerate(PEN):
                    grid[mindex[m],j,l]+=macro(fit(k,len(it),y[it],pen,dim).argmax(1),y[iv])/len(inner)
                del k
        del g
    g=makegrams(raw,tr,te,compact);archive={'test':te,'train':tr,'val_grid':grid}
    records={}
    for jj,(it,iv) in enumerate(inner):archive['inner_train_'+str(jj)]=it;archive['inner_val_'+str(jj)]=iv
    for i,m in enumerate(METHODS):
        j,l=np.unravel_index(np.argmax(grid[i]),grid[i].shape)
        k=kernel(g,PROFILES[j],m);scores=fit(k,len(tr),y[tr],PEN[l],dim)
        archive[m]=scores.argmax(1);archive[m+'_scores']=scores.astype('float32')
        records[m]={'p':int(j),'lambda':PEN[l],'validation':float(grid[i,j,l]),'recall':macro(archive[m],y[te])}
        del k
    np.savez_compressed(out/(name+'.npz'),**archive)
    (out/(name+'.json')).write_text(json.dumps(records,indent=2)+'\n')
    print(name,'seconds',round(time.time()-start),{m:round(v['recall'],2) for m,v in records.items()},flush=True)
def run(args):
    sys.path.insert(0,str((args.supp/'dtd').resolve()))
    from predire import load_blocks,compact,partitions
    out=args.output;out.mkdir(exist_ok=True,parents=True)
    if args.task=='DTD':
        ids=dict(np.load(args.supp/'dtd/results/identites.npz'));y=ids['y'];groups=ids['groupes']
        raw=load_blocks(args.cache/'hsmt',ids['ids'],['F7','F8','rgb'])
        for k,tr,te in partitions(ids,groups):
            tag=f'DTD_{k:02}';it=ids['train'+str(k)];iv=ids['val'+str(k)]
            iv=iv[np.isin(iv,tr)];it=it[np.isin(it,tr)&~np.isin(groups[it],groups[iv])]
            if not(out/(tag+'.json')).exists():runfold(tag,tr,te,[(it,iv)],y,raw,47,compact,out,groups)
    elif args.task=='KTH':
        base=args.supp/'external/data';raw=dict(np.load(base/'kth.npz'))
        rows=list(csv.DictReader((base/'kth_images.csv').open()))
        y=np.unique([r['materiau'] for r in rows],return_inverse=True)[1];sp=np.array([r['specimen'] for r in rows]);h=np.array([r['sha_pixels'] for r in rows])
        for k in 'abcd':
            te=np.flatnonzero(sp==k);tr=np.flatnonzero((sp!=k)&~np.isin(h,h[te]));inner=[]
            for g in sorted(set('abcd')-{k}):
                iv=tr[sp[tr]==g];it=tr[(sp[tr]!=g)&~np.isin(h[tr],h[iv])];inner.append((it,iv))
            tag='KTH_'+k
            if not(out/(tag+'.json')).exists():runfold(tag,tr,te,inner,y,raw,11,compact,out,h)
    else:
        base=args.supp/'external/data'; parts=[np.load(base/(tag+'.npz')) for tag in ('tf','plantnet','nouveau_2026')]
        raw={v:np.concatenate([q[v] for q in parts]) for v in ('F7','F8','rgb')}
        y=(np.concatenate([q['classe'] for q in parts])[:,None]==[10,12,9]).argmax(1)
        groups=np.array([int(r['credit_group']) for r in csv.DictReader((base/'tf_manifest.csv').open())]);keys=np.unique(groups)
        fold=np.random.default_rng(CFG['seed']).permutation(keys);tr=np.arange(len(groups));te=np.arange(len(tr),len(y))
        inn=[(tr[~np.isin(groups,fold[k::3])],tr[np.isin(groups,fold[k::3])]) for k in range(3)]
        if not(out/'Flowers.json').exists():runfold('Flowers',tr,te,inn,y,raw,3,compact,out,groups)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('task',choices=['DTD','Flowers','KTH']);p.add_argument('supp',type=Path);p.add_argument('cache',type=Path);p.add_argument('output',type=Path);p.add_argument('--threads',type=int,default=3);a=p.parse_args()
    with threadpool_limits(a.threads):run(a)
