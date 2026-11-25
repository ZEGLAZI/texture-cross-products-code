"""Classical and frozen-encoder DTD references with a training-only ridge search."""
import argparse,ctypes,csv,json,sys,time,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from numpy.ctypeslib import ndpointer
from PIL import Image
from threadpoolctl import threadpool_limits
from nested import fit,macro,PEN,kernel
from ablations import embedded
def prep(a):
    a.work.mkdir(parents=True,exist_ok=True)
    so=a.work/'operator.so';subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','-fopenmp','-ffp-contract=off',str(a.supp/'dtd/reference/operateurs.cpp'),'-o',str(so)],check=True)
    fun=ctypes.CDLL(str(so.resolve())).oclbp;fun.argtypes=[ndpointer(np.uint8,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,ndpointer(np.int32,flags='C_CONTIGUOUS')];fun.restype=None
    yy,xx=np.mgrid[0:224,0:224]/223
    def features(key):
        with Image.open(a.images/key) as im:pix=np.asarray(im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC))
        count=np.zeros(432,np.int32);fun(np.ascontiguousarray(pix),224,224,count);count=count.reshape(12,36);oc=np.sqrt(count/count.sum(1,keepdims=True)/12).ravel()
        rgb=pix/255;lum=rgb.mean(2);dy,dx=np.gradient(lum)
        v=np.stack([xx,yy,*rgb.transpose(2,0,1),dx,dy,np.hypot(dx,dy)],2);coords=[];tri=np.triu_indices(8)
        for g in (4,2,1):
            size=224//g
            for i in range(g):
                for j in range(g):
                    x=v[i*size:(i+1)*size,j*size:(j+1)*size].reshape(-1,8);x=x-x.mean(0);m=x.T@x/(len(x)-1)+np.eye(8)*1e-4;l,u=np.linalg.eigh(m);coords.append(((u*np.log(l))@u.T)[tri])
        return oc,np.array(coords)
    ids=np.load(a.supp/'dtd/results/identites.npz')['ids']
    for start in range(0,len(ids),96):
        batch=ids[start:start+96];fn=a.work/f'{start:05d}.npz'
        if fn.exists():continue
        with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(features,batch))
        np.savez(fn,ids=batch,oc=np.stack([r[0] for r in rows]),cov=np.stack([r[1] for r in rows])[:,None])
def score(a):
    sys.path.insert(0,str((a.supp/'dtd').resolve()));from predire import load_blocks,partitions,normal,compact
    ids=dict(np.load(a.supp/'dtd/results/identites.npz'));keys=ids['ids'];y=ids['y'];features=load_blocks(a.work,keys,['cov','oc'])
    count=load_blocks(a.cache/'clbp',keys,['clbp'])['clbp'];features['clbp']=np.sqrt(count/count.sum(1,keepdims=True))
    for key,folder in [('crop',a.cache/'dino'),('full',a.dino)]:features[key]=normal(load_blocks(folder,keys,['dino'])['dino'])
    raw=load_blocks(a.cache/'hsmt',keys,['F7','F8','rgb']);methods=['clbp','oc','cov','crop','full','full_add','full_W'];a.out.mkdir(exist_ok=True,parents=True)
    for k,tr,te in partitions(ids,ids['groupes']):
        file=a.out/f'fold_{k:02}.npz'
        if file.exists():continue
        it=ids['train'+str(k)];iv=ids['val'+str(k)];iv=iv[np.isin(iv,tr)];it=it[np.isin(it,tr)&~np.isin(ids['groupes'][it],ids['groupes'][iv])]
        archive={'test':te}
        for stage,(t,v) in enumerate([(it,iv),(tr,te)]):
            order=np.r_[t,v]
            def gram(f):return f[order]@f[t].T
            blocks,_=compact(raw,training=t);blocks*=np.sqrt(2)
            for name in methods:
                if name.startswith('full_'):
                    mode='W' if name=='full_W' else 'Add'
                    K=(gram(features['full'])+kernel([gram(blocks[:,:945]),gram(blocks[:,945:]),gram(raw['rgb'])],[.25,.25,.5],mode))/2
                else:K=gram(embedded(features['cov'],8,t)) if name=='cov' else gram(features[name])
                if stage==0:
                    val=np.array([macro(fit(K,len(t),y[t],lam,47).argmax(1),y[v]) for lam in PEN]);archive[name+'_val']=val
                else:
                    for label,lam in [('selected',PEN[np.argmax(archive[name+'_val'])]),('fixed',1)]:
                        archive[name+'_'+label]=fit(K,len(t),y[t],lam,47).argmax(1).astype('uint8')
                del K
        np.savez_compressed(file,**archive);print('reference fold',k,flush=True)
    old=np.load(a.supp/'dtd/results/decisions.npz');arc=[np.load(a.out/f'fold_{k:02}.npz') for k in range(1,11)];tests=old['tests'];out={}
    for name in methods:
        out[name]={z:float(100*np.mean(np.stack([a[name+'_'+z] for a in arc])==y[tests])) for z in ('fixed','selected')}
    (a.out/'scores.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','score'])
    for n in ('supp','cache','images','work','dino','out'):p.add_argument(n,type=Path)
    a=p.parse_args()
    with threadpool_limits(1 if a.stage=='prepare' else 3):(prep if a.stage=='prepare' else score)(a)
