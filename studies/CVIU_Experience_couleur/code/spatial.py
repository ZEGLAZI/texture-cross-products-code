"""A secondary test of local colour arrangement, with an exact permutation-expectation control."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from pathlib import Path
import numpy as np
from numpy.ctypeslib import ndpointer
from threadpoolctl import threadpool_limits
from extract_palette import sample
from fit import get,standardized,gram,kernels,classify
from analyse import ci_dtd,ci_external,correctness,macro,csv_out

VARIANTS=('local_pairs','permuted_pairs')
def load_spatial(binary):
    lib=ctypes.CDLL(str(binary.resolve()));routine=lib.spatial
    routine.argtypes=[ndpointer(np.uint8,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,ndpointer(np.float64,flags='C_CONTIGUOUS')]
    routine.restype=None
    def transform(im):
        assert im.dtype==np.uint8 and im.ndim==3 and im.shape[2]==3 and min(im.shape[:2])>8
        value=np.empty((2,378))
        routine(np.ascontiguousarray(im),im.shape[0],im.shape[1],value)
        np.testing.assert_allclose(np.square(value).sum(axis=1),1.,rtol=0,atol=1e-12)
        return value
    return transform

def extract(name,images,cache,binary,workers):
    design=get(cache/(name+'_design.npz'));ids=design['ids']
    other=get(cache/(name+'_palettes.npz'));np.testing.assert_array_equal(ids,other['ids'])
    transform=load_spatial(binary)
    def one(i):
        im,digest,_,_=sample(images,ids[i],name)
        assert digest==other['hash224'][i]
        return transform(im)
    with ThreadPoolExecutor(workers) as pool:values=np.stack(list(pool.map(one,range(len(ids)))))
    np.savez_compressed(cache/(name+'_pairs.npz'),ids=ids,values=values)
    print(name,len(ids),'images with local and permutation-expected colour pairs',flush=True)

def fit(name,cache,out):
    corpora=('FMD','Lytro') if name=='Lytro' else (name,)
    features=[get(cache/(c+('_features.npz' if name=='Lytro' else '_original.npz'))) for c in corpora]
    pp=[get(cache/(c+'_palettes.npz')) for c in corpora]
    zz=[get(cache/(c+'_pairs.npz')) for c in corpora]
    for f,p,z in zip(features,pp,zz):np.testing.assert_array_equal(f['ids'],p['ids']);np.testing.assert_array_equal(p['ids'],z['ids'])
    f7=np.concatenate([f['F7'] for f in features]);f8=np.concatenate([f['F8'] for f in features])
    rgb=np.concatenate([p['values'][:,0] for p in pp]);pairs=np.concatenate([z['values'] for z in zz])
    design=get(cache/(name+'_design.npz'));main=get(out/(name+'_predictions.npz'))
    if name=='Lytro':
        src=get(cache/'FMD_design.npz');n=len(src['y'])
        splits=[(main['refit'],np.arange(n,n+len(design['y'])))]
        y=np.r_[src['y'],design['y']]
    else:
        splits=[(design['train'+str(i+1)],te) for i,te in enumerate(main['indices'])]
        y=design['y']
    result={k:[] for k in VARIANTS}
    for tr,te in splits:
        a,b=(gram(standardized(x,tr,dim),tr,te) for x,dim in ((f7,9),(f8,3)))
        base=gram(rgb,tr,te)
        for i,key in enumerate(VARIANTS):
            c=(base+gram(pairs[:,i],tr,te))*.5
            result[key].append(classify(kernels(a,b,c),len(tr),y[tr],len(design['classes']),1.))
    pred={k:(v[0] if name=='Lytro' else np.stack(v)) for k,v in result.items()}
    np.savez_compressed(out/(name+'_spatial_predictions.npz'),ids=main['ids'],y=main['y'],groups=main['groups'],**pred)
    print(name,'spatial models fitted',flush=True)

def summary(cache,out):
    rows=[];det=[]
    for k in ('DTD','KTH','Lytro'):
        main=get(out/(k+'_predictions.npz'));sp=get(out/(k+'_spatial_predictions.npz'));y=main['y']
        for key in ('ids','y','groups'):np.testing.assert_array_equal(sp[key],main[key])
        classes=get(cache/(k+'_design.npz'))['classes']
        vals={p:correctness(sp[p],y) for p in VARIANTS};vals['RGB_hard']=correctness(main['RGB_hard'],y)
        for comparator in ('RGB_hard','permuted_pairs'):
            delta=vals['local_pairs']-vals[comparator]
            ci=(ci_dtd if k=='DTD' else ci_external)(delta,y,sp['groups'],12)
            rows.append(dict(corpus=k,comparator=comparator,local_recall_pct=macro(vals['local_pairs'],y),
                comparator_recall_pct=macro(vals[comparator],y),gain_pp=ci[0],low_adjusted=ci[1],high_adjusted=ci[2]))
        for j,cls in enumerate(classes):
            keep=y==j
            det.append(dict(corpus=k,label=cls,**{p:float(v[keep].mean()*100) for p,v in vals.items()}))
    csv_out(out/'spatial.csv',rows);csv_out(out/'spatial_by_class.csv',det)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);q=p.add_subparsers(dest='cmd',required=True)
    e=q.add_parser('extract');e.add_argument('name');e.add_argument('images',type=Path)
    e.add_argument('cache',type=Path);e.add_argument('binary',type=Path);e.add_argument('--workers',type=int,default=3)
    f=q.add_parser('fit');f.add_argument('name');f.add_argument('cache',type=Path);f.add_argument('out',type=Path);f.add_argument('--threads',type=int,default=3)
    s=q.add_parser('summary');s.add_argument('cache',type=Path);s.add_argument('out',type=Path)
    args=vars(p.parse_args());cmd=args.pop('cmd')
    if cmd=='fit':
        with threadpool_limits(args.pop('threads')):fit(**args)
    elif cmd=='extract':extract(**args)
    else:summary(**args)
