"""Run the pre-specified Outex_TC_00031 transfer if its provider archive is available."""
import argparse,csv,hashlib,io,json,tarfile
from pathlib import Path,PurePosixPath
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
from assess import run_fold,select
from extracteur import make_estimator,compact_features
from histogramme_oxford import describe
import torch

def locate(names,suffix):
    found=[n for n in names if n.lower().endswith(suffix)]
    if len(found)!=1:raise ValueError(f'Expected one {suffix}, found {len(found)}')
    return found[0]
def assignments(txt):
    lines=[l.split() for l in txt.strip().splitlines() if l.strip()]
    if len(lines[0])==1:
        if int(lines[0][0])!=len(lines)-1:raise ValueError('Inconsistent list size')
        lines=lines[1:]
    if any(len(l)!=2 for l in lines):raise ValueError('Expected image name and class')
    return [(PurePosixPath(name).name,int(label)) for name,label in lines]
def distances(pred_a,pred_b,y,seed=19112026):
    d=(pred_a==y).astype(int)-(pred_b==y)
    v=np.array([d[y==cl].mean()*100 for cl in np.unique(y)])
    rng=np.random.default_rng(seed);sample=v[rng.integers(0,len(v),(100000,len(v)))].mean(1)
    return [float(v.mean()),*np.quantile(sample,[.0125,.9875]).tolist()]
def run(a):
    if a.archive.name!='Outex_TC_00031.tar.gz':raise ValueError('This protocol is locked to the official Outex_TC_00031.tar.gz')
    a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    cache=a.out/'outex.npz'
    h=hashlib.sha256()
    with a.archive.open('rb') as stream:
        for part in iter(lambda:stream.read(1<<20),b''):h.update(part)
    digest=h.hexdigest()
    if not cache.exists():
        with tarfile.open(a.archive,'r:gz') as ar:
            members=ar.getmembers();names=[m.name for m in members if m.isfile()]
            layout=[]
            for split in ('train','test'):
                ent=locate(names,'000/'+split+'.txt');layout.append(assignments(ar.extractfile(ent).read().decode('utf-8-sig')))
            trr,ter=layout;keys=np.array([k for k,c in trr+ter]);y=np.array([c for k,c in trr+ter])
            if len(set(keys))!=2720 or not np.array_equal(np.unique(y),np.arange(68)) or len(trr)!=1360 or len(ter)!=1360:raise ValueError('Official dimensions do not match locked suite')
            tr=np.arange(1360);te=tr+1360;iv=[]
            for c in range(68):
                nums=tr[y[tr]==c]
                if len(nums)!=20 or (y[te]==c).sum()!=20:raise ValueError('Unexpected per-class counts')
                order=sorted(nums,key=lambda i:hashlib.sha256(('HSMT-Outex31-v1|'+str(keys[i])).encode()).digest())
                iv.extend(order[::5])
            iv=np.array(sorted(iv));it=np.setdiff1d(tr,iv);features={k:[] for k in ('F7','F8','rgb')};ex=make_estimator()
            # Read the compressed archive once in member order; then restore official split order.
            bykey={PurePosixPath(n).name:n for n in names if '/images/' in '/'+n.lower() and n.lower().endswith('.bmp')}
            if any(k not in bykey for k in keys):raise ValueError('Official image entries missing')
            official_keys=set(keys);wanted={n:k for k,n in bykey.items() if k in official_keys};payload={}
            for m in members:
                if m.name in wanted:payload[wanted[m.name]]=ar.extractfile(m).read()
            for j in range(0,len(keys),6):
                pics=[]
                for k in keys[j:j+6]:
                    with Image.open(io.BytesIO(payload[k])) as im:
                        if im.size!=(128,128) or im.mode not in ('RGB','RGBA','P'):raise ValueError('Expected 128×128 colour pixels')
                        pics.append(im.convert('RGB'))
                blocks=compact_features(pics,ex)
                for key in ('F7','F8'):features[key].append(blocks[key].astype('f2'))
                features['rgb'].append(np.stack([describe(x)[0] for x in pics]))
            np.savez_compressed(cache,keys=keys,y=y,train=tr,test=te,in_train=it,val=iv,archive_sha256=digest,**{k:np.concatenate(v) for k,v in features.items()})
    with np.load(cache) as z:
        if str(z['archive_sha256'])!=digest:raise ValueError('Archive differs from extraction cache')
        z={k:z[k] for k in z.files}
    run_fold(a.out/'Outex.npz',z,z['y'],z['train'],z['test'],[(z['in_train'],z['val'])])
    with np.load(a.out/'Outex.npz') as f:
        yy=f['labels'];v=f['fixed'][:,0];s=select(f);methods=['L','S','X','Q','RS','RX','RBF']
        report={'official_archive_sha256':digest,'training':1360,'test':1360,'classes':68,'fixed_p0_lambda1':dict(zip(methods,100*(v==yy).mean(1))),
                'add_crosses_simultaneous_interval_pp':distances(v[3],v[1],yy),'add_squares_simultaneous_interval_pp':distances(v[3],v[2],yy),
                'selected_recall_pct':dict(zip(methods,100*(s==yy).mean(1))),'redistributed_squares_minus_crosses_pp':float(100*((s[4]==yy).mean()-(s[5]==yy).mean()))}
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    with threadpool_limits(2):run(a)
