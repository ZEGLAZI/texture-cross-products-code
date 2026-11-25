"""Locked follow-up on the original quarter-resolution ALOT acquisitions."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import numpy as np
from threadpoolctl import threadpool_limits
from fit import standard_wavelet,compact_map,polynomial,classify
from stats import write
CONDITIONS=('l8','i','l1','l2','l3','l4','l5')
PATTERN=re.compile(r'^png4/(\d+)/(\d+)_c([1-4])(i|l8|l[1-5])(?:r0)?[.]png$')
CODES=('AW','RGB','Lab6','RGB_J','RGB_J_perm','RGB_J_mean')
def selection():
    return [(i,c,k) for i in range(1,251) for c in range(1,5) for k in (CONDITIONS if c==1 else CONDITIONS[:2])]
def keyname(i,c,k):return f'{i:03}_c{c}{k}.png'
def unpack(archive,images):
    images.mkdir(exist_ok=True,parents=True);seen=set()
    with tarfile.open(archive,'r:') as tar:
        for member in tar:
            m=PATTERN.fullmatch(member.name.removeprefix('./'))
            if not m or not member.isfile():continue
            i,i2,c=map(int,m.group(1,2,3));k=m.group(4)
            if not 1<=i<=250 or (c!=1 and k not in ('l8','i')):continue
            key=(i,c,k)
            if i!=i2 or key in seen:raise ValueError(member.name)
            (images/keyname(*key)).write_bytes(tar.extractfile(member).read());seen.add(key)
    if seen!=set(selection()):raise ValueError('Incomplete ALOT selection: '+str(sorted(set(selection())-seen)[:5]))
    print(len(seen),'ALOT photographs unpacked',flush=True)

def extract(images,binary,output,threads):
    from PIL import Image
    import os
    for k in ('ATEN_CPU_CAPABILITY','ONEDNN_MAX_CPU_ISA','DNNL_MAX_CPU_ISA'):os.environ[k]='avx2' if k.startswith('ATEN') else 'AVX2'
    import torch
    from extract import routine
    from wavelets_original import compact_features,make_estimator
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    wave=make_estimator();lib=routine(binary);allkeys=selection()
    vals={key:[] for key in ('ids','y','camera','light','hash224','file_hash','rgb','lab','edges','F7','F8','sizes')}
    for start in range(0,len(allkeys),8):
        photos=[]
        for idx,cam,light in allkeys[start:start+8]:
            key=keyname(idx,cam,light);path=images/key
            with Image.open(path) as im:
                native=im.convert('RGB');dims=native.size;im=native.resize((224,224),Image.Resampling.BICUBIC)
            p=np.ascontiguousarray(im);digest=hashlib.sha256(p.tobytes()).hexdigest();photos.append(im)
            lab=np.empty(9);edges=np.empty((2,3,30));diag=np.empty((6,6))
            lib.colour_describe(p,224,224,int(digest[:16],16),lab,edges,diag)
            codes=p.astype(np.int32)//64;bins=16*codes[:,:,0]+4*codes[:,:,1]+codes[:,:,2]
            rgb=np.sqrt(np.bincount(bins.ravel(),minlength=64)/bins.size)
            for k,v in dict(ids=key,y=idx-1,camera=cam,light=light,hash224=digest,
                file_hash=hashlib.sha256(path.read_bytes()).hexdigest(),sizes=dims,rgb=rgb,lab=lab,edges=edges).items():
                vals[k].append(v)
        b=compact_features(photos,wave)
        for key in ('F7','F8'):vals[key].extend(b[key])
        if start%400==0:print('ALOT',start+len(photos),'/',len(allkeys),flush=True)
    packed={k:np.array(v) for k,v in vals.items()}
    assert len(set(packed['ids']))==3250
    temp=output.with_suffix('.partial.npz');np.savez_compressed(temp,**packed);temp.replace(output)

def maps(rec,tr):
    e=rec['edges'][:,1]
    ret={'RGB':rec['rgb'],'Lab6':compact_map(rec['lab'][:,3:],tr)}
    for i,k in enumerate(('RGB_J','RGB_J_mean','RGB_J_perm')):
        ret[k]=np.concatenate((rec['rgb'],compact_map(e[:,i],tr)),axis=1)/np.sqrt(2)
    return ret

def run(data,out):
    with np.load(data) as z:rec=dict(z)
    y=rec['y'];cam=rec['camera'];lights=rec['light'];items=[];details=[]
    for i in range(1,5):
        tr=np.flatnonzero((cam==i)&(lights=='l8'))
        te=np.flatnonzero((cam==i)&(lights!='l8'))
        assert len(tr)==250 and len(te)==(1500 if i==1 else 250)
        ix=np.r_[tr,te];nt=len(tr)
        for key in ('hash224','file_hash'):
            assert not np.isin(rec[key][te],rec[key][tr]).any(),key
        np.testing.assert_array_equal(y[tr],np.arange(250))
        hand=[standard_wavelet(rec[k],tr,d) for k,d in (('F7',9),('F8',3))]
        aw=sum(x[ix]@x[tr].T for x in hand)
        m=maps(rec,tr);truth=np.eye(250)[y[tr]];prior=truth.mean(axis=0)
        preds={}
        for key in CODES:
            sums=aw if key=='AW' else aw+m[key][ix]@m[key][tr].T
            K=polynomial(sums);factor=(11/13)/np.diag(K[:nt]).mean()
            preds[key]=classify(K,nt,truth-prior,prior,1./factor)
        for j,idx in enumerate(te):
            row=dict(id=rec['ids'][idx],texture=int(y[idx])+1,camera=i,light=lights[idx],**{key:int(v[j])+1 for key,v in preds.items()})
            items.append(row)
        details.append(dict(camera=i,train_ids=rec['ids'][tr].tolist(),test_ids=rec['ids'][te].tolist()))
    write(out/'ALOT_decisions.csv',items)
    (out/'ALOT_splits.json').write_text(json.dumps(details,indent=1)+'\n')
    scores=[];contrasts=[]
    for name in ('temperature','direction_c1'):
        rows=[r for r in items if (r['light']=='i')==(name=='temperature')]
        count=4 if name=='temperature' else 5
        assert len(rows)==250*count
        group=np.array([r['texture'] for r in rows]);groups=np.unique(group)
        pred={k:np.array([r[k] for r in rows]) for k in CODES}
        hit={k:np.array([(z[group==g]==g).mean() for g in groups]) for k,z in pred.items()}
        for key in CODES:
            conf=np.bincount((group-1)*250+(pred[key]-1),minlength=250**2).reshape(250,250)
            prec=np.divide(np.diag(conf),conf.sum(0),out=np.zeros(250),where=conf.sum(0)>0)
            scores.append(dict(protocol=name,method=key,appearances=len(rows),texture_groups=250,
                               macro_recall_pct=float(100*hit[key].mean()),top1_pct=float(100*(pred[key]==group).mean()),macro_precision_pct=float(100*prec.mean())))
        for comp in ('RGB','RGB_J_perm','RGB_J_mean'):
            diff=hit['RGB_J']-hit[comp];u,counts=np.unique(np.rint(count*diff).astype(int),return_counts=True)
            rng=np.random.default_rng(20261120)
            sims=rng.multinomial(250,counts/250.,100000)@u/(count*250.)
            lo,hi=100*np.quantile(sims,[.025/6,1-.025/6])
            contrasts.append(dict(protocol=name,method='RGB_J',reference=comp,delta_pp=float(diff.mean()*100),
                                  low_pp=float(lo),high_pp=float(hi),bonferroni_family=6))
    write(out/'ALOT_scores.csv',scores);write(out/'ALOT_contrasts.csv',contrasts)
    print('ALOT',len(items),'held-out predictions per model',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);q=p.add_subparsers(dest='cmd',required=True)
    u=q.add_parser('unpack');u.add_argument('archive',type=Path);u.add_argument('images',type=Path)
    e=q.add_parser('extract')
    for k in ('images','binary','output'):e.add_argument(k,type=Path)
    e.add_argument('--threads',type=int,default=3)
    f=q.add_parser('fit');f.add_argument('data',type=Path);f.add_argument('out',type=Path)
    a=vars(p.parse_args());cmd=a.pop('cmd')
    if cmd=='unpack':unpack(**a)
    elif cmd=='extract':extract(**a)
    else:
        with threadpool_limits(2):run(**a)
