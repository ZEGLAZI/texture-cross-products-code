"""Lock the documented FMD split after visual overlap screening of the original images."""
import argparse,csv,io,zipfile,hashlib,json
from pathlib import Path
from PIL import Image,ImageOps
import numpy as np
from fingerprints import sig
from histogramme_oxford import describe
def collect(a):
    d=np.load(a.dtd);ph=d['h'].tolist();thu=list(d['v']);ids=list(d['ids'])
    def append(image,k):
        for pic in (image,ImageOps.mirror(image)):
            h,v=sig(pic);ph.append(h);thu.append(v);ids.append(k)
    for r in csv.DictReader(a.retained.open()):
        path=a.textile/(r['key']+'.png')
        with Image.open(path) as im:append(im,'cloth/'+r['key'])
    archive=zipfile.ZipFile(a.zip);z=np.load(a.native);keys=np.array(z['ids'] if 'ids' in z else z['keys']);lookup={k:i for i,k in enumerate(keys)}
    manifest=[];zrgb=z['rgb']
    for k in sorted(keys,key=lambda n:hashlib.sha256(('HSMT-fixed-FMD-v1|'+n).encode()).hexdigest()):
        with Image.open(io.BytesIO(archive.read(k))) as im:pix=im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC)
        np.testing.assert_allclose(describe(pix)[0],zrgb[lookup[k]],rtol=1e-7,atol=1e-8)
        match=''
        for scan in (pix,ImageOps.mirror(pix)):
            h,v=sig(scan);near=np.flatnonzero(np.bitwise_count(np.array(ph,dtype=np.uint64)^np.uint64(h))<=6)
            for j in near:
                if np.mean((v.astype(float)-thu[j])**2)<=256:match=ids[j];break
            if match:break
        manifest.append((k,k.split('/')[1],hashlib.sha256(pix.tobytes()).hexdigest(),match))
        if not match:append(pix,k)
    classes=sorted({r[1] for r in manifest});assert len(classes)==10
    keep=[r for r in manifest if not r[3]];kidx=np.array([lookup[r[0]] for r in keep]);y=np.array([classes.index(r[1]) for r in keep])
    tr=[];te=[];iv=[]
    for c in range(10):
        idx=np.flatnonzero(y==c);tr.extend(idx[::2]);te.extend(idx[1::2]);iv.extend(idx[::2][::5])
    tr,te,iv=map(lambda x:np.array(sorted(x)),(tr,te,iv))
    with a.output.with_suffix('.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','class','pixel_hash','excluded_match']);w.writerows(manifest)
    np.savez_compressed(a.output,keys=keys[kidx],classes=classes,y=y,train=tr,test=te,in_train=np.setdiff1d(tr,iv),val=iv,**{k:z[k][kidx] for k in ('F7','F8','rgb')})
    print(len(keep),len(tr),len(te),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('zip','native','dtd','textile','retained','output'):p.add_argument(n,type=Path)
    collect(p.parse_args())
