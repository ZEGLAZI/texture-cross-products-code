"""Re-extract the four Oxford views on the published sample using the reference CPU configuration."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile

for name in ('ATEN_CPU_CAPABILITY','ONEDNN_MAX_CPU_ISA','DNNL_MAX_CPU_ISA'):
    os.environ[name] = 'avx2' if name=='ATEN_CPU_CAPABILITY' else 'AVX2'
import numpy as np
from PIL import Image
import torch
from threadpoolctl import threadpool_limits

VIEWS=('whole','foreground','background','outline')
def archive_hash(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def extract(data,originals,out):
    historic=data.resolve()/'historical'
    ref=historic/'revision/results/masks'
    expected=json.loads((ref/'scores.json').read_text())['archives']
    if any(archive_hash(originals/name)!=digest for name,digest in expected.items()):
        raise ValueError('Oxford provider archive checksum differs')
    sys.path.insert(0,str(historic/'dtd/reference/extracteur'))
    sys.path.insert(0,str(historic/'revision/code'))
    from extracteur import make_estimator,compact_features
    from histogramme_oxford import describe
    from cohort.reference.signatures import signatures
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with (ref/'verified.csv').open() as f:rows=[r for r in csv.DictReader(f) if r['excluded']=='0']
    out.mkdir(parents=True,exist_ok=True)
    with tarfile.open(originals/'17flowers.tgz') as photos, tarfile.open(originals/'trimaps.tgz') as masks:
        trimaps={Path(m.name).stem:m for m in masks if m.isfile() and m.name.endswith('.png')}
        ims={v:[] for v in VIEWS}
        for row in rows:
            blob=photos.extractfile(row['image']).read()
            raw=masks.extractfile(trimaps[Path(row['image']).stem]).read()
            if hashlib.sha256(raw).hexdigest()!=row['mask_sha256'] or signatures(blob)['sha256_jpeg']!=row['sha256_jpeg']:
                raise ValueError(row['image'])
            rgb=np.array(Image.open(io.BytesIO(blob)).convert('RGB'));mask=np.array(Image.open(io.BytesIO(raw)))
            if rgb.shape[:2]!=mask.shape or set(np.unique(mask))!={0,1,3}:raise ValueError(row['image'])
            images=[Image.fromarray(rgb),Image.fromarray(np.where((mask==1)[:,:,None],rgb,np.uint8(128))),
                    Image.fromarray(np.where((mask==3)[:,:,None],rgb,np.uint8(128))),
                    Image.fromarray(np.where(mask==1,192,128).astype('uint8')).convert('RGB')]
            for v,img in zip(VIEWS,images,strict=True):ims[v].append(img.resize((224,224),Image.Resampling.BICUBIC))
    estimator=make_estimator()
    y=(np.array([int(r['class_code']) for r in rows])[:,None]==[10,12,9]).argmax(1)
    for v in VIEWS:
        chunks={}
        for i in range(0,len(rows),8):
            batch=ims[v][i:i+8];raw=compact_features(batch,estimator);raw['rgb']=np.stack([describe(p)[0] for p in batch])
            for k,x in raw.items():chunks.setdefault(k,[]).append(x)
        np.savez_compressed(out/(v+'.npz'),image=[r['image'] for r in rows],y=y,**{k:np.concatenate(x) for k,x in chunks.items()})
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('data','archives','output'):p.add_argument(n,type=Path)
    a=p.parse_args()
    with threadpool_limits(1):extract(a.data,a.archives,a.output)
