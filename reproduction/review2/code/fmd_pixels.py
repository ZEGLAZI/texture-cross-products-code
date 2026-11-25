"""Check FMD measurement parity against original material photographs."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import argparse,json,hashlib,csv
import torch
import numpy as np
from PIL import Image
from extracteur import make_estimator,compact_features
from histogramme_oxford import describe

def run(a):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with np.load(a.features) as z: prior={k:z[k] for k in ('keys','F7','F8','rgb')}
    with a.manifest.open() as h: sig={x['id']:x['pixel_hash'] for x in csv.DictReader(h)}
    ex=make_estimator();rows=[];ids=prior['keys']
    with ZipFile(a.archive) as ar:
        for j in range(0,len(ids),6):
            ims=[]
            for name in ids[j:j+6]:
                with Image.open(BytesIO(ar.read(str(name)))) as im:im=im.convert('RGB')
                assert hashlib.sha256(im.resize((224,224),Image.Resampling.BICUBIC).tobytes()).hexdigest()==sig[name]
                ims.append(im)
            v=compact_features(ims,ex);v['rgb']=np.stack([describe(im)[0] for im in ims])
            for k,b in v.items():
                ref=prior[k][j:j+len(ims)];diff=b.astype(float)-ref.astype(float)
                rows.append((k,int(np.count_nonzero(diff)),diff.size,float(abs(diff).max())))
    result={}
    for k in ('F7','F8','rgb'):
        r=[a for a in rows if a[0]==k];result[k]={'different_stored_coordinates':sum(a[1] for a in r),'total_coordinates':sum(a[2] for a in r),'max_absolute_difference':max(a[3] for a in r)}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps({'images':len(ids),'blocks':result},indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('archive','features','manifest','out'):p.add_argument(n,type=Path)
    run(p.parse_args())
