"""Re-extract both representations from the same FMD and Lytro full-field images."""
import argparse
import hashlib
import os
from pathlib import Path
import time
from types import SimpleNamespace
for k,v in dict(ATEN_CPU_CAPABILITY="avx2",ONEDNN_MAX_CPU_ISA="AVX2",DNNL_MAX_CPU_ISA="AVX2").items():
    os.environ[k]=v
import numpy as np
from PIL import Image
import torch
from dino_original import load_model,LAYERS
from wavelets_original import compact_features,make_estimator

def run(a):
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(19112026)
    a.temp.mkdir(exist_ok=True,parents=True)
    e=make_estimator()
    model=load_model(SimpleNamespace(source=a.source,weights=a.weights,out=a.temp,device="cpu"),torch)
    mean=torch.tensor([.485,.456,.406])[None,:,None,None]
    sd=torch.tensor([.229,.224,.225])[None,:,None,None]
    with np.load(a.cache/(a.corpus+'_design.npz')) as z: ids=z['ids']
    parts=a.temp/('parts_'+a.corpus)
    parts.mkdir(exist_ok=True)
    filenames=[]
    with torch.inference_mode():
        for start in range(0,len(ids),40):
            subset=ids[start:start+40];file=parts/f'{start:05}.npz';filenames.append(file)
            if file.exists():
                with np.load(file) as old:np.testing.assert_array_equal(old['ids'],subset)
                continue
            values={k:[] for k in ('F7','F8','patch3','patch6','patch12')}
            hashes=[]
            for off in range(0,len(subset),8):
                ims=[]
                for key in subset[off:off+8]:
                    p=a.images/key
                    for retry in range(180 if a.wait else 1):
                        if p.exists():break
                        time.sleep(2)
                    with Image.open(p) as im:ims.append(im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC))
                hashes.extend(hashlib.sha256(im.tobytes()).hexdigest() for im in ims)
                wave=compact_features(ims,e)
                for k,x in wave.items():values[k].append(x)
                pixels=np.stack([np.asarray(im) for im in ims])
                inp=torch.from_numpy(pixels).permute(0,3,1,2).contiguous().float()/255.
                inp=(inp-mean)/sd
                layers=model.get_intermediate_layers(inp,n=LAYERS,norm=True,return_class_token=True)
                if start==0 and off==0:torch.testing.assert_close(layers[-1][1],model(inp),rtol=1e-5,atol=1e-5)
                for k,(patch,cls) in zip(('patch3','patch6','patch12'),layers,strict=True):
                    assert patch.shape[1:]==(256,384)
                    values[k].append(patch.mean(dim=1).numpy())
            vals={k:np.concatenate(v) for k,v in values.items()}
            if any(not np.isfinite(x).all() for x in vals.values()):raise ValueError('Nonfinite features')
            tmp=file.with_suffix('.partial.npz');np.savez(tmp,ids=subset,hash224=hashes,**vals);tmp.replace(file)
            print(a.corpus,start+len(subset),flush=True)
    batches=[]
    for p in filenames:
        with np.load(p) as z:batches.append(dict(z))
    joined={k:np.concatenate([z[k] for z in batches]) for k in batches[0]}
    np.testing.assert_array_equal(ids,joined['ids'])
    if a.corpus=='FMD':
        with np.load(a.cache/'FMD_design.npz') as z: np.testing.assert_array_equal(joined['hash224'],z['expected_pixels'])
    np.savez_compressed(a.cache/(a.corpus+'_features.npz'),**joined)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('corpus',choices=['Lytro','FMD'])
    for v in ('images','cache','source','weights','temp'):p.add_argument(v,type=Path)
    p.add_argument('--threads',type=int,default=3);p.add_argument('--wait',action='store_true')
    run(p.parse_args())
