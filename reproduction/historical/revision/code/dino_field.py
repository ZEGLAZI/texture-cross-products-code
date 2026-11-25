"""Apply the pinned DINOv2 extractor to exactly the full resized field used by the statistics."""
import argparse,sys,time,resource,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
def main(a):
    sys.path.insert(0,str((a.supp/'dtd').resolve()));from extraire import setup_dino
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);a.output.mkdir(parents=True,exist_ok=True)
    # setup_dino applies a centred 224 square crop; supplying a 224 square makes this the identity.
    encode=setup_dino(a.archives,a.output)
    ids=np.load(a.data/'index.npz')['ids']
    for start in range(0,len(ids),96):
        dest=a.output/f'{start:05d}.npz';keys=ids[start:start+96]
        if dest.exists():continue
        result=[];seconds=[]
        for i in range(0,len(keys),8):
            pix=[]
            for k in keys[i:i+8]:
                with Image.open(a.data/'images'/k) as im:pix.append(im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC))
            t=time.perf_counter();result.append(encode(pix)['dino']);seconds.append(time.perf_counter()-t)
        np.savez(dest,ids=keys,dino=np.concatenate(result),seconds=seconds)
        if start%480==0:print('DINO full',start+len(keys),'rss_KiB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('supp','data','archives','output'):p.add_argument(n,type=Path)
    p.add_argument('--threads',type=int,default=2);main(p.parse_args())
