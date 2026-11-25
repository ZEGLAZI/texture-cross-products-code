"""Apply the earlier FMD fingerprint screen to source-target and within-target pairs."""
import argparse
import csv
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image,ImageOps
from scipy.fft import dctn

def signature(im):
    gray=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS))
    coeff=dctn(gray,norm='ortho')[:8,:8].ravel()
    ph=int.from_bytes(np.packbits(coeff>np.median(coeff[1:])).tobytes(),'big')
    thumb=np.asarray(im.resize((16,16),Image.Resampling.BILINEAR),dtype=np.int16)
    return ph,thumb

def run(src,target,cache,out):
    data=[]
    for name,root in [('FMD',src),('Lytro',target)]:
        with np.load(cache/(name+'_design.npz')) as z: ids=z['ids']
        def one(key):
            with Image.open(root/key) as im:pix=im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC)
            return signature(pix),signature(ImageOps.mirror(pix))
        with ThreadPoolExecutor(4) as pool:res=list(pool.map(one,ids))
        hashes=np.array([[s[0] for s in row] for row in res],dtype=np.uint64);thumb=np.array([[s[1] for s in row] for row in res])
        data.append((ids,hashes,thumb))
    records=[];counts={}
    lt,ht,tt=data[1]
    for label,(ids,h,v) in zip(('FMD-Lytro','within-Lytro'),data):
        total=0
        for i in range(len(lt)):
            found={}
            for flip in (0,1):
                dist=np.bitwise_count(h[:,0]^ht[i,flip])
                ind=np.flatnonzero(dist<=6)
                if label=='within-Lytro':ind=ind[ind<i]
                for j in ind:
                    mse=float(np.mean((tt[i,flip].astype(float)-v[j,0])**2))
                    if mse<=256:
                        val=(int(dist[j]),mse,flip)
                        if j not in found or val<found[j]:found[j]=val
            for j,(ham,mse,flip) in found.items():
                records.append([label,lt[i],ids[j],ham,mse,flip]);total+=1
        counts[label]=total
    with (out/'overlap_candidates.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['comparison','target','other','hamming','thumbnail_mse','mirrored']);w.writerows(records)
    (out/'overlap_summary.json').write_text(json.dumps(counts,indent=2)+'\n')
    print(counts)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('FMD','Lytro','cache','output'):p.add_argument(k,type=Path)
    a=p.parse_args();run(a.FMD,a.Lytro,a.cache,a.output)
