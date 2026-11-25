"""Read original RGB images, verify available manuscript fingerprints, compute global palettes."""
import argparse
import hashlib
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from palettes import load

def sample(root,name,corpus,expected=None,source=None):
    blob=(root/name).read_bytes()
    source_hash=hashlib.sha256(blob).hexdigest()
    if source is not None and source_hash != source: raise ValueError("File hash: "+name)
    with Image.open(io.BytesIO(blob)) as im:
        rgb=im.convert("RGB")
        size=rgb.size
        arr=np.asarray(rgb.resize((224,224),Image.Resampling.BICUBIC))
        small_hash=hashlib.sha256(arr.tobytes()).hexdigest()
        if expected is not None:
            check=hashlib.sha256(np.asarray(size,dtype="<u4").tobytes()+rgb.tobytes()).hexdigest() if corpus=="KTH" else small_hash
            if check!=expected: raise ValueError("Pixel hash: "+name)
    return arr,small_hash,source_hash,size

def extract(name,root,cache,binary,workers):
    with np.load(cache/(name+"_design.npz")) as z:
        ids=z['ids']; expected=z.get('expected_pixels');source=z.get('source_hash')
    transform=load(binary)
    def one(i):
        arr,h,s,size=sample(root,ids[i],name,None if expected is None else expected[i],None if source is None else source[i])
        return transform(arr),h,s,size
    with ThreadPoolExecutor(workers) as pool: results=list(pool.map(one,range(len(ids))))
    vals,hashes,sources,sizes=zip(*results); vals=np.stack(vals)
    if name in ("DTD","KTH","FMD"):
        ref=cache/(name+("_design.npz" if name=="FMD" else "_original.npz"))
        with np.load(ref) as z: np.testing.assert_array_equal(vals[:,0],z["rgb"])
    np.savez_compressed(cache/(name+"_palettes.npz"),ids=ids,values=vals,hash224=np.array(hashes),file_hash=np.array(sources),sizes=np.array(sizes))
    print(name,len(ids),"checked; native size range",np.min(sizes,0),np.max(sizes,0),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('name',choices=['FMD','Lytro','DTD','KTH'])
    for arg in ('images','cache','binary'):p.add_argument(arg,type=Path)
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();extract(a.name,a.images,a.cache,a.binary,a.workers)
