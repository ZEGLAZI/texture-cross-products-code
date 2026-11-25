"""Measure the proposed colour features from original images after fingerprint checks."""
import argparse
import ctypes
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from numpy.ctypeslib import ndpointer
from PIL import Image

def routine(binary):
    lib=ctypes.CDLL(str(binary.resolve()))
    image_type=ndpointer(np.uint8,flags='C_CONTIGUOUS');number_type=ndpointer(np.float64,flags='C_CONTIGUOUS')
    lib.colour_describe.argtypes=[image_type,ctypes.c_int,ctypes.c_int,ctypes.c_uint64,number_type,number_type,number_type]
    lib.colour_describe.restype=None
    lib.from_log_rgb.argtypes=[number_type,ctypes.c_int,ctypes.c_int,ctypes.c_uint64,number_type,number_type]
    lib.from_log_rgb.restype=None
    return lib

def extract(corpus,root,data,binary,workers):
    lib=routine(binary)
    with np.load(data/(corpus+'_inputs.npz')) as z:ids=z['ids'];expected=z['hash224']
    def one(item):
        i,key=item
        with Image.open(root/key) as im:rgb=np.ascontiguousarray(im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC))
        digest=hashlib.sha256(rgb.tobytes()).hexdigest()
        if digest!=expected[i]:raise ValueError('Image fingerprint: '+key)
        seed=int(digest[:16],16)
        lab=np.empty(9);edge=np.empty((2,3,30));diag=np.empty((6,6))
        lib.colour_describe(rgb,224,224,seed,lab,edge,diag)
        return lab,edge,diag
    with ThreadPoolExecutor(workers) as ex: outputs=list(ex.map(one,enumerate(ids)))
    lab,edges,diag=(np.stack(x) for x in zip(*outputs))
    assert np.isfinite(lab).all() and np.isfinite(edges).all()
    np.testing.assert_allclose(diag[:,:,2:].sum(axis=2),diag[:,:,0],atol=0,rtol=0)
    path=data/(corpus+'_new.npz');tmp=path.with_suffix('.partial.npz')
    np.savez_compressed(tmp,ids=ids,lab=lab,edges=edges,diagnostics=diag);tmp.replace(path)
    print(corpus,'images',len(ids),'empty orientation-scale cells',int((diag[:,:,0]==0).sum()),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('corpus',choices=('DTD','KTH','FMD','Lytro'))
    for k in ('images','data','binary'):p.add_argument(k,type=Path)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();extract(a.corpus,a.images,a.data,a.binary,a.workers)
