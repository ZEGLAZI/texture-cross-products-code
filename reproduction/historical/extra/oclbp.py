"""Evaluate opponent-colour LBP on the exact DTD folds with the common ridge reader."""
import argparse
import ctypes
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer
from PIL import Image
from threadpoolctl import threadpool_limits

def main(a):
    sys.path.insert(0,str(a.pipeline.resolve()))
    from predire import base_lecteur
    where = a.output
    where.mkdir(exist_ok=True,parents=True)
    so = where / "operators.so"
    subprocess.run(["c++","-O3","-std=c++17","-shared","-fPIC","-fopenmp","-ffp-contract=off",str(a.operator),"-o",str(so)],check=True)
    dll=ctypes.CDLL(str(so.resolve()))
    fun=dll.oclbp
    fun.argtypes=[ndpointer(np.uint8,flags="C_CONTIGUOUS"),ctypes.c_int,ctypes.c_int,ndpointer(np.int32,flags="C_CONTIGUOUS")]
    fun.restype=None
    index=dict(np.load(a.results/"identites.npz"))
    baseline=dict(np.load(a.results/"decisions.npz"))
    def feature(name):
        with Image.open(a.images/str(name)) as im:
            rgb=np.ascontiguousarray(im.convert("RGB").resize((224,224),Image.Resampling.BICUBIC),dtype=np.uint8)
        counts=np.zeros(432,np.int32)
        fun(rgb,224,224,counts)
        counts=counts.reshape(12,36)
        sizes=counts.sum(1,keepdims=True)
        np.testing.assert_array_equal(sizes[:,0],[(224-2*r)**2 for r in (1,1,1,2,2,2,3,3,3,1,1,1)])
        return np.sqrt(counts/sizes/12).flatten()
    with ThreadPoolExecutor(max_workers=4) as pool:
        feat=np.stack(list(pool.map(feature,index["ids"])))
    base_lecteur("OCLBP",feat,index,index["groupes"],where)
    tests=baseline["tests"]
    pred=np.stack([np.load(where/f"OCLBP_{i}.npz")["pred"] for i in range(1,11)])
    y=index["y"][tests]
    group=index["groupes"][tests]
    _,inv=np.unique(group,return_inverse=True)
    inv=inv.reshape(group.shape)
    delta=(baseline["HSMT"]==y).astype("int8")-(pred==y)
    inf=np.zeros(inv.max()+1)
    for i in range(10):
        mu=np.bincount(y[i],weights=delta[i],minlength=47)/40
        np.add.at(inf,inv[i],(delta[i]-mu[y[i]])*(100/tests.size))
    rng=np.random.default_rng(19112026)
    gains=[]
    for i in range(150):
        signs=2*rng.integers(0,2,size=(200,len(inf)),dtype="int8")-1
        gains.extend(delta.mean()*100+signs.astype(float)@inf)
    summary=dict(score=float((pred==y).mean()*100),delta=float(delta.mean()*100),nominal_ci=np.quantile(gains,[.025,.975]).tolist(),oclbp_folds=((pred==y).mean(1)*100).tolist(),corrected=int((delta==1).sum()),lost=int((delta==-1).sum()))
    np.savez_compressed(where/"oclbp_decisions.npz",tests=tests,OCLBP=pred)
    (where/"oclbp.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(summary,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("images","pipeline","operator","results","output"):
        p.add_argument("--"+name,required=True,type=Path)
    with threadpool_limits(3):
        main(p.parse_args())
