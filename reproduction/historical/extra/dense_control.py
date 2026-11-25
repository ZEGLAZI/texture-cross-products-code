"""Evaluate a uniformly weighted three-edge kernel with matched interaction budget."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

def main(a):
    sys.path.insert(0,str(a.pipeline.resolve()))
    from predire import load_blocks,compact,partitions,ridge
    ids=dict(np.load(a.results/"identites.npz"))
    old=dict(np.load(a.results/"decisions.npz"))
    raw=load_blocks(a.cache/"hsmt",ids["ids"],["F7","F8","rgb"])
    methods=("structure","wavelet","colour","all_pairs")
    guesses={name:[] for name in methods}
    for fold,tr,te in partitions(ids,ids["groupes"]):
        order=np.r_[tr,te]
        f,_=compact(raw,training=tr)
        f*=np.sqrt(2)
        s=f[order,:945]@f[tr,:945].T
        w=f[order,945:]@f[tr,945:].T
        c=raw["rgb"][order]@raw["rgb"][tr].T
        joins=(.5*(s+w)+c+.25*(s*w+s*c+w*c))/3.25
        for name,gram in zip(methods,(s,w,c,joins)):
            guesses[name].append(ridge(gram,len(tr),ids["y"][tr])[0])
        del s,w,c,joins,gram
    tests=old["tests"]
    truth=ids["y"][tests]
    group, inv=np.unique(ids["groupes"][tests],return_inverse=True)
    inv=inv.reshape(tests.shape)
    predictions={name:np.stack(rows) for name,rows in guesses.items()}
    counts={name:float((g==truth).mean()*100) for name,g in predictions.items()}
    delta=(old["HSMT"]==truth).astype(np.int8)-(predictions["all_pairs"]==truth)
    inf=np.zeros(len(group))
    for i in range(10):
        mu=np.bincount(truth[i],weights=delta[i],minlength=47)/40
        np.add.at(inf,inv[i],(delta[i]-mu[truth[i]])*100/tests.size)
    rng=np.random.default_rng(19112026)
    vals=[]
    for j in range(150):
        signs=2*rng.integers(0,2,(200,len(group)),dtype="int8")-1
        vals.extend(100*delta.mean()+signs.astype(float)@inf)
    out=dict(scores=counts,delta=float(delta.mean()*100),ci=np.quantile(vals,[.025,.975]).tolist(),wins=int((delta.mean(1)>0).sum()),agreement=int((old["HSMT"]==predictions["all_pairs"]).sum()))
    a.output.mkdir(exist_ok=True,parents=True)
    (a.output/"dense.json").write_text(json.dumps(out,indent=2)+"\n")
    np.savez_compressed(a.output/"dense_decisions.npz",tests=tests,**predictions)
    print(out,flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for field in ("pipeline","results","cache","output"): ap.add_argument("--"+field,type=Path,required=True)
    with threadpool_limits(3):
        main(ap.parse_args())
