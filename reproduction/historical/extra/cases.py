"""Reconstruct two signed DTD class margins from the five kernel terms on split 1."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from threadpoolctl import threadpool_limits
def main(a):
    sys.path.insert(0,str(a.pipeline.resolve()))
    from predire import partitions,load_blocks,compact
    ids=dict(np.load(a.results/"identites.npz"))
    d=dict(np.load(a.results/"decisions.npz"))
    raw=load_blocks(a.cache/"hsmt",ids["ids"],["F7","F8","rgb"])
    _,tr,te=next(partitions(ids,ids["groupes"]))
    f,_=compact(raw,training=tr); f*=np.sqrt(2)
    x=[f[:,:945],f[:,945:],raw["rgb"]]
    def components(rows):
        s,w,c=(v[rows]@v[tr].T for v in x)
        return [s/6.5,w/6.5,c/3.25,s*w/13,w*c/6.5]
    rc=components(tr)
    gram=sum(rc); del rc
    y=np.eye(47)[ids["y"][tr]]; prior=y.mean(0)
    gram.flat[::len(tr)+1]+=1
    alpha=cho_solve(cho_factor(gram,lower=True,overwrite_a=True,check_finite=False),y-prior,check_finite=False)
    del gram
    rc=components(te)
    scores=sum(rc)@alpha+prior
    truth=ids["y"][te]; nw=d["HSMT"][0]; old=d["Additif_fixe"][0]; names=ids["traits"]
    np.testing.assert_array_equal(scores.argmax(1),nw)
    cases=[]
    for cls,gained in (("fibrous",True),("zigzagged",False)):
        idx=np.flatnonzero((names[truth]==cls)&((nw==truth) if gained else (old==truth))&((old!=truth) if gained else (nw!=truth)))
        if not len(idx):raise ValueError(cls)
        other=old if gained else nw
        m=scores[idx,truth[idx]]-scores[idx,other[idx]]
        order=sorted(range(len(idx)), key=lambda j:(m[j],str(ids["ids"][te[idx[j]]])))
        z=idx[order[(len(idx)-1)//2]]
        a1=int(truth[z]);a2=int(other[z])
        vals=[float(t[z]@(alpha[:,a1]-alpha[:,a2])) for t in rc]+[float(prior[a1]-prior[a2])]
        margin=scores[z,a1]-scores[z,a2]
        np.testing.assert_allclose(sum(vals),margin,atol=1e-12)
        cases.append(dict(image=str(ids["ids"][te[z]]),label=str(names[a1]),other=str(names[a2]),n_candidates=len(idx),terms=vals,margin=float(margin),direction="recovered" if gained else "lost"))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"cases.json").write_text(json.dumps(cases,indent=2)+"\n")
    print(cases)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("pipeline","results","cache","output"):p.add_argument("--"+k,required=True,type=Path)
    with threadpool_limits(3):main(p.parse_args())
