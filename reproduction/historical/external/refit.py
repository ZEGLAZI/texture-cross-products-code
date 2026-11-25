"""Refit fixed kernels on the source flowers and the four held-out material specimens."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
METHODS=("HSMT","Pivot_F7","Pivot_RGB","Add","dense","sw","wc")
def grams_batch(train,test,y,nclass,ridge):
    both=[np.vstack((a,b)) for a,b in zip(train,test)]
    s,w,c=(b@tr.T for b,tr in zip(both,train))
    t=(s+w)/2+c
    pairs=(s*w*.25,w*c*.5,s*c)
    sw,wc,sc=pairs
    terms=(sw+wc,sw+.5*sc,.25*sc+wc,0,sw+.5*wc+.25*sc,sw,wc)
    return {k:ridge((t+v)/3.25,len(y),y,nclass)[0] for k,v in zip(METHODS,terms)}
def run(a):
    sys.path.insert(0,str(a.pipeline.resolve()))
    from predire import compact,ridge
    raw=np.load(a.data/"tf.npz")
    params=np.load(a.data/"flower_normalization.npz")
    def fblocks(r):
        v,_=compact(r,params=params)
        return v[:,:945]*np.sqrt(2),v[:,945:]*np.sqrt(2),r["rgb"]
    x=fblocks(raw)
    def labels(r):return (r["classe"][:,None]==np.array([10,12,9])).argmax(1)
    yt=labels(raw)
    allrows={}
    for tag in ("plantnet","nouveau_2026"):
        t=np.load(a.data/(tag+".npz"))
        y=labels(t)
        allrows[tag]=(y,np.arange(len(y)),grams_batch(x,fblocks(t),yt,3,ridge))
    kt=np.load(a.data/"kth.npz")
    with (a.data/"kth_images.csv").open() as f: rows=list(csv.DictReader(f))
    np.testing.assert_array_equal(kt["path"],[r["chemin"] for r in rows])
    y=np.unique([r["materiau"] for r in rows],return_inverse=True)[1]
    group=np.array([r["specimen"] for r in rows]); h=np.array([r["sha_pixels"] for r in rows])
    res={m:np.zeros(len(y),int) for m in METHODS}
    for g in "abcd":
        te=np.flatnonzero(group==g);tr=np.flatnonzero((group!=g)&~np.isin(h,h[te]))
        z,_=compact(kt,training=tr)
        blocks=(z[:,:945]*np.sqrt(2),z[:,945:]*np.sqrt(2),kt["rgb"])
        ps=grams_batch([b[tr] for b in blocks],[b[te] for b in blocks],y[tr],11,ridge)
        for m in METHODS:res[m][te]=ps[m]
    allrows["KTH"]=(y,group,res)
    reference=np.load(a.reference)
    rng=np.random.default_rng(19112026)
    out={}; bundle={}
    for tag,(y,group,preds) in allrows.items():
        np.testing.assert_array_equal(y,reference[tag+"_vrai"])
        for m in ("HSMT","Pivot_F7","Pivot_RGB"):np.testing.assert_array_equal(preds[m],reference[tag+"_"+m])
        classes=np.unique(y); ok=np.stack([preds[m]==y for m in METHODS],axis=-1)
        diff=ok[:,0].astype(float)-ok[:,METHODS.index("dense")]
        dist=np.zeros(30000);scores=[]
        for c in classes:
            ix=y==c;scores.append(ok[ix].mean(0))
            if tag=="KTH":
                v=np.array([diff[ix][group[ix]==j].mean() for j in np.unique(group[ix])]); n=len(v)
                dist+=rng.multinomial(n,np.ones(n)/n,size=len(dist))@v/n
            else:
                v,counts=np.unique(diff[ix],return_counts=True); n=ix.sum()
                dist+=rng.multinomial(n,counts/n,size=len(dist))@v/n
        vals=dict(zip(METHODS,(100*np.mean(scores,axis=0)).tolist()))
        out[tag]=dict(scores=vals,delta=vals["HSMT"]-vals["dense"],ci=np.quantile(dist*(100/len(classes)),[.025,.975]).tolist())
        bundle.update({tag+"_"+m:v for m,v in dict(y=y,group=group,**preds).items()})
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"external.json").write_text(json.dumps(out,indent=2)+"\n")
    np.savez_compressed(a.output/"external.npz",**bundle)
    print(out)
if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for field in ("pipeline","data","reference","output"):ap.add_argument("--"+field,type=Path,required=True)
    with threadpool_limits(3):run(ap.parse_args())
