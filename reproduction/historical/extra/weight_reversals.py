"""Complete the six fixed-weight, two-edge assignments without selecting a new model."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

def main(a):
    sys.path.insert(0,str(a.pipeline.resolve()))
    from predire import partitions, load_blocks, compact, ridge
    import csv
    # Coefficients are ordered SW, WC, SC. Each centre has two possible weight orders.
    weights={"W_reversed":(.5,.25,0.),"S_reversed":(.5,0.,.25),"C_reversed":(0.,.25,.5)}
    def evaluate(xt,xv,y,nclasses):
        allx=[np.concatenate([b,v]) for b,v in zip(xt,xv)]
        s,w,c=[u@b.T for u,b in zip(allx,xt)]
        main=(s+w)*.5+c
        sw,wc,sc=s*w,w*c,s*c
        del s,w,c
        return {m:ridge((main+a*sw+b*wc+h*sc)/3.25,len(y),y,nclasses)[0] for m,(a,b,h) in weights.items()}
    identity=dict(np.load(a.results/"identites.npz")); old=np.load(a.results/"decisions.npz")
    f=load_blocks(a.cache/"hsmt",identity["ids"],["F7","F8","rgb"])
    def vectors(r,tr=None,params=None):
        g,_=compact(r,training=tr,params=params)
        return [g[:,:945]*np.sqrt(2),g[:,945:]*np.sqrt(2),r["rgb"]]
    guesses={m:[] for m in weights}
    for _,tr,te in partitions(identity,identity["groupes"]):
        b=vectors(f,tr)
        for name,z in evaluate([u[tr] for u in b],[u[te] for u in b],identity["y"][tr],47).items():
            guesses[name].append(z)
    allsets={"DTD":(identity["y"][old["tests"]],{m:np.stack(z) for m,z in guesses.items()})}
    tf=np.load(a.data/"tf.npz");prm=np.load(a.data/"flower_normalization.npz")
    label=lambda r:(r["classe"][:,None]==np.array([10,12,9])).argmax(1)
    train=vectors(tf,params=prm)
    for tag in ("plantnet","nouveau_2026"):
        r=np.load(a.data/(tag+".npz"))
        allsets[tag]=(label(r),evaluate(train,vectors(r,params=prm),label(tf),3))
    kt=np.load(a.data/"kth.npz")
    with (a.data/"kth_images.csv").open() as stream: rows=list(csv.DictReader(stream))
    np.testing.assert_array_equal([r["chemin"] for r in rows],kt["path"])
    y=np.unique([r["materiau"] for r in rows],return_inverse=True)[1];group=np.array([r["specimen"] for r in rows]);ph=np.array([r["sha_pixels"] for r in rows])
    pred={m:np.full(len(y),-1) for m in weights}
    for spec in "abcd":
        te=np.flatnonzero(group==spec);tr=np.flatnonzero((group!=spec)&~np.isin(ph,ph[te]));b=vectors(kt,tr)
        new=evaluate([u[tr] for u in b],[u[te] for u in b],y[tr],11)
        for m in weights:pred[m][te]=new[m]
    allsets["KTH"]=(y,pred)
    rows={};bundle={}
    for tag,(lab,pred) in allsets.items():
        rows[tag]={}
        for m,z in pred.items():
            assert (z>=0).all()
            rows[tag][m]=float(100*np.mean([(z[lab==c]==c).mean() for c in np.unique(lab)]))
            bundle[tag+"_"+m]=z
    a.output.mkdir(exist_ok=True,parents=True);(a.output/"reversals.json").write_text(json.dumps(rows,indent=2)+"\n")
    np.savez_compressed(a.output/"reversals.npz",**bundle);print(rows)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("pipeline","cache","results","data","output"):p.add_argument("--"+name,required=True,type=Path)
    with threadpool_limits(3):main(p.parse_args())
