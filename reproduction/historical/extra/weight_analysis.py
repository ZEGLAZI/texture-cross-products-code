"""Paired, exploratory 95% intervals for reversing the two W-centred product weights."""
import argparse
import json
from pathlib import Path
import numpy as np

def main(root,out):
    r=np.load(root/"extra/reversals.npz"); ext=np.load(root/"external/results/external.npz")
    old=np.load(root/"dtd/results/decisions.npz"); ids=np.load(root/"dtd/results/identites.npz")
    result={}
    for name in ("DTD","plantnet","nouveau_2026","KTH"):
        rng=np.random.default_rng(19112026)
        if name=="DTD":
            te=old["tests"]; y=ids["y"][te]; keys,inv=np.unique(ids["groupes"][te],return_inverse=True);inv=inv.reshape(te.shape)
            d=(old["HSMT"]==y).astype(np.int8)-(r["DTD_W_reversed"]==y)
            inf=np.zeros(len(keys))
            for i in range(10):
                means=np.bincount(y[i],weights=d[i],minlength=47)/40
                np.add.at(inf,inv[i],(d[i]-means[y[i]])*100/te.size)
            point=100*d.mean(); draws=np.empty(30000)
            for i in range(0,len(draws),200):
                signs=2*rng.integers(0,2,(200,len(keys)),dtype="int8")-1
                draws[i:i+200]=point+signs.astype(float)@inf
        else:
            y=ext[name+"_y"]; group=ext[name+"_group"]; d=(ext[name+"_HSMT"]==y).astype(np.int8)-(r[name+"_W_reversed"]==y)
            cats=np.unique(y);point=100*np.mean([d[y==c].mean() for c in cats]); draws=np.zeros(30000)
            for c in cats:
                ix=y==c
                if name=="KTH":
                    v=np.array([d[ix][group[ix]==g].mean() for g in np.unique(group[ix])]);n=len(v)
                    draws+=rng.multinomial(n,np.ones(n)/n,len(draws))@v/n
                else:
                    v,numbers=np.unique(d[ix],return_counts=True);n=ix.sum()
                    draws+=rng.multinomial(n,numbers/n,len(draws))@v/n
            draws*=100/len(cats)
        result[name]=dict(delta=float(point),adjusted=np.quantile(draws,[.05/8,1-.05/8]).tolist())
    out.write_text(json.dumps(result,indent=2)+"\n");print(result)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("evidence",type=Path);p.add_argument("output",type=Path);a=p.parse_args();main(a.evidence,a.output)
