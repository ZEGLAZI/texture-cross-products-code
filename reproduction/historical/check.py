"""Verify source identifiers, empirical decisions, operating weights and displayed scores."""
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import gamma
R=Path(__file__).resolve().parent
def j(name):return json.loads((R/name).read_text())
def audit():
    old=np.load(R/"dtd/results/decisions.npz"); ids=np.load(R/"dtd/results/identites.npz")
    te=old["tests"]; y=ids["y"][te]; grouped=ids["groupes"][te]
    assert te.shape==(10,1880) and len(np.unique(te))==5546 and len(np.unique(grouped))==5530
    np.testing.assert_array_equal(np.sum(np.bincount(ids["groupes"])>1),17)
    for i in range(10):
        train=np.r_[ids["train"+str(i+1)],ids["val"+str(i+1)]]
        train=train[~np.isin(ids["groupes"][train],grouped[i])]
        assert len(train)==old["n_train"][i]
        np.testing.assert_array_equal(np.bincount(y[i]),np.full(47,40))
    summary=j("dtd/results/bilan.json")
    for m,val in summary["methodes"].items():np.testing.assert_allclose(100*(old[m]==y).mean(1),val["par_partition"],atol=1e-12)
    for pair,result in summary["contrastes"].items():
        a,b=pair.split(" - "); d=(old[a]==y).astype(int)-(old[b]==y)
        assert (d==1).sum()==result["corrigees"] and (d==-1).sum()==result["degradees"]
        np.testing.assert_allclose(d.mean()*100,result["ecart"])
    for fname,source in (("oclbp","oclbp_decisions.npz"),("dense","dense_decisions.npz")):
        new=np.load(R/"extra"/source);np.testing.assert_array_equal(te,new["tests"])
        ref=j("extra/"+fname+".json");values=ref.get("scores",{"OCLBP":ref.get("score")})
        for m,v in values.items():np.testing.assert_allclose((new[m]==y).mean()*100,v)
    ext=np.load(R/"external/results/external.npz"); rev=np.load(R/"extra/reversals.npz"); rv=j("extra/reversals.json")
    for t in ("DTD","KTH","plantnet","nouveau_2026"):
        yy=y if t=="DTD" else ext[t+"_y"]
        for key,target in rv[t].items():
            pred=rev[t+"_"+key]
            np.testing.assert_allclose(100*np.mean([(pred[yy==v]==v).mean() for v in np.unique(yy)]),target)
        if t!="DTD":
            for key,val in j("external/results/external.json")[t]["scores"].items():
                pred=ext[t+"_"+key]
                np.testing.assert_allclose(100*np.mean([(pred[yy==c]==c).mean() for c in np.unique(yy)]),val)
    d=(old["HSMT"]==y).astype(int)-(old["Additif_fixe"]==y)
    class_d=[d[y==k].mean() for k in range(47)];assert sum(v>0 for v in class_d)==36 and sum(v<0 for v in class_d)==11
    for case in j("extra/cases.json"):
        np.testing.assert_allclose(sum(case["terms"]),case["margin"],atol=1e-12)
    def norm(t):return t/np.linalg.norm(t,axis=1)[:,None]
    rng=np.random.default_rng(9);u,v,c=[norm(rng.standard_normal((24,d))) for d in (5,3,4)];c=np.abs(c)
    s,w,h=[f@f.T for f in (u,v,c)];T=.5*(s+w)+h;d=3.25
    kw=(T+s*w/4+w*h/2)/d;ks=(T+s*w/4+s*h/2)/d;kc=(T+s*h/4+w*h/2)/d
    np.testing.assert_allclose(kw,((1+w/2)*(1+s/2+h)-1)/d,atol=1e-13)
    np.testing.assert_allclose(kw-ks,h*(w-s)/(2*d),atol=1e-13);np.testing.assert_allclose(kw-kc,s*(w-h)/(4*d),atol=1e-13)
    outer=lambda a,b:np.einsum("ij,ik->ijk",a,b).reshape(len(a),-1)
    feat=np.hstack([u/math.sqrt(2),v/math.sqrt(2),c,outer(u,v)/2,outer(v,c)/math.sqrt(2)])/math.sqrt(d)
    np.testing.assert_allclose(kw,feat@feat.T,atol=1e-13)
    for gram in [kw,ks,kc,(T+(s*w+w*h+s*h)/4)/d]:assert np.min(np.linalg.eigvalsh(gram))>=-1e-12
    np.testing.assert_allclose(np.diag(kw),11/13);np.testing.assert_allclose(np.diag(T/d),8/13)
    np.testing.assert_allclose(gamma(7/2)*gamma(3/2)/gamma(5/2)**2,5/3)
    with (R/"external/data/tf_manifest.csv").open() as f:source=list(csv.DictReader(f))
    assert len(source)==2213 and len(set(v["credit_group"] for v in source))==1075
    with (R/"external/data/kth_readouts.csv").open() as f: kr=list(csv.DictReader(f))
    assert len(kr)==4752
    rates=j("external/results/material_reference.json")["regimes"]["trois_vers_un"]["moyennes"]
    for key in ("clbp_ridge","oclbp_ridge","clbp_chi2","oclbp_l1"):
        np.testing.assert_allclose(100*np.mean([r[key]==r["verite"] for r in kr]),rates[key],atol=1e-12)
    print("Verified all official split counts, stored recall values, corrected/lost counts, six weight assignments, case margins and kernel identities.")
if __name__=="__main__":audit()
