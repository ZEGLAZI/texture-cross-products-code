"""Reproduce the two product-deletion experiments and their specified uncertainty units."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
root=Path(__file__).resolve().parent
data=np.load(root/"results/external.npz")
def intervals():
    result={}
    y=data["plantnet_y"]; ok=(data["plantnet_HSMT"]==y).astype(int)
    # Deleting SW leaves WC; deleting WC leaves SW.
    ds=np.stack([ok-(data["plantnet_"+k]==y) for k in ("wc","sw")],axis=1)
    rng=np.random.default_rng(20261119); class_draw=[]
    for cl in range(3):
        r=ds[y==cl];n=len(r); sets=[]
        for _ in range(160):
            weights=rng.multinomial(n,np.ones(n)/n,250)
            sets.append(100*(weights@r)/n)
        class_draw.append(np.vstack(sets))
    boot=np.mean(class_draw,axis=0)
    result["plantnet_adjusted"]=np.quantile(boot,[.0125,.9875],axis=0).T.tolist()
    y=data["KTH_y"]; group=data["KTH_group"];p=data["KTH_HSMT"]==y
    w=np.random.default_rng(20261118).multinomial(4,[.25]*4,size=(50000,11))
    for m,quant in (("wc",[.0125,.9875]),("sw",[.025,.975]),("Add",[.025,.975])):
        v=p.astype(int)-(data["KTH_"+m]==y)
        matrix=np.array([[v[(y==c)&(group==i)].mean() for i in "abcd"] for c in range(11)])
        b=100*np.einsum("rci,ci->r",w,matrix)/44
        result["KTH_minus_"+m]=np.quantile(b,quant).tolist()
    np.testing.assert_array_equal(np.round(result["plantnet_adjusted"],2),[[-.13,2.76],[1.48,5.33]])
    np.testing.assert_array_equal(np.round(result["KTH_minus_wc"],2),[.59,4.55])
    np.testing.assert_array_equal(np.round(result["KTH_minus_sw"],2),[-1.58,1.94])
    np.testing.assert_array_equal(np.round(result["KTH_minus_Add"],2),[-.36,4.73])
    print(json.dumps(result,indent=2))
if __name__=="__main__":
    with threadpool_limits(2):intervals()
