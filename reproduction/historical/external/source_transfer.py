"""Refit the original additive and two-interaction flower models and check every image."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"dtd"))
from predire import compact,ridge
NAMES=np.array(["pâquerette","pissenlit","tournesol"])
CODES=np.array([10,12,9])
def run(folder):
    tf=np.load(folder/"tf.npz");params=np.load(folder/"flower_normalization.npz")
    x,standard=compact(tf,training=np.arange(len(tf["classe"])))
    for k,v in standard.items():np.testing.assert_allclose(v,params[k],atol=3e-10,rtol=1e-7)
    def parts(i):
        v,_=compact(i,params=params)
        return [v[:,:945]*np.sqrt(2),v[:,945:]*np.sqrt(2),i["rgb"]]
    xt=parts(tf); yt=(tf["classe"][:,None]==CODES).argmax(1)
    with (folder/"fleurs.csv").open() as stream: records=list(csv.DictReader(stream))
    for tag,title,key in [("inat_2018_2020","iNaturalist 2018–2020","identite"),("inat_2025","iNaturalist 2025","identite"),("plantnet","Pl@ntNet","occurrence"),("nouveau_2026","iNaturalist 2026","ids")]:
        t=np.load(folder/(tag+".npz")); rows=[v for v in records if v["lot"]==title]; fn=[str(v) for v in t[key]]
        np.testing.assert_array_equal(fn,[v["photo"] for v in rows])
        yl=(t["classe"][:,None]==CODES).argmax(1);np.testing.assert_array_equal(NAMES[yl],[k["vraie_classe"] for k in rows])
        s,w,c=[np.vstack([a,b])@a.T for a,b in zip(xt,parts(t))]
        base=(s+w)/2+c
        for label,gram in (("deux_liens",(base+s*w/4+w*c/2)/3.25),("additif",base/2)):
            guess=ridge(gram,len(yt),yt,3)[0]
            np.testing.assert_array_equal(NAMES[guess],[k[label] for k in rows])
        print(title,len(rows),"source-fixed predictions verified")
if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("data",type=Path)
    with threadpool_limits(3):run(ap.parse_args().data)
