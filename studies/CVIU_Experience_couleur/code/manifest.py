"""Prepare row order, pixel references and splits from the v5 journal reviewer bundle."""
import argparse
import csv
from pathlib import Path
import shutil
import numpy as np
from fetch_target import CLASSES

def rows(p):
    with p.open(encoding="utf8",newline="") as f: return list(csv.DictReader(f))

def prepare(root,pixel,cache):
    cache.mkdir(parents=True,exist_ok=True)
    for name in ("DTD","KTH"):
        with np.load(pixel / (name+"_colour.npz")) as z:
            # Pinned full wavelet coordinates and reference histogram, regenerated for v5.
            np.savez_compressed(cache/(name+"_original.npz"),**dict(z))
        with np.load(pixel/(name+"_predictions.npz")) as z:
            reference={k:z[k] for k in ("test","labels","ids","groups","H_Q")}
        splits={}
        if name=="DTD":
            with np.load(root/"historical/dtd/results/identites.npz") as z: m=dict(z)
            ids,y=m["ids"],m["y"]
            classes=np.array(sorted(set(k.split("/")[0] for k in ids)))
            for i in range(1,11):
                pool=np.concatenate((m["train"+str(i)],m["val"+str(i)]))
                te=m["test"+str(i)]
                keep=~(np.isin(m["source_sha256"][pool],m["source_sha256"][te])|np.isin(m["pixels224"][pool],m["pixels224"][te]))
                splits["train"+str(i)]=pool[keep]; splits["test"+str(i)]=te
            extras=dict(expected_pixels=m["pixels224"],source_hash=m["source_sha256"])
        else:
            m=rows(root/"historical/external/data/kth_images.csv")
            ids=np.array([r["chemin"] for r in m]); classes,y=np.unique([r["materiau"] for r in m],return_inverse=True)
            specimens=np.array([r["specimen"] for r in m]); hashes=np.array([r["sha_pixels"] for r in m])
            for i,sp in enumerate(sorted(set(specimens)),1):
                te=np.flatnonzero(specimens==sp)
                splits["train"+str(i)]=np.flatnonzero((specimens!=sp)&~np.isin(hashes,hashes[te]))
                splits["test"+str(i)]=te
            extras=dict(expected_pixels=hashes,specimens=specimens)
        for i,v in enumerate(reference["test"],1):
            np.testing.assert_array_equal(splits["test"+str(i)],v)
            np.testing.assert_array_equal(reference["labels"][i-1],y[v])
            np.testing.assert_array_equal(reference["ids"][i-1],ids[v])
        np.savez_compressed(cache/(name+"_design.npz"),ids=ids,y=y,classes=classes,baseline=reference["H_Q"],
                            occurrence_group=reference["groups"],**extras,**splits)
    m=rows(root/"review2/extra/fmd.csv")
    with np.load(root/"review2/extra/fmd.npz") as z:
        np.testing.assert_array_equal(z["keys"],[r["id"] for r in m])
        np.savez_compressed(cache/"FMD_design.npz",ids=z["keys"],y=z["y"],classes=z["classes"],fit=z["train"],
            val=z["test"],expected_pixels=np.array([r["pixel_hash"] for r in m]),rgb=z["rgb"])
        classes=z["classes"]
    assert [k.lower() for k in CLASSES]==list(classes)
    ids=np.array([f"{c}/{n:03}.jpg" for c in CLASSES for n in range(1,101)])
    np.savez_compressed(cache/"Lytro_design.npz",ids=ids,classes=classes,y=np.repeat(np.arange(10),100))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("data","pixel_check","output"):p.add_argument(arg,type=Path)
    a=p.parse_args();prepare(a.data,a.pixel_check,a.output)
