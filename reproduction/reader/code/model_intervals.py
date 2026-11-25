"""Pointwise macro-recall intervals and fold-level tuning at the original profile.

For DTD use image-clustered multipliers, retaining repeated test appearances.
For the clothing cohort average each photograph's 30 correctness indicators before
the class-stratified bootstrap. On KTH resample the four specimens within material.
Use the manuscript's per-corpus inference code and its deterministic seed.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"review2/code"))
from assess import COEFF, PEN, select
from inference import dtd_interval, external

REGIMES = ("fixed", "tuned", "unit")
SELECTED = ("RS","RX","Q","RBF")
SPEC = [(r, name) for r in REGIMES for name in COEFF[:4]] + [("selected", name) for name in SELECTED]


def save(path, rows):
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def load(name, penalties):
    with np.load(ROOT/"review2/results/kernels"/(name+".npz")) as f:
        assert tuple(f["modes"]) == COEFF
        chosen = select(f)
        predictions = np.stack([f[regime][COEFF.index(method),0] if regime != "selected"
                                else chosen[COEFF.index(method)] for regime,method in SPEC])
        for i,method in enumerate(COEFF[:4]):
            grid = f["grid"][i,0]
            j = int(np.argmax(grid))
            penalties.append(dict(fold=name, method=method, profile=0,
                                  fixed_ridge=1, unit_ridge=1, tuned_ridge=PEN[j],
                                  inner_recall_pct=grid[j]))
        return f["test"], f["labels"], predictions


def build(out):
    out.mkdir(exist_ok=True, parents=True)
    penalties,results=[],[]
    def record(corpus, labels, pred, style="photos", groups=None):
        for (regime,method),p in zip(SPEC,pred,strict=True):
            correct=(p==labels).astype("int8")
            if style=="DTD":
                score,lo,hi=dtd_interval(correct,labels,groups,m=1)
            elif style=="clothing":
                score,lo,hi=external(correct.mean(0),labels[0],m=1)
            else:
                score,lo,hi=external(correct,labels,groups=groups,m=1)
            results.append(dict(corpus=corpus,regime=regime,method=method,
                                recall_pct=score,pointwise_lo95=lo,pointwise_hi95=hi))

    dtd=[load(f"DTD_{i:02}",penalties) for i in range(1,11)]
    tests=np.stack([r[0] for r in dtd])
    labels=np.stack([r[1] for r in dtd])
    with np.load(ROOT/"historical/dtd/results/identites.npz") as features:
        np.testing.assert_array_equal(features["y"][tests],labels)
        groups=features["groupes"][tests]
    record("DTD",labels,np.stack([r[2] for r in dtd],axis=1),"DTD",groups)
    kth=[load("KTH_"+c,penalties) for c in "abcd"]
    with (ROOT/"historical/external/data/kth_images.csv").open() as f:
        specimens=np.array([r["specimen"] for r in csv.DictReader(f)])
    indices=np.concatenate([r[0] for r in kth])
    record("KTH",np.concatenate([r[1] for r in kth]),np.concatenate([r[2] for r in kth],axis=1),
           groups=specimens[indices])
    _,labels,pred=load("Flower",penalties)
    start=0
    for name,ref in [("Pl@ntNet","plantnet"),("iNat 2026","nouveau_2026"),("iNat 2015-17",None)]:
        if ref:
            with np.load(ROOT/f"historical/external/data/{ref}.npz") as f:
                stop=start+len(f["classe"])
        else:
            stop=len(labels)
        record(name,labels[start:stop],pred[:,start:stop])
        start=stop
    _,labels,pred=load("FMD",penalties)
    record("FMD",labels,pred)
    with np.load(ROOT/"clothing/data/features.npz") as f:
        y=f["y"]
    pred=np.full((len(SPEC),30,len(y)),-1,dtype=np.int16)
    for r in range(30):
        for k in range(5):
            indices,truth,part=load(f"Textile_{r:02}_{k}",penalties)
            assert (pred[:,r,indices] == -1).all()
            np.testing.assert_array_equal(truth,y[indices])
            pred[:,r,indices]=part
    assert np.all(pred>=0)
    record("Clothing",np.tile(y,(30,1)),pred,"clothing")
    save(out/"model_intervals.csv",results)
    save(out/"p0_model_penalties.csv",penalties)
    print(len(results),"marginal intervals;",len(penalties),"fold and model settings")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    with threadpool_limits(1):
        build(parser.parse_args().output)
