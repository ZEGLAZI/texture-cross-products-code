"""Rebuild every Gaussian validation grid and refit the selected DTD and clothing models."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"review2/code"))
sys.path.insert(0,str(ROOT/"historical/dtd"))
from assess import stage
from nested import PEN, fit, macro
from predire import load_blocks

GAMMAS=(0.25,1.,4.,16.)


def distance(blocks,train,evaluation):
    grams,norms=stage(blocks,train,evaluation)
    weights=(.25,.25,.5)
    result=np.zeros_like(grams[0])
    for weight,gram,d in zip(weights,grams,norms):
        result+=weight*(d[:,None]+d[:len(train)][None,:]-2*gram)
    return np.maximum(result,0,out=result)


def check(folds,blocks,y):
    records=[]
    for path in folds:
        with np.load(path) as saved:
            it,iv=saved["inner_train_0"],saved["inner_val_0"]
            assert "inner_train_1" not in saved
            train,test=saved["train"],saved["test"]
            nc=len(np.unique(y))
            truth=y[test]
            np.testing.assert_array_equal(truth,saved["labels"])
            distances=distance(blocks,it,iv)
            grid=np.empty((len(GAMMAS),len(PEN)))
            for i,gamma in enumerate(GAMMAS):
                kernel=np.exp(-gamma*distances)
                for j,penalty in enumerate(PEN):
                    decisions=fit(kernel,len(it),y[it],penalty,nc).argmax(1)
                    grid[i,j]=macro(decisions,y[iv])
            pos=saved["modes"].tolist().index("RBF")
            delta=float(np.max(np.abs(saved["grid"][pos]-grid)))
            np.testing.assert_allclose(saved["grid"][pos],grid,rtol=0,atol=1e-10)
            i,j=np.unravel_index(np.argmax(grid),grid.shape)
            del distances,kernel
            kernel=distance(blocks,train,test)
            np.multiply(kernel,-GAMMAS[i],out=kernel)
            np.exp(kernel,out=kernel)
            pred=fit(kernel,len(train),y[train],PEN[j],nc).argmax(1)
            mismatch=int(np.count_nonzero(saved["tuned"][pos,i]!=pred))
            if mismatch:
                raise ValueError(f"{path.stem}: {mismatch} selected predictions differ")
            records.append(dict(fold=path.stem,gamma=GAMMAS[i],ridge=PEN[j],test_appearances=len(test),
                                validation_cells=grid.size,max_validation_difference=delta,prediction_differences=mismatch))
    return records


def run(out):
    folds=ROOT/"review2/results/kernels"
    with np.load(ROOT/"historical/dtd/results/identites.npz") as z:
        y=z["y"]
        blocks=load_blocks(ROOT/"review2/dtd_cache/hsmt",z["ids"],["F7","F8","rgb"])
    dtd=check(sorted(folds.glob("DTD_*.npz")),blocks,y)
    del blocks
    with np.load(ROOT/"clothing/data/features.npz") as z:
        blocks={k:z[k] for k in ("F7","F8","rgb")}
        clothing=check(sorted(folds.glob("Textile_*.npz")),blocks,z["y"])
    assert len(dtd)==10 and len(clothing)==150
    out.mkdir(exist_ok=True,parents=True)
    with (out/"gaussian_refits.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=dtd[0]);w.writeheader();w.writerows(dtd+clothing)
    info={name:{key:sum(r[key] for r in records) for key in ("test_appearances","validation_cells","prediction_differences")}
          for name,records in (("DTD",dtd),("Clothing",clothing))}
    (out/"gaussian_refits.json").write_text(json.dumps(info,indent=2)+"\n")
    print(info)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    with threadpool_limits(3):
        run(parser.parse_args().output)
