"""Refit native-scale fixed kernels from HSMT descriptors and compare archived decisions."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from compare import METHODS, metadata


def main(data, out, threads):
    sys.path.insert(0,str(data.resolve()/"review2/code"))
    sys.path.insert(1,str(data.resolve()/"historical/dtd"))
    from assess import stage, kernels
    from nested import PEN, fit, macro
    from predire import load_blocks
    from verify_primary_refits import refit_fold

    summary = {}
    with threadpool_limits(threads):
        for corpus in ("DTD","KTH"):
            ids,y,groups,hashes,names = metadata(corpus,data)
            if corpus=="DTD":
                raw = load_blocks(data/"review2/dtd_cache/hsmt",ids,["F7","F8","rgb"])
                n_test, diff = refit_fold(raw,y,data/"review2/results/kernels/DTD_01.npz")
                summary["DTD_split1_original_weights"] = dict(n_test=n_test,different_predictions=diff)
                if any(diff.values()):
                    raise ValueError("Original-profile decisions differ on split 1.")
            else:
                with np.load(data/"historical/external/data/kth.npz") as f:
                    raw = {k:f[k] for k in ("F7","F8","rgb")}
            folds = []
            selections = []
            for path in sorted((data/"review2/results/kernels").glob(corpus+"_*.npz")):
                with np.load(path) as f:
                    tr,te = f["train"],f["test"]
                    np.testing.assert_array_equal(f["labels"],y[te])
                    np.testing.assert_array_equal(f["modes"][:4],METHODS)
                    expected = f["fixed"][:,1]
                    expected_tuned = f["tuned"][[1,3],1]
                    expected_grid = f["grid"][[1,3],1]
                    inner = [(f[f"inner_train_{i}"], f[f"inner_val_{i}"])
                             for i in range(1 if corpus == "DTD" else 3)]
                if np.intersect1d(groups[tr],groups[te]).size or np.intersect1d(hashes[tr],hashes[te]).size:
                    raise ValueError("Shared source across an outer boundary")
                grid = np.zeros((2, len(PEN)))
                for fitting, validation in inner:
                    if not np.isin(np.r_[fitting,validation],tr).all() or np.intersect1d(
                            groups[fitting],groups[validation]).size:
                        raise ValueError(f"Unexpected source in inner split of {path.stem}")
                    inner_grams, inner_diags = stage(raw, fitting, validation)
                    for name, kernel in kernels(inner_grams, inner_diags, 1):
                        if name not in ("S", "Q"):
                            continue
                        j = ("S", "Q").index(name)
                        for k, penalty in enumerate(PEN):
                            labels = fit(kernel, len(fitting), y[fitting], penalty, len(names)).argmax(1)
                            grid[j,k] += macro(labels,y[validation]) / len(inner)
                        if name == "Q":
                            break
                    del inner_grams, inner_diags, kernel
                np.testing.assert_allclose(grid, expected_grid, atol=1e-10, rtol=0)
                best = grid.argmax(axis=1)
                np.testing.assert_array_equal(best, expected_grid.argmax(axis=1))
                selected_diff = {}
                gs,diags = stage(raw,tr,te)
                diff = {}
                for i,(mode,k) in enumerate(kernels(gs,diags,1)):
                    assert mode == METHODS[i]
                    pred = fit(k,len(tr),y[tr],1.,len(names)).argmax(1)
                    diff[mode] = int(np.count_nonzero(pred!=expected[i]))
                    if mode in ("S", "Q"):
                        j = ("S", "Q").index(mode)
                        selected = pred if PEN[best[j]] == 1 else fit(
                            k, len(tr), y[tr], PEN[best[j]], len(names)).argmax(1)
                        selected_diff[mode] = int(np.count_nonzero(selected != expected_tuned[j]))
                    if mode=="Q": break
                folds.append(dict(split=path.stem,n_test=len(te),different_predictions=diff))
                del gs,diags,k
                selections.append(dict(fold=path.stem, grid_cells_verified=int(grid.size),
                                       penalties={name:float(PEN[index]) for name,index in zip(("S","Q"),best)},
                                       test_prediction_mismatches=selected_diff))
                if any(diff.values()) or any(selected_diff.values()):
                    raise ValueError(f"Equal-weight refits differ in {path.stem}: {diff}, {selected_diff}")
            summary[corpus+"_equal_weights"] = folds
            summary[corpus+"_equal_weights_selection"] = selections
            print(corpus,len(folds),"all refits agree",flush=True)
    out.write_text(json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data",type=Path)
    ap.add_argument("out",type=Path)
    ap.add_argument("--threads",type=int,default=2)
    a=ap.parse_args()
    main(a.data,a.out,a.threads)
