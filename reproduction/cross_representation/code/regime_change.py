"""Measure a post-hoc change of two paired representation contrasts by data set."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "review2/code"))
from inference import dtd_interval, external


def calculate(corpus, source):
    with np.load(source / (corpus + "_predictions.npz"), allow_pickle=False) as saved:
        y, groups = saved["labels"], saved["groups"]
        expected_shape = (10, 1880) if corpus == "DTD" else (4, 1188)
        if y.shape != expected_shape or groups.shape != y.shape:
            raise ValueError("Unexpected test set or grouping")
        if corpus == "DTD":
            with np.load(ROOT / "historical/dtd/results/identites.npz") as identities:
                for key, original in [("ids","ids"),("labels","y"),("groups","groupes")]:
                    np.testing.assert_array_equal(saved[key], identities[original][saved["test"]])
        else:
            with (ROOT / "historical/external/data/kth_images.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            names, ys = np.unique([r["materiau"] for r in rows], return_inverse=True)
            ix = saved["test"]
            np.testing.assert_array_equal(ys[ix], y)
            np.testing.assert_array_equal(np.array([r["chemin"] for r in rows])[ix], saved["ids"])
            np.testing.assert_array_equal(np.array([r["materiau"]+"/"+r["specimen"] for r in rows])[ix], groups)
            assert len(names)==11 and len(np.unique(groups))==44
        def change(family,regime):
            prefix = family + ("_tuned_" if regime == "selected" else "_")
            before = saved[prefix+"S"]
            after = saved[prefix+"Q"]
            assert before.shape == after.shape == y.shape
            return (after==y).astype(np.int16) - (before==y).astype(np.int16)
        fixed = change("D","fixed") - change("H","fixed")
        selected = change("D","selected") - change("H","selected")
    difference = selected - fixed
    intervals = {}
    for family_size, label in [(1,"pointwise_95_pp"),(2,"bonferroni_two_datasets_95_pp")]:
        if corpus=="DTD":
            estimate, lo, hi = dtd_interval(difference,y,groups,m=family_size,n=100000,seed=19112026)
        else:
            estimate, lo, hi = external(difference.ravel(),y.ravel(),groups.ravel(),m=family_size,n=100000)
        intervals[label] = [float(lo),float(hi)]
    measured = 100*sum(difference[i][y[i]==c].mean() for i in range(len(y)) for c in np.unique(y[i]))/(len(y)*len(np.unique(y)))
    np.testing.assert_allclose(measured,estimate,atol=1e-12)
    return dict(test_appearances=y.size, groups=int(np.unique(groups).size),
                D_fixed_pp=float(fixed.mean()*100),D_selected_pp=float(selected.mean()*100),
                T_pp=float(estimate),**intervals)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("predictions",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    results = {name:calculate(name,args.predictions) for name in ("DTD","KTH")}
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
