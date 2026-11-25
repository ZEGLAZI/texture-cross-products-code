"""Rebuild representation and regime comparisons from paired, full-precision decisions."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from compare import metadata
from regime_change import ROOT, calculate
sys.path.insert(0, str(ROOT / "review2/code"))
from inference import dtd_interval, external


def write_tables(source, output):
    output.mkdir(exist_ok=True, parents=True)
    effects, regime_changes = [], {}
    for corpus in ("DTD","KTH"):
        ids,y,groups,_,names = metadata(corpus,ROOT)
        with np.load(source/(corpus+"_predictions.npz"),allow_pickle=False) as f:
            test,labels=f["test"],f["labels"]
            np.testing.assert_array_equal(labels,y[test])
            np.testing.assert_array_equal(f["ids"],ids[test])
            np.testing.assert_array_equal(f["groups"],groups[test])
            def ci(diff,m=1):
                if corpus=="DTD":
                    return dtd_interval(diff,labels,groups[test],m=m,n=100000,seed=19112026)
                return external(diff.ravel(),labels.ravel(),groups[test].ravel(),m=m,n=100000)
            for regime,infix in (("fixed","_"),("selected","_tuned_")):
                row={"dataset":corpus,"penalty":regime}
                gains=[]
                for fam in ("H","D"):
                    scores=[]
                    for method in ("S","Q"):
                        correct=(f[fam+infix+method]==labels).astype(np.int16)
                        row[fam+"_"+method]=float(correct.mean()*100)
                        scores.append(correct)
                    gain=scores[1]-scores[0]
                    gains.append(gain)
                    row[fam+"_gain"],row[fam+"_lo"],row[fam+"_hi"]=ci(gain)
                row["contrast"],row["contrast_lo"],row["contrast_hi"]=ci(gains[1]-gains[0],2 if regime=="fixed" else 1)
                effects.append(row)
        regime_changes[corpus]=calculate(corpus,source)
    with (output/"representation_comparison.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=effects[0])
        writer.writeheader()
        writer.writerows(effects)
    (output/"regime_change.json").write_text(json.dumps(regime_changes,indent=2)+"\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("predictions",type=Path)
    ap.add_argument("output",type=Path)
    a=ap.parse_args()
    write_tables(a.predictions,a.output)
