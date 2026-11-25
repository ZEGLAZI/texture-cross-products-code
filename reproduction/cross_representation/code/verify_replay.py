"""Check each test prediction and the numeric summaries produced in a new run."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare_json(left,right,key):
    if isinstance(left,dict):
        assert set(left)==set(right),key
        for k in left:
            compare_json(left[k],right[k],key+"."+k)
    elif isinstance(left,list):
        assert len(left)==len(right),key
        for i,(a,b) in enumerate(zip(left,right,strict=True)):
            compare_json(a,b,f"{key}[{i}]")
    elif isinstance(left,float):
        np.testing.assert_allclose(left,right,rtol=0.,atol=1e-10,err_msg=key)
    else:
        assert left==right,key


def main(ref,run):
    for corpus in ("DTD","KTH"):
        name=corpus+"_predictions.npz"
        with np.load(ref/name) as a,np.load(run/name) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:
                np.testing.assert_array_equal(a[key],b[key],err_msg=corpus+" "+key)
        for suffix in ("summary","settings"):
            name=corpus+"_"+suffix+".json"
            compare_json(json.loads((ref/name).read_text()),json.loads((run/name).read_text()),name)
        print(corpus,"all predictions, settings and summary values match")
    name="HSMT_refits.json"
    compare_json(json.loads((ref/name).read_text()),json.loads((run/name).read_text()),name)
    print("All HSMT fixed-kernel refits match")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference",type=Path)
    ap.add_argument("rerun",type=Path)
    a=ap.parse_args()
    main(a.reference,a.rerun)
