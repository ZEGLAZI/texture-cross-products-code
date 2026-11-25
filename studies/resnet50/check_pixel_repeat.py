"""Compare all pooled feature coordinates from two complete frozen CNN extractions."""
import argparse
import json
from pathlib import Path
import numpy as np

def compare(reference, new):
    reports={}
    for ds in ("DTD","KTH","FMD","Lytro"):
        with np.load(reference/(ds+"_stages.npz"),allow_pickle=False) as a, np.load(new/(ds+"_stages.npz"),allow_pickle=False) as b:
            if set(a.files)!=set(b.files): raise ValueError("Stage file schema differs for "+ds)
            for key in a.files: np.testing.assert_array_equal(a[key],b[key],err_msg=ds+" "+key)
            reports[ds]={"images":len(a["ids"]),"coordinates":sum(a[k].size for k in ("layer2","layer3","layer4"))}
    return {"datasets":reports,"images":sum(x["images"] for x in reports.values()),
            "coordinates":sum(x["coordinates"] for x in reports.values()),"differing_coordinates":0}
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference",type=Path);p.add_argument("new",type=Path);p.add_argument("--report",type=Path)
    a=p.parse_args()
    text=json.dumps(compare(a.reference,a.new),indent=2)+"\n"
    if a.report:a.report.write_text(text,encoding="utf-8")
    print(text)
