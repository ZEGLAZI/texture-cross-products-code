"""Regenerate five manuscript tables from archived fold decisions, with integrity checks."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/"historical/revision/results"
sys.path.insert(0, str(ROOT/"review2/code"))
from inference import dtd_interval

MODELS=("Add","W","S","C","U","Q")


def csv_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def save(path, rows):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def macro(truth, pred):
    return 100*np.mean([(truth[truth==c] == pred[truth==c]).mean() for c in np.unique(truth)])


def selected_scores():
    sel=RESULTS/"selection"
    def read(name):
        with np.load(sel/(name+".npz")) as f:
            return f["test"], {m:f[m] for m in MODELS}
    def row(name,y,models):
        return dict(corpus=name,**{m:macro(y,p) for m,p in models.items()})
    def join(folds):
        return np.concatenate([f[0] for f in folds]),{m:np.concatenate([f[1][m] for f in folds]) for m in MODELS}
    idx,pred=join([read(f"DTD_{i:02}") for i in range(1,11)])
    with np.load(ROOT/"historical/dtd/results/identites.npz") as f:
        rows=[row("DTD",f["y"][idx],pred)]
    idx,pred=read("Flowers")
    pos=0
    for label,data in (("Pl@ntNet","plantnet"),("iNat2026","nouveau_2026")):
        with np.load(ROOT/f"historical/external/data/{data}.npz") as f:
            y=(f["classe"][:,None]==[10,12,9]).argmax(1)
        end=pos+len(y)
        rows.append(row(label,y,{m:p[pos:end] for m,p in pred.items()}))
        pos=end
    assert pos==len(idx)
    with np.load(RESULTS/"cohort/predictions.npz") as f:
        rows.append(row("iNat2015-17",f["y"],{m:f["selected_"+m] for m in MODELS}))
    idx,pred=join([read("KTH_"+s) for s in "abcd"])
    kth=csv_rows(ROOT/"historical/external/data/kth_images.csv")
    y=np.unique([r["materiau"] for r in kth],return_inverse=True)[1]
    assert len(np.unique(idx))==len(kth)==len(idx)
    rows.append(row("KTH",y[idx],pred))
    ref=json.loads((RESULTS/"selected.json").read_text())
    ref["iNat2015-17"]=json.loads((RESULTS/"cohort/scores.json").read_text())["selected"]
    for r in rows:
        np.testing.assert_allclose([r[m] for m in MODELS],[ref[r["corpus"]][m] for m in MODELS],atol=1e-10)
    return rows


def clothing_scores():
    subprocess.run([sys.executable,str(ROOT/"clothing/code/check_results.py")],check=True)
    with np.load(ROOT/"clothing/results/predictions.npz") as f:
        y=f["y"]
        models=f["methods"].tolist()
        recalls=np.array([[macro(y,p) for p in repeat] for repeat in f["pred"]])
    return [dict(model=m,mean=recalls[:,i].mean(),lowest=recalls[:,i].min(),
                 highest=recalls[:,i].max()) for i,m in enumerate(models)]


def ablation_scores():
    with np.load(ROOT/"historical/dtd/results/identites.npz") as f:
        labels,groups=f["y"],f["groupes"]
    old=RESULTS/"ablations"
    acc={}
    for k in range(1,11):
        with np.load(old/f"fold_{k:02}.npz") as a, np.load(ROOT/f"review2/results/radial/Radial_{k:02}.npz") as b:
            np.testing.assert_array_equal(a["test"],b["test"])
            np.testing.assert_array_equal(a["grouped_half"],b["original"])
            np.testing.assert_array_equal(a["diag"],b["shape_diagonal"])
            for m in ("test","grouped_half","no_shape","no_scale","no_radial","diag"):
                acc.setdefault(m,[]).append(a[m])
            for m in ("upstream_diagonal","expanded_grid","log_ratio","boundary_indicator"):
                acc.setdefault(m,[]).append(b[m])
    arrays={m:np.stack(parts) for m,parts in acc.items()}
    test=arrays.pop("test");y=labels[test];reference=arrays["grouped_half"]
    source={r["variant"]:(r["recall"],r["delta"],r["lo"],r["hi"],r["changed_label"])
            for r in csv_rows(old/"ablation.csv")}
    source.update({r["method"]:(r["recall"],r["diff"],r["lo95"],r["hi95"],r["changed"])
                   for r in csv_rows(ROOT/"review2/results/radial/radial_scores.csv") if r["reference"]=="original"})
    rows=[]
    for name,pred in arrays.items():
        delta=(pred==y).astype("int8")-(reference==y)
        effect,lo,hi=dtd_interval(delta,y,groups[test],m=1,n=60000)
        values=(100*(pred==y).mean(),effect,lo,hi,int(np.count_nonzero(reference!=pred)))
        np.testing.assert_allclose(values,np.array(source[name],dtype=float),atol=1e-10)
        rows.append(dict(intervention=name,recall_pct=values[0],difference_pp=effect,
                         pointwise_lo95=lo,pointwise_hi95=hi,changed=values[-1]))
    return rows


def main(out):
    out.mkdir(exist_ok=True,parents=True)
    subprocess.run([sys.executable,str(ROOT/"review2/code/replay.py"),"aggregate","--output",str(out/"aggregate")],check=True)
    for name in ("primary.csv","redistributed_rbf.csv","all_controls.csv","selected_settings.csv","per_repeat.csv","parity.json"):
        expected=(ROOT/"review2/results/aggregate"/name).read_text()
        if (out/"aggregate"/name).read_text()!=expected:
            raise ValueError(f"Replayed output differs from archive: {name}")
    save(out/"clothing_scores.csv",clothing_scores())
    save(out/"selected_restricted_scores.csv",selected_scores())
    save(out/"wavelet_interventions.csv",ablation_scores())
    print("Verified the fixed terms, redistributed models, clothing scores, selected restricted models and wavelet interventions.")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    with threadpool_limits(1):
        main(parser.parse_args().output)
