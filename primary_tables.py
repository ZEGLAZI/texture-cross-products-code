"""Recompute the equal-weight comparison from the reference pixel extraction."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from threadpoolctl import threadpool_limits
import numpy as np


def prepare(data, pixels, out):
    code = data / "cross_representation/code"
    fits = out / "primary_fits"
    fits.mkdir(parents=True, exist_ok=True)
    for corpus in ("DTD", "KTH"):
        for suffix in ("predictions", "matched_predictions"):
            shutil.copyfile(pixels / f"{corpus}_predictions.npz", fits / f"{corpus}_{suffix}.npz")
    subprocess.run([sys.executable, str(code / "table_rows.py"), str(fits), str(out / "cross_tables")], check=True)
    sys.path.insert(0, str(code.resolve()))
    from diagonal_tables import interval, macro, save_csv
    summaries, penalties = [], []
    with threadpool_limits(limits=1):
        for corpus in ("DTD", "KTH"):
            with np.load(fits / f"{corpus}_predictions.npz") as f:
                y, groups = f["labels"], f["groups"]
                row = {"dataset":corpus}
                gains, native = {}, {}
                for fam in ("H", "D"):
                    correct = {}
                    for key in ("S", "Q", "matched_Q"):
                        field = fam + "_" + key
                        correct[key] = (f[field] == y).astype(int)
                        row[field + "_recall_pct"] = np.mean([macro(a,b) for a,b in zip(correct[key],y)])
                    gains[fam] = correct["matched_Q"] - correct["S"]
                    native[fam] = correct["Q"] - correct["S"]
                    for suffix, value in zip(("gain_pp","pointwise_low","pointwise_high"),interval(corpus,gains[fam],y,groups,1)):
                        row[fam+"_"+suffix] = value
                for key,value in zip(("matched_D_pp","matched_D_simultaneous_low","matched_D_simultaneous_high"),
                                     interval(corpus,gains["D"]-gains["H"],y,groups,2)):
                    row[key] = value
                summaries.append(row)
            splits = {}
            for fit in json.loads((pixels/f"{corpus}_refit.json").read_text())["splits"]:
                split = splits.setdefault(fit["split"],dict(dataset=corpus,fold=fit["split"]))
                for kernel, penalty in zip(("S","Q"), fit["selections"]):
                    split[fit["representation"]+"_"+kernel] = penalty
            penalties.extend(splits.values())
    scale = out / "scale_tables"
    scale.mkdir(parents=True,exist_ok=True)
    save_csv(scale/"scale_comparison.csv",summaries)
    save_csv(scale/"selected_penalties.csv",penalties)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("data","pixels","out"):
        p.add_argument(key,type=Path)
    a = p.parse_args()
    prepare(a.data,a.pixels,a.out)
