"""Check the inputs to the numerical figures using test decisions and their pairing."""
from collections import defaultdict
from pathlib import Path
from numerical_fmd import verify_fmd_decisions
import numpy as np
from numerical_checks import csv_rows


def check_figure_data(data, out, mode):
    native_dir = out/"primary_fits"
    matched_dir = out/"primary_fits"
    native_table = csv_rows(out/"cross_tables/representation_comparison.csv")
    matched_table = csv_rows(out/"scale_tables/scale_comparison.csv")
    gains = []
    shifts = {}
    for ds in ("DTD", "KTH"):
        with (np.load(native_dir/f"{ds}_predictions.npz", allow_pickle=False) as base,
              np.load(matched_dir/f"{ds}_matched_predictions.npz", allow_pickle=False) as scaled):
            for key in ("labels","test","groups","ids"):
                np.testing.assert_array_equal(base[key],scaled[key])
            contrast = {}
            for setting, fitted, comparison in [
                    ("fixed", base, ("_S","_Q")), ("matched",scaled,("_S","_matched_Q")),
                    ("selected",base,("_tuned_S","_tuned_Q"))]:
                row = next(r for r in (matched_table if setting=="matched" else native_table)
                           if r["dataset"] == ds and (setting=="matched" or r["penalty"] == setting))
                results = []
                for rep in ("H","D"):
                    split = []
                    for fold,truth in enumerate(fitted["labels"]):
                        before,after = [fitted[rep+key][fold] == truth for key in comparison]
                        split.append(100*np.mean([after[truth==cls].mean()-before[truth==cls].mean()
                                                 for cls in np.unique(truth)]))
                    mean = float(np.mean(split))
                    key = rep+("_gain_pp" if setting == "matched" else "_gain")
                    np.testing.assert_allclose(mean,float(row[key]),rtol=0,atol=1e-9)
                    ends = ("_pointwise_low","_pointwise_high") if setting == "matched" else ("_lo","_hi")
                    lo,hi = [float(row[rep+x]) for x in ends]
                    if not lo <= mean <= hi:
                        raise ValueError(f"Interval does not contain its point estimate for {ds} {rep} {setting}.")
                    results.append(np.asarray(split))
                    gains.append({"dataset":ds,"setting":setting,"representation":rep,
                                  "gain_pp":mean,"pointwise_interval_pp":[lo,hi]})
                contrast[setting] = results[1]-results[0]
            changes = contrast["selected"]-contrast["fixed"]
            if not (np.all(changes < 0) if ds == "DTD" else np.all(changes > 0)):
                raise ValueError(f"Split directions differ from the reported {ds} comparisons.")
            shifts[ds] = changes.tolist()
    clothing = {}
    identity = {}
    for protocol,filename,truth,key,table in [
            ("photograph", data/"clothing/results/predictions.csv","class","source_key",data/"reader/tables/clothing_by_class.csv"),
            ("host",(out/"hosts" if mode=="full" else data/"reader/hosts")/"decisions.csv","truth","key",out/"host_inference/host_classes.csv")]:
        observations = defaultdict(dict)
        for r in csv_rows(filename):
            k = (r["seed"],r[key])
            if k in observations[r[truth]]:
                raise ValueError("Duplicate prediction for the same class, partition and photo.")
            observations[r[truth]][k] = int(r["squares"]==r[truth]) - int(r["crosses"]==r[truth])
        identity[protocol] = {(c,k) for c,rows in observations.items() for k in rows}
        values = {}
        for record in csv_rows(table):
            c=record["pattern"]
            decisions=observations[c]
            if len(decisions)!=int(record["photos"])*30:
                raise ValueError(f"Mismatch in repeated decisions for {c}.")
            effect=100*sum(decisions.values())/len(decisions)
            np.testing.assert_allclose(effect,float(record["difference_pp"]),rtol=0,atol=1e-10)
            values[c]=effect
        if len(values) != 6:
            raise ValueError("Expected six clothing classes.")
        clothing[protocol]=values
    if identity["photograph"] != identity["host"]:
        raise ValueError("Different photographs or seed labels between the clothing protocols.")
    return {"individual_representation_gains":gains, "selected_minus_fixed_by_split_pp":shifts,
            "clothing_class_differences_pp":clothing,
            "FMD_numerical_decisions":verify_fmd_decisions(data, mode)}
