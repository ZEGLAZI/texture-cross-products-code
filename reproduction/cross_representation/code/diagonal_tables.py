"""Rebuild matched-diagonal gains and each selected ridge penalty from aligned records."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from compare import metadata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"review2/code"))
from inference import dtd_interval, external, macro

FAMILIES = ("H", "D")


def interval(corpus, difference, labels, groups, family_size):
    if corpus == "DTD":
        return dtd_interval(difference, labels, groups, m=family_size, n=100000, seed=19112026)
    return external(difference.ravel(), labels.ravel(), groups.ravel(), m=family_size, n=100000)


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def build(matched_dir, native_dir, out):
    out.mkdir(parents=True, exist_ok=True)
    summaries, penalties, splits = [], [], []
    colour_fits = json.loads((native_dir/"HSMT_refits.json").read_text())
    for corpus in ("DTD","KTH"):
        ids, labels, groups, _, classes = metadata(corpus, ROOT)
        with np.load(matched_dir/(corpus+"_matched_predictions.npz")) as f:
            truth, test, test_groups = f["labels"], f["test"], f["groups"]
            np.testing.assert_array_equal(f["ids"], ids[test])
            np.testing.assert_array_equal(truth, labels[test])
            np.testing.assert_array_equal(test_groups, groups[test])
            with np.load(native_dir/(corpus+"_predictions.npz")) as native:
                np.testing.assert_array_equal(test, native["test"])
                for fam in FAMILIES:
                    for name in ("S","Q"):
                        key = fam+"_"+name
                        np.testing.assert_array_equal(f[key], native[key])
            row={"dataset":corpus}
            correct = {}
            gains = {}
            for fam in FAMILIES:
                for name in ("S", "Q", "matched_Q"):
                    key = fam+"_"+name
                    correct[key] = (f[key] == truth).astype(np.int16)
                    row[key+"_recall_pct"] = float(np.mean([macro(a,b) for a,b in zip(correct[key],truth)]))
                gains[fam] = correct[fam+"_matched_Q"] - correct[fam+"_S"]
                gain, low, high = interval(corpus, gains[fam], truth, test_groups, 1)
                row.update({fam+"_gain_pp":gain, fam+"_pointwise_low":low, fam+"_pointwise_high":high})
            contrast, low, high = interval(corpus, gains["D"] - gains["H"], truth, test_groups, 2)
            original = correct["D_Q"] - correct["D_S"] - correct["H_Q"] + correct["H_S"]
            native_contrast = float(np.mean([macro(a,b) for a,b in zip(original,truth)]))
            row.update(native_D_pp=native_contrast, matched_D_pp=contrast, matched_D_simultaneous_low=low,
                       matched_D_simultaneous_high=high, matched_minus_native_D_pp=contrast-native_contrast)
            summaries.append(row)
            settings = json.loads((native_dir/(corpus+"_settings.json")).read_text())
            checks = colour_fits[corpus+"_equal_weights_selection"]
            if len(settings) != len(truth) or len(checks) != len(truth):
                raise ValueError("Missing outer-fold settings")
            for i, (config, checked) in enumerate(zip(settings,checks,strict=True)):
                if config["fold"] != checked["fold"] or config["H"] != checked["penalties"]:
                    raise ValueError("Selected colour–texture penalties disagree")
                grid = np.asarray(config["lambda_grid"])
                setting = {"dataset":corpus,"fold":config["fold"]}
                for fam in FAMILIES:
                    for name in ("S","Q"):
                        penalty = config[fam][name]
                        if fam=="D" and penalty != grid[np.argmax(config["D_inner_macro_recall_fraction"][name])]:
                            raise ValueError("Selected DINOv2 penalty disagrees with the first inner maximum")
                        setting[fam+"_"+name] = penalty
                penalties.append(setting)
                split = {"dataset":corpus,"fold":config["fold"],"n_test":len(truth[i])}
                np.testing.assert_array_equal(np.unique(truth[i]),np.arange(len(classes)))
                for name, values in correct.items():
                    split[name+"_pct"] = macro(values[i],truth[i])
                splits.append(split)
    save_csv(out/"scale_comparison.csv", summaries)
    save_csv(out/"selected_penalties.csv", penalties)
    save_csv(out/"scale_by_split.csv", splits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("matched_dir","native_dir","output"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    build(args.matched_dir,args.native_dir,args.output)
