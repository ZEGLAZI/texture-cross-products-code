"""Direct penalty-regime contrast and a validation-only kernel choice on the CVIU archive."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def save(path, entries):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)


def validation_choice(results, dataset):
    scores = defaultdict(list)
    for row in rows(results / f"{dataset}_validation.csv"):
        key = row["fold"], row["kernel"], float(row["penalty"])
        scores[key].append(float(row["macro_recall_pct"]))
    choices = []
    for setting in rows(results / f"{dataset}_settings.csv"):
        fold = setting["fold"]
        best = {}
        for kernel in ("S", "Q"):
            profile = {p: np.mean(v) for (label, k, p), v in scores.items()
                       if label == fold and kernel == k}
            if len(profile) != 6:
                raise ValueError(f"Incomplete grid for {fold} {kernel}")
            best[kernel] = max(profile.values())
            chosen = float(setting["lambda_" + kernel])
            np.testing.assert_allclose(best[kernel], profile[chosen], rtol=0, atol=1e-12)
        choices.append((fold, best["Q"] > best["S"] + 1e-12))
    return choices


def run(root, output):
    sys.path.insert(0, str(root))
    from summarize_pixel_check import macro, paired_interval
    sys.path.insert(0, str(root / "studies/CVIU_Experience_couleur/code"))
    from analyse import ci_external

    results = root / "studies/resnet50/results"
    published = {(r["dataset"], r["contrast"]): float(r["estimate_pp"])
                 for r in rows(results / "contrasts.csv")}
    changes, rules = [], []
    for dataset in ("DTD", "KTH", "Lytro"):
        conv = arrays(results / f"{dataset}_decisions.npz")
        y, groups = conv["labels"], conv["groups"]
        if dataset == "Lytro":
            ref = arrays(root / "studies/CVIU_Experience_couleur/results/Lytro_predictions.npz")
            for key, old in (("ids", "ids"), ("labels", "y"), ("groups", "groups")):
                np.testing.assert_array_equal(conv[key][0], ref[old])
            transformer = {k: ref["D_" + k][None, :] for k in
                           ("fixed_S", "fixed_Q", "selected_S", "selected_Q")}
        else:
            ref = arrays(root / "pixel_check" / f"{dataset}_predictions.npz")
            for key in ("ids", "labels", "groups", "test"):
                np.testing.assert_array_equal(conv[key], ref[key])
            transformer = {mode + "_" + k: ref["D_" + prefix + k]
                           for mode, prefix in (("fixed", ""), ("selected", "tuned_"))
                           for k in ("S","Q")}

        def correctness(labels):
            return (labels == y).astype(np.int16)

        paired = {}
        for mode, prefix in (("fixed", ""), ("selected", "selected_")):
            conv_effect = correctness(conv[prefix+"Q"]) - correctness(conv[prefix+"S"])
            trans_effect = correctness(transformer[mode+"_Q"]) - correctness(transformer[mode+"_S"])
            paired[mode] = conv_effect - trans_effect
            np.testing.assert_allclose(
                macro(paired[mode], y), published[dataset, "R_minus_D_"+mode], atol=1e-10, rtol=0)
        shift = paired["selected"] - paired["fixed"]
        effect = macro(shift, y)
        np.testing.assert_allclose(effect, published[dataset,"R_minus_D_selected"]
                                   - published[dataset,"R_minus_D_fixed"], atol=1e-10, rtol=0)
        if dataset == "Lytro":
            low, high = ci_external(shift.ravel(), y.ravel(), groups.ravel(), 3)[1:]
        else:
            low, high = paired_interval(dataset, shift, y, groups, family=3)
        changes.append(dict(dataset=dataset, common_penalty_gap_pp=macro(paired["fixed"],y),
                            selected_penalties_gap_pp=macro(paired["selected"],y), shift_pp=effect,
                            adjusted_low_pp=low, adjusted_high_pp=high))
        choices = validation_choice(results, dataset)
        fold_order = {"DTD": [f"DTD_{i:02}" for i in range(1,11)],
                      "KTH": ["KTH_"+letter for letter in "abcd"], "Lytro": ["FMD_to_Lytro"]}
        if [fold for fold, _ in choices] != fold_order[dataset] or len(choices) != len(y):
            raise ValueError("Validation folds and held-out decisions do not match")
        choose_q = np.array([q for _, q in choices])[:, None]
        selected = np.where(choose_q, conv["selected_Q"], conv["selected_S"])
        choose_recall = macro(correctness(selected), y)
        square = macro(correctness(conv["selected_S"]), y)
        full = macro(correctness(conv["selected_Q"]), y)
        rules.append(dict(dataset=dataset, folds=len(y), full_kernel_chosen=int(choose_q.sum()),
                          squares_recall_pct=square, full_recall_pct=full, selection_recall_pct=choose_recall,
                          selection_minus_squares_pp=choose_recall-square,
                          selection_minus_full_pp=choose_recall-full))
    output.mkdir(exist_ok=True, parents=True)
    save(output/"contrast.csv",changes)
    save(output/"validation.csv",rules)
    for collection in (changes, rules):
        for record in collection:
            print({k: round(v,4) if isinstance(v,(float,np.floating)) else v for k,v in record.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviewer_package", type=Path, help="Unpacked IVC_Reviewer.zip, version with ResNet-50")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.reviewer_package.resolve(), args.output)
