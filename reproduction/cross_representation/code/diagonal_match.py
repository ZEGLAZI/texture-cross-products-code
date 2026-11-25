"""Refit the full kernel at the original square kernel's mean training diagonal."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from compare import metadata, normalize_block

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "review2/code"))
sys.path.insert(1, str(ROOT / "historical/dtd"))
from assess import forms, kernels, stage
from nested import PROFILES, fit
from predire import load_blocks

LAYERS = ("patch3", "patch6", "patch12")


def representation_grams(family, raw, layers, train, test):
    if family == "H":
        return stage(raw, train, test)
    order = np.r_[train, test]
    blocks = [normalize_block(layers[layer], train) for layer in LAYERS]
    grams = [block[order] @ block[train].T for block in blocks]
    diagonals = [np.einsum("ij,ij->i", block[order], block[order]) for block in blocks]
    return grams, diagonals


def compare_fold(family, raw, layers, train, test, labels, n_classes, native):
    grams, blocks = representation_grams(family, raw, layers, train, test)
    n_train = len(train)
    squared_norms = np.asarray(blocks, dtype=np.float64)
    if not np.isfinite(squared_norms).all() or np.min(squared_norms) < 0:
        raise ValueError("Squared block norms must be finite and nonnegative")
    diagonal = dict(forms(squared_norms, PROFILES[1]))
    row = {"n_train": n_train, "n_test": len(test)}
    for role, indices in (("train", slice(0, n_train)), ("test", slice(n_train, None))):
        selected = squared_norms[:, indices]
        for b, values in enumerate(selected):
            row[f"{role}_zero_block_{b+1}"] = int(np.count_nonzero(values == 0))
        row[role+"_max_abs_sqnorm_minus_one"] = float(np.max(np.abs(selected-1)))
    predictions = {}
    for name, kernel in kernels(grams, blocks, 1):
        if name not in ("S", "Q"):
            continue
        training_diagonal = np.diag(kernel[:n_train])
        np.testing.assert_allclose(training_diagonal, diagonal[name][:n_train], rtol=0, atol=2e-6)
        training_mean = float(np.mean(training_diagonal, dtype=np.float64))
        if training_mean <= 0:
            raise ValueError("Cannot match a zero mean training diagonal")
        row[name+"_mean_train_diagonal"] = training_mean
        scores = fit(kernel, n_train, labels[train], 1., n_classes)
        predictions[name] = np.argmax(scores, axis=1).astype(np.int16)
        np.testing.assert_array_equal(predictions[name], native[name])
        if name != "Q":
            continue
        factor = row["S_mean_train_diagonal"] / training_mean
        matched = fit(factor * kernel, n_train, labels[train], 1., n_classes)
        equivalent = fit(kernel, n_train, labels[train], 1./factor, n_classes)
        np.testing.assert_allclose(matched, equivalent, atol=1e-10, rtol=1e-10)
        np.testing.assert_array_equal(matched.argmax(axis=1), equivalent.argmax(axis=1))
        predictions["matched_Q"] = np.argmax(matched, axis=1).astype(np.int16)
        row["factor"] = factor
        row["equivalent_penalty"] = 1./factor
        # Record numerical score agreement and verify discrete labels separately.
        row["max_score_equivalence_error"] = float(np.max(np.abs(matched-equivalent)))
        row["matched_mean_train_diagonal"] = float((factor*training_diagonal).mean())
        row["max_paired_diagonal_error_train_and_test"] = float(
            np.max(np.abs(factor * diagonal["Q"] - diagonal["S"])))
        theoretical = fit(kernel, n_train, labels[train], 11./9, n_classes).argmax(axis=1)
        row["label_differences_at_theoretical_penalty"] = int(
            np.count_nonzero(theoretical != predictions["matched_Q"]))
        break
    return predictions, row


def run(corpus, data, output, features_dir, threads):
    ids, labels, groups, hashes, class_names = metadata(corpus, data)
    with np.load(features_dir/(corpus+"_features.npz")) as f:
        np.testing.assert_array_equal(f["ids"], ids)
        layers = {layer: f[layer] for layer in LAYERS}
    if corpus == "DTD":
        raw = load_blocks(data/"review2/dtd_cache/hsmt", ids, ["F7","F8","rgb"])
    else:
        with np.load(data/"historical/external/data/kth.npz") as f:
            raw = {k: f[k] for k in ("F7", "F8", "rgb")}
    paths = sorted((data/"review2/results/kernels").glob(corpus+"_*.npz"))
    expected_folds = 10 if corpus == "DTD" else 4
    if len(paths) != expected_folds:
        raise ValueError(f"Expected {expected_folds} splits")
    record = []
    predictions = []
    folder = data/"cross_representation/results"
    with np.load(folder/(corpus+"_predictions.npz")) as original, threadpool_limits(threads):
        for j, path in enumerate(paths):
            with np.load(path) as f:
                train, test = f["train"], f["test"]
                np.testing.assert_array_equal(labels[test], f["labels"])
            if np.intersect1d(groups[train],groups[test]).size or np.intersect1d(hashes[train],hashes[test]).size:
                raise ValueError("Train and test share a source")
            for name, values in (("test",test),("labels",labels[test]),("groups",groups[test]),("ids",ids[test])):
                np.testing.assert_array_equal(values, original[name][j])
            fold = {"test":test, "labels":labels[test], "groups":groups[test], "ids":ids[test]}
            for family in ("H", "D"):
                native = {k: original[family+"_"+k][j] for k in ("S","Q")}
                results, diagnostics = compare_fold(
                    family, raw, layers, train, test, labels, len(class_names), native)
                fold.update({family+"_"+k: v for k,v in results.items()})
                record.append({"dataset":corpus, "fold":path.stem, "representation":family, **diagnostics})
            predictions.append(fold)
            print(path.stem, "matched full kernel and checked native labels", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/(corpus+"_matched_predictions.npz"),
                        **{k:np.stack([fold[k] for fold in predictions]) for k in predictions[0]})
    with (output/(corpus+"_diagonals.csv")).open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=record[0]); writer.writeheader(); writer.writerows(record)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus",choices=["DTD","KTH"])
    parser.add_argument("output",type=Path)
    parser.add_argument("--threads",type=int,default=2)
    args=parser.parse_args()
    np.testing.assert_array_equal(PROFILES[1], [1/3]*3)
    run(args.corpus, ROOT, args.output, ROOT/"cross_representation/features", args.threads)
