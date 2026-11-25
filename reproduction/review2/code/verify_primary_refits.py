"""Refit the four fixed-coefficient kernels on DTD and clothing from their descriptors."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from calcul_historique import compact


def refit_fold(raw, labels, fold_path):
    with np.load(fold_path) as saved:
        train, test = saved["train"], saved["test"]
        expected = saved["fixed"][:, 0]
        assert saved["modes"][:4].tolist() == ["L", "S", "X", "Q"]
        np.testing.assert_array_equal(saved["labels"], labels[test])

    normalized, _ = compact(raw, training=train)
    normalized *= np.sqrt(2.0)
    blocks = (normalized[:, :945], normalized[:, 945:], raw["rgb"])
    order = np.concatenate([train, test])
    weights = (.25, .25, .5)
    weighted_grams = [weight * (block[order] @ block[train].T)
                      for weight, block in zip(weights, blocks)]
    s, w, c = weighted_grams
    q = s + w + c
    n = len(train)
    targets = np.eye(int(labels.max()) + 1)[labels[train]]
    prior = targets.mean(axis=0)
    targets -= prior
    mismatches = {}
    for index, (name, use_squares, use_products) in enumerate([
        ("linear", False, False), ("linear_squares", True, False),
        ("linear_products", False, True), ("full_quadratic", True, True),
    ]):
        kernel = 2 * q
        if use_squares:
            for gram in weighted_grams:
                kernel += .75 * np.square(gram)
        if use_products:
            for left, right in ((s, w), (s, c), (w, c)):
                kernel += 1.5 * (left * right)
        kernel /= 3.25
        regularized = np.array(kernel[:n], order="F", copy=True)
        regularized.flat[::n + 1] += 1.0
        factor = cho_factor(regularized, lower=True, overwrite_a=True, check_finite=False)
        coefficients = cho_solve(factor, targets, check_finite=False)
        prediction = np.argmax(kernel[n:] @ coefficients + prior, axis=1)
        mismatches[name] = int(np.count_nonzero(prediction != expected[index]))
        del kernel, regularized, factor, coefficients
    return len(test), mismatches


def check_corpus(raw, labels, fold_paths):
    total = 0
    mismatches = dict(linear=0, linear_squares=0, linear_products=0, full_quadratic=0)
    per_fold = []
    for path in fold_paths:
        size, fold_counts = refit_fold(raw, labels, path)
        total += size
        for name, count in fold_counts.items():
            mismatches[name] += count
        per_fold.append(dict(fold=path.stem, test_appearances=size, differences=fold_counts))
    return dict(test_appearances=total, differences=mismatches, folds=per_fold)


def run(root):
    sys.path.insert(0, str(root / "historical/dtd"))
    from predire import load_blocks

    folds = root / "review2/results/kernels"
    with np.load(root / "historical/dtd/results/identites.npz") as ids:
        labels = ids["y"]
        raw = load_blocks(root / "review2/dtd_cache/hsmt", ids["ids"], ["F7", "F8", "rgb"])
    dtd = check_corpus(raw, labels, sorted(folds.glob("DTD_*.npz")))
    assert len(dtd["folds"]) == 10 and dtd["test_appearances"] == 18800
    del raw
    with np.load(root / "clothing/data/features.npz") as saved:
        raw = {key: saved[key] for key in ("F7", "F8", "rgb")}
        labels = saved["y"]
    clothing = check_corpus(raw, labels, sorted(folds.glob("Textile_*.npz")))
    assert len(clothing["folds"]) == 150 and clothing["test_appearances"] == 6330
    return {"DTD": dtd, "Clothing": clothing}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="output JSON for counts by fold and corpus")
    args = parser.parse_args()
    with threadpool_limits(3):
        result = run(Path(__file__).resolve().parents[2])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, row in result.items():
        print(name, row["test_appearances"], row["differences"])
    if any(any(row["differences"].values()) for row in result.values()):
        raise SystemExit("One or more refitted decisions differ from the archived decisions.")
