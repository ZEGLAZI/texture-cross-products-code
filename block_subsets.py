"""Refit additive colour–texture block subsets using the outer splits of main Table 2."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from independent_refit import load_inputs, scaled_block, targets

BLOCKS = ("S", "W", "C")
SUBSETS = ["".join(s) for size in (1, 2, 3) for s in combinations(BLOCKS, size)]
CORPORA = ("DTD", "KTH")
DIVISOR = 13 / 4
FAMILY_SIZE = 6
DRAWS = 100_000
SEED = 19_112_026


def refit(data, pixels, corpus, out):
    ids, labels, splits, raw, _ = load_inputs(data, corpus, pixels)
    blocks_predicted = {subset: [] for subset in SUBSETS}
    test_indices = []
    with np.load(pixels / f"{corpus}_predictions.npz") as reference:
        for fold, (name, train, test, _) in enumerate(splits):
            np.testing.assert_array_equal(test, reference["test"][fold])
            np.testing.assert_array_equal(labels[test], reference["labels"][fold])
            np.testing.assert_array_equal(ids[test], reference["ids"][fold])
            blocks = (scaled_block(raw["F7"], train, 9), scaled_block(raw["F8"], train, 3),
                      np.asarray(raw["rgb"], dtype=np.float64))
            for block in blocks:
                np.testing.assert_allclose(np.linalg.norm(block, axis=1), 1., rtol=0., atol=2.e-12)
            xtr = np.concatenate([block[train] for block in blocks], axis=1)
            xte = np.concatenate([block[test] for block in blocks], axis=1)
            boundaries = np.cumsum([0, *[block.shape[1] for block in blocks]])
            columns = dict(zip(BLOCKS, np.split(np.arange(xtr.shape[1]), boundaries[1:-1]), strict=True))
            centered, prior = targets(labels[train], int(labels.max()) + 1)
            gram, response = xtr.T @ xtr, xtr.T @ centered
            for subset in SUBSETS:
                keep = np.concatenate([columns[key] for key in subset])
                # K = 2 XX' / (d |J|). With unit dual penalty the primal ridge
                # for the unscaled columns of X is d |J| / 2.
                system = gram[np.ix_(keep, keep)]
                system.flat[::len(keep) + 1] += DIVISOR * len(subset) / 2
                beta = solve(system, response[keep], assume_a="pos", check_finite=False)
                yhat = (xte[:, keep] @ beta + prior).argmax(axis=1)
                blocks_predicted[subset].append(yhat.astype(np.int16))
            test_indices.append(test)
            print(f"{name}: seven block subsets fitted", flush=True)
    out.mkdir(exist_ok=True, parents=True)
    arrays = {subset: np.stack(rows) for subset, rows in blocks_predicted.items()}
    np.savez_compressed(out / f"{corpus}_predictions.npz", test=np.stack(test_indices), **arrays)


def summarize(data, pixels, predictions, out):
    # Apply exactly the grouping and interval routines used by the main experiment.
    sys.path.insert(0, str((data / "review2/code").resolve()))
    from inference import dtd_interval, external

    out.mkdir(exist_ok=True, parents=True)
    result = {}
    for corpus in CORPORA:
        with (np.load(pixels / f"{corpus}_predictions.npz") as metadata,
              np.load(predictions / f"{corpus}_predictions.npz") as fits):
            np.testing.assert_array_equal(fits["test"], metadata["test"])
            labels, groups = metadata["labels"], metadata["groups"]
            if set(fits.files) != {"test", *SUBSETS}:
                raise ValueError(f"{corpus}: unexpected block subsets")
            expected_shape = (10, 1880) if corpus == "DTD" else (4, 1188)
            assert labels.shape == groups.shape == expected_shape
            correct = {}
            scores, by_split = {}, {}
            for subset in SUBSETS:
                pred = fits[subset]
                assert pred.shape == labels.shape and np.isin(pred, np.unique(labels)).all()
                hit = (pred == labels).astype(np.int16)
                correct[subset] = hit
                fold_scores = [100 * np.mean([h[y == c].mean() for c in np.unique(y)])
                               for h, y in zip(hit, labels, strict=True)]
                by_split[subset] = list(map(float, fold_scores))
                scores[subset] = float(np.mean(fold_scores))
            differences = {}
            for pair in ("SW", "SC", "WC"):
                delta = correct["SWC"] - correct[pair]
                if corpus == "DTD":
                    for y in labels:
                        np.testing.assert_array_equal(np.bincount(y), np.full(47, 40))
                    interval = dtd_interval(delta, labels, groups, m=FAMILY_SIZE, n=DRAWS, seed=SEED)
                else:
                    assert np.unique(groups).size == 44
                    for y in np.unique(labels):
                        _, counts = np.unique(groups[labels == y], return_counts=True)
                        np.testing.assert_array_equal(counts, np.full(4, 108))
                    interval = external(delta.ravel(), labels.ravel(), groups.ravel(), m=FAMILY_SIZE, n=DRAWS)
                np.testing.assert_allclose(interval[0], scores["SWC"] - scores[pair], atol=1.e-12)
                differences[pair] = list(map(float, interval))
            result[corpus] = dict(macro_recall_pct=scores, full_minus_pair_pp=differences,
                                  recall_pct_by_split=by_split, test_appearances=labels.size,
                                  groups=int(np.unique(groups).size))
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def prepare(data, root, output, mode):
    from check_reference import compare_npz

    reference = root / "block_subsets"
    pixels = root / "pixel_check"
    dest = output / "block_subsets"
    predictions = reference
    with threadpool_limits(limits=2):
        if mode == "full":
            for corpus in CORPORA:
                refit(data, pixels, corpus, dest)
                filename = f"{corpus}_predictions.npz"
                compare_npz(dest / filename, reference / filename)
            predictions = dest
        summary = summarize(data, pixels, predictions, dest)
    archived = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    for corpus in CORPORA:
        for field in ("macro_recall_pct", "full_minus_pair_pp", "recall_pct_by_split"):
            if set(archived[corpus][field]) != set(summary[corpus][field]):
                raise ValueError(f"{corpus}: differing subset names for {field}")
            for key, measured in summary[corpus][field].items():
                np.testing.assert_allclose(measured, archived[corpus][field][key], atol=1.e-10, rtol=0.)
        for field in ("groups", "test_appearances"):
            if summary[corpus][field] != archived[corpus][field]:
                raise ValueError(f"{corpus}: differing {field}")
    return {"mode": mode, "fits": 98 if mode == "full" else 0,
            "decisions_evaluated": sum(len(SUBSETS) * summary[c]["test_appearances"] for c in CORPORA)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Unpacked data_and_code_r4.zip")
    parser.add_argument("output", type=Path)
    parser.add_argument("--pixels", type=Path, default=Path(__file__).resolve().parent / "pixel_check")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        for dataset in CORPORA:
            refit(args.data, args.pixels, dataset, args.output)
        summary = summarize(args.data, args.pixels, args.output, args.output)
    for dataset, row in summary.items():
        print(dataset, row["macro_recall_pct"], row["full_minus_pair_pp"])
