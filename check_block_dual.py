"""Check each additive block-subset decision using the released dual Gram-matrix solver."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits


def check(data, pixels, predictions):
    sys.path.insert(0, str((data / "review2/code").resolve()))
    sys.path.insert(1, str((data / "cross_representation/code").resolve()))
    from assess import stage
    from nested import fit
    from compare import metadata

    checks = {}
    for corpus in ("DTD", "KTH"):
        ids, y, groups, hashes, classes = metadata(corpus, data)
        with np.load(pixels / f"{corpus}_colour.npz") as z:
            np.testing.assert_array_equal(z["ids"], ids)
            raw = {key: z[key] for key in ("F7", "F8", "rgb")}
        checked = 0
        folds = sorted((data / "review2/results/kernels").glob(f"{corpus}_*.npz"))
        with (np.load(predictions / f"{corpus}_predictions.npz") as saved,
              np.load(pixels / f"{corpus}_predictions.npz") as reference):
            for index, path in enumerate(folds):
                with np.load(path) as f:
                    train, test = f["train"], f["test"]
                for g in (groups, hashes):
                    assert not np.intersect1d(g[train], g[test]).size
                np.testing.assert_array_equal(saved["test"][index], test)
                for name, expected in (("ids", ids), ("groups", groups), ("labels", y)):
                    np.testing.assert_array_equal(reference[name][index], expected[test])
                grams, diagonals = stage(raw, train, test)
                np.testing.assert_allclose(diagonals, 1., rtol=0., atol=3.e-12)
                for subset in ("S", "W", "C", "SW", "SC", "WC", "SWC"):
                    kernel = sum(grams["SWC".index(k)] for k in subset) * (8 / (13 * len(subset)))
                    labels = fit(kernel, len(train), y[train], 1., len(classes)).argmax(axis=1)
                    np.testing.assert_array_equal(labels, saved[subset][index], err_msg=f"{path.stem}: {subset}")
                    checked += len(test)
                print(path.stem, "all seven prediction arrays matched", flush=True)
        checks[corpus] = dict(fits=len(folds) * 7, test_label_comparisons=checked)
    return checks


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path, help="Unpacked data_and_code_r4.zip")
    p.add_argument("predictions", type=Path, help="Block-subset decisions to check")
    p.add_argument("--pixels", type=Path, default=Path(__file__).resolve().parent / "pixel_check")
    args = p.parse_args()
    with threadpool_limits(limits=2):
        print(json.dumps(check(args.data, args.pixels, args.predictions)))
