"""Check FMD decisions without loading any rendered figure or article."""
import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from independent_refit import scaled_block, targets


def refit_pair(features, train, test):
    """Fit both native-scale kernels at p0 and lambda=1 from raw FMD blocks."""
    row_ids = np.concatenate((train, test))
    blocks = (scaled_block(features["F7"], train, 9),
              scaled_block(features["F8"], train, 3), features["rgb"])
    weighted = [p * (block[row_ids] @ block[train].T)
                for p, block in zip((.25, .25, .5), blocks, strict=True)]
    linear = sum(weighted)
    squares = sum(block**2 for block in weighted)
    rhs, prior = targets(features["y"][train], len(features["classes"]))
    predictions = []
    for quadratic in (squares, linear**2):
        kernel = (2 * linear + .75 * quadratic) / 3.25
        n = len(train)
        alpha = solve(kernel[:n] + np.eye(n), rhs, assume_a="pos")
        predictions.append((kernel[n:] @ alpha + prior).argmax(axis=1))
    return predictions


def verify_fmd_decisions(data, mode):
    with np.load(data / "review2/extra/fmd.npz", allow_pickle=False) as features, \
            np.load(data / "review2/results/kernels/FMD.npz", allow_pickle=False) as reference, \
            threadpool_limits(limits=2):
        train, test = features["train"], features["test"]
        for indices, key in ((train, "train"), (test, "test")):
            np.testing.assert_array_equal(indices, reference[key])
        expected = [reference["fixed"][reference["modes"].tolist().index(key), 0] for key in ("S", "Q")]
        predictions = refit_pair(features, train, test) if mode == "full" else expected
        for actual, wanted in zip(predictions, expected, strict=True):
            np.testing.assert_array_equal(actual, wanted)
        truth = features["y"][test]
        np.testing.assert_array_equal(np.bincount(truth), np.full(len(features["classes"]), 50))
        square, full = predictions
        corrected = int(np.sum((square != truth) & (full == truth)))
        lost = int(np.sum((square == truth) & (full != truth)))
        same = int(np.sum(square == full))
        other = int(np.sum((square != full) & (square != truth) & (full != truth)))
        counts = [same, corrected, lost, other]
        right = [int(np.sum(guess == truth)) for guess in predictions]
        if sum(counts) != len(test) or counts != [470, 9, 9, 12] or right != [222, 222]:
            raise ValueError("FMD decision partition differs from the archived numerical reference.")
    return {"refitted_test_labels": 2 * len(test) if mode == "full" else 0,
            "summarized_test_labels": 2 * len(test), "correct_per_kernel": right,
            "same_label": same, "corrected": corrected, "newly_wrong": lost,
            "changed_still_wrong": other}
