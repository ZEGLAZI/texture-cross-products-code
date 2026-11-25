"""Check a held-out fold in each task with expanded kernels and a dual eigensolver."""
from pathlib import Path
import csv
import numpy as np
from scipy.linalg import eigh
from cnn_models import load_task, load_npz, PENALTIES


def expanded_grams(features, train, test):
    blocks = []
    for vectors in features:
        train_values = vectors[train].astype(np.float64)
        center = np.mean(train_values, axis=0)
        std = np.clip(np.std(train_values, axis=0), 1e-4, None)
        unit = np.clip((vectors[np.r_[train, test]] - center) / std, -12, 12)
        norms = np.linalg.norm(unit, axis=1)
        unit /= np.where(norms == 0, 1, norms)[:, None]
        blocks.append(unit @ unit[:len(train)].T)
    a, b, c = blocks
    shared = (8*(a+b+c) + a*a + b*b + c*c) / 39
    # The equation has linear coefficient 8/39, and square coefficient 1/39.
    return shared, shared + (2/39)*(a*b + b*c + c*a)


def resolve(kernel, train_y, penalties, class_count):
    n = len(train_y)
    target = np.eye(class_count)[train_y]
    prior = np.mean(target, axis=0)
    eigenvalues, eigenvectors = eigh(kernel[:n], driver="evd", check_finite=False)
    left = kernel[n:] @ eigenvectors
    right = eigenvectors.T @ (target-prior)
    return [np.argmax(prior + left @ (right/(eigenvalues[:,None]+value)), axis=1)
            for value in penalties]


def run(root, folder, dataset):
    _, y, matrices, _, folds = load_task(root, folder/"data", dataset)
    fold, train, test, _ = folds[0]
    with (folder/"results"/(dataset+"_settings.csv")).open(newline="") as stream:
        first = next(csv.DictReader(stream))
    if first["fold"] != fold:
        raise ValueError("Fold order differs in the selection table")
    square, full = expanded_grams(matrices, train, test)
    ratio = np.trace(square[:len(train)]) / np.trace(full[:len(train)])
    np.testing.assert_allclose(ratio, 9/11, rtol=0, atol=2e-14)
    square_keys = ("S", "selected_S")
    full_keys = ("Q", "matched_Q", "selected_Q")
    labels = resolve(square, y[train], [1., float(first["lambda_S"])], len(np.unique(y)))
    labels.extend(resolve(full, y[train], [1., 1./ratio, float(first["lambda_Q"])], len(np.unique(y))))
    ref = load_npz(folder/"results"/(dataset+"_decisions.npz"))
    for method, predictions in zip(square_keys + full_keys, labels, strict=True):
        np.testing.assert_array_equal(predictions, ref[method][0])
    return len(test)*len(labels)
