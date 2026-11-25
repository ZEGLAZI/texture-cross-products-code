"""Cross-check the two equal-weight kernels from their cached, unstandardized blocks.

Usage: python independent_refit.py PATH_TO_EXTRACTED_DATA [--threads 2]
Runs all outer and inner DTD and KTH-TIPS2-b splits. Uses no experiment-code imports.
"""
import argparse
import csv
import json
from pathlib import Path

from pixel_files import transformer_file

import numpy as np
from scipy.linalg import eigh, solve
from threadpoolctl import threadpool_limits

PENALTIES = np.array([1., 0.1, 10., 0.01, 100., 0.001])
MODES = ("S", "Q")


def load_inputs(root, corpus, pixel_dir=None):
    if corpus == "DTD":
        with np.load(root / "historical/dtd/results/identites.npz") as z:
            meta = dict(z)
        parts = sorted((root / "review2/dtd_cache/hsmt").glob("[0-9]*.npz"))
        tiles = {key: [] for key in ("ids", "F7", "F8", "rgb")}
        for path in parts:
            with np.load(path) as z:
                for key in tiles:
                    tiles[key].append(z[key])
        raw = {key: np.concatenate(values) for key, values in tiles.items()}
        ids, y = meta["ids"], meta["y"]
        np.testing.assert_array_equal(raw.pop("ids"), ids)
        def exclude(indices, other):
            conflict = np.isin(meta["source_sha256"][indices], meta["source_sha256"][other])
            conflict |= np.isin(meta["pixels224"][indices], meta["pixels224"][other])
            return indices[~conflict]
        splits = []
        for number in range(1, 11):
            tr, va, te = (meta[key + str(number)] for key in ("train", "val", "test"))
            pool = exclude(np.concatenate((tr, va)), te)
            val = exclude(va, te)
            fit = exclude(exclude(tr, te), val)
            splits.append((f"DTD_{number:02d}", pool, te, [(fit, val)]))
    else:
        base = root / "historical/external/data"
        with (base / "kth_images.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        ids = np.array([r["chemin"] for r in rows])
        y = np.unique([r["materiau"] for r in rows], return_inverse=True)[1]
        specimens = np.array([r["specimen"] for r in rows])
        hashes = np.array([r["sha_pixels"] for r in rows])
        with np.load(base / "kth.npz") as z:
            np.testing.assert_array_equal(z["path"], ids)
            raw = {k: z[k] for k in ("F7", "F8", "rgb")}
        splits = []
        for letter in sorted(set(specimens)):
            te = np.flatnonzero(specimens == letter)
            pool = np.flatnonzero((specimens != letter) & ~np.isin(hashes, hashes[te]))
            inner = []
            for held in sorted(set(specimens[pool])):
                val = pool[specimens[pool] == held]
                fit = pool[(specimens[pool] != held) & ~np.isin(hashes[pool], hashes[val])]
                inner.append((fit, val))
            splits.append(("KTH_" + letter, pool, te, inner))
    feature_file = root / f"cross_representation/features/{corpus}_features.npz"
    if pixel_dir is not None:
        with np.load(pixel_dir / (corpus+"_colour.npz")) as z:
            np.testing.assert_array_equal(z["ids"], ids)
            raw = {k: z[k] for k in ("F7","F8","rgb")}
        feature_file = transformer_file(root, pixel_dir, corpus)
    with np.load(feature_file) as z:
        np.testing.assert_array_equal(z["ids"], ids)
        layers = tuple(z["patch" + str(k)] for k in (3, 6, 12))
    return ids, y, splits, raw, layers


def scaled_block(array, train, dimension=None):
    values = array.astype(np.float64)
    if dimension is None:
        axes = 0
    else:
        row, col = np.triu_indices(dimension)
        values[..., np.flatnonzero(row != col)] *= np.sqrt(2.)
        axes = (0, 2)
    sample = values[train]
    values -= sample.mean(axis=axes, keepdims=True)
    values /= np.maximum(sample.std(axis=axes, keepdims=True), 1.e-4)
    np.clip(values, -12., 12., out=values)
    flat = values.reshape(len(values), -1)
    length = np.linalg.norm(flat, axis=1, keepdims=True)
    return np.divide(flat, length, out=np.zeros_like(flat), where=length != 0)


def kernel_pair(blocks, tr, ev):
    # Expanding the three distinct products explicitly avoids using the squared
    # weighted sum that constructs the full kernel in the experiment code.
    rows = np.concatenate((tr, ev))
    train_blocks = [np.ascontiguousarray(b[tr]) for b in blocks]
    for block in blocks:
        np.testing.assert_allclose(np.linalg.norm(block[rows], axis=1), 1., rtol=0., atol=2.e-12)
    squares = np.empty((len(rows), len(tr)))
    full = np.empty_like(squares)
    for start in range(0, len(rows), 192):
        section = slice(start, start+192)
        a, b, c = (block[rows[section]] @ ref.T for block, ref in zip(blocks, train_blocks))
        squares[section] = ((8./3.)*(a+b+c) + (a*a+b*b+c*c)/3.)/13.
        full[section] = squares[section] + (2./39.)*(a*b+a*c+b*c)
    return squares, full


def targets(y, classes):
    encoded = np.eye(classes)[y]
    prior = np.mean(encoded, axis=0)
    return encoded-prior, prior


def check_corpus(root, corpus, pixel_dir=None, output_dir=None):
    ids, y, splits, raw, layers = load_inputs(root, corpus, pixel_dir)
    decisions = {}
    classes = int(y.max())+1
    cross = root / "cross_representation"
    settings = json.loads((cross / f"results/{corpus}_settings.json").read_text())
    records = []
    with np.load(cross / f"results/{corpus}_predictions.npz") as published, np.load(
        cross / f"scale_results/{corpus}_matched_predictions.npz"
    ) as matched:
        for index, (name, train, test, inner) in enumerate(splits):
            with np.load(root / f"review2/results/kernels/{name}.npz") as fold:
                for key, indices in (("train", train), ("test", test)):
                    np.testing.assert_array_equal(fold[key], indices)
                np.testing.assert_array_equal(published["ids"][index], ids[test])
                np.testing.assert_array_equal(published["labels"][index], y[test])
                for j, (subtrain, val) in enumerate(inner):
                    np.testing.assert_array_equal(fold["inner_train_"+str(j)], subtrain)
                    np.testing.assert_array_equal(fold["inner_val_"+str(j)], val)
                expected_h_grid = fold["grid"][[1,3], 1] / 100.
            if pixel_dir is not None:
                for field in ("ids","labels","groups","test"):
                    decisions.setdefault(field,[]).append(published[field][index])
            for family in ("H", "D"):
                def matrices(fit, ev):
                    if family == "H":
                        blocks = (scaled_block(raw["F7"], fit, 9), scaled_block(raw["F8"], fit, 3), raw["rgb"])
                    else:
                        blocks = [scaled_block(layer, fit) for layer in layers]
                    return kernel_pair(blocks, fit, ev)
                grid = np.zeros((2, len(PENALTIES)))
                for fit, val in inner:
                    square, full = matrices(fit, val)
                    target, prior = targets(y[fit], classes)
                    for j, kernel in enumerate((square, full)):
                        eigval, eigvec = eigh(kernel[:len(fit)], driver="evd", check_finite=False)
                        right, left = eigvec.T @ target, kernel[len(fit):] @ eigvec
                        for col, pen in enumerate(PENALTIES):
                            labels = (left @ (right/(eigval[:,None]+pen)) + prior).argmax(axis=1)
                            grid[j,col] += np.mean([np.mean(labels[y[val]==k]==k) for k in range(classes)]) / len(inner)
                    del square, full, kernel, eigvec, left
                published_grid = expected_h_grid if family == "H" else np.array([
                    settings[index]["D_inner_macro_recall_fraction"][mode] for mode in MODES
                ])
                if pixel_dir is None:
                    np.testing.assert_allclose(grid, published_grid, rtol=0., atol=1.e-12)
                # First candidate at the maximum; use a roundoff tolerance only for exact score ties.
                chosen = [float(PENALTIES[np.flatnonzero(row >= row.max()-1.e-13)[0]]) for row in grid]
                original_selection = [settings[index][family][k] for k in MODES]
                if pixel_dir is None and chosen != original_selection:
                    raise ValueError(f"{name} {family}: ridge selection {chosen} disagrees with {original_selection}")
                square, full = matrices(train, test)
                test_count = len(test)
                n = len(train)
                factor = np.trace(square[:n])/np.trace(full[:n])
                np.testing.assert_allclose(factor, 9./11., rtol=0., atol=1.e-12)
                target, prior = targets(y[train], classes)
                label_changes = {}
                for j, kernel in enumerate((square, full)):
                    mode = MODES[j]
                    native_name, tuned_name = family+"_"+mode, family+"_tuned_"+mode
                    expected = [(1., native_name, published[native_name][index]),
                                (chosen[j], tuned_name, published[tuned_name][index])]
                    if mode == "Q":
                        matched_name = family+"_matched_Q"
                        expected.append((1./factor, matched_name, matched[matched_name][index]))
                    by_penalty = {}
                    for pen, field, truth in expected:
                        if pen not in by_penalty:
                            system = np.array(kernel[:n], order="F", copy=True)
                            system.flat[::n+1] += pen
                            alpha = solve(system, target, assume_a="sym", overwrite_a=True, check_finite=False)
                            by_penalty[pen] = (kernel[n:] @ alpha + prior).argmax(axis=1)
                        labels = by_penalty[pen]
                        label_changes[field] = int(np.count_nonzero(labels != truth))
                        if pixel_dir is None:
                            np.testing.assert_array_equal(labels, truth)
                        else:
                            decisions.setdefault(field, []).append(labels.astype(np.int16))
                del square, full, kernel
                records.append(dict(split=name, representation=family, test_appearances=test_count, selections=chosen,
                                    inner_scores=2*len(PENALTIES), predictions_compared=5*test_count,
                                    original_selections=original_selection, different_test_labels=label_changes,
                                    max_grid_difference_pp=100*float(np.max(np.abs(grid-published_grid)))))
                print(name, family, "checked", flush=True)
    if pixel_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(output_dir/(corpus+"_predictions.npz"), **{k:np.stack(v) for k,v in decisions.items()})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--corpus", choices=("DTD","KTH","both"), default="both")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pixel-dir", type=Path, help="Use newly extracted CORPUS_colour.npz and CORPUS_features.npz")
    parser.add_argument("--output-dir", type=Path, help="Save fresh held-out decisions for the pixel sensitivity")
    args = parser.parse_args()
    if (args.pixel_dir is None) != (args.output_dir is None):
        parser.error("Supply --pixel-dir and --output-dir together")
    if args.threads < 1:
        parser.error("--threads must be positive")
    with threadpool_limits(limits=args.threads):
        rows = []
        for corpus in (("DTD","KTH") if args.corpus == "both" else (args.corpus,)):
            rows.extend(check_corpus(args.data, corpus, args.pixel_dir, args.output_dir))
    results = dict(split_representation_fits=len(rows), inner_scores_compared=sum(r["inner_scores"] for r in rows),
                   predictions_compared=sum(r["predictions_compared"] for r in rows), splits=rows)
    results["different_test_labels"] = sum(sum(r["different_test_labels"].values()) for r in rows)
    results["source"] = "released descriptors" if args.pixel_dir is None else "freshly extracted descriptors"
    if args.output:
        args.output.parent.mkdir(exist_ok=True, parents=True)
        args.output.write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
    print(results["inner_scores_compared"], "validation scores,", results["predictions_compared"],
          "test-label comparisons,", results["different_test_labels"], "label differences relative to the earlier r4 extraction", flush=True)


if __name__ == "__main__":
    main()
