"""Rebuild the CLBP and opponent-LBP KTH baselines for the held-out specimens."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent


def check(r4, image_root):
    historical = r4.resolve() / "historical"
    sys.path.insert(0, str(historical / "revision/code"))
    from nested import fit, macro

    with (historical / "external/data/kth_images.csv").open() as file:
        manifest = list(csv.DictReader(file))
    with (historical / "external/data/kth_readouts.csv").open() as file:
        outcomes = list(csv.DictReader(file))
    ids = [row["chemin"] for row in manifest]
    hashes = [row["sha_pixels"] for row in manifest]
    # The prediction CSV is grouped by test fold; the feature file is grouped by material.
    by_path = {row["chemin"]: row for row in outcomes}
    if len(by_path) != len(outcomes) or len(set(ids)) != len(ids) or set(ids) != set(by_path):
        raise ValueError("KTH identities differ between features and predictions")
    outcomes = [by_path[name] for name in ids]
    np.testing.assert_array_equal([r["verite"] for r in outcomes], [r["materiau"] for r in manifest])
    np.testing.assert_array_equal([r["lettre_du_pli"] for r in outcomes], [r["specimen"] for r in manifest])
    classes, y = np.unique([row["materiau"] for row in manifest], return_inverse=True)
    groups = np.array([row["specimen"] for row in manifest])
    with np.load(HERE / "temoins.npz") as archived:
        np.testing.assert_array_equal(archived["path"], ids)
        np.testing.assert_array_equal(archived["sha_pixels"], hashes)
        counts = {key: archived[key] for key in ("clbp", "oclbp")}
    if image_root:
        from baselines import features
        for i, (name, expected) in enumerate(zip(ids, hashes, strict=True)):
            with Image.open(image_root / name) as raw:
                image = raw.convert("RGB")
            actual = hashlib.sha256(np.array(image.size, dtype="<u4").tobytes() + image.tobytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"The image fingerprint differs for {name}")
            for kind, vector in zip(("clbp", "oclbp"), features(image), strict=True):
                np.testing.assert_array_equal(vector, counts[kind][i], err_msg=name)
            if (i+1) % 600 == 0:
                print(i+1, "verified pixel extractions", flush=True)
    results = {}
    for kind, values in counts.items():
        a = values.astype(np.float64)
        if kind == "clbp":
            features = a / a.sum(1, keepdims=True)
        else:
            parts = a.reshape(len(a), 12, 36)
            features = (parts / parts.sum(2, keepdims=True) / 12).reshape(len(a), -1)
        vectors = np.sqrt(features)
        gram = vectors @ vectors.T
        predicted = np.full(len(y), -1)
        split_recalls = []
        for g in "abcd":
            test = np.flatnonzero(groups == g)
            train = np.flatnonzero((groups != g) & ~np.isin(hashes, np.asarray(hashes)[test]))
            ordered = np.r_[train, test]
            scores = fit(gram[np.ix_(ordered, train)], len(train), y[train], 1, len(classes))
            predicted[test] = scores.argmax(1)
            split_recalls.append(macro(predicted[test], y[test]))
        np.testing.assert_array_equal(classes[predicted], [row[kind+"_ridge"] for row in outcomes])
        results[kind] = {"matched_labels": len(y), "mean_recall_pct": float(np.mean(split_recalls))}
        del gram
    return {"images_verified_from_pixels": len(y) if image_root else 0, "kernels": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("r4", type=Path)
    parser.add_argument("--image-root", type=Path, help="Directory containing KTH-TIPS2-b")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        result = check(args.r4, args.image_root)
    text = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
