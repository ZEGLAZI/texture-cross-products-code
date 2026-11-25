"""Compare the historical flower fits with freshly extracted source and target blocks."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BLOCKS = ("F7", "F8", "rgb")


def source_from_parts(parts, historical):
    """Verify identity and order before using newly extracted source descriptors."""
    with (historical / "external/data/tf_manifest.csv").open() as stream:
        names = [row["source_image"] for row in csv.DictReader(stream)]
    observed = []
    blocks = {name: [] for name in BLOCKS}
    for path in sorted(parts.glob("*.npz")):
        log = json.loads(path.with_suffix(".json").read_text())
        if log["failed"] or log["available"] != log["expected"]:
            raise ValueError(f"Incomplete extraction in {path.name}")
        with np.load(path) as chunk:
            observed.extend(chunk["ids"].tolist())
            for name in BLOCKS:
                blocks[name].append(chunk[name])
    if observed != names:
        raise ValueError("Source images differ from the TF manifest")
    result = {name: np.concatenate(chunks) for name, chunks in blocks.items()}
    with np.load(historical / "external/data/tf.npz") as ref:
        result["classe"] = ref["classe"]
    return result


def compare(r4, source_parts, target_path):
    historical = r4.resolve() / "historical"
    sys.path.insert(0, str(historical / "revision/code"))
    from flowers_new import classifiers
    from nested import macro

    cohort = historical / "revision/results/cohort"
    choices = json.loads((historical / "revision/results/selection/Flowers.json").read_text())
    with np.load(HERE / "features/source.npz") as supplied:
        source = {key: supplied[key] for key in supplied.files}
    if source_parts is not None:
        source = source_from_parts(source_parts, historical)
    with np.load(target_path) as target, np.load(cohort / "predictions.npz") as reference:
        np.testing.assert_array_equal(target["photo"], reference["photo"])
        y = (target["classe"][:, None] == [10, 12, 9]).argmax(1)
        np.testing.assert_array_equal(y, reference["y"])
        with threadpool_limits(1):
            predictions = classifiers(historical, source, target, choices)
        result = {}
        for key, labels in predictions.items():
            if key.endswith("_scores"):
                continue
            result[key] = {
                "original_recall_pct": macro(reference[key], y),
                "pixel_recall_pct": macro(labels, y),
                "different_labels": int(np.count_nonzero(labels != reference[key])),
            }
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("r4", type=Path)
    parser.add_argument("--source-parts", type=Path)
    parser.add_argument("--target", type=Path, default=HERE / "features/target.npz")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    actual = compare(args.r4, args.source_parts, args.target)
    if args.report:
        args.report.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
    if actual != json.loads((HERE / "reference.json").read_text()):
        raise ValueError("The flower source/target sensitivity differs from the reference")
    print("Verified all 12 flower classifier comparisons.")
