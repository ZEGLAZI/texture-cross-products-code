"""Count paired label disagreements on the ten official DTD test sets."""
import argparse
import csv
from pathlib import Path

import numpy as np


def count_disagreements(labels, squares, full):
    different = squares != full
    squares_correct = squares == labels
    full_correct = full == labels
    counts = {
        "test_appearances": labels.size,
        "different_labels": int(different.sum()),
        "only_squares_correct": int((squares_correct & ~full_correct).sum()),
        "only_full_correct": int((full_correct & ~squares_correct).sum()),
        "different_both_wrong": int((different & ~squares_correct & ~full_correct).sum()),
        "squares_correct": int(squares_correct.sum()),
        "full_correct": int(full_correct.sum()),
    }
    assert counts["different_labels"] == sum(
        counts[key] for key in ("only_squares_correct", "only_full_correct", "different_both_wrong")
    )
    return counts


def recount(root):
    rows = []
    with np.load(root / "historical/dtd/results/identites.npz") as identities:
        target_labels = identities["y"]
    for split in range(1, 11):
        name = f"DTD_{split:02d}.npz"
        with (
            np.load(root / "review2/results/kernels" / name) as fits,
            np.load(root / "historical/revision/results/degrees" / name) as historical_squares,
            np.load(root / "historical/revision/results/selection" / name) as historical_full,
        ):
            tests = fits["test"]
            y = target_labels[tests]
            np.testing.assert_array_equal(y, fits["labels"])
            for historical in (historical_squares, historical_full):
                np.testing.assert_array_equal(tests, historical["test"])
            selected = {}
            for model, historical_labels in (
                ("RS", historical_squares["squares"]), ("Q", historical_full["Q"])
            ):
                index = fits["modes"].tolist().index(model)
                grid = fits["grid"][index]
                profile, _ = np.unravel_index(np.argmax(grid), grid.shape)
                # The stored predictions at each profile use its validation-selected ridge.
                selected[model] = fits["tuned"][index, profile]
                np.testing.assert_array_equal(selected[model], historical_labels)
            rows.append({"split": split, **count_disagreements(y, selected["RS"], selected["Q"])})
    columns = [key for key in rows[0] if key != "split"]
    rows.append({"split": "all", **{key: sum(row[key] for row in rows) for key in columns}})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="destination CSV")
    args = parser.parse_args()
    rows = recount(Path(__file__).resolve().parents[2])
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(rows[-1])
