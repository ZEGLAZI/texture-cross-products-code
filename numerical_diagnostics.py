"""Independent numerical diagnostics for splits, kernels and stored decisions."""
from pathlib import Path

import numpy as np

from numerical_checks import columns, csv_rows, read_json


def check_diagnostics(directory: Path, data: Path, out: Path, mode: str):
    report = {}
    fits = out / "scale_fits" if mode == "full" else data / "cross_representation/scale_results"
    diagnostics = [r for ds in ("DTD", "KTH") for r in csv_rows(fits / (ds + "_diagonals.csv"))]
    if len(diagnostics) != 28:
        raise ValueError("Expected 28 representation-by-split diagonal checks.")
    for row in diagnostics:
        if any(int(v) for k, v in row.items() if "zero_block" in k or k == "label_differences_at_theoretical_penalty"):
            raise ValueError("A zero feature block or a prediction mismatch was found.")
        np.testing.assert_allclose(
            columns(row, ("S_mean_train_diagonal", "Q_mean_train_diagonal", "factor", "equivalent_penalty")),
            [9/13, 11/13, 9/11, 11/9], rtol=0, atol=2e-14)
    report["diagonal_checks"] = len(diagnostics)

    pixel = read_json(directory / "pixel_check/DTD_refit.json")
    switch = [row for row in pixel["splits"] if row["selections"] != row["original_selections"]]
    if len(switch) != 1 or switch[0]["split"] != "DTD_06" or switch[0]["representation"] != "H":
        raise ValueError("The reference pixel-extraction penalty switch differs.")
    with np.load(data / "cross_representation/results/DTD_predictions.npz") as earlier, \
            np.load(directory / "pixel_check/DTD_predictions.npz") as current:
        for field in ("ids", "labels", "test"):
            np.testing.assert_array_equal(earlier[field], current[field])
        y = earlier["labels"]
        gains = []
        for records, square, full in [(earlier, "H_p0_S", "H_p0_Q"),
                                      (earlier, "H_S", "H_Q"), (current, "H_S", "H_Q")]:
            improvement = (records[full] == y).astype(int) - (records[square] == y).astype(int)
            gains.append(float(100 * improvement.mean()))
        changed_selected = int(np.count_nonzero(earlier["H_tuned_S"][5] != current["H_tuned_S"][5]))
        changed_fixed = int(np.count_nonzero(earlier["H_S"][5] != current["H_S"][5]))
    if changed_selected != switch[0]["different_test_labels"]["H_tuned_S"]:
        raise ValueError("DTD split 6: paired predictions disagree with the refit record.")
    report["weight_and_extraction_gains_pp"] = gains
    report["split_6_changed_counts"] = {"selected_penalty": changed_selected, "fixed_penalty": changed_fixed}

    for dataset in ("DTD", "KTH"):
        counts = []
        with np.load(directory / "pixel_check" / f"{dataset}_predictions.npz") as arrays:
            y = arrays["labels"]
            for family in ("H", "D"):
                square = arrays[f"{family}_S"] == y
                full = arrays[f"{family}_Q"] == y
                counts.extend([int(np.count_nonzero(full & ~square)), int(np.count_nonzero(square & ~full))])
        report[f"{dataset}_corrected_introduced"] = counts

    with np.load(data / "historical/dtd/results/decisions.npz") as historical, \
            np.load(data / "historical/dtd/results/identites.npz") as identities:
        truth = identities["y"][historical["tests"]]
        additive = historical["Additif_fixe"] == truth
        restricted = historical["HSMT"] == truth
        changes = np.array([(restricted[truth == k].sum() - additive[truth == k].sum())
                            for k in np.unique(truth)])
        corrected = int(np.count_nonzero(restricted & ~additive))
        lost = int(np.count_nonzero(additive & ~restricted))
        report["historical_restricted_decisions"] = {
            "corrected": corrected, "new_errors": lost, "net_corrected": corrected - lost,
            "attributes_improved": int((changes > 0).sum()),
            "attributes_declined": int((changes < 0).sum()), "attributes": int(len(changes))}

    with np.load(data / "review2/extra/fmd.npz", allow_pickle=False) as archive:
        if set(archive["train"]) & set(archive["test"]):
            raise ValueError("Overlapping FMD training and test indices.")
        report["FMD_split"] = {"train": len(archive["train"]), "test": len(archive["test"]),
                               "overlap": 0}

    from numerical_figure_checks import check_figure_data
    report["paired_result_checks"] = check_figure_data(data, out, mode)
    return report
