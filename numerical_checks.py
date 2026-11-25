"""Recalculate summaries and independently check released numerical evidence.

Table labels retain the numerical reference identifiers. No manuscript,
supplement, typeset table, or publication source is read by this module.
"""
import csv
import json
from pathlib import Path
import numpy as np


EXPECTED_CELLS = {"2": 36, "4": 49, "5": 8, "6": 21, "7": 24,
                  "8": 30, "9": 14, "S3": 24, "S4": 56, "S16": 43}
EXPECTED_CELLS.update({"S1":40,"S2":14,"S5":28,"S6":36,"S7":63,"S8":30,"S9":35,
                       "S18":12,"S10":100,"S11":30,"S12":19,"S13":24,"S14":12,"S15":24})
EXPECTED_CELLS["S19"] = 32
DATASET_LABELS = {"DTD": "DTD", "KTH": "KTH-TIPS2-b"}


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def columns(row, keys):
    return [float(row[k]) for k in keys]


class SummaryRows:
    """Collect calculated numerical summaries; this does not check any document.

    The numerical functions below retain their independent consistency checks.
    A recorded value is output evidence, not a successful PDF comparison.
    """
    def __init__(self):
        self.counts = {}
        self.rows = []

    def record(self, table, label, values, count=None):
        values = [float(value) for value in values]
        if count is not None:
            values.append(int(count))
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite numerical summary: {table}/{label}")
        self.counts[table] = self.counts.get(table, 0) + len(values)
        self.rows.append({"table": table, "row": label, "values": values})

    def penalties(self, label, values):
        self.record("S4", label, values)


def paired_macro_gain(truth, base, full):
    """Average the paired change in recall across classes within each outer split."""
    fold_means = []
    for labels, base_labels, full_labels in zip(truth, base, full, strict=True):
        delta = (full_labels == labels).astype(float) - (base_labels == labels).astype(float)
        by_class = [delta[labels == c].mean() for c in np.unique(labels)]
        fold_means.append(np.mean(by_class))
    return float(100 * np.mean(fold_means))


def check_representation_results(tables, data, out, mode):
    native_rows = {(v["dataset"], v["penalty"]): v
                   for v in csv_rows(out / "cross_tables/representation_comparison.csv")}
    matched_rows = {v["dataset"]: v for v in csv_rows(out / "scale_tables/scale_comparison.csv")}
    shifts = read_json(out / "cross_tables/regime_change.json")
    native_path = out/"primary_fits"
    matched_path = out/"primary_fits"
    primary = []
    for dataset, label in DATASET_LABELS.items():
        with (np.load(native_path / f"{dataset}_predictions.npz", allow_pickle=False) as native,
              np.load(matched_path / f"{dataset}_matched_predictions.npz", allow_pickle=False) as matched):
            for field in ("test", "ids", "labels", "groups"):
                np.testing.assert_array_equal(native[field], matched[field], err_msg=f"{dataset}: {field}")
            labels = native["labels"]
            contrasts = {}
            for setting, predictions, square_key, full_key in (
                ("Native", native, "_S", "_Q"),
                ("Matched", matched, "_S", "_matched_Q"),
                ("Selected", native, "_tuned_S", "_tuned_Q"),
            ):
                gains = [paired_macro_gain(labels, predictions[f + square_key], predictions[f + full_key])
                         for f in ("H", "D")]
                contrast = gains[1] - gains[0]
                contrasts[setting] = contrast
                if setting == "Matched":
                    row = matched_rows[dataset]
                    estimates = columns(row, ("H_gain_pp", "D_gain_pp", "matched_D_pp"))
                    limits = columns(row, ("matched_D_simultaneous_low", "matched_D_simultaneous_high"))
                else:
                    row = native_rows[dataset, {"Native": "fixed", "Selected": "selected"}[setting]]
                    estimates = columns(row, ("H_gain", "D_gain", "contrast"))
                    limits = columns(row, ("contrast_lo", "contrast_hi"))
                np.testing.assert_allclose([*gains, contrast], estimates, rtol=0, atol=1e-10,
                                           err_msg=f"{label}, {setting}: direct paired class means")
                conv = Path(__file__).resolve().parent / "studies/resnet50/results" / (dataset+"_decisions.npz")
                with np.load(conv, allow_pickle=False) as cnn:
                    for field in ("test","ids","labels","groups"):
                        np.testing.assert_array_equal(cnn[field],native[field])
                    keys = {"Native":("S","Q"), "Matched":("S","matched_Q"), "Selected":("selected_S","selected_Q")}
                    rs,rq = keys[setting]
                    cnn_gain = paired_macro_gain(labels,cnn[rs],cnn[rq])
                tables.record("2", label + " " + setting, [*gains, cnn_gain, contrast, *limits])
                primary.append(dict(dataset=label, setting=setting, test_appearances=int(labels.size),
                                    colour_texture_gain_pp=gains[0], dinov2_gain_pp=gains[1], resnet50_gain_pp=cnn_gain,
                                    dinov2_minus_colour_pp=contrast))
            shift = contrasts["Selected"] - contrasts["Native"]
            np.testing.assert_allclose(shift, shifts[dataset]["T_pp"], rtol=0, atol=1e-10,
                                       err_msg=f"{label}: selected-minus-fixed representation contrast")
        for key, name in (("H", "Colour–texture"), ("D", "DINOv2")):
            fields = (key + "_" + s for s in ("S_recall_pct", "Q_recall_pct", "matched_Q_recall_pct",
                                               "gain_pp", "pointwise_low", "pointwise_high"))
            tables.record("S3", label + " " + name, columns(matched_rows[dataset], fields))
    for row in csv_rows(out / "scale_tables/selected_penalties.csv"):
        label = DATASET_LABELS[row["dataset"]] + " " + row["fold"].split("_")[-1]
        tables.penalties(label, columns(row, ("H_S", "H_Q", "D_S", "D_Q")))
    return primary


def check_colour_texture_tables(tables, out):
    for row in csv_rows(out / "paper/aggregate/primary.csv"):
        if row["profile"] == "0" and row["mode"] == "fixed":
            keys = ("Q", "Q_minus_S", "Q_minus_S_lo", "Q_minus_S_hi", "Q_minus_X", "Q_minus_X_lo", "Q_minus_X_hi")
            tables.record("4", row["corpus"].replace("2015-17", "2015–17"), columns(row, keys))
    for row in csv_rows(out / "paper/aggregate/redistributed_rbf.csv"):
        if row["corpus"] in ("Clothing", "DTD"):
            tables.record("5", row["corpus"], columns(row, ("RS", "RX", "Q", "RBF")))
    clothing = {"additive": "Linear", "crosses": "Redistributed cross-products", "squares": "Redistributed squares",
                "clbp": "Joint grayscale CLBP", "dino": "Frozen DINOv2", "dino_additive": "DINOv2 + linear",
                "dino_squares": "DINOv2 + redistributed squares"}
    for row in csv_rows(out / "paper/clothing_scores.csv"):
        tables.record("6", clothing[row["model"]], columns(row, ("mean", "lowest", "highest")))
    long_names = {"iNat2015-17": "iNaturalist 2015–17", "iNat2026": "iNaturalist 2026"}
    for row in csv_rows(out / "paper/selected_restricted_scores.csv"):
        name = long_names.get(row["corpus"], row["corpus"])
        tables.record("8", name, columns(row, ("Add", "W", "S", "C", "U", "Q")))
    interventions = {"grouped_half": "Full signed block", "no_shape": "Remove scatter shape",
                     "no_scale": "Remove log scale", "no_radial": "Remove radial index", "diag": "Diagonal shape only",
                     "upstream_diagonal": "Upstream diagonal", "expanded_grid": "Expanded radial grid",
                     "log_ratio": "Log moment ratio", "boundary_indicator": "Lower-limit indicator"}
    for row in csv_rows(out / "paper/wavelet_interventions.csv"):
        keys = ["recall_pct", "difference_pp"]
        if row["intervention"] != "grouped_half":
            keys += ["pointwise_lo95", "pointwise_hi95"]
        tables.record("S16", interventions[row["intervention"]], columns(row, keys), count=int(row["changed"]))


def check_historical_tables(tables, archive):
    base = read_json(archive / "dtd/results/bilan.json")["methodes"]
    single = read_json(archive / "extra/dense.json")["scores"]
    contrasts = {v["comparaison"]: v for v in csv_rows(archive / "dtd/results/contrastes.csv")}
    for key, label in (("structure", "Achromatic structure, S"), ("wavelet", "Opponent wavelets, W"),
                       ("colour", "RGB palette, C")):
        tables.record("7", label, [single[key]])
    for key, label in (("Additif_historique", "Additive KA,2"), ("Additif_fixe", "Additive K∗A"),
                       ("Additif_diag_egal", "Additive, matched diagonal"), ("Pivot_F7", "Structure centre K∗S"),
                       ("Pivot_RGB", "Colour centre K∗C")):
        diff = contrasts["HSMT - " + key]
        tables.record("7", label, [base[key]["rappel"], *columns(diff, ("ecart", "ic_bas", "ic_haut"))])
    tables.record("7", "Opponent-wavelet centre KW", [base["HSMT"]["rappel"]])
    references = read_json(archive / "revision/results/references/scores.json")
    labels = {"clbp": "Grayscale completed LBP", "oc": "Opponent-colour LBP", "cov": "Regional log covariance",
              "crop": "DINOv2, centre crop", "full": "DINOv2, full field", "full_add": "Full-field DINOv2 + additive",
              "full_W": "Full-field DINOv2 + two products"}
    for key, label in labels.items():
        tables.record("9", label, columns(references[key], ("fixed", "selected")))


def check_numerical_results(root: Path, data: Path, out: Path, mode: str):
    tables = SummaryRows()
    primary = check_representation_results(tables, data, out, mode)
    check_colour_texture_tables(tables, out)
    check_historical_tables(tables, data / "historical")
    from supplementary_checks import check_supplementary_tables
    check_supplementary_tables(tables, data, out, mode)
    from summarize_pixel_check import summarize
    from threadpoolctl import threadpool_limits
    pixel_folder = root / "pixel_check"
    measured = json.loads((pixel_folder / "summary.json").read_text(encoding="utf-8"))
    with threadpool_limits(limits=1):
        for corpus, label in DATASET_LABELS.items():
            audit = summarize(corpus, data, pixel_folder, pixel_folder)
            reference = measured[corpus]
            for branch in ("new_pixels", "released"):
                for setting in ("native", "matched", "selected", "shift"):
                    for field, value in audit[branch][setting].items():
                        np.testing.assert_allclose(value, reference[branch][setting][field], rtol=0., atol=1.e-10)
            for field in ("test_label_changes","colour_extraction","transformer_blocks","selected_penalties_changed"):
                if audit[field] != reference[field]:
                    raise ValueError(f"{corpus}: discrepancy in the pixel check for {field}")
            results = audit["new_pixels"]
            values = [results[s]["colour_gain_pp"] for s in ("native","matched","selected")]
            tables.record("S18", label, [*values, results["shift"]["estimate_pp"], *results["shift"]["adjusted_interval_pp"]])
    subsets = read_json(out / "block_subsets/summary.json")
    for subset in ("S", "W", "C", "SW", "SC", "WC", "SWC"):
        cells = []
        for corpus in DATASET_LABELS:
            cells.append(subsets[corpus]["macro_recall_pct"][subset])
            if len(subset) == 2:
                cells.extend(subsets[corpus]["full_minus_pair_pp"][subset])
        tables.record("S19", " + ".join(subset), cells)
    if tables.counts != EXPECTED_CELLS:
        raise ValueError(f"Incomplete numerical summary coverage: {tables.counts}")
    from numerical_diagnostics import check_diagnostics
    diagnostics = check_diagnostics(root, data, out, mode)
    return {"numerical_diagnostics": diagnostics, "calculated_values_by_table": tables.counts, "representation_gains_from_predictions": primary,
            "calculated_summary_rows": tables.rows, "publication_documents_checked": False}
