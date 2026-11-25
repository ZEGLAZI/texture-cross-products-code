"""Compare fixed DTD models and validation-selected references with the per-image release."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(reference, measured):
    fields = {}
    with np.load(reference, allow_pickle=False) as ref, np.load(measured, allow_pickle=False) as new:
        if set(ref.files) != set(new.files):
            raise ValueError(f"Array names differ in {measured}")
        for key in ref.files:
            a, b = new[key], ref[key]
            if a.dtype.kind == "f":
                np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-9, err_msg=key)
            else:
                np.testing.assert_array_equal(a, b, err_msg=key)
            fields[key] = a.size
    return fields


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("r4", "fixed", "selected"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--report", type=Path)
    a = parser.parse_args()
    historic = a.r4 / "historical"
    n = compare(historic/"dtd/results/decisions.npz", a.fixed/"decisions.npz")
    fixed = sum(v for k,v in n.items() if k not in ("tests","n_train"))
    labels = validation = 0
    reference = historic/"revision/results/references"
    for i in range(1, 11):
        name = f"fold_{i:02}.npz"
        n = compare(reference/name, a.selected/name)
        labels += sum(v for k,v in n.items() if k.endswith(("_fixed","_selected")))
        validation += sum(v for k,v in n.items() if k.endswith("_val"))
    if json.loads((reference/"scores.json").read_text()) != json.loads((a.selected/"scores.json").read_text()):
        raise ValueError("The DTD reference summaries differ")
    result = {"fixed_model_labels": fixed, "reference_labels_at_two_penalty_rules": labels,
              "inner_validation_scores": validation, "folds": 10}
    if a.report:
        a.report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(result)
