"""Reproduce the manuscript tables or all numerical checks in Supplement S7."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent


def equal_value(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            equal_value(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            equal_value(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, (float, int)):
        return isinstance(actual, (float, int)) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-10)
    if isinstance(actual, str) and isinstance(expected, str) and actual != expected:
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-10)
        except ValueError:
            return False
    return actual == expected


def compare(actual, expected):
    if actual.suffix == ".npz":
        with np.load(actual) as a, np.load(expected) as b:
            assert set(a.files) == set(b.files), expected
            for field in a.files:
                if a[field].dtype.kind == "f":
                    np.testing.assert_allclose(a[field], b[field], atol=1e-10, rtol=1e-11)
                else:
                    np.testing.assert_array_equal(a[field], b[field])
        return
    if actual.suffix == ".json":
        match = equal_value(json.loads(actual.read_text()), json.loads(expected.read_text()))
    elif actual.suffix == ".csv":
        with actual.open(newline="", encoding="utf-8") as a, expected.open(newline="", encoding="utf-8") as b:
            match = equal_value(list(csv.reader(a)), list(csv.reader(b)))
    else:
        match = actual.read_bytes() == expected.read_bytes()
    if not match:
        raise ValueError(f"Replayed values differ from {expected.relative_to(ROOT)}")


def replay(mode, out):
    checked = []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

    def run(script, *args):
        subprocess.run([sys.executable, "-B", str(ROOT / script), *map(str, args)],
                       cwd=ROOT, env=env, check=True)

    def verify_files(folder, archived):
        files = sorted(p for p in (out / folder).iterdir() if p.is_file())
        if not files:
            raise ValueError(f"No output in {folder}")
        for f in files:
            compare(f, ROOT / archived / f.name)
            checked.append(str(f.relative_to(out)))

    run("verify.py")
    run("reader/code/paper_tables.py", out / "paper")
    verify_files("paper", "reader/tables")
    verify_files("paper/aggregate", "review2/results/aggregate")
    run("review2/code/decision_agreement.py", out / "agreement.csv")
    compare(out / "agreement.csv", ROOT / "review2/results/aggregate/decision_agreement.csv")
    checked.append("agreement.csv")
    host_results = ROOT / "reader/hosts"
    if mode == "full":
        run("review2/code/verify_primary_refits.py", out / "refits.json")
        compare(out / "refits.json", ROOT / "review2/results/provenance/primary_refits.json")
        checked.append("refits.json")
        run("review2/code/replay.py", "radial", "--output", out / "radial")
        verify_files("radial", "review2/results/radial")
        run("reader/code/verify_gaussian.py", out / "gaussian")
        verify_files("gaussian", "reader/tables")
        run("reader/code/cohort_tables.py", out / "cohort")
        verify_files("cohort", "reader/tables")
        run("reader/code/model_intervals.py", out / "intervals")
        verify_files("intervals", "reader/tables")
        run("clothing/code/summarize.py", "--output", out / "photograph_bootstrap")
        verify_files("photograph_bootstrap", "clothing/results")
        host_results = out / "hosts"
        run("reader/code/clothing_hosts.py", "prepare", host_results)
        run("reader/code/clothing_hosts.py", "fit", host_results)
        verify_files("hosts", "reader/hosts")
    run("reader/code/host_inference.py", host_results, out / "host_inference")
    verify_files("host_inference", "reader/host_inference")
    cross = ROOT / "cross_representation"
    predictions = cross / "results"
    if mode == "full":
        predictions = out / "cross_fits"
        predictions.mkdir()
        run("cross_representation/code/check_classical.py", ROOT, predictions / "HSMT_refits.json")
        for corpus in ("DTD", "KTH"):
            run("cross_representation/code/compare.py", corpus, ROOT,
                cross / "features" / (corpus + "_features.npz"), predictions)
        verify_files("cross_fits", "cross_representation/results")
    run("cross_representation/code/table_rows.py", predictions, out / "cross_tables")
    verify_files("cross_tables", "cross_representation/results")
    scaled_predictions = cross / "scale_results"
    if mode == "full":
        scaled_predictions = out / "scale_fits"
        for corpus in ("DTD", "KTH"):
            run("cross_representation/code/diagonal_match.py", corpus, scaled_predictions)
        verify_files("scale_fits", "cross_representation/scale_results")
    run("cross_representation/code/diagonal_tables.py", scaled_predictions, predictions, out / "scale_tables")
    verify_files("scale_tables", "cross_representation/scale_results")
    report = {"mode": mode, "status": "passed", "files_compared": checked,
              "input_level": "archived numerical inputs and predictions; no source photograph downloads"}
    (out / "replay_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(checked)} output files against the distribution.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tables", "full"))
    parser.add_argument("--output", type=Path, default=Path("reproduced"))
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A new directory for each run also prevents reuse of earlier radial fits.
    with tempfile.TemporaryDirectory(prefix="hsmt-replay-", dir=destination.parent) as temp:
        replay(args.mode, Path(temp))
        shutil.copytree(temp, destination, dirs_exist_ok=True)
    print(f"Results: {destination}")
