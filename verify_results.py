"""Run historical numerical replay and verify the released code/data evidence."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
ARCHIVE = "data_and_code_r4.zip"
ARCHIVE_SHA256 = "ad1a3c2cda703498ca2bca2e2f5fa6718ad071931a0d5dbd8869c4f46dfbf385"
MANIFEST_SHA256 = "7fa27470dab59573e0933bf1ade04cea903c04483756e9ab5744022350c03283"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def numerical_root(stack):
    directory = ROOT / "reproduction"
    if not directory.is_dir():
        archive = ROOT / ARCHIVE
        if not archive.is_file():
            raise FileNotFoundError(f"Place {ARCHIVE} next to this script or provide the extracted reproduction/ directory.")
        if digest(archive) != ARCHIVE_SHA256:
            raise ValueError("The numerical ZIP differs from the documented r4 release.")
        directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="hsmt-r4-")))
        with ZipFile(archive) as bundle:
            bundle.extractall(directory)
    if digest(directory / "SHA256SUMS") != MANIFEST_SHA256:
        raise ValueError("The numerical manifest differs from the r4 reference.")
    from numerical_release import verify_release
    verify_release(ROOT)
    return directory


def run(mode):
    from numerical_checks import check_numerical_results

    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix=mode + "-", dir=runs))
    print(f"Results will be written to {destination}", flush=True)
    with ExitStack() as stack:
        data = numerical_root(stack)
        env = os.environ.copy()
        env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", OMP_NUM_THREADS="2")
        env.pop("PYTHONOPTIMIZE", None)  # The original numerical checks include assertions.
        subprocess.run([sys.executable, "-B", str(data / "reproduce.py"), mode, "--output", str(destination)],
                       cwd=data, env=env, check=True)
        report = json.loads((destination / "replay_report.json").read_text(encoding="utf-8"))
        reference = json.loads((ROOT / "r4_reference.json").read_text(encoding="utf-8"))
        checked = [name.replace(chr(92), "/") for name in report["files_compared"]]
        expected = set(reference["files_compared"])
        expected_count = 72 if mode == "full" else 21
        if (report["status"] != "passed" or report["mode"] != mode or len(checked) != expected_count
                or len(set(checked)) != expected_count or not set(checked).issubset(expected)):
            raise ValueError("The replay did not cover the expected reference files.")
        if mode == "full" and set(checked) != expected:
            raise ValueError("The output list differs from the original full replay report.")
        from primary_tables import prepare
        pixels = ROOT / "pixel_check"
        if mode == "full":
            from check_reference import refit_and_verify
            pixels = refit_and_verify(data, pixels, pixels, destination / "reference_refits")
        prepare(data, pixels, destination)
        from block_subsets import prepare as prepare_subsets
        block_report = prepare_subsets(data, ROOT, destination, mode)
        paper_report = check_numerical_results(ROOT, data, destination, mode)
        paper_report["block_subset_replay"] = block_report
    paper_report.update(status="passed", replay_mode=mode, numerical_outputs_checked=len(checked),
                        historical_files_compared=checked)
    (destination / "numerical_check.json").write_text(json.dumps(paper_report, indent=2, allow_nan=False) + "\n",
                                                encoding="utf-8")
    print("\nTable 2. Recalculated macro-recall gains in percentage points")
    print(f"{'Dataset':<12} {'Setting':<10} {'Colour-texture':>14} {'DINOv2':>8} {'ResNet-50':>10} {'DINOv2–colour':>15}")
    for row in paper_report["representation_gains_from_predictions"]:
        print(f"{row['dataset']:<12} {row['setting']:<10} {row['colour_texture_gain_pp']:>+14.2f}"
              f" {row['dinov2_gain_pp']:>+8.2f} {row['resnet50_gain_pp']:>+10.2f} {row['dinov2_minus_colour_pp']:>+15.2f}")
    print(f"PASS | {len(checked)} replay outputs; {sum(paper_report['calculated_values_by_table'].values())} calculated summary values; no publication audit.")
    print(f"Detailed checks: {destination / 'numerical_check.json'}")
    return paper_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tables", "full"), default="full",
                        help="tables uses the archived decisions; full refits the classifiers included in the numerical release")
    parser.add_argument("--report",type=Path,help="Optional path to a second copy of the verification report")
    args = parser.parse_args()
    report = run(args.mode)
    if args.report is not None:
        args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
