"""Reproduce texture-recognition experiments from code and numerical data.

tables: recalculate summaries and consistency checks from archived decisions.
full: additionally repeat classifier fits and the complete dense DTD study.
No manuscript, supplement, or publication source is required or checked.
Each invocation writes a new run directory; full never reuses dense fits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from numerical_release import verify_release

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tables", "full"), default="tables")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path, help="Parent directory for this invocation's new run directory")
    args = parser.parse_args()
    if args.jobs < 1 or args.threads < 1:
        parser.error("jobs and threads must be positive")
    if not (ROOT / "additional_features/dino_r50/DTD_stages.npz").is_file():
        parser.error("The numerical data are incomplete: additional_features/dino_r50/DTD_stages.npz is missing")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parent = args.output.resolve() if args.output else ROOT / "runs"
    run = parent / f"numerical-{args.mode}-{stamp}-{uuid.uuid4().hex[:6]}"
    run.mkdir(parents=True)
    print(f"Run directory: {run}", flush=True)
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads))
    env.pop("PYTHONOPTIMIZE", None)

    def execute(label: str, script: Path, *options):
        print(label, flush=True)
        log_path = run / (label + ".log")
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-B", str(script), *map(str, options)],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{label} failed; inspect {log_path}")

    release = verify_release(ROOT)
    historical_report = run / "historical.json"
    execute("historical", ROOT / "replay_submission.py", "--mode", args.mode,
            "--jobs", args.jobs, "--report", historical_report)

    ridge = ROOT / "studies/ridge_sensitivity"
    ridge_output = run / "ridge"
    if args.mode == "tables":
        shutil.copytree(ridge / "results", ridge_output,
                        ignore=shutil.ignore_patterns("*.log", "*.png", "*.pdf", "__pycache__"))
    else:
        ridge_output.mkdir()
    ridge_report = run / "ridge.json"
    options = ["--mode", args.mode, "--root", ROOT, "--output", ridge_output,
               "--threads", args.threads, "--report", ridge_report]
    if args.mode == "full":
        options.append("--force")
    execute("ridge", ridge / "replay.py", *options)
    execute("ridge_export", ridge / "export_checks.py", "--input", ridge_output / "audit_v3.json",
            "--output", ridge_output)
    provenance_report = run / "provenance.json"
    execute("provenance", ROOT / "verify_provenance.py", "--package", ROOT, "--report", provenance_report)

    paths = {"historical": historical_report, "ridge": ridge_report, "provenance": provenance_report}
    sections = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    if any(section.get("status") != "passed" for section in sections.values()):
        raise ValueError("A required numerical verification stage did not pass")
    report = {
        "status": "passed", "mode": args.mode,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "performed_classifier_refits_in_this_invocation": args.mode == "full",
        "scope": "Refits from released descriptors" if args.mode == "full" else "Summaries from stored decisions",
        "fresh_pixel_extraction_in_this_invocation": False,
        "publication_documents_checked": False,
        "numerical_release": release,
        "historical_output_files_compared": sections["historical"]["historical"]["numerical_outputs_checked"],
        "ridge_numeric_reference_values_checked": sections["ridge"]["numeric_values_checked"],
        "ridge_numeric_mismatches": sections["ridge"]["numeric_mismatches"],
        "ridge_evidence_counts": sections["ridge"]["counts"],
        "ridge_evidence_counts_origin": "freshly recomputed" if args.mode == "full" else "archived fitting records",
        "provenance_rows_checked": sections["provenance"]["provenance_rows_checked"],
        "deterministic_examples_checked": sections["provenance"]["s42"]["rows_checked"],
        "stage_reports": {name: str(path.relative_to(run)) for name, path in paths.items()},
        "run_directory": str(run),
    }
    destination = args.report.resolve() if args.report else run / "verification.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"PASS: numerical reproduction ({args.mode}); {destination}")


if __name__ == "__main__":
    main()
