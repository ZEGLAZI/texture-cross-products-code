"""Replay historical numerical experiments from the released code and data."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from numerical_extensions import check_transfer_and_colour, check_resnet_diagnostics, check_matched_diagnostics
from numerical_release import verify_release

ROOT = Path(__file__).resolve().parent

def run(mode, jobs, report_path):
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    env.pop("PYTHONOPTIMIZE", None)
    (ROOT / "runs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="texture-numerical-") as temp:
        tmp = Path(temp)
        def execute(script, *args):
            subprocess.run([sys.executable, "-B", str(script), *map(str,args)],check=True,cwd=ROOT,env=env)
        core = tmp / "core.json"
        execute(ROOT / "verify_results.py", "--mode", mode, "--report", core)
        report = {"mode":mode, "historical":json.loads(core.read_text(encoding="utf-8")),
                  "transfer_and_colour_consistency":check_transfer_and_colour(ROOT)}
        cnn_log = tmp/"resnet50.json"
        execute(ROOT/"studies/resnet50/replay.py","--mode",mode,"--jobs",jobs,"--report",cnn_log)
        report["resnet50"] = json.loads(cnn_log.read_text())
        report["resnet50_consistency"] = check_resnet_diagnostics(ROOT)
        if mode == "full":
            from contextlib import ExitStack
            from verify_results import numerical_root
            from verify_legacy import run as verify_legacy
            with ExitStack() as resources:
                numerical = numerical_root(resources)
                report["legacy_refits_and_simulation"] = verify_legacy(numerical, tmp / "legacy.json")
                for name in ("kth_lbp", "flower_pixels"):
                    log = tmp / (name + ".json")
                    execute(ROOT / "studies" / name / "check.py", numerical, "--report", log)
                    report[name] = json.loads(log.read_text(encoding="utf-8"))
            studies = ROOT / "studies"
            transfer = studies / "CVIU_Experience_couleur"
            colour = studies / "CVIU_Dependances_chromatiques"
            for name, script, options in [
                ("transfer", transfer/"replay.py", ["--threads",2]),
                ("colour", colour/"replay.py", ["--threads",2,"--jobs",1])]:
                log = tmp / (name + ".json")
                execute(script,*options,"--log",log)
                report[name] = json.loads(log.read_text(encoding="utf-8"))
            refits = studies / "source_refit"
            execute(refits/"refit.py",transfer/"cache",transfer/"results",tmp/"refits","--jobs",jobs)
            expected = refits / "results"
            fresh = tmp / "refits"
            summary = json.loads((fresh / "bootstrap_summary.json").read_text())
            if summary != json.loads((expected / "bootstrap_summary.json").read_text()):
                raise ValueError("The bootstrap summary differs from the reference.")
            with (fresh / "bootstrap_draws.csv").open(newline="") as a, (expected / "bootstrap_draws.csv").open(newline="") as b:
                if list(csv.reader(a)) != list(csv.reader(b)):
                    raise ValueError("The 999 draw records differ from the reference.")
            report["source_refit"] = summary
        matched = ROOT / "studies/matched_pretraining"
        extension_report=tmp/"matched.json"
        execute(matched/"replay.py","--mode",mode,"--features",ROOT/"additional_features","--report",extension_report)
        report["matched_pretraining_and_garments"]=json.loads(extension_report.read_text())
        report["matched_numerical_consistency"] = check_matched_diagnostics(ROOT, matched)
        report["numerical_release_verification"] = verify_release(ROOT)
        report["publication_documents_checked"] = False
        report["performed_classifier_refits_in_this_invocation"] = mode == "full"
        report["status"] = "passed"
        destination = report_path
        if destination is None:
            destination = Path(tempfile.mkdtemp(prefix="numerical-",dir=ROOT/"runs"))/"verification.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        print("Passed. Detailed report:",destination)

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("tables","full"),default="full")
    p.add_argument("--jobs",type=int,default=1,help="Workers for the paired source refits")
    p.add_argument("--report",type=Path)
    args=p.parse_args()
    if args.jobs<1:
        p.error("--jobs must be positive")
    run(args.mode,args.jobs,args.report)
