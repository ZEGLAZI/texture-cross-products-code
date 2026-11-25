"""Recreate the frozen CNN analysis from distributed features and aligned decisions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from threadpoolctl import threadpool_limits
from cnn_models import load_npz, fit_task, METHODS, STAGES
from paired_results import tables, DATASETS
from independent import run as independent_fit


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare_csv(reference, actual):
    with reference.open(newline="") as expected, actual.open(newline="") as rebuilt:
        a, b = list(csv.reader(expected)), list(csv.reader(rebuilt))
    if len(a) != len(b) or a[0] != b[0]:
        raise ValueError(f"CSV row or column count changed in {reference.name}")
    for ar, br in zip(a[1:], b[1:], strict=True):
        for left, right in zip(ar, br, strict=True):
            if left == right:
                continue
            np.testing.assert_allclose(float(left),float(right),rtol=0,atol=2e-10,
                                       err_msg=reference.name)
    return len(a)-1


def verify(root, folder, mode, jobs):
    expected = folder/"results"
    checks = {"mode":mode, "pooled_coordinates":0, "image_identities":0,
              "validation_scores":0, "penalty_choices":0,
              "refitted_test_labels":0, "independent_solver_test_labels":0}
    for dataset in ("FMD","Lytro","DTD","KTH"):
        report = read_json(folder/"data"/(dataset+"_stages.json"))
        features = load_npz(folder/"data"/(dataset+"_stages.npz"))
        hashes = [hashlib.sha256(features[key].tobytes()).hexdigest() for key in STAGES]
        if hashes != report["pooled_arrays_sha256"]:
            raise ValueError("Feature checksum mismatch for "+dataset)
        count = len(features["ids"])
        with np.load(root/"studies/CVIU_Experience_couleur/cache"/(dataset+"_palettes.npz")) as manifest:
            for key in ("ids","hash224"):
                np.testing.assert_array_equal(features[key],manifest[key])
        if [features[k].shape for k in STAGES] != [(count,d) for d in (512,1024,2048)]:
            raise ValueError("Unexpected pooled dimensions for "+dataset)
        checks["pooled_coordinates"] += sum(features[k].size for k in STAGES)
        checks["image_identities"] += count
    with tempfile.TemporaryDirectory(prefix="resnet-replay-") as tmp:
        run = Path(tmp)
        if mode == "full":
            for dataset in DATASETS:
                fit_task(root, folder/"data", dataset, run)
                reference = load_npz(expected/(dataset+"_decisions.npz"))
                fresh = load_npz(run/(dataset+"_decisions.npz"))
                if set(fresh) != set(reference):
                    raise ValueError("Missing predictions or identity arrays")
                for key in reference:
                    np.testing.assert_array_equal(reference[key],fresh[key],err_msg=dataset+" "+key)
                checks["refitted_test_labels"] += sum(fresh[key].size for key in METHODS)
                for suffix, count in (("settings", "penalty_choices"), ("validation","validation_scores")):
                    filename=dataset+"_"+suffix+".csv"
                    rows = compare_csv(expected/filename, run/filename)
                    checks[count] += rows*(2 if suffix=="settings" else 1)
                checks["independent_solver_test_labels"] += independent_fit(root,folder,dataset)
        tables(root,run if mode=="full" else expected,run)
        for table in ("gains","contrasts","per_fold"):
            checks[table+"_rows"]=compare_csv(expected/(table+".csv"),run/(table+".csv"))
        if mode=="full":
            subprocess.run([sys.executable,"-B",str(folder/"refit_transfer.py"),str(root),str(folder/"data"),
                            str(run),str(run),"--jobs",str(jobs)],check=True)
            checks["source_target_paired_draws"]=compare_csv(expected/"resnet_bootstrap_draws.csv",run/"resnet_bootstrap_draws.csv")
            if read_json(expected/"resnet_bootstrap.json")!=read_json(run/"resnet_bootstrap.json"):
                raise ValueError("The full source/target bootstrap differs from the released result.")
    checks["status"]="passed"
    return checks


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    folder=Path(__file__).resolve().parent
    parser.add_argument("--root",type=Path,default=folder.parent.parent)
    parser.add_argument("--mode",choices=("tables","full"),default="full")
    parser.add_argument("--jobs",type=int,default=2)
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    if min(args.jobs,args.threads)<1: parser.error("Use at least one worker and one CPU thread.")
    with threadpool_limits(args.threads):
        result=verify(args.root.resolve(),folder,args.mode,args.jobs)
    output=json.dumps(result,indent=2,allow_nan=False)+"\n"
    if args.report: args.report.write_text(output,encoding="utf-8")
    print(output)
