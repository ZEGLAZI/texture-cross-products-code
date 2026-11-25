"""Verify reference descriptors and refitted predictions exactly, including all selected penalties."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from pixel_files import array_digest
from primary_tables import prepare


def compare_npz(actual, expected):
    with np.load(actual) as measured, np.load(expected) as reference:
        if set(measured.files) != set(reference.files):
            raise ValueError(f"{actual}: array names differ")
        for name in reference.files:
            a, b = measured[name], reference[name]
            if a.dtype != b.dtype or a.shape != b.shape or array_digest(a) != array_digest(b):
                raise ValueError(f"{actual}: {name} differs from {expected}")


def verify_features(inputs, reference, corpus, require_new_transformer):
    compare_npz(inputs / f"{corpus}_colour.npz", reference / f"{corpus}_colour.npz")
    if require_new_transformer:
        manifest = json.loads((reference/f"{corpus}_transformer_hashes.json").read_text())
        with np.load(inputs/f"{corpus}_features.npz") as z:
            if {key:array_digest(z[key]) for key in z.files} != manifest:
                raise ValueError(f"{corpus}: transformer features differ from the reference")


def refit_and_verify(data, reference, inputs, out):
    out.mkdir(parents=True,exist_ok=True)
    for corpus in ("DTD","KTH"):
        verify_features(inputs, reference, corpus, inputs.resolve() != reference.resolve())
        fit = out/f"{corpus}_refit.json"
        script = Path(__file__).with_name("independent_refit.py")
        subprocess.run([sys.executable,str(script),str(data),"--pixel-dir",str(inputs),
                        "--output-dir",str(out),"--corpus",corpus,"--output",str(fit)],check=True)
        compare_npz(out/f"{corpus}_predictions.npz",reference/f"{corpus}_predictions.npz")
        def settings(path):
            return [(r["split"],r["representation"],r["selections"]) for r in json.loads(path.read_text())["splits"]]
        if settings(fit) != settings(reference/fit.name):
            raise ValueError(f"{corpus}: selected penalties differ")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("inputs", type=Path, help="Freshly extracted DTD and KTH vectors")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    ref = Path(__file__).resolve().parent / "pixel_check"
    for corpus in ("DTD","KTH"):
        verify_features(args.inputs,ref,corpus,True)
    fits = refit_and_verify(args.data,ref,args.inputs,args.output/"fits")
    prepare(args.data,fits,args.output)
    print("PASS: both feature families, all 56 selected penalties and all 235520 test-label comparisons.")
