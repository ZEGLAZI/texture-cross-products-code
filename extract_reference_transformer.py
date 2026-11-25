"""Recompute the published DINOv2 vectors with the reference CPU arithmetic."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile

import numpy as np
from pixel_files import array_digest

HERE = Path(__file__).resolve().parent
CPU_ENV = {
    "ATEN_CPU_CAPABILITY": "avx2",
    "ONEDNN_MAX_CPU_ISA": "AVX2",
    "DNNL_MAX_CPU_ISA": "AVX2",
    "MKL_ENABLE_INSTRUCTIONS": "AVX2",
    "MKL_CBWR": "AVX2",
}


def copy_atomic(source, target):
    with NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        tmp = Path(handle.name)
    try:
        shutil.copyfile(source, tmp)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)


def extract(corpus, data, images, source, weights, output):
    output.mkdir(exist_ok=True, parents=True)
    # Keep cached chunks separate from extractions with native CPU arithmetic.
    work = output / "dinov2_reference_avx2" / corpus
    work.mkdir(exist_ok=True, parents=True)
    settings = {"cpu_environment": CPU_ENV, "threads": 3, "batch": 12}
    config = work / "cpu_configuration.json"
    if config.is_file() and json.loads(config.read_text()) != settings:
        raise ValueError(f"An incompatible DINOv2 extraction exists in {work}")
    config.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    program = data / "cross_representation/code/extract_features.py"
    subprocess.run([sys.executable, "-B", str(program), corpus, str(data), str(images),
                    str(source), str(weights), str(work), "--threads", "3", "--batch", "12"],
                   check=True, env={**os.environ, **CPU_ENV})
    expected = json.loads((HERE / "pixel_check" / (corpus+"_transformer_hashes.json")).read_text())
    filename = corpus + "_features.npz"
    with np.load(work / filename) as fresh:
        observed = {key: array_digest(fresh[key]) for key in fresh.files}
    if observed != expected:
        raise ValueError(f"{corpus} DINOv2 coordinates differ from the reference")
    for name in (filename, corpus+"_extraction.json"):
        copy_atomic(work / name, output / name)
    (output / (corpus+"_transformer_cpu.json")).write_text(
        json.dumps({**settings, "array_sha256": observed}, indent=2) + "\n", encoding="utf-8")
    print(corpus, "DINOv2 reference vectors verified exactly.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", choices=("DTD", "KTH"))
    for name in ("data", "images", "source", "weights", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    extract(args.corpus, args.data, args.images, args.source, args.weights, args.output)
