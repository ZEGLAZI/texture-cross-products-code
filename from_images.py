"""Rebuild the primary comparison from original photographs in a new directory."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys

from verify_results import numerical_root


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", type=Path, help="Contains dtd/images, kth, dinov2.zip and dino.pth")
    ap.add_argument("output", type=Path, help="New destination for computed features and results")
    args = ap.parse_args()
    source = args.inputs.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    def run(script, *params):
        subprocess.run([sys.executable,"-B",str(script),*map(str,params)],check=True)
    with ExitStack() as stack:
        data = numerical_root(stack)
        for corpus, images in (("DTD",source/"dtd/images"),("KTH",source/"kth")):
            run(here/"check_colour_pixels.py",corpus,data,images,out/"features","--threads",2)
            run(here/"extract_reference_transformer.py",corpus,data,images,
                source/"dinov2.zip",source/"dino.pth",out/"features")
        run(here/"check_reference.py",data,out/"features",out/"analysis")


if __name__ == "__main__":
    main()
