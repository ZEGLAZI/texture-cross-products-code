"""Recover only the 4,752 fingerprint-matched original images from a public mirror."""
import argparse
import csv
import hashlib
import io
import subprocess
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


COMMIT = "e1da4def9b5694ad17970054dc02bd84454cb436"


def main(data, repo, output):
    with (data / "historical/external/data/kth_images.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    expected = {(r["materiau"],Path(r["chemin"]).name):r for r in rows}
    assert len(expected)==len(rows)==4752
    command = ["git","-C",str(repo),"archive",COMMIT,"KTH/train1","KTH/test1"]
    seen = set()
    with subprocess.Popen(command, stdout=subprocess.PIPE) as proc:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for member in tar:
                p = Path(member.name)
                key = p.parent.name,p.name
                if not member.isfile() or key not in expected:
                    continue
                row = expected[key]
                buf = tar.extractfile(member).read()
                with Image.open(io.BytesIO(buf)) as image:
                    rgb = image.convert("RGB")
                    fingerprint = hashlib.sha256(np.asarray(rgb.size,dtype="<u4").tobytes()+rgb.tobytes()).hexdigest()
                if fingerprint != row["sha_pixels"] or key in seen:
                    raise ValueError(f"Clashing image identity: {key}")
                dest=output/row["chemin"]
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(buf)
                seen.add(key)
        if proc.wait()!=0:
            raise RuntimeError("Git could not extract the two folders.")
    if len(seen)!=len(rows):
        raise ValueError(f"Recovered {len(seen)} out of {len(rows)} images.")
    print("Verified KTH RGB identities",len(seen))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ("data","repo","output"):
        ap.add_argument(arg,type=Path)
    args=ap.parse_args()
    main(args.data,args.repo,args.output)
