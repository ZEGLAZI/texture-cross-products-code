"""Préparer les images et les dix partitions à partir des archives Oxford officielles."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
PROTO = json.loads((ROOT / "protocole.json").read_text())


def empreinte(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prepare(src, out):
    out.mkdir(parents=True, exist_ok=True)
    for key in ("dtd", "labels", "decaf"):
        if empreinte(src / (key + ".tar.gz")) != PROTO["sources"][key + "_sha256"]:
            raise ValueError(f"Archive inattendue : {key}")
    labels = {}
    with tarfile.open(src / "dtd.tar.gz", "r|gz") as t:
        for entry in t:
            name = Path(entry.name)
            if not entry.isfile() or ".." in name.parts or name.is_absolute():
                continue
            if name.parts[:2] == ("dtd", "images") and name.suffix.lower() == ".jpg":
                dest = out / name.relative_to("dtd")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(t.extractfile(entry).read())
            if name.parts[:2] == ("dtd", "labels") and name.suffix == ".txt":
                labels[name.name] = t.extractfile(entry).read()
            if name.name == "imdb.mat":
                (out / "imdb.mat").write_bytes(t.extractfile(entry).read())
    ids = sorted(
        p.relative_to(out / "images").as_posix()
        for p in (out / "images").glob("*/*.jpg")
    )
    names = sorted({s.split("/")[0] for s in ids})
    if len(ids) != 5640 or len(names) != 47:
        raise ValueError("Inventaire DTD incomplet")
    idx = {s: i for i, s in enumerate(ids)}
    y = np.array([names.index(s.split("/")[0]) for s in ids], dtype=np.int16)
    imdb = loadmat(out / "imdb.mat", simplify_cells=True)["images"]
    np.testing.assert_array_equal(imdb["name"], ids)
    np.testing.assert_array_equal(imdb["id"], np.arange(1, 5641))
    np.testing.assert_array_equal(imdb["class"] - 1, y)
    split = {}
    with tarfile.open(src / "labels.tar.gz") as t:
        for k in range(1, 11):
            parts = []
            for role in ("train", "val", "test"):
                name = f"{role}{k}.txt"
                data = t.extractfile("labels/" + name).read()
                if data != labels[name]:
                    raise ValueError(f"Partitions officielles divergentes : {name}")
                pos = np.array(
                    [
                        idx[s.strip()]
                        for s in data.decode("ascii").splitlines()
                        if s.strip()
                    ]
                )
                np.testing.assert_array_equal(np.bincount(y[pos]), np.full(47, 40))
                split[role + str(k)] = pos
                parts.append(pos)
            np.testing.assert_array_equal(
                np.sort(np.concatenate(parts)), np.arange(5640)
            )
    np.savez_compressed(out / "index.npz", ids=ids, traits=names, y=y, **split)
    with tarfile.open(src / "decaf.tar.gz") as t:
        data = t.extractfile("dtd.may14/decaf_feats.mat").read()
    feats = loadmat(io.BytesIO(data), variable_names=["feats"])["feats"]
    assert feats.shape == (5640, 4096) and np.isfinite(feats).all()
    np.save(out / "decaf.npy", feats)
    print(
        "5 640 JPG, 47 classes, 30 listes officielles identiques dans les deux archives, 5 640 vecteurs DeCAF.",
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archives", type=Path)
    ap.add_argument("destination", type=Path)
    a = ap.parse_args()
    prepare(a.archives, a.destination)
