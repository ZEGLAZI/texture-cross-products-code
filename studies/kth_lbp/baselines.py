"""CLBP de Guo et al. et OCLBP couleur de Bello-Cerezo et al., indépendants de HSMT."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
import tarfile
from tempfile import TemporaryDirectory

import numpy as np
from numpy.ctypeslib import ndpointer
from PIL import Image

from images import ROOT, PROTOCOL, index, read, sha_file


@lru_cache(maxsize=1)
def native():
    # The library has no dependencies beyond a C++17 compiler and OpenMP.
    with TemporaryDirectory() as scratch:
        path = Path(scratch) / "operateurs.so"
        subprocess.run(
            [
                "c++",
                "-O3",
                "-std=c++17",
                "-fPIC",
                "-shared",
                "-fopenmp",
                "-march=native",
                "-ffp-contract=off",
                str(ROOT / "operateurs.cpp"),
                "-o",
                str(path),
            ],
            check=True,
        )
        dll = ctypes.CDLL(str(path))
    for fun in (dll.clbp, dll.oclbp):
        fun.argtypes = [
            ndpointer(np.uint8, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            ndpointer(np.int32, flags="C_CONTIGUOUS"),
        ]
        fun.restype = None
    dll.distances.argtypes = [
        ndpointer(np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ]
    dll.distances.restype = None
    return dll


def features(photo):
    resized = photo.resize((224, 224), Image.Resampling.BICUBIC)
    gray = np.ascontiguousarray(resized.convert("L"), dtype=np.uint8)
    rgb = np.ascontiguousarray(resized, dtype=np.uint8)
    cl = np.zeros(1352, dtype=np.int32)
    oc = np.zeros(432, dtype=np.int32)
    native().clbp(gray, *gray.shape, cl)
    native().oclbp(rgb, *gray.shape, oc)
    if cl.sum() != 218**2:
        raise ValueError("Support CLBP incorrect")
    np.testing.assert_array_equal(
        oc.reshape(12, 36).sum(1),
        [(224 - 2 * r) ** 2 for r in (1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1)],
    )
    return cl, oc


def pairwise(x, is_chi, threads=3):
    values = np.ascontiguousarray(x, dtype=np.float64)
    output = np.empty((len(x), len(x)), dtype=np.float64)
    native().distances(values, *values.shape, is_chi, threads, output)
    return output


def extract(archive, out, threads):
    if sha_file(archive) != PROTOCOL["sha256_archive"]:
        raise ValueError("Archive KTH inattendue")
    native()
    lists = {"clbp": [], "oclbp": []}
    pixel_hashes = []
    with tarfile.open(archive) as tar, ThreadPoolExecutor(max_workers=threads) as pool:
        items = index(tar)
        for start in range(0, len(items), 48):
            payload = [read(tar, x[-1]) for x in items[start : start + 48]]
            pixel_hashes.extend(x[1][-1] for x in payload)
            for cl, oc in pool.map(features, [x[0] for x in payload]):
                lists["clbp"].append(cl)
                lists["oclbp"].append(oc)
            if start % 432 == 0:
                print(start, "/", len(items), flush=True)
    out.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(
        out / "temoins.npz",
        path=[r[-1] for r in items],
        sha_pixels=pixel_hashes,
        **{k: np.stack(v) for k, v in lists.items()},
    )
    print("CLBP et OCLBP extraits sur", len(items), "photos", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("--sortie", type=Path, default=ROOT / "donnees")
    p.add_argument("--threads", type=int, default=3)
    a = p.parse_args()
    extract(a.archive, a.sortie, a.threads)
