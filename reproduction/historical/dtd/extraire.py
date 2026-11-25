"""Extraire HSMT, CLBP et DINOv2 sur les photographies DTD, avec reprise par blocs."""

import argparse
import ctypes
import hashlib
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer
from PIL import Image, ImageOps
from preparer import PROTO, ROOT, empreinte


def setup_hsmt():
    sys.path.insert(0, str(ROOT / "reference/extracteur"))
    from extracteur import compact_features, make_estimator
    from histogramme_oxford import describe

    estimator = make_estimator()

    def run(images):
        result = compact_features(images, estimator)
        result["rgb"] = np.stack([describe(im)[0] for im in images])
        return result

    return run


def setup_clbp(tmp):
    dyn = tmp / "pr_clbp.so"
    subprocess.run(
        [
            "c++",
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-fopenmp",
            "-ffp-contract=off",
            str(ROOT / "reference/operateurs.cpp"),
            "-o",
            str(dyn),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(dyn))
    lib.clbp.argtypes = [
        ndpointer(np.uint8, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
        ndpointer(np.int32, flags="C_CONTIGUOUS"),
    ]
    lib.clbp.restype = None

    def run(images):
        counts = np.zeros((len(images), 1352), dtype=np.int32)
        hashes = []
        for i, im in enumerate(images):
            rgb = im.resize((224, 224), Image.Resampling.BICUBIC)
            hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            lum = np.ascontiguousarray(rgb.convert("L"), dtype=np.uint8)
            lib.clbp(lum, *lum.shape, counts[i])
        np.testing.assert_array_equal(counts.sum(1), np.full(len(images), 218**2))
        return {"clbp": counts, "pixels224": np.array(hashes)}

    return run


def setup_dino(src, tmp):
    import torch

    archive = src / "dinov2.zip"
    weights = src / "dino.pth"
    if (
        empreinte(archive)
        != "209c4f04fb0d63a869f7ea5c18d61dc7d8196c88cb1cb4a53bf3b0607146685c"
        or empreinte(weights) != PROTO["sources"]["dino_sha256"]
    ):
        raise ValueError("Empreinte de DINOv2 différente du modèle figé")
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            p = Path(member)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(member)
        z.extractall(tmp)
    sys.path.insert(0, str(tmp / ("dinov2-" + PROTO["sources"]["dinov2_commit"])))
    os.environ["XFORMERS_DISABLED"] = "1"
    from dinov2.hub.backbones import dinov2_vits14

    model = dinov2_vits14(pretrained=False)
    model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True), strict=True
    )
    model.eval().requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

    def run(images):
        pixels = [
            np.asarray(
                ImageOps.fit(
                    im,
                    (224, 224),
                    method=Image.Resampling.BICUBIC,
                    centering=(0.5, 0.5),
                ),
                dtype=np.uint8,
            )
            for im in images
        ]
        x = (
            torch.from_numpy(np.stack(pixels)).permute(0, 3, 1, 2).contiguous().float()
            / 255.0
        )
        with torch.inference_mode():
            values = model((x - mean) / std).cpu().numpy()
        assert values.shape == (len(images), 384) and np.isfinite(values).all()
        return {"dino": values}

    return run


def extract(mode, data, archives, cache, threads):
    if mode in ("dino", "hsmt"):
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    where = cache / mode
    where.mkdir(parents=True, exist_ok=True)
    with np.load(data / "index.npz") as f:
        ids = f["ids"]
    function = {
        "hsmt": setup_hsmt,
        "clbp": lambda: setup_clbp(where),
        "dino": lambda: setup_dino(archives, where),
    }[mode]()
    batch = 8 if mode != "dino" else 12
    for start in range(0, len(ids), 96):
        keys = ids[start : start + 96]
        part = where / f"{start:05d}.npz"
        if part.exists():
            with np.load(part) as f:
                np.testing.assert_array_equal(f["ids"], keys)
            continue
        chunks = {}
        hashes = []
        for i in range(0, len(keys), batch):
            images = []
            for key in keys[i : i + batch]:
                raw = (data / "images" / key).read_bytes()
                if mode == "clbp":
                    hashes.append(hashlib.sha256(raw).hexdigest())
                with Image.open(io.BytesIO(raw)) as im:
                    if mode == "dino":
                        im = ImageOps.exif_transpose(im)
                    images.append(im.convert("RGB"))
            for name, values in function(images).items():
                chunks.setdefault(name, []).append(values)
        arrays = {name: np.concatenate(seq) for name, seq in chunks.items()}
        if mode == "clbp":
            arrays["source_sha256"] = np.array(hashes)
        temp = part.with_suffix(".pending.npz")
        np.savez(temp, ids=keys, **arrays)
        temp.replace(part)
        if start % 480 == 0:
            print(mode, start + len(keys), "/", len(ids), flush=True)
    print(mode, "terminé", len(ids), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["hsmt", "clbp", "dino"])
    ap.add_argument("data", type=Path)
    ap.add_argument("archives", type=Path)
    ap.add_argument("cache", type=Path)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    extract(a.mode, a.data, a.archives, a.cache, a.threads)
