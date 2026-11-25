"""Pinned manuscript extractors for the two external representations."""
import ctypes
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
PROTO = {"sources": {
    "dino_sha256": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
    "dinov2_commit": "e1277af2ba9496fbadf7aec6eba56e8d882d1e35"}}

def empreinte(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

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
            str(ROOT / "frozen/operateurs.cpp"),
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


