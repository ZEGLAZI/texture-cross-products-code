"""Lecture des PNG officiels, avec identité du matériau, du spécimen et de la vue."""

import csv
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "protocole.json").read_text())
PATTERN = re.compile(
    r"KTH-TIPS2-b/([^/]+)/sample_([a-d])/[^/]+-scale_(\d+)_im_(\d+)_col\.png"
)


def sha_file(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def index(tar):
    result = []
    for member in tar.getmembers():
        if not member.isfile() or not member.name.endswith(".png"):
            continue
        match = PATTERN.fullmatch(member.name)
        if not match:
            raise ValueError(member.name)
        material, sample, scale, view = match.groups()
        result.append((material, sample, int(scale), int(view), member.name))
    result.sort()
    if len(result) != 4752 or len({tuple(x[:4]) for x in result}) != 4752:
        raise ValueError("Inventaire incomplet")
    classes = sorted({i[0] for i in result})
    expected = [
        (c, s, z, v)
        for c in classes
        for s in "abcd"
        for z in range(2, 11)
        for v in range(1, 13)
    ]
    if len(classes) != 11 or [x[:4] for x in result] != expected:
        raise ValueError("Plan incomplet")
    return result


def read(tar, pathname):
    raw = tar.extractfile(pathname).read()
    with Image.open(io.BytesIO(raw)) as im:
        image = im.convert("RGB")
    pixel_hash = hashlib.sha256(
        np.asarray(image.size, dtype="<u4").tobytes() + image.tobytes()
    ).hexdigest()
    return image, (
        image.width,
        image.height,
        hashlib.sha256(raw).hexdigest(),
        pixel_hash,
    )


def write_csv(path, header, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
