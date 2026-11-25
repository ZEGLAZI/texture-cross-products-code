"""Choose fresh DINOv2 vectors or a byte-identical released copy when rebuilding the CPU sensitivity."""
import hashlib
import json
from pathlib import Path

import numpy as np


def array_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def transformer_file(data, folder, corpus):
    fresh = folder/(corpus+"_features.npz")
    if fresh.is_file():
        return fresh
    cached = data/f"cross_representation/features/{corpus}_features.npz"
    manifest = folder/(corpus+"_transformer_hashes.json")
    if not manifest.is_file():
        raise FileNotFoundError(f"Supply {fresh} or the released digest record {manifest}.")
    expected = json.loads(manifest.read_text(encoding="utf-8"))
    with np.load(cached, allow_pickle=False) as z:
        if set(expected) != set(z.files) or any(array_digest(z[k])!=v for k,v in expected.items()):
            raise ValueError(f"{corpus}: the released transformer vectors differ from the pixel check.")
    return cached
