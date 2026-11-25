"""Rebuild A, W and the 64-bin palette in the reference order and pixel-check every coordinate."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.update(ATEN_CPU_CAPABILITY="avx2", ONEDNN_MAX_CPU_ISA="AVX2", DNNL_MAX_CPU_ISA="AVX2")
import numpy as np
from PIL import Image
import torch
from wavelets_original import compact_features, make_estimator


def rebuild(corpus, images, reference_folder, destination, report, threads):
    with np.load(reference_folder / f"{corpus}_inputs.npz") as archive:
        design = dict(archive)
    fields = ("F7", "F8", "rgb")
    original = {key: design.pop(key) for key in fields}
    fresh = {key: np.empty_like(value) for key, value in original.items()}
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    extractor = make_estimator()
    ids = design["ids"]
    for start in range(0, len(ids), 8):
        photos = []
        for i in range(start, min(start + 8, len(ids))):
            with Image.open(images / ids[i]) as source:
                im = source.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
            if hashlib.sha256(im.tobytes()).hexdigest() != design["hash224"][i]:
                raise ValueError(f"Image fingerprint differs for {ids[i]}")
            codes = np.asarray(im).astype(np.int32) // 64
            bins = 16 * codes[:, :, 0] + 4 * codes[:, :, 1] + codes[:, :, 2]
            fresh["rgb"][i] = np.sqrt(np.bincount(bins.ravel(), minlength=64) / bins.size)
            photos.append(im)
        wavelets = compact_features(photos, extractor)
        for key in ("F7", "F8"):
            fresh[key][start:start + len(photos)] = wavelets[key]
        if start % 800 == 0:
            print(corpus, start + len(photos), "/", len(ids), flush=True)
    stats = {key: {"coordinates": fresh[key].size,
                   "different": int(np.count_nonzero(fresh[key] != original[key])),
                   "max_abs": float(np.abs(fresh[key].astype(float) - original[key]).max())} for key in fields}
    report.write_text(json.dumps({"corpus": corpus, "images": len(ids), "blocks": stats}, indent=2) + "\n")
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{corpus}_inputs.npz"
    tmp = output.with_suffix(".partial.npz")
    np.savez_compressed(tmp, **design, **fresh)
    tmp.replace(output)
    print(json.dumps(stats), flush=True)
    if any(entry["different"] for entry in stats.values()):
        raise ValueError(f"Inspect {report} for raw feature mismatches")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("corpus", choices=("DTD", "KTH", "FMD", "Lytro"))
    for key in ("images", "reference_folder", "destination", "report"):
        p.add_argument(key, type=Path)
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    rebuild(**vars(a))
