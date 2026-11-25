"""Measure archived versus freshly extracted colour--texture coordinates from provider pixels.

Examples (from the reviewer package, after extracting its data archive):
 python check_colour_pixels.py DTD reproduction inputs/dtd/images regenerated
 python check_colour_pixels.py KTH reproduction inputs/kth regenerated
Use the optional image-extraction dependencies listed in the README.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

# Use the reference AVX2 kernels before importing PyTorch or loading oneDNN.
import os
os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX2"
os.environ["DNNL_MAX_CPU_ISA"] = "AVX2"

import numpy as np
from PIL import Image
import torch


def extract(corpus, data, images_dir, output, threads):
    sys.path.insert(0, str((data/"historical/dtd/reference/extracteur").resolve()))
    from extracteur import make_estimator, compact_features
    from histogramme_oxford import describe

    keys = ("F7", "F8", "rgb")
    if corpus == "DTD":
        with np.load(data/"historical/dtd/results/identites.npz") as z:
            ids, file_hash, pixel_hash = (z[k] for k in ("ids", "source_sha256", "pixels224"))
        blocks = {k: [] for k in ("ids", *keys)}
        for p in sorted((data/"review2/dtd_cache/hsmt").glob("[0-9]*.npz")):
            with np.load(p) as z:
                for k in blocks: blocks[k].append(z[k])
        ref = {k: np.concatenate(arrays) for k, arrays in blocks.items()}
        np.testing.assert_array_equal(ids, ref.pop("ids"))
    else:
        with (data/"historical/external/data/kth_images.csv").open(encoding="utf-8", newline="") as handle:
            meta = list(csv.DictReader(handle))
        ids = np.array([r["chemin"] for r in meta])
        pixel_hash = [r["sha_pixels"] for r in meta]
        with np.load(data/"historical/external/data/kth.npz") as z:
            np.testing.assert_array_equal(ids, z["path"])
            ref = {k: z[k] for k in keys}
    output.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    estimator = make_estimator()
    result = {k: np.empty_like(ref[k]) for k in keys}
    for start in range(0, len(ids), 8):
        batch = []
        for i in range(start, min(start+8, len(ids))):
            p = images_dir/ids[i]
            if corpus == "DTD":
                with p.open("rb") as handle:
                    if hashlib.file_digest(handle, "sha256").hexdigest() != file_hash[i]:
                        raise ValueError(f"Changed file: {ids[i]}")
            with Image.open(p) as im:
                rgb = im.convert("RGB")
            if corpus == "DTD":
                pixels = rgb.resize((224,224), Image.Resampling.BICUBIC).tobytes()
            else:
                pixels = np.asarray(rgb.size, dtype="<u4").tobytes()+rgb.tobytes()
            if hashlib.sha256(pixels).hexdigest() != pixel_hash[i]:
                raise ValueError(f"Pixel fingerprint mismatch: {ids[i]}")
            batch.append(rgb)
        values = compact_features(batch, estimator)
        values["rgb"] = np.stack([describe(im)[0] for im in batch])
        for k in keys:
            result[k][start:start+len(batch)] = values[k]
        if (start % 512) == 0:
            print(corpus, start + len(batch), "/", len(ids), flush=True)
    stats = {}
    for k in keys:
        d = np.abs(result[k].astype(np.float64) - ref[k].astype(np.float64))
        stats[k] = dict(coordinates=int(d.size), different=int(np.count_nonzero(d)),
                        max_absolute=float(d.max()), mean_absolute=float(d.mean()),
                        differing_images=int(np.count_nonzero(d.reshape(len(ids),-1).any(axis=1))))
    report = dict(corpus=corpus, images=len(ids), verified_pixel_hashes=len(ids), blocks=stats,
                  torch=str(torch.__version__), threads=threads, batch=8,
                  cpu_kernels=torch.backends.cpu.get_cpu_capability())
    np.savez_compressed(output/(corpus+"_colour.npz"), ids=ids, **result)
    (output/(corpus+"_colour_check.json")).write_text(json.dumps(report, indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", choices=("DTD","KTH"))
    for name in ("data","images","output"): parser.add_argument(name,type=Path)
    parser.add_argument("--threads",type=int,default=2)
    args=parser.parse_args()
    extract(args.corpus,args.data,args.images,args.output,args.threads)
