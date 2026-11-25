"""Extract full-field, frozen DINOv2-S/14 features in the order of the HSMT archive."""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version


COMMIT = "e1277af2ba9496fbadf7aec6eba56e8d882d1e35"
WEIGHTS = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
SOURCE = "209c4f04fb0d63a869f7ea5c18d61dc7d8196c88cb1cb4a53bf3b0607146685c"
LAYERS = [2, 5, 11]  # Public API uses zero-based block indices.
FEATURES = ("patch3", "patch6", "patch12", "cls12")


def file_digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def image_records(corpus, data):
    if corpus == "DTD":
        with np.load(data / "historical/dtd/results/identites.npz") as f:
            return list(zip(f["ids"].tolist(), f["source_sha256"].tolist(),
                            f["pixels224"].tolist(), strict=True))
    with (data / "historical/external/data/kth_images.csv").open() as handle:
        return [(row["chemin"], None, row["sha_pixels"]) for row in csv.DictReader(handle)]


def resized_image(root, record):
    key, file_sha, pixel_sha = record
    contents = (root / key).read_bytes()
    if file_sha and hashlib.sha256(contents).hexdigest() != file_sha:
        raise ValueError(f"Changed DTD file: {key}")
    with Image.open(io.BytesIO(contents)) as image:
        rgb = image.convert("RGB")
        full = rgb.resize((224, 224), Image.Resampling.BICUBIC)
        data = full.tobytes() if file_sha else np.asarray(rgb.size, dtype="<u4").tobytes() + rgb.tobytes()
        if hashlib.sha256(data).hexdigest() != pixel_sha:
            raise ValueError(f"RGB fingerprint disagrees with the HSMT archive: {key}")
        return np.asarray(full)


def load_model(args, torch):
    if file_digest(args.weights) != WEIGHTS or file_digest(args.source) != SOURCE:
        raise ValueError("The DINOv2 weights or implementation differ from the manuscript.")
    with zipfile.ZipFile(args.source) as src:
        # Only source files for the pinned package are required.
        for name in src.namelist():
            p = Path(name)
            if p.parts[:2] == ("dinov2-" + COMMIT, "dinov2") and ".." not in p.parts and p.suffix == ".py":
                src.extract(name, args.out / "model_code")
    sys.path.insert(0, str(args.out / "model_code" / ("dinov2-" + COMMIT)))
    os.environ["XFORMERS_DISABLED"] = "1"
    from dinov2.hub.backbones import dinov2_vits14
    model = dinov2_vits14(pretrained=False)
    model.load_state_dict(torch.load(args.weights, weights_only=True, map_location="cpu"), strict=True)
    return model.eval().requires_grad_(False).to(args.device)


def main(args):
    import torch
    if args.batch < 1 or args.threads < 1:
        raise ValueError("Batch size and thread count must be positive.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(19112026)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    records = image_records(args.corpus, args.data)
    assert len(records) == (5640 if args.corpus == "DTD" else 4752)
    args.out.mkdir(parents=True, exist_ok=True)
    model = load_model(args, torch)
    mean = torch.tensor([.485, .456, .406], device=args.device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=args.device)[None, :, None, None]
    provenance = dict(corpus=args.corpus, source=SOURCE, weights=WEIGHTS, commit=COMMIT,
                      device=args.device, torch=torch.__version__, pillow=pillow_version,
                      numpy=np.__version__, layers=LAYERS, fp32=True, script=file_digest(Path(__file__)),
                      manifest=hashlib.sha256(json.dumps(records).encode()).hexdigest())
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    parts = args.out / "parts" / args.corpus / signature
    parts.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    validated = False
    with torch.inference_mode():
        for start in range(0, len(records), 96):
            subset = records[start:start+96]
            keys = np.array([r[0] for r in subset])
            dest = parts / f"{start:05d}.npz"
            if dest.exists():
                with np.load(dest) as f:
                    np.testing.assert_array_equal(f["ids"], keys)
                continue
            vectors = {key: [] for key in FEATURES}
            for pos in range(0, len(subset), args.batch):
                pixels = np.stack([resized_image(args.images, row) for row in subset[pos:pos+args.batch]])
                inputs = torch.from_numpy(pixels).to(args.device).permute(0, 3, 1, 2).contiguous().float() / 255.
                inputs = (inputs - mean) / std
                values = model.get_intermediate_layers(inputs, n=LAYERS, norm=True, return_class_token=True)
                if not validated:
                    assert len(values) == 3 and all(v[0].shape[1:] == (256,384) for v in values)
                    torch.testing.assert_close(values[-1][1], model(inputs), rtol=1e-5, atol=1e-5)
                    validated = True
                pooled = [patch.mean(dim=1) for patch, cls in values] + [values[-1][1]]
                for name, value in zip(FEATURES, pooled, strict=True):
                    vectors[name].append(value.cpu().numpy())
            arrays = {k: np.concatenate(v) for k,v in vectors.items()}
            assert all(a.shape == (len(subset),384) and np.isfinite(a).all() for a in arrays.values())
            tmp = dest.with_suffix(".partial.npz")
            np.savez(tmp, ids=keys, **arrays)
            tmp.replace(dest)
            if start % 480 == 0:
                print(args.corpus, start + len(subset), "/", len(records),
                      "elapsed seconds", round(time.perf_counter()-start_time), flush=True)
    collected = {k: [] for k in ("ids", *FEATURES)}
    for path in sorted(parts.glob("*.npz")):
        with np.load(path) as values:
            for key in collected:
                collected[key].append(values[key])
    joined = {k: np.concatenate(v) for k,v in collected.items()}
    np.testing.assert_array_equal(joined["ids"], np.asarray([r[0] for r in records]))
    dest = args.out / f"{args.corpus}_features.npz"
    tmp = dest.with_suffix(".partial.npz")
    np.savez_compressed(tmp, **joined)
    tmp.replace(dest)
    provenance.update(images=len(records), output_sha256=file_digest(dest))
    (args.out / f"{args.corpus}_extraction.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print("Finished", args.corpus, len(records), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", choices=["DTD","KTH"])
    for name in ("data","images","source","weights","out"):
        ap.add_argument(name, type=Path)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    main(ap.parse_args())
