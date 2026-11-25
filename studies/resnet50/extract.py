"""Extract the three frozen ResNet-50 stages from the exact images in the reviewer archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, __version__ as pillow_version

CHECKPOINT = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
STAGES = ("layer2", "layer3", "layer4")
CHANNELS = (512, 1024, 2048)

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def read_rgb(args):
    path, file_sha, rgb_sha = args
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != file_sha:
        raise ValueError(f"Changed file {path}")
    with Image.open(io.BytesIO(raw)) as im:
        rgb = np.asarray(im.convert("RGB").resize((224,224), Image.Resampling.BICUBIC))
    if hashlib.sha256(rgb.tobytes()).hexdigest() != rgb_sha:
        raise ValueError(f"Changed 224-pixel input {path}")
    return rgb

def features(model, x):
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    x = model.layer1(x)
    maps = []
    for stage in STAGES:
        x = getattr(model,stage)(x)
        maps.append(x)
    return maps

def run(args):
    import torch
    import torchvision
    from torchvision.models import resnet50
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if digest(args.checkpoint) != CHECKPOINT:
        raise ValueError("The checkpoint differs from the locked TorchVision V2 file")
    model = resnet50(weights=None).eval().requires_grad_(False)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    mean = torch.tensor([.485,.456,.406])[None,:,None,None]
    sd = torch.tensor([.229,.224,.225])[None,:,None,None]
    with np.load(args.cache / (args.dataset + "_palettes.npz")) as cache:
        ids, file_hashes, pixel_hashes = (cache[k] for k in ("ids","file_hash","hash224"))
    rows = [(args.images / str(key), f, p) for key,f,p in zip(ids,file_hashes,pixel_hashes,strict=True)]
    values = [np.empty((len(ids),c),dtype=np.float32) for c in CHANNELS]
    with ThreadPoolExecutor(max_workers=2) as readers, torch.inference_mode():
        for start in range(0,len(ids),args.batch):
            pixels = np.stack(list(readers.map(read_rgb,rows[start:start+args.batch])))
            x = (torch.from_numpy(pixels).permute(0,3,1,2).contiguous().float()/255. - mean)/sd
            maps = features(model,x)
            if start == 0:
                if [tuple(v.shape[1:]) for v in maps] != [(512,28,28),(1024,14,14),(2048,7,7)]:
                    raise ValueError("Unexpected residual stage shapes")
                torch.testing.assert_close(model.fc(model.avgpool(maps[-1]).flatten(1)),model(x),rtol=0,atol=0)
            for target, array in zip(values, maps, strict=True):
                target[start:start+len(x)] = array.mean((2,3)).numpy()
            if start % (args.batch*20) == 0: print(args.dataset,start,len(ids),flush=True)
    if not all(np.isfinite(v).all() for v in values):
        raise ValueError("Nonfinite stage values")
    args.out.mkdir(exist_ok=True, parents=True)
    output = args.out / (args.dataset + "_stages.npz")
    temp = output.with_suffix(".partial.npz")
    np.savez_compressed(temp, ids=ids, hash224=pixel_hashes, **dict(zip(STAGES,values,strict=True)))
    temp.replace(output)
    report = {"dataset":args.dataset, "images":len(ids), "checkpoint_sha256":CHECKPOINT,
              "stages":STAGES, "channels":CHANNELS, "pooled_arrays_sha256":[hashlib.sha256(v.tobytes()).hexdigest() for v in values],
              "torch":torch.__version__, "torchvision":torchvision.__version__, "Pillow":pillow_version,
              "numpy":np.__version__, "batch":args.batch, "threads":args.threads, "device":"cpu",
              "dtype":"float32","file_sha_checks":len(ids),"pixel_sha_checks":len(ids),"classification_head_check":"passed"}
    output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    print("Extracted",args.dataset,len(ids),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset",choices=("DTD","KTH","FMD","Lytro"))
    for field in ("images","cache","checkpoint","out"):p.add_argument(field,type=Path)
    p.add_argument("--threads",type=int,default=6)
    p.add_argument("--batch",type=int,default=16)
    a=p.parse_args()
    if min(a.batch,a.threads)<1:p.error("Batch and thread count must be positive")
    run(a)
