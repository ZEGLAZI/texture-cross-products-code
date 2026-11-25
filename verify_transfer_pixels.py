"""Verify original FMD/Lytro pixels, the RGB baseline and the S12 colour features."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "studies/CVIU_Dependances_chromatiques/code"))
from extract import routine


def verify(images, dataset, binary, workers):
    transfer = ROOT / "studies/CVIU_Experience_couleur/cache"
    colour = ROOT / "studies/CVIU_Dependances_chromatiques/data"
    with np.load(transfer / (dataset + "_palettes.npz")) as z:
        ids, pixels, originals, rgb, sizes = (z[k] for k in ("ids","hash224","file_hash","values","sizes"))
    with np.load(colour / (dataset + "_new.npz")) as z:
        np.testing.assert_array_equal(ids,z["ids"])
        lab,edges,diagnostics = (z[k] for k in ("lab","edges","diagnostics"))
    library = routine(binary)
    def one(i):
        blob=(images / str(ids[i])).read_bytes()
        if hashlib.sha256(blob).hexdigest() != originals[i]:
            raise ValueError("Original file mismatch for " + str(ids[i]))
        with Image.open(io.BytesIO(blob)) as im:
            np.testing.assert_array_equal(im.size,sizes[i])
            arr=np.ascontiguousarray(im.convert("RGB").resize((224,224),Image.Resampling.BICUBIC))
        digest=hashlib.sha256(arr.tobytes()).hexdigest()
        if digest != pixels[i]:
            raise ValueError("Resized pixel mismatch for " + str(ids[i]))
        indices=arr.astype(np.int32)//64
        bins=16*indices[:,:,0] + 4*indices[:,:,1] + indices[:,:,2]
        hist=np.sqrt(np.bincount(bins.ravel(),minlength=64)/bins.size)
        np.testing.assert_array_equal(hist,rgb[i,0])
        measured_lab=np.empty_like(lab[i])
        measured_edges=np.empty_like(edges[i])
        measured_diagnostics=np.empty_like(diagnostics[i])
        library.colour_describe(arr,224,224,int(digest[:16],16),measured_lab,measured_edges,measured_diagnostics)
        for fresh, reference in [(measured_lab, lab[i]),(measured_edges, edges[i]),(measured_diagnostics, diagnostics[i])]:
            np.testing.assert_array_equal(fresh, reference, err_msg=str(ids[i]))
        return len(hist)+measured_lab.size+measured_edges.size
    with ThreadPoolExecutor(workers) as pool:
        total=sum(pool.map(one, range(len(ids))))
    return {"original_images_checked":len(ids),"resized_pixel_hashes_checked":len(ids),
            "colour_coordinates_checked":total,"support_diagnostics_checked":int(diagnostics.size),"mismatches":0}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for field in ("fmd","lytro","binary","report"):
        p.add_argument(field,type=Path)
    p.add_argument("--workers",type=int,default=2)
    a=p.parse_args()
    result={ds:verify(path,ds,a.binary,a.workers) for ds,path in (("FMD",a.fmd),("Lytro",a.lytro))}
    a.report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
