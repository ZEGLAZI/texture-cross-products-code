"""Download the complete ten-class overlap, using one 2D pinhole preview per original capture."""
import argparse
import concurrent.futures
import io
from pathlib import Path
import time
from urllib.request import urlopen
from PIL import Image

CLASSES = "Fabric Foliage Glass Leather Metal Paper Plastic Stone Water Wood".split()
ROOTS = ("https://visualcomp.ucsd.edu/projects/LF/papers/ECCV16/",
         "https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV16/")
def fetch(args):
    root, name = args
    p = root / name
    if p.is_file():
        with Image.open(p) as image:
            image.verify()
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(6):
        try:
            with urlopen(ROOTS[attempt % len(ROOTS)] + "images/" + name, timeout=22) as handle:
                data = handle.read()
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            tmp = p.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(p)
            return
        except Exception as error:
            if attempt == 5:
                raise RuntimeError(name) from error
            time.sleep(0.3*(attempt+1))
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    jobs = [(a.output, f"{cls}/{i:03}.jpg") for cls in CLASSES for i in range(1,101)]
    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        futures={pool.submit(fetch,j):j[1] for j in jobs}
        errors=[]
        for n,f in enumerate(concurrent.futures.as_completed(futures),1):
            try: f.result()
            except Exception as e: errors.append(futures[f])
            if n%50==0: print(n, "completed,",len(errors),"errors",flush=True)
    if errors: raise RuntimeError(', '.join(errors))
