"""Check all saved coordinates against 48 fixed public DTD reference images."""
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"reference/extracteur"))
from extracteur import make_estimator,compact_features
from histogramme_oxford import describe
p=argparse.ArgumentParser(description=__doc__);p.add_argument("images",type=Path);args=p.parse_args()
torch.set_num_threads(2);torch.set_num_interop_threads(1)
reference=np.load(ROOT/"results/golden_48.npz"); estimator=make_estimator()
for i in range(0,48,8):
    names=reference["ids"][i:i+8]
    images=[]
    for name in names:
        with Image.open(args.images/str(name)) as im:images.append(im.convert("RGB"))
    measured=compact_features(images,estimator)
    measured["rgb"]=np.stack([describe(im)[0] for im in images])
    for k in measured:np.testing.assert_array_equal(measured[k],reference[k][i:i+8])
print("48 images, 69,552 wavelet coordinates and 3,072 histogram coordinates matched exactly.")
