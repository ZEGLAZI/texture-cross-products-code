"""Extract CLBP and frozen DINOv2 from the 211 locked RGB crops."""
import argparse
import csv
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from reference_extractors import setup_clbp, setup_dino
from frozen.histogramme_oxford import describe

ROOT=Path(__file__).resolve().parents[1]

def extract(photos, weights):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with np.load(ROOT/'data/features.npz') as frozen:
        keys=frozen['keys']
        rgb=frozen['rgb']
    with (ROOT/'data/sources.csv').open() as handle:
        sources=list(csv.DictReader(handle))
    assert [r['key'] for r in sources]==keys.tolist()
    counts, tokens = [], []
    with tempfile.TemporaryDirectory() as cache:
        clbp=setup_clbp(Path(cache))
        dino=setup_dino(weights,Path(cache))
        for start in range(0,len(keys),8):
            images=[]
            for index in range(start,min(start+8,len(keys))):
                with Image.open(photos/(keys[index]+'.png')) as source:
                    images.append(source.convert('RGB'))
                np.testing.assert_allclose(describe(images[-1])[0],rgb[index],rtol=1e-7,atol=1e-8)
            values=clbp(images)
            assert values['pixels224'].tolist()==[r['sha256_rgb224'] for r in sources[start:start+len(images)]]
            counts.append(values['clbp'])
            tokens.append(dino(images)['dino'])
    counts=np.concatenate(counts)
    tokens=np.concatenate(tokens)
    assert counts.shape==(211,1352) and tokens.shape==(211,384)
    assert np.isfinite(tokens).all() and np.all(np.linalg.norm(tokens,axis=1)>0)
    np.savez_compressed(ROOT/'data/reference_features.npz',keys=keys,clbp_counts=counts,dino_cls=tokens)
    print('Verified RGB and extracted both references for',len(keys),'sources.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('photos',type=Path)
    parser.add_argument('weights',type=Path)
    a=parser.parse_args()
    extract(a.photos,a.weights)
