"""Freeze the image identities and folds before computing classifier outcomes."""
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]

def create(image_dir):
    with np.load(ROOT / 'data/features.npz') as data:
        keys, y, names = [data[name] for name in ('keys','y','classes')]
    assert len(keys) == len(set(keys)) == 211
    assert list(np.bincount(y)) == [43,58,23,14,39,34]
    seeds = np.arange(2026111900,2026111930)
    folds = np.full((len(seeds), len(keys)), -1, dtype=np.int8)
    for r, seed in enumerate(seeds):
        split = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
        for k, (_, test) in enumerate(split.split(keys, y)):
            folds[r, test] = k
        for c in range(len(names)):
            count = np.bincount(folds[r, y==c], minlength=5)
            assert count.max()-count.min() <= 1
    assert set(folds.ravel()) == set(range(5))
    assert len(set(map(bytes, folds))) == len(seeds)
    with (ROOT/'data/sources.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['key','class','sha256_rgb224'])
        for key, label in zip(keys,y):
            with Image.open(image_dir / (key+'.png')) as im:
                assert im.mode == 'RGB' and im.size == (224,224)
                digest = hashlib.sha256(im.tobytes()).hexdigest()
            writer.writerow([key,names[label],digest])
    np.savez_compressed(ROOT/'data/partitions.npz', keys=keys, y=y, seeds=seeds, folds=folds)
    paths = ['PROTOCOL.txt','data/features.npz','data/partitions.npz','data/sources.csv',
             'code/frozen/calcul_historique.py','code/frozen/degree_two.py',
             'code/frozen/nested.py','code/frozen/protocol.json','code/frozen/operateurs.cpp']
    with (ROOT/'LOCK.sha256').open('w') as f:
        for path in paths:
            digest=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
            f.write(f'{digest}  {path}\n')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('photos',type=Path)
    create(parser.parse_args().photos)
