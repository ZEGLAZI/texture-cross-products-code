"""Compare the frozen HSMT matrices with a fresh extraction from each locked crop."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from frozen.extracteur import compact_features, make_estimator
from frozen.histogramme_oxford import describe

ROOT=Path(__file__).resolve().parents[1]
def run(photos):
    torch.set_num_threads(2)
    model=make_estimator()
    with np.load(ROOT/'data/features.npz') as f:
        fields=('F7','F8','rgb')
        stored={k:f[k] for k in fields}
        keys=f['keys']
    batches={k:[] for k in fields}
    for start in range(0,len(keys),8):
        images=[]
        for key in keys[start:start+8]:
            with Image.open(photos/(key+'.png')) as photo:
                images.append(photo.convert('RGB'))
        batch=compact_features(images,model)
        batch['rgb']=np.stack([describe(im)[0] for im in images])
        for k in fields:
            batches[k].append(batch[k])
    audit={}
    for k in fields:
        field=np.concatenate(batches[k])
        difference=np.abs(field.astype(float)-stored[k].astype(float))
        audit[k]={'dimensions':list(field.shape),'identical_values':int(np.count_nonzero(difference==0)),
                  'total_values':int(difference.size),'maximum_absolute_difference':float(difference.max()),
                  'images_with_a_difference':int((difference.reshape(len(keys),-1).max(1)>0).sum())}
    (ROOT/'results/pixel_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(audit)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('photos',type=Path)
    run(p.parse_args().photos)
