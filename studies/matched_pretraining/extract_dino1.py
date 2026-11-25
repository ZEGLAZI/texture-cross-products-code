"""Spatially pool three depths of the original ImageNet-1K DINO checkpoints."""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet50

HERE = Path(__file__).resolve().parent
STAGES = ('layer2', 'layer3', 'layer4')
HASHES = {'dino_r50':'156f8c4166a23dc2951ae811e39d76a06269c565932edf647c0187e65cd7aa7c',
          'dino_vits16':'1566d50496f27f52f07fea6094fa29b2fdd6fae89da65bdd3ebc3b24ef6b7eb7'}

def get_model(name, checkpoint):
    with checkpoint.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != HASHES[name]:
            raise ValueError('DINO checkpoint fingerprint differs from the locked file')
    if name == 'dino_r50':
        model = resnet50(weights=None)
        model.fc = torch.nn.Identity()
    else:
        sys.path.insert(0, str(HERE / 'vendor'))
        from vision_transformer import vit_small
        model = vit_small(patch_size=16, num_classes=0)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return model.eval().requires_grad_(False)

def pool(model, name, x, check):
    if name == 'dino_r50':
        v = model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))
        result = []
        for stage in STAGES:
            v = getattr(model, stage)(v)
            result.append(v.mean((2,3)))
        if check:
            torch.testing.assert_close(result[-1], model(x), rtol=1e-5, atol=1e-5)
    else:
        layers = model.get_intermediate_layers(x, n=12)
        result = []
        for index in (2,5,11):
            if layers[index].shape[1:] != (197,384):
                raise ValueError('Unexpected spatial token count')
            result.append(layers[index][:,1:].mean(1))
        if check:
            torch.testing.assert_close(layers[11][:,0], model(x), rtol=1e-5, atol=1e-5)
    return [v.numpy() for v in result]

def run(a):
    sys.path.insert(0, str(a.package / 'studies/resnet50'))
    from extract import read_rgb
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    models = [get_model(n,a.weights / c) for n,c in zip(HASHES,('official_r50.pth','official_vit.pth'))]
    mean = torch.tensor([.485,.456,.406])[None,:,None,None]
    sd = torch.tensor([.229,.224,.225])[None,:,None,None]
    cache = a.package / 'studies/CVIU_Experience_couleur/cache'
    folders={'FMD':'FMD','Lytro':'lytro','DTD':'DTD/dtd/images','KTH':'KTH'}
    with torch.inference_mode(), ThreadPoolExecutor(2) as readers:
        for dataset, folder in folders.items():
            if a.datasets and dataset not in a.datasets: continue
            with np.load(cache / (dataset + '_palettes.npz')) as z:
                ids,original,pixels=(z[k] for k in ('ids','file_hash','hash224'))
            rows=[(a.images / folder / str(i), h, p) for i,h,p in zip(ids,original,pixels,strict=True)]
            buffers={n:[[],[],[]] for n in HASHES}
            for i in range(0,len(ids),a.batch):
                batch=np.stack(list(readers.map(read_rgb, rows[i:i+a.batch])))
                x=(torch.from_numpy(batch).permute(0,3,1,2).contiguous().float()/255.-mean)/sd
                for name,model in zip(HASHES,models):
                    for slot,v in zip(buffers[name],pool(model,name,x,i==0),strict=True):
                        slot.append(v)
                if i%(25*a.batch)==0: print(dataset,i,len(ids),flush=True)
            for name,blocks in buffers.items():
                folder_out=a.output / name;folder_out.mkdir(parents=True,exist_ok=True)
                vals=dict(zip(STAGES,[np.concatenate(q) for q in blocks],strict=True))
                if not all(np.isfinite(v).all() for v in vals.values()):raise ValueError('Non-finite features')
                dest=folder_out / (dataset+'_stages.npz')
                tmp=dest.with_suffix('.pending.npz')
                np.savez_compressed(tmp,ids=ids,hash224=pixels,**vals);tmp.replace(dest)
                dest.with_suffix('.json').write_text(json.dumps({'model':name,'images':len(ids),
                   'source_and_resized_hashes_checked':len(ids),'checkpoint':HASHES[name],
                   'array_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in vals.items()}},indent=2)+'\n')
            print('Extracted',dataset,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('package','images','weights','output'):p.add_argument(key,type=Path)
    p.add_argument('--threads',type=int,default=6);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--datasets',nargs='*');run(p.parse_args())
