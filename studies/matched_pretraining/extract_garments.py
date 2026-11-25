"""Read locked garment regions and extract every representation from the same pixels."""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
for name in ('ATEN_CPU_CAPABILITY','ONEDNN_MAX_CPU_ISA','DNNL_MAX_CPU_ISA'):
    os.environ[name]='avx2'
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet50
from extract_dino1 import get_model,pool

def run(a):
    path=a.package/'studies/CVIU_Experience_couleur/code'
    sys.path.insert(0,str(path.resolve()))
    from dino_original import load_model,LAYERS
    from wavelets_original import make_estimator,compact_features
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    a.out.mkdir(exist_ok=True,parents=True)
    lock=json.loads((a.garments/'cohort.json').read_text())
    manifest=a.garments/'manifest.csv'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=lock['manifest_sha256']:raise ValueError('Changed cohort')
    with manifest.open() as f:rows=list(csv.DictReader(f))
    s_checkpoint=a.weights/'resnet50-11ad3fa6.pth'
    with s_checkpoint.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!='11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca':
            raise ValueError('Incorrect supervised checkpoint')
    supervised=resnet50(weights=None).eval().requires_grad_(False)
    supervised.load_state_dict(torch.load(s_checkpoint,weights_only=True,map_location='cpu'),strict=True)
    supervised.fc=torch.nn.Identity()
    d2=load_model(SimpleNamespace(source=a.weights/'dinov2.zip',weights=a.weights/'dino.pth',out=a.out,device='cpu'),torch)
    dr=get_model('dino_r50',a.weights/'official_r50.pth')
    dv=get_model('dino_vits16',a.weights/'official_vit.pth')
    estimator=make_estimator()
    files=[];mean=torch.tensor([.485,.456,.406])[None,:,None,None];sd=torch.tensor([.229,.224,.225])[None,:,None,None]
    with torch.inference_mode():
        for start in range(0,len(rows),a.batch):
            batch=rows[start:start+a.batch];parts=a.out/'parts';parts.mkdir(exist_ok=True)
            name=parts/(str(start)+'.npz');files.append(name)
            ids=[r['photo'] for r in batch];hashes=[r['rgb224_sha256'] for r in batch]
            if name.exists():
                with np.load(name) as prior:
                    np.testing.assert_array_equal(prior['hash224'],hashes);np.testing.assert_array_equal(prior['ids'],ids)
                continue
            ims=[]
            for entry,want in zip(ids,hashes,strict=True):
                with Image.open(a.garments/'crops'/(Path(entry).stem+'.png')) as im: rgb=im.convert('RGB')
                if rgb.size!=(224,224) or hashlib.sha256(rgb.tobytes()).hexdigest()!=want:raise ValueError('Changed crop')
                ims.append(rgb)
            items={'ids':ids,'hash224':hashes}
            pixels=np.stack([np.asarray(im) for im in ims])
            x=(torch.from_numpy(pixels).permute(0,3,1,2).contiguous().float()/255.-mean)/sd
            items.update(compact_features(ims,estimator))
            rgb=pixels.astype(np.int16)//64
            index=(16*rgb[:,:,:,0]+4*rgb[:,:,:,1]+rgb[:,:,:,2]).reshape(len(batch),-1)
            items['rgb']=np.stack([np.sqrt(np.bincount(i,minlength=64)/len(i)) for i in index])
            patches=d2.get_intermediate_layers(x,n=LAYERS,norm=True,return_class_token=True)
            if start==0:torch.testing.assert_close(patches[-1][1],d2(x),rtol=1e-5,atol=1e-5)
            for j,(v,_) in enumerate(patches):
                if v.shape[1:]!=(256,384):raise ValueError('DINOv2 feature grid')
                items['D'+str(j)]=v.mean(1).numpy()
            for label,model,arch in [('R',supervised,'dino_r50'),('DR',dr,'dino_r50'),('DV',dv,'dino_vits16')]:
                for j,v in enumerate(pool(model,arch,x,start==0)):items[label+str(j)]=v
            temp=name.with_suffix('.partial.npz');np.savez(temp,**items);temp.replace(name)
            if start%(a.batch*24)==0:print(start,len(rows),flush=True)
    with np.load(files[0]) as z:keys=z.files
    arrays={}
    for key in keys:
        parts=[]
        for file in files:
            with np.load(file) as z:parts.append(z[key])
        v=np.concatenate(parts)
        if v.dtype.kind=='f' and not np.isfinite(v).all():raise ValueError('Nonfinite '+key)
        arrays[key]=v
    dest=a.out/'features.npz';tmp=a.out/'features.pending.npz'
    np.savez_compressed(tmp,**arrays);tmp.replace(dest)
    (a.out/'verified.json').write_text(json.dumps({k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()},indent=2)+'\n')
    print('Extracted all five representations',len(rows),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('package','garments','weights','out'):p.add_argument(k,type=Path)
    p.add_argument('--threads',type=int,default=3);p.add_argument('--batch',type=int,default=8)
    run(p.parse_args())
