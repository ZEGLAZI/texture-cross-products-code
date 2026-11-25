"""Two fixed families of 8-bit texture pairs with exactly the same complete RGB multiset.
The test is explanatory and does not enter model selection or validation.
"""
import argparse
import csv
from pathlib import Path
import os
for k in ('ATEN_CPU_CAPABILITY','ONEDNN_MAX_CPU_ISA','DNNL_MAX_CPU_ISA'):os.environ[k]='avx2' if k.startswith('ATEN') else 'AVX2'
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from PIL import Image
from wavelets_original import compact_features,make_estimator
from extract import routine
W=224
def pair(family,seed):
    rng=np.random.default_rng(seed)
    y,x=np.mgrid[:W,:W];field=gaussian_filter(rng.normal(size=(W,W)),1.5)+2*gaussian_filter(rng.normal(size=(W,W)),5)
    field+=.15*np.sin(.29*x+.015*y)+.09*np.cos(.075*x+.21*y)+.06*np.sin(.91*y+.007*x)
    loc=np.argsort(field.ravel(),kind='stable')
    if family=='opponent_swap':
        lo=rng.integers(12,124,W*W//2,dtype=np.uint8)
        t=np.sort(np.concatenate((lo,255-lo)));v=np.empty(W*W,dtype=np.uint8);v[loc]=t;v=v.reshape(W,W)
        original=np.stack((v,255-v,np.full_like(v,128)),axis=-1);reversed_=original[:,:,[1,0,2]]
    else:
        raw=rng.integers(8,248,(W*W//2,3),dtype=np.uint8);allc=np.concatenate((raw,255-raw))
        direction=allc.astype(float)@np.array([.7,.2,-.4])+rng.normal(0,3.,len(allc))
        original=np.zeros((W*W,3),dtype=np.uint8)
        original[loc]=allc[np.argsort(direction,kind='stable')];original=original.reshape(W,W,3)
        reversed_=255-original
    return np.ascontiguousarray(original),np.ascontiguousarray(reversed_)

def run(binary,out):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    lib=routine(binary);wavelet=make_estimator();records=[]
    for family in ('opponent_swap','contrast_reversal'):
        for seed in (19,113,257,422,727,811,1069,1811):
            pair_=pair(family,seed);ims=[Image.fromarray(a) for a in pair_]
            wave=compact_features(ims,wavelet)
            duplicate=compact_features([ims[0],ims[0]],wavelet)
            same_slot=[compact_features([im],wavelet) for im in ims]
            fv=[];lv=[]
            for a in pair_:
                v=np.empty((2,3,30));m=np.empty(9);q=np.empty((6,6))
                lib.colour_describe(a,W,W,65537,m,v,q);fv.append(v);lv.append(m)
            codes=[((x[:,:,0].astype(np.int32)*256+x[:,:,1])*256+x[:,:,2]).ravel() for x in pair_]
            np.testing.assert_array_equal(np.sort(codes[0]),np.sort(codes[1]))
            r={'family':family,'seed':seed,'same_RGB_multiset':True,'max_Lab9_error':float(np.max(abs(lv[0]-lv[1])))}
            for k in ('F7','F8'):
                a,b=wave[k].astype(np.float64)
                r[k+'_different']=int(np.count_nonzero(a!=b));r[k+'_max_abs']=float(abs(a-b).max())
            for k in ('F7','F8'):
                a,b=[z[k][0].astype(np.float64) for z in same_slot]
                r[k+'_same_slot_different']=int(np.count_nonzero(a!=b));r[k+'_same_slot_max']=float(abs(a-b).max())
                a,b=duplicate[k].astype(np.float64);r[k+'_identical_copy_in_batch_max']=float(abs(a-b).max())
            for i,name in enumerate(('P','J')):r[name+'_L2']=float(np.linalg.norm(fv[0][i,0]-fv[1][i,0]))
            records.append(r)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    print('Collision diagnostics',len(records),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('binary',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.binary,a.output)
