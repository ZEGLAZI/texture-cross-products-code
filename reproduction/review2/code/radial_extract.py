"""Recompute diagonal interventions and raw radial ratios from signed opponent coefficients."""
import argparse,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.special import gammaln
import torch
from pytorch_wavelets import DTCWTForward
import extracteur as e

def cells(fields):
    b,_,_,h,w=fields.shape
    regions=e.pyramid_regions(h,w)
    for lo,hi in ((0,16),(16,20),(20,21)):
        blocks=torch.stack([fields[:,:,:,i:j,l:m] for i,j,l,m in regions[lo:hi]],1)
        x=blocks.permute(0,1,3,4,5,2).reshape(b*(hi-lo),-1,3)
        yield x-x.mean(1,keepdim=True),hi-lo

def ratios(x,cov):
    q=torch.einsum('bnd,bde,bne->bn',x,torch.linalg.inv(cov),x).clamp_min(1e-8)
    return q.square().mean(1)/q.mean(1).square().clamp_min(1e-8)

def coords(cov,ratio):
    scale=cov.det().clamp_min(1e-12)**(1/6)
    shape=cov/scale[:,None,None].square()
    idx=(ratio[:,None]-e.MG_RATIO).abs().argmin(1)
    return torch.cat((e.vech(e.spd_logm(e.safe_spd(shape))),scale.log()[:,None],e.MG_BETA_GRID[idx].log()[:,None]),1),idx

def main(a):
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    tf=DTCWTForward(J=3,biort='near_sym_b',qshift='qshift_b').eval()
    keys=np.load(a.data/'index.npz')['ids'];a.dest.mkdir(exist_ok=True,parents=True)
    for start in range(0,len(keys),48):
        path=a.dest/f'{start:05d}.npz'
        if path.exists(): continue
        out={k:[] for k in ('diag','ratio','native_beta','diag_beta')}
        for j in range(start,min(len(keys),start+48),6):
            pics=[]
            for k in keys[j:j+6]:
                with Image.open(a.data/'images'/k) as im:pics.append(im.convert('RGB'))
            b=len(pics);base=e.input_tensor(pics)
            with torch.no_grad():
                _,coeffs=tf(e.rgb_to_opponent(base));sc={k:[] for k in out}
                for c in coeffs:
                    level={k:[] for k in out}
                    for x,g in cells(c[...,0].float()):
                        cov=e.safe_spd(x.transpose(1,2)@x/(x.shape[1]-1))
                        orig=ratios(x,cov);dc=torch.diag_embed(cov.diagonal(dim1=1,dim2=2))
                        dcoord,di=coords(dc,ratios(x,dc));idx=(orig[:,None]-e.MG_RATIO).abs().argmin(1)
                        for tag,v in [('diag',dcoord),('ratio',orig),('native_beta',idx),('diag_beta',di)]:
                            level[tag].append(v.reshape(b,g,-1))
                    for tag in sc:sc[tag].append(torch.cat(level[tag],1))
                for tag in out:
                    x=torch.stack(sc[tag],1).numpy()
                    if tag!='diag':x=x[...,0]
                    out[tag].append(x.astype('f2' if tag=='diag' else 'f4' if tag=='ratio' else 'u1'))
        np.savez_compressed(path,ids=keys[start:start+48],**{k:np.concatenate(v) for k,v in out.items()})
        if start%480==0:print(start+len(out['ratio'])*6,'of',len(keys),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('dest',type=Path);p.add_argument('--threads',type=int,default=2);main(p.parse_args())
