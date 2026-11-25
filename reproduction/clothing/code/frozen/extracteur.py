"""CPU implementation of the two wavelet descriptor blocks used in the paper."""
import math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward

DEVICE = torch.device("cpu")
N_REGIONS = 21


def pyramid_regions(h, w):
    out = []
    # 4x4
    for gy in range(4):
        for gx in range(4):
            y0 = round(gy*h/4); y1 = round((gy+1)*h/4)
            x0 = round(gx*w/4); x1 = round((gx+1)*w/4)
            out.append((y0,y1,x0,x1))
    # 2x2
    for gy in range(2):
        for gx in range(2):
            y0 = round(gy*h/2); y1 = round((gy+1)*h/2)
            x0 = round(gx*w/2); x1 = round((gx+1)*w/2)
            out.append((y0,y1,x0,x1))
    # global
    out.append((0,h,0,w))
    assert len(out) == N_REGIONS
    return out


def rgb_to_opponent(raw):
    r,g,b = raw[:,0:1], raw[:,1:2], raw[:,2:3]
    return torch.cat([
        (r+g+b)/math.sqrt(3.0),
        (r-g)/math.sqrt(2.0),
        (r+g-2*b)/math.sqrt(6.0),
    ], dim=1)


def safe_spd(A, eps=1e-4):
    d = A.shape[-1]
    A = 0.5*(A + A.transpose(-1,-2))
    return A + eps*torch.eye(d, device=A.device, dtype=A.dtype)


def spd_logm(A):
    A = safe_spd(A)
    e, V = torch.linalg.eigh(A)
    e = e.clamp_min(1e-6)
    return (V * torch.log(e).unsqueeze(-2)) @ V.transpose(-1,-2)


def vech(A):
    d = A.shape[-1]
    idx = torch.triu_indices(d,d,device=A.device)
    return A[...,idx[0],idx[1]]


def rank_gaussian(X):
    # X B,N,D
    N = X.shape[1]
    order = torch.argsort(X,dim=1)
    ranks = torch.argsort(order,dim=1).float()+1.0
    u = (ranks-0.5)/N
    return math.sqrt(2.0)*torch.erfinv((2*u-1).clamp(-0.99999,0.99999))


def fit_copula(X):
    Z = rank_gaussian(X)
    Z = Z-Z.mean(1,keepdim=True)
    cov = torch.einsum("bnd,bne->bde",Z,Z)/max(1,Z.shape[1]-1)
    sd = torch.sqrt(torch.diagonal(cov,dim1=-2,dim2=-1).clamp_min(1e-8))
    R = cov/(sd[:,:,None]*sd[:,None,:]).clamp_min(1e-8)
    R = safe_spd(R)
    coord = vech(spd_logm(R))
    return coord,R

MG_BETA_GRID = torch.logspace(
    math.log10(0.3),math.log10(3.5),160,device=DEVICE
)
_D3 = 3.0
MG_RATIO = torch.exp(
    torch.lgamma((_D3+4)/(2*MG_BETA_GRID))
    + torch.lgamma(_D3/(2*MG_BETA_GRID))
    - 2*torch.lgamma((_D3+2)/(2*MG_BETA_GRID))
)



def fit_mggd(X):
    # X B,N,3
    X = X-X.mean(1,keepdim=True)
    cov = torch.einsum("bnd,bne->bde",X,X)/max(1,X.shape[1]-1)
    cov = safe_spd(cov)

    det = torch.linalg.det(cov).clamp_min(1e-12)
    scale = det**(1.0/6.0)
    C = cov/(scale[:,None,None]**2)
    C = safe_spd(C)

    inv = torch.linalg.inv(cov)
    r2 = torch.einsum("bnd,bde,bne->bn",X,inv,X).clamp_min(1e-8)
    ratio = (r2*r2).mean(1)/(r2.mean(1)**2).clamp_min(1e-8)
    idx = (ratio[:,None]-MG_RATIO[None,:]).abs().argmin(1)
    beta = MG_BETA_GRID[idx]

    coord = torch.cat([
        vech(spd_logm(C)),
        torch.log(scale)[:,None],
        torch.log(beta)[:,None]
    ],1)
    extras = torch.stack([torch.log(scale),torch.log(beta)],1)
    return coord,C,extras

def make_estimator():
    transform = DTCWTForward(J=3, biort="near_sym_b", qshift="qshift_b").eval()
    @torch.inference_mode()
    def extract(raw):
        batch=raw.shape[0]
        _, coefficients=transform(rgb_to_opponent(raw))
        real=[t[...,0].float() for t in coefficients]
        imag=[t[...,1].float() for t in coefficients]
        mag=[torch.sqrt(r*r+i*i+1e-10) for r,i in zip(real,imag)]
        mid=mag[1][:,0]
        height,width=mid.shape[-2:]
        means=[F.interpolate(v[:,0].mean(1,keepdim=True),size=(height,width),mode="bilinear",align_corners=False) for v in mag]
        coords9=torch.cat([mid,*means],1)
        structure=[]
        for top,bottom,left,right in pyramid_regions(height,width):
            s=coords9[:,:,top:bottom,left:right].permute(0,2,3,1).reshape(batch,-1,9)
            structure.append(fit_copula(s)[0])
        scales=[]
        for t in real:
            h,w=t.shape[-2:]; cells=[]
            for top,bottom,left,right in pyramid_regions(h,w):
                observations=t[:,:,:,top:bottom,left:right].permute(0,2,3,4,1).reshape(batch,-1,3)
                cells.append(fit_mggd(observations)[0])
            scales.append(torch.stack(cells,1))
        return {"F7":torch.stack(structure,1)[:,None],"F8":torch.stack(scales,1)}
    return extract


def input_tensor(photos):
    x = np.stack([np.asarray(im.convert("RGB").resize((224,224), Image.Resampling.BICUBIC), dtype=np.float32) / np.float32(255.) for im in photos])
    return torch.from_numpy(x).permute(0,3,1,2).contiguous()


def compact_features(photos, extractor):
    with torch.inference_mode():
        values = extractor(input_tensor(photos))
    result = {k: values[k].cpu().numpy().astype(np.float16) for k in ("F7","F8")}
    if not all(np.isfinite(x).all() for x in result.values()):
        raise ValueError("Nonfinite native features")
    return result
