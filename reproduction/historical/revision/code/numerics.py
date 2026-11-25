"""Reconstruct native DTD wavelet features and counterfactuals before quantization.
Usage: numerics.py SUPPLEMENT DTD_DATA OUT [--threads 2]. The result is resumable.
"""
import argparse,json,math,sys,time,platform,resource
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward

def cell_groups(x,e):
    b,d,h,w=x.shape
    bounds=e.pyramid_regions(h,w)
    for lo,hi in ((0,16),(16,20),(20,21)):
        window=torch.stack([x[:,:,i:j,l:m] for i,j,l,m in bounds[lo:hi]],1)
        yield window.permute(0,1,3,4,2).reshape(b*(hi-lo),-1,d),hi-lo

def analyse(raw,transform,e):
    b=len(raw);_,coef=transform(e.rgb_to_opponent(raw));mag=[(c[...,0]**2+c[...,1]**2+1e-10).sqrt() for c in coef]
    h,w=mag[1].shape[-2:];v=torch.cat([mag[1][:,0]]+[F.interpolate(m[:,0].mean(1,keepdim=True),(h,w),mode='bilinear',align_corners=False) for m in mag],1)
    mid=[];old=[];ties=[]
    for x,g in cell_groups(v,e):
        n=x.shape[1];order=x.argsort(1);vals=x.gather(1,order);eq=vals[:,1:]==vals[:,:-1]
        pad=torch.zeros_like(vals[:,:1],dtype=torch.bool);left=torch.cat([pad,eq],1);right=torch.cat([eq,pad],1)
        pos=torch.arange(n)[None,:,None]
        first=torch.where(~left,pos,0).cummax(1).values;last=torch.where(~right,pos,n-1).flip([1]).cummin(1).values.flip([1])
        ranks=torch.zeros_like(x).scatter_(1,order,1+(first+last)*.5)
        z=math.sqrt(2)*torch.erfinv((2*(ranks-.5)/n-1).clamp(-.99999,.99999));z-=z.mean(1,keepdim=True)
        cov=z.transpose(1,2)@z/(n-1);sd=cov.diagonal(dim1=1,dim2=2).clamp_min(1e-8).sqrt()
        corr=cov/(sd[:,:,None]*sd[:,None,:]).clamp_min(1e-8);corr.diagonal(dim1=1,dim2=2).fill_(1)
        mid.append(e.vech(e.spd_logm(e.safe_spd(corr))).reshape(b,g,45));old.append(e.fit_copula(x)[0].reshape(b,g,45))
        const=vals[:,0]==vals[:,-1]
        ties.append(torch.stack([eq.sum((0,1)),(left|right).sum((0,1)),const.sum(0),torch.full((9,),len(x)*n),torch.full((9,),len(x))]))
    original=[];diag=[];magnitude=[];betas=[];counts=[]
    for signed,absolute in zip([q[...,0] for q in coef],mag):
        level=[[],[],[],[]];stats=[]
        for (t,g),(a,_) in zip(cell_groups(signed.flatten(1,2),e),cell_groups(absolute.flatten(1,2),e)):
            def orient(t):return t.reshape(len(t),-1,3,6).transpose(2,3).reshape(len(t),-1,3)
            t=orient(t);a=orient(a);x=t-t.mean(1,keepdim=True)
            cov=e.safe_spd(x.transpose(1,2)@x/(x.shape[1]-1));scale=cov.det().clamp_min(1e-12)**(1/6)
            scatter=cov/scale[:,None,None].square();cholinv=torch.linalg.inv(cov)
            r2=torch.einsum('bnd,bde,bne->bn',x,cholinv,x);q=r2.clamp_min(1e-8);avg=q.mean(1)
            ratio=q.square().mean(1)/avg.square().clamp_min(1e-8)
            beta=(ratio[:,None]-e.MG_RATIO).abs().argmin(1)
            base=torch.cat([e.vech(e.spd_logm(e.safe_spd(scatter))),scale.log()[:,None],e.MG_BETA_GRID[beta].log()[:,None]],1)
            d=base.clone();d[:,:6]=0;d[:,[0,3,5]]=(scatter.diagonal(dim1=1,dim2=2)+2e-4).log()
            low=avg<=1e-4;new=base[:,7].clone();new[low]=0
            stats.append(torch.tensor([len(q),q.numel(),(r2<1e-8).sum().item(),low.sum().item(),(beta==0).sum().item(),(beta==159).sum().item()]))
            for out,v in zip(level,(base,d,e.fit_mggd(a)[0],new)):out.append(v.reshape(b,g,-1))
        for out,part in zip((original,diag,magnitude,betas),level):out.append(torch.cat(part,1))
        counts.append(torch.stack(stats))
    values=dict(F7full=torch.cat(old,1)[:,None],mid=torch.cat(mid,1)[:,None],F8full=torch.stack(original,1),diag=torch.stack(diag,1),mag=torch.stack(magnitude,1),beta=torch.stack(betas,1)[...,0])
    return {k:v.numpy().astype('float32' if k.endswith('full') else 'float16') for k,v in values.items()},torch.stack(ties).numpy(),torch.stack(counts).numpy()
def main(a):
    sys.path.insert(0,str((a.supp/'dtd/reference/extracteur').resolve()));import extracteur as e
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);tf=DTCWTForward(J=3,biort='near_sym_b',qshift='qshift_b').eval()
    ids=np.load(a.data/'index.npz')['ids'];a.output.mkdir(exist_ok=True,parents=True)
    for start in range(0,len(ids),96):
        keys=ids[start:start+96];dest=a.output/f'{start:05d}.npz'
        if dest.exists():continue
        result={};st=0;sc=0;times=[];decodes=[]
        for i in range(0,len(keys),6):
            t=time.perf_counter();photos=[]
            for key in keys[i:i+6]:
                with Image.open(a.data/'images'/key) as im:photos.append(im.convert('RGB'))
            raw=e.input_tensor(photos);decodes.append(time.perf_counter()-t);t=time.perf_counter()
            with torch.inference_mode():v,s,c=analyse(raw,tf,e)
            times.append(time.perf_counter()-t);st+=s;sc+=c
            for k,f in v.items():result.setdefault(k,[]).append(f)
        np.savez(dest.with_suffix('.tmp.npz'),ids=keys,ties=st,radii=sc,seconds=times,decode_seconds=decodes,**{k:np.concatenate(v) for k,v in result.items()});dest.with_suffix('.tmp.npz').replace(dest)
        if start%480==0:print('images',start+len(keys),'rss_KiB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for s in ('supp','data','output'):p.add_argument(s,type=Path)
    p.add_argument('--threads',type=int,default=2);main(p.parse_args())
