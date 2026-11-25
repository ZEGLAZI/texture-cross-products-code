"""DTD conditional ablations with matched numerical reference, train-only preprocessing and fixed ridge."""
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from nested import fit
from inference import dtd_interval
GROUPS={'shape':[0,1,2,3,4,5],'scale':[6],'radial':[7],'no_shape':[6,7],'no_scale':[0,1,2,3,4,5,7],'no_radial':[0,1,2,3,4,5,6]}
def embedded(raw,dim,tr,coords=None):
    x=raw.astype('float64');ii,jj=np.triu_indices(dim);x[...,np.flatnonzero(ii!=jj)]*=np.sqrt(2)
    if coords is not None:x=x[...,coords]
    train=x[tr];mu=train.mean((0,2),keepdims=True);scale=np.sqrt(np.maximum(1e-8,train.var((0,2),keepdims=True)));del train
    x-=mu;x/=scale;np.clip(x,-12,12,out=x);x=x.reshape(len(x),-1)
    norm=np.linalg.norm(x,axis=1,keepdims=True);return np.divide(x,norm,out=np.zeros_like(x),where=norm!=0)
def run(a):
    sys.path.insert(0,str((a.supp/'dtd').resolve()));from predire import load_blocks,partitions
    ids=dict(np.load(a.supp/'dtd/results/identites.npz'));files=sorted(a.features.glob('[0-9][0-9][0-9][0-9][0-9].npz'));assert len(files)==59
    vals=load_blocks(a.features,ids['ids'],['mid','F7full','F8full','diag','mag','beta']);raw=load_blocks(a.cache/'hsmt',ids['ids'],['rgb','F7','F8'])
    old=np.load(a.supp/'dtd/results/decisions.npz');a.out.mkdir(exist_ok=True,parents=True);base7=vals['F7full'].astype('f2');base8=vals['F8full'].astype('f2')
    stats={'ties':sum(np.load(f)['ties'] for f in files).tolist(),'radii':sum(np.load(f)['radii'] for f in files).tolist()}
    stats['coordinate_reconstruction']={k:{'half_different':int((b!=raw[k]).sum()),'count':b.size,'max_abs':float(np.max(abs(b.astype(float)-raw[k].astype(float))))} for k,b in [('F7',base7),('F8',base8)]}
    nyt=base8.copy();nyt[...,7]=vals['beta'];stats['beta_changed']=int((nyt[...,7]!=base8[...,7]).sum());orderrows={}
    for k,tr,te in partitions(ids,ids['groupes']):
        f=a.out/f'fold_{k:02}.npz'
        if f.exists():continue
        order=np.r_[tr,te];n=len(tr);lab=ids['y'][tr];pred={}
        def linear(x):return x[order]@x[tr].T
        def block(x,d,coords=None):return linear(embedded(x,d,tr,coords))
        def read(s,w,c):
            total=(.5*s+.5*w+c)/3.25
            for lo in range(0,len(total),128):
                z=slice(lo,lo+128);total[z]+=(.25*s[z]*w[z]+.5*w[z]*c[z])/3.25
            return fit(total,n,lab,1,47).argmax(1).astype('uint8')
        c=linear(raw['rgb']);s=block(base7,9);w=block(base8,3);pred['grouped_half']=read(s,w,c)
        alt_s=block(vals['mid'],9);pred['mid']=read(alt_s,w,c);alt_w=block(nyt,3);pred['mid_beta']=read(alt_s,alt_w,c);pred['beta_neutral']=read(s,alt_w,c);del alt_s,alt_w,w
        for name,coords in GROUPS.items():pred[name]=read(s,block(base8,3,coords),c)
        for name in ('diag','mag'):pred[name]=read(s,block(vals[name],3),c)
        del s
        pred['grouped_single']=read(block(vals['F7full'],9),block(vals['F8full'],3),c)
        pred['native']=old['HSMT'][k-1];np.savez_compressed(f,test=te,**pred);del c
        print('ablations fold',k,flush=True)
    arc=[np.load(a.out/f'fold_{k:02}.npz') for k in range(1,11)];tests=old['tests'];yy=ids['y'][tests];groups=ids['groupes'][tests];ref=np.stack([q['grouped_half'] for q in arc]);rows=[]
    for m in arc[0].files:
        if m=='test':continue
        p=np.stack([q[m] for q in arc]);d=(p==yy).astype('int8')-(ref==yy);ci=dtd_interval(d,yy,groups,m=1,n=60000)
        rows.append(dict(variant=m,recall=100*float((p==yy).mean()),delta=ci[0],lo=ci[1],hi=ci[2],changed_label=int((p!=ref).sum())))
    with (a.out/'ablation.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (a.out/'numeric.json').write_text(json.dumps(stats,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('supp','cache','features','out'):p.add_argument(n,type=Path)
    p.add_argument('--threads',type=int,default=3)
    a=p.parse_args()
    with threadpool_limits(a.threads):run(a)
