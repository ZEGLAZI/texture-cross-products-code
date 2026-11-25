"""Score the upstream covariance intervention and alternatives to the bounded radial coordinate."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from scipy.special import gammaln
from threadpoolctl import threadpool_limits
from nested import fit
from inference import dtd_interval

def write(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def theory(b):
    return np.exp(gammaln(7/(2*b))+gammaln(3/(2*b))-2*gammaln(5/(2*b)))
def nearest(rat,extend=True):
    beta=np.geomspace(.3,3.5,160)
    if extend:beta=np.r_[np.geomspace(.12,.3,97)[:-1],beta]
    val=theory(beta)[::-1];beta=beta[::-1]
    hi=np.searchsorted(val,rat).clip(1,len(beta)-1);lo=hi-1
    return beta[np.where(rat-val[lo]<=val[hi]-rat,lo,hi)]
def embed(a,tr):
    x=a.astype(float);dim=9 if x.shape[-1]==45 else 3
    ii,jj=np.triu_indices(dim);x[...,np.flatnonzero(ii!=jj)]*=np.sqrt(2)
    t=x[tr];mu=t.mean((0,2),keepdims=True);sd=np.sqrt(np.maximum(1e-8,t.var((0,2),keepdims=True)));del t
    x-=mu;x/=sd;np.clip(x,-12,12,out=x);x=x.reshape(len(x),-1);norm=np.linalg.norm(x,axis=1,keepdims=True)
    return np.divide(x,norm,out=x,where=norm!=0)
def main(a):
    sys.path.insert(0,str(a.supp/'dtd'))
    from predire import load_blocks,partitions
    ids=np.load(a.supp/'dtd/results/identites.npz');y=ids['y'];keys=ids['ids']
    files=sorted(a.rad.glob('[0-9]*.npz'));assert len(files)==118
    def collected(path,k):
        out=[]
        for p in path:
            with np.load(p) as f:out.append(f[k])
        return np.concatenate(out)
    np.testing.assert_array_equal(collected(files,'ids'),keys)
    rat=collected(files,'ratio');idx=collected(files,'native_beta');diag=collected(files,'diag')
    oldfiles=sorted(a.old.glob('[0-9]*.npz'))
    np.testing.assert_array_equal(collected(oldfiles,'ids'),keys)
    def native_numeric(key):
        parts=[]
        for f in oldfiles:
            with np.load(f) as v:
                if key+'full' in v:
                    parts.append(v[key+'full'].astype('f2'))
                else:
                    with np.load(a.cache/'hsmt'/f.name) as cached:
                        np.testing.assert_array_equal(v['ids'],cached['ids'])
                        arr=cached[key].astype('f2',copy=True)
                    arr.view('uint16').reshape(-1)[v[key+'_index']]=v[key+'_bits']
                    parts.append(arr)
        return np.concatenate(parts)
    F7=native_numeric('F7');F8=native_numeric('F8')
    rgb=load_blocks(a.cache/'hsmt',keys,['rgb'])['rgb'];held=collected(oldfiles,'diag')
    a.dest.mkdir(exist_ok=True,parents=True); lo,hi=theory(np.array([3.5,.3]))
    bits=idx==0;above=rat>hi;below=rat<lo
    rows=[]
    def stat(typ,b,up,dn):
        return dict(group=typ,regions=b.size,bound=int(b.sum()),above=int(up.sum()),below=int(dn.sum()),bound_pct=100*float(b.mean()),above_pct=100*float(up.mean()),below_pct=100*float(dn.mean()))
    for sk in range(3):
        rows.append(stat('scale'+str(sk+1),bits[:,sk],above[:,sk],below[:,sk]))
        for tag,sl in [('4x4',slice(0,16)),('2x2',slice(16,20)),('global',slice(20,21))]:
            rows.append(stat(f'scale{sk+1}_{tag}',bits[:,sk,sl],above[:,sk,sl],below[:,sk,sl]))
    rows.append(stat('all',bits,above,below));write(a.dest/'radial_strata.csv',rows)
    write(a.dest/'radial_images.csv',[dict(image=str(k),lower=int(bits[i].sum()),above=int(above[i].sum()),below=int(below[i].sum()),n=63) for i,k in enumerate(keys)])
    b2=nearest(rat.astype(float))
    report={'range':[float(lo),float(hi)],'bound_and_in_range':int((bits&~above&~below).sum()),'wide_at_min':int((b2==.12).sum()),'old_at_max':int((idx==159).sum()),
            'matched_native_indices':int(np.isclose(np.log(np.geomspace(.3,3.5,160)[idx]),F8[...,7],atol=.001).sum()),'n':int(bits.size),
            'per_image_bound_fraction_quantiles':np.quantile(bits.mean((1,2)),[0,.25,.5,.75,1]).tolist()}
    (a.dest/'radial_rates.json').write_text(json.dumps(report,indent=2));print(rows[-1],report,flush=True)
    altered={}
    for name,vec in [('expanded_grid',np.log(b2)),('log_ratio',np.log(np.maximum(rat,1e-8))),('boundary_indicator',bits)]:
        alt=F8.copy();alt[...,7]=vec.astype('f2');altered[name]=alt
    variants={'original':F8,'shape_diagonal':held,'upstream_diagonal':diag,**altered}
    for k,tr,te in partitions(ids,ids['groupes']):
        path=a.dest/f'Radial_{k:02}.npz'
        if path.exists():continue
        order=np.r_[tr,te];n=len(tr);s=embed(F7,tr);gram_s=s[order]@s[tr].T;del s
        gram_c=rgb[order]@rgb[tr].T
        save={'test':te}
        for name,feat in variants.items():
            w=embed(feat,tr);kernel=w[order]@w[tr].T;del w
            for start in range(0,len(order),64):
                sl=slice(start,start+64);wv=kernel[sl];sv=gram_s[sl];cv=gram_c[sl]
                kernel[sl]=(.5*sv+.5*wv+cv+.25*sv*wv+.5*wv*cv)/3.25
            save[name]=fit(kernel,n,y[tr],1,47).argmax(1).astype('u1');del kernel
        np.savez_compressed(path,**save);del gram_s,gram_c
        print('radial decisions',k,flush=True)
    arcs=[np.load(a.dest/f'Radial_{k:02}.npz') for k in range(1,11)]
    te=np.stack([f['test'] for f in arcs]);yy=y[te];gg=ids['groupes'][te]
    preds={name:np.stack([f[name] for f in arcs]) for name in variants}
    rows=[];cl=[]
    for name,base in [(m,'original') for m in variants]+[('upstream_diagonal','shape_diagonal')]:
        p=preds[name];ref=preds[base];delta=(p==yy).astype('i1')-(ref==yy)
        di,low,high=dtd_interval(delta,yy,gg,m=1,n=60000)
        rows.append(dict(method=name,reference=base,recall=100*(p==yy).mean(),diff=di,lo95=low,hi95=high,changed=int((p!=ref).sum())))
        for j,cn in enumerate(ids['traits']):
            z=(yy==j);cl.append(dict(method=name,reference=base,category=str(cn),appearances=int(z.sum()),diff=100*delta[z].mean(),changed=int((p[z]!=ref[z]).sum())))
    write(a.dest/'radial_scores.csv',rows);write(a.dest/'radial_classes.csv',cl)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['supp','cache','rad','old','dest']:p.add_argument(n,type=Path)
    a=p.parse_args()
    with threadpool_limits(2):main(a)
