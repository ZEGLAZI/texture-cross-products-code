"""Rebuild paired statistics from the ordered predictions. No fold treats repeated images as independent."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.stats import binom
SEED=19112026
METHODS=['Add','W','S','C','U','Q']
def assess(pred,y):return (pred==y).astype('int8')
def macro(d,y):return float(np.mean([d[y==c].mean() for c in np.unique(y)])*100)
def dtd_interval(d,y,groups,m=3,n=100000,seed=SEED,report_mc=False):
    assert d.shape==(10,1880)
    keys,inv=np.unique(groups,return_inverse=True);inv=inv.reshape(y.shape)
    # DTD is balanced (40 images/class/fold): use integer group weights for an exactly
    # equivalent distribution of the multiplier, aggregating signs of equal magnitude.
    u=np.zeros(len(keys),dtype=int)
    for j in range(10):
        sums=np.bincount(y[j],weights=d[j],minlength=47).astype(int)
        np.add.at(u,inv[j],40*d[j]-sums[y[j]])
    w,counts=np.unique(abs(u[u!=0]),return_counts=True);rng=np.random.default_rng(seed);sim=np.zeros(n)
    for v,k in zip(w,counts):sim+=v*(2*rng.binomial(k,.5,n)-k)
    delta=float(d.mean()*100);sim=delta+sim*100/(d.size*40);probs=[.025/m,1-.025/m]
    ci=np.quantile(sim,probs)
    if report_mc:
        vals=np.sort(sim);rank=binom.ppf([.025,.975],n,probs[1]).astype(int)
        return {'delta':delta,'CI':ci.tolist(),'upper_MC_rank_95':vals[np.clip(rank,0,n-1)].tolist(),'upper_each_100k':[float(np.quantile(s,probs[1])) for s in np.split(sim,5)]}
    return [delta,*map(float,ci)]
def external(d,y,groups=None,m=3,n=100000):
    rng=np.random.default_rng(SEED);classes=np.unique(y);vals=np.zeros(n)
    for c in classes:
        keep=y==c;z=d[keep]
        if groups is not None:
            key=groups[keep];z=np.array([z[key==k].mean() for k in np.unique(key)])
        v,ct=np.unique(z,return_counts=True);vals+=(rng.multinomial(len(z),ct/len(z),n)@v)/len(z)
    return [macro(d,y),*np.quantile(vals*100/len(classes),[.025/m,1-.025/m]).tolist()]
def main(a):
    supp=a.supp;sel=a.selection;ids=np.load(supp/'dtd/results/identites.npz');old=np.load(supp/'dtd/results/decisions.npz');te=old['tests'];y=ids['y'][te];grp=ids['groupes'][te]
    arr=[dict(np.load(sel/f'DTD_{k:02}.npz')) for k in range(1,11)]
    for idx,b in enumerate(arr):np.testing.assert_array_equal(b['test'],te[idx])
    pre={m:np.stack([s[m] for s in arr]) for m in METHODS};result={};rows=[]
    def tab(name,pr,yy,kind,g=None):
        by={m:assess(pr[m],yy) for m in METHODS};res={m:macro(by[m],yy) for m in METHODS}
        for k in ('Add','U','Q'):
            d=by['W']-by[k];res['W-'+k]=dtd_interval(d,yy,g) if kind=='DTD' else external(d,yy,g)
        result[name]=res
    tab('DTD',pre,y,'DTD',grp)
    for j in range(10):
        row={'split':j+1}
        for m in METHODS:row[m]=100*assess(pre[m][j],y[j]).mean()
        row['W_Add_fixed']=100*(assess(old['HSMT'][j],y[j])-assess(old['Additif_fixe'][j],y[j])).mean()
        row['W_Add_selected']=row['W']-row['Add'];row['W_Q_selected']=row['W']-row['Q'];rows.append(row)
    a.out.mkdir(exist_ok=True,parents=True)
    def write(name,rows):
        with (a.out/name).open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    write('dtd_folds.csv',rows)
    rev=np.load(supp/'extra/reversals.npz')['DTD_W_reversed']
    result['reversal_MC']=dtd_interval(assess(old['HSMT'],y)-assess(rev,y),y,grp,4,500000,report_mc=True)
    f=np.load(sel/'Flowers.npz');trn=len(f['train']);offset=0
    for name,key in [('Pl@ntNet','plantnet'),('iNat2026','nouveau_2026')]:
        target=np.load(supp/f'external/data/{key}.npz');yy=(target['classe'][:,None]==[10,12,9]).argmax(1);sl=slice(offset,offset+len(yy));tab(name,{m:f[m][sl] for m in METHODS},yy,'flowers');offset+=len(yy)
    assert offset==len(f['test'])
    if all((sel/f'KTH_{c}.npz').exists() for c in 'abcd'):
        rr=list(csv.DictReader((supp/'external/data/kth_images.csv').open()));classes,lab=np.unique([r['materiau'] for r in rr],return_inverse=True);spec=np.array([r['specimen'] for r in rr])
        kk=[np.load(sel/f'KTH_{c}.npz') for c in 'abcd'];ii=np.concatenate([k['test'] for k in kk]);assert len(np.unique(ii))==4752
        yy=lab[ii];gp=spec[ii];pr={m:np.concatenate([k[m] for k in kk]) for m in METHODS};tab('KTH',pr,yy,'KTH',gp);rows=[]
        for j,c in enumerate(classes):
            for letter in 'abcd':
                ix=(yy==j)&(gp==letter);assert ix.sum()==108
                rows.append({'material':c,'specimen':letter,'views':int(ix.sum()),**{m:int((v[ix]==yy[ix]).sum()) for m,v in pr.items()}})
        write('kth_specimens.csv',rows)
    (a.out/'selected.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('supp','selection','out'):p.add_argument(n,type=Path)
    main(p.parse_args())
