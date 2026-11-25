"""Collect every assigned test appearance and quantify paired differences by the sampling unit."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from inference import dtd_interval,external
from assess import COEFF,PROFILES,PEN,GAMMA,select
from threadpoolctl import threadpool_limits

def rows_csv(path,recs):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=recs[0]);w.writeheader();w.writerows(recs)
def run(a):
    out=a.output;out.mkdir(exist_ok=True,parents=True);rt=a.kernels
    prim=[];long=[];selected=[];selection=[];splitrows=[];audit={}
    def load(name):
        with np.load(rt/(name+'.npz')) as f: z={k:f[k] for k in f.files}
        assert tuple(z['modes'])==COEFF
        assert len(np.intersect1d(z['train'],z['test']))==0
        for k in [v for v in z if v.startswith('inner_train_')]:
            t=z[k];v=z[k.replace('train','val')]
            assert set(t).isdisjoint(v) and np.isin(np.r_[t,v],z['train']).all()
        z['sel']=select(z)
        for m,g in zip(COEFF,z['grid']):
            j,h=np.unravel_index(np.argmax(g),g.shape)
            selection.append(dict(fold=name,method=m,profile=int(j) if m!='RBF' else '',gamma=GAMMA[j] if m=='RBF' else '',penalty=PEN[h],val=float(g[j,h])))
        return z
    def summarise(name,yd,g,joined,kind):
        # joined(pred_selector) returns an array of test decisions in yd order.
        yy=yd.ravel()
        def measure(v):
            c=(v.ravel()==yy);return float(100*np.mean([c[yy==label].mean() for label in np.unique(yy)]))
        def interval(pa,pb,m):
            delta=(pa==yd).astype('int8')-(pb==yd)
            if kind=='DTD':return dtd_interval(delta,yd,g,m=m)
            if kind=='Textile':return external(delta.mean(0),yd[0],m=m)
            return external(delta.ravel(),yy,g,m=m)
        for mode in ('fixed','tuned','unit'):
            for j in range(4):
                vals=[joined(lambda q,i=i:q[mode][i,j]) for i in range(4)]
                scores=[measure(v) for v in vals];pair=[interval(vals[3],vals[k],2) for k in (1,2)]
                row=dict(corpus=name,mode=mode,profile=j,**dict(zip(COEFF[:4],scores)))
                for tag,ci in zip(['Q_minus_S','Q_minus_X'],pair):
                    row[tag],row[tag+'_lo'],row[tag+'_hi']=ci
                long.append(row)
                if mode=='fixed' and j==0:
                    prim.append(row)
                    if kind in ('DTD','Textile'):
                        for r in range(len(yd)):
                            def mr(p):
                                return 100*np.mean([(p[yd[r]==j]==j).mean() for j in np.unique(yd[r])])
                            splitrows.append(dict(corpus=name,repeat=r+1,Q_minus_S=mr(vals[3][r])-mr(vals[1][r]),Q_minus_X=mr(vals[3][r])-mr(vals[2][r])))
        vals={m:joined(lambda q,i=i:q['sel'][i]) for i,m in enumerate(COEFF)}
        row={'corpus':name,**{m:measure(v) for m,v in vals.items()}}
        for l,r,t in [('RS','RX','RS_minus_RX'),('RS','Q','RS_minus_Q'),('RBF','RS','RBF_minus_RS')]:
            est,lo,hi=interval(vals[l],vals[r],1)
            row[t]=est;row[t+'_lo']=lo;row[t+'_hi']=hi
        selected.append(row)
    d=[load(f'DTD_{i:02}') for i in range(1,11)]
    ids=np.load(a.supp/'dtd/results/identites.npz');tests=np.stack([v['test'] for v in d])
    y=ids['y'][tests];g=ids['groupes'][tests]
    summarise('DTD',y,g,lambda fun:np.stack([fun(z) for z in d]),'DTD')
    matched={}
    for ref,name,i in [('selection','Q',3),('degrees','squares',4),('degrees','crosses',5),('selection','Add',0)]:
        old=np.stack([np.load(a.prev/ref/f'DTD_{k:02}.npz')[name] for k in range(1,11)])
        now=np.stack([z['sel'][i] for z in d]);matched['DTD_'+name]=int((now!=old).sum())
    del d
    k=[load('KTH_'+c) for c in 'abcd'];ii=np.concatenate([z['test'] for z in k]);rr=list(csv.DictReader((a.supp/'external/data/kth_images.csv').open()))
    assert len(set(ii))==len(ii)==4752
    labs=np.concatenate([z['labels'] for z in k]);groups=np.array([x['specimen'] for x in rr])[ii]
    summarise('KTH',labs,groups,lambda fun:np.concatenate([fun(z) for z in k]),'KTH')
    f=load('Flower');start=0
    for lab,nm in [('Pl@ntNet','plantnet'),('iNat 2026','nouveau_2026'),('iNat 2015-17',None)]:
        size=len(np.load(a.supp/f'external/data/{nm}.npz')['classe']) if nm else len(f['labels'])-start
        sl=slice(start,start+size);yy=f['labels'][sl]
        summarise(lab,yy,None,lambda fun:fun(f)[sl],'Flower');start+=size
    for nm in ['FMD','Outex']:
        if not (rt/(nm+'.npz')).exists():continue
        z=load(nm);summarise(nm,z['labels'],None,lambda fun:fun(z),nm)
    folds=[[load(f'Textile_{r:02}_{k}') for k in range(5)] for r in range(30)]
    y=np.load(a.textile/'features.npz')['y']
    def textile(fun):
        ans=np.full((30,211),-1,int)
        for r,repeat in enumerate(folds):
            for f in repeat:
                assert (ans[r,f['test']]==-1).all()
                ans[r,f['test']]=fun(f);np.testing.assert_array_equal(y[f['test']],f['labels'])
        assert np.all(ans>=0);return ans
    summarise('Clothing',np.tile(y,(30,1)),None,textile,'Textile')
    old=np.load(a.textile.parent/'results/predictions.npz')
    for name,i in [('additive',0),('squares',4),('crosses',5)]:
        matched['Clothing_'+name]=int((textile(lambda z:z['sel'][i])!=old['pred'][:,old['methods'].tolist().index(name)]).sum())
    rows_csv(out/'primary.csv',prim);rows_csv(out/'all_controls.csv',long);rows_csv(out/'redistributed_rbf.csv',selected)
    rows_csv(out/'selected_settings.csv',selection);rows_csv(out/'per_repeat.csv',splitrows)
    (out/'parity.json').write_text(json.dumps(matched,indent=2)+'\n')
    print('Parity divergences:',matched)
    for x in prim:print(x['corpus'],*[round(x[k],3) for k in ('L','S','X','Q','Q_minus_S','Q_minus_S_lo','Q_minus_S_hi','Q_minus_X','Q_minus_X_lo','Q_minus_X_hi')])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('supp','prev','textile','kernels','output'):p.add_argument(n,type=Path)
    a=p.parse_args()
    with threadpool_limits(1):run(a)
