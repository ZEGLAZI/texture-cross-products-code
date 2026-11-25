"""Verify the shipped v2 predictions, data separation, typed tables and formula identities."""
from pathlib import Path
import csv,json
import numpy as np
here=Path(__file__).resolve().parent;root=here.parent;r=here/'results'
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def js(p):return json.loads(p.read_text())
def check(x,y):np.testing.assert_allclose(x,y,rtol=0,atol=2e-8)
def macro(x,y):return np.mean([100*np.mean(x[y==j]==j) for j in np.unique(y)])
def run():
    ids=np.load(root/'dtd/results/identites.npz');frozen=np.load(root/'dtd/results/decisions.npz');score=js(r/'selected.json')
    methods=['Add','W','S','C','U','Q'];maskp=np.load(r/'masks/predictions.npz');masked=js(r/'masks/scores.json')
    for view in ['whole','foreground','background','outline']:
        for m in methods:
            for stage in ('fixed','selected'):
                key=stage+'_'+m;check(macro(maskp[view+'_'+key],maskp['y']),masked[view][key])
    masks=read(r/'masks/verified.csv');assert len(masks)==183 and sum(z['excluded']=='0' for z in masks)==168
    oldkeys=set();co=read(r/'cohort/nouveau_manifeste.csv');prev=read(r/'cohort/observations_anterieures.csv')
    for field in ['compte','id','photo']:
        keys={v[field] for v in co};assert len(keys)==420 and not keys&{v[field] for v in prev}
    assert min(int(v['d_anterieur']) for v in co)==8 and min(int(v['d_lot']) for v in co)==12
    new=np.load(r/'cohort/predictions.npz');assert len(set(new['photo']))==420
    for stage,v in js(r/'cohort/scores.json').items():
        for m in methods:
            pred=new[stage+'_'+m];assert np.array_equal(pred,new[stage+'_'+m+'_scores'].argmax(1));check(macro(pred,new['y']),v[m])
    raw=js(r/'ablations/numeric.json');num=np.array(raw['radii']).sum(axis=(0,1));assert num.tolist()==[355320,1671425280,374000,1,216661,0]
    tab=read(r/'ablations/ablation.csv');fi=[np.load(r/f'ablations/fold_{j:02}.npz') for j in range(1,11)]
    te=frozen['tests'];y=ids['y'][te];np.testing.assert_array_equal(np.stack([q['grouped_half'] for q in fi]),frozen['HSMT'])
    for row in tab:check(100*np.mean(np.stack([q[row['variant']] for q in fi])==y),float(row['recall']))
    conf=js(here/'code/protocol.json');folds=read(r/'dtd_folds.csv');dt=[]
    for j in range(1,11):
        z=np.load(r/f'selection/DTD_{j:02}.npz');dt.append(z)
        np.testing.assert_array_equal(te[j-1],z['test'])
        for lft,rgt in [('train','test'),('inner_train_0','inner_val_0')]:
            assert not np.intersect1d(ids['groupes'][z[lft]],ids['groupes'][z[rgt]]).size
        choices=js(r/f'selection/DTD_{j:02}.json')
        for mi,m in enumerate(methods):
            pr=z[m];assert np.array_equal(pr,z[m+'_scores'].argmax(1));check(macro(pr,y[j-1]),float(folds[j-1][m]))
            i,k=np.unravel_index(z['val_grid'][mi].argmax(),(4,6));assert i==choices[m]['p'] and conf['ridge_grid'][k]==choices[m]['lambda']
    for m in methods:check(100*np.mean(np.stack([q[m] for q in dt])==y),score['DTD'][m])
    kk=read(r/'kth_specimens.csv');assert len(kk)==44
    for m in methods:check(np.mean([100*int(x[m])/int(x['views']) for x in kk]),score['KTH'][m])
    ks=read(root/'external/data/kth_images.csv');sp=np.array([v['specimen'] for v in ks])
    for g in 'abcd':
        z=np.load(r/f'selection/KTH_{g}.npz');assert np.all(sp[z['test']]==g) and g not in sp[z['train']]
        for k in range(3):assert not set(sp[z[f'inner_train_{k}']])&set(sp[z[f'inner_val_{k}']])
    flowers=np.load(r/'selection/Flowers.npz');groups=np.array([q['credit_group'] for q in read(root/'external/data/tf_manifest.csv')])
    for k in range(3):assert not set(groups[flowers[f'inner_train_{k}']])&set(groups[flowers[f'inner_val_{k}']])
    off=0
    for tag,name in [('plantnet','Pl@ntNet'),('nouveau_2026','iNat2026')]:
        lab=(np.load(root/f'external/data/{tag}.npz')['classe'][:,None]==[10,12,9]).argmax(1)
        for m in methods:check(macro(flowers[m][off:off+len(lab)],lab),score[name][m])
        off+=len(lab)
    for dirname,name,desired in [('degrees','squares','squares'),('degrees','crosses','crosses'),('additive_grid','Add','recall')]:
        val=np.stack([np.load(r/f'{dirname}/DTD_{k:02}.npz')[name] for k in range(1,11)])
        check(100*np.mean(val==y),js(r/dirname/'scores.json')[desired])
    ref=js(r/'references/scores.json')
    for m,st in ref.items():
        for stage,aim in st.items():
            val=np.stack([np.load(r/f'references/fold_{k:02}.npz')[m+'_'+stage] for k in range(1,11)]);check(100*np.mean(val==y),aim)
    rng=np.random.default_rng(42);G=[]
    for n in (9,5,3):
        a=rng.normal(size=(18,n));a/=np.linalg.norm(a,axis=1)[:,None];G.append(a@a.T)
    s,w,c=G;d=3.25;T=(s+w)/2+c;kw=(T+.25*s*w+.5*w*c)/d;check(kw,((1+w/2)*(1+s/2+c)-1)/d)
    for p in conf['profiles']:
        q=sum(v*z for v,z in zip(p,G));assert np.linalg.eigvalsh((2*q+.75*q*q)/d).min()>-1e-10
    assert 945+504+64+945*504+504*64==510049
    print('Verified all six selected models, 44 specimen rows, 420 disjoint accounts, 168 masked photographs, numerical ablations, reference scores and the kernel identities.')
if __name__=='__main__':run()
