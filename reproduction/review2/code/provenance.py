"""Tabulate radial-bound counts, flower attrition and actual historical settings."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from nested import PROFILES

def write(path,rows):
    with path.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def run(a):
    a.out.mkdir(exist_ok=True,parents=True);r=[];s=[]
    def summarize(name,blocks):
        b=np.concatenate(blocks);vals=b[...,7].astype(float);l=np.isclose(vals,np.log(.3),atol=.001);u=np.isclose(vals,np.log(3.5),atol=.001)
        for label,x,y in [('all',l,u)]+[(f'scale{j+1}',l[:,j],u[:,j]) for j in range(3)]:
            r.append(dict(corpus=name,stratum=label,images=len(b),regions=x.size,lower=int(x.sum()),upper=int(y.sum()),lower_pct=100*x.mean(),median_fraction=np.median(x.reshape(len(x),-1).mean(1))))
    def f8(f):
        with np.load(f) as z:return z['F8']
    summarize('DTD',[f8(f) for f in sorted((a.cache/'hsmt').glob('*.npz'))])
    for name,f in [('KTH','kth'),('TF source','tf'),('Pl@ntNet','plantnet'),('iNat 2026','nouveau_2026'),('iNat 2018-20','inat_2018_2020'),('iNat 2025','inat_2025')]:
        summarize(name,[f8(a.supp/'external/data'/(f+'.npz'))])
    for name,f in [('iNat 2015-17',a.cohort/'features.npz'),('Clothing',a.clothing/'features.npz'),('FMD',a.extra/'fmd.npz')]:summarize(name,[f8(f)])
    write(a.out/'corpus_radial_endpoints.csv',r)
    for f in sorted(a.selection.glob('*.json')):
        for m,ent in json.loads(f.read_text()).items():
            idx=ent['p'];p=PROFILES[idx];s.append(dict(fold=f.stem,model=m,profile=idx,p_s=p[0],p_w=p[1],p_c=p[2],ridge=ent['lambda'],validation_recall=ent['validation'],common_slope_factor='yes' if m=='W' and abs(p[2]-2*p[0])<1e-12 else 'no' if m=='W' else ''))
    write(a.out/'original_selected_profiles.csv',s)
    with np.load(a.cohort/'empreintes_anterieures.npz') as h:
        b=h['sha_j'];px=h['sha_p'];vals=h['hashes']
        write(a.out/'flower_reference_fingerprints.csv',[dict(row=i,byte_sha256=b[i],rgb_sha256=px[i],phash=f'{v[0]:016x}',phash_mirror=f'{v[1]:016x}') for i,v in enumerate(vals)])
    stages=json.loads((a.cohort/'selection.json').read_text())['classes']
    rows=[]
    for cl,b in stages.items():
        x=b['exclusions'];n=b['retenues'];rows.append(dict(taxon=cl,candidates=n+sum(x.values()),earlier_account=x.get('compte_antérieur',0),already_selected_account=x.get('compte_déjà_retenu',0),duplicate_image=x.get('image_identique_ou_proche',0),retained=n))
    assert sum(x['candidates'] for x in rows)==942;write(a.out/'flower_attrition.csv',rows)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('supp','cache','selection','cohort','clothing','extra','out'):p.add_argument(n,type=Path)
    run(p.parse_args())
