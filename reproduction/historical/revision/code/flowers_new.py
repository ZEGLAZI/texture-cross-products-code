"""Frozen source classifiers on newly ascertained observations, with one retained image per account."""
import argparse,csv,json,sys,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
from nested import kernel,fit,macro,METHODS,PROFILES
from inference import external
def featurize(a):
    import torch
    sys.path.insert(0,str((a.supp/'dtd/reference/extracteur').resolve()));from extracteur import make_estimator,compact_features
    from histogramme_oxford import describe
    torch.set_num_threads(2);torch.set_num_interop_threads(1);extract=make_estimator()
    rows=list(csv.DictReader((a.cohort/'nouveau_manifeste.csv').open()));out={}
    for i in range(0,len(rows),8):
        images=[]
        for x in rows[i:i+8]:
            fname=a.photos/(x['photo']+'.jpg');assert hashlib.sha256(fname.read_bytes()).hexdigest()==x['sha256_jpeg']
            with Image.open(fname) as im:images.append(im.convert('RGB'))
        blocks=compact_features(images,extract);blocks['rgb']=np.stack([describe(p)[0] for p in images])
        for k,v in blocks.items():out.setdefault(k,[]).append(v)
    np.savez_compressed(a.cohort/'features.npz',photo=[r['photo'] for r in rows],classe=[int(r['code']) for r in rows],**{k:np.concatenate(v) for k,v in out.items()})
def classifiers(supp,source,test,choices):
    sys.path.insert(0,str((supp/'dtd').resolve()));from predire import compact
    y=(source['classe'][:,None]==[10,12,9]).argmax(1);n=len(y);tr=np.arange(n);raw={k:np.concatenate([source[k],test[k]]) for k in ('F7','F8','rgb')}
    f,_=compact(raw,training=tr);f*=np.sqrt(2);grams=[t@t[tr].T for t in (f[:,:945],f[:,945:],raw['rgb'])];out={}
    for protocol,pars in [('fixed',{m:{'p':0,'lambda':1} for m in METHODS}),('selected',choices)]:
        for m in METHODS:
            c=pars[m];scores=fit(kernel(grams,PROFILES[c['p']],m),n,y,c['lambda'],3);out[protocol+'_'+m+'_scores']=scores;out[protocol+'_'+m]=scores.argmax(1)
    return out
def predict(a):
    rows=list(csv.DictReader((a.cohort/'nouveau_manifeste.csv').open()));test=np.load(a.cohort/'features.npz');np.testing.assert_array_equal(test['photo'],[r['photo'] for r in rows]);yy=(test['classe'][:,None]==[10,12,9]).argmax(1)
    pred=classifiers(a.supp,np.load(a.supp/'external/data/tf.npz'),test,json.loads(a.choices.read_text()))
    np.savez_compressed(a.cohort/'predictions.npz',photo=test['photo'],y=yy,**pred)
    result={}
    for p in ('fixed','selected'):
        by={m:(pred[p+'_'+m]==yy).astype('int8') for m in METHODS};r={m:macro(pred[p+'_'+m],yy) for m in METHODS}
        r['W-S']=external(by['W']-by['S'],yy,m=2);r['W-C']=external(by['W']-by['C'],yy,m=2)
        for k in ('Add','U','Q'):r['W-'+k]=external(by['W']-by[k],yy,m=3)
        r['recall_by_class']={m:[float(100*by[m][yy==c].mean()) for c in range(3)] for m in METHODS};result[p]=r
    (a.cohort/'scores.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('supp','cohort','photos','choices'):p.add_argument(n,type=Path)
    a=p.parse_args()
    with threadpool_limits(1):
        if not(a.cohort/'features.npz').exists():featurize(a)
        predict(a)
