"""Rebuild the scores for Supplementary Table S14 from the four masked descriptors."""
import argparse
from pathlib import Path
import json
import sys
import numpy as np
from threadpoolctl import threadpool_limits

VIEWS=('whole','foreground','background','outline')
def fit(data,features,out):
    hist=data.resolve()/'historical'
    sys.path.insert(0,str(hist/'revision/code'))
    from flowers_new import classifiers
    from nested import macro
    from inference import external
    pars=json.loads((hist/'revision/results/selection/Flowers.json').read_text())
    scores={};arrays={}
    with np.load(hist/'external/data/tf.npz') as source, threadpool_limits(1):
        for view in VIEWS:
            with np.load(features/(view+'.npz')) as blocks:
                if view==VIEWS[0]:
                    arrays['image']=blocks['image'];arrays['y']=blocks['y']
                for key in ('image','y'):np.testing.assert_array_equal(blocks[key],arrays[key])
                pred=classifiers(hist,source,blocks,pars)
            scores[view]={k:macro(v,arrays['y']) for k,v in pred.items() if not k.endswith('_scores')}
            arrays.update({view+'_'+k:v for k,v in pred.items()})
    y=arrays['y']
    scores['background_minus_outline']=external((arrays['background_fixed_W']==y).astype('int8')-(arrays['outline_fixed_W']==y),y,m=1)
    scores['counts']=np.bincount(y,minlength=3).tolist()
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/'predictions.npz',**arrays)
    (out/'scores.json').write_text(json.dumps(scores,indent=2)+'\n')
    return scores
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','features','out'):p.add_argument(name,type=Path)
    a=p.parse_args();fit(a.data,a.features,a.out)
