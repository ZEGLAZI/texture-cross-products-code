"""Compare independent expanded-kernel eigenfits with each archived decision on the existing tasks."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

def run(root,features,results,dataset):
    sys.path.insert(0,str((root/'studies/resnet50').resolve()))
    from independent import expanded_grams,resolve
    from cnn_models import load_task,PENALTIES,choose,macro
    ids,y,blocks,groups,folds=load_task(root,features,dataset)
    with (results/(dataset+'_settings.csv')).open() as f:settings={r['fold']:r for r in csv.DictReader(f)}
    with (results/(dataset+'_validation.csv')).open() as f:
        expected={(r['fold'],int(r['inner']),r['kernel'],float(r['penalty'])):float(r['macro_recall_pct']) for r in csv.DictReader(f)}
    nv=0;nt=0
    with np.load(results/(dataset+'_decisions.npz')) as ref:
        np.testing.assert_array_equal(groups,ref['groups'])
        for j,(fold,train,test,inner) in enumerate(folds):
            grids={'S':[],'Q':[]}
            for inner_id,(fit,val) in enumerate(inner):
                pair=expanded_grams(blocks,fit,val)
                for label,k in zip(('S','Q'),pair):
                    seq=resolve(k,y[fit],PENALTIES,len(np.unique(y)))
                    scores=[macro(v,y[val]) for v in seq]
                    for p,s in zip(PENALTIES,scores):
                        np.testing.assert_allclose(100*s,expected[fold,inner_id,label,p],atol=1e-10,rtol=0)
                        nv+=1
                    grids[label].append(scores)
                del pair,k
            best={k:choose(np.mean(v,axis=0)) for k,v in grids.items()}
            for k,v in best.items():
                if v!=float(settings[fold]['lambda_'+k]):raise ValueError('Penalty choice differs')
            pair=expanded_grams(blocks,train,test);factor=pair[0][:len(train)].diagonal().mean()/pair[1][:len(train)].diagonal().mean()
            np.testing.assert_allclose(factor,float(settings[fold]['matched_factor']),atol=2e-14,rtol=0)
            for k,keys,penalties in [(pair[0],('S','selected_S'),(1.,best['S'])),
                           (pair[1],('Q','matched_Q','selected_Q'),(1.,1./factor,best['Q']))]:
                for name,value in zip(keys,resolve(k,y[train],penalties,len(np.unique(y))),strict=True):
                    np.testing.assert_array_equal(value,ref[name][j]);nt+=len(value)
            np.testing.assert_array_equal(ids[test],ref['ids'][j]);np.testing.assert_array_equal(y[test],ref['labels'][j])
            del pair,k
            print('Independent fit',fold,flush=True)
    return dict(dataset=dataset,folds=len(folds),validation_scores=nv,matched_test_labels=nt)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('package','features','results'):p.add_argument(k,type=Path)
    p.add_argument('dataset');p.add_argument('report',type=Path);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    with threadpool_limits(a.threads):r=run(a.package,a.features,a.results,a.dataset)
    a.report.write_text(json.dumps(r,indent=2)+'\n')
