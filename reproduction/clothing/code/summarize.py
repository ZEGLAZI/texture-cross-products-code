"""Report paired macro recall, split sensitivity and a source-stratified bootstrap."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
PAIRS=(('squares','crosses'),('squares','additive'),('squares','clbp'),
       ('squares','dino'),('dino_squares','dino'),('dino_squares','dino_additive'))

def write_csv(output,name,columns,rows):
    with (output/name).open('w',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)

def quantiles(draws,lower):
    return [float(v) for v in np.quantile(draws,[lower,1-lower])]

def bootstrap(source_differences,y):
    rng=np.random.default_rng(19112647)
    simulated=np.zeros((100000,source_differences.shape[1]))
    for label in np.unique(y):
        values=source_differences[y==label]
        n=len(values)
        for start in range(0,len(simulated),1000):
            counts=rng.multinomial(n,np.full(n,1/n),1000)
            simulated[start:start+1000]+=counts@values*(100/(6*n))
    return simulated

def main(output):
    output.mkdir(exist_ok=True, parents=True)
    with np.load(OUT/'predictions.npz') as z:
        methods=z['methods'].tolist()
        y,keys,classes,seeds,folds,pred=[z[k] for k in ('y','keys','classes','seeds','folds','pred')]
    assert pred.shape==(30,7,211) and set(pred.ravel())==set(range(6))
    correct=pred==y[None,None,:]
    # Compute source-level correctness within each repeat before giving equal weight to classes.
    class_recalls=np.stack([correct[:,:,y==c].mean(axis=2)*100 for c in range(6)],axis=2)
    recalls=class_recalls.mean(axis=2)
    summary={'n_sources':len(y),'counts':dict(zip(classes.tolist(),np.bincount(y).tolist())),
             'repeats':len(seeds),'outer_folds':5,'scores':{},'paired':{}}
    for i,m in enumerate(methods):
        values=recalls[:,i]
        summary['scores'][m]={'mean':float(values.mean()),'min':float(values.min()),
                              'max':float(values.max()),'median':float(np.median(values))}
    source_correct=correct.mean(axis=0)
    differences=np.column_stack([source_correct[methods.index(a)]-source_correct[methods.index(b)] for a,b in PAIRS])
    draws=bootstrap(differences,y)
    for j,(a,b) in enumerate(PAIRS):
        values=recalls[:,methods.index(a)]-recalls[:,methods.index(b)]
        estimate=float(values.mean())
        independent_estimate=np.mean([differences[y==c,j].mean() for c in range(6)])*100
        np.testing.assert_allclose(estimate,independent_estimate,atol=1e-12)
        row={'mean':estimate,'ci95':quantiles(draws[:,j],0.025),'min':float(values.min()),
             'max':float(values.max()),'median':float(np.median(values)),
             'wins':int(np.sum(values>1e-10)),'ties':int(np.sum(np.abs(values)<=1e-10)),
             'losses':int(np.sum(values< -1e-10))}
        if j>0:
            row['ci99_family']=quantiles(draws[:,j],0.005)
        summary['paired'][a+' - '+b]=row
    # Keep decisions available as ordinary text for external auditing.
    write_csv(output,'predictions.csv',['seed','outer_test_fold','source_key','class',*methods],
              ([int(seed),int(folds[r,i]),key,classes[y[i]],*[classes[j] for j in pred[r,:,i]]]
               for r,seed in enumerate(seeds) for i,key in enumerate(keys)))
    write_csv(output,'by_class.csv',['class','n_sources',*methods],
              ([name,int(np.sum(y==c)),*class_recalls[:,:,c].mean(0)]
               for c,name in enumerate(classes)))
    write_csv(output,'by_partition.csv',['seed',*methods,*[a+' minus '+b for a,b in PAIRS]],
              ([int(seed),*recalls[r],*[recalls[r,methods.index(a)]-recalls[r,methods.index(b)]
                                       for a,b in PAIRS]] for r,seed in enumerate(seeds)))
    write_csv(output,'paired.csv',['left','right','mean_pp','low95','high95','low99_family','high99_family','min','max','wins','ties','losses'],
              ([a,b,row['mean'],*row['ci95'],*row.get('ci99_family',['','']),
                row['min'],row['max'],row['wins'],row['ties'],row['losses']]
               for a,b in PAIRS for row in [summary['paired'][a+' - '+b]]))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for m,v in summary['scores'].items():
        print(m,round(v['mean'],3),flush=True)
    for m,v in summary['paired'].items():
        print(m,round(v['mean'],3),[round(x,2) for x in v['ci95']],v['wins'],v['ties'],v['losses'],
              [round(v['min'],3),round(v['max'],3)],v.get('ci99_family'),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args=parser.parse_args()
    with threadpool_limits(1):
        main(args.output)
