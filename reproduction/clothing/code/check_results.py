"""Cross-check the reported contrasts using the plain-text decisions independently."""
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    for line in (ROOT/'LOCK.sha256').read_text().splitlines():
        digest,path=line.split('  ',maxsplit=1)
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    with np.load(ROOT/'results/predictions.npz') as z:
        methods,y,classes,seeds,keys,folds,pred,grid,choices=[z[k] for k in (
            'methods','y','classes','seeds','keys','folds','pred','validation','choices')]
    assert grid.shape==(30,5,7,4,6)
    for r in range(30):
        for k in range(5):
            for m in range(len(methods)):
                j,h=choices[r,k,m]
                assert np.nanmax(grid[r,k,m])==grid[r,k,m,j,h]
                idx=np.ravel_multi_index((j,h),(4,6))
                assert idx==np.nanargmax(grid[r,k,m])
    raw=json.loads((ROOT/'results/summary.json').read_text())
    truth=dict(zip(keys,classes[y])); index=dict(zip(keys,range(len(keys))))
    repeat=dict(zip(seeds,range(30)))
    totals=Counter(); hits=Counter(); unique=set()
    with (ROOT/'results/predictions.csv').open() as handle:
        for row in csv.DictReader(handle):
            seed=int(row['seed']); source=row['source_key']; label=row['class']
            assert truth[source]==label and (seed,source) not in unique
            unique.add((seed,source))
            r=repeat[seed]; i=index[source]
            assert folds[r,i]==int(row['outer_test_fold'])
            totals[seed,label]+=1
            for m,name in enumerate(methods):
                assert classes[pred[r,m,i]]==row[name]
                hits[seed,label,name]+=row[name]==label
    assert len(unique)==211*30
    scores={name:np.asarray([100*np.mean([hits[s,c,name]/totals[s,c] for c in classes])
                            for s in seeds]) for name in methods}
    for name,values in scores.items():
        np.testing.assert_allclose([values.mean(),values.min(),values.max()],
                                   [raw['scores'][name][k] for k in ('mean','min','max')])
    for pair,row in raw['paired'].items():
        left,right=pair.split(' - ')
        diff=scores[left]-scores[right]
        np.testing.assert_allclose(diff.mean(),row['mean'],atol=1e-12)
        assert (diff>1e-10).sum()==row['wins']
        assert (abs(diff)<=1e-10).sum()==row['ties']
        assert (diff< -1e-10).sum()==row['losses']
    # Algebra in the manuscript uses the original frozen pair of kernel functions.
    import sys
    sys.path.insert(0,str(ROOT/'code/frozen'))
    from degree_two import decomposition
    rng=np.random.default_rng(10)
    s,w,c=rng.uniform(-1,1,(3,10000)); p=[1/3]*3
    delta=decomposition((s,w,c),p,'squares')-decomposition((s,w,c),p,'crosses')
    np.testing.assert_allclose(delta,((s-w)**2+(s-c)**2+(w-c)**2)/26,atol=1e-15)
    print('Verified locked inputs, 150 tuning grids for each method, 6330 image/partition rows, seven recalls, all paired differences and the equal-weight identity.')
if __name__=='__main__':
    main()
