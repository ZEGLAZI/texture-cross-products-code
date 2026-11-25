"""Validate S14 from frozen descriptor inputs, optionally comparing a new pixel extraction."""
from pathlib import Path
import argparse
import importlib.util
import json
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent
def refit_module():
    spec=importlib.util.spec_from_file_location('oxford_mask_source_refit',HERE/'refit.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def check(data,fresh_features=None):
    ref=HERE/'results'
    if fresh_features:
        for view in ('whole','foreground','background','outline'):
            with np.load(HERE/'features'/(view+'.npz')) as a,np.load(fresh_features/(view+'.npz')) as b:
                if set(a.files)!=set(b.files):raise ValueError(view+' schema differs')
                for name in a.files:np.testing.assert_array_equal(a[name],b[name],err_msg=view+' '+name)
    with tempfile.TemporaryDirectory() as temp:
        tmp=Path(temp)
        scores=refit_module().fit(data,fresh_features or HERE/'features',tmp)
        if scores!=json.loads((ref/'scores.json').read_text()):raise ValueError('Oxford scores or interval differ')
        n=0
        with np.load(tmp/'predictions.npz') as a,np.load(ref/'predictions.npz') as b:
            if set(a.files)!=set(b.files):raise ValueError('Incomplete Oxford fit')
            for k in b.files:
                if b[k].dtype.kind=='f':
                    np.testing.assert_allclose(a[k],b[k],rtol=1e-8,atol=1e-9,err_msg=k)
                else:
                    np.testing.assert_array_equal(a[k],b[k],err_msg=k)
                    if k not in ('y','image'):n+=b[k].size
        return scores,{'photos':sum(scores['counts']),'views':4,'paired_model_decisions':n,'mismatches':0}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path);p.add_argument('--fresh-features',type=Path);p.add_argument('--report',type=Path)
    a=p.parse_args();_,r=check(a.data,a.fresh_features)
    if a.report:a.report.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
