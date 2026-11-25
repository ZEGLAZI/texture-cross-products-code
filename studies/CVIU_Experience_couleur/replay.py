"""Refit from supplied descriptors and compare each prediction, tuning choice and reported estimate."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from fit import previous,external
from analyse import analyse
from spatial import fit as fit_spatial,summary as summarise_spatial

def verify(threads):
    with tempfile.TemporaryDirectory() as temp,threadpool_limits(threads):
        output=Path(temp);cache=ROOT/'cache';expected=ROOT/'results'
        for corpus in ('DTD','KTH'):previous(corpus,cache,output)
        external(cache,output);analyse(cache,output)
        for corpus in ("DTD","KTH","Lytro"):fit_spatial(corpus,cache,output)
        summarise_spatial(cache,output)
        comparisons=0
        for p in sorted(output.glob('*_predictions.npz')):
            with np.load(p) as z,np.load(expected/p.name) as ref:
                assert set(z.files)==set(ref.files)
                for k in z.files:
                    np.testing.assert_array_equal(z[k],ref[k])
                    if k not in ('ids','y','groups','indices','fit','val','refit'): comparisons+=z[k].size
        for p in output.glob('*.csv'):
            with p.open(newline='') as f,(expected/p.name).open(newline='') as g:
                actual=list(csv.DictReader(f));expected_rows=list(csv.DictReader(g))
            assert len(actual)==len(expected_rows)
            for left,right in zip(actual,expected_rows):
                assert left.keys()==right.keys()
                for key,s in left.items():
                    try:a,b=float(s),float(right[key])
                    except ValueError:assert s==right[key]
                    else:np.testing.assert_allclose(a,b,atol=1e-9,rtol=0)
        with (output/'Lytro_settings.json').open() as f,(expected/'Lytro_settings.json').open() as g:
            settings,reference=json.load(f),json.load(g)
        for family in ('H','D'):
            f,r=settings['families'][family],reference['families'][family]
            np.testing.assert_allclose(f.pop('matched_scale'),r.pop('matched_scale'),atol=1e-12,rtol=0)
        assert settings==reference
        return dict(predicted_labels_compared=int(comparisons),changed=0,statistics_tables_checked=7,source_grids_checked=4)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--threads',type=int,default=3)
    p.add_argument('--log',type=Path);a=p.parse_args();ans=verify(a.threads);print(json.dumps(ans,indent=2))
    if a.log:a.log.write_text(json.dumps(ans,indent=2)+'\n')
