"""Recompute every classifier and numerical table from the packaged feature arrays.
With --data, additionally compare an independent pixel extraction before refitting.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parent
TASKS=('DTD','KTH','omit_ambient','omit_front','omit_top','omit_side','Lytro')
CODES=ROOT/'code'

def equal_tables(path,reference):
    with path.open(newline='') as f,reference.open(newline='') as g:
        a,b=list(csv.DictReader(f)),list(csv.DictReader(g))
    if len(a)!=len(b):raise ValueError(path)
    for left,right in zip(a,b,strict=True):
        if left.keys()!=right.keys():raise AssertionError(path)
        for key,val in left.items():
            try:x,y=float(val),float(right[key])
            except ValueError:assert val==right[key],(path,key,val,right[key])
            else:
                if abs(x-y)>1e-9 or not np.isfinite(x+y):raise AssertionError((path,key,x,y))

def verify(data,threads,jobs):
    expected=ROOT/'results';baseline_labels=0;target_decisions=0;fit_count=0;array_values=0
    nimages=0;native_values=0
    if data.resolve()!=(ROOT/'data').resolve():
        for name in ('DTD','KTH','FMD','Lytro'):
            with np.load(data/(name+'_inputs.npz')) as measured,np.load(ROOT/'data'/(name+'_inputs.npz')) as expected_inputs:
                assert set(measured.files)==set(expected_inputs.files)
                for key in measured.files:
                    np.testing.assert_array_equal(measured[key],expected_inputs[key])
                    if key in ('F7','F8','rgb'):native_values+=measured[key].size
            p=name+'_new.npz'
            with np.load(data/p) as a,np.load(ROOT/'data'/p) as b:
                assert a.files==b.files
                for k in a.files:np.testing.assert_array_equal(a[k],b[k])
                nimages+=len(a['ids']);array_values+=a['edges'].size+a['lab'].size
    env=dict(os.environ,OPENBLAS_NUM_THREADS=str(threads),OMP_NUM_THREADS=str(threads))
    with tempfile.TemporaryDirectory() as d:
        out=Path(d)
        def run(task):
            subprocess.run([sys.executable,str(CODES/'fit.py'),task,str(data),str(out),'--threads',str(threads)],env=env,check=True)
        with ThreadPoolExecutor(jobs) as pool:list(pool.map(run,TASKS))
        single=dict(env,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        subprocess.run([sys.executable,str(CODES/'stats.py'),str(out)],env=single,check=True)
        subprocess.run([sys.executable,str(CODES/'lighting_pairs.py'),str(data),str(out)],env=single,check=True)
        for task in TASKS:
            p=task+'_predictions.npz'
            with np.load(out/p,allow_pickle=False) as a,np.load(expected/p,allow_pickle=False) as b:
                assert set(a.files)==set(b.files)
                for k in a.files:
                    np.testing.assert_array_equal(a[k],b[k])
                    if k not in {'ids','y','groups','indices','classes','light','pose'}:
                        target_decisions+=int(a[k].size);fit_count+=len(a[k])
                if task in ('KTH','DTD','Lytro'):baseline_labels+=a['AW_RGB'].size
            p=task+'_fits.json';new=json.loads((out/p).read_text());old=json.loads((expected/p).read_text())
            assert len(new)==len(old)
            for a,b in zip(new,old):
                for key in ('fold','train','test'):assert a[key]==b[key]
                assert a['scale'].keys()==b['scale'].keys()
                np.testing.assert_allclose(list(a['scale'].values()),list(b['scale'].values()),rtol=0.,atol=1e-12)
        tables=list(sorted(out.glob('*.csv')))
        for p in tables:equal_tables(p,expected/p.name)
    return dict(newly_reextracted_images=nimages,new_feature_coordinates_recompared=int(array_values),different_feature_coordinates=0 if nimages else None,native_coordinates_recompared=int(native_values),
                predicted_label_comparisons=target_decisions,different_labels=0,model_fold_outputs=fit_count,
                prior_RGB_labels_checked=int(baseline_labels),tables_verified=[p.name for p in tables])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=ROOT/'data')
    p.add_argument('--threads',type=int,default=3);p.add_argument('--jobs',type=int,default=1);p.add_argument('--log',type=Path)
    a=p.parse_args();result=verify(a.data,a.threads,a.jobs);print(json.dumps(result,indent=2))
    sys.path.insert(0,str(CODES))
    if (ROOT/'data/ALOT_inputs.npz').exists():
        from check_alot import verify as verify_followup
        result['independent_acquisition']=verify_followup(ROOT/'data/ALOT_inputs.npz',a.data/'ALOT_inputs.npz',ROOT/'results',equal_tables)
        print(json.dumps(result['independent_acquisition'],indent=2))
        from weight_control import run as fit_weight_control
        from threadpoolctl import threadpool_limits
        with tempfile.TemporaryDirectory() as temp,threadpool_limits(a.threads):
            produced=Path(temp)
            fit_weight_control(a.data,ROOT/'results',produced)
            equal_tables(produced/'weight_control.csv',ROOT/'results/weight_control.csv')
            name='weight_control_decisions.npz'
            with np.load(produced/name) as fresh,np.load(ROOT/'results'/name) as archived:
                assert set(fresh.files)==set(archived.files)
                total=0
                for key in fresh.files:
                    np.testing.assert_array_equal(fresh[key],archived[key])
                    if not key.endswith('_ids'):total+=fresh[key].size
            result['weight_control']=dict(label_comparisons=int(total),different_labels=0,contrasts=6)
        if (ROOT/'results/kernel_contrast.csv').exists():
            from check_interaction import verify as verify_kernels
            result['kernel_contrast']=verify_kernels(a.data,ROOT/'results',a.threads)
    if a.log:a.log.write_text(json.dumps(result,indent=2)+'\n')
