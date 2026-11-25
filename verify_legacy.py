"""Refit the textile and flower models and regenerate the radial simulation from the r4 sources."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import numpy as np

def same_npz(actual, reference):
    with np.load(actual, allow_pickle=False) as a, np.load(reference, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError(f"Array names differ in {actual.name}")
        labels = floats = 0
        for field in b.files:
            fresh, expected = a[field], b[field]
            if expected.dtype.kind == 'f':
                np.testing.assert_allclose(fresh,expected,rtol=1e-8,atol=1e-9,err_msg=field)
                floats += fresh.size
            else:
                np.testing.assert_array_equal(fresh,expected,err_msg=field)
                if fresh.dtype.kind in 'iu' and fresh.ndim==1 and field not in ('test','train','y','labels'):
                    labels += fresh.size
    return {"integer_labels_and_indices_checked_in_1d_arrays":labels,"floating_point_values_checked":floats}

def run(base, out):
    root=base.resolve()
    env=dict(os.environ,PYTHONUTF8="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    def call(script,*args,cwd=None):
        subprocess.run([sys.executable,"-B",str(script),*map(str,args)],cwd=cwd,env=env,check=True)
    historical=root/'historical'
    with TemporaryDirectory(prefix="texture-legacy-") as folder:
        tmp=Path(folder)
        clothing=tmp/'clothing'
        shutil.copytree(root/'clothing',clothing)
        call(clothing/'code/evaluate_partitions.py',cwd=clothing)
        call(clothing/'code/check_results.py',cwd=clothing)
        original=root/'clothing/results/predictions.csv'
        fitted=clothing/'results/predictions.csv'
        if fitted.read_bytes()!=original.read_bytes():raise ValueError("Clothing decisions differ")
        with fitted.open() as stream: rows=list(csv.DictReader(stream))
        call(historical/'external/source_transfer.py',historical/'external/data')
        call(historical/'external/refit.py','--pipeline',historical/'dtd','--data',historical/'external/data',
             '--reference',historical/'external/results/pivots_archives.npz','--output',tmp/'external')
        cohort=tmp/'cohort'
        ref=historical/'revision/results/cohort'
        cohort.mkdir()
        for filename in ('nouveau_manifeste.csv','features.npz'):
            shutil.copy2(ref/filename,cohort/filename)
        call(historical/'revision/code/flowers_new.py',historical,cohort,tmp,historical/'revision/results/selection/Flowers.json')
        comparisons=same_npz(cohort/'predictions.npz',ref/'predictions.npz')
        if json.loads((cohort/'scores.json').read_text())!=json.loads((ref/'scores.json').read_text()):
            raise ValueError("Flower statistics differ")
        call(root/'review2/code/replay.py','simulate','--output',tmp/'sim')
        sim=tmp/'sim/synthetic.csv'
        target=root/'review2/results/radial/synthetic.csv'
        if not target.is_file():target=next(p for p in (root/'review2').rglob('synthetic.csv'))
        with sim.open() as f1,target.open() as f2:
            a,b=list(csv.DictReader(f1)),list(csv.DictReader(f2))
        if len(a)!=len(b) or not a:raise ValueError("Simulation size differs")
        for row,reference in zip(a,b,strict=True):
            if row.keys()!=reference.keys():raise ValueError("Simulation columns differ")
            for k,x in row.items():
                try: target_value=float(reference[k])
                except ValueError:
                    if x!=reference[k]:raise ValueError(k)
                else:np.testing.assert_allclose(float(x),target_value,rtol=1e-10,atol=1e-10)
        report={'status':'passed','clothing':{'test_rows':len(rows),'models':7,'sha256':hashlib.sha256(fitted.read_bytes()).hexdigest()},
                'flower_source_transfer':{'target_photos':[750,750,1070,711],'exact_label_check':True},
                'external_restricted_refits':['Pl@ntNet','iNaturalist 2026','KTH specimens'],
                'additional_flower_cohort':dict(photos=420,**comparisons),'radial_simulation_rows':len(a)}
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path,help='Extracted data_and_code_r4.zip')
    p.add_argument('report',type=Path)
    a=p.parse_args();print(json.dumps(run(a.data,a.report),indent=2))
