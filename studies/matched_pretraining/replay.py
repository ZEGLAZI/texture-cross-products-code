"""Verify manuscript estimates, the predeclared comparisons, and optionally refit frozen features."""
import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from analyse_extension import analyze

def rows(path):
    with path.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))

def compare_rows(a,b):
    x,y=rows(a),rows(b)
    if len(x)!=len(y) or not x:raise ValueError(f'Row count differs: {a}')
    for i,(first,second) in enumerate(zip(x,y,strict=True)):
        if first.keys()!=second.keys():raise ValueError(f'Fields differ: {a}')
        for key,value in first.items():
            want=second[key]
            try:f,g=float(value),float(want)
            except ValueError:
                if value==want:continue
            else:
                if math.isclose(f,g,abs_tol=1e-10,rel_tol=0.):continue
            raise ValueError(f'{a}, row {i+1}, {key}: {value} vs {want}')
    return len(x)

def replay(root,here,output,mode,features):
    from check_fits import run as independent
    import shutil
    output.mkdir(exist_ok=True,parents=True)
    results=here/'results';fresh=output/'summaries'
    analyze(root,results,fresh)
    checked={name:compare_rows(fresh/(name+'.csv'),results/(name+'.csv'))
             for name in ('gains','contrasts','garment_classes','garment_image_macro')}
    retrospective=output/'retrospective'
    subprocess.run([sys.executable,'-B',str(here/'retrospective.py'),str(root),str(retrospective)],check=True)
    for name in ('contrast','validation'):compare_rows(retrospective/(name+'.csv'),here/'retrospective_results'/(name+'.csv'))
    extra={}
    if mode=='full':
        for encoder in ('dino_r50','dino_vits16'):
            if not (features/encoder/'DTD_stages.npz').is_file():
                raise FileNotFoundError('Restore the additional_features/ directory from IVC_Reviewer.zip or specify --features.')
            for dataset in ('DTD','KTH','Lytro'):
                check=independent(root,features/encoder,results/encoder,dataset)
                reference=json.loads((here/'audit'/(encoder+'_'+dataset+'.json')).read_text())
                if check!=reference:raise ValueError('Refit coverage differs')
                extra[encoder+'_'+dataset]=check
    return {'summary_rows':checked,'paired_test_products':len(rows(results/'garment_test_evidence.csv')),
            'independent_refits':extra,'status':'passed'}
def main():
    here=Path(__file__).resolve().parent;p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,default=here.parents[1])
    p.add_argument('--features',type=Path);p.add_argument('--mode',choices=['tables','full'],default='tables')
    p.add_argument('--report',type=Path);p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    root=a.package.resolve();features=(a.features or root/'additional_features').resolve()
    with tempfile.TemporaryDirectory() as temp,threadpool_limits(a.threads):
        report=replay(root,here,Path(temp),a.mode,features)
    if a.report:a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
