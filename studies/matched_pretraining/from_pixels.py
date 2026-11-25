"""Recreate the held-out product experiment from the sources and check its published evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
PROVIDER={'labels.zip':'aa911ac1753f906df292a5a7a77aa32c029d3e5c09414772c64fff6a643d59b9',
          'masks.zip':'8898baa9975749cbf3ae694f0e8c11ac12d907ee62ede7af9099e559f0234f3d',
          'images.zip':'f27aa9f1a7e024f0d88df871c419543390e74c96a38261d3d69e237983179cb5'}

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(a):
    root=a.package.resolve();inputs=a.inputs.resolve();output=a.output.resolve()
    for filename,expected in PROVIDER.items():
        if sha(inputs/filename)!=expected:raise ValueError('Provider file differs: '+filename)
    output.mkdir(exist_ok=True,parents=True)
    def call(name,*args):
        subprocess.run([sys.executable,'-B',str(HERE/name),*map(str,args)],check=True)
    reference=HERE/'audit'
    prior=a.history_index.resolve() if a.history_index else reference/'history_screen.json'
    garments=output/'garments';features=output/'features';fits=output/'results'
    call('garments.py','prepare',inputs,prior,garments,'--threads',a.threads)
    cohort=json.loads((garments/'cohort.json').read_text())
    expected=json.loads((reference/'cohort.json').read_text())
    if cohort['manifest_sha256']!=expected['manifest_sha256']:raise ValueError('The catalogue split differs from the reference')
    call('extract_garments.py',root,garments,inputs,features,'--threads',a.threads)
    call('fit_garments.py',root,garments,features/'features.npz',fits,
         '--threads',2,'--private-decisions',output/'garment_decisions.npz','--independent')
    with (fits/'garment_test_evidence.csv').open() as f,(HERE/'results/garment_test_evidence.csv').open() as r:
        if list(csv.reader(f))!=list(csv.reader(r)):raise ValueError('Held-out product evidence differs')
    from replay import compare_rows
    for name in ('garment_validation','garment_settings','garment_development_precision'):
        compare_rows(fits/(name+'.csv'),HERE/'results'/(name+'.csv'))
    digest=sha(fits/'garment_test_evidence.csv')
    report={'manifest_sha256':cohort['manifest_sha256'],'test_evidence_sha256':digest,
            'matched_test_products':len((fits/'garment_test_evidence.csv').read_text().splitlines())-1,
            'status':'passed'}
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified catalogue assignments, validation scores, and all product-level comparisons.')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('package','inputs','output'):p.add_argument(k,type=Path)
    p.add_argument('--threads',default=3,type=int);p.add_argument('--history-index',type=Path,
        help='Optional independently reconstructed historical photo-signature index; by default reuse the locked cross-corpus screening record.')
    main(p.parse_args())
