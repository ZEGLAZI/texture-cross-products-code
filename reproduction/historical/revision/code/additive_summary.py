"""Compare the broader, validation-selected additive model to the matched-budget models."""
from pathlib import Path
import argparse,numpy as np,json
from inference import dtd_interval
def run(a):
    orig=np.load(a.supp/'dtd/results/identites.npz');sel=[np.load(a.selected/f'DTD_{k:02}.npz') for k in range(1,11)];te=np.stack([r['test'] for r in sel]);y=orig['y'][te];groups=orig['groupes'][te]
    add=np.stack([np.load(a.broader/f'DTD_{k:02}.npz')['Add'] for k in range(1,11)]);out={'recall':float(100*(add==y).mean())}
    for name in ('Add','W'):
        z=(np.stack([r[name] for r in sel])==y).astype('int8')-(add==y);out[name+'-broad']=dtd_interval(z,y,groups,2)
    (a.broader/'scores.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('supp','selected','broader'):p.add_argument(n,type=Path)
    run(p.parse_args())
