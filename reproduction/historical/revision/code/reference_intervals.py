"""Full-field DINO comparisons from the ten DTD test assignments."""
from pathlib import Path
import argparse,json
import numpy as np
from inference import dtd_interval
def main(a):
    f=[np.load(a.refs/f'fold_{j:02}.npz') for j in range(1,11)];te=np.stack([q['test'] for q in f]);p=np.load(a.supp/'dtd/results/identites.npz');y=p['y'][te];g=p['groupes'][te];out={}
    for stage in ('fixed','selected'):
        z={m:(np.stack([q[m+'_'+stage] for q in f])==y).astype('int8') for m in ('full','full_add','full_W')}
        out[stage]={m:dtd_interval(z['full_W']-z[m],y,g,m=2) for m in ('full','full_add')}
    (a.refs/'intervals.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('supp',type=Path);p.add_argument('refs',type=Path);main(p.parse_args())
