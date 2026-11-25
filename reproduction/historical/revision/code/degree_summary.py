"""Exploratory paired DTD decomposition results. Family of two contrasts for squares."""
import argparse,json
from pathlib import Path
import numpy as np
from inference import dtd_interval
def main(a):
    ids=np.load(a.supp/'dtd/results/identites.npz');te=np.load(a.supp/'dtd/results/decisions.npz')['tests'];y=ids['y'][te];g=ids['groupes'][te]
    def gather(folder,name):return np.stack([np.load(folder/f'DTD_{i:02}.npz')[name] for i in range(1,11)])
    pred={k:gather(a.degrees,k) for k in ('squares','crosses')};pred.update({k:gather(a.selection,k) for k in ('W','Q')})
    z={k:(v==y).astype('int8') for k,v in pred.items()};summary={k:float(v.mean()*100) for k,v in z.items()}
    for k in ('crosses','Q'):summary['squares-'+k]=dtd_interval(z['squares']-z[k],y,g,2)
    (a.degrees/'scores.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('supp','degrees','selection'):p.add_argument(n,type=Path)
    main(p.parse_args())
