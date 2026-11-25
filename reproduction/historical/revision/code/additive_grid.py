"""Adversarially generous additive comparison. The simplex lattice includes pure descriptors."""
import argparse,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import nested
def main(a):
    previous=nested.PROFILES.copy()
    nested.PROFILES=np.vstack([previous,[[i/5,j/5,(5-i-j)/5] for i in range(6) for j in range(6-i)]])
    assert len(nested.PROFILES)==25
    nested.METHODS=['Add'];a.output.mkdir(exist_ok=True,parents=True)
    (a.output/'profiles.json').write_text(json.dumps(nested.PROFILES.tolist())+'\n')
    with threadpool_limits(a.threads):nested.run(a)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('task',choices=['DTD','Flowers'])
    for n in ('supp','cache','output'):p.add_argument(n,type=Path)
    p.add_argument('--threads',type=int,default=3);main(p.parse_args())
