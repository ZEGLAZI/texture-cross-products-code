"""Refit the six restricted/profile-selected models on every outer fold in Table 8."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

def run(root,dest,threads):
    old=root.resolve()/'historical'; models=old/'revision/results/selection'
    sys.path[:0]=[str(old/'dtd'),str(old/'revision/code')]
    from predire import compact,load_blocks
    from nested import fit,kernel,makegrams,METHODS,PROFILES,PEN
    counts={}
    with np.load(old/'dtd/results/identites.npz') as ids:
        dtd_raw=load_blocks(root/'review2/dtd_cache/hsmt',ids['ids'],('F7','F8','rgb'));dtd_y=ids['y']
    with np.load(old/'external/data/kth.npz') as z:kth={k:z[k] for k in ('F7','F8','rgb')}
    with (old/'external/data/kth_images.csv').open() as f:kth_y=np.unique([r['materiau'] for r in csv.DictReader(f)],return_inverse=True)[1]
    flowers={k:[] for k in ('F7','F8','rgb','classe')}
    for ds in ('tf','plantnet','nouveau_2026'):
        with np.load(old/'external/data'/(ds+'.npz')) as z:
            for k in flowers:flowers[k].append(z[k])
    flowers={k:np.concatenate(v) for k,v in flowers.items()};flower_y=(flowers['classe'][:,None]==[10,12,9]).argmax(1)
    with threadpool_limits(threads):
        for ds,raw,y,files in [('Flowers',flowers,flower_y,[models/'Flowers.npz']),('KTH',kth,kth_y,sorted(models.glob('KTH_*.npz'))),('DTD',dtd_raw,dtd_y,sorted(models.glob('DTD_*.npz')))]:
            checked=0;choices=0
            for file in files:
                selected=json.loads(file.with_suffix('.json').read_text())
                with np.load(file) as archive:
                    train,test=archive['train'],archive['test']
                    grams=makegrams(raw,train,test,compact)
                    for i,method in enumerate(METHODS):
                        grid=archive['val_grid'][i]
                        j,l=np.unravel_index(np.argmax(grid),grid.shape)
                        if (j,PEN[l])!=(selected[method]['p'],selected[method]['lambda']):raise ValueError(file.stem+' '+method)
                        # Construct one kernel in row blocks to bound temporary storage.
                        complete=np.empty(grams[0].shape,dtype=np.result_type(*grams))
                        for first in range(0,len(complete),128):
                            block=slice(first,first+128)
                            complete[block]=kernel([g[block] for g in grams],PROFILES[j],method)
                        scores=fit(complete,len(train),y[train],PEN[l],len(np.unique(y)))
                        del complete
                        np.testing.assert_array_equal(scores.argmax(1),archive[method],err_msg=file.stem+' '+method)
                        checked+=len(test);choices+=1
                del grams
                print(file.stem,'checked',flush=True)
            counts[ds]={'test_decisions':checked,'selected_settings_checked':choices,'outer_fits':len(files)}
    dest.write_text(json.dumps(dict(status='passed',datasets=counts),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('r4',type=Path);p.add_argument('report',type=Path)
    p.add_argument('--threads',type=int,default=1)
    a=p.parse_args();run(a.r4,a.report,a.threads)
