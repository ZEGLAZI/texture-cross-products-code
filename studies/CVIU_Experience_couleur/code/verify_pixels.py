"""Compare a separate pixel extraction to the published numerical cache, field by field."""
import argparse
import json
from pathlib import Path
import numpy as np

def verify(reference,candidate):
    datasets=('DTD','KTH','FMD','Lytro')
    report={}
    mismatches=0
    for corpus in datasets:
        fields=('palettes','pairs','features') if corpus in ('FMD','Lytro') else ('palettes','pairs')
        for kind in fields:
            filename=f'{corpus}_{kind}.npz'
            with np.load(reference/filename) as truth,np.load(candidate/filename) as actual:
                if set(truth.files)!=set(actual.files):raise ValueError(filename+' columns')
                records={}
                for k in truth.files:
                    x,y=truth[k],actual[k]
                    if x.dtype!=y.dtype or x.shape!=y.shape:raise ValueError(filename+' '+k)
                    count=int(np.count_nonzero(x!=y))
                    if count or k in ('values','F7','F8','patch3','patch6','patch12'):
                        records[k]={'coordinates':int(x.size),'differing':count}
                    mismatches+=count
                report[filename]=records
    return dict(source_images=12392,external_images_with_both_representations=2000,
                array_values_differing=mismatches,details=report)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('reference','candidate','output'):p.add_argument(k,type=Path)
    a=p.parse_args();report=verify(a.reference,a.candidate);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(report['array_values_differing'],'different values')
    if report['array_values_differing']:raise SystemExit(1)
