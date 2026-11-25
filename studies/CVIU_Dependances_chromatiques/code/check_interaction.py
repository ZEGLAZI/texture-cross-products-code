"""Refit the secondary kernel analysis and check ordered decisions and intervals."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from threadpoolctl import threadpool_limits
from interaction import run

def verify(data, reference, threads=2):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from replay import equal_tables
    total = 0
    with TemporaryDirectory() as temp, threadpool_limits(threads):
        out = Path(temp)
        run(data, reference, out)
        equal_tables(out / 'kernel_contrast.csv', reference / 'kernel_contrast.csv')
        paths = sorted(out.glob('*_interaction.npz'))
        assert len(paths) == 6
        for path in paths:
            with np.load(path) as new, np.load(reference / path.name) as old:
                assert set(new.files) == set(old.files)
                for key in new.files:
                    np.testing.assert_array_equal(new[key], old[key])
                    if key!='ids':
                        total += new[key].size
    return dict(ordered_label_checks=int(total),different_labels=0,intervals=12,protocols=6)

if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data','reference','report'):
        p.add_argument(key,type=Path)
    p.add_argument('--threads',type=int,default=2)
    args=p.parse_args()
    answer=verify(args.data,args.reference,args.threads)
    args.report.write_text(json.dumps(answer,indent=2)+'\n')
