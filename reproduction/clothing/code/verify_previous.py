"""Verify classifier and kernel parity before evaluating the newly locked partitions."""
import numpy as np
from threadpoolctl import threadpool_limits
from evaluate_partitions import ROOT,outer_fold,representations
def check():
    raw,fixed,keys,y,_=representations()
    with np.load(ROOT/'data/previous_check.npz') as old:
        assert np.array_equal(keys,old['keys'])
        for k in range(5):
            test,pred,_,_=outer_fold(old['fold'],k,raw,fixed,y,('squares','crosses','additive'))
            np.testing.assert_array_equal(pred,old['pred'][:,test])
    print('All 633 prior decisions reproduced exactly.')
if __name__=='__main__':
    with threadpool_limits(2):
        check()
