"""Descriptive cross-light retrieval with fixed material, specimen, pose and scale."""
import argparse
from pathlib import Path
import re
import numpy as np
from fit import rows,colour_maps,light_and_pose
from stats import write

def paired(data,out):
    rec=rows(data,'KTH');ids=rec['ids'];y=rec['y'];light,pose=light_and_pose(ids)
    scale=np.array([int(re.search(r'scale_(\d+)_im_',s).group(1)) for s in ids])
    sums={}
    for j in range(1,5):
        tr=rec['train'+str(j)];tr=tr[light[tr]!=3]
        te=rec['test'+str(j)]
        blocks=colour_maps(rec,tr)
        for key in ('RGB','Lab6','P','J','Lab_P','Lab_J','Lab_P_mean','Lab_J_mean','RGB_P','RGB_J'):
            desc=blocks[key];v=sums.setdefault(key,np.zeros(4))
            for s in np.unique(scale):
                for o in range(3):
                    front=te[(light[te]==0)&(scale[te]==s)&(pose[te]==o)]
                    ambient=te[(light[te]==3)&(scale[te]==s)&(pose[te]==o)]
                    front=front[np.argsort(y[front])];ambient=ambient[np.argsort(y[ambient])]
                    assert len(front)==len(ambient)==11
                    np.testing.assert_array_equal(y[ambient],np.arange(11));np.testing.assert_array_equal(y[front],y[ambient])
                    distance=np.maximum(0,1.-desc[front]@desc[ambient].T)
                    v[0]+=11;v[1]+=np.trace(distance);v[2]+=(distance.sum()-np.trace(distance))/10
                    wins=np.isclose(distance,distance.min(axis=1,keepdims=True),atol=1e-12,rtol=0)
                    v[3]+=(np.diag(wins)/wins.sum(axis=1)).sum()
    lines=[]
    for key,(n,within,between,wins) in sums.items():
        lines.append(dict(colour_block=key,matched_queries=int(n),other_material_comparisons=int(10*n),
            mean_same_specimen_distance=within/n,mean_other_material_distance=between/n,
            ratio_same_to_other=within/between,nearest_material_top1_pct=100*wins/n))
    write(out/'lighting_pairs.csv',lines)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();paired(a.data,a.output)
