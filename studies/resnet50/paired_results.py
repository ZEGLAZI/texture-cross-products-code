"""Pair all models on the same test identities and evaluate the two prespecified families."""
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
from cnn_models import load_npz,save_csv

DATASETS=("DTD","KTH","Lytro")
REGIMES=("fixed","matched","selected")
FAMILIES=("H","D","R")

def tables(root,decisions,out):
    sys.path.insert(0,str(root.resolve()))
    from summarize_pixel_check import paired_interval,macro
    sys.path.insert(0,str(root.resolve()/"studies/CVIU_Experience_couleur/code"))
    from analyse import ci_external
    results=[];contrasts=[];by_fold=[]
    for ds in DATASETS:
        cnn=load_npz(decisions/(ds+"_decisions.npz"))
        y=cnn["labels"];groups=cnn["groups"]
        if ds=="Lytro":
            old=load_npz(root/"studies/CVIU_Experience_couleur/results/Lytro_predictions.npz")
            np.testing.assert_array_equal(y[0],old["y"])
            np.testing.assert_array_equal(cnn["ids"][0],old["ids"])
            np.testing.assert_array_equal(groups[0],old["groups"])
        else:
            old=load_npz(root/"pixel_check"/(ds+"_predictions.npz"))
            for field in ("labels","groups","test","ids"):np.testing.assert_array_equal(cnn[field],old[field])
        def pred(family,regime,suffix):
            variant=regime if regime=="selected" or (regime=="matched" and suffix=="Q") else "fixed"
            if family=="R":
                prefix=variant+"_" if variant!="fixed" else ""
                return cnn[prefix+suffix]
            if ds=="Lytro":return old[family+"_"+variant+"_"+suffix][None,:]
            prefix={"selected":"tuned_","matched":"matched_","fixed":""}[variant]
            return old[family+"_"+prefix+suffix]
        def interval(delta,size):
            if ds=="Lytro":return ci_external(delta.ravel(),y.ravel(),groups.ravel(),size)[1:]
            return paired_interval(ds,delta,y,groups,size)
        deltas={}
        for regime in REGIMES:
            for family in FAMILIES:
                before=(pred(family,regime,"S")==y).astype(np.int16)
                after=(pred(family,regime,"Q")==y).astype(np.int16)
                delta=after-before;deltas[family,regime]=delta
                lo,hi=interval(delta,1)
                results.append(dict(dataset=ds,regime=regime,representation=family,
                  square_pct=macro(before,y),full_pct=macro(after,y),gain_pp=macro(delta,y),point_low_pp=lo,point_high_pp=hi))
                for i,labels in enumerate(y):
                    by_fold.append(dict(dataset=ds,fold=i+1,regime=regime,representation=family,
                        square_pct=macro(before[i:i+1],y[i:i+1]),full_pct=macro(after[i:i+1],y[i:i+1]),
                        corrected=int(np.count_nonzero(delta[i]>0)),introduced=int(np.count_nonzero(delta[i]<0))))
        hypotheses=[("primary","R_native_gain",deltas["R","fixed"]),
            ("primary","T_R_minus_H",deltas["R","selected"]-deltas["H","selected"]-deltas["R","fixed"]+deltas["H","fixed"]),
            ("secondary","R_minus_D_fixed",deltas["R","fixed"]-deltas["D","fixed"]),
            ("secondary","R_minus_D_selected",deltas["R","selected"]-deltas["D","selected"])]
        for family,contrast,delta in hypotheses:
            lo,hi=interval(delta,6)
            contrasts.append(dict(dataset=ds,family=family,contrast=contrast,estimate_pp=macro(delta,y),
                    adjusted_low_pp=lo,adjusted_high_pp=hi,comparisons_per_family=6,
                    inference_unit={"DTD":"image/duplicate group","KTH":"specimen within material","Lytro":"capture within material"}[ds]))
    out.mkdir(exist_ok=True,parents=True)
    for name,rows in [("gains",results),("contrasts",contrasts),("per_fold",by_fold)]:save_csv(out/(name+".csv"),rows)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("root","decisions","output"):p.add_argument(k,type=Path)
    a=p.parse_args();tables(a.root,a.decisions,a.output)
