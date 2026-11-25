"""Fit the frozen CNN representation using the published image and specimen partitions."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import eigh, solve
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from numerical_release import ReleasedPayload

STAGES=("layer2","layer3","layer4")
PENALTIES=(1.,.1,10.,.01,100.,.001)
METHODS=("S","Q","matched_Q","selected_S","selected_Q")

def load_npz(path):
    with np.load(path,allow_pickle=False) as archive:
        return dict(archive)

def acquisition(root, features, name):
    cache=root/"studies/CVIU_Experience_couleur/cache"
    item=load_npz(features/(name+"_stages.npz"))
    design=load_npz(cache/(name+"_design.npz"))
    with np.load(cache/(name+"_palettes.npz")) as palette:
        np.testing.assert_array_equal(item["ids"],palette["ids"])
        np.testing.assert_array_equal(item["hash224"],palette["hash224"])
    np.testing.assert_array_equal(item["ids"],design["ids"])
    return item,design

def load_task(root, features, name):
    if name=="Lytro":
        fmd,source=acquisition(root,features,"FMD")
        lytro,target=acquisition(root,features,"Lytro")
        np.testing.assert_array_equal(source["classes"],target["classes"])
        reference=load_npz(root/"studies/CVIU_Experience_couleur/results/Lytro_predictions.npz")
        n=len(source["y"])
        raw=[np.concatenate([fmd[k],lytro[k]]) for k in STAGES]
        ids=np.concatenate([fmd["ids"],lytro["ids"]])
        hashes=np.concatenate([fmd["hash224"],lytro["hash224"]])
        if len(np.unique(hashes))!=len(hashes):raise ValueError("Repeated source or target image")
        y=np.concatenate([source["y"],target["y"]])
        np.testing.assert_array_equal(target["ids"],reference["ids"])
        np.testing.assert_array_equal(target["y"],reference["y"])
        folds=[("FMD_to_Lytro",reference["refit"],np.arange(n,len(y)),[(reference["fit"],reference["val"])])]
        groups=reference["groups"][None,:]
    else:
        record,design=acquisition(root,features,name)
        raw=[record[k] for k in STAGES]
        y=design["y"];ids=record["ids"];hashes=record["hash224"]
        ref=load_npz(root/"pixel_check"/(name+"_predictions.npz"))
        names=[f"DTD_{i:02}" for i in range(1,11)] if name=="DTD" else ["KTH_"+v for v in "abcd"]
        groups=ref["groups"];folds=[]
        with ReleasedPayload(root) as z:
            for i,label in enumerate(names):
                f=load_npz(io.BytesIO(z.read("review2/results/kernels/"+label+".npz")))
                train=f["train"];test=f["test"]
                np.testing.assert_array_equal(train,design["train"+str(i+1)])
                np.testing.assert_array_equal(test,design["test"+str(i+1)])
                np.testing.assert_array_equal(ids[test],ref["ids"][i])
                np.testing.assert_array_equal(y[test],ref["labels"][i])
                local=[(f["inner_train_"+str(j)],f["inner_val_"+str(j)]) for j in range(1 if name=="DTD" else 3)]
                folds.append((label,train,test,local))
    for _,train,test,inner in folds:
        if np.intersect1d(hashes[train],hashes[test]).size:raise ValueError("Train/test pixels overlap")
        for fit,val in inner:
            if np.intersect1d(hashes[fit],hashes[val]).size:raise ValueError("Fit/validation pixels overlap")
            if not np.isin(np.r_[fit,val],train).all():raise ValueError("Validation outside outer-training pool")
    return ids,y,raw,groups,folds

def kernels(raw, fitting_sample, train, evaluation):
    """Train-only coordinate scaling followed by the two equal-weight similarities."""
    indices=np.r_[train,evaluation]
    linear=np.zeros((len(indices),len(train)),dtype=np.float64)
    squared=np.zeros_like(linear)
    for layer in raw:
        sample=layer[fitting_sample].astype(np.float64)
        mu=sample.mean(axis=0)
        sd=np.maximum(sample.std(axis=0),1e-4)
        values=layer[indices].astype(np.float64)
        values-=mu
        values/=sd
        np.clip(values,-12.,12.,out=values)
        lengths=np.linalg.norm(values,axis=1,keepdims=True)
        np.divide(values,lengths,out=values,where=lengths!=0)
        gram=values @ values[:len(train)].T
        linear+=gram
        squared+=gram*gram
    squares=((8./3.)*linear+squared/3.)/13.
    full=((8./3.)*linear+linear**2/3.)/13.
    return squares,full

def macro(labels, y):
    return np.mean([np.mean(labels[y==c]==c) for c in np.unique(y)])

def classify(kernel,y,classes,penalty,counts=None):
    n=len(y)
    counts=np.ones(n,dtype=int) if counts is None else counts
    target=np.eye(classes)[y]
    prior=np.average(target,axis=0,weights=counts)
    matrix=kernel[:n].copy(order="F")
    matrix.flat[::n+1]+=penalty/counts
    coef=solve(matrix,target-prior,assume_a="pos",overwrite_a=True,check_finite=False)
    return (kernel[n:] @ coef+prior).argmax(axis=1).astype(np.int16)

def validation(kernel, yfit, yeval, classes):
    n=len(yfit);target=np.eye(classes)[yfit];prior=target.mean(axis=0)
    vals,vecs=eigh(kernel[:n],driver="evd",check_finite=False)
    left=kernel[n:] @ vecs
    right=vecs.T @ (target-prior)
    return np.array([macro((left @ (right/(vals[:,None]+p))+prior).argmax(axis=1),yeval) for p in PENALTIES])

def choose(scores):
    return PENALTIES[np.flatnonzero(scores>=max(scores)-1e-13)[0]]

def save_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def fit_task(root,features,name,out):
    ids,y,raw,groups,folds=load_task(root,features,name)
    classes=len(np.unique(y)); records=[];settings=[]
    results={k:[] for k in ("test","ids","labels",*METHODS)}
    for fold,train,test,inner in folds:
        grid=np.zeros((2,len(PENALTIES)))
        for inner_index,(fit,val) in enumerate(inner):
            pair=kernels(raw,fit,fit,val)
            for j,k in enumerate(pair):
                scores=validation(k,y[fit],y[val],classes)
                grid[j]+=scores/len(inner)
                for p,value in zip(PENALTIES,scores,strict=True):
                    records.append(dict(fold=fold,inner=inner_index,kernel=("S","Q")[j],penalty=p,macro_recall_pct=100*value))
            del k,pair
        s,q=kernels(raw,train,train,test)
        n=len(train)
        diag_s=np.diagonal(s[:n]);diag_q=np.diagonal(q[:n])
        factor=float(diag_s.mean()/diag_q.mean())
        ps,pq=map(choose,grid)
        settings.append(dict(fold=fold,fit_images=n,test_appearances=len(test),lambda_S=ps,lambda_Q=pq,
             mean_diag_S=float(diag_s.mean()),mean_diag_Q=float(diag_q.mean()),matched_factor=factor,
             unit_block_diagonal_S_error=float(np.max(np.abs(diag_s-9/13))),unit_block_diagonal_Q_error=float(np.max(np.abs(diag_q-11/13)))))
        values={"test":test,"ids":ids[test],"labels":y[test]}
        for key,kernel,penalty in [("S",s,1.),("Q",q,1.),("matched_Q",q,1./factor),("selected_S",s,ps),("selected_Q",q,pq)]:
            values[key]=classify(kernel,y[train],classes,penalty)
        for key,data in values.items():results[key].append(data)
        print("Fitted",fold,flush=True)
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/(name+"_decisions.npz"),groups=groups,**{k:np.stack(v) for k,v in results.items()})
    save_csv(out/(name+"_validation.csv"),records)
    save_csv(out/(name+"_settings.csv"),settings)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset",choices=("DTD","KTH","Lytro"))
    for key in ("root","features","output"):p.add_argument(key,type=Path)
    p.add_argument("--threads",type=int,default=2)
    a=p.parse_args()
    with threadpool_limits(a.threads):fit_task(a.root,a.features,a.dataset,a.output)
