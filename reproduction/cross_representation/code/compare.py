"""Evaluate fixed kernels and paired contrasts on the HSMT outer test splits."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

METHODS = ("L", "S", "X", "Q")
SEED = 19112026


def metadata(corpus, data):
    if corpus == "DTD":
        with np.load(data / "historical/dtd/results/identites.npz") as f:
            return f["ids"], f["y"], f["groupes"], f["groupes"], f["traits"]
    with (data / "historical/external/data/kth_images.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    names, y = np.unique([r["materiau"] for r in rows], return_inverse=True)
    keys = np.array([r["chemin"] for r in rows])
    specimens = np.array([r["materiau"] + "/" + r["specimen"] for r in rows])
    fingerprints = np.array([r["sha_pixels"] for r in rows])
    return keys, y, specimens, fingerprints, names


def norm(x):
    lengths = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, lengths, out=np.zeros_like(x), where=lengths > 0)


def normalize_block(x, train):
    x = x.astype(np.float64)
    mean = x[train].mean(axis=0)
    sd = np.maximum(x[train].std(axis=0), 1e-4)
    return norm(np.clip((x - mean) / sd, -12., 12.))


def select_penalties(features, splits, y, groups, hashes, outer_train, fit, kernels, penalties):
    grids = {name:np.zeros(len(penalties)) for name in ("S","Q")}
    for tr,val in splits:
        if not np.isin(np.r_[tr,val],outer_train).all() or np.intersect1d(groups[tr],groups[val]).size or np.intersect1d(hashes[tr],hashes[val]).size:
            raise ValueError("Inner split is not isolated by source.")
        order = np.r_[tr,val]
        blocks = [normalize_block(features[k],tr) for k in ("patch3","patch6","patch12")]
        grams = [b[order] @ b[tr].T for b in blocks]
        diagonals = [np.einsum("ij,ij->i",b[order],b[order]) for b in blocks]
        for name,k in kernels(grams,diagonals,1):
            if name in grids:
                for j,pen in enumerate(penalties):
                    p = fit(k,len(tr),y[tr],pen,int(y.max())+1).argmax(1)
                    grids[name][j] += np.mean([np.mean(p[y[val]==c]==c) for c in np.unique(y[val])]) / len(splits)
            if name=="Q":
                break
    return {name:float(penalties[np.argmax(values)]) for name,values in grids.items()}, grids


def fold_predictions(npz_path, features, data, corpus, y, groups, hashes, fit, kernels, penalties):
    with np.load(npz_path) as f:
        tr, te = f["train"], f["test"]
        truth = f["labels"]
        np.testing.assert_array_equal(truth, y[te])
        assert tuple(f["modes"][:4]) == METHODS
        predictions = {"H_" + name: arr for name, arr in zip(METHODS, f["fixed"][:,1], strict=True)}
        predictions["H_unit_S"], predictions["H_unit_Q"] = f["unit"][[1,3],1]
        predictions["H_p0_S"], predictions["H_p0_Q"] = f["fixed"][[1,3],0]
        predictions["H_tuned_S"], predictions["H_tuned_Q"] = f["tuned"][[1,3],1]
        h_pens = {name:float(penalties[np.argmax(f["grid"][i,1])]) for name,i in (("S",1),("Q",3))}
        inner = [(f["inner_train_"+str(i)],f["inner_val_"+str(i)])
                 for i in range(1 if corpus=="DTD" else 3)]
    if np.intersect1d(groups[tr], groups[te]).size or np.intersect1d(hashes[tr], hashes[te]).size:
        raise ValueError("Shared source group across train and test")
    expected_counts = 40 if corpus == "DTD" else 108
    np.testing.assert_array_equal(np.bincount(truth), np.full(int(y.max())+1,expected_counts))
    order = np.r_[tr,te]
    n = len(tr)

    d_pens, d_grids = select_penalties(features, inner, y, groups, hashes, tr, fit, kernels, penalties)

    def predict(k,penalty=1.):
        return fit(k, n, y[tr], penalty, int(y.max())+1).argmax(1).astype(np.int16)

    def gram(x):
        return x[order] @ x[tr].T

    blocks = [normalize_block(features[k],tr) for k in ("patch3","patch6","patch12")]
    grams = [gram(x) for x in blocks]
    diagonals = [np.einsum("ij,ij->i",x[order],x[order]) for x in blocks]
    # All DINO block vectors are nonzero; check this explicitly before scalar diagonal scaling.
    for diagonal in diagonals:
        np.testing.assert_allclose(diagonal,1.,rtol=0.,atol=2e-12)
    for name, k in kernels(grams,diagonals,1):
        if name not in METHODS:
            raise ValueError(f"Unexpected kernel before all fixed kernels were processed: {name}")
        predictions["D_"+name] = predict(k)
        if name in ("S","Q"):
            predictions["D_tuned_"+name] = predictions["D_"+name] if d_pens[name]==1. else predict(k,d_pens[name])
            d = (2 + .75 * (1/3 if name=="S" else 1)) / 3.25
            np.testing.assert_allclose(np.diag(k[:n]), d, atol=2e-12, rtol=0.)
            predictions["D_unit_"+name] = predict(k/d)
        if name == "Q":
            break
    del grams, diagonals, k
    predictions["D_patch12_linear"] = predict(gram(blocks[-1])*(2/3.25))
    predictions["D_cls12_linear"] = predict(gram(normalize_block(features["cls12"],tr))*(2/3.25))
    parity = None
    if corpus == "DTD":
        predictions["D_cls12_historical"] = predict(gram(norm(features["cls12"].astype(np.float64))))
        fold = int(npz_path.stem.split("_")[1])
        ref = data / f"historical/revision/results/references/fold_{fold:02d}.npz"
        with np.load(ref) as f:
            np.testing.assert_array_equal(f["test"],te)
            parity = int(np.count_nonzero(predictions["D_cls12_historical"] != f["full_fixed"]))
    tuning = dict(fold=npz_path.stem, H=h_pens, D=d_pens, lambda_grid=list(penalties),
                  D_inner_macro_recall_fraction={k:v.tolist() for k,v in d_grids.items()})
    return dict(test=te, labels=truth, groups=groups[te], **predictions), parity, tuning


def aggregate(corpus, results, class_names, out, dtd_interval, external):
    labels = results["labels"]
    nc = len(class_names)
    groups = results["groups"]
    methods = [n for n in results if n not in ("labels","test","groups","ids")]
    correct = {n:(results[n]==labels).astype(np.int8) for n in methods}
    scores = {n:float(100*ok.mean()) for n,ok in correct.items()}  # Equal class counts in every split.
    class_rows = []
    for c,name in enumerate(class_names):
        select = labels==c
        row = dict(category=str(name), test_appearances=int(select.sum()))
        for fam in ("H","D"):
            s,q = correct[fam+"_S"][select], correct[fam+"_Q"][select]
            for mode in METHODS:
                row[fam+"_"+mode] = float(correct[fam+"_"+mode][select].mean()*100)
            row[fam+"_cross_gain_pp"] = float((q-s).mean()*100)
            row[fam+"_corrected"] = int(np.sum(q>s))
            row[fam+"_introduced"] = int(np.sum(q<s))
        row["D_minus_H_gain_pp"] = row["D_cross_gain_pp"]-row["H_cross_gain_pp"]
        class_rows.append(row)
    write_csv(out/f"{corpus}_classes.csv",class_rows)
    for key in ("D_Q","H_Q","D_S","H_S"):
        np.testing.assert_allclose(np.mean([r[key] for r in class_rows]),scores[key],atol=1e-12)

    def interval(diff, m=1):
        if corpus=="DTD":
            values = dtd_interval(diff,labels,groups,m=m,n=100000,seed=SEED)
        else:
            values = external(diff.ravel(),labels.ravel(),groups.ravel(),m=m,n=100000)
        np.testing.assert_allclose(values[0],np.mean(diff)*100,atol=1e-12)
        return dict(estimate_pp=float(values[0]), interval_pp=[float(v) for v in values[1:]])

    h = correct["H_Q"]-correct["H_S"]
    d = correct["D_Q"]-correct["D_S"]
    unit_h = correct["H_unit_Q"]-correct["H_unit_S"]
    unit_d = correct["D_unit_Q"]-correct["D_unit_S"]
    tuned_h = correct["H_tuned_Q"]-correct["H_tuned_S"]
    tuned_d = correct["D_tuned_Q"]-correct["D_tuned_S"]
    effects = dict(HSMT=interval(h), DINO=interval(d), DINO_minus_HSMT=interval(d-h),
                   DINO_minus_HSMT_adjusted_two_corpora=interval(d-h,m=2),
                   H_original_profile=interval(correct["H_p0_Q"]-correct["H_p0_S"]),
                   unit_H=interval(unit_h), unit_D=interval(unit_d),
                   unit_D_minus_H=interval(unit_d-unit_h),
                   tuned_H=interval(tuned_h),tuned_D=interval(tuned_d),tuned_D_minus_H=interval(tuned_d-tuned_h),
                   add_squares_H=interval(correct["H_Q"]-correct["H_X"]),
                   add_squares_D=interval(correct["D_Q"]-correct["D_X"]))
    transitions = {}
    for fam in ("H","D"):
        base,full = results[fam+"_S"],results[fam+"_Q"]
        before,after = correct[fam+"_S"],correct[fam+"_Q"]
        transitions[fam] = dict(corrected=int(np.sum(after>before)),introduced=int(np.sum(after<before)),
                                changed_label=int(np.sum(base!=full)))
    rows = []
    for i in range(len(labels)):
        row = {"fold":i+1 if corpus=="DTD" else "abcd"[i],
               "n_test":len(labels[i]),"H_gain_pp":float(h[i].mean()*100),
               "D_gain_pp":float(d[i].mean()*100),"D_minus_H_gain_pp":float((d[i]-h[i]).mean()*100)}
        row.update({k:float(ok[i].mean()*100) for k,ok in correct.items()})
        rows.append(row)
    write_csv(out/f"{corpus}_folds.csv",rows)
    return dict(corpus=corpus, folds=len(labels), test_appearances=int(labels.size),
                distinct_test_ids=int(np.unique(results["test"]).size),
                represented_groups=int(np.unique(groups).size), classes=nc,
                macro_recall_percent=scores, changes_pp=effects, cross_product_decisions=transitions)


def write_csv(path,rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)


def main(args):
    code = args.data.resolve()/"review2/code"
    sys.path.insert(0,str(code))
    from assess import kernels
    from nested import PROFILES, PEN, fit
    from inference import dtd_interval, external
    np.testing.assert_array_equal(PROFILES[1],[1/3]*3)
    keys,y,groups,hashes,names = metadata(args.corpus,args.data)
    with np.load(args.features) as f:
        np.testing.assert_array_equal(f["ids"],keys)
        features = {k:f[k] for k in ("patch3","patch6","patch12","cls12")}
    args.out.mkdir(parents=True,exist_ok=True)
    paths = sorted((args.data/"review2/results/kernels").glob(args.corpus+"_*.npz"))
    assert len(paths) == (10 if args.corpus=="DTD" else 4)
    all_folds,parities,tuning = [],[],[]
    with threadpool_limits(args.threads):
        for path in paths:
            fold,parity,settings = fold_predictions(path,features,args.data,args.corpus,y,groups,hashes,fit,kernels,PEN)
            all_folds.append(fold)
            tuning.append(settings)
            if parity is not None:
                parities.append(parity)
            print(path.stem,"complete",flush=True)
        results = {k:np.stack([r[k] for r in all_folds]) for k in all_folds[0]}
        results["ids"] = keys[results["test"]]
        np.savez_compressed(args.out/f"{args.corpus}_predictions.npz",**results)
        summary = aggregate(args.corpus,results,names,args.out,dtd_interval,external)
    if parities:
        summary["archived_full_field_CLS_disagreements_by_split"] = parities
    (args.out/f"{args.corpus}_settings.json").write_text(json.dumps(tuning,indent=2)+"\n")
    (args.out/f"{args.corpus}_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(args.corpus,summary["changes_pp"]["DINO_minus_HSMT"],flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus",choices=["DTD","KTH"])
    for name in ("data","features","out"):
        ap.add_argument(name,type=Path)
    ap.add_argument("--threads",type=int,default=3)
    main(ap.parse_args())
