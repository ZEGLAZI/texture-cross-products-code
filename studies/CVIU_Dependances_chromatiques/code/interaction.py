"""Paired difference of the cross-product gains under two alternative definitions of the colour block."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from fit import classify, compact_map, load_design, polynomial, read_npz, standard_wavelet
from stats import correct, inference, macro, write


def predict(rec, tr, te, nlabels):
    ix = np.r_[tr, te]
    a, w = [standard_wavelet(rec[key], tr, width) for key, width in (('F7', 9), ('F8', 3))]
    a, w = a[ix] @ a[tr].T, w[ix] @ w[tr].T
    rgb = rec['rgb'][ix] @ rec['rgb'][tr].T
    j = compact_map(rec['edges'][:, 1, 0], tr)
    j = j[ix] @ j[tr].T
    yy = np.eye(nlabels)[rec['y'][tr]]
    prior = yy.mean(axis=0)
    common = a + w
    squares = a * a + w * w
    n = len(tr)
    outputs = {}
    for key, m in (('RGB', rgb), ('RGB_J', .5 * (rgb + j))):
        full = polynomial(common + m)
        square = (8.0 * (common + m) + squares + m * m) / 39.0
        ratio = np.diag(square[:n]).mean() / np.diag(full[:n]).mean()
        for mode, kernel, penalty in [('full', full, 1.0), ('square', square, 1.0),
                                       ('square_scale', square, ratio)]:
            outputs[key + '_' + mode] = classify(kernel,n,yy-prior,prior,penalty)
    return outputs


def compare(task, labels, groups, outputs, summary):
    for mode in ('square', 'square_scale'):
        changes = {}
        metrics = {}
        for form in ('RGB', 'RGB_J'):
            full = correct(outputs[form+'_full'],labels)
            squares = correct(outputs[form+'_'+mode],labels)
            changes[form] = full - squares
            metrics[form+'_square_pct'] = macro(squares, labels)
            metrics[form+'_full_pct'] = macro(full, labels)
            metrics[form+'_gain_pp'] = macro(changes[form], labels)
        diff = changes['RGB_J'] - changes['RGB']
        if task in ('direction_c1','temperature'):
            units = np.unique(groups)
            count = labels.size // len(units)
            sums = [int(diff[groups==k].sum()) for k in units]
            values, freq = np.unique(sums, return_counts=True)
            rng = np.random.default_rng(20261120)
            samples = rng.multinomial(len(units),freq / len(units),100000) @ values
            lo, hi = np.quantile(samples * 100 / (len(units)*count),[.025/12,1-.025/12])
            est = 100*diff.mean()
        else:
            est,lo,hi = inference(diff,labels,groups,task,12)
        summary.append(dict(task=task,scale_regime=mode,**metrics,difference_pp=est,
                            low_pp=lo,high_pp=hi,family=12,analysis='post_hoc'))


def run(data, reference, out):
    import csv
    out.mkdir(exist_ok=True,parents=True)
    summary = []
    for task in ('DTD','KTH','omit_ambient','Lytro'):
        rec,splits,classes,groups,_ = load_design(data,task)
        archive=read_npz(reference/(task+'_predictions.npz'))
        p=[predict(rec,tr,te,len(classes)) for tr,te in splits]
        outputs={k:np.stack([fold[k] for fold in p]) for k in p[0]}
        for form in ('RGB','RGB_J'):
            np.testing.assert_array_equal(outputs[form+'_full'],archive['AW_'+form])
        if task in ('DTD','KTH'):
            with np.load(Path(__file__).resolve().parents[1]/'data/native_square_reference.npz') as historical:
                np.testing.assert_array_equal(outputs['RGB_square'],historical[task])
        compare(task,archive['y'],groups,outputs,summary)
        np.savez_compressed(out/(task+'_interaction.npz'),**outputs)
        print('Kernel contrast',task,flush=True)
    rec = read_npz(data/'ALOT_inputs.npz')
    with (reference/'ALOT_decisions.csv').open(newline='') as handle:
        ref={r['id']:r for r in csv.DictReader(handle)}
    complete = {}
    for cam in (1,2,3,4):
        tr=np.flatnonzero((rec['camera']==cam)&(rec['light']=='l8'))
        te=np.flatnonzero((rec['camera']==cam)&(rec['light']!='l8'))
        outputs=predict(rec,tr,te,250)
        for form in ('RGB','RGB_J'):
            np.testing.assert_array_equal(outputs[form+'_full'],[int(ref[i][form])-1 for i in rec['ids'][te]])
        for idx,k in enumerate(te):complete[k]={name:value[idx] for name,value in outputs.items()}
    for task in ('temperature','direction_c1'):
        ids=sorted(k for k in complete if (rec['light'][k]=='i')==(task=='temperature'))
        values={key:np.array([complete[k][key] for k in ids]) for key in complete[ids[0]]}
        compare(task,rec['y'][ids],rec['y'][ids],values,summary)
        np.savez_compressed(out/(task+'_interaction.npz'),ids=rec['ids'][ids],**values)
    write(out/'kernel_contrast.csv',summary)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','reference','out'):
        p.add_argument(name,type=Path)
    p.add_argument('--threads',type=int,default=2)
    args=p.parse_args()
    with threadpool_limits(args.threads):
        run(args.data,args.reference,args.out)
