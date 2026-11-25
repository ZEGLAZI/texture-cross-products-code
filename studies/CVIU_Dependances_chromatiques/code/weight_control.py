"""Post-hoc control for the half-weight of RGB in the mixed colour similarity."""
import argparse
import csv
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from fit import classify, load_design, polynomial, read_npz, standard_wavelet
from stats import correct, inference, macro, write


def fit_constant(rec, train, test, nclasses):
    order = np.r_[train, test]
    gram = np.ones((len(order), len(train)), dtype=float) * .5
    rgb = rec["rgb"]
    gram += .5 * (rgb[order] @ rgb[train].T)
    for key, width in (("F7", 9), ("F8", 3)):
        wave = standard_wavelet(rec[key], train, width)
        gram += wave[order] @ wave[train].T
    kernel = polynomial(gram)
    ntrain = len(train)
    scale = (11 / 13) / np.diag(kernel[:ntrain]).mean()
    labels = np.eye(nclasses)[rec["y"][train]]
    prior = labels.mean(axis=0)
    return classify(kernel, ntrain, labels - prior, prior, 1 / scale)


def record(task, ref, y, p, out, group, test_task):
    difference = correct(ref, y) - correct(p, y)
    if test_task == "ALOT":
        units = np.unique(group)
        m = int(y.size // units.size)
        increments = np.array([difference[group == k].sum() for k in units])
        values, counts = np.unique(increments, return_counts=True)
        rng = np.random.default_rng(20261120)
        draws = rng.multinomial(len(units), counts / len(units), 100000) @ values
        lo, hi = np.quantile(100 * draws / (len(units) * m), [.025 / 6, 1 - .025 / 6])
        delta = 100 * difference.mean()
    else:
        delta, lo, hi = inference(difference, y, group, test_task, 6)
    out.append(dict(task=task, half_RGB_plus_constant=macro(correct(p,y),y),
                    RGB_plus_J=macro(correct(ref,y),y), effect_pp=delta, low_pp=lo, high_pp=hi,
                    family=6, analysis="post_hoc"))


def run(data, reference, out):
    out.mkdir(exist_ok=True, parents=True)
    table = []
    decisions = {}
    for name in ("DTD", "KTH", "omit_ambient", "Lytro"):
        rec, splits, classes, groups, _ = load_design(data, name)
        pred = np.stack([fit_constant(rec, tr, te, len(classes)) for tr, te in splits])
        archived = read_npz(reference / f"{name}_predictions.npz")
        decisions[name] = pred
        record(name, archived["AW_RGB_J"], archived["y"], pred, table, groups, name)
    rec = read_npz(data / "ALOT_inputs.npz")
    with (reference / "ALOT_decisions.csv").open(newline="") as f:
        archived = {row["id"]: int(row["RGB_J"]) - 1 for row in csv.DictReader(f)}
    output = {}
    for camera in (1, 2, 3, 4):
        train = np.flatnonzero((rec["camera"] == camera) & (rec["light"] == "l8"))
        test = np.flatnonzero((rec["camera"] == camera) & (rec["light"] != "l8"))
        output.update(zip(test, fit_constant(rec, train, test, 250), strict=True))
    for name in ("temperature", "direction_c1"):
        idx = sorted(j for j in output if (rec["light"][j] == "i") == (name == "temperature"))
        y = rec["y"][idx]
        ref = np.array([archived[s] for s in rec["ids"][idx]])
        pred = np.array([output[j] for j in idx])
        decisions[name] = pred
        decisions[name+"_ids"] = rec["ids"][idx]
        record(name, ref, y, pred, table, y, "ALOT")
    np.savez_compressed(out / "weight_control_decisions.npz", **decisions)
    write(out / "weight_control.csv", table)
    for row in table:
        print(row, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "reference", "out"):
        p.add_argument(key, type=Path)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    with threadpool_limits(args.threads):
        run(args.data,args.reference,args.out)
