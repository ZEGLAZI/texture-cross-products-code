"""Describe the 30 disjoint-host fits and resample whole image hosts for a paired interval.

Each photograph has one held-out prediction per model and repeat. Resample 74 hosts
with replacement, carrying their photos, labels, both models and all repeats together.
Renormalize the six class recalls in each draw. Report draws missing a class, per-class
host support, deletion diagnostics and Monte Carlo variability. Fitted models stay fixed.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits

METHODS = ("squares", "crosses")
SEED = 2026111907
STREAMS = 4
DRAWS_PER_STREAM = 50000


def read(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_predictions(folder):
    rows = read(folder / "decisions.csv")
    seeds = sorted({int(r["seed"]) for r in rows})
    keys = sorted({r["key"] for r in rows})
    key_to_i = {key: i for i, key in enumerate(keys)}
    seed_to_i = {seed: i for i, seed in enumerate(seeds)}
    info = {r["key"]: (r["hostname"], r["truth"]) for r in rows}
    assert len(rows) == len(seeds) * len(keys) == 30 * 211
    hosts, group = np.unique([info[k][0] for k in keys], return_inverse=True)
    classes, y = np.unique([info[k][1] for k in keys], return_inverse=True)
    assert len(hosts) == 74 and len(classes) == 6
    scores = np.full((len(seeds), len(keys), len(METHODS)), -1, dtype=np.int8)
    folds = np.full((len(seeds), len(keys)), -1, dtype=np.int8)
    for row in rows:
        seed = seed_to_i[int(row["seed"])]
        key = key_to_i[row["key"]]
        assert folds[seed, key] == -1 and info[row["key"]] == (row["hostname"], row["truth"])
        folds[seed,key] = int(row["fold"])
        scores[seed,key] = [row[m] == row["truth"] for m in METHODS]
    assert np.all(scores >= 0)
    return seeds, hosts, classes, group, y, folds, scores


def fold_counts(seeds, classes, group, y, folds):
    records = []
    for repeat, seed in enumerate(seeds):
        f = folds[repeat]
        assert all(len(np.unique(f[group == h])) == 1 for h in np.unique(group))
        for test in range(5):
            valid = (test + 1) % 5
            masks = {"inner_train": (f != test) & (f != valid),
                     "inner_validation": f == valid, "outer_test": f == test, "outer_refit": f != test}
            for role, inside in masks.items():
                n = np.bincount(y[inside], minlength=len(classes))
                assert np.all(n > 0)
                row = dict(seed=seed, outer_test_fold=test, role=role,
                           images=int(inside.sum()), hosts=int(len(np.unique(group[inside]))))
                for i, label in enumerate(classes):
                    row[label + "_images"] = int(n[i])
                    row[label + "_hosts"] = len(np.unique(group[inside & (y == i)]))
                records.append(row)
    return records


def pooled(hits, counts, weights):
    numerator = weights @ hits
    denominator = weights @ counts
    valid = (denominator > 0).all(axis=1)
    return 100 * (numerator[valid] / denominator[valid]).mean(axis=1), valid


def analyze(folder, out):
    out.mkdir(parents=True, exist_ok=True)
    seeds, hosts, classes, group, y, folds, correct = load_predictions(folder)
    ng, nc = len(hosts), len(classes)
    correctness = correct.mean(axis=0)
    photo_delta = correctness[:,0] - correctness[:,1]
    counts = np.zeros((ng, nc), dtype=int)
    delta_sums = np.zeros((ng, nc), dtype=float)
    np.add.at(counts, (group, y), 1)
    np.add.at(delta_sums, (group, y), photo_delta)
    class_n = counts.sum(axis=0)
    host_n = counts.sum(axis=1)
    total = 100 * (delta_sums.sum(axis=0) / class_n).mean()
    archived = read(folder / "recalls.csv")
    assert len(archived) == len(seeds)
    np.testing.assert_allclose(total, np.mean([float(r["difference_pp"]) for r in archived]), atol=1e-11)
    classes_out = []
    for c, label in enumerate(classes):
        count = counts[:,c]
        per_method = 100 * correctness[y==c].mean(axis=0)
        d = per_method[0] - per_method[1]
        classes_out.append(dict(pattern=label, photos=int(class_n[c]),
                                observed_hosts=int(np.count_nonzero(count)),
                                effective_hosts_from_sizes=float(count.sum()**2 / np.square(count).sum()),
                                largest_host_photos=int(count.max()),
                                squares_pct=per_method[0], products_pct=per_method[1],
                                difference_pp=d, contribution_to_macro_pp=d/nc))
    write(out/"host_classes.csv", classes_out)
    folds_out = fold_counts(seeds, classes, group, y, folds)
    write(out/"host_fold_composition.csv", folds_out)
    joint = Counter((r["model"], int(r["profile"]), float(r["ridge"])) for r in read(folder/"settings.csv"))
    assert all(sum(v for (m,_,_),v in joint.items() if m==method) == 150 for method in METHODS)
    write(out/"host_selection_frequencies.csv",
          [dict(model=m, profile=j, ridge=lam, folds=v) for (m,j,lam),v in sorted(joint.items())])

    drops = []
    rank = np.lexsort((hosts, -host_n))
    remaining = np.ones(ng, dtype=int)
    for removed in [[int(i)] for i in rank] + [rank[:2].tolist()]:
        weights = remaining.copy()
        weights[removed] = 0
        result,valid = pooled(delta_sums, counts, weights[None, :])
        assert valid[0]
        row = dict(removed_host="|".join(hosts[removed]), images_removed=int(host_n[removed].sum()),
                   retained_images=int(weights @ host_n), retained_hosts=int(weights.sum()),
                   macro_delta_pp=result[0], change_from_full_pp=result[0]-total)
        row.update({label+"_removed":int(counts[removed,c].sum()) for c,label in enumerate(classes)})
        drops.append(row)
    write(out/"host_deletions.csv", drops)
    loo = np.array([r["macro_delta_pp"] for r in drops[:-1]])
    jack_se = np.sqrt((ng-1)/ng * np.square(loo-loo.mean()).sum())
    jack_half = t.ppf(.975,ng-1) * jack_se

    # Multinomial cluster multiplicities are equivalent to drawing 74 complete host IDs.
    # Do not resample separately by class: a host may contain several distinct patterns.
    estimates,streams = [],[]
    for stream,child in enumerate(np.random.SeedSequence(SEED).spawn(STREAMS)):
        rng = np.random.default_rng(child)
        sample,missing_per_class = [],np.zeros(nc,dtype=int)
        for start in range(0,DRAWS_PER_STREAM,1000):
            batch = min(1000, DRAWS_PER_STREAM - start)
            weights = rng.multinomial(ng, np.full(ng,1/ng), size=batch)
            delta,valid = pooled(delta_sums,counts,weights)
            missing_per_class += (weights @ counts == 0).sum(axis=0)
            if start==0:
                # Direct photo-level calculation independently checks the count-matrix formula.
                for j in np.flatnonzero(valid)[:3]:
                    photo_w = weights[j, group]
                    direct = np.mean([np.average(photo_delta[y==c], weights=photo_w[y==c]) for c in range(nc)]) * 100
                    np.testing.assert_allclose(direct,delta[valid[:j].sum()],atol=1e-12)
            sample.append(delta)
        sample = np.concatenate(sample)
        q = np.quantile(sample,[.025,.975])
        streams.append(dict(stream=stream, attempted=DRAWS_PER_STREAM, retained=len(sample),
                            missing_any=DRAWS_PER_STREAM-len(sample), lower95_pp=q[0],upper95_pp=q[1],
                            **{c+"_missing":int(v) for c,v in zip(classes,missing_per_class)}))
        estimates.append(sample)
    write(out/"host_bootstrap_streams.csv",streams)
    simulated = np.concatenate(estimates)
    low,high = np.quantile(simulated,[.025,.975])
    summary = dict(method="paired percentile bootstrap of exact hostnames, conditional on six represented classes and fixed out-of-fold predictions",
                   seed=SEED, streams=STREAMS, photos=int(len(y)), hosts=ng, repeats=len(seeds),
                   effective_hosts_from_sizes=float(host_n.sum()**2 / np.square(host_n).sum()),
                   attempted=STREAMS*DRAWS_PER_STREAM, retained=len(simulated),
                   missing_class_draws=STREAMS*DRAWS_PER_STREAM-len(simulated), difference_pp=float(total),
                   percentile95_pp=[float(low),float(high)],
                   leave_one_host_range_pp=[float(loo.min()),float(loo.max())],
                   delete_two_largest_pp=float(drops[-1]["macro_delta_pp"]),
                   jackknife_se_pp=float(jack_se), jackknife_t95_pp=[float(total-jack_half),float(total+jack_half)])
    (out/"host_uncertainty.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host_results",type=Path,help="folder with decisions.csv, settings.csv and recalls.csv")
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    with threadpool_limits(1):
        analyze(args.host_results,args.output)
