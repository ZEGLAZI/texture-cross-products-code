"""Exploratory sensitivity to the image-serving hostname on the 211 archived photographs.

Freeze the first 30 distinct five-fold allocations with all six patterns in each fold.
The search uses only labels and original URL hostnames. Begin at seed 2026111900
and increment by one. Never choose a split using a prediction or a performance metric.
The comparator, training-only grids, cyclic validation fold and photograph-level macro
recall are identical to the repeated clothing analysis. Report all 30 outcomes.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
SEED = 2026111900
N_REPEATS = 30
MODELS = ("squares", "crosses")


def inputs():
    with np.load(ROOT / "clothing/data/features.npz") as z:
        raw = {k: z[k] for k in ("F7", "F8", "rgb")}
        keys, y, classes = (z[k] for k in ("keys", "y", "classes"))
    with (ROOT / "clothing/data/attempts.csv").open() as f:
        hosts = {(r["key"], r["class"]): urlsplit(r["URL"]).hostname
                 for r in csv.DictReader(f)}
    groups = np.array([hosts[k, classes[c]] for k, c in zip(keys, y)])
    assert len(keys) == len(set(keys)) == 211 and all(groups)
    return raw, keys, y, classes, groups


def save_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def prepare(out):
    _, keys, y, classes, groups = inputs()
    folds, seeds, seen = [], [], set()
    for seed in range(SEED, SEED + 10000):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        assignment = np.full(len(y), -1, dtype=np.int8)
        for fold, (_, test) in enumerate(splitter.split(keys, y, groups)):
            assignment[test] = fold
        if not all(len(np.unique(y[assignment == k])) == len(classes) for k in range(5)):
            continue
        assert all(len(np.unique(assignment[groups == h])) == 1 for h in np.unique(groups))
        signature = tuple(sorted(tuple(np.flatnonzero(assignment == f)) for f in range(5)))
        if signature in seen:
            continue
        seen.add(signature)
        folds.append(assignment)
        seeds.append(seed)
        if len(folds) == N_REPEATS:
            break
    assert len(folds) == N_REPEATS
    save_csv(out / "assignments.csv", ("seed", "key", "class", "hostname", "fold"),
             ((seed, k, classes[c], h, f) for seed, assignment in zip(seeds, folds)
              for k, c, h, f in zip(keys, y, groups, assignment)))
    print(f"{len(folds)} allocations, {len(np.unique(groups))} hostnames, seeds through {seeds[-1]}")


def evaluate(out):
    sys.path.insert(0, str(ROOT / "clothing/code"))
    from evaluate_partitions import PEN, PROFILES, macro, outer_fold

    raw, keys, y, classes, groups = inputs()
    path = out / "assignments.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    seeds = sorted({int(r["seed"]) for r in rows})
    assert len(seeds) == N_REPEATS and len(rows) == len(keys) * N_REPEATS
    results, choices, predictions, all_grids = [], [], [], []
    for seed in seeds:
        records = [row for row in rows if int(row["seed"]) == seed]
        np.testing.assert_array_equal(keys, [r["key"] for r in records])
        np.testing.assert_array_equal(groups, [r["hostname"] for r in records])
        np.testing.assert_array_equal(classes[y], [r["class"] for r in records])
        assignment = np.array([int(r["fold"]) for r in records])
        assert all(len(set(assignment[groups == h])) == 1 for h in set(groups))
        ordered = np.full((len(MODELS), len(keys)), -1, dtype=np.int8)
        for k in range(5):
            test, pred, selected, grid = outer_fold(assignment, k, raw, {}, y, methods=MODELS)
            assert (ordered[:, test] == -1).all()
            ordered[:, test] = pred
            all_grids.append(grid)
            for method, (j, v) in zip(MODELS, selected):
                choices.append((seed, k, method, j, *PROFILES[j], PEN[v]))
        assert np.all(ordered >= 0)
        square, cross = [macro(pred, y) for pred in ordered]
        results.append((seed, square, cross, square - cross))
        predictions.extend((seed, k, h, t, classes[c], classes[s], classes[x])
                           for k, h, t, c, s, x in zip(keys, groups, assignment, y, *ordered))
    save_csv(out / "recalls.csv", ("seed","squares","crosses","difference_pp"), results)
    save_csv(out / "settings.csv", ("seed","fold","model","profile","p_s","p_w","p_c","ridge"), choices)
    save_csv(out / "decisions.csv", ("seed","key","hostname","fold","truth","squares","crosses"), predictions)
    np.savez_compressed(out / "validation_grids.npz", validation=np.array(all_grids).reshape(N_REPEATS,5,2,4,6),
                        seeds=seeds, methods=MODELS, profiles=PROFILES, penalties=PEN)
    scores = np.array(results)[:, 1:]
    delta = scores[:, 2]
    info = {"unit":"exact hostname of original manifest URL","n_photos":len(keys),
            "n_hosts":int(len(np.unique(groups))),"n_partitions":N_REPEATS,
            "assignments_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "squares_mean":scores[:, 0].mean(),"crosses_mean":scores[:, 1].mean(),
            "delta_mean_pp":delta.mean(),"delta_min_pp":delta.min(),"delta_max_pp":delta.max(),
            "wins":int((delta > 0).sum()),"ties":int((delta == 0).sum()),
            "losses":int((delta < 0).sum())}
    (out / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "fit"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    with threadpool_limits(1):
        (prepare if args.action == "prepare" else evaluate)(args.output)
