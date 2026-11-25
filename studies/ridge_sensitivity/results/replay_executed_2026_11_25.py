"""Reconstruct the v3 DTD ridge sensitivity from the released descriptor arrays.

From the reviewer package root:
    python studies/ridge_sensitivity/replay.py --mode full --report ridge.json
    python studies/ridge_sensitivity/replay.py --mode tables --report ridge.json

The validation stage is completed and fingerprinted for all 100 kernel/split
choices before any outer fit.  ``select`` and ``fit`` expose resumable stages;
``tables`` reads only saved predictions and curves, and never refits a model.
This file was reconstructed from the published v3 protocol; it is not an
unrecovered original script.  Historical scripts and arrays are unchanged.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time
from zipfile import ZipFile

import numpy as np
import scipy
from scipy.linalg import eigh, solve
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[1]
MODELS = ("H", "R", "DR", "DV", "D")
NAMES = {
    "H": "Colour–texture", "R": "Supervised ResNet-50",
    "DR": "DINO ResNet-50", "DV": "DINO ViT-S/16", "D": "DINOv2 ViT-S/14",
}
SIX = (1., .1, 10., .01, 100., .001)
KERNELS = ("S", "Q")
SETTINGS = ("common_0.1", "common_1", "common_10", "six", "dense")
PROTOCOL = {
    "version": "v3-reconstruction-1", "initial_j": [-12, 8],
    "penalty": "10**(j/4)", "extension_j": 4, "limits_j": [-32, 24],
    "recall_fraction_tolerance": 1e-13,
    "dense_tie_break": "minimum (abs(j), j)", "six_order": SIX,
    "common_penalties": [.1, 1., 10.], "bootstrap_seed": 19112026,
    "bootstrap_draws": 100000,
    "source_of_seed": "unchanged summarize_pixel_check.paired_interval",
    "uncertainty": "class/split-centred exact-image-group Rademacher multipliers",
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def zip_npz(archive, name):
    return load_npz(io.BytesIO(archive.read(name)))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(root):
    paths = ["data_and_code_r4.zip", "pixel_check/DTD_colour.npz",
             "studies/resnet50/data/DTD_stages.npz",
             "additional_features/dino_r50/DTD_stages.npz",
             "additional_features/dino_vits16/DTD_stages.npz"]
    files = {p: sha256(root / p) for p in paths}
    payload = {"protocol": PROTOCOL, "source_sha256": files,
               "implementation_sha256": sha256(Path(__file__))}
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def design(root):
    with ZipFile(root / "data_and_code_r4.zip") as z:
        meta = zip_npz(z, "historical/dtd/results/identites.npz")
        folds = []
        def exclude(indices, other):
            conflict = np.isin(meta["source_sha256"][indices], meta["source_sha256"][other])
            conflict |= np.isin(meta["pixels224"][indices], meta["pixels224"][other])
            return indices[~conflict]
        for number in range(1, 11):
            train, val, test = (meta[k + str(number)] for k in ("train", "val", "test"))
            outer = exclude(np.r_[train, val], test)
            validation = exclude(val, test)
            fit = exclude(exclude(train, test), validation)
            old = zip_npz(z, f"review2/results/kernels/DTD_{number:02}.npz")
            for key, values in (("train", outer), ("test", test),
                                ("inner_train_0", fit), ("inner_val_0", validation)):
                np.testing.assert_array_equal(values, old[key])
            for key in ("source_sha256", "pixels224"):
                assert not np.intersect1d(meta[key][outer], meta[key][test]).size
                assert not np.intersect1d(meta[key][fit], meta[key][validation]).size
            folds.append({"name": f"DTD_{number:02}", "train": outer, "test": test,
                          "fit": fit, "val": validation})
    return meta, folds


def features(root, model, ids):
    if model == "H":
        item = load_npz(root / "pixel_check/DTD_colour.npz")
        raw = [item[k] for k in ("F7", "F8", "rgb")]
    elif model == "D":
        with ZipFile(root / "data_and_code_r4.zip") as z:
            item = zip_npz(z, "cross_representation/features/DTD_features.npz")
        raw = [item["patch" + str(k)] for k in (3, 6, 12)]
    else:
        folder = {"R": "studies/resnet50/data", "DR": "additional_features/dino_r50",
                  "DV": "additional_features/dino_vits16"}[model]
        item = load_npz(root / folder / "DTD_stages.npz")
        raw = [item["layer" + str(k)] for k in (2, 3, 4)]
    np.testing.assert_array_equal(item["ids"], ids)
    return raw


def normalize(raw, fit, model):
    # Same train-only feature definition as independent_refit.py, implemented here
    # explicitly so this new analysis does not depend on a prior experiment runner.
    result = []
    for index, array in enumerate(raw):
        if model == "H" and index == 2:
            result.append(array.astype(np.float64, copy=False))
            continue
        values = array.astype(np.float64)
        axes = 0
        if model == "H":
            row, col = np.triu_indices((9, 3)[index])
            values[..., np.flatnonzero(row != col)] *= np.sqrt(2.)
            axes = (0, 2)
        sample = values[fit]
        values -= sample.mean(axis=axes, keepdims=True)
        values /= np.maximum(sample.std(axis=axes, keepdims=True), 1e-4)
        np.clip(values, -12., 12., out=values)
        values = values.reshape(len(values), -1)
        norm = np.linalg.norm(values, axis=1, keepdims=True)
        np.divide(values, norm, out=values, where=norm != 0)
        result.append(values)
    return result


def kernels(raw, fit, evaluation, model):
    blocks = normalize(raw, fit, model)
    rows = np.r_[fit, evaluation]
    train_blocks = [np.ascontiguousarray(b[fit]) for b in blocks]
    square = np.empty((len(rows), len(fit)), dtype=np.float64)
    full = np.empty_like(square)
    for start in range(0, len(rows), 192):
        section = slice(start, start + 192)
        a, b, c = (block[rows[section]] @ ref.T for block, ref in zip(blocks, train_blocks))
        square[section] = ((8. / 3.) * (a + b + c) + (a*a + b*b + c*c) / 3.) / 13.
        full[section] = square[section] + (2. / 39.) * (a*b + a*c + b*c)
    np.testing.assert_allclose(np.diag(square), 9. / 13., rtol=0, atol=2e-12)
    np.testing.assert_allclose(np.diag(full), 11. / 13., rtol=0, atol=2e-12)
    return square, full


def target(y):
    enc = np.eye(47)[y]
    prior = enc.mean(axis=0)
    return enc - prior, prior


def macro(labels, truth):
    return float(np.mean([np.mean(labels[truth == c] == c) for c in np.unique(truth)]))


def validation(kernel, yfit, yval):
    eig, vec = eigh(kernel[:len(yfit)], driver="evd", check_finite=False)
    t, prior = target(yfit)
    left, right = kernel[len(yfit):] @ vec, vec.T @ t
    scores = {}
    def score(j):
        if j not in scores:
            penalty = 10. ** (j / 4.)
            pred = (left @ (right / (eig[:, None] + penalty)) + prior).argmax(axis=1)
            scores[j] = macro(pred, yval)
        return scores[j]
    lo, hi = -12, 8
    for j in range(lo, hi + 1):
        score(j)
    lower_steps = upper_steps = 0
    while True:
        maximum = max(scores.values())
        low_edge = scores[lo] >= maximum - 1e-13 and lo > -32
        high_edge = scores[hi] >= maximum - 1e-13 and hi < 24
        if not low_edge and not high_edge:
            break
        if low_edge:
            new_lo = max(lo - 4, -32)
            for j in range(new_lo, lo):
                score(j)
            lo = new_lo
            lower_steps += 1
        if high_edge:
            new_hi = min(hi + 4, 24)
            for j in range(hi + 1, new_hi + 1):
                score(j)
            hi = new_hi
            upper_steps += 1
    maximum = max(scores.values())
    tied = sorted(j for j, v in scores.items() if v >= maximum - 1e-13)
    selected_j = min(tied, key=lambda j: (abs(j), j))
    original_j = (0, -4, 4, -8, 8, -12)
    six_max = max(scores[j] for j in original_j)
    six_j = next(j for j in original_j if scores[j] >= six_max - 1e-13)
    return {
        "selected_j": selected_j, "selected_lambda": 10. ** (selected_j / 4.),
        "six_j": six_j, "six_lambda": 10. ** (six_j / 4.), "max_recall_fraction": maximum,
        "tied_j": tied, "lower_extension_steps": lower_steps, "upper_extension_steps": upper_steps,
        "lower_cap_tied": -32 in tied, "upper_cap_tied": 24 in tied,
        "edf_at_1": float(np.sum(eig / (eig + 1.))),
        "edf_at_11_over_9": float(np.sum(eig / (eig + 11. / 9.))),
        "minimum_eigenvalue": float(eig.min()),
        "curve": [{"j": j, "lambda": 10. ** (j / 4.), "macro_recall_fraction": scores[j]}
                  for j in sorted(scores)],
    }


def selection(root, out, manifest, force=False):
    meta, folds = design(root)
    records = []
    for model in MODELS:
        path = out / f"selection_{model}.json"
        if path.exists() and not force:
            cached = json.loads(path.read_text())
            if cached["fingerprint"] != manifest["fingerprint"]:
                raise ValueError(f"Stale selection cache {path}; choose a new output or use --force")
            records.extend(cached["records"])
            print(f"Reused validation {model}: {len(cached['records'])} choices", flush=True)
            continue
        raw = features(root, model, meta["ids"])
        model_records = []
        for fold in folds:
            start = time.monotonic()
            pair = kernels(raw, fold["fit"], fold["val"], model)
            for code, kernel in zip(KERNELS, pair):
                record = validation(kernel, meta["y"][fold["fit"]], meta["y"][fold["val"]])
                record.update(representation=model, split=fold["name"], kernel=code,
                              fit_images=len(fold["fit"]), validation_images=len(fold["val"]))
                model_records.append(record)
            del pair, kernel
            print(f"Selected {model} {fold['name']}: "
                  f"{[r['selected_j']/4 for r in model_records[-2:]]} "
                  f"({time.monotonic()-start:.1f}s)", flush=True)
        dump(path, {"fingerprint": manifest["fingerprint"], "records": model_records})
        records.extend(model_records)
        del raw
    assert len(records) == 100
    # This file is completed before fit() may run. It records this reconstruction,
    # without asserting recovery of the original authors' historical timestamp.
    frozen = {**manifest, "frozen_utc": utc(), "records": records}
    path = out / "choices_frozen.json"
    if path.exists() and not force:
        old = json.loads(path.read_text())
        if old["fingerprint"] == manifest["fingerprint"] and old["records"] == records:
            return old
    dump(path, frozen)
    (out / "choices_frozen.sha256").write_text(sha256(path) + "  choices_frozen.json\n")
    return frozen


def archived_predictions(root, model):
    if model in ("H", "D"):
        a = load_npz(root / "pixel_check/DTD_predictions.npz")
        return {**{k: a[k] for k in ("test", "ids", "labels", "groups")},
                **{f"common_1_{k}": a[f"{model}_{k}"] for k in KERNELS},
                **{f"six_{k}": a[f"{model}_tuned_{k}"] for k in KERNELS}}
    folder = "studies/resnet50/results" if model == "R" else (
        "studies/matched_pretraining/results/" + {"DR": "dino_r50", "DV": "dino_vits16"}[model])
    a = load_npz(root / folder / "DTD_decisions.npz")
    return {**{k: a[k] for k in ("test", "ids", "labels", "groups")},
            **{f"common_1_{k}": a[k] for k in KERNELS},
            **{f"six_{k}": a[f"selected_{k}"] for k in KERNELS}}


def fit(root, out, manifest, force=False):
    frozen_path = out / "choices_frozen.json"
    frozen = json.loads(frozen_path.read_text())
    frozen_hash = sha256(frozen_path)
    if frozen["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Validation choices do not correspond to the present implementation/inputs")
    if (out / "choices_frozen.sha256").read_text().split()[0] != frozen_hash:
        raise ValueError("Validation choices changed after fingerprinting")
    choices = {(r["representation"], r["split"], r["kernel"]): r for r in frozen["records"]}
    assert len(choices) == 100
    meta, folds = design(root)
    for model in MODELS:
        audit_file = out / f"fit_{model}.json"
        pred_file = out / f"DTD_{model}_predictions.npz"
        if audit_file.exists() and pred_file.exists() and not force:
            previous = json.loads(audit_file.read_text())
            if previous["choices_sha256"] != frozen_hash or previous["predictions_sha256"] != sha256(pred_file):
                raise ValueError(f"Stale or changed outer predictions for {model}")
            print(f"Reused outer fits {model}", flush=True)
            continue
        raw = features(root, model, meta["ids"])
        archive = archived_predictions(root, model)
        rows, evidence = {}, []
        started = utc()
        for f, fold in enumerate(folds):
            start = time.monotonic()
            train, test = fold["train"], fold["test"]
            n = len(train)
            pair = kernels(raw, train, test, model)
            target_y, prior = target(meta["y"][train])
            data = {"test": test, "ids": meta["ids"][test], "labels": meta["y"][test],
                    "groups": meta["groupes"][test]}
            compared = mismatches = 0
            max_score_error = 0.
            solver_mismatches = 0
            for key in ("test", "ids", "labels", "groups"):
                np.testing.assert_array_equal(data[key], archive[key][f])
            for code, kernel in zip(KERNELS, pair):
                choice = choices[model, fold["name"], code]
                penalties = {"common_0.1": .1, "common_1": 1., "common_10": 10.,
                             "six": choice["six_lambda"], "dense": choice["selected_lambda"]}
                eig, vec = eigh(kernel[:n], driver="evd", check_finite=False)
                left, right = kernel[n:] @ vec, vec.T @ target_y
                cache = {}
                for setting, penalty in penalties.items():
                    if penalty not in cache:
                        cache[penalty] = left @ (right / (eig[:, None] + penalty)) + prior
                    data[f"{setting}_{code}"] = cache[penalty].argmax(axis=1).astype(np.int16)
                    if setting in ("common_1", "six"):
                        expected = archive[f"{setting}_{code}"][f]
                        differences = int(np.count_nonzero(data[f"{setting}_{code}"] != expected))
                        compared += len(expected)
                        mismatches += differences
                # A distinct positive-definite solve checks every dense outer fit,
                # including very small validation-selected penalties.
                matrix = kernel[:n].copy(order="F")
                penalty = choice["selected_lambda"]
                matrix.flat[::n+1] += penalty
                alpha = solve(matrix, target_y, assume_a="pos", overwrite_a=True, check_finite=False)
                direct = kernel[n:] @ alpha + prior
                max_score_error = max(max_score_error, float(np.max(np.abs(direct - cache[penalty]))))
                solver_mismatches += int(np.count_nonzero(direct.argmax(axis=1) != data[f"dense_{code}"]))
                del eig, vec, left, right, matrix, alpha, direct, cache
            for key, value in data.items():
                rows.setdefault(key, []).append(value)
            evidence.append({"split": fold["name"], "predicted_labels": 10 * len(test),
                             "archived_labels_compared": compared, "archived_label_mismatches": mismatches,
                             "dense_solver_labels_compared": 2 * len(test),
                             "dense_solver_label_mismatches": solver_mismatches,
                             "maximum_dense_solver_score_difference": max_score_error})
            del pair, kernel
            print(f"Fitted {model} {fold['name']}: {mismatches} archived / "
                  f"{solver_mismatches} solver differences ({time.monotonic()-start:.1f}s)", flush=True)
        np.savez_compressed(pred_file, **{key: np.stack(value) for key, value in rows.items()})
        dump(audit_file, {"representation": model, "started_utc": started, "finished_utc": utc(),
                         "choices_sha256": frozen_hash, "predictions_sha256": sha256(pred_file),
                         "records": evidence})
        del raw


def old_extraction_control(root, out, force=False):
    path = out / "extraction_control.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    meta, folds = design(root)
    fold = folds[5]
    with ZipFile(root / "data_and_code_r4.zip") as z:
        names = sorted(n for n in z.namelist() if n.startswith("review2/dtd_cache/hsmt/") and n.endswith(".npz"))
        pieces = [zip_npz(z, name) for name in names]
        np.testing.assert_array_equal(np.concatenate([p["ids"] for p in pieces]), meta["ids"])
        raw = [np.concatenate([p[k] for p in pieces]) for k in ("F7", "F8", "rgb")]
        archive = zip_npz(z, "cross_representation/results/DTD_predictions.npz")
    pair = kernels(raw, fold["fit"], fold["val"], "H")
    old = validation(pair[0], meta["y"][fold["fit"]], meta["y"][fold["val"]])
    del pair
    records = json.loads((out / "choices_frozen.json").read_text())["records"]
    new = next(r for r in records if (r["representation"], r["split"], r["kernel"]) == ("H", "DTD_06", "S"))
    current = load_npz(out / "DTD_H_predictions.npz")
    pair = kernels(raw, fold["train"], fold["test"], "H")
    n = len(fold["train"])
    t, prior = target(meta["y"][fold["train"]])
    matrix = pair[0][:n].copy(order="F")
    matrix.flat[::n+1] += old["six_lambda"]
    alpha = solve(matrix, t, assume_a="pos", overwrite_a=True, check_finite=False)
    predicted = (pair[0][n:] @ alpha + prior).argmax(axis=1)
    old_mismatch = int(np.count_nonzero(predicted != archive["H_tuned_S"][5]))
    result = {"old_outer_labels_compared": len(predicted), "old_outer_label_mismatches": old_mismatch,
              "old_vs_reference_selected_labels_changed": int(np.count_nonzero(archive["H_tuned_S"][5] != current["six_S"][5])),
              "old_vs_reference_common_1_labels_changed": int(np.count_nonzero(archive["H_S"][5] != current["common_1_S"][5])),
              "rows": []}
    for label, record in (("Earlier r4", old), ("Reference", new)):
        scores = {r["j"]: r["macro_recall_fraction"] for r in record["curve"]}
        result["rows"].append({"extraction": label, "lambda_0.1_pct": 100*scores[-4],
                               "lambda_1_pct": 100*scores[0], "margin_pp": 100*(scores[-4]-scores[0]),
                               "selected_lambda": record["six_lambda"]})
    for setting, original in (("common_1", "H_"), ("six", "H_tuned_")):
        for branch, source, prefix in (("earlier", archive, original), ("reference", current, setting+"_")):
            result[branch+"_"+setting+"_gain_pp"] = 100*float(np.mean(
                (source[prefix+"Q"] == current["labels"]).astype(int) -
                (source[prefix+"S"] == current["labels"]).astype(int)))
    dump(path, result)
    return result


def figures(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(10.7, 4.5))
    positions = (("H", 0), ("R", 1), ("D", 2), ("DR", 3), ("DV", 4))
    for model, position in positions:
        ax = axes.flat[position]
        for code, color, name in (("S", "#1c587a", "Squares"), ("Q", "#bd6b2a", "Full")):
            subset = [r for r in rows if r["representation"] == model and r["kernel"] == code]
            values = np.array([[100*c["macro_recall_fraction"] for c in r["curve"] if -12 <= c["j"] <= 8] for r in subset])
            penalties = 10. ** (np.arange(-12, 9) / 4)
            ax.plot(penalties, values.T, color=color, lw=.55, alpha=.25)
            ax.plot(penalties, values.mean(axis=0), color=color, lw=1.7, label=name)
        ax.set_xscale("log")
        ax.axvline(1., color=".45", ls=":", lw=.8)
        ax.set_title(NAMES[model])
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].axis("off")
    axes.flat[5].legend(handles, labels, loc="center", frameon=False)
    fig.supxlabel("Ridge penalty λ", fontsize=10)
    fig.supylabel("Validation macro recall (%)", fontsize=10)
    fig.tight_layout(pad=1.5)
    fig.savefig(out / "ridge_validation.pdf", bbox_inches="tight")
    fig.savefig(out / "ridge_validation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def summarize(root, out):
    sys.path.insert(0, str(root))
    # Retain the released Monte Carlo ordering, not merely the same limiting law.
    from summarize_pixel_check import paired_interval, macro as percent
    frozen = json.loads((out / "choices_frozen.json").read_text())
    records = frozen["records"]
    pred = {m: load_npz(out / f"DTD_{m}_predictions.npz") for m in MODELS}
    y, groups = pred["H"]["labels"], pred["H"]["groups"]
    for p in pred.values():
        for key in ("ids", "test", "labels", "groups"):
            np.testing.assert_array_equal(p[key], pred["H"][key])
    gains, delta = [], {}
    for model in MODELS:
        for setting in SETTINGS:
            before = (pred[model][setting+"_S"] == y).astype(int)
            after = (pred[model][setting+"_Q"] == y).astype(int)
            diff = after - before
            delta[model, setting] = diff
            lo, hi = paired_interval("DTD", diff, y, groups, 1)
            gains.append({"representation": model, "setting": setting,
                          "squares_pct": percent(before, y), "full_pct": percent(after, y),
                          "gain_pp": percent(diff, y), "low_pp": lo, "high_pp": hi})
    contrasts = []
    for name, a, b, family in (("DINO pair", "DR", "DV", 8),
                               ("ResNet pair", "DR", "R", 8),
                               ("Supervised / DINOv2", "R", "D", 3)):
        for common in ("0.1", "1", "10"):
            for setting in ("six", "dense"):
                diff = delta[a, setting] - delta[a, "common_"+common] - delta[b, setting] + delta[b, "common_"+common]
                lo, hi = paired_interval("DTD", diff, y, groups, family)
                contrasts.append({"pair": name, "common_lambda": float(common), "selection": setting,
                                  "estimate_pp": percent(diff, y), "low_pp": lo, "high_pp": hi,
                                  "family_size": family})
    edf = []
    for model in MODELS:
        s = [r for r in records if r["representation"] == model and r["kernel"] == "S"]
        q = [r for r in records if r["representation"] == model and r["kernel"] == "Q"]
        edf.append({"representation": model, "native_S": np.mean([r["edf_at_1"] for r in s]),
                    "native_Q": np.mean([r["edf_at_1"] for r in q]),
                    "matched_Q": np.mean([r["edf_at_11_over_9"] for r in q]),
                    "extended_S": sum(r["lower_extension_steps"] > 0 for r in s),
                    "extended_Q": sum(r["lower_extension_steps"] > 0 for r in q),
                    "cap_tied_S": sum(r["lower_cap_tied"] for r in s),
                    "cap_tied_Q": sum(r["lower_cap_tied"] for r in q)})
    all_curves = [{"representation": r["representation"], "split": r["split"], "kernel": r["kernel"],
                   **c} for r in records for c in r["curve"]]
    choices = [{"representation": r["representation"], "split": r["split"], "kernel": r["kernel"],
                "log10_lambda": r["selected_j"]/4, "six_lambda": r["six_lambda"],
                "tied_maxima": len(r["tied_j"]), "lower_extension_steps": r["lower_extension_steps"],
                "upper_extension_steps": r["upper_extension_steps"]} for r in records]
    for name, rows in (("gains", gains), ("contrasts", contrasts), ("effective_degrees", edf),
                       ("validation_curves", all_curves), ("selected_penalties", choices)):
        write_csv(out / (name + ".csv"), rows)
    meta, _ = design(root)
    candidates = {"correction": [], "new_error": []}
    seen = set()
    p = pred["H"]
    for f in range(10):
        for i in np.argsort(p["ids"][f]):
            group = int(groups[f, i])
            if group in seen:
                continue
            seen.add(group)
            s, q, t = (int(v) for v in (p["common_1_S"][f, i], p["common_1_Q"][f, i], y[f, i]))
            category = "correction" if s != t and q == t else "new_error" if s == t and q != t else None
            if category:
                identifier = str(p["ids"][f, i])
                candidates[category].append({"sha256": hashlib.sha256(identifier.encode()).hexdigest(),
                    "identifier": identifier, "filename": identifier.split("/")[-1], "split": f+1,
                    "attribute": str(meta["traits"][t]), "squares": str(meta["traits"][s]),
                    "full": str(meta["traits"][q]), "category": category})
    examples = [v for c in ("correction", "new_error") for v in sorted(candidates[c], key=lambda x:x["sha256"])[:2]]
    write_csv(out / "examples.csv", examples)
    figures(out, records)
    result = {"gains": gains, "contrasts": contrasts, "effective_degrees": edf, "examples": examples,
              "selection_counts": {"choices": len(records), "changed_from_six": sum(r["selected_j"] != r["six_j"] for r in records),
                "searches_with_tied_maxima": sum(len(r["tied_j"]) > 1 for r in records),
                "lower_extensions": sum(r["lower_extension_steps"] > 0 for r in records),
                "upper_extensions": sum(r["upper_extension_steps"] > 0 for r in records),
                "searched_validation_values": len(all_curves)},
              "test_appearances_per_representation": int(y.size), "unique_test_groups": int(np.unique(groups).size),
              "extraction_control": json.loads((out / "extraction_control.json").read_text())}
    dump(out / "summary.json", result)
    return result


def audit_published(out, summary):
    expected = json.loads((HERE / "published_v3.json").read_text())
    checks = []
    def compare(table, label, fields, observed, truth, decimals=2):
        for field, actual, printed in zip(fields, observed, truth, strict=True):
            ok = abs(float(actual)-float(printed)) <= .5*10.**(-decimals) + 1e-10
            checks.append({"table": table, "row": label, "field": field, "observed": float(actual),
                           "published": float(printed), "decimals": decimals, "matches_rounding": bool(ok)})
    gains = {(r["representation"], r["setting"]): r for r in summary["gains"]}
    for model in MODELS:
        g = gains[model, "dense"]
        compare("4", model, ("common_gain", "six_gain", "squares", "full", "dense_gain", "low", "high"),
                [gains[model, "common_1"]["gain_pp"], gains[model, "six"]["gain_pp"],
                 *[g[k] for k in ("squares_pct", "full_pct", "gain_pp", "low_pp", "high_pp")]], expected["4"][model])
    records = json.loads((out / "choices_frozen.json").read_text())["records"]
    chosen = {(r["representation"], r["split"], r["kernel"]): r["selected_j"]/4 for r in records}
    for f in range(10):
        fields = [m+"_"+k for m in MODELS for k in KERNELS]
        observed = [chosen[m, f"DTD_{f+1:02}", k] for m in MODELS for k in KERNELS]
        compare("S38", f"DTD_{f+1:02}", fields, observed, expected["S38"][f])
    contrasts = {(r["pair"], r["common_lambda"], r["selection"]): r for r in summary["contrasts"]}
    for row in expected["S39"]:
        observed = [contrasts[row["pair"], row["common_lambda"], setting][k]
                    for setting in ("six", "dense") for k in ("estimate_pp", "low_pp", "high_pp")]
        compare("S39", row["pair"]+" "+str(row["common_lambda"]),
                ("six", "six_low", "six_high", "dense", "dense_low", "dense_high"), observed, row["values"])
    for row in summary["effective_degrees"]:
        fields = ("native_S", "native_Q", "matched_Q", "extended_S", "extended_Q", "cap_tied_S", "cap_tied_Q")
        compare("S40", row["representation"], fields, [row[k] for k in fields], expected["S40"][row["representation"]], 1)
    for row in summary["extraction_control"]["rows"]:
        fields = ("lambda_0.1_pct", "lambda_1_pct", "margin_pp", "selected_lambda")
        compare("S41", row["extraction"], fields, [row[k] for k in fields], expected["S41"][row["extraction"]], 4)
    assert len(checks) == 232
    example_fields = ("filename", "split", "attribute", "squares", "full")
    actual_examples = [[row[k] for k in example_fields] for row in summary["examples"]]
    other = {
        "selection_counts": summary["selection_counts"]["choices"] == 100 and
            summary["selection_counts"]["changed_from_six"] == 75 and
            summary["selection_counts"]["searches_with_tied_maxima"] == 17 and
            summary["selection_counts"]["lower_extensions"] == 3 and
            summary["selection_counts"]["upper_extensions"] == 0,
        "S42_examples": actual_examples == expected["S42"],
        "S41_selected_labels_changed_542": summary["extraction_control"]["old_vs_reference_selected_labels_changed"] == 542,
        "S41_common_labels_changed_2": summary["extraction_control"]["old_vs_reference_common_1_labels_changed"] == 2,
        "old_split6_outer_reproduced": summary["extraction_control"]["old_outer_label_mismatches"] == 0,
    }
    audits = [json.loads((out / f"fit_{m}.json").read_text()) for m in MODELS]
    counters = {k: sum(r[k] for a in audits for r in a["records"]) for k in
        ("predicted_labels", "archived_labels_compared", "archived_label_mismatches",
         "dense_solver_labels_compared", "dense_solver_label_mismatches")}
    other["all_test_predictions_saved"] = counters["predicted_labels"] == 940000
    other["archived_predictions_reproduced"] = counters["archived_labels_compared"] == 376000 and counters["archived_label_mismatches"] == 0
    other["dense_independent_solver_agrees"] = counters["dense_solver_labels_compared"] == 188000 and counters["dense_solver_label_mismatches"] == 0
    failed = [c for c in checks if not c["matches_rounding"]]
    report = {"status": "passed" if not failed and all(other.values()) else "mismatch",
              "checked_utc": utc(), "new_numeric_tables": 5, "numeric_values_checked": len(checks),
              "numeric_mismatches": len(failed), "failed_numeric_checks": failed,
              "additional_checks": other, "counts": counters, "selection_counts": summary["selection_counts"],
              "implementation": "Reconstructed from v3 PDF and existing feature definitions; original dense script was absent.",
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "uncertainty": {"draws": 100000, "seed": 19112026, "source": "unchanged summarize_pixel_check.py"},
              "checks": checks}
    dump(out / "audit_v3.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "tables", "select", "fit"), default="tables")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--force", action="store_true", help="Rebuild the selected stages; full rebuilds both stages")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(args.threads):
        if args.mode in ("full", "select", "fit"):
            manifest = fingerprint(root)
            dump(out / "input_manifest.json", manifest)
        if args.mode in ("full", "select"):
            selection(root, out, manifest, args.force)
        if args.mode == "select":
            print("All 100 validation choices frozen before any outer fit.", flush=True)
            return
        if args.mode in ("full", "fit"):
            fit(root, out, manifest, args.force)
            old_extraction_control(root, out, args.force)
        summary = summarize(root, out)
        report = audit_published(out, summary)
        if args.report:
            dump(args.report, report)
        print(json.dumps({k: report[k] for k in ("status", "numeric_values_checked", "numeric_mismatches", "counts")}), flush=True)
        if report["status"] != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
