"""Mesurer les écarts appariés en conservant ensemble toutes les répétitions d'une photo."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from preparer import PROTO, ROOT
from threadpoolctl import threadpool_limits


def table(file, rows):
    with file.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyse(src, out):
    out.mkdir(exist_ok=True, parents=True)
    a = dict(np.load(src / "identites.npz"))
    d = dict(np.load(src / "decisions.npz"))
    tests = d["tests"]
    groups = a["groupes"]
    truth = a["y"][tests]
    names = list(PROTO["methodes"]) + list(
        json.loads((ROOT / "controle_echelle.json").read_text())["methodes"]
    )
    primary = PROTO["principales"]
    extra = json.loads((ROOT / "controle_echelle.json").read_text())["secondaires"]
    pairs = primary + PROTO["secondaires"] + extra
    assert tests.shape == (10, 1880) and all(
        d[name].shape == tests.shape for name in names
    )
    folds = []
    classes = []
    overall = {}
    correct = {}
    for k in range(10):
        np.testing.assert_array_equal(tests[k], a["test" + str(k + 1)])
        np.testing.assert_array_equal(np.bincount(truth[k]), np.full(47, 40))
    for name in names:
        pred = d[name]
        assert np.all((pred >= 0) & (pred < 47))
        ok = pred == truth
        correct[name] = ok.astype("int8")
        mean_fold = [float(ok[k].mean() * 100) for k in range(10)]
        class_scores = []
        for c, cn in enumerate(a["traits"]):
            hits = int(np.sum(ok & (truth == c)))
            val = 100 * hits / (10 * 40)
            class_scores.append(val)
            classes.append(
                dict(methode=name, classe=cn, correctes=hits, decisions=400, rappel=val)
            )
        macro = float(np.mean(class_scores))
        np.testing.assert_allclose(macro, np.mean(mean_fold), atol=1e-13)
        overall[name] = dict(rappel=macro, par_partition=mean_fold)
        for i, score in enumerate(mean_fold):
            folds.append(
                dict(
                    methode=name,
                    partition=i + 1,
                    train=int(d["n_train"][i]),
                    test=1880,
                    rappel=score,
                )
            )
    key, inv = np.unique(groups[tests].ravel(), return_inverse=True)
    inv = inv.reshape(tests.shape)
    influence = np.zeros((len(key), len(pairs)))
    points = []
    contrasts = {}
    for j, tag in enumerate(pairs):
        model, base = tag.split(" - ")
        delta = correct[model] - correct[base]
        points.append(100 * float(delta.mean()))
        diffs = 100 * delta.mean(1)
        for k in range(10):
            means = np.bincount(truth[k], weights=delta[k], minlength=47) / 40
            values = (delta[k] - means[truth[k]]) * (100.0 / tests.size)
            np.add.at(influence[:, j], inv[k], values)
        contrasts[tag] = dict(
            ecart=points[-1],
            plis_favorables=int((diffs > 0).sum()),
            plis_egaux=int((diffs == 0).sum()),
            corrigees=int((delta == 1).sum()),
            degradees=int((delta == -1).sum()),
            par_partition=diffs.tolist(),
        )
        np.testing.assert_allclose(
            points[-1], overall[model]["rappel"] - overall[base]["rappel"], atol=1e-12
        )
    np.testing.assert_allclose(influence.sum(0), 0, atol=1e-11)
    # Un seul multiplicateur par groupe, commun aux dix partitions et à tous les modèles.
    rng = np.random.default_rng(19112026)
    estimates = np.empty((30000, len(pairs)))
    for first in range(0, len(estimates), 200):
        n = min(200, len(estimates) - first)
        signs = 2 * rng.integers(0, 2, size=(n, len(key)), dtype="int8") - 1
        estimates[first : first + n] = signs.astype(float) @ influence + points
    for j, tag in enumerate(pairs):
        alpha = 0.05 / (len(primary) if tag in primary else 1)
        contrasts[tag]["ic"] = np.quantile(
            estimates[:, j], [alpha / 2, 1 - alpha / 2]
        ).tolist()
        contrasts[tag]["type_ic"] = "simultane_3" if tag in primary else "nominal"
    # Incertitude conditionnelle vérifiée par la formule analytique de variance des multiplicateurs.
    np.testing.assert_allclose(
        estimates.std(0), np.linalg.norm(influence, axis=0), rtol=0.03, atol=1e-10
    )
    gn = np.bincount(groups)
    nonempty = gn[gn > 0]
    counts = dict(
        n_photos=len(a["ids"]),
        n_photos_en_test=len(np.unique(tests)),
        decisions_par_methode=tests.size,
        groupes_en_test=len(key),
        groupes_exact_doublons=int((nonempty > 1).sum()),
        photos_dans_groupes_doublons=int(nonempty[nonempty > 1].sum()),
        photos_apprentissage_ecartees_par_partition=(3760 - d["n_train"]).tolist(),
    )
    result = dict(comptages=counts, methodes=overall, contrastes=contrasts)
    (out / "bilan.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    table(out / "scores_partitions.csv", folds)
    table(out / "scores_classes.csv", classes)
    table(
        out / "contrastes.csv",
        [
            dict(
                comparaison=k,
                ecart=v["ecart"],
                ic_bas=v["ic"][0],
                ic_haut=v["ic"][1],
                type_ic=v["type_ic"],
                plis_favorables=v["plis_favorables"],
                corrigees=v["corrigees"],
                degradees=v["degradees"],
            )
            for k, v in contrasts.items()
        ],
    )
    print(
        "Calcul terminé pour",
        len(overall),
        "méthodes et",
        len(contrasts),
        "contrastes.",
    )
    print(
        json.dumps(
            {
                k: [round(v["ecart"], 2), *[round(t, 2) for t in v["ic"]]]
                for k, v in contrasts.items()
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("predictions", type=Path)
    ap.add_argument("sortie", type=Path)
    args = ap.parse_args()
    with threadpool_limits(2):
        analyse(args.predictions, args.sortie)
