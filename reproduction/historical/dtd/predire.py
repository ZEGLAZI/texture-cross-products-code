"""Ajuster les lecteurs figés de DTD et enregistrer une décision par image et méthode."""

import argparse
import json
from pathlib import Path

import numpy as np
from preparer import PROTO, ROOT, empreinte
from reference.calcul_historique import compact
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

EXTRA = json.loads((ROOT / "controle_echelle.json").read_text())
METHODES = list(PROTO["methodes"]) + list(EXTRA["methodes"])
FAMILLE = (
    "Additif_fixe",
    "Pivot_F7",
    "Pivot_RGB",
    "Additif_historique",
    "HSMT",
    "DINO_Additif",
    "DINO_HSMT",
    "Additif_diag_egal",
    "DINO_Additif_diag_egal",
)


def load_blocks(folder, ids, names):
    chunks = {n: [] for n in ("ids", *names)}
    for p in sorted(folder.glob("[0-9][0-9][0-9][0-9][0-9].npz")):
        with np.load(p) as a:
            for n in chunks:
                chunks[n].append(a[n])
    result = {k: np.concatenate(v) for k, v in chunks.items()}
    np.testing.assert_array_equal(result.pop("ids"), ids)
    return result


def groups_exact(byte, pix):
    parent = np.arange(len(byte))
    first = {}

    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, row in enumerate(zip(byte, pix)):
        for typ, value in enumerate(row):
            j = first.setdefault((typ, value), i)
            a, b = root(i), root(j)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return np.array([root(i) for i in range(len(byte))])


def normal(x):
    x = np.asarray(x, dtype=np.float64)
    k = np.linalg.norm(x, axis=1, keepdims=True)
    if not (k > 0).all():
        raise ValueError("Vecteur nul")
    return x / k


def ridge(kernel, n, y_train, n_classes=47):
    target = np.eye(n_classes)[y_train]
    prior = target.mean(0)
    target -= prior
    train = kernel[:n]
    matrix = np.array(train, dtype=np.float64, order="F", copy=True)
    matrix.flat[:: n + 1] += 1.0
    fact = cho_factor(matrix, lower=True, overwrite_a=True, check_finite=False)
    alpha = cho_solve(fact, target, check_finite=False)
    err = np.max(np.abs(train @ alpha + alpha - target))
    if not np.isfinite(err) or err > 1e-8:
        raise ValueError(f"Résidu du lecteur {err}")
    scores = kernel[n:] @ alpha + prior
    if not np.isfinite(scores).all():
        raise ValueError("Scores non finis")
    return np.argmax(scores, axis=1).astype(np.uint8), float(err)


def partitions(index, groups):
    for k in range(1, 11):
        dev = np.r_[index["train" + str(k)], index["val" + str(k)]]
        te = index["test" + str(k)]
        tr = dev[~np.isin(groups[dev], groups[te])]
        assert len(np.intersect1d(groups[tr], groups[te])) == 0 and len(te) == 1880
        yield k, tr, te


def base_lecteur(name, feat, index, groups, where):
    gram = feat @ feat.T
    np.testing.assert_allclose(np.diag(gram), 1, atol=1e-12)
    for k, tr, te in partitions(index, groups):
        kernel = gram[np.ix_(np.r_[tr, te], tr)]
        pred, err = ridge(kernel, len(tr), index["y"][tr])
        np.savez(where / f"{name}_{k}.npz", test=te, train=tr, pred=pred, err=err)
        del kernel
    print(name, "10 lecteurs résolus", flush=True)


def remplir(name, dest, grams, ratio, dino, tr):
    ss, ww, cc = grams
    for start in range(0, len(dest), 256):
        q = slice(start, start + 256)
        s, w, c = ss[q], ww[q], cc[q]
        t = (s + w) * 0.5 + c
        if name in ("HSMT", "DINO_HSMT"):
            t += s * w * 0.25 + w * c * 0.5
        elif name == "Pivot_F7":
            t += s * w * 0.25 + s * c * 0.5
        elif name == "Pivot_RGB":
            t += s * c * 0.25 + w * c * 0.5
        elif name not in (
            "Additif_fixe",
            "Additif_historique",
            "DINO_Additif",
            "Additif_diag_egal",
            "DINO_Additif_diag_egal",
        ):
            raise ValueError(name)
        t /= 2.0 if name == "Additif_historique" else 3.25
        if name in ("Additif_diag_egal", "DINO_Additif_diag_egal"):
            t *= ratio[q, None] * ratio[None, : len(tr)]
        if name.startswith("DINO_"):
            t = (t + dino[q]) * 0.5
        dest[q] = t


def famille(index, raw, dinofeat, groups, where):
    for k, tr, te in partitions(index, groups):
        order = np.r_[tr, te]
        n = len(tr)
        fitted, params = compact(raw, training=tr)
        # La même transformation est applicable à une photo de test isolée.
        probe = te[::151]
        transformed, _ = compact(
            {name: raw[name][probe] for name in ("F7", "F8")}, params=params
        )
        np.testing.assert_allclose(fitted[probe], transformed, rtol=0, atol=1e-14)
        f = fitted[order] * np.sqrt(2.0)
        rgb = raw["rgb"][order]
        blocks = (f[:, :945], f[:, 945:], rgb)
        assert [x.shape[1] for x in blocks] == [945, 504, 64]
        norms = [np.einsum("ij,ij->i", x, x) for x in blocks]
        grams = [x @ x[:n].T for x in blocks]
        for g, diag in zip(grams, norms):
            np.testing.assert_allclose(np.diag(g[:n]), diag[:n], atol=1e-12)
        s, w, c = norms
        t = (s + w) * 0.5 + c
        ratio = np.sqrt((t + s * w * 0.25 + w * c * 0.5) / t)
        d = dinofeat[order]
        dk = d @ d[:n].T
        kernel = np.empty_like(grams[0])
        preds = {}
        for name in FAMILLE:
            remplir(name, kernel, grams, ratio, dk, tr)
            if name == "HSMT":
                gs, gw, gc = [g[::121, :8] for g in grams]
                expected = ((1 + gw * 0.5) * (1 + gs * 0.5 + gc) - 1) / 3.25
                np.testing.assert_allclose(kernel[::121, :8], expected, atol=5e-15)
            if name == "Additif_diag_egal":
                np.testing.assert_allclose(
                    np.diag(kernel[:n]),
                    (t + s * w * 0.25 + w * c * 0.5)[:n] / 3.25,
                    atol=1e-12,
                )
            pred, err = ridge(kernel, n, index["y"][tr])
            preds[name] = pred
            preds["erreur_" + name] = err
        np.savez(where / f"famille_{k}.npz", test=te, train=tr, **preds)
        del grams, dk, kernel
        print("HSMT, partition", k, "sur 10", flush=True)


def main(data, cache, sortie, phase):
    sortie.mkdir(parents=True, exist_ok=True)
    cache_preds = sortie / "lecteurs"
    cache_preds.mkdir(exist_ok=True)
    index = dict(np.load(data / "index.npz"))
    ids = index["ids"]
    cc = load_blocks(cache / "clbp", ids, ["clbp", "source_sha256", "pixels224"])
    groups = groups_exact(cc["source_sha256"], cc["pixels224"])
    np.savez_compressed(
        sortie / "identites.npz",
        ids=ids,
        traits=index["traits"],
        y=index["y"],
        groupes=groups,
        source_sha256=cc["source_sha256"],
        pixels224=cc["pixels224"],
        **{k: v for k, v in index.items() if k.startswith(("train", "val", "test"))},
    )
    if phase in ("bases", "tout"):
        counts = cc.pop("clbp")
        hist = np.sqrt(counts / counts.sum(1, keepdims=True))
        del counts
        base_lecteur("CLBP", hist, index, groups, cache_preds)
        del hist
        decaf = normal(np.load(data / "decaf.npy", mmap_mode="r"))
        base_lecteur("DeCAF", decaf, index, groups, cache_preds)
        del decaf
    if phase in ("famille", "tout"):
        dino = normal(load_blocks(cache / "dino", ids, ["dino"])["dino"])
        base_lecteur("DINOv2", dino, index, groups, cache_preds)
        raw = load_blocks(cache / "hsmt", ids, ["F7", "F8", "rgb"])
        np.testing.assert_allclose(np.square(raw["rgb"]).sum(1), 1, atol=2e-14)
        famille(index, raw, dino, groups, cache_preds)
    if phase == "tout" or phase == "assembler":
        preds = {name: [] for name in METHODES}
        tests = []
        trains = []
        errors = []
        for k, tr, te in partitions(index, groups):
            tests.append(te)
            trains.append(len(tr))
            for name in ("CLBP", "DeCAF", "DINOv2"):
                with np.load(cache_preds / f"{name}_{k}.npz") as read:
                    np.testing.assert_array_equal(read["test"], te)
                    np.testing.assert_array_equal(read["train"], tr)
                    preds[name].append(read["pred"])
                    errors.append(float(read["err"]))
            with np.load(cache_preds / f"famille_{k}.npz") as read:
                np.testing.assert_array_equal(read["test"], te)
                np.testing.assert_array_equal(read["train"], tr)
                for name in FAMILLE:
                    preds[name].append(read[name])
                    errors.append(float(read["erreur_" + name]))
        np.savez_compressed(
            sortie / "decisions.npz", tests=tests, n_train=trains, **preds
        )
        (sortie / "controles.json").write_text(
            json.dumps(
                {
                    "residu_ridge_max": max(errors),
                    "lecteurs": len(errors),
                    "protocole_sha256": empreinte(ROOT / "protocole.json"),
                    "controle_echelle_sha256": empreinte(
                        ROOT / "controle_echelle.json"
                    ),
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", type=Path)
    ap.add_argument("cache", type=Path)
    ap.add_argument("sortie", type=Path)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--phase", choices=["tout", "bases", "famille", "assembler"], default="tout"
    )
    a = ap.parse_args()
    with threadpool_limits(a.threads):
        main(a.data, a.cache, a.sortie, a.phase)
