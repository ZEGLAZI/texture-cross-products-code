"""Vérifier indépendamment la première partition avec l'identité factorisée du noyau HSMT."""

import argparse
import json
from pathlib import Path

import numpy as np
from predire import load_blocks
from reference.calcul_historique import compact
from scipy.linalg import solve
from threadpoolctl import threadpool_limits


def verifier(cache, result):
    meta = dict(np.load(result / "identites.npz"))
    prediction = dict(np.load(result / "decisions.npz"))
    te = meta["test1"]
    tr = np.r_[meta["train1"], meta["val1"]]
    tr = tr[~np.isin(meta["groupes"][tr], meta["groupes"][te])]
    f = load_blocks(cache / "hsmt", meta["ids"], ["F7", "F8", "rgb"])
    x, _ = compact(f, training=tr)
    blocks = (x[:, :945] * np.sqrt(2.0), x[:, 945:] * np.sqrt(2.0), f["rgb"])

    def factorise(ii):
        s, w, c = (b[ii] @ b[tr].T for b in blocks)
        return ((1 + w * 0.5) * (1 + s * 0.5 + c) - 1) / 3.25

    # Solve symétrique indépendant de la routine Cholesky employée par le calcul principal.
    gram = factorise(tr)
    np.fill_diagonal(gram, gram.diagonal() + 1)
    y = np.eye(47)[meta["y"][tr]]
    prior = y.mean(0)
    alpha = solve(gram, y - prior, assume_a="sym", check_finite=True)
    pred = np.argmax(factorise(te) @ alpha + prior, axis=1)
    np.testing.assert_array_equal(te, prediction["tests"][0])
    np.testing.assert_array_equal(pred, prediction["HSMT"][0])
    report = {
        "partition": 1,
        "train": len(tr),
        "test": len(te),
        "decisions_identiques": int((pred == prediction["HSMT"][0]).sum()),
    }
    (result / "verification_independante.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("resultats", type=Path)
    args = p.parse_args()
    with threadpool_limits(3):
        verifier(args.cache, args.resultats)
