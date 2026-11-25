"""Relire les décisions déjà produites. Les intervalles conditionnent sur les modèles ajustés."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
LOTS = ("iNaturalist 2018–2020", "iNaturalist 2025", "Pl@ntNet", "iNaturalist 2026")
FLEURS = ("pâquerette", "pissenlit", "tournesol")
NBOOT = 60000


def read(path):
    with path.open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def col(rows, k):
    return np.array([r[k] for r in rows])


def recall(truth, pred):
    return float(
        np.mean([100 * (pred[truth == c] == c).mean() for c in np.unique(truth)])
    )


def diff(truth, a, b):
    return (a == truth).astype(np.int8) - (b == truth).astype(np.int8)


def interval(sims, coverage=0.95):
    q = (1 - coverage) / 2
    return np.quantile(sims, [q, 1 - q]).tolist()


def compare(rows, a, b, rng, group=None):
    truth = col(rows, "vraie_classe")
    pa, pb = col(rows, a), col(rows, b)
    d = diff(truth, pa, pb)
    estimates = []
    for label in np.unique(truth):
        sel = truth == label
        if group:
            g = col(rows, group)[sel]
            vals = np.array([d[sel][g == v].mean() for v in np.unique(g)])
            count = len(vals)
            mean_draws = (
                rng.multinomial(count, np.full(count, 1 / count), size=NBOOT)
                @ vals
                / count
            )
        else:
            vals, counts = np.unique(d[sel], return_counts=True)
            count = int(counts.sum())
            mean_draws = (
                rng.multinomial(count, counts / count, size=NBOOT) @ vals / count
            )
        estimates.append(mean_draws)
    sims = 100 * np.mean(estimates, axis=0)
    ra, rb = recall(truth, pa), recall(truth, pb)
    return dict(
        methode=a,
        temoin=b,
        n=len(d),
        score=ra,
        reference=rb,
        gain=ra - rb,
        ic95=interval(sims),
        corrigees=int((d > 0).sum()),
        degradees=int((d < 0).sum()),
    ), sims


def flower_audit(rows, rng):
    summary = []
    byclass = []
    stored = {}
    for lot in LOTS:
        subset = [r for r in rows if r["lot"] == lot]
        stats, sims = compare(subset, "deux_liens", "additif", rng)
        summary.append(
            dict(lot=lot, **stats, ic_simultane_quatre=interval(sims, 1 - 0.05 / 4))
        )
        truth = col(subset, "vraie_classe")
        for c in FLEURS:
            sub = [r for r in subset if r["vraie_classe"] == c]
            row, draws = compare(sub, "deux_liens", "additif", rng)
            comparators = {
                key: 100 * (col(sub, key) == c).mean()
                for key in ("RGB", "F7_RGB", "F8_RGB", "additif")
            }
            best = max(comparators.values())
            byclass.append(
                dict(
                    lot=lot,
                    classe=c,
                    **row,
                    ic_simultane_douze=interval(draws, 1 - 0.05 / 12),
                    meilleur_temoin=" / ".join(
                        k for k in comparators if comparators[k] == best
                    ),
                    meilleur_score=float(best),
                    ecart_meilleur=float(row["score"] - best),
                )
            )
        assert len(np.unique(truth)) == 3
        if lot == "Pl@ntNet":
            for key in ["F7xF8_seul", "F8xRGB_seul", "additif_diviseur_commun"]:
                stored[key] = compare(subset, "deux_liens", key, rng)[0]
    return summary, byclass, stored


if __name__=="__main__":
    p=argparse.ArgumentParser(description="Recompute the four class-stratified transfer comparisons from per-photo decisions.")
    p.add_argument("data",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    a.output.mkdir(exist_ok=True,parents=True)
    fs,fc,_=flower_audit(read(a.data),np.random.default_rng(19112026))
    for name,rows in [("fleurs_par_lot",fs),("fleurs_par_classe",fc)]:
        out=[]
        for r in rows:
            dest={}
            for k,v in r.items():
                if isinstance(v,list):dest[k+"_bas"],dest[k+"_haut"]=v
                else:dest[k]=v
            out.append(dest)
        write(a.output/(name+".csv"),out)
