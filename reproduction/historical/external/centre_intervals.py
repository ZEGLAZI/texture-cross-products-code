"""Reproduce the six retrospective simultaneous intervals from paired predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
PIVOTS = ("HSMT", "Pivot_F7", "Pivot_RGB")
def run(file,out):
    z=np.load(file)
    rows={t:(z[t+"_vrai"], z[t+"_groupe"], {k:z[t+"_"+k] for k in PIVOTS}) for t in ("plantnet","nouveau_2026","KTH")}
    out.mkdir(parents=True,exist_ok=True)
    result = {}
    all_decisions = {}
    rng = np.random.default_rng(19112026)
    for tag, (y, group, res) in rows.items():
        cats = np.unique(y)
        assert all((v >= 0).all() for v in res.values())
        correct = np.array([res[k] == y for k in PIVOTS], dtype=np.int8).T
        diffs = correct[:, [0]] - correct[:, 1:]
        sims = np.zeros((30000, 2))
        scores = []
        for c in cats:
            ix = y == c
            scores.append(correct[ix].mean(0))
            values = diffs[ix]
            if tag == "KTH":
                v = np.stack(
                    [values[group[ix] == s].mean(0) for s in np.unique(group[ix])]
                )
                n = len(v)
                sims += (rng.multinomial(n, np.full(n, 1 / n), size=len(sims)) @ v) / n
            else:
                v, counts = np.unique(values, axis=0, return_counts=True)
                n = len(values)
                sims += (rng.multinomial(n, counts / n, size=len(sims)) @ v) / n
        points = 100 * np.mean(scores, axis=0)
        sims *= 100 / len(cats)
        alpha = 0.05 / 6
        ic = np.quantile(sims, [alpha / 2, 1 - alpha / 2], axis=0)
        result[tag] = {
            "n": len(y),
            "scores": dict(zip(PIVOTS, points.tolist())),
            "vs": {
                k: {
                    "ecart": float(points[0] - points[j]),
                    "ic_simultane_six": ic[:, j - 1].tolist(),
                }
                for j, k in enumerate(PIVOTS[1:], 1)
            },
        }
        all_decisions.update(
            {tag + "_" + k: v for k, v in {"vrai": y, "groupe": group, **res}.items()}
        )
    (out / "pivots_archives.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("file",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();run(a.file,a.output)
