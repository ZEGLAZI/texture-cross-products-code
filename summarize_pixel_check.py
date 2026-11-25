"""Compare a second CPU extraction and its new fits with the released descriptors and decisions."""
import argparse
import json
from pathlib import Path

import numpy as np
from pixel_files import transformer_file
from threadpoolctl import threadpool_limits


def macro(delta, labels):
    return 100.*float(np.mean([np.mean([a[b==c].mean() for c in np.unique(b)])
                               for a, b in zip(delta, labels)]))


def paired_interval(corpus, diff, y, groups, family=2):
    # The DTD multipliers are aggregated by magnitude, and the KTH bootstrap draws
    # four specimens per material. This reproduces the original Monte Carlo order.
    rng = np.random.default_rng(19112026)
    count = 100000
    estimates = np.zeros(count)
    if corpus == "DTD":
        _, group_index = np.unique(groups, return_inverse=True)
        group_index = group_index.reshape(y.shape)
        influence = np.zeros(int(group_index.max()) + 1, dtype=np.int64)
        for labels, change, indices in zip(y, diff, group_index):
            class_counts = np.bincount(labels)
            np.testing.assert_array_equal(class_counts, np.repeat(40,47))
            totals = np.bincount(labels, weights=change).astype(int)
            np.add.at(influence, indices, 40*change-totals[labels])
        absolute, frequencies = np.unique(np.abs(influence[influence != 0]), return_counts=True)
        for mag, total in zip(absolute, frequencies):
            estimates += mag*(2*rng.binomial(total,.5,count)-total)
        estimates = macro(diff,y)+estimates*100./(diff.size*40)
    else:
        classes = np.unique(y)
        for c in classes:
            ok = y==c
            specimens = groups[ok]
            change = diff[ok]
            specimen_means = [change[specimens==s].mean() for s in np.unique(specimens)]
            if len(specimen_means) != 4:
                raise ValueError("Expected four KTH specimens per material")
            values, multiplicities = np.unique(specimen_means, return_counts=True)
            weights = rng.multinomial(4,multiplicities/4.,size=count)
            estimates += (weights@values)/4.
        estimates *= 100./len(classes)
    return np.quantile(estimates,[.025/family,1.-.025/family]).tolist()


def compare_colour_arrays(data, fresh, corpus):
    fresh = Path(fresh)
    saved = json.loads((fresh/(corpus+"_colour_check.json")).read_text())
    with np.load(fresh/(corpus+"_colour.npz")) as measured:
        keys = ("F7","F8","rgb")
        if corpus == "DTD":
            chunks = {key: [] for key in ("ids", *keys)}
            for f in sorted((data/"review2/dtd_cache/hsmt").glob("[0-9]*.npz")):
                with np.load(f) as z:
                    for key in chunks:
                        chunks[key].append(z[key])
            values = {key:np.concatenate(parts) for key,parts in chunks.items()}
            np.testing.assert_array_equal(measured["ids"],values.pop("ids"))
        else:
            with np.load(data/"historical/external/data/kth.npz") as z:
                np.testing.assert_array_equal(measured["ids"],z["path"])
                values={key:z[key] for key in keys}
        for key in keys:
            diff=np.abs(measured[key].astype(np.float64)-values[key].astype(np.float64))
            observed=(diff.size,np.count_nonzero(diff),diff.max(),diff.mean(),
                      np.count_nonzero(diff.reshape(len(diff),-1).any(axis=1)))
            record=saved["blocks"][key]
            expected=tuple(record[k] for k in ("coordinates","different","max_absolute","mean_absolute","differing_images"))
            np.testing.assert_allclose(observed,expected,rtol=0.,atol=1.e-13)
        if corpus == "DTD":
            with np.load(data/"historical/dtd/results/golden_48.npz") as gold:
                index={name:i for i,name in enumerate(measured["ids"])}
                ix=[index[k] for k in gold["ids"]]
                saved["reference_48"]={k:{"coordinates":int(gold[k].size),
                                          "different":int(np.count_nonzero(measured[k][ix]!=gold[k]))} for k in keys}
    return saved


def summarize(corpus, data, fresh, fits):
    cross = data/"cross_representation"
    with np.load(cross/f"features/{corpus}_features.npz") as ref, np.load(transformer_file(data,fresh,corpus)) as pixel:
        np.testing.assert_array_equal(pixel["ids"], ref["ids"])
        coords = {}
        for key in ("patch3","patch6","patch12","cls12"):
            a,b=pixel[key],ref[key]
            coords[key]=dict(coordinates=int(a.size),different=int(np.count_nonzero(a!=b)),
                             max_absolute=float(np.max(np.abs(a-b))))
    with np.load(fits/(corpus+"_predictions.npz")) as z,np.load(cross/f"results/{corpus}_predictions.npz") as ref,np.load(
            cross/f"scale_results/{corpus}_matched_predictions.npz") as original_matched:
        y,groups=z["labels"],z["groups"]
        for key in ("test","ids","labels","groups"):
            np.testing.assert_array_equal(z[key],ref[key])
        out={}
        for branch in ("released","new_pixels"):
            v=ref if branch=="released" else z
            vm=original_matched if branch=="released" else z
            rows={}
            contrasts={}
            for setting, prefix, source, q_suffix in [("native","_",v,"_Q"),("matched","_",vm,"_matched_Q"),
                                                      ("selected","_tuned_",v,"_tuned_Q")]:
                differences={}
                for family in ("H","D"):
                    sq=family+prefix+"S"
                    q=family+q_suffix
                    differences[family]=(source[q]==y).astype(int)-(v[sq]==y).astype(int)
                direct=differences["D"]-differences["H"]
                contrasts[setting]=direct
                rows[setting]=dict(colour_gain_pp=macro(differences["H"],y),DINO_gain_pp=macro(differences["D"],y),
                                   contrast_pp=macro(direct,y),interval_pp=paired_interval(corpus,direct,y,groups,1 if setting=="selected" else 2))
            shift=contrasts["selected"]-contrasts["native"]
            rows["shift"]=dict(estimate_pp=macro(shift,y), adjusted_interval_pp=paired_interval(corpus,shift,y,groups,2))
            out[branch]=rows
        original=json.loads((cross/"results/regime_change.json").read_text())[corpus]
        np.testing.assert_allclose(out["released"]["shift"]["adjusted_interval_pp"],original["bonferroni_two_datasets_95_pp"],rtol=0,atol=1.e-10)
        out["test_appearances"]=int(y.size)
        changes={}
        for key in (k for k in z.files if k not in ("test","ids","labels","groups")):
            original_z=original_matched if "_matched_" in key else ref
            changes[key]=int(np.count_nonzero(original_z[key]!=z[key]))
        out["test_label_changes"]=changes
    out["transformer_blocks"]=coords
    out["colour_extraction"]=compare_colour_arrays(data, fresh, corpus)
    info=json.loads((fits/(corpus+"_refit.json")).read_text())
    out["selected_penalties_changed"]=sum(sum(a!=b for a,b in zip(row["selections"],row["original_selections"])) for row in info["splits"])
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("data","fresh","fits","output"):
        parser.add_argument(key,type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        results={c:summarize(c,args.data,args.fresh,args.fits) for c in ("DTD","KTH")}
    args.output.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    for c, result in results.items():
        print(c, result["new_pixels"]["shift"], flush=True)
