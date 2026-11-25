"""Recalculate additional empirical summaries from decisions and stored results."""
from collections import Counter
from pathlib import Path
from statistics import mean
from numerical_checks import DATASET_LABELS, columns, csv_rows, read_json

PATTERNS = {"checkered": "Checkered", "dotted": "Dotted", "floral": "Floral",
            "solid": "Solid", "striped": "Striped", "zig zag": "Zigzag"}
RIDGES = [1, .1, 10, .01, 100, .001]


def representations(tables, out):
    for row in csv_rows(out / "cross_tables/representation_comparison.csv"):
        dataset = DATASET_LABELS[row["dataset"]]
        for key, name in [("H", "Colour–texture"), ("D", "DINOv2")]:
            label = f"{dataset} {row['penalty'].title()} {name}"
            tables.record("S1", label, columns(row, [key + suffix for suffix in ("_S", "_Q", "_gain", "_lo", "_hi")]))
    for dataset, row in read_json(out / "cross_tables/regime_change.json").items():
        values = [row["D_fixed_pp"], row["D_selected_pp"], row["T_pp"],
                  *row["pointwise_95_pp"], *row["bonferroni_two_datasets_95_pp"]]
        tables.record("S2", DATASET_LABELS[dataset], values)
    corpus_names = {"KTH":"KTH-TIPS2-b", "iNat 2026":"iNaturalist 2026",
                    "iNat 2015-17":"iNaturalist 2015–2017"}
    regimes = {"fixed":"Original coefficients", "unit":"Unit diagonal", "tuned":"Selected ridge"}
    for row in csv_rows(out / "paper/aggregate/all_controls.csv"):
        if row["profile"] != "0":
            continue
        corpus, mode = row["corpus"], row["mode"]
        if mode == "fixed":
            tables.record("S5", corpus_names.get(corpus, corpus), columns(row, "LSXQ"))
        if corpus in ("DTD", "Clothing") and mode in regimes:
            keys = ("Q_minus_S","Q_minus_S_lo","Q_minus_S_hi","Q_minus_X","Q_minus_X_lo","Q_minus_X_hi")
            tables.record("S6", corpus + " " + regimes[mode], columns(row, keys))


def clothing(tables, data, out, mode):
    ledger = csv_rows(data / "reader/tables/clothing_attempts.csv")
    for key, label in [*PATTERNS.items(), (None, "Total")]:
        records = [r for r in ledger if key is None or r["pattern"] == key]
        counts = Counter(r["outcome"] for r in records)
        order = ("http","connection","decode","geometry","label_conflict","duplicate","dtd_overlap","retained")
        tables.record("S7", label, [len(records), *(counts[k] for k in order)])
    host_dir = out / "hosts" if mode == "full" else data / "reader/hosts"
    for table, records, truth_key, photo_key in [
        ("S8",csv_rows(data / "clothing/results/predictions.csv"),"class","source_key"),
        ("S9",csv_rows(host_dir / "decisions.csv"),"truth","key"),
    ]:
        scores = []
        for key, label in PATTERNS.items():
            group = [r for r in records if r[truth_key] == key]
            n = len({r[photo_key] for r in group})
            if len(group) != 30*n:
                raise ValueError(f"{table} {key}: incomplete set of repeated predictions")
            square, product = [100 * mean(r[m] == key for r in group) for m in ("squares","crosses")]
            diff = square - product
            hosts = len({r["hostname"] for r in group}) if table == "S9" else None
            numbers = [n, square, product, diff, diff/6] if hosts is None else [n, hosts, square, product, diff]
            tables.record(table, label, numbers)
            scores.append((square, product, diff))
        if table == "S9":
            tables.record(table, "Macro recall", [len({r[photo_key] for r in records}),
                               len({r["hostname"] for r in records}), *map(mean, zip(*scores, strict=True))])
    selected = csv_rows(out / "paper/aggregate/selected_settings.csv")
    for dataset, prefix in [("DTD","DTD_"),("Clothing","Textile_")]:
        for key, name in [("RS","Squares"),("RX","Products"),("Q","Full quadratic"),("RBF","Gaussian")]:
            fits = [r for r in selected if r["fold"].startswith(prefix) and r["method"] == key]
            settings = Counter(float(r["gamma" if key == "RBF" else "profile"]) for r in fits)
            penalties = Counter(float(r["penalty"]) for r in fits)
            order = [.25,1,4,16] if key == "RBF" else range(4)
            tables.record("S10", dataset + " " + name, [*(settings[i] for i in order), *(penalties[i] for i in RIDGES)])
    for key, name in [("squares","Squares"),("crosses","Products")]:
        fits = [r for r in csv_rows(host_dir / "settings.csv") if r["model"] == key]
        settings = Counter(int(r["profile"]) for r in fits)
        penalties = Counter(float(r["ridge"]) for r in fits)
        tables.record("S10", "Host " + name, [*(settings[i] for i in range(4)), *(penalties[i] for i in RIDGES)])


def historical(tables, data, mode):
    h = data / "historical"
    for row in csv_rows(h / "revision/results/dtd_folds.csv"):
        tables.record("S11",row["split"],columns(row,("W_Add_fixed","W_Add_selected","W_Q_selected")))
    base = read_json(h / "dtd/results/bilan.json")["methodes"]
    oc = read_json(h / "extra/oclbp.json")["score"]
    tables.record("S12","OCLBP",[432, oc])
    baselines = [("CLBP","CLBP",[1352]),("Fixed KW","HSMT",[1513]),("DeCAF","DeCAF",[4096]),
                 ("DINO","DINOv2",[384]),("DINO and KA","DINO_Additif",[384,1513]),
                 ("DINO and diagonal-matched KA","DINO_Additif_diag_egal",[384,1513]),
                 ("DINO and KW","DINO_HSMT",[384,1513])]
    for label, key, sizes in baselines:
        tables.record("S12",label,[*sizes,base[key]["rappel"]])
    dense = read_json(h / "extra/dense.json")
    tables.record("S13","DTD",[base["Additif_fixe"]["rappel"],dense["scores"]["all_pairs"],
                              base["HSMT"]["rappel"],dense["delta"],*dense["ci"]])
    others = read_json(h / "external/results/external.json")
    for key, label in [("KTH","KTH-TIPS2-b"),("plantnet","Pl@ntNet"),("nouveau_2026","iNaturalist 2026")]:
        row = others[key]
        tables.record("S13",label,[*columns(row["scores"],("Add","dense","HSMT")),row["delta"],*row["ci"]])
    from studies.oxford_masks.check import check as refit_oxford
    masks = refit_oxford(data)[0] if mode == "full" else read_json(Path(__file__).resolve().parent / "studies/oxford_masks/results/scores.json")
    for key, name in [("whole","Whole image"),("foreground","Foreground only"),
                      ("background","Background only"),("outline","Silhouette")]:
        tables.record("S14",name,columns(masks[key],("fixed_Add","fixed_W","fixed_Q")))
    for row in csv_rows(h / "external/results/fleurs_par_lot.csv"):
        values = columns(row, ("n","reference","score","gain","ic_simultane_quatre_bas","ic_simultane_quatre_haut"))
        tables.record("S15", row["lot"], values)


def check_supplementary_tables(tables, data, out, mode):
    representations(tables, out)
    clothing(tables, data, out, mode)
    historical(tables, data, mode)
