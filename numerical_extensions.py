"""Check numerical consistency in transfer, encoder and catalogue experiments.

This module reads only data/results. It does not compare typeset documents.
Collected summary rows are outputs, not document-verification claims.
"""
from collections import Counter
from pathlib import Path
import sys
import numpy as np
from numerical_checks import SummaryRows, csv_rows, read_json, columns


def check_transfer_and_colour(root):
    transfer = root / "studies/CVIU_Experience_couleur/results"
    colour = root / "studies/CVIU_Dependances_chromatiques/results"
    fits = root / "studies/source_refit/results"
    table = SummaryRows()
    transfer_scores = {(r["regime"], r["representation"]): r for r in csv_rows(transfer / "external_kernels.csv")}
    transfer_contrasts = {r["contrast"]: r for r in csv_rows(transfer / "external_contrasts.csv")}
    for regime, label in [("fixed","Native"), ("matched","Matched"), ("selected","Selected")]:
        estimates = [transfer_scores[regime, representation][kernel] for representation in ("H","D")
                     for kernel in ("S_pct","Q_pct")]
        contrast = transfer_contrasts["DINO_gain_minus_handcrafted_" + regime]
        table.record("S20", label, estimates + [contrast[k] for k in ("gain_pp", "low", "high")])

    summary = read_json(fits / "bootstrap_summary.json")
    draws = csv_rows(fits / "bootstrap_draws.csv")
    if list(map(int, (row["replicate"] for row in draws))) != list(range(999)):
        raise ValueError("Bootstrap draws are incomplete or unordered.")
    penalties = (1., .1, 10., .01, 100., .001)
    classifiers = [("H_S", "Colour squares"), ("H_Q", "Colour full"),
                   ("D_S", "DINOv2 squares"), ("D_Q", "DINOv2 full")]
    for key, label in classifiers:
        counts = Counter(float(row["lambda_" + key]) for row in draws)
        expected = [counts[p] for p in penalties]
        if expected != [summary["penalty_counts"][key][str(p)] for p in penalties] or sum(expected) != len(draws):
            raise ValueError("Penalty frequency mismatch for " + label)
        table.record("S21", label, expected)
    recomputed = []
    for row in draws:
        contrasts = []
        for mode in ("fixed","selected"):
            d = (float(row[f"D_{mode}_Q"]) - float(row[f"D_{mode}_S"])
                 - float(row[f"H_{mode}_Q"]) + float(row[f"H_{mode}_S"]))
            if abs(d - float(row["D_" + mode])) > 1e-8:
                raise ValueError("Inconsistent contrast in draw " + row["replicate"])
            contrasts.append(d)
        shift = contrasts[1] - contrasts[0]
        if abs(shift - float(row["T"])) > 1e-8:
            raise ValueError("Inconsistent shift in draw " + row["replicate"])
        recomputed.append(shift)
    endpoints = np.percentile(recomputed, (2.5,97.5))
    np.testing.assert_allclose(endpoints, summary["refit_percentile95_pp"], rtol=0, atol=1e-10)
    joint_ones = sum(all(float(row["lambda_" + k]) == 1 for k, _ in classifiers) for row in draws)
    if joint_ones != summary["all_four_choose_one"]:
        raise ValueError("Joint penalty count does not match the draws.")

    colour_scores = {(r["method"],r["task"]):r for r in csv_rows(colour / "scores.csv")}
    methods = [("AW", "Wavelets only"), ("AW_RGB", "RGB"), ("AW_Lab6", "Lab moments"),
               ("AW_P", "Local colour"), ("AW_J", "Oriented jump"), ("AW_Lab_P", "Lab with local colour"),
               ("AW_Lab_J", "Lab with jump"), ("AW_RGB_J", "RGB with jump")]
    for method, label in methods:
        dims = int(colour_scores[method,"DTD"]["raw_dimensions"]) - int(colour_scores["AW","DTD"]["raw_dimensions"])
        values = [colour_scores[method, d]["macro_recall_pct"] for d in ("DTD","KTH","omit_ambient","Lytro")]
        table.record("S22", label, [dims, *values])
    directions = ("temperature","direction_c1")
    alot = {(r["method"],r["protocol"]):r for r in csv_rows(colour / "ALOT_scores.csv")}
    alot_labels = [("AW","Wavelets only"), ("RGB","RGB"), ("Lab6","Lab moments"), ("RGB_J","RGB with jump"),
                   ("RGB_J_perm","RGB with permuted jump"), ("RGB_J_mean","RGB with local mean")]
    for method, label in alot_labels:
        table.record("S23", label, [alot[method,d]["macro_recall_pct"] for d in directions])
    weights = {r["task"]:r for r in csv_rows(colour / "weight_control.csv")}
    table.record("S23", "RGB with constant", [weights[d]["half_RGB_plus_constant"] for d in directions])

    direct_shift = (float(transfer_contrasts["DINO_gain_minus_handcrafted_selected"]["gain_pp"])
                    - float(transfer_contrasts["DINO_gain_minus_handcrafted_fixed"]["gain_pp"]))
    if direct_shift != summary["original_T_pp"] or direct_shift != 0:
        raise ValueError("Observed source-split shift is inconsistent.")
    if table.counts != {"S20":21, "S21":24, "S22":40, "S23":14}:
        raise ValueError("Unexpected numerical summary coverage.")
    return {"calculated_values_by_table": table.counts,
            "calculated_summary_rows": table.rows,
            "original_shift_pp": direct_shift, "bootstrap_draws": len(draws),
            "bootstrap_interval_pp": endpoints.tolist(), "four_common_penalties": joint_ones}


def check_resnet_diagnostics(root):
    results = root/"studies/resnet50/results"
    tables = SummaryRows()
    regime_labels = {"fixed":"native", "matched":"matched", "selected":"selected"}
    for row in csv_rows(results/"gains.csv"):
        if row["representation"] == "R":
            label = row["dataset"]+" "+regime_labels[row["regime"]]
            tables.record("S24",label,columns(row,("square_pct","full_pct","gain_pp","point_low_pp","point_high_pp")))
    names = {"R_native_gain":"native CNN gain","T_R_minus_H":"CNN minus colour–texture, shift",
             "R_minus_D_fixed":"CNN minus DINOv2, native","R_minus_D_selected":"CNN minus DINOv2, selected"}
    for row in csv_rows(results/"contrasts.csv"):
        tables.record("S25",row["dataset"]+" "+names[row["contrast"]],
                     columns(row,("estimate_pp","adjusted_low_pp","adjusted_high_pp")))
    max_diag_error = 0.
    for ds in ("DTD","KTH","Lytro"):
        for row in csv_rows(results/(ds+"_settings.csv")):
            label = row["fold"].replace("_"," ") if ds!="Lytro" else "FMD to Lytro"
            tables.record("S26",label,columns(row,("lambda_S","lambda_Q")))
            max_diag_error = max(max_diag_error,*columns(row,("unit_block_diagonal_S_error","unit_block_diagonal_Q_error")))
            np.testing.assert_allclose(float(row["matched_factor"]),9/11,atol=2e-14,rtol=0)
    if max_diag_error > 2e-14 or tables.counts!={"S24":45,"S25":36,"S26":30}:
        raise ValueError("Check the training diagonals or the CNN table coverage")
    bootstrap = read_json(results / "resnet_bootstrap.json")
    with np.load(results / "DTD_decisions.npz", allow_pickle=False) as cases:
        sq, full = [cases[k] == cases["labels"] for k in ("S", "Q")]
        corrected = int(np.sum(full & ~sq))
        introduced = int(np.sum(sq & ~full))
    folder = root / "studies/resnet50"
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    from check_pairing import compare
    return {"source_target_draw_pairing": compare(root, folder),
            "calculated_values_by_table": tables.counts,
            "calculated_summary_rows": tables.rows,
            "max_absolute_outer_diagonal_error": max_diag_error,
            "source_target_intervals_pp": bootstrap["source_target_percentiles_pp"],
            "DTD_corrected": corrected, "DTD_new_errors": introduced}


def check_matched_diagnostics(root,folder):
    sys.path.insert(0,str(root.resolve()))
    t=SummaryRows();results=folder/'results'
    texts={'DTD':'DTD','KTH':'KTH-TIPS2-b','Lytro':'FMD to Lytro','Garments':'Catalogue motifs'}
    names={'H':'Colour–texture','D':'DINOv2 ViT-S/14','R':'Supervised ResNet-50','DR':'DINO ResNet-50','DV':'DINO ViT-S/16'}
    regimes={'fixed':'Native','matched':'Matched','selected':'Selected'}
    c={(r['dataset'],r['contrast']):r for r in csv_rows(results/'contrasts.csv')}
    old={r['dataset']:r for r in csv_rows(folder/'retrospective_results/contrast.csv')}
    for ds,label in texts.items():
        fields=('estimate_pp','low_pp','high_pp')
        x=columns(c[ds,'T'],fields) if ds=='Garments' else columns(old[ds],('shift_pp','adjusted_low_pp','adjusted_high_pp'))
        t.record('3',label,x+columns(c[ds,'A'],fields)+columns(c[ds,'B'],fields))
    counts=csv_rows(folder/'audit/counts.csv')
    for r in counts:
        t.record('S27',r['motif'].capitalize(),columns(r,('fit_products','fit_photos','validation_products','validation_photos','test_products','test_photos')))
    for r in csv_rows(results/'gains.csv'):
        model=names[r['representation']];mode=regimes[r['regime']]
        if r['dataset']=='Garments':table='S28';label=model+' '+mode
        else:table='S30';label=texts[r['dataset']]+' '+model+' '+mode
        t.record(table,label,columns(r,('square_pct','full_pct','gain_pp','point_low_pp','point_high_pp')))
    for r in csv_rows(results/'garment_settings.csv'):
        t.record('S29',names[r['representation']],columns(r,('lambda_S','lambda_Q')))
        np.testing.assert_allclose(float(r['matched_factor']),9/11,atol=1e-13,rtol=0)
    for ds in ('DTD','KTH','Lytro'):
        rr=csv_rows(results/'dino_r50'/(ds+'_settings.csv'));vv=csv_rows(results/'dino_vits16'/(ds+'_settings.csv'))
        for r,v in zip(rr,vv,strict=True):
            lab=r['fold'].replace('_',' ') if ds!='Lytro' else 'FMD to Lytro'
            t.record('S31',lab,columns(r,('lambda_S','lambda_Q'))+columns(v,('lambda_S','lambda_Q')))
    per_class={(r['motif'],r['representation'],r['regime']):r for r in csv_rows(results/'garment_classes.csv')}
    shifts=[]
    for motif in sorted({k[0] for k in per_class}):
        fixed,selected=[float(per_class[motif,'R',r]['gain_pp'])-float(per_class[motif,'D',r]['gain_pp']) for r in ('fixed','selected')]
        t.record('S32',motif.capitalize(),columns(per_class[motif,'R','fixed'],('products','views'))+[fixed,selected,selected-fixed]);shifts.append(selected-fixed)
    np.testing.assert_allclose(np.mean(shifts),float(c['Garments','T']['estimate_pp']),atol=1e-11,rtol=0)
    expected={'3':36,'S27':36,'S28':75,'S29':10,'S30':90,'S31':60,'S32':30}
    if t.counts!=expected:raise ValueError('Incomplete numerical summary coverage: '+str(t.counts))
    return {"calculated_values_by_table": t.counts,
            "calculated_summary_rows": t.rows,
            "garment_class_shift_mean_pp": float(np.mean(shifts))}
