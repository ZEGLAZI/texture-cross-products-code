"""Recreate all manuscript tables with measurements and all four vector figures."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

TEAL="#146764"; BLUE="#325888"; OCHRE="#986029"; INK="#1d2b35"; MUTED="#5a6770"; GRID="#dbe1e3"
mpl.rcParams.update({"font.family":"DejaVu Sans","mathtext.fontset":"dejavusans","font.size":9.2,
    "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#9aa6ac",
    "axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"axes.titlesize":11,"axes.titleweight":"semibold",
    "axes.titlelocation":"left","pdf.fonttype":42,"ps.fonttype":42,"text.color":INK,"axes.unicode_minus":True})
def readjson(path):return json.loads(path.read_text())
def csvrows(path):
    with path.open() as f:return list(csv.DictReader(f))
def f(x):return f"{float(x):.2f}"
def band(x,lo,hi):
    return f"${x:+.2f}\\;[{lo:.2f}, {hi:.2f}]$"
def write_rows(path, rows):
    lines=[v if isinstance(v,str) else " & ".join(v)+r" \\" for v in rows]
    layouts = {
        "dtd": ("lrr",r"Image comparison & Recall (\%) & $\KW$ minus comparison (pp)"),
        "baselines": ("lcr",r"Representation or kernel & Coordinates & Mean recall (\%)"),
        "flowers": ("lrrrr",r"Collection & Images & Additive (\%) & $\KW$ (\%) & Difference [interval]"),
        "uniform": ("lrrrr",r"Test set & Additive (\%) & $K_U$ (\%) & $\KW$ (\%) & Difference [interval]"),
    }
    layouts["orders"]=("lccccccc",r"Centre & $sw$ & $wc$ & $sc$ & DTD & KTH & Pl@ntNet & iNat. 2026")
    cols,header=layouts[path.stem]
    wrapper = [r"\begin{tabular}{@{}"+cols+r"@{}}\toprule", header+r" \\ \midrule", *lines, r"\bottomrule\end{tabular}"]
    path.write_text("\n".join(wrapper)+"\n")
def save(fig,path):
    fig.savefig(path,bbox_inches="tight",pad_inches=.045,metadata={"CreationDate":None,"ModDate":None,"Creator":"Matplotlib"})
    plt.close(fig)
def run(src,out):
    figs=out/"figures";tabs=out/"tables";figs.mkdir(exist_ok=True,parents=True);tabs.mkdir(exist_ok=True,parents=True)
    data=readjson(src/"dtd/results/bilan.json");method=data["methodes"]; comp=data["contrastes"]
    dense=readjson(src/"extra/dense.json");oc=readjson(src/"extra/oclbp.json");others=readjson(src/"external/results/external.json")
    piv=readjson(src/"external/results/pivots_archives.json"); flower=csvrows(src/"external/results/fleurs_par_lot.csv")
    cases=readjson(src/"extra/cases.json");score=lambda key:method[key]["rappel"]
    rows=[[r"Luminance structure, $S$",f(dense["scores"]["structure"]),r"---"],[r"Opponent wavelets, $W$",f(dense["scores"]["wavelet"]),r"---"],[r"RGB palette, $C$",f(dense["scores"]["colour"]),r"---"],r"\midrule"]
    for nm,lab in [("Additif_historique",r"Additive $K_{A,2}$"),("Additif_fixe",r"Additive $K_A^{*}$"),("Additif_diag_egal",r"Additive, matched diagonal"),("Pivot_F7",r"Structure centre $K_S^{*}$"),("Pivot_RGB",r"Colour centre $K_C^{*}$")]:
        v=comp["HSMT - "+nm];rows.append([lab,f(score(nm)),band(v["ecart"],*v["ic"])])
    rows+=[r"\midrule",[r"Opponent-wavelet centre $K_W$",f(score("HSMT")),r"Reference"]]
    write_rows(tabs/"dtd.tex",rows)
    rows=[]
    for name,dim,val in [("OCLBP","432",oc["score"]),("CLBP","1,352",score("CLBP")),(r"Proposed $K_W$","1,513",score("HSMT")),("DeCAF","4,096",score("DeCAF")),("DINO","384",score("DINOv2")),(r"DINO and $K_A$","384 + 1,513",score("DINO_Additif")),(r"DINO and diagonal-matched $K_A$","384 + 1,513",score("DINO_Additif_diag_egal")),(r"DINO and $K_W$","384 + 1,513",score("DINO_HSMT"))]:
        if name.startswith("DINO and") and len(rows)==5:rows.append(r"\midrule")
        rows.append([name,dim,f(val)])
    write_rows(tabs/"baselines.tex",rows)
    rows=[]
    for i,r in enumerate(flower):
        if i==2:rows.append(r"\midrule")
        rows.append([r["lot"].replace("–","--"),f'{int(r["n"]):,}',f(r["reference"]),f(r["score"]),band(float(r["gain"]),float(r["ic_simultane_quatre_bas"]),float(r["ic_simultane_quatre_haut"]))])
    write_rows(tabs/"flowers.tex",rows)
    rows=[["DTD",f(score("Additif_fixe")),f(dense["scores"]["all_pairs"]),f(score("HSMT")),band(dense["delta"],*dense["ci"])]]
    for name,tag in [("KTH-TIPS2-b","KTH"),("Pl@ntNet","plantnet"),("iNaturalist 2026","nouveau_2026")]:
        x=others[tag]; rows.append([name,f(x["scores"]["Add"]),f(x["scores"]["dense"]),f(x["scores"]["HSMT"]),band(x["delta"],*x["ci"])])
    write_rows(tabs/"uniform.tex",rows)
    alt=readjson(src/"extra/reversals.json")
    ordrows=[]
    for label,wts,key,rev in [("Wavelets",("1/4","1/2","0"),"HSMT","W"),("Structure",("1/4","0","1/2"),"Pivot_F7","S"),("Colour",("0","1/2","1/4"),"Pivot_RGB","C")]:
        if ordrows:ordrows.append(r"\addlinespace")
        a,b,c=wts
        swapped=lambda x:{"1/4":"1/2","1/2":"1/4","0":"0"}[x]
        ordrows.append([label,*wts,f(score(key)),*[f(others[tag]["scores"][key]) for tag in ("KTH","plantnet","nouveau_2026")]])
        ordrows.append([label,*[swapped(v) for v in wts],*[f(alt[tag][rev+"_reversed"]) for tag in ("DTD","KTH","plantnet","nouveau_2026")]])
    write_rows(tabs/"orders.tex",ordrows)
    fig,axes=plt.subplots(1,3,figsize=(7.9,2.34))
    positions={"S":(.17,.70),"C":(.83,.70),"W":(.50,.20)}
    for ax,head,edge in zip(axes,["(a)  Opponent wavelets","(b)  Structure","(c)  Colour"],[[("S","W","¼"),("W","C","½")],[("S","W","¼"),("S","C","½")],[("S","C","¼"),("W","C","½")]]):
        ax.set(xlim=(-.02,1.02),ylim=(.00,1.00),aspect="equal");ax.axis("off")
        ax.text(.50,.98,head,ha="center",va="top",fontsize=10.1,fontweight="semibold")
        for x,y,frac in edge:
            x1,y1=positions[x];x2,y2=positions[y]
            ax.plot([x1,x2],[y1,y2],linewidth=2.3 if frac=="½" else 1.6,color=INK,zorder=0)
            ax.text((x1+x2)/2,(y1+y2)/2,frac,ha="center",va="center",fontsize=10.4,bbox=dict(facecolor="white",edgecolor="none",pad=1.5))
        for label,xy in positions.items():
            col={"S":BLUE,"W":TEAL,"C":OCHRE}[label]
            ax.add_patch(Circle(xy,.079,facecolor="white",edgecolor=col,lw=1.8))
            ax.text(*xy,label,color=col,ha="center",va="center",weight="bold",fontsize=11.5)
    fig.subplots_adjust(wspace=.15);save(fig,figs/"arrangements.pdf")
    by={}
    for r in csvrows(src/"dtd/results/scores_classes.csv"):
        by.setdefault(r["classe"],{})[r["methode"]]=float(r["rappel"])
    diff=sorted(((a["HSMT"]-a["Additif_fixe"],i)for i,a in by.items()),key=lambda x:(-x[0],x[1]))
    fig,axs=plt.subplots(1,2,figsize=(7.9,6.30))
    for ix,(ax,items) in enumerate(zip(axs,(diff[:24],diff[24:]))):
        for i,(v,n) in enumerate(items):
            co=TEAL if v>0 else OCHRE
            ax.plot([0,v],[i,i],color=co,lw=1.4);ax.scatter([v],[i],c=co,s=20,zorder=3)
        ax.axvline(0,color=INK,lw=.9);ax.set(yticks=np.arange(len(items)),yticklabels=[i[1] for i in items],xlim=(-10.3,16.7),ylim=(23.6,-.65),xticks=[-10,-5,0,5,10,15])
        ax.set_axisbelow(True);ax.grid(axis="x",color=GRID,lw=.55);ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0,pad=3,labelsize=9.3)
        ax.set_xlabel("Recall difference (pp)")
    fig.subplots_adjust(left=.15,right=.99,bottom=.095,top=.99,wspace=.61);save(fig,figs/"attributes.pdf")
    fig,ax=plt.subplots(figsize=(7.65,3.55))
    store=[]
    for i,(key,title) in enumerate([("dtd","DTD"),("plantnet","Pl@ntNet"),("nouveau_2026","iNaturalist 2026"),("KTH","KTH-TIPS2-b")]):
        for h,(name,col,marker) in enumerate([("Pivot_F7",BLUE,"o"),("Pivot_RGB",OCHRE,"s")]):
            if key=="dtd":
                q=comp["HSMT - "+name];val=q["ecart"];lo,hi=q["ic"]
            else:q=piv[key]["vs"][name];val=q["ecart"];lo,hi=q["ic_simultane_six"]
            y=3-i+.13-.26*h
            ax.errorbar(val,y,xerr=[[val-lo],[hi-val]],fmt=marker,ms=5.5,color=col,ecolor=col,capsize=3,lw=1.35)
            ax.text(7.68,y,f"{val:+.2f}  [{lo:.2f}, {hi:.2f}]".replace("-","−"),va="center",fontsize=8.5,color=col)
        store.append(title)
    ax.axvline(0,color=INK,lw=1);ax.set_yticks([3,2,1,0],labels=store);ax.tick_params(axis="y",length=0);ax.set(xlim=(-3.4,11.5),ylim=(-.55,3.63),xticks=[-2,0,2,4,6])
    ax.spines["left"].set_visible(False);ax.spines["bottom"].set_bounds(-3,7.2);ax.set_xlabel("Difference in macro recall (pp)",x=.38)
    ax.text(7.68,3.48,"Estimate [interval]",fontsize=8.6,fontweight="semibold")
    ax.legend(handles=[Line2D([],[],color=BLUE,marker="o",label="W minus S centre"),Line2D([],[],color=OCHRE,marker="s",label="W minus C centre")],frameon=False,loc="upper left",bbox_to_anchor=(.00,-.19),ncol=2,handlelength=1.6)
    fig.subplots_adjust(left=.17,right=.99,top=.98,bottom=.27);save(fig,figs/"centres.pdf")
    fig,axes=plt.subplots(1,2,figsize=(7.9,3.05),sharex=True)
    for i,(ax,z) in enumerate(zip(axes,cases)):
        for j,v in enumerate(z["terms"][:5]):
            ax.barh(j,v,color=TEAL if v>0 else OCHRE,height=.48)
        ax.plot(z["margin"],5,marker="D",ms=6.5,color=INK);ax.axvline(0,color=INK,lw=.8)
        ax.axhline(4.54,color=GRID,lw=1)
        ax.set(xlim=(-.071,.077),ylim=(5.6,-.7),yticks=range(6),yticklabels=["Structure","Wavelets","Palette","S × W","W × C","Total"],xticks=[-.05,0,.05],xlabel="Class-score contribution")
        ax.set_title(("Recovered" if i==0 else "Lost")+" ("+z["label"]+" vs. "+z["other"]+")",fontsize=9.8)
        ax.tick_params(axis="y",length=0);ax.spines["left"].set_visible(False);ax.spines["bottom"].set_bounds(-.06,.07)
    fig.subplots_adjust(left=.12,right=.99,bottom=.22,top=.88,wspace=.43);save(fig,figs/"margins.pdf")
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("evidence",type=Path);parser.add_argument("output",type=Path)
    args=parser.parse_args();run(args.evidence,args.output)
