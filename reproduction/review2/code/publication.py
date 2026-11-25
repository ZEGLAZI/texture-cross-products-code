"""Make the principal figures and fixed-coefficient table from archived decisions."""
from pathlib import Path
import argparse,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

BLUE='#275277';GREEN='#006C64';INK='#192a33';GREY='#acb7be'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
def pipeline(path):
    fig = plt.figure(figsize=(4.48, 3.47))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(.02, .97, "Query and reference photographs", size=11,
            weight="bold", va="top")
    ax.text(.02, .865, "Image measurement", size=8.8, color=BLUE)
    ax.text(.96, .865, "Pair similarity", size=8.8, color=BLUE, ha="right")
    blocks = [
        ("Achromatic structure", "945", "$s$"),
        ("Signed opponent wavelets", "504", "$w$"),
        ("RGB histogram", "64", "$c$"),
    ]
    for row, (title, coordinates, similarity) in enumerate(blocks):
        y = .78 - row * .095
        ax.plot([.02, .96], [y-.04, y-.04], color="#d5dade", lw=.6)
        ax.text(.02, y, title, size=9.6, va="center")
        ax.text(.71, y, coordinates + " values", size=8.9, ha="right", va="center")
        ax.text(.91, y, similarity, size=11, ha="right", va="center")
    ax.annotate("", xy=(.50, .465), xytext=(.50, .542),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=.9))
    for x, heading, line1, line2 in [
        (.02, "Fixed coefficients", "Add squares or products", "at their original weights."),
        (.51, "Redistributed weight", "Allocate all quadratic", "weight to one family."),
    ]:
        ax.add_patch(FancyBboxPatch((x, .238), .452, .217,
                     boxstyle="round,pad=0.004,rounding_size=0.008",
                     facecolor="#f3f6f7", edgecolor="#a3b1b9", lw=.7))
        ax.text(x+.021, .409, heading, size=9.2, weight="bold", va="center")
        ax.text(x+.021, .337, line1, size=8.75, va="center")
        ax.text(x+.021, .282, line2, size=8.75, va="center")
    ax.annotate("", xy=(.50, .137), xytext=(.50, .212),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=.9))
    ax.text(.50, .074, "Kernel ridge fit and query class scores", size=9.7,
            weight="bold", ha="center", va="center")
    fig.savefig(path)
    plt.close(fig)

def clothing(rows,path):
    fig,ax=plt.subplots(figsize=(4.48,2.74));fig.subplots_adjust(left=.15,right=.98,bottom=.20,top=.79)
    n=np.arange(1,len(rows)+1);one=np.array([float(x['squares minus crosses']) for x in rows]);two=np.array([float(x['squares minus clbp']) for x in rows])
    ax.axhline(0,c=GREY,lw=1);ax.scatter(n,one,s=20,c=GREEN,label='Squares minus cross-products',marker='o')
    ax.scatter(n,two,s=19,edgecolor=BLUE,c='white',label='Squares minus CLBP',marker='^',linewidth=.9)
    ax.legend(loc='lower left',bbox_to_anchor=(-.09,1.0),frameon=False,ncol=1,fontsize=9.0,handletextpad=.4,labelspacing=.1)
    ax.set(xlabel='Repeated five-fold partition',ylabel='Difference (pp)',xlim=(0,31),xticks=[1,5,10,15,20,25,30]);ax.yaxis.grid(True,c='#e1e7e9',linewidth=.5);ax.set_axisbelow(True)
    fig.savefig(path);plt.close(fig)
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def table(primary,dest):
    names={'DTD':'DTD','Clothing':'Clothing','KTH':'KTH','Pl@ntNet':r'\ptnet{}','iNat 2026':'iNat 2026','iNat 2015-17':'iNat 2015--17','FMD':'FMD'}
    out=[r'\begin{tabular}{@{}lrrr@{}}\toprule',r'Test set & $K_Q$ (\%) & Add cross-products & Add squares \\ \midrule']
    def fmt(x):
        v=float(x)
        if abs(v)<.005:return '0.00'
        return f'{v:+.2f}'.replace('-','{-}').replace('+','{+}')
    for name in names:
        b=next(x for x in primary if x['corpus']==name)
        def col(k):return '$'+fmt(b[k])+r'\;['+fmt(b[k+'_lo'])+', '+fmt(b[k+'_hi'])+']$'
        out.append(f"{names[name]} & {float(b['Q']):.2f} & "+col('Q_minus_S')+' & '+col('Q_minus_X')+r' \\')
    out.append(r'\bottomrule\end{tabular}')
    dest.write_text('\n'.join(out)+'\n')
def redistribution(path, destination):
    by_name = {row["corpus"]: row for row in read(path)}
    lines = [r"\begin{tabular}{@{}lrrrr@{}}\toprule",
             r"Test set & Squares & Cross-products & Full quadratic & Gaussian \\ \midrule"]
    for corpus in ("DTD", "Clothing"):
        row = by_name[corpus]
        values = [f"{float(row[key]):.2f}" for key in ("RS", "RX", "Q", "RBF")]
        lines.append(" & ".join([corpus, *values]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    destination.write_text("\n".join(lines) + "\n")

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('clothing_csv',type=Path);p.add_argument('manuscript',type=Path);p.add_argument('--primary',type=Path);p.add_argument('--redistribution',type=Path)
    a=p.parse_args();f=a.manuscript/'figures';f.mkdir(exist_ok=True,parents=True);pipeline(f/'pipeline.pdf');clothing(read(a.clothing_csv),f/'clothing_partitions.pdf')
    if a.primary:table(read(a.primary),a.manuscript/'tables/nested.tex')
    if a.redistribution:redistribution(a.redistribution,a.manuscript/'tables/redistribution.tex')
