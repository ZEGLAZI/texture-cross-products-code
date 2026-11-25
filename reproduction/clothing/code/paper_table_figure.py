"""Generate the manuscript table and figure from audited out-of-fold predictions."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'manuscript'
NAMES={'additive':'Additive HSMT','crosses':'HSMT, cross-products','squares':'HSMT, squares',
       'clbp':'Joint grayscale CLBP','dino':'Frozen DINOv2','dino_additive':'DINOv2 + additive HSMT',
       'dino_squares':'DINOv2 + HSMT squares'}
def main():
    summary=json.loads((ROOT/'results/summary.json').read_text())
    with (PAPER/'tables/clothing.tex').open('w') as out:
        print(r'\begin{tabular}{@{}lrr@{}}\toprule',file=out)
        print(r'Model & Mean recall (\%) & Range over partitions (\%) \\ \midrule',file=out)
        for method,label in NAMES.items():
            row=summary['scores'][method]
            print(f"{label} & {row['mean']:.2f} & {row['min']:.2f}--{row['max']:.2f} "+r'\\',file=out)
        print(r'\bottomrule\end{tabular}',file=out)
    with (ROOT/'results/by_partition.csv').open() as f:
        rows=list(csv.DictReader(f))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'pdf.fonttype':42,'axes.labelsize':10})
    fig,ax=plt.subplots(figsize=(6.65,2.65),layout='constrained')
    x=np.arange(1,len(rows)+1)
    ax.axhline(0,color='#575757',linewidth=0.9)
    ax.scatter(x-.13,[float(r['squares minus crosses']) for r in rows],
               marker='o',s=24,color='#174f72',label='HSMT squares minus cross-products',zorder=3)
    ax.scatter(x+.13,[float(r['squares minus clbp']) for r in rows],
               facecolors='none',edgecolors='#aa4d1c',marker='^',s=27,
               linewidth=1.05,label='HSMT squares minus CLBP',zorder=3)
    ax.set(xlabel='Stratified five-fold partition',ylabel='Recall difference (pp)',
           xlim=(0,31),ylim=(-10,16),xticks=[1,5,10,15,20,25,30])
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#d8dde1',linewidth=.6)
    ax.set_axisbelow(True)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncols=2,frameon=False,
              handletextpad=.25,columnspacing=1.4,fontsize=8.6)
    fig.savefig(PAPER/'figures/clothing_partitions.pdf')
    plt.close(fig)
if __name__=='__main__':
    main()
