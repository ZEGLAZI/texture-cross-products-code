"""Uncertainty of two decisive contrasts at matched search budget."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
def draw(r,out):
    selected=json.loads((r/'selected.json').read_text());new=json.loads((r/'cohort/scores.json').read_text())['selected']
    cases=[('DTD',selected['DTD']),('KTH-TIPS2-b',selected['KTH']),('Pl@ntNet',selected['Pl@ntNet']),('iNaturalist 2026',selected['iNat2026']),('iNaturalist 2015–17',new)]
    plt.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans','font.size':9})
    fig,ax=plt.subplots(1,2,figsize=(7.25,2.65),sharey=True)
    for a,contrast,title,color in zip(ax,('W-Add','W-Q'),('Two products minus additive','Two products minus full quadratic'),('#147575','#a75f2b')):
        yy=np.arange(len(cases))[::-1];vals=np.array([q[contrast] for _,q in cases]);a.axvline(0,color='#687078',lw=.9)
        a.errorbar(vals[:,0],yy,xerr=np.stack([vals[:,0]-vals[:,1],vals[:,2]-vals[:,0]]),fmt='o',color=color,capsize=2.8,ms=4)
        a.set_title(title,fontsize=9.4,fontweight='bold',pad=8);a.set_ylim(-.55,4.55);a.set_xlabel('Difference in macro recall (pp)',fontsize=8.0);a.spines[['top','right','left']].set_visible(False);a.tick_params(axis='y',length=0);a.grid(axis='x',color='#e3e8e8',lw=.5);a.set_axisbelow(True)
    ax[0].set(yticks=np.arange(5)[::-1],yticklabels=[n for n,_ in cases],xlim=(-3,9),xticks=[-2,0,2,4,6,8]);ax[1].set(xlim=(-12,4),xticks=[-12,-8,-4,0,4]);fig.subplots_adjust(left=.20,right=.98,bottom=.23,top=.83,wspace=.22);fig.savefig(out,bbox_inches='tight',pad_inches=.035);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('out',type=Path);a=p.parse_args();draw(a.results,a.out)
