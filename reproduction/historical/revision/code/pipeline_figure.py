"""Vector diagram of the exact computation, without empirical or generated imagery."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
def draw(out):
    plt.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans','font.size':9})
    fig,ax=plt.subplots(figsize=(7.15,3.75));ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
    def box(x,y,w,h,t,fill='#f5f7f8'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07,rounding_size=0.06',facecolor=fill,edgecolor='#74828a',lw=.9));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8.4,linespacing=1.22)
    def link(a,b):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color='#43575f',lw=.85,shrinkA=2,shrinkB=2))
    ax.text(.18,5.76,'A   Measurements of each image',fontweight='bold',va='center')
    ax.text(5.08,5.76,'B   Similarity for a query and one reference',fontweight='bold',va='center')
    for y,l,r in [(4.45,'Achromatic wavelet magnitudes','Rank dependence  •  945'),(3.12,'Signed opponent wavelets','Scatter, scale, radial index  •  504'),(1.80,'Global RGB bin counts','Square-root histogram  •  64')]:
        box(.15,y,2.20,.73,l.replace('wavelet magnitudes','wavelet\nmagnitudes').replace('opponent wavelets','opponent\nwavelets'));box(2.7,y,2.19,.73,r.replace('  •  ','\n').replace('Scatter, scale, radial index','Scatter, scale, radial index'));link((2.45,y+.365),(2.59,y+.365))
    for y,h in [(4.8,'s'),(3.48,'w'),(2.16,'c')]:
        box(5.28,y-.24,.50,.46,h,'#e6f0ef');link((4.99,y),(5.20,y))
    box(6.67,2.75,2.99,1.45,'Additive  •  weighted sum\nTwo products  •  W, S or C centre\nUniform  •  all three pairs\nQuadratic  •  pairs and squares','#e6f0ef')
    for y in [4.8,3.48,2.16]:link((5.9,y),(6.57,3.47))
    box(6.67,.76,2.99,.82,'Kernel ridge class scores\nfor all training references');link((8.17,2.60),(8.17,1.66))
    ax.text(.18,.72,'Both wavelet blocks use 21 spatial regions.\nThe RGB histogram covers the full photograph.',fontsize=8.2,color='#38484e',va='bottom')
    fig.subplots_adjust(left=.012,right=.995,top=.98,bottom=.04);fig.savefig(out,bbox_inches='tight',pad_inches=.055);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);draw(p.parse_args().out)
