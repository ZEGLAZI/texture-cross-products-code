"""Offline reproduction of the nested recognition and radial comparisons."""
import argparse,subprocess,sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['aggregate','refit','radial','radial-from-pixels','simulate','figures','details'])
    p.add_argument('--output',type=Path,required=True,help='writable output directory')
    p.add_argument('--corpus',choices=['DTD','Flower','KTH','Textile','FMD','all'],default='all')
    p.add_argument('--pixels',type=Path,help='folder with the original DTD images/ and index.npz')
    p.add_argument('--decisions',type=Path,help='directory of refitted fold NPZs; defaults to supplied decisions')
    a=p.parse_args();d=Path(__file__).resolve().parent;r=d.parent;base=r.parent;old=base/'historical';cloth=base/'clothing';a.output.mkdir(exist_ok=True,parents=True)
    def run(name,*arg):subprocess.run([sys.executable,str(d/name),*map(str,arg)],check=True)
    if a.mode=='aggregate':
        run('summary.py',old,old/'revision/results',cloth/'data',a.decisions or r/'results/kernels',a.output)
    elif a.mode=='refit':
        for task in (['DTD','Flower','KTH','Textile','FMD'] if a.corpus=='all' else [a.corpus]):
            run('assess.py',task,'--supp',old,'--cache',r/'dtd_cache','--cohort',old/'revision/results/cohort','--textile',cloth/'data','--input',r/'extra/fmd.npz','--out',a.output)
    elif a.mode=='radial':
        run('radial_assess.py',old,r/'dtd_cache',r/'radial_cache',r/'original_numeric',a.output)
    elif a.mode=='radial-from-pixels':
        if not a.pixels:p.error('--pixels is required for radial-from-pixels')
        run('radial_extract.py',a.pixels,a.output)
    elif a.mode=='simulate':run('calibration.py',a.output/'synthetic.csv')
    elif a.mode=='details':run('provenance.py',old,r/'dtd_cache',old/'revision/results/selection',old/'revision/results/cohort',cloth/'data',r/'extra',a.output)
    else:
        (a.output/'tables').mkdir(exist_ok=True);run('publication.py',cloth/'results/by_partition.csv',a.output,'--primary',r/'results/aggregate/primary.csv','--redistribution',r/'results/aggregate/redistributed_rbf.csv')
if __name__=='__main__':main()
