"""Post hoc decomposition of the full quadratic DTD comparator, with equal validation budgets."""
import argparse,sys
from pathlib import Path
from threadpoolctl import threadpool_limits
import nested
def decomposition(gs,p,method):
    s,w,c=gs;a,b,e=p;q=a*s+b*w+e*c
    if method=='squares':
        z=((a*s)**2+(b*w)**2+(e*c)**2)/(a*a+b*b+e*e)
    elif method=='crosses':
        z=(a*b*s*w+a*e*s*c+b*e*w*c)/(a*b+a*e+b*e)
    else:raise ValueError(method)
    return (2*q+.75*z)/3.25
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('supp','cache','output'):p.add_argument(n,type=Path)
    p.add_argument('--threads',type=int,default=2);a=p.parse_args();a.task='DTD'
    nested.METHODS=['squares','crosses'];nested.kernel=decomposition
    with threadpool_limits(a.threads):nested.run(a)
