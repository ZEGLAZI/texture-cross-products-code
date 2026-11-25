"""Fit all five garment representations, retaining the held-out unit and the validation grid."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve
from threadpoolctl import threadpool_limits
from garments import write_csv

MODELS=('H','D','R','DR','DV')
VARIANTS=('S','Q','matched_Q','selected_S','selected_Q')
def read_csv(path):
    with path.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))

def product_weights(y,products):
    groups,inverse,counts=np.unique(products,return_inverse=True,return_counts=True)
    gclass=np.zeros(len(groups),int)
    gclass[inverse]=y
    np.testing.assert_array_equal(gclass[inverse],y)
    return 1./(len(np.unique(y))*np.bincount(gclass)[y]*counts[inverse])

def predictions(k,trainlabels,classes,p):
    n=len(trainlabels);targets=np.eye(classes)[trainlabels];prior=targets.mean(0)
    g=k[:n].copy(order='F');g.flat[::n+1]+=p
    coef=cho_solve(cho_factor(g,lower=True,overwrite_a=True,check_finite=False),targets-prior,check_finite=False)
    return (k[n:]@coef+prior).argmax(1)

def run(a):
    historic=a.package/'studies/CVIU_Experience_couleur/code'
    sys.path.insert(0,str(historic.resolve()))
    from fit import standardized,kernels,gram
    sys.path.insert(0,str((a.package/'studies/resnet50').resolve()))
    from cnn_models import PENALTIES,choose,classify
    rows=read_csv(a.garments/'manifest.csv')
    y_names=np.array([r['motif'] for r in rows]);class_names,y=np.unique(y_names,return_inverse=True)
    products=np.array([r['product'] for r in rows]);role=np.array([r['role'] for r in rows])
    fit=np.nonzero(role=='fit')[0];val=np.nonzero(role=='validation')[0]
    train=np.nonzero(role!='test')[0];test=np.nonzero(role=='test')[0]
    weights=product_weights(y[val],products[val]);np.testing.assert_allclose(weights.sum(),1.)
    for x in (fit,val,test):np.testing.assert_array_equal(np.unique(y[x]),np.arange(len(class_names)))
    if set(products[fit]) & set(products[val]) or set(products[test]) & set(products[train]):raise ValueError('Product crosses split')
    full={};grids=[];settings=[];development={}
    with np.load(a.features) as features:
        np.testing.assert_array_equal(features['ids'],[r['photo'] for r in rows])
        np.testing.assert_array_equal(features['hash224'],[r['rgb224_sha256'] for r in rows])
        for name in MODELS:
            if name=='H': blocks=[features[k] for k in ('F7','F8','rgb')]
            else:blocks=[features[name+str(i)] for i in range(3)]
            def build(ix,ev):
                grams=[]
                for j,v in enumerate(blocks):
                    if name=='H' and j==2:values=v
                    else:values=standardized(v,ix,(9,3)[j] if name=='H' else None)
                    grams.append(gram(values,ix,ev))
                pair=kernels(*grams,both=True)
                if a.independent:
                    a1,b1,c1=grams
                    reference=(8*(a1+b1+c1)+a1*a1+b1*b1+c1*c1)/39
                    np.testing.assert_allclose(pair[0],reference,atol=1e-14,rtol=0)
                    np.testing.assert_allclose(pair[1],reference+(2/39)*(a1*b1+a1*c1+b1*c1),atol=1e-14,rtol=0)
                return pair
            pair=build(fit,val);best={}
            for model,k in zip(('S','Q'),pair,strict=True):
                n=len(fit);t=np.eye(len(class_names))[y[fit]];mu=t.mean(0)
                eig,u=eigh(k[:n],driver='evd',check_finite=False)
                left=k[n:]@u;right=u.T@(t-mu)
                scores=[];candidate_labels={}
                for p in PENALTIES:
                    predicted=(left@(right/(eig[:,None]+p))+mu).argmax(1)
                    score=float(weights@(predicted==y[val]));scores.append(score)
                    candidate_labels[p]=predicted
                    if a.independent:
                        np.testing.assert_array_equal(predicted,predictions(k,y[fit],len(class_names),p))
                    grids.append(dict(representation=name,kernel=model,penalty=p,product_recall_pct=score*100))
                best[model]=choose(np.array(scores))
                if name in ('R','D'):
                    for tag,ridge in [('fixed',1.),('selected',best[model])]:
                        development[name,model,tag]=(candidate_labels[ridge]==y[val]).astype(int)
            del k,pair
            s,q=build(train,test);n=len(train);ds=float(s[:n].diagonal().mean());dq=float(q[:n].diagonal().mean())
            factor=ds/dq
            settings.append(dict(representation=name,lambda_S=best['S'],lambda_Q=best['Q'],
                                 mean_diag_S=ds,mean_diag_Q=dq,matched_factor=factor))
            alternate={}
            if a.independent:
                target=np.eye(len(class_names))[y[train]];prior=target.mean(0)
                for kernel,settings_to_test in [(s,{'S':1.,'selected_S':best['S']}),
                            (q,{'Q':1.,'matched_Q':1./factor,'selected_Q':best['Q']})]:
                    eigen,vectors=eigh(kernel[:n],driver='evd',check_finite=False)
                    left=kernel[n:]@vectors;right=vectors.T@(target-prior)
                    for variant,penalty in settings_to_test.items():
                        alternate[variant]=(left@(right/(eigen[:,None]+penalty))+prior).argmax(1)
            for tag,k,p in [('S',s,1.),('Q',q,1.),('matched_Q',q,1./factor),('selected_S',s,best['S']),('selected_Q',q,best['Q'])]:
                predicted=classify(k,y[train],len(class_names),p);full[name+'_'+tag]=predicted
                if a.independent:np.testing.assert_array_equal(predicted,alternate[tag])
            del s,q,k
            print('Fitted garment',name,flush=True)
    a.out.mkdir(exist_ok=True,parents=True);write_csv(a.out/'garment_validation.csv',grids);write_csv(a.out/'garment_settings.csv',settings)
    if a.private_decisions:
        np.savez_compressed(a.private_decisions,classes=class_names,labels=y[test],products=products[test],
                            ids=np.array([r['photo'] for r in rows])[test],**full)
    # Precision diagnostic based on the development cases and actual test group sizes.
    def shift(model):
        return (development[model,'Q','selected']-development[model,'S','selected']
                -development[model,'Q','fixed']+development[model,'S','fixed'])
    val_contrast=shift('R')-shift('D');precision=[]
    for rho in (0.,.5,1.):
        var=0.
        for c in np.unique(y):
            x=val_contrast[y[val]==c];_,m=np.unique(products[test][y[test]==c],return_counts=True)
            var+=np.var(x,ddof=1)*np.sum((1+(m-1)*rho)/m)/(len(m)**2)
        precision.append(dict(within_product_correlation=rho,approx_halfwidth_pp=196*np.sqrt(var)/len(class_names)))
    write_csv(a.out/'garment_development_precision.csv',precision)
    evidence=[]
    for i,prod in enumerate(np.unique(products[test])):
        take=products[test]==prod;label=np.unique(y[test][take])
        if len(label)!=1:raise ValueError('Conflicting retained product')
        entry={'test_product':i+1,'motif':class_names[label[0]],'views':int(take.sum())}
        for key,pred in full.items():entry[key]=int(np.count_nonzero(pred[take]==label[0]))
        evidence.append(entry)
    write_csv(a.out/'garment_test_evidence.csv',evidence)
    if a.independent:print('All 60 validation and 25 test decision vectors agreed across Cholesky and eigen fits.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('package','garments','features','out'):p.add_argument(k,type=Path)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--independent',action='store_true')
    p.add_argument('--private-decisions',type=Path)
    a=p.parse_args()
    with threadpool_limits(a.threads):run(a)
