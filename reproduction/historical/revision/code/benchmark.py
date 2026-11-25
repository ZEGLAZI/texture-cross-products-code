"""CPU wall times for native image extraction, a fixed Cholesky fit and batched test scores."""
import argparse,json,sys,time,resource,platform
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.linalg import cho_factor,cho_solve
from threadpoolctl import threadpool_limits
def main(a):
    import torch
    sys.path.insert(0,str((a.supp/'dtd').resolve()));from extraire import setup_hsmt
    from predire import load_blocks,compact,partitions
    from nested import kernel
    torch.set_num_threads(2);torch.set_num_interop_threads(1);extract=setup_hsmt();ids=dict(np.load(a.supp/'dtd/results/identites.npz'));y=ids['y']
    ix=np.concatenate([np.flatnonzero(y==j)[:4] for j in range(47)]);images=[]
    for key in ids['ids'][ix]:
        with Image.open(a.images/key) as im:images.append(im.convert('RGB'))
    extract(images[:8]);times=[]
    for i in range(0,len(images),8):
        t=time.perf_counter();extract(images[i:i+8]);times.append(time.perf_counter()-t)
    rss_extract=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    raw=load_blocks(a.cache/'hsmt',ids['ids'],['F7','F8','rgb']);_,tr,te=next(partitions(ids,ids['groupes']));n=len(tr)
    with threadpool_limits(2):
        f,_=compact(raw,training=tr);f*=np.sqrt(2);blocks=[f[:,:945],f[:,945:],raw['rgb']]
        trblocks=[v[tr] for v in blocks];queries=[v[te] for v in blocks]
        fit_s=[];pred_s=[]
        for j in range(4):
            t=time.perf_counter();gram=kernel([x@x.T for x in trblocks],[.25,.25,.5],'W');gram.flat[::n+1]+=1;lab=np.eye(47)[y[tr]];prior=lab.mean(0)
            alpha=cho_solve(cho_factor(gram,lower=True,check_finite=False),lab-prior,check_finite=False);fit_s.append(time.perf_counter()-t);del gram
            t=time.perf_counter();q=kernel([x@v.T for x,v in zip(queries,trblocks)],[.25,.25,.5],'W');pred=(q@alpha+prior).argmax(1)
            pred_s.append(time.perf_counter()-t);del q
        np.testing.assert_array_equal(pred,np.load(a.supp/'dtd/results/decisions.npz')['HSMT'][0])
    cpu=[v.split(':',1)[1].strip() for v in Path('/proc/cpuinfo').read_text().splitlines() if v.startswith('model name')]
    out={'CPU':cpu[0] if cpu else platform.machine(),'threads':2,'torch':str(torch.__version__),'extraction_images':len(images),'batch':8,'decode_excluded':True,'seconds_per_image':sum(times)/len(images),'seconds_per_batch':times,'train_n':n,'test_n':len(te),'first_repetition_warmup':True,'fit_s_median':float(np.median(fit_s[1:])),'batch_test_s_median':float(np.median(pred_s[1:])),'extract_process_peak_MiB':rss_extract/1024,'whole_process_peak_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'raw_gram_MiB':8*n*n/(1024**2)}
    a.out.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('supp','cache','images','out'):p.add_argument(n,type=Path)
    main(p.parse_args())
