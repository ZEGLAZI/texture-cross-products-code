"""Oxford human trimaps. Four fixed interventions, with the unchanged TF source classifier."""
import argparse,csv,io,tarfile,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
from flowers_new import classifiers
from nested import macro,METHODS
from inference import external
from cohort.reference.signatures import signatures
def images(path,suffix):
    with tarfile.open(path) as t:return {m.name:t.extractfile(m).read() for m in t if m.isfile() and m.name.endswith(suffix)}
def run(a):
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);sys.path.insert(0,str((a.supp/'dtd/reference/extracteur').resolve()));from extracteur import make_estimator,compact_features
    from histogramme_oxford import describe
    ext=make_estimator();data=images(a.archives/'17flowers.tgz','.jpg');masks=images(a.archives/'trimaps.tgz','.png');refs=list(csv.DictReader((a.out/'source_hashes.csv').open()));ph=np.array([[int(r['phash'],16),int(r['phash_miroir'],16)] for r in refs],np.uint64).ravel();exact={r[k] for r in refs for k in ('sha256_jpeg','sha256_rgb')};rows=[];kept=[];xx=[];photo=[];images_by_view={k:[] for k in ('whole','foreground','background','outline')}
    for filename in sorted(masks):
        num=int(Path(filename).stem.split('_')[-1]);c=(num-1)//80
        if c not in [9,10,12]:continue
        key=f'jpg/image_{num:04}.jpg';raw=data[key];sig=signatures(raw);pair=np.array([int(sig[k],16) for k in ('phash','phash_miroir')],dtype=np.uint64);dist=int(min(np.bitwise_count(ph^h).min() for h in pair));nearprev=bool(kept and min(np.bitwise_count(np.array(kept).ravel()^h).min() for h in pair)<=6)
        im=Image.open(io.BytesIO(raw)).convert('RGB');trim=np.array(Image.open(io.BytesIO(masks[filename])));assert set(np.unique(trim))=={0,1,3} and im.size==trim.shape[::-1]
        excluded=dist<=6 or sig['sha256_jpeg'] in exact or sig['sha256_rgb'] in exact or nearprev
        rows.append({'image':key,'class_code':c,'min_source_distance':dist,'excluded':int(excluded),'foreground_fraction':float(np.mean(trim==1)),'background_fraction':float(np.mean(trim==3)),'mask_sha256':hashlib.sha256(masks[filename]).hexdigest(),**sig})
        if excluded:continue
        kept.append(pair);photo.append(key);xx.append(c);arr=np.array(im);images_by_view['whole'].append(im.resize((224,224),Image.Resampling.BICUBIC))
        for name,v in [('foreground',1),('background',3)]:images_by_view[name].append(Image.fromarray(np.where((trim==v)[:,:,None],arr,np.uint8(128))).resize((224,224),Image.Resampling.BICUBIC))
        images_by_view['outline'].append(Image.fromarray(np.where(trim==1,192,128).astype('uint8')).convert('RGB').resize((224,224),Image.Resampling.BICUBIC))
    yy=(np.array(xx)[:,None]==[10,12,9]).argmax(1);source=np.load(a.supp/'external/data/tf.npz');choices=json.loads(a.choices.read_text());out={};archive={'y':yy,'image':photo}
    for name,photos in images_by_view.items():
        chunks={}
        for i in range(0,len(photos),8):
            batch=photos[i:i+8];v=compact_features(batch,ext);v['rgb']=np.stack([describe(p)[0] for p in batch])
            for k,z in v.items():chunks.setdefault(k,[]).append(z)
        pred=classifiers(a.supp,source,{k:np.concatenate(z) for k,z in chunks.items()},choices)
        out[name]={k:macro(v,yy) for k,v in pred.items() if not k.endswith('_scores')};archive.update({name+'_'+k:v for k,v in pred.items()})
    out['background_minus_outline']=external((archive['background_fixed_W']==yy).astype('int8')-(archive['outline_fixed_W']==yy),yy,m=1)
    out['counts']=np.bincount(yy,minlength=3).tolist();out['archives']={n:hashlib.sha256((a.archives/n).read_bytes()).hexdigest() for n in ('trimaps.tgz','17flowers.tgz')}
    with (a.out/'verified.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    np.savez_compressed(a.out/'predictions.npz',**archive);(a.out/'scores.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('supp','archives','choices','out'):p.add_argument(k,type=Path)
    a=p.parse_args()
    with threadpool_limits(1):run(a)
