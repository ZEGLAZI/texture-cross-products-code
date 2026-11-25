"""Re-extract earlier external study descriptors from manifest-verified public images.
For KTH or TF supply an extracted image tree; for Pl@ntNet and iNaturalist
the program reconstructs missing JPEGs from the manifest URL. It never replaces
a missing observation with another. Unavailable or changed files are recorded.
"""
import argparse,csv,hashlib,io,json,sys,urllib.request
from pathlib import Path
from PIL import Image
import numpy as np

def rows(a):
    b=a.supp/'external/data'
    def read(p):
        with p.open() as f:return list(csv.DictReader(f))
    if a.task=='kth':
        return [(r['chemin'],r['chemin'],'','',r['sha_pixels']) for r in read(b/'kth_images.csv')]
    if a.task=='tf':
        src=read(b/'tf_manifest.csv');hashes={r['path']:r for r in read(a.supp/'revision/results/masks/source_hashes.csv')}
        return [(r['source_image'],r['source_image'],'',hashes[r['source_image']]['sha256_jpeg'],hashes[r['source_image']]['sha256_rgb']) for r in src]
    name,field=('plantnet_manifest.csv','occurrence_id') if a.task=='plantnet' else ('inaturalist2026_manifest.csv','observation_id')
    return [(r[field],r['file_sha256']+'.jpg',r['original_url'],r['file_sha256'],r['rgb_sha256']) for r in read(b/name)]
def run(a):
    import torch
    sys.path.insert(0,str((a.supp/'dtd/reference/extracteur').resolve()));from extracteur import make_estimator,compact_features
    from histogramme_oxford import describe
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);model=make_estimator();record=rows(a);a.out.mkdir(parents=True,exist_ok=True);a.images.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(record),96):
        dest=a.out/f'{start:05d}.npz';journal=a.out/f'{start:05d}.json'
        if dest.exists() and journal.exists() and not json.loads(journal.read_text())['failed']:continue
        ids=[];photos=[];fail=[];vectors={}
        def flush():
            if not photos:return
            v=compact_features(photos,model);v['rgb']=np.stack([describe(x)[0] for x in photos])
            for k,z in v.items():vectors.setdefault(k,[]).append(z)
            photos.clear()
        for key,rel,url,sha,pixels in record[start:start+96]:
            try:
                path=a.images/rel
                if path.is_file():data=path.read_bytes()
                elif url:
                    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Texture research (archived observation reconstruction)'}),timeout=20) as f:data=f.read()
                else:raise FileNotFoundError(rel)
                if sha and hashlib.sha256(data).hexdigest()!=sha:raise ValueError('file checksum changed')
                with Image.open(io.BytesIO(data)) as im:image=im.convert('RGB')
                digest=hashlib.sha256(np.array(image.size,dtype='<u4').tobytes()+image.tobytes()).hexdigest()
                if digest!=pixels:raise ValueError('RGB checksum changed')
                if url and not path.exists():path.write_bytes(data)
                ids.append(key);photos.append(image)
                if len(photos)==8:flush()
            except (OSError,ValueError) as exc:fail.append([str(key),str(exc)])
        flush();np.savez_compressed(dest,ids=ids,**{k:np.concatenate(v) for k,v in vectors.items()})
        journal.write_text(json.dumps({'expected':len(record[start:start+96]),'available':len(ids),'failed':fail},indent=2)+'\n')
        print(start,len(ids),len(fail),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('task',choices=['kth','tf','plantnet','nouveau_2026'])
    for n in ('supp','images','out'):p.add_argument(n,type=Path)
    p.add_argument('--threads',type=int,default=2);run(p.parse_args())
