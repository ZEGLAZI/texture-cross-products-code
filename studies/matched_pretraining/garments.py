"""Reconstruct the catalogue-product experiment from the providers' three ZIP files."""
import argparse
import csv
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.ndimage import distance_transform_cdt, find_objects

TOPS = ('Tees_Tanks','Graphic_Tees','Blouses_Shirts','Shirts_Polos','Sweaters','Sweatshirts_Hoodies')
MAP = {c:(0,1) for c in TOPS}
MAP.update({c:(2,2) for c in ('Jackets_Vests','Jackets_Coats')})
MAP.update({c:(1,5) for c in ('Shorts','Denim','Pants')})
MAP.update(Skirts=(1,3),Leggings=(1,6))
MOTIFS = {0:'floral',1:'graphic',2:'striped',3:'pure colour',4:'lattice',6:'colour block'}
BICUBIC = Image.Resampling.BICUBIC

def sha(value):
    return hashlib.sha256(value).hexdigest()

def order(kind, value):
    return sha(('dfmm-cviu-v1|'+kind+'|'+value).encode())

def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf8') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)

def fingerprint(im):
    grey=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float)
    coef=dctn(grey,type=2,norm='ortho')[:8,:8].ravel()
    bits=np.packbits(coef>np.median(coef)).tobytes()
    small=np.asarray(im.resize((16,16),Image.Resampling.BILINEAR)).copy()
    return int.from_bytes(bits,'big'),small

def band_keys(number):
    # Five disjoint bands ensure a candidate with four changed bits shares a band.
    for i in range(5):
        yield (i, (number>>(13*i)) & (4095 if i==4 else 8191))

def historical(a):
    folders={'FMD':'FMD','Lytro':'lytro','DTD':'DTD/dtd/images','KTH':'KTH'}
    rows=[]
    for ds,folder in folders.items():
        with np.load(a.package/'studies/CVIU_Experience_couleur/cache'/(ds+'_palettes.npz')) as cache:
            rows+= [(ds,str(i),str(h),a.history/folder/str(i)) for i,h in zip(cache['ids'],cache['hash224'],strict=True)]
    sys.path.insert(0, str(a.package.resolve()))
    from numerical_release import ReleasedPayload
    with ReleasedPayload(a.package) as z:
        table=csv.DictReader(io.StringIO(z.read('clothing/data/sources.csv').decode()))
        rows += [('Clothing',r['key'],r['sha256_rgb224'],a.clothing/(r['key']+'.png')) for r in table]
    def read(r):
        ds,key,expected,p=r
        if not p.exists():
            if ds=='Clothing': return ds,expected,0,np.zeros((16,16,3),dtype=np.uint8),False
            raise FileNotFoundError(p)
        with Image.open(p) as src: im=src.convert('RGB').resize((224,224),BICUBIC)
        if sha(im.tobytes())!=expected: raise ValueError(f'Historical pixels differ: {ds} {key}')
        h,t=fingerprint(im)
        return ds,expected,h,t,True
    with ThreadPoolExecutor(a.threads) as pool: result=list(pool.map(read,rows))
    ds,exact,phash,thumb,available=zip(*result)
    np.savez_compressed(a.out,ds=ds,exact=exact,phash=np.asarray(phash,dtype=np.uint64),
                        thumb=np.stack(thumb),available=available)

def square(mask, label):
    binary=mask==label
    pieces=find_objects(binary.astype(np.uint8),max_label=1)
    if not pieces or pieces[0] is None:return None
    ys,xs=pieces[0]
    if min(ys.stop-ys.start,xs.stop-xs.start)<65:return None
    distance=distance_transform_cdt(np.pad(binary[ys,xs],1),metric='chessboard')[1:-1,1:-1]
    i,j=np.unravel_index(distance.argmax(),distance.shape)
    radius=int(distance[i,j])-1
    if radius<32:return None
    x=int(xs.start+j);y=int(ys.start+i)
    box=(x-radius,y-radius,x+radius+1,y+radius+1)
    if not binary[box[1]:box[3],box[0]:box[2]].all():raise ValueError('Crop outside mask')
    return box

def near_records(records):
    candidates=defaultdict(list)
    for idx, (_,h,_) in enumerate(records):
        for k in band_keys(h):candidates[k].append(idx)
    def search(h,thumbnail):
        ids=set()
        for k in band_keys(h):ids.update(candidates.get(k,()))
        for j in ids:
            name,other,t=records[j]
            if (h^other).bit_count()<=4 and np.abs(thumbnail.astype(np.int16)-t).mean()<=8.:
                yield name
    return search

def cohort(a):
    a.out.mkdir(parents=True,exist_ok=True);crops=a.out/'crops';crops.mkdir(exist_ok=True)
    counts=Counter();raw_patterns=defaultdict(set);catalogue=defaultdict(set);entries=[]
    with ZipFile(a.source/'labels.zip') as labels:
        lines=labels.read('labels/texture/pattern_ann.txt').decode().splitlines()
    counts['annotated_photos']=len(lines)
    for line in lines:
        image,*p=line.split();gender,cat,product,_=image.split('-')
        catalogue[product].add((gender,cat))
        if cat not in MAP:counts['ambiguous_catalogue_role']+=1;continue
        column,mask=MAP[cat];label=int(p[column])
        if label not in MOTIFS:counts['other_or_missing_pattern']+=1;continue
        raw_patterns[product].add(label)
        entries.append((image,product,cat,mask,label))
    with ZipFile(a.source/'masks.zip') as seg, ZipFile(a.source/'images.zip') as images:
        masknames=set(seg.namelist());imagenames=set(images.namelist());eligible=[]
        for item in entries:
            image,product,cat,mask,label=item
            if len(catalogue[product])!=1 or len(raw_patterns[product])!=1:
                counts['inconsistent_product']+=1;continue
            mn='segm/'+image[:-4]+'_segm.png'
            if mn not in masknames:counts['no_manual_mask']+=1;continue
            if 'images/'+image not in imagenames:counts['no_original_photo']+=1;continue
            eligible.append((*item,mn))
        def region(item):
            image,product,cat,label,pattern,mn=item
            with Image.open(io.BytesIO(seg.read(mn))) as s:
                mask=np.asarray(s);box=square(mask,label)
            if box is None:return None
            with Image.open(io.BytesIO(images.read('images/'+image))) as src:
                if src.size!=(mask.shape[1],mask.shape[0]):raise ValueError('Unaligned parsing mask '+image)
                thumb=src.convert('RGB').crop(box).resize((224,224),BICUBIC)
            code=sha(thumb.tobytes());local=crops/(image[:-4]+'.png')
            thumb.save(local,compress_level=1)
            h,t=fingerprint(thumb)
            record={'photo':image,'product':product,'category':cat,'motif':MOTIFS[pattern],'mask_label':label,
                    'left':box[0],'top':box[1],'right':box[2],'bottom':box[3],'rgb224_sha256':code}
            return record,h,t
        with ThreadPoolExecutor(a.threads) as pool:regions=list(pool.map(region,eligible))
    counts['mask_square_below_65']=sum(r is None for r in regions)
    regions=[r for r in regions if r is not None]
    invalid=set()
    candidate_sha=sha(''.join(sorted(r['photo']+' '+r['rgb224_sha256']+'\n' for r,_,_ in regions)).encode())
    if a.prior.suffix=='.json':
        screen=json.loads(a.prior.read_text())
        if candidate_sha!=screen['eligible_crop_sha256']:raise ValueError('New eligible crop pixels differ from the locked comparison')
        removed=set(screen['excluded_crop_hashes'])
        invalid.update(r['product'] for r,_,_ in regions if r['rgb224_sha256'] in removed)
        prior_counts=screen['previous_pixels_available'];past_matches=screen['historical_matches']
        unique_exact=screen['prior_exact_keys']
    else:
        with np.load(a.prior) as z:
            idx=np.where(z['available'])[0];ds=z['ds'];ph=z['phash'];tm=z['thumb']
            history=[(str(ds[i]),int(ph[i]),tm[i]) for i in idx]
            prior_counts=dict(Counter(str(ds[i]) for i in idx))
            prior_exact=set(z['exact'].tolist())
        unique_exact=len(prior_exact);lookup=near_records(history);past_matches=Counter();removed=set()
        for row,h,t in regions:
            matched=set(lookup(h,t))
            if row['rgb224_sha256'] in prior_exact:matched.add('exact')
            if matched:
                invalid.add(row['product']);removed.add(row['rgb224_sha256'])
                for source in matched:past_matches[source]+=1
    screen=dict(eligible_crop_sha256=candidate_sha,previous_pixels_available=prior_counts,
                historical_matches=dict(past_matches),prior_exact_keys=unique_exact,excluded_crop_hashes=sorted(removed))
    (a.out/'history_screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    counts['historical_match_products']=len(invalid)
    lookup=near_records([(r['product'],h,t) for r,h,t in regions])
    duplicates=set()
    for row,h,t in regions:
        duplicates.update(tuple(sorted((row['product'],other))) for other in lookup(h,t) if other!=row['product'])
    for left,right in duplicates:invalid.update((left,right))
    counts['duplicate_product_pairs']=len(duplicates)
    counts['photos_removed_with_duplicate_products']=sum(r['product'] in invalid for r,_,_ in regions)
    by_product=defaultdict(list)
    for row,_,_ in regions:
        if row['product'] not in invalid:by_product[row['product']].append(row)
    by_motif=defaultdict(list)
    for prod,views in by_product.items():
        unique={}
        for r in sorted(views,key=lambda r:order('view',r['photo'])):
            unique.setdefault(r['rgb224_sha256'],r)
        counts['repeated_crop_within_product']+=len(views)-len(unique)
        kept=list(unique.values())[:2]
        counts['view_cap']+=len(unique)-len(kept)
        by_motif[kept[0]['motif']].append((prod,kept))
    final=[];attrition=[];excluded_products=[]
    for motif,group in sorted(by_motif.items()):
        total=len(group)
        if total<25:
            counts['below_minimum_class_size']+=sum(len(views) for _,views in group)
            excluded_products.append((motif,total));continue
        group.sort(key=lambda p:order('product-cap',p[0]))
        counts['product_cap']+=sum(len(views) for _,views in group[600:]);group=group[:600]
        group.sort(key=lambda p:order('split',p[0]))
        n=len(group);nt=max(1,int(.35*n));nv=max(1,int(.20*n))
        for i,(prod,views) in enumerate(group):
            role='test' if i<nt else 'validation' if i<nt+nv else 'fit'
            final.extend([{**view,'role':role} for view in views])
        attrition.append(dict(motif=motif,eligible_products=total,included_products=n))
    final.sort(key=lambda r:r['photo']);write_csv(a.out/'manifest.csv',final)
    seen=defaultdict(set)
    for r in final:seen[r['product']].add(r['role'])
    if not all(len(x)==1 for x in seen.values()):raise ValueError('Shared product across roles')
    if len({r['rgb224_sha256'] for r in final})!=len(final):raise ValueError('Repeated selected pixels')
    for item in attrition:
        for role in ('fit','validation','test'):
            rows=[r for r in final if r['motif']==item['motif'] and r['role']==role]
            item[role+'_photos']=len(rows);item[role+'_products']=len({r['product'] for r in rows})
    write_csv(a.out/'counts.csv',attrition)
    counts['retained_photos']=len(final);counts['retained_products']=len(seen)
    reasons=('ambiguous_catalogue_role','other_or_missing_pattern','inconsistent_product','no_manual_mask',
             'no_original_photo','mask_square_below_65','photos_removed_with_duplicate_products',
             'repeated_crop_within_product','view_cap','below_minimum_class_size','product_cap','retained_photos')
    if sum(counts[k] for k in reasons)!=len(lines):raise ValueError('Attrition does not sum to the source')
    report=dict(counts=dict(counts),previous_pixels_available=prior_counts,prior_exact_keys=unique_exact,
                historical_matches=dict(past_matches),omitted_classes=excluded_products,
                manifest_sha256=sha((a.out/'manifest.csv').read_bytes()))
    (a.out/'cohort.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='action',required=True)
    for action,fields in [('history',('package','history','clothing','out')),('prepare',('source','prior','out'))]:
        p=sub.add_parser(action)
        for f in fields:p.add_argument(f,type=Path)
        p.add_argument('--threads',type=int,default=3)
    a=ap.parse_args()
    historical(a) if a.action=='history' else cohort(a)
