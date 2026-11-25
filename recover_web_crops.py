"""Recover the retained clothing crops when providers still return the archived pixels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen
from PIL import Image

def variants(url):
    parts=urlsplit(url)
    if parts.scheme not in ('http','https'):raise ValueError("Unsupported source URL")
    hosts=[parts.netloc]
    if parts.hostname=='img.etsystatic.com':hosts.append('i.etsystatic.com')
    return list(dict.fromkeys(urlunsplit(('https',host,parts.path,query,'')) for host in hosts for query in (parts.query,'')))

def recover(data,output,workers):
    output.mkdir(parents=True,exist_ok=True)
    with (data/'sources.csv').open(encoding='utf-8') as f:retained=list(csv.DictReader(f))
    with (data/'manifest.csv').open(encoding='utf-8-sig') as f:
        records={(hashlib.sha256(('HSMT-textile-v1|'+r['URL']).encode()).hexdigest(),r['Class Name']):r for r in csv.DictReader(f)}
    def one(source):
        key,expected=source['key'],source['sha256_rgb224']
        path=output/(key+'.png')
        if path.exists():
            with Image.open(path) as im:
                if im.size==(224,224) and hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()==expected:
                    return key,'matched existing'
            return key,'existing file has a different fingerprint'
        r=records[key,source['class']]
        ow,oh,x,y,cw,ch=(int(r[n]) for n in ('Original Width','Original Height','Crop X','Crop Y','Crop Width','Crop Height'))
        for url in variants(r['URL']):
            for accept in ('*/*','image/jpeg','image/webp,image/*;q=0.8'):
                try:
                    req=Request(url,headers={'User-Agent':'Mozilla/5.0','Accept':accept})
                    with urlopen(req,timeout=15) as response:raw=response.read(12000001)
                    if len(raw)>12000000:continue
                    with Image.open(io.BytesIO(raw)) as im:
                        w,h=im.size
                        box=(round(x*w/ow),round(y*h/oh),round((x+cw)*w/ow),round((y+ch)*h/oh))
                        clipped=(max(0,box[0]),max(0,box[1]),min(w,box[2]),min(h,box[3]))
                        for cropbox in dict.fromkeys((clipped,box)):
                            crop=im.convert('RGB').crop(cropbox).resize((224,224),Image.Resampling.BICUBIC)
                            if hashlib.sha256(crop.tobytes()).hexdigest()==expected:
                                crop.save(path);return key,'matched source'
                except (OSError,ValueError,Image.DecompressionBombError):
                    continue
        return key,'no matching pixels recovered'
    with ThreadPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(one,retained))
    with (output/'recovery_extended.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(('key','status'));writer.writerows(rows)
    print(f"{sum(v.startswith('matched') for _,v in rows)} / {len(rows)} retained crops have the expected RGB fingerprint.")
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path,help='Extracted r4/clothing/data directory')
    p.add_argument('output',type=Path);p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();recover(a.data,a.output,a.workers)
