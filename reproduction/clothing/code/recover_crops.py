"""Recover retained crops when the source URL still serves matching pixels."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
from urllib.request import Request, urlopen
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def recover(destination):
    destination.mkdir(parents=True,exist_ok=True)
    with (ROOT/'data/sources.csv').open() as f:
        retained=list(csv.DictReader(f))
    with (ROOT/'data/manifest.csv').open(encoding='utf-8-sig') as f:
        manifest={(hashlib.sha256(('HSMT-textile-v1|'+row['URL']).encode()).hexdigest(),row['Class Name']):row
                  for row in csv.DictReader(f)}
    def one(source):
        key=source['key']
        record=manifest[key,source['class']]
        path=destination/(key+'.png')
        try:
            if path.is_file():
                with Image.open(path) as im:
                    if im.size==(224,224) and hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()==source['sha256_rgb224']:
                        return key,'verified'
                return key,'existing file differs, preserved'
            url=record['URL']
            if url.startswith('http://'):
                url='https://'+url[7:]
            if not url.startswith('https://'):
                raise ValueError('Unsupported URL scheme')
            with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=20) as response:
                data=response.read(12000001)
            if len(data)>12000000:
                raise ValueError('Oversized response')
            with Image.open(io.BytesIO(data)) as im:
                w,h=im.size
                ow,oh=[int(record['Original '+k]) for k in ('Width','Height')]
                x,y,cw,ch=[int(record['Crop '+k]) for k in ('X','Y','Width','Height')]
                box=(max(0,round(x*w/ow)),max(0,round(y*h/oh)),
                     min(w,round((x+cw)*w/ow)),min(h,round((y+ch)*h/oh)))
                crop=im.convert('RGB').crop(box).resize((224,224),Image.Resampling.BICUBIC)
            if hashlib.sha256(crop.tobytes()).hexdigest()!=source['sha256_rgb224']:
                return key,'pixels differ'
            crop.save(path)
            return key,'verified'
        except (OSError,ValueError,Image.DecompressionBombError) as exc:
            return key,str(exc)
    with ThreadPoolExecutor(max_workers=8) as pool:
        status=list(pool.map(one,retained))
    with (destination/'recovery.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(('key','status'))
        w.writerows(status)
    n=sum(s=='verified' for _,s in status)
    print(f'{n}/{len(retained)} original crops verified.')
    return n==len(retained)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination',type=Path)
    raise SystemExit(0 if recover(p.parse_args().destination) else 1)
