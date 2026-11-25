"""Compute the pHash and thumbnail check used for clothing/FMD overlap screening."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.fft import dctn

def sig(im):
    gray=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS))
    d=dctn(gray,norm='ortho')[:8,:8].ravel()
    ph=int.from_bytes(np.packbits(d>np.median(d[1:])).tobytes(),'big')
    thumb=np.asarray(im.convert('RGB').resize((16,16),Image.Resampling.BILINEAR))
    return ph,thumb
def main(root,output):
    all_h=[];all_v=[];keys=[]
    for p in sorted(root.glob('*/*.jpg')):
        with Image.open(p) as im:x=im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC)
        h,v=sig(x);keys.append(p.name);all_h.append(h);all_v.append(v)
    if len(keys)!=5640:raise ValueError('Expected the official 5,640 DTD photographs')
    np.savez(output,h=np.array(all_h,dtype='uint64'),v=np.stack(all_v),ids=keys)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dtd_images',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.dtd_images,a.output)
