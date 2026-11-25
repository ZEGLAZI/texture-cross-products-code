"""Check the unordered-pair indexing, offsets and exact shuffled expectation on known colours."""
import argparse
from pathlib import Path
import numpy as np
from spatial import load_spatial

def reference(labels):
    h,w=labels.shape;local=np.zeros(378);permuted=np.zeros(378)
    pair=lambda a,b: max(a,b)*(max(a,b)+1)//2+min(a,b)
    for r in (2,8):
        for dy,dx in ((0,r),(r,0),(r,r),(r,-r)):
            keys=[pair(labels[y,x],labels[y+dy,x+dx]) for y in range(h-dy) for x in range(max(0,-dx),w-max(0,dx))]
            local+=np.bincount(keys,minlength=378)/(8*len(keys))
    flat=labels.flatten()
    for i in range(len(flat)):
        for j in range(i):
            permuted[pair(flat[i],flat[j])]+=2/(len(flat)*(len(flat)-1))
    return np.sqrt(np.stack((local,permuted)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('binary',type=Path);binary=p.parse_args().binary
    encode=load_spatial(binary)
    rng=np.random.default_rng(20261120)
    gray=np.array([0,120,255],dtype=np.uint8)
    grids=[np.zeros((15,17),dtype=int),np.indices((15,17))[0]%3,np.indices((15,17))[1]%3,
        rng.integers(0,3,(15,17)),rng.integers(0,3,(20,22))]
    for labels in grids:
        pixels=np.repeat(gray[labels][...,None],3,axis=-1)
        np.testing.assert_allclose(encode(pixels),reference(9*labels+4),atol=2e-12,rtol=0)
    x=grids[-1];pic=lambda g:np.repeat(gray[g][...,None],3,axis=-1)
    actual,permuted=encode(pic(x)),encode(pic(rng.permutation(x.ravel()).reshape(x.shape)))
    np.testing.assert_array_equal(actual[1],permuted[1]);assert np.linalg.norm(actual[0]-permuted[0])>.001
    print('Five independent enumeration checks and a pixel-permutation control passed.')
