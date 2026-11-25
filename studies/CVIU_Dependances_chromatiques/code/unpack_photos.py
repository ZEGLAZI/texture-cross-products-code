"""Extract only manifest-listed images from the released DTD/KTH/FMD source archives."""
import argparse
from pathlib import Path, PurePosixPath
import tarfile
import zipfile
import numpy as np

def unpack(corpus,archive,cache,output):
    with np.load(cache/(corpus+'_inputs.npz')) as z: keys=set(z['ids'].tolist())
    def write(key,buf):
        if key not in keys:raise ValueError(key)
        dest=output/key;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(buf)
        keys.remove(key)
    if corpus=='FMD':
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if name in keys:write(name,z.read(name))
    else:
        with tarfile.open(archive,'r:*') as z:
            for m in z:
                if not m.isfile():continue
                name=PurePosixPath(m.name).as_posix()
                if corpus=='DTD' and name.startswith('dtd/images/'):name=name[len('dtd/images/'):]
                if name in keys:write(name,z.extractfile(m).read())
    if keys: raise RuntimeError(f'{len(keys)} images missing, first: '+str(sorted(keys)[:3]))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('corpus',choices=('FMD','KTH','DTD'))
    for a in ('archive','cache','output'):p.add_argument(a,type=Path)
    a=p.parse_args();unpack(a.corpus,a.archive,a.cache,a.output)
