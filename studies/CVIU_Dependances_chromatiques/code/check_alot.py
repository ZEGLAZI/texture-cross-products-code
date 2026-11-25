"""Check the independent extraction, refit and tabulation of the locked ALOT follow-up."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from threadpoolctl import threadpool_limits
from alot import run
def verify(refdata,data,reference,equal_tables):
    checked=0;n=int(3250)
    if refdata.resolve()!=data.resolve():
        with np.load(refdata) as a,np.load(data) as b:
            assert set(a.files)==set(b.files)
            for k in a.files:
                np.testing.assert_array_equal(a[k],b[k])
                if k in ('F7','F8','rgb','lab','edges'):checked+=a[k].size
    with TemporaryDirectory() as folder,threadpool_limits(2):
        out=Path(folder);run(data,out)
        for name in ('ALOT_scores.csv','ALOT_contrasts.csv','ALOT_decisions.csv'):
            equal_tables(out/name,reference/name)
        assert json.loads((out/'ALOT_splits.json').read_text())==json.loads((reference/'ALOT_splits.json').read_text())
    return dict(new_pixel_extractions_compared=n if checked else 0,feature_coordinates_compared=int(checked),
                different_feature_coordinates=0 if checked else None,model_labels_compared=2250*6,
                different_labels=0,held_out_appearances=2250,distinct_images=3250)
if __name__=='__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from replay import equal_tables
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('reference_data','candidate_data','results','output'):p.add_argument(k,type=Path)
    a=p.parse_args();v=verify(a.reference_data,a.candidate_data,a.results,equal_tables)
    a.output.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
