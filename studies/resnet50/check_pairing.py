"""Verify the ordered bootstrap indices used by the CNN and the original transfer analysis."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import numpy as np
import refit_transfer


def compare(root, folder):
    original = root/"studies/source_refit/refit.py"
    spec = spec_from_file_location("colour_transformer_source_refit", original)
    original_fit = module_from_spec(spec)
    spec.loader.exec_module(original_fit)
    acquisition = root/"studies/CVIU_Experience_couleur"
    original_fit.initialize(acquisition/"cache", acquisition/"results")
    refit_transfer.initialize(root,folder/"data",folder/"results")
    checked = 0
    for key in range(refit_transfer.REPLICATES):
        for old, cnn in zip(original_fit.draws(key),refit_transfer.draw(key),strict=True):
            np.testing.assert_array_equal(old,cnn,err_msg=f"Unpaired image indices on bootstrap draw {key}")
            checked += len(cnn)
    return {"paired_draws":refit_transfer.REPLICATES,"identical_ordered_image_indices":checked}


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    print(compare(folder.parent.parent,folder))
