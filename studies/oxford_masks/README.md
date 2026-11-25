# Oxford flower mask experiment

This is the extraction used in Table S14 of the IVC edition. Use the Python 3.12 numerical requirements at the package root. The full verification recomputes the fixed and validation-selected KRR models from features/ and the training descriptors in the immutable r4 numerical archive. Four views of each of the 168 accepted photographs are included, with three flower categories. The source, class order, transforms, eight-image batch and 224-pixel input are held fixed.

From the reviewer package root:

    python -m zipfile -e data_and_code_r4.zip reproduction
    python studies/oxford_masks/check.py reproduction

To repeat from photographs, obtain 17flowers.tgz and trimaps.tgz from https://www.robots.ox.ac.uk/~vgg/data/flowers/17/ and install the pixel dependencies named in the main README. On the reference x86-64 CPU environment, run:

    python studies/oxford_masks/pixels.py reproduction oxford_archives fresh_oxford
    python studies/oxford_masks/check.py reproduction --fresh-features fresh_oxford

The pixel script checks both provider archive fingerprints and the individual retained image and mask identities. The older r4 masks.py script regenerates the inclusion and exclusion list from the 183 annotations. The current edition uses the four views in this directory for Table S14. Its numerical reference is results/predictions.npz.
