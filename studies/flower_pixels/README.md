# Flower transfer from the source and target photographs

The historical flower classifiers use the 2,213 source descriptors in `historical/external/data/tf.npz`. The 420 retained iNaturalist observations provide a distinct target. Supplement section S9 also reports a sensitivity using descriptors re-extracted from the pixels of both collections. The two feature files here are numerical derivatives of the authenticated source and target images. The 12 class and kernel settings use the archived source-validation choices.

From the unpacked reviewer package:

    python -m zipfile -e data_and_code_r4.zip reproduction
    python studies/flower_pixels/check.py reproduction --report flower_pixel_recall.json

To repeat the image extraction, use the TensorFlow Flowers JPEG archive and the 420 original iNaturalist JPEGs referenced by their retained photo IDs and hashes in `historical/revision/results/cohort/nouveau_manifeste.csv`. Use the CPU extraction dependencies listed in the package README. Set the command paths relative to the reviewer package:

    python reproduction/historical/revision/code/external_images.py tf reproduction/historical flower_photos measured_source --threads 2
    mkdir fresh_cohort
    cp reproduction/historical/revision/results/cohort/nouveau_manifeste.csv fresh_cohort/
    python reproduction/historical/revision/code/flowers_new.py reproduction/historical fresh_cohort target_jpegs reproduction/historical/revision/results/selection/Flowers.json
    python studies/flower_pixels/check.py reproduction --source-parts measured_source --target fresh_cohort/features.npz --report flower_pixel_recall.json

The TF extractor checks the source identity and pixel hashes. The target extractor checks the original JPEG hash. In twelve comparisons on all 420 observations, a change to a wrong class in the fixed two-product model preserves its recall and a changed decision for the selected colour-centred model changes its recall. The other ten label sets agree exactly. The paper's reported fixed two-product and additive recalls use the original source-trained classifier.
