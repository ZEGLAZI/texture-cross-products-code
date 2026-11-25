# Cross-products in colour–texture descriptors and frozen transformer layers

This package contains a retrospective, paired evaluation on DTD and KTH-TIPS2-b. Read Table 2 of the manuscript and Supplement S8. All scores are macro-recall in percent. All gains are differences expressed in percentage points.

## What is included

* `results/*_summary.json` reports four fixed-coefficient recalls, conditional paired intervals, the effect of selecting the ridge penalty, unit-diagonal results and changes in decisions. H means the archived HSMT blocks. D means the patch means of frozen DINOv2 layers 3, 6 and 12. S means linear plus squared similarities. X means linear plus cross-products. Q includes all terms. The principal gain adds cross-products to S.
* `results/*_folds.csv`, `*_classes.csv` and `*_settings.json` give the split results, per-class transitions and selection grids. `*_predictions.npz` contains both families' predictions, labels, source identities and test indices in the same order.
* `features/` contains derived, unstandardized, float32 DINOv2 vectors and image identifiers, obtained from provider pixels. The enclosing numerical release supplies the archived colour–texture inputs under `historical/` and `review2/`. Paths in the commands below start at the outer release root. `code/` holds the new extraction and evaluation programs. The parent numerical archive from which the original colour–texture inputs were obtained has SHA-256 `954a52a008c60d2df93fac284bb2678d69e53f99b76ace9e7a7839aa3096b49f`.
* `protocol.md` gives the choices recorded before reading new DINOv2 classifier results, the estimands and the scope of the intervals. The outer `SHA256SUMS` covers every distributed file. Intervals are conditional on the fitted classifiers. Both datasets were already familiar from the manuscript.

## Refit from the included numerical inputs

Use Python 3.12 and run these commands from the unzipped package root. The execution used NumPy 2.3.5, SciPy 1.17.0, threadpoolctl 3.6.0, Pillow 13.0.0 and PyTorch 2.5.1+cpu. PyTorch and Pillow are needed only for extracting image features or preparing the KTH mirror. Replaying decisions from the included features needs neither a model download nor a GPU.

```sh
python -m pip install -r requirements-replay.txt
python reproduce.py full --output reproduced
```

## Re-extract DINOv2 vectors from provider images

The DTD release is `dtd-r1.0.1.tar.gz` (SHA-256 `e42855a52a4950a3b59612834602aa253914755c95b0cff9ead6d07395f8e205`). Extract it into `inputs`, so that `inputs/dtd/images` contains the 47 class directories. The KTH layout is `inputs/kth/KTH-TIPS2-b/<material>/sample_<letter>/<file>.png`. The extraction validates DTD file bytes and resized RGB pixels, and validates the native RGB pixels and dimensions of every KTH image against the manuscript's manifest.

* [Oxford DTD release](https://thor.robots.ox.ac.uk/dtd/dtd-r1.0.1.tar.gz)
* [KTH provider](https://www.csc.kth.se/cvap/databases/kth-tips/download.html)
* [DINOv2 implementation archive](https://codeload.github.com/facebookresearch/dinov2/zip/e1277af2ba9496fbadf7aec6eba56e8d882d1e35)
* [DINOv2-S/14 weights](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth)

Save the model archive and weights in `inputs/dinov2.zip` and `inputs/dino.pth`. Both hashes are checked against the pinned implementation and model used in the HSMT manuscript. The provider photographs and model weights remain available separately under their respective terms.

The KTH provider download returned HTTP 502 during this run. The following alternative recovers all images from a specific public Git revision. Its train/test assignments are ignored. The included script admits each image only if its native RGB fingerprint matches the archived manifest. Folds are always taken from the physical-specimen protocol under `historical/` and `review2/`.

```sh
git clone --no-checkout https://github.com/mayank-gubba/KTH-TIPS2-b.git inputs/KTH_repository
python cross_representation/code/prepare_kth_mirror.py . inputs/KTH_repository inputs/kth
python cross_representation/code/extract_features.py DTD . inputs/dtd/images inputs/dinov2.zip inputs/dino.pth fresh_features
python cross_representation/code/extract_features.py KTH . inputs/kth inputs/dinov2.zip inputs/dino.pth fresh_features
mkdir -p rerun
python cross_representation/code/check_classical.py . rerun/HSMT_refits.json
python cross_representation/code/compare.py DTD . fresh_features/DTD_features.npz rerun
python cross_representation/code/compare.py KTH . fresh_features/KTH_features.npz rerun
python cross_representation/code/verify_replay.py cross_representation/results rerun
python cross_representation/code/table_rows.py rerun rerun/tables
```

The extraction uses full-image 224 × 224 bicubic resizing, with checkpoint channel normalization. The numerical verification of the last-layer CLS uses the historical full-field fixed-penalty classifier. The patch tokens of each layer and the CLS control use the same decoded images. The two adjusted primary intervals use Bonferroni across the two datasets. Intervals for separate gains and for the validation-selected penalty sensitivity are pointwise. There is no independent acquisition in this package.

## Regime change from paired predictions

The original protocol was recorded before scoring the new DINOv2 classifiers on two previously inspected datasets. The follow-up contrast was defined after reading their fixed and selected regimes. With D denoting the DINOv2 gain minus the colour–texture gain, T is D under separate penalty selection minus D at the fixed penalty. The eight predictions for a given test appearance stay together. T has its own exploratory Bonferroni family across DTD and KTH. Its pointwise intervals are also archived.

`results/representation_comparison.csv` gives absolute recalls, the individual gains with pointwise limits, and the representation contrasts with their stated limits. `results/regime_change.json` gives T from unrounded decisions and both interval types. The DTD and KTH test sets are balanced within every split. The grouping units are the 5,530 represented exact-image groups on DTD and 44 physical specimens stratified within 11 KTH materials. These intervals condition on the fitted classifiers. To calculate just these tables from saved decisions, run the following from the outer release root.

```sh
python cross_representation/code/table_rows.py cross_representation/results reproduced_cross_tables
```

## Matching the mean training diagonals

Release r4 adds the follow-up in `scale_protocol.md`. The archive stores the matched per-image decisions and measured diagonal and zero-block diagnostics for every representation and split in `scale_results/`. Original folds, preprocessing and square-kernel scaling are retained. In the fixed-penalty sensitivity the complete training and test-to-training kernels share a scalar learned on the corresponding training set. The tables list all native-scale selected penalties separately from the reciprocal penalty for the matched full kernel.

From the outer numerical archive root, the following commands reproduce just this sensitivity and compare the other fit metadata with the existing selection records.

```sh
python cross_representation/code/diagonal_match.py DTD matched_refit
python cross_representation/code/diagonal_match.py KTH matched_refit
python cross_representation/code/diagonal_tables.py matched_refit cross_representation/results matched_tables
```

The full reproduction entry point compares the recomputed predictions, diagnostics and summaries against the archive. The tables-only entry point regenerates the same paired statistics and penalty records from saved outputs.
