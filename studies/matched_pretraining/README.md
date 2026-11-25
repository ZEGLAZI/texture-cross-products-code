# Product holdout and matched pretraining

The two original DINO checkpoints add a ResNet-50 and a ViT-S/16 trained on ImageNet-1K. The catalogue study compares those two encoders, TorchVision's supervised ResNet-50, DINOv2 ViT-S/14 and the three-block wavelet/RGB descriptor. The two original kernels, three penalty regimes and coordinates are defined in the article. PROTOCOL.md records the design before any new classifier outputs. locked_utc.txt and audit/manifest_locked_utc.txt document the internal chronology. The three earlier tasks had already been inspected. The historical contrast that motivated the new endpoint is reconstructed by retrospective.py.

## Regenerate the numerical results

Run from the directory containing replay_submission.py. After installing requirements.txt the following uses the paired decisions on the three previous tasks and the correctness counts of every held-out catalogue product. Each product has one or two views. Each model column is the number of correct views, so the paired comparisons and both averaging conventions can be reconstructed without redistributing provider records.

    python studies/matched_pretraining/replay.py --mode tables --report matched_tables.json

To repeat every validation grid and outer fit for the two original DINO encoders on DTD, KTH and FMD-to-Lytro, use the additional_features/ directory at the root of this package. The verifier independently expands the cross-products and uses an eigensolver for predictions. The original fits used a positive-definite linear solve. Each check compares all five test-decision vectors, internal validation scores and the retained penalties by fold.

    python studies/matched_pretraining/replay.py --mode full --report matched_refits.json

## Reconstruct all five garment representations from source pixels

The DeepFashion-MultiModal authors distribute images, manual segmentation masks and texture annotations at https://github.com/yumingj/DeepFashion-MultiModal . Obtain the three provider archives and place them in an inputs directory as images.zip, masks.zip (provider name segm.zip) and labels.zip. Use the unmodified ZIPs. The executable verifies their SHA-256 fingerprints. Provider images, labels and masks are read on the reviewer's machine under the dataset terms. The results directory contains our own test performance records, validation scores and choices.

Also place the following provider files in inputs, under the names on the left.

| File | Public source |
|---|---|
| official_r50.pth | https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth |
| official_vit.pth | https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth |
| resnet50-11ad3fa6.pth | https://download.pytorch.org/models/resnet50-11ad3fa6.pth |
| dino.pth | https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth |
| dinov2.zip | https://codeload.github.com/facebookresearch/dinov2/zip/e1277af2ba9496fbadf7aec6eba56e8d882d1e35 |

Use Python 3.12, the top-level numerical requirements, PyTorch 2.5.1 CPU, TorchVision 0.20.1 CPU, pytorch-wavelets 1.3.0 and PyWavelets 1.8.0, as in the pixel-replay instructions at the root. With the eight files available locally:

    python studies/matched_pretraining/from_pixels.py . inputs garment_replay --threads 3

The command checks the three source archives, regenerates the aligned crops and exact catalogue split, verifies every crop on extraction, recomputes all five representations, refits the models and checks every product-level correctness vector and all validation scores. The individual decisions and private source manifest can be inspected in garment_replay/. SHA-256 fingerprints in audit/ identify the released cohort. The first run recovered 194 of the earlier 211 Web crops for perceptual comparison and checked every earlier exact fingerprint. For reconstruction without all four previous image collections, the command reuses the locked historical cross-collection screening outcome after verifying a fingerprint over *all* candidate crops. It repeats the within-catalogue duplicate screen. To repeat the cross-collection search as well, obtain the previous provider images and the earlier Web crops using the root instructions. Supply exactly the scope reported in audit/cohort.json for a like-for-like check, and use:

    python studies/matched_pretraining/garments.py history . previous_images old_web_crops previous_signatures.npz
    python studies/matched_pretraining/from_pixels.py . inputs garment_replay --history-index previous_signatures.npz

The four directories inside previous_images are DTD/dtd/images, KTH, FMD and lytro. Missing Web crops still contribute exact hashes. A changed availability scope should be reported when interpreting a fresh historical screen.

## Re-extract the original DINO features for the earlier tasks

The original pixels use the same four directories described above. Hashes of the original and resized inputs are checked per image. The extraction supports --datasets to select a subset. The public upstream implementation in vendor/ is pinned to facebookresearch/dino commit 7c446df5b9f45747937fb0d72314eb9f7b66930a, under its included licence.

    python studies/matched_pretraining/extract_dino1.py . previous_images inputs fresh_features --threads 7
    python studies/matched_pretraining/replay.py --mode full --features fresh_features --report fresh_fits.json

## Reading the results

The main quantities are full minus squares at the native scale, after matching the mean training diagonal, and after separate penalty selection. In contrasts.csv, T is the primary external shift between the supervised ResNet-50 and DINOv2. A compares the shift of DINO ResNet-50 with supervised ResNet-50 and B compares DINO ResNet-50 with DINO ViT-S/16. A and B share one family of eight conditional intervals; T has a separate prespecified interval. The three retrospective shifts are in retrospective_results/contrast.csv. The interval procedure preserves physical specimens, repeated photographs, captures or catalogue products as appropriate. The CSV files retain unrounded values in percentage points. garment_classes.csv and garment_image_macro.csv give the class and photograph-weighted breakdowns.
