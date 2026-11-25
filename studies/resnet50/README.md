# Three stages of frozen ResNet-50 for texture recognition

The two kernels differ by the three products between stage similarities. The weights, stages, pool, validation rule and contrasts were fixed before computing the CNN results on these previously studied tasks. Read PROTOCOL.md for the full design. The photo identity, grouping and inner/outer assignments agree with the original colour–texture and DINOv2 comparisons.

## Refit offline

From the unpacked reviewer package, install the versions in its top-level requirements.txt and use

    python studies/resnet50/replay.py --mode full

This reads data/ and the distributed source, target and split records. It refits all 15 outer folds, checks each of the five test prediction arrays per fold, recomputes every validation score and all 999 source/target resamples, and checks the paired statistics and Figure 2 inputs. A separate calculation on the first fold of each task expands the kernel into its individual products and solves the ridge problem by eigendecomposition. The tables-only mode uses the distributed decisions. The package-wide command python replay_submission.py --mode full also verifies the other numerical experiments without requiring manuscript PDFs.

## Extract the CNN vectors from images

The numerical replay above uses the frozen vectors in this distribution. For extraction obtain the original DTD, KTH-TIPS2-b, FMD and Lytro pinhole images with the paths given by the IDs in studies/CVIU_Experience_couleur/cache/*_palettes.npz. DTD_ROOT is the folder containing the 47 attribute directories. KTH_ROOT, FMD_ROOT and LYTRO_ROOT are the provider folders relative to the corresponding IDs. The extractor checks both source-file and resized-RGB SHA-256 before using any image. Duplicate groups in DTD and KTH are retained as recorded by the study.

The reference CPU environment used Python 3.12, PyTorch 2.5.1+cpu, TorchVision 0.20.1+cpu, NumPy 2.3.5 and Pillow 13.0.0. Install the CPU wheels from the PyTorch index. Obtain the unmodified TorchVision ResNet50_Weights.IMAGENET1K_V2 file from https://download.pytorch.org/models/resnet50-11ad3fa6.pth and keep it as resnet50-11ad3fa6.pth. Its SHA-256 is 11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca. The TorchVision reference is https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.resnet50.html . For the classifier comparison both networks receive identically resized full RGB fields.

Example commands below use shell variables for your actual dataset roots. Run from the reviewer package root.

    python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
    cnn=studies/resnet50
    manifests=studies/CVIU_Experience_couleur/cache
    python "$cnn/extract.py" DTD "$DTD_ROOT" "$manifests" resnet50-11ad3fa6.pth pixel_run
    python "$cnn/extract.py" KTH "$KTH_ROOT" "$manifests" resnet50-11ad3fa6.pth pixel_run
    python "$cnn/extract.py" FMD "$FMD_ROOT" "$manifests" resnet50-11ad3fa6.pth pixel_run
    python "$cnn/extract.py" Lytro "$LYTRO_ROOT" "$manifests" resnet50-11ad3fa6.pth pixel_run
    python "$cnn/check_pixel_repeat.py" "$cnn/data" pixel_run

All reference arrays are float32, in stages layer2, layer3 and layer4 ordered by the archived photo ID. The classifier uses float64. The reference extraction uses 16 images per CPU batch and six Torch threads. Each JSON in data/ records the checkpoint, extraction versions and coordinate hashes. The exact coordinate comparison is intended for the reference environment.

## Outputs and uncertainty

results/gains.csv gives 27 square/full recalls, paired gain estimates and pointwise intervals. results/contrasts.csv contains the two families of six prespecified CNN contrasts. results/per_fold.csv records corrected and introduced errors for the same cases. The dataset-specific validation, settings and decision files describe every fitted CNN. Fixed and matched use lambda=1, with the full kernel scaled by the training-trace ratio for matched. Selected uses independent source validation for each kernel. DTD repeated appearances and exact duplicates remain together during inference. KTH holds out physical specimens, and the transfer is evaluated on Lytro captures. resnet_bootstrap_draws.csv records the 999 paired source/target draws; their summary is in resnet_bootstrap.json. The source and target models and predictions can be rebuilt using refit_transfer.py and the shared bootstrap draw implementation in studies/source_refit/.

The package-wide publication check also compares the ordered fitting, validation and target image indices of every one of the 999 draws with those of the original colour–texture and DINOv2 program. To repeat this test alone, run `python studies/resnet50/check_pairing.py`.
