# Colour encoding and transfer to an independently acquired material collection
This addendum can be reviewed independently of the manuscript submission. It contains full numerical inputs for classifier fitting, including the original manuscript’s two wavelet blocks on DTD and KTH-TIPS2-b, plus every new held-out decision. Raw photographs and DINOv2 weights are obtained from their providers. PROTOCOLE.md and ADDITIF_SPATIAL.md document when each question was fixed.

## Refit offline
From the extracted directory, on a Python 3.12 installation, run:

```bash
python -m pip install -r requirements-replay.txt
python replay.py --threads 3
```

The replay refits the five global-palette models and the two spatial comparison models. It also reruns the FMD validation grid and all external kernel fits, regenerates seven CSV tables in a temporary folder, and compares every predicted label and every displayed estimate. The reference is the bundled decisions, never an input to fitting. For the two manuscript datasets the original RGB decisions are also checked independently within every fold. Tested environment and verification output are in environment.json and results/replay_verified.json.

## Output files
- results/palettes.csv gives the five palette scores and the two planned comparisons against RGB on each of the three datasets. Units are percentage points for differences. Bounds are simultaneous across the six contrasts (Bonferroni).
- results/spatial.csv gives local pairs versus the original RGB, and versus a matched-dimension unordered-pair expectation under pixel permutation. The interval correction counts twelve contrasts. DTD global-palette outcomes had already been viewed when this extension was chosen.
- results/external_kernels.csv and results/external_contrasts.csv give test performance and the interaction between representations and ridge selection. The prespecified external contrast is the row primary_captures. Other 95% intervals in these two files are pointwise.
- results/by_class.csv, results/by_fold.csv and results/spatial_by_class.csv retain the material/class or split-level breakdown. The three *_predictions.npz and three *_spatial_predictions.npz files align every test label with its ID and group. results/Lytro_settings.json includes the source-only validation grids and selected penalties. All reported recalls give equal weight to classes.
- cache/*_design.npz and cache/*_palettes.npz preserve sample IDs, groups or splits, image fingerprints and original dimensions. The two FMD source subsets are denoted fit and val. The first consists of 500 images previously used for training in the source archive. The other 500, previously used as a source-dataset test set, serve as inner validation in this new cross-dataset transfer. Both are pooled for final fitting. Lytro is solely the new test set.

## Rebuild the newly measured descriptors from provider files
Use a separate working directory if you wish to preserve your reference outputs. The following example assumes the extracted addendum is the working directory. A C++17 compiler and PyTorch 2.5.1 CPU with PyWavelets 1.8.0 and pytorch-wavelets 1.3.0 are additionally required for this path. The original wavelet implementation uses pkg_resources; setuptools 75.8.0 was used. All comparisons run on RGB-decoded, full-field 224 × 224 bicubic images; no masks, crops or augmentations are applied here.

Download the three archives from the providers below, save them under inputs/dtd.tar.gz, inputs/kth.tar and inputs/FMD.zip. Download the pinned DINO weights as inputs/dino.pth and the source archive as inputs/dinov2.zip. The expected SHA-256 values are below or in environment.json. Do not unzip the DINO source before applying the fingerprint check.
- DTD: https://www.robots.ox.ac.uk/~vgg/data/dtd/  · archive https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
- KTH-TIPS2-b: https://www.csc.kth.se/cvap/databases/kth-tips/download.html  · archive https://www.csc.kth.se/cvap/databases/kth-tips/kth-tips2-b_col_200x200.tar
- FMD: https://people.csail.mit.edu/lavanya/fmd.html  · archive https://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip
- Lytro 2D pinhole views: https://visualcomp.ucsd.edu/projects/LF/papers/ECCV16/dataset.html · Wang et al., ECCV 2016, https://people.csail.mit.edu/junyanz/projects/lfmr/eccv16_lfmr.pdf
- DINOv2 weights: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth  · code https://codeload.github.com/facebookresearch/dinov2/zip/e1277af2ba9496fbadf7aec6eba56e8d882d1e35

```bash
mkdir -p work inputs measured
cp cache/*_design.npz cache/*_original.npz measured/
g++ -std=c++17 -O3 -shared -fPIC code/spatial.cpp -o work/colour.so
python code/check_palettes.py work/colour.so
python code/check_spatial.py work/colour.so
python code/unpack_photos.py DTD inputs/dtd.tar.gz measured inputs/DTD
python code/unpack_photos.py KTH inputs/kth.tar measured inputs/KTH
python code/unpack_photos.py FMD inputs/FMD.zip measured inputs/FMD
python code/fetch_target.py inputs/Lytro
for dataset in DTD KTH FMD Lytro; do
  python code/extract_palette.py "$dataset" "inputs/$dataset" measured work/colour.so
  python code/spatial.py extract "$dataset" "inputs/$dataset" measured work/colour.so
done
for dataset in FMD Lytro; do
  python code/extract_external.py "$dataset" "inputs/$dataset" measured inputs/dinov2.zip inputs/dino.pth "work/$dataset" --threads 3
done
for dataset in DTD KTH Lytro; do
  python code/fit.py "$dataset" measured recomputed
  python code/spatial.py fit "$dataset" measured recomputed
done
python code/analyse.py measured recomputed
python code/spatial.py summary measured recomputed
```

The addendum holds the two existing wavelet blocks fixed on DTD and KTH to isolate the new colour measurement. For a full fresh extraction of those two historical blocks, the v5 reviewer bundle supplies check_colour_pixels.py; manifest.py can also reconstruct the design files from that bundle. FMD and Lytro wavelets and transformer features were re-extracted for the present experiment. In each source split, wavelet and DINO scaling parameters use only that split's fitting observations. The root-histogram colour vectors use unit norm and are not standardised across photos.

## Scope of inference
The ten FMD material names are shared with the Lytro dataset. Fur and sky are outside the common label set. Each Lytro capture contributes one 2D pinhole view. This is whole-image cross-acquisition transfer; the original Wang et al. segmented 4D patch benchmark has a different task and unit. The source classes are balanced and the target categories each contain 100 images. Intervals condition on fitted models. DTD inference groups repeated photograph occurrences and exact duplicates using the manuscript's multiplier; KTH resamples four entire specimens within each material; Lytro resamples complete capture groups within each material. The grouped sensitivity combines five successive IDs per category; physical object or scene identities are unavailable. Source/test hashes exclude exact overlap at the resized-image and raw-file levels in cross-dataset fitting.

Verify file integrity after extraction with `sha256sum -c SHA256SUMS` on systems providing this utility.

The source archive checksums (SHA-256) are:

```text
254475ddb37bf0fe1383ff9ea61eb71042db33af1f9a9aa8747391a72135b8f8  FMD.zip
b2f70055f916af92be3761d868fb386944ec4e1473b71ebe39765d69ad358c1f  kth.tar
e42855a52a4950a3b59612834602aa253914755c95b0cff9ead6d07395f8e205  dtd.tar.gz
```

After computing all descriptors under measured/, `python code/verify_pixels.py cache measured pixel_check.json` compares every new palette, every local-pair block, every re-extracted external wavelet and transformer coordinate, and all their IDs and pixel fingerprints. The second complete extraction in the documented environment is recorded in results/pixel_replay_verified.json.

For the additional source/target overlap diagnostic, run `python code/screen_overlap.py inputs/FMD inputs/Lytro measured recomputed`. Its two files (overlap_summary.json and overlap_candidates.csv) use the original FMD screen: perceptual hash Hamming distance at most six and mean squared RGB difference at most 256 on 16 × 16 thumbnails, allowing horizontal reflection. This diagnostic does not change the fitted models.

The 23,552 historical test appearances (18,800 DTD + 4,752 KTH) reproduce all RGB reference labels at equal block weights and fixed λ = 1. The larger replay count includes decisions from several models on the same examples.
