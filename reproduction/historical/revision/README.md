# Controlled nonlinear comparisons, revision 2

These archived numerical experiments are dated 19 November 2026. Macro recall is in percent, and differences are percentage points. S, W and C denote achromatic wavelet rank dependence (F7), opponent-wavelet scatter/scale/radial index (F8) and a global RGB histogram. Run the commands below from `reproduction/historical/`. Install the pinned dependencies listed in the repository README. For image-based checks these include PyTorch and pytorch-wavelets. This numerical release does not grant new rights to third-party photographs or software.

## Check results without downloading images

~~~sh
python check.py
python revision/check_revision.py
python revision/code/inference.py . revision/results/selection output/statistics
python revision/code/degree_summary.py . revision/results/degrees revision/results/selection
python revision/code/additive_summary.py . revision/results/selection revision/results/additive_grid
python revision/code/reference_intervals.py . revision/results/references
~~~

The last three commands regenerate the corresponding small JSON files. To preserve an untouched reference, run them against a working copy or your new outputs. The DTD intervals condition on the observed classes and fitted functions. The archive includes complete validation grids, deterministic tie resolution and image indices. Selection archives store class scores in float32. Diagnostics designed after observing prior results are identified in the article. The 2015–2017 observations were acquired after setting the eligibility rules and freezing the source procedures. The revision protocol has no external registration.

## Refit the classifiers

Build the DTD index and cached feature blocks using the top-level README. Below, imageset contains index.npz and the images subdirectory, downloads contains the verified archives and model checkpoint, and cache contains the feature blocks.

~~~sh
python revision/code/nested.py DTD . cache output/selection --threads 3
python revision/code/nested.py Flowers . cache output/selection --threads 3
python revision/code/nested.py KTH . cache output/selection --threads 3
python revision/code/inference.py . output/selection output/statistics
python revision/code/degree_two.py . cache output/degrees --threads 2
python revision/code/degree_summary.py . output/degrees output/selection
python revision/code/additive_grid.py DTD . cache output/additive_grid --threads 3
python revision/code/additive_summary.py . output/selection output/additive_grid
~~~

DTD_01 to DTD_10 use the official splits. KTH_a to KTH_d each withhold one physical specimen per class. The Flowers.npz targets are ordered Pl@ntNet and then iNaturalist 2026. Image identities and true labels are in external/data. The 150-configuration additive sensitivity analysis includes simplex vertices. The square-only and cross-product-only models each have 24 configurations. The six-model comparison specification is in revision/code/protocol.json.

## Recompute the numerical and image controls

~~~sh
python revision/code/numerics.py . imageset output/numeric_features --threads 2
python revision/code/ablations.py . cache output/numeric_features output/ablations --threads 3
python revision/code/dino_field.py . imageset downloads output/dino_full --threads 2
python revision/code/references.py prepare . cache imageset/images output/local_features output/dino_full output/references
python revision/code/references.py score . cache imageset/images output/local_features output/dino_full output/references
python revision/code/reference_intervals.py . output/references
python revision/code/benchmark.py . cache imageset/images output/cpu_benchmark.json
~~~

Run the large phases sequentially on memory-limited machines. Cell grouping changes the floating-point summation order slightly. The grouped half-precision reference reproduced all 18,800 original DTD decisions. Precision, scatter and tie interventions are interpreted relative to this same grouped reference. The diagonal variant diagonalizes the normalized scatter before its matrix log and preserves the full-covariance scale and radii. The magnitude variant re-estimates scatter, scale and radial index. In numeric.json the tie axes are grid (4×4, 2×2, full), statistic and coordinate. The five rows are redundant values, observations participating in a tie, constant columns, total observations and total columns. In radii the axes are scale, grid and statistic. The six entries count regions, radii, sub-floor radii, near-flat denominators, lower grid endpoints and upper grid endpoints. The supplement ships the resulting decisions, numerical counts and indices. Large intermediate feature blocks can be regenerated.

## New flower observations and human-labelled masks

The new cohort has 420 distinct iNaturalist accounts. The query snapshot, exclusion log, merged prior observation identities and previous exact/perceptual hashes are in revision/results/cohort. To use the original API snapshot, extract its JSON files into an empty cohort-pages directory. Media changes or unavailability are recorded by the selection code. No replacement from a changed provider is required to check the stored decisions.

~~~sh
mkdir -p output/cohort output/cohort-pages
python -m zipfile -e revision/results/cohort/instantane_inaturalist.zip output/cohort-pages
cp revision/results/cohort/observations_anterieures.csv revision/results/cohort/empreintes_anterieures.npz output/cohort/
python revision/code/cohort/collect.py tout --cache output/cohort-pages --photos output/flower-images --donnees output/cohort
python revision/code/flowers_new.py . output/cohort output/flower-images output/selection/Flowers.json
~~~

For a prediction-only replay, run flowers_new.py against a working copy of the shipped cohort directory. If features.npz is present, its image order is verified and the program proceeds directly to the source classifiers. The two preceding flower protocols and snapshots are in revision/provenance.

Oxford hosts 17flowers.tgz and trimaps.tgz at https://www.robots.ox.ac.uk/~vgg/data/flowers/17/. Place them in oxford17. Their SHA-256 identifiers are in revision/results/masks/scores.json. The file verified.csv contains the 183 examined trimaps and the 15 exclusions. Trimap codes for the three examined classes are 1 (foreground), 3 (background) and 0 (uncertain). The full photograph, flower only, background only and silhouette use the same classifiers.

~~~sh
mkdir -p output/masks
cp revision/results/masks/source_hashes.csv output/masks/
python revision/code/masks.py . oxford17 output/selection/Flowers.json output/masks
~~~

To reconstruct earlier external descriptors, external_images.py accepts a TF image root containing the class subdirectories, or a KTH root containing KTH-TIPS2-b. For the other two sources it uses archived media URLs. Bytes and decoded pixels are hash-verified. If a file is unavailable or has changed, its failure is recorded without substituting a different image. Previously distributed feature blocks remain usable for refitting the earlier experiments.

~~~sh
python revision/code/external_images.py tf . tf-flowers-image-root output/tf-check
python revision/code/external_images.py kth . kth-image-root output/kth-check
python revision/code/external_images.py plantnet . plantnet-image-cache output/plantnet-check
python revision/code/external_images.py nouveau_2026 . inat2026-image-cache output/inat2026-check
~~~

## Rebuild the numerical figures

The following commands generate the pipeline diagram and the empirical comparison figure from the archived numerical inputs. The manuscript-prose renderer is not included in this code-and-data distribution. The numerical tables remain available as CSV and JSON files and through the repository's numerical replay entry point.

~~~sh
mkdir -p output/figures
python revision/code/pipeline_figure.py output/figures/pipeline.pdf
python revision/code/selection_figure.py revision/results output/figures/selected_contrasts.pdf
~~~
