# Historical numerical reproduction
## Cross-block similarity products for texture recognition with colour–texture descriptors and frozen DINOv2 features

Oussama ZEGLAZI and Mohammed RZIZA. This directory contains numerical inputs, frozen decisions, settings and source provenance. It is code-and-data documentation; no manuscript or supplementary article PDF is included. The [repository README](../README.md) gives the current entry point. Paths below start at this `reproduction/` directory.

### Reading and reproducing this version

**Historical numerical release HSMT-PR-2026-11-19-r4.** Dataset editions, provenance and fingerprints are in `reader/dataset_sources.csv`. The numerical inputs support offline refitting. Historical table and section numbers below identify the corresponding numerical analyses; they do not imply that a publication document is distributed here. The original SHA-256 manifest is retained as a reference. Verification explicitly accounts for the removed manuscript-prose renderer and the adapted public-distribution files.

From the extracted archive root, create an isolated Python 3.12 environment. On macOS or Linux use the following commands. On Windows use the Python executable under .venv/Scripts/python.exe after creating the environment.

~~~bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-replay.txt
.venv/bin/python reproduce.py full --output reproduced
~~~

The full run first verifies the SHA-256 manifest. For DTD and KTH-TIPS2-b, it refits the four equal-weight kernels on both representations, recomputes the inner square and full kernel grids and checks the selected test decisions. It regenerates the representation contrast and post-hoc regime change. It also refits the matched-diagonal full kernel in both representations on every DTD and KTH split. It checks the equivalent reciprocal-penalty solution and reproduces the per-split penalty settings and scale-sensitivity tables. It also checks the original colour–texture fits, Gaussian validation grids and test labels, radial interventions, and photograph- and host-level clothing analyses. Each group of checks is listed in `reproduced/replay_report.json`. No original photographs or model weights are needed for this numerical replay. The host deletion check uses the already fitted classifiers.

The lightweight command `python reproduce.py tables --output reproduced_tables` rebuilds the results below, the DTD disagreement counts and the host interval from saved outputs.

| Location | Subject | File under the output directory |
| --- | --- | --- |
| Main Table 2 and S8 | Change after adding products in each representation, and difference of changes | cross_tables/representation_comparison.csv |
| S8 | Post-hoc paired difference between the two penalty regimes | cross_tables/regime_change.json |
| Main Table 2 and S8 | Matched-diagonal recalls, gains and differences | scale_tables/scale_comparison.csv |
| S8 | All selected native-scale penalties, by representation and split | scale_tables/selected_penalties.csv |
| S8 | All native and matched recalls by split | scale_tables/scale_by_split.csv |
| Main Table 3 | Original-profile colour–texture kernels and their two paired effects | paper/aggregate/primary.csv |
| Main Table 4 | Redistributed kernels and Gaussian | paper/aggregate/redistributed_rbf.csv |
| Main Table 5 | Seven clothing classifiers under photograph-level splits | paper/clothing_scores.csv |
| Main Table 7 | Selected two-product model and controls | paper/selected_restricted_scores.csv |
| S9, Table S16 | Covariance and radial interventions | paper/wavelet_interventions.csv |

See `cross_representation/README.md` for the per-image predictions, optional extraction commands, archival source fingerprints and separate inferential families. Earlier logs are preserved as `reader/replay_verification_r2.json` and `reader/replay_verification_r3.json`. The accompanying report for this release is distributed next to the numerical ZIP and identifies its exact SHA-256.

#### Individual commands

All paths below are relative to the extracted root. Activate the Python environment first. The two host commands must run in the order shown. The last command reads the resulting decisions and settings. Standard output and the comparison report indicate the level of each check. Use a fresh destination for direct radial replay to recompute every fold. The full wrapper automatically stages its calculations in a new directory.

~~~bash
python verify.py
python reader/code/paper_tables.py reproduced/paper
python review2/code/decision_agreement.py reproduced/agreement.csv
python review2/code/verify_primary_refits.py reproduced/refits.json
python review2/code/replay.py radial --output reproduced/radial
python reader/code/verify_gaussian.py reproduced/gaussian
python reader/code/cohort_tables.py reproduced/cohort
python reader/code/model_intervals.py reproduced/intervals
python clothing/code/summarize.py --output reproduced/photograph_bootstrap
python reader/code/clothing_hosts.py prepare reproduced/hosts
python reader/code/clothing_hosts.py fit reproduced/hosts
python reader/code/host_inference.py reproduced/hosts reproduced/host_inference
~~~

All 3,840 Gaussian validation cells and the selected Gaussian test labels on 18,800 DTD and 6,330 clothing test appearances are checked against the archived fits. The clothing disposition ledger covers all 720 attempts. The dated annotation manifest and source fingerprints are in reader/tables/clothing_summary.json. The folder reader/hosts contains all 150 host-disjoint assignments and settings, complete validation matrices and 6,330 paired out-of-fold predictions. The folder reader/host_inference provides per-fit image and host counts by class and role, per-class recalls, joint tuning frequencies, each host's influence and whole-host bootstrap diagnostics. Physical products and vendors are not identified across hostnames.

Source images remain with their providers. Re-extraction from original images, recreating masks and repeating visual checks depend on access to those source pixels. The numerical checks above use the archived inputs.

### Compare the results

- review2/results/aggregate/primary.csv — the four nested model recalls, both paired differences and within-corpus family-wise 95% intervals. Q minus S adds the cross-products. Q minus X adds the individual squares. L, S and X in these machine-readable tables include the common linear term.
- review2/results/aggregate/all_controls.csv — all four profiles and three regularization regimes. "fixed" uses the same ridge penalty of one. "tuned" independently selects the ridge on training data at each fixed profile. "unit" applies unit-diagonal normalization and fixes the ridge at one.
- review2/results/aggregate/redistributed_rbf.csv — training-selected results for all seven kernels. RS and RX reassign the quadratic weight to squares or to cross-products. RBF uses the original weighted feature distance, four absolute inverse scales gamma = 0.25, 1, 4, 16 and six penalties. All intervals in this file are pointwise. Historical DTD adjusted and clothing primary intervals in the manuscript use the original simulation seeds and families from historical/ and clothing/, respectively.
- review2/results/aggregate/selected_settings.csv and review2/results/provenance/original_selected_profiles.csv — each selected profile or Gaussian width, the selected ridge and the validation recall. Profile 0 is (1/4,1/4,1/2), 1 is uniform, 2 is (1/2,1/4,1/4) and 3 is (1/4,1/2,1/4).
- review2/results/kernels/ — per-test-image predictions including every profile and regularization regime, inner train/validation indices and full validation grids. Each fold records test indices and ground truth. For flowers the indices address the concatenation of TF source, Pl@ntNet, iNaturalist 2026 and iNaturalist 2015–17, in this order. Clothing source-level decisions at the historical redistributed settings are also provided in the tidy clothing/results/predictions.csv.
- review2/results/radial/ — upstream covariance and radial interventions, per-region-scale counts, per-image lower selections and observed out-of-range counts, class-level/split decisions, and synthetic recovery. The reported intervals here are pointwise.
- review2/results/provenance/ — rates of selecting either native grid endpoint per collection and scale, flower candidate counts, earlier reference fingerprints and selected settings. Filter masks and raw moment ratios for all 355,320 DTD region-scale cells are in review2/radial_cache/. Each NPZ has the image IDs and the 3 × 21 raw ratio, original grid index, diagonal grid index and recalculated diagonal features.
- review2/protocol.txt, review2/amendment.txt and review2/chronology.csv — experimental sequence and frozen contrasts. The chronological CSV uses local file modification times and content hashes. There was no public preregistration. Across all 1,000 FMD photographs, fresh extraction reproduced every stored F7, F8 and RGB coordinate. Two independent kernel solves (FMD and one clothing fold) and Gram identities are recorded in review2/results/provenance/. Previous development use makes the Flickr Material Database evaluation retrospective. The nominated Outex_TC_00031 acquisition has no experimental score in this release.
- review2/figures/ — supplementary vector figures for the fixed DTD attribute changes, alternative centres, selected comparisons and individual class margins. Their numerical inputs and original plotting programs are in historical/ and historical/revision/.
- historical/ — the earlier experiments, executable analyses, figure sources, classes, provenance, and raw prediction files. The earlier kernel names refer to the two-product centre swaps described in the manuscript. historical/README.md explains their mapping. clothing/ contains the source URLs, overlap decisions, crops' pixel hashes, fixed folds and all earlier clothing fits. For descriptive and mask experiments, use the corresponding original subdirectory's README.

The final manuscript also reports paired label disagreements between the validation-selected redistributed square and full quadratic kernels on DTD. The split-specific counts and total are in `review2/results/aggregate/decision_agreement.csv`. This is a descriptive comparison of previously saved predictions and includes repeated test appearances across splits. Reproduce it with

```bash
python review2/code/decision_agreement.py reproduced/decision_agreement.csv
```

The script aligns test image indices, checks both prediction vectors against the separate historical runs, and counts which classifier was correct on each disagreement.

### Run or verify offline

1. With Python 3.12, install historical/requirements-analysis.txt. These packages suffice for the main kernel refits and aggregate replay. Install PyTorch and pytorch-wavelets only for new pixel extraction. The figures use Matplotlib, and source images remain subject to their providers' licences.
2. Run python verify.py to verify the public numerical distribution against the original SHA-256 manifest and the explicitly recorded public-distribution changes.
3. Run python review2/code/replay.py aggregate --output reproduced/summary to reproduce the score tables and paired intervals using the archived decisions. Run python review2/code/replay.py details --output reproduced/provenance for the supplementary settings and cohort tables.
4. Run python review2/code/replay.py refit --corpus DTD --output reproduced/folds to refit the classifiers from cached descriptors without downloads. Other options are Flower, KTH, Textile and FMD. Use --corpus all to fit the full set sequentially. Then run python review2/code/replay.py aggregate --decisions reproduced/folds --output reproduced/refit_summary. The solver is deterministic for a fixed numerical environment. Float summation and tied decisions may depend on the BLAS implementation.
5. Run python review2/code/replay.py radial --output reproduced/radial. This recomputes the radial and covariance fits from the cached wavelet/radius measurements. The original grouped half-precision wavelet features are stored as exact bit-level differences against the DTD descriptor cache to avoid duplicating the same arrays. The replay restores the arrays before fitting. Use simulate for the finite-sample radial simulations, figures to regenerate the two manuscript figures and the two main result tables for the fixed and redistributed comparisons, and radial-from-pixels --pixels PATH for fresh DTD radial extraction (DTD root containing images/ and index.npz).
6. Run python review2/code/fmd_pixels.py FMD.zip review2/extra/fmd.npz review2/extra/fmd.csv reproduced/fmd_pixels.json to check the retrospective FMD descriptor cache against provider pixels. The inclusion and split can also be regenerated with review2/code/fingerprints.py IMAGE_ROOT dtd_signatures.npz followed by review2/code/fmd_prepare.py FMD.zip review2/extra/fmd.npz dtd_signatures.npz CLOTHING_CROPS clothing/data/sources.csv fmd_rebuilt.npz. Clothing and previous external-image checks have their own source manifests and programs.
7. If the provider archive is obtained, python review2/code/outex.py Outex_TC_00031.tar.gz reproduced/outex applies the recorded 68-class protocol. It requires the official images and lists. The conditional program is included for the open acquisition only.

### Independent refits of the fixed comparisons

Run the following command to recompute the four fixed DTD and clothing classifiers from the archived descriptors. It evaluates the full quadratic as an explicit sum of the square and cross-product terms. The original implementation evaluates the squared weighted sum. Checks compare predictions by test appearance and provide a JSON count for each split and method.

    python review2/code/verify_primary_refits.py reproduced/primary_refits.json

The run reported with the final manuscript is in review2/results/provenance/primary_refits.json.

### Flower enrolment

The list historical/revision/results/cohort/candidats.csv records all 942 candidates in deterministic order. The class filter and the supported licences are in historical/revision/code/cohort/collect.py. The literal ordering input is HSMT-new-observations-2015-2017-v1|TAXON|OBSERVATION_ID, encoded in UTF-8 and hashed with SHA-256. The ordering salt is separate from the perceptual fingerprint. Images are converted to grayscale, resized to 32 × 32 with Lanczos and transformed with an orthonormal type-II DCT. The top-left 8 × 8 coefficients are thresholded at the median of the 63 non-DC coefficients. All 64 bits are packed. The process also uses horizontal reflections. Hamming distance ≤6 or a matching exact byte/decoded-RGB hash triggers duplicate exclusion. The 9,211 earlier fingerprint records need not represent distinct people. Earlier iNaturalist observation, photo and account IDs are in observations_anterieures.csv; provider-specific manifests are in historical/external/data/ and historical/revision/provenance/. A complete cross-platform identity map is unavailable. The raw provider snapshot and every disposition are preserved.
