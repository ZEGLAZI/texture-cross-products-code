# DTD ridge sensitivity: reconstruction of the v3 analysis

This directory supplies the analysis missing from the earlier reviewer code
archive. `replay.py` was reconstructed from the protocol and numerical results
in the supplied v3 manuscript and supplement. It reads the original released
feature arrays, independently recalculates the validation curves and outer
predictions, and then compares the results with the recorded reference values. The
recorded reference numbers in `published_v3.json` are used only by the final audit.
They are not inputs to selection, fitting, or uncertainty calculations.

This reconstruction does not recover the original dense-study source code or
the authors' original timestamp. Its own validation freeze records the time of
this repeat calculation. It must not be described as a new blind experiment.

## Running the study

Use the Python environment described in the package-root `requirements.txt`.
Run these commands from the corrected package root. The supplied
`additional_features/` directory already contains the companion descriptors:

```sh
python studies/ridge_sensitivity/replay.py --mode full --threads 2 --report ridge.json
python studies/ridge_sensitivity/replay.py --mode tables --report ridge.json
```

A standalone numerical figure and a cell-by-cell numerical export can be
produced without additional fitting:

```sh
python studies/ridge_sensitivity/plot_validation.py
python studies/ridge_sensitivity/export_checks.py
```

The publication trace distinguishes the full kernel with dashes as well as
colour and records its source-data fingerprint. Keeping this renderer separate
allows typography changes without invalidating the numerical selection freeze.

`full` repeats validation, fitting, the split-6 extraction control, table
recalculation, and audit even when saved results already exist. `tables`
recalculates the summaries and intervals from
the saved validation curves and test predictions; it does not claim to refit
the models. Both commands return a nonzero exit code if a reference numerical
value or another required check does not agree.

The intermediate stages can also be resumed:

```sh
python studies/ridge_sensitivity/replay.py --mode select --force --threads 2
python studies/ridge_sensitivity/replay.py --mode fit --threads 2 --report ridge.json
```

`select` completes all 100 validation choices before writing
`results/choices_frozen.json` and its SHA-256 fingerprint. `fit` verifies that
fingerprint before starting an outer fit. Complete per-representation stages
are reused only when their recorded input/protocol fingerprints agree. For
`full`, reuse additionally requires the explicit `--resume` option. A new
output directory may be chosen with `--output PATH`. Use `--force` to repeat
completed stages. Do not run concurrent writers in the same output directory.

The historical numerical files are read from `reproduction/`. Its original
SHA-256 manifest is retained alongside an explicit public-variant change record.
The unused manuscript-prose renderer is excluded; documentation and the integrity
wrapper are adapted and checked under their new hashes. All other retained
original files are checked against their historical hashes. The original ZIP
digest is a provenance identifier, not a freshly measured archive digest.
The public replay requires no manuscript or supplement PDF.

## Inputs and exact protocol

| Code | Representation | Released arrays |
|---|---|---|
| H | Reference colour–texture | `pixel_check/DTD_colour.npz` |
| R | Supervised ResNet-50 | `studies/resnet50/data/DTD_stages.npz` |
| DR | Original DINO ResNet-50 | `additional_features/dino_r50/DTD_stages.npz` |
| DV | Original DINO ViT-S/16 | `additional_features/dino_vits16/DTD_stages.npz` |
| D | DINOv2 ViT-S/14 | `reproduction/cross_representation/features/DTD_features.npz` |

Image order and all ten official DTD splits come from the r4 identities.
Train/test and fit/validation exclusions are reconstructed using both exact
source-file and decoded-pixel hashes, then checked against the archived split
indices. Statistics for feature centring, standardisation, clipping, and block
normalisation are estimated only on the appropriate fitting sample. For the
colour–texture representation, the square-root RGB histogram is retained
without a second coordinate-standardisation step.

For each kernel and split, the initial grid is `lambda = 10**(j/4)` for every
integer `j` from −12 through 8. A maximum at an endpoint triggers an extension
of four candidates on that side, with limits `j = -32` and `j = 24`. Scores
within `1e-13` in macro-recall fraction count as tied. A dense-grid tie is
resolved by the smallest absolute log penalty and then the smaller penalty.
The earlier six-candidate rule retains its original order
`[1, 0.1, 10, 0.01, 100, 0.001]` and first-maximum tie rule.

All curves and choices are saved before outer fitting. The outer fitting
sample is the duplicate-cleaned train–validation pool. Each fitted kernel
produces decisions under five settings: common penalties 0.1, 1, and 10;
six-candidate selection; and dense selection. Across five representations,
two kernels, ten splits, and 1,880 test appearances per split, this is 940,000
saved labels.

The common-1 and six-candidate labels are compared with 376,000 archived
decisions. Every dense outer fit is also checked using a separate positive
definite linear solver, covering 188,000 dense labels. The main computation
uses an eigendecomposition and the check uses a Cholesky-based solve.

Uncertainty uses the unchanged released function
`summarize_pixel_check.paired_interval`: 100,000 draws and seed 19112026.
For DTD it aggregates centred integer influences by absolute magnitude and
draws their sums with binomial variables. This has exactly the distribution
of independent Rademacher signs per exact-image group, and preserves the
Monte Carlo ordering used by the earlier release. Each S39 common reference
retains its stated family correction (eight, eight, and three); these are not
one simultaneous family across all three common penalties.

## Outputs and audit coverage

All calculated outputs are written under `results/` by default.

| Output | Content |
|---|---|
| `input_manifest.json` | Protocol, script hash, and source-array archive hashes |
| `selection_MODEL.json` | Curves, selected penalties, ties, extensions, and spectra summaries |
| `choices_frozen.json` / `.sha256` | The complete validation freeze before outer fitting |
| `DTD_MODEL_predictions.npz` | Test identities and all ten decision arrays for one representation |
| `fit_MODEL.json` | Per-split counts, archived comparisons, solver comparisons, and prediction hashes |
| `validation_curves.csv` | Every searched value, including extensions |
| `selected_penalties.csv` | S38 and selection bookkeeping |
| `gains.csv` | All five settings, including the five-representation comparison |
| `contrasts.csv` | S39 contrasts and conditional intervals |
| `effective_degrees.csv` | S40 effective degrees of freedom and extension counts |
| `extraction_control.json` | S41 recomputation from the original r4 and reference wavelets |
| `examples.csv` | Deterministic S42 example selection |
| `ridge_validation.pdf` / `.png` | Figure S2 recomputed from the validation curves |
| `ridge_validation_publication.pdf` / `.png` / `.json` | Publication rendering, including dashed full-kernel curves and provenance |
| `summary.json` | Combined calculated summaries |
| `audit_v3.json` | The final numerical comparison, including every mismatch if any |
| `published_table_values.json` / `.csv` | Unrounded calculated values and exact display precision for numerical reference comparisons |

The numerical audit checks 232 newly recorded reference values in five tables:
Table 4 (35), S38 (100), S39 (54), S40 (35), and S41 (8). It also checks the
selection totals, S42 identifiers and decisions, split-6 decision changes,
and archived and independent-solver agreement. A printed value passes only
when the independently calculated result lies within its stated rounding
precision. The final report records actual run outcomes; this README does not
assert a successful result before that report exists.

This is a refit from archived descriptors. It does not rerun frozen image
encoders from image pixels or establish any new external validation. The
root package's historical audit covers the earlier experiments separately.

## Prior verification record (25 November 2026)

The supplied results were recalculated, not filled from the expected-value
file. All 232 numerical values in the five new tables agree at the recorded display
precision. The run evaluated 2,144 validation values, reproduced all 100 dense
choices, and saved 940,000 test predictions. All 376,000 comparisons with the
earlier common-1 and six-candidate predictions agreed. The independent
positive-definite solver agreed on all 188,000 dense test predictions.
The split-6 extraction control and the four deterministic examples also
agreed. These outcomes are recorded in `results/audit_v3_full_refit.json`.

The full refit used the original numerical ZIP. Its exact executed source is
preserved as `results/replay_executed_2026_11_25.py`, matching the implementation
digest in the original input manifest and validation freeze. Afterwards, the
input reader was adapted to the unpacked public payload and `full` was made
unconditional by default. The numerical functions were checked to be
unchanged. A table replay was then executed with the numerical ZIP absent
from the test root: all 796 source files verified and all 232 values again
agreed. This separate portability check is recorded in
`results/backend_adaptation.json` and `results/unpacked_tables_report.json`;
it is not presented as a second full refit.

## Corrected code/data distribution

The prior verification records above describe the earlier complete author
package. The corrected distribution excludes the article and supplement PDFs,
their sources and manuscript-prose generation. Its public-variant integrity
check explicitly records the omitted/adapted historical files. Current commands
verify numerical results only; they do not check printed publication documents.
