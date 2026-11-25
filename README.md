# Texture recognition with cross-products of block similarities

Research code, derived descriptors, predictions and numerical reproducibility
materials by **Oussama ZEGLAZI and Mohammed RZIZA**.

This distribution contains the computational materials only. The unpublished
article, supplementary document, their LaTeX projects, old draft versions and
internal submission audits are kept separately. Six one-page numerical plot
PDFs are included as data visualizations; none is an article or supplement.

## Start here

Use a checkout of the corrected code repository or extract the complete clean
archive into a **new, empty directory**. The archive already contains
`additional_features/`; no second archive is needed. Do not overlay it on an
older working directory, because extraction does not delete unwanted files.

Python **3.12** is the validated interpreter. Install the pinned numerical and
plotting dependencies from this directory:

```sh
python -m pip install -r requirements.txt
```

First verify the supplied files and the reviewed document inventory. This
check uses only the Python standard library:

```sh
python check_release.py
```

Recalculate numerical summaries and consistency checks from the stored
decisions:

```sh
python replay_paa.py --mode tables
```

Repeat the supported classifier fits from the supplied descriptors, then
recalculate their summaries:

```sh
python replay_paa.py --mode full
```

Both commands return a nonzero exit code on failure. Each invocation creates a
fresh directory under `runs/` with its stage logs and JSON verification report.
The report identifies whether that invocation actually performed classifier
refits. Full mode starts the dense DTD analysis from a fresh output directory;
it does not silently reuse its archived fitted predictions. The standalone
dense-study runner has an explicit `--resume` option for interrupted work.

The default numerical settings use two BLAS threads and one worker for the
source-refit stage. Full mode requires substantially more time than table mode.
Increase `--jobs` only when enough memory is available.

## What the commands verify

The offline chain covers the original kernel comparisons, paired contrasts,
block subsets, colour controls, source and capture resampling, supervised
ResNet features, original DINO and DINOv2 comparisons, garment controls and
the dense DTD ridge study. Results are rebuilt from the released descriptors
or stored per-example decisions, according to the selected mode.

The dense DTD analysis checks its numerical reference values, validation
choices, archived-decision agreement, a second linear solver, extraction
control and deterministic examples. Expected values are terminal comparison
targets; they are not inputs to fitting, penalty selection or intervals.
Provenance checks retain the source records and deterministic example rules.

This public numerical replay does not open or validate article PDFs. It does
not claim that printed tables, captions or prose were checked during these
commands. Those editorial checks belong to the authors' separate submission
materials. A numerical pass also does not claim a new extraction of every
image encoder from source pixels or a rerun of the original machine timings.

The optional pixel-reconstruction programs are retained with their individual
study instructions. They need separately obtained provider images, masks,
model weights and extra dependencies. They are not prerequisites for the
offline commands above.

Executed validation reports are retained under `verification/`. The full
validation record identifies its separate execution segments and the correction
of an old archive-only input reader; it does not claim a single uninterrupted
run. The released full command itself always performs fresh fits.

## Directory guide

| Path | Contents |
|---|---|
| `replay_paa.py` | Complete code/data replay entry point |
| `replay_submission.py` | Historical and extended numerical stages |
| `check_release.py` | Distribution hashes, reviewed document/archive inventory and optional Git checks |
| `reproduction/` | Historical numerical inputs, original code, decisions and explicit public-distribution adaptations |
| `pixel_check/` | Reference wavelet/RGB descriptors and shared-transformer controls |
| `additional_features/` | Original DINO stage descriptors and their own manifest |
| `studies/resnet50/` | Supervised ResNet descriptors, refits and source/target resampling |
| `studies/matched_pretraining/` | Original DINO comparisons and catalogue product evidence |
| `studies/ridge_sensitivity/` | Reconstructed dense DTD analysis, recorded decisions and provenance |
| `verification/` | Recorded executions and release-gate checks of this corrected distribution |
| `SHA256SUMS` | Hashes of the files in this corrected distribution |
| `runs/` | Locally generated output; excluded from version control and distribution |

## Historical provenance and public adaptation

The original numerical release is identified by archive SHA-256:

```text
ad1a3c2cda703498ca2bca2e2f5fa6718ad071931a0d5dbd8869c4f46dfbf385
```

That original ZIP is not included or needed. Its original `reproduction/SHA256SUMS`
is retained as historical provenance. The public variant removes an unused
script that generated manuscript prose and adapts the associated documentation
and integrity wrapper. The public verification records the exceptions
explicitly and checks the retained original bytes and adapted files against
their respective hashes. It does not claim that the complete original archive
or every historical file is present unchanged.

The missing dense-study implementation was reconstructed from the documented
protocol and reference results, then executed and independently checked. The
reconstruction and its own validation freeze are identified in the study
README and saved records. This is not a recovery of an unavailable original
script and is not presented as a newly blinded experiment.

Some unchanged historical documentation retains older study names and table
identifiers. The commands in this root README are the current package-level
entry points. Generated numeric LaTeX table fragments are computational
outputs; the article's LaTeX source is not distributed.

## Data provenance and licences

The package supplies derived descriptors, identities, splits and predictions
for offline reproduction. The study documentation identifies the DTD,
KTH-TIPS2-b, FMD, Lytro, ALOT, flower, clothing and DeepFashion-MultiModal inputs
needed for optional reconstruction from pixels. Provider photographs,
annotations and model weights remain subject to their own terms and
availability. Included third-party licence files are preserved; this package
does not assign a new blanket licence to those materials.

Three small provider-provenance ZIP snapshots are retained with their original
bytes. They contain 51 JSON responses in total, with readable copies in the
adjacent `*_contents/` directories. They contain no paper or source project.

## Before a GitHub upload

Keep publication and submission documents outside this directory. The
`.gitignore` rules exclude them, and `check_release.py` rejects unexpected
files even if Git would ignore them. After an intentional code change, review
the change and regenerate the distribution manifest before making a release.

Use a new empty repository when moving from an older repository that included
the manuscript. Do not copy its `.git` directory or merge its old history.
Deleting a file in a later commit does not remove it from earlier commits.
After staging and committing this reviewed distribution in a fresh repository,
run:

```sh
python check_release.py --check-git
```

For this first upload, this option requires exactly one new root commit, no
parent history, and identical files in the working directory, Git index and
committed tree. It rejects old or shallow history. For later development,
the plain `python check_release.py` remains the distribution check; the
`--check-git` option is specifically for a fresh initial repository. Neither
command changes repository visibility or publishes anything.
