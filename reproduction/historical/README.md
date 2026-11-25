# Quadratic interactions between structure, opponent-wavelet statistics and colour for texture recognition

Supplementary data and software for the article by Oussama ZEGLAZI and Mohammed RZIZA.

## Revision 2, 19 November 2026

Start with revision/README.md for validation-selected kernels, square-only and cross-only controls, new flower observations, human-labelled masks, descriptor ablations, encoder controls and measured CPU cost. The archive-wide SHA256SUMS covers both versions of the experiments. Run python revision/check_revision.py to verify the revised results.

## Reproduce the original fixed-setting results

Run the commands from the extracted root, using Python 3.12. The decision-level analysis and the external refits use NumPy, SciPy, threadpoolctl, Matplotlib and Pillow. Versions used here appear in requirements-analysis.txt. The DTD split and estimator definitions are supplied with the code. Directory names in the commands are relative to the extracted root. Each output directory can be created from scratch.

~~~sh
python -m pip install -r requirements-analysis.txt
python check.py
python dtd/analyser.py dtd/results output/dtd
python external/flower_intervals.py external/data/fleurs.csv output/flowers
python external/centre_intervals.py external/results/pivots_archives.npz output/centres
python external/deletion_intervals.py
python external/source_transfer.py external/data
python external/refit.py --pipeline dtd --data external/data --reference external/results/pivots_archives.npz --output output/external
python extra/weight_analysis.py . output/weight_intervals.json
python publication/displays.py . output/manuscript
~~~

The first check compares scores against individual predictions, verifies the duplicate handling and case margins, and checks the kernel expansions using explicit tensor products. The DTD analyst reconstructs scores for 12 original methods and all 11 contrasts. The four source-transfer refits check both the original additive model and the proposed model photograph by photograph. The second external refit calculates both alternative centres, the two individual product deletions, the common-divisor additive model and the uniformly weighted three-product model. The figures are deterministic vector PDFs. The tables in the LaTeX source are computed from these same inputs. Table 1 specifies the experimental design and is typeset directly in the manuscript.

## Recompute DTD from images

The archived decisions can be checked independently of downloads. For raw-image extraction, a CPU build of PyTorch 2.5.1, PyWavelets 1.8.0, pytorch-wavelets 1.3.0 and setuptools 75.8.0 were used. A C++17 compiler with OpenMP is needed for the local binary patterns. In a separate environment:

~~~sh
python -m pip install -r requirements-analysis.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install PyWavelets==1.8.0 pytorch-wavelets==1.3.0 setuptools==75.8.0
python dtd/telecharger.py downloads
python dtd/preparer.py downloads imageset
python dtd/check_extractor.py imageset/images
python dtd/extraire.py hsmt imageset downloads cache --threads 2
python dtd/extraire.py clbp imageset downloads cache
python dtd/extraire.py dino imageset downloads cache --threads 4
python dtd/predire.py imageset cache output/raw --threads 3
python dtd/analyser.py output/raw output/raw_statistics
python dtd/verifier_partition.py cache output/raw
python extra/oclbp.py --pipeline dtd --images imageset/images --operator dtd/reference/operateurs.cpp --results dtd/results --output output/oclbp
python extra/dense_control.py --pipeline dtd --results dtd/results --cache cache --output output/dense
python extra/cases.py --pipeline dtd --results dtd/results --cache cache --output output/cases
python extra/weight_reversals.py --pipeline dtd --results dtd/results --cache cache --data external/data --output output/reversals
~~~

Oxford distributes the DTD images, label lists and DeCAF activations. The downloader verifies SHA-256 hashes for those archives and the DINOv2 checkpoint, and obtains the DINOv2 source at the pinned commit in dtd/protocole.json. The 48 reference images exercise every block and each scale over six batches. The public two-block implementation reproduces all 69,552 wavelet coordinates in the full-extractor check at their stored half precision. CPU linear algebra backends and PyTorch releases can differ in floating-point results and in rank ties. For exact comparison, inspect the per-image decisions and reference coordinates.

## Contents and conventions

| Location | Contents |
| --- | --- |
| dtd/results/identites.npz | Ordered DTD filenames, true classes, exact-image groups, image hashes and thirty official split arrays. |
| dtd/results/decisions.npz | Per-method labels for ten by 1,880 official test appearances, together with training counts. |
| dtd/results/contrastes.csv | Paired differences, intervals, favourable folds, corrections and regressions. Positive values favour the first model. |
| external/data | Source/target wavelet and palette summaries, flower decisions, credit-group index, target provenance manifests and material specimen identities. Class codes 10, 12 and 9 mean daisy, dandelion and sunflower. |
| external/results | Cohort results, material reference summaries, and measured and refitted centre and product comparisons. |
| extra | Predictions and code for the OCLBP, all-pairs, singleton, six weight assignments, selected margin examples, and paired weight reversal checks. |
| publication | Script for the four vector figures and five numerical tables. |
| SHA256SUMS | Digests of the distributed files. |

The experiment keys in the prediction archives are HSMT (the proposed W centre), Pivot_F7 (structure centre), Pivot_RGB (colour centre), Additif_fixe (common divisor 13/4), Additif_historique (divisor 2) and Additif_diag_egal (matched diagonal). Interactions are products of similarities between the same two images. Some historical column names and comments are in French. Additif or additif in the original four-collection flower table denotes the original divisor 2. The all-pairs and weight-reversal studies were performed after the primary contrasts were established. None of these alternatives changes the submitted method.

Dataset sources are listed in the article. The Pl@ntNet and iNaturalist 2026 confirmation manifests contain public observation or media IDs, source URLs, taxon labels, licence metadata and content hashes. Original photos remain with their respective providers. The earlier iNaturalist image IDs are preserved alongside their features and decisions. TF source identities and encoded photographer groups are listed in tf_manifest.csv. The material dataset has 44 specimens. Each specimen remains intact during resampling. No third-party photographs or pretrained model weights are redistributed here.
