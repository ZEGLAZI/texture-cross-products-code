# DTD conventional and frozen-encoder baselines

The DTD reference experiments in Tables 7 and 9 and Supplement S7 can be rebuilt from the public DTD release, Oxford's published DeCAF activations, the original DINOv2 checkpoint and the supplied historical colour–texture descriptors. Start from the reviewer package with r4 unpacked as `reproduction`. The input directory below contains `dtd.tar.gz`, `labels.tar.gz`, `decaf.tar.gz`, `dinov2.zip` and `dino.pth`. Download locations and checksums are provided in `reproduction/historical/dtd/protocole.json`. The pixel stage also requires the CPU PyTorch installation in the main README and a C++ compiler with OpenMP for the LBP operators.

Run from the package root in a new working directory tree. The copy command provides the historical three-block inputs for the mixed representations.

    python reproduction/historical/dtd/preparer.py dtd_inputs dtd_prepared
    python -c "import shutil; shutil.copytree('reproduction/review2/dtd_cache/hsmt', 'dtd_cache/hsmt')"
    python reproduction/historical/dtd/extraire.py clbp dtd_prepared dtd_inputs dtd_cache --threads 2
    python reproduction/historical/dtd/extraire.py dino dtd_prepared dtd_inputs dtd_cache --threads 2
    python reproduction/historical/revision/code/references.py prepare reproduction/historical dtd_cache dtd_prepared/images local_dtd_features dino_full dtd_references
    python reproduction/historical/revision/code/dino_field.py reproduction/historical dtd_prepared dtd_inputs dino_full --threads 2
    python reproduction/historical/revision/code/references.py score reproduction/historical dtd_cache dtd_prepared/images local_dtd_features dino_full dtd_references
    python reproduction/historical/dtd/predire.py dtd_prepared dtd_cache dtd_fixed --threads 2
    python studies/dtd_references/compare.py reproduction dtd_fixed dtd_references --report dtd_comparison.json

The final command checks all assignments and class predictions exactly and compares all 420 inner reference scores and the resulting summaries. The earlier fixed set includes 12 classifiers. The second comparison uses seven representations under fixed and separately selected penalties, including the centre-cropped and full-field DINOv2 readouts.
