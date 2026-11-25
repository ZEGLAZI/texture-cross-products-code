# Conventional KTH texture descriptors

The manuscript compares the fixed two-product kernel with grayscale CLBP and opponent-colour LBP on the four held-out specimen folds. The raw histogram counts in `temoins.npz` come from the original KTH control experiment. The program checks image identity and order, fits both ridge classifiers and compares all 9,504 decisions with the released per-image record.

    python studies/kth_lbp/check.py reproduction --report kth_lbp.json

To reconstruct all histogram counts first, add `--image-root inputs/kth`, where that directory contains `KTH-TIPS2-b`. This option compiles the original C++17 operators with OpenMP. Each image is checked against its decoded RGB fingerprint. The paper uses macro recall averaged over the four specimen folds.
