# Complete numerical validation, executed in segments

The code/data distribution was validated on 25 November 2026 using Python
3.12 and the pinned numerical environment. Its public `full` command always
starts fresh model fits. The validation recorded here was completed in
segments after correcting a legacy input-reader path; it is not described as
one uninterrupted invocation of that command.

The historical numerical core was first run in full and its 72 expected
outputs passed. A subsequent CNN loading stage exposed an old archive-only
path. The reader was adapted to the supplied unpacked data; its numerical
algorithms were unchanged. The remaining historical/extended stages then ran
with the previously completed core evidence explicitly identified as reused.
The dense DTD study ran separately in full with fresh selection and fitting.

The composite record identifies these segments and their evidence. Archived
fit counts in the quick `tables` run must not be confused with the newly
calculated fits recorded here. Neither path validates publication PDFs or
claims a fresh extraction of all image encoders from provider images.

`input_preservation.json` records byte-for-byte comparisons of retained
numerical inputs and checks of the unchanged computational implementations.
Raw report paths identify the verification workspace; generated intermediate
fit outputs are not duplicated in this distribution.
