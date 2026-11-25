# Executed checks of the code/data distribution

`tables/verification.json` records the successful quick execution of the
corrected package entry point. The historical, ridge and provenance stage
reports are stored beside it. Fitting counts shown by the quick run come from
archived records; the report explicitly says that this invocation did not
refit classifiers.

`full/validation.json` records the complete numerical validation executed in
documented segments after an input-reader correction. It links and fingerprints
`historical_core.json`, `historical_full.json`, `dense_full.json`,
`provenance_after_cleanup.json` and `input_preservation.json`. See the README
inside `full/` for the continuation boundary and the distinction from a single
uninterrupted invocation. The public full command always starts fresh fits.

Raw report paths identify the verification workspace. Large generated fit
outputs are not duplicated here; the root README commands create new local
outputs and reports. Publication documents and full image-encoder extraction
are outside the numerical verification scope.

`release_gate_tests.json` records negative checks using synthetic data,
including rejection of a deleted unreviewed file retained in older history.
`privacy_review.json` and `initial_git_check.json` record independent content
and clean-initial-history checks of the corrected source snapshot before the
final reports were added. Their manifest fingerprints identify that snapshot;
the root SHA256SUMS inventories the final distribution including the reports.
None of these records contains manuscript prose or private Git objects.
