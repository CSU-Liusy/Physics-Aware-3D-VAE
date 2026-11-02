# GeoVAE source data for the revised supplement

This directory contains the formally adopted saved-scalar source-data supplement.
It is included in the local revision package; no external deposit or upload is
claimed. `SOURCE_DATA_INDEX.json` lists all 46 data files with SHA256, row counts,
origins, transformations, denominators and undefined-value policies. Together
with this README, `LICENSE_AND_SCOPE.md` and `SOURCE_INPUT_HASHES.json`, the
directory contains 50 files. All 46 data payloads preserve the reviewed candidate
bytes. The 824 recorded input hashes are preserved provenance, not a new raw-source
audit. The one-time completed formal integrity report is referenced by its hash.

The index `file` entries resolve relative to this directory. `logical_sources`
and paths in `SOURCE_INPUT_HASHES.json` identify records in the author archive;
they are provenance identifiers, not promises that those archive paths are
included or externally accessible. Main text and SI sections should cite the
corresponding `files/` group and the index.

## File groups and statistical units

| Group | Files | Scope and unit |
|---|---:|---|
| primary | 6 | 750 synthetic volumes from 150 held-out parents; three learned-model fits; classical copies counted once in marginal summaries |
| ablation | 17 | The same 750 volumes and 150 parents; reused basic/full-model fits are not new training runs |
| secondary | 3 | All 20 post-results paired training-arm contrasts; pointwise intervals conditional on three fits |
| robustness | 5 | 30 selected parents, seed 11, ten dependent retention/mask/noise conditions and ten methods; 3,000 scalar case-condition-method records |
| diagnostic | 7 | Ten validation and 30 test parents, three optimized starts, three proposal variances and three filter thresholds; candidate pools are reused across thresholds |
| thresholds | 1 | Frozen validation-selection curves, not sealed-test performance |
| training | 2 | 18 distinct fits, checkpoint identities and recorded stage times, without checkpoint tensors |
| real_aggregate | 2 | Only anonymous descriptive aggregates for 113 eligible of 122 related interpreted segments from one mining area; nine excluded |
| external150 | 3 | 150 saved-only scalar outcomes and all 50 method-level summaries from one external synthetic parent, five nested Cu cutoffs and three masks |

Synthetic overlap is calculated per volume and averaged within generating parent
before equally weighting parents. Training seeds, variants, optimization starts,
mask repeats and external Cu cutoffs are not independent geological examples.
Reliability-bin voxel counts are descriptive counts. Empty-surface distances stay
undefined with their valid denominators. Zero accepted candidates have no ensemble
mean; fewer than two have no sample variance. Local filtered candidates do not
establish a conditional posterior, calibrated total uncertainty or a drilling
decision benefit. All fixed settings and unfavorable outcomes are retained.

## Reproduction and confidentiality limits

The separate code package supplies the procedural generator, frozen 5,000-volume
manifest and workflow for a new synthetic run. Arrays are obtained separately or
reconstructed from the recorded parameters; their hashes must match the manifest. This source-data directory
does not include trained weights, probability/occupancy arrays, meshes, coordinates,
private case-level predictions or real-case adapters. Its hashes and scalar
records support result inspection and provenance; they do not provide all inputs
needed to regenerate the completed author predictions from scratch.

The only real-data files are `files/real_aggregate/real_summary.json` and
`files/real_aggregate/real_aggregate_metrics.csv`. They cannot independently
reproduce the private retrospective evaluation. The real coordinate units remain
unverified; outcomes use voxel units and do not establish independent-deposit
inference or raw-assay field validation.

The external150 files are saved-only derived scalars from one synthetic parent
under separately recorded upstream CC BY-NC-SA 4.0 terms. No upstream block model,
drillhole/assay file, original or derived geometry, prepared occupancy/observation
array, weight or prediction array is included. Read `LICENSE_AND_SCOPE.md` before
redistribution. Ten author-checkpoint production-parity predictions and CPU
recomputation of 150 saved author outcomes are bounded software evidence; neither
is a public from-scratch training run or an independent public full-150 execution.

The formal primary snapshot has five CSV tables plus `analysis.json`; no extra
primary CSV is implied. The native GENESIM executable version was not recorded.
The recorded scikit-mps 0.6.0 version is the Python package version, not a claim
about the native executable. Final external publication locations and author-held
release declarations remain outside this local adoption step.
