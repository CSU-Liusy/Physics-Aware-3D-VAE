# Saved-scalar and synthetic-showcase supplements

These separate portable entry points cover two pieces of the revised supplement
that are additional to the main training/evaluation pipeline. They do not train,
select checkpoints, start inference, alter the750case primary test, or audit
confidential real data.

## All20 secondary cross-arm contrasts

From the outer repository root:

```text
python revision/supplemental/secondary_contrasts.py --verify-only
python revision/supplemental/secondary_contrasts.py --out runs/secondary_reanalysis
```

The inputs are the distributed `revision/evidence/source_data` scalar CSVs and
the frozen `revision/protocol/manifest.json`. Four contrasts times five metrics
are retained, with three fitted-model seeds and five variants averaged within
each of150generating parents. The original pure computation is reused from its
byte-identical reference copy:2000parent bootstrap draws, seed20261025. This
analysis was specified after inspecting point estimates. Intervals are pointwise
95%percentile intervals conditional on the three fits, without multiplicity
adjustment, significance/equivalence or general architecture-superiority claims.

The entry checks the fixed index, manifest and every required scalar input hash;
rejects missing/duplicate sample/seed/method pairings and nonfinite values; and
compares all20estimates plus all seed/parent differences against the saved
distributed outputs. Its explicit comparison tolerances are relative1e-12 and
absolute1e-15; the actual maximum differences are printed. `--verify-only` writes
nothing. `--out` refuses an existing destination or a destination inside the
distributed `revision` sources. No grid, probability, checkpoint or private
case-level array is read. This is saved-scalar reanalysis, not an independent
authentication of the original raw prediction provenance or new science.

## Four complex-example figures from a new completed public run

```text
python revision/supplemental/complex_examples.py --run runs/esi_revision_v2 --out runs/complex_examples --check-only
python revision/supplemental/complex_examples.py --run runs/esi_revision_v2 --out runs/complex_examples
```

The primary run must already have completed the public synthetic pipeline and
its verified `public_handoff.json`. The tool cannot replace it with the author's
restricted run or a CPU smoke run. It checks the public handoff and original
reference selection, then reuses only the3frozen targets and their30existing
seed11saved method predictions from that public run. Cases, planes, masks and
method-specific validation thresholds are fixed; all ten methods stay visible.
It adds no prediction or independent test case. The highlighted auxiliary lobe
and disconnected voxel example do not establish geological branching, fault
recovery or external generalization.

The archived evidence loader and four-figure renderer are retained unchanged in
`reference/render_complex_synthetic.py`. The portable wrapper adapts only the
save seam to the current public plotting implementation and an explicit fresh
output path. It writes PDF, SVG and PNG plus provenance. The original author
alignment/font/collision tools are optional external tools, not bundled here.
Without them, the public exporter records **NOT AUDITABLE**, not a QA pass.
Regardless of those optional tools, this portable entry does not certify the
final embedded manuscript PDF or publication layout. It is mechanically tested
with guards/fixtures; a newly trained public full750run and this wrapper's full
four-figure rendering have not been executed as part of the repository layout
change. The existing author's manuscript figures retain their separate original
generation and QA evidence.

`reference/SOURCE_PROVENANCE.json` identifies the exact archived helpers and
selection record. Their original author-only CLI paths and full-archive gates
are historical transparency references; run the portable commands above rather
than those archived CLIs. No author clock/budget ledger, private geometry or
third-party geometallurgical input/derived geometry is bundled.
