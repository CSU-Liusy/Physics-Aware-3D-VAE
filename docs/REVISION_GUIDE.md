# Guide to the revision files

The full inner tree was moved together so relative imports, configuration paths,
checkpoint-engine identities and release attestations remain compatible.

| Task | Files under revision/ |
| --- | --- |
| CLI and stage orchestration | main.py, repro_run_all.py, repro_cli.py, revision_runner.py |
| Models and supervised training | model.py, repro_pipeline.py, repro_cgan.py, comparisons/models/unet.py |
| Procedural benchmark and sampling | experiment_data.py, protocol/manifest.json, downloaded protocol/grids/, r12_sampling.py |
| Baselines and validation selection | repro_baselines.py, repro_hyper_tuning.py, repro_tuning.py |
| Ablation and main analysis | repro_ablation.py, repro_report.py, revision_analysis.py |
| Adaptation, local candidates and retrospective hooks | revision_diagnostics.py, r12_data.py |
| Retention, repeated masks and label noise | revision_robustness.py |
| Strict evaluation policy | revision_numerics.py |
| Public synthetic finalization | public_finish.py |
| Figures and author-side audits/tables | revision_figures.py, repro_figures.py, revision_integrity.py, revision_tables.py |
| Optional external synthetic case | porphyry/ |
| Paired secondary contrasts and complex synthetic display helpers | supplemental/ |
| Saved scalar paper evidence and per-file source map | evidence/source_data/SOURCE_DATA_INDEX.json |
| Tests, CPU fixture flow and historical verification | tests/, revision_smoke.py, repro_smoke.py, run_tests.py, VERIFICATION.json, EXPORT_PROVENANCE.json |

The complete author-side integrity/table audit intentionally requires restricted
author artifacts. It is supplied for transparency and does not mark absent real
data as audited. The supported public route uses public_finish.py instead.

Use the root main.py from the repository root. The inner README's direct commands
apply only after changing into revision/. For example, after a completed public
run, native synthetic figures can be generated from the root with:

```text
python revision/revision_figures.py --run runs/esi_revision_v2 --out runs/esi_revision_v2/figures
```

For the external synthetic case, use root `main.py external` rather than the
historical direct command. The root launcher supplies/checks `--code-root revision`
and enforces outside-repository input/cache/output paths. One external parent
with 15 dependent level/mask conditions does not add 15 independent deposits.

The legacy implementation remains separate. It is not used to replace adverse
current results, recover an old score trend or select winning test cases.
