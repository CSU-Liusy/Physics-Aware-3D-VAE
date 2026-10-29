# Historical mesh-training workflow

This directory preserves the original repository's mesh-based training and general result-export workflow. The default repository entry point uses the revised procedural benchmark in `revision/`; this legacy workflow is separate and does **not** produce the revised paper's experimental evidence.

No original datasets, generated meshes, checkpoints or results are included here. Supply only external inputs you are authorized to process. Legacy inputs, trusted checkpoints, outputs and caches must all remain outside this public repository, including paths supplied through YAML. The confidential122-case cohort is not distributed or authorized for release.

## Dependencies and help

Use the repository's supported Python environment and install the legacy extras when needed:

```text
python -m pip install -r legacy/requirements.txt
python main.py legacy --help
```

Help does not import Torch or visualization libraries. Normal execution reads `legacy/config/default.yaml` before applying CLI overrides. The original configuration parser did not actually read its claimed default file; the parser here corrects that interface defect. Consequently, do not treat a new default legacy run as an exact replay of undocumented historical command-line defaults.

## Complete training and export

```text
python main.py legacy --mode train --config legacy/config/default.yaml --ply-dir EXTERNAL_MESH_DIRECTORY --output-dir EXTERNAL_OUTPUT_DIRECTORY
```

The original training closure is retained: `dataset.py` reads meshes and builds sparse observations; `model.py` supplies standard/octree VAE and LoRA layers; `model_factory.py` constructs the model/optimizers; `trainer.py` performs training, saves weights/logs and evaluates; `output_result.py` exports reconstructions, sections and reports; `showresult.py` provides optional spin rendering.

`--model-type standard` and `--model-type octree` choose historical architectures. `--use-lora` and `--lora-preset` retain the historical training adapter options; they are distinct from the revised benchmark's case-specific test-time adaptation. Factory resource/speed recommendations are unvalidated legacy heuristics, not measured paper results.

For an explicitly selected single mesh, use `--ply-file EXTERNAL_MESH_FILE`; it now selects that file only, rather than every neighboring mesh. `--generate-only --ply-file EXTERNAL_MESH_FILE --num-samples NUMBER` retains the historical simulated-observation export. Output geometries and coordinates remain in the chosen external output directory.

Warm-start a trusted legacy checkpoint with:

```text
python main.py legacy --mode train --ply-dir EXTERNAL_MESH_DIRECTORY --output-dir EXTERNAL_NEW_OUTPUT_DIRECTORY --checkpoint EXTERNAL_TRUSTED_LEGACY_CHECKPOINT
```

This is the original weight warm start, not restoration of optimizer/scaler/RNG state. Only saved architecture fields are applied; a checkpoint cannot redirect input or output paths. A revision-protocol checkpoint is rejected here. Checkpoints can contain serialized objects, so supply only files you trust. The revised workflow implements its own stricter, source-bound resume mechanism.

Caches are stored below the chosen external output, in a namespace bound to the input file contents and voxel dimensions. `--force-regen-cache` only removes matching voxel-cache filenames after validating that namespace's identity. It never clears the input directory, an input-parent cache folder or unrelated NPY files.

## Other preserved operations

```text
python main.py legacy --mode benchmark --ply-dir EXTERNAL_MESH_DIRECTORY --checkpoint EXTERNAL_TRUSTED_LEGACY_CHECKPOINT --output-dir EXTERNAL_BENCHMARK_OUTPUT
python main.py legacy --mode ablation --ply-dir EXTERNAL_MESH_DIRECTORY --output-dir EXTERNAL_NEW_ABLATION_OUTPUT --epochs 50 --patience 10
python legacy/comparisons/train_unet.py --ply-dir EXTERNAL_MESH_DIRECTORY --output-dir EXTERNAL_UNET_OUTPUT
python legacy/showresult.py --ply-file EXTERNAL_PLY_FILE_OR_DIRECTORY --out-dir EXTERNAL_VISUALIZATION_OUTPUT
python legacy/comparisons/plot_ablation_summary.py --results-dir EXTERNAL_ABLATION_OUTPUT --output-dir EXTERNAL_PLOT_OUTPUT
```

The legacy benchmark requires an explicit checkpoint containing architecture `hparams` and loads all model state keys strictly. It does not guess a1024-dimensional octree architecture or search previous results for weights. `--unet-checkpoint` is optional at the unified legacy entry point; an omitted U-Net is reported as not evaluated. The four-arm ablation driver retains its original arm definitions, writes each training arm directly to its specified output and refuses to overwrite an existing arm. It does not delete or move the newest result directory.

Benchmark mode also honors `--ply-file` as an exact single-mesh selection. Its `benchmark_execution.json` records failed cases and the number of successful evaluations; a run with no successful evaluations fails. The standalone U-Net helper writes `unet_execution.json` and rejects empty or all-failed training epochs instead of saving an artificial zero-loss best model. Failed batches remain in the record, and any partial-epoch mean uses successful batches only. Ablation plotting creates a new external output directory when needed.

`robust_eval` is explicitly unsupported because the original repository did not contain the referenced `run_robust_analysis.py`. Use the documented `revision/` robustness workflow for the revised experiment protocol. No missing algorithm or result has been invented.

## Scientific scope and exclusions

Legacy data splitting is at the simulated-sample level and the old trainer chooses its saved best model by training BCE. These are not the revised benchmark's generating-parent partitions and validation-frozen selection. The legacy benchmark reports whole-grid overlap; its historical Chamfer is a squared bidirectional sum and it uses a999empty-distance penalty. The revised evaluation instead reports explicit blind-region metrics, an unsquared surface Chamfer and undefined-distance coverage. Outputs from these two workflows must not be merged or compared as though their definitions matched.

The old paper-specific `plot_comparison.py` and `plot_dataset_overview.py` are intentionally omitted. They were not needed by the training/export closure; they could select/show real geometries, automatically train U-Net while plotting and retain unsupported uncertainty/prospectivity wording. Use the audited `revision/revision_figures.py` and the supplementary figure helpers described in the revision documentation. The old README's unverified ADR/VoxelOreGen/data-lineage claims and download link are not imported into this release.

The accompanying lightweight interface tests check configuration, path protection, import closure, owned-cache behavior and a fake checkpoint-loader interface. They do not establish a new legacy training run, GPU parity or historical-score reproduction. See `MIGRATION.md` and `ORIGINAL_SOURCE_MAP.json` for the source-to-current changes.

Additional execution verification: a newly created toy cube mesh was used for one tiny CPU epoch and a separate trusted-checkpoint warm start. Both exited successfully and the toy input was unchanged. An initial warm-start logger ordering failure was preserved, corrected and rerun. Final visualization path routing was checked with a renderer fixture. These checks establish software functionality only; they do not reproduce historical scores or test the full GPU training/export workload. New layout checks and their scope are recorded in the outer `VERIFICATION.json`.
