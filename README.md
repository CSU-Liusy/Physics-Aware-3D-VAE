# Physics-Aware-3D-VAE

Release package for the 3D VAE training and sparse-observation reconstruction workflow used in this repository.

The original general mesh-training workflow is retained in `legacy/`. The revised GeoVAE training, evaluation, ablation and reviewer-reproduction files are collected in `revision/`, with a common `main.py` launcher. The historical repository name is retained; the revised benchmark evaluates a learned shape prior on procedural geometric shapes.

## What Is Included

- 3D VAE training and inference pipelines (`revision/` and `legacy/`)
- Dataset loading, sparse drill-hole simulation and cache management
- Result export and visualization utilities (`legacy/output_result.py`, `legacy/showresult.py`)
- Revised paper figures and supplemental analysis (`revision/revision_figures.py`, `revision/supplemental/`)
- Benchmark, ablation and robustness scripts, tests and saved scalar evidence for review (`revision/`)

## Environment

Recommended Python: `3.10+`

Core dependencies include:

- `torch`
- `numpy<2`
- `scipy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-image`
- `scikit-learn`
- `tqdm`
- `PyYAML`
- `imageio`
- `ezdxf`
- `trimesh`

Install the appropriate PyTorch build for your CPU/GPU, then install the remaining dependencies and inspect the environment:

```bash
python -m pip install -r requirements.txt
python main.py doctor
```

The full classical comparison also requires a compatible MPSlib GENESIM executable. See [revision/README.md](revision/README.md) for native dependencies and the tested execution environment.

## Data And Paths

- Revised benchmark manifest: `revision/protocol/manifest.json`; download the data from the Baidu link below and place its input arrays in `revision/protocol/grids/` (excluded from Git)
- Revised run outputs: an explicit directory such as `runs/esi_revision_v2/`
- Original mesh workflow: supply your own authorized PLY inputs with `--ply-dir` or `--ply-file`, and an explicit `--output-dir`

Mesh inputs, checkpoints, reconstruction outputs and caches for the original workflow must stay outside the public repository. Input files remain read-only. Confidential real data and third-party restricted geometry are not distributed; the saved real-data evidence contains anonymous aggregates only.

Run the following commands from the repository root, `Physics-Aware-3D-VAE/`, so relative paths resolve correctly. The two workflows have separate models, configurations and checkpoint formats.

## Quick Start

```bash
cd Physics-Aware-3D-VAE
python main.py --help
python main.py --all --smoke --out runs/cpu_smoke
```

The smoke command runs a small synthetic functional check without the 5,000-volume download. Before a complete run or the full test suite, download the Baidu data as described in [Download](#download). Then run:

```bash
python main.py --all --out runs/esi_revision_v2
```

This includes preparation, training, validation-only selection, main and ablation tests, local-candidate diagnostics, robustness and public finalization. Matching saved stages can resume. A smoke check does not reproduce the paper's scientific experiments. Running `python main.py` without arguments prints help and starts no training.

## Main Entry Modes

The root `main.py` supports:

- Default or `revision`: revised GeoVAE training and evaluation
- `legacy`: original mesh-based `train`, `benchmark` and `ablation` modes
- `external`: optional external synthetic-case preparation and evaluation

Examples:

```bash
python main.py --prepare --out runs/esi_revision_v2
python main.py --train --model geovae --seed 11 --out runs/esi_revision_v2
python main.py legacy --mode train --config legacy/config/default.yaml --ply-dir EXTERNAL_PLY_DIRECTORY --output-dir EXTERNAL_OUTPUT_DIRECTORY
python main.py legacy --mode benchmark --ply-dir EXTERNAL_PLY_DIRECTORY --checkpoint EXTERNAL_TRUSTED_CHECKPOINT --output-dir EXTERNAL_BENCHMARK_OUTPUT
python main.py external --help
```

The original `--mode` spelling is also recognized by the root launcher. For complete stage prerequisites and command migration, see [docs/REVISION_GUIDE.md](docs/REVISION_GUIDE.md) and [legacy/README.md](legacy/README.md). Keep historical and revised metrics/results separate.

## Figure Generation

Generate the revised synthetic figure suite from a completed matching public run:

```bash
python revision/revision_figures.py --run runs/esi_revision_v2 --out runs/esi_revision_v2/figures
```

Supplementary contrasts and reference-selected complex examples are grouped in [revision/supplemental/](revision/supplemental/README.md). The old paper-specific plot scripts were replaced by these documented routes; the original training/export utilities remain available in `legacy/`.

## File Structure

```text
Physics-Aware-3D-VAE/
  main.py
  requirements.txt
  run_tests.py
  README.md
  LICENSING.md
  REPOSITORY_MANIFEST.json
  VERIFICATION.json
  docs/
  tests/
  legacy/
    comparisons/
    config/
    config_loader.py
    dataset.py
    model.py
    model_factory.py
    trainer.py
    output_result.py
    showresult.py
  revision/
    main.py
    model.py
    repro_pipeline.py
    protocol/
    comparisons/
    porphyry/
    supplemental/
    evidence/source_data/
    tests/
```

All revision-specific implementations are in `revision/`. The outer `VERIFICATION.json` records checks for the reorganized repository; the inner verification and provenance files retain their original, narrower scope. Run the software tests with `python run_tests.py`. These checks do not claim a new independent full training or external evaluation. See [LICENSING.md](LICENSING.md) for release and reuse permissions.

Current test-fixture changes remain recorded in `revision/test_fixture_changes.json`; production source and release evidence are preserved.

## Related Work: VoxelOreGen

For the related VoxelOreGen project, see:

> [https://github.com/CSU-Liusy/VoxelOreGen](https://github.com/CSU-Liusy/VoxelOreGen)

The revised benchmark distributed here uses procedural geometric shapes. Its provenance is defined by the bundled protocol; it is not asserted to originate from VoxelOreGen or an ADR simulation.

## Download

The 5,000 synthesized virtual geological datasets used for the prior learning phase are available through the author's Baidu Netdisk share:

> **Download link:** [https://pan.baidu.com/s/1yfQY1he1xiwN17aZ6kE0jw?pwd=gq2p](https://pan.baidu.com/s/1yfQY1he1xiwN17aZ6kE0jw?pwd=gq2p)  
> **Extraction Code:** `gq2p`

Data are downloaded separately and are not included in the GitHub code directory. For the revised workflow, place the 5,000 matching `.npy` inputs in `revision/protocol/grids/`, keeping the supplied `revision/protocol/manifest.json` unchanged. Run the preparation check before training:

```bash
python main.py --prepare --out runs/esi_revision_v2
```
