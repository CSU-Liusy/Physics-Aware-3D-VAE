# Physics-Aware-3D-VAE

Open-source release package for the 3D VAE workflow used in this repository.

This folder is a public-facing copy of `3dvae/` with the same functionality, English comments/docstrings, and a standalone usage guide.

## What Is Included

- 3D VAE training and inference pipeline (`main.py`, `trainer.py`, `model.py`)
- Mining dataset loading, sparse drill-hole simulation, and cache management (`dataset.py`)
- Result export and visualization utilities (`output_result.py`, `showresult.py`)
- Figure generation scripts for comparison and paper plots (`plot_comparison.py`, `plot_dataset_overview.py`)
- Benchmark and ablation scripts (`comparisons/`)

## Environment

Recommended Python: `3.9+`

Core dependencies:

- `torch`
- `numpy`
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

Example install command:

```bash
pip install torch numpy scipy pandas matplotlib seaborn scikit-image scikit-learn tqdm pyyaml imageio ezdxf
```

## Data And Paths

This code keeps the original path conventions from the project:

- Input data: `../data/`
- Output artifacts: `../results/`

Typical mining dataset locations:

- `../data/mining_ply/`
- `../data/mining_ply_pretrain/`

Important: run commands from inside `Physics-Aware-3D-VAE/` so relative paths resolve correctly.

## Quick Start

```bash
cd Physics-Aware-3D-VAE
python main.py
```

This starts the default training flow using `config/default.yaml`.

## Main Entry Modes

`main.py` supports these modes:

- `train`: train the VAE model
- `benchmark`: run baseline comparisons

Examples:

```bash
python main.py --mode train
python main.py --mode benchmark --checkpoint ../results/3dvae/<run>/checkpoints/best_model.pth
```

## Figure Generation

Generate the end-to-end figure suite:

```bash
python plot_dataset_overview.py
```

Generate comparison matrix only:

```bash
python plot_comparison.py
```

## File Structure

```text
Physics-Aware-3D-VAE/
  comparisons/
  config/
  config_loader.py
  dataset.py
  main.py
  model.py
  model_factory.py
  trainer.py
  output_result.py
  plot_comparison.py
  plot_dataset_overview.py
  README.md
```

## Related Work: VoxelOreGen

The virtual prior dataset provided in this repository originates from our companion project, **VoxelOreGen**, a physics-driven generative pipeline for 3D orebody benchmark data generation. VoxelOreGen couples an Advection-Diffusion-Reaction (ADR) physical simulation engine with a conditional Wasserstein GAN (cWGAN-GP) to produce large-scale, physically grounded 3D orebody voxel datasets. The workflow synthesizes high-throughput 3D tensors by solving coupled fluid-flow, heat-transport, and reactive-mass-transfer equations, embedding metallogenic constraints directly into the training data.

For details on the physics engine, generative architecture, quantitative evaluation, and the complete benchmark dataset, see the VoxelOreGen repository:

> [https://github.com/CSU-Liusy/VoxelOreGen](https://github.com/CSU-Liusy/VoxelOreGen)

## Download

The 5,000 synthesized virtual geological datasets used for the prior learning phase are publicly available via Baidu Netdisk:

> **Download link:** [https://pan.baidu.com/s/1yfQY1he1xiwN17aZ6kE0jw?pwd=gq2p](https://pan.baidu.com/s/1yfQY1he1xiwN17aZ6kE0jw?pwd=gq2p)  
> **Extraction Code:** `gq2p`
