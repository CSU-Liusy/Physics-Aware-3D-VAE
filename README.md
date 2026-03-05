# Physics-Aware-3D-VAE

Open-source release package for the 3D VAE workflow used in this repository.

This folder is a public-facing copy of `3dvae/` with the same functionality, English comments/docstrings, and a standalone usage guide.

## What Is Included

- 3D VAE training and inference pipeline (`main.py`, `trainer.py`, `model.py`)
- Mining dataset loading, sparse drill-hole simulation, and cache management (`dataset.py`)
- Result export and visualization utilities (`output_result.py`, `showresult.py`)
- Figure generation scripts for comparison and paper plots (`plot_comparison.py`, `plot_dataset_overview.py`, `paper_plot.py`)
- Robustness analysis and automated workflow scripts (`run_robust_analysis.py`, `run_auto_pipeline.py`)
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
- `ablation`: run ablation study pipeline
- `robust_eval`: run robustness analysis

Examples:

```bash
python main.py --mode train
python main.py --mode benchmark --checkpoint ../results/3dvae/<run>/checkpoints/best_model.pth
python main.py --mode ablation
python main.py --mode robust_eval --checkpoint ../results/3dvae/<run>/checkpoints/best_model.pth
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
  trainer.py
  output_result.py
  plot_comparison.py
  plot_dataset_overview.py
  run_auto_pipeline.py
  run_robust_analysis.py
  README.md
```

## Notes

- This folder is synchronized from `3dvae/` for public release packaging.
- Relative path assumptions were intentionally preserved to avoid behavior changes.
- If you move this folder outside the original repository layout, update data and output paths in config/CLI arguments.
