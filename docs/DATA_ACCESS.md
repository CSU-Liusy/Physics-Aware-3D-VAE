# Data download and placement

The author confirms the original Baidu Netdisk share as the download location for the 5,000 datasets:

> **Download link:** [https://pan.baidu.com/s/1yfQY1he1xiwN17aZ6kE0jw?pwd=gq2p](https://pan.baidu.com/s/1yfQY1he1xiwN17aZ6kE0jw?pwd=gq2p)  
> **Extraction Code:** `gq2p`

Upload this repository directory directly to GitHub. Dataset files are downloaded separately; no new ZIP distribution is required by this code release.

## Revised workflow

Place the matching input files in the following layout, relative to the repository root:

```text
revision/protocol/
  manifest.json
  grids/
    syn-p000000-v0.npy
    ...
```

The complete file list, split assignments and file hashes are in the supplied `manifest.json`. The runtime expects 5,000 uint8 32x32x32 arrays: 1,000 parents with five variants, split into 3,500/750/750 cases and 700/150/150 parents. Retain the checked-in manifest (SHA256 `48ec0924b76a1cca26715bb77566646dacb5fe9a3252471f8e4690aae8e01030`).

From the repository root:

```text
python main.py --prepare --out runs/esi_revision_v2
python main.py --all --out runs/esi_revision_v2
```

Preparation validates the files against the frozen protocol. A missing or mismatched input is an error; changing filenames or replacing the manifest does not reproduce the revised experiments. Mesh inputs for the separate original workflow use `main.py legacy` as documented in `legacy/README.md`.

`revision/protocol/grids/` is ignored by Git. Full `python run_tests.py` requires these inputs. Help, `--dry-run`, `--smoke` and `python run_tests.py --scope layout` do not require them. A small smoke run is not a replacement for the published benchmark.

## Exact reconstruction when matching downloads are unavailable

From the outer repository root:

```text
python revision/restore_benchmark.py
python main.py --prepare --out runs/esi_revision_v2
```

The optional first command reconstructs the 5,000 frozen procedural inputs from
their recorded parameters. It checks the manifest and archived generator SHA256,
then checks both occupancy bytes and serialized NPY bytes before installing each
missing file. Existing differing files cause failure and are never replaced. The
manifest, split and scientific results remain unchanged. This does not convert
old PLY meshes into the revised benchmark.

`--check-only` writes no input files and returns exit code 2 for missing but exactly
replayable inputs; `--case-id syn-p000000-v0` selects one manifest case for a small
software check. Omit case selection for the full benchmark. Completed matching
files can be reused after an interruption. The recorded formal environment is
Python 3.10.20 with NumPy 1.26.4; any byte mismatch stops with an error instead of
weakening identity checks. This is data preparation, not a new model experiment.

## Link provenance

The location above is supplied and confirmed by the author. Its remote contents were not independently downloaded and compared during this repository edit; local file validation remains mandatory. No scientific input, score or comparison was changed by restoring the link. See `DATA_ACCESS.json` for the machine-readable distribution record.
