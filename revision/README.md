# GeoVAE synthetic revision reproducibility package

This package reproduces the newly specified synthetic experiments. It includes
the exact 5,000-case protocol used for the revision, not the original confidential
reserve-model meshes. Numerical conclusions must be taken from completed,
verified formal runs. A CPU smoke run is a functional check only.

## Installation

The reference formal environment uses Python 3.10. The CPU integration is also
checked with Python 3.12. Use an isolated environment and install a PyTorch build
suited to the intended device, then install the remaining dependencies:

```text
python -m pip install -r requirements.txt
python main.py doctor
```

`requirements.txt` accepts an already installed compatible PyTorch build. Formal
training requires CUDA and the full GSTools/MPSlib dependencies. The MPSlib native
executable must work locally before starting; the runner never downloads it or
installs anything. The reference compute environment uses an RTX 4060 Ti 16 GB.
Keep NumPy below 2 because the tested scikit-mps 0.6.0 uses the removed `np.NaN` alias.

The commands are the same in Windows CMD, PowerShell and a WSL conda environment.
To use an existing WSL environment, activate it first with `conda activate wsl_vae`.
No user-specific drive or Linux home path is embedded in the public entry point.
The formal CUDA experiments are executed in WSL/Linux. Windows is verified for
the CPU integration draft. Native Windows formal GPU/MPS execution and complete
external-process termination at the budget deadline are not validated. Use WSL
for the formal workflow when the process-tree budget must be enforced.

Evaluation uses an explicit strict deterministic-backend policy, configured
before CUDA initialization. Its helper hash, actual flags and stage proofs are
recorded separately from the unchanged training recipe. A same-device two-step
GPU latent-only/joint-adaptation repeat probe produced exactly matching arrays
under this policy; this is a narrow numerical check, not completion of the full
CUDA experiment. Training itself is not claimed to be bitwise deterministic.
Prediction cache identities include the actual numerical policy, without thread
counts, process IDs or phase names. Old pre-policy result directories must not be
mixed with new runs. The recipe remains `config/revision_v1.yaml`; v2 names the
new evaluation archive, not a changed architecture or training objective.

## Start with a CPU functional run

```text
python main.py --prepare --smoke --out runs/cpu_smoke
python main.py --all --smoke --out runs/cpu_smoke
```

This uses 20 independent synthetic grids at 16 cubed resolution, 14/3/3 cases,
one epoch, one seed and small networks. It checks four model families, two training
ablations, eight methods, draft validation threshold selection, saved test
predictions, local-variability outputs and checkpoint restoration. IK and MPS are
excluded from this small CPU draft. It does not run or imitate formal 50-parent
hyperparameter tuning. `smoke_status.json` is always labelled as draft evidence.
Run training in a fresh process: the smoke helper records unchanged training
flags, trains all six networks and checks restoration before enabling strict
evaluation. It never silently resets backend flags after an evaluation.

## Formal CUDA reproduction

First download the Baidu data using [the root download guide](../docs/DATA_ACCESS.md). Run from this directory. Reserve adequate disk space for checkpoints and prediction
arrays (the preflight requires at least 24 GiB free):

```text
python main.py --all --out runs/esi_revision_v2
```

The complete pipeline uses GeoVAE and basic VAE latent dimension 256, 32 cubed
grids, seeds 11/23/37, microbatch 2, effective batch 16, up to 200 epochs and
validation early stopping. A saved 72-hour experiment budget controls `--all`.
Completed matching stages resume; changed data, code or settings require a new
output directory. Reproduction duration depends on the hardware and native MPS
runtime. Reaching the budget preserves completed checkpoints and does not imply
that the formal result is complete.

The equivalent modules, all using the same output directory, are:

```text
python main.py --prepare --out runs/esi_revision_v2
python main.py --train --out runs/esi_revision_v2
python main.py --tune --out runs/esi_revision_v2
python main.py --validate --out runs/esi_revision_v2
python main.py --ablation-train --out runs/esi_revision_v2
python main.py --ablation-validate --out runs/esi_revision_v2
python main.py --test --out runs/esi_revision_v2
python main.py --ablation-test --out runs/esi_revision_v2
python main.py --uncertainty --out runs/esi_revision_v2
python main.py --robustness --out runs/esi_revision_v2
python main.py --finalize --out runs/esi_revision_v2
```

Before `--test`, both main and ablation validation settings must be frozen. The
formal all-run overlaps CPU classical prediction with GPU ablation training;
separate modules may perform that work sequentially. `--pretrain` aliases main
training; it is not a separate pretraining objective. Example filtered training:

```text
python main.py --train --model geovae --seed 11 --out runs/esi_revision_v2
```

Missing prerequisites cause an explicit error rather than hidden training.
No arguments print help. Further arguments for a particular module are documented
by its underlying `repro_run_all.py --help` or `revision_runner.py --help`.
Public `--finalize` verifies the synthetic results and writes `public_handoff.json`.
After `--all` completes, generate the independent synthetic quantitative figures:

```text
python revision_figures.py --run runs/esi_revision_v2 --out runs/esi_revision_v2/figures
```

The figure gate requires the matching public handoff, completed controller status,
formal 750-case analysis, and unchanged protocol and engine hashes. These figures
support synthetic reproduction only; neither this command nor its metadata mark the
manuscript ready or audit the confidential real cohort. For a modular run, resume
`--all` after `--finalize` so the completed controller status is recorded.
The author-side `revision_integrity.py` and `revision_tables.py` are also supplied
for transparency. Their complete-submission audit/table workflow requires the
restricted author archive and intentionally refuses incomplete public-only runs.
It is not invoked by the public default pipeline and does not mark absent real
data as verified.

## Data and measurement boundaries

`protocol/manifest.json` is included. Obtain the exact 5,000 uint8 arrays separately using the author-designated Baidu link, or run `python revision/restore_benchmark.py` from the outer repository root to reconstruct them from the frozen parameters. Remote file identity is not independently verified; strict manifest checks remain mandatory. Arrays belong in `protocol/grids/` before a formal run and stay outside the uploaded Git files.
They contain 1,000 procedural parents with five variants each, split by parent
into 700/150/150 parents and 3,500/750/750 cases. The fixed domain is normalized.
The generator is a stylized geometric family, not a physical geology simulator.
No real mesh contributes its shape or extent to this independent protocol.

All methods use identical target-independent observed and hidden drill masks.
Primary observations contain eight holes with 12 samples each. Test truth is used
for scoring, not prediction or parameter selection. Classical methods are cached
once per matching case/mask/settings; their copies across model seeds do not count
as independent repeats. CIs resample generator parents, not voxels or model runs.

HD, HD95 and symmetric unsquared Chamfer distances use voxel units. Empty-surface
distances are undefined and are reported with their failure counts. Local latent
perturbations describe local variability, not an established conditional posterior.
Zero accepted candidates remain zero-acceptance cases, not zero uncertainty.

## Restricted real data

The confidential real meshes, coordinates, predictions, sections, real-case LoRA
weights and original raw-data audit inventories are not distributed. Public
preparation records `not_available_restricted`; it does not fabricate a passed
audit. `--all` skips the real-data stage and cannot reproduce that cohort.

Only a user with authorized external input files can explicitly invoke:

```text
python main.py --real-cases --data-root EXTERNAL_DATA_DIRECTORY --private-out ABSOLUTE_PRIVATE_DIRECTORY --out runs/esi_revision_v2
```

Both paths must be supplied explicitly. Private outputs must be outside this
repository and any submission directory. Only allow-listed aggregate metrics are
exported to the public run directory. Shape-level real data are never copied here.

## Optional external synthetic porphyry case

`porphyry/` contains the reviewed source bridge for one independently
obtained synthetic porphyry field. Read `porphyry/README.md` before use. Its
third-party CSV, license file, prepared arrays, weights and predictions are absent.
The source-bound registry records a completed author-checkpoint ten-method
production parity check and CPU recomputation of 150 saved author outcomes.
These are bounded software checks. They do not establish public from-scratch
training or an independently executed public full-150 evaluation. A new normal
public evaluation requires a separately trained and finalized primary run and
externally obtained licensed data; it retains all five levels and three masks.

## Verification and export identity

Download the required data before running the full suite below. The strict input-presence tests are retained. For code-only checks from the repository root, use `python run_tests.py --scope layout`.

```text
python run_tests.py
```

This entry runs the main and optional-bridge CPU tests in separate fresh
processes. The bridge tests verify that Torch has not been imported; the main
tests exercise Torch numerical policies. Combining both domains in one pytest
process violates those test preconditions. The CPU runner sets the declared
workspace policy before imports and hides CUDA, so CUDA and opt-in native MPS
tests are reported as skipped. These are fixture-based functional checks, not
paper experiments. Native MPS integration is opt-in
(`GEOVAE_MPS_INTEGRATION=1`). Training engine files are byte-identical to the
revision engine. `EXPORT_PROVENANCE.json` records the hashes and the small public
controller changes. This preserves checkpoint-source compatibility without
pretending that development and public controller identities are identical.

Runs, checkpoints, caches and private data are ignored by Git. No repository
license has been invented; see [the repository license status](../LICENSING.md) before choosing release terms.
