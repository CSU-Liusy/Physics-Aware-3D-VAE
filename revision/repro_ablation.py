"""Matched-budget ablations with arm-specific immutable configurations."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from experiment_data import validate_manifest
from repro_pipeline import (evaluate, load_config, save_json, sha256_file,
                            source_hash, stable_hash, train)
from repro_report import _read_run, report
from repro_tuning import select_thresholds

ARMS = {
    "base_no_observed": ("base_vae", {"lambda_observed": 0.0}),
    "base_observed": ("base_vae", {}),
    "attention_no_position": ("geovae", {"use_transformer": True,
                                        "use_relative_position": False}),
    "attention_position": ("geovae", {"use_transformer": True,
                                     "use_relative_position": True}),
}


def prepare_configs(base_config_path: str, out_dir: str) -> dict[str, Path]:
    base = load_config(base_config_path)
    if float(base["lambda_observed"]) <= 0:
        raise ValueError("Base configuration must have a positive observed loss")
    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = {}
    for arm, (_, overrides) in ARMS.items():
        config = dict(base)
        config.update(overrides)
        path = output / "configs" / f"{arm}.json"
        save_json(path, config)
        configs[arm] = path
    save_json(output / "ablation_design.json", {
        "base_config": str(Path(base_config_path).resolve()),
        "arms": {arm: {"model_kind": kind, "overrides": overrides}
                 for arm, (kind, overrides) in ARMS.items()},
        "contrasts": [
            ["base_no_observed", "base_observed", "observed loss"],
            ["base_observed", "attention_no_position", "plain bottleneck attention"],
            ["attention_no_position", "attention_position", "relative position bias"],
            ["attention_position/geovae", "attention_position/geovae_latent",
             "latent TTLO"],
            ["attention_position/geovae_latent",
             "attention_position/geovae_joint", "decoder LoRA"],
        ],
        "selection_rule": "Choose adaptation hyperparameters on validation; run sealed test once.",
    })
    return configs


def train_arms(manifest: str, base_config_path: str, out_dir: str,
               seeds: list[int], max_epochs: int | None = None) -> None:
    configs = prepare_configs(base_config_path, out_dir)
    for seed in seeds:
        for arm, (kind, _) in ARMS.items():
            run_dir = Path(out_dir) / "models" / arm / f"seed{seed}"
            if (run_dir / "best.pt").exists():
                raise RuntimeError(f"Existing checkpoint would be overwritten: {run_dir}")
            train(manifest, str(configs[arm]), str(run_dir), kind, seed,
                  max_epochs=max_epochs)


def evaluate_arms(manifest: str, base_config_path: str, out_dir: str,
                  seeds: list[int], split: str = "val", limit: int = 0) -> list[str]:
    if split not in ("val", "test"):
        raise ValueError("Ablations must use val or test split")
    configs = prepare_configs(base_config_path, out_dir)
    threshold_files = {}
    if split == "test":
        for arm, (kind, _) in ARMS.items():
            validation_dirs = [
                Path(out_dir) / "evaluations" / "val" / arm / f"seed{seed}"
                for seed in seeds
            ]
            if not all(path.is_dir() for path in validation_dirs):
                raise RuntimeError(f"Validation ablation results missing for {arm}")
            selected_path = Path(out_dir) / "thresholds" / f"{arm}.json"
            select_thresholds([str(path) for path in validation_dirs],
                              str(configs[arm]), str(selected_path),
                              draft=limit > 0, allow_subset=True)
            threshold_files[arm] = str(selected_path)
    directories = []
    for seed in seeds:
        for arm, (kind, _) in ARMS.items():
            checkpoint = Path(out_dir) / "models" / arm / f"seed{seed}" / "best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing ablation checkpoint: {checkpoint}")
            methods = ([kind, "geovae_latent", "geovae_joint"]
                       if arm == "attention_position" else [kind])
            directory = Path(out_dir) / "evaluations" / split / arm / f"seed{seed}"
            evaluate(manifest, str(configs[arm]), str(directory), split,
                     methods, {kind: str(checkpoint)}, seed, limit=limit,
                     threshold_file=threshold_files.get(arm))
            directories.append(str(directory))
    return directories


def report_arms(out_dir: str, split: str, seeds: list[int],
                limit: int = 0) -> Path:
    root = Path(out_dir).resolve()
    arm_rows = []
    case_sets = []
    for arm, (kind, _) in ARMS.items():
        directories = [root / "evaluations" / split / arm / f"seed{seed}"
                       for seed in seeds]
        for directory in directories:
            with (directory / "evaluation.json").open(encoding="utf-8") as stream:
                metadata = json.load(stream)
            if not metadata.get("complete"):
                raise RuntimeError(f"Incomplete ablation arm: {directory}")
            with (directory / "per_case.csv").open(newline="", encoding="utf-8") as stream:
                case_sets.append({row["sample_id"] for row in csv.DictReader(stream)})
        required = ([kind, "geovae_latent", "geovae_joint"]
                    if arm == "attention_position" else [kind])
        summary = report([str(path) for path in directories],
                         str(root / "summaries" / split / arm),
                         draft=(split != "test" or limit > 0),
                         required_methods=tuple(required))
        with summary.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                arm_rows.append({"arm": arm, **row})
    if len({frozenset(cases) for cases in case_sets}) != 1:
        raise RuntimeError("Ablation arms used different cases")
    output = root / "summaries" / split / "ablation_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(arm_rows[0]))
        writer.writeheader()
        writer.writerows(arm_rows)
    return output