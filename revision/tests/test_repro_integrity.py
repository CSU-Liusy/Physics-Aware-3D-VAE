"""Scientific-integrity checks for config selection and sealed evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from config_loader import parse_args
from experiment_data import prepare_protocol
from repro_pipeline import evaluate, training_config_hash
from repro_report import report


def _tiny_config(path: Path) -> None:
    path.write_text(
        """
grid_size: [16, 16, 16]
latent_dim: 16
base_channels: 4
use_transformer: true
use_relative_position: true
num_holes: 2
samples_per_hole: 4
mask_seed: 42
device: cpu
microbatch: 2
effective_batch: 4
num_workers: 0
amp: false
epochs: 1
patience: 1
lr: 0.0001
lambda_observed: 5
beta: 0.1
threshold: 0.5
""".strip() + "\n",
        encoding="utf-8",
    )


def test_only_evaluation_settings_can_change_without_retraining():
    trained = {
        "grid_size": [32, 32, 32], "mask_seed": 42,
        "base_channels": 16, "lambda_observed": 5.0,
        "idw_power": 2.0, "test_time_steps": 50, "threshold": 0.5,
        "device": "cuda",
    }
    tuned = dict(trained, idw_power=3.0, test_time_steps=25,
                 latent_test_time_steps=25, latent_test_time_lr=0.01,
                 joint_test_time_steps=50, joint_test_time_lora_lr=0.0003,
                 threshold=0.35, device="cpu")
    assert training_config_hash(trained) == training_config_hash(tuned)
    changed_training = dict(trained, mask_seed=99)
    assert training_config_hash(trained) != training_config_hash(changed_training)

def test_config_is_single_named_file_then_cli_override(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("epochs: 7\nnum_holes: 2\n", encoding="utf-8")
    b.write_text("epochs: 99\nnum_holes: 16\n", encoding="utf-8")
    parsed = parse_args(["--config", str(a)])
    assert (parsed.epochs, parsed.num_holes, parsed.config) == (7, 2, str(a))
    overridden = parse_args(["--config", str(a), "--epochs", "9"])
    assert (overridden.epochs, overridden.num_holes) == (9, 2)
    with pytest.raises(FileNotFoundError):
        parse_args(["--config", str(tmp_path / "missing.yaml")])


def test_repeated_evaluation_and_prediction_hashes(tmp_path):
    manifest = prepare_protocol(str(tmp_path), str(tmp_path / "protocol"),
                                count=20, seed=42, grid_size=(16, 16, 16),
                                audit_raw=False)
    config = tmp_path / "config.yaml"
    _tiny_config(config)
    first = tmp_path / "eval1"
    second = tmp_path / "eval2"
    evaluate(manifest, str(config), str(first), "test",
             ["idw", "rbf"], {}, seed=11, limit=2)
    evaluate(manifest, str(config), str(second), "test",
             ["idw", "rbf"], {}, seed=11, limit=2)
    with (first / "per_case.csv").open(newline="", encoding="utf-8") as stream:
        rows1 = list(csv.DictReader(stream))
    with (second / "per_case.csv").open(newline="", encoding="utf-8") as stream:
        rows2 = list(csv.DictReader(stream))
    assert [(r["sample_id"], r["method"], r["prediction_sha256"])
            for r in rows1] == [(r["sample_id"], r["method"], r["prediction_sha256"])
                               for r in rows2]
    assert len(rows1) == 4
    assert json.loads((first / "evaluation.json").read_text(encoding="utf-8"))["complete"]
    draft = report([str(first)], str(tmp_path / "report"), draft=True)
    assert draft.is_file()
    assert (tmp_path / "report" / "reliability.csv").is_file()
    with pytest.raises(RuntimeError, match="Final report requires"):
        report([str(first)], str(tmp_path / "strict"))
    one_prediction = first / rows1[0]["prediction_file"]
    one_prediction.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        report([str(first)], str(tmp_path / "tampered"), draft=True)
