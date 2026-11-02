"""CPU integration draft of the synthetic revision, never manuscript evidence.

This bypasses formal 50-parent hyperparameter tuning rather than pretending a
tiny validation set contains 50 parents. It trains all four model families and
two training ablations for one epoch, freezes draft validation thresholds, then
checks saved test predictions, local-variability artifacts and checkpoint resume.
No confidential source files are read.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import torch

from experiment_data import prepare_protocol, verify_protocol
from repro_pipeline import ROOT, evaluate, save_json, sha256_file, source_hash, stable_hash, train
from repro_report import report
from repro_tuning import select_thresholds
from revision_analysis import analyse_evaluations
from revision_diagnostics import uncertainty_run
from revision_numerics import (DECLARED_POLICY, configure_evaluation,
                               helper_sha256 as numerics_sha256, record_training_policy)

METHODS = ("idw", "rbf", "unet", "base_vae", "cgan", "geovae",
           "geovae_latent", "geovae_joint")
CONFIG = {
    "grid_size": [16, 16, 16], "latent_dim": 8, "base_channels": 2,
    "use_transformer": True, "use_relative_position": True,
    "unet_features": 2, "cgan_base_channels": 4,
    "num_holes": 2, "samples_per_hole": 4, "mask_seed": 42,
    "device": "cpu", "microbatch": 2, "effective_batch": 4,
    "num_workers": 0, "amp": False, "epochs": 1, "patience": 1,
    "lr": .0001, "weight_decay": .00001, "lambda_observed": 5.,
    "beta": .1, "kl_warmup_epochs": 1, "threshold": .5,
    "test_time_steps": 2, "test_time_lr": .03, "test_time_lora_lr": .001,
    "test_time_prior_weight": .01, "test_time_lora_weight": .01,
    "test_time_lora_rank": 2, "cgan_lr_g": .0002, "cgan_lr_d": .0002,
    "cgan_lambda_recon": 10., "cgan_lambda_observed": 5.,
    "cgan_eval_realizations": 1,
}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _train(manifest: str, config: Path, folder: Path, kind: str) -> Path:
    latest = folder / "latest.pt"
    if kind == "cgan":
        from repro_cgan import train_cgan
        return train_cgan(manifest, str(config), str(folder), 11,
                          resume=str(latest) if latest.is_file() else None)
    return train(manifest, str(config), str(folder), kind, 11,
                 resume=str(latest) if latest.is_file() else None)


def run(out: str | Path, prepare_only: bool = False) -> Path:
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("CPU smoke training requires a fresh process with unchanged training flags")
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {"profile": "cpu_revision_smoke_v1", "config": CONFIG,
                "training_source_sha256": source_hash(), "helper_sha256": sha256_file(__file__),
                "diagnostic_sha256": sha256_file(ROOT / "revision_diagnostics.py"),
                "evaluation_numerical_policy": DECLARED_POLICY,
                "evaluation_numerics_source_sha256": numerics_sha256(),
                "seed": 11, "synthetic_count": 20, "methods": list(METHODS),
                "restricted_real_data_status": "not_available_restricted"}
    identity_path = root / "smoke_identity.json"
    if identity_path.is_file() and _read(identity_path) != identity:
        raise RuntimeError("CPU draft identity differs; use a new output directory")
    save_json(identity_path, identity)
    record_training_policy(root, phase="cpu_smoke_training")
    save_json(root / "smoke_status.json", {"status": "running", "formal_evidence": False})
    config_path = root / "config.json"
    save_json(config_path, CONFIG)
    manifest = prepare_protocol(str(root), str(root / "protocol"), count=20,
                                seed=42, grid_size=(16, 16, 16), audit_raw=False)
    protocol = verify_protocol(manifest, verify_files=True)
    if prepare_only:
        save_json(root / "smoke_status.json", {"status": "draft_smoke_prepared", "formal_evidence": False,
                  "case_count": 20, "split_counts": protocol["split_counts"],
                  "restricted_real_data_status": "not_available_restricted"})
        return root / "smoke_status.json"
    started = time.perf_counter()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        checkpoints = {}
        for kind in ("geovae", "base_vae", "unet", "cgan"):
            print(f"[CPU draft] train {kind}, seed11, one epoch", flush=True)
            folder = root / "models" / kind / "seed11"
            checkpoints[kind] = str(_train(manifest, config_path, folder, kind))
        arms = []
        for arm, kind, overrides in (
                ("base_no_observed", "base_vae", {"lambda_observed": 0.}),
                ("attention_no_position", "geovae", {"use_relative_position": False})):
            arm_root = root / "ablation" / arm
            arm_config = arm_root / "config.json"
            save_json(arm_config, dict(CONFIG, **overrides))
            checkpoint = _train(manifest, arm_config, arm_root / "model", kind)
            arms.append((arm, kind, arm_root, arm_config, checkpoint))
        # Restore while training backend flags are still unchanged. Evaluation
        # flags remain strict for the rest of this process; they are never reset.
        _train(manifest, config_path, root / "models" / "geovae" / "seed11", "geovae")
        configure_evaluation(root, phase="cpu_smoke_validation")
        validation = root / "validation" / "seed11"
        evaluate(manifest, str(config_path), str(validation), "val", list(METHODS),
                 checkpoints, seed=11)
        thresholds = root / "thresholds.json"
        select_thresholds([str(validation)], str(config_path), str(thresholds),
                          draft=True, allow_subset=True)
        testing = root / "test" / "seed11"
        configure_evaluation(root, phase="cpu_smoke_test")
        evaluate(manifest, str(config_path), str(testing), "test", list(METHODS),
                 checkpoints, seed=11, threshold_file=str(thresholds))
        report([str(testing)], str(root / "report"), draft=True, required_methods=METHODS)
        analyse_evaluations(manifest, [testing], root / "analysis", formal=False,
                            required_methods=METHODS, config_path=config_path, replicates=64)
        ablation_rows = []
        for arm, kind, arm_root, arm_config, checkpoint in arms:
            arm_val, arm_test = arm_root / "validation", arm_root / "test"
            configure_evaluation(root, phase="cpu_smoke_ablation_validation")
            evaluate(manifest, str(arm_config), str(arm_val), "val", [kind],
                     {kind: str(checkpoint)}, seed=11)
            arm_threshold = arm_root / "thresholds.json"
            select_thresholds([str(arm_val)], str(arm_config), str(arm_threshold),
                              draft=True, allow_subset=True)
            configure_evaluation(root, phase="cpu_smoke_ablation_test")
            evaluate(manifest, str(arm_config), str(arm_test), "test", [kind],
                     {kind: str(checkpoint)}, seed=11, threshold_file=str(arm_threshold))
            summary = report([str(arm_test)], str(arm_root / "report"), draft=True,
                             required_methods=(kind,))
            with summary.open(newline="", encoding="utf-8") as stream:
                ablation_rows.extend({"arm": arm, **row} for row in csv.DictReader(stream))
        with (root / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ablation_rows[0]))
            writer.writeheader(); writer.writerows(ablation_rows)
        for split in ("val", "test"):
            uncertainty_run(manifest, config_path, checkpoints["geovae"], root / "uncertainty",
                            split=split, seed=11, smoke=True)
        save_json(root / "smoke_status.json", {
            "status": "draft_smoke_complete", "formal_evidence": False,
            "manifest_sha256": sha256_file(manifest), "config_sha256": stable_hash(CONFIG),
            "training_source_sha256": source_hash(), "models": list(checkpoints),
            "evaluation_numerics_source_sha256": numerics_sha256(),
            "training_ablations": ["base_no_observed", "attention_no_position"],
            "methods": list(METHODS), "seed": 11, "case_count": 20,
            "split_counts": protocol["split_counts"], "epochs": 1,
            "uncertainty": "one validation and one test parent, three starts, four proposals per variance",
            "checkpoint_restore_checked": True, "elapsed_seconds": time.perf_counter() - started,
            "hyperparameter_search": "not performed in the CPU draft; no 50-parent requirement is bypassed for formal runs",
            "restricted_real_data_status": "not_available_restricted"})
    except Exception as error:
        save_json(root / "smoke_status.json", {"status": "draft_smoke_failed", "formal_evidence": False,
                  "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        torch.set_num_threads(previous_threads)
    return root / "smoke_status.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "runs" / "public_cpu_smoke"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.out, prepare_only=args.prepare_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
