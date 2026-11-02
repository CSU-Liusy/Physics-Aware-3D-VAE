"""One-command, offline, provenance-checked GeoVAE experiment runner.

No network access, package installation, manuscript changes, or raw-data edits.
The formal path uses the frozen procedural protocol. Confidential raw source
files are not distributed or audited in this public synthetic package.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import torch

from experiment_data import prepare_protocol, verify_protocol
from repro_pipeline import (
    DEFAULT_CONFIG, DEFAULT_DATA, DEFAULT_OUT, ROOT, doctor, environment_info,
    EVALUATION_ONLY_CONFIG_KEYS, evaluate, load_config, save_json, sha256_file,
    source_hash, stable_hash, train, training_config_hash,
    verify_prediction_cache,
)
from repro_report import REQUIRED_METHODS, _read_run, report
from revision_numerics import DECLARED_POLICY, helper_sha256

DEFAULT_MANIFEST = ROOT / "protocol" / "manifest.json"
SEEDS = (11, 23, 37)
ROBUST_METHODS = ("idw", "unet", "geovae", "geovae_latent", "geovae_joint")
SNAPSHOT_FILES = (
    "main.py", "repro_run_all.py", "repro_cli.py", "repro_pipeline.py",
    "experiment_data.py", "model.py", "repro_cgan.py", "repro_baselines.py",
    "repro_metrics.py", "repro_tuning.py", "repro_report.py",
    "repro_ablation.py", "repro_figures.py", "repro_smoke.py",
    "repro_hyper_tuning.py",
    "revision_numerics.py",
    "config_loader.py", "requirements-core.txt", "requirements-baselines.txt",
    "comparisons/models/unet.py",
)

SNAPSHOT_FILES += ("revision_runner.py", "revision_diagnostics.py", "revision_analysis.py",
                   "revision_robustness.py", "revision_smoke.py", "public_finish.py")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_hashes(paths: list[Path], root: Path) -> dict[str, str]:
    result = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise RuntimeError(f"Missing or escaped run artifact: {path}")
        result[str(resolved.relative_to(root)).replace("\\", "/")] = sha256_file(resolved)
    return result


class _StageLog:
    """Keep detailed output in a file while showing bounded case progress."""

    def __init__(self, stream, terminal, stage: str, verbose: bool,
                 max_epochs: int) -> None:
        self.stream = stream
        self.terminal = terminal
        self.stage = stage
        self.verbose = verbose
        self.started = time.monotonic()
        self.pending = ""
        self.tune_candidate = (0, 0, 0)
        self.max_epochs = max_epochs

    def _progress(self, completed: int, total: int, unit: str) -> None:
        elapsed = time.monotonic() - self.started
        remaining = elapsed / completed * (total - completed) if completed else 0.0
        print(f"[PROGRESS] {self.stage}: {completed}/{total} {unit}; "
              f"elapsed={elapsed / 60:.1f}m, ETA={remaining / 60:.1f}m",
              file=self.terminal, flush=True)

    def write(self, value: str) -> int:
        self.stream.write(value)
        if self.verbose:
            self.terminal.write(value)
            self.terminal.flush()
        else:
            self.pending += value
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                announced = re.match(r"^\[tune\] (\d+)/(\d+) .*: (\d+) cases$", line)
                if announced:
                    self.tune_candidate = tuple(int(value) for value in announced.groups())
                tuned = re.match(r"^\[tune\] .*: (\d+)/(\d+)$", line)
                if tuned and self.tune_candidate[0]:
                    index, candidates, per_candidate = self.tune_candidate
                    done = (index - 1) * per_candidate + int(tuned.group(1))
                    self._progress(done, candidates * per_candidate, "tuning cases")
                    continue
                match = re.search(r"(\d+)/(\d+) cases", line)
                if match:
                    completed, total = (int(part) for part in match.groups())
                    self._progress(completed, total, "cases")
                    continue
                if line.startswith("{") and '"epoch"' in line:
                    try:
                        epoch = int(json.loads(line)["epoch"])
                    except (ValueError, TypeError, KeyError):
                        continue
                    if epoch == 1 or epoch % 10 == 0:
                        self._progress(epoch, self.max_epochs,
                                       "epochs (upper bound)")
        return len(value)

    def flush(self) -> None:
        self.stream.flush()
        if self.verbose:
            self.terminal.flush()


class OfflineRun:
    def __init__(self, manifest: str | Path, config: str | Path,
                 out: str | Path, smoke: bool = False, dry_run: bool = False,
                 prepare_mode: bool = False, verbose: bool = False):
        self.manifest = Path(manifest).resolve()
        self.config_path = Path(config).resolve()
        self.out = Path(out).resolve()
        self.smoke = bool(smoke)
        self.dry_run = bool(dry_run)
        self.prepare_mode = bool(prepare_mode)
        self.verbose = bool(verbose)
        self.seeds = (11,) if smoke else SEEDS
        self.methods = ("idw", "rbf", "unet", "base_vae", "geovae") if smoke else REQUIRED_METHODS
        self.robust_methods = ("idw", "geovae") if smoke else ROBUST_METHODS
        self.limit = 3 if smoke else 0
        self.failed_stage: str | None = None
        self._raw_verified: dict | None = None
        self.identity: dict = {}
        self.identity_sha: str = ""
        self.config = load_config(self.config_path)
        # Public synthetic reproduction does not require restricted source files.
        if not self.manifest.is_file():
            raise FileNotFoundError(
                f"Frozen protocol is missing: {self.manifest}. Restore the audited "
                "protocol before running modules; this runner never silently "
                "generates a new experiment cohort.")
        self.protocol = verify_protocol(
            self.manifest, verify_files=self.prepare_mode and not dry_run)
        if tuple(self.protocol["grid_size"]) != tuple(self.config["grid_size"]):
            raise ValueError("Protocol and configuration grid sizes differ")
        if not smoke:
            if (self.protocol["count"] != 5000
                    or self.protocol["split_counts"] !=
                    {"train": 3500, "val": 750, "test": 750}
                    or self.protocol["parent_counts"] !=
                    {"train": 700, "val": 150, "test": 150}
                    or tuple(self.config["grid_size"]) != (32, 32, 32)):
                raise RuntimeError("Formal run requires the frozen 5000-case 32^3 protocol")
        raw_inventory = ROOT / "protocol" / "raw_inventory.jsonl"
        raw_summary = ROOT / "protocol" / "raw_audit_summary.json"
        # Restricted raw inventories are intentionally not distributed.
        code_hashes = {
            name: sha256_file(ROOT / name)
            for name in SNAPSHOT_FILES if (ROOT / name).is_file()
        }
        self.identity = {
            "format_version": 1,
            "mode": "draft_smoke" if smoke else "formal",
            "data_root": "not_available_restricted", "public_mode": "synthetic_only",
            "manifest": str(self.manifest),
            "manifest_sha256": sha256_file(self.manifest),
            "raw_inventory_sha256": sha256_file(raw_inventory) if raw_inventory.is_file() else None,
            "raw_audit_sha256": sha256_file(raw_summary) if raw_summary.is_file() else None,
            "config": str(self.config_path),
            "config_sha256": stable_hash(self.config),
            "source_sha256": source_hash(),
            "evaluation_numerical_policy": DECLARED_POLICY,
            "evaluation_numerics_source_sha256": helper_sha256(),
            "code_files": code_hashes,
            "seeds": list(self.seeds),
            "methods": list(self.methods),
            "robustness_methods": list(self.robust_methods),
            "limit": self.limit,
            "test_label_policy": "sealed_until_validation_thresholds_frozen",
        }
        self.identity_sha = stable_hash(self.identity)
        if not dry_run:
            self._initialize_output()

    def _initialize_output(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        identity_file = self.out / "run_identity.json"
        if identity_file.is_file():
            old = _json(identity_file)
            if stable_hash(old) != self.identity_sha:
                raise RuntimeError(
                    "Existing output belongs to different code, configuration, data "
                    "or experiment settings. Use a NEW --out directory; do not mix runs.")
        else:
            save_json(identity_file, self.identity)
        snapshot = self.out / "frozen_inputs"
        for name, digest in self.identity["code_files"].items():
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file():
                if sha256_file(destination) != digest:
                    raise RuntimeError(f"Frozen source snapshot changed: {destination}")
            else:
                shutil.copy2(ROOT / name, destination)
        for source, name in ((self.manifest, "manifest.json"),
                             (self.config_path, "evaluation_config.yaml")):
            destination = snapshot / name
            if destination.is_file():
                if sha256_file(destination) != sha256_file(source):
                    raise RuntimeError(f"Frozen input snapshot changed: {destination}")
            else:
                shutil.copy2(source, destination)

    def _event(self, event: str, stage: str, detail: str = "") -> None:
        print(f"[{event}] {stage}" + (f": {detail}" if detail else ""), flush=True)
        if self.dry_run:
            return
        path = self.out / "stage_events.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": event, "stage": stage, "detail": detail,
            }, ensure_ascii=False) + "\n")

    def _stage(self, name: str, inputs: dict, check: Callable[[], list[Path] | None],
               run: Callable[[], None]) -> None:
        if self.dry_run:
            self._event("PLAN", name)
            return
        input_sha = stable_hash({"identity": self.identity_sha, "stage": name,
                                 "inputs": inputs})
        marker = self.out / "stages" / (name.replace("/", "__") + ".json")
        self.failed_stage = name
        if marker.is_file():
            saved = _json(marker)
            if saved.get("input_sha256") != input_sha:
                raise RuntimeError(f"Stage inputs changed after completion: {name}")
            expected = saved.get("artifacts", {})
            for relative, digest in expected.items():
                artifact = (self.out / relative).resolve()
                if not artifact.is_relative_to(self.out) or not artifact.is_file():
                    raise RuntimeError(f"Missing completed artifact: {artifact}")
                if sha256_file(artifact) != digest:
                    raise RuntimeError(f"Completed artifact changed: {artifact}")
            artifacts = check()
            if artifacts is None:
                raise RuntimeError(f"Completed stage now fails validation: {name}")
            self._event("SKIP", name, "verified completed outputs")
            return
        artifacts = check()
        if artifacts is None:
            self._event("RUN", name)
            log_path = self.out / "logs" / (name.replace("/", "__") + ".log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                          f"{name} ===\n")
                stage_log = _StageLog(log, sys.stdout, name, self.verbose,
                                      int(self.config["epochs"]))
                try:
                    with contextlib.redirect_stdout(stage_log), contextlib.redirect_stderr(stage_log):
                        run()
                except BaseException:
                    traceback.print_exc(file=log)
                    self._event("ERROR", name, f"details: {log_path}")
                    raise
            artifacts = check()
            if artifacts is None:
                raise RuntimeError(f"Stage did not produce complete outputs: {name}")
        else:
            self._event("ADOPT", name, "verified existing complete outputs")
        hashes = _file_hashes(artifacts, self.out)
        save_json(marker, {
            "status": "complete", "stage": name, "input_sha256": input_sha,
            "artifacts": hashes,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        self._event("DONE", name)
        self.failed_stage = None

    def _preflight(self) -> list[Path] | None:
        path = self.out / "preflight" / "environment.json"
        if not path.is_file():
            return None
        state = _json(path)
        if state.get("identity_sha256") != self.identity_sha:
            raise RuntimeError("Preflight belongs to different inputs")
        if not state.get("passed"):
            return None
        if not self.smoke:
            recorded = state.get("raw_data_verification") or {}
            if recorded.get("inventory_sha256") != self.identity["raw_inventory_sha256"]:
                raise RuntimeError("Preflight raw inventory identity changed")
        return [path]

    def _run_preflight(self) -> None:
        checks = doctor(require_cuda=not self.smoke)
        required = ("numpy", "scipy", "yaml")
        if not self.smoke:
            required += ("gstools", "mpslib")
        missing = [name for name in required if not checks[name]]
        if importlib.util.find_spec("matplotlib") is None:
            missing.append("matplotlib")
        if missing:
            raise RuntimeError(
                "Missing offline Python dependencies: " + ", ".join(missing)
                + ". Install requirements-core.txt and requirements-baselines.txt "
                  "before starting; this runner never downloads packages.")
        free_gib = round(shutil.disk_usage(self.out).free / 1024 ** 3, 2)
        memory = checks["gpu_memory_gib"]
        if not self.smoke and memory is not None and memory < 12:
            raise RuntimeError(
                f"GPU has {memory} GiB; use a GPU with at least 12 GiB and "
                "run the CUDA batch preflight.")
        if not self.smoke and free_gib < 24:
            raise RuntimeError(
                f"Only {free_gib} GiB free on the results drive; the full "
                "prediction archive needs about 24 GiB. Free space or use "
                "a new --out on a larger drive.")
        raw_check = self._verify_raw_inventory() if not self.smoke else None
        state = {
            **checks, "raw_data_verification": raw_check, "passed": True, "identity_sha256": self.identity_sha,
            "data_root": "not_available_restricted", "public_mode": "synthetic_only",
            "protocol_grid_count": self.protocol["count"],
            "free_disk_gib_at_start": free_gib,
            "environment": environment_info(),
            "network_required": False,
        }
        save_json(self.out / "preflight" / "environment.json", state)

    def _smoke_check(self) -> list[Path] | None:
        path = self.out / "preflight" / "smoke.json"
        if not path.is_file():
            return None
        meta = _json(path)
        if (not meta.get("passed")
                or meta.get("manifest_sha256") != self.identity["manifest_sha256"]
                or meta.get("config_sha256") != self.identity["config_sha256"]
                or meta.get("source_sha256") != self.identity["source_sha256"]
                or meta.get("device") != ("cpu" if self.smoke else "cuda")):
            raise RuntimeError("CUDA/CPU smoke result differs from frozen inputs")
        checkpoint = Path(meta["checkpoint"])
        if not checkpoint.is_file():
            raise RuntimeError("Smoke checkpoint is missing")
        return [path, checkpoint]

    def _run_smoke(self) -> None:
        from repro_smoke import smoke_batch
        smoke_batch(str(self.manifest), str(self.config_path),
                    str(self.out / "preflight"), "cpu" if self.smoke else "cuda")

    def _source_for_kind(self, kind: str) -> str:
        if kind == "cgan":
            from repro_cgan import _source_sha256
            return _source_sha256()
        return source_hash()

    def _train_check(self, run_dir: Path, kind: str, seed: int,
                     config_path: Path) -> list[Path] | None:
        best, latest = run_dir / "best.pt", run_dir / "latest.pt"
        if not best.is_file() and not latest.is_file():
            return None
        if not best.is_file() or not latest.is_file():
            return None
        config = load_config(config_path)
        expected = {
            "model_kind": kind, "seed": seed,
            "manifest_sha256": self.identity["manifest_sha256"],
            "config_sha256": stable_hash(config),
            "source_sha256": self._source_for_kind(kind),
        }
        states = []
        for checkpoint in (best, latest):
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            for key, value in expected.items():
                if state.get(key) != value:
                    raise RuntimeError(f"Checkpoint {checkpoint} has mismatched {key}")
            states.append(state)
        current = states[1]
        stale = int(current.get("stale_epochs", current.get("no_improve", 0)))
        if int(current["epoch"]) < int(config["epochs"]) and stale < int(config["patience"]):
            return None
        meta_path = run_dir / "run.json"
        history = run_dir / "history.jsonl"
        if not meta_path.is_file() or not history.is_file():
            raise RuntimeError(f"Completed checkpoint lacks training metadata: {run_dir}")
        return [best, latest, meta_path, history]

    def _train_one(self, run_dir: Path, kind: str, seed: int,
                   config_path: Path, reuse_from: Path | None = None) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        if reuse_from is not None:
            source_files = self._train_check(reuse_from, kind, seed, self.config_path)
            if source_files is None:
                raise RuntimeError(f"Cannot reuse unfinished main model: {reuse_from}")
            if stable_hash(load_config(config_path)) != stable_hash(self.config):
                raise RuntimeError(f"Ablation arm differs from the reusable main model: {config_path}")
            for source in source_files:
                destination = run_dir / source.name
                if destination.is_file() and sha256_file(destination) != sha256_file(source):
                    raise RuntimeError(f"Existing reused checkpoint differs: {destination}")
                if not destination.is_file():
                    shutil.copy2(source, destination)
            save_json(run_dir / "reused_from.json", {
                "source_run": str(reuse_from),
                "source_best_sha256": sha256_file(reuse_from / "best.pt"),
                "reason": "exact same model kind, seed, manifest, training config and source",
            })
            return
        latest = run_dir / "latest.pt"
        best = run_dir / "best.pt"
        if latest.is_file():
            resume = str(latest)
        elif best.is_file():
            resume = str(best)
        elif any(run_dir.iterdir()):
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            for filename in ("run.json", "history.jsonl"):
                orphan = run_dir / filename
                if orphan.is_file():
                    target = run_dir / (filename + ".interrupted." + stamp)
                    if target.exists():
                        raise RuntimeError(f"Interrupted training backup exists: {target}")
                    orphan.replace(target)
            resume = None
        else:
            resume = None
        if kind == "cgan":
            from repro_cgan import train_cgan
            train_cgan(str(self.manifest), str(config_path), str(run_dir),
                       seed, resume=resume)
        else:
            train(str(self.manifest), str(config_path), str(run_dir),
                  kind, seed, resume=resume)

    def _verify_raw_inventory(self) -> dict:
        """Explicit absence record, never a fabricated successful raw-data audit."""
        self._raw_verified = {"status": "not_available_restricted", "audited": False,
                              "scope": "public_synthetic_only", "files": None,
                              "inventory_sha256": None,
                              "reason": "Confidential source files and inventories are not distributed"}
        return self._raw_verified


    def _baseline_check(self) -> list[Path] | None:
        return self._eval_check(
            self.out / "preflight" / "classical_one",
            "val", ("ik", "mps"), 11, self.config_path,
            {}, None, 8, 0.0, 1, None)

    def _run_baseline_check(self) -> None:
        evaluate(str(self.manifest), str(self.config_path),
                 str(self.out / "preflight" / "classical_one"),
                 "val", ["ik", "mps"], {}, 11, limit=1)

    def _protocol_archive_check(self) -> list[Path] | None:
        archive = self.out / "frozen_inputs"
        metadata = archive / "protocol_archive.json"
        if not metadata.is_file():
            return None
        record = _json(metadata)
        if record.get("manifest_sha256") != self.identity["manifest_sha256"]:
            raise RuntimeError("Copied protocol manifest does not match the run")
        if sha256_file(archive / "manifest.json") != self.identity["manifest_sha256"]:
            raise RuntimeError("Copied protocol manifest changed")
        for row in self.protocol["samples"]:
            original = (self.manifest.parent / row["path"]).resolve()
            copied = (archive / row["path"]).resolve()
            if not copied.is_relative_to(archive) or not copied.is_file():
                raise RuntimeError(f"Missing protocol archive grid: {copied}")
            if copied.stat().st_size != original.stat().st_size:
                raise RuntimeError(f"Protocol archive grid size changed: {copied}")
        return [archive / "manifest.json", metadata]

    def _full_archive_check(self) -> None:
        """Final handoff checks contents, rather than only archive file sizes."""
        archive = self.out / "frozen_inputs"
        if self._protocol_archive_check() is None:
            raise RuntimeError("Frozen protocol archive is incomplete")
        verify_protocol(self.manifest, verify_files=True)
        for index, row in enumerate(self.protocol["samples"], 1):
            copied = (archive / row["path"]).resolve()
            if sha256_file(copied) != row["sha256"]:
                raise RuntimeError(f"Archived protocol grid hash mismatch: {row['sample_id']}")
            if index % 1000 == 0:
                self._event("PROGRESS", "final_archive_hashes",
                            f"{index}/{len(self.protocol['samples'])}")

    def _run_protocol_archive(self) -> None:
        archive = self.out / "frozen_inputs"
        for index, row in enumerate(self.protocol["samples"], 1):
            source = (self.manifest.parent / row["path"]).resolve()
            destination = (archive / row["path"]).resolve()
            if not destination.is_relative_to(archive):
                raise RuntimeError("Manifest grid escaped protocol archive")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file():
                shutil.copy2(source, destination)
            if sha256_file(destination) != row["sha256"]:
                raise RuntimeError(f"Copied protocol grid hash differs: {row['sample_id']}")
            if index % 1000 == 0:
                self._event("PROGRESS", "protocol_archive", f"{index}/{len(self.protocol['samples'])}")
        save_json(archive / "protocol_archive.json", {
            "manifest_sha256": self.identity["manifest_sha256"],
            "count": len(self.protocol["samples"]),
            "source": "independent procedural protocol; confidential raw files are not distributed or audited",
            "grid_hashes": "per-sample SHA256 in manifest.json",
        })

    def _eval_check(
        self, path: Path, split: str, methods: tuple[str, ...], seed: int,
        config_path: Path, checkpoints: dict[str, str],
        thresholds: Path | None, holes: int, noise: float, limit: int,
        sample_ids: list[str] | None, deep: bool = False,
    ) -> list[Path] | None:
        metadata_path = path / "evaluation.json"
        csv_path = path / "per_case.csv"
        if not metadata_path.is_file() or not csv_path.is_file():
            return None
        metadata = _json(metadata_path)
        if not metadata.get("complete"):
            return None
        expected = {
            "manifest_sha256": self.identity["manifest_sha256"],
            "config_sha256": stable_hash(load_config(config_path)),
            "source_sha256": source_hash(),
            "split": split, "seed": seed,
            "methods": list(methods), "mask_seed": int(load_config(config_path)["mask_seed"]),
            "num_holes": holes, "noise_rate": noise, "limit": limit,
            "project_observed": False,
            "threshold_selection_sha256": sha256_file(thresholds) if thresholds else None,
            "checkpoints": {name: sha256_file(path) for name, path in checkpoints.items()},
            "case_selection_sha256": stable_hash(sample_ids) if sample_ids else None,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"Existing evaluation {path} mismatches {key}")
        expected_count = (len(sample_ids) if sample_ids else
                          (min(limit, self.protocol["split_counts"][split]) if limit else
                           self.protocol["split_counts"][split]))
        if int(metadata["expected_cases"]) != expected_count:
            raise RuntimeError(f"Existing evaluation has wrong case count: {path}")
        if deep:
            _read_run(path)
        else:
            with csv_path.open(newline="", encoding="utf-8") as stream:
                row_count = sum(1 for _ in csv.DictReader(stream))
            if row_count != expected_count * len(methods):
                raise RuntimeError(f"Evaluation row count differs: {path}")
        return [metadata_path, csv_path, path / "failures.json",
                path / "reliability.json"]

    def _run_eval(
        self, path: Path, split: str, methods: tuple[str, ...], seed: int,
        config_path: Path, checkpoints: dict[str, str],
        thresholds: Path | None = None, holes: int = 8, noise: float = 0.0,
        limit: int = 0, sample_ids: list[str] | None = None,
    ) -> None:
        evaluate(
            str(self.manifest), str(config_path), str(path), split,
            list(methods), checkpoints, seed, num_holes=holes,
            noise_rate=noise, limit=limit,
            threshold_file=str(thresholds) if thresholds else None,
            sample_ids=sample_ids,
            prediction_cache_root=str(self.out / "prediction_cache"),
        )

    def _threshold_check(self, path: Path, methods: tuple[str, ...],
                         config_path: Path, seeds: tuple[int, ...],
                         draft: bool) -> list[Path] | None:
        if not path.is_file():
            return None
        result = _json(path)
        expected = {
            "status": "draft" if draft else "validation_frozen",
            "manifest_sha256": self.identity["manifest_sha256"],
            "config_sha256": stable_hash(load_config(config_path)),
            "evaluation_source_sha256": source_hash(),
            "seeds": list(seeds),
            "cases_per_seed": self.limit if self.limit else 750,
        }
        for key, value in expected.items():
            if result.get(key) != value:
                raise RuntimeError(f"Threshold file {path} mismatches {key}")
        if set(result["thresholds"]) != set(methods):
            raise RuntimeError(f"Threshold file {path} has different methods")
        return [path]

    def _report_check(
        self, path: Path, methods: tuple[str, ...], seeds: tuple[int, ...],
        case_count: int, split: str, draft: bool,
    ) -> list[Path] | None:
        metadata = path / "report.json"
        if not metadata.is_file():
            return None
        result = _json(metadata)
        expected = {
            "status": "draft" if draft else "complete_sealed_test",
            "manifest_sha256": self.identity["manifest_sha256"],
            "source_sha256": source_hash(),
            "seeds": list(seeds),
            "methods": sorted(methods),
            "case_count": case_count,
        }
        for key, value in expected.items():
            if result.get(key) != value:
                raise RuntimeError(f"Report {path} mismatches {key}")
        if result["settings"]["split"] != split:
            raise RuntimeError(f"Report {path} has wrong split")
        files = [metadata, path / "summary.csv", path / "reliability.csv"]
        if not all(file.is_file() for file in files):
            return None
        pairwise = path / "paired_differences.csv"
        if pairwise.is_file():
            files.append(pairwise)
        calibration = path / "reliability.png"
        if calibration.is_file():
            files.append(calibration)
        return files

    def _ablation_config_check(self, root: Path) -> list[Path] | None:
        from repro_ablation import ARMS
        design = root / "ablation_design.json"
        if not design.is_file():
            return None
        paths = [design]
        for arm, (_, overrides) in ARMS.items():
            path = root / "configs" / f"{arm}.json"
            if not path.is_file():
                return None
            expected = dict(self.config)
            expected.update(overrides)
            if load_config(path) != expected:
                raise RuntimeError(f"Ablation configuration changed: {arm}")
            paths.append(path)
        return paths

    def _figure_check(self, path: Path) -> list[Path] | None:
        source = path / "figure_sources.json"
        if not source.is_file():
            return None
        record = _json(source)
        if record.get("manifest_sha256") != self.identity["manifest_sha256"]:
            raise RuntimeError("Figure source manifest differs")
        selected = self.out / "tuning" / "selected_config.json"
        expected_config = (stable_hash(load_config(selected)) if selected.is_file()
                           else self.identity["config_sha256"])
        if record.get("config_sha256") != expected_config:
            raise RuntimeError("Figure source configuration differs")
        files = [source]
        for name in ("figure7_cases", "figure8_performance_calibration"):
            for suffix in (".pdf", ".svg", ".png", ".tiff"):
                files.append(path / (name + suffix))
        files += [path / "figure7_alignment.json",
                  path / "figure8_alignment.json"]
        return files if all(file.is_file() for file in files) else None

    def _robust_ids(self) -> list[str]:
        selected: dict[str, str] = {}
        for row in self.protocol["samples"]:
            if row["split"] == "test":
                selected.setdefault(row["parent_id"], row["sample_id"])
        ids = sorted(selected.values())
        if not self.smoke and len(ids) != 150:
            raise RuntimeError("Robustness needs one predeclared case from each of 150 test parents")
        return ids

    def _robust_ids_check(self, path: Path, ids: list[str]) -> list[Path] | None:
        if not path.is_file():
            return None
        record = _json(path)
        if record.get("manifest_sha256") != self.identity["manifest_sha256"]:
            raise RuntimeError("Robustness case selection uses another protocol")
        if record.get("sample_ids") != ids:
            raise RuntimeError("Robustness case selection changed")
        return [path]

    def _ablation_summary_check(self, root: Path, split: str) -> list[Path] | None:
        path = root / "summaries" / split / "ablation_summary.csv"
        if not path.is_file():
            return None
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        from repro_ablation import ARMS
        expected = {(arm, method) for arm, (kind, _) in ARMS.items()
                    for method in ((kind, "geovae_latent", "geovae_joint")
                                   if arm == "attention_position" else (kind,))}
        if {(row["arm"], row["method"]) for row in rows} != expected:
            raise RuntimeError("Ablation summary has missing or extra contrasts")
        files = [path]
        for arm in ARMS:
            summary_dir = root / "summaries" / split / arm
            files.extend([summary_dir / "summary.csv",
                          summary_dir / "report.json",
                          summary_dir / "reliability.csv"])
        return files if all(file.is_file() for file in files) else None

    def _handoff_check(self) -> list[Path] | None:
        path = self.out / "handoff.json"
        if not path.is_file():
            return None
        record = _json(path)
        if record.get("status") != ("draft_smoke" if self.smoke else "complete_formal"):
            raise RuntimeError("Handoff status is inconsistent")
        if record.get("identity_sha256") != self.identity_sha:
            raise RuntimeError("Handoff identity changed")
        for relative, digest in record["key_artifacts"].items():
            item = (self.out / relative).resolve()
            if not item.is_relative_to(self.out) or sha256_file(item) != digest:
                raise RuntimeError(f"Handoff artifact changed: {relative}")
        return [path]

    def _write_handoff(self) -> None:
        key = [
            self.out / "run_identity.json",
            self.out / "frozen_inputs" / "manifest.json",
            self.out / "frozen_inputs" / "evaluation_config.yaml",
            self.out / "tuning" / "selection.json",
            self.out / "tuning" / "selected_config.json",
            self.out / "thresholds.json",
            self.out / "report" / "summary.csv",
            self.out / "report" / "report.json",
            self.out / "report" / "paired_differences.csv",
            self.out / "report" / "reliability.csv",
            self.out / "figures" / "figure_sources.json",
            self.out / "ablation" / "summaries" / "test" / "ablation_summary.csv",
            self.out / "robustness" / "case_selection.json",
            self.out / "robustness" / "summary.csv",
            self.out / "integrity_audit.json",
        ]
        key = [path for path in key if path.is_file()]
        save_json(self.out / "handoff.json", {
            "status": "draft_smoke" if self.smoke else "complete_formal",
            "identity_sha256": self.identity_sha,
            "key_artifacts": _file_hashes(key, self.out),
            "copy_this_entire_directory": str(self.out),
            "contains": [
                "frozen protocol manifest and all 5000 source grids",
                "training checkpoints and epoch histories",
                "validation and sealed test per-case predictions",
                "three-seed comparison and calibration report",
                "matched ablation results and one-case-per-parent robustness grid",
                "traceable manuscript figures and all stage markers",
            ],
            "data_folder_role": (
                "Confidential raw files are not distributed or audited here; the formal "
                "training target is the independent 3dvae/protocol/grids dataset."
            ),
            "interpretation": (
                "Synthetic and retrospective pseudo-drill evidence only; "
                "do not claim prospective field validation or measured metre units."
            ),
        })

    def run(self) -> Path:
        from repro_ablation import ARMS, prepare_configs, report_arms
        from repro_figures import figures
        from repro_tuning import select_thresholds

        holes = int(self.config["num_holes"])
        if not self.smoke and holes != 8:
            raise RuntimeError("Formal protocol requires the preregistered eight-hole main setting")
        methods = tuple(self.methods)
        val_dirs = [self.out / "validation" / f"seed{seed}" for seed in self.seeds]
        test_dirs = [self.out / "test" / f"seed{seed}" for seed in self.seeds]
        ablation_root = self.out / "ablation"
        robust_root = self.out / "robustness"
        robust_ids = self._robust_ids()
        try:
            self._stage("preflight", {"cuda": not self.smoke},
                        self._preflight, self._run_preflight)
            if not self.smoke:
                self._stage("classical_one_case",
                            {"methods": ["ik", "mps"], "split": "val"},
                            self._baseline_check, self._run_baseline_check)
            if not self.smoke and not self.dry_run:
                with (self.out / "preflight" / "classical_one" /
                      "per_case.csv").open(newline="", encoding="utf-8") as stream:
                    timing = {row["method"]: float(row["seconds"])
                              for row in csv.DictReader(stream)}
                case_runs = 2 * 750 * len(self.seeds)
                for method in ("ik", "mps"):
                    hours = timing[method] * case_runs / 3600
                    self._event("ESTIMATE", "classical_baselines",
                                f"{method}: one case {timing[method]:.1f}s; "
                                f"main val+test about {hours:.1f}h at that rate")
            self._stage("protocol_archive",
                        {"manifest": self.identity["manifest_sha256"]},
                        self._protocol_archive_check, self._run_protocol_archive)
            self._stage("cuda_or_cpu_batch",
                        {"device": "cpu" if self.smoke else "cuda"},
                        self._smoke_check, self._run_smoke)

            for seed in self.seeds:
                for kind in ("geovae", "base_vae", "unet", "cgan"):
                    folder = self.out / "models" / kind / f"seed{seed}"
                    self._stage(
                        f"train/{kind}/seed{seed}",
                        {"seed": seed, "kind": kind, "config": self.identity["config_sha256"]},
                        lambda folder=folder, kind=kind, seed=seed:
                            self._train_check(folder, kind, seed, self.config_path),
                        lambda folder=folder, kind=kind, seed=seed:
                            self._train_one(folder, kind, seed, self.config_path),
                    )

            for seed, path in zip(self.seeds, val_dirs):
                checkpoints = {
                    kind: str(self.out / "models" / kind / f"seed{seed}" / "best.pt")
                    for kind in ("geovae", "base_vae", "unet", "cgan")
                    if kind in methods or (kind == "geovae" and
                       set(methods) & {"geovae_latent", "geovae_joint"})
                }
                self._stage(
                    f"validation/seed{seed}",
                    {"methods": methods, "seed": seed, "checkpoints": checkpoints},
                    lambda path=path, seed=seed, checkpoints=checkpoints:
                        self._eval_check(path, "val", methods, seed, self.config_path,
                                         checkpoints, None, holes, 0.0, self.limit, None),
                    lambda path=path, seed=seed, checkpoints=checkpoints:
                        self._run_eval(path, "val", methods, seed, self.config_path,
                                       checkpoints, holes=holes, limit=self.limit),
                )

            threshold_path = self.out / "thresholds.json"
            self._stage(
                "validation_thresholds",
                {"evaluations": [str(path) for path in val_dirs]},
                lambda: self._threshold_check(
                    threshold_path, methods, self.config_path, self.seeds, self.smoke),
                lambda: select_thresholds(
                    [str(path) for path in val_dirs], str(self.config_path),
                    str(threshold_path), draft=self.smoke),
            )

            for seed, path in zip(self.seeds, test_dirs):
                checkpoints = {
                    kind: str(self.out / "models" / kind / f"seed{seed}" / "best.pt")
                    for kind in ("geovae", "base_vae", "unet", "cgan")
                    if kind in methods or (kind == "geovae" and
                       set(methods) & {"geovae_latent", "geovae_joint"})
                }
                self._stage(
                    f"sealed_test/seed{seed}",
                    {"methods": methods, "seed": seed,
                     "threshold_sha256": sha256_file(threshold_path) if not self.dry_run else "planned"},
                    lambda path=path, seed=seed, checkpoints=checkpoints:
                        self._eval_check(path, "test", methods, seed, self.config_path,
                                         checkpoints, threshold_path, holes, 0.0,
                                         self.limit, None),
                    lambda path=path, seed=seed, checkpoints=checkpoints:
                        self._run_eval(path, "test", methods, seed, self.config_path,
                                       checkpoints, threshold_path, holes=holes,
                                       limit=self.limit),
                )

            report_root = self.out / "report"
            self._stage(
                "main_report", {"test": [str(path) for path in test_dirs]},
                lambda: self._report_check(report_root, methods, self.seeds,
                                           self.limit or 750, "test", self.smoke),
                lambda: report([str(path) for path in test_dirs],
                               str(report_root), draft=self.smoke,
                               required_methods=methods),
            )
            figure_root = self.out / "figures"
            figure_methods = ("idw", "unet", "geovae")
            self._stage(
                "manuscript_figures",
                {"report": str(report_root), "methods": figure_methods},
                lambda: self._figure_check(figure_root),
                lambda: figures(str(test_dirs[0]), str(report_root),
                                str(self.config_path), str(figure_root),
                                list(figure_methods), draft=self.smoke),
            )

            self._stage(
                "ablation/configs", {"base": self.identity["config_sha256"]},
                lambda: self._ablation_config_check(ablation_root),
                lambda: prepare_configs(str(self.config_path), str(ablation_root)),
            )
            for seed in self.seeds:
                for arm, (kind, _) in ARMS.items():
                    config_path = ablation_root / "configs" / f"{arm}.json"
                    folder = ablation_root / "models" / arm / f"seed{seed}"
                    reuse = (self.out / "models" / kind / f"seed{seed}"
                             if arm in ("base_observed", "attention_position") else None)
                    self._stage(
                        f"ablation/train/{arm}/seed{seed}",
                        {"arm": arm, "seed": seed,
                         "config_sha256": stable_hash(load_config(config_path))
                         if not self.dry_run else arm},
                        lambda folder=folder, kind=kind, seed=seed, config_path=config_path:
                            self._train_check(folder, kind, seed, config_path),
                        lambda folder=folder, kind=kind, seed=seed,
                               config_path=config_path, reuse=reuse:
                            self._train_one(folder, kind, seed, config_path, reuse),
                    )
            for arm, (kind, _) in ARMS.items():
                config_path = ablation_root / "configs" / f"{arm}.json"
                arm_methods = ((kind, "geovae_latent", "geovae_joint")
                               if arm == "attention_position" else (kind,))
                for seed in self.seeds:
                    checkpoint = str(ablation_root / "models" / arm /
                                     f"seed{seed}" / "best.pt")
                    path = ablation_root / "evaluations" / "val" / arm / f"seed{seed}"
                    self._stage(
                        f"ablation/validation/{arm}/seed{seed}",
                        {"arm": arm, "seed": seed},
                        lambda path=path, seed=seed, config_path=config_path,
                               arm_methods=arm_methods, kind=kind, checkpoint=checkpoint:
                            self._eval_check(path, "val", arm_methods, seed, config_path,
                                             {kind: checkpoint}, None, holes, 0.0,
                                             self.limit, None),
                        lambda path=path, seed=seed, config_path=config_path,
                               arm_methods=arm_methods, kind=kind, checkpoint=checkpoint:
                            self._run_eval(path, "val", arm_methods, seed, config_path,
                                           {kind: checkpoint}, holes=holes,
                                           limit=self.limit),
                    )
                threshold_arm = ablation_root / "thresholds" / f"{arm}.json"
                validation_dirs = [ablation_root / "evaluations" / "val" / arm /
                                   f"seed{seed}" for seed in self.seeds]
                self._stage(
                    f"ablation/thresholds/{arm}",
                    {"arm": arm, "validation": [str(p) for p in validation_dirs]},
                    lambda threshold_arm=threshold_arm, arm_methods=arm_methods,
                           config_path=config_path:
                        self._threshold_check(threshold_arm, arm_methods,
                                              config_path, self.seeds, self.smoke),
                    lambda threshold_arm=threshold_arm, config_path=config_path,
                           validation_dirs=validation_dirs:
                        select_thresholds([str(p) for p in validation_dirs],
                                          str(config_path), str(threshold_arm),
                                          draft=self.smoke, allow_subset=True),
                )
                for seed in self.seeds:
                    checkpoint = str(ablation_root / "models" / arm /
                                     f"seed{seed}" / "best.pt")
                    path = ablation_root / "evaluations" / "test" / arm / f"seed{seed}"
                    self._stage(
                        f"ablation/test/{arm}/seed{seed}",
                        {"arm": arm, "seed": seed, "threshold": str(threshold_arm)},
                        lambda path=path, seed=seed, config_path=config_path,
                               arm_methods=arm_methods, kind=kind,
                               checkpoint=checkpoint, threshold_arm=threshold_arm:
                            self._eval_check(path, "test", arm_methods, seed, config_path,
                                             {kind: checkpoint}, threshold_arm, holes,
                                             0.0, self.limit, None),
                        lambda path=path, seed=seed, config_path=config_path,
                               arm_methods=arm_methods, kind=kind,
                               checkpoint=checkpoint, threshold_arm=threshold_arm:
                            self._run_eval(path, "test", arm_methods, seed, config_path,
                                           {kind: checkpoint}, threshold_arm,
                                           holes=holes, limit=self.limit),
                    )

            self._stage(
                "ablation/report", {"seeds": self.seeds, "split": "test"},
                lambda: self._ablation_summary_check(ablation_root, "test"),
                lambda: report_arms(str(ablation_root), "test",
                                    list(self.seeds), self.limit),
            )

            selection = robust_root / "case_selection.json"
            self._stage(
                "robustness/case_selection",
                {"ids_sha256": stable_hash(robust_ids)},
                lambda: self._robust_ids_check(selection, robust_ids),
                lambda: save_json(selection, {
                    "manifest_sha256": self.identity["manifest_sha256"],
                    "selection_rule": "first sample ID for every independent test parent",
                    "sample_ids": robust_ids,
                    "parent_count": len(robust_ids),
                }),
            )
            robust_rows = []
            for count_holes in (2, 4, 8, 16):
                for noise in (0.0, 0.05, 0.10, 0.20):
                    label = f"holes{count_holes:02d}_noise{round(noise * 100):02d}"
                    condition_dirs = []
                    for seed in self.seeds:
                        path = robust_root / label / f"seed{seed}"
                        condition_dirs.append(path)
                        checkpoints = {
                            kind: str(self.out / "models" / kind /
                                      f"seed{seed}" / "best.pt")
                            for kind in ("geovae", "unet", "base_vae", "cgan")
                            if kind in self.robust_methods or
                               (kind == "geovae" and
                                set(self.robust_methods) &
                                {"geovae_latent", "geovae_joint"})
                        }
                        self._stage(
                            f"robustness/{label}/seed{seed}",
                            {"holes": count_holes, "noise": noise, "seed": seed,
                             "ids_sha256": stable_hash(robust_ids)},
                            lambda path=path, seed=seed, checkpoints=checkpoints,
                                   count_holes=count_holes, noise=noise:
                                self._eval_check(path, "test", tuple(self.robust_methods),
                                                 seed, self.config_path, checkpoints,
                                                 threshold_path, count_holes, noise,
                                                 0, robust_ids),
                            lambda path=path, seed=seed, checkpoints=checkpoints,
                                   count_holes=count_holes, noise=noise:
                                self._run_eval(path, "test", tuple(self.robust_methods),
                                               seed, self.config_path, checkpoints,
                                               threshold_path, count_holes, noise,
                                               0, robust_ids),
                        )
                    summary_root = robust_root / label / "summary"
                    self._stage(
                        f"robustness/{label}/report",
                        {"evaluations": [str(path) for path in condition_dirs]},
                        lambda summary_root=summary_root:
                            self._report_check(summary_root,
                                               tuple(self.robust_methods), self.seeds,
                                               len(robust_ids), "test", True),
                        lambda condition_dirs=condition_dirs, summary_root=summary_root:
                            report([str(path) for path in condition_dirs],
                                   str(summary_root), draft=True,
                                   required_methods=tuple(self.robust_methods)),
                    )
                    summary_file = summary_root / "summary.csv"
                    if not self.dry_run:
                        with summary_file.open(newline="", encoding="utf-8") as stream:
                            for row in csv.DictReader(stream):
                                robust_rows.append({
                                    "num_holes": count_holes, "noise_rate": noise,
                                    "case_policy": "one variant per test parent",
                                    **row,
                                })
            robust_summary = robust_root / "summary.csv"

            def check_robust_summary() -> list[Path] | None:
                if not robust_summary.is_file():
                    return None
                with robust_summary.open(newline="", encoding="utf-8") as stream:
                    previous = list(csv.DictReader(stream))
                expected = [{key: str(value) for key, value in row.items()}
                            for row in robust_rows]
                if previous != expected:
                    raise RuntimeError("Existing robustness summary differs from verified condition reports")
                return [robust_summary]

            def write_robust_summary() -> None:
                robust_summary.parent.mkdir(parents=True, exist_ok=True)
                with robust_summary.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(robust_rows[0]))
                    writer.writeheader()
                    writer.writerows(robust_rows)

            self._stage("robustness/combined_summary",
                        {"condition_count": 16, "seeds": self.seeds},
                        check_robust_summary, write_robust_summary)

            self._stage("handoff", {"report": str(report_root)},
                        self._handoff_check, self._write_handoff)
            if not self.dry_run:
                save_json(self.out / "run_status.json", {
                    "status": "complete", "identity_sha256": self.identity_sha,
                    "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   time.gmtime()),
                    "handoff": str(self.out / "handoff.json"),
                })
            return self.out / "handoff.json"
        except Exception as error:
            if not self.dry_run:
                save_json(self.out / "run_status.json", {
                    "status": "failed", "failed_stage": self.failed_stage,
                    "error_type": type(error).__name__, "message": str(error),
                    "identity_sha256": self.identity_sha,
                    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime()),
                })
            raise

    # Modular entry points share the same identity and stage markers as the
    # original one-command workflow.  A prerequisite is inspected, never run.
    def _require_stage(self, name: str, *artifacts: Path) -> None:
        if self.dry_run:
            return
        marker = self.out / "stages" / (name.replace("/", "__") + ".json")
        if not marker.is_file() or _json(marker).get("status") != "complete":
            raise RuntimeError(
                f"Missing prerequisite stage '{name}'. Run its module first; "
                "no expensive prerequisite will start automatically.")
        for path in artifacts:
            if not path.is_file():
                raise RuntimeError(f"Prerequisite '{name}' lacks artifact: {path}")

    def _require_prepare(self) -> None:
        for stage in ("preflight", "protocol_archive", "cuda_or_cpu_batch"):
            self._require_stage(stage)
        if not self.smoke:
            self._require_stage("classical_one_case")

    def _require_tune(self) -> Path:
        selected = self.out / "tuning" / "selected_config.json"
        self._require_stage("validation_hyperparameters", selected,
                            self.out / "tuning" / "selection.json")
        if not self.dry_run:
            tuned = load_config(selected)
            if training_config_hash(tuned) != training_config_hash(self.config):
                raise RuntimeError("Tuned config changed training parameters")
        return selected

    def _model_checkpoint(self, kind: str, seed: int) -> str:
        path = self.out / "models" / kind / f"seed{seed}" / "best.pt"
        self._require_stage(f"train/{kind}/seed{seed}", path)
        return str(path)

    def _main_checkpoints(self, seed: int) -> dict[str, str]:
        return {kind: self._model_checkpoint(kind, seed)
                for kind in ("geovae", "base_vae", "unet", "cgan")}

    def _arm_config(self, arm: str, evaluation: bool = False) -> Path:
        root = self.out / "ablation"
        self._require_stage("ablation/configs")
        path = root / ("evaluation_configs" if evaluation else "configs") / f"{arm}.json"
        if evaluation:
            self._require_stage("ablation/evaluation_configs", path)
        elif not self.dry_run and not path.is_file():
            raise RuntimeError(f"Missing ablation training config: {path}")
        return path

    def _ensure_arm_eval_configs(self) -> None:
        from repro_ablation import ARMS
        selected = self._require_tune()
        root = self.out / "ablation"
        self._require_stage("ablation/configs")
        paths = [root / "evaluation_configs" / f"{arm}.json" for arm in ARMS]

        def expected(arm: str) -> dict:
            base = load_config(root / "configs" / f"{arm}.json")
            tuned = load_config(selected)
            for key in EVALUATION_ONLY_CONFIG_KEYS:
                if key in tuned:
                    base[key] = tuned[key]
            return base

        def check() -> list[Path] | None:
            if not all(path.is_file() for path in paths):
                return None
            for arm, path in zip(ARMS, paths):
                if load_config(path) != expected(arm):
                    raise RuntimeError(f"Ablation evaluation config changed: {arm}")
            return paths

        def write() -> None:
            for arm, path in zip(ARMS, paths):
                save_json(path, expected(arm))

        self._stage("ablation/evaluation_configs",
                    {"selection_sha256": sha256_file(selected) if not self.dry_run else "planned"},
                    check, write)

    def _mark_partial(self, module: str, filtered: bool) -> None:
        if self.dry_run:
            return
        save_json(self.out / "run_status.json", {
            "status": "partial", "module": module, "filtered": filtered,
            "identity_sha256": self.identity_sha,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    def _run_prepare_module(self) -> None:
        self._stage("preflight", {"cuda": not self.smoke},
                    self._preflight, self._run_preflight)
        if not self.smoke:
            self._stage("classical_one_case",
                        {"methods": ["ik", "mps"], "split": "val"},
                        self._baseline_check, self._run_baseline_check)
        self._stage("protocol_archive",
                    {"manifest": self.identity["manifest_sha256"]},
                    self._protocol_archive_check, self._run_protocol_archive)
        self._stage("cuda_or_cpu_batch",
                    {"device": "cpu" if self.smoke else "cuda"},
                    self._smoke_check, self._run_smoke)

    def _run_train_module(self, seed_filter: int | None,
                          model_filter: str | None) -> None:
        self._require_prepare()
        for seed in self.seeds:
            if seed_filter is not None and seed != seed_filter:
                continue
            for kind in ("geovae", "base_vae", "unet", "cgan"):
                if model_filter is not None and kind != model_filter:
                    continue
                folder = self.out / "models" / kind / f"seed{seed}"
                self._stage(
                    f"train/{kind}/seed{seed}",
                    {"seed": seed, "kind": kind, "config": self.identity["config_sha256"]},
                    lambda folder=folder, kind=kind, seed=seed:
                        self._train_check(folder, kind, seed, self.config_path),
                    lambda folder=folder, kind=kind, seed=seed:
                        self._train_one(folder, kind, seed, self.config_path),
                )

    def _run_tune_module(self) -> None:
        self._require_prepare()
        checkpoint = self._model_checkpoint("geovae", self.seeds[0])
        from repro_hyper_tuning import tune
        root = self.out / "tuning"
        selection, selected = root / "selection.json", root / "selected_config.json"

        def check() -> list[Path] | None:
            if not selection.is_file() or not selected.is_file():
                return None
            record = _json(selection)
            if record.get("status") != "validation_selected":
                raise RuntimeError("Hyperparameter selection is not complete")
            if training_config_hash(load_config(selected)) != training_config_hash(self.config):
                raise RuntimeError("Tuning changed model training settings")
            if record.get("manifest_sha256") != self.identity["manifest_sha256"]:
                raise RuntimeError("Tuning used another data protocol")
            if record.get("checkpoint_sha256") != sha256_file(checkpoint):
                raise RuntimeError("Tuning used another GeoVAE checkpoint")
            return [selection, selected]

        self._stage("validation_hyperparameters",
                    {"checkpoint": sha256_file(checkpoint) if not self.dry_run else "planned",
                     "base_config": self.identity["config_sha256"]},
                    check,
                    lambda: tune(str(self.manifest), str(self.config_path),
                                 str(root), checkpoint, dry_run=False))

    def _run_validate_module(self, seed_filter: int | None) -> None:
        from repro_tuning import select_thresholds
        self._require_prepare()
        selected = self._require_tune()
        methods = tuple(self.methods)
        holes = int(self.config["num_holes"])
        for seed in self.seeds:
            if seed_filter is not None and seed != seed_filter:
                continue
            checkpoints = self._main_checkpoints(seed)
            path = self.out / "validation" / f"seed{seed}"
            self._stage(
                f"validation/seed{seed}",
                {"methods": methods, "seed": seed, "checkpoints": checkpoints,
                 "selected_config_sha256": sha256_file(selected) if not self.dry_run else "planned"},
                lambda path=path, seed=seed, checkpoints=checkpoints:
                    self._eval_check(path, "val", methods, seed, selected,
                                     checkpoints, None, holes, 0.0, self.limit, None),
                lambda path=path, seed=seed, checkpoints=checkpoints:
                    self._run_eval(path, "val", methods, seed, selected,
                                   checkpoints, holes=holes, limit=self.limit),
            )
        if seed_filter is not None:
            return
        for seed in self.seeds:
            self._require_stage(f"validation/seed{seed}",
                                self.out / "validation" / f"seed{seed}" / "evaluation.json")
        paths = [self.out / "validation" / f"seed{seed}" for seed in self.seeds]
        threshold_path = self.out / "thresholds.json"
        self._stage(
            "validation_thresholds",
            {"evaluations": [str(path) for path in paths],
             "selected_config_sha256": sha256_file(selected) if not self.dry_run else "planned"},
            lambda: self._threshold_check(threshold_path, methods,
                                          selected, self.seeds, self.smoke),
            lambda: select_thresholds([str(path) for path in paths],
                                      str(selected), str(threshold_path), draft=self.smoke),
        )

    def _run_test_module(self, seed_filter: int | None) -> None:
        from repro_figures import figures
        self._require_prepare()
        selected = self._require_tune()
        thresholds = self.out / "thresholds.json"
        self._require_stage("validation_thresholds", thresholds)
        if not self.smoke:
            for arm in ("base_no_observed", "base_observed",
                        "attention_no_position", "attention_position"):
                self._require_stage(f"ablation/thresholds/{arm}",
                                    self.out / "ablation" / "thresholds" / f"{arm}.json")
        methods = tuple(self.methods)
        holes = int(self.config["num_holes"])
        for seed in self.seeds:
            if seed_filter is not None and seed != seed_filter:
                continue
            checkpoints = self._main_checkpoints(seed)
            path = self.out / "test" / f"seed{seed}"
            self._stage(
                f"sealed_test/seed{seed}",
                {"methods": methods, "seed": seed,
                 "threshold_sha256": sha256_file(thresholds) if not self.dry_run else "planned",
                 "selected_config_sha256": sha256_file(selected) if not self.dry_run else "planned"},
                lambda path=path, seed=seed, checkpoints=checkpoints:
                    self._eval_check(path, "test", methods, seed, selected,
                                     checkpoints, thresholds, holes, 0.0, 0, None),
                lambda path=path, seed=seed, checkpoints=checkpoints:
                    self._run_eval(path, "test", methods, seed, selected,
                                   checkpoints, thresholds, holes=holes, limit=0),
            )
        if seed_filter is not None:
            return
        for seed in self.seeds:
            self._require_stage(f"sealed_test/seed{seed}",
                                self.out / "test" / f"seed{seed}" / "evaluation.json")
        paths = [self.out / "test" / f"seed{seed}" for seed in self.seeds]
        report_root = self.out / "report"
        self._stage(
            "main_report", {"test": [str(path) for path in paths]},
            lambda: self._report_check(report_root, methods, self.seeds,
                                       self.limit or 750, "test", self.smoke),
            lambda: report([str(path) for path in paths], str(report_root),
                           draft=self.smoke, required_methods=methods),
        )
        figure_root = self.out / "figures"
        figure_methods = ("idw", "unet", "geovae")
        self._stage(
            "manuscript_figures",
            {"report": str(report_root), "methods": figure_methods},
            lambda: self._figure_check(figure_root),
            lambda: figures(str(paths[0]), str(report_root),
                            str(selected), str(figure_root),
                            list(figure_methods), draft=self.smoke),
        )

    def _run_ablation_train_module(self, seed_filter: int | None,
                                   arm_filter: str | None) -> None:
        from repro_ablation import ARMS, prepare_configs
        self._require_prepare()
        root = self.out / "ablation"
        self._stage("ablation/configs", {"base": self.identity["config_sha256"]},
                    lambda: self._ablation_config_check(root),
                    lambda: prepare_configs(str(self.config_path), str(root)))
        for arm, (kind, _) in ARMS.items():
            if arm_filter is not None and arm != arm_filter:
                continue
            config_path = root / "configs" / f"{arm}.json"
            for seed in self.seeds:
                if seed_filter is not None and seed != seed_filter:
                    continue
                folder = root / "models" / arm / f"seed{seed}"
                reuse = (Path(self._model_checkpoint(kind, seed)).parent
                         if arm in ("base_observed", "attention_position") else None)
                self._stage(
                    f"ablation/train/{arm}/seed{seed}",
                    {"arm": arm, "seed": seed,
                     "config_sha256": stable_hash(load_config(config_path))
                     if not self.dry_run else "planned"},
                    lambda folder=folder, kind=kind, seed=seed, config_path=config_path:
                        self._train_check(folder, kind, seed, config_path),
                    lambda folder=folder, kind=kind, seed=seed,
                           config_path=config_path, reuse=reuse:
                        self._train_one(folder, kind, seed, config_path, reuse),
                )

    def _run_ablation_validate_module(self, seed_filter: int | None,
                                      arm_filter: str | None) -> None:
        from repro_ablation import ARMS
        from repro_tuning import select_thresholds
        self._require_prepare()
        self._ensure_arm_eval_configs()
        root = self.out / "ablation"
        holes = int(self.config["num_holes"])
        for arm, (kind, _) in ARMS.items():
            if arm_filter is not None and arm != arm_filter:
                continue
            config_path = self._arm_config(arm, evaluation=True)
            methods = ((kind, "geovae_latent", "geovae_joint")
                       if arm == "attention_position" else (kind,))
            for seed in self.seeds:
                if seed_filter is not None and seed != seed_filter:
                    continue
                checkpoint = root / "models" / arm / f"seed{seed}" / "best.pt"
                self._require_stage(f"ablation/train/{arm}/seed{seed}", checkpoint)
                path = root / "evaluations" / "val" / arm / f"seed{seed}"
                checkpoints = {kind: str(checkpoint)}
                self._stage(
                    f"ablation/validation/{arm}/seed{seed}",
                    {"arm": arm, "seed": seed,
                     "selected_config_sha256": sha256_file(config_path)
                     if not self.dry_run else "planned"},
                    lambda path=path, seed=seed, config_path=config_path,
                           methods=methods, checkpoints=checkpoints:
                        self._eval_check(path, "val", methods, seed, config_path,
                                         checkpoints, None, holes, 0.0, self.limit, None),
                    lambda path=path, seed=seed, config_path=config_path,
                           methods=methods, checkpoints=checkpoints:
                        self._run_eval(path, "val", methods, seed, config_path,
                                       checkpoints, holes=holes, limit=self.limit),
                )
            if seed_filter is not None:
                continue
            paths = [root / "evaluations" / "val" / arm / f"seed{seed}"
                     for seed in self.seeds]
            for seed, path in zip(self.seeds, paths):
                self._require_stage(f"ablation/validation/{arm}/seed{seed}",
                                    path / "evaluation.json")
            threshold_path = root / "thresholds" / f"{arm}.json"
            self._stage(
                f"ablation/thresholds/{arm}",
                {"arm": arm, "validation": [str(path) for path in paths],
                 "config_sha256": sha256_file(config_path) if not self.dry_run else "planned"},
                lambda threshold_path=threshold_path, methods=methods,
                       config_path=config_path:
                    self._threshold_check(threshold_path, methods, config_path,
                                          self.seeds, self.smoke),
                lambda threshold_path=threshold_path, config_path=config_path,
                       paths=paths:
                    select_thresholds([str(path) for path in paths],
                                      str(config_path), str(threshold_path),
                                      draft=self.smoke, allow_subset=True),
            )

    def _run_ablation_test_module(self, seed_filter: int | None,
                                  arm_filter: str | None) -> None:
        from repro_ablation import ARMS, report_arms
        self._require_prepare()
        self._require_stage("main_report", self.out / "report" / "report.json")
        root = self.out / "ablation"
        holes = int(self.config["num_holes"])
        for arm, (kind, _) in ARMS.items():
            if arm_filter is not None and arm != arm_filter:
                continue
            config_path = self._arm_config(arm, evaluation=True)
            methods = ((kind, "geovae_latent", "geovae_joint")
                       if arm == "attention_position" else (kind,))
            threshold_path = root / "thresholds" / f"{arm}.json"
            self._require_stage(f"ablation/thresholds/{arm}", threshold_path)
            for seed in self.seeds:
                if seed_filter is not None and seed != seed_filter:
                    continue
                checkpoint = root / "models" / arm / f"seed{seed}" / "best.pt"
                self._require_stage(f"ablation/train/{arm}/seed{seed}", checkpoint)
                path = root / "evaluations" / "test" / arm / f"seed{seed}"
                checkpoints = {kind: str(checkpoint)}
                self._stage(
                    f"ablation/test/{arm}/seed{seed}",
                    {"arm": arm, "seed": seed,
                     "threshold_sha256": sha256_file(threshold_path)
                     if not self.dry_run else "planned"},
                    lambda path=path, seed=seed, config_path=config_path,
                           methods=methods, checkpoints=checkpoints,
                           threshold_path=threshold_path:
                        self._eval_check(path, "test", methods, seed, config_path,
                                         checkpoints, threshold_path, holes, 0.0, 0, None),
                    lambda path=path, seed=seed, config_path=config_path,
                           methods=methods, checkpoints=checkpoints,
                           threshold_path=threshold_path:
                        self._run_eval(path, "test", methods, seed, config_path,
                                       checkpoints, threshold_path, holes=holes, limit=0),
                )
        if arm_filter is not None or seed_filter is not None:
            return
        for arm in ARMS:
            for seed in self.seeds:
                self._require_stage(f"ablation/test/{arm}/seed{seed}",
                                    root / "evaluations" / "test" / arm /
                                    f"seed{seed}" / "evaluation.json")
        self._stage(
            "ablation/report", {"seeds": self.seeds, "split": "test"},
            lambda: self._ablation_summary_check(root, "test"),
            lambda: report_arms(str(root), "test", list(self.seeds), self.limit),
        )

    def _run_robustness_module(self, seed_filter: int | None,
                               holes_filter: int | None,
                               noise_filter: float | None) -> None:
        selected = self._require_tune()
        thresholds = self.out / "thresholds.json"
        self._require_stage("main_report", self.out / "report" / "report.json")
        self._require_stage("validation_thresholds", thresholds)
        root = self.out / "robustness"
        ids = self._robust_ids()
        selection = root / "case_selection.json"
        self._stage(
            "robustness/case_selection", {"ids_sha256": stable_hash(ids)},
            lambda: self._robust_ids_check(selection, ids),
            lambda: save_json(selection, {
                "manifest_sha256": self.identity["manifest_sha256"],
                "selection_rule": "first sample ID for every independent test parent",
                "sample_ids": ids, "parent_count": len(ids),
            }),
        )
        methods = tuple(self.robust_methods)
        all_conditions = [(holes, noise)
                          for holes in (2, 4, 8, 16)
                          for noise in (0.0, 0.05, 0.10, 0.20)]
        for holes, noise in all_conditions:
            if holes_filter is not None and holes != holes_filter:
                continue
            if noise_filter is not None and noise != noise_filter:
                continue
            label = f"holes{holes:02d}_noise{round(noise * 100):02d}"
            paths = []
            for seed in self.seeds:
                if seed_filter is not None and seed != seed_filter:
                    continue
                checkpoints = {
                    kind: self._model_checkpoint(kind, seed)
                    for kind in ("geovae", "unet", "base_vae", "cgan")
                    if kind in methods or
                    (kind == "geovae" and set(methods) &
                     {"geovae_latent", "geovae_joint"})
                }
                path = root / label / f"seed{seed}"
                paths.append(path)
                self._stage(
                    f"robustness/{label}/seed{seed}",
                    {"holes": holes, "noise": noise, "seed": seed,
                     "ids_sha256": stable_hash(ids),
                     "selected_config_sha256": sha256_file(selected)
                     if not self.dry_run else "planned"},
                    lambda path=path, seed=seed, checkpoints=checkpoints,
                           holes=holes, noise=noise:
                        self._eval_check(path, "test", methods, seed, selected,
                                         checkpoints, thresholds, holes, noise, 0, ids),
                    lambda path=path, seed=seed, checkpoints=checkpoints,
                           holes=holes, noise=noise:
                        self._run_eval(path, "test", methods, seed, selected,
                                       checkpoints, thresholds, holes, noise, 0, ids),
                )
            if seed_filter is not None:
                continue
            paths = [root / label / f"seed{seed}" for seed in self.seeds]
            for seed, path in zip(self.seeds, paths):
                self._require_stage(f"robustness/{label}/seed{seed}",
                                    path / "evaluation.json")
            summary_root = root / label / "summary"
            self._stage(
                f"robustness/{label}/report",
                {"evaluations": [str(path) for path in paths]},
                lambda summary_root=summary_root:
                    self._report_check(summary_root, methods, self.seeds,
                                       len(ids), "test", True),
                lambda paths=paths, summary_root=summary_root:
                    report([str(path) for path in paths], str(summary_root),
                           draft=True, required_methods=methods),
            )
        if seed_filter is not None or holes_filter is not None or noise_filter is not None:
            return
        combined = root / "summary.csv"

        def collect_rows() -> list[dict]:
            rows = []
            for holes, noise in all_conditions:
                label = f"holes{holes:02d}_noise{round(noise * 100):02d}"
                summary = root / label / "summary" / "summary.csv"
                self._require_stage(f"robustness/{label}/report", summary)
                if self.dry_run:
                    continue
                with summary.open(newline="", encoding="utf-8") as stream:
                    for row in csv.DictReader(stream):
                        rows.append({"num_holes": holes, "noise_rate": noise,
                                     "case_policy": "one variant per test parent", **row})
            return rows

        rows = collect_rows()

        def check_combined() -> list[Path] | None:
            if not combined.is_file():
                return None
            with combined.open(newline="", encoding="utf-8") as stream:
                previous = list(csv.DictReader(stream))
            expected = [{key: str(value) for key, value in row.items()}
                        for row in rows]
            if previous != expected:
                raise RuntimeError("Robustness combined summary differs from condition reports")
            return [combined]

        def write_combined() -> None:
            if not rows:
                raise RuntimeError("No robustness rows to summarize")
            combined.parent.mkdir(parents=True, exist_ok=True)
            with combined.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        self._stage("robustness/combined_summary",
                    {"condition_count": len(all_conditions), "seeds": self.seeds},
                    check_combined, write_combined)

    def _run_finalize_module(self) -> None:
        from repro_ablation import ARMS
        required = ["validation_thresholds", "main_report", "manuscript_figures",
                    "ablation/report", "robustness/combined_summary"]
        for name in required:
            self._require_stage(name)
        evaluation_paths = []
        for seed in self.seeds:
            for split, prefix in (("val", "validation"), ("test", "sealed_test")):
                self._require_stage(f"{prefix}/seed{seed}")
                evaluation_paths.append(self.out /
                                        ("validation" if split == "val" else "test") /
                                        f"seed{seed}")
            for arm in ARMS:
                for split, prefix in (("val", "validation"), ("test", "test")):
                    self._require_stage(f"ablation/{prefix}/{arm}/seed{seed}")
                    evaluation_paths.append(self.out / "ablation" / "evaluations" /
                                            split / arm / f"seed{seed}")
            for holes in (2, 4, 8, 16):
                for noise in (0.0, 0.05, 0.10, 0.20):
                    label = f"holes{holes:02d}_noise{round(noise * 100):02d}"
                    self._require_stage(f"robustness/{label}/seed{seed}")
                    evaluation_paths.append(self.out / "robustness" / label /
                                            f"seed{seed}")
        audit = self.out / "integrity_audit.json"
        verified: dict[str, object] = {}

        def verify_all() -> None:
            if verified:
                return
            self._full_archive_check()
            raw = self._verify_raw_inventory()
            evaluated = 0
            for path in evaluation_paths:
                _read_run(path)  # Includes every prediction and raw-prediction SHA.
                evaluated += 1
                if evaluated % 10 == 0:
                    self._event("PROGRESS", "final_integrity",
                                f"{evaluated}/{len(evaluation_paths)} evaluations")
            cache_root = self.out / "prediction_cache"
            cache_summary = verify_prediction_cache(cache_root)
            for record_path in cache_root.glob("*/*.json"):
                record = _json(record_path)
                identity = record.get("identity")
                if identity.get("manifest_sha256") != self.identity["manifest_sha256"]:
                    raise RuntimeError(f"Prediction cache protocol differs: {record_path}")
                relative = record.get("source_prediction", "")
                source = (self.out / relative).resolve()
                source_eval = source.parents[2] / "evaluation.json"
                if not source_eval.is_file() or not _json(source_eval).get("complete"):
                    raise RuntimeError(f"Prediction cache source evaluation incomplete: {source}")
                if (_json(source_eval).get("journal_identity_sha256") !=
                        record.get("source_evaluation_identity_sha256")):
                    raise RuntimeError(f"Prediction cache provenance differs: {record_path}")
            verified["raw"] = raw
            verified["evaluated"] = evaluated
            verified["cache"] = cache_summary

        def audit_check() -> list[Path] | None:
            verify_all()
            if not audit.is_file():
                return None
            record = _json(audit)
            if record.get("identity_sha256") != self.identity_sha:
                raise RuntimeError("Integrity audit belongs to a different run")
            if record.get("evaluation_directories") != len(evaluation_paths):
                raise RuntimeError("Integrity audit has wrong evaluation count")
            return [audit]

        def audit_run() -> None:
            save_json(audit, {
                "status": "passed", "identity_sha256": self.identity_sha,
                "protocol_grid_count": len(self.protocol["samples"]),
                "raw_data_verification": verified["raw"], "scope": "synthetic_protocol_and_predictions_only",
                "evaluation_directories": verified["evaluated"],
                "prediction_integrity": "every recorded prediction SHA256 rechecked",
                "prediction_cache": verified["cache"],
            })

        self._stage("final_integrity",
                    {"identity": self.identity_sha,
                     "evaluations": len(evaluation_paths)},
                    audit_check, audit_run)
        self._stage("handoff", {"report": str(self.out / "report")},
                    self._handoff_check, self._write_handoff)
        if not self.dry_run:
            save_json(self.out / "run_status.json", {
                "status": "complete", "identity_sha256": self.identity_sha,
                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "handoff": str(self.out / "handoff.json"),
            })

    def run_module(self, module: str, *, seed: int | None = None,
                   model: str | None = None, arm: str | None = None,
                   holes: int | None = None, noise_rate: float | None = None) -> Path:
        """Run one explicit module, or all modules in the preregistered order."""
        modules = ("prepare", "train", "tune", "validate", "ablation-train",
                   "ablation-validate", "test", "ablation-test", "robustness",
                   "finalize") if module == "all" else (module,)
        filtered = any(value is not None for value in
                       (seed, model, arm, holes, noise_rate))
        try:
            for current in modules:
                if current == "prepare":
                    self._run_prepare_module()
                elif current == "train":
                    self._run_train_module(seed, model)
                elif current == "tune":
                    self._run_tune_module()
                elif current == "validate":
                    self._run_validate_module(seed)
                elif current == "ablation-train":
                    self._run_ablation_train_module(seed, arm)
                elif current == "ablation-validate":
                    self._run_ablation_validate_module(seed, arm)
                elif current == "test":
                    self._run_test_module(seed)
                elif current == "ablation-test":
                    self._run_ablation_test_module(seed, arm)
                elif current == "robustness":
                    self._run_robustness_module(seed, holes, noise_rate)
                elif current == "finalize":
                    self._run_finalize_module()
                else:
                    raise ValueError(f"Unknown GeoVAE module: {current}")
                if current != "finalize":
                    self._mark_partial(current, filtered)
            return self.out / ("handoff.json" if "finalize" in modules else "run_status.json")
        except Exception as error:
            if not self.dry_run:
                save_json(self.out / "run_status.json", {
                    "status": "failed", "failed_stage": self.failed_stage,
                    "failed_module": current, "error_type": type(error).__name__,
                    "message": str(error), "identity_sha256": self.identity_sha,
                    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
            raise


def _automatic_output(manifest: str, config: str, smoke: bool) -> Path:
    """Keep an old default run intact when the code or protocol changes."""
    base = ROOT / "runs" / ("offline_smoke" if smoke else "full_offline")
    identity_file = base / "run_identity.json"
    if not identity_file.is_file() or not Path(manifest).is_file():
        return base
    preview = OfflineRun(manifest, config, base, smoke=smoke, dry_run=True)
    if stable_hash(_json(identity_file)) == preview.identity_sha:
        return base
    return base.parent / f"{base.name}_{preview.identity_sha[:16]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run GeoVAE modules offline; --all runs the complete ordered workflow.")
    stages = parser.add_mutually_exclusive_group(required=True)
    for flag in ("prepare", "tune", "validate", "ablation-train",
                 "ablation-validate", "test", "ablation-test",
                 "robustness", "finalize", "all"):
        stages.add_argument("--" + flag, action="store_true")
    stages.add_argument("--train", "--pretrain", dest="train", action="store_true",
                        help="Train all four main models; --pretrain is an alias, "
                             "not a separate pretraining objective")
    parser.add_argument("--manifest", default=None,
                        help="Frozen protocol manifest; default: protocol/manifest.json")
    parser.add_argument("--config", default=None,
                        help="Experiment config; default: config/revision_v1.yaml")
    parser.add_argument("--out", default=str(ROOT / "runs" / "esi_revision_v2"),
                        help="New output directory shared by every module; default: runs/esi_revision_v2")
    parser.add_argument("--smoke", action="store_true",
                        help="Small CPU draft workflow, never manuscript evidence")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stages without training or writing results")
    parser.add_argument("--verbose", action="store_true",
                        help="Also stream detailed per-epoch logs to the terminal")
    parser.add_argument("--seed", type=int, choices=SEEDS,
                        help="Run only this training/evaluation seed")
    parser.add_argument("--model", choices=("geovae", "base_vae", "unet", "cgan"),
                        help="With --train, run only this model")
    parser.add_argument("--arm", choices=("base_no_observed", "base_observed",
                                          "attention_no_position", "attention_position"),
                        help="With an ablation module, run only this arm")
    parser.add_argument("--holes", type=int, choices=(2, 4, 8, 16),
                        help="With --robustness, select one drillhole count")
    parser.add_argument("--noise-rate", type=float,
                        choices=(0.0, 0.05, 0.10, 0.20),
                        help="With --robustness, select one label-flip rate")
    args = parser.parse_args(argv)
    module = next(name for name in ("prepare", "train", "tune", "validate",
                                   "ablation-train", "ablation-validate", "test",
                                   "ablation-test", "robustness", "finalize", "all")
                  if getattr(args, name.replace("-", "_")))
    if not args.dry_run:
        from revision_numerics import configure_evaluation, record_training_policy
        phases = {"validate": "validation", "ablation-validate": "ablation_validation",
                  "ablation-test": "ablation_test"}
        if module in {"tune", "validate", "test", "ablation-validate",
                      "ablation-test", "robustness", "finalize"}:
            configure_evaluation(args.out, phase=phases.get(module, module))
        else:
            record_training_policy(args.out, phase=module.replace("-", "_"))
    if args.model is not None and module != "train":
        parser.error("--model requires --train or --pretrain")
    if args.arm is not None and module not in {
            "ablation-train", "ablation-validate", "ablation-test"}:
        parser.error("--arm requires an ablation module")
    if (args.holes is not None or args.noise_rate is not None) and module != "robustness":
        parser.error("--holes and --noise-rate require --robustness")
    if args.seed is not None and module not in {
            "train", "validate", "test", "ablation-train",
            "ablation-validate", "ablation-test", "robustness"}:
        parser.error("--seed is not supported for this module")
    if args.smoke and args.seed not in (None, 11):
        parser.error("--smoke has only seed 11")
    manifest = args.manifest or (
        str(ROOT / "protocol_smoke" / "manifest.json") if args.smoke else
        str(DEFAULT_MANIFEST))
    config = args.config or (
        str(ROOT / "config" / "smoke16.yaml") if args.smoke else
        str(DEFAULT_CONFIG))
    workflow = OfflineRun(manifest, config, args.out,
                          smoke=args.smoke, dry_run=args.dry_run,
                          prepare_mode=module in ("prepare", "all"),
                          verbose=args.verbose)
    print("[offline] data=public synthetic protocol; restricted raw data not distributed", flush=True)
    print(f"[offline] protocol={workflow.manifest}", flush=True)
    print(f"[offline] output={workflow.out}", flush=True)
    print(f"[offline] mode={'CPU draft' if args.smoke else 'formal CUDA'}",
          flush=True)
    print(f"[offline] module={module}", flush=True)
    print(workflow.run_module(module, seed=args.seed, model=args.model,
                              arm=args.arm, holes=args.holes,
                              noise_rate=args.noise_rate), flush=True)
    return 0


def run_all(argv: list[str] | None = None) -> int:
    """Compatibility hook; the parser now requires an explicit module flag."""
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
