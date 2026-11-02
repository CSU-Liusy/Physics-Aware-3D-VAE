"""Bounded ESI revision workflow. No network operations or raw-data edits.

Public results use the independent synthetic protocol. Retrospective real
predictions always stay in the separate private directory supplied by the user.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from repro_pipeline import ROOT, DEFAULT_CONFIG, evaluate, save_json, sha256_file

SEEDS = (11, 23, 37)
CLASSICAL = ("idw", "rbf", "ik", "mps")


def _classical_shard(arguments: tuple) -> str:
    manifest, config, out, split, thresholds, index, workers = arguments
    from revision_numerics import configure_evaluation
    configure_evaluation(out, phase=f"classical_{split}_worker{index}")
    budget_file = Path(out) / "revision_budget.json"
    if os.name == "posix" and budget_file.exists():
        # Workers form their own process groups. A deadline also stops an MPS
        # executable, rather than leaving it orphaned after its parent exits.
        if os.getpgrp() != os.getpid():
            os.setsid()
        budget = json.loads(budget_file.read_text())
        seconds_left = budget["started_unix"] + budget["hours"] * 3600 - time.time()
        if seconds_left <= 0:
            raise TimeoutError("Experiment budget exhausted before classical prediction")
        def deadline_handler(signum, frame):
            os.killpg(os.getpgrp(), signal.SIGTERM)
        signal.signal(signal.SIGALRM, deadline_handler)
        signal.alarm(max(1, math.ceil(seconds_left)))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    records = json.loads(Path(manifest).read_text(encoding="utf-8"))["samples"]
    ids = [r["sample_id"] for r in records if r["split"] == split][index::workers]
    destination = Path(out) / "classical_precompute" / split / f"shard{index}"
    evaluate(manifest, config, str(destination), split, list(CLASSICAL), {}, 11,
             sample_ids=ids, threshold_file=thresholds,
             prediction_cache_root=str(Path(out) / "prediction_cache"))
    return str(destination / "evaluation.json")


def precompute_classical(manifest: str, config: str, out: str, split: str,
                         thresholds: str | None, workers: int = 2) -> list[str]:
    """Each process has its own evaluation journal and MPS temporary directory."""
    # Set before spawning so NumPy/BLAS import in each child sees the limit.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    arguments = [(manifest, config, out, split, thresholds, i, workers)
                 for i in range(workers)]
    if workers == 1:
        return [_classical_shard(arguments[0])]
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(_classical_shard, arguments))


class RevisionRun:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.started_path = self.out / "revision_budget.json"
        if self.started_path.exists():
            self.budget = json.loads(self.started_path.read_text())
            if self.budget["config_sha256"] != sha256_file(args.config):
                raise RuntimeError("Revision budget belongs to another configuration")
        else:
            self.budget = {"started_unix": time.time(), "hours": args.budget_hours,
                           "config_sha256": sha256_file(args.config)}
            save_json(self.started_path, self.budget)
        # Controller repairs are versioned separately from the immutable training
        # engine. They may not change a checkpoint's training-source identity.
        controller_sha = sha256_file(__file__)
        controller_copy = self.out / "controller_sources" / (controller_sha + ".py")
        controller_copy.parent.mkdir(exist_ok=True)
        if not controller_copy.exists():
            controller_copy.write_bytes(Path(__file__).read_bytes())

    def remaining(self) -> float:
        return self.budget["hours"] * 3600 - (time.time() - self.budget["started_unix"])

    def command(self, name: str, arguments: list[str]) -> None:
        import threading

        if self.remaining() <= 0:
            raise TimeoutError("72-hour experiment budget exhausted; checkpoints preserved")
        print(f"[REVISION] {name}; budget remaining {self.remaining()/3600:.2f} h", flush=True)
        log = self.out / "controller_logs" / f"{name}.log"
        log.parent.mkdir(exist_ok=True)
        with log.open("a", encoding="utf-8", buffering=1) as stream:
            child_environment = dict(getattr(os, "environ", {}))
            child_environment["PYTHONIOENCODING"] = "utf-8"
            process = subprocess.Popen([sys.executable, "-u", *arguments],
                                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1,
                                       env=child_environment,
                                       start_new_session=(os.name == "posix"))
            terminal = sys.stdout
            verbose = bool(getattr(self.args, "verbose", False))
            progress_prefixes = ("[RUN]", "[DONE]", "[SKIP]", "[PARTIAL]",
                                 "[ERROR]", "[PROGRESS]")
            relay_errors = []

            def relay_output():
                try:
                    for line in process.stdout:
                        stream.write(line)
                        if verbose or line.startswith(progress_prefixes):
                            terminal.write(line)
                            terminal.flush()
                except BaseException as error:
                    relay_errors.append(error)

            def stop_owned_child(force=False):
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                elif force:
                    process.kill()
                else:
                    process.terminate()

            def stop_and_reap():
                stop_owned_child()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    stop_owned_child(force=True)
                    process.wait(timeout=30)
                else:
                    if os.name == "posix":
                        # The leader can exit before a descendant that ignored
                        # SIGTERM. Reap the remaining owned group as well.
                        stop_owned_child(force=True)

            # Reading a line must never block the deadline-monitoring thread.
            # The reader is joined before closing the complete on-disk log.
            reader = threading.Thread(target=relay_output, name=f"revision-log-{process.pid}",
                                      daemon=True)
            reader.start()
            try:
                code = process.wait(timeout=max(1, self.remaining()))
            except subprocess.TimeoutExpired:
                stop_and_reap()
                raise TimeoutError("Experiment budget reached; latest completed checkpoints preserved")
            except BaseException:
                stop_and_reap()
                raise
            finally:
                reader.join(timeout=5)
                if reader.is_alive():
                    # An inherited pipe must not leave this task's children or
                    # a reader writing into a closed log after the leader exits.
                    stop_owned_child(force=True)
                    reader.join(timeout=5)
                if reader.is_alive():
                    raise RuntimeError(f"{name} output pipe did not close; inspect {log}")
                process.stdout.close()
            if relay_errors:
                raise RuntimeError(f"{name} output relay failed; inspect {log}") from relay_errors[0]
        if code:
            raise RuntimeError(f"{name} failed (exit {code}); inspect {log}")

    def module(self, flag: str) -> None:
        arguments = ["repro_run_all.py", "--" + flag,
                           "--manifest", self.args.manifest, "--config", self.args.config,
                           "--out", str(self.out)]
        if self.args.verbose:
            arguments.append("--verbose")
        self.command(flag, arguments)

    def uncertainty(self) -> None:
        from revision_diagnostics import uncertainty_run
        config = self.out / "tuning" / "selected_config.json"
        checkpoint = self.out / "models" / "geovae" / "seed11" / "best.pt"
        destination = self.out / "uncertainty"
        for split in ("val", "test"):
            uncertainty_run(self.args.manifest, str(config), str(checkpoint),
                            str(destination), split=split, seed=11,
                            deadline_epoch=self.budget["started_unix"] + self.budget["hours"] * 3600)

    def real_cases(self) -> None:
        # Implementation validates that private_out is outside publication folders.
        from revision_diagnostics import realcases_run
        checkpoints = {kind: str(self.out / "models" / kind / "seed11" / "best.pt")
                       for kind in ("geovae", "base_vae", "unet", "cgan")}
        realcases_run(self.args.manifest,
                      str(self.out / "tuning" / "selected_config.json"), checkpoints,
                      self.args.private_out, str(self.out / "real_aggregate"),
                      data_root=self.args.data_root,
                      threshold_file=str(self.out / "thresholds.json"),
                      deadline_epoch=self.budget["started_unix"] + self.budget["hours"] * 3600)

    def all(self) -> None:
        self.module("prepare")
        self.module("train")
        self.module("tune")
        # Independent MPS/kriging work proceeds on CPU while ablation uses CUDA.
        command = [sys.executable, "-u", str(ROOT / "revision_runner.py"),
                   "--classical", "--split", "val", "--manifest", self.args.manifest,
                   "--config", str(self.out / "tuning" / "selected_config.json"),
                   "--out", str(self.out)]
        log = self.out / "controller_logs" / "classical_val.log"
        with log.open("a", encoding="utf-8", buffering=1) as stream:
            classical = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                         stderr=subprocess.STDOUT)
            try:
                self.module("ablation-train")
                if classical.wait(timeout=max(1, self.remaining())):
                    raise RuntimeError(f"Classical validation failed; inspect {log}")
            finally:
                if classical.poll() is None:
                    classical.terminate()
                    classical.wait(timeout=30)
        self.module("validate")
        self.module("ablation-validate")
        # Every validation choice is now frozen, before any new test prediction.
        self.command("classical_test", ["revision_runner.py", "--classical", "--split", "test",
                     "--manifest", self.args.manifest, "--config",
                     str(self.out / "tuning" / "selected_config.json"), "--out", str(self.out)])
        self.module("test")
        self.module("ablation-test")
        self.uncertainty()
        if self.args.include_real:
            self.real_cases()
        else:
            save_json(self.out / "real_aggregate" / "status.json", {
                "status": "not_available_restricted", "performed": False,
                "reason": "Confidential real inputs are not part of the public package"})
        from revision_robustness import run as robustness_run
        robustness_run(self.args.manifest, str(self.out),
                       str(self.out / "tuning" / "selected_config.json"),
                       deadline_epoch=self.budget["started_unix"] + self.budget["hours"] * 3600)
        from revision_analysis import analyze
        analyze(str(self.out))
        from public_finish import finalize
        finalize(self.out)
        save_json(self.out / "revision_status.json", {
            "status": "public_synthetic_experiments_complete", "restricted_real_data_status": "externally_supplied_aggregate_only" if self.args.include_real else "not_available_restricted", "elapsed_hours":
            (time.time() - self.budget["started_unix"]) / 3600,
            "note": "Manuscript and response verification is a separate delivery gate"})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", "--revision-all", dest="all", action="store_true")
    group.add_argument("--uncertainty", action="store_true")
    group.add_argument("--real-cases", action="store_true")
    group.add_argument("--classical", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--manifest", default=str(ROOT / "protocol" / "manifest.json"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(ROOT / "runs" / "esi_revision_v2"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--private-out", default=None)
    parser.add_argument("--include-real", action="store_true",
                        help="Explicitly include authorized external real data; requires both paths")
    parser.add_argument("--budget-hours", type=float, default=72.)
    parser.add_argument("--smoke", action="store_true",
                        help="Small complete CPU draft, never manuscript evidence")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args(argv)
    if (args.real_cases or args.include_real) and (not args.data_root or not args.private_out):
        parser.error("Real cases require explicit --data-root and absolute --private-out")
    if args.real_cases or args.include_real:
        from revision_diagnostics import assert_private_real_path
        assert_private_real_path(args.private_out)
        if Path(args.data_root).resolve().is_relative_to(ROOT):
            parser.error("Restricted real input data must remain outside the public repository")
    if args.dry_run:
        stages = ("prepare", "train", "tune", "classical_val + ablation-train",
                  "validate", "ablation-validate", "classical_test", "test",
                  "ablation-test", "uncertainty val/test", ("external private real cases" if args.include_real or args.real_cases else "restricted real cases skipped"),
                  "subset robustness", "independent metric reanalysis")
        print("[PLAN] " + " -> ".join(stages if args.all else
              ("uncertainty val/test",) if args.uncertainty else (("external private real cases" if args.include_real or args.real_cases else "restricted real cases skipped"),)))
        print("[PLAN] No files written or experiments started")
        return 0
    if args.smoke:
        if not args.all:
            parser.error("--smoke requires --all")
        from revision_smoke import run as smoke_run
        print(smoke_run(args.out))
        return 0
    if args.classical:
        from revision_numerics import configure_evaluation
        configure_evaluation(args.out, phase="classical_" + args.split)
        thresholds = str(Path(args.out) / "thresholds.json") if args.split == "test" else None
        precompute_classical(args.manifest, args.config, args.out, args.split, thresholds)
        return 0
    runner = RevisionRun(args)
    try:
        if args.all:
            runner.all()
        elif args.uncertainty:
            runner.uncertainty()
        else:
            runner.real_cases()
    except Exception as error:
        save_json(runner.out / "revision_status.json", {
            "status": "budget_exhausted" if isinstance(error, TimeoutError) else "failed",
            "error": str(error), "error_type": type(error).__name__})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
