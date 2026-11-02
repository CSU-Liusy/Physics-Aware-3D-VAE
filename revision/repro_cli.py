"""Command-line interface for the sealed GeoVAE experiment protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from repro_pipeline import DEFAULT_CONFIG, ROOT, doctor, evaluate, train
from repro_report import report

DEFAULT_MANIFEST = ROOT / "protocol" / "manifest.json"


def _checkpoints(items: list[str]) -> dict[str, str]:
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Checkpoint must use kind=path: {item}")
        kind, path = item.split("=", 1)
        if not kind or not path or kind in parsed:
            raise ValueError(f"Invalid or duplicate checkpoint: {item}")
        parsed[kind] = path
    return parsed


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible GeoVAE data, training, evaluation and reporting")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("doctor", help="Check Python, CUDA and optional baselines")
    preflight.add_argument("--require-cuda", action="store_true")
    smoke = commands.add_parser("smoke-batch", help="Check one batch and checkpoint resume")
    smoke.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    smoke.add_argument("--config", default=str(DEFAULT_CONFIG))
    smoke.add_argument("--out", required=True)
    smoke.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    prepare = commands.add_parser("prepare", help="Audit raw data and generate parent-disjoint protocol")
    prepare.add_argument("--data-root", default=str(ROOT.parent / "data"))
    prepare.add_argument("--protocol-dir", default=str(ROOT / "protocol"))
    prepare.add_argument("--count", type=int, default=5000)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--grid-size", nargs=3, type=int, default=(32, 32, 32))
    prepare.add_argument("--skip-raw-audit", action="store_true")
    training = commands.add_parser("train", help="Train one model on the fixed train/val splits")
    training.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    training.add_argument("--config", default=str(DEFAULT_CONFIG))
    training.add_argument("--model", required=True,
                          choices=("geovae", "base_vae", "unet", "cgan"))
    training.add_argument("--run-dir", required=True)
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--resume")
    training.add_argument("--max-epochs", type=int)
    scoring = commands.add_parser("evaluate", help="Run common cases, masks and domain")
    scoring.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    scoring.add_argument("--config", default=str(DEFAULT_CONFIG))
    scoring.add_argument("--out", required=True)
    scoring.add_argument("--split", choices=("val", "test"), default="test")
    scoring.add_argument("--methods", nargs="+", required=True,
                         choices=("idw", "rbf", "ik", "mps", "unet", "base_vae",
                                  "cgan", "geovae", "geovae_latent", "geovae_joint"))
    scoring.add_argument("--checkpoint", action="append", default=[],
                         metavar="KIND=PATH")
    scoring.add_argument("--seed", type=int, required=True)
    scoring.add_argument("--num-holes", type=int)
    scoring.add_argument("--noise-rate", type=float, default=0.0)
    scoring.add_argument("--limit", type=int, default=0,
                         help="Smoke tests only; 0 evaluates the full split")
    scoring.add_argument("--thresholds", help="Frozen validation threshold JSON")
    scoring.add_argument("--project-observed", action="store_true")
    tuning = commands.add_parser("tune-thresholds", help="Choose thresholds only on validation runs")
    tuning.add_argument("--evaluations", nargs="+", required=True)
    tuning.add_argument("--config", default=str(DEFAULT_CONFIG))
    tuning.add_argument("--out", required=True)
    tuning.add_argument("--draft", action="store_true",
                        help="Permit fewer methods and cases for smoke tests")
    summary = commands.add_parser("report", help="Reject incomplete/mismatched runs and summarize")
    summary.add_argument("--evaluations", nargs="+", required=True)
    summary.add_argument("--out", required=True)
    summary.add_argument("--draft", action="store_true",
                         help="Allow fewer than 3 seeds/methods for smoke tests")
    robust = commands.add_parser("robustness", help="Run fixed-case hole/noise grid")
    robust.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    robust.add_argument("--config", default=str(DEFAULT_CONFIG))
    robust.add_argument("--out", required=True)
    robust.add_argument("--split", choices=("val", "test"), default="test")
    robust.add_argument("--methods", nargs="+", required=True,
                        choices=("idw", "rbf", "ik", "mps", "unet", "base_vae",
                                 "cgan", "geovae", "geovae_latent", "geovae_joint"))
    robust.add_argument("--checkpoint", action="append", default=[], metavar="KIND=PATH")
    robust.add_argument("--seed", type=int, required=True)
    robust.add_argument("--limit", type=int, default=0)
    robust.add_argument("--thresholds", help="Frozen validation threshold JSON")
    robust.add_argument("--project-observed", action="store_true")
    ablation = commands.add_parser("ablation", help="Run the matched four-arm mechanism study")
    ablation.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ablation.add_argument("--config", default=str(DEFAULT_CONFIG))
    ablation.add_argument("--out", required=True)
    ablation.add_argument("--stage", choices=("prepare", "train", "evaluate", "report"),
                          required=True)
    ablation.add_argument("--seeds", nargs="+", type=int, default=(11, 23, 37))
    ablation.add_argument("--split", choices=("val", "test"), default="val")
    ablation.add_argument("--max-epochs", type=int)
    ablation.add_argument("--limit", type=int, default=0)
    figures = commands.add_parser("figures", help="Build traceable Figures 7 and 8")
    figures.add_argument("--evaluation", required=True)
    figures.add_argument("--report", required=True)
    figures.add_argument("--config", default=str(DEFAULT_CONFIG))
    figures.add_argument("--out", required=True)
    figures.add_argument("--methods", nargs="+", default=["idw", "unet", "geovae"])
    figures.add_argument("--sample-ids", nargs=3)
    figures.add_argument("--draft", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if args.command in {"evaluate", "robustness"} or (
            args.command == "ablation" and args.stage == "evaluate"):
        from revision_numerics import configure_evaluation
        configure_evaluation(args.out, phase="diagnostic_" + args.command)
    if args.command == "doctor":
        print(json.dumps(doctor(args.require_cuda), ensure_ascii=False, indent=2))
    elif args.command == "smoke-batch":
        from repro_smoke import smoke_batch
        print(smoke_batch(args.manifest, args.config, args.out, args.device))
    elif args.command == "prepare":
        from experiment_data import prepare_protocol
        manifest = prepare_protocol(
            args.data_root, args.protocol_dir, args.count, args.seed,
            tuple(args.grid_size), audit_raw=not args.skip_raw_audit)
        print(manifest)
    elif args.command == "train":
        if args.model == "cgan":
            from repro_cgan import train_cgan
            best = train_cgan(args.manifest, args.config, args.run_dir,
                              args.seed, args.resume, args.max_epochs)
        else:
            best = train(args.manifest, args.config, args.run_dir,
                         args.model, args.seed, args.resume, args.max_epochs)
        print(best)
    elif args.command == "evaluate":
        path = evaluate(
            args.manifest, args.config, args.out, args.split, args.methods,
            _checkpoints(args.checkpoint), args.seed, args.num_holes,
            args.noise_rate, args.limit, args.project_observed, args.thresholds)
        print(path)
    elif args.command == "tune-thresholds":
        from repro_tuning import select_thresholds
        print(select_thresholds(args.evaluations, args.config, args.out, args.draft))
    elif args.command == "report":
        print(report(args.evaluations, args.out, draft=args.draft))
    elif args.command == "ablation":
        from repro_ablation import (prepare_configs, train_arms,
                                   evaluate_arms, report_arms)
        if args.stage == "prepare":
            print(prepare_configs(args.config, args.out))
        elif args.stage == "train":
            train_arms(args.manifest, args.config, args.out,
                       args.seeds, args.max_epochs)
        elif args.stage == "evaluate":
            print(evaluate_arms(args.manifest, args.config, args.out,
                                args.seeds, args.split, args.limit))
        else:
            print(report_arms(args.out, args.split, args.seeds, args.limit))
    elif args.command == "figures":
        from repro_figures import figures
        print(figures(args.evaluation, args.report, args.config, args.out,
                      args.methods, args.sample_ids, args.draft))
    elif args.command == "robustness":
        runs = []
        for holes in (2, 4, 8, 16):
            for noise in (0.0, 0.05, 0.10, 0.20):
                path = Path(args.out) / f"holes{holes:02d}_noise{round(noise * 100):02d}"
                evaluate(args.manifest, args.config, str(path), args.split,
                         args.methods, _checkpoints(args.checkpoint), args.seed,
                         holes, noise, args.limit, args.project_observed, args.thresholds)
                runs.append(str(path))
        (Path(args.out) / "robustness_runs.json").write_text(
            json.dumps({"seed": args.seed, "runs": runs}, indent=2), encoding="utf-8")
        print(Path(args.out) / "robustness_runs.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
