"""Public synthetic-only GeoVAE revision entry point."""
from __future__ import annotations
import argparse
import sys

HELP = """GeoVAE public synthetic revision

CPU integration draft (no manuscript evidence)
  python main.py --prepare --smoke --out runs/cpu_smoke
  python main.py --all --smoke --out runs/cpu_smoke

Formal CUDA reproduction (new runs, 72-hour default experiment budget)
  python main.py --all --out runs/esi_revision_v2

Modules, sharing the same --out
  --prepare  --train (alias --pretrain)  --tune  --validate
  --ablation-train  --ablation-validate  --test  --ablation-test
  --uncertainty  --robustness  --finalize
Training supports --model and --seed. See README.md for prerequisites.

Restricted real data are absent and skipped by default, not audited as passed.
An optional --real-cases run requires explicit --data-root and --private-out.
"""

def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or "--help" in args or "-h" in args:
        print(HELP)
        return 0
    if args[0] == "all":
        args[0] = "--all"
    if "--smoke" in args:
        parser = argparse.ArgumentParser(description="CPU integration draft; no formal data selection")
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument("--all", action="store_true")
        mode.add_argument("--prepare", action="store_true")
        parser.add_argument("--smoke", action="store_true", required=True)
        parser.add_argument("--out", default="runs/cpu_smoke")
        options = parser.parse_args(args)
        from revision_smoke import run
        print(run(options.out, prepare_only=options.prepare))
        return 0
    if any(flag in args for flag in ("--all", "--revision-all", "--uncertainty", "--real-cases")):
        from revision_runner import main as run
    elif any(flag in args for flag in ("--robustness", "--finalize")):
        from public_finish import main as run
    elif args[0].startswith("-"):
        from repro_run_all import main as run
    else:
        from repro_cli import main as run
    return run(args)

if __name__ == "__main__":
    raise SystemExit(main())
