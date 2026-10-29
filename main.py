"""Unified GeoVAE launcher with isolated current and historical implementations."""
from __future__ import annotations

from pathlib import Path
import os
import json
import subprocess
import sys

from repository_safety import validate_arguments

ROOT = Path(__file__).resolve().parent
HELP = """GeoVAE training and revision reproduction

Current manuscript workflow (default):
  python main.py --all --out runs/esi_revision_v2
  python main.py --all --smoke --out runs/cpu_smoke
  python main.py revision --help
  python main.py doctor

Historical general PLY workflow (separate models/configuration):
  python main.py legacy --help
  python main.py legacy --mode train --config legacy/config/default.yaml --ply-dir EXTERNAL_PLY_DIRECTORY --output-dir EXTERNAL_OUTPUT_DIRECTORY

Optional external synthetic case:
  python main.py external --help

The current implementation, protocol, tests and historical verification are
collected in revision/. The historical general workflow is in legacy/.
No-argument invocation prints help and never starts training.
"""


def select_route(arguments):
    args = list(arguments)
    if args and args[0] in ('legacy', '--legacy'):
        return 'legacy', args[1:]
    if args and args[0] in ('external', 'porphyry'):
        return 'external', args[1:]
    if args and args[0] == 'revision':
        return 'revision', args[1:]
    if '--mode' in args or any(arg.startswith('--mode=') for arg in args):
        return 'legacy', args
    return 'revision', args


def require_downloaded_protocol(root, manifest=None):
    """Presence preflight only; the inner engine retains all identity checks."""
    manifest = Path(manifest) if manifest is not None else Path(root) / 'revision/protocol/manifest.json'
    protocol = manifest.parent
    try:
        data = json.loads(manifest.read_text(encoding='utf-8'))
        samples = data['samples']
        if not samples:
            raise ValueError('empty manifest')
        missing = next((row['path'] for row in samples if not (protocol / row['path']).is_file()), None)
        if missing is None:
            return
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError('Missing or unreadable protocol. See docs/DATA_ACCESS.md.') from error
    raise ValueError('Download the matching Baidu input files into revision/protocol/grids before a formal run or full tests; missing ' + missing + '. See docs/DATA_ACCESS.md. No run has started.')


def selected_protocol_manifest(root, arguments):
    selected = Path(root) / 'revision/protocol/manifest.json'
    for index, argument in enumerate(arguments):
        if argument == '--manifest':
            if index + 1 == len(arguments) or arguments[index + 1].startswith('--'):
                raise ValueError('--manifest requires a path')
            selected = Path(arguments[index + 1])
        elif argument.startswith('--manifest='):
            value = argument.split('=', 1)[1]
            if not value:
                raise ValueError('--manifest requires a path')
            selected = Path(value)
    return selected if selected.is_absolute() else Path(root) / selected


def needs_downloaded_protocol(arguments):
    if not arguments or arguments[0] == 'doctor':
        return False
    if any(arg in ('-h', '--help', '--dry-run', '--smoke') for arg in arguments):
        return False
    stages = {'--all', '--revision-all', '--prepare', '--train', '--pretrain', '--tune', '--validate',
              '--test', '--ablation-train', '--ablation-validate', '--ablation-test',
              '--uncertainty', '--robustness', '--real-cases'}
    if arguments[0] == 'ablation':
        stage = None
        for index, argument in enumerate(arguments):
            if argument == '--stage' and index + 1 < len(arguments):
                stage = arguments[index + 1]
            elif argument.startswith('--stage='):
                stage = argument.split('=', 1)[1]
        return stage in ('train', 'evaluate')
    consumers = {'all', 'train', 'evaluate', 'robustness', 'smoke-batch'}
    # Diagnostic `prepare` can intentionally generate a new user protocol; its
    # unchanged inner implementation rejects any incomplete existing manifest.
    return arguments[0] in consumers or bool(stages.intersection(arguments))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ('-h', '--help'):
        print(HELP)
        return 0
    route, forwarded = select_route(args)
    try:
        forwarded = validate_arguments(route, forwarded, ROOT)
        if route == 'revision' and needs_downloaded_protocol(forwarded):
            require_downloaded_protocol(ROOT, selected_protocol_manifest(ROOT, forwarded))
    except ValueError as error:
        print(f'GeoVAE: {error}', file=sys.stderr)
        return 2
    entry = {'revision': ROOT/'revision/main.py',
             'legacy': ROOT/'legacy/main.py',
             'external': ROOT/'revision/porphyry/porphyry_public.py'}[route]
    if not entry.is_file():
        print(f'GeoVAE: required entry is missing: {entry.relative_to(ROOT)}', file=sys.stderr)
        return 2
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    # Separate interpreters prevent model/configuration modules in legacy/ and
    # revision/ from shadowing one another. Each child script owns sys.path[0].
    result = subprocess.run([sys.executable, '-B', str(entry), *forwarded],
                            cwd=ROOT, env=env)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
