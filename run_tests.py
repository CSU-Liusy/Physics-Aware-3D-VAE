"""Run layout/legacy checks and the preserved current CPU domains separately."""
from pathlib import Path
import argparse
import os
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('all', 'layout', 'legacy', 'revision'), default='all')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    if args.scope in ('all', 'revision'):
        from main import require_downloaded_protocol
        try:
            require_downloaded_protocol(root)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               PYTHONDONTWRITEBYTECODE='1')
    commands = []
    if args.scope in ('all', 'layout', 'legacy'):
        tests = 'tests/test_legacy_cli.py' if args.scope == 'legacy' else 'tests'
        commands.append([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', tests])
    if args.scope in ('all', 'revision'):
        commands.append([sys.executable, '-B', str(root/'revision/run_tests.py')])
    failed = False
    for command in commands:
        print('Running: ' + ' '.join(command[2:]), flush=True)
        result = subprocess.run(command, cwd=root, env=env)
        failed = failed or result.returncode != 0
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
