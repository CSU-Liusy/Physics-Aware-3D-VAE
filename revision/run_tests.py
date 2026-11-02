"""Run public CPU checks in fresh processes with their intended import policies.

These functional tests use small fixtures; they do not reproduce paper results.
"""
from pathlib import Path
import os
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               PYTHONDONTWRITEBYTECODE='1')
    commands = [
        [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests'],
        [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'porphyry'],
    ]
    failed = False
    for command in commands:
        print('Running CPU test domain: ' + command[-1], flush=True)
        result = subprocess.run(command, cwd=root, env=env)
        failed = failed or result.returncode != 0
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
