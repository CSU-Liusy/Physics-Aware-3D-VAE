"""Bounded legacy I/O/failure checks; no scientific model or data is evaluated."""
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / 'legacy'


def function(filename, name, namespace):
    tree = ast.parse((LEGACY / filename).read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<legacy-interface-fixture>', 'exec'), namespace)
    return namespace[name]


def path_helpers():
    spec = importlib.util.spec_from_file_location('legacy_failure_paths', LEGACY / 'legacy_paths.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Tensor:
    def unsqueeze(self, *_): return self
    def to(self, *_): return self
    def float(self): return self
    def squeeze(self): return self
    def cpu(self): return self
    def numpy(self): return [0]


class Progress:
    def __init__(self, values): self.values = values
    def __iter__(self): return iter(self.values)
    def set_postfix(self, *_): pass


class LegacyFailureReportingTests(unittest.TestCase):
    def test_unified_benchmark_preserves_exact_file_selection(self):
        paths = path_helpers()
        capture = {}
        fake = ModuleType('comparisons.run_benchmark')
        fake.execute_benchmark = lambda **kwargs: capture.update(kwargs)
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / 'inputs'
            source.mkdir()
            selected = source / 'one.ply'
            selected.write_bytes(b'fixture')
            (source / 'neighbor.ply').write_bytes(b'not selected')
            checkpoint = base / 'toy.pth'
            checkpoint.write_bytes(b'fake, never loaded')
            args = SimpleNamespace(mode='benchmark', output_dir=str(base / 'output'),
                                   ply_dir=None, ply_file=str(selected), checkpoint=str(checkpoint),
                                   unet_checkpoint=None, generate_only=False, num_samples=1, cuda=False)
            call = function('main.py', 'main', {'Path': Path, 'parse_args': lambda _: args,
                            'external_input': paths.external_input,
                            'resolve_output_directory': paths.resolve_output_directory})
            with patch.dict(sys.modules, {'comparisons.run_benchmark': fake}):
                self.assertEqual(call([]), 0)
            self.assertEqual(capture['file_list'], str(selected.resolve()))

    def benchmark_fixture(self, base, failed_indices):
        paths = path_helpers()
        source = base / 'inputs'
        source.mkdir()
        selected = source / 'one.ply'
        selected.write_bytes(b'fixture')
        checkpoint = base / 'toy.pth'
        checkpoint.write_bytes(b'not loaded')
        calls = {}

        class Dataset:
            def __init__(self, **kwargs): calls.update(kwargs)
            def __len__(self): return 2
            def __getitem__(self, index):
                if index in failed_indices: raise ValueError('intentional synthetic fixture failure')
                return Tensor(), Tensor(), None, None, index

        class Model:
            def __call__(self, _): return Tensor(), None, None

        class Frame:
            def __init__(self, rows): self.rows = rows
            def to_csv(self, filename, index): Path(filename).write_text('fixture summary\n', encoding='utf-8')

        namespace = {'os': os, 'json': json, 'torch': SimpleNamespace(device=lambda value: value,
                     cuda=SimpleNamespace(is_available=lambda: False), no_grad=nullcontext,
                     sigmoid=lambda value: value), 'MiningDataset': Dataset,
                     'external_input': paths.external_input,
                     'resolve_output_directory': paths.resolve_output_directory,
                     'load_vae': lambda *_: Model(), 'tqdm': lambda values, **_: values,
                     'calculate_metrics': lambda *_args, **_kwargs: (.5, .4, 1., 2.),
                     'np': SimpleNamespace(mean=lambda values: sum(values) / len(values)),
                     'pd': SimpleNamespace(DataFrame=Frame), 'print': lambda *_args, **_kwargs: None}
        call = function('comparisons/run_benchmark.py', 'execute_benchmark', namespace)
        return call, calls, selected, checkpoint

    def test_benchmark_records_all_failures_and_raises_without_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            call, calls, selected, checkpoint = self.benchmark_fixture(base, {0, 1})
            output = base / 'output'
            with self.assertRaisesRegex(RuntimeError, 'no successful evaluations'):
                call(vae_ckpt=str(checkpoint), data_dir=str(selected.parent), output_dir=str(output),
                     file_list=str(selected), use_cuda=False)
            record = json.loads((output / 'benchmark_execution.json').read_text(encoding='utf-8'))
            self.assertEqual(calls['file_list'], str(selected))
            self.assertEqual(record['status'], 'failed_no_successful_evaluations')
            self.assertEqual((record['successful_evaluations'], len(record['failures'])), (0, 2))
            self.assertFalse((output / 'benchmark_summary_final.csv').exists())

    def test_benchmark_partial_failure_is_retained_with_success_count(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            call, _, selected, checkpoint = self.benchmark_fixture(base, {1})
            output = base / 'output'
            call(vae_ckpt=str(checkpoint), data_dir=str(selected.parent), output_dir=str(output),
                 file_list=str(selected), use_cuda=False)
            record = json.loads((output / 'benchmark_execution.json').read_text(encoding='utf-8'))
            self.assertEqual(record['status'], 'completed_with_failures')
            self.assertEqual(record['successful_evaluations'], 1)
            self.assertEqual(record['failures'][0]['sample_index'], 1)
            self.assertTrue((output / 'benchmark_summary_final.csv').is_file())

    def unet_fixture(self, base, outcomes, epochs=1):
        paths = path_helpers()
        inputs = base / 'inputs'
        inputs.mkdir()
        output = base / 'output'
        saves = []
        sequence = iter(outcomes)

        class Model:
            def to(self, _): return self
            def parameters(self): return []
            def train(self): pass
            def __call__(self, _): return Tensor()

        class Criterion:
            def __call__(self, *_):
                value = next(sequence)
                if isinstance(value, Exception): raise value
                return SimpleNamespace(item=lambda: value, backward=lambda: None)

        optimizer = SimpleNamespace(zero_grad=lambda: None, step=lambda: None)
        def save(_state, filename):
            saves.append(filename)
            Path(filename).write_bytes(b'fake checkpoint from mechanical fixture')
        namespace = {'getattr': getattr, 'os': os, 'json': json,
                     'torch': SimpleNamespace(device=lambda value: value,
                        cuda=SimpleNamespace(is_available=lambda: False), save=save),
                     'external_input': paths.external_input, 'resolve_output_directory': paths.resolve_output_directory,
                     'MiningDataset': lambda **_: [],
                     'DataLoader': lambda *_args, **_kwargs: [(Tensor(), Tensor(), None, None, 0) for _ in outcomes],
                     'UNet3D': lambda **_: Model(),
                     'optim': SimpleNamespace(Adam=lambda *_args, **_kwargs: optimizer),
                     'nn': SimpleNamespace(BCELoss=Criterion), 'tqdm': lambda values, **_: Progress(values),
                     'print': lambda *_args, **_kwargs: None}
        Model.state_dict = lambda self: {'toy': 1}
        call = function('comparisons/train_unet.py', 'train_unet', namespace)
        args = SimpleNamespace(ply_dir=str(inputs), output_dir=str(output), epochs=epochs)
        return call, args, output, saves

    def test_unet_all_failed_or_empty_epoch_never_saves_best(self):
        for outcomes in ([ValueError('intentional batch failure')], []):
            with self.subTest(outcomes=len(outcomes)), tempfile.TemporaryDirectory() as temp:
                call, args, output, saves = self.unet_fixture(Path(temp), outcomes)
                with self.assertRaisesRegex(RuntimeError, 'no successful batches'):
                    call(args)
                record = json.loads((output / 'unet_execution.json').read_text(encoding='utf-8'))
                self.assertEqual(record['status'], 'failed_no_successful_batches')
                self.assertEqual(saves, [])

    def test_unet_partial_epoch_loss_uses_only_successful_batches(self):
        with tempfile.TemporaryDirectory() as temp:
            call, args, output, saves = self.unet_fixture(Path(temp), [2., ValueError('intentional failure')])
            self.assertEqual(call(args), str(output / 'unet_best.pth'))
            record = json.loads((output / 'unet_execution.json').read_text(encoding='utf-8'))
            self.assertEqual(record['status'], 'completed_with_failures')
            self.assertEqual(record['epochs'][0]['mean_successful_batch_loss'], 2.)
            self.assertEqual(record['epochs'][0]['successful_batches'], 1)
            self.assertEqual(len(saves), 1)

    def test_unet_zero_epochs_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            call, args, output, saves = self.unet_fixture(Path(temp), [], epochs=0)
            with self.assertRaisesRegex(RuntimeError, 'at least one completed'):
                call(args)
            self.assertEqual(json.loads((output / 'unet_execution.json').read_text(encoding='utf-8'))['status'], 'failed_no_epochs')
            self.assertEqual(saves, [])

    def test_ablation_plot_creates_a_new_external_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / 'inputs' / 'Baseline'
            source.mkdir(parents=True)
            (source / 'metrics_summary.csv').write_text('split,iou,dice,accuracy\ntest,0.4,0.5,0.8\ntest,0.6,0.7,0.9\n', encoding='utf-8')
            output = base / 'new_output'
            env = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run([sys.executable, '-B', str(LEGACY / 'comparisons/plot_ablation_summary.py'),
                                    '--results-dir', str(source.parent), '--output-dir', str(output)],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / 'ablation_summary_plot.png').is_file())
            self.assertTrue((output / 'ablation_summary_table.csv').is_file())


if __name__ == '__main__':
    unittest.main()
