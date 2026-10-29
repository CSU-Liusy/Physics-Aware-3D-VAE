"""Legacy interface/path tests, using temporary bytes and no model training."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
LEGACY=ROOT/'legacy'


def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,LEGACY/filename)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class LegacyInterfaceTests(unittest.TestCase):
    def test_default_yaml_is_read_and_has_no_author_paths(self):
        args=module('legacy_config_test','config_loader.py').parse_args([])
        self.assertEqual((args.epochs,args.patience,args.latent_dim),(120,7,256))
        self.assertIsNone(args.checkpoint)
        self.assertIsNone(args.ply_dir)
        self.assertIsNone(args.ply_file)
        self.assertIsNone(args.output_dir)

    def test_yaml_then_cli_override(self):
        config=module('legacy_config_test','config_loader.py')
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'config.yaml';p.write_text('epochs: 3\nlatent_dim: 64\ncuda: true\n',encoding='utf-8')
            args=config.parse_args(['--config',str(p),'--epochs','4','--no-cuda','--no-skip-vis'])
            self.assertEqual((args.epochs,args.latent_dim,args.cuda,args.skip_vis),(4,64,False,False))

    def test_invalid_yaml_and_missing_config_fail(self):
        config=module('legacy_config_test','config_loader.py')
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'config.yaml';p.write_text('[]',encoding='utf-8')
            with self.assertRaises(ValueError): config.parse_args(['--config',str(p)])
            with self.assertRaises(FileNotFoundError): config.parse_args(['--config',str(p.with_name('missing.yaml'))])

    def test_help_and_missing_implementation_never_import_torch(self):
        script="import runpy,sys;sys.path.insert(0,sys.argv.pop(1));sys.argv=['legacy/main.py',*sys.argv[1:]];\ntry: runpy.run_path(sys.path[0]+'/main.py',run_name='__main__')\nfinally: assert 'torch' not in sys.modules, 'legacy help/preflight imported torch'"
        for args,expected in [(['--help'],0),(['--mode','robust_eval'],1),([],1)]:
            result=subprocess.run([sys.executable,'-B','-c',script,str(LEGACY),*args],capture_output=True,text=True)
            self.assertEqual(result.returncode,expected,result.stderr)
            if args==['--help']: self.assertIn('Historical mesh-based',result.stdout)
            if 'robust_eval' in args:self.assertIn('unsupported',result.stderr)

    def test_explicit_file_does_not_include_neighboring_mesh(self):
        paths=module('legacy_paths_test','legacy_paths.py')
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);one=folder/'one.ply';two=folder/'two.ply'
            one.write_bytes(b'one');two.write_bytes(b'two')
            self.assertEqual(paths.select_mesh_files(folder,str(one)),[str(one.resolve())])
            self.assertEqual(len(paths.select_mesh_files(folder)),2)
            with self.assertRaises(ValueError): paths.select_mesh_files(folder,folder/'missing.ply')

    def test_legacy_inputs_outputs_must_stay_external(self):
        paths=module('legacy_paths_test','legacy_paths.py')
        with self.assertRaises(ValueError): paths.resolve_output_directory(None,[])
        with self.assertRaises(ValueError): paths.resolve_output_directory(ROOT/'runs/legacy',[])
        with self.assertRaises(ValueError): paths.external_input(ROOT/'revision/protocol/manifest.json')
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'inputs';source.mkdir()
            with self.assertRaises(ValueError): paths.resolve_output_directory(source/'cache',[source])
            self.assertEqual(paths.resolve_output_directory(Path(temp)/'outputs',[source]),Path(temp)/'outputs')

    def test_owned_cache_reset_preserves_inputs_and_unrelated_files(self):
        paths=module('legacy_paths_test','legacy_paths.py')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);inputs=root/'inputs';inputs.mkdir();source=inputs/'toy.ply';source.write_bytes(b'fixture, not a mesh')
            before=hashlib.sha256(source.read_bytes()).hexdigest()
            namespace=Path(paths.prepare_mesh_cache(root/'outputs/cache',[source],(4,4,4)))
            own=namespace/'toy.ply.vox4x4x4.npy';own.write_bytes(b'old owned fixture')
            unrelated=namespace/'unrelated.npy';unrelated.write_bytes(b'must remain')
            paths.prepare_mesh_cache(root/'outputs/cache',[source],(4,4,4),reset=True)
            self.assertFalse(own.exists());self.assertEqual(unrelated.read_bytes(),b'must remain')
            self.assertEqual(before,hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(list(inputs.iterdir()),[source])

    def test_cache_rejects_unowned_or_mismatched_identity(self):
        paths=module('legacy_paths_test','legacy_paths.py')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);inputs=root/'inputs';inputs.mkdir();source=inputs/'toy.ply';source.write_bytes(b'toy')
            namespace=Path(paths.prepare_mesh_cache(root/'outputs/cache',[source],(4,4,4)))
            own=namespace/'toy.ply.vox4x4x4.npy';own.write_bytes(b'preserve')
            marker=namespace/'cache_identity.json';marker.write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):paths.prepare_mesh_cache(root/'outputs/cache',[source],(4,4,4),reset=True)
            self.assertEqual(own.read_bytes(),b'preserve')
            marker.unlink()
            with self.assertRaises(ValueError):paths.prepare_mesh_cache(root/'outputs/cache',[source],(4,4,4),reset=True)
            self.assertEqual(own.read_bytes(),b'preserve')

    def test_changed_source_uses_a_new_cache_namespace(self):
        paths=module('legacy_paths_test','legacy_paths.py')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);inputs=root/'inputs';inputs.mkdir();source=inputs/'toy.ply';source.write_bytes(b'first')
            one=paths.prepare_mesh_cache(root/'outputs/cache',[source],(4,4,4))
            source.write_bytes(b'second')
            two=paths.prepare_mesh_cache(root/'outputs/cache',[source],(4,4,4))
            self.assertNotEqual(one,two)

    def test_yaml_cannot_bypass_external_path_checks(self):
        script="import runpy,sys;sys.path.insert(0,sys.argv.pop(1));sys.argv=['legacy/main.py',*sys.argv[1:]];\ntry: runpy.run_path(sys.path[0]+'/main.py',run_name='__main__')\nfinally: assert 'torch' not in sys.modules, 'invalid YAML paths imported torch'"
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'config.yaml'
            p.write_text(f'ply_dir: {LEGACY.as_posix()}\noutput_dir: {Path(temp).as_posix()}/output\n',encoding='utf-8')
            result=subprocess.run([sys.executable,'-B','-c',script,str(LEGACY),'--config',str(p)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('outside the public repository',result.stderr)
            self.assertFalse((Path(temp)/'output').exists())

    def test_benchmark_uses_checkpoint_architecture_and_strict_state_fixture(self):
        source=(LEGACY/'comparisons/run_benchmark.py').read_text(encoding='utf-8')
        function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='load_vae')
        code=compile(ast.Module(body=[function],type_ignores=[]),'<legacy-loader-fixture>','exec')
        calls={};payload={'hparams':{'model_type':'standard','grid_size':[32,32,32],'latent_dim':64,'base_channels':8},'state_dict':{'module.toy':1}}
        class Model:
            def load_state_dict(self,state,strict):calls.update(state=state,strict=strict)
            def eval(self):calls['eval']=True
        def create(**kwargs):calls.update(kwargs);return Model(),None,None
        namespace={'os':__import__('os'),'torch':SimpleNamespace(load=lambda *a,**kw:payload),'create_model':create}
        exec(code,namespace)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint=Path(temp)/'fixture.pth';checkpoint.write_bytes(b'not a real checkpoint; fake loader only')
            namespace['load_vae'](str(checkpoint),'cpu')
            self.assertEqual((calls['model_type'],calls['latent_dim'],calls['base_channels']),('standard',64,8))
            self.assertTrue(calls['strict']);self.assertEqual(calls['state'],{'toy':1})
            payload.clear();payload.update(format_version=2,model_kind='geovae',manifest_sha256='fixture')
            with self.assertRaisesRegex(ValueError,'revision workflow'):namespace['load_vae'](str(checkpoint),'cpu')

    def test_legacy_local_training_import_closure_exists(self):
        local={'config_loader','legacy_paths','trainer','dataset','model','model_factory','output_result','showresult','comparisons'}
        for path in LEGACY.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
                names=([node.module] if isinstance(node,ast.ImportFrom) and node.module else [n.name for n in node.names] if isinstance(node,ast.Import) else [])
                for name in names:
                    if name.split('.')[0] not in local:continue
                    target=LEGACY.joinpath(*name.split('.'))
                    self.assertTrue(target.with_suffix('.py').is_file() or (target/'__init__.py').is_file(),f'{path.name}: {name}')

    def test_no_skip_vis_uses_guarded_external_run_directory(self):
        paths=module('legacy_paths_test','legacy_paths.py')
        source=ast.parse((LEGACY/'trainer.py').read_text(encoding='utf-8-sig'))
        train=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        render_stage=next(n for n in reversed(train.body) if isinstance(n,ast.Try) and any(isinstance(c,ast.Call) and isinstance(c.func,ast.Name) and c.func.id=='render_outputs_to_vis' for c in ast.walk(n)))
        code=compile(ast.Module(body=[render_stage],type_ignores=[]),'<legacy-final-render-fixture>','exec')
        calls=[]
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);inputs=base/'inputs';inputs.mkdir();output=base/'output';output.mkdir()
            namespace={'os':__import__('os'),'args':SimpleNamespace(skip_vis=False),'model':SimpleNamespace(cpu=lambda:None),'torch':SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False)),'base_out':str(output),'ply_dir':str(inputs),'outputs_dir':str(output/'outputs'),'resolve_output_directory':paths.resolve_output_directory,'render_outputs_to_vis':lambda *a,**kw:calls.append((a,kw)),'print':lambda *a,**kw:None}
            exec(code,namespace)
            self.assertEqual(Path(calls[0][0][1]),output/'vis')
            self.assertFalse((ROOT/'results').exists())


if __name__=='__main__':
    unittest.main()
