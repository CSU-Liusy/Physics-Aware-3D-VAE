"""Public launcher checks without training, data access or model imports."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import subprocess
import sys
import pytest
import main

ROOT=Path(__file__).resolve().parents[1]


def test_no_arguments_print_help_without_subprocess(monkeypatch,capsys):
    monkeypatch.setattr(main.subprocess,'run',lambda *a,**k:pytest.fail('help launched a child'))
    assert main.main([])==0
    assert 'revision/' in capsys.readouterr().out


@pytest.mark.parametrize('args,route,forwarded',[
    (['--all','--out','runs/test'],'revision',['--all','--out','runs/test']),
    (['revision','--help'],'revision',['--help']),
    (['legacy','--help'],'legacy',['--help']),
    (['--legacy','--mode','train'],'legacy',['--mode','train']),
    (['--mode=benchmark'],'legacy',['--mode=benchmark']),
    (['--mode','train'],'legacy',['--mode','train']),
    (['external','--help'],'external',['--help']),
])
def test_explicit_routes(args,route,forwarded):
    assert main.select_route(args)==(route,forwarded)


@pytest.mark.parametrize('args,entry',[
    (['revision','--help'],'revision/main.py'),
    (['legacy','--help'],'legacy/main.py'),
    (['external','--help'],'revision/porphyry/porphyry_public.py'),
])
def test_child_isolated_and_relative_outputs_use_outer_root(monkeypatch,args,entry):
    calls=[]
    def run(command,**kwargs): calls.append((command,kwargs));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(main.subprocess,'run',run)
    assert main.main(args)==0
    command,settings=calls[0]
    assert Path(command[2])==ROOT/entry
    assert settings['cwd']==ROOT
    assert settings['env']['PYTHONDONTWRITEBYTECODE']=='1'


def test_launcher_import_has_no_torch_side_effect():
    result=subprocess.run([sys.executable,'-B','-c',"import sys; import main; assert 'torch' not in sys.modules"],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_inner_core_identity_and_historical_paths_are_preserved():
    inner=ROOT/'revision'
    meta=json.loads((inner/'EXPORT_PROVENANCE.json').read_text())
    digest=hashlib.sha256()
    for name in ('repro_pipeline.py','experiment_data.py','model.py','repro_cgan.py','repro_baselines.py','repro_metrics.py','comparisons/models/unet.py'):
        actual=hashlib.sha256((inner/name).read_bytes()).hexdigest()
        assert actual==meta['checkpoint_engine_files'][name]
        digest.update(name.encode());digest.update(actual.encode())
    assert digest.hexdigest()==meta['engine_source_sha256']=='2bbb423c62e8ca17cd8045d98cc62fcd123d0a0ade27c6dd2ef72cd52c91d5c3'
