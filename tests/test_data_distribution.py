"""Downloaded-input presence checks; no real arrays, training or scores."""
import json
from pathlib import Path
import pytest
import main
import run_tests

@pytest.mark.parametrize('args', [['--all'], ['--revision-all'], ['all'], ['train'], ['evaluate'], ['robustness'], ['smoke-batch'], ['ablation','--stage','train'], ['--prepare'], ['--train','--model','geovae']])
def test_formal_stage_needs_archive(args):
    assert main.needs_downloaded_protocol(args)

@pytest.mark.parametrize('args', [[],['doctor'],['--help'],['--all','--smoke'],['--all','--dry-run']])
def test_help_smoke_and_plan_do_not_need_archive(args):
    assert not main.needs_downloaded_protocol(args)

def test_missing_archive_prevents_subprocess_and_budget(monkeypatch,tmp_path,capsys):
    monkeypatch.setattr(main,'ROOT',tmp_path)
    monkeypatch.setattr(main.subprocess,'run',lambda *a,**k:pytest.fail('child started without dataset'))
    assert main.main(['--all','--out',str(tmp_path/'run')])==2
    assert 'DATA_ACCESS' in capsys.readouterr().err
    assert not (tmp_path/'run').exists()

def test_presence_check_rejects_partial_archive(tmp_path):
    protocol=tmp_path/'revision/protocol';protocol.mkdir(parents=True)
    (protocol/'manifest.json').write_text(json.dumps({'samples':[{'path':'grids/one.npy'},{'path':'grids/two.npy'}]}))
    (protocol/'grids').mkdir();(protocol/'grids/one.npy').touch()
    with pytest.raises(ValueError,match='two.npy'):
        main.require_downloaded_protocol(tmp_path)
    (protocol/'grids/two.npy').touch()
    main.require_downloaded_protocol(tmp_path) # Existence only; not a scientific-integrity claim.

def test_full_tests_do_not_silently_skip_missing_archive(monkeypatch,tmp_path):
    def absent(root): raise ValueError('download required')
    monkeypatch.setattr(main,'require_downloaded_protocol',absent)
    monkeypatch.setattr(run_tests.subprocess,'run',lambda *a,**k:pytest.fail('suite started'))
    assert run_tests.main(['--scope','all'])==2

@pytest.mark.parametrize('args', [['--manifest','custom/manifest.json'], ['--manifest=custom/manifest.json']])
def test_explicit_relative_manifest_resolves_from_outer_root(tmp_path,args):
    assert main.selected_protocol_manifest(tmp_path,args)==tmp_path/'custom/manifest.json'

def test_external_installed_protocol_works_without_bundled_arrays(tmp_path):
    external=tmp_path/'external';external.mkdir()
    (external/'grid.npy').touch()
    manifest=external/'manifest.json'
    manifest.write_text(json.dumps({'samples':[{'path':'grid.npy'}]}))
    root=tmp_path/'code';root.mkdir()
    selected=main.selected_protocol_manifest(root,['--all','--manifest',str(manifest)])
    main.require_downloaded_protocol(root,selected)

@pytest.mark.parametrize('args',[['--manifest'],['--manifest='],['--manifest','--all']])
def test_malformed_explicit_manifest_rejected(tmp_path,args):
    with pytest.raises(ValueError,match='requires a path'):
        main.selected_protocol_manifest(tmp_path,args)

@pytest.mark.parametrize('args', [['--finalize'],['report'],['tune-thresholds'],['figures'],['prepare'],['ablation','--stage','prepare'],['ablation','--stage=report']])
def test_independent_finalization_and_metadata_routes_keep_existing_behavior(args):
    assert not main.needs_downloaded_protocol(args)
