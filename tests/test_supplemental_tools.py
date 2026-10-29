"""Portable supplemental tools: saved scalars and explicit public-run guards only."""
from pathlib import Path
import importlib.util, json, subprocess, sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
SUPPLEMENTAL=ROOT/'revision/supplemental'
def load(name):
    spec=importlib.util.spec_from_file_location('test_'+name,SUPPLEMENTAL/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_secondary_is_fixed_original_twenty_item_family():
    module=load('secondary_contrasts');helper=module.reference()
    assert len(helper.CONTRASTS)==4 and len(helper.METRICS)==5
    assert helper.REPLICATES==2000 and helper.BOOTSTRAP_SEED==20261025
    assert helper.SEEDS==(11,23,37)

def test_secondary_pure_compute_matches_constant_parent_difference():
    module=load('secondary_contrasts');helper=module.reference()
    parents={f'case-p{p}-v{v}':f'parent{p}' for p in range(2) for v in range(5)}
    arms={}
    arm_order={'base_no_observed':0.,'base_observed':.1,'attention_no_position':.2,'attention_position':.3}
    for arm in helper.ARMS:
        arms[arm]={(seed,sid):{metric:arm_order[arm] for metric in helper.METRICS} for seed in helper.SEEDS for sid in parents}
    rows,seeds,groups,_=helper.compute(arms,parents)
    assert len(rows)==20 and len(seeds)==60 and len(groups)==40
    for row in rows:
        want=arm_order[row['candidate']]-arm_order[row['reference']]
        assert row['mean_candidate_minus_reference']==pytest.approx(want)
        assert row['ci_low']==pytest.approx(want) and row['ci_high']==pytest.approx(want)

def test_secondary_comparison_rejects_changed_value_or_identifier():
    module=load('secondary_contrasts')
    assert module.compare([{'id':'a','x':.1}],[{'id':'a','x':'0.1'}])['max_absolute_difference']==0
    with pytest.raises(ValueError):module.compare([{'id':'a','x':.2}],[{'id':'a','x':'0.1'}])
    with pytest.raises(ValueError):module.compare([{'id':'b','x':.1}],[{'id':'a','x':'0.1'}])

def test_secondary_missing_data_cannot_produce_results(tmp_path):
    module=load('secondary_contrasts');out=tmp_path/'out'
    assert module.main(['--source-data',str(tmp_path/'missing'),'--out',str(out)])==2
    assert not out.exists()

def test_complex_rejects_author_run_before_array_or_module_import(tmp_path):
    module=load('complex_examples');run=tmp_path/'author';run.mkdir();(run/'run_identity.json').write_text(json.dumps({'mode':'formal','source_sha256':'unchanged-author-source'}))
    out=tmp_path/'figures'
    with pytest.raises(ValueError,match='independent public synthetic run'):module.render(run,out)
    assert not out.exists()

def test_complex_requires_complete_public_handoff(tmp_path):
    module=load('complex_examples');run=tmp_path/'public';run.mkdir()
    (run/'run_identity.json').write_text(json.dumps({'public_mode':'synthetic_only','data_root':'not_available_restricted'}))
    (run/'public_handoff.json').write_text(json.dumps({'status':'incomplete','raw_real_audit_passed':False}))
    with pytest.raises(ValueError,match='handoff'):module.guard_public_run(run,tmp_path/'figures')

def test_complex_reference_and_public_plot_modules_are_loadable():
    module=load('complex_examples');helper,style,selection=module.checked_modules()
    assert helper.CASE_IDS==('syn-p000736-v4','syn-p000654-v1','syn-p000651-v2')
    assert len(helper.METHODS)==10 and len(helper.LABELS)==3
    assert module.sha(selection)==module.SELECTION_SHA
    assert callable(helper.render) and callable(helper.load_evidence)
    assert all(callable(getattr(style,name)) for name in ['_plane','_plane_record','_section','_plane_panel','_panel_labels','_alignment','export','_guard_public_output'])

@pytest.mark.parametrize('name',['secondary_contrasts','complex_examples'])
def test_supplemental_help_does_not_load_torch_or_create_output(tmp_path,name):
    command=[sys.executable,'-B',str(SUPPLEMENTAL/(name+'.py')),'--help']
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0 and 'usage:' in result.stdout.lower()
    assert not list(tmp_path.iterdir())
