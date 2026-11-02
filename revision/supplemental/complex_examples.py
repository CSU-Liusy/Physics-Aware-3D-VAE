"""Render the fixed three synthetic showcase cases from an already completed public run.

No model loading or inference. The archived evidence loader and four-figure
renderer are reused; a new portable output adapter uses the distributed plotting
module. Actual alignment/font/collision QA remains explicitly optional and its
absence is recorded, never represented as publication-ready validation.
"""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent;REVISION=HERE.parent
REFERENCE_SHA='80a973a393eb75e3624ff9317d5a5807b496f8ca2866771ecbadc36947f21d64'
SELECTION_SHA='f9b6cee797e702ca9b7e26ab33af578c093fa5e26d5c7683b0b34ad0005861ab'
PLOT_SHA='04c5ad838b5d26bd72fd94d8c9495e570fd62418423b799cb71caaa8ba0759f7'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value
def checked_modules():
    source=HERE/'reference/render_complex_synthetic.py'; selection=HERE/'reference/complex_synthetic_selection.json';plot=REVISION/'revision_figures.py'
    require(sha(source)==REFERENCE_SHA,'Archived complex helper differs')
    require(sha(selection)==SELECTION_SHA,'Fixed reference-selected examples differ')
    require(sha(plot)==PLOT_SHA,'Distributed plotting implementation differs')
    sys.path.insert(0,str(REVISION))
    return module('geovae_complex_reference',source),module('geovae_complex_plotting',plot),selection
def guard_public_run(run,out):
    run=Path(run).resolve();out=Path(out).resolve()
    require(not out.exists(),'Output exists; refusing stale/overwritten figures')
    require(not out.is_relative_to(REVISION),'Figure outputs must be outside distributed revision sources')
    identity=json.loads((run/'run_identity.json').read_text())
    require(identity.get('public_mode')=='synthetic_only' and identity.get('data_root')=='not_available_restricted','A completed independent public synthetic run is required; author/private runs are not substituted')
    handoff=json.loads((run/'public_handoff.json').read_text())
    require(handoff.get('status')=='public_synthetic_reproduction_complete' and handoff.get('raw_real_audit_passed') is False,'Matching public-synthetic handoff required')
    return run,out
def render(run,out,check_only=False):
    run,out=guard_public_run(run,out)
    helper,style,selection=checked_modules()
    manifest=run/'frozen_inputs/manifest.json'
    style._guard_public_output(run,out,manifest,False)
    cases,inputs,metadata=helper.load_evidence(run,selection,style)
    result={'status':'ready_checked_saved_synthetic_cases','fixed_cases':list(helper.CASE_IDS),'all_methods':list(helper.METHODS),'model_calls':0,'checkpoint_tensors_loaded':False,
            'reused_primary_parents':3,'new_independent_test_cases':0,'historical_helper_sha256':REFERENCE_SHA,'plot_helper_sha256':PLOT_SHA,'portable_wrapper_sha256':sha(__file__),
            'scope':'Same original reference-selected three cases and planes, all ten methods; saved public predictions only. No real or external geometallurgical input accepted.',
            'final_publication_layout_certified':False}
    if check_only:return result
    # The reference loader/renderer remain unchanged on disk. Only the save seam
    # is adapted for a public output path and the public plot exporter's explicit QA metadata.
    def portable_save(fig,destination,name,plot,contract,source_files,draft):
        require(not draft,'Toy layout fixtures cannot become scientific exports')
        plot.export(fig,destination,name,contract,source_files,alignment_options=plot._alignment([a for a in fig.axes if a.get_visible()],4 if len(fig.axes)==12 else 3),draft=False)
        p=destination/(name+'.provenance.json');v=json.loads(p.read_text())
        v.update(evidence_role='illustrative_subset_of_completed_primary_test',no_inference=True,selection_sha256=SELECTION_SHA,reference_helper_sha256=REFERENCE_SHA,portable_wrapper_sha256=sha(__file__),new_independent_test_cases=0,final_publication_layout_certified=False)
        p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
    helper.save=portable_save
    helper.render(cases,out,style,inputs,metadata,draft=False)
    result['status']='generated_saved_synthetic_figures_not_publication_QA_certification'
    result['generated']=['complex_case_1_sections','complex_case_2_sections','complex_case_3_sections','complex_reference_geometries']
    (out/'portable_readiness.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args(argv)
    try:result=render(args.run,args.out,args.check_only)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
        print(json.dumps({'status':'blocked_no_figures','error':str(error),'model_calls':0}));return 2
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
