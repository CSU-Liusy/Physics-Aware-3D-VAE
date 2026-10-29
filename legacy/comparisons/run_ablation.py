"""Historical four-arm training comparison with explicit, non-overwriting outputs.

This is not the revised paper's parent-disjoint ablation experiment.
"""
from pathlib import Path
import subprocess
import sys
import json

ROOT_DIR=Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path: sys.path.insert(0,str(ROOT_DIR))
from legacy_paths import external_input, resolve_output_directory

EXPERIMENTS={
    'Baseline':['--model-type','octree','--lambda-drill','50.0','--kl-ratio','0.5'],
    'No_Constraint':['--model-type','octree','--lambda-drill','0.0','--kl-ratio','0.5'],
    'No_Octree':['--model-type','standard','--lambda-drill','50.0','--kl-ratio','0.5'],
    'No_KL':['--model-type','octree','--lambda-drill','50.0','--kl-ratio','0.0'],
}


def main(args=None):
    if args is None:
        from config_loader import parse_args
        args=parse_args()
    if not args.ply_dir:
        raise SystemExit('Legacy ablation requires an explicit external --ply-dir')
    source=external_input(args.ply_dir)
    if not source.is_dir(): raise FileNotFoundError(source)
    if args.checkpoint: raise ValueError('Legacy ablation starts fresh models; do not supply --checkpoint')
    output=resolve_output_directory(args.output_dir,[source])
    for name in EXPERIMENTS:
        case_out=output/name
        if case_out.exists(): raise FileExistsError(f'Refusing to overwrite an existing ablation arm: {case_out}')
    output.mkdir(parents=True,exist_ok=True)
    receipt={'scope':'historical_four_arm_workflow_not_revision_evidence','commands':[],'status':'running'}
    marker=output/'legacy_ablation_run.json'
    if marker.exists(): raise FileExistsError(f'Refusing to overwrite an existing ablation record: {marker}')
    for name,specific in EXPERIMENTS.items():
        command=[sys.executable,str(ROOT_DIR/'main.py'),'--config',args.config,'--mode','train','--ply-dir',str(source),'--output-dir',str(output/name),'--epochs',str(args.epochs),'--patience',str(args.patience),'--num-workers','0','--load-mode','sequential','--batch-to-mem','0.3','--save-every',str(args.save_every),'--skip-vis','--cuda' if args.cuda else '--no-cuda',*specific]
        receipt['commands'].append({'arm':name,'argv':command,'status':'running'})
        marker.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        try:
            subprocess.run(command,cwd=ROOT_DIR,check=True)
        except subprocess.CalledProcessError:
            receipt['status']='failed';receipt['commands'][-1]['status']='failed'
            marker.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
            raise
        receipt['commands'][-1]['status']='complete'
    receipt['status']='training_complete'
    marker.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('Historical ablation training complete. Existing outputs were not moved or deleted.')
    print('Plot these external outputs with comparisons/plot_ablation_summary.py --results-dir EXTERNAL_OUTPUT_DIRECTORY --output-dir EXTERNAL_PLOT_DIRECTORY')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
