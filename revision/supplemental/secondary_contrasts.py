"""Reproduce all20 secondary contrasts from the distributed saved scalar data.

This is post-results scalar reanalysis, not training, new predictions or a
private/raw-source integrity audit. The historical helper is retained unchanged;
only its pure fixed compute function is used after explicit public-input checks.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
REVISION=HERE.parent
INDEX_SHA='88c14f0b05fe29bbcec47a91c9852a67105bf8d946638cdb43bd70b7bf223b04'
MANIFEST_SHA='48ec0924b76a1cca26715bb77566646dacb5fe9a3252471f8e4690aae8e01030'
REFERENCE_SHA='632aa4a42500ef1d1591bc32dea78f418c2dbde0fe6dd5b31a1c448cc74cd781'
REL_TOL, ABS_TOL=1e-12,1e-15

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(value,message):
    if not value: raise ValueError(message)
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        require(reader.fieldnames and len(reader.fieldnames)==len(set(reader.fieldnames)), 'Invalid CSV headers')
        result=list(reader)
        require(all(None not in row and None not in row.values() for row in result),'Malformed CSV')
        return result
def reference():
    p=HERE/'reference/secondary_ablation_contrasts.py'
    require(sha(p)==REFERENCE_SHA,'Historical secondary helper differs')
    spec=importlib.util.spec_from_file_location('geovae_secondary_reference',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def load_inputs(source_data,manifest):
    source_data,manifest=Path(source_data).resolve(),Path(manifest).resolve()
    require(sha(source_data/'SOURCE_DATA_INDEX.json')==INDEX_SHA,'Expected distributed source-data index differs')
    require(sha(manifest)==MANIFEST_SHA,'Expected frozen synthetic manifest differs')
    index=json.loads((source_data/'SOURCE_DATA_INDEX.json').read_text())
    records={row['file']:row for row in index['source_data_files']}
    protocol=json.loads(manifest.read_text())
    require(protocol['protocol_version']=='procedural-lens-v1' and protocol['count']==5000,'Wrong synthetic protocol')
    parents={x['sample_id']:x['parent_id'] for x in protocol['samples'] if x['split']=='test'}
    require(len(parents)==750 and Counter(Counter(parents.values()).values())=={5:150},'Expected750cases/150parents')
    helper=reference();arms={};inputs={'manifest.json':sha(manifest),'SOURCE_DATA_INDEX.json':INDEX_SHA}
    for arm,methods in helper.ARMS.items():
        relative=f'files/ablation/{arm}/recomputed_per_case.csv'
        path=(source_data/relative).resolve()
        require(path.is_relative_to(source_data) and relative in records,'Unregistered source path')
        require(sha(path)==records[relative]['sha256'],'Ablation saved-scalar input differs: '+arm)
        inputs[relative]=sha(path)
        all_rows=rows(path); require(len(all_rows)==2250*len(methods),'Incomplete ablation method coverage: '+arm)
        keys=[(int(x['seed']),x['sample_id'],x['method']) for x in all_rows]
        wanted={(seed,sid,method) for seed in helper.SEEDS for sid in parents for method in methods}
        require(len(keys)==len(set(keys)) and set(keys)==wanted,'Duplicate/missing ablation pairing: '+arm)
        for row in all_rows:
            require(row['parent_id']==parents[row['sample_id']] and row['mask_seed']=='42','Parent/mask mismatch')
            for metric in helper.METRICS:
                value=float(row[metric]);require(math.isfinite(value) and 0<=value<=1,'Invalid metric: '+metric)
        arms[arm]={(int(row['seed']),row['sample_id']):row for row in all_rows if row['method']==methods[0]}
    expected={}
    for name in ['contrasts.csv','seed_differences.csv','parent_differences.csv']:
        relative='files/secondary/'+name;path=source_data/relative
        require(sha(path)==records[relative]['sha256'],'Distributed secondary result differs: '+name)
        inputs[relative]=sha(path);expected[name]=rows(path)
    return helper,arms,parents,inputs,expected
def compare(actual,expected):
    require(len(actual)==len(expected),'Saved result row count differs')
    largest=0.0; numeric=0
    for a,b in zip(actual,expected):
        require(set(a)==set(b),'Saved result columns differ')
        for key,value in a.items():
            if isinstance(value,(int,float)):
                ref=float(b[key]);delta=abs(float(value)-ref);largest=max(largest,delta);numeric+=1
                require(math.isfinite(ref) and math.isclose(float(value),ref,rel_tol=REL_TOL,abs_tol=ABS_TOL),'Saved numerical result differs: '+key)
            else: require(str(value)==b[key],'Saved result identifier/order differs: '+key)
    return {'numeric_values_compared':numeric,'max_absolute_difference':largest}
def analyze(source_data,manifest,out=None):
    helper,arms,parents,inputs,expected=load_inputs(source_data,manifest)
    estimates,seeds,groups,numpy_version=helper.compute(arms,parents)
    actual={'contrasts.csv':estimates,'seed_differences.csv':seeds,'parent_differences.csv':groups}
    comparisons={name:compare(value,expected[name]) for name,value in actual.items()}
    result={'status':'passed_saved_scalar_reanalysis','not_preregistered':True,'contrasts':20,'metrics':list(helper.METRICS),'contrast_definitions':[list(x) for x in helper.CONTRASTS],
            'bootstrap':{'replicates':2000,'seed':20261025,'unit':'generator_parent','parents':150,'conditional_on_three_executed_fits':True},
            'numeric_comparison_policy':{'relative_tolerance':REL_TOL,'absolute_tolerance':ABS_TOL},'comparisons':comparisons,'numpy_version':numpy_version,
            'source_sha256':inputs,'historical_compute_helper_sha256':REFERENCE_SHA,'portable_wrapper_sha256':sha(__file__),
            'model_calls':0,'prediction_or_grid_arrays_read':0,'private_case_records_read':0,
            'interpretation':'Four contrasts times five metrics, retaining all directions; pointwise percentile intervals, not multiplicity-adjusted. Reanalysis of distributed saved scalars does not independently authenticate private/raw prediction provenance.'}
    if out is not None:
        out=Path(out).resolve();require(not out.exists(),'Output exists; refusing overwrite')
        require(not out.is_relative_to(REVISION),'Outputs must be outside distributed revision sources')
        out.mkdir(parents=True)
        for name,value in actual.items():helper.write_csv(out/name,value)
        (out/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-data',type=Path,default=REVISION/'evidence/source_data')
    parser.add_argument('--manifest',type=Path,default=REVISION/'protocol/manifest.json')
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--verify-only',action='store_true');mode.add_argument('--out',type=Path)
    args=parser.parse_args(argv)
    try: result=analyze(args.source_data,args.manifest,args.out)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
        print(json.dumps({'status':'blocked_no_reanalysis','error':str(error),'model_calls':0}));return 2
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
