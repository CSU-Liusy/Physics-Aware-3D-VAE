"""Configuration for the isolated historical mesh-training workflow.

This interface is independent of the revised paper's frozen procedural protocol.
The YAML file is actually read before command-line overrides are applied.
"""
import argparse
from pathlib import Path
import sys

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent / 'config/default.yaml')


def load_config(config_path):
    if config_path is None:
        return {}
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Legacy configuration does not exist: {path}')
    import yaml
    with path.open(encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    if data is None: data = {}
    if not isinstance(data, dict):
        raise ValueError('Legacy configuration must be a YAML mapping')
    return data


def build_parser(defaults):
    p = argparse.ArgumentParser(description='Historical mesh-based VAE training; not the revised-paper experiment protocol', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    get = defaults.get
    p.add_argument('--lr', type=float, default=float(get('lr', 1e-4)), help='Optimizer learning rate')
    p.add_argument('--max-batch-size', type=int, default=int(get('max_batch_size', get('batch_size', 128))), help='Upper bound used by memory-based batch sizing')
    p.add_argument('--batch-to-mem', type=float, default=float(get('batch_to_mem', .8)), help='Target fraction of available device memory')
    p.add_argument('--epochs', type=int, default=int(get('epochs', 300)))
    p.add_argument('--patience', type=int, default=int(get('patience', 30)))
    p.add_argument('--patience-start', type=int, default=int(get('patience_start', 80)))
    p.add_argument('--start-epoch', type=int, default=int(get('start_epoch', 0)), help='Epoch offset when continuing a checkpoint')
    p.add_argument('--kl-cycle', type=int, default=int(get('kl_cycle', 80)))
    p.add_argument('--kl-ratio', type=float, default=float(get('kl_ratio', .5)))
    p.add_argument('--cuda', action=argparse.BooleanOptionalAction, default=bool(get('cuda', True)), help='Use CUDA if available; --no-cuda requests CPU')
    p.add_argument('--split-seed', type=int, default=int(get('split_seed', 42)))
    p.add_argument('--grid-size', type=int, nargs='+', default=list(get('grid_size', [32,32,32])))
    p.add_argument('--latent-dim', type=int, default=int(get('latent_dim', 1024)))
    p.add_argument('--base-channels', type=int, default=int(get('base_channels', 32)))
    p.add_argument('--lambda-drill', type=float, default=float(get('lambda_drill', 50.0)))
    p.add_argument('--mode', default=str(get('mode','train')), choices=['train','benchmark','ablation','robust_eval'], help='robust_eval is explicitly unsupported because its original implementation was absent')
    p.add_argument('--model-type', default=str(get('model_type','standard')), choices=['standard','octree'])
    p.add_argument('--use-lora', action=argparse.BooleanOptionalAction, default=bool(get('use_lora',False)))
    p.add_argument('--lora-preset', default=str(get('lora_preset','light')), choices=['minimal','light','standard','aggressive'])
    p.add_argument('--octree-levels', type=int, default=int(get('octree_levels',4)))
    p.add_argument('--show-recommendations', action=argparse.BooleanOptionalAction, default=bool(get('show_recommendations',False)))
    p.add_argument('--num-holes', type=int, default=int(get('num_holes',8)))
    p.add_argument('--samples-per-hole', type=int, default=int(get('samples_per_hole',12)))
    p.add_argument('--augment', type=int, default=int(get('augment',0)))
    p.add_argument('--num-samples', type=int, default=int(get('num_samples',0)))
    p.add_argument('--train-frac', type=float, default=float(get('train_frac',1)))
    p.add_argument('--base-kb-per-sample', type=float, default=float(get('base_kb_per_sample',20.0)))
    p.add_argument('--max-samples-per-file', type=int, default=int(get('max_samples_per_file',30)))
    p.add_argument('--min-samples-per-file', type=int, default=int(get('min_samples_per_file',1)))
    p.add_argument('--output-dir', default=get('output_dir'), help='Required external run output directory; no repository-local fallback')
    p.add_argument('--save-every', type=int, default=int(get('save_every',50)))
    p.add_argument('--vis-every', type=int, default=int(get('vis_every',1000)))
    p.add_argument('--skip-vis', action=argparse.BooleanOptionalAction, default=bool(get('skip_vis',True)), help='Use --no-skip-vis to enable final spin rendering')
    p.add_argument('--save-svg', action=argparse.BooleanOptionalAction, default=bool(get('save_svg',False)))
    p.add_argument('--max-output-points', type=int, default=int(get('max_output_points',0)))
    p.add_argument('--max-output-per-file', type=int, default=int(get('max_output_per_file',3)))
    p.add_argument('--output-workers', type=int, default=int(get('output_workers',12)))
    p.add_argument('--checkpoint', default=get('checkpoint'), help='Explicit trusted legacy checkpoint; no machine-specific fallback')
    p.add_argument('--unet-checkpoint', default=get('unet_checkpoint'), help='Optional trusted legacy U-Net checkpoint for benchmark mode')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=bool(get('amp',False)))
    p.add_argument('--log-mode', default=str(get('log_mode','full')), choices=['full','brief'])
    p.add_argument('--verbosity', default=get('verbosity'), choices=['full','brief'])
    p.add_argument('--num-workers', type=int, default=int(get('num_workers',0)))
    p.add_argument('--force-regen-cache', action=argparse.BooleanOptionalAction, default=bool(get('force_regen_cache',False)), help='Rebuild only matching files in this run-owned mesh cache')
    p.add_argument('--load-mode', default=str(get('load_mode','parallel')), choices=['parallel','sequential'])
    p.add_argument('--generate-only', action=argparse.BooleanOptionalAction, default=bool(get('generate_only',False)))
    p.add_argument('--ply-file', default=get('ply_file'), help='One explicitly supplied external PLY mesh')
    p.add_argument('--ply-dir', default=get('ply_dir'), help='Explicit external directory of authorized PLY inputs; never modified')
    p.add_argument('--train-after-generate', action=argparse.BooleanOptionalAction, default=bool(get('train_after_generate',False)))
    return p


def parse_args(argv=None):
    tokens=list(sys.argv[1:] if argv is None else argv)
    if '--help' in tokens or '-h' in tokens:
        parser=build_parser({})
        parser.add_argument('--config',default=DEFAULT_CONFIG_PATH,help='YAML configuration, read before CLI overrides')
        parser.print_help()
        raise SystemExit(0)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=DEFAULT_CONFIG_PATH)
    known, _ = pre.parse_known_args(tokens)
    config_path = str(Path(known.config).expanduser().resolve())
    parser = build_parser(load_config(config_path))
    parser.add_argument('--config', default=config_path, help='YAML configuration, read before CLI overrides')
    args = parser.parse_args(tokens)
    args.config = str(Path(args.config).expanduser().resolve())
    return args
