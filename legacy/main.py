"""Isolated historical mesh-training entry point; imports heavy modules lazily."""
from pathlib import Path
from config_loader import parse_args
from legacy_paths import external_input, resolve_output_directory


def main(argv=None):
    args = parse_args(argv)
    if args.mode == 'robust_eval':
        raise SystemExit('Legacy robust_eval is unsupported: run_robust_analysis.py was absent from the original repository. Use revision/main.py --robustness with a completed matching revision run; it is a separate protocol.')
    if args.mode == 'ablation':
        from comparisons.run_ablation import main as run
        return run(args)
    if args.output_dir is None:
        raise SystemExit('Supply an explicit external --output-dir for legacy outputs and caches')
    if not args.ply_dir and args.ply_file:
        args.ply_dir = str(Path(args.ply_file).expanduser().resolve().parent)
    if not args.ply_dir:
        raise SystemExit('Supply an explicit external --ply-dir or --ply-file. No private data or implicit data-directory fallback is distributed.')
    if not Path(args.ply_dir).expanduser().is_dir():
        raise SystemExit(f'PLY directory does not exist: {args.ply_dir}')
    if args.ply_file and not Path(args.ply_file).expanduser().is_file():
        raise SystemExit(f'PLY file does not exist: {args.ply_file}')
    for name in ('checkpoint','unet_checkpoint'):
        path = getattr(args,name,None)
        if path and not Path(path).expanduser().is_file():
            raise SystemExit(f'Checkpoint does not exist: {path}')
    args.ply_dir = str(external_input(args.ply_dir))
    if args.ply_file: args.ply_file = str(external_input(args.ply_file))
    for name in ('checkpoint','unet_checkpoint'):
        path = getattr(args,name,None)
        if path: setattr(args,name,str(external_input(path)))
    if args.generate_only and not args.ply_file:
        raise SystemExit('--generate-only requires an explicit --ply-file')
    if args.mode == 'benchmark' and not args.checkpoint:
        raise SystemExit('Legacy benchmark requires an explicit trusted --checkpoint; no checkpoint is selected from previous output folders.')
    args.output_dir = str(resolve_output_directory(args.output_dir,[args.ply_dir]))
    if args.mode == 'train':
        from trainer import train
        train(args)
    elif args.mode == 'benchmark':
        from comparisons.run_benchmark import execute_benchmark
        execute_benchmark(vae_ckpt=args.checkpoint,unet_ckpt=args.unet_checkpoint,data_dir=args.ply_dir,output_dir=args.output_dir,limit=args.num_samples if args.num_samples>0 else 10,use_cuda=args.cuda,file_list=args.ply_file)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
