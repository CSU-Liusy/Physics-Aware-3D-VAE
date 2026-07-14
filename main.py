"""
main.py - 3D VAE entry point: parse config and invoke training pipeline.
Supported modes: train, benchmark, ablation, robust_eval
"""

import sys
import os

from config_loader import parse_args
from trainer import train

# Dynamic import of feature modules
try:
    from comparisons.run_benchmark import execute_benchmark
    from comparisons.run_ablation import main as run_ablation_main
except ImportError:
    # Attempt path correction
    sys.path.append(os.path.join(os.path.dirname(__file__), 'comparisons'))
    try:
        from comparisons.run_benchmark import execute_benchmark
        from comparisons.run_ablation import main as run_ablation_main
    except ImportError:
        pass


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'train':
        print("Starting mode: Train")
        train(args)
        print('Training finished')
        
    elif args.mode == 'benchmark':
        print("Starting mode: Benchmark")
        limit = args.num_samples if args.num_samples > 0 else 10
        execute_benchmark(
            vae_ckpt=args.checkpoint,
            limit=limit,
            data_dir=args.ply_dir if args.ply_dir else None
        )
        print('Benchmark finished')
        
    elif args.mode == 'ablation':
        print("Starting mode: Ablation Study")
        # Ablation script runs independently, we just invoke it
        run_ablation_main()
        print('Ablation study finished')
        
    elif args.mode == 'robust_eval':
        # Deferred import to avoid circular dependency
        from run_robust_analysis import run_robustness_analysis
        run_robustness_analysis(args)
        
    else:
        print(f"Unknown mode: {args.mode}")
