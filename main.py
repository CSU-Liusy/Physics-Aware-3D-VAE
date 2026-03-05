"""Entry point for 3D VAE training and evaluation workflows.

Supported modes: ``train``, ``benchmark``, ``ablation``, and ``robust_eval``.
"""

import sys
import os

from config_loader import parse_args
from trainer import train

# English comment for public release.
try:
    from comparisons.run_benchmark import execute_benchmark
    from comparisons.run_ablation import main as run_ablation_main
except ImportError:
    # English comment for public release.
    sys.path.append(os.path.join(os.path.dirname(__file__), 'comparisons'))
    try:
        from comparisons.run_benchmark import execute_benchmark
        from comparisons.run_ablation import main as run_ablation_main
    except ImportError:
        pass


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'train':
        print("Mode: train")
        train(args)
        print('Training finished')
        
    elif args.mode == 'benchmark':
        print("Mode: benchmark")
        limit = args.num_samples if args.num_samples > 0 else 10
        execute_benchmark(
            vae_ckpt=args.checkpoint,
            limit=limit,
            data_dir=args.ply_dir if args.ply_dir else None
        )
        print('Benchmark finished')
        
    elif args.mode == 'ablation':
        print("Mode: ablation")
        # Ablation script runs independently, we just invoke it
        run_ablation_main()
        print('Ablation finished')
        
    elif args.mode == 'robust_eval':
        # English comment for public release.
        from run_robust_analysis import run_robustness_analysis
        run_robustness_analysis(args)
        
    else:
        print(f"Unknown mode: {args.mode}")
