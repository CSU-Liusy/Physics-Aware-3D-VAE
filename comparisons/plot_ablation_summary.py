
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

# Try to use seaborn for better style, fallback to matplotlib if not installed
try:
    import seaborn as sns
    sns.set(style="whitegrid")
except ImportError:
    plt.style.use('ggplot')

import argparse
# ... existing imports ...

# Configuration
# Assuming this script is run from project root, or we find the path relative to this file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Default: 3dvae/comparisons/../../results/ablations -> results/ablations
DEFAULT_RESULTS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../results/ablations'))

def main():
    parser = argparse.ArgumentParser(description='Generate comparison plots for ablation results.')
    parser.add_argument('--results-dir', type=str, default=DEFAULT_RESULTS_DIR, help='Directory containing experiment subdirectories.')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for plots (default: results-dir)')
    args = parser.parse_args()

    RESULTS_DIR = os.path.abspath(args.results_dir)
    OUTPUT_DIR = os.path.abspath(args.output_dir) if args.output_dir else RESULTS_DIR

    print(f"Reading ablation results from: {RESULTS_DIR}")
    
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Directory {RESULTS_DIR} does not exist.")
        return

    # ... existing imports ...
    # This replacement fixes the syntax as args is now used in main
    # and RESULTS_DIR is local to main
    
    experiment_dirs = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    
    if not experiment_dirs:
        print("No experiment directories found.")
        return

    print(f"Found experiments: {experiment_dirs}")

    summary_data = []

    for exp_name in experiment_dirs:
        metrics_path = os.path.join(RESULTS_DIR, exp_name, 'metrics_summary.csv')
        
        if not os.path.exists(metrics_path):
            print(f"Warning: No metrics_summary.csv found for {exp_name}, skipping.")
            continue
            
        try:
            df = pd.read_csv(metrics_path)
            
            # Check if 'split' column exists
            if 'split' not in df.columns:
                print(f"Warning: 'split' column missing in {exp_name}, assuming all are test data.")
                df['split'] = 'test'
            
            # Filter for Test set (or validation if test is empty)
            test_df = df[df['split'] == 'test']
            if test_df.empty:
                print(f"Warning: No test split found for {exp_name}, using all data.")
                test_df = df
            
            # Calculate metrics
            # Available: dice, iou, precision, recall, accuracy
            metrics = {
                'Experiment': exp_name,
                'IoU_Mean': test_df['iou'].mean(),
                'IoU_Std': test_df['iou'].std(),
                'Dice_Mean': test_df['dice'].mean(),
                'Dice_Std': test_df['dice'].std(),
                'Accuracy_Mean': test_df['accuracy'].mean(),
                'Accuracy_Std': test_df['accuracy'].std(),
                'Sample_Count': len(test_df)
            }
            summary_data.append(metrics)
            print(f"Loaded {exp_name}: IoU={metrics['IoU_Mean']:.4f}, Samples={metrics['Sample_Count']}")
            
        except Exception as e:
            print(f"Error reading {metrics_path}: {e}")

    if not summary_data:
        print("No valid data found to plot.")
        return

    # Convert to DataFrame for easier plotting
    df_summary = pd.DataFrame(summary_data)
    
    # Sort for consistent plotting order (e.g., Baseline first)
    # Custom sort if needed
    order = ['Baseline', 'No_Constraint', 'No_Octree', 'No_KL']
    # Create a categorical type for sorting
    df_summary['Experiment'] = pd.Categorical(df_summary['Experiment'], categories=[x for x in order if x in df_summary['Experiment'].values], ordered=True)
    df_summary = df_summary.sort_values('Experiment')


    # --- Plotting ---
    
    # Setup CN font if possible (referenced from user preference for Chinese output)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False # Fix minus sign

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(df_summary)))

    # Plot 1: IoU
    bars = axes[0].bar(df_summary['Experiment'].astype(str), df_summary['IoU_Mean'], yerr=df_summary['IoU_Std'], capsize=5, color=colors, alpha=0.8)
    axes[0].set_title('translated_text IoU translated_text (translated_text)', fontsize=14)
    axes[0].set_ylabel('Mean IoU')
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom')

    # Plot 2: Dice
    bars2 = axes[1].bar(df_summary['Experiment'].astype(str), df_summary['Dice_Mean'], yerr=df_summary['Dice_Std'], capsize=5, color=colors, alpha=0.8)
    axes[1].set_title('translated_text Dice translated_text (translated_text)', fontsize=14)
    axes[1].set_ylabel('Mean Dice')
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'ablation_summary_plot.png')
    plt.savefig(output_path, dpi=300)
    print(f"Summary plot saved to: {output_path}")

    # Also save a CSV summary
    csv_path = os.path.join(OUTPUT_DIR, 'ablation_summary_table.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"Summary table saved to: {csv_path}")

if __name__ == '__main__':
    main()

