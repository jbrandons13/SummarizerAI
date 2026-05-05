import sys
import os
import argparse
import yaml
import logging
from pathlib import Path
import glob
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
print(f"DEBUG: sys.path = {sys.path}")

from src.eval.run_ablation import AblationRunner

def main():
    parser = argparse.ArgumentParser(description="Run Evaluation and Display Results in Terminal")
    parser.add_argument("--videos", type=str, required=True, help="Glob pattern for eval videos (e.g. 'data/eval/*.mp4')")
    parser.add_argument("--arms", type=str, default="random,caption_temporal,siglip_direct,siglip_temporal,siglip_temporal_hungarian,siglip_temporal_dp", help="Comma-separated list of arms to evaluate")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default="results/", help="Directory to save evaluation results")
    parser.add_argument("--force", action="store_true", help="Force re-run all evaluations")
    
    args = parser.parse_args()
    
    video_paths = [Path(p) for p in glob.glob(args.videos)]
    if not video_paths:
        print(f"\n❌ Error: No videos found matching pattern: {args.videos}")
        return
        
    print(f"\n🚀 Found {len(video_paths)} videos for evaluation.")
    
    # Load config
    if not Path(args.config).exists():
        print(f"❌ Error: Config file not found at {args.config}")
        return
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    if "paths" not in config:
        config["paths"] = {}
    config["paths"]["results_dir"] = args.output
    
    arms = [a.strip() for a in args.arms.split(",")]
    
    print(f"Evaluating {len(arms)} arms: {', '.join(arms)}\n")
    
    try:
        runner = AblationRunner(config)
        run_dir, _ = runner.run(video_paths, arms, force=args.force)
        
        csv_path = run_dir / "ablation_results.csv"
        if not csv_path.exists():
            print(f"❌ Error: Ablation results CSV not found at {csv_path}")
            return
            
        # Display the results beautiful terminal table
        df = pd.read_csv(csv_path)
        
        print("\n" + "="*80)
        print(" " * 28 + "EVALUATION RESULTS")
        print("="*80)
        
        # Determine the columns to display
        important_cols = ["video_id", "arm", "rouge_l", "bertscore", "clipscore_mean", "temporal_acc_15s", "visual_coherence_mean", "total_time_sec"]
        cols_to_print = [c for c in important_cols if c in df.columns]
        
        header = f"{'Arm':<26} | {'ROUGE-L':<9} | {'BERTScore':<9} | {'ClipScore':<9} | {'Temp Acc':<9} | {'Vis Coher':<9}"
        print(header)
        print("-"*80)
        
        for idx, row in df.iterrows():
            arm = str(row.get("arm", ""))
            rouge_l = float(row.get("rouge_l", 0.0))
            bertscore = float(row.get("bertscore", 0.0))
            clipscore = float(row.get("clipscore_mean", 0.0))
            temp_acc = float(row.get("temporal_acc_15s", 0.0))
            vis_coher = float(row.get("visual_coherence_mean", 0.0))
            
            line = f"{arm:<26} | {rouge_l:<9.4f} | {bertscore:<9.4f} | {clipscore:<9.4f} | {temp_acc:<9.4f} | {vis_coher:<9.4f}"
            print(line)
            
        print("="*80)
        print(f"All files and charts saved in directory: {run_dir}\n")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")

if __name__ == "__main__":
    main()
