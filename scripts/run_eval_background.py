import sys
import os
import argparse
import yaml
import logging
from pathlib import Path
import glob
from datetime import datetime
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.eval.run_ablation import AblationRunner

def main():
    parser = argparse.ArgumentParser(description="Run Evaluation in the Background with Logging")
    parser.add_argument("--videos", type=str, required=True, help="Glob pattern for eval videos")
    parser.add_argument("--arms", type=str, default="random,caption_temporal,siglip_direct,siglip_temporal,siglip_temporal_hungarian,siglip_temporal_dp", help="Comma-separated list of arms to evaluate")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default="results/", help="Directory to save evaluation results")
    parser.add_argument("--force", action="store_true", help="Force re-run all evaluations")
    
    args = parser.parse_args()
    
    results_dir = Path(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = results_dir / f"eval_run_{timestamp}.log"
    
    # Setup dual-output logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Write to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Write to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info("=" * 60)
    logger.info("BACKGROUND EVALUATION STARTING")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)
    
    video_paths = [Path(p) for p in glob.glob(args.videos)]
    if not video_paths:
        logger.error(f"No videos found matching pattern: {args.videos}")
        return
        
    logger.info(f"Found {len(video_paths)} videos for evaluation.")
    
    if not Path(args.config).exists():
        logger.error(f"Config file not found at {args.config}")
        return
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    if "paths" not in config:
        config["paths"] = {}
    config["paths"]["results_dir"] = args.output
    
    arms = [a.strip() for a in args.arms.split(",")]
    
    logger.info(f"Evaluating {len(arms)} arms: {', '.join(arms)}")
    
    try:
        runner = AblationRunner(config)
        run_dir, _ = runner.run(video_paths, arms, force=args.force)
        
        csv_path = run_dir / "ablation_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            logger.info("="*60)
            logger.info("EVALUATION RESULTS SUMMARY")
            logger.info("="*60)
            
            for idx, row in df.iterrows():
                arm = str(row.get("arm", ""))
                rouge_l = float(row.get("rouge_l", 0.0))
                bertscore = float(row.get("bertscore", 0.0))
                clipscore = float(row.get("clipscore_mean", 0.0))
                temp_acc = float(row.get("temporal_acc_15s", 0.0))
                vis_coher = float(row.get("visual_coherence_mean", 0.0))
                logger.info(f"Arm: {arm:<26} | ROUGE-L: {rouge_l:.4f} | BERTScore: {bertscore:.4f} | ClipScore: {clipscore:.4f} | TempAcc: {temp_acc:.4f} | VisCoher: {vis_coher:.4f}")
            logger.info("="*60)
            
        logger.info(f"Evaluation complete. Files saved to: {run_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
