import sys
import os
import argparse
import yaml
import logging
from pathlib import Path
import glob

# Add project root to sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.eval.run_ablation import AblationRunner

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EvalCLI")

def main():
    # Load .env if exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Run Evaluation Ablation Study")
    parser.add_argument("--videos", type=str, required=True, help="Glob pattern for eval videos (e.g. 'data/eval/*.mp4')")
    parser.add_argument("--arms", type=str, default="random,caption_cosine,siglip_direct", help="Comma-separated list of arms to evaluate")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default="results/", help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # 1. Resolve videos
    video_paths = [Path(p) for p in glob.glob(args.videos)]
    if not video_paths:
        logger.error(f"No videos found matching pattern: {args.videos}")
        return
    
    logger.info(f"Found {len(video_paths)} videos for evaluation.")
    
    # 2. Resolve arms
    arms = [a.strip() for a in args.arms.split(",")]
    
    # 3. Load config
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        return
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Update paths in config
    if "paths" not in config:
        config["paths"] = {}
    config["paths"]["results_dir"] = args.output
    
    # 4. Run Evaluation
    try:
        runner = AblationRunner(config)
        results_path = runner.run(video_paths, arms)
        
        logger.info("==========================================")
        logger.info("EVALUATION COMPLETE")
        logger.info(f"Results saved to: {results_path}")
        logger.info("==========================================")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
