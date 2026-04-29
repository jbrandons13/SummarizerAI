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

import warnings

# Aggressive Silence
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from src.eval.run_ablation import AblationRunner

# Setup Logging - ULTRA CLEAN
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Silence EVERYTHING except our errors/info
for logger_name in ["httpx", "urllib3", "absl", "transformers", "pytorch_lightning", "lightning_fabric", "scenedetect", "pyannote", "kokoro"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

logger = logging.getLogger("EvalCLI")

def main():
    # Load .env if exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # Manual load if dotenv not installed
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path, "r") as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        key, _, value = line.partition("=")
                        os.environ[key.strip()] = value.strip()

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
        results_path, _ = runner.run(video_paths, arms)
        
        logger.info("==========================================")
        logger.info("EVALUATION COMPLETE")
        logger.info(f"Results saved to: {results_path}")
        logger.info("==========================================")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
