import sys
import os
from pathlib import Path

# Add project root to sys.path to allow running from scripts/ directory
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import argparse
import yaml
import logging
from src.pipeline import VideoSummarizerPipeline

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PipelineRunner")

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

    parser = argparse.ArgumentParser(description="Run the Video Summarizer Pipeline")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--method", type=str, default="siglip_direct", choices=["siglip_direct", "caption_cosine", "random", "grouping_gate"], help="Retrieval method")
    parser.add_argument("--phases", type=str, default="1,2,3,4,5", help="Comma-separated list of phases to run (e.g., '1,2,3' or '1,2,3,4,5')")
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Ensure mandatory paths in config match implementation expectations
    if "paths" in config:
        config["output_dir"] = config["paths"].get("output_dir", "data/output")
        config["intermediate_dir"] = config["paths"].get("intermediate_dir", "data/intermediate")

    try:
        logger.info(f"Starting pipeline for: {video_path}")
        pipeline = VideoSummarizerPipeline(config)
        
        phases = [int(p.strip()) for p in args.phases.split(",") if p.strip().isdigit()]
        stop_after_phase = max(phases) if phases else 5
        
        output = pipeline.run(video_path, method=args.method, stop_after_phase=stop_after_phase)
        
        logger.info("==========================================")
        logger.info("PIPELINE SUCCESSFUL")
        if output:
            logger.info(f"Video: {output.output_path}")
            logger.info(f"Total Duration: {output.total_duration_seconds:.2f}s")
        else:
            logger.info(f"Pipeline stopped early after Phase {stop_after_phase}")
        logger.info("==========================================")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
