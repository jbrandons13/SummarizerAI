import sys
import os
import argparse
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
import yaml

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
logger = logging.getLogger("AggregateEval")

def main():
    parser = argparse.ArgumentParser(description="Aggregate individual evaluation results")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, help="Directory to save final results (optional)")
    
    args = parser.parse_args()
    
    # 1. Load config
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        return
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    intermediate_dir = Path(config.get("paths", {}).get("intermediate_dir", "data/intermediate"))
    results_dir = Path(config.get("paths", {}).get("results_dir", "results"))
    
    # 2. Find all individual result JSONs
    result_files = list(intermediate_dir.glob("*/eval_results_*.json"))
    if not result_files:
        logger.error(f"No evaluation results found in {intermediate_dir}")
        return
    
    all_results = []
    for file_path in result_files:
        try:
            with open(file_path, "r") as f:
                all_results.append(json.load(f))
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            
    if not all_results:
        logger.error("No valid results collected.")
        return
        
    df = pd.DataFrame(all_results)
    logger.info(f"Aggregated {len(df)} results across {df['video_id'].nunique()} videos and {df['arm'].nunique()} arms.")
    
    # 3. Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        run_dir = Path(args.output)
    else:
        run_dir = results_dir / f"aggregated_{timestamp}"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Save CSV
    csv_path = run_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    
    # 5. Generate Summary & Plots (Reusing AblationRunner logic)
    runner = AblationRunner(config)
    runner._generate_summary(df, run_dir)
    runner._generate_plots(df, run_dir)
    
    logger.info("==========================================")
    logger.info("AGGREGATION COMPLETE")
    logger.info(f"Results saved to: {run_dir}")
    logger.info("==========================================")

if __name__ == "__main__":
    main()
