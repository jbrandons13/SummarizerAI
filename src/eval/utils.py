import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Loads config from the given YAML path."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def log_error(video_id: str, metric_id: str, error_msg: str, tb: str):
    """Write an error log to data/evaluation/errors.log"""
    log_dir = Path("data/evaluation")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "errors.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"video_id: {video_id}\n")
        f.write(f"metric_id: {metric_id}\n")
        f.write(f"error_msg: {error_msg}\n")
        f.write("traceback:\n")
        f.write(tb)
        f.write("\n" + "="*80 + "\n")

def get_video_ids() -> list:
    """Returns sorted video IDs (review_1, review_2, ..., review_10)."""
    output_dir = Path("data/output")
    video_ids = []
    if output_dir.exists():
        for d in output_dir.iterdir():
            if d.is_dir() and d.name.startswith("review_"):
                # Ensure it has a numeric suffix
                parts = d.name.split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    video_ids.append(d.name)
    video_ids.sort(key=lambda x: int(x.split("_")[1]))
    return video_ids
