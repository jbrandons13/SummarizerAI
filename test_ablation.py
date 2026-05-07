import yaml
from pathlib import Path
from src.eval.run_ablation import AblationRunner

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

print("Init AblationRunner")
runner = AblationRunner(config)
print("Runner initialized")

video_paths = [Path("data/raw/review_7.mp4")]
arms = ["caption_temporal_cvalign"]
print("Running ablation...")
runner.run(video_paths, arms, force=False)
print("Ablation finished")
