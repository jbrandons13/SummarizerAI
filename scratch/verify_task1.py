import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

try:
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("YAML parses cleanly.")
    
    # Try importing the pipeline
    import pipeline
    print("Pipeline imports cleanly.")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
