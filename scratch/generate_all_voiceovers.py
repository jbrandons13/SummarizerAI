import sys
import os
import yaml
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.phase3_voiceover import Phase3Voiceover
from src.utils.vram import VRAMManager

def main():
    config_path = project_root / "configs/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize VRAMManager
    vram_manager = VRAMManager(
        device_id=config.get("vram", {}).get("device_id", 0),
        limit_gb=config.get("vram", {}).get("limit_gb", 22.0)
    )

    intermediate_dir = project_root / "data/intermediate"
    review_dirs = sorted(list(intermediate_dir.glob("review_*")))

    print(f"Found {len(review_dirs)} review directories.")

    for review_dir in review_dirs:
        video_id = review_dir.name
        manifest_path = review_dir / "audio_manifest.json"
        script_path = review_dir / "summary_script.json"

        if not script_path.exists():
            print(f"[{video_id}] Skipping - summary_script.json not found.")
            continue

        # Check and fix schema issue in summary_script.json
        try:
            with open(script_path, "r") as f:
                script_data = json.load(f)
            
            modified = False
            if "style" not in script_data:
                script_data["style"] = "informative"
                modified = True
            
            # Make sure all sentences have estimated_duration_seconds
            for sent in script_data.get("sentences", []):
                if "estimated_duration_seconds" not in sent:
                    sent["estimated_duration_seconds"] = 3.0
                    modified = True

            if modified:
                with open(script_path, "w") as f:
                    json.dump(script_data, f, indent=4)
                print(f"[{video_id}] Fixed summary_script.json schema (added style field).")
        except Exception as e:
            print(f"[{video_id}] Failed to check/fix summary_script.json: {e}")
            continue

        if manifest_path.exists():
            print(f"[{video_id}] Manifest already exists at {manifest_path}. Skipping generation.")
            continue

        print(f"[{video_id}] Running Phase 3 Voiceover...")
        try:
            p3 = Phase3Voiceover(config, vram_manager)
            p3.run(script_path)
            print(f"[{video_id}] Phase 3 complete!")
        except Exception as e:
            print(f"[{video_id}] Failed with error: {e}")

if __name__ == "__main__":
    main()
