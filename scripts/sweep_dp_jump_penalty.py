"""
Sweep jump_penalty values for DP, run on review_3, and collect metrics.
"""
import numpy as np
from pathlib import Path
import yaml
import subprocess
import json
import shutil

JUMP_PENALTIES = [0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
VIDEO_ID = "review_1"
VIDEO_PATH = f"data/eval_videos/{VIDEO_ID}.mp4"
ARM_NAME = "siglip_temporal_dp"
CONFIG_PATH = Path("configs/default.yaml")

def run_with_jump_penalty(jp: float):
    """Run pipeline with a specific jump_penalty and return metrics."""
    # 1. Update Config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    original_jp = config["retrieval"]["dp_jump_penalty"]
    config["retrieval"]["dp_jump_penalty"] = jp

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)

    try:
        # 2. Clear caches for this specific arm to force re-retrieval
        # We don't use --force to avoid re-running Phase 1-3
        eval_file = Path(f"data/intermediate/{VIDEO_ID}/eval_results_{ARM_NAME}.json")
        match_file = Path(f"data/intermediate/{VIDEO_ID}/scene_matches_{ARM_NAME}.json")
        
        if eval_file.exists(): eval_file.unlink()
        if match_file.exists(): match_file.unlink()

        # 3. Run Evaluation
        print(f"   Executing evaluation for jp={jp}...")
        cmd = [
            "conda", "run", "-n", "sumarizer", "python", "scripts/run_eval_terminal.py",
            "--videos", VIDEO_PATH,
            "--arms", ARM_NAME
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        # 4. Load results
        if not eval_file.exists():
            print(f"❌ Error: Evaluation failed to produce {eval_file}")
            return None
            
        with open(eval_file) as f:
            metrics = json.load(f)
        return metrics

    finally:
        # Restore Config
        config["retrieval"]["dp_jump_penalty"] = original_jp
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(config, f)

def main():
    results = []
    print(f"🚀 Starting DP jump_penalty sweep on {VIDEO_ID}")
    print(f"Targeting ARM: {ARM_NAME}")
    
    for jp in JUMP_PENALTIES:
        print(f"\n=== Running jump_penalty = {jp} ===")
        metrics = run_with_jump_penalty(jp)
        if metrics:
            results.append({
                "jump_penalty": jp,
                "clipscore": metrics.get("clipscore_mean"),
                "temp_acc": metrics.get("temporal_acc_30s"),
                "vis_coher": metrics.get("visual_coherence_mean"),
            })

    # Print Results Table
    print("\n" + "="*50)
    print(f"{'JP':<8} | {'CLIPScore':<10} | {'TempAcc':<10} | {'VisCoher':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['jump_penalty']:<8.2f} | {r['clipscore']:<10.4f} | {r['temp_acc']:<10.4f} | {r['vis_coher']:<10.4f}")
    print("="*50)

    # Save findings
    Path("notes").mkdir(exist_ok=True)
    with open("notes/dp_jump_penalty_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Sweep complete. Results saved to notes/dp_jump_penalty_sweep.json")

if __name__ == "__main__":
    main()
