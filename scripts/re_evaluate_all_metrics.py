import sys
import os
import json
import pandas as pd
from pathlib import Path

# Fix python path
sys.path.append(os.getcwd())

from src.eval.metrics import temporal_alignment_score, visual_coherence_score
from src.utils.io import load_json_as_model
from src.schemas import SummaryScript, KeyframesManifest, RetrievalOutput

# 1. Iterate over all video directories in data/intermediate
intermediate_dir = Path("data/intermediate")
video_dirs = [d for d in intermediate_dir.iterdir() if d.is_dir()]

print(f"Found {len(video_dirs)} video directories in intermediate data.")

# We will collect the updated metrics to also sync with csv files
updated_rows = {}

for video_dir in video_dirs:
    video_id = video_dir.name
    print(f"\nProcessing video ID: {video_id}")
    
    manifest_path = video_dir / "keyframes_manifest.json"
    summary_path = video_dir / "summary_script.json"
    
    if not manifest_path.exists() or not summary_path.exists():
        print(f"Skipping {video_id}: manifest or summary missing.")
        continue
        
    manifest = load_json_as_model(manifest_path, KeyframesManifest)
    summary = load_json_as_model(summary_path, SummaryScript)
    
    import joblib
    model_slug = "google_siglip2_so400m_patch16_naflex"
    cache_path = video_dir / f"embeddings_{model_slug}.joblib"
    frame_embeddings = {}
    if cache_path.exists():
        try:
            frame_embeddings = joblib.load(cache_path)
        except Exception as e:
            print(f"Warning: Failed to load joblib: {e}")
            
    arms = ["random", "caption_temporal", "siglip_direct", "siglip_temporal", "siglip_temporal_hungarian", "siglip_temporal_dp"]
    
    for arm in arms:
        matches_path = video_dir / f"scene_matches_{arm}.json"
        if not matches_path.exists():
            continue
            
        try:
            matches = load_json_as_model(matches_path, RetrievalOutput)
            
            temporal = temporal_alignment_score(matches.matches, summary, manifest)
            coherence = visual_coherence_score(matches.matches, frame_embeddings)
            
            eval_result_path = video_dir / f"eval_results_{arm}.json"
            if eval_result_path.exists():
                with open(eval_result_path, "r") as f:
                    eval_data = json.load(f)
            else:
                eval_data = {"video_id": video_id, "arm": arm}
                
            eval_data.update({
                "temporal_mean_error_s": temporal.get("mean_temporal_error_seconds"),
                "temporal_acc_5s":  temporal.get("temporal_accuracy_within_5s"),
                "temporal_acc_15s": temporal.get("temporal_accuracy_within_15s"),
                "temporal_acc_30s": temporal.get("temporal_accuracy_within_30s"),
                "temporal_acc_60s": temporal.get("temporal_accuracy_within_60s"),
                "visual_coherence_mean": coherence.get("visual_coherence_mean", 0.0),
            })
            
            with open(eval_result_path, "w") as f:
                json.dump(eval_data, f, indent=2)
                
            print(f"  Updated {eval_result_path.name}: coherence={coherence.get('visual_coherence_mean'):.4f}")
            
            # Save for csv syncing
            updated_rows[(video_id, arm)] = {
                "temporal_mean_error_s": temporal.get("mean_temporal_error_seconds"),
                "temporal_acc_5s":  temporal.get("temporal_accuracy_within_5s"),
                "temporal_acc_15s": temporal.get("temporal_accuracy_within_15s"),
                "temporal_acc_30s": temporal.get("temporal_accuracy_within_30s"),
                "temporal_acc_60s": temporal.get("temporal_accuracy_within_60s"),
                "visual_coherence_mean": coherence.get("visual_coherence_mean", 0.0),
            }
        except Exception as e:
            print(f"Error updating {arm} for {video_id}: {e}")

# 2. Sync with all ablation_results.csv files in results directory
results_dir = Path("results")
if results_dir.exists():
    csv_files = list(results_dir.glob("**/ablation_results.csv"))
    print(f"\nSyncing {len(csv_files)} ablation_results.csv files")
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            for idx, row in df.iterrows():
                vid = str(row.get("video_id", ""))
                arm = str(row.get("arm", ""))
                if (vid, arm) in updated_rows:
                    metrics = updated_rows[(vid, arm)]
                    for col, val in metrics.items():
                        df.at[idx, col] = val
            df.to_csv(csv_file, index=False)
            print(f"Successfully updated {csv_file}")
        except Exception as e:
            print(f"Error updating CSV {csv_file}: {e}")
print("\nDone!")
