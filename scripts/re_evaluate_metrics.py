import json
from pathlib import Path
from src.eval.metrics import temporal_alignment_score, visual_coherence_score
from src.utils.io import load_json_as_model
from src.schemas import SummaryScript, KeyframesManifest, RetrievalOutput

video_id = "501c3e27-b34c-48b4-bffa-de0e61b97ecd"
video_dir = Path("data/intermediate") / video_id

print(f"Re-evaluating metrics for video: {video_id}")

# Load Manifest
manifest_path = video_dir / "keyframes_manifest.json"
manifest = load_json_as_model(manifest_path, KeyframesManifest)
print(f"Loaded manifest with {len(manifest.scenes)} scenes")

# Load Summary
summary_path = video_dir / "summary_script.json"
summary = load_json_as_model(summary_path, SummaryScript)
print(f"Loaded summary script with {len(summary.sentences)} sentences")

# Load embeddings
import joblib
model_slug = "google_siglip2_so400m_patch16_naflex"
cache_path = video_dir / f"embeddings_{model_slug}.joblib"
frame_embeddings = {}
if cache_path.exists():
    frame_embeddings = joblib.load(cache_path)
print(f"Loaded frame embeddings with {len(frame_embeddings)} keys")

# Iterate over all arms
arms = ["random", "caption_temporal", "siglip_direct", "siglip_temporal", "siglip_temporal_hungarian", "siglip_temporal_dp"]

for arm in arms:
    # Look at existing scene matches or eval results
    matches_path = video_dir / f"scene_matches_{arm}.json"
    if not matches_path.exists():
        print(f"❌ Match output for arm {arm} does not exist at {matches_path}")
        continue
        
    matches = load_json_as_model(matches_path, RetrievalOutput)
    print(f"\nArm: {arm}")
    print(f"Total matches: {len(matches.matches)}")
    
    # 1. Temporal Alignment
    temporal = temporal_alignment_score(matches.matches, summary, manifest)
    print(f"  Temporal results: {temporal}")

    # 2. Visual Coherence
    coherence = visual_coherence_score(matches.matches, frame_embeddings)
    print(f"  Visual coherence results: {coherence}")

    # Load existing eval result if it exists to preserve ROUGE/BERTScore
    eval_result_path = video_dir / f"eval_results_{arm}.json"
    if eval_result_path.exists():
        with open(eval_result_path, "r") as f:
            eval_data = json.load(f)
    else:
        eval_data = {"video_id": video_id, "arm": arm}

    # Update metrics
    eval_data.update({
        "temporal_mean_error_s": temporal.get("mean_temporal_error_seconds"),
        "temporal_acc_5s":  temporal.get("temporal_accuracy_within_5s"),
        "temporal_acc_15s": temporal.get("temporal_accuracy_within_15s"),
        "temporal_acc_30s": temporal.get("temporal_accuracy_within_30s"),
        "temporal_acc_60s": temporal.get("temporal_accuracy_within_60s"),
        "visual_coherence_mean": coherence.get("visual_coherence_mean", 0.0),
    })

    # Write back
    with open(eval_result_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"  Updated {eval_result_path}")
