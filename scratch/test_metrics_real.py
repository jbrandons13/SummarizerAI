import sys
import os
sys.path.append(os.getcwd())

from pathlib import Path
import joblib
import json

from src.eval.metrics import temporal_alignment_score, visual_coherence_score
from src.utils.io import load_json_as_model
from src.schemas import SummaryScript, KeyframesManifest, RetrievalOutput, Phase5Output

video_id = "f72b0f8d-f76c-4767-8dd9-f8d938df7c0c"
video_dir = Path("data/intermediate") / video_id

print(f"Loading files from {video_dir}")
manifest = load_json_as_model(video_dir / "keyframes_manifest.json", KeyframesManifest)
summary = load_json_as_model(video_dir / "summary_script.json", SummaryScript)

model_slug = "google_siglip2_so400m_patch16_naflex"
cache_path = video_dir / f"embeddings_{model_slug}.joblib"
frame_embeddings = joblib.load(cache_path)

print(f"Keys type check: {type(list(frame_embeddings.keys())[0][0])}, {type(list(frame_embeddings.keys())[0][1])}")

arms = ["random", "caption_temporal", "siglip_direct", "siglip_temporal", "siglip_temporal_hungarian", "siglip_temporal_dp"]
for arm in arms:
    eval_path = video_dir / f"eval_results_{arm}.json"
    with open(eval_path, "r") as f:
        eval_data = json.load(f)
    print(f"\nARM: {arm}")
    print(f"Original eval: {eval_data.get('temporal_acc_15s')}, {eval_data.get('visual_coherence_mean')}")

    # Load from scene_matches
    matches_path = video_dir / f"scene_matches_{arm}.json"
    matches = load_json_as_model(matches_path, RetrievalOutput)
    
    # Calculate
    t_res = temporal_alignment_score(matches.matches, summary, manifest)
    c_res = visual_coherence_score(matches.matches, frame_embeddings)
    print(f"Calculated: temporal_acc_15s={t_res.get('temporal_accuracy_within_15s')}, visual_coherence={c_res.get('visual_coherence_mean')}")
