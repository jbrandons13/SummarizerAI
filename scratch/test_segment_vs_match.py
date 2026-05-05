import sys
import os
sys.path.append(os.getcwd())

from pathlib import Path
import joblib
import json

from src.eval.metrics import visual_coherence_score
from src.utils.io import load_json_as_model
from src.schemas import SummaryScript, KeyframesManifest, RetrievalOutput, Phase5Output

video_id = "f72b0f8d-f76c-4767-8dd9-f8d938df7c0c"
video_dir = Path("data/intermediate") / video_id

model_slug = "google_siglip2_so400m_patch16_naflex"
cache_path = video_dir / f"embeddings_{model_slug}.joblib"
frame_embeddings = joblib.load(cache_path)

# Load existing eval results or outputs
arm = "random"
print(f"Checking for {arm}")

matches_path = video_dir / f"scene_matches_{arm}.json"
matches = load_json_as_model(matches_path, RetrievalOutput)

# There may not be a saved Phase5Output in the intermediate folder, but let's check
# Or we can recreate output.segments from existing files
# Let's see what is in metadata in data/output
out_dir = Path("data/output") / video_id
meta_path = list(out_dir.glob(f"*_summary_{arm}_metadata.json"))[0]
output = load_json_as_model(meta_path, Phase5Output)

print(f"Total segments: {len(output.segments)}, total matches: {len(matches.matches)}")

print("\n--- Match vs Segment for Random ---")
for m, seg in zip(matches.matches, output.segments):
    print(f"Match: scene={m.matched_scene_id}, ts={m.best_frame_timestamp}")
    print(f"Seg  : scene={seg.source_scene_id}, ts={seg.best_frame_timestamp}")

c_match = visual_coherence_score(matches.matches, frame_embeddings)
c_seg = visual_coherence_score(output.segments, frame_embeddings)
print(f"\nc_match: {c_match}")
print(f"c_seg  : {c_seg}")
