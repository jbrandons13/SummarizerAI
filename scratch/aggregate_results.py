import os
import json
import numpy as np

base_dir = "data/intermediate"
videos = [d for d in os.listdir(base_dir) if d.startswith("review_") and os.path.isdir(os.path.join(base_dir, d))]
arms = [
    "random", "caption_direct", "caption_temporal", "caption_temporal_dp",
    "siglip_direct", "siglip_temporal", "siglip_temporal_hungarian", "siglip_temporal_dp"
]

results = {arm: {"clip": [], "temp": [], "vc": []} for arm in arms}

for video in videos:
    for arm in arms:
        path = os.path.join(base_dir, video, f"eval_results_{arm}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                results[arm]["clip"].append(data.get("clipscore_mean", 0))
                results[arm]["temp"].append(data.get("temporal_acc_15s", 0))
                results[arm]["vc"].append(data.get("visual_coherence_mean", 0))

print("| Arm | CLIPScore | TempAcc (15s) | VisCoher |")
print("|-----|-----------|---------------|----------|")
for arm in arms:
    clip_mean = np.mean(results[arm]["clip"]) if results[arm]["clip"] else 0
    temp_mean = np.mean(results[arm]["temp"]) if results[arm]["temp"] else 0
    vc_mean = np.mean(results[arm]["vc"]) if results[arm]["vc"] else 0
    print(f"| {arm} | {clip_mean:.3f} | {temp_mean:.3f} | {vc_mean:.3f} |")
