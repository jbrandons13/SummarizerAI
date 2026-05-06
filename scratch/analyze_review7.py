import json
from pathlib import Path

review_7_dir = Path("data/output/review_7")
results = {}

for json_file in review_7_dir.glob("*.json"):
    with open(json_file, "r") as f:
        try:
            data = json.load(f)
            if "segments" in data:
                scene_ids = [s["source_scene_id"] for s in data["segments"]]
                results[json_file.name] = {
                    "method": data.get("method", "unknown"),
                    "scenes": scene_ids,
                    "num_sentences": len(data["segments"])
                }
        except:
            continue

# Check manifest for scene count
manifest_path = Path("data/intermediate/review_7/keyframes_manifest.json")
num_scenes = 0
if manifest_path.exists():
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        num_scenes = len(manifest.get("scenes", []))

print(f"Video ID: review_7")
print(f"Total Scenes in Video: {num_scenes}")
print("-" * 50)
for file, res in results.items():
    unique = len(set(res["scenes"]))
    total = len(res["scenes"])
    status = "LOOPING" if unique < total else "CLEAN"
    print(f"{file:50} | {status} | Scenes: {res['scenes']}")
