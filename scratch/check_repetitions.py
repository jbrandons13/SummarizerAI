import json
import os
from pathlib import Path

output_dir = Path("data/output")
results = {}

for video_dir in output_dir.iterdir():
    if not video_dir.is_dir():
        continue
    
    video_id = video_dir.name
    results[video_id] = {}
    
    for json_file in video_dir.glob("*.json"):
        with open(json_file, "r") as f:
            try:
                data = json.load(f)
                if "segments" in data:
                    scene_ids = [s["source_scene_id"] for s in data["segments"]]
                    num_total = len(scene_ids)
                    num_unique = len(set(scene_ids))
                    if num_total > num_unique:
                        results[video_id][json_file.name] = {
                            "total": num_total,
                            "unique": num_unique,
                            "repeated": num_total - num_unique,
                            "scenes": scene_ids
                        }
            except:
                continue

print(json.dumps(results, indent=2))
