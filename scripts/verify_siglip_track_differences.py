import json
from pathlib import Path

videos = [f"review_{i}" for i in range(1, 11)]

print(f"\n{'Video':<12}{'C-G≠DP':<10}{'S-G≠DP':<10}{'Total':<8}")
print("-" * 44)

for video in videos:
    assignments = {}
    valid_video = True
    for arm in ["caption_temporal", "caption_temporal_dp", "siglip_temporal", "siglip_temporal_dp"]:
        path = Path(f"data/intermediate/{video}/scene_matches_{arm}.json")
        if not path.exists():
            valid_video = False
            break
        with open(path) as f:
            data = json.load(f)
        matches = data.get("matches", data) if isinstance(data, dict) else data
        assignments[arm] = [m["matched_scene_id"] for m in sorted(matches, key=lambda x: x["sentence_id"])]

    if not valid_video:
        continue

    cg = assignments["caption_temporal"]
    cd = assignments["caption_temporal_dp"]
    sg = assignments["siglip_temporal"]
    sd = assignments["siglip_temporal_dp"]

    diff_c = sum(1 for a, b in zip(cg, cd) if a != b)
    diff_s = sum(1 for a, b in zip(sg, sd) if a != b)
    n = len(cg)

    print(f"{video:<12}{diff_c:<10}{diff_s:<10}{n:<8}")

    if video == "review_1":
        print(f"  Caption G: {cg}")
        print(f"  Caption D: {cd}")
        print(f"  SigLIP  G: {sg}")
        print(f"  SigLIP  D: {sd}")
