import os
import json
import shutil
import csv

for vid in ["geology", "ecology"]:
    sweep_dir = f"runs/{vid}/sweep"
    daca_csv = f"runs/{vid}/daca/adaptive_anchor.csv"
    manifest_path = f"{sweep_dir}/manifest.json"
    
    if not os.path.exists(daca_csv) or not os.path.exists(manifest_path):
        continue
        
    man = json.load(open(manifest_path))
    
    # parse daca csv for w* per shot
    w_star = {}
    with open(daca_csv) as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].startswith("shot_"):
                # shot, adaptive_w*, content_at_w*, concept_at_w*
                w_star[row[0].strip('"')] = float(row[1])
                
    # parse manifest to get image paths
    img_map = {} # (shot_id, weight) -> path
    if isinstance(man, list):
        for item in man:
            img_map[(item.get("shot_id", "shot"), round(float(item["weight"]), 4))] = item["path"]
    else:
        for r in man["rows"]:
            label = r.get("label", "shot")
            for c in r["cells"]:
                img_map[(label, round(float(c["weight"]), 4))] = c["image"]
                
    # copy files
    fixed_dir = f"runs/{vid}/images_fixed_w02"
    daca_dir = f"runs/{vid}/images_daca"
    os.makedirs(fixed_dir, exist_ok=True)
    os.makedirs(daca_dir, exist_ok=True)
    
    for shot, w in w_star.items():
        fixed_path = img_map.get((shot, 0.2))
        daca_path = img_map.get((shot, w))
        
        if fixed_path and os.path.exists(fixed_path):
            shutil.copy(fixed_path, f"{fixed_dir}/{shot}.png")
        if daca_path and os.path.exists(daca_path):
            shutil.copy(daca_path, f"{daca_dir}/{shot}.png")
