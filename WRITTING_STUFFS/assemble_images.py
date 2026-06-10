import os
import sys
import json
import shutil

def assemble(vid_dir):
    manifest_path = os.path.join(vid_dir, "sweep", "manifest.json")
    picks_path = os.path.join(vid_dir, "daca", "adaptive_anchor.csv")
    
    if not os.path.exists(manifest_path) or not os.path.exists(picks_path):
        print(f"Missing data in {vid_dir}")
        return
        
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    picks = {}
    with open(picks_path, "r") as f:
        lines = f.readlines()
        in_shots = False
        for line in lines:
            if line.startswith("shot,adaptive_w*"):
                in_shots = True
                continue
            if in_shots and line.strip() == "":
                break
            if in_shots:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    shot_id = parts[0].strip('"')
                    w_star = float(parts[1])
                    picks[shot_id] = w_star
        
    fixed_dir = os.path.join(vid_dir, "images_fixed_w02")
    daca_dir = os.path.join(vid_dir, "images_daca")
    os.makedirs(fixed_dir, exist_ok=True)
    os.makedirs(daca_dir, exist_ok=True)
    
    for shot_id, w_star in picks.items():
        # Find w=0.2
        w02_path = None
        for item in manifest:
            if item["shot_id"] == shot_id and abs(item["weight"] - 0.2) < 0.01:
                w02_path = item["path"]
                break
        
        # Find w_star
        wstar_path = None
        for item in manifest:
            if item["shot_id"] == shot_id and abs(item["weight"] - w_star) < 0.01:
                wstar_path = item["path"]
                break
                
        if w02_path and os.path.exists(w02_path):
            shutil.copy(w02_path, os.path.join(fixed_dir, f"{shot_id}.png"))
        if wstar_path and os.path.exists(wstar_path):
            shutil.copy(wstar_path, os.path.join(daca_dir, f"{shot_id}.png"))

if __name__ == "__main__":
    assemble(sys.argv[1])
