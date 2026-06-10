import os
import json
import csv
from glob import glob

VIDEOS = ["geology", "ecology", "photosynthesis", "iphone"]

print("=== FINAL VIDEO STATUS ===")
completed_finals = 0
for vid in VIDEOS:
    for method in ["fixed_w02", "daca"]:
        path = f"runs/{vid}/video_{method}.mp4"
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024*1024)
            clips_dir = f"runs/{vid}/clips_{method.replace('_w02', '')}"
            num_shots = len(glob(f"{clips_dir}/*_av.mp4"))
            print(f"[OK] {path} ({size_mb:.1f} MB) - {num_shots} shots")
            completed_finals += 1
        else:
            print(f"[MISSING] {path}")

print(f"\nTotal finals completed: {completed_finals}/8")

print("\n=== DACA PICKS & PUSH ===")
for vid in VIDEOS:
    csv_path = f"runs/{vid}/daca/adaptive_anchor.csv"
    if not os.path.exists(csv_path):
        print(f"[{vid}] No DACA picks found.")
        continue
    
    with open(csv_path) as f:
        reader = csv.reader(f)
        w_star_list = []
        for row in reader:
            if row and row[0].startswith("shot_"):
                w_star_list.append(float(row[1]))
                
    if w_star_list:
        pushed = sum(1 for w in w_star_list if w > 0.2)
        print(f"[{vid}] Shots pushed above w=0.2: {pushed}/{len(w_star_list)}")
        print(f"      Picks: {w_star_list}")

print("\n=== FAILED I2V SHOTS ===")
# Look for error logs or just missing shots compared to storyboard
for vid in VIDEOS:
    sb_path = f"runs/{vid}/storyboard.json"
    if not os.path.exists(sb_path):
        continue
    sb = json.load(open(sb_path))
    expected_shots = [s["shot_id"] for s in sb["shots"]]
    
    for method in ["fixed", "daca"]:
        clips_dir = f"runs/{vid}/clips_{method}"
        if not os.path.exists(clips_dir):
            continue
        missing = []
        for shot in expected_shots:
            if not os.path.exists(f"{clips_dir}/{shot}_av.mp4"):
                missing.append(shot)
        if missing:
            print(f"[{vid} - {method}] Missing shots (potentially failed): {missing}")

print("\n=== OVERALL RUN STATUS ===")
try:
    with open("run_overnight.log", "r") as f:
        lines = f.readlines()
        print(f"Total log lines: {len(lines)}")
        if lines:
            print(f"Last log line: {lines[-1].strip()}")
except Exception as e:
    print(f"Could not read log file: {e}")
