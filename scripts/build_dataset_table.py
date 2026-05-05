# scripts/build_dataset_table.py
import json
from pathlib import Path
import csv

VIDEO_DIR = Path("data/eval_videos")
OUTPUT_CSV = Path("data/dataset_summary.csv")

rows = []
for info_file in sorted(VIDEO_DIR.glob("*.info.json")):
    with open(info_file) as f:
        info = json.load(f)

    video_id = info_file.stem.replace(".info", "")
    rows.append({
        "video_id": video_id,
        "channel": info.get("uploader", ""),
        "title": info.get("title", ""),
        "duration_seconds": info.get("duration", 0),
        "duration_minutes": round(info.get("duration", 0) / 60, 2),
        "upload_date": info.get("upload_date", ""),
        "view_count": info.get("view_count", 0),
        "url": info.get("webpage_url", ""),
    })

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"✓ Dataset table saved to {OUTPUT_CSV}")
print(f"Total videos: {len(rows)}")