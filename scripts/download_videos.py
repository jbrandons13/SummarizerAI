# scripts/download_videos.py
import yt_dlp
from pathlib import Path

# (video_id, url, channel, expected_title_keyword)
# expected_title_keyword optional, just for sanity check
videos = [
    ("review_1",  "https://www.youtube.com/watch?v=wRaDbRjVrc4", "Marques Brownlee", "Earphone"),
    ("review_2",  "https://www.youtube.com/watch?v=iGeXGdYE7UE", "Marques Brownlee", "Macbook"),
    ]

OUTPUT_DIR = Path("data/eval_videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for video_id, url, channel, keyword in videos:
    output_path = OUTPUT_DIR / f"{video_id}.mp4"

    # Skip kalau sudah ada (biar bisa re-run kalau gagal di tengah)
    if output_path.exists():
        print(f"✓ {video_id} already exists, skipping")
        continue

    print(f"\n→ Downloading {video_id} ({channel})...")

    ydl_opts = {
        'format': 'best[ext=mp4][height<=720]/best[height<=720]',
        'outtmpl': str(OUTPUT_DIR / f"{video_id}.%(ext)s"),
        'writeinfojson': True,         # save metadata for thesis table
        'writesubtitles': False,        # tidak butuh, kita pakai WhisperX
        'quiet': False,
        'no_warnings': False,
        'merge_output_format': 'mp4',   # force mp4 even if separate streams
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"✓ {video_id} downloaded")
    except Exception as e:
        print(f"✗ {video_id} FAILED: {e}")
        continue

print("\n=== Done ===")
print(f"Check: ls -lh {OUTPUT_DIR}/")