import os
import json
import logging
import argparse
import datetime
from pathlib import Path
import re
import yt_dlp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def slugify(value: str) -> str:
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    value = str(value).lower()
    value = re.sub(r'[^\w\s-]', '', value)
    value = re.sub(r'[\s_-]+', '-', value)
    value = re.sub(r'^-+|-+$', '', value)
    return value

def download_youtube_video(url: str, base_dir: str = "data/intermediate") -> str:
    """
    Downloads a YouTube video and extracts its metadata.
    Returns the generated video_id.
    """
    ydl_opts_meta = {
        'quiet': True,
        'extract_flat': False,
        'no_warnings': True,
    }

    logging.info(f"Fetching metadata for {url}...")
    with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl:
        try:
            info_dict = ydl.extract_info(url, download=False)
        except Exception as e:
            logging.error(f"Failed to fetch metadata: {e}")
            raise

    yt_id = info_dict.get('id', 'unknown_id')
    title = info_dict.get('title', 'video')
    
    slug_title = slugify(title)[:40]
    video_id = f"{yt_id}_{slug_title}"
    
    out_dir = Path(base_dir) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    source_mp4 = out_dir / "source.mp4"
    source_meta = out_dir / "source_meta.json"
    
    # Caching check
    if source_mp4.exists() and source_meta.exists():
        logging.info(f"Video {video_id} already exists in cache. Skipping download.")
        return video_id

    license_info = info_dict.get('license', '')
    is_cc = license_info and 'Creative Commons' in license_info
    
    if not is_cc:
        logging.warning(f"Video {video_id} license is not Creative Commons. License found: {license_info or 'None'}. Proceeding anyway...")

    meta_data = {
        "video_id": video_id,
        "url": url,
        "title": title,
        "channel": info_dict.get('uploader', ''),
        "duration_sec": info_dict.get('duration', 0),
        "license": license_info,
        "download_timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }

    with open(source_meta, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2, ensure_ascii=False)

    ydl_opts_download = {
        'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/best[ext=mp4]',
        'outtmpl': str(source_mp4),
        'merge_output_format': 'mp4',
        'quiet': False,
    }

    logging.info(f"Downloading video {video_id} (max 1080p mp4)...")
    with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
        ydl.download([url])
        
    logging.info(f"Download complete! Saved to {source_mp4}")
    return video_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0: Download YouTube Video")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    args = parser.parse_args()
    
    video_id = download_youtube_video(args.url)
    print(f"\nSuccess. video_id: {video_id}")
