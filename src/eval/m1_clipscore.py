import os
import csv
import logging
import traceback
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

from src.eval.utils import log_error, get_video_ids

logger = logging.getLogger(__name__)

def get_video_duration(video_path: Path) -> float:
    """Get video duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        try:
            return float(result.stdout.strip())
        except ValueError:
            pass
    return 0.0

def extract_frame_at_time(video_path: Path, timestamp_sec: float) -> Image.Image:
    """Extract single frame at given timestamp using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{timestamp_sec:.3f}",
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        img = Image.open(tmp_path).convert("RGB")
        return img
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def compute_clipscore(image: Image.Image, text: str, model, processor, device: str) -> float:
    """
    Standard CLIPScore: max(0, 2.5 * cos(image_emb, text_emb))
    Reference: Hessel et al. 2021, "CLIPScore: A Reference-free Evaluation Metric for Image Captioning"
    """
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    image_emb = outputs.image_embeds  # (1, 768) L2-normalized
    text_emb = outputs.text_embeds    # (1, 768) L2-normalized
    cos_sim = (image_emb * text_emb).sum(dim=-1).item()  # scalar
    return max(0.0, 2.5 * cos_sim)

def run_m1(config: dict, vram_manager) -> int:
    """
    Run M1 (CLIPScore) evaluation for all 10 videos.
    Returns the number of successfully processed videos.
    """
    logger.info("Initializing M1 CLIPScore evaluation...")
    
    # 1. Resolve configurations
    clip_cfg = config.get("evaluation", {}).get("clipscore", {})
    model_name = clip_cfg.get("model", "openai/clip-vit-large-patch14")
    
    device = f"cuda:{vram_manager.device_id}" if torch.cuda.is_available() else "cpu"
    cache_dir = os.path.expanduser("~/models/clip_vit_l14")
    
    def load_clip():
        logger.info(f"Loading CLIP model '{model_name}' (cache_dir={cache_dir}) on device {device}...")
        proc = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        mod = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
        mod.to(device)
        mod.eval()
        return mod, proc

    # Load model via VRAMManager
    model, processor = vram_manager.load_model("CLIPScore", load_clip)
    
    video_ids = get_video_ids()
    logger.info(f"Identified {len(video_ids)} videos for M1: {video_ids}")
    
    eval_dir = Path("data/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    group_csv_path = eval_dir / "m1_clipscore_per_group.csv"
    video_csv_path = eval_dir / "m1_clipscore_per_video.csv"
    
    group_rows = []
    video_rows = []
    
    success_count = 0
    
    for video_id in video_ids:
        logger.info(f"Processing video {video_id} for M1...")
        video_path = Path("data/output") / video_id / "summary_grouping_gate.mp4"
        assignments_path = Path("data/intermediate") / video_id / "p4_assignments.json"
        manifest_path = Path("data/intermediate") / video_id / "audio_manifest.json"
        
        try:
            if not video_path.exists():
                raise FileNotFoundError(f"Output video not found: {video_path}")
            if not assignments_path.exists():
                raise FileNotFoundError(f"Assignments file not found: {assignments_path}")
            if not manifest_path.exists():
                raise FileNotFoundError(f"Audio manifest not found: {manifest_path}")
            
            with open(assignments_path, "r", encoding="utf-8") as f:
                assignments = json.load(f)
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            
            sentence_durations = {s["id"]: s["duration_seconds"] for s in manifest["sentences"]}
            sentence_texts = {s["id"]: s["text"] for s in manifest["sentences"]}
            
            # Reconstruct timeline
            cursor = 0.0
            groups = []
            for group in assignments:
                group_start = cursor
                group_text = " ".join([sentence_texts[sid] for sid in group["sentence_ids"]])
                for sid in group["sentence_ids"]:
                    cursor += sentence_durations[sid]
                group_end = cursor
                group_mid = (group_start + group_end) / 2
                groups.append({
                    "start": group_start,
                    "end": group_end,
                    "mid": group_mid,
                    "text": group_text
                })
            
            video_duration = get_video_duration(video_path)
            logger.info(f"Video {video_id} duration: {video_duration:.3f}s. Reconstructed audio sum: {cursor:.3f}s")
            
            scores = []
            for idx, g in enumerate(groups):
                # Clamp timestamp to video duration
                ts = max(0.0, min(g["mid"], video_duration - 0.01))
                frame = extract_frame_at_time(video_path, ts)
                
                # Check for debug print & save on review_1 group 0
                if video_id == "review_1" and idx == 0:
                    debug_img_path = eval_dir / "debug_review_1_group_0.png"
                    frame.save(debug_img_path)
                    logger.info(f"Saved review_1 group 0 frame to {debug_img_path}")
                
                score = compute_clipscore(frame, g["text"], model, processor, device)
                scores.append(score)
                
                if video_id == "review_1" and idx == 0:
                    logger.info(f"review_1 group 0 computed CLIPScore: {score:.4f} (expected range 0.5 - 2.5)")
                    if score < 0.0 or score > 2.5:
                        logger.warning(f"CLIPScore {score:.4f} for review_1 group 0 is outside expected [0.0, 2.5] range!")
                
                group_rows.append({
                    "video_id": video_id,
                    "group_id": idx,
                    "score": f"{score:.4f}"
                })
            
            if len(scores) > 0:
                mean_val = np.mean(scores)
                std_val = np.std(scores)
                n_groups = len(scores)
                video_rows.append({
                    "video_id": video_id,
                    "mean": f"{mean_val:.4f}",
                    "std": f"{std_val:.4f}",
                    "n_groups": str(n_groups)
                })
                success_count += 1
            else:
                raise ValueError("No groups found in assignments.")
                
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error evaluating M1 for {video_id}: {e}")
            log_error(video_id, "M1", str(e), tb)
            video_rows.append({
                "video_id": video_id,
                "mean": "NaN",
                "std": "NaN",
                "n_groups": "0"
            })
            
    # Write CSVs
    with open(group_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "group_id", "score"])
        writer.writeheader()
        writer.writerows(group_rows)
        
    with open(video_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "mean", "std", "n_groups"])
        writer.writeheader()
        writer.writerows(video_rows)
        
    logger.info(f"M1 CLIPScore evaluation complete. Success: {success_count}/{len(video_ids)}")
    return success_count

import json
