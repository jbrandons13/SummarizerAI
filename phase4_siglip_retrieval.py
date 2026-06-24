import os
import sys
import json
import subprocess
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel

def extract_frames(video_path, output_dir, fps=1):
    os.makedirs(output_dir, exist_ok=True)
    # Extract frames at 1 fps
    print(f"Extracting frames from {video_path} at {fps} fps...")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps}", f"{output_dir}/%05d.jpg"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")])
    return frames

def main():
    if len(sys.argv) < 2:
        print("Usage: python phase4_siglip_retrieval.py <video_id>")
        sys.exit(1)
        
    video_id = sys.argv[1]
    video_path = f"data/raw_videos/{video_id}.mp4"
    intermediate_dir = f"data/intermediate/{video_id}"
    script_path = f"{intermediate_dir}/summary_script.json"
    
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found.")
        sys.exit(1)
        
    frames_dir = f"{intermediate_dir}/frames"
    images_dir = f"{intermediate_dir}/retrieved_images"
    os.makedirs(images_dir, exist_ok=True)
    
    frames = extract_frames(video_path, frames_dir, fps=1)
    if not frames:
        print("No frames extracted!")
        sys.exit(1)
        
    with open(script_path, "r") as f:
        data = json.load(f)
        
    shots = data.get("sentences") or data.get("segments") or data.get("shots") or (data if isinstance(data, list) else [])
    
    print("Loading SigLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    
    # Pre-compute frame features in batches
    print("Computing frame features...")
    frame_features = []
    batch_size = 64
    for i in range(0, len(frames), batch_size):
        batch_paths = frames[i:i+batch_size]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            frame_features.append(feats)
    frame_features = torch.cat(frame_features, dim=0)
    
    print("Retrieving best frames...")
    for shot in shots:
        shot_id = f"shot_{shot.get('id'):03d}" if 'id' in shot else shot.get("shot_id")
        kws = shot.get("keywords", [])
        text = ", ".join(kws) if kws else (shot.get("image_prompt") or shot.get("text", ""))
        
        inputs = processor(text=[text], padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            txt_feat = model.get_text_features(**inputs)
            txt_feat = txt_feat / txt_feat.norm(p=2, dim=-1, keepdim=True)
            
        sims = torch.matmul(frame_features, txt_feat.T).squeeze()
        best_idx = sims.argmax().item()
        best_frame = frames[best_idx]
        
        out_path = f"{images_dir}/{shot_id}.png"
        Image.open(best_frame).save(out_path)
        print(f"Shot {shot_id}: matched frame {best_frame} (sim: {sims[best_idx]:.3f})")
        
    print(f"Saved {len(shots)} images to {images_dir}")

if __name__ == "__main__":
    main()
