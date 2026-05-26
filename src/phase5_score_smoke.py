import os
import sys
import json
import yaml
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image

# Setup paths and import modules
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from src.utils.vram import VRAMManager
from src.models.siglip import SigLIPEncoder

def extract_all_frames(video_path: Path) -> list:
    """Extract all frames from a video path as PIL Images."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

def encode_images_in_batch(images: list, siglip: SigLIPEncoder, batch_size: int = 16) -> np.ndarray:
    """Encode PIL images in batches and return L2-normalized numpy features."""
    siglip._load()
    processor = siglip.processor
    model = siglip.model
    
    embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt").to("cuda")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = getattr(features, "pooler_output", features[0])
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

def compute_hsv_hist(img: Image.Image) -> np.ndarray:
    """Compute normalized 3D HSV histogram for an image."""
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 8 bins per dimension for robustness
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    
    # L1 normalization using pure numpy
    hist_sum = np.sum(hist)
    if hist_sum > 0:
        hist = hist / hist_sum
    return hist

def compute_color_dist(cond_img: Image.Image, gen_frames: list) -> float:
    """Calculate the average Bhattacharyya distance between the conditioning frame and all video frames in HSV."""
    cond_hist = compute_hsv_hist(cond_img)
    dists = []
    for frame in gen_frames:
        frame_hist = compute_hsv_hist(frame)
        dist = cv2.compareHist(cond_hist, frame_hist, cv2.HISTCMP_BHATTACHARYYA)
        dists.append(dist)
    return float(np.mean(dists))

def main():
    print("=== STARTING SMOKE TEST SCORING ===")
    
    # Paths definition
    workspace_root = repo_root.parent
    inputs_dir = repo_root / "phase5_smoke_inputs"
    outputs_dir = workspace_root / "phase5_smoke_outputs"
    
    # 1. Verification of inputs and outputs
    models = ["wan22_5b", "cogvideox_5b"]
    inputs = ["input_a_strong", "input_b_marginal", "input_c_generate"]
    
    for input_name in inputs:
        cond_frame = inputs_dir / input_name / "frame.jpg"
        prompt_file = inputs_dir / input_name / "text.txt"
        
        if not cond_frame.exists():
            raise FileNotFoundError(f"Missing conditioning frame: {cond_frame}")
        if not prompt_file.exists():
            raise FileNotFoundError(f"Missing prompt text file: {prompt_file}")
            
        for model in models:
            video_path = outputs_dir / model / f"{input_name}.mp4"
            if not video_path.exists():
                raise FileNotFoundError(f"Missing generated video: {video_path}")
                
    print("All inputs, conditioning frames, prompts, and output videos VERIFIED.")
    
    # 2. Load configs and initialize SigLIPEncoder
    with open(repo_root / "configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    vram_manager = VRAMManager(
        device_id=config.get("vram", {}).get("device_id", 0),
        limit_gb=config.get("vram", {}).get("limit_gb", 22.0)
    )
    
    siglip_model_name = config.get("models", {}).get("siglip", {}).get("model_name", "google/siglip2-so400m-patch16-naflex")
    print(f"Initializing SigLIPEncoder with model: {siglip_model_name}")
    siglip = SigLIPEncoder(vram_manager, siglip_model_name)
    
    # 3. Score loop
    scores = {}
    
    for model in models:
        scores[model] = {}
        for input_name in inputs:
            print(f"Scoring Model: {model} | Input: {input_name}...")
            
            # Load conditioning frame and prompt text
            cond_frame_path = inputs_dir / input_name / "frame.jpg"
            prompt_file_path = inputs_dir / input_name / "text.txt"
            video_path = outputs_dir / model / f"{input_name}.mp4"
            
            cond_img = Image.open(cond_frame_path).convert("RGB")
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
                
            # Extract video frames
            gen_frames = extract_all_frames(video_path)
            print(f"  Extracted {len(gen_frames)} frames from video.")
            
            # Compute SigLIP embeddings
            print("  Encoding conditioning frame and prompt text...")
            cond_embedding = encode_images_in_batch([cond_img], siglip)[0]
            text_embedding = siglip.encode(prompt_text)
            
            print("  Encoding video frames...")
            gen_embeddings = encode_images_in_batch(gen_frames, siglip, batch_size=16)
            
            # Compute similarity metrics
            print("  Calculating metrics...")
            text_sims = gen_embeddings @ text_embedding
            text_sim = float(np.mean(text_sims))
            
            style_sims = gen_embeddings @ cond_embedding
            style_sim = float(np.mean(style_sims))
            
            # Compute color distance
            color_dist = compute_color_dist(cond_img, gen_frames)
            
            scores[model][input_name] = {
                "text_sim": text_sim,
                "style_sim": style_sim,
                "color_dist": color_dist,
                "num_frames": len(gen_frames)
            }
            print(f"  Results -> text_sim: {text_sim:.4f} | style_sim: {style_sim:.4f} | color_dist: {color_dist:.4f}")
            
    # Cleanup VRAM after inference
    vram_manager.unload_current_model()
    
    # 4. Save results to scores.json
    scores_path = outputs_dir / "scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print(f"Saved scores to {scores_path}")
    
    # 5. Print Markdown Table and stats
    print("\n")
    print("| Model | Input | Text Sim | Style Sim | Color Dist |")
    print("| :--- | :--- | :---: | :---: | :---: |")
    for model in models:
        for input_name in inputs:
            s = scores[model][input_name]
            print(f"| {model} | {input_name} | {s['text_sim']:.4f} | {s['style_sim']:.4f} | {s['color_dist']:.4f} |")
            
    print("\n=== PER-MODEL MEANS ===")
    for model in models:
        text_sims = [scores[model][inp]["text_sim"] for inp in inputs]
        style_sims = [scores[model][inp]["style_sim"] for inp in inputs]
        color_dists = [scores[model][inp]["color_dist"] for inp in inputs]
        print(f"\nModel: {model}")
        print(f"  Mean Text Sim:  {np.mean(text_sims):.4f}")
        print(f"  Mean Style Sim: {np.mean(style_sims):.4f}")
        print(f"  Mean Color Dist: {np.mean(color_dists):.4f}")

if __name__ == "__main__":
    main()
