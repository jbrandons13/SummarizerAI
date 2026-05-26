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

PROMPTS = {
    "input_a_original": "A Xiaomi smartphone with a custom gaming case attached, rear screen displaying game controller buttons that glow softly, slow camera pan around the device, tech product review style, soft studio lighting, clean dark background, shallow depth of field",
    "input_b_original": "Two modern smartphones lying side by side on a clean surface, camera slowly tilts down revealing their identical sleek metal frames and rounded edges, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field",
    "input_b_v1": "Two identical phones placed motionless side by side on a clean surface, camera completely static, no zoom no pan, both phones remain unchanged, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field",
    "input_b_v2": "Two identical phones lying side by side on a clean surface, focus on the leftmost phone, both phones remain unchanged throughout, no scene transitions, camera slowly tilts down, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field",
    "input_c_original": "Close-up of a smartphone's rear screen lighting up to display music playback controls with album art, fingers gently tap the screen, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus",
    "input_c_v1": "Small secondary display in the camera module area on the back of phone, showing music playback controls and album art, fingers gently tap the small display, rest of phone back is matte aluminum, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus",
    "input_c_v2": "Phone back with camera bump containing a tiny circular display showing album art and music controls, main body of phone back is plain dark metal, no buttons, no full touchscreen, fingers tap the tiny display, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus",
    "input_c_v3": "Xiaomi 13 Ultra style mini rear display near camera lens showing music album art and playback controls, fingers tap the mini display, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus",
    "input_c_v4": "Phone with mini display next to camera lens showing music playback, finger taps the mini display, smooth camera movement, warm lighting, dark background"
}

KEYFRAME_IMAGES = {
    "input_a": repo_root / "phase5_smoke_inputs/input_a_strong/frame_ltx_preprocessed.jpg",
    "input_b": repo_root / "phase5_smoke_inputs/input_b_marginal/frame_ltx_preprocessed.jpg",
    "input_c": repo_root / "phase5_smoke_inputs/input_c_generate/frame_ltx_preprocessed.jpg"
}

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

def get_sampled_frames(frames: list) -> list:
    """Sample 4 frames uniformly from a list of frames: t=0, 1/3, 2/3, end."""
    n = len(frames)
    if n == 0:
        raise ValueError("Video has 0 frames")
    idx0 = 0
    idx1 = int(round((n - 1) / 3))
    idx2 = int(round(2 * (n - 1) / 3))
    idx3 = n - 1
    print(f"Sampling frames at indices: [{idx0}, {idx1}, {idx2}, {idx3}] out of {n} frames")
    return [frames[idx0], frames[idx1], frames[idx2], frames[idx3]]

def encode_images(images: list, siglip: SigLIPEncoder) -> np.ndarray:
    """Encode PIL images and return L2-normalized numpy features."""
    siglip._load()
    processor = siglip.processor
    model = siglip.model
    
    embeddings = []
    for img in images:
        inputs = processor(images=img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = getattr(features, "pooler_output", features[0])
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

def main():
    print("=== STARTING SWEEP VERIFICATION AND SCORING ===")
    
    outputs_dir = repo_root / "phase5_smoke_outputs/ltx_prompt_refine"
    results_file = outputs_dir / "results.json"
    
    if not results_file.exists():
        print(f"Error: {results_file} not found.")
        sys.exit(1)
        
    with open(results_file, "r") as f:
        results = json.load(f)
        
    # Load config to initialize SigLIP
    with open(repo_root / "configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    vram_manager = VRAMManager(
        device_id=config.get("vram", {}).get("device_id", 0),
        limit_gb=config.get("vram", {}).get("limit_gb", 22.0)
    )
    
    siglip_model_name = config.get("models", {}).get("siglip", {}).get("model_name", "google/siglip2-so400m-patch16-naflex")
    print(f"Initializing SigLIPEncoder with model: {siglip_model_name}")
    siglip = SigLIPEncoder(vram_manager, siglip_model_name)
    
    updated_results = {}
    
    for run_key, run_info in results.items():
        print(f"\nProcessing run key: {run_key}...")
        input_id = run_info["input_id"]
        variant = run_info["variant"]
        num_frames = run_info["num_frames"]
        
        # Get video path
        video_path = Path(run_info["output_path_sub"])
        if not video_path.exists():
            video_path = Path(run_info["output_path_root"])
        if not video_path.exists():
            print(f"Error: Video file not found for {run_key} at {video_path}")
            continue
            
        # Get prompt
        prompt_key = f"{input_id}_{variant}"
        if prompt_key not in PROMPTS:
            print(f"Warning: Prompt key {prompt_key} not found, checking if variant is standard...")
            prompt_text = PROMPTS.get(variant, "")
        else:
            prompt_text = PROMPTS[prompt_key]
            
        if not prompt_text:
            print(f"Error: No prompt found for {run_key}")
            continue
            
        # Get keyframe (conditioning image)
        cond_image_path = KEYFRAME_IMAGES.get(input_id)
        if not cond_image_path or not cond_image_path.exists():
            print(f"Error: Conditioning image not found for {input_id} at {cond_image_path}")
            continue
            
        print(f"  Extracting frames from {video_path}...")
        all_frames = extract_all_frames(video_path)
        sampled_frames = get_sampled_frames(all_frames)
        
        print("  Encoding sampled frames...")
        sampled_embeddings = encode_images(sampled_frames, siglip) # shape: (4, D)
        
        print("  Encoding prompt...")
        prompt_embedding = siglip.encode(prompt_text) # shape: (D,)
        
        print("  Encoding keyframe...")
        cond_img = Image.open(cond_image_path).convert("RGB")
        cond_embedding = encode_images([cond_img], siglip)[0] # shape: (D,)
        
        # Compute similarities
        prompt_sims = sampled_embeddings @ prompt_embedding
        prompt_score = float(np.mean(prompt_sims))
        
        keyframe_sims = sampled_embeddings @ cond_embedding
        keyframe_score = float(np.mean(keyframe_sims))
        
        print(f"  Result -> prompt_score: {prompt_score:.4f} | keyframe_score: {keyframe_score:.4f}")
        
        # Update run info
        run_info["prompt_score"] = prompt_score
        run_info["keyframe_score"] = keyframe_score
        updated_results[run_key] = run_info
        
    # Unload model
    vram_manager.unload_current_model()
    
    # Save back to results.json (overwrite/update)
    with open(results_file, "w") as f:
        json.dump(updated_results, f, indent=2)
    print(f"\nSuccessfully updated results.json with verified scores at {results_file}")

if __name__ == "__main__":
    main()
