import time
import json
import torch
import os
import sys
import traceback
from pathlib import Path
from diffusers import HunyuanVideo15ImageToVideoPipeline
from PIL import Image
import imageio
import numpy as np

# Resolution and frame count config
WIDTH, HEIGHT = 720, 480
STEPS = 20
FPS = 16
SEED = 42

MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_distilled"
OUTDIR = Path("/home/wins053/smoke_tests/num_frames_sweep")
OUTDIR.mkdir(parents=True, exist_ok=True)

def prepare_frame_hunyuan(path, target_w=720, target_h=480):
    img = Image.open(path).convert("RGB")
    target_ar = target_w / target_h
    src_w, src_h = img.size
    src_ar = src_w / src_h
    if src_ar < target_ar:
        new_h = int(src_w / target_ar)
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))
    elif src_ar > target_ar:
        new_w = int(src_h * target_ar)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return img

print(f"Loading model: {MODEL_ID}")
try:
    pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    print("Enabling model CPU offloading...")
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        print("Enabling VAE tiling...")
        pipe.vae.enable_tiling()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    traceback.print_exc()
    exit(1)

# Input parameters
name = "input_a_strong"
image_path = Path("/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/phase5_smoke_inputs/input_a_strong/frame.jpg")
prompt = "A Xiaomi smartphone with a custom gaming case attached, rear screen displaying game controller buttons that glow softly, slow camera pan around the device, tech product review style, soft studio lighting, clean dark background, shallow depth of field"

if not image_path.exists():
    print(f"Error: conditioning frame missing at {image_path}")
    exit(1)

image = prepare_frame_hunyuan(image_path, target_w=WIDTH, target_h=HEIGHT)

sweep_runs = [61, 81, 121]
results = {}

for num_frames in sweep_runs:
    print(f"\n==========================================")
    print(f"Running sweep with num_frames = {num_frames}...")
    print(f"==========================================")
    
    # Reset peak VRAM tracking
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    
    t0 = time.time()
    oom_occurred = False
    error_msg = ""
    peak_vram_gb = 0.0
    elapsed = 0.0
    
    try:
        out = pipe(
            image=image,
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=STEPS,
            generator=generator,
        )
        elapsed = time.time() - t0
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
        
        video = out.frames[0]
        if isinstance(video[0], Image.Image):
            video_np = [np.array(img) for img in video]
        else:
            video_np = video
            
        outpath = OUTDIR / f"run_{num_frames}.mp4"
        imageio.mimsave(outpath, video_np, fps=FPS)
        print(f"Success! Saved video to {outpath}")
        
        results[num_frames] = {
            "num_frames": num_frames,
            "duration_s": num_frames / FPS,
            "peak_vram_gb": peak_vram_gb,
            "latency_s": elapsed,
            "latency_per_frame_ms": (elapsed / num_frames) * 1000,
            "oom": False,
            "output_path": str(outpath),
            "status": "success"
        }
        
    except torch.cuda.OutOfMemoryError as oom_err:
        print(f"\n!!! OUT OF MEMORY !!! occurred during num_frames = {num_frames}")
        traceback.print_exc()
        oom_occurred = True
        error_msg = str(oom_err)
    except Exception as e:
        print(f"\nAn error occurred during num_frames = {num_frames}: {e}")
        traceback.print_exc()
        if "out of memory" in str(e).lower() or "oom" in str(e).lower() or "CUDA out of memory" in str(e):
            oom_occurred = True
        error_msg = str(e)
        
    if oom_occurred or error_msg:
        results[num_frames] = {
            "num_frames": num_frames,
            "duration_s": num_frames / FPS,
            "peak_vram_gb": peak_vram_gb or (torch.cuda.max_memory_allocated() / 1e9),
            "latency_s": time.time() - t0,
            "latency_per_frame_ms": -1.0,
            "oom": oom_occurred,
            "output_path": "",
            "status": "oom" if oom_occurred else "failure",
            "error": error_msg
        }
        
        # Stop sweep immediately as per the rules
        print(f"Stopping the sweep because of OOM or failure at num_frames = {num_frames}")
        break

# Save results json in output directory
manifest_path = OUTDIR / "results.json"
manifest_path.write_text(json.dumps(results, indent=2))
print(f"\nAll sweep results saved to {manifest_path}")
print("=== SWEEP DONE ===")
