import sys
import os
import gc
import time
import json
import torch
import shutil
import numpy as np
from PIL import Image
from pathlib import Path

# 1. Monkeypatch diffusers to fix the LTX retrieve_timesteps bug in both pipelines
import diffusers.pipelines.ltx.pipeline_ltx as pl_t2v
import diffusers.pipelines.ltx.pipeline_ltx_image2video as pl_i2v

orig_t2v_retrieve = pl_t2v.retrieve_timesteps
orig_i2v_retrieve = pl_i2v.retrieve_timesteps

def patched_t2v(scheduler, num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kwargs):
    if timesteps is not None:
        sigmas = None
    return orig_t2v_retrieve(scheduler, num_inference_steps, device, timesteps, sigmas, **kwargs)

def patched_i2v(scheduler, num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kwargs):
    if timesteps is not None:
        sigmas = None
    return orig_i2v_retrieve(scheduler, num_inference_steps, device, timesteps, sigmas, **kwargs)

pl_t2v.retrieve_timesteps = patched_t2v
pl_i2v.retrieve_timesteps = patched_i2v

from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video

# Define directories
root_output_dir = "/home/wins053/smoke_tests/ltx_prompt_refine"
workspace_output_dir = "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/phase5_smoke_outputs/ltx_prompt_refine"
model_path = "/home/wins053/models/ltx_video_distilled"

os.makedirs(root_output_dir, exist_ok=True)
os.makedirs(workspace_output_dir, exist_ok=True)

runs = [
    # Input C (rear screen issue)
    {
        "input_id": "input_c",
        "variant": "v1",
        "conditioning_image": "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/phase5_smoke_inputs/input_c_generate/frame_ltx_preprocessed.jpg",
        "prompt": "Small secondary display in the camera module area on the back of phone, showing music playback controls and album art, fingers gently tap the small display, rest of phone back is matte aluminum, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus",
        "num_frames": 121
    },
    {
        "input_id": "input_c",
        "variant": "v1",
        "conditioning_image": "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/phase5_smoke_inputs/input_c_generate/frame_ltx_preprocessed.jpg",
        "prompt": "Small secondary display in the camera module area on the back of phone, showing music playback controls and album art, fingers gently tap the small display, rest of phone back is matte aluminum, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus",
        "num_frames": 241
    },
    {
        "input_id": "input_c",
        "variant": "v2",
        "conditioning_image": "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/phase5_smoke_inputs/input_c_generate/frame_ltx_preprocessed.jpg",
        "prompt": "Phone back with camera bump containing a tiny circular display showing album art and music controls, main body of phone back is plain dark metal, no buttons, no full touchscreen, fingers tap the tiny display, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus",
        "num_frames": 121
    },
    {
        "input_id": "input_c",
        "variant": "v2",
        "conditioning_image": "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/phase5_smoke_inputs/input_c_generate/frame_ltx_preprocessed.jpg",
        "prompt": "Phone back with camera bump containing a tiny circular display showing album art and music controls, main body of phone back is plain dark metal, no buttons, no full touchscreen, fingers tap the tiny display, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus",
        "num_frames": 241
    },
    {
        "input_id": "input_c",
        "variant": "v3",
        "conditioning_image": "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/phase5_smoke_inputs/input_c_generate/frame_ltx_preprocessed.jpg",
        "prompt": "Xiaomi 13 Ultra style mini rear display near camera lens showing music album art and playback controls, fingers tap the mini display, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus",
        "num_frames": 121
    },
    {
        "input_id": "input_c",
        "variant": "v3",
        "conditioning_image": "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/phase5_smoke_inputs/input_c_generate/frame_ltx_preprocessed.jpg",
        "prompt": "Xiaomi 13 Ultra style mini rear display near camera lens showing music album art and playback controls, fingers tap the mini display, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus",
        "num_frames": 241
    },
    {
        "input_id": "input_c",
        "variant": "v4",
        "conditioning_image": "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/phase5_smoke_inputs/input_c_generate/frame_ltx_preprocessed.jpg",
        "prompt": "Phone with mini display next to camera lens showing music playback, finger taps the mini display, smooth camera movement, warm lighting, dark background",
        "num_frames": 121
    },
    {
        "input_id": "input_c",
        "variant": "v4",
        "conditioning_image": "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/phase5_smoke_inputs/input_c_generate/frame_ltx_preprocessed.jpg",
        "prompt": "Phone with mini display next to camera lens showing music playback, finger taps the mini display, smooth camera movement, warm lighting, dark background",
        "num_frames": 241
    },
    # Input B (scene drift issue)
    {
        "input_id": "input_b",
        "variant": "v1",
        "conditioning_image": "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/phase5_smoke_inputs/input_b_marginal/frame_ltx_preprocessed.jpg",
        "prompt": "Two identical phones placed motionless side by side on a clean surface, camera completely static, no zoom no pan, both phones remain unchanged, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field",
        "num_frames": 121
    },
    {
        "input_id": "input_b",
        "variant": "v1",
        "conditioning_image": "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/phase5_smoke_inputs/input_b_marginal/frame_ltx_preprocessed.jpg",
        "prompt": "Two identical phones placed motionless side by side on a clean surface, camera completely static, no zoom no pan, both phones remain unchanged, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field",
        "num_frames": 241
    },
    {
        "input_id": "input_b",
        "variant": "v2",
        "conditioning_image": "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/phase5_smoke_inputs/input_b_marginal/frame_ltx_preprocessed.jpg",
        "prompt": "Two identical phones lying side by side on a clean surface, focus on the leftmost phone, both phones remain unchanged throughout, no scene transitions, camera slowly tilts down, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field",
        "num_frames": 121
    },
    {
        "input_id": "input_b",
        "variant": "v2",
        "conditioning_image": "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/phase5_smoke_inputs/input_b_marginal/frame_ltx_preprocessed.jpg",
        "prompt": "Two identical phones lying side by side on a clean surface, focus on the leftmost phone, both phones remain unchanged throughout, no scene transitions, camera slowly tilts down, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field",
        "num_frames": 241
    }
]

results_file = os.path.join(workspace_output_dir, "results.json")
results = {}

if os.path.exists(results_file):
    try:
        with open(results_file, "r") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {results_file}")
    except Exception as e:
        print("Failed to load existing results:", e)

custom_timesteps = [1000, 993, 987, 981, 975, 909, 725, 0.03]

def run_single(run, force_sequential=False):
    input_id = run["input_id"]
    variant = run["variant"]
    num_frames = run["num_frames"]
    prompt = run["prompt"]
    cond_image_path = run["conditioning_image"]
    
    run_key = f"{input_id}_{variant}_{num_frames}f"
    offload_style = "Sequential Offload" if (num_frames == 241 or force_sequential) else "Model CPU Offload"
    
    print(f"\nExecuting {run_key} using {offload_style}...")
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    pipe = LTXImageToVideoPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )
    pipe.vae.enable_tiling()
    
    if offload_style == "Model CPU Offload":
        pipe.enable_model_cpu_offload()
    else:
        pipe.enable_sequential_cpu_offload()
        
    image = Image.open(cond_image_path).convert("RGB").resize((768, 512))
    generator = torch.Generator("cuda").manual_seed(42)
    
    start_time = time.time()
    output = pipe(
        prompt=prompt,
        image=image,
        num_frames=num_frames,
        width=768,
        height=512,
        guidance_scale=1.0,
        timesteps=custom_timesteps,
        generator=generator
    )
    latency = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    frames = output.frames[0] if isinstance(output.frames[0], list) else output.frames
    
    temp_out_path = os.path.join(root_output_dir, f"{run_key}.mp4")
    workspace_out_path = os.path.join(workspace_output_dir, f"{run_key}.mp4")
    
    export_to_video(frames, temp_out_path, fps=30)
    shutil.copy2(temp_out_path, workspace_out_path)
    
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "input_id": input_id,
        "variant": variant,
        "num_frames": num_frames,
        "duration_s": num_frames / 30.0,
        "peak_vram_gb": peak_vram,
        "latency_s": latency,
        "oom": False,
        "output_path_root": temp_out_path,
        "output_path_sub": workspace_out_path,
        "status": "success",
        "offload_style": offload_style,
        "fallback_to_sequential": force_sequential
    }

for run_idx, run in enumerate(runs, 1):
    input_id = run["input_id"]
    variant = run["variant"]
    num_frames = run["num_frames"]
    run_key = f"{input_id}_{variant}_{num_frames}f"
    
    # Check if run exists and is successful
    if run_key in results and results[run_key].get("status") == "success":
        # Check if output files actually exist
        root_path = os.path.join(root_output_dir, f"{run_key}.mp4")
        workspace_path = os.path.join(workspace_output_dir, f"{run_key}.mp4")
        if os.path.exists(root_path) and os.path.exists(workspace_path):
            print(f"[{run_idx}/{len(runs)}] Skipping already completed run: {run_key}")
            continue

    print(f"\n==================================================")
    print(f"Running [{run_idx}/{len(runs)}]: {run_key}")
    
    try:
        # Attempt with standard config (Model CPU Offload for 121f)
        res = run_single(run, force_sequential=False)
        results[run_key] = res
    except Exception as e:
        err_msg = str(e)
        is_oom = "OutOfMemoryError" in err_msg or "OOM" in err_msg or "out of memory" in err_msg.lower()
        
        # If it was Model CPU Offload and it OOMed, attempt fallback
        if is_oom and num_frames == 121:
            print(f"WARNING: OOM detected for {run_key} with Model CPU Offload. Retrying with Sequential Offload...")
            try:
                res = run_single(run, force_sequential=True)
                results[run_key] = res
            except Exception as retry_e:
                print(f"FAILED RETRY FOR {run_key}: {retry_e}")
                results[run_key] = {
                    "input_id": input_id,
                    "variant": variant,
                    "num_frames": num_frames,
                    "duration_s": num_frames / 30.0,
                    "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
                    "latency_s": 0.0,
                    "oom": True,
                    "status": "failed",
                    "error": str(retry_e),
                    "offload_style": "Sequential Offload",
                    "fallback_to_sequential": True
                }
        else:
            print(f"FAILED RUN FOR {run_key}: {e}")
            results[run_key] = {
                "input_id": input_id,
                "variant": variant,
                "num_frames": num_frames,
                "duration_s": num_frames / 30.0,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
                "latency_s": 0.0,
                "oom": is_oom,
                "status": "failed",
                "error": err_msg,
                "offload_style": "Sequential Offload" if num_frames == 241 else "Model CPU Offload",
                "fallback_to_sequential": False
            }
            
    # Save results state after each run
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

print("\nSweep execution completed! Results saved to", results_file)
