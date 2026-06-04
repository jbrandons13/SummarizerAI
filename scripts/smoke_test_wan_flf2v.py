import torch
import numpy as np
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
import time
import json
from pathlib import Path
import gc

# Reset GPU memory baseline
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
start_time = time.time()

# Config
VIDEO_ID = "lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge"
BASE_PATH = Path(f"data/intermediate/{VIDEO_ID}/phase4")
FIRST_FRAME_PATH = BASE_PATH / "semantic_triggered/images/shot_009.png"
LAST_FRAME_PATH = BASE_PATH / "semantic_triggered/images/shot_010.png"
OUTPUT_PATH = BASE_PATH / "_smoke_test/shot_010_FLF2V_test.mp4"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Get prompt from storyboard
with open(BASE_PATH / "storyboard.json") as f:
    storyboard = json.load(f)
shot_010 = next(s for s in storyboard["shots"] if s["shot_id"] == "shot_010")
PROMPT = shot_010["visual_description"]

# Model setup — LOCAL PATH
model_id = "./models/wan2.1-flf2v-14b"

print(f"[{time.time()-start_time:.1f}s] Loading image encoder...")
image_encoder = CLIPVisionModel.from_pretrained(
    model_id, subfolder="image_encoder", torch_dtype=torch.float32
)

print(f"[{time.time()-start_time:.1f}s] Loading VAE...")
vae = AutoencoderKLWan.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.float32
)

# Load frames
first_frame = load_image(str(FIRST_FRAME_PATH))
last_frame = load_image(str(LAST_FRAME_PATH))

# Attempt strategy logic
attempts = [
    ("default_cuda", lambda p: p.to("cuda")),
    ("model_cpu_offload", lambda p: p.enable_model_cpu_offload()),
    ("sequential_cpu_offload", lambda p: p.enable_sequential_cpu_offload()),
]

last_error = None
success = False
winning_strategy = None
output = None
gen_time = 0
vram_peak = 0

for strategy_name, apply_strategy in attempts:
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        print(f"\n=== ATTEMPT: {strategy_name} ===")
        
        # Re-load pipeline fresh (DELETE old pipe object first)
        if 'pipe' in globals():
            del globals()['pipe']
            gc.collect()
            torch.cuda.empty_cache()
            
        print(f"[{time.time()-start_time:.1f}s] Loading pipeline...")
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=torch.bfloat16
        )
        apply_strategy(pipe)
        
        vram_after_load = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM after load: {vram_after_load:.2f} GB")
        
        # Try generation
        print(f"[{time.time()-start_time:.1f}s] Starting generation with {strategy_name}...")
        gen_start = time.time()
        output = pipe(
            image=first_frame,
            last_image=last_frame,
            prompt=PROMPT,
            height=480, width=832,
            num_frames=81,
            guidance_scale=5.5,
            num_inference_steps=30,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]
        gen_time = time.time() - gen_start
        vram_peak = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"✅ SUCCESS with strategy '{strategy_name}'")
        print(f"Gen time: {gen_time:.1f}s, VRAM peak: {vram_peak:.2f}GB")
        success = True
        winning_strategy = strategy_name
        break
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ OOM with '{strategy_name}': {str(e)[:200]}")
        last_error = e
        continue
    except Exception as e:
        print(f"❌ Other error with '{strategy_name}': {type(e).__name__}: {str(e)[:200]}")
        last_error = e
        continue

if 'pipe' in globals():
    del globals()['pipe']
gc.collect()
torch.cuda.empty_cache()

if not success:
    print(f"\n=== ALL STRATEGIES FAILED ===")
    print(f"Last error: {last_error}")
    raise SystemExit(1)

# Export only if success
export_to_video(output, str(OUTPUT_PATH), fps=16)
print(f"[{time.time()-start_time:.1f}s] Saved to {OUTPUT_PATH}")

# Summary
print("\n=== SMOKE TEST SUMMARY ===")
print(f"Total wall time: {time.time() - start_time:.1f}s")
print(f"Generation time only: {gen_time:.1f}s")
print(f"VRAM peak: {vram_peak:.2f} GB")
print(f"Winning Strategy: {winning_strategy}")
print(f"Output: {OUTPUT_PATH}")
print(f"Output size: {OUTPUT_PATH.stat().st_size / 1e6:.2f} MB")
