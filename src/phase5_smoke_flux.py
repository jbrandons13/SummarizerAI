import time
import json
import torch
import os
from pathlib import Path
from diffusers import FluxPipeline
from PIL import Image

# Output directory setup
OUTDIR = Path("phase5_smoke_outputs/flux_ipadapter")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
CACHE_DIR = os.path.expanduser("~/models/flux_schnell")
IP_ADAPTER_ID = "XLabs-AI/flux-ip-adapter"
IMAGE_ENCODER_ID = "openai/clip-vit-large-patch14"
IP_SCALE = 0.7
SEED = 42

def prepare_frame_flux(path, target_w=832, target_h=480):
    """
    Center-crops the frame to 1.733 aspect ratio, then upscales/resizes to 832x480.
    Identical to the crop-and-resize logic verified in previous smoke tests.
    """
    img = Image.open(path).convert("RGB")
    target_ar = target_w / target_h  # 1.733
    src_w, src_h = img.size
    src_ar = src_w / src_h
    if src_ar < target_ar:
        # Source too tall — crop top/bottom
        new_h = int(src_w / target_ar)
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))
    elif src_ar > target_ar:
        # Source too wide — crop sides
        new_w = int(src_h * target_ar)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    # Resize to exact target
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return img

print(f"Loading base model: {MODEL_ID} (cache: {CACHE_DIR})")
try:
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
    )
    print("Base pipeline loaded successfully.")
except Exception as e:
    print(f"Failed to load base model: {e}")
    exit(1)

print(f"Loading IP-Adapter from {IP_ADAPTER_ID} with image encoder {IMAGE_ENCODER_ID}")
try:
    pipe.load_ip_adapter(
        IP_ADAPTER_ID,
        weight_name="ip_adapter.safetensors",
        image_encoder_pretrained_model_name_or_path=IMAGE_ENCODER_ID,
    )
    pipe.set_ip_adapter_scale(IP_SCALE)
    print("IP-Adapter loaded successfully.")
except Exception as e:
    print(f"Failed to load IP-Adapter: {e}")
    exit(1)

# Use sequential CPU offloading for robust execution within 24GB VRAM
print("Enabling sequential CPU offloading...")
pipe.enable_sequential_cpu_offload()

# 3 input configurations
inputs = [
    (
        "input_a_strong",
        "phase5_smoke_inputs/input_a_strong",
        "A Xiaomi smartphone with a custom gaming case attached, rear screen displaying game controller buttons that glow softly, slow camera pan around the device, tech product review style, soft studio lighting, clean dark background, shallow depth of field"
    ),
    (
        "input_b_marginal",
        "phase5_smoke_inputs/input_b_marginal",
        "Two modern smartphones lying side by side on a clean surface, camera slowly tilts down revealing their identical sleek metal frames and rounded edges, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field"
    ),
    (
        "input_c_generate",
        "phase5_smoke_inputs/input_c_generate",
        "Close-up of a smartphone's rear screen lighting up to display music playback controls with album art, fingers gently tap the screen, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus"
    ),
]

results = {}

for name, path_str, prompt in inputs:
    print(f"\nProcessing {name}...")
    path = Path(path_str)
    image_path = path / "frame.jpg"
    
    if not image_path.exists():
        print(f"Skipping {name}: original frame.jpg missing at {image_path}")
        continue
        
    # Preprocess conditioning frame
    preprocessed_img = prepare_frame_flux(image_path)
    preprocessed_path = path / "frame_flux_preprocessed.jpg"
    preprocessed_img.save(preprocessed_path, quality=95)
    print(f"Preprocessed conditioning frame saved to: {preprocessed_path}")

    # Set generator seed for strict reproducibility
    generator = torch.Generator(device="cpu").manual_seed(SEED)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    t0 = time.time()
    try:
        out = pipe(
            prompt=prompt,
            ip_adapter_image=preprocessed_img,
            num_inference_steps=4,      # default for FLUX.1-schnell
            guidance_scale=0.0,         # default for FLUX.1-schnell
            height=480,
            width=832,
            generator=generator,
        )
        elapsed = time.time() - t0
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

        out_img = out.images[0]
        outpath = OUTDIR / f"{name}.png"
        out_img.save(outpath)

        results[name] = {
            "time_seconds": elapsed,
            "peak_vram_gb": peak_vram_gb,
            "output_path": str(outpath),
            "preprocessed_path": str(preprocessed_path),
            "status": "success"
        }
        print(f"{name}: {elapsed:.1f}s, peak VRAM {peak_vram_gb:.2f}GB. Saved to {outpath}")
    except Exception as e:
        print(f"Error processing {name}: {e}")
        results[name] = {
            "status": "failure",
            "error": str(e)
        }

# Write consolidated results
results_path = OUTDIR / "results.json"
results_path.write_text(json.dumps(results, indent=2))
print(f"\nAll smoke test results saved to {results_path}")
