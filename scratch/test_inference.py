import time
import torch
import os
from diffusers import FluxPipeline
from PIL import Image
import numpy as np

def prepare_frame(path, target_w=832, target_h=480):
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
    # Now aspect matches; resize to exact target
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return img

print("1. Loading pipeline...")
model_id = "black-forest-labs/FLUX.1-schnell"
cache_dir = os.path.expanduser("~/models/flux_schnell")

t0 = time.time()
pipe = FluxPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir,
)
print(f"Pipeline loaded in {time.time() - t0:.2f}s")

print("2. Loading IP-Adapter...")
pipe.load_ip_adapter(
    "XLabs-AI/flux-ip-adapter",
    weight_name="ip_adapter.safetensors",
    image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14",
)
pipe.set_ip_adapter_scale(0.7)
print("IP-Adapter loaded.")

print("3. Enabling sequential CPU offload...")
pipe.enable_sequential_cpu_offload()
print("Sequential CPU offload enabled.")

print("4. Preprocessing conditioning frame...")
input_img_path = "phase5_smoke_inputs/input_a_strong/frame.jpg"
preprocessed_img = prepare_frame(input_img_path)
preprocessed_img.save("scratch/test_preprocessed.jpg")
print("Preprocessed frame saved to scratch/test_preprocessed.jpg")

print("5. Generating image...")
prompt = "A Xiaomi smartphone with a custom gaming case attached, rear screen displaying game controller buttons that glow softly, slow camera pan around the device, tech product review style, soft studio lighting, clean dark background, shallow depth of field"

# Set seed for reproducibility
generator = torch.Generator(device="cpu").manual_seed(42)

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
t0 = time.time()

try:
    # Let's run inference
    out = pipe(
        prompt=prompt,
        ip_adapter_image=preprocessed_img,
        num_inference_steps=4,
        guidance_scale=0.0,
        height=480,
        width=832,
        generator=generator,
    )
    elapsed = time.time() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"Generation successful in {elapsed:.2f}s! Peak VRAM: {peak_vram_gb:.2f} GB")
    
    out_img = out.images[0]
    out_img.save("scratch/test_inference_output.png")
    print("Output saved to scratch/test_inference_output.png")

except Exception as e:
    print(f"Error during inference: {e}")
    import traceback
    traceback.print_exc()

print("Script complete.")
