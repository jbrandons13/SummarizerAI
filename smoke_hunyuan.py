import time, json, torch
from pathlib import Path
from diffusers import HunyuanVideo15ImageToVideoPipeline
from PIL import Image
import imageio
import numpy as np

OUTDIR = Path("phase5_smoke_outputs/hunyuan")
OUTDIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_distilled"

print(f"Loading model: {MODEL_ID}")
try:
    pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

inputs = [
    ("input_a_high", "phase5_smoke_inputs/input_a_high"),
    ("input_b_mid",  "phase5_smoke_inputs/input_b_mid"),
    ("input_c_low",  "phase5_smoke_inputs/input_c_low"),
]

results = {}
for name, path in inputs:
    print(f"\nProcessing {name}...")
    image_path = Path(f"{path}/frame.jpg")
    text_path = Path(f"{path}/text.txt")
    
    if not image_path.exists() or not text_path.exists():
        print(f"Skipping {name}: input files missing.")
        continue
        
    image = Image.open(image_path).convert("RGB")
    image = image.resize((720, 480))
    prompt = text_path.read_text().strip()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    t0 = time.time()
    try:
        # HunyuanVideo parameters
        out = pipe(
            image=image,
            prompt=prompt,
            num_frames=41,        # Still keep it shorter for smoke test
            num_inference_steps=20,
        )
        elapsed = time.time() - t0
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

        video = out.frames[0]
        
        if isinstance(video[0], Image.Image):
            video_np = [np.array(img) for img in video]
        else:
            video_np = video

        outpath = OUTDIR / f"{name}.mp4"
        imageio.mimsave(outpath, video_np, fps=16)

        results[name] = {
            "time_seconds": elapsed,
            "peak_vram_gb": peak_vram_gb,
            "output_path": str(outpath),
            "status": "success"
        }
        print(f"{name}: {elapsed:.1f}s, peak VRAM {peak_vram_gb:.2f}GB")
    except Exception as e:
        print(f"Error processing {name}: {e}")
        results[name] = {
            "status": "failure",
            "error": str(e)
        }

(OUTDIR / "results.json").write_text(json.dumps(results, indent=2))
print(f"\nResults saved to {OUTDIR / 'results.json'}")
