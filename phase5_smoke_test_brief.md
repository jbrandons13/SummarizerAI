# Phase 5 Smoke Test — Wan 2.2 TI2V-5B Feasibility (Bandwidth-Constrained)

## Goal

Test whether Wan 2.2 TI2V-5B (the lighter 5B variant of Wan 2.2, not the 14B) can run on RTX 3090 (22GB VRAM) for image-to-video generation. This is a feasibility check before designing Phase 5 of the pipeline.

The previous version of this brief tested 2 models (Wan 2.2 14B + HunyuanVideo 1.5), but bandwidth on the test server is limited to ~20 GB per session. Wan 2.2 TI2V-5B total size is ~10-12 GB, fitting within budget.

## Rules of engagement

- Be direct.
- Print `=== TASK N DONE ===` after each task.
- All downloads must be resume-able. If interrupted, the script must be re-runnable without re-downloading.
- Do not commit anything.
- Smoke test scripts go in repo root (e.g. `smoke_wan22_5b.py`), not in `src/`.

---

## Task 1: Environment check

Before installing anything, check the existing environment.

### Steps

1. GPU check:
   ```bash
   nvidia-smi --query-gpu=name,memory.total --format=csv
   ```

2. Python version (need 3.10+):
   ```bash
   python --version
   ```

3. Installed packages:
   ```bash
   pip list 2>/dev/null | grep -iE "torch|diffusers|transformers|accelerate|bitsandbytes|optimum|xformers|imageio"
   ```

4. CUDA / PyTorch:
   ```bash
   python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('available:', torch.cuda.is_available())"
   ```

5. Disk space and HF cache location:
   ```bash
   df -h ~/.cache/huggingface 2>/dev/null || df -h ~
   du -sh ~/.cache/huggingface 2>/dev/null || echo "no HF cache yet"
   ```

### Report

Report all outputs. Do not install anything yet.

---

## Task 2: Install only what is missing

Required minimum stack:

- `torch>=2.4` with CUDA support (do NOT reinstall if already working)
- `diffusers>=0.32`
- `transformers>=4.46`
- `accelerate>=1.0`
- `imageio`, `imageio-ffmpeg`, `Pillow` (for output saving)
- `huggingface_hub>=0.25` (for resume-able download)

### Steps

1. Only install packages that are missing or out-of-date in Task 1 output.
2. Do NOT install `optimum-quanto`, `bitsandbytes`, or `xformers` unless Wan 2.2 5B explicitly requires them — keep the install minimal.
3. Total install bandwidth budget: ~3-5 GB. If installing pulls more than that, report and STOP.

### Definition of done

```bash
python -c "import torch, diffusers, transformers, accelerate, imageio, huggingface_hub; print('all imports OK')"
```

### If blocked

- Conflict with existing torch / CUDA version → report and STOP. Do not force-resolve.

---

## Task 3: Resume-able model download

Wan 2.2 TI2V-5B is the lighter variant. Approximate size: 10-12 GB total.

### Steps

1. Search HuggingFace Hub for the exact id of Wan 2.2 TI2V-5B. Likely candidates:
   - `Wan-AI/Wan2.2-TI2V-5B`
   - `Wan-AI/Wan2.2-TI2V-5B-Diffusers`

   Verify the id by visiting the Hub page in the browser (you can `curl` the HF API). Report the exact id used.

2. Download with `huggingface_hub.snapshot_download` and resume-able flag. Use this script `download_wan5b.py`:

```python
from huggingface_hub import snapshot_download
import os

MODEL_ID = "<the resolved Wan 2.2 TI2V-5B id>"
local_dir = os.path.expanduser("~/models/wan22_ti2v_5b")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=local_dir,
    resume_download=True,
    max_workers=2,           # gentle on the connection
    ignore_patterns=["*.md", "*.png", "*.gif", "*.jpg", "example*"],  # skip non-essential
)
print(f"Downloaded to {local_dir}")
```

3. Run the script. If interrupted, re-run; it must continue from where it stopped.

4. After download, check the local size:
   ```bash
   du -sh ~/models/wan22_ti2v_5b
   ```

### Definition of done

- Folder `~/models/wan22_ti2v_5b` exists.
- Size between 8-15 GB (5B model + encoders + VAE).
- All expected files present: `model_index.json`, `transformer/`, `vae/`, `text_encoder/`, `tokenizer/`, `scheduler/`.

### If blocked

- Download exceeds 15 GB → bandwidth budget broken. STOP, report, do not proceed to Task 4.
- Model id not findable → report what variants exist on Hub for "Wan2.2" and STOP.

---

## Task 4: Prepare 3 smoke test inputs from Phase 4 output

Identify 3 assignments from existing Phase 4 output across the 10 evaluation videos.

### Steps

1. Pick 3 assignments spanning similarity buckets:
   - **Input A:** retrieve action, `weighted_similarity >= 0.15` (strong)
   - **Input B:** retrieve action, `weighted_similarity in [0.12, 0.14]` (marginal)
   - **Input C:** generate action, `weighted_similarity < 0.10` (genuine generation use case)

2. For each, extract the middle frame of the locked scene (use `FrameSelector` "middle" strategy from `src/phase4_retrieve.py`). Save as JPG resized to 720x480.

3. Save the joined sentence text per input.

4. Output layout:
   ```
   phase5_smoke_inputs/
     input_a_strong/
       frame.jpg
       text.txt
       meta.json     # {video_id, group_id, sentence_ids, weighted_sim, action}
     input_b_marginal/
       ... (same)
     input_c_generate/
       ... (same)
   ```

### Definition of done

- 3 input folders exist with `frame.jpg`, `text.txt`, `meta.json`.
- Frames are exactly 720x480 px.

---

## Task 5: Smoke test Wan 2.2 TI2V-5B

### Goal

Load the model, generate one short clip per input, measure VRAM and time.

### Script: `smoke_wan22_5b.py`

```python
import time, json, torch, os
from pathlib import Path
from PIL import Image
import imageio

# Resolve correct pipeline class from installed diffusers.
# Likely candidates (try in order):
#   from diffusers import WanImageToVideoPipeline
#   from diffusers import WanPipeline
#   from diffusers import Wan22Pipeline
# If none of these exist, search:
#   import diffusers; print([x for x in dir(diffusers) if "Wan" in x or "TI2V" in x])
# Report which class was found.

from diffusers import WanImageToVideoPipeline  # adapt if needed

MODEL_PATH = os.path.expanduser("~/models/wan22_ti2v_5b")

OUTDIR = Path("phase5_smoke_outputs/wan22_5b")
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"Loading Wan 2.2 TI2V-5B from {MODEL_PATH}...")
pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()
if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
    pipe.vae.enable_tiling()

inputs = [
    ("input_a_strong",    "phase5_smoke_inputs/input_a_strong"),
    ("input_b_marginal",  "phase5_smoke_inputs/input_b_marginal"),
    ("input_c_generate",  "phase5_smoke_inputs/input_c_generate"),
]

results = {}
for name, path in inputs:
    image = Image.open(f"{path}/frame.jpg").convert("RGB")
    prompt = Path(f"{path}/text.txt").read_text().strip()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = pipe(
        image=image,
        prompt=prompt,
        height=480,
        width=720,
        num_frames=49,           # ~3 seconds at 16fps; conservative for 5B
        num_inference_steps=30,
        guidance_scale=5.0,
    )
    elapsed = time.time() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    # Save output. Adapt attribute name to whatever Wan returns.
    frames = out.frames[0] if hasattr(out, "frames") else out.videos[0]
    outpath = OUTDIR / f"{name}.mp4"
    imageio.mimsave(outpath, frames, fps=16)

    results[name] = {
        "time_seconds": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "output_path": str(outpath),
    }
    print(f"{name}: {elapsed:.1f}s, peak VRAM {peak_vram_gb:.2f}GB, saved {outpath}")

(OUTDIR / "results.json").write_text(json.dumps(results, indent=2))
print("DONE")
```

### Steps

1. Verify the correct pipeline class name before running. If `WanImageToVideoPipeline` does not exist, search and adapt as commented in the script.

2. Run the script.

3. If OOM occurs:
   - Try `num_frames=33` (about 2 seconds) and re-run that input.
   - If still OOM, mark that input as failed and continue with the next.

4. After all 3 runs, capture stdout and `results.json`.

### Definition of done

- 3 MP4 files (or fewer if some failed) in `phase5_smoke_outputs/wan22_5b/`.
- `results.json` exists.

---

## Task 6: Report

```
=== PHASE 5 SMOKE TEST REPORT (Wan 2.2 TI2V-5B) ===

Environment:
  GPU: <model, total VRAM>
  Python: <version>
  CUDA: <version>
  Free disk (HF cache): <GB>
  Key packages: torch=<v>, diffusers=<v>, transformers=<v>, accelerate=<v>

Model:
  Hub id: <resolved id>
  Pipeline class: <name>
  Local size: <GB>

Inputs:
  A (strong):   video=<id>, group=<id>, weighted_sim=<x>, prompt="<text>"
  B (marginal): video=<id>, group=<id>, weighted_sim=<x>, prompt="<text>"
  C (generate): video=<id>, group=<id>, weighted_sim=<x>, prompt="<text>"

Results:
  A: time=<s>s, peak VRAM=<GB>, num_frames=<n>, output=<path>
  B: time=<s>s, peak VRAM=<GB>, num_frames=<n>, output=<path>
  C: time=<s>s, peak VRAM=<GB>, num_frames=<n>, output=<path>

Mean time per clip: <s>
Mean peak VRAM: <GB>

Scale projection:
  Assuming ~6 generate-action groups per video, 10 videos → 60 generations.
  Total time estimate: <mean_time_seconds> * 60 = <hours>

Console output excerpts (any artifact warnings, NaN warnings, etc.): <list, or "none">

Blockers: <list, or "none">
```

## Hard constraints

- Do not modify code in `src/`.
- Do not download Wan 2.2 14B or HunyuanVideo. Only Wan 2.2 TI2V-5B.
- Total bandwidth budget: ~20 GB session. If anything exceeds budget, STOP.
- All downloads must use `resume_download=True`.

End of brief.
