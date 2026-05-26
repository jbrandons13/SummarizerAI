# Phase 5 Smoke Test (Comparison Model) — CogVideoX-5B I2V

## Goal

Test a second image-to-video model on the same inputs already used for Wan 2.2 TI2V-5B. This gives a side-by-side comparison so we can choose the best model for Phase 5 with empirical evidence, not just trusting Wan.

Model: **CogVideoX-5B I2V** by THUDM. Comparable size (~10-12 GB), different architecture (DiT-based), proven on RTX 3090.

## Rules of engagement

- Be direct. Print `=== TASK N DONE ===` after each task.
- Reuse the existing `phase5_smoke_inputs/` from the Wan 2.2 test. Do NOT regenerate inputs.
- All downloads resume-able with `resume_download=True`.
- Smoke test script goes in repo root: `smoke_cogvideox.py`. Do not touch `src/`.
- Do not commit.

---

## Task 1: Verify inputs from previous Wan 2.2 test still exist

```bash
ls -la phase5_smoke_inputs/input_a_strong/
ls -la phase5_smoke_inputs/input_b_marginal/
ls -la phase5_smoke_inputs/input_c_generate/
```

Each folder must have `frame.jpg`, `text.txt`, `meta.json`. Report yes/no per folder.

If any folder is missing, STOP and report. The point of this test is to reuse exact same inputs.

---

## Task 2: Resume-able model download

CogVideoX-5B I2V hub id: `THUDM/CogVideoX-5b-I2V`. Verify the id on the Hub before downloading.

Script `download_cogvideox.py`:

```python
from huggingface_hub import snapshot_download
import os

MODEL_ID = "THUDM/CogVideoX-5b-I2V"
local_dir = os.path.expanduser("~/models/cogvideox_5b_i2v")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=local_dir,
    resume_download=True,
    max_workers=2,
    ignore_patterns=["*.md", "*.png", "*.gif", "*.jpg", "example*"],
)
print(f"Downloaded to {local_dir}")
```

Run it. After download:

```bash
du -sh ~/models/cogvideox_5b_i2v
```

### Definition of done

- Folder exists with model files.
- Size between 8-15 GB.

### If blocked

- Download exceeds 15 GB → STOP, report.
- Model id not findable → search for alternative variants and report.

---

## Task 3: Smoke test on the 3 inputs

Use the same input resolution Wan ended up at (832x480) for fair comparison, but verify CogVideoX supports it. CogVideoX-5B I2V's native resolution is **720x480**, so if 832x480 fails, fall back to 720x480.

Script `smoke_cogvideox.py`:

```python
import time, json, torch, os
from pathlib import Path
from PIL import Image
import imageio
from diffusers import CogVideoXImageToVideoPipeline

MODEL_PATH = os.path.expanduser("~/models/cogvideox_5b_i2v")

OUTDIR = Path("phase5_smoke_outputs/cogvideox_5b")
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"Loading CogVideoX-5B I2V from {MODEL_PATH}...")
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,  # match Wan test for fair comparison
)
pipe.enable_model_cpu_offload()
if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
    pipe.vae.enable_tiling()

inputs = [
    ("input_a_strong",    "phase5_smoke_inputs/input_a_strong"),
    ("input_b_marginal",  "phase5_smoke_inputs/input_b_marginal"),
    ("input_c_generate",  "phase5_smoke_inputs/input_c_generate"),
]

# CogVideoX-5B I2V native settings:
# - resolution 720x480
# - num_frames 49 (8 fps native, but Diffusers exposes fps via output saving)
# - num_inference_steps 50 recommended

WIDTH, HEIGHT = 720, 480
NUM_FRAMES = 49
STEPS = 50
FPS = 8  # CogVideoX native

results = {}
for name, path in inputs:
    image = Image.open(f"{path}/frame.jpg").convert("RGB").resize((WIDTH, HEIGHT))
    prompt = Path(f"{path}/text.txt").read_text().strip()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = pipe(
        image=image,
        prompt=prompt,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=STEPS,
        guidance_scale=6.0,
    )
    elapsed = time.time() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    frames = out.frames[0] if hasattr(out, "frames") else out.videos[0]
    outpath = OUTDIR / f"{name}.mp4"
    imageio.mimsave(outpath, frames, fps=FPS)

    results[name] = {
        "time_seconds": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "output_path": str(outpath),
        "fps": FPS,
        "num_frames": NUM_FRAMES,
        "duration_seconds": NUM_FRAMES / FPS,
    }
    print(f"{name}: {elapsed:.1f}s, peak VRAM {peak_vram_gb:.2f}GB, saved {outpath}")

(OUTDIR / "results.json").write_text(json.dumps(results, indent=2))
print("DONE")
```

### Steps

1. If `CogVideoXImageToVideoPipeline` import fails, search installed diffusers for the right class and adapt.
2. Run the script.
3. If OOM occurs, try `num_inference_steps=30` (lower steps = lower memory peak in some pipelines, but mostly affects time). If still OOM, report and stop that input.
4. Capture stdout and `results.json`.

### Definition of done

- 3 MP4 files (or fewer if some failed) in `phase5_smoke_outputs/cogvideox_5b/`.
- `results.json` exists.

---

## Task 4: Comparison report

```
=== PHASE 5 SMOKE TEST REPORT (CogVideoX-5B I2V) ===

Inputs reused from Wan 2.2 test: yes / no

Model:
  Hub id: THUDM/CogVideoX-5b-I2V (or actual id used)
  Local size: <GB>
  Pipeline class: <name>

Settings:
  Resolution: 720x480
  num_frames: 49
  fps: 8 (native)
  Output duration per clip: 6.125s
  num_inference_steps: 50
  dtype: bfloat16

Results:
  A: time=<s>s, peak VRAM=<GB>, output=<path>
  B: time=<s>s, peak VRAM=<GB>, output=<path>
  C: time=<s>s, peak VRAM=<GB>, output=<path>

Mean time per clip: <s>
Mean peak VRAM: <GB>

Scale projection:
  60 generations total -> <hours>

Side-by-side vs Wan 2.2 TI2V-5B (from previous report):
  | Metric             | Wan 2.2 5B      | CogVideoX-5B I2V |
  |--------------------|-----------------|------------------|
  | Resolution         | 832x480         | 720x480          |
  | Frames / fps / dur | 49 / 16fps / 3s | 49 / 8fps / 6s   |
  | Mean time / clip   | 90s             | <s>              |
  | Mean peak VRAM     | 13.3 GB         | <GB>             |
  | Total est. 60 clip | 1.5 hours       | <hours>          |

Output sample notes (objective only, do not judge visual quality):
  - Any warnings, NaN, artifact reports in console: <list, or "none">
  - Frame count actually produced: <n>

Blockers: <list, or "none">
```

## Hard constraints

- Do not download any other models.
- Do not modify `src/` or any existing pipeline code.
- Do not regenerate `phase5_smoke_inputs/`. Reuse them.
- Bandwidth budget: ~15 GB for CogVideoX download + minor pip if needed. STOP if over.
- Do not commit.

End of brief.
