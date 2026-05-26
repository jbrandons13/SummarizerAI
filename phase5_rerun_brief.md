# Phase 5 Smoke Test Re-run Brief

## Context

The first smoke test produced severe warping, flickering, and melting artifacts in both Wan 2.2 5B and CogVideoX 5B outputs. The diagnostic report identified two root causes:

1. **Prompts were raw summary sentences** — descriptive product facts with no motion verbs, camera cues, or visual style guidance. I2V models need explicit motion direction.
2. **Aspect ratio mismatch on Wan** — 720x480 input frames (AR 1.5) were silently stretched to 832x480 (AR 1.733) by Diffusers, distorting the conditioning frame before generation even started.

This brief is a re-run of the smoke test with both issues fixed. Same 3 inputs, same models, same settings — only prompts and frame preprocessing change. Goal: isolate whether the artifacts were caused by inputs or by the models themselves.

## Scope

Re-run the smoke test for **both** Wan 2.2 5B and CogVideoX 5B on the same 3 inputs (`input_a_strong`, `input_b_marginal`, `input_c_generate`).

Do NOT change:
- Models, dtype, inference steps, guidance scale, num_frames, resolution, optimizations (CPU offload, VAE tiling).
- Conditioning frame source files.

DO change:
- Prompt text (see below).
- Frame preprocessing for Wan (see below).

## Change 1: Prompt engineering

For each of the 3 inputs, construct a new I2V-friendly prompt by combining three elements:

1. **Visual subject** — concrete nouns from the original sentence (e.g., "smartphone", "rear screen", "phone case").
2. **Motion / action cue** — a verb describing what the camera or subject does (e.g., "slowly rotating", "camera pans across", "screen lights up", "fingers tap the surface").
3. **Style / scene cue** — short descriptor matching the source video style (e.g., "tech product review style, soft studio lighting, clean background, shallow depth of field").

Use this template, but adapt to each input:

> `{visual subject with concrete details}, {motion or camera action}, {style and lighting cues}`

### Proposed prompts (use these verbatim)

- **input_a_strong** (original: "The Xiaomi phones offer a unique gaming experience with a custom case that turns the back screen into a controller."):
  > `A Xiaomi smartphone with a custom gaming case attached, rear screen displaying game controller buttons that glow softly, slow camera pan around the device, tech product review style, soft studio lighting, clean dark background, shallow depth of field`

- **input_b_marginal** (original: "Xiaomi's 17s Pro and Pro Max phones closely resemble the iPhone 17 Pro and Pro Max, including the design and user interface."):
  > `Two modern smartphones lying side by side on a clean surface, camera slowly tilts down revealing their identical sleek metal frames and rounded edges, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field`

- **input_c_generate** (original: "The rear screen can be used for various functions, including controlling music, displaying notifications, and serving as a camera viewfinder."):
  > `Close-up of a smartphone's rear screen lighting up to display music playback controls with album art, fingers gently tap the screen, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus`

These prompts apply to **both** Wan and CogVideoX runs (no model-specific formatting needed for this test).

## Change 2: Frame preprocessing fix for Wan

The 720x480 conditioning frames must be explicitly converted to 832x480 (Wan's expected resolution) before passing to the pipeline. Use **center crop after resize** to preserve aspect, not stretch.

Recommended approach:

```python
from PIL import Image

def prepare_frame_for_wan(path, target_w=832, target_h=480):
    img = Image.open(path).convert("RGB")
    # Resize so height matches target, preserving aspect
    src_w, src_h = img.size
    scale = target_h / src_h
    new_w = int(src_w * scale)
    img = img.resize((new_w, target_h), Image.LANCZOS)
    # Now img is (new_w, 480). If new_w < target_w, pad. If >, center crop.
    if new_w >= target_w:
        left = (new_w - target_w) // 2
        img = img.crop((left, 0, left + target_w, target_h))
    else:
        # Pad with black (rare case)
        from PIL import ImageOps
        pad = (target_w - new_w) // 2
        img = ImageOps.expand(img, border=(pad, 0, target_w - new_w - pad, 0), fill="black")
    return img
```

For 720x480 input specifically: rescaling height stays 480, width becomes 720 (no change since scale=1). Then center crop won't help because new_w (720) < target_w (832). So this falls into the **pad** branch, adding ~56px black on each side.

**However**, padding with black is also not ideal — it gives the model a weird letterboxed frame. Better alternative for this specific case: **center crop the original frame to a 1.733 aspect ratio first, then upscale to 832x480**. This trims a bit of the top/bottom but keeps the frame full and undistorted.

Use this version instead:

```python
def prepare_frame_for_wan(path, target_w=832, target_h=480):
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
    img = img.resize((target_w, target_h), Image.LANCZOS)
    return img
```

Save the preprocessed frame for each input to `phase5_smoke_inputs/{input_name}/frame_wan_preprocessed.jpg` so we can visually verify it before generation.

CogVideoX run: keep as-is (no aspect mismatch). But also save the preprocessed frame to `phase5_smoke_inputs/{input_name}/frame_cogvideox_preprocessed.jpg` for parity.

## Deliverables

1. The 6 newly generated videos, overwriting (or alongside, with `_v2` suffix) the previous outputs:
   - `phase5_smoke_outputs/wan22_5b/input_*_v2.mp4`
   - `phase5_smoke_outputs/cogvideox_5b/input_*_v2.mp4`
2. The 6 preprocessed frames (3 wan + 3 cogvideox) saved alongside originals.
3. A short markdown report containing:
   - Confirmation that new prompts and preprocessed frames were used.
   - Per-clip generation time (just to confirm no regression).
   - Filepaths of the 6 outputs.
   - No quality interpretation — user will visual-judge.

## Constraints

- Do not change inference settings, models, or any other config.
- Do not re-run the scoring script yet. We'll only score if visual judgment passes.
- If anything in this brief is ambiguous or conflicts with the existing codebase, stop and ask before improvising.

=== END OF BRIEF ===
