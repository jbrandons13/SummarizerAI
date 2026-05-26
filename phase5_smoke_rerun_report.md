# Phase 5 Smoke Test Re-run Report

## Context & Objectives

The first smoke test run for the video summarization pipeline's image-to-video (I2V) stage yielded visual artifacts (melting, warping, and flickering) in both the **Wan 2.2 TI2V-5B** and **CogVideoX-5B I2V** outputs. Two primary root causes were identified:
1. **Lack of motion guidance**: The initial prompts were raw product summary facts with no motion verbs or camera movement cues.
2. **Aspect ratio mismatch on Wan**: The 720x480 conditioning frames were stretched silently to 832x480 by the diffusers pipeline, distorting the conditioning prior.

This re-run executed both smoke tests on the same 3 inputs (`input_a_strong`, `input_b_marginal`, `input_c_generate`) with the identical hardware, models, dtypes, and inference settings, but implemented:
- **Change 1 (Prompt Engineering)**: Direct, high-fidelity engineered prompts featuring explicit camera/subject motion cues and clean visual aesthetics.
- **Change 2 (Frame Preprocessing)**: Crop-and-resize of conditioning frames for Wan to perfectly match the target 832x480 aspect ratio (1.733) before pipeline entry, and standard 720x480 resizing for CogVideoX-5B for parity.

---

## 1. Verifications & Confirmations

- [x] **New Prompts Used**: Constructed verbatim per the rerun brief instructions:
  - **input_a_strong**:
    > *A Xiaomi smartphone with a custom gaming case attached, rear screen displaying game controller buttons that glow softly, slow camera pan around the device, tech product review style, soft studio lighting, clean dark background, shallow depth of field*
  - **input_b_marginal**:
    > *Two modern smartphones lying side by side on a clean surface, camera slowly tilts down revealing their identical sleek metal frames and rounded edges, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field*
  - **input_c_generate**:
    > *Close-up of a smartphone's rear screen lighting up to display music playback controls with album art, fingers gently tap the screen, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus*
- [x] **Preprocessed Frames Created & Saved**:
  - Saved Wan preprocessed frames (832x480 cropped & resized) to: `video-summarizer/phase5_smoke_inputs/{input_name}/frame_wan_preprocessed.jpg`
  - Saved CogVideoX preprocessed frames (720x480 resized) to: `video-summarizer/phase5_smoke_inputs/{input_name}/frame_cogvideox_preprocessed.jpg`
- [x] **Identical Hardware & Settings Preserved**:
  - GPU: NVIDIA GeForce RTX 3090 (24 GB)
  - Precision: `torch.bfloat16`
  - CPU Offload & VAE Tiling: Active on both runs.
  - Wan settings: 832x480 resolution, 49 frames, 30 inference steps, guidance scale 5.0.
  - CogVideoX settings: 720x480 resolution, 49 frames, 50 inference steps, guidance scale 6.0.

---

## 2. Quantitative Results & Timings

### Wan 2.2 TI2V-5B
- **Run Environment**: `wan21` Conda environment (PyTorch 2.7.1 + CUDA 11.8)
- **Mean Generation Time**: **92.8 seconds** (90.4 seconds in the first run - no regression)
- **Mean Peak VRAM**: **15.23 GB** (13.30 GB in the first run - expected increase due to larger input frame feed instead of silent pipeline stretching)

| Input Name | Generation Time (s) | Peak VRAM (GB) | Output MP4 Path | Status |
|---|---|---|---|---|
| **input_a_strong** | 97.48s | 15.23 GB | `phase5_smoke_outputs/wan22_5b/input_a_strong_v2.mp4` | Success |
| **input_b_marginal** | 90.57s | 15.23 GB | `phase5_smoke_outputs/wan22_5b/input_b_marginal_v2.mp4` | Success |
| **input_c_generate** | 90.40s | 15.23 GB | `phase5_smoke_outputs/wan22_5b/input_c_generate_v2.mp4` | Success |

### CogVideoX-5B I2V
- **Run Environment**: `cogvideo5b` Conda environment (PyTorch 2.6.0 + CUDA 12.4)
- **Mean Generation Time**: **562.5 seconds** (565.0 seconds in the first run - no regression)
- **Mean Peak VRAM**: **14.58 GB** (14.58 GB in the first run - no regression)

| Input Name | Generation Time (s) | Peak VRAM (GB) | Output MP4 Path | Status |
|---|---|---|---|---|
| **input_a_strong** | 570.95s (9.5 min) | 14.58 GB | `phase5_smoke_outputs/cogvideox_5b/input_a_strong_v2.mp4` | Success |
| **input_b_marginal** | 558.48s (9.3 min) | 14.58 GB | `phase5_smoke_outputs/cogvideox_5b/input_b_marginal_v2.mp4` | Success |
| **input_c_generate** | 558.17s (9.3 min) | 14.58 GB | `phase5_smoke_outputs/cogvideox_5b/input_c_generate_v2.mp4` | Success |

---

## 3. Side-by-Side Model Profile Comparison

Below is the comparative profile between Wan 2.2 TI2V-5B and CogVideoX-5B I2V:

| Metric | Wan 2.2 TI2V-5B (Rerun) | CogVideoX-5B I2V (Rerun) |
| :--- | :--- | :--- |
| **Inference Resolution** | 832x480 (AR 1.733) | 720x480 (AR 1.5) |
| **Frames / FPS / Duration** | 49 frames / 16 fps / 3.0s | 49 frames / 8 fps / 6.125s |
| **Inference Steps** | 30 steps | 50 steps |
| **Mean Time / Clip** | **92.8s** | **562.5s (9.38 min)** |
| **Mean Peak VRAM** | **15.23 GB** | **14.58 GB** |
| **60-Clip Pipeline Projection** | **~1.55 hours** | **~9.38 hours** |

---

## 4. Deliverables Directory Structure

All outputs and preprocessed frames have been fully written to the workspace.

```
video-summarizer/
└── phase5_smoke_inputs/
    ├── input_a_strong/
    │   ├── frame.jpg (Original)
    │   ├── frame_wan_preprocessed.jpg
    │   └── frame_cogvideox_preprocessed.jpg
    ├── input_b_marginal/
    │   ├── frame.jpg (Original)
    │   ├── frame_wan_preprocessed.jpg
    │   └── frame_cogvideox_preprocessed.jpg
    └── input_c_generate/
        ├── frame.jpg (Original)
        ├── frame_wan_preprocessed.jpg
        └── frame_cogvideox_preprocessed.jpg

phase5_smoke_outputs/
├── wan22_5b/
│   ├── input_a_strong_v2.mp4
│   ├── input_b_marginal_v2.mp4
│   ├── input_c_generate_v2.mp4
│   └── results.json (Updated)
└── cogvideox_5b/
    ├── input_a_strong_v2.mp4
    ├── input_b_marginal_v2.mp4
    ├── input_c_generate_v2.mp4
    └── results.json (Updated)
```

> [!NOTE]
> Per the constraints of the rerun brief, no visual quality judgment has been made. The user can now compare the outputs side-by-side visually to evaluate details, temporal coherence, and motion quality under identical conditions.
