# Phase 5 Smoke Test Report: FLUX.1-schnell + IP-Adapter

## Context & Objectives

Following the challenges with motion coherence and catastrophic color-drifting in earlier image-to-video (I2V) smoke tests (Wan 2.2 5B and CogVideoX-5B I2V), the video summarization pipeline's generation phase was pivoted to **T2I diffusion conditioned on retrieved keyframes via IP-Adapter**. 

Motion generation has been decoupled into a separate downstream stage (e.g., Ken Burns + depth parallax). This smoke test evaluated the Still-Image Generation step using **FLUX.1-schnell** and a FLUX-compatible **IP-Adapter** to ensure frame-grounded, visually consistent still image outputs on the RTX 3090 hardware across the same 3 inputs.

---

## 1. System & Model Configuration

- **Base Diffusion Model**: `black-forest-labs/FLUX.1-schnell` (Loaded in `torch.bfloat16` precision)
- **Local Cache Path**: `~/models/flux_schnell`
- **IP-Adapter Checkpoint**: `XLabs-AI/flux-ip-adapter` (Safetensors weight format)
- **CLIP Vision Encoder**: `openai/clip-vit-large-patch14`
- **IP-Adapter Conditioning Scale**: `0.7`
- **Generator Seed**: `42` (Set explicitly for strict output reproducibility)
- **Inference Resolution**: `832x480` (Landscape, matching final video aspect ratio of `1.733`)
- **Optimization Strategy**: Sequential CPU Offloading (`pipe.enable_sequential_cpu_offload()`) for minimal peak VRAM overhead.

---

## 2. Quantitative Results & Timings

All three test cases completed successfully in a single batch. By utilizing sequential CPU offloading, peak PyTorch active memory on the GPU was maintained under **1.00 GB**, leaving substantial headroom on the RTX 3090.

| Input Name | Prompts (Verbatim Cues) | Generation Time | Peak VRAM | Status |
|---|---|---|---|---|
| **input_a_strong** | *A Xiaomi smartphone with a custom gaming case attached, rear screen displaying game controller buttons that glow softly, slow camera pan around the device, tech product review style, soft studio lighting, clean dark background, shallow depth of field* | **16.3 seconds** | **0.99 GB** | Success |
| **input_b_marginal** | *Two modern smartphones lying side by side on a clean surface, camera slowly tilts down revealing their identical sleek metal frames and rounded edges, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field* | **15.9 seconds** | **0.99 GB** | Success |
| **input_c_generate** | *Close-up of a smartphone's rear screen lighting up to display music playback controls with album art, fingers gently tap the screen, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus* | **15.9 seconds** | **0.99 GB** | Success |

---

## 3. Output Filepaths & Deliverables

All outputs and preprocessed conditioning frames have been fully generated and written to their designated locations in the workspace.

```
video-summarizer/
├── phase5_smoke_inputs/
│   ├── input_a_strong/
│   │   ├── frame.jpg (Original)
│   │   └── frame_flux_preprocessed.jpg (832x480 crop-and-resize)
│   ├── input_b_marginal/
│   │   ├── frame.jpg (Original)
│   │   └── frame_flux_preprocessed.jpg (832x480 crop-and-resize)
│   └── input_c_generate/
│       ├── frame.jpg (Original)
│       └── frame_flux_preprocessed.jpg (832x480 crop-and-resize)
│
└── phase5_smoke_outputs/
    └── flux_ipadapter/
        ├── input_a_strong.png (Generated still)
        ├── input_b_marginal.png (Generated still)
        ├── input_c_generate.png (Generated still)
        └── results.json (Timing & VRAM metrics)
```

### Exact Artifact Paths:
- **input_a_strong (Generated Still)**: `phase5_smoke_outputs/flux_ipadapter/input_a_strong.png`
- **input_b_marginal (Generated Still)**: `phase5_smoke_outputs/flux_ipadapter/input_b_marginal.png`
- **input_c_generate (Generated Still)**: `phase5_smoke_outputs/flux_ipadapter/input_c_generate.png`
- **input_a_strong (Preprocessed conditioning frame)**: `phase5_smoke_inputs/input_a_strong/frame_flux_preprocessed.jpg`
- **input_b_marginal (Preprocessed conditioning frame)**: `phase5_smoke_inputs/input_b_marginal/frame_flux_preprocessed.jpg`
- **input_c_generate (Preprocessed conditioning frame)**: `phase5_smoke_inputs/input_c_generate/frame_flux_preprocessed.jpg`

> [!NOTE]
> Per the constraints of the brief, no quality interpretation of the generated still images has been performed. The visual results are ready for manual inspection to assess how well the IP-Adapter preserves the visual identity of the original conditioning frame.
