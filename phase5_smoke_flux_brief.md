# Phase 5 Smoke Test Brief: FLUX + IP-Adapter

## Context

Previous I2V smoke tests (Wan 2.2 5B, CogVideoX 5B) failed to produce usable output even after prompt engineering and aspect ratio fixes. Wan exhibits catastrophic drift (fade to brown), CogVideoX produces near-static output. Both are unsuitable for the Phase 5 generation slot.

New plan: replace I2V diffusion with **T2I diffusion (FLUX) conditioned on the retrieved keyframe via IP-Adapter**. Motion will be added in a separate downstream step (Ken Burns + depth parallax), not in this smoke test. This brief covers only the still-image generation step.

Goal of this smoke test: confirm that FLUX + IP-Adapter produces high-quality, frame-grounded still images that preserve the visual identity of the conditioning frame, on the same 3 inputs used for previous smoke tests.

## Scope

Run FLUX + IP-Adapter on the same 3 inputs (`input_a_strong`, `input_b_marginal`, `input_c_generate`). Generate 1 still image per input. Apple-to-apple with previous smoke tests so the user can directly compare outputs.

Do NOT:
- Add motion / Ken Burns / parallax in this smoke test.
- Modify any other phase.
- Run the scoring script yet.

## Model selection

Use **FLUX.1-schnell** (`black-forest-labs/FLUX.1-schnell`) as primary candidate.

Rationale:
- ~12 GB VRAM footprint, fits RTX 3090 22 GB comfortably with headroom for IP-Adapter.
- Fast inference (4 steps default, ~5-10 seconds per image on RTX 3090).
- Apache 2.0 license, no gating.

If FLUX.1-schnell + IP-Adapter integration runs into compatibility issues (e.g., no compatible IP-Adapter checkpoint for schnell), fall back to **FLUX.1-dev** (`black-forest-labs/FLUX.1-dev`). Report which model was actually used.

For IP-Adapter, use the most recent FLUX-compatible IP-Adapter checkpoint available on HuggingFace. Common candidates as of early 2026:
- `XLabs-AI/flux-ip-adapter`
- `InstantX/FLUX.1-dev-IP-Adapter`

Check which is currently maintained and well-documented in Diffusers. Report the exact checkpoint used.

## Inputs

For each of the 3 inputs:

- **Conditioning frame**: `phase5_smoke_inputs/{input_name}/frame.jpg` (the same frame used in previous smoke tests).
- **Prompt**: use the engineered prompts from the re-run smoke test (the ones with motion/style cues). Verbatim:
  - `input_a_strong`: `A Xiaomi smartphone with a custom gaming case attached, rear screen displaying game controller buttons that glow softly, slow camera pan around the device, tech product review style, soft studio lighting, clean dark background, shallow depth of field`
  - `input_b_marginal`: `Two modern smartphones lying side by side on a clean surface, camera slowly tilts down revealing their identical sleek metal frames and rounded edges, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field`
  - `input_c_generate`: `Close-up of a smartphone's rear screen lighting up to display music playback controls with album art, fingers gently tap the screen, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus`

Note: motion words ("slow camera pan", "tilts down", "dolly forward") in the prompts are OK to keep — FLUX will treat them as style/composition cues rather than motion, which is fine for our purposes.

## Inference settings

Standard FLUX defaults:
- `num_inference_steps`: 4 (schnell) or 28 (dev) — use model-recommended default.
- `guidance_scale`: 0.0 (schnell) or 3.5 (dev) — use model-recommended default.
- IP-Adapter scale / strength: start at 0.7. If output drifts too far from conditioning frame, increase to 0.8-0.9. If output is too close to a literal copy of the conditioning frame, decrease to 0.5-0.6.
- Output resolution: 832x480 (match Wan resolution, landscape, matches final video format).
- Seed: set explicitly (e.g., 42) for reproducibility. Report it.
- Dtype: bfloat16.

## Frame preprocessing

Use the same center-crop-then-resize approach validated in the re-run smoke test, target 832x480. Save the preprocessed conditioning frame to `phase5_smoke_inputs/{input_name}/frame_flux_preprocessed.jpg` for verification.

## Implementation notes

- New script: `src/phase5_smoke_flux.py`. Do not modify existing smoke test files.
- Use HuggingFace Diffusers `FluxPipeline` with IP-Adapter loaded via `pipe.load_ip_adapter(...)`.
- Enable `pipe.enable_model_cpu_offload()` if VRAM tight.
- Cache model at `~/models/flux_schnell` (or `flux_dev` if fallback).

## Deliverables

1. 3 generated still images saved as:
   - `phase5_smoke_outputs/flux_ipadapter/input_a_strong.png`
   - `phase5_smoke_outputs/flux_ipadapter/input_b_marginal.png`
   - `phase5_smoke_outputs/flux_ipadapter/input_c_generate.png`
2. 3 preprocessed conditioning frames saved alongside.
3. A short markdown report containing:
   - Exact model used (schnell or dev), IP-Adapter checkpoint, IP-Adapter scale used.
   - Inference time per image, peak VRAM.
   - Seed used.
   - Filepaths of outputs.
   - No quality interpretation — user will visual-judge.

## Constraints

- Do not regenerate Wan or CogVideoX outputs.
- Do not download more than necessary. FLUX-schnell ~12 GB + IP-Adapter ~few hundred MB.
- If model download fails or IP-Adapter integration breaks, stop and report — do not improvise with a different architecture.
- If both FLUX-schnell and FLUX-dev fail integration with IP-Adapter, stop and report; the user will decide next step.

=== END OF BRIEF ===
