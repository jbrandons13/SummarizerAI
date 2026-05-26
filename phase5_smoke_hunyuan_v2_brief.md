# Phase 5 Smoke Test Brief: HunyuanVideo 1.5 I2V (Apple-to-Apple Re-run)

## Context

HunyuanVideo 1.5 480p I2V was previously smoke-tested in an earlier session with different inputs (`input_a_high`, `input_b_mid`, `input_c_low`). Results: ~175 sec/clip, peak VRAM 21.79 GB, output saved to `phase5_smoke_outputs/hunyuan/`.

Problem: those inputs are **not the same** as the inputs used for Wan 2.2 5B, CogVideoX 5B, and FLUX smoke tests in this session (`input_a_strong`, `input_b_marginal`, `input_c_generate`). Direct comparison is not possible.

This brief is a re-run of the HunyuanVideo smoke test using the same 3 inputs and engineered prompts as the most recent smoke tests, for apple-to-apple visual comparison.

## Scope

Run HunyuanVideo 1.5 480p I2V on `input_a_strong`, `input_b_marginal`, `input_c_generate`. Generate 1 clip per input. Save outputs alongside (NOT overwriting) the previous HunyuanVideo outputs.

Do NOT:
- Run other models.
- Run scoring scripts yet.
- Delete or move previous HunyuanVideo outputs (they may still be referenced for the earlier-session comparison).

## Model

Use the same model as before: `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_distilled`.

If the model is still cached locally (from the prior smoke test), reuse the cache to save bandwidth.

## Prerequisite: Free VRAM

The previous run hit peak 21.79 GB out of 22 GB usable. Any other process consuming GPU memory will cause OOM. Before running:

1. Check `nvidia-smi` for GPU memory usage.
2. If Ollama, background orchestrators, or other GPU consumers are running, pause them as done in the FLUX smoke test (`SIGSTOP` on the relevant PIDs, then `SIGCONT` after).
3. Verify GPU memory is mostly free (<2 GB used) before starting inference.

## Inputs

For each of the 3 inputs:

- **Conditioning frame**: `phase5_smoke_inputs/{input_name}/frame.jpg`
- **Frame preprocessing**: center-crop-then-resize to match HunyuanVideo's required input resolution. Check what resolution the model expects (likely 832x480 or 720x480 — verify from the pipeline's documentation or prior smoke test code). Save preprocessed frame as `phase5_smoke_inputs/{input_name}/frame_hunyuan_preprocessed.jpg`.

### Prompts (verbatim, same as recent smoke tests)

- **input_a_strong**:
  > `A Xiaomi smartphone with a custom gaming case attached, rear screen displaying game controller buttons that glow softly, slow camera pan around the device, tech product review style, soft studio lighting, clean dark background, shallow depth of field`

- **input_b_marginal**:
  > `Two modern smartphones lying side by side on a clean surface, camera slowly tilts down revealing their identical sleek metal frames and rounded edges, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field`

- **input_c_generate**:
  > `Close-up of a smartphone's rear screen lighting up to display music playback controls with album art, fingers gently tap the screen, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus`

## Inference settings

Match the prior smoke test as closely as possible (since it ran successfully):
- `num_inference_steps`: 20 (distilled variant)
- Frame count: same as before (likely 49 or similar, verify)
- Resolution: 832x480 if model supports landscape, else 720x480 (whatever matches the prior config)
- Dtype: bfloat16
- Seed: 42 (explicit, for reproducibility)
- VRAM optimizations: `enable_sequential_cpu_offload` (since peak was 21.79 GB last time, sequential offload is mandatory)
- VAE tiling: enabled if pipeline supports it

If the prior `smoke_hunyuan.py` script exists, base the new run on that script with only the inputs and seed changed. New script name: `smoke_hunyuan_v2.py` to distinguish from the prior run.

## Deliverables

1. 3 generated videos saved as:
   - `phase5_smoke_outputs/hunyuan_v2/input_a_strong.mp4`
   - `phase5_smoke_outputs/hunyuan_v2/input_b_marginal.mp4`
   - `phase5_smoke_outputs/hunyuan_v2/input_c_generate.mp4`
2. 3 preprocessed conditioning frames saved alongside originals.
3. A short markdown report containing:
   - Model id, inference steps, seed, resolution used.
   - Per-clip generation time, peak VRAM.
   - Any deviations from the prior smoke test config (and why).
   - Filepaths of outputs.
   - No quality interpretation — user will visual-judge.

## Constraints

- Do not modify the inference pipeline. Use it as configured in the prior smoke test.
- If OOM occurs despite freeing VRAM, stop and report. Do not try smaller resolutions or shorter clips without confirming with the user — apple-to-apple comparison requires consistent settings.
- If model download is required (cache missing), confirm download size first; HunyuanVideo 1.5 distilled is ~16 GB.

=== END OF BRIEF ===
