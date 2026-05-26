# Phase 5 Smoke Test Diagnostic Brief

## Context

Visual inspection of the 6 generated videos (3 Wan 2.2 5B + 3 CogVideoX 5B) shows severe quality issues across both models: warping, flickering, melting artifacts. Since both models are publicly released and used at scale, two failing in the same way suggests the issue is likely in the **inputs or settings**, not the models themselves.

Before deciding to switch models, change approach, or tune anything, we need to know exactly what was used in the smoke tests. This brief is **diagnostic only**. Do not regenerate videos. Do not change code. Just report.

## Deliverable

A single markdown report with the three sections below. If any info is missing or you cannot find it, say so explicitly — do not guess or fabricate.

### Section 1: Prompts (verbatim)

For each of the 3 inputs (`input_a_strong`, `input_b_marginal`, `input_c_generate`), report:

- The exact prompt string passed to the model. Verbatim. Quoted.
- Where the prompt came from: was it the raw sentence text from the Phase 2 summary, or was it engineered (keywords appended, style cues, motion verbs added, etc)?
- The character/token length of each prompt.

If the prompts differ between Wan and CogVideoX runs (e.g., model-specific formatting), report both.

### Section 2: Inference settings

For each model, report the inference call as actually executed:

- `num_inference_steps`
- `guidance_scale`
- `negative_prompt` (if any)
- `num_frames`
- Resolution (height x width) actually passed
- `seed` (if set)
- Any other non-default kwargs

If these are buried in a config file or hardcoded, paste the relevant code snippet.

### Section 3: Conditioning frame handling

For each of the 3 conditioning frames, report:

- File path and original dimensions (height x width).
- Aspect ratio of the original frame.
- Aspect ratio expected by each model (832x480 = 1.733 for Wan; 720x480 = 1.5 for CogVideoX).
- What was done to resolve any aspect ratio mismatch: resize, center crop, pad, or stretch? Paste the preprocessing code if any.
- Whether the frame was saved after preprocessing (so we can visually inspect what the model actually received).

## Format

Markdown report, ~1 page. Tables are fine where useful. No commentary or interpretation needed — just facts. We will analyze afterward.

## Constraints

- Do not run any new generation.
- Do not download anything.
- If a piece of information is genuinely not recoverable from the existing code/logs, write "NOT RECOVERABLE" and explain why.

=== END OF BRIEF ===
