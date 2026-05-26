# Phase 5 Smoke Test Scoring Brief

## Goal

Score the 6 generated videos from Phase 5 smoke tests (3 Wan 2.2 5B + 3 CogVideoX 5B) on three standard metrics to compare model quality side-by-side.

## Metrics

For each generated video, compute:

1. **text_sim**: SigLIP cosine similarity between the prompt text and generated frames, averaged across all frames.
2. **style_sim**: SigLIP cosine similarity between the conditioning frame (the input image used for I2V) and generated frames, averaged across all frames.
3. **color_dist**: Histogram distance (chi-square or Bhattacharyya, your call) between the conditioning frame and generated frames in HSV space, averaged across all frames. Lower = more similar.

## Inputs

- Generated videos:
  - `phase5_smoke_outputs/wan22_5b/input_a_strong.mp4`
  - `phase5_smoke_outputs/wan22_5b/input_b_marginal.mp4`
  - `phase5_smoke_outputs/wan22_5b/input_c_generate.mp4`
  - `phase5_smoke_outputs/cogvideox_5b/input_a_strong.mp4`
  - `phase5_smoke_outputs/cogvideox_5b/input_b_marginal.mp4`
  - `phase5_smoke_outputs/cogvideox_5b/input_c_generate.mp4`
- Conditioning frames and prompt texts: in `phase5_smoke_inputs/`. Confirm exact filenames and metadata format before running. If prompts are in a JSON/YAML manifest, use that as source of truth.

## Encoder

Use the existing `SigLIPEncoder` from `src/models/siglip.py` (`google/siglip2-so400m-patch16-naflex`, dim 1152, L2-normalized). Same encoder used in Phase 4 — keep consistent so scores are comparable across phases.

## Frame sampling

Extract frames from each video at native fps. If a video has many frames (CogVideoX 49 frames is fine, no subsampling needed). Encode all frames in batch.

## Implementation notes

- New script: `src/phase5_score_smoke.py`. Do not modify existing Phase 5 smoke test files.
- Reuse `SigLIPEncoder`. Do not reload model per video.
- Histogram: use OpenCV `cv2.calcHist` on HSV, normalize, then `cv2.compareHist` with `HISTCMP_BHATTACHARYYA`.
- Output: print a markdown table with columns `model | input | text_sim | style_sim | color_dist`. Also save raw JSON to `phase5_smoke_outputs/scores.json`.

## Deliverable

A markdown report with:
1. The comparison table (6 rows).
2. Per-model means for each metric.
3. One-paragraph observation: which model wins on which axis, and any anomaly worth flagging (e.g., suspiciously high or low values, NaN, etc.).

## Constraints

- Do not download new models. SigLIP is already cached.
- Do not regenerate videos. Score existing outputs only.
- If conditioning frame or prompt manifest is missing/ambiguous, stop and report — do not guess.

=== END OF BRIEF ===
