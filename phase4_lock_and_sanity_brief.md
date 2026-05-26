# Phase 4 Threshold Lock + Visual Sanity Check

## Goal

Two small tasks:
1. Lock the retrieval gate threshold at 0.12 (lowered from 0.13 based on aggregate calibration across 10 videos).
2. Visually verify that the scenes selected for `action="retrieve"` actually match the narration text.

## Rules of engagement

- Be direct.
- Do not commit.
- Print `=== TASK N DONE ===` after each task.

---

## Task 1: Lock threshold at 0.12

In `configs/default.yaml`, update the Phase 4 block:

```yaml
phase4:
  gate_threshold: 0.12         # lowered from 0.13 after 10-video calibration
  extend_epsilon: 0.03
  max_group_size: 5
  join_sep: " "
  temporal_sigma: 30.0
  enable_temporal_prior: true
```

### Definition of done

- Config updated.
- YAML parses cleanly.
- Pipeline imports cleanly.

---

## Task 2: Visual sanity check on retrieve assignments

The aggregate stats show 41.1% retrieve fraction at threshold 0.12. We need to confirm that these "retrieve" decisions are actually pairing narration sentences with visually relevant scenes — not just passing the threshold by luck.

### Steps

1. Re-run the pipeline (or reuse cached Phase 4 outputs if available) on all 10 evaluation videos with threshold 0.12.

2. Across all videos, collect every assignment with `action == "retrieve"`.

3. **Random sample 5 retrieve assignments**, stratified by similarity bucket to get a mix:
   - 2 assignments with weighted_similarity in `[0.12, 0.15)` (marginal matches)
   - 2 assignments with weighted_similarity in `[0.15, 0.20)` (moderate matches)
   - 1 assignment with weighted_similarity `>= 0.20` (strong matches, if any exist)

   If a bucket has fewer than the requested count, take what is available and note it.

4. For each of the 5 sampled assignments, save:
   - The joined sentence text (concatenation of the sentences in the group)
   - The selected scene's middle frame as a JPG image (filename: `sanity_<video_id>_g<group_id>_<sim>.jpg`)
   - Place all 5 images in a folder `sanity_check_threshold_012/` at repo root

5. For each sample, also report in plain text:
   - Video id
   - Sentence ids in the group
   - Joined sentence text
   - Selected scene id and its time range (start, end)
   - Weighted similarity, raw cosine, temporal weight
   - Path to the saved frame image

### Report format

```
=== SANITY CHECK REPORT ===

Threshold used: 0.12
Total retrieve assignments across 10 videos: <n>

Sampled assignments:

[1] video=<id>, group_id=<id>, sents=[<ids>]
    text: "<joined text>"
    scene: id=<id>, time=(<start>, <end>)
    similarity: weighted=<x>, raw=<x>, temporal_weight=<x>
    frame: sanity_check_threshold_012/sanity_<...>.jpg

[2] ... (same structure)

[3] ... (same structure)

[4] ... (same structure)

[5] ... (same structure)

Bucket distribution of sample:
  [0.12, 0.15): <n>
  [0.15, 0.20): <n>
  [0.20, +inf): <n>

Blockers: <list, or "none">
```

## Hard constraints

- Do not modify code in `src/`.
- Do not modify Phase 5 (it does not exist yet).
- Do not commit.
- Frame images should be at reasonable resolution (e.g., 480p or 720p), not full source quality.

End of brief.
