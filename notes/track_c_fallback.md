# Track C Fallback Documentation

## Overview
Track C aimed to eliminate scene reuse looping in DP arms by re-tuning the `dp_backward_penalty` parameter. A pilot parameter sweep was conducted on `review_7` (Caption DP) and `review_2` (SigLIP DP) to find a value that minimizes reuse without significantly degrading temporal accuracy.

## Pilot Sweep Results

| Video | Arm | BP | Assignment | Unique | Reuse Rate | TempAcc (15s) | VisCoher | CLIPScore |
|-------|-----|----|------------|--------|------------|--------------|----------|-----------|
| review_7 | caption_temporal_dp | 0.05 | [2, 4, 4, 4, 4, 4] | 2 | 0.667 | 0.667 | 0.903 | 0.547 |
| review_7 | caption_temporal_dp | 0.1 | [2, 4, 4, 4, 4, 4] | 2 | 0.667 | 0.667 | 0.903 | 0.547 |
| review_7 | caption_temporal_dp | 0.2 | [2, 4, 4, 4, 4, 4] | 2 | 0.667 | 0.667 | 0.903 | 0.547 |
| review_7 | caption_temporal_dp | 0.3 | [2, 4, 4, 4, 4, 4] | 2 | 0.667 | 0.667 | 0.903 | 0.547 |
| review_7 | caption_temporal_dp | 0.5 | [2, 4, 4, 4, 4, 4] | 2 | 0.667 | 0.667 | 0.903 | 0.547 |
| review_2 | siglip_temporal_dp | 0.05 | [3, 18, 18, 45, 62, 81] | 5 | 0.167 | 0.833 | 0.662 | 0.471 |
| review_2 | siglip_temporal_dp | 0.1 | [3, 18, 18, 45, 62, 81] | 5 | 0.167 | 0.833 | 0.662 | 0.471 |
| review_2 | siglip_temporal_dp | 0.2 | [3, 18, 18, 45, 62, 81] | 5 | 0.167 | 0.833 | 0.662 | 0.471 |
| review_2 | siglip_temporal_dp | 0.3 | [3, 18, 18, 45, 62, 81] | 5 | 0.167 | 0.833 | 0.662 | 0.471 |
| review_2 | siglip_temporal_dp | 0.5 | [3, 18, 18, 45, 62, 81] | 5 | 0.167 | 0.833 | 0.662 | 0.471 |

## Justification for Fallback
The parameter sweep failed to resolve the looping issue in `review_7` Caption DP. Even at a minimal `dp_backward_penalty` of 0.05, the DP algorithm continued to select scene 4 for five consecutive sentences (`[2, 4, 4, 4, 4, 4]`). This indicates that the semantic signal for scene 4 is overwhelmingly dominant or that alternatives are extremely weak, and simple penalty adjustments are insufficient to break the loop.

Since the success criteria (reuse rate ≤ 1/6 for `review_7`) were not met by any of the tested values, Track C is abandoned in favor of **Track B Fallback**.

## Next Steps
- Compute **Scene Diversity** metrics on the existing `cleanrun_v1` dataset.
- Compute **Modified Visual Coherence (VisCoher_strict)** to account for the inflation caused by scene reuse.
- Update statistical tests and interpretation based on these new insights.
