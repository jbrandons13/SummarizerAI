# Parameter Sweep Pilot Results (Track C)

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

## Verdict

**FAILURE**: No backward penalty value satisfied all three criteria.

---

**Diagnostic Note (2026-05-06)**: A sanity check was performed to verify if the parameters were correctly applied. Test results (bp=0.0, bp=-1.0) on `review_7` confirmed that the matching algorithm responds correctly to parameter changes, but the semantic signal for scene reuse in the Caption track is strong enough to override penalties in the 0.05 - 0.5 range. The identical assignments observed in the sweep are a result of this signal dominance, not a code defect.
