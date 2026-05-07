# Visual Coherence Strict Results

## VisCoher vs VisCoher_Strict Comparison

| arm                       |   visual_coherence_mean |   viscoher_strict |       gap |
|:--------------------------|------------------------:|------------------:|----------:|
| caption_direct            |                0.655739 |          0.655739 | 0         |
| caption_temporal          |                0.662249 |          0.662249 | 0         |
| caption_temporal_dp       |                0.709538 |          0.674314 | 0.035224  |
| random                    |                0.659628 |          0.659628 | 0         |
| siglip_direct             |                0.593703 |          0.593703 | 0         |
| siglip_temporal           |                0.626297 |          0.626297 | 0         |
| siglip_temporal_dp        |                0.632029 |          0.612724 | 0.0193046 |
| siglip_temporal_hungarian |                0.627601 |          0.627601 | 0         |

### Analysis
The largest gap is observed in the **caption_temporal_dp** arm, indicating it is most affected by scene reuse artifacting.

## Statistical Significance (Paired T-test for VisCoher_Strict)

- **caption_temporal vs caption_temporal_dp**: p-value = 0.5176 (Not Significant)
- **siglip_temporal vs siglip_temporal_dp**: p-value = 0.1659 (Not Significant)
- **caption_temporal_dp vs siglip_temporal_dp**: p-value = 0.0188 (Significant)
