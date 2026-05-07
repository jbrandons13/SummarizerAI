# Significance Tests for CV-Align Implementation

This report contains paired t-tests and Wilcoxon signed-rank tests for the newly added CV-Align matching algorithm across 10 videos.

## CV-Align vs Vanilla DP (Caption) (caption_temporal_dp vs caption_temporal_cvalign)

| Metric | Mean (Vanilla/Greedy) | Mean (CV-Align) | T-stat | T p-value | Wilcoxon p-value | Sig |
|--------|----------------------|-----------------|--------|-----------|-------------------|-----|
| clipscore_mean | 0.5658 | 0.5634 | 1.3719 | 0.2124 | 0.1094 | NO |
| temporal_acc_15s | 0.7827 | 0.8036 | -1.0000 | 0.3506 | 1.0000 | NO |
| visual_coherence_mean | 0.7401 | 0.7406 | -0.0622 | 0.9522 | 0.8750 | NO |
| viscoher_strict | 0.6961 | 0.6766 | 0.7422 | 0.4821 | 0.7500 | NO |
| scene_diversity | 0.8110 | 0.8318 | -1.0000 | 0.3506 | 1.0000 | NO |

## CV-Align vs Vanilla DP (SigLIP) (siglip_temporal_dp vs siglip_temporal_cvalign)

| Metric | Mean (Vanilla/Greedy) | Mean (CV-Align) | T-stat | T p-value | Wilcoxon p-value | Sig |
|--------|----------------------|-----------------|--------|-----------|-------------------|-----|
| clipscore_mean | 0.5493 | 0.5493 | 1.0346 | 0.3407 | 0.3750 | NO |
| temporal_acc_15s | 0.8605 | 0.8605 | nan | 1.0000 | 1.0000 | NO |
| visual_coherence_mean | 0.6602 | 0.6602 | nan | 1.0000 | 1.0000 | NO |
| viscoher_strict | 0.6406 | 0.6406 | nan | 1.0000 | 1.0000 | NO |
| scene_diversity | 0.9558 | 0.9558 | nan | 1.0000 | 1.0000 | NO |

## CV-Align vs Greedy (Caption) (caption_temporal vs caption_temporal_cvalign)

| Metric | Mean (Vanilla/Greedy) | Mean (CV-Align) | T-stat | T p-value | Wilcoxon p-value | Sig |
|--------|----------------------|-----------------|--------|-----------|-------------------|-----|
| clipscore_mean | 0.5714 | 0.5634 | 2.0615 | 0.0782 | 0.0547 | NO |
| temporal_acc_15s | 0.7381 | 0.8036 | -0.9538 | 0.3719 | 0.5000 | NO |
| visual_coherence_mean | 0.6940 | 0.7406 | -2.7641 | 0.0279 | 0.0625 | YES |
| viscoher_strict | 0.6940 | 0.6766 | 1.0483 | 0.3294 | 0.5625 | NO |
| scene_diversity | 1.0000 | 0.8318 | 2.4376 | 0.0449 | 0.0625 | YES |

## CV-Align vs Greedy (SigLIP) (siglip_temporal vs siglip_temporal_cvalign)

| Metric | Mean (Vanilla/Greedy) | Mean (CV-Align) | T-stat | T p-value | Wilcoxon p-value | Sig |
|--------|----------------------|-----------------|--------|-----------|-------------------|-----|
| clipscore_mean | 0.5490 | 0.5493 | -0.0594 | 0.9545 | 0.9375 | NO |
| temporal_acc_15s | 0.8844 | 0.8605 | 0.4201 | 0.6891 | 1.0000 | NO |
| visual_coherence_mean | 0.6477 | 0.6602 | -0.6913 | 0.5152 | 0.7500 | NO |
| viscoher_strict | 0.6477 | 0.6406 | 0.7024 | 0.5087 | 0.5000 | NO |
| scene_diversity | 1.0000 | 0.9558 | 1.5428 | 0.1738 | 0.5000 | NO |

