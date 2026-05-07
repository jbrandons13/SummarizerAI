# Significance Tests V2 (Track C Fallback)

This report contains paired t-tests and Wilcoxon signed-rank tests for key comparisons on the cleanrun_v1 dataset, including the new Scene Diversity and Strict Visual Coherence metrics.

## T on Caption (caption_direct vs caption_temporal)

| Metric | T-stat | T p-value | Wilcoxon p-value | Significance |
|--------|--------|-----------|-------------------|--------------|
| clipscore_mean | 1.4931 | 0.1696 | 0.2324 | NO |
| temporal_acc_15s | -7.0019 | 0.0001 | 0.0020 | YES |
| visual_coherence_mean | -0.2768 | 0.7882 | 0.8457 | NO |
| viscoher_strict | -0.2768 | 0.7882 | 0.8457 | NO |
| scene_diversity | nan | nan | 1.0000 | NO |
| visual_relevance | 1.0000 | 0.3434 | 0.6250 | NO |

## T on SigLIP (siglip_direct vs siglip_temporal)

| Metric | T-stat | T p-value | Wilcoxon p-value | Significance |
|--------|--------|-----------|-------------------|--------------|
| clipscore_mean | 0.3019 | 0.7696 | 0.4922 | NO |
| temporal_acc_15s | -7.7396 | 0.0000 | 0.0020 | YES |
| visual_coherence_mean | -2.1650 | 0.0586 | 0.1055 | NO |
| viscoher_strict | -2.1650 | 0.0586 | 0.1055 | NO |
| scene_diversity | nan | nan | 1.0000 | NO |
| visual_relevance | -0.3612 | 0.7263 | 1.0000 | NO |

## DP on Caption (caption_temporal vs caption_temporal_dp)

| Metric | T-stat | T p-value | Wilcoxon p-value | Significance |
|--------|--------|-----------|-------------------|--------------|
| clipscore_mean | 1.3345 | 0.2148 | 0.3750 | NO |
| temporal_acc_15s | -0.8851 | 0.3991 | 0.5000 | NO |
| visual_coherence_mean | -2.5808 | 0.0297 | 0.0312 | YES |
| viscoher_strict | -0.6734 | 0.5176 | 0.6875 | NO |
| scene_diversity | 2.1181 | 0.0632 | 0.0625 | NO |
| visual_relevance | 0.8018 | 0.4433 | 0.7500 | NO |

## DP on SigLIP (siglip_temporal vs siglip_temporal_dp)

| Metric | T-stat | T p-value | Wilcoxon p-value | Significance |
|--------|--------|-----------|-------------------|--------------|
| clipscore_mean | -0.6992 | 0.5021 | 0.4375 | NO |
| temporal_acc_15s | 0.4215 | 0.6833 | 0.6250 | NO |
| visual_coherence_mean | -0.4407 | 0.6698 | 0.8125 | NO |
| viscoher_strict | 1.5076 | 0.1659 | 0.1875 | NO |
| scene_diversity | 1.9378 | 0.0846 | 0.2500 | NO |
| visual_relevance | -1.5000 | 0.1679 | 0.5000 | NO |

## Caption-best vs SigLIP-best (caption_temporal_dp vs siglip_temporal_dp)

| Metric | T-stat | T p-value | Wilcoxon p-value | Significance |
|--------|--------|-----------|-------------------|--------------|
| clipscore_mean | 1.0455 | 0.3231 | 0.3223 | NO |
| temporal_acc_15s | -1.0550 | 0.3189 | 0.3438 | NO |
| visual_coherence_mean | 3.5333 | 0.0064 | 0.0020 | YES |
| viscoher_strict | 2.8603 | 0.0188 | 0.0195 | YES |
| scene_diversity | -1.3491 | 0.2103 | 0.3125 | NO |
| visual_relevance | 0.2308 | 0.8227 | 0.7812 | NO |

