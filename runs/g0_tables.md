### Geology Legacy - Fixed Frontier
| w | mean_concept | c̄ | ref_sim |
|---|---|---|---|
| 0.00 | 0.2462 | 1.0000 (n=3) | 0.5381 |
| 0.20 | 0.2801 | 0.8527 (n=3) | 0.7426 |
| 0.30 | 0.2927 | 0.6568 (n=3) | 0.8749 |
| 0.40 | 0.2874 | 0.7323 (n=3) | 0.8241 |
| 0.50 | 0.2991 | 0.5770 (n=3) | 0.9166 |
| 0.60 | 0.2903 | 0.6054 (n=3) | 0.8049 |
| 0.80 | 0.2865 | 0.5729 (n=3) | 0.7879 |

### Geology Collided - Fixed Frontier
| w | mean_concept | c̄ | ref_sim |
|---|---|---|---|
| 0.00 | 0.2425 | 1.0000 | 0.4810 |
| 0.20 | 0.2664 | 0.6776 | 0.7753 |
| 0.30 | 0.2804 | 0.5930 | 0.8685 |
| 0.40 | 0.2851 | 0.5594 | 0.9016 |
| 0.50 | 0.2923 | 0.5153 | 0.9195 |
| 0.60 | 0.2925 | 0.4973 | 0.9206 |
| 0.80 | 0.2926 | 0.4828 | 0.9099 |

### Geology v2 - Fixed Frontier
| w | mean_concept | c̄ | ref_sim |
|---|---|---|---|
| 0.00 | 0.2403 | 1.0000 | 0.5270 |
| 0.20 | 0.2662 | 0.7373 | 0.7678 |
| 0.30 | 0.2782 | 0.6580 | 0.8539 |
| 0.40 | 0.2894 | 0.6131 | 0.8980 |
| 0.50 | 0.2906 | 0.5637 | 0.9188 |
| 0.60 | 0.2914 | 0.5529 | 0.9254 |
| 0.80 | 0.2897 | 0.5222 | 0.9172 |

### Ecology v2 - Fixed Frontier
| w | mean_concept | c̄ | ref_sim |
|---|---|---|---|
| 0.00 | 0.2200 | 1.0000 | 0.2934 |
| 0.20 | 0.2275 | 0.7347 | 0.4747 |
| 0.30 | 0.2230 | 0.6367 | 0.5974 |
| 0.40 | 0.2152 | 0.5082 | 0.7551 |
| 0.50 | 0.2111 | 0.3924 | 0.8247 |
| 0.60 | 0.2112 | 0.3247 | 0.8562 |
| 0.80 | 0.2129 | 0.2994 | 0.8573 |

### Summary: Method Advantage vs Interpolated Fixed Frontier
| Sweep | Method | mc | c̄ | Δc̄ vs Frontier | n |
|---|---|---|---|---|---|
| Geology Legacy | Original DACA (max w) | 0.2638 | 0.7733 | -0.0794 | n=3 (anchors exist only for shots 005/011/013) |
| Geology Legacy | Benefit-gated (δ=0.01) | 0.2754 | 0.7821 | -0.0707 | n=3 (anchors exist only for shots 005/011/013) |
| Geology Legacy | Pure argmax | 0.2754 | 0.7821 | -0.0707 | n=3 (anchors exist only for shots 005/011/013) |
| Geology Collided | Original DACA (max w) | 0.2696 | 0.6539 | -0.0045 | n=14 |
| Geology Collided | Benefit-gated (δ=0.01) | 0.2693 | 0.6692 | +0.0090 | n=14 |
| Geology Collided | Pure argmax | 0.2701 | 0.6553 | +0.0001 | n=14 |
| Geology v2 | Original DACA (max w) | 0.2677 | 0.6813 | -0.0462 | n=14 |
| Geology v2 | Benefit-gated (δ=0.01) | 0.2711 | 0.7073 | +0.0024 | n=14 |
| Geology v2 | Pure argmax | 0.2713 | 0.7003 | -0.0033 | n=14 |
| Ecology v2 | Original DACA (max w) | 0.2281 | 0.7034 | -0.0312 | n=16 |
| Ecology v2 | Benefit-gated (δ=0.01) | 0.2304 | 0.7203 | -0.0144 | n=16 |
| Ecology v2 | Pure argmax | 0.2304 | 0.7203 | -0.0144 | n=16 |