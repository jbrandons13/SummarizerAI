# Paired Significance Tests (n=10)

With n=10 videos, statistical power is limited. Effects with p>0.05 may still be real but undetectable at this sample size. Conversely, with 7 comparisons × 6 metrics = 42 tests, expect ~2 false positives at α=0.05 without correction. Bonferroni-corrected α = 0.0012. We report uncorrected p-values for transparency but interpret cautiously.

## T on Caption (caption_direct vs caption_temporal)

|    | Metric                |   Mean A |   Mean B |   Diff |   t-stat |   p (t-test) |    W |   p (Wilcoxon) |   Cohen d | Sig   |
|---:|:----------------------|---------:|---------:|-------:|---------:|-------------:|-----:|---------------:|----------:|:------|
|  0 | clipscore_mean        |    0.586 |    0.575 | -0.011 |    -1.49 |       0.1696 | 15   |         0.2324 |     -0.47 | ns    |
|  1 | temporal_acc_15s      |    0.268 |    0.77  |  0.503 |     7    |       0.0001 |  0   |         0.002  |      2.21 | **    |
|  2 | visual_coherence_mean |    0.656 |    0.662 |  0.007 |     0.28 |       0.7882 | 25   |         0.8457 |      0.09 | ns    |
|  3 | visual_relevance      |    3     |    2.8   | -0.2   |    -1    |       0.3434 |  2.5 |         0.625  |     -0.32 | ns    |
|  4 | information_retention |    3.4   |    3.4   |  0     |     0    |       1      |  1.5 |         1      |      0    | ns    |
|  5 | factual_faithfulness  |    2.8   |    2.8   |  0     |     0    |       1      |  5   |         1      |      0    | ns    |

## T on SigLIP (siglip_direct vs siglip_temporal)

|    | Metric                |   Mean A |   Mean B |   Diff |   t-stat |   p (t-test) |    W |   p (Wilcoxon) |   Cohen d | Sig   |
|---:|:----------------------|---------:|---------:|-------:|---------:|-------------:|-----:|---------------:|----------:|:------|
|  0 | clipscore_mean        |    0.554 |    0.55  | -0.003 |    -0.3  |       0.7696 | 20   |         0.4922 |     -0.1  | ns    |
|  1 | temporal_acc_15s      |    0.315 |    0.902 |  0.588 |     7.74 |       0      |  0   |         0.002  |      2.45 | **    |
|  2 | visual_coherence_mean |    0.594 |    0.626 |  0.033 |     2.17 |       0.0586 | 11   |         0.1055 |      0.68 | ns    |
|  3 | visual_relevance      |    2.2   |    2.3   |  0.1   |     0.36 |       0.7263 |  4   |         1      |      0.11 | ns    |
|  4 | information_retention |    3.3   |    3.3   |  0     |     0    |       1      |  1.5 |         1      |      0    | ns    |
|  5 | factual_faithfulness  |    2.6   |    2.7   |  0.1   |     0.56 |       0.5911 |  2   |         1      |      0.18 | ns    |

## DP on Caption (caption_temporal vs caption_temporal_dp)

|    | Metric                |   Mean A |   Mean B |   Diff |   t-stat |   p (t-test) |   W |   p (Wilcoxon) |   Cohen d | Sig   |
|---:|:----------------------|---------:|---------:|-------:|---------:|-------------:|----:|---------------:|----------:|:------|
|  0 | clipscore_mean        |    0.575 |    0.57  | -0.005 |    -1.33 |       0.2148 | 8   |         0.375  |     -0.42 | ns    |
|  1 | temporal_acc_15s      |    0.77  |    0.806 |  0.036 |     0.89 |       0.3991 | 1   |         0.5    |      0.28 | ns    |
|  2 | visual_coherence_mean |    0.662 |    0.71  |  0.047 |     2.58 |       0.0297 | 1   |         0.0312 |      0.82 | *     |
|  3 | visual_relevance      |    2.8   |    2.6   | -0.2   |    -0.8  |       0.4433 | 1.5 |         0.75   |     -0.25 | ns    |
|  4 | information_retention |    3.4   |    3.4   |  0     |     0    |       1      | 1.5 |         1      |      0    | ns    |
|  5 | factual_faithfulness  |    2.8   |    2.7   | -0.1   |    -1    |       0.3434 | 0   |         1      |     -0.32 | ns    |

## DP on SigLIP (siglip_temporal vs siglip_temporal_dp)

|    | Metric                |   Mean A |   Mean B |   Diff |   t-stat |   p (t-test) |   W |   p (Wilcoxon) |   Cohen d | Sig   |
|---:|:----------------------|---------:|---------:|-------:|---------:|-------------:|----:|---------------:|----------:|:------|
|  0 | clipscore_mean        |    0.55  |    0.554 |  0.003 |     0.7  |       0.5021 |   4 |         0.4375 |      0.22 | ns    |
|  1 | temporal_acc_15s      |    0.902 |    0.882 | -0.02  |    -0.42 |       0.6833 |   3 |         0.625  |     -0.13 | ns    |
|  2 | visual_coherence_mean |    0.626 |    0.632 |  0.006 |     0.44 |       0.6698 |   6 |         0.8125 |      0.14 | ns    |
|  3 | visual_relevance      |    2.3   |    2.5   |  0.2   |     1.5  |       0.1679 |   0 |         0.5    |      0.47 | ns    |
|  4 | information_retention |    3.3   |    3.3   |  0     |   nan    |     nan      |   0 |         1      |    nan    | ns    |
|  5 | factual_faithfulness  |    2.7   |    2.7   |  0     |   nan    |     nan      |   0 |         1      |    nan    | ns    |

## Hungarian on SigLIP (siglip_temporal vs siglip_temporal_hungarian)

|    | Metric                |   Mean A |   Mean B |   Diff |   t-stat |   p (t-test) |   W |   p (Wilcoxon) |   Cohen d | Sig   |
|---:|:----------------------|---------:|---------:|-------:|---------:|-------------:|----:|---------------:|----------:|:------|
|  0 | clipscore_mean        |    0.55  |    0.551 |  0.001 |        1 |       0.3434 |   0 |              1 |      0.32 | ns    |
|  1 | temporal_acc_15s      |    0.902 |    0.919 |  0.017 |        1 |       0.3434 |   0 |              1 |      0.32 | ns    |
|  2 | visual_coherence_mean |    0.626 |    0.628 |  0.001 |        1 |       0.3434 |   0 |              1 |      0.32 | ns    |
|  3 | visual_relevance      |    2.3   |    2.3   |  0     |      nan |     nan      |   0 |              1 |    nan    | ns    |
|  4 | information_retention |    3.3   |    3.3   |  0     |      nan |     nan      |   0 |              1 |    nan    | ns    |
|  5 | factual_faithfulness  |    2.7   |    2.7   |  0     |      nan |     nan      |   0 |              1 |    nan    | ns    |

## Caption-best vs SigLIP-best (caption_temporal_dp vs siglip_temporal_dp)

|    | Metric                |   Mean A |   Mean B |   Diff |   t-stat |   p (t-test) |    W |   p (Wilcoxon) |   Cohen d | Sig   |
|---:|:----------------------|---------:|---------:|-------:|---------:|-------------:|-----:|---------------:|----------:|:------|
|  0 | clipscore_mean        |    0.57  |    0.554 | -0.016 |    -1.05 |       0.3231 | 17   |         0.3223 |     -0.33 | ns    |
|  1 | temporal_acc_15s      |    0.806 |    0.882 |  0.076 |     1.06 |       0.3189 |  8   |         0.3438 |      0.33 | ns    |
|  2 | visual_coherence_mean |    0.71  |    0.632 | -0.078 |    -3.53 |       0.0064 |  0   |         0.002  |     -1.12 | **    |
|  3 | visual_relevance      |    2.6   |    2.5   | -0.1   |    -0.23 |       0.8227 |  8.5 |         0.7812 |     -0.07 | ns    |
|  4 | information_retention |    3.4   |    3.3   | -0.1   |    -0.43 |       0.6783 |  6   |         1      |     -0.14 | ns    |
|  5 | factual_faithfulness  |    2.7   |    2.7   |  0     |     0    |       1      |  3   |         1      |      0    | ns    |

## Caption-best vs SigLIP-best (greedy) (caption_temporal_dp vs siglip_temporal)

|    | Metric                |   Mean A |   Mean B |   Diff |   t-stat |   p (t-test) |    W |   p (Wilcoxon) |   Cohen d | Sig   |
|---:|:----------------------|---------:|---------:|-------:|---------:|-------------:|-----:|---------------:|----------:|:------|
|  0 | clipscore_mean        |    0.57  |    0.55  | -0.02  |    -1.31 |       0.2239 | 15   |         0.2324 |     -0.41 | ns    |
|  1 | temporal_acc_15s      |    0.806 |    0.902 |  0.096 |     1.32 |       0.2181 |  9.5 |         0.2812 |      0.42 | ns    |
|  2 | visual_coherence_mean |    0.71  |    0.626 | -0.083 |    -3.51 |       0.0066 |  1   |         0.0039 |     -1.11 | **    |
|  3 | visual_relevance      |    2.6   |    2.3   | -0.3   |    -0.9  |       0.3938 |  4.5 |         0.5625 |     -0.28 | ns    |
|  4 | information_retention |    3.4   |    3.3   | -0.1   |    -0.43 |       0.6783 |  6   |         1      |     -0.14 | ns    |
|  5 | factual_faithfulness  |    2.7   |    2.7   |  0     |     0    |       1      |  3   |         1      |      0    | ns    |

