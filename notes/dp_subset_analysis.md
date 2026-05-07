# DP Empirical Subset Analysis

## Section 1.1: Per-video DP vs Greedy comparison

### Table 1.1a: Caption DP vs Caption Greedy per video

| Video     |   TempAcc (15s) |   VisCoher (raw) |   VisCoher_strict |   Scene Diversity |    CLIPScore |   LLM Visual Relevance |
|:----------|----------------:|-----------------:|------------------:|------------------:|-------------:|-----------------------:|
| review_1  |        0        |       0.0509518  |        0.0382389  |         -0.125    |  0.0081783   |                     -1 |
| review_10 |        0.166667 |      -0.00242047 |       -0.00242047 |          0        | -0.0127693   |                      0 |
| review_2  |        0        |       0.0412936  |       -0.0240651  |         -0.166667 |  0.000637261 |                      0 |
| review_3  |        0        |       0          |        0          |          0        |  0           |                      0 |
| review_4  |        0        |       0          |        0          |          0        |  0           |                      0 |
| review_5  |       -0.142857 |       0.0859664  |       -0.0775236  |         -0.428571 | -0.0252891   |                     -2 |
| review_6  |        0        |       0.104401   |        0.104401   |          0        | -0.00105966  |                      1 |
| review_7  |        0.333333 |       0.171908   |        0.10437    |         -0.666667 |  0.00389706  |                      0 |
| review_8  |        0        |       0          |        0          |          0        |  0           |                      0 |
| review_9  |        0        |       0.0207984  |       -0.0223408  |         -0.125    | -0.0189539   |                      0 |

**Positive diff counts (DP > Greedy):**
- TempAcc (15s): 2/10
- VisCoher (raw): 6/10
- VisCoher_strict: 3/10
- Scene Diversity: 0/10
- CLIPScore: 3/10
- LLM Visual Relevance: 1/10

- **Consistent DP improvement:** None
- **Consistent DP regression:** None

### Table 1.1b: SigLIP DP vs SigLIP Greedy per video

| Video     |   TempAcc (15s) |   VisCoher (raw) |   VisCoher_strict |   Scene Diversity |   CLIPScore |   LLM Visual Relevance |
|:----------|----------------:|-----------------:|------------------:|------------------:|------------:|-----------------------:|
| review_1  |        0        |       0          |         0         |          0        |  0          |                      0 |
| review_10 |        0.166667 |      -0.0228966  |        -0.0228966 |          0        |  0.0236085  |                      0 |
| review_2  |        0.166667 |       0.117487   |         0.0329924 |         -0.166667 |  0.00182336 |                      0 |
| review_3  |        0        |       0          |         0         |          0        |  0          |                      0 |
| review_4  |        0        |       0          |         0         |          0        |  0          |                      0 |
| review_5  |        0        |       0.00430923 |        -0.0485415 |         -0.142857 | -0.0266457  |                      1 |
| review_6  |       -0.2      |      -0.00770733 |        -0.0634085 |         -0.2      |  0.0064237  |                      0 |
| review_7  |       -0.333333 |      -0.0338713  |        -0.0338713 |          0        |  0.027379   |                      1 |
| review_8  |        0        |       0          |         0         |          0        |  0          |                      0 |
| review_9  |        0        |       0          |         0         |          0        |  0          |                      0 |

**Positive diff counts (DP > Greedy):**
- TempAcc (15s): 2/10
- VisCoher (raw): 2/10
- VisCoher_strict: 1/10
- Scene Diversity: 0/10
- CLIPScore: 4/10
- LLM Visual Relevance: 2/10

- **Consistent DP improvement:** None
- **Consistent DP regression:** None

## Section 1.2: Video characteristic correlation

Characteristics and DP_strict_gain (viscoher_strict DP - Greedy):

| Video     |   Num Scenes |   Num Sents |    Ratio |   Avg Scene Dur (s) |   Caption Gain |   SigLIP Gain |
|:----------|-------------:|------------:|---------:|--------------------:|---------------:|--------------:|
| review_1  |           67 |           8 |  8.375   |             7.91886 |     0.0382389  |     0         |
| review_10 |          249 |           6 | 41.5     |             3.61807 |    -0.00242047 |    -0.0228966 |
| review_2  |          100 |           6 | 16.6667  |             7.72438 |    -0.0240651  |     0.0329924 |
| review_3  |          224 |           6 | 37.3333  |             4.7768  |     0          |     0         |
| review_4  |          156 |           3 | 52       |             4.55156 |     0          |     0         |
| review_5  |           65 |           7 |  9.28571 |             7.66308 |    -0.0775236  |    -0.0485415 |
| review_6  |          103 |           5 | 20.6     |             8.2356  |     0.104401   |    -0.0634085 |
| review_7  |          168 |           6 | 28       |             4.62024 |     0.10437    |    -0.0338713 |
| review_8  |          182 |           8 | 22.75    |             4.89011 |     0          |     0         |
| review_9  |          144 |           8 | 18       |             3.40549 |    -0.0223408  |     0         |

### Spearman Correlations with viscoher_strict gain:

**Caption track:**
- num_scenes: rho=0.202 (p=0.575)
- num_sentences: rho=-0.282 (p=0.430)
- ratio: rho=0.227 (p=0.528)
- avg_scene_dur: rho=0.288 (p=0.419)

**Siglip track:**
- num_scenes: rho=0.058 (p=0.873)
- num_sentences: rho=0.237 (p=0.511)
- ratio: rho=-0.045 (p=0.901)
- avg_scene_dur: rho=-0.136 (p=0.708)

## Section 1.3: Sentence-position analysis

Aggregate probability of DP differing from Greedy by summary half:

| half        |   differs |
|:------------|----------:|
| First Half  |  0.15     |
| Second Half |  0.242424 |

## Section 1.4: Joint subset analysis

### Caption DP Success Regime
Criteria: viscoher_strict gain > 0.01, scene_diversity >= 0.9, temp_acc_15s gain >= 0

- **Qualifying videos:** review_6
- **Paired t-test:** N/A (less than 3 videos)

### Siglip DP Success Regime
Criteria: viscoher_strict gain > 0.01, scene_diversity >= 0.9, temp_acc_15s gain >= 0

- **Qualifying videos:** None
- **Paired t-test:** N/A (less than 3 videos)

## Section 1.5: Honest summary

The empirical analysis fails to identify a clear, statistically significant "win regime" for DP over Greedy matching. No measured video characteristic (scene count, ratio, or duration) significantly predicts DP performance gains, with all Spearman correlations showing p > 0.4. While DP differs from Greedy more frequently in the second half of summaries (24% vs 15%), this divergence rarely translates to improved strict metrics. Only one video (`review_6`) met the success criteria on the Caption track, and none on the SigLIP track. Instead, DP often achieves raw visual coherence gains by sacrificing scene diversity, confirming that its primary "advantage" in the current pipeline is a propensity for scene reuse rather than superior temporal ordering.
