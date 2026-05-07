# Per-Video Analysis

### clipscore_mean

| video_id   |   random |   caption_direct |   caption_temporal |   caption_temporal_dp |   siglip_direct |   siglip_temporal |   siglip_temporal_hungarian |   siglip_temporal_dp |
|:-----------|---------:|-----------------:|-------------------:|----------------------:|----------------:|------------------:|----------------------------:|---------------------:|
| review_1   | 0.502851 |         0.556987 |           0.568177 |              0.576356 |        0.540265 |          0.492185 |                    0.492185 |             0.492185 |
| review_10  | 0.509813 |         0.63969  |           0.617477 |              0.604708 |        0.587617 |          0.557818 |                    0.557818 |             0.581426 |
| review_2   | 0.369186 |         0.490466 |           0.51078  |              0.511417 |        0.491081 |          0.478746 |                    0.489145 |             0.480569 |
| review_3   | 0.361082 |         0.53992  |           0.512495 |              0.512495 |        0.480055 |          0.459012 |                    0.459012 |             0.459012 |
| review_4   | 0.523994 |         0.582491 |           0.539655 |              0.539655 |        0.609273 |          0.583899 |                    0.583899 |             0.583899 |
| review_5   | 0.544029 |         0.603856 |           0.605638 |              0.580349 |        0.602902 |          0.601526 |                    0.601526 |             0.57488  |
| review_6   | 0.50633  |         0.685775 |           0.661997 |              0.660938 |        0.56913  |          0.64282  |                    0.64282  |             0.649244 |
| review_7   | 0.504241 |         0.612569 |           0.569039 |              0.572936 |        0.609836 |          0.61426  |                    0.61426  |             0.641639 |
| review_8   | 0.407765 |         0.611952 |           0.626067 |              0.626067 |        0.55152  |          0.544535 |                    0.544535 |             0.544535 |
| review_9   | 0.380342 |         0.533274 |           0.533979 |              0.515025 |        0.494138 |          0.527712 |                    0.527712 |             0.527712 |

### temporal_acc_15s

| video_id   |   random |   caption_direct |   caption_temporal |   caption_temporal_dp |   siglip_direct |   siglip_temporal |   siglip_temporal_hungarian |   siglip_temporal_dp |
|:-----------|---------:|-----------------:|-------------------:|----------------------:|----------------:|------------------:|----------------------------:|---------------------:|
| review_1   | 0        |         0.25     |           0.75     |              0.75     |        0.5      |          1        |                    1        |             1        |
| review_10  | 0.333333 |         0.333333 |           0.833333 |              1        |        0.666667 |          0.833333 |                    0.833333 |             1        |
| review_2   | 0        |         0.5      |           1        |              1        |        0        |          0.666667 |                    0.833333 |             0.833333 |
| review_3   | 0.166667 |         0.166667 |           1        |              1        |        0.333333 |          1        |                    1        |             1        |
| review_4   | 0        |         0        |           0.666667 |              0.666667 |        0        |          1        |                    1        |             1        |
| review_5   | 0.285714 |         0.285714 |           0.571429 |              0.428571 |        0.571429 |          0.857143 |                    0.857143 |             0.857143 |
| review_6   | 0.2      |         0.6      |           0.8      |              0.8      |        0.2      |          1        |                    1        |             0.8      |
| review_7   | 0        |         0.166667 |           0.333333 |              0.666667 |        0        |          0.666667 |                    0.666667 |             0.333333 |
| review_8   | 0.125    |         0.125    |           0.875    |              0.875    |        0.5      |          1        |                    1        |             1        |
| review_9   | 0        |         0.25     |           0.875    |              0.875    |        0.375    |          1        |                    1        |             1        |

### visual_coherence_mean

| video_id   |   random |   caption_direct |   caption_temporal |   caption_temporal_dp |   siglip_direct |   siglip_temporal |   siglip_temporal_hungarian |   siglip_temporal_dp |
|:-----------|---------:|-----------------:|-------------------:|----------------------:|----------------:|------------------:|----------------------------:|---------------------:|
| review_1   | 0.593345 |         0.732778 |           0.738141 |              0.789093 |        0.678559 |          0.654145 |                    0.654145 |             0.654145 |
| review_10  | 0.601812 |         0.572611 |           0.586672 |              0.584251 |        0.524847 |          0.587323 |                    0.587323 |             0.564426 |
| review_2   | 0.50669  |         0.588939 |           0.68097  |              0.722264 |        0.555647 |          0.544535 |                    0.557578 |             0.662022 |
| review_3   | 0.716089 |         0.634532 |           0.519483 |              0.519483 |        0.494839 |          0.509253 |                    0.509253 |             0.509253 |
| review_4   | 0.680621 |         0.675184 |           0.650591 |              0.650591 |        0.655787 |          0.628512 |                    0.628512 |             0.628512 |
| review_5   | 0.658272 |         0.721801 |           0.742204 |              0.82817  |        0.688882 |          0.731437 |                    0.731437 |             0.735746 |
| review_6   | 0.884063 |         0.574816 |           0.550815 |              0.655216 |        0.51937  |          0.632826 |                    0.632826 |             0.625118 |
| review_7   | 0.693483 |         0.574195 |           0.731253 |              0.903162 |        0.679856 |          0.750818 |                    0.750818 |             0.716947 |
| review_8   | 0.582775 |         0.79063  |           0.770945 |              0.770945 |        0.572699 |          0.58253  |                    0.58253  |             0.58253  |
| review_9   | 0.679126 |         0.6919   |           0.651411 |              0.67221  |        0.566543 |          0.641591 |                    0.641591 |             0.641591 |

### Per-Video Winners (Excluding Random)

|    | video_id   | clipscore_mean                        | temporal_acc_15s                                                                                      | visual_coherence_mean                                        |
|---:|:-----------|:--------------------------------------|:------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------|
|  0 | review_1   | caption_temporal_dp                   | siglip_temporal, siglip_temporal_hungarian, siglip_temporal_dp                                        | caption_temporal_dp                                          |
|  1 | review_2   | caption_temporal, caption_temporal_dp | caption_temporal, caption_temporal_dp                                                                 | caption_temporal_dp                                          |
|  2 | review_3   | caption_direct                        | caption_temporal, caption_temporal_dp, siglip_temporal, siglip_temporal_hungarian, siglip_temporal_dp | caption_direct                                               |
|  3 | review_4   | siglip_direct                         | siglip_temporal, siglip_temporal_hungarian, siglip_temporal_dp                                        | caption_direct                                               |
|  4 | review_5   | caption_temporal                      | siglip_temporal, siglip_temporal_hungarian, siglip_temporal_dp                                        | caption_temporal_dp                                          |
|  5 | review_6   | caption_direct                        | siglip_temporal, siglip_temporal_hungarian                                                            | caption_temporal_dp                                          |
|  6 | review_7   | siglip_temporal_dp                    | caption_temporal_dp, siglip_temporal, siglip_temporal_hungarian                                       | caption_temporal_dp                                          |
|  7 | review_8   | caption_temporal, caption_temporal_dp | siglip_temporal, siglip_temporal_hungarian, siglip_temporal_dp                                        | caption_direct                                               |
|  8 | review_9   | caption_temporal, caption_direct      | siglip_temporal, siglip_temporal_hungarian, siglip_temporal_dp                                        | caption_direct                                               |
|  9 | review_10  | caption_direct                        | caption_temporal_dp, siglip_temporal_dp                                                               | caption_temporal, siglip_temporal, siglip_temporal_hungarian |

### Differentiation Analysis

- SigLIP DP vs SigLIP Greedy: 5/10 videos differ.
- Caption DP vs Caption Greedy: 7/10 videos differ.
- SigLIP Hungarian vs SigLIP Greedy: 1/10 videos differ.

### Outlier Case Study: review_7 SigLIP DP

(TBD: Add 1-2 paragraphs here manually after checking stats)
