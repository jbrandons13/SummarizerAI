# Scene Diversity Results

## Per-Arm Aggregate Scene Diversity

| Arm                       |   Mean Scene Diversity |
|:--------------------------|-----------------------:|
| caption_direct            |               1        |
| caption_temporal          |               1        |
| caption_temporal_dp       |               0.84881  |
| random                    |               1        |
| siglip_direct             |               1        |
| siglip_temporal           |               1        |
| siglip_temporal_dp        |               0.949048 |
| siglip_temporal_hungarian |               1        |

## Looping Cases (Max Consecutive Reuse >= 3)

| video_id   | arm                 |   max_consecutive_reuse |   scene_diversity |
|:-----------|:--------------------|------------------------:|------------------:|
| review_5   | caption_temporal_dp |                       3 |          0.571429 |
| review_7   | caption_temporal_dp |                       5 |          0.333333 |

## Per-Video Breakdown

| video_id   |   caption_direct |   caption_temporal |   caption_temporal_dp |   random |   siglip_direct |   siglip_temporal |   siglip_temporal_dp |   siglip_temporal_hungarian |
|:-----------|-----------------:|-------------------:|----------------------:|---------:|----------------:|------------------:|---------------------:|----------------------------:|
| review_1   |                1 |                  1 |              0.875    |        1 |               1 |                 1 |             1        |                           1 |
| review_10  |                1 |                  1 |              1        |        1 |               1 |                 1 |             1        |                           1 |
| review_2   |                1 |                  1 |              0.833333 |        1 |               1 |                 1 |             0.833333 |                           1 |
| review_3   |                1 |                  1 |              1        |        1 |               1 |                 1 |             1        |                           1 |
| review_4   |                1 |                  1 |              1        |        1 |               1 |                 1 |             1        |                           1 |
| review_5   |                1 |                  1 |              0.571429 |        1 |               1 |                 1 |             0.857143 |                           1 |
| review_6   |                1 |                  1 |              1        |        1 |               1 |                 1 |             0.8      |                           1 |
| review_7   |                1 |                  1 |              0.333333 |        1 |               1 |                 1 |             1        |                           1 |
| review_8   |                1 |                  1 |              1        |        1 |               1 |                 1 |             1        |                           1 |
| review_9   |                1 |                  1 |              0.875    |        1 |               1 |                 1 |             1        |                           1 |

