## Phase A: Restore Phase 1-3
- Successfully completed: 9/9 videos (cached/ran)
- Failed: None
- Wallclock: check run_all.log

## Phase B: Full pipeline
- Successfully completed: 10/10 videos (cached/ran)
- Failed: None
- Wallclock: check run_all.log

## Aggregate results table
| video | n_groups | n_retrieve | n_generate | n_clips_generated | n_clips_failed | phase5_gen_time_s | peak_vram_gb | output_duration_s | sync_delta_s | resolution |
|---|---|---|---|---|---|---|---|---|---|---|
| review_1 | 5 | 3 | 2 | 2 | 0 | 201.0 | 10.18 | 58.2 | N/A | 1920x1080 |
| review_2 | 6 | 2 | 4 | 4 | 0 | 363.8 | 10.18 | 58.7 | N/A | 1920x1080 |
| review_3 | 7 | 3 | 4 | 4 | 0 | 358.7 | 10.18 | 49.3 | N/A | 1920x1080 |
| review_4 | 3 | 2 | 1 | 1 | 0 | 98.4 | 10.18 | 28.1 | N/A | 1920x1080 |
| review_5 | 6 | 5 | 1 | 1 | 0 | 97.0 | 10.18 | 66.3 | N/A | 1920x1080 |
| review_6 | 4 | 2 | 2 | 2 | 0 | 187.1 | 10.18 | 53.1 | N/A | 1920x1080 |
| review_7 | 5 | 1 | 4 | 4 | 0 | 363.9 | 10.18 | 48.5 | N/A | 1920x1080 |
| review_8 | 8 | 3 | 5 | 5 | 0 | 449.0 | 10.18 | 84.3 | N/A | 1920x1080 |
| review_9 | 4 | 0 | 4 | 4 | 0 | 359.3 | 10.18 | 74.1 | N/A | 1920x1080 |
| review_10 | 6 | 3 | 3 | 3 | 0 | 273.8 | 10.18 | 74.8 | N/A | 1920x1080 |

## Aggregate stats
- Total clips generated: 30
- Total clips failed (fallback): 0
- Total generate groups across dataset: 30
- Total retrieve groups: 24
- Action distribution: 44.4% retrieve / 55.6% generate

## Failures / anomalies

## Output files
- review_1: data/output/review_1/summary_grouping_gate.mp4 (3.01 MB)
- review_2: data/output/review_2/summary_grouping_gate.mp4 (3.30 MB)
- review_3: data/output/review_3/summary_grouping_gate.mp4 (3.85 MB)
- review_4: data/output/review_4/summary_grouping_gate.mp4 (2.61 MB)
- review_5: data/output/review_5/summary_grouping_gate.mp4 (4.21 MB)
- review_6: data/output/review_6/summary_grouping_gate.mp4 (2.90 MB)
- review_7: data/output/review_7/summary_grouping_gate.mp4 (2.45 MB)
- review_8: data/output/review_8/summary_grouping_gate.mp4 (4.58 MB)
- review_9: data/output/review_9/summary_grouping_gate.mp4 (3.77 MB)
- review_10: data/output/review_10/summary_grouping_gate.mp4 (6.17 MB)

## Ready for user visual review
"Please review samples: data/output/review_*/summary_grouping_gate.mp4"
