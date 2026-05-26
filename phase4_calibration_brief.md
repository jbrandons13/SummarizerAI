# Phase 4 Calibration — Aggregate Stats Across All Videos

## Goal

Run the current Phase 4 (with Gaussian temporal prior) on all available evaluation videos and produce aggregate statistics. The output of this brief feeds threshold and grouping parameter tuning decisions. No code changes.

## Rules of engagement

- Do not modify any code in `src/` or `configs/`.
- Do not commit anything.
- Print `=== TASK N DONE ===` after each task.
- If a video fails, skip it and continue. Report failures at the end.

---

## Task 1: Identify the video list

Find all evaluation videos. Likely path: `data/eval_videos/`. List every `.mp4` (or video file) under that directory.

Report the list and count. Target: 10 videos. If fewer or more, report the actual count.

---

## Task 2: Run the pipeline on each video and collect per-video stats

For every video in the list from Task 1, run the full pipeline. For each video, capture:

- **Phase 2 sentence count** (from the Phase 2 output JSON for that video)
- **Phase 4 assignments and summary** (from the smoke-test print block that should already be in place from the previous task)

If the pipeline is already cached for some videos (no need to re-run preprocessing), use the cache; only re-run Phase 2 + Phase 4 if needed.

Save the per-video results to memory (or a temporary file) for aggregation in Task 3.

If a video fails, log the video id and the error, then skip.

---

## Task 3: Aggregate report

Produce a single aggregate report across all successful videos. Do NOT paste per-video full assignment lists; that is too long. Aggregate only.

### What to compute

For each video, you should already have a `summarise_assignments(...)` dict. Aggregate across all videos:

```
For each numeric field in the summary dict:
  - report min across videos
  - report max across videos
  - report mean across videos

For weighted_sim, raw_cosine, temporal_weight:
  - additionally report a global histogram: count of values in bins [0.00-0.05, 0.05-0.10, 0.10-0.15, 0.15-0.20, 0.20-0.25, 0.25+]
  - aggregate over ALL assignments across ALL videos, not per video

For action distribution:
  - report total retrieve count and total generate count across all videos
  - report retrieve fraction = retrieve / (retrieve + generate)

For group size distribution:
  - count of singleton (size=1) across all assignments in all videos
  - count of size=2 across all videos
  - count of size=3+ across all videos
  - max group size observed
```

### Report format

```
=== AGGREGATE PHASE 4 STATS ===

Videos processed: <n successful> / <n total>
Failed videos: <list, or "none">

Per-video Phase 2 sentence counts:
  min: <n>
  max: <n>
  mean: <n>
  total sentences across all videos: <n>

Per-video group counts:
  min: <n>
  max: <n>
  mean: <n>
  total groups across all videos: <n>

Group size distribution (across all assignments in all videos):
  singletons (size=1): <n> (<pct>%)
  size=2: <n> (<pct>%)
  size=3: <n> (<pct>%)
  size=4: <n> (<pct>%)
  size=5: <n> (<pct>%)
  max size observed: <n>

Weighted similarity (cosine * temporal_weight) distribution:
  min: <x>
  max: <x>
  mean: <x>
  histogram:
    [0.00, 0.05): <n>
    [0.05, 0.10): <n>
    [0.10, 0.15): <n>
    [0.15, 0.20): <n>
    [0.20, 0.25): <n>
    [0.25, +inf): <n>

Raw cosine distribution:
  min: <x>
  max: <x>
  mean: <x>
  histogram:
    [0.00, 0.05): <n>
    [0.05, 0.10): <n>
    [0.10, 0.15): <n>
    [0.15, 0.20): <n>
    [0.20, 0.25): <n>
    [0.25, +inf): <n>

Temporal weight distribution:
  min: <x>
  max: <x>
  mean: <x>
  histogram:
    [0.00, 0.20): <n>
    [0.20, 0.40): <n>
    [0.40, 0.60): <n>
    [0.60, 0.80): <n>
    [0.80, 1.00]: <n>

Action distribution (at threshold 0.13):
  retrieve total: <n>
  generate total: <n>
  retrieve fraction: <pct>%

Hypothetical thresholds (count what would be retrieve if threshold changed):
  threshold = 0.08: retrieve count = <n> (<pct>%)
  threshold = 0.10: retrieve count = <n> (<pct>%)
  threshold = 0.12: retrieve count = <n> (<pct>%)
  threshold = 0.13: retrieve count = <n> (<pct>%)  (current)
  threshold = 0.15: retrieve count = <n> (<pct>%)

Blockers: <list, or "none">
```

## Hard constraints

- Do not modify code or config.
- Do not commit.
- Do not paste per-video full assignment lists. Aggregate only.
- If preprocessing cache exists, reuse it; only re-run Phase 2 + Phase 4 when needed.

End of brief.
