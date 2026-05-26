# Brief Update: Run Phase 1-3 for Missing Videos, Then Full Pipeline 10 Videos

**Task type:** Execution (full pipeline run, 10 videos).
**Goal:** Run Phase 1-3 untuk review_2 through review_10 (intermediate outputs hilang karena user cleanup). Lalu run full pipeline (Phase 4-5 + assembler) untuk semua 10 videos.

## Context

Per pre-flight check sebelumnya:
- review_1: Phase 1-3 outputs **exist** (skip rebuild)
- review_2 through review_10: Phase 1-3 outputs **missing** (cleanup di chat sebelumnya)

User confirmed: rebuild Phase 1-3 untuk 9 videos yang missing.

## Updated execution plan

### Phase A: Restore Phase 1-3 (9 videos)

Run Phase 1-3 sequentially untuk review_2 to review_10:

```bash
for i in 2 3 4 5 6 7 8 9 10; do
  python scripts/run_pipeline.py --video review_$i --phases 1,2,3 2>&1 | tee logs/p1to3_review_$i.log
done
```

(Adjust CLI flag to match actual interface. Goal: run Phase 1-3 only, stop before Phase 4.)

**Important:**
- Verify Phase 1-3 deterministic output. Phase 1 (WhisperX) & Phase 3 (Kokoro) should be deterministic given same input & config. Phase 2 (Groq LLM) bisa vary karena LLM stochastic.
- Each video Phase 1-3: ~5-10 menit total. 9 videos: ~50-90 menit.

### Phase B: Full pipeline (Phase 4-5 + assembler) for all 10 videos

Setelah Phase A complete, run Phase 4 onward untuk semua 10 videos:

```bash
for i in 1 2 3 4 5 6 7 8 9 10; do
  python scripts/run_pipeline.py --video review_$i --method grouping_gate 2>&1 | tee logs/pipeline_review_$i.log
done
```

(For review_1: re-runs Phase 4-5+assembler dengan output sebelumnya. Should pick up cached LTX clips if exists, otherwise regenerate.)

### Phase C: Aggregate report

Same as previous brief (`brief_scale_10_videos.md` Step 4):

| video | n_groups | n_retrieve | n_generate | n_clips_generated | n_clips_failed | phase5_gen_time_s | peak_vram_gb | output_duration_s | sync_delta_s | resolution |
|---|---|---|---|---|---|---|---|---|---|---|

Plus:
- Phase A wallclock total
- Phase B wallclock total
- Grand total

## Hard rules

- **Phase A and B separate.** Don't interleave (run all Phase 1-3 first, then all Phase 4-5).
- **Sequential within each phase.** Don't parallelize.
- **Pause Ollama** before Phase B (LTX generation needs VRAM headroom).
- **Continue on crash.** Log full traceback, mark video as failed in report, continue with next video.
- **Don't tweak parameters.** Use locked config (gate_threshold 0.12, num_frames adaptive, etc).
- **Don't claim quality.** Just metrics.
- **Cache liberal.** Don't regenerate completed outputs.

## What to report back

```
## Phase A: Restore Phase 1-3
- Successfully completed: <N>/9 videos
- Failed: <list with reason>
- Wallclock: X hours Y minutes

## Phase B: Full pipeline
- Successfully completed: <N>/10 videos
- Failed: <list with reason>
- Wallclock: X hours Y minutes

## Aggregate results table
[As specified in Step 4 of previous brief]

## Aggregate stats
- Total clips generated: N
- Total clips failed (fallback): N
- Total generate groups across dataset: N
- Total retrieve groups: N
- Action distribution: X% retrieve / Y% generate

## Failures / anomalies
<list per-video>

## Total wallclock
- Phase A: X hours
- Phase B: X hours
- Grand total: X hours

## Output files
[List of 10 final mp4 paths with sizes]

## Ready for user visual review
"Please review samples: data/output/review_*/summary_grouping_gate.mp4"
```

## Anti-hallucination

- Quote actual durations & VRAM from measurement, not estimate
- If video fails to produce final mp4, mark explicitly — don't quietly skip
- If sync delta > 0.5s anywhere, flag specifically

## Crash recovery

Same as previous brief: STOP entire run, report state, wait for user. Don't auto-restart.

## Out of scope

- Re-running review_1 Phase 1-3 (already exists)
- Tweaking any parameters mid-run
- Visual quality judgment
- Evaluation metrics
- Cross-video analysis (just aggregate stats)
