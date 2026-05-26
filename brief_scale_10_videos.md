# Brief: Scale Phase 5 LTX Integration to All 10 Videos

**Task type:** Execution (full dataset run + monitoring + per-video verification).
**Goal:** Run full pipeline (Phase 1-5 + assembler) di semua 10 evaluation videos. Confirm pipeline robust, capture metrics, identify per-video issues.

## Context

Phase 5 LTX integration verified working pada `review_1`:
- ✅ Bug freeze 50 detik fixed (resolution unification + hybrid duration handling)
- ✅ User visual review: hampir tidak bisa membedakan generated vs retrieved
- ✅ Known limitation: text/UI elements illegible (expected video diffusion limitation)

Sekarang scale ke `review_1` through `review_10` untuk dapat full dataset output.

## Execution plan

### Step 1: Pre-flight check

Verify per-video Phase 1-3 outputs sudah ada (lo pernah scale Phase 1-3 sebelumnya):
```bash
for i in {1..10}; do
  echo "=== review_$i ==="
  ls data/intermediate/review_$i/audio/ 2>/dev/null | wc -l
  ls data/intermediate/review_$i/summary_script.json 2>/dev/null && echo "phase2 OK" || echo "phase2 MISSING"
  ls data/intermediate/review_$i/keyframes_manifest.json 2>/dev/null && echo "phase4 prep OK" || echo "phase4 prep MISSING"
done
```

Kalau ada video yang missing Phase 1-3 output, STOP dan report. Don't auto-run Phase 1-3.

### Step 2: Sequential pipeline run

Run pipeline per video, sequential (NOT parallel — VRAM constraint).

For each video review_1 to review_10:

```bash
python scripts/run_pipeline.py --video review_N --method grouping_gate 2>&1 | tee logs/pipeline_review_N.log
```

(Adjust command to match actual CLI of `scripts/run_pipeline.py`).

**Important:**
- Use cached LTX clips if exist (resume support).
- Use cached prompts.json if exist.
- If you've never run Phase 4-5 on a video, full pipeline runs from Phase 4 onward.

### Step 3: Monitor per-video

After each video completes (sequential), capture:

1. **Phase 4 stats** dari `p4_assignments.json`:
   - Total groups
   - N retrieve / N generate
   - Group sizes distribution

2. **Phase 5 (LTX gen) stats** dari generation_metrics.json atau log:
   - N clips generated
   - N clips failed (if any)
   - Total LTX generation time
   - Peak VRAM
   - Offload mode used (model/sequential)

3. **Final output stats** via ffprobe:
   - Output mp4 path
   - Duration (video stream + audio stream)
   - Sync delta
   - Resolution (should match each source's native resolution)

4. **Error log**: any warnings, exceptions, fallbacks

### Step 4: Aggregate report

After all 10 videos done, compile summary table:

| video | n_groups | n_retrieve | n_generate | n_clips_generated | n_clips_failed | total_time_s | peak_vram_gb | output_duration_s | sync_delta_s | resolution |
|---|---|---|---|---|---|---|---|---|---|---|

Plus aggregate stats:
- Total wallclock time across all 10 videos
- Total clips generated
- Total fallback-to-retrieve count (clips failed → assembler fell back)
- Distribution of generate/retrieve actions across dataset

## Hard rules

- **Sequential, not parallel.** VRAM tight.
- **Pause Ollama** before each pipeline run (per Phase 5 LTX brief — kalau orchestration udah implemented, ini automatic).
- **Don't skip any video** without flagging. If a video crashes, log full traceback, then continue with next video. Don't abort entire run.
- **Don't tweak prompts, num_frames, gate_threshold, atau parameter lain.** Use exactly the locked config.
- **Don't claim quality** ("looks great", "high-fidelity"). Cuma report metrics. User akan visual review sample outputs sendiri.
- **Cache liberal.** Don't regenerate Phase 1-3 or existing LTX clips unless explicit flag.

## Crash recovery

Kalau pipeline crashes mid-run (misal video 5):
1. STOP, don't auto-restart
2. Report:
   - Which video crashed
   - Full traceback
   - State of intermediate outputs (mana yang berhasil saved)
   - Whether Ollama is still paused (SIGCONT manually if needed)
3. Wait for user decision (fix-then-resume, atau skip-this-video-continue)

## What to report back

```
## Pre-flight
- All videos have Phase 1-3 outputs: yes/no
- Missing: <list if any>

## Run progression
[Real-time-ish updates per video — paste log summary]

## Aggregate results table
[As specified in Step 4]

## Failures / anomalies
- Per-video issues encountered
- Fallbacks triggered
- VRAM peaks vs constraint

## Total run time
- Wallclock: X hours Y minutes
- Per-video average: X minutes
- Bottleneck phases (Qwen-VL vs LTX vs assembly)

## Output files
[List of 10 final mp4 paths]

## Ready for user visual review
"Please review: data/output/review_1/... through review_10/..."
```

## Anti-hallucination

- All durations from ffprobe verbatim
- VRAM from `torch.cuda.max_memory_allocated()` or pipeline metrics, not estimate
- If a video fails to produce final output, mark explicitly in table (don't quietly skip)
- If sync delta > 0.5s in any video, flag specifically — it's a regression

## Out of scope

- Re-running review_1 (already done)
- Tweaking any parameters
- Cross-video comparison analysis (just per-video stats)
- Visual quality judgment
- Evaluation metrics (CLIPScore, LLM-judge) — separate phase
