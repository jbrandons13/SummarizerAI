# GEMINI BRIEF — full overnight run (v2): 4 educational videos x 2 methods, full I2V

> **What changed from v1 (read this first):**
> - **Dataset replaced.** v1 had photosynthesis + an iPhone review. The new
>   showcase set is four **educational** videos: geology, ecology, **The Sun**,
>   **The Heart**. (Reasoning: keeps every showcase video inside the educational
>   concept-consistency niche the thesis is built on.)
> - **Metric framework is now LOCKED** (section 2.1). Do not invent a "calibrated
>   gap" metric. Content-kept = DINOv2 sim-to-own, concept = CLIP-text.
> - **Fine weight grid is mandatory** and the **per-shot fine CSV must be saved**
>   (section 4, A3/A4). This is the file the DACA per-shot figure needs; v1's run is
>   what made that file, but it was almost lost, so it is now explicit.
> - **STOP-and-report if any reused per-shot data is coarse-only** (A0).
> - **Section 9 (full-I2V vs policy) is now RESOLVED** in the thesis text, see below.
> - **Contrast control included** (section 11): one subject-centric review video,
>   **Phase A only** (no I2V render). It demonstrates the reward-collapse is
>   scope-specific to educational concept content, the main defense of C3.

## 0. Goal (read first)

Produce **8 final summary videos**: for each of **4 source videos**, render **two**
final videos, one per anchoring method.

| # | source video    | method = FIXED (w=0.2) | method = DACA (content-floor) |
|---|-----------------|------------------------|-------------------------------|
| 1 | geology (have)  | video_fixed_w02.mp4    | video_daca.mp4                |
| 2 | ecology (have)  | video_fixed_w02.mp4    | video_daca.mp4                |
| 3 | sun (new)       | video_fixed_w02.mp4    | video_daca.mp4                |
| 4 | heart (new)     | video_fixed_w02.mp4    | video_daca.mp4                |

Every shot is animated by **Wan I2V (full generative, NO Ken Burns)** via the
built-in `--all-i2v` flag. Start at **T2I** (SDXL) and go through **full I2V**.

### HONEST TIME REALITY — do not skip
Full I2V on all shots x 2 methods x 4 videos is roughly **16-20 h** of GPU time
(~16 min per I2V shot, the project's own estimate). It will **not** finish in one
night on a single 3090. The plan is built so that:
- **Phase A** (cheap, ~1-2 h) finishes overnight and secures ALL data + stills.
- **Phase B** (I2V, expensive) is **resumable**: re-running skips any shot whose
  clip already exists, so an interrupted night continues next session.

If you want a run that **fits one night**, see "Reduced option" in section 8.

---

## 1. The 4 source videos

- Video 1 — geology (CrashCourse, "where do rocks come from"). Already processed.
- Video 2 — ecology (CrashCourse, hydrologic/carbon cycles). Already processed.
- Video 3 — **The Sun** (CrashCourse Astronomy #10, NEW):
  `https://www.youtube.com/watch?v=b22HKFMIfWo`
  Recurring concept = the Sun. Expect a mix of Sun close-ups (fragile under
  anchoring) and Earth/scale-context shots (tolerate more), i.e. varied fragility.
- Video 4 — **The Heart, Part 1** (CrashCourse A&P #25, NEW):
  `https://www.youtube.com/watch?v=X9ZZ6tcxArI`
  Recurring concept = the heart. Same expectation, organ close-ups vs body-context
  shots.

All four are CrashCourse, ~10 min, narration-driven, single dominant recurring
concept. This keeps the source channel a controlled variable.

**FIRST STEP (pre-flight, section 6): confirm both new URLs resolve** with
`yt-dlp --simulate <url>` before committing the night.

---

## 2. The two methods (what differs)

Both methods differ ONLY in the still image fed to I2V (the IP-Adapter anchoring
weight per shot). The I2V step is identical, just pointed at a different image set.

- **FIXED** — every shot uses anchoring weight **w = 0.2** (the operating point).
- **DACA** — per shot, pick the **largest weight whose content-kept (sim-to-own) >= tau**,
  with **tau = 0.70** (content-floor argmax). Exactly the validated method
  (`adaptive_anchor.py`). If no weight clears tau, fall back to the smallest weight.

Both image sets come from **one weight sweep** per video (so we pay T2I once):
- FIXED set  = the sweep image at **w=0.2** for every shot.
- DACA set   = the sweep image at **w\*** for every shot (w\* from `adaptive_anchor.py`).

### 2.1 Metric framework — LOCKED (do not change)
Use exactly these three measures, all as defined for the geology/ecology runs. Do
**not** introduce a "calibrated gap" or "same minus floor" metric, that framing is
superseded.
- **content-kept** `c_s(w)` = **DINOv2** cosine similarity between shot `s` at weight
  `w` and the **same shot at w=0** (its own un-anchored scene). Falls as `w` rises.
  This is the counter-metric and the DACA controller.
- **similarity-to-reference** = **DINOv2** cosine to the canonical reference image.
  Rises with `w`. This is the **misleading** score (rewards copying the reference).
- **inter-shot similarity** = mean **DINOv2** cosine among the shots at a given `w`.
  Rises with `w` (shots converge to one image, mode collapse).
- **concept** = **CLIP image-to-text** similarity between the generated shot and the
  video's concept phrase (section 4, A5). This is the y-axis of the frontier plane.
If DINOv2 via `torch.hub` fails in your env (it has before), use the HF `AutoModel`
path. Keep the metric identical across all videos.

---

## 3. Output layout (create per video)

```
runs/<vid>/
  storyboard.json          # upstream; every shot must have an i2v_prompt field
  summary_script.json      # upstream (narration, for subtitles)
  audio/                   # upstream (Kokoro wavs per shot)
  reference.png            # the canonical concept-anchor reference image
  sweep/manifest.json      # weight_sweep output (image path per shot per weight)
  sweep/...                # the swept images
  collapse_metrics.csv     # per (shot, weight) DINOv2 sim_to_reference + sim_to_own
                           #   PLUS an aggregate means block (see A4). FINE grid.
  daca/                    # adaptive_anchor output (w* per shot + comparison)
  images_fixed_w02/        # assembled: each shot's w=0.2 image
  images_daca/             # assembled: each shot's w* image
  clips_fixed/  clips_daca/# phase5 work dirs (separate, so resume is per-method)
  video_fixed_w02.mp4      # FINAL (FIXED)
  video_daca.mp4           # FINAL (DACA)
  log_<vid>.txt
```

`<vid>` in `{geology, ecology, sun, heart}`.

---

## 4. PHASE A — cheap, run for ALL 4 videos first (secures the data)

Do this for every video before any I2V. It finishes overnight easily.

### A0. Pre-check on reused data (geology, ecology) — STOP if coarse
Before reusing the existing geology/ecology per-shot CSVs, confirm the per-shot
sweep is the **FINE** grid (0.1 steps, including w=0.0), the same grid this brief
produces in A3. If the only per-shot data you have for a reused video is the
**coarse 0.2-step** sweep, **STOP and report it** — do not mix grids and do not
plot a per-shot DACA figure from coarse data, because the chosen `w\*` would not
line up with the fine-grid values in the thesis tables. Re-run the fine sweep (A3)
for that video instead.

### A1. Upstream (ADAPT to your existing scripts)
For **videos 1 and 2**: reuse existing `storyboard.json`, `summary_script.json`,
`audio/`, and the canonical `reference.png`. Copy them into `runs/<vid>/`.

For **videos 3 and 4 (sun, heart)**: run your normal pipeline P1-P3 + storyboard
from the URL — the same steps that produced videos 1 and 2 (download -> WhisperX
transcribe -> Qwen summarize -> storyboard -> Kokoro voiceover -> pick recurring
concept + reference image via your concept-recurrence / anchor-policy step). Land
the same artifacts in `runs/<vid>/`.

### A2. I2V motion prompts for EVERY shot (needed for full I2V)
```
python generate_i2v_prompts.py --storyboard runs/<vid>/storyboard.json --in-place
```
(`--in-place` writes the `i2v_prompt` field for all shots; a `.bak` is saved.)

### A3. Weight sweep over ALL shots (this is the T2I cost) — FINE GRID
List every shot id in `--shots` (comma-separated). The grid **must** include
**0.0** (the un-anchored reference point for content-kept) and **fine 0.1 steps**
through the low range so the per-shot DACA figure is smooth and the chosen `w\*`
match the thesis tables, plus the high end for the collapse curve:
```
python weight_sweep.py \
  --config <PIPELINE_CONFIG> \
  --storyboard runs/<vid>/storyboard.json \
  --reference runs/<vid>/reference.png \
  --shots <shot_001,shot_002,...ALL> \
  --weights 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0 \
  --out runs/<vid>/sweep
```
(9 weights. If T2I time is tight, the load-bearing ones are 0.0,0.1,0.2,0.3,0.4,0.5,0.6;
0.8 and 1.0 only feed the collapse curve's tail. Do not drop below 0.1 resolution
in the 0.0-0.6 range.)

### A4. Metrics per (shot, weight) — save BOTH per-shot and means
Produce `runs/<vid>/collapse_metrics.csv` with two parts (same schema as the
geology/ecology files):
1. **Per-shot block**: one row per (shot, weight) with `sim_to_reference` and
   `sim_to_own` (DINOv2). This per-shot fine-grid block is the source for the
   per-shot DACA figure — it **must** be saved, not just summarized.
2. **Means block** (after a blank line): per weight,
   `mean_sim_to_reference, mean_sim_to_own_w0, mean_inter_shot_sim`. This feeds the
   collapse / frontier curve for the video.
```
python collapse_metrics.py --manifest runs/<vid>/sweep/manifest.json --out runs/<vid>/collapse_metrics.csv
```
(Adapt flags to your interface; the requirement is the two-part CSV above, fine grid.)

### A5. DACA selection (w* per shot)
```
python adaptive_anchor.py \
  --manifest runs/<vid>/sweep/manifest.json \
  --metrics-csv runs/<vid>/collapse_metrics.csv \
  --tau 0.70 \
  --concept "<concept phrase for this video>" \
  --out runs/<vid>/daca
```
Concept phrase per video (refine from the actual recurring concept):
- geology: `a colorful cartoon illustration of rocks, rocky terrain, boulders and stones`
- ecology: `a colorful cartoon illustration of water, the water cycle, rivers and oceans`
- sun:     `a colorful cartoon illustration of the Sun, a bright glowing star in space`
- heart:   `a colorful cartoon illustration of a human heart, the organ that pumps blood`

`adaptive_anchor.py` prints/writes **w\* per shot**. Capture it (`daca/picks.json`).

### A6. Assemble the two image sets (from the ONE sweep)
For every shot, the sweep manifest has the image path at each weight. Build:
- `images_fixed_w02/<shot>.png` = sweep image at weight **0.2**
- `images_daca/<shot>.png`      = sweep image at weight **w\*** (from A5)

A tiny assembly script reading `sweep/manifest.json` + `daca/picks.json` and copying
the right file per shot is enough. (w\* is always one of the swept weights.)

**End of Phase A: all experiment data + both still-image sets exist for all 4 videos.**
Even if Phase B never finishes, this is the part that secures the thesis numbers
(collapse curves, frontier, DACA tables, the per-shot DACA figure).

---

## 5. PHASE B — full I2V (expensive, resumable). Two renders per video.

Requires **ComfyUI running** (Wan I2V via its API at `--comfy-url`, default
`http://127.0.0.1:8188`). Start it first.

For each video, run the two renders (FIXED then DACA):
```
# FIXED
python render_summary_video.py --all-i2v \
  --storyboard runs/<vid>/storyboard.json \
  --script     runs/<vid>/summary_script.json \
  --images-dir runs/<vid>/images_fixed_w02 \
  --audio-dir  runs/<vid>/audio \
  --work       runs/<vid>/clips_fixed \
  --final      runs/<vid>/video_fixed_w02.mp4 \
  --workflow   wan_i2v_workflow.json

# DACA
python render_summary_video.py --all-i2v \
  --storyboard runs/<vid>/storyboard.json \
  --script     runs/<vid>/summary_script.json \
  --images-dir runs/<vid>/images_daca \
  --audio-dir  runs/<vid>/audio \
  --work       runs/<vid>/clips_daca \
  --final      runs/<vid>/video_daca.mp4 \
  --workflow   wan_i2v_workflow.json
```
- `--all-i2v` forces every shot to I2V (no Ken Burns).
- **Resume is automatic**: an I2V clip that already exists is reused unless you pass
  `--force`. Re-run the exact same command after any interruption and it continues.
- Separate `--work` dirs per method keep the two methods' clips from colliding.

Recommended run order (so progress is useful if the night ends early):
DACA-geology, DACA-ecology, DACA-sun, DACA-heart, then the four FIXED.
(DACA is the proposed method; the four DACA videos are the higher-value half.)

---

## 6. PRE-FLIGHT SMOKE (auto-proceed) — protects your night

Run BEFORE the marathon. Minutes, not hours.
1. **Compile**: `python -m py_compile *.py` (all touched scripts). Stop on error.
2. **URLs**: `yt-dlp --simulate <sun_url>` and `<heart_url>`. Stop if either fails.
3. **Wiring dry-run** (no GPU): on video 1, both image sets,
   `render_summary_video.py --all-i2v --only <one_shot> --dry-run ...` for
   `images_fixed_w02` and `images_daca`. Confirm it finds the still, the audio, and
   the i2v_prompt for that shot.
4. **One real I2V** (~16 min): same command WITHOUT `--dry-run`, one shot, DACA set.
   Confirm a valid clip is produced and ComfyUI responds.

If 1-4 pass, **auto-continue** to full Phase A then Phase B. If any fails, stop and
report the exact error.

---

## 7. How to run it unattended

Wrap the whole thing (smoke -> Phase A all videos -> Phase B all videos/methods) in
one driver `run_overnight.sh`, then:
```
nohup bash run_overnight.sh > run_overnight.log 2>&1 &
```
(or inside `tmux` / `screen`). **Log every step with timestamps**, and log
**peak VRAM per phase** (e.g. `nvidia-smi --query-gpu=memory.used --format=csv -l 5`
in the background, or `torch.cuda.max_memory_allocated()` printed at each phase end)
so the runtime table can be filled from real measurements, not estimates. After each
(video, method) render, append a one-line status to `run_overnight.log`.

---

## 8. Time / VRAM notes + reduced option

- Phase A: ~1-2 h total (SDXL sweeps + metrics; cheap).
- Phase B: ~16-20 h total (8 full-I2V videos). Sequential on one 3090. Resumable.
- VRAM: load models sequentially (SDXL for T2I, then ComfyUI/Wan for I2V). Do not
  hold SDXL and Wan resident at once. Unload between phases. Log the peak per phase.

**Reduced option (fits one night, ~8-10 h):** run Phase A for all 4 videos, then in
Phase B render **DACA only** for all 4 videos (skip the 4 FIXED I2V videos for now).
FIXED is still fully represented by its **stills + metrics** from Phase A; its final
video can render later. Pick this if you need finished videos by morning.

---

## 9. Thesis consistency note (RESOLVED — no action needed)

Earlier this was an open worry: full-I2V (no Ken Burns) seemed to contradict the
content-aware animation policy described in Bab 1-3. **This is now resolved in the
thesis text.** Bab 3 states the delivered videos are rendered full-I2V on purpose
for the richest motion, and the content-aware policy is the mechanism that keeps the
pipeline feasible on longer inputs, with its at-scale evaluation left to future work.
So rendering full-I2V here is consistent with what is written. Nothing to decide.

---

## 10. Report back when done (or interrupted)

- Which of the 8 finals completed (path + duration + #shots).
- **Confirm the per-shot fine-grid block + the means block exist in each
  `collapse_metrics.csv`** (these feed the per-shot DACA figure and the collapse
  curves). Flag any video where only coarse data was available.
- The DACA `picks.json` per video (w\* per shot) and how many shots DACA pushed above
  w=0.2 (the "shots vary -> DACA helps" evidence per video).
- Any shot that failed I2V (id + error), so it can be re-run with `--force`.
- **Per-phase wall-clock and peak VRAM from the logs** (real numbers for tab:runtime),
  and whether any phase peaked above 24 GB.
- Total wall-clock and where it stopped if the night ran out.

---

## 11. Contrast control (Phase A ONLY — do this)

**Run this. It is Phase A only, no I2V render, so it costs ~1 h and adds no time to
the marathon.** Stop after Phase A for this video.

Add ONE **subject-centric** video as a control to show the reward-collapse is
**scope-specific** to educational concept content. For a product/subject video the
recurring element IS a specific object, so high anchoring that copies the reference
is appropriate rather than pathological. Showing that contrast is the strongest
single defense of the C3 claim.

- Suggested: a short product review, e.g. MKBHD iPhone review
  `https://www.youtube.com/watch?v=MRtg6A1f2Ko` (verify with `yt-dlp --simulate`,
  or swap for any current single-product review).
- Run **Phase A only** for it: `<vid> = review`. Upstream (A1) + sweep (A3, same fine
  grid) + metrics (A4) + DACA selection (A5, concept phrase e.g.
  `a cartoon illustration of a smartphone`). **Do NOT run Phase B** (no showcase
  video needed; its value is the collapse curve, not a rendered video).
- Report its **collapse curve** (sim-to-reference rising, content-kept falling) and
  its DACA picks. Expected contrast: content-kept stays high even at high `w`
  (copying the one object does not destroy "content"), so DACA brakes little and
  high fixed anchoring looks fine — the opposite of the educational videos.
