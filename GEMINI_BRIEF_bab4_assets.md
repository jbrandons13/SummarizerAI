# GEMINI BRIEF — Bab 4 assets (figures + tables) [CONSOLIDATED]

**This SUPERSEDES `GEMINI_BRIEF_bab4_figures.md`.** Changes since that file:
- Block-wise was demoted, so the two Ide-1 scatters are **dropped** (do not make them).
- Claude is building the adaptive scatters (Fig 4.6, 4.7) and all schematic diagrams himself.
- Added: `daca-curve`, plus the post-run items (qual, filmstrip, VLM-judge, runtime).

All figures must MATCH the style of `collapse_curve.png` (same fonts, color palette,
axis ranges [0,1] where applicable, grid). Save PNGs at >=150 dpi to outputs.

================================================================
## PART 1 — no GPU, data already exists (do anytime)
================================================================

**1. Fig `frontier-eco`  ->  fig4.4_frontier_ecology.png**
Ecology collapse curve. From the ecology collapse CSV (same schema as geology
`collapse_metrics.csv`: weight, mean_sim_to_reference, mean_sim_to_own_w0,
mean_inter_shot_sim). Three lines over w: sim->ref (rising), content-kept (falling),
inter-shot (rising). Mark the crossover (~0.3). Same look as collapse_curve.png.

**2. Fig `daca-curve`  ->  fig3.3_daca_curve.png**  (IMPORTANT, consistency)
Per-shot content-kept selection curve for GEOLOGY. Plot content-kept c_s(w) for each
shot across the **FINE weight grid used for the adaptive selection** (the grid that
produced w* = 0.2 / 0.3 / 0.5). One curve per shot, x = w, y = content-kept, a
horizontal line at tau = 0.7, and mark each shot's chosen w* (the largest w with
c_s(w) >= tau).
- MUST use the fine-grid per-shot data, NOT the coarse 0.2-step sweep, so the marked
  w* equal 0.2 / 0.3 / 0.5 and stay consistent with Table `tab:adaptive`.
- If only the coarse 0.2-step per-shot data exists, STOP and tell us (do not silently
  use coarse data, it would contradict the table).

**3. `tab:frontier` ecology rows**
Print (or CSV) the ecology means at w = 0.2, 0.4, 0.6, 0.8 for sim->ref, content-kept,
inter-shot, so the intermediate ecology rows of the table can be filled. Geology is
already in the table; we only need ecology's middle rows.

**4. `tab:anim-rules` routing criteria**
Read `classify_animation.py` and report the actual routing rules in plain words, i.e.
which shot characteristics send a shot to Wan I2V vs to Ken Burns. Two short rows is
enough (motion-implying -> I2V; static/diagrammatic -> Ken Burns), but report what the
code actually checks.

**5. (OPTIONAL) tau-sensitivity ablation**
For tau in {0.6, 0.7, 0.8} on geology AND ecology, recompute per-shot w* from the
existing per-shot sweep CSV and the resulting (CLIP-concept, content-kept) point. Emit
`tab_tau_ablation.csv` and `fig4.8_tau_ablation.png` (operating point vs tau, both
videos). No image generation. Purpose: show tau=0.7 is not cherry-picked.

Pre-flight for Part 1: confirm each input CSV exists and has the expected columns
before plotting; if any is missing, STOP and report exactly which.

================================================================
## PART 2 — AFTER the full-I2V run finishes (uses its outputs)
================================================================

**6. Fig `qual`  ->  fig4.1_qual.png**
Qualitative grid. Pick one final summary (geology is fine). Show its generated shots in
a grid, demonstrating a recurring concept held recognizable at the operating point
w = 0.2. Use the final rendered shot images.

**7. Fig `filmstrip`  ->  fig4.2_filmstrip.png**
Source vs generated. A strip of source-video frames (top row) beside the generated
summary frames (bottom row), with the narration line as a caption under each generated
frame. One summary is enough.

**8. `tab:vlm` — VLM-as-judge scores**
Run a capable vision-language model as an automated judge (Gemini's own model is fine)
over each final summary video. Rubric, score 1 to 5 each:
  - Faithfulness to the source (does it convey the source's main content)
  - Visual coherence (is the recurring concept recognizable, is the style consistent)
  - Narration-visual alignment (do the visuals match what is being said)
Report per-video scores (geology, ecology). Keep the judge prompt in the output so it
is reproducible. We can refine the rubric later.

**9. `tab:runtime` — measured per-phase cost**
From the run logs and nvidia-smi peaks, report wall-clock time and peak VRAM per phase:
  1 WhisperX, 2 Qwen2.5-14B-AWQ, 3 Kokoro, 4 SDXL+LoRA+IP-Adapter,
  5 Wan I2V (per dynamic shot), 5 Ken Burns (per static shot).

================================================================
## OUTPUT
================================================================
Part 1 now (figures frontier-eco + daca-curve, the ecology rows, the anim-rules,
optionally tau). Part 2 after the full-I2V run (qual, filmstrip, VLM scores, runtime).
Report what was produced and what was blocked (and why).
