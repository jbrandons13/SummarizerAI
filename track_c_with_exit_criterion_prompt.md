# TASK: Track C — Parameter Re-Tuning with Exit Criterion to Track B

## Context

Investigation revealed that some videos exhibit scene reuse looping in DP arms,
particularly review_7 Caption DP which assigned scene 4 to five consecutive
sentences. Root cause: high backward_penalty (0.5) makes backward jumps
expensive, forcing DP to stay at current scene when forward alternatives are
weaker. This interacts poorly with Caption signal, which scores high on early
scenes for late sentences.

Looping has two issues:
1. Visual quality: same-scene reuse looks like a video glitch
2. Metric inflation: VisCoher rewards same-scene transitions (cosine = 1.0),
   so VisCoher gains can be partly artifact of reuse rather than real
   cross-scene smoothing

Track C goal: re-tune parameters to eliminate or minimize looping, then re-run
cleanrun. If parameter tuning fails, fallback to Track B.

---

## Phase 1: Pilot Parameter Sweep (Exit Criterion)

Run a small sweep on 2 pilot videos: review_7 (worst looping case) and
review_2 (mid-case with some reuse). Do NOT touch other videos in this phase.

Parameters to sweep:
- dp_backward_penalty: [0.05, 0.1, 0.2, 0.3]
- Keep other parameters fixed: dp_jump_penalty=0.01, dp_reuse_bonus=0.01

Run only Caption DP and SigLIP DP arms (4 conditions × 2 videos × 4 backward
penalty values = 32 runs). Use existing embeddings cache; do not regenerate
captions or SigLIP embeddings.

For each (video, arm, backward_penalty), record:
- Scene assignment sequence (e.g., [2, 4, 4, 4, 4, 4])
- Number of unique scenes used (e.g., 2 unique out of 6 sentences)
- Reuse rate: 1 - (unique_scenes / num_sentences)
- TempAcc, VisCoher, CLIPScore on this video

Generate a sweep table. The decision criterion is:

**SUCCESS** if at least one backward_penalty value produces:
- review_7 Caption DP reuse rate ≤ 1/6 (at most one reuse out of 6 sentences)
- AND TempAcc on review_7 Caption DP does not drop more than 0.2 from baseline (0.667)
- AND review_2 SigLIP DP reuse rate stays at current baseline (≤1/6)

**FAILURE** if no value satisfies all three. In that case, exit Track C and
proceed to Phase B (Track B fallback) below.

Report sweep results in `notes/parameter_sweep_pilot.md` with the table and
the SUCCESS/FAILURE verdict.

---

## Phase 2A: Full Re-run (only if Phase 1 SUCCESS)

If Phase 1 succeeds with a chosen backward_penalty value (call it BP*):

1. Update `configs/default.yaml` with dp_backward_penalty: BP*
2. Save snapshot: `notes/config_snapshot_cleanrun_v2.yaml`
3. Run full cleanrun: 10 videos × 8 arms (use embeddings cache to skip
   Phase 4 re-embed; only re-run matching algorithm and downstream metrics)
4. Save results to `results/cleanrun_v2/ablation_results.csv`
5. Generate comparison `notes/cleanrun_v1_vs_v2_comparison.md` showing
   how aggregate metrics changed and how looping incidents changed

After full re-run, proceed to Phase 3 below (Scene Diversity computation).

---

## Phase 2B: Track B Fallback (only if Phase 1 FAILURE)

If parameter sweep cannot resolve looping with reasonable parameters,
do NOT re-tune or re-run. Instead:

1. Document in `notes/track_c_fallback.md` why Track C was abandoned and
   show the sweep table that justifies the fallback.
2. Proceed directly to Phase 3 (Scene Diversity) and Phase 4 (Modified VisCoher)
   on the EXISTING cleanrun_v1 data.

---

## Phase 3: Compute Scene Diversity (always run, regardless of 2A or 2B)

For each (video, arm) combination in the chosen results dataset (v2 if 2A,
v1 if 2B), compute:

- num_unique_scenes_used
- num_sentences
- scene_diversity = num_unique_scenes_used / num_sentences (range 0 to 1)
- max_consecutive_reuse = longest run of identical scene picks

Add these as new columns to ablation_results.csv. Generate
`notes/scene_diversity_results.md` with:
- Per-arm aggregate scene_diversity (mean across 10 videos)
- Per-video breakdown
- Identification of looping cases (consecutive_reuse ≥ 3)

---

## Phase 4: Modified Visual Coherence (always run)

Define VisCoher_strict as:
- Take consecutive matched-keyframe pairs
- Exclude pairs where both keyframes belong to the SAME scene (same scene_id)
- Compute mean cosine similarity over remaining cross-scene pairs only
- If all pairs are same-scene (degenerate case), VisCoher_strict = NaN or 0
  (decide and document the choice)

Compute VisCoher_strict for all 80 (video, arm) combinations. Add as new
column to ablation_results.csv.

Generate `notes/viscoher_strict_results.md`:
- Aggregate VisCoher vs VisCoher_strict per arm
- Identify arms where the gap is largest (these are arms most affected by reuse)
- Re-run paired t-test for VisCoher_strict on the key DP comparisons

---

## Phase 5: Update Statistical Tests

Re-run paired t-tests and Wilcoxon for all key comparisons using the new
data (v2 if 2A, v1 if 2B), plus the new metrics (scene_diversity,
viscoher_strict). Save to `notes/significance_tests_v2.md`.

Key comparisons:
- T on Caption (caption_direct vs caption_temporal)
- T on SigLIP (siglip_direct vs siglip_temporal)
- DP on Caption (caption_temporal vs caption_temporal_dp)
- DP on SigLIP (siglip_temporal vs siglip_temporal_dp)
- Caption-best vs SigLIP-best (caption_temporal_dp vs siglip_temporal_dp)

Metrics: clipscore_mean, temporal_acc_15s, visual_coherence_mean,
viscoher_strict, scene_diversity, visual_relevance.

---

## Phase 6: Updated Interpretation

Generate `notes/cleanrun_interpretation_v2.md` (or update v1 file) with:
- Headline findings using new data
- Trade-off table updated with strict VisCoher and scene_diversity
- Honest discussion of how findings changed
- Honest discussion of looping resolution (or persistence if 2B)

---

## Critical Rules

- Do NOT re-run pipeline phases other than Phase 4 matching. Embeddings,
  transcripts, summaries, TTS audio are all cached and unchanged.
- If Phase 1 fails the criterion, do not re-attempt with different parameter
  ranges. Fallback to Track B as instructed.
- Document the SUCCESS or FAILURE decision explicitly so we know which
  branch was taken.
- Be honest. If new findings are weaker than v1 findings, report so.
- Do not change anything in cleanrun_v1 files; v2 should be a separate set.

## Output Deliverables

After completion, the following files exist:

Track C SUCCESS branch:
- notes/parameter_sweep_pilot.md
- notes/config_snapshot_cleanrun_v2.yaml
- results/cleanrun_v2/ablation_results.csv (with scene_diversity and viscoher_strict)
- notes/cleanrun_v1_vs_v2_comparison.md
- notes/scene_diversity_results.md
- notes/viscoher_strict_results.md
- notes/significance_tests_v2.md
- notes/cleanrun_interpretation_v2.md

Track B FALLBACK branch:
- notes/parameter_sweep_pilot.md
- notes/track_c_fallback.md
- ablation_results.csv updated with scene_diversity and viscoher_strict columns
- notes/scene_diversity_results.md
- notes/viscoher_strict_results.md
- notes/significance_tests_v2.md
- notes/cleanrun_interpretation_v2.md (updated based on new metrics on v1 data)

Confirm completion with a brief summary of which branch was taken and
the headline findings.
