# TASK: Statistical Analysis & Honest Interpretation of Final Ablation (Cleanrun)

## Context

Cleanrun is complete (10 videos × 8 arms, deterministic, verified). Aggregate
results are in `results/cleanrun_aggregated_20260505.csv` (or wherever the final
CSV lives). Pre-cleanrun comparison is documented in
`notes/cleanrun_vs_pre_cleanrun_comparison.md`.

Before writing thesis Chapter 4, we need: (a) per-video breakdown, (b) paired
significance tests, (c) honest interpretation that does NOT oversell DP and
correctly characterizes Hungarian degeneracy.

Critical constraints:
- Do NOT re-run ablation
- Do NOT change config (jp=0.01, rb=0.01, sigma=30 are locked)
- Be honest. If a difference is not significant, say so plainly.

---

## Background findings already verified by user (do not re-derive these,
## just confirm consistency):

**SigLIP DP vs Greedy across 10 videos:**
- 5/10 videos identical (review_1, 3, 4, 8, 9 — all reach TempAcc=1.0,
  no headroom for DP to differ)
- 5/10 videos differ. Mixed direction:
  - DP helps: review_2 (+0.167 TA), review_10 (+0.167 TA)
  - DP hurts: review_6 (-0.200 TA), review_7 (-0.333 TA)
  - Marginal: review_5 (~0 TA, +0.004 VC)
- Mean across 10: TA -0.020, VC +0.006 → marginal effect

**Caption DP vs Greedy across 10 videos:**
- VisCoher: DP helps in 7/10, neutral in 2/10, hurts marginally in 1/10
  (review_10, -0.002). Consistent positive direction.
- TempAcc: DP helps in 2/10 (review_7 +0.333, review_10 +0.167),
  hurts in 1/10 (review_5 -0.143), neutral in rest.
- Mean: TA +0.036, VC +0.047

**Hungarian vs Greedy (SigLIP track):**
- 9/10 videos IDENTICAL (not "small differences")
- Only review_2 differs: Hungarian +0.167 TempAcc, +0.013 VC
- Aggregate selisih 0.017 in TempAcc comes from this single video.
- This is consistent with the hypothesized degeneracy: when scenes >> sentences
  (typical case in our dataset), bipartite optimum coincides with Greedy.

---

## Task 1: Per-Video Breakdown

Generate two files:

**`notes/per_video_results.csv`** with columns:
- video_id, arm, clipscore_mean, temporal_acc_15s, visual_coherence_mean,
  visual_relevance, information_retention, factual_faithfulness

**`notes/per_video_analysis.md`** containing:

1. Per-arm × per-video table (8 arms as columns, 10 videos as rows) for
   each of the three primary metrics (CLIPScore, TempAcc, VisCoher).

2. Per-video winners table (already partially computed by user):
   - For each video, identify the best-performing arm per metric
     (excluding `random`).
   - Note ties (within 0.001 of max).

3. Differentiation analysis:
   - Count videos where SigLIP DP differs from SigLIP Greedy
     (any metric, threshold 0.001). Expected: 5/10.
   - Count videos where Caption DP differs from Caption Greedy.
     Expected: 8/10 (only review_3 and review_4 fully identical based on
     preliminary check).
   - Count videos where SigLIP Hungarian differs from SigLIP Greedy.
     Expected: 1/10.

4. **Outlier case study: review_7 SigLIP DP**
   - SigLIP Greedy TempAcc = 0.667, SigLIP DP TempAcc = 0.333 (drops by 0.333).
   - But Caption DP on the same video: TempAcc 0.333 → 0.667 (improves by 0.333).
   - Hypothesize: what is different about review_7 (Xiaomi 17 Pro Max,
     Mrwhosetheboss, 12.93 min) that causes opposite DP behavior on the
     two signal tracks?
   - Look at: number of scenes detected, number of sentences, scene-to-sentence
     ratio, embedding signal quality if accessible.
   - Output: 1-2 paragraphs in per_video_analysis.md.

---

## Task 2: Paired Significance Tests

Use scipy.stats.ttest_rel and scipy.stats.wilcoxon (paired, n=10 — Wilcoxon is
more conservative for small n and non-normal distributions; report both).

```python
from scipy.stats import ttest_rel, wilcoxon

comparisons = [
    # Effect of temporal prior
    ("caption_direct", "caption_temporal", "T on Caption"),
    ("siglip_direct",  "siglip_temporal",  "T on SigLIP"),
    # Effect of DP
    ("caption_temporal", "caption_temporal_dp", "DP on Caption"),
    ("siglip_temporal",  "siglip_temporal_dp",  "DP on SigLIP"),
    # Effect of Hungarian
    ("siglip_temporal", "siglip_temporal_hungarian", "Hungarian on SigLIP"),
    # Cross-track comparison at best-of-each
    ("caption_temporal_dp", "siglip_temporal_dp", "Caption-best vs SigLIP-best"),
    ("caption_temporal_dp", "siglip_temporal",    "Caption-best vs SigLIP-best (greedy)"),
]

metrics = ["clipscore_mean", "temporal_acc_15s", "visual_coherence_mean",
           "visual_relevance", "information_retention", "factual_faithfulness"]
```

For each (comparison, metric):
- Compute mean_a, mean_b, mean_diff
- Compute t-statistic, p-value (paired t-test)
- Compute Wilcoxon W, p-value
- Report effect size: Cohen's d for paired samples
- Mark significance: ** p<0.01, * p<0.05, (.) p<0.10, ns otherwise

Save as `notes/significance_tests.md` with one section per comparison. Use a
clean table format. Do NOT bold p-values that are not significant.

**Important caveat to include in the document:**
"With n=10 videos, statistical power is limited. Effects with p>0.05 may
still be real but undetectable at this sample size. Conversely, with 6
comparisons × 6 metrics = 36 tests, expect ~2 false positives at α=0.05
without correction. Bonferroni-corrected α = 0.0014. We report uncorrected
p-values for transparency but interpret cautiously."

---

## Task 3: Re-write Interpretation

Update `notes/two_track_ablation_results.md`. The structure should be:

### Section 1: Headline findings (3-5 bullets)
- Random baseline confirms metrics are not trivially saturated
  (random TempAcc=0.111, vs best ~0.92).
- Temporal prior is the dominant factor: Caption 0.27→0.77 TA,
  SigLIP 0.31→0.90 TA. Effect size massive on both tracks.
- DP provides meaningful, consistent improvement on Caption track
  (VisCoher +0.047, helps 7/10 videos).
- DP has marginal/mixed effect on SigLIP track. SigLIP+temporal already
  achieves near-ceiling TempAcc (5/10 at 1.0), leaving no headroom.
- Hungarian is degenerate as predicted: 9/10 identical to Greedy.
  Document as expected finding, not anomaly.

### Section 2: Trade-off characterization
Caption vs SigLIP — present as complementary, not as winner/loser:
- Caption track: higher CLIPScore (+0.014), higher VisCoher (+0.077),
  but lower TempAcc (-0.076).
- SigLIP track: higher TempAcc, but lower semantic match and lower
  visual coherence.
- Per-video: VisCoher winners are 9/10 Caption-track, TempAcc winners
  are 8/10 SigLIP-track. Almost no overlap.
- Implication: signal choice should depend on application priorities.
  For documentary-style summaries where temporal precision matters most,
  SigLIP. For semantic-coherence summaries where the visual story should
  flow smoothly, Caption + DP.

### Section 3: When does DP help?
Hypothesis grounded in observation, not over-claimed:
- DP helps when underlying signal produces noisy or temporally-
  discontinuous matches that benefit from sequential smoothing.
- Caption signal (Qwen-VL → MiniLM, text-text matching) is noisier
  on a per-sentence basis, which is why DP helps it more.
- SigLIP signal already produces sequentially-coherent paths under
  the Gaussian temporal prior, so DP's regularization adds little.
- This is consistent with — but not proven by — n=10 evidence.
  Frame as "we hypothesize" and "consistent with the intuition that".

### Section 4: Hungarian degeneracy (honest framing)
- Hungarian is mathematically expected to coincide with Greedy when
  scenes >> sentences (our setting: 12-25× more scenes than sentences).
- 9/10 videos identical confirms this. The single exception (review_2)
  shows Hungarian's bipartite optimum can diverge from Greedy when
  scene density is unusual, but this is the exception not the rule.
- Document as methodological observation: bipartite assignment is not
  the right tool when one side has abundant supply. DP/sequence
  alignment is the more appropriate inductive bias.

### Section 5: LLM-as-Judge corroboration
- LLM judge (Qwen 2.5) on 1-5 scale was previously degenerate (always 3).
  Now produces variable scores (random=1.1, best arms=2.6-3.0).
- Visual relevance: caption_direct=3.0, caption_temporal=2.8,
  caption_temporal_dp=2.6 — slight DECREASE despite VisCoher INCREASE.
  Hypothesis: VisCoher measures consecutive-frame similarity (visual
  smoothness), LLM judge measures narration-frame relevance (semantic
  match). DP optimizes the former at slight cost to the latter.
  This is a legitimate trade-off worth flagging.
- Information retention and factual faithfulness are roughly constant
  across arms (3.3-3.4 IR, 2.6-2.8 FF). This is expected — these
  metrics depend on Phase 2 (summarization) and Phase 3 (TTS), not
  Phase 4 (visual retrieval). Confirms metric isolation.

### Section 6: Limitations to acknowledge
- n=10 limits statistical power
- Single domain (gadget reviews from 5 channels)
- Single dataset, no cross-dataset generalization claimed
- DP tuned on review_1; though verified on review_2, review_3 in Task 3,
  some leakage risk remains. Honestly note this.
- Hungarian arm not pursued further given degeneracy

### Critical writing rules
- Don't claim "winner" between Caption and SigLIP — they're trade-offs
- Don't oversell DP — meaningful on Caption, marginal on SigLIP, say so
- Don't undersell Hungarian degeneracy — 9/10 identical is the finding,
  not "small differences"
- When statistical tests are non-significant, write "we did not detect a
  significant difference (p=X.XX)" — never "no difference"
- Include effect sizes, not just p-values

---

## Output Deliverables
1. `notes/per_video_results.csv`
2. `notes/per_video_analysis.md`
3. `notes/significance_tests.md`
4. Updated `notes/two_track_ablation_results.md` (or new file
   `notes/cleanrun_interpretation.md` — your choice, but make sure
   the canonical file is clearly marked)

Confirm completion with a brief summary of what changed in interpretation
versus what was originally drafted pre-cleanrun.
