# TASK: Verification & Revision of Cleanrun Interpretation

## Context

The cleanrun statistical analysis is mostly accurate, but a manual review by
the user identified 3 issues that need verification or revision before this
document is used for Thesis Chapter 4. Do NOT re-run any pipeline. Do NOT
re-compute statistics already done. Only verify what is asked, then revise
the document accordingly.

---

## Issue 1: Verify review_7 scene/sentence ratio

The current `notes/per_video_analysis.md` states:

> "review_7 has an unusually high scene-to-sentence ratio of 78:1
> (468 scenes for 6 sentences), compared to the average ~24:1."

This claim cannot be verified from `ablation_results.csv` alone. Please verify
from intermediate data:

1. Open `data/intermediate/review_7/scenes.json` (or whatever the canonical
   scene-list filename is — could be `keyframes.json`, `scene_list.json`, etc.)
   and count the number of scenes/keyframes.
2. Open the SummaryScript output for review_7 (likely
   `data/intermediate/review_7/summary_script.json` or similar) and count
   the number of sentences.
3. Compute the actual ratio.
4. Repeat for all 10 videos to get the actual average ratio (the doc claims
   ~24:1).

**Output:** Add a small table to `notes/per_video_analysis.md` like:

| video_id | n_scenes | n_sentences | ratio |
|----------|---------:|------------:|------:|
| review_1 | ... | ... | ... |
| ... | | | |

Then:
- If 78:1 and ~24:1 are correct → keep the case study as-is
- If numbers are wrong → correct the numbers in the case study, but keep
  the qualitative argument if review_7 still has a noticeably higher ratio
- If review_7 is NOT an outlier in scene/sentence ratio → remove that
  causal claim and replace with: "We do not have a definitive explanation
  for review_7's anomalous DP behavior. Hypotheses include: scene density,
  embedding signal quality, or interaction with hallucinated summary content.
  Further investigation is left for future work."

## Issue 2: Soften the "DP found distractors" claim

The case study currently asserts:

> "the DP optimizer... finds 'distractor' scenes elsewhere in the video that
> look more like the hallucinated descriptions"

This is post-hoc reasoning without direct evidence. Without manually inspecting
the actual scenes selected by DP for review_7, we cannot claim DP is matching
hallucinations specifically. Please revise to:

> "We hypothesize that DP, by optimizing for global similarity and smoothness,
> may have selected temporally distant scenes that locally maximize the
> sequence-level objective at the cost of temporal alignment. We do not
> verify this claim by inspecting individual scene matches; this is left
> for future qualitative analysis."

Use this softer framing or equivalent. Keep the observation that VisCoher
DROPS for SigLIP DP on review_7 (0.751 → 0.717) — that's a fact, not a
hypothesis.

## Issue 3: Revise the trade-off table for statistical honesty

Current table in `notes/cleanrun_interpretation.md` Section 2:

| Feature | Caption Track (Best) | SigLIP Track (Best) |
| :--- | :--- | :--- |
| **CLIPScore** | Higher (0.570) | Lower (0.554) |
| **Visual Coherence** | **Significantly Higher (0.710)** | Lower (0.632) |
| **Temporal Accuracy** | Lower (0.806) | **Higher (0.882)** |

Two problems:

**Problem 3a — "Best Caption" for CLIPScore is wrong arm.**
The table uses caption_temporal_dp (0.570) as "Caption best." But for
CLIPScore, caption_direct (0.586) is actually the highest in the Caption track.
This is misleading. Either:
- (a) Pick "best arm per metric per track" honestly, OR
- (b) Pick a single "best arm per track" (justified by composite criterion)
  and stick with it consistently.

Recommend option (b): use caption_temporal_dp and siglip_temporal_dp as the
"best comparable" arms (both with full pipeline: temporal + DP). Then add
a note explaining that in the Caption track, removing DP (caption_temporal)
or removing temporal prior (caption_direct) achieves slightly higher CLIPScore
but at the cost of TempAcc and VisCoher. This is an honest trade-off
within the Caption track itself.

**Problem 3b — Significance markers are inconsistent.**
According to `significance_tests.md` for `caption_temporal_dp vs siglip_temporal_dp`:
- CLIPScore diff: -0.016, p=0.323 → NOT significant
- TempAcc diff: +0.076, p=0.319 → NOT significant
- VisCoher diff: -0.078, p=0.006, d=-1.12 → SIGNIFICANT

So only VisCoher has statistical support. The TempAcc and CLIPScore
differences are real in mean but not statistically detectable at n=10.

Revised table should look like:

| Feature | Caption + DP | SigLIP + DP | Difference |
| :--- | ---: | ---: | :--- |
| CLIPScore | 0.570 | 0.554 | +0.016 (n.s., p=0.32) |
| Visual Coherence | **0.710** | 0.632 | **+0.078 (p=0.006, d=1.12)** |
| Temporal Accuracy | 0.806 | 0.882 | -0.076 (n.s., p=0.32) |

Then revise the surrounding paragraph to say:

> "The two signal tracks differ significantly in Visual Coherence, with
> Caption + DP producing more visually coherent narration paths. Differences
> in CLIPScore and Temporal Accuracy are not statistically significant at
> n=10, although the directional pattern (SigLIP slightly favors temporal
> alignment, Caption slightly favors semantic match) is consistent with
> the per-video winners distribution. With a larger sample, these
> directional trends may reach significance, but we do not claim them as
> established findings."

## Issue 4 (additional, please add): DP's CLIPScore cost in Caption track

Add a new short subsection to `cleanrun_interpretation.md` Section 3
("When does DP help?"), covering this internal Caption-track trade-off:

Within the Caption track, CLIPScore decreases monotonically as more
constraints are added:
- caption_direct: 0.586
- caption_temporal: 0.575 (-0.011)
- caption_temporal_dp: 0.570 (-0.005)

The differences are not statistically significant individually, but the
direction is consistent: adding temporal prior and DP both slightly reduce
the per-frame semantic match score, while substantially improving temporal
alignment and (for DP) visual coherence. This is a within-method trade-off
worth flagging: DP optimizes sequence-level coherence at marginal cost
to per-frame semantic match.

## Output Deliverables

1. Updated `notes/per_video_analysis.md`:
   - Scene/sentence ratio table for all 10 videos
   - Corrected/softened review_7 case study
2. Updated `notes/cleanrun_interpretation.md`:
   - Revised trade-off table (Issue 3)
   - Revised paragraph for statistical honesty (Issue 3)
   - New subsection on DP's CLIPScore cost (Issue 4)
3. A brief change log file `notes/interpretation_revisions.md` summarizing
   what changed and why.

## Critical Rules

- Do NOT re-run any pipeline or recompute existing statistics.
- Do NOT change the headline finding (DP-on-Caption VisCoher is significant).
- Be honest. If review_7 is NOT actually an outlier in scene density,
  say so and remove the causal claim.
- Preserve the Hungarian degeneracy framing — that's correct.
- Do NOT add new metrics or new comparisons.

Confirm completion with a one-paragraph summary of what changed.
