# TASK: Option B - Subset Analysis & Theoretical Formalization for DP

## Context

The cleanrun_v2 findings show:
- DP-on-Caption raw VisCoher gain (+0.047) is largely artifact of same-scene reuse
- DP-on-Caption VisCoher_strict gain is non-significant (p=0.52)
- DP-on-SigLIP gain is marginal at the aggregate level

Current state of DP as a contribution: documents failure modes but does not show
clear win conditions. Goal of Option B: identify (a) empirical conditions where
DP outperforms Greedy in our existing data, and (b) a theoretical characterization
of when DP should outperform Greedy in general.

This task does NOT require re-running any pipeline. All analysis is on existing
cleanrun_v1 data (with the v2 metrics scene_diversity and viscoher_strict already
computed).

---

## Part 1: Empirical Subset Analysis

Generate `notes/dp_subset_analysis.md`.

### Section 1.1: Per-video DP vs Greedy comparison

For each of 10 videos, compute the difference (DP minus Greedy) for these metrics:
- TempAcc (15s)
- VisCoher (raw)
- VisCoher_strict
- Scene Diversity
- CLIPScore
- LLM Visual Relevance

Do this separately for Caption track and SigLIP track. Output two tables:

Table 1.1a: Caption DP vs Caption Greedy per video (10 rows × 6 metric diffs)
Table 1.1b: SigLIP DP vs SigLIP Greedy per video (10 rows × 6 metric diffs)

After the tables, a short summary:
- How many videos show DP improvement on each metric (count of positive diffs)?
- Which videos show consistent DP improvement across metrics?
- Which videos show DP regression?

### Section 1.2: Video characteristic correlation

For each of 10 videos, compute:
- num_scenes (from intermediate scene data)
- num_sentences (from summary script)
- scene_to_sentence_ratio = num_scenes / num_sentences
- avg_scene_duration = source_video_duration / num_scenes
- summary_target_duration = 90 seconds (constant for all)

Then correlate these characteristics with DP gain (DP minus Greedy) for the
viscoher_strict metric on Caption and SigLIP tracks.

Output: a table showing each video with its characteristics and its DP_strict_gain
on each track. Compute Spearman correlation between characteristics and gain.

Question to answer: is there a video characteristic that predicts when DP helps?

### Section 1.3: Sentence-position analysis

For each (video, arm) pair where DP and Greedy differ in assignment, identify
which sentence positions (first half vs second half of summary) have the most
disagreement. Aggregate across videos.

Question to answer: does DP tend to differ from Greedy more at certain sentence
positions (e.g., later sentences where forward progression matters more)?

### Section 1.4: Joint subset analysis

Define a "DP success regime" as videos where ALL of:
- viscoher_strict (DP - Greedy) > 0.01
- scene_diversity (DP) >= 0.9
- temp_acc_15s (DP - Greedy) >= 0

Count how many of 10 videos meet this on each track. If at least 3 videos meet
this on either track, run a paired t-test on viscoher_strict over just that
subset and report whether DP advantage is significant on this regime.

Output the list of qualifying videos and the test result.

### Section 1.5: Honest summary

Write a 3-4 sentence summary at the end of dp_subset_analysis.md:
- Did we identify a video characteristic that predicts DP success?
- Is there a subset of videos where DP genuinely outperforms Greedy on strict metrics?
- What does this tell us about when DP is worth using?

If the analysis does NOT find a clear DP success regime, say so honestly. Do not
manufacture a story.

---

## Part 2: Theoretical Analysis

Generate `notes/dp_theoretical_analysis.md`.

### Section 2.1: DP recurrence formal recap

State the DP recurrence and transition cost we use:

V(i,j) = S(i,j) + max_k [V(i-1, k) - T(k, j)]

T(k,j) = jp * dt           if dt > 0  (forward jump)
        jp * |dt| + bp     if dt < 0  (backward jump)
        -rb                if k == j  (reuse)

with our values jp=0.01, rb=0.01, bp=0.5.

### Section 2.2: When DP differs from Greedy (sufficient condition)

Greedy picks j*_i = argmax_j S(i,j) for each sentence i independently.

DP picks a sequence that maximizes sum of S minus sum of T transitions.

Formalize: under what condition does DP pick the same path as Greedy?

Hint: When the score gap between Greedy's pick and DP's alternative pick is
smaller than the transition cost differential between the two paths, DP
chooses something different from Greedy.

Derive a sufficient condition for DP and Greedy to coincide:
- For all sentences i, S(i, j*_i) - S(i, j_alt) > T(j_alt, j*_i+1) - T(j*_i, j*_i+1)
  for any alternative j_alt.

State this clearly with notation and brief proof sketch.

### Section 2.3: When does DP help (heuristic argument)

Three regimes worth characterizing:

**Regime A**: Score gap between best and second-best is large for every sentence.
Greedy picks are confident. DP gives same answer. DP marginal.

**Regime B**: Score gap is small but per-sentence picks happen to be temporally
ordered. Greedy picks already form a smooth path. DP has nothing to fix. DP
marginal.

**Regime C**: Score gap is small AND per-sentence picks are temporally
disordered. Greedy picks form a non-monotonic path. DP can re-order to favor
forward progression. DP genuinely helps.

For each regime, explain in 2-3 sentences and connect to our empirical findings:
- SigLIP track behaves like Regime A or B (DP marginal)
- Caption track has elements of Regime C BUT with a degenerate failure mode
  (scene-attractor problem) that pushes DP toward reuse instead of healthy
  re-ordering

### Section 2.4: Scene-attractor failure mode

Formalize the looping failure:

If there exists a scene j* such that S(i, j*) is dominantly the highest score
for multiple consecutive sentences i, AND the scene's center timestamp is
within the temporal prior window for those sentences, AND the score advantage
of j* exceeds the reuse_bonus, then DP will reuse j* across those sentences
because:
- Reusing j* costs only -rb (small bonus) in transition
- Switching to alternative scene with lower score costs S_gap + transition penalty

Result: DP pathway = j*, j*, ..., j* across the affected segment.

Explicitly note this is the mechanism behind review_7 Caption DP looping.

### Section 2.5: Implications for DP design

In 2-3 paragraphs, discuss design implications:
- Standard DP transition costs (time-based) are insufficient when one scene
  dominates similarity scores for multiple sentences
- Diversity penalty (lambda * 1[k=j]) could break attractor loops, at the
  cost of explicit hyperparameter
- Visual-similarity-aware transitions could provide finer-grained smoothness
  without depending on time alone
- These motivate future work direction (do not implement here)

---

## Part 3: Update Interpretation Document

Update `notes/cleanrun_interpretation_v2.md` to incorporate findings from Part 1
and Part 2. Add a new section "When does DP help?" that synthesizes:
- Empirical answer from subset analysis
- Theoretical answer from formalization
- Honest acknowledgment if no strong DP win regime was found

Keep the existing v2 interpretation content intact; just add the new section
toward the end.

---

## Critical Rules

- Do NOT re-run any pipeline phase or compute new metric values from raw data.
  Use only existing CSV files and intermediate JSON.
- Do NOT modify cleanrun_v1 data files.
- Be honest about findings. If subset analysis shows DP does not have a clean
  win regime, document that. Do not manufacture a story.
- Theoretical analysis should be mathematically precise but understandable
  by a CS reader unfamiliar with sequence alignment internals.
- Time budget: 4-6 hours. If approaching budget without clean conclusions,
  stop and report what was found.

## Output Deliverables

1. `notes/dp_subset_analysis.md` (Part 1)
2. `notes/dp_theoretical_analysis.md` (Part 2)
3. Updated `notes/cleanrun_interpretation_v2.md` with new "When does DP help?" section

Confirm completion with a 4-5 line summary:
- Was a DP success regime found empirically?
- What is the sufficient condition for DP and Greedy to coincide?
- What is the theoretical characterization of when DP genuinely helps?
- Was the scene-attractor failure mode formalized?
