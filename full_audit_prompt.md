# Full Correctness Audit Task — CCMA Implementation and Evaluation Pipeline

## CONTEXT

You are auditing a Master's thesis video summarization pipeline that uses CCMA (Capacity-Constrained Monotonic Alignment) — a state-augmented Viterbi variant adapted from Constrained Viterbi literature (Esterle, Chang & Rush 2014) — as the main matching algorithm in Phase 4.

The user is preparing to write Chapter 3 (Methodology) and Chapter 4 (Experiments) of their thesis. Before writing, they want to verify that:
1. The CCMA implementation is mathematically correct
2. All evaluation metrics are computed correctly
3. The data in results CSV is consistent and reproducible
4. There are no silent bugs that would invalidate empirical claims at defense

**Defense is ~8 weeks away. Budget for this audit: 2 days maximum. Stop and report any issues — do not attempt to fix bugs beyond simple obvious ones without user approval.**

## DELIVERABLES REQUIRED

At the end of the audit, produce a single markdown report (`audit_report.md`) with the following structure:

```
# CCMA Pipeline Audit Report

## Executive Summary
- Pass/Fail per scope (A, B, C, D)
- Critical issues found (if any)
- Recommended actions before writing

## Scope A: Implementation Correctness
[detailed findings]

## Scope B: Data Integrity
[detailed findings]

## Scope C: Evaluation Pipeline
[detailed findings]

## Scope D: End-to-End Reproducibility
[detailed findings]

## Appendix: Test Output Logs
[raw output from each test]
```

---

## SCOPE A: IMPLEMENTATION CORRECTNESS (Target: 4 hours)

### A.1 — CCMA Reduction Property

The CCMA algorithm must reduce to vanilla DP under relaxed constraints. This is non-negotiable: if it fails, there is a bug.

**Test:** Run CCMA with `c_max=1000`, `reuse_penalty=-0.01` (equal to -reuse_bonus in vanilla DP), `forward_jump_penalty=0.01`, `backward_jump_penalty=0.5`. Compare to vanilla DP output.

**Procedure:**
```python
for video_id in ["review_2", "review_5", "review_7"]:
    sim_matrix, scenes, video_dur = load_test_data(video_id, "caption_temporal")
    
    assign_dp = backend.dp_sequence_align(
        sim_matrix, scenes, video_dur,
        jump_penalty=0.01, reuse_bonus=0.01, backward_penalty=0.5
    )
    
    assign_ccma_relaxed = backend.ccma_align_sequence(
        sim_matrix, scenes, video_dur,
        c_max=1000, reuse_penalty=-0.01, 
        forward_jump_penalty=0.01, backward_jump_penalty=0.5
    )
    
    if assign_dp != assign_ccma_relaxed:
        REPORT FAILURE with both sequences printed
```

**Expected:** PASS for all 3 videos. If FAIL, the CCMA recurrence has a sign error or state transition bug.

### A.2 — Constraint Satisfaction

CCMA with `c_max=k` must NEVER produce assignments with more than `k` consecutive identical scene IDs.

**Test:** Run CCMA with `c_max=2` and `c_max=3` on all 10 videos. Compute max consecutive reuse for each output. Assert ≤ c_max.

**Procedure:**
```python
for c_max in [2, 3]:
    for video_id in all_10_videos:
        for track in ["caption_temporal", "siglip_temporal"]:
            assign = backend.ccma_align_sequence(
                sim_matrix, scenes, video_dur,
                c_max=c_max, ...
            )
            max_consec = compute_max_consecutive(assign)
            assert max_consec <= c_max, f"VIOLATION: {video_id} {track} c_max={c_max} got {max_consec}"
```

**Expected:** PASS for all 40 combinations (2 c_max × 10 videos × 2 tracks). If FAIL, the hard cap enforcement has a bug.

### A.3 — Edge Cases

Verify CCMA handles boundary conditions correctly.

**Test cases:**
1. `N = 1` (single sentence) — should return single argmax, no DP needed
2. `N = M` (sentences = scenes) — should still work, capacity constraint may bind
3. `c_max = 1` (no reuse allowed) — equivalent to Hungarian without tiling for `M >= N`
4. All-tied similarity scores — should return valid assignment (no NaN, no infinite loop)
5. Single scene M=1 with N>1 — should assign all sentences to scene 0 if c_max >= N

**Procedure:** Construct synthetic test inputs for each case. Run CCMA. Verify output is valid (no -inf, no NaN, length=N, all values in [0, M-1]).

**Expected:** All cases produce valid output. If any case crashes or produces invalid output, REPORT.

### A.4 — Backward Pointer Reconstruction

The backpointers must reconstruct the actual optimal path, not just any valid path.

**Test:** For 3 videos, run CCMA and verify:
1. The reconstructed assignment's total score (sum of similarities + transition costs) matches the DP table's optimum at the terminal state
2. Manual verification: trace the path and recompute its score independently

**Procedure:**
```python
assign, dp_table = backend.ccma_align_sequence(..., return_dp_table=True)
# Manually compute score of the reconstructed assignment
manual_score = compute_path_score(assign, sim_matrix, scenes, video_dur, params)
# Compare to DP optimum
final_state_score = dp_table[N-1, assign[-1], compute_final_r(assign)]
assert abs(manual_score - final_state_score) < 1e-6
```

**Expected:** PASS. If FAIL, the backpointer logic is broken — the algorithm finds a good path but reports a different one.

### A.5 — Determinism

Same inputs must produce same outputs across reruns.

**Test:** Run CCMA on review_7 caption_temporal 5 times in a row. Verify all 5 outputs are identical.

**Expected:** PASS. If FAIL, there is nondeterminism in tie-breaking or numerical precision issues.

---

## SCOPE B: DATA INTEGRITY (Target: 4 hours)

### B.1 — Resolve Numerical Inconsistency

Background: Previous reports show Scene Diversity for `caption_temporal_dp` arm as both 0.811 (full ablation report v1) AND 0.832 (after CCMA addition report) AND 0.680 (in CCMA full ablation report). These are three different values for the same arm and metric. Only one can be correct.

**Test:** 
1. Locate the canonical `ablation_results.csv` that the user will cite in thesis
2. For `caption_temporal_dp` arm, extract Scene Diversity per video
3. Recompute Scene Diversity from raw `scene_matches_caption_temporal_dp.json` files for each video
4. Compare: do they match?
5. If multiple CSVs exist with different values, determine which one corresponds to the latest valid run

**Expected:** Match. If MISMATCH, REPORT with all values and identify the canonical version.

### B.2 — Cache Consistency

`scene_matches_<arm>.json` files should be consistent with what the algorithm would produce now.

**Test:** For 3 random videos × 3 arms (caption_temporal_dp, caption_temporal_ccma, siglip_temporal_ccma):
1. Load cached `scene_matches_<arm>.json`
2. Re-run Phase 4 only for that video+arm (do NOT use cache)
3. Compare cached vs fresh output

**Expected:** Identical. If MISMATCH, the cache is stale and results CSV may be invalid.

### B.3 — Embedding Cache Validation

`embeddings_*.joblib` files must contain correct embeddings.

**Test:** For 2 videos:
1. Load cached embeddings
2. Re-extract embeddings from scratch (Phase 4 keyframe pass)
3. Compare element-wise (allow numerical tolerance 1e-4)

**Expected:** Match within tolerance.

### B.4 — Force-Delete Audit

User mentioned `--force` flag previously deleted some data and was restored from backup. Verify no orphaned or stale files exist.

**Test:**
1. List all `scene_matches_*.json` files
2. List all arms in `ARM_CONFIGS`
3. Verify no scene_matches files for deprecated arm names
4. Check timestamps: files should be consistent within each video's intermediate directory

**Expected:** Clean. If orphans exist, REPORT (do not delete without user approval).

---

## SCOPE C: EVALUATION PIPELINE (Target: 4 hours)

### C.1 — VisCoher_strict Correctness

Formula: average cosine similarity between consecutive matched FRAMES, EXCLUDING same-scene pairs.

**Test:**
1. Create synthetic test case: 5 matches, 3 with scene_a == scene_b, 2 with scene_a != scene_b
2. Manually compute expected VisCoher_strict (only 2 pairs counted)
3. Run `compute_strict_viscoher` function
4. Compare

**Expected:** Match. If MISMATCH, the metric computation has a bug.

### C.2 — Scene Diversity Correctness

Formula: `num_unique_scenes / num_sentences`.

**Test:**
1. Synthetic case: assignment `[2, 4, 4, 11, 11, 16]` → unique = {2, 4, 11, 16}, num_unique=4, num_sentences=6, diversity=4/6=0.667
2. Synthetic case: assignment `[2, 2, 2, 2, 2, 2]` → diversity = 1/6 = 0.167
3. Synthetic case: all unique [1, 2, 3, 4, 5] → diversity = 1.0

**Expected:** All match.

### C.3 — Max Consecutive Reuse Correctness

Formula: longest run of identical consecutive values.

**Test:**
1. Synthetic `[1, 1, 1, 2, 2]` → max_consec = 3
2. Synthetic `[1, 2, 1, 2, 1]` → max_consec = 1
3. Synthetic `[1, 1, 2, 2, 2]` → max_consec = 3
4. Single element `[5]` → max_consec = 1
5. Empty list `[]` → max_consec = 0

**Expected:** All match.

### C.4 — Statistical Test Validity

The user uses paired t-test (scipy.stats.ttest_rel) and Wilcoxon (scipy.stats.wilcoxon) for arm comparisons.

**Test:** 
1. Verify input data is paired correctly (same video_id ordering between arms)
2. Verify no missing values that would silently skew results
3. Verify alpha=0.05 used consistently
4. Spot-check 2 reported p-values from the latest results by re-running the test manually

**Expected:** Match within numerical tolerance.

### C.5 — LLM Judge Validation

LLM Judge should return real scores, not placeholder (1.0, 3.0, or 0.0).

**Test:**
1. Sample 5 cached `eval_results_<arm>.json` files
2. Check if `information_retention`, `factual_faithfulness`, `visual_relevance` are non-default
3. Verify by re-running LLM Judge on the same input — should produce similar score (±1 due to LLM variance)

**Expected:** Real, non-placeholder scores.

### C.6 — CLIPScore Computation

CLIPScore implementation uses `2.5 * max(cos_sim, 0)` formula.

**Test:**
1. Verify formula matches published CLIPScore (Hessel et al. 2021): `w * max(cos_sim, 0)` where `w = 2.5`
2. Spot-check 3 image-text pairs manually

**Expected:** Match published formula.

### C.7 — Temporal Accuracy Validation

`temporal_alignment_score` uses thresholds (5, 15, 30, 60s). Error = distance from matched frame timestamp to source_timestamp_hint range.

**Test:**
1. Synthetic case: source_hint=[100, 200], retrieved_ts=150 → error=0 (within range)
2. Synthetic case: source_hint=[100, 200], retrieved_ts=120 → error=0
3. Synthetic case: source_hint=[100, 200], retrieved_ts=80 → error=20
4. Synthetic case: source_hint=[100, 200], retrieved_ts=250 → error=50

**Expected:** All match.

---

## SCOPE D: END-TO-END REPRODUCIBILITY (Target: 4 hours)

### D.1 — Full Pipeline Determinism

Goal: verify that re-running the entire pipeline produces identical CSV results.

**Test:**
1. Pick 2 videos (small/medium duration to fit in time budget)
2. Delete ALL intermediate data for these videos: `rm -rf data/intermediate/review_X/`
3. Re-run pipeline from scratch for these 2 videos × 2 arms (caption_temporal_dp, caption_temporal_ccma)
4. Compare fresh CSV results with cached results
5. Verify scene_matches, embeddings, retrieval outputs all match (within numerical tolerance)

**Expected:** Match. If MISMATCH, there is non-determinism in the pipeline (random seeds, model nondeterminism, etc.) that needs to be characterized.

### D.2 — Seed Verification

Verify seeds are properly set across all phases.

**Test:** Grep codebase for `random_seed`, `torch.manual_seed`, `np.random.seed`. Verify they are set consistently.

**Expected:** Seeds set in Phase 2 (LLM), Phase 4 (random retrieval baseline), and any sampling operations.

### D.3 — Configuration Coherence

`default.yaml` parameters must match what is reported in results.

**Test:**
1. Read `default.yaml` retrieval section
2. Read latest results CSV header/metadata
3. Verify CCMA parameters used in CSV match config: `ccma_c_max`, `ccma_reuse_penalty`, `ccma_forward_jump_penalty`, `ccma_backward_jump_penalty`

**Expected:** Match.

### D.4 — Phase Dependency Order

Pipeline must run Phase 1 → 2 → 3 → 4 → 5 in order. Each phase must use the previous phase's output.

**Test:** Verify Phase 4 uses Phase 1 transcript, Phase 2 summary, and Phase 4 keyframes. Trace data flow for 1 video.

**Expected:** Correct order, no shortcut.

---

## CRITICAL RULES FOR THE AUDIT AGENT

1. **DO NOT FIX BUGS WITHOUT APPROVAL.** If you find an issue, REPORT it. User decides whether to fix.

2. **DO NOT MODIFY production code.** Audit tests can be in a separate `audit_tests/` directory.

3. **TIME BOX EACH SCOPE.** If a scope test exceeds its time budget, stop and report what was completed.

4. **TRANSPARENT REPORTING.** For each test, report:
   - PASS / FAIL / SKIPPED (with reason)
   - Actual values vs expected
   - Severity if FAIL: BLOCKING (defense risk) / MODERATE (needs note in thesis) / MINOR (cosmetic)

5. **DO NOT TRUST CACHED RESULTS BLINDLY.** If you suspect cache corruption, prioritize Scope B.

6. **AT MAXIMUM, USE 2 DAYS OF WORK.** If audit cannot complete in 2 days, prioritize Scope A > B > C > D.

7. **IF YOU FIND A BLOCKING BUG**, stop immediately and report. Do not continue auditing downstream scopes — the bug may invalidate them.

8. **PROVIDE RAW DATA.** Include all test command outputs in the appendix. The user wants to verify your verification.

## FINAL OUTPUT

Single file: `audit_report.md` in the project root.

If the audit passes all scopes: report explicitly "AUDIT PASSED — SAFE TO PROCEED WITH WRITING."

If issues found: report severity, specific test that failed, and recommended action (FIX / DOCUMENT IN THESIS / IGNORE).

END OF PROMPT
