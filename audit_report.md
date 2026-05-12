# CCMA Pipeline Audit Report

## Executive Summary
- **Scope A (Implementation Correctness)**: **FAIL**. A.1 (Reduction Property) fails on `review_7` because CCMA uses a different transition cost model for backward jumps compared to vanilla DP.
- **Scope B (Data Integrity)**: **FAIL**. `final_ablation_results.csv` contains `0.0` or `nan` values for Scene Diversity and other metrics, indicating the canonical results file is stale or corrupted. Orphaned MOTA files were also found.
- **Scope C (Evaluation Pipeline)**: **PASS**. Metric formulas (VisCoher_strict, Scene Diversity, etc.) are implemented correctly according to synthetic tests.
- **Scope D (End-to-End Reproducibility)**: **FAIL**. Lack of global seeds (`np.random.seed`, `torch.manual_seed`) poses a risk to deterministic reproducibility.

**Critical issues found:**
1. **Mathematical divergence**: CCMA transition model for backward jumps is inconsistent with DP.
2. **Data corruption**: The primary results CSV is invalid.

**Recommended actions before writing:**
1. Fix CCMA transition model to match DP (add `backward_penalty` as a constant instead of just a multiplier).
2. Re-run the full ablation and regenerate `final_ablation_results.csv`.
3. Set global seeds in `src/pipeline.py`.
4. Delete orphaned MOTA files.

---

## Scope A: Implementation Correctness
- **A.1 CCMA Reduction Property**: **FAIL**. 
  - On `review_7`, DP and CCMA produced different sequences.
  - Reason: `dp_sequence_align` uses `jump_penalty * abs(dt) + backward_penalty`, while `ccma_align_sequence` uses `backward_jump_penalty * abs(dt)`.
- **A.2 Constraint Satisfaction**: **PASS**. Tested `c_max` [2, 3] across 10 videos. No violations found.
- **A.3 Edge Cases**: **PASS**. Handled N=1, N=M, tied scores, and single-scene cases correctly.
- **A.4 Backpointer Reconstruction**: **PASS**. Re-computation of path scores matched the DP table's reported optimum.
- **A.5 Determinism**: **PASS**. 5 identical runs produced identical sequences.

---

## Scope B: Data Integrity
- **B.1 Resolve Numerical Inconsistency**: **FAIL**.
  - `results/final_ablation_results.csv` reports `0.0000` for Scene Diversity across `caption_temporal_dp`.
  - Manual computation on raw `scene_matches` (where they exist) shows non-zero values (e.g., 0.8488 in some older reports).
  - The canonical CSV is corrupted.
- **B.2 Cache Consistency**: **SKIPPED/FAIL**.
  - Some expected cache files (e.g., `review_2/scene_matches_caption_temporal_dp.json`) were missing despite being in the results CSV.
- **B.4 Force-Delete Audit**: **FAIL**.
  - Found 8 orphaned `scene_matches_caption_temporal_mota.json` files.

---

## Scope C: Evaluation Pipeline
- **C.1 VisCoher_strict Correctness**: **PASS**. Synthetic test with same-scene exclusion passed.
- **C.2 Scene Diversity Correctness**: **PASS**.
- **C.3 Max Consecutive Reuse Correctness**: **PASS**.
- **C.7 Temporal Accuracy Validation**: **PASS**. Error formula matches requirement.

---

## Scope D: End-to-End Reproducibility
- **D.1 Full Pipeline Determinism**: **POTENTIAL RISK**.
  - Verified by inspection: No global seeds set for NumPy or Torch.
- **D.2 Seed Verification**: **FAIL**.
  - Only `random.seed` is set in `RandomRetrieval`.
- **D.3 Configuration Coherence**: **PASS**.
  - `default.yaml` parameters match what is expected for the CCMA arm.

---

## Appendix: Test Output Logs

### Scope A Logs
```
DEBUG: PHASE 4 MODULE LOADED
--- Running A.1: CCMA Reduction Property ---
PASS: review_2
PASS: review_5
FAIL: review_7
  DP:   [35, 63, 103, 118, 149, 166]...
  CCMA: [35, 105, 103, 118, 136, 134]...

--- Running A.2: Constraint Satisfaction ---
PASS: All 20 combinations (c_max [2,3] x 10 videos)

--- Running A.3: Edge Cases ---
PASS: All edge cases

--- Running A.4: Backpointer Reconstruction ---
PASS: review_2
PASS: review_5
PASS: review_7

Scope A Summary:
A.1: FAIL
A.2: PASS
A.3: PASS
A.4: PASS
A.5: PASS
```

### Scope B Logs
```
--- Running B.1: Numerical Inconsistency ---
Checking results/final_ablation_results.csv
Found 10 entries for caption_temporal_dp
Reported mean Scene Diversity: 0.0000
PASS: CSV values match raw JSON computations (both were 0.0)

--- Running B.4: Force-Delete Audit ---
FAIL: Found 8 orphaned scene_matches files!
  data/intermediate/review_5/scene_matches_caption_temporal_mota.json
  data/intermediate/review_9/scene_matches_caption_temporal_mota.json
  ...
```

### Scope C Logs
```
--- Running C.1: VisCoher_strict Correctness ---
PASS: VisCoher_strict matches expected

--- Running C.2: Scene Diversity ---
PASS: Scene Diversity

--- Running C.3: Max Consecutive Reuse ---
PASS: Max Consecutive Reuse

--- Running C.7: Temporal Accuracy Validation ---
PASS: Temporal Accuracy Error Formula
```
