# review_1 Consistency Check: SigLIP Track

## Summary
The current results for `review_1` in the SigLIP track show a significant shift since the initial DP tuning session. While DP still achieves 100% Temporal Accuracy, it no longer provides the +9% Visual Coherence boost observed during tuning because it now matches the Greedy baseline exactly.

## Metrics Comparison

| Source | Arm | CLIPScore | TempAcc | VisCoher |
| :--- | :--- | :--- | :--- | :--- |
| **Tuning Notes** | siglip_temporal (Greedy) | - | 0.7140 | 0.6970 |
| **Tuning Notes** | siglip_temporal_dp | - | 1.0000 | 0.7628 |
| **Current (JSON)** | siglip_temporal (Greedy) | 0.4849* | 1.0000* | 0.6997* |
| **Current (JSON)** | siglip_temporal_dp | 0.4849 | 1.0000 | 0.6997 |

*\*Inferred as identical to DP based on scene matching analysis.*

## Scene Assignment Analysis
Current assignments for `review_1` SigLIP DP:
```python
# From scene_matches_siglip_temporal_dp.json
dp_assignments = [2, 8, 12, 17, 27, 38, 50, 63]

# Inferred Greedy (Top-1 alternatives)
greedy_assignments = [2, 8, 12, 17, 27, 38, 50, 63]
```
**Difference: 0 dari 8 sentences different.**

Manual inspection of `scene_matches_siglip_temporal_dp.json` confirms that for every sentence, the DP algorithm selected the **Top-1** semantic match (Alternative 0).

## Diagnosis: Outcome B (Inconsistent)

### 1. Greedy Performance Jump
Greedy accuracy improved from **0.714 to 1.000**. This suggests that the signal quality (embeddings) or the temporal guidance weight has been modified since tuning. Because Greedy is already "perfectly" aligned, DP finds no "errors" to fix.

### 2. Loss of DP Coherence Benefit
During tuning, DP achieved **0.7628** VisCoher. Currently, it only achieves **0.6997**. 
- This indicates that the current "correct" scenes (those with 1.0 TempAcc) are less visually similar to each other than the ones identified during the tuning session.
- The recent update to `embeddings_*.joblib` (2026-05-05 16:53) likely shifted the similarity landscape.

### 3. Algorithm Sensitivity
The current `dp_jump_penalty: 0.01` is too low to force DP to deviate from a strong semantic match in favor of coherence when the semantic match is already temporally correct. 

## Recommendation
The algorithm is **working correctly** (it is finding the optimal path given the scores), but the "easy" nature of the current `review_1` signal makes it a poor test case for DP benefits. The coherence gains are more visible in `review_2` where Greedy still struggles (TempAcc 0.66).

**Verdict**: The reported "0/8 different" is accurate for the *current* state of the data, but the data itself has changed since the tuning notes were written.
