# DP Tuning Decision Report

## Goal
Balance `Visual Coherence` (smoothness) and `Temporal Accuracy` (alignment with script hints). The previous settings were too aggressive, causing the DP to "stick" to scenes and ignore temporal guidance.

## Sweep Results (Video: review_1)
A fine-grained sweep identified a sharp performance cliff between `0.016` and `0.018`.

| Jump Penalty (JP) | Reuse Bonus (RB) | TempAcc | VisCoher | Result |
| :--- | :--- | :--- | :--- | :--- |
| 0.00 (Greedy) | - | 0.7140 | 0.6970 | Baseline |
| 0.30 | 0.30 | 0.1430 | 0.9750 | **Too Aggressive** |
| 0.01 | 0.01 | **1.0000** | **0.7628** | **Optimal (Win-Win)** |
| 0.016 | 0.01 | 1.0000 | 0.7628 | Safe |
| 0.018 | 0.01 | 0.5000 | 0.9226 | Stickiness begins |

## Decision
**Chosen Value: `dp_jump_penalty: 0.01` and `dp_reuse_bonus: 0.01`**

### Justification:
1. **Recovery of Temporal Accuracy**: At `0.01`, we recovered and even improved the `TempAcc` (1.0 vs 0.714 baseline) compared to the greedy approach.
2. **Preserved Coherence Benefit**: While `VisCoher` is lower than the aggressive setting (0.76 vs 0.97), it is still **+9.4% higher than the Greedy baseline** (0.697), fulfilling the goal of improving smoothness without breaking alignment.
3. **Cross-Video Stability**: Verified on `review_2` that these lower values prevent the catastrophic 85% drop in `TempAcc` seen previously.

## Configuration Update
Updated `configs/default.yaml` with the new values.
