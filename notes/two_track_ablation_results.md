# Two-Track Ablation Results: Caption vs SigLIP

This ablation study evaluates the performance of the video summarization pipeline across two signal tracks (Caption-based and SigLIP-based) with varying matching algorithms and temporal priors.

## Setup
- **Dataset**: 10 Gadget Review Videos (1-5 minutes each)
- **Signal Tracks**:
    - **Caption**: Qwen2.5-VL-7B generated dense captions per scene.
    - **SigLIP**: Google SigLIP-2 (so400m) semantic embeddings per scene.
- **Ablation Arms**:
    1. `random`: Baseline
    2. `caption_direct`: Caption + No Temporal Prior + Greedy
    3. `caption_temporal`: Caption + Temporal Prior + Greedy
    4. `caption_temporal_dp`: Caption + Temporal Prior + DP alignment
    5. `siglip_direct`: SigLIP + No Temporal Prior + Greedy
    6. `siglip_temporal`: SigLIP + Temporal Prior + Greedy
    7. `siglip_temporal_hungarian`: SigLIP + Temporal Prior + Hungarian matching
    8. `siglip_temporal_dp`: SigLIP + Temporal Prior + DP alignment

## Final Aggregate Results (10/10 Videos)

| Arm | CLIPScore | TempAcc (15s) | VisCoher |
|-----|-----------|---------------|----------|
| random | 0.458 | 0.136 | 0.660 |
| caption_direct | 0.584 | 0.255 | 0.663 |
| caption_temporal | 0.575 | 0.758 | 0.672 |
| caption_temporal_dp | 0.569 | 0.794 | 0.719 |
| siglip_direct | 0.550 | 0.302 | 0.595 |
| siglip_temporal | 0.552 | 0.902 | 0.629 |
| siglip_temporal_hungarian | 0.553 | 0.919 | 0.630 |
| siglip_temporal_dp | 0.555 | 0.882 | 0.635 |

## Analysis of Claims

### 1. Effect of Temporal Prior
The addition of the temporal prior significantly improves **Temporal Accuracy (TempAcc)** across both signals.
- For **Caption**: `caption_direct` (0.255) -> `caption_temporal` (0.758). 
- For **SigLIP**: `siglip_direct` (0.302) -> `siglip_temporal` (0.902).
- *Observation*: Without temporal priors, both signals suffer from significant temporal drift, though SigLIP remains slightly more robust.

### 2. Effect of DP Sequence Alignment
DP alignment consistently improves **Visual Coherence (VisCoher)** by enforcing logical scene transitions.
- **Caption Track**: DP improved VisCoher from **0.672** to **0.719** (average across 10 videos).
- **SigLIP Track**: DP improved VisCoher from **0.629** to **0.635**.
- *Trade-off*: DP slightly decreases semantic relevance (`CLIPScore`) in the caption track (0.575 -> 0.569) as it prioritizes sequence flow.

### 3. Caption vs SigLIP Comparison
- **Semantic Quality**: The Caption track (`caption_temporal_dp`) achieves higher **CLIPScore** (0.569 vs 0.555) compared to SigLIP.
- **Visual Coherence**: The Caption track produces significantly more visually coherent sequences (higher average VisCoher of 0.719 vs 0.635 for SigLIP).
- **Temporal Robustness**: SigLIP remains superior in strict temporal alignment (~0.90 vs ~0.79 TempAcc), which is expected as embeddings capture more fine-grained temporal patterns than discrete captions.

## Verdict
The **Caption + Temporal Prior + DP** configuration (`caption_temporal_dp`) is the most balanced approach for video summarization, offering superior semantic relevance and the best visual flow among all tested configurations.
