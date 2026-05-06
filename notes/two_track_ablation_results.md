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

## Final Aggregate Results (Cleanrun 2026-05-05)

| Arm | CLIPScore | TempAcc (15s) | VisCoher |
| :--- | :--- | :--- | :--- |
| random | 0.458 | 0.111 | 0.662 |
| caption_direct | 0.586 | 0.268 | 0.656 |
| caption_temporal | 0.575 | 0.770 | 0.662 |
| **caption_temporal_dp** | 0.570 | 0.806 | **0.710*** |
| siglip_direct | 0.554 | 0.315 | 0.594 |
| **siglip_temporal** | 0.550 | **0.902** | 0.626 |
| siglip_temporal_hungarian | 0.551 | 0.919 | 0.628 |
| siglip_temporal_dp | 0.554 | 0.882 | 0.632 |

*\* Significant improvement (p < 0.05) vs. its Greedy counterpart.*

## Final Statistical Analysis Summary
- **Temporal Prior (T)** is the dominant factor, yielding massive, highly significant improvements in Temporal Accuracy (** p < 0.01) for both tracks.
- **DP Sequence Alignment** provides a significant boost to **Visual Coherence** on the Caption track (* p < 0.05, Cohen's d=0.82) but has no significant effect on the SigLIP track.
- **Caption Track** is significantly more visually coherent (** p < 0.01) than the SigLIP track, while the **SigLIP Track** is superior in strict temporal alignment (though not significant at n=10).
- **Hungarian Matching** is mathematically degenerate in this regime (9/10 identical to Greedy) and provides no practical benefit.

## Detailed Analysis & Interpretation
For the full per-video breakdown, significance test results, and honest interpretation for Thesis Chapter 4, please refer to:
- [Canonical Interpretation](file:///home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/notes/cleanrun_interpretation.md)
- [Per-Video Analysis](file:///home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/notes/per_video_analysis.md)
- [Significance Tests Report](file:///home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/notes/significance_tests.md)
