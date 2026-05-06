# Final Ablation Study: Statistical Analysis & Interpretation
**Date:** 2026-05-05
**Dataset:** 10 videos × 8 arms (Deterministic Cleanrun)

## 1. Headline Findings
- **Random Baseline Confirmation:** Metric scores for the `random` arm (e.g., TempAcc=0.111, VisCoher=0.66) are significantly lower than the best retrieval arms (TempAcc ~0.90, VisCoher ~0.71), confirming that the metrics are not trivially saturated and capture meaningful retrieval quality.
- **Dominance of Temporal Prior:** The Gaussian temporal prior is the single most impactful factor. It improves TempAcc from 0.27 → 0.77 (+0.50) on the Caption track and 0.31 → 0.90 (+0.59) on the SigLIP track. These effects are highly significant (p < 0.001) with massive effect sizes (Cohen's d > 2.2).
- **DP provides consistent gain on Caption track:** For Caption-based retrieval, DP improves Visual Coherence by +0.047 (p=0.03, d=0.82) and helps in 7/10 videos. This confirms DP's role in smoothing noisy textual signals.
- **DP has marginal/mixed effect on SigLIP track:** SigLIP already achieves near-ceiling TempAcc (5/10 videos reach 1.0), leaving little headroom for DP. Aggregate effects are non-significant, with mixed results per video.
- **Hungarian Degeneracy:** As mathematically predicted for the scenes >> sentences regime, the Hungarian algorithm produced results identical to Greedy in 9/10 videos.

## 2. Trade-off: Caption vs. SigLIP
The two signal tracks should be viewed as complementary rather than as winner/loser:

| Feature | Caption + DP | SigLIP + DP | Difference |
| :--- | ---: | ---: | :--- |
| CLIPScore | 0.570 | 0.554 | +0.016 (n.s., p=0.32) |
| Visual Coherence | **0.710** | 0.632 | **+0.078 (p=0.006, d=1.12)** |
| Temporal Accuracy | 0.806 | 0.882 | -0.076 (n.s., p=0.32) |

The two signal tracks differ significantly in Visual Coherence, with Caption + DP producing more visually coherent narration paths. Differences in CLIPScore and Temporal Accuracy are not statistically significant at n=10, although the directional pattern (SigLIP slightly favors temporal alignment, Caption slightly favors semantic match) is consistent with the per-video winners distribution. With a larger sample, these directional trends may reach significance, but we do not claim them as established findings.

## 3. When does DP help?
We hypothesize that DP is most effective when the underlying similarity signal is noisy or temporally discontinuous. 
- The **Caption signal** (Qwen-VL → MiniLM) relies on cross-modal text matching which is inherently noisier on a per-sentence basis. DP provides the necessary regularization to enforce sequential coherence.
- The **SigLIP signal** is naturally more aligned with the visual sequence when combined with a temporal prior, so DP's additional regularization adds little value and can occasionally "over-correct" (as seen in `review_7`).

### 3.1 DP's CLIPScore cost in Caption track
Within the Caption track, CLIPScore decreases monotonically as more constraints are added:
- `caption_direct`: 0.586
- `caption_temporal`: 0.575 (-0.011)
- `caption_temporal_dp`: 0.570 (-0.005)

The differences are not statistically significant individually, but the direction is consistent: adding temporal prior and DP both slightly reduce the per-frame semantic match score, while substantially improving temporal alignment and (for DP) visual coherence. This is a within-method trade-off worth flagging: DP optimizes sequence-level coherence at marginal cost to per-frame semantic match.

## 4. Understanding Hungarian Degeneracy
The finding that Hungarian coincides with Greedy in 90% of cases confirms that bipartite matching is not the optimal inductive bias for video retrieval where "supply" (scenes) vastly exceeds "demand" (sentences). In this regime, the bipartite optimum naturally collapses to the local optimum (Greedy). Dynamic Programming or Sequence Alignment remains the more appropriate tool for this domain.

## 5. LLM-as-Judge Corroboration
- **Metric Isolation:** Information Retention (3.3-3.4) and Factual Faithfulness (2.6-2.8) remained constant across all retrieval arms. This confirms that these metrics correctly isolate Phase 2 (Summarization) quality from Phase 4 (Retrieval) performance.
- **Visual Relevance vs. Coherence:** Interestingly, Caption DP showed a slight decrease in LLM-judged Visual Relevance despite a significant increase in Visual Coherence. This suggests a trade-off: DP may occasionally favor a "smooth" transition over a slightly more "relevant" but visually jarring scene.

## 6. Limitations & Scope
- **Sample Size:** n=10 videos limits statistical power; non-significant results (p > 0.05) should be interpreted as "no detectable difference" rather than "no difference."
- **Domain Specificity:** Results are limited to gadget reviews; cross-domain generalization (e.g., to movies or vlogs) remains to be tested.
- **Optimization Leakage:** DP parameters were tuned on a subset of the data; while verified on others, some risk of over-fitting to the "review video" aesthetic exists.

---
**Canonical analysis for Thesis Chapter 4.**
Data verified against cleanrun logs and aggregated CSVs.
