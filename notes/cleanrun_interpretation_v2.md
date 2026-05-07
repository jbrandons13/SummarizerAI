# Cleanrun Interpretation V2 (Track C Fallback)

## Headline Findings

The inclusion of **Scene Diversity** and **Strict Visual Coherence** metrics provides a more nuanced and honest appraisal of the retrieval arms. The primary finding is that the apparent "Visual Coherence" superiority of the Dynamic Programming (DP) algorithm in the Caption track was partially inflated by redundant scene reuse (looping).

1.  **DP Visual Coherence is Partially Artificial**: In the Caption track, DP improved standard `visual_coherence_mean` significantly (p=0.03). However, when using `viscoher_strict` (which excludes same-scene transitions), the significance disappears (p=0.52). This confirms that DP's smoothing effect in the Caption track is heavily driven by its tendency to "stick" to a single high-scoring scene when alternatives are weak.
2.  **SigLIP is More Robust to Looping**: SigLIP arms exhibited significantly higher scene diversity than Caption DP. While Caption DP reached a low of 0.33 diversity in `review_7`, SigLIP DP maintained higher diversity (mean 0.95 vs 0.85).
3.  **Temporal Priors remain the Strongest Win**: The application of temporal priors (Temporal track) remains the most statistically robust improvement across all metrics, particularly `temporal_acc_15s` (p < 0.001), regardless of the matching algorithm used.
4.  **Caption vs SigLIP**: Even with strict metrics, Caption DP remains "more coherent" than SigLIP DP (p=0.019), but the gap is smaller than previously thought.

## Trade-off Table (Updated)

| Metric | Caption Temporal (G) | Caption Temporal (DP) | SigLIP Temporal (G) | SigLIP Temporal (DP) |
| :--- | :--- | :--- | :--- | :--- |
| **CLIPScore** | 0.575 | 0.570 | 0.550 | 0.554 |
| **TempAcc (15s)** | **0.770** | **0.806** | **0.902** | **0.882** |
| **VisCoher (Mean)** | 0.662 | 0.710 | 0.626 | 0.632 |
| **VisCoher (Strict)**| 0.662 | 0.674 | 0.626 | 0.613 |
| **Scene Diversity** | 1.000 | 0.849 | 1.000 | 0.949 |
| **Visual Relevance**| 2.8 | 2.6 | 2.3 | 2.5 |

## Looping Analysis

The parameter sweep (Track C Pilot) confirmed that looping in the Caption DP track is not easily resolved by simple penalty tuning. Even with `dp_backward_penalty` reduced to 0.05, the algorithm persisted in picking the same scene for 5 consecutive sentences in `review_7`. This suggests a "sink" effect where one scene (e.g., scene 4 in `review_7`) has a similarity score so much higher than its temporal neighbors that no reasonable transition penalty can dislodge it.

## When does DP help?

A targeted subset analysis of 10 videos (20 arm comparisons) reveals that Dynamic Programming (DP) rarely provides a "clean win" over Greedy matching in our current pipeline. 

### Empirical Reality
Only 1 out of 10 videos on the Caption track (`review_6`) met the success regime for DP (improving strict coherence and temporal accuracy while maintaining high scene diversity). On the SigLIP track, zero videos met these criteria. Across all videos, there was no significant correlation between video characteristics (length, scene density) and DP performance. Instead, DP's primary impact is observed in the second half of summaries, where it diverges from Greedy to enforce temporal consistency, often at the cost of diversity.

### Theoretical Regimes
The theoretical characterization of DP matching identifies three regimes:
1.  **Regime A (High Certainty):** Similarity scores are highly discriminative. Greedy is already optimal. DP is redundant. (Most SigLIP cases).
2.  **Regime B (Natural Order):** Best matches naturally follow chronological order. Greedy is coherent by accident. DP is redundant.
3.  **Regime C (Disordered Ambiguity):** Scores are ambiguous and greedy picks are jittery. DP should provide a smoother, forward-moving path.

While the Caption track often enters Regime C, it is plagued by a **"Scene Attractor" failure mode**. When one scene dominates similarity scores across several sentences, DP interprets the "reuse bonus" and low-cost self-transition as the optimal path, resulting in the observed looping artifacts.

### Synthesis
DP is most valuable when the semantic signal is weak enough to require temporal guidance but strong enough to avoid single-scene domination. In its current form, DP's theoretical promise is largely neutralized by the degeneracy of our similarity score distributions in the Caption track and the sufficiency of Greedy matching in the SigLIP track.

## Conclusion

While DP provides a smoother visual experience, users and researchers should be aware that this smoothness comes at the cost of visual variety. For content where scene diversity is critical (e.g., fast-paced reviews), Greedy matching with SigLIP or Hungarian matching (which enforces uniqueness) may be preferable to DP. For "cinematic" summaries where transitions matter more than variety, DP remains the superior choice, provided the looping is monitored.
