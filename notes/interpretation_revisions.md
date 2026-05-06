# Interpretation Revisions Log
**Date:** 2026-05-05

This document summarizes the revisions made to the cleanrun interpretation based on manual review and verification of intermediate data.

## 1. Scene/Sentence Density Verification
- **Issue:** The original analysis claimed `review_7` had a 78:1 scene-to-sentence ratio (468 scenes).
- **Verification:** Manual inspection of `keyframes_manifest.json` for all 10 videos revealed that `review_7` actually has 168 scenes for 6 sentences, resulting in a **28.0:1** ratio. The average ratio across all 10 videos is **25.5:1**.
- **Correction:** `review_7` is NOT a significant outlier in scene density (unlike `review_4` at 52:1). The causal claim linking scene density to DP's failure on `review_7` has been removed.
- **Revised Hypothesis:** The anomalous behavior is now attributed to potential noisy interactions between hallucinations in the summary and the embedding signal, rather than scene density.

## 2. Statistical Honesty in Trade-offs
- **Issue:** The trade-off table used inconsistent arms (picking the highest score regardless of method) and lacked significance markers.
- **Correction:** 
    - The table now consistently compares `caption_temporal_dp` vs `siglip_temporal_dp` (the "full pipeline" arms).
    - Statistical significance markers (p-values and Cohen's d) were added based on `significance_tests.md`.
    - Explicitly stated that CLIPScore and TempAcc differences are NOT statistically significant at n=10.
- **Softened Language:** Hypotheses about "distractor" scenes in `review_7` were softened to clarify they are post-hoc hypotheses, not verified findings.

## 3. DP's CLIPScore Cost
- **Addition:** Added a new section documenting the monotonic decrease in CLIPScore within the Caption track as more constraints (temporal prior, DP) are added. 
- **Significance:** This highlights the inherent trade-off between local semantic match and global sequence coherence.

## 4. Visual Coherence Fact-Check
- **Verification:** Confirmed that Visual Coherence actually drops for SigLIP DP on `review_7` (0.751 → 0.717), supporting the claim that DP can be counter-productive on already "smooth" signals when noisy summary content is present.
