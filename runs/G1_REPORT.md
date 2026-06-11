# FACET Stage 1 (G1) Report

## 1. V-A1 Equivalence Check
- **Methodology**: `callback_on_step_end` + `StopIteration` at $k=9$.
- **Verdict**: **PASSED** (0.0 diff).

## 2. Audit: Hung Interactive Prompt
- **Verdict**: Unclosed string quote in interactive Python session (`>`). 100% idle GPU.

## 3. Alt-1 Fast-Forward Selection Evaluation (v2 & Finals)

### Final Checks & Binding End Conditions
1. **Anchor Inspection & Asymmetry Fix**:
   - *Issue*: $\hat{c}(w=0.2)$ median was initially 0.54–0.60.
   - *Fix*: Replaced conditional `w=0` reference image with an unconditional `_prep_reference(...)` (black image was causing asymmetric latents).
   - *Result*: $\hat{c}(0.2)$ restored to >0.70. Re-ran $k \in \{9, 12\}$. Still failed structural/semantic correlation hurdles.
2. **$k=30$ TAESD-Decoding Ceiling Control**:
   - *Result*: Median $\hat{c}$ Spearman = 0.7714; Median $mc$ Spearman = 0.4857.
   - *Conclusion*: Failed to reach the $>0.95$ threshold. TAESD inherently obliterates scale-ranking information even on final latents.
3. **Conditional VAE-Scoring Variant ($k=12$)**:
   - *Condition Triggered*: TAESD confirmed as binding constraint.
   - *Run*: Re-scored $k=12$ using the full pipeline VAE over a 5-shot subset.
   - *Result*: Median $\hat{c}$ Spearman = 1.0 (perfect structural preservation). Median $mc$ Spearman = 0.0857 (semantic ranking completely chaotic/uncorrelated at step 12).
   - *Verdict*: Even without TAESD corruption, early-step `pred-x0` previews do not carry scale-ranking information under this pipeline.

## 4. G1 Verdict
**Verdict: FAIL. Alt-1 is permanently dropped.**
Early-step `pred-x0` previews do not carry stable scale-ranking information under this pipeline. The $k=30$ TAESD control failed, and the $k=12$ VAE control proved that even with pure VAE decoding, semantic concepts remain completely unordered at step 12. Full sweeps revert to being the selection engine. G1 is permanently CLOSED.
